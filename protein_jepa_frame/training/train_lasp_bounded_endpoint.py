"""Opt-in bounded-energy endpoint development trainer (never a promotion gate).

No legacy loader or sampler is changed. Full BPTT covers every physical macro
step and microstep. Checkpoints deliberately cannot load into legacy energy.
Only train/validation manifests are opened; there is no test or resume path.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from protein_jepa_frame.models.lasp_bounded_energy import BoundedRidgeEnergy
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.lasp_run_state import rng_state, restore_rng, step_batch
from protein_jepa_frame.training.train_lasp_ca_dynamics import (
    CADynamics, anchor_noise_factor, make_loader, displacement_scales,
)
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.training.train_utils import _move_batch, _set_reproducibility
from protein_jepa_frame.data.protocol import validate_manifest_split


CHECKPOINT_KIND = 'lasp_bounded_endpoint_v1'
SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
GATE_PATH = REPO / 'results/lasp_bounded_v1/numerical_gate.json'


def energy_config(latent_dim, reference_temperature_k=348.):
    return dict(latent_dim=latent_dim, features=128, curvature_budget=.5,
                tail_init=.01, reference_temperature_k=reference_temperature_k)


def _all_finite_json(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite_json(v) for v in value.values())
    if isinstance(value, list):
        return all(_all_finite_json(v) for v in value)
    return True


def verify_numerical_gate(path, checkpoint, val_manifest, horizons, microsteps, component=None):
    """Fail closed on incomplete, stale-source, nonfinite or failed screening."""
    path = Path(path).resolve()
    if component is None and path != GATE_PATH.resolve():
        raise ValueError(f'numerical gate must be saved at {GATE_PATH}')
    raw = path.read_bytes()
    gate = json.loads(raw)
    if (gate.get('screening_passed') is not True
            or gate.get('test_opened') is not False
            or gate.get('optimizer_steps') != 0 or not _all_finite_json(gate)):
        raise ValueError('requires passed finite no-update validation screening')
    protocol = gate['protocol']
    if component is not None:
        if (Path(protocol.get('component', '')).resolve() != Path(component).resolve()
                or gate.get('component_sha256') != hashlib.sha256(Path(component).read_bytes()).hexdigest()
                or gate.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                or gate.get('manifest_sha256') != hashlib.sha256(Path(val_manifest).read_bytes()).hexdigest()):
            raise ValueError('new component numerical provenance mismatch')
    if Path(protocol['checkpoint']).resolve() != Path(checkpoint).resolve():
        raise ValueError('screened checkpoint differs from initialization')
    if Path(protocol['manifest']).resolve() != Path(val_manifest).resolve():
        raise ValueError('screened validation manifest differs')
    if not set(horizons) <= set(protocol['horizons']):
        raise ValueError('unscreened horizon')
    # The existing screening acceptance explicitly compares 8 against 16.
    if microsteps != 8 or not {4, 8, 16} <= set(protocol['microsteps']):
        raise ValueError('this gate authorizes only the screened 8-microstep path')
    windows = gate.get('windows', [])
    temperatures = set(protocol['temperatures'])
    if not windows or not temperatures:
        raise ValueError('empty screening panel')
    for horizon in horizons:
        for temperature in temperatures:
            if not any(w['horizon'] == horizon and w['temperature'] == temperature for w in windows):
                raise ValueError('incomplete horizon/temperature screening panel')
    for window in windows:
        modes = window.get('modes', {})
        if not {'4', '8', '16'} <= modes.keys():
            raise ValueError('missing screening refinement modes')
        for mode in modes.values():
            gain = mode.get('joint_backward_gain')
            if (mode.get('forward_finite') is not True
                    or mode.get('gradient_elements_finite') is not True
                    or not isinstance(gain, (float, int)) or not 0 <= gain <= 10.):
                raise ValueError('failed forward/backward screening')
        refinement = modes['8'].get('endpoint_ca_rms_to_microsteps_max')
        if not isinstance(refinement, (float, int)) or not 0 <= refinement <= .05:
            raise ValueError('failed 8-to-16 endpoint refinement screening')
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                screening_only=True, full_promotion=False, report=gate)


def load_bounded_checkpoint(checkpoint_path, component_path=None, device='cuda'):
    """Load only a bounded-endpoint checkpoint for evaluation callers.

    The explicit kind/schema check prevents the legacy CADynamics loader from
    interpreting the incompatible ``energy.*`` state as a legacy transformer.
    """
    checkpoint_path = Path(checkpoint_path)
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if source.get('checkpoint_kind') != CHECKPOINT_KIND or source.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('not a bounded endpoint v1 checkpoint')
    if source.get('test_opened') is not False:
        raise ValueError('bounded checkpoint has invalid test provenance')
    if component_path is None:
        component_path = checkpoint_path.parent / 'component.pt'
    model = BoundedEndpointDynamics(str(component_path), device, checkpoint_path,
                                    energy_seed=int(source['energy_seed']),
                                    microsteps=int(source['integrator']['microsteps']))
    model.kernel.load_state_dict(source['kernel'], strict=True)
    return model, source


class BoundedEndpointDynamics(CADynamics):
    """Isolated fixed-anchor, history-free CADynamics with fresh bounded energy."""

    def __init__(self, component, device, init_kernel, energy_seed=42, microsteps=8):
        source = torch.load(init_kernel, map_location='cpu', weights_only=False)
        if source.get('checkpoint_kind') not in ('lasp_ca_dynamics', CHECKPOINT_KIND):
            raise ValueError('initialization requires a stopped-best or bounded dynamics checkpoint')
        old = source.get('args', {})
        bounded_source = source.get('checkpoint_kind') == CHECKPOINT_KIND
        noise_meta = source.get('integrator', {}) if bounded_source else old
        noise_mix = float(noise_meta.get('noise_mix', 0.))
        noise_length = float(noise_meta.get('noise_length', 8.))
        if bounded_source:
            # CADynamics' init_kernel path strictly loads legacy HSpaceEnergy.
            # Build the kernel from its config without that load, then restore
            # only the compatible bath/history tensors below.
            config = StageB1bConfig(**source['kernel_config'])
            super().__init__(component, device, init_kernel=None,
                             history_conditioning=False,
                             noise_mix=noise_mix, noise_length=noise_length,
                             bath_dim=config.bath_dim,
                             collective_mobility=True)
            self.kernel = StageB1b(config).to(device)
            compatible = {k: v for k, v in source['kernel'].items()
                          if not k.startswith('energy.')}
            self.kernel.load_state_dict(compatible, strict=False)
        else:
            super().__init__(component, device, init_kernel=init_kernel,
                             history_conditioning=False,
                             noise_mix=old.get('noise_mix', 0.),
                             noise_length=old.get('noise_length', 8.),
                             collective_mobility=True)
        cfg = self.kernel.cfg
        if not cfg.use_bath or cfg.bath_coupling != 'velocity' or cfg.force_clip is not None:
            raise ValueError('bounded midpoint requires retained velocity bath and unclipped force')
        if isinstance(microsteps, bool) or not isinstance(microsteps, int) or microsteps < 1:
            raise ValueError('microsteps must be a positive integer')
        self.microsteps = microsteps
        self.noise_mix = noise_mix
        self.noise_length = noise_length
        self.energy_seed = energy_seed
        self.energy_config = energy_config(cfg.latent_dim, cfg.reference_temperature_k)
        # Construct on CPU without consuming/changing global sampling RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(energy_seed)
            fresh = BoundedRidgeEnergy(**self.energy_config)
        self.kernel.energy = fresh.to(device)
        cfg.rollout_grad_steps = None
        # Unused history posterior is retained for provenance, never optimized.
        self.kernel.history_posterior.requires_grad_(False)
        self.bundle.eval()
        self.last_sample_diagnostics = {}

    def integrator_metadata(self):
        return dict(sampler='rollout_microstep_v1', integrator='coupled_midpoint_v1',
                    microsteps=self.microsteps, macro_step_ps=self.kernel.cfg.step_ps,
                    microstep_ps=self.kernel.cfg.step_ps / self.microsteps,
                    noise_mix=self.noise_mix, noise_length=self.noise_length,
                    fixed_anchor=True, history_conditioning=False,
                    collective_mobility=True, full_bptt=True, grad_steps=None,
                    solver_tolerance=1e-6, solver_max_iterations=128)

    def sample(self, observed, mask, aa, ri, ci, temp, steps, n_samples,
               generator=None, grad_steps=None):
        if grad_steps is not None:
            raise ValueError('bounded endpoint requires full BPTT; grad_steps must be None')
        if isinstance(steps, bool) or int(steps) != steps or steps < 1 or n_samples < 2:
            raise ValueError('positive physical steps and at least two samples required')
        self.bundle.eval()
        b, _, l = observed.shape[:3]
        with torch.no_grad():
            h = self.bundle.encode(observed[:, -1], mask, aa, ri, ci)
        def repeat(x):
            return x[:, None].expand(b, n_samples, *x.shape[1:]).reshape(b*n_samples, *x.shape[1:])
        start, rm = repeat(h), repeat(mask)
        factor = None
        if self.noise_mix:
            factor = repeat(anchor_noise_factor(observed[:, -1, :, 1], mask,
                                                self.noise_mix, self.noise_length))
        temperature = None if temp is None else repeat(temp[:, None]).flatten()
        traj, bath = self.kernel.rollout_microstep_v1(
            start, rm, temperature, steps, seq_feature=start,
            H_hist=None, dt_hist=None, generator=generator,
            detach=not self.training, return_bath=True, grad_steps=None,
            noise_factor=factor, collective_mobility=True, microsteps=self.microsteps,
            integrator='coupled_midpoint_v1', solver_tolerance=1e-6,
            solver_max_iterations=128)
        if traj.shape[1] != steps + 1:
            raise ValueError('rollout must return the anchor plus physical macro observations')
        endpoint = traj[:, -1]
        if getattr(self, '_sample_time_mixture', False):
            # Independent uniform observation times turn independent paths into
            # iid mixture samples. Clone the RNG so path-noise budgets stay paired.
            time_rng = torch.Generator(device=start.device)
            time_rng.set_state(generator.get_state())
            times = torch.randint(1, steps + 1, (b*n_samples,), device=start.device,
                                  generator=time_rng)
            endpoint = traj[torch.arange(b*n_samples, device=start.device), times]
        else:
            times = torch.full((b*n_samples,), steps, device=start.device, dtype=torch.long)
        rotation, anchor = _extract_backbone_frames(observed[:, -1])
        ca = self.bundle.decode_ca(endpoint, start, repeat(aa), repeat(ri), repeat(ci),
                                   repeat(rotation), repeat(anchor), rm)
        # Log the requested local curvature and a conservative mobility bound
        # at the fixed anchor. These are diagnostics, not a global stability proof.
        with torch.enable_grad():
            D0, _, _ = self.kernel.bath(start.detach(), temperature)
        residual_bound = self.kernel.energy.residual_hessian_bound().detach()
        max_d = D0.detach().max()
        if factor is None:
            factor_norm_sq = start.new_tensor(1.)
        else:
            factor_norm_sq = torch.linalg.matrix_norm(factor.detach(), ord=2).square().max()
        mobility_bound = self.kernel.cfg.step_ps * max_d * factor_norm_sq * residual_bound
        bounds = dict(residual_hessian_bound=float(residual_bound),
                      mobility_curvature_bound=float(mobility_bound),
                      mobility_factor_norm_sq=float(factor_norm_sq),
                      mobility_bound_formula='step_ps*maxD*factor_operator_norm²*residual_hessian_bound',
                      tail_min=float(torch.nn.functional.softplus(self.kernel.energy.log_tail).detach().min()),
                      tail_max=float(torch.nn.functional.softplus(self.kernel.energy.log_tail).detach().max()),
                      bath_D_min=float(D0.detach().min()), bath_D_max=float(D0.detach().max()))
        flags = dict(latent_elements_finite=bool(torch.isfinite(traj).all()),
                     bath_elements_finite=bool(torch.isfinite(bath).all()),
                     ca_elements_finite=bool(torch.isfinite(ca).all()),
                     bounds_finite=all(math.isfinite(v) for v in bounds.values() if isinstance(v, float)))
        if not all(flags.values()):
            self.last_sample_diagnostics = flags
            raise FloatingPointError(f'nonfinite bounded rollout: {flags}')
        # Adjacent valid same-chain residues only; gaps/padding are not bonds.
        bonds = (rm[:, 1:] & rm[:, :-1] & (repeat(ci)[:, 1:] == repeat(ci)[:, :-1])
                 & (repeat(ri)[:, 1:] - repeat(ri)[:, :-1] == 1))
        lengths = (ca[:, 1:] - ca[:, :-1]).double().norm(dim=-1)[bonds]
        geometry = dict(bond_count=lengths.numel(), bond_min=None, bond_max=None,
                        bond_rms_error=None, geometry_checked=bool(lengths.numel()),
                        geometry_passed=None)
        if lengths.numel():
            geometry.update(bond_min=float(lengths.min().detach()),
                            bond_max=float(lengths.max().detach()),
                            bond_rms_error=float((lengths-3.8).square().mean().sqrt().detach()),
                            geometry_passed=bool(((lengths >= 2.5) & (lengths <= 5.)).all()))
        self.last_sample_diagnostics = dict(**flags, **bounds, **geometry,
            prediction_kind='time_mixture' if getattr(self, '_sample_time_mixture', False) else 'endpoint',
            observation_steps=times.detach().cpu().tolist())
        return ca.reshape(b, n_samples, l, 3), endpoint.reshape(b, n_samples, l, -1)

    def losses(self, batch, horizon, args, generator):
        self._sample_time_mixture = args.target == 'path_mixture'
        try:
            terms = super().losses(batch, horizon, args, generator)
            if getattr(args, 'score_recipe', 'original') == 'train_scaled_v2':
                terms['training_loss'] = terms['loss']
                terms['loss'] = terms['coord_es'] + .1*terms['latent_es'] + .01*terms['clash']
            return terms
        finally:
            self._sample_time_mixture = False


def training_score_scales(model, loaders, device):
    """Freeze endpoint displacement scales on every fixed training-subset window."""
    from evaluation.ensemble_metrics_ext import _align_to
    state = rng_state()
    was_training = model.training
    rows = []
    model.eval()
    try:
        with torch.no_grad():
            for horizon, loader in loaders.items():
                for index, raw in enumerate(loader):
                    batch = _move_batch(raw, device)
                    _validate_batch_clock(batch, horizon, model.kernel.cfg.step_ps)
                    mask = batch['residue_mask'].all(1)
                    # Scale estimation needs only the two endpoints, not all
                    # 64 intermediate encodings; preserve the full-clip mask.
                    endpoints = dict(batch)
                    endpoints['coords'] = batch['coords'][:, [3, -1]]
                    endpoints['residue_mask'] = mask[:, None].expand(-1, 2, -1)
                    h, _ = model.encode_clip(endpoints)
                    anchor = batch['coords'][0, 3, :, 1]
                    target = _align_to(batch['coords'][0, -1:, :, 1], anchor, mask[0])[0]
                    hs, xs = displacement_scales(h[0, 0], h[0, -1], anchor, target, mask[0])
                    rows.append(dict(horizon=horizon, index=index, **_identity(raw),
                                     latent=float(hs), coordinate=float(xs)))
                print(f'score scales: horizon={horizon} windows={len(loader)}', flush=True)
    finally:
        restore_rng(state)
        model.train(was_training)
        model.bundle.eval()
    means = {}
    for horizon in loaders:
        selected = [r for r in rows if r['horizon'] == horizon]
        means[str(horizon)] = {k: sum(r[k] for r in selected)/len(selected)
                               for k in ('latent', 'coordinate')}
    scales = {k: sum(r[k] for r in means.values())/len(means)
              for k in ('latent', 'coordinate')}
    if any(not math.isfinite(v) or v <= 1e-6 for v in scales.values()):
        raise ValueError('invalid aggregate training displacement scales')
    alpha = scales['coordinate']/scales['latent']
    if not math.isfinite(alpha) or alpha <= .1:
        raise ValueError('train-scaled recipe requires latent weight greater than 0.1')
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(**scales, latent_weight=alpha, by_horizon=means,
                window_count=len(rows), window_sha256=digest, per_window=rows,
                source='fixed training subsets only; equal windows within horizon, equal horizons')


def calibration_eligible(metrics, initial):
    rows, start = metrics['by_horizon'], initial['by_horizon']
    if (rows['64']['latent_es'] > .98*start['64']['latent_es']
            or rows['64']['loss'] > start['64']['loss']):
        return False
    return all(rows[h][key] <= 1.02*start[h][key]
               for h in ('1', '4', '16') for key in ('latent_es', 'coord_es'))


def calibration_regressed(metrics, initial):
    row, start = metrics['by_horizon']['64'], initial['by_horizon']['64']
    return row['latent_es'] > 1.05*start['latent_es'] or row['loss'] > 1.05*start['loss']


def gradient_audit_and_clip(kernel, max_norm=1.):
    """Element checks and robust float64 group/global norms before explicit clipping.

    Never square/reduce float32 gradients: their norm can overflow even when
    every element is finite. A float64 clipping scalar is cast only after its
    multiplication with the float64 gradient, avoiding float32 underflow.
    """
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('invalid clipping threshold')
    groups = {'energy': [], 'bath': [], 'other': []}
    flags = {}
    for name, parameter in kernel.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        flags[name] = bool(torch.isfinite(grad).all())
        groups['energy' if name.startswith('energy.') else
               'bath' if name.startswith('bath.') else 'other'].append(grad)
    finite = bool(flags) and all(flags.values())
    norms = {name: math.hypot(*(float(torch.linalg.vector_norm(g.double())) for g in grads))
             for name, grads in groups.items()}
    norm = math.hypot(*norms.values())
    report = dict(gradient_elements_finite=finite, gradient_parameter_flags=flags,
                  preclip_norm64=norm if math.isfinite(norm) else None,
                  preclip_group_norms64={k: v if math.isfinite(v) else None for k, v in norms.items()},
                  norm64_finite=math.isfinite(norm))
    if not finite or not math.isfinite(norm):
        return report
    scale = min(1., max_norm / max(norm, 1e-300))
    with torch.no_grad():
        for grads in groups.values():
            for grad in grads:
                grad.copy_((grad.double() * scale).to(grad.dtype))
    report['clip_scale64'] = scale
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', help='new bounded-edge component; requires its own numerical gate')
    p.add_argument('--init-checkpoint', required=True)
    p.add_argument('--numerical-gate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/train.jsonl')
    p.add_argument('--val-manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--val-every', type=int, default=250)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--microsteps', type=int, default=8)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--train-windows', type=int, default=1024)
    p.add_argument('--val-windows', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clip-norm', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--energy-seed', type=int, default=42)
    p.add_argument('--validation-seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--target', choices=['endpoint', 'path_mixture'], default='endpoint')
    p.add_argument('--score-recipe', choices=['original', 'train_scaled_v2'], default='original')
    p.add_argument('--paired-reference', help='original endpoint metrics.jsonl for step0/data pairing')
    # Mixture changes both generated and reference observation times; no truncation.
    p.set_defaults(grad_steps=None, latent_weight=.1, coord_weight=1.,
                   contact_weight=0., rmsf_weight=0., distance_weight=0., clash_weight=.01,
                   mode_weight=0., mode_distribution_weight=0., contact_objective='mae',
                   stateless_data=True, history_conditioning=False)
    return p


def _identity(batch):
    return dict(protein=batch['protein_id'], trajectory=batch['trajectory_id'],
                temperature=batch['temperature_k'].tolist(),
                time_ps=batch['time_ps'].tolist())


def _validate_batch_clock(batch, horizon, step_ps):
    dt = batch['time_ps'].double().diff(dim=1)
    if (batch['coords'].shape[1] != 4+horizon or not torch.isfinite(dt).all()
            or not torch.allclose(dt, torch.full_like(dt, step_ps), atol=1e-3, rtol=1e-6)):
        raise ValueError('batch does not match the unchanged physical macro clock')
    temperature = batch['temperature_k']
    if not bool((torch.isfinite(temperature) & (temperature > 0)).all()):
        raise ValueError('invalid batch temperature')


def main(argv=None):
    a = parser().parse_args(argv)
    if (not 1 <= a.steps <= 2000 or not 1 <= a.val_every <= 250
            or a.n_samples < 2 or not a.horizons or len(set(a.horizons)) != len(a.horizons)
            or not set(a.horizons) <= {1, 4, 16, 64}
            or a.train_windows < 0 or a.val_windows < 1 or a.max_residues < 4
            or not math.isfinite(a.lr) or a.lr <= 0
            or not math.isfinite(a.clip_norm) or a.clip_norm <= 0):
        raise ValueError('invalid development budget or optimization configuration')
    if a.score_recipe == 'train_scaled_v2' and (
            a.target != 'endpoint' or a.horizons != [1, 4, 16, 64]
            or a.n_samples != 12 or a.seed != 42 or a.energy_seed != 42
            or a.validation_seed != 123 or a.train_windows != 1024 or a.val_windows != 4
            or a.max_residues != 96 or (a.steps, a.val_every) not in ((4, 4), (2000, 250))
            or a.lr != 1e-4 or a.clip_norm != 1. or not a.paired_reference):
        raise ValueError('train-scaled v2 requires the frozen paired seed42 endpoint protocol')
    gate = verify_numerical_gate(a.numerical_gate, a.init_checkpoint, a.val_manifest,
                                 a.horizons, a.microsteps, a.component)
    audit = validate_manifest_split(a.manifest, a.val_manifest)
    torch.set_num_threads(4)
    _set_reproducibility(a.seed, False)
    source = torch.load(a.init_checkpoint, map_location='cpu', weights_only=False)
    if (source.get('checkpoint_kind') != 'lasp_ca_dynamics'
            or source.get('component_checkpoint', {}).get('checkpoint_kind') != 'ca_component_probe'):
        raise ValueError('requires stopped-best dynamics with embedded CA component')
    if a.component is not None:
        replacement = torch.load(a.component, map_location='cpu', weights_only=False)
        mode = replacement.get('decoder_mode')
        projection = replacement.get('calibration_args', {}).get('bond_projection_steps')
        if (replacement.get('checkpoint_kind') != 'ca_component_probe'
                or not ((mode == 'bounded_edge_ca' and projection == 0)
                        or (mode == 'geometric_ca' and projection == 512))):
            raise ValueError('component override requires an audited bounded-edge or projection-aware representation')
        source['component_checkpoint'] = replacement
    out = Path(a.output)
    # New output only: no ambiguous partial restarts or overwritten components.
    out.mkdir(parents=True, exist_ok=False)
    component_path = out / 'component.pt'
    torch.save(source['component_checkpoint'], component_path)
    model = BoundedEndpointDynamics(str(component_path), a.device, a.init_checkpoint,
                                    a.energy_seed, a.microsteps).to(a.device)
    opt = torch.optim.AdamW([p for p in model.kernel.parameters() if p.requires_grad],
                           lr=a.lr, weight_decay=.01)
    generator = torch.Generator(device=a.device).manual_seed(a.seed)
    train = {h: make_loader(a, a.manifest, h, True) for h in a.horizons}
    val = {h: make_loader(a, a.val_manifest, h, False) for h in a.horizons}
    if any(len(loader.dataset) == 0 for loader in [*train.values(), *val.values()]):
        raise ValueError('empty training or validation windows')
    scales = None
    paired_rows = None
    if a.score_recipe == 'train_scaled_v2':
        reference_run = json.loads((Path(a.paired_reference).parent / 'run.json').read_text())
        for key in ('train_manifest_sha256', 'val_manifest_sha256'):
            if audit[key] != reference_run['split_audit'][key]:
                raise ValueError(f'paired manifest mismatch: {key}')
        paired_rows = [json.loads(line) for line in Path(a.paired_reference).read_text().splitlines() if line.strip()]
        scales = training_score_scales(model, train, a.device)
        a.latent_weight = scales['latent_weight']
        (out / 'score_scales.json').write_text(json.dumps(scales, indent=2, allow_nan=False))
    execution_scope = 'smoke' if a.steps == 4 and a.score_recipe == 'train_scaled_v2' else 'development_pilot'
    metadata = dict(execution_scope=execution_scope, score_scales=scales, args=vars(a), split_audit=audit, numerical_gate=gate,
                    source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (Path(__file__).resolve(),
                                                REPO / 'protein_jepa_frame/training/train_lasp_ca_dynamics.py')},
                    paired_reference_sha256=(hashlib.sha256(Path(a.paired_reference).read_bytes()).hexdigest()
                                             if a.score_recipe == 'train_scaled_v2' else None),
                    checkpoint_kind=CHECKPOINT_KIND, schema_version=SCHEMA_VERSION,
                    energy_config=model.energy_config, integrator=model.integrator_metadata(),
                    screening_only=True, full_promotion=False, test_opened=False)
    (out / 'run.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))

    def record(row):
        text = json.dumps(row, allow_nan=False)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(text + '\n')
        print(text, flush=True)

    def evaluate_terms(batch, horizon):
        _validate_batch_clock(batch, horizon, model.kernel.cfg.step_ps)
        objective_args = argparse.Namespace(**vars(a))
        if not model.training:
            objective_args.target = 'endpoint'
        terms = model.losses(_move_batch(batch, a.device), horizon, objective_args, generator)
        return terms, {k: bool(torch.isfinite(v).all()) for k, v in terms.items()}

    def validate():
        model.eval()
        rows = {}
        # Restore training generator even on failure; fixed validation noise and
        # deterministic window indices make all checkpoints directly comparable.
        state = generator.get_state()
        generator.manual_seed(a.validation_seed)
        try:
            with torch.no_grad():
                for horizon, loader in val.items():
                    values = []
                    for index, batch in enumerate(loader):
                        terms, flags = evaluate_terms(batch, horizon)
                        geometry = dict(model.last_sample_diagnostics)
                        row = dict(phase='validation_window', horizon=horizon, window=index,
                                   **_identity(batch), terms={k: float(v) for k, v in terms.items()},
                                   term_elements_finite=flags, sample=geometry)
                        record(row)
                        if not all(flags.values()) or geometry.get('geometry_passed') is False:
                            raise FloatingPointError('nonfinite or invalid CA geometry in validation')
                        values.append(row['terms'])
                    rows[str(horizon)] = {k: sum(v[k] for v in values)/len(values) for k in values[0]}
        finally:
            generator.set_state(state)
        return dict(by_horizon=rows, loss=sum(r['loss'] for r in rows.values())/len(rows),
                    geometry_checked=True, test_opened=False)

    best = math.inf

    def save(tag, step, metrics):
        payload = dict(checkpoint_kind=CHECKPOINT_KIND, schema_version=SCHEMA_VERSION,
                       recipe=a.score_recipe, score_scales_sha256=(
                           hashlib.sha256((out / 'score_scales.json').read_bytes()).hexdigest()
                           if scales is not None else None),
                       run_metadata=metadata,
                       kernel=model.kernel.state_dict(), kernel_config=asdict(model.kernel.cfg),
                       energy_config=model.energy_config, energy_seed=a.energy_seed,
                       integrator=model.integrator_metadata(), component_checkpoint=source['component_checkpoint'],
                       args=vars(a), step=step, validation=metrics, best_loss=best,
                       optimizer=opt.state_dict(), rng_state=rng_state(),
                       sampling_rng=generator.get_state(), numerical_gate=gate,
                       split_audit=audit, test_opened=False, full_promotion=False)
        temporary = out / f'checkpoint_{tag}.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / f'checkpoint_{tag}.pt')

    metrics = validate()
    initial = metrics
    calibrated = a.score_recipe == 'train_scaled_v2'
    reference_train = {}
    if calibrated:
        reference = next(r['val'] for r in paired_rows if r.get('step') == 0 and 'val' in r)
        for horizon, row in reference['by_horizon'].items():
            for key in ('loss', 'latent_es', 'coord_es'):
                if abs(metrics['by_horizon'][horizon][key] - row[key]) > 1e-6:
                    raise ValueError(f'initial paired validation mismatch: {horizon}/{key}')
        reference_train = {r['step']: r for r in paired_rows if r.get('phase') == 'train'}
    best = math.inf if calibrated else metrics['loss']
    eligible_seen = False
    regression_count = 0
    record(dict(step=0, val=metrics, paired_initial_verified=calibrated))
    if not calibrated:
        save('best', 0, metrics)
    save('latest', 0, metrics)
    for step in range(1, a.steps+1):
        started = time.perf_counter()
        horizon = a.horizons[(step-1) % len(a.horizons)]
        batch = step_batch(train[horizon], step, a.seed)
        row = dict(step=step, phase='train', horizon=horizon, **_identity(batch))
        if calibrated:
            reference = reference_train.get(step)
            if reference is None or any(row[key] != reference[key] for key in
                                         ('horizon', 'protein', 'trajectory', 'temperature', 'time_ps')):
                raise ValueError(f'paired training identity mismatch at step {step}')
            row['paired_identity_verified'] = True
        try:
            model.train()
            model.bundle.eval()
            opt.zero_grad(set_to_none=True)
            terms, flags = evaluate_terms(batch, horizon)
            row.update(terms={k: float(v.detach()) for k, v in terms.items()},
                       term_elements_finite=flags, sample=dict(model.last_sample_diagnostics))
            if not all(flags.values()):
                raise FloatingPointError('nonfinite training terms')
            if (calibrated or a.component is not None) and model.last_sample_diagnostics.get('geometry_passed') is not True:
                raise FloatingPointError('invalid training CA geometry')
            terms.get('training_loss', terms['loss']).backward()
            grad = gradient_audit_and_clip(model.kernel, a.clip_norm)
            row.update(grad)
            if not grad['gradient_elements_finite'] or not grad['norm64_finite']:
                raise FloatingPointError('nonfinite gradient elements or float64 norm')
            opt.step()
            row['parameter_elements_finite'] = all(bool(torch.isfinite(p).all()) for p in model.kernel.parameters())
            if not row['parameter_elements_finite']:
                raise FloatingPointError('nonfinite updated parameters')
        except Exception as error:
            # Invalid values remain represented by explicit flags, not invalid JSON.
            if 'terms' in row:
                row['terms'] = {k: v if math.isfinite(v) else None for k, v in row['terms'].items()}
            row.update(error=f'{type(error).__name__}: {error}', seconds=time.perf_counter()-started)
            record(row)
            # This snapshot may contain invalid post-update parameters; never resume it.
            save('failure', step if 'parameter_elements_finite' in row else step-1, metrics)
            status = dict(status='rejected', reason=row['error'], attempted_step=step,
                          last_successful_step=step-1, failed_phase='train',
                          optimizer_step_returned='parameter_elements_finite' in row,
                          diagnostic_checkpoint='checkpoint_failure.pt',
                          test_opened=False, quality_promotion=False)
            temporary = out / 'status.tmp'
            temporary.write_text(json.dumps(status, indent=2, allow_nan=False))
            temporary.replace(out / 'status.json')
            record(status)
            raise
        if str(a.device).startswith('cuda'):
            torch.cuda.synchronize(a.device)
        row['seconds'] = time.perf_counter()-started
        record(row)
        if step % a.val_every == 0 or step == a.steps:
            metrics = validate()
            eligible = not calibrated or calibration_eligible(metrics, initial)
            eligible_seen = eligible_seen or eligible
            regression_count = (regression_count + 1 if calibrated and calibration_regressed(metrics, initial)
                                else 0)
            record(dict(step=step, val=metrics, eligible=eligible,
                        consecutive_h64_regressions=regression_count))
            if eligible and metrics['loss'] < best:
                best = metrics['loss']
                save('best', step, metrics)
            save('latest', step, metrics)
            reason = ('h64_regression' if regression_count >= 2 else
                      'futility_no_eligible_checkpoint' if calibrated and step >= 1000 and not eligible_seen else None)
            if reason:
                status = dict(status='rejected', reason=reason, step=step,
                              test_opened=False, quality_promotion=False)
                (out / 'status.json').write_text(json.dumps(status, indent=2))
                record(status)
                break
    else:
        status = dict(status=('smoke_passed' if execution_scope == 'smoke' else
                              'completed' if not calibrated or eligible_seen else 'rejected'),
                      execution_scope=execution_scope, step=a.steps,
                      eligible_checkpoint=eligible_seen, test_opened=False,
                      quality_promotion=False)
        (out / 'status.json').write_text(json.dumps(status, indent=2))
        record(status)


if __name__ == '__main__':
    main()
