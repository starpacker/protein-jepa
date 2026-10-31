"""No-update development gate for the isolated bounded-ridge energy prototype."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from protein_jepa_frame.models.lasp_bounded_energy import BoundedRidgeEnergy
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from scripts.diagnose_lasp_microsteps import _jsonable, _panel_windows, _run_window


def geometry_summary(ca, mask, ri, ci):
    """Locate the worst valid adjacent bond, retaining sample and residue IDs."""
    valid = mask[:, 1:] & mask[:, :-1] & (ci[:, 1:] == ci[:, :-1]) & (ri[:, 1:] - ri[:, :-1] == 1)
    lengths = (ca[:, 1:] - ca[:, :-1]).double().norm(dim=-1)
    if not bool(valid.any()):
        raise ValueError('geometry diagnostic requires adjacent residues')
    selected = lengths[valid]
    worst = lengths.masked_fill(~valid, -float('inf')).flatten().argmax()
    sample, edge = divmod(int(worst), lengths.shape[1])
    return dict(finite=bool(torch.isfinite(ca).all()), bond_min=float(selected.min()),
                bond_max=float(selected.max()),
                invalid_bonds=int(((selected < 2.5) | (selected > 5.)).sum()),
                bond_count=selected.numel(), worst_sample=sample,
                worst_residue_indices=ri[sample, edge:edge+2].tolist())


def run_geometry_screen(run_path, output_path, seed):
    """Screen the recorded failed training window; never reconstruct failed weights."""
    from argparse import Namespace
    from protein_jepa_frame.training.train_lasp_bounded_endpoint import (
        BoundedEndpointDynamics, load_bounded_checkpoint, _identity, _validate_batch_clock,
    )
    from protein_jepa_frame.training.train_lasp_ca_dynamics import make_loader
    from protein_jepa_frame.training.lasp_run_state import step_batch
    from protein_jepa_frame.training.train_utils import _move_batch
    from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
    from scripts.diag_lasp_ca_ceiling import _same_component

    run = Path(run_path)
    metadata = json.loads((run / 'run.json').read_text())
    args = Namespace(**metadata['args'])
    approved = Path('artifacts/mdcath_manifest_full/train.jsonl').resolve()
    if Path(args.manifest).resolve() != approved:
        raise ValueError('geometry screen requires approved training manifest')
    failed = next(row for line in (run / 'metrics.jsonl').read_text().splitlines()
                  if (row := json.loads(line)).get('error'))
    torch.set_num_threads(4)
    device = torch.device(args.device)
    loader = make_loader(args, str(approved), failed['horizon'], True)
    batch = step_batch(loader, failed['step'], args.seed)
    identity = _identity(batch)
    if any(identity[key] != failed[key] for key in identity):
        raise ValueError('failed training window identity mismatch')
    batch = _move_batch(batch, device)
    component = run / 'component.pt'
    models = dict(fresh=BoundedEndpointDynamics(
        str(component), device, args.init_checkpoint, args.energy_seed, args.microsteps).to(device))
    reference = Path(args.paired_reference).parent / 'checkpoint_best.pt'
    models['endpoint_best'], source = load_bounded_checkpoint(reference, component, device)
    if not _same_component(source['component_checkpoint'], torch.load(component, map_location='cpu', weights_only=False)):
        raise ValueError('reference component mismatch')
    model = models['fresh']
    _validate_batch_clock(batch, failed['horizon'], model.kernel.cfg.step_ps)
    mask = batch['residue_mask'].all(1)
    aa, ri, ci = (batch[key] for key in ('aatype', 'residue_index', 'chain_id'))
    anchor_bb = batch['coords'][:, 3]
    rotation, anchor = _extract_backbone_frames(anchor_bb)
    result = dict(scope='failed training window geometry screen; not exact failed-state replay',
                  optimizer_steps=0, test_opened=False, failed_step=failed['step'],
                  identity=identity, sampling_seed=seed, recorded_failure=failed['sample'],
                  references={}, models={})
    with torch.no_grad():
        ah = model.bundle.encode(anchor_bb, mask, aa, ri, ci)
        th = model.bundle.encode(batch['coords'][:, -1], mask, aa, ri, ci)
        for name, coords in [('observed', anchor), ('target', batch['coords'][:, -1, :, 1])]:
            result['references'][name] = geometry_summary(coords, mask, ri, ci)
        for name, latent in [('anchor_reconstruction', ah), ('oracle_endpoint_reconstruction', th)]:
            raw = model.bundle.decoder(latent, ah, aa, ri, ci, rotation, anchor, mask)
            projected = model.bundle.decode_ca(latent, ah, aa, ri, ci, rotation, anchor, mask)
            result['references'][name] = dict(raw=geometry_summary(raw, mask, ri, ci),
                                              projected=geometry_summary(projected, mask, ri, ci))
        for name, model in models.items():
            model.eval()
            rng = torch.Generator(device=device).manual_seed(seed)
            ca, latent = model.sample(batch['coords'][:, :4], mask, aa, ri, ci,
                                      batch['temperature_k'], failed['horizon'], args.n_samples, rng)
            n = args.n_samples
            repeat = lambda x: x.expand(n, *x.shape[1:])
            raw = model.bundle.decoder(latent[0], repeat(ah), repeat(aa), repeat(ri),
                                       repeat(ci), repeat(rotation), repeat(anchor), repeat(mask))
            from protein_jepa_frame.models.ca_geometry import project_ca_bonds
            projection = {}
            default_steps = model.bundle.projection_steps
            for count in sorted(set([default_steps, max(1, default_steps)*2,
                                     max(1, default_steps)*4, max(1, default_steps)*8])):
                refined = project_ca_bonds(raw, repeat(anchor), repeat(mask),
                                          repeat(ri), repeat(ci), count)
                correction = (refined - ca[0])[repeat(mask)]
                projection[str(count)] = dict(
                    geometry=geometry_summary(refined, repeat(mask), repeat(ri), repeat(ci)),
                    rms_displacement_from_default=float(correction.double().square().sum(-1).mean().sqrt()))
            result['models'][name] = dict(
                raw=geometry_summary(raw, repeat(mask), repeat(ri), repeat(ci)),
                projected=geometry_summary(ca[0], repeat(mask), repeat(ri), repeat(ci)),
                default_projection_steps=default_steps, projection_refinement=projection,
                diagnostics=model.last_sample_diagnostics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


def bond_length_probe(anchor, future, mask, ri, ci):
    """Oracle representation diagnostic, never a forecast or capacity bound."""
    from protein_jepa_frame.models.ca_geometry import retract_ca_bond_lengths
    valid = (mask[:, 1:] & mask[:, :-1] & (ci[:, 1:] == ci[:, :-1])
             & (ri[:, 1:]-ri[:, :-1] == 1))
    if not bool(valid.any()):
        raise ValueError('bond probe requires adjacent residues')
    leaf = future.detach().double().requires_grad_(True)
    anchor = anchor.double()
    reconstructed = retract_ca_bond_lengths(leaf, anchor, mask, ri, ci)
    change = (reconstructed-leaf)[mask]
    mismatch = (leaf.diff(dim=1).norm(dim=-1)-anchor.diff(dim=1).norm(dim=-1))[valid]
    # Deterministic probes; sampled VJP gains are not a Jacobian norm bound.
    generator = torch.Generator(device=leaf.device).manual_seed(26)
    gains = []
    for _ in range(4):
        probe = torch.randn(leaf.shape, dtype=leaf.dtype, device=leaf.device, generator=generator)
        probe = probe*mask[..., None]
        probe = probe/probe.norm()
        gradient, = torch.autograd.grad((reconstructed*probe).sum(), leaf, retain_graph=True)
        gains.append(float(gradient.norm()))
    from evaluation.ensemble_metrics_ext import _align_to
    aligned_future = _align_to(leaf.detach(), anchor[0], mask[0])
    motion_rms = float((aligned_future-anchor)[mask].square().sum(-1).mean().sqrt())
    return dict(residues=int(mask.sum()), valid_bonds=int(valid.sum()),
                aligned_true_motion_rms=motion_rms,
                bond_mismatch_rms=float(mismatch.detach().square().mean().sqrt()),
                bond_mismatch_max=float(mismatch.detach().abs().max()),
                oracle_coordinate_rms_shift=float(change.detach().square().sum(-1).mean().sqrt()),
                max_sampled_vjp_gain=max(gains),
                finite=bool(torch.isfinite(reconstructed).all()) and all(math.isfinite(g) for g in gains))


def run_bond_representation_probe(manifest, output_path):
    from evaluation.data import EvalProtocol, iter_windows
    approved = {Path('artifacts/mdcath_manifest_full/train.jsonl').resolve(),
                Path('artifacts/mdcath_manifest_full/val.jsonl').resolve()}
    if Path(manifest).resolve() not in approved:
        raise ValueError('bond probe requires approved train/development manifest')
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    protocol = EvalProtocol(manifest=str(Path(manifest).resolve()), horizons_frames=(1, 4, 16, 64),
                            context_frames=8, frame_stride=1, max_residues=96,
                            max_anchors_per_trajectory=4, seed=0,
                            round_robin_proteins=True, round_robin_conditions=True)
    counts, rows = {}, []
    for w in iter_windows(protocol):
        key = (float(w.temperature_k), int(w.horizon))
        if counts.get(key, 0) >= 2:
            continue
        mask = torch.as_tensor(w.context['residue_mask']).all(0).bool()[None]
        future_mask = torch.as_tensor(w.future_path['residue_mask']).bool()
        if int(mask.sum()) < 4 or not bool(future_mask[:, mask[0]].all()):
            continue
        anchor = torch.as_tensor(w.context['coords'][-1, :, 1])[None]
        future = torch.as_tensor(w.future_path['coords'][-1, :, 1])[None]
        ri, ci = [torch.as_tensor(w.context[k]).long()[None] for k in ('residue_index', 'chain_id')]
        row = bond_length_probe(anchor, future, mask, ri, ci)
        if Path(manifest).resolve().name == 'train.jsonl' and bool(mask[:, :96].all()) and mask.shape[1] >= 96:
            row['prefix_sensitivity'] = {
                str(length): bond_length_probe(anchor[:, :length], future[:, :length],
                                              mask[:, :length], ri[:, :length], ci[:, :length])
                for length in (16, 32, 64, 96)
            }
        row.update(protein=w.protein_id, trajectory=w.trajectory_id, anchor=int(w.anchor_frame),
                   horizon=int(w.horizon), temperature=float(w.temperature_k))
        rows.append(row)
        counts[key] = counts.get(key, 0)+1
        if len(rows) == 40:
            break
    if len(rows) != 40 or len(counts) != 20:
        raise ValueError('incomplete balanced bond panel')
    result = dict(scope='oracle fixed-anchor-length approximation; not forecast evidence',
                  manifest=str(Path(manifest).resolve()), optimizer_steps=0, test_opened=False,
                  protocol=dict(windows=40, per_temperature_horizon=2, max_residues=96,
                                vjp_probes=4, dtype='float64', seed=26), windows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('bond representation probe completed', len(rows), flush=True)


def retraction_acceptance(rows, once, twice):
    """Apply the frozen preservation gates to once-retracted coordinates."""
    groups = {'overall': rows}
    groups.update({str(h): [r for r in rows if r['horizon'] == h] for h in (1, 4, 16, 64)})
    if any(not group for group in groups.values()):
        raise ValueError('retraction acceptance requires all four horizons')
    deltas = {
        name: {metric: sum(r['modes'][once][metric]-r['modes']['32'][metric]
                           for r in group)/len(group)
               for metric in ('ca_angle_cos_mae', 'ca_clash_fraction')}
        for name, group in groups.items()
    }
    checks = dict(
        bonds=all(r['modes'][once]['bonds']['invalid_bonds'] == 0 for r in rows),
        finite=all(m['gradient_finite'] and m['bonds']['finite']
                   for r in rows for m in r['modes'].values()),
        endpoint_score=all(r['modes'][once]['coordinate_endpoint_es']
                           <= 1.02*r['modes']['32']['coordinate_endpoint_es'] for r in rows),
        nonlocal_clashes=all(r['modes'][once]['clash_fraction']
                            <= r['modes']['32']['clash_fraction']+.0001 for r in rows),
        angles=all(d['ca_angle_cos_mae'] <= .0001 for d in deltas.values()),
        calibration_clashes=all(d['ca_clash_fraction'] <= .0001 for d in deltas.values()),
        idempotence=all(r['rms_finest_pair'] <= .01 for r in rows),
    )
    return dict(passed=all(checks.values()), checks=checks, geometry_deltas=deltas,
                candidate_mode=once, idempotence_mode=twice,
                scope='decoder preservation only; full BPTT required before training')


def run_projection_panel(checkpoint, component, output_path, iterations, standard200=False, retraction=False):
    """Fixed-latent development screen; never changes the deployed decoder."""
    from collections import Counter
    from evaluation.data import EvalProtocol, iter_windows
    from evaluation.ensemble_metrics_ext import _align_to, extended_ensemble_summary
    from evaluation.geometry import ca_geometry_summary
    from protein_jepa_frame.models.ca_geometry import project_ca_bonds
    from protein_jepa_frame.training.train_lasp_bounded_endpoint import load_bounded_checkpoint
    from protein_jepa_frame.training.train_lasp_ca_dynamics import energy_score
    from protein_jepa_frame.training.train_lasp_calibration import calibration_terms
    from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
    from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed, _same_component
    from scripts.eval_lasp_final import METRICS

    if (iterations != sorted(set(iterations)) or len(iterations) < 2
            or iterations[0] != 32):
        raise ValueError('projection iterations must be sorted unique, starting at32')
    if retraction and len(iterations) != 3:
        raise ValueError('retraction requires three labels: baseline, once, twice')
    mode_definitions = {
        str(count): (dict(projection_iterations=32, retractions=index)
                     if retraction else dict(projection_iterations=count, retractions=0))
        for index, count in enumerate(iterations)
    }
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    manifest = _development_manifest('artifacts/mdcath_manifest_full/val.jsonl')
    protocol = EvalProtocol(manifest=manifest, horizons_frames=(1, 4, 16, 64),
                            context_frames=8, frame_stride=1, max_residues=96,
                            max_anchors_per_trajectory=4, seed=0,
                            round_robin_proteins=True, round_robin_conditions=True)
    counts, windows = Counter(), []
    budget = 200 if standard200 else 40
    for window in iter_windows(protocol):
        key = (float(window.temperature_k), int(window.horizon))
        if (not standard200 and counts[key] >= 2) or int(window.context['residue_mask'].all(0).sum()) < 4:
            continue
        counts[key] += 1
        windows.append(window)
        if len(windows) == budget:
            break
    if len(windows) != budget or (not standard200 and len(counts) != 20):
        raise ValueError('incomplete projection panel')
    if standard200:
        reference = json.loads(Path('results/lasp_bounded_v1/latent_transition_probe.json').read_text())
        expected = [(r['protein'], r['trajectory'], r['anchor_frame'], r['horizon'])
                    for r in reference['per_window']]
        actual = [(w.protein_id, w.trajectory_id, int(w.anchor_frame), int(w.horizon)) for w in windows]
        if actual != expected:
            raise ValueError('standard200 identity mismatch')
    result = dict(scope='development projection screening only; no training or promotion',
                  checkpoint=str(checkpoint), test_opened=False, optimizer_steps=0,
                  protocol=dict(manifest=manifest, n_windows=budget, n_samples=12,
                                panel='standard200' if standard200 else 'balanced40',
                                iterations=iterations, mode_definitions=mode_definitions,
                                retraction=retraction, evaluator_seed=0,
                                shared_generated_latents=True, cpu_threads=4),
                  acceptance=dict(bond_range=[2.5, 5.], finite_gradients_required=True,
                                  max_coordinate_es_relative_increase=.02,
                                  max_clash_fraction_increase=.0001,
                                  max_finest_pair_rms_angstrom=.05,
                                  scope='screening only; full BPTT and standard200 required before training'),
                  windows=[], status='running')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False))
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    model, source = load_bounded_checkpoint(checkpoint, component, device)
    if not _same_component(source['component_checkpoint'], torch.load(component, map_location='cpu', weights_only=False)):
        raise ValueError('projection panel component mismatch')
    model.eval()
    for window in windows:
        context = torch.as_tensor(window.context['coords'], device=device)
        future = torch.as_tensor(window.future_path['coords'], device=device)
        mask = torch.as_tensor(window.context['residue_mask'], device=device).all(0).bool()[None]
        future_mask = torch.as_tensor(window.future_path['residue_mask'], device=device).bool()
        if not bool(future_mask[:, mask[0]].all()):
            raise ValueError('future mask differs on evaluated residues')
        aa, ri, ci = [torch.as_tensor(window.context[k], device=device).long()[None]
                      for k in ('aatype', 'residue_index', 'chain_id')]
        repeat = lambda x: x.expand(12, *x.shape[1:])
        with torch.no_grad():
            ah = model.bundle.encode(context[-1:], mask, aa, ri, ci)
            rotation, anchor = _extract_backbone_frames(context[-1:])
            rng = torch.Generator(device=device).manual_seed(_window_seed(0, window))
            _, latent = model.sample(context[None], mask, aa, ri, ci,
                                     torch.tensor([float(window.temperature_k)], device=device),
                                     int(window.horizon), 12, rng)
            raw = model.bundle.decoder(latent[0], repeat(ah), repeat(aa), repeat(ri),
                                       repeat(ci), repeat(rotation), repeat(anchor), repeat(mask))
            target = _align_to(future[-1:, :, 1], anchor[0], mask[0])[:, mask[0]].flatten(1)
        modes, coordinates = {}, {}
        for count in iterations:
            leaf = raw.detach().requires_grad_(True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            base_count = 32 if retraction else count
            ca = project_ca_bonds(leaf, repeat(anchor), repeat(mask), repeat(ri), repeat(ci), base_count)
            if retraction and count != 32:
                from protein_jepa_frame.models.ca_geometry import retract_ca_bond_lengths
                ca = retract_ca_bond_lengths(ca, repeat(anchor), repeat(mask), repeat(ri), repeat(ci))
                if count == iterations[-1]:
                    ca = retract_ca_bond_lengths(ca, repeat(anchor), repeat(mask), repeat(ri), repeat(ci))
            score = energy_score(ca[:, mask[0]].flatten(1), target)
            gradient, = torch.autograd.grad(score, leaf)
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            detached = ca.detach()
            coordinates[count] = detached
            with torch.no_grad():
                decoded = extended_ensemble_summary(detached[:, mask[0]], future[:, mask[0], 1], None,
                                                    cutoff=8., pca_k=3)
                geometry = [ca_geometry_summary(sample, mask[0], chain_id=ci[0], residue_index=ri[0])
                            for sample in detached]
                aligned_target = _align_to(future[-1:, :, 1], anchor[0], mask[0])
                validity = calibration_terms(detached[None], aligned_target[None].expand(1, 12, -1, -1),
                                             mask, ri, ci)
                modes[str(count)] = dict(
                    ca_angle_cos_mae=float(validity['ca_angle_cos_mae']),
                    ca_clash_fraction=float(validity['ca_clash_fraction']),
                    bonds=geometry_summary(detached, repeat(mask), repeat(ri), repeat(ci)),
                    coordinate_endpoint_es=float(score.detach()),
                    gradient_finite=bool(torch.isfinite(gradient).all()),
                    raw_coordinate_vjp_norm64=float(gradient.double().norm()),
                    projection_and_vjp_seconds=seconds,
                    clash_fraction=sum(g['ca_nonlocal_clash_fraction'] for g in geometry)/12,
                    decoded={k: float(decoded[k]) for k in METRICS})
        refinement = float((coordinates[iterations[-2]]-coordinates[iterations[-1]])[:, mask[0]].double().square().sum(-1).mean().sqrt())
        row = dict(protein=window.protein_id, trajectory=window.trajectory_id,
                   anchor=int(window.anchor_frame), temperature=float(window.temperature_k),
                   horizon=int(window.horizon), modes=modes, rms_finest_pair=refinement)
        result['windows'].append(row)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(output)
        print(f'projection panel {len(result["windows"])}/{budget}', flush=True)
    result['status'] = 'completed'
    result['screening_passed'] = all(
        r['rms_finest_pair'] <= .05
        and r['modes'][str(iterations[-1])]['bonds']['invalid_bonds'] == 0
        and all(m['gradient_finite'] and m['bonds']['finite'] for m in r['modes'].values())
        and r['modes'][str(iterations[-1])]['coordinate_endpoint_es'] <= 1.02*r['modes']['32']['coordinate_endpoint_es']
        and r['modes'][str(iterations[-1])]['clash_fraction'] <= r['modes']['32']['clash_fraction']+.0001
        for r in result['windows'])
    if retraction:
        result['retraction_acceptance'] = retraction_acceptance(
            result['windows'], str(iterations[1]), str(iterations[2]))
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(output)
    print('projection_screening_passed', result['screening_passed'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--bond-representation-probe', action='store_true')
    parser.add_argument('--projection-panel', action='store_true')
    parser.add_argument('--standard200', action='store_true')
    parser.add_argument('--retraction', action='store_true')
    parser.add_argument('--projection-iterations', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--component')
    parser.add_argument('--geometry-run', help='screen the first failed window in an existing training run')
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--microsteps', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--temperatures', type=float, nargs='+', default=[348., 413.])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bond_representation_probe:
        if args.projection_panel or args.geometry_run or args.checkpoint:
            parser.error('bond representation probe excludes model/projection modes')
        run_bond_representation_probe(args.manifest, args.out)
        return
    if args.projection_panel:
        if not args.checkpoint or not args.component or args.geometry_run:
            parser.error('projection panel requires checkpoint/component and excludes geometry-run')
        run_projection_panel(args.checkpoint, args.component, args.out, args.projection_iterations, args.standard200, args.retraction)
        return
    if args.geometry_run:
        run_geometry_screen(args.geometry_run, args.out, args.seed)
        return
    if not args.checkpoint:
        parser.error('--checkpoint is required without --geometry-run')
    if args.microsteps != sorted(set(args.microsteps)) or args.microsteps[0] < 1:
        raise ValueError('microsteps must be sorted unique positive counts')
    if any(max(args.microsteps) % n for n in args.microsteps):
        raise ValueError('all microstep counts must divide the finest grid')
    if not args.horizons or min(args.horizons) < 1:
        raise ValueError('horizons must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    component = Path(args.component) if args.component else output.parent / 'bounded_gate_component.pt'
    if not args.component:
        if component.exists():
            raise FileExistsError(component)
        torch.save(checkpoint['component_checkpoint'], component)
    old = checkpoint['args']
    if checkpoint.get('checkpoint_kind') == 'lasp_bounded_endpoint_v1':
        from protein_jepa_frame.training.train_lasp_bounded_endpoint import load_bounded_checkpoint
        model, _ = load_bounded_checkpoint(args.checkpoint, component, device)
        model.eval()
        scope = 'trained bounded energy numerical audit, no quality claim'
    elif checkpoint.get('checkpoint_kind') == 'lasp_ca_dynamics':
        model = CADynamics(str(component), device, args.checkpoint,
                           history_conditioning=False, noise_mix=old.get('noise_mix', 0.),
                           noise_length=old.get('noise_length', 8.), collective_mobility=True).to(device).eval()
        model.kernel.energy = BoundedRidgeEnergy(model.kernel.cfg.latent_dim).to(device)
        scope = 'fresh energy with stopped-best bath, no training or quality claim'
    else:
        raise ValueError('unsupported checkpoint kind')
    panel = _panel_windows(args.manifest, args.temperatures, args.horizons, 96, 2, 2)
    result = {'scope': scope,
              'protocol': vars(args), 'optimizer_steps': 0, 'test_opened': False,
              'acceptance': {'max_joint_gain': 10., 'max_endpoint_rms_8_to_16': 0.05,
                             'finite_required': True}, 'windows': []}
    if args.component:
        result['component_sha256'] = hashlib.sha256(component.read_bytes()).hexdigest()
        result['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        result['manifest_sha256'] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    started = time.perf_counter()
    for index, window in enumerate(panel):
        modes = _run_window(model, window, 2, args.microsteps, args.seed + index,
                            device, type('Args', (), old)(), 'coupled_midpoint_v1')
        row = {'protein': window.protein_id, 'temperature': float(window.temperature_k),
               'horizon': int(window.horizon), 'trajectory': window.trajectory_id,
               'anchor_frame': int(window.anchor_frame), 'modes': _jsonable(modes)}
        result['windows'].append(row)
        result['seconds'] = time.perf_counter() - started
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        print(f'window={index+1}/{len(panel)} protein={window.protein_id} h={window.horizon}', flush=True)
    result['screening_passed'] = all(
        r['gradient_elements_finite'] and r['forward_finite'] and r['joint_backward_gain'] <= 10.
        for w in result['windows'] for r in w['modes'].values())
    if 8 in args.microsteps and max(args.microsteps) == 16:
        result['screening_passed'] &= all(w['modes']['8']['endpoint_ca_rms_to_microsteps_max'] <= .05
                                          for w in result['windows'])
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('screening_passed', result['screening_passed'], flush=True)


if __name__ == '__main__':
    main()
