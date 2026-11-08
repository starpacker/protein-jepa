"""CA endpoint diagnostics on the standard200 development window protocol.

Each forecast receives observed context only; its reference is the ONE endpoint
at the requested horizon, rigidly aligned to the observed anchor. Scores are
interpreted in expectation across held-out windows. One realized endpoint does
not estimate a per-window conditional ensemble, RMSF, or dynamical correctness.

Existing score definitions are preserved: coordinate ES uses the training
off-diagonal U-statistic; physical-distance ES uses evaluation.geometry's
empirical V-statistic; contact Brier scores empirical hard-contact frequencies.
The latter two are finite-ensemble scores, not unbiased estimates of the latent
generator's score. No historical seven-metric result is changed or superseded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import ca_geometry_summary, kabsch_align, physical_distance_energy_score
from evaluation.loaders import sha256_file
from protein_jepa_frame.training.train_lasp_ca_dynamics import (
    CADynamics, contact_probability_loss, energy_score,
)
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed
from scripts.gate_ca_representation import SOURCE_PATHS as GATE_SOURCES, audit_sources


SCORES = ('coordinate_energy_score', 'physical_distance_energy_score', 'contact_brier')
TASK_VERSION = 'ca_conditional_endpoint_scores_v1'
SOURCES = tuple(dict.fromkeys((*GATE_SOURCES, 'scripts/eval_ca_endpoint_scores.py',
    'scripts/diag_lasp_ca_ceiling.py', 'evaluation/loaders.py',
    'protein_jepa_frame/data/dataset.py', 'protein_jepa_frame/models/stage_b1b/model.py',
    'protein_jepa_frame/models/correlated_bath_posterior.py',
    'protein_jepa_frame/training/train_lasp_ca_dynamics.py',
    'protein_jepa_frame/training/train_stage_ensemble.py')))


def canonical_pairs(mask, residue_index, chain_id):
    """Require the canonical distance metric's index-pair rule to be physical.

    Standard200 CA domains satisfy this condition. Gapped/multichain layouts
    with a different physical pair set fail visibly instead of silently changing
    evaluation.geometry.physical_distance_energy_score's established definition.
    """
    i, j = torch.triu_indices(len(mask), len(mask), offset=1, device=mask.device)
    valid = mask[i] & mask[j]
    canonical = valid & (j - i >= 3)
    physical = valid & ((chain_id[i] != chain_id[j]) | ((residue_index[i] - residue_index[j]).abs() >= 3))
    if not torch.equal(canonical, physical):
        raise ValueError('canonical index pairs differ from physical residue/chain pairs; unsupported layout')
    if not bool(canonical.any()):
        raise ValueError('no valid nonlocal pairs')
    return i[canonical], j[canonical]


@torch.no_grad()
def endpoint_scores(samples, endpoint, anchor, mask, residue_index, chain_id):
    """One endpoint label, fixed anchor gauge, and unchanged canonical metrics."""
    if (samples.ndim != 3 or len(samples) < 2 or samples.shape[1:] != endpoint.shape
            or endpoint.shape != anchor.shape or endpoint.shape != (len(mask), 3)
            or mask.dtype != torch.bool or int(mask.sum()) < 4):
        raise ValueError('scores require >=2 samples, >=4 valid residues and matching CA coordinates')
    if not all(bool(torch.isfinite(x[..., mask, :]).all()) for x in (samples, endpoint, anchor)):
        raise ValueError('nonfinite evaluated CA coordinates')
    i, j = canonical_pairs(mask, residue_index, chain_id)
    aligned_endpoint = kabsch_align(endpoint, anchor, mask)
    # Float64 metric arithmetic; no independently fitted prediction alignment.
    x, y = samples.double(), aligned_endpoint.double()
    predicted_contacts = ((x[:, i] - x[:, j]).norm(dim=-1) < 8.).double().mean(0)
    true_contacts = ((endpoint.double()[i] - endpoint.double()[j]).norm(dim=-1) < 8.).double()
    result = dict(
        coordinate_energy_score=float(energy_score(x[:, mask].flatten(1), y[None, mask].flatten(1))),
        physical_distance_energy_score=physical_distance_energy_score(
            x, endpoint.double(), mask, scale_angstrom=1., max_pairs=512),
        contact_brier=float(contact_probability_loss(predicted_contacts, true_contacts, 'brier')),
    )
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError('nonfinite endpoint score')
    return result


@torch.no_grad()
def score_window(model, window, *, device='cpu', n_samples=12, seed=0):
    if getattr(model, 'prediction_kind', None) != 'endpoint':
        raise ValueError('endpoint evaluator requires endpoint sampling')
    if (window.target_frame != window.anchor_frame + window.horizon
            or not math.isclose(window.horizon_ps, window.horizon * model.kernel.cfg.step_ps,
                                rel_tol=1e-6, abs_tol=1e-3)):
        raise ValueError('requested endpoint and sampler physical horizon disagree')
    context = window.context
    mask = context['residue_mask'].to(device).all(0).bool()
    if int(mask.sum()) < 4:
        raise ValueError('endpoint window needs >=4 context-valid residues')
    ri, ci = [context[key].to(device) for key in ('residue_index', 'chain_id')]
    sampling_seed = _window_seed(seed, window)
    generator = torch.Generator(device=device).manual_seed(sampling_seed)
    # The sampler signature has no target, target mask, future H, or future path.
    samples, _ = model.sample(context['coords'][None].to(device), mask[None],
        *[context[key][None].to(device) for key in ('aatype', 'residue_index', 'chain_id')],
        torch.tensor([window.temperature_k], device=device), window.horizon, n_samples, generator)
    if samples.shape != (1, n_samples, len(mask), 3):
        raise ValueError('sampler returned a different sample/residue budget')
    steps = getattr(model, 'last_observation_steps', None)
    if steps != [window.horizon] * n_samples:
        raise ValueError('sampler did not return exactly one requested endpoint per path')
    target = window.target
    if target['coords'].shape != (1, len(mask), 4, 3):
        raise ValueError('reference must be exactly one backbone endpoint')
    if not bool(target['residue_mask'].to(device)[:, mask].all()):
        raise ValueError('target is missing a context-valid residue')
    anchor = context['coords'][-1, :, 1].to(device)
    endpoint = target['coords'][0, :, 1].to(device)
    persistence = anchor[None].expand(n_samples, -1, -1)
    metrics = {name: endpoint_scores(x, endpoint, anchor, mask, ri, ci)
               for name, x in (('model', samples[0]), ('persistence', persistence))}
    geometry_args = dict(chain_id=ci, residue_index=ri)
    per_sample = [ca_geometry_summary(x, mask, **geometry_args) for x in samples[0]]
    geometry_mean = {key: (float(np.mean([r[key] for r in per_sample]))
                           if all(r[key] is not None for r in per_sample) else None)
                     for key in per_sample[0]}
    return dict(protein=window.protein_id, trajectory=window.trajectory_id,
        temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
        target_frame=int(window.target_frame), horizon=int(window.horizon), horizon_ps=float(window.horizon_ps),
        context_frames=window.context_frames.tolist(), valid_residues=int(mask.sum()),
        nonlocal_pairs=int(len(canonical_pairs(mask, ri, ci)[0])), actual_n_samples=n_samples,
        sampling_seed=sampling_seed, observation_steps=steps, scores=metrics,
        sample_geometry=per_sample, mean_sample_geometry=geometry_mean,
        persistence_geometry=ca_geometry_summary(anchor, mask, **geometry_args),
        target_geometry=ca_geometry_summary(endpoint, mask, **geometry_args))


def summarize(rows):
    """Equal windows within protein, then equal proteins; no CI/pass claim."""
    groups, seen = defaultdict(list), set()
    for row in rows:
        key = (row['protein'], row['trajectory'], row['temperature_k'], row['anchor_frame'], row['horizon'])
        if key in seen:
            raise ValueError('duplicate endpoint evaluation window')
        seen.add(key)
        groups[(row['horizon'], row['protein'])].append(row)
    result = {}
    for horizon in sorted({h for h, _ in groups}):
        proteins = []
        for (h, protein), windows in sorted(groups.items()):
            if h != horizon:
                continue
            values = {arm: {metric: float(np.mean([w['scores'][arm][metric] for w in windows]))
                            for metric in SCORES} for arm in ('model', 'persistence')}
            proteins.append(dict(protein=protein, n_windows=len(windows), scores=values))
        means = {arm: {metric: float(np.mean([p['scores'][arm][metric] for p in proteins]))
                       for metric in SCORES} for arm in ('model', 'persistence')}
        result[str(horizon)] = dict(n_proteins=len(proteins), n_windows=sum(p['n_windows'] for p in proteins),
            means=means, paired_model_minus_persistence={m: means['model'][m] - means['persistence'][m] for m in SCORES},
            per_protein=proteins)
    return result


def load_model(path, checkpoint, device):
    if checkpoint.get('checkpoint_kind') != 'lasp_ca_dynamics' or 'component_checkpoint' not in checkpoint:
        raise ValueError('requires lasp_ca_dynamics with an embedded CA component')
    args = checkpoint['args']
    model = CADynamics(path, device, path,
        history_conditioning=args.get('history_conditioning', False),
        noise_mix=args.get('noise_mix', 0.), noise_length=args.get('noise_length', 8.),
        collective_mobility=args.get('collective_mobility', False),
        noise_kernel=args.get('noise_kernel'), noise_cutoff=args.get('noise_cutoff'),
        s0_mode=args.get('s0_mode', 'gru')).eval()
    model.prediction_kind = 'endpoint'
    if state_sha256(model.kernel) != state_sha256(checkpoint['kernel']):
        raise ValueError('loaded kernel differs from checkpoint')
    if state_sha256(model.bundle.encoder) != state_sha256(checkpoint['component_checkpoint']['encoder']):
        raise ValueError('loaded encoder differs from checkpoint')
    component = checkpoint['component_checkpoint']
    decoder_state = component.get('delta_ca_decoder', component.get('decoder'))
    if decoder_state is None or state_sha256(model.bundle.decoder) != state_sha256(decoder_state):
        raise ValueError('loaded decoder differs from embedded component')
    return model


def run(args):
    args.manifest = _development_manifest(args.manifest)
    if args.max_windows < 4 or args.max_windows % 4 or args.torch_threads < 1:
        raise ValueError('standard200 prefixes require complete four-horizon groups and positive threads')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    checkpoint_hash = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = load_model(args.checkpoint, checkpoint, args.device)
    provenance = audit_sources(checkpoint, checkpoint['component_checkpoint'],
                              'artifacts/mdcath_manifest_full/train.jsonl', args.manifest)
    source_hashes = {path: sha256_file(path) for path in SOURCES}
    model_hash = state_sha256(model)
    protocol = EvalProtocol(manifest=args.manifest, context_frames=8, horizons_frames=(1, 4, 16, 64),
        max_residues=96, max_anchors_per_trajectory=4, temperatures=(320., 348., 379., 413., 450.),
        seed=0, round_robin_proteins=True, round_robin_conditions=True)
    rows = []
    for window in iter_windows(protocol):
        if len(rows) == args.max_windows:
            break
        rows.append(score_window(model, window, device=args.device, n_samples=12, seed=0))
        if len(rows) % 10 == 0:
            print(f'Endpoint scores {len(rows)}/{args.max_windows}', flush=True)
    if len(rows) != args.max_windows or set(r['horizon'] for r in rows) != {1, 4, 16, 64}:
        raise RuntimeError('incomplete endpoint panel')
    if state_sha256(model) != model_hash or sha256_file(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('model/checkpoint changed during endpoint evaluation')
    if any(sha256_file(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError('metric/model source changed during endpoint evaluation')
    for audit in provenance['training_splits']:
        if (sha256_file(audit['manifest']) != audit['train_manifest_sha256']
                or sha256_file(args.manifest) != audit['val_manifest_sha256']):
            raise RuntimeError('manifest changed during endpoint evaluation')
    report = dict(task_version=TASK_VERSION, dataset_role='development_validation',
        prediction_kind='endpoint', reference_kind='single_matched_future_endpoint',
        optimizer_steps=0, test_opened=False, future_information_given_to_sampler=False,
        checkpoint=args.checkpoint, checkpoint_sha256=checkpoint_hash, model_tensor_sha256=model_hash,
        checkpoint_step=checkpoint.get('step'), checkpoint_training_target=checkpoint['args'].get('target'),
        sampling=dict(seed=0, n_samples=12, model_history_frames=4 if model.history_conditioning else 1,
                      s0_mode=model.kernel.s0_mode, collective_mobility=model.collective_mobility,
                      noise_mix=model.noise_mix, noise_kernel=model.noise_kernel,
                      window_seed_rule='same stateless SHA256 rule as eval_lasp_ca_dynamics'),
        protocol=asdict(protocol), args=vars(args), source_hashes=source_hashes, provenance=provenance,
        metric_semantics=dict(
            coordinate_energy_score='training.energy_score: off-diagonal U-statistic, norm/sqrt(3*N_valid); angstrom',
            physical_distance_energy_score='evaluation.geometry: diagonal-inclusive empirical V-statistic, up to512 deterministic nonlocal pairs, norm/sqrt(N_pairs), distance/1 angstrom',
            contact_brier='mean squared error of empirical P(CA distance<8 angstrom) versus one binary endpoint outcome; all nonlocal pairs; dimensionless',
            finite_sample='coordinate U-score is unbiased for iid generator draws; distance V-score and empirical Brier include finite-M effects; M fixed at12',
            coordinate_gauge='only real endpoint fitted onto observed anchor; prediction coordinates never independently aligned',
            geometry='per-generated-conformation evaluation.geometry.ca_geometry_summary; bonds3.3-4.3 angstrom, nonlocal clashes<2.5 angstrom',
            interpretation='single endpoint scores assess conditional forecasts in expectation across windows; no per-window conditional ensemble estimate or physical-memory proof',
            aggregation='equal windows within protein then equal proteins separately per horizon; per-window rows retained for paired analyses; no significance/pass claim'),
        n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}), per_window=rows, by_horizon=summarize(rows))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), n_windows=len(rows), n_proteins=report['n_proteins']), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, choices=[0], default=0,
                        help='This protocol fixes the sampling/window seed to0.')
    parser.add_argument('--n-samples', type=int, choices=[12], default=12,
                        help='This protocol fixes the common finite ensemble budget to12.')
    parser.add_argument('--max-windows', type=int, default=200,
                        help='Use4 for a smoke; every prefix keeps complete horizon groups. Seed0 and12 samples are fixed.')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
