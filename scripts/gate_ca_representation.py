"""Held-out CA latent readout gate; future-H decoding is an oracle, never a forecast.

The default windows match the standard200 development protocol. All conditions
share the observed anchor, decoder, residue mask, and actual MD path. No optimizer
or latent fitting is used. A pass certifies only this representation's readout on
this development panel, not predictive sufficiency or autonomous dynamics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import _align_to, extended_ensemble_summary
from evaluation.geometry import aligned_rmsd
from evaluation.loaders import sha256_file
from protein_jepa_frame.data.protocol import validate_manifest_split
from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
from protein_jepa_frame.training.train_lasp_calibration import calibration_terms
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest


CONDITIONS = ('true_future_H', 'anchor_H', 'wrong_time_H', 'persistence')
PRIMARY = ('endpoint_rmsd', 'nonlocal_distance_change_rmse')
SOURCE_PATHS = (
    'scripts/gate_ca_representation.py', 'evaluation/data.py',
    'evaluation/ensemble_metrics.py', 'evaluation/ensemble_metrics_ext.py',
    'evaluation/geometry.py', 'protein_jepa_frame/data/protocol.py',
    'protein_jepa_frame/models/ca_component_bundle.py',
    'protein_jepa_frame/models/encoder.py',
    'protein_jepa_frame/models/geometric_delta_decoder.py',
    'protein_jepa_frame/models/latent_delta_decoder.py',
    'protein_jepa_frame/models/ca_geometry.py',
    'protein_jepa_frame/training/train_lasp_calibration.py',
    'protein_jepa_frame/utils/contracts.py',
)


def readout_metrics(pred, truth, anchor, mask, residue_index, chain_id):
    """Score a time-matched path, including its anchor, in the observed gauge.

    Future-coordinate alignment depends on real coordinates only. The primary
    coordinate error never realigns a prediction independently. Ensemble metrics
    retain the repository's standard anchor-inclusive path convention.
    """
    if pred.shape != truth.shape or pred.ndim != 3 or len(pred) < 2:
        raise ValueError('readout requires matching [anchor,...,future] CA paths')
    if int(mask.sum()) < 4 or not torch.isfinite(pred[:, mask]).all():
        raise ValueError('readout requires >=4 valid residues and finite predictions')
    target = _align_to(truth, anchor, mask)
    error = pred[:, mask] - target[:, mask]
    displacement_error = ((pred[1:] - pred[:1]) - (target[1:] - target[:1]))[:, mask]
    metrics = dict(
        endpoint_rmsd=float(error[-1].square().sum(-1).mean().sqrt()),
        endpoint_shape_rmsd=aligned_rmsd(pred[-1], truth[-1], mask),
        path_rmsd=float(error[1:].square().sum(-1).mean().sqrt()),
        displacement_rmse=float(displacement_error.square().sum(-1).mean().sqrt()),
        anchor_rmsd=float(error[0].square().sum(-1).mean().sqrt()),
    )
    metrics.update(extended_ensemble_summary(pred.float(), truth.float(), mask, cutoff=8., pca_k=3))
    i, j = torch.triu_indices(len(mask), len(mask), offset=1, device=mask.device)
    nonlocal_pair = mask[i] & mask[j] & (
        (chain_id[i] != chain_id[j]) | ((residue_index[i] - residue_index[j]).abs() >= 3))
    i, j = i[nonlocal_pair], j[nonlocal_pair]
    if not len(i):
        raise ValueError('no nonlocal pairs in gate window')
    pd = (pred[:, i] - pred[:, j]).norm(dim=-1)
    td = (truth[:, i] - truth[:, j]).norm(dim=-1)
    metrics.update(
        nonlocal_distance_rmse=float((pd[1:] - td[1:]).square().mean().sqrt()),
        nonlocal_distance_change_rmse=float(
            ((pd[1:] - pd[:1]) - (td[1:] - td[:1])).square().mean().sqrt()),
        nonlocal_contact_error=float(((pd[1:] < 8) != (td[1:] < 8)).float().mean()),
        nonlocal_contact_probability_mae=float(
            ((pd < 8).float().mean(0) - (td < 8).float().mean(0)).abs().mean()),
    )
    geometry = calibration_terms(pred[None], target[None], mask[None],
                                 residue_index[None], chain_id[None])
    for name in ('ca_bond', 'ca_clash_fraction', 'true_ca_clash_fraction', 'ca_angle_cos_mae'):
        metrics[name] = float(geometry[name])
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('nonfinite CA gate metric')
    return metrics


@torch.no_grad()
def score_window(bundle, window, device):
    context = window.context['coords'].to(device)
    coords = window.future_path['coords'].to(device)
    mask = window.context['residue_mask'].to(device).all(0).bool()
    if not bool(window.future_path['residue_mask'].to(device)[:, mask].all()):
        raise ValueError('future mask differs on evaluated residues')
    if not torch.equal(coords[0], context[-1]):
        raise ValueError('future_path must begin at the observed anchor')
    aa, ri, ci = [window.context[key][None].to(device) for key in ('aatype', 'residue_index', 'chain_id')]
    n = len(coords)
    expand = lambda value: value.expand(n, *value.shape[1:])
    future_h = bundle.encode(coords, expand(mask[None]), expand(aa), expand(ri), expand(ci))
    anchor_h = bundle.encode(context[-1:], mask[None], aa, ri, ci)
    wrong_h = bundle.encode(context[:1], mask[None], aa, ri, ci)
    rotation, anchor = _extract_backbone_frames(context[-1:])

    def decode(h):
        count = len(h)
        repeat = lambda value: value.expand(count, *value.shape[1:])
        return bundle.decode_ca(h, repeat(anchor_h), repeat(aa), repeat(ri), repeat(ci),
                                repeat(rotation), repeat(anchor), repeat(mask[None]))

    decoded = dict(true_future_H=decode(future_h),
                   anchor_H=decode(anchor_h).expand(n, -1, -1),
                   wrong_time_H=decode(wrong_h).expand(n, -1, -1),
                   persistence=anchor.expand(n, -1, -1))
    row = dict(protein=window.protein_id, trajectory=window.trajectory_id,
               temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
               target_frame=int(window.target_frame), horizon=int(window.horizon),
               horizon_ps=float(window.horizon_ps), context_frames=window.context_frames.tolist(),
               path_frames=n, valid_residues=int(mask.sum()),
               latent_delta_rms=float((future_h[-1, mask] - anchor_h[0, mask]).square().mean().sqrt()))
    row['conditions'] = {name: readout_metrics(pred, coords[:, :, 1], anchor[0], mask, ri[0], ci[0])
                         for name, pred in decoded.items()}
    return row


def assess_rows(rows, *, seed=20261101, bootstrap_reps=10000,
                min_proteins=4, max_oracle_ratio=1/3, bond_limit=.1):
    """Equal protein weighting; paired differences resample protein clusters.

    CI95 are marginal two-sided intervals, not a familywise superiority claim.
    Duplicate windows fail closed instead of increasing apparent sample size.
    """
    if not rows or bootstrap_reps < 1 or min_proteins < 2 or not 0 < max_oracle_ratio < 1:
        raise ValueError('invalid gate rows or bootstrap/threshold parameters')
    groups, seen = defaultdict(list), set()
    metric_names = tuple(rows[0]['conditions']['true_future_H'])
    for row in rows:
        key = (row['protein'], row['trajectory'], row['temperature_k'], row['anchor_frame'], row['horizon'])
        if key in seen:
            raise ValueError(f'duplicate gate window: {key}')
        seen.add(key)
        for condition in CONDITIONS:
            if set(row['conditions'][condition]) != set(metric_names):
                raise ValueError('gate metric schema changed across windows')
            if not all(math.isfinite(value) for value in row['conditions'][condition].values()):
                raise ValueError('nonfinite gate input')
        groups[(row['horizon'], row['protein'])].append(row)
    proteins = defaultdict(list)
    for (horizon, protein), windows in sorted(groups.items()):
        values = {condition: {metric: float(np.mean([r['conditions'][condition][metric] for r in windows]))
                              for metric in metric_names} for condition in CONDITIONS}
        proteins[horizon].append(dict(protein=protein, windows=len(windows), conditions=values))
    result = {}
    for horizon, summaries in sorted(proteins.items()):
        count = len(summaries)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(horizon)]))
        idx = rng.integers(0, count, size=(bootstrap_reps, count))
        arrays = {condition: {metric: np.array([p['conditions'][condition][metric] for p in summaries])
                               for metric in metric_names} for condition in CONDITIONS}

        def estimate(values):
            return dict(mean=float(values.mean()),
                        ci95=np.quantile(values[idx].mean(1), [.025, .975]).tolist())

        metrics = {condition: {metric: estimate(value) for metric, value in values.items()}
                   for condition, values in arrays.items()}
        paired = {condition: {metric: estimate(arrays['true_future_H'][metric] - value)
                              for metric, value in arrays[condition].items()}
                  for condition in CONDITIONS if condition != 'true_future_H'}
        persistence = metrics['persistence']['endpoint_rmsd']['mean']
        ratio = metrics['true_future_H']['endpoint_rmsd']['mean'] / persistence if persistence > 1e-6 else None
        checks = dict(
            enough_proteins=count >= min_proteins,
            nontrivial_motion=persistence > 1e-6,
            oracle_error_budget=ratio is not None and ratio <= max_oracle_ratio,
            paired_endpoint_improvement=all(paired[c]['endpoint_rmsd']['ci95'][1] < 0 for c in paired),
            distance_change_improvement=paired['persistence']['nonlocal_distance_change_rmse']['ci95'][1] < 0,
            ca_bond_budget=metrics['true_future_H']['ca_bond']['mean'] <= bond_limit,
        )
        result[str(horizon)] = dict(n_proteins=count, n_windows=sum(p['windows'] for p in summaries),
                                    metrics=metrics, paired_true_minus_control=paired,
                                    oracle_persistence_ratio=ratio, checks=checks, passed=all(checks.values()))
    return dict(passed=all(value['passed'] for value in result.values()), by_horizon=result,
                per_protein=[p | {'horizon': h} for h, values in sorted(proteins.items()) for p in values])


def audit_sources(checkpoint, component, train_manifest, val_manifest):
    """Audit declared training splits and snapshot upstream files actually loaded.

    Legacy components do not record immutable hashes for all historical parents;
    current snapshots are recorded explicitly rather than asserting that history.
    """
    manifests = {str(Path(train_manifest).resolve())}
    upstream = {}
    for source in (checkpoint, component):
        for key in ('args', 'train_args', 'calibration_args'):
            args = source.get(key, {})
            if args.get('manifest'):
                manifests.add(str(Path(args['manifest']).resolve()))
    args = component['calibration_args']
    for key in ('stage_a_ckpt', 'b1a_ckpt'):
        path = args[key]
        upstream[key] = dict(path=path, current_sha256=sha256_file(path))
        source = torch.load(path, map_location='cpu', weights_only=False)
        config = source.get('train_args', source.get('args', {}))
        if config.get('synthetic') or not config.get('manifest'):
            raise ValueError(f'{key} lacks real-data training provenance')
        manifest = str(Path(config['manifest']).resolve())
        expected = source.get('train_manifest_sha256', config.get('train_manifest_sha256'))
        if expected and sha256_file(manifest) != expected:
            raise ValueError(f'{key} training manifest hash changed')
        manifests.add(manifest)
    audits = [dict(manifest=path, **validate_manifest_split(path, val_manifest)) for path in sorted(manifests)]
    return dict(training_splits=audits, upstream_checkpoints=upstream,
                lineage_scope='current checkpoint snapshots and declared training splits; legacy historical parent hashes unavailable')


def run(args):
    args.manifest = _development_manifest(args.manifest)
    if args.max_windows < len(args.horizons) or args.max_windows % len(args.horizons):
        raise ValueError('window budget must contain complete groups of horizons')
    if args.history_frames < 2 or args.torch_threads < 1:
        raise ValueError('at least two history frames and one torch thread required')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    checkpoint_hash = sha256_file(args.component)
    checkpoint = torch.load(args.component, map_location='cpu', weights_only=False)
    component = checkpoint.get('component_checkpoint', checkpoint)
    if component.get('checkpoint_kind') != 'ca_component_probe':
        raise ValueError('requires a CA component or dynamics checkpoint embedding one')
    provenance = audit_sources(checkpoint, component, args.train_manifest, args.manifest)
    source_hashes = {path: sha256_file(path) for path in SOURCE_PATHS}
    protocol = EvalProtocol(manifest=args.manifest, context_frames=args.history_frames,
                            horizons_frames=tuple(args.horizons), frame_stride=1,
                            max_residues=args.max_residues, max_anchors_per_trajectory=4,
                            temperatures=(320., 348., 379., 413., 450.), seed=args.seed,
                            round_robin_proteins=True, round_robin_conditions=True)
    rows = []
    device = torch.device(args.device)
    with tempfile.TemporaryDirectory(prefix='ca_representation_gate_') as temp:
        path = Path(temp) / 'component.pt'
        torch.save(component, path)
        bundle = CAComponentBundle(path, device).eval()
        encoder_hash, decoder_hash = state_sha256(bundle.encoder), state_sha256(bundle.decoder)
        if encoder_hash != state_sha256(component['encoder']):
            raise ValueError('loaded encoder differs from selected component')
        for window in iter_windows(protocol):
            if len(rows) == args.max_windows:
                break
            rows.append(score_window(bundle, window, device))
            if len(rows) % 10 == 0:
                print(f'CA oracle readout scored {len(rows)}/{args.max_windows}', flush=True)
        if state_sha256(bundle.encoder) != encoder_hash or state_sha256(bundle.decoder) != decoder_hash:
            raise RuntimeError('readout gate changed model parameters')
    if len(rows) != args.max_windows or set(r['horizon'] for r in rows) != set(args.horizons):
        raise RuntimeError('incomplete gate window panel')
    if sha256_file(args.component) != checkpoint_hash or any(sha256_file(p) != h for p, h in source_hashes.items()):
        raise RuntimeError('gate checkpoint or metric/model source changed during evaluation')
    for audit in provenance['training_splits']:
        if sha256_file(audit['manifest']) != audit['train_manifest_sha256']:
            raise RuntimeError('training manifest changed during gate')
    if sha256_file(args.manifest) != provenance['training_splits'][0]['val_manifest_sha256']:
        raise RuntimeError('validation manifest changed during gate')
    assessment = assess_rows(rows, seed=args.bootstrap_seed, bootstrap_reps=args.bootstrap_reps,
                             min_proteins=args.min_proteins, max_oracle_ratio=args.max_oracle_ratio,
                             bond_limit=args.bond_limit)
    result = dict(schema_version=1, purpose='true_H_readout_gate_only_not_forecast_evidence',
                  dataset_role='development_validation', optimizer_steps=0, test_opened=False,
                  checkpoint=args.component, checkpoint_sha256=checkpoint_hash,
                  encoder_tensor_sha256=encoder_hash, decoder_tensor_sha256=decoder_hash,
                  component_step=component.get('step'), checkpoint_step=checkpoint.get('step'),
                  checkpoint_kind=checkpoint.get('checkpoint_kind'), source_hashes=source_hashes,
                  train_manifest_sha256=sha256_file(args.train_manifest),
                  val_manifest_sha256=sha256_file(args.manifest), provenance=provenance,
                  protocol=asdict(protocol), args=vars(args),
                  metric_semantics=dict(reference='MD path including anchor; same-time oracle reconstructions',
                      coordinate_gauge='MD truth aligned to observed anchor; primary errors do not fit predictions',
                      wrong_time_H='first observed history H repeated; observed anchor decoder geometry fixed',
                      paired_difference='true_future_H minus control; lower is better except correlations/Jaccards',
                      confidence='equal-protein paired cluster bootstrap, marginal two-sided CI95',
                      thresholds_origin='oracle ratio/min_proteins from existing H gate; CA bond budget from calibration'),
                  n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}),
                  per_window=rows, **assessment)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(output), passed=result['passed'], n_windows=len(rows),
                         by_horizon={h: dict(ratio=v['oracle_persistence_ratio'], checks=v['checks'])
                                     for h, v in result['by_horizon'].items()}), indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--train-manifest', default='artifacts/mdcath_manifest_full/train.jsonl')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bootstrap-seed', type=int, default=20261101)
    parser.add_argument('--bootstrap-reps', type=int, default=10000)
    parser.add_argument('--max-windows', type=int, default=200)
    parser.add_argument('--max-residues', type=int, default=96)
    parser.add_argument('--history-frames', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--min-proteins', type=int, default=4)
    parser.add_argument('--max-oracle-ratio', type=float, default=1/3)
    parser.add_argument('--bond-limit', type=float, default=.1)
    parser.add_argument('--require-pass', action='store_true', help='Exit 2 when the completed gate fails.')
    args = parser.parse_args()
    report = run(args)
    if args.require_pass and not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
