"""True-H readout diagnostic. This is never a future forecasting benchmark.

All interventions share the same observed anchor and decoder. A passing report
permits B1b calibration only for this exact frozen representation and protocol.
Validation proteins used here are development data, not a final test set.
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
from evaluation.geometry import aligned_rmsd, ca_from_backbone, common_anchor_rmsd
from evaluation.loaders import build_bundle, sha256_file
from evaluation.run_eval import _batch, set_determinism, source_provenance, window_seed
from evaluation import variants as V
from protein_jepa_frame.data.protocol import validate_manifest_split


CONDITIONS = ('true_future_H', 'anchor_H', 'wrong_time_H', 'zero_H', 'shuffled_H')


def _observables(pred, target, mask, *, chain_id=None, residue_index=None):
    """Nonlocal distance, contact-change error, signed CA pseudo-torsion error."""
    L = target.shape[0]
    i, j = torch.triu_indices(L, L, offset=3, device=target.device)
    good = mask[i] & mask[j]
    i, j = i[good], j[good]
    if not len(i):
        raise ValueError('no valid nonlocal residue pairs for H gate')
    pd = (pred[i]-pred[j]).norm(dim=-1)
    td = (target[i]-target[j]).norm(dim=-1)
    # Subtracting the same anchor observable from prediction and target cancels;
    # these errors therefore also measure internal distance/contact changes.
    distance = float((pd-td).square().mean().sqrt())
    contact = float(((pd < 8) != (td < 8)).float().mean())

    def torsion(x):
        a, b, c = x[1:-2]-x[:-3], x[2:-1]-x[1:-2], x[3:]-x[2:-1]
        unit = b/b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u = a-(a*unit).sum(-1, keepdim=True)*unit
        v = c-(c*unit).sum(-1, keepdim=True)*unit
        return torch.atan2((torch.cross(unit, u, dim=-1)*v).sum(-1), (u*v).sum(-1))

    keep = mask[:-3] & mask[1:-2] & mask[2:-1] & mask[3:]
    if chain_id is not None:
        same = chain_id[1:] == chain_id[:-1]
        keep &= same[:-2] & same[1:-1] & same[2:]
    if residue_index is not None:
        contiguous = residue_index[1:]-residue_index[:-1] == 1
        keep &= contiguous[:-2] & contiguous[1:-1] & contiguous[2:]
    angle = torsion(pred)-torsion(target)
    circ = float(torch.atan2(angle.sin(), angle.cos())[keep].abs().mean()) if keep.any() else None
    return dict(distance_change_rmse=distance, contact_change_error=contact,
                ca_pseudotorsion_mae_rad=circ)


def _mean(rows, name):
    values = [r[name] for r in rows if r.get(name) is not None]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError(f'empty/nonfinite gate metric: {name}')
    return float(np.mean(values))


def assess_rows(rows, *, max_oracle_ratio=1/3, min_proteins=4, seed=0, bootstrap_reps=2000):
    """Equal protein weight and paired cluster bootstrap, separately per horizon."""
    if not rows:
        raise ValueError('H gate emitted no windows')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['horizon_frames'], row['protein_id'])].append(row)
    by_horizon = defaultdict(list)
    metrics = ('rmsd', 'shape_rmsd', 'distance_change_rmse', 'contact_change_error')
    for (h, p), windows in sorted(grouped.items()):
        summary = dict(protein_id=p, horizon_frames=h, windows=len(windows))
        for condition in (*CONDITIONS, 'persistence'):
            summary[condition] = {m: _mean([r[condition] for r in windows], m) for m in metrics}
        by_horizon[h].append(summary)
    rng = np.random.default_rng(seed)
    results = {}
    for h, proteins in sorted(by_horizon.items()):
        n = len(proteins)
        means = {c: {m: float(np.mean([p[c][m] for p in proteins])) for m in metrics}
                 for c in (*CONDITIONS, 'persistence')}
        comparisons = {}
        for control in ('anchor_H', 'wrong_time_H', 'persistence'):
            delta = np.array([p['true_future_H']['rmsd']-p[control]['rmsd'] for p in proteins])
            boots = delta[rng.integers(0, n, size=(bootstrap_reps, n))].mean(1)
            comparisons[control] = dict(delta=float(delta.mean()),
                                        ci95=np.quantile(boots, [.025, .975]).tolist())
        persistence = means['persistence']['rmsd']
        ratio = means['true_future_H']['rmsd']/persistence if persistence > 1e-6 else None
        checks = dict(enough_proteins=n >= min_proteins,
                      nontrivial_motion=persistence > 1e-6,
                      oracle_error_budget=ratio is not None and ratio <= max_oracle_ratio,
                      paired_improvement=all(v['ci95'][1] < 0 for v in comparisons.values()),
                      distance_change_improvement=means['true_future_H']['distance_change_rmse']
                                                   < means['persistence']['distance_change_rmse'])
        results[str(h)] = dict(proteins=n, windows=sum(p['windows'] for p in proteins),
                               metrics=means, oracle_persistence_ratio=ratio,
                               paired_rmsd=comparisons, checks=checks, passed=all(checks.values()))
    return dict(passed=all(r['passed'] for r in results.values()), by_horizon=results,
                per_protein=[p for ps in by_horizon.values() for p in ps])


def run(args):
    torch.set_num_threads(args.torch_threads)
    set_determinism(args.seed)
    bundle = build_bundle(variant='oracle', b1a_ckpt=args.b1a_ckpt, device=args.device)
    if bundle.diagnostic_only and args.scope == 'heldout_protein':
        raise ValueError('diagnostic-only bundles cannot certify an H gate')
    proto = EvalProtocol(manifest=args.manifest, context_frames=args.history_frames,
                         frame_stride=args.frame_stride, horizons_frames=tuple(args.horizons_frames),
                         max_residues=args.max_residues, seed=args.seed,
                         anchor_stride=args.anchor_stride, max_proteins=args.max_proteins,
                         max_trajectories_per_protein=args.max_trajectories,
                         max_anchors_per_trajectory=args.max_anchors, skip_head_frames=args.skip_head_frames)
    if args.scope == 'heldout_protein' and not args.train_manifest:
        raise ValueError('heldout gate requires --train-manifest; training-fit diagnostics cannot pass')
    audit = validate_manifest_split(args.train_manifest, args.manifest) if args.scope == 'heldout_protein' else None
    ck = torch.load(args.b1a_ckpt, map_location='cpu', weights_only=False)
    # Verify every known upstream training manifest against the heldout proteins.
    upstream_audits = []
    declared = ck.get('train_args', {}).get('manifest')
    upstream = [declared] if declared and not ck.get('train_args', {}).get('synthetic') else []
    provenance = bundle.representation_contract.get('source') or {}
    source = provenance.get('stage_a_checkpoint')
    if source:
        expected = provenance.get('stage_a_checkpoint_sha256')
        if args.scope == 'heldout_protein' and not expected:
            raise ValueError('missing source checkpoint hash; cannot certify upstream lineage')
        if expected and sha256_file(source) != expected:
            raise ValueError('Stage-A source checkpoint has changed since B1a calibration')
        ac = torch.load(source, map_location='cpu', weights_only=False)
        aa = ac.get('train_args', ac.get('args', {}))
        if args.scope == 'heldout_protein' and not aa.get('manifest'):
            raise ValueError('missing actual Stage-A training manifest provenance')
        if aa.get('synthetic') and args.scope == 'heldout_protein':
            raise ValueError('synthetic-trained source is a diagnostic, not a real-data gate')
        if aa.get('manifest') and not aa.get('synthetic'):
            expected_manifest = provenance.get('stage_a_train_manifest_sha256')
            if args.scope == 'heldout_protein' and not expected_manifest:
                raise ValueError('missing Stage-A training manifest hash')
            if expected_manifest and sha256_file(aa['manifest']) != expected_manifest:
                raise ValueError('Stage-A training manifest changed')
            upstream.append(aa['manifest'])
    if args.scope == 'heldout_protein':
        if not upstream or ck.get('train_args', {}).get('synthetic') or not source:
            raise ValueError('missing upstream training provenance; cannot certify heldout gate')
        expected_manifest = provenance.get('b1a_train_manifest_sha256')
        if not expected_manifest or sha256_file(declared) != expected_manifest:
            raise ValueError('B1a training manifest missing hash or changed')
        for manifest in sorted(set(upstream)):
            upstream_audits.append(dict(manifest=manifest, **validate_manifest_split(manifest, args.manifest)))

    before_source = source_provenance()
    rows, seen, latent, frame_intervals = [], set(), defaultdict(list), set()
    with torch.no_grad():
        for w in iter_windows(proto):
            key = (w.protein_id, w.trajectory_id, w.temperature_k, w.anchor_frame, w.target_frame)
            if key in seen:
                raise ValueError(f'duplicate gate window: {key}')
            seen.add(key)
            frame_intervals.add(round(w.horizon_ps/w.horizon, 6))
            ctx, tgt = _batch(w.context, bundle.device), _batch(w.target, bundle.device)
            hctx, htgt = V.encode_clip(bundle, ctx), V.encode_clip(bundle, tgt)[:, -1]
            mask = ctx['residue_mask'].all(1).bool() & tgt['residue_mask'][:, -1].bool()
            if int(mask.sum()) < 4:
                raise ValueError('gate needs >=4 valid residues')
            R, t = V.anchor_frames(ctx)
            anchor, target = ca_from_backbone(ctx['coords'][:, -1])[0], ca_from_backbone(tgt['coords'][:, -1])[0]
            gen = torch.Generator(device=bundle.device).manual_seed(window_seed(args.seed, w))
            valid = torch.where(mask[0])[0]
            permutation = valid[torch.randperm(len(valid), generator=gen, device=bundle.device)]
            shuffled = htgt.clone()
            shuffled[:, valid] = htgt[:, permutation]
            values = dict(true_future_H=htgt, anchor_H=hctx[:, -1], wrong_time_H=hctx[:, 0],
                          zero_H=torch.zeros_like(htgt), shuffled_H=shuffled)
            row = dict(protein_id=w.protein_id, trajectory_id=w.trajectory_id, temperature_k=w.temperature_k,
                       anchor_frame=w.anchor_frame, target_frame=w.target_frame, horizon_frames=w.horizon,
                       horizon_ps=w.horizon_ps, context_frames=w.context_frames.tolist())
            observable_args = dict(chain_id=ctx['chain_id'][0], residue_index=ctx['residue_index'][0])
            for name, h in values.items():
                pred = V.decode(bundle, h, hctx[:, -1], ctx, R, t, mask)[0]
                if not torch.isfinite(pred[mask[0]]).all():
                    raise FloatingPointError(f'nonfinite {name} readout')
                row[name] = dict(rmsd=common_anchor_rmsd(pred, anchor, target, mask[0]),
                                 shape_rmsd=aligned_rmsd(pred, target, mask[0]),
                                 **_observables(pred, target, mask[0], **observable_args))
            row['persistence'] = dict(rmsd=common_anchor_rmsd(anchor, anchor, target, mask[0]),
                                      shape_rmsd=aligned_rmsd(anchor, target, mask[0]),
                                      **_observables(anchor, target, mask[0], **observable_args))
            # Current-frame shape readout is separate from future-change gating.
            current = V.decode(bundle, hctx[:, -1], hctx[:, -1], ctx, R, t, mask)[0]
            row['current_frame_shape_rmsd'] = aligned_rmsd(current, anchor, mask[0])
            delta = (htgt-hctx[:, -1])[0, mask[0]]
            row['latent_delta_rms'] = float(delta.square().mean().sqrt())
            latent[w.protein_id].append(delta.cpu().flatten())
            rows.append(row)
    assessment = assess_rows(rows, max_oracle_ratio=args.max_oracle_ratio,
                              min_proteins=args.min_proteins, seed=args.seed)
    ranks = {}
    for protein, deltas in latent.items():
        # Mask changes may change vector size; never pad arbitrary residues into rank.
        if len({d.numel() for d in deltas}) != 1:
            ranks[protein] = None
            continue
        X = torch.stack(deltas)
        s = torch.linalg.svdvals(X-X.mean(0, keepdim=True))
        p = s.square()/s.square().sum().clamp_min(1e-20)
        ranks[protein] = float(torch.exp(-(p*p.clamp_min(1e-20).log()).sum())) if s.square().sum() > 0 else 0.
    if len(frame_intervals) != 1:
        raise ValueError('one B1b gate requires a common physical frame interval')
    if source_provenance()['source_sha256'] != before_source['source_sha256']:
        raise RuntimeError('source code changed during H gate; rerun with a stable source tree')
    report = dict(schema_version=2, passed=assessment['passed'] and args.scope == 'heldout_protein',
                  seed=args.seed, bootstrap_reps=2000, torch_threads=args.torch_threads, device=str(bundle.device),
                  diagnostic_only=args.scope != 'heldout_protein', scope=args.scope,
                  purpose='true_H_readout_gate_only_not_forecast_evidence', dataset_role='development_validation',
                  representation_id=bundle.representation_contract['representation_id'],
                  checkpoint_sha256=sha256_file(args.b1a_ckpt), protocol={**asdict(proto), 'history_frames':args.history_frames,
                  'frame_interval_ps':next(iter(frame_intervals))},
                  train_manifest_sha256=sha256_file(args.train_manifest) if args.train_manifest else None,
                  val_manifest_sha256=sha256_file(args.manifest), dataset_audit=audit,
                  upstream_split_audits=upstream_audits, thresholds=dict(max_oracle_ratio=args.max_oracle_ratio,
                  min_proteins=args.min_proteins), source=before_source,
                  latent_delta_effective_rank=ranks, **{k:v for k,v in assessment.items() if k != 'passed'})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'per_window.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    lines = ['# H readout gate', '', f"Passed: **{report['passed']}**. Scope: `{args.scope}`.", '',
             'True-future-H oracle diagnostic; this is not autonomous forecast evidence.', '',
             f"Representation: `{report['representation_id']}`. Seed: {args.seed}.", '',
             '| Horizon (raw frames) | Proteins | Windows | True-H RMSD | Persistence RMSD | Oracle ratio | Passed |',
             '|---:|---:|---:|---:|---:|---:|---|']
    for h, r in report['by_horizon'].items():
        ratio = f"{r['oracle_persistence_ratio']:.3f}" if r['oracle_persistence_ratio'] is not None else 'N/A'
        lines.append(f"| {h} | {r['proteins']} | {r['windows']} | {r['metrics']['true_future_H']['rmsd']:.3f} | "
                     f"{r['metrics']['persistence']['rmsd']:.3f} | {ratio} | {r['passed']} |")
    lines += ['', 'All numerical thresholds, paired protein intervals, upstream hashes and source identity are in report.json.', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=report['passed'], proteins=len(latent), windows=len(rows),
                         report=str(output/'report.json')), indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--b1a-ckpt', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--train-manifest')
    p.add_argument('--scope', choices=['heldout_protein', 'diagnostic_fit'], default='heldout_protein')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--history-frames', type=int, default=8)
    p.add_argument('--frame-stride', type=int, default=1)
    p.add_argument('--horizons-frames', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--anchor-stride', type=int, default=16)
    p.add_argument('--skip-head-frames', type=int, default=0)
    p.add_argument('--max-proteins', type=int, default=4)
    p.add_argument('--max-residues', type=int, default=None,
                   help='Optional explicit crop; default evaluates complete proteins.')
    p.add_argument('--max-trajectories', type=int, default=1)
    p.add_argument('--max-anchors', type=int, default=8)
    p.add_argument('--max-oracle-ratio', type=float, default=1/3)
    p.add_argument('--min-proteins', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not 0 < args.max_oracle_ratio < 1 or args.min_proteins < 2 or args.history_frames < 2:
        p.error('require 0<ratio<1, min_proteins>=2 and history_frames>=2')
    report = run(args)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
