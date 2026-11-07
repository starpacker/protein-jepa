"""Correct-noise ensemble eval for the S_0 arms (noise_mix=0 reconstruction).

The exact/prior/gru S_0-arm checkpoints record ``args.noise_mix=0.8``, but the
training launch scripts (``run_scripts/run_lasp_s0_arms.sh`` and
``run_scripts/drive_lasp_s0_arms.sh``) pass ``--noise-mix 0``. The
``s0_mode='exact'`` closed-form Kalman posterior also *requires* diagonal
transition noise (R = 2 h D), so ``noise_mix=0`` is the only self-consistent
value. The committed ``scripts/eval_lasp_ca_dynamics.py`` re-reads the stale
``args.noise_mix`` and reconstructs a spatially-correlated (non-identity) noise
covariance, which the current ``model.py`` correctly rejects.

This wrapper reproduces ``eval_lasp_ca_dynamics.py`` exactly, except it
reconstructs ``CADynamics`` with ``noise_mix=0.0`` (the value training actually
used). All other reconstruction and every metric is byte-for-byte the same
``extended_ensemble_summary`` code path. It is a documented evaluation entry,
not a change to the sealed training or metric definitions.

Only the val manifest is read; test.jsonl is never opened.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.training.train_lasp_calibration import calibration_terms
from scripts.eval_lasp_final import METRICS


def anchor_target(context, n_samples):
    return context['coords'][-1, :, 1][None, None].cuda().expand(1, n_samples, -1, -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--max-windows', type=int, default=200)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--temperatures', type=float, nargs='+', default=[320., 348., 379., 413., 450.])
    p.add_argument('--anchors-per-trajectory', type=int, default=4)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    a = p.parse_args()
    torch.set_num_threads(4)

    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('checkpoint_kind') != 'lasp_ca_dynamics':
        raise ValueError('incorrect checkpoint kind')

    component = Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck['component_checkpoint'], component)

    # noise_mix=0.0 is the training-launch value; the checkpoint's recorded
    # args.noise_mix=0.8 is stale metadata inconsistent with s0_mode='exact'.
    model = CADynamics(str(component), 'cuda', a.checkpoint,
                       history_conditioning=ck['args'].get('history_conditioning', False),
                       noise_mix=0.0,
                       noise_length=ck['args'].get('noise_length', 8.),
                       collective_mobility=ck['args'].get('collective_mobility', False),
                       noise_kernel=ck['args'].get('noise_kernel', 'gaussian'),
                       noise_cutoff=ck['args'].get('noise_cutoff', 10.)).eval()

    protocol = EvalProtocol(
        manifest=a.manifest, horizons_frames=tuple(a.horizons),
        context_frames=8, frame_stride=1, max_residues=a.max_residues,
        max_anchors_per_trajectory=a.anchors_per_trajectory,
        temperatures=tuple(a.temperatures), seed=a.seed,
        round_robin_proteins=True, round_robin_conditions=True)
    rows = []
    for w in iter_windows(protocol):
        if len(rows) >= a.max_windows:
            break
        mask = w.context['residue_mask'].all(0).bool().cuda()
        if int(mask.sum()) < 4:
            continue
        key = f'{w.protein_id}|{w.trajectory_id}|{w.anchor_frame}|{w.horizon}'
        seed = (a.seed * 1000003 + int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)) % (2 ** 31 - 1)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        with torch.no_grad():
            ca, _ = model.sample(w.context['coords'][None].cuda(), mask[None],
                                 *[w.context[k][None].cuda() for k in ('aatype', 'residue_index', 'chain_id')],
                                 torch.tensor([w.temperature_k], device='cuda'), w.horizon, a.n_samples, generator)
            geometry = calibration_terms(ca, anchor_target(w.context, ca.shape[1]), mask[None],
                                         w.context['residue_index'][None].cuda(), w.context['chain_id'][None].cuda())
            geometry = {k: float(geometry[k]) for k in ('ca_bond', 'ca_clash_fraction', 'ca_angle_cos_mae')}
            pred = ca[0, :, mask]
            truth = w.future_path['coords'][:, :, 1].cuda()[:, mask]
            anchor = w.context['coords'][-1, :, 1].cuda()[mask]
            metrics = extended_ensemble_summary(pred.float(), truth.float(), None, cutoff=8., pca_k=3)
            persist = extended_ensemble_summary(anchor[None].expand(len(truth), -1, -1), truth.float(), None, cutoff=8., pca_k=3)
        rows.append(dict(protein=w.protein_id, traj=w.trajectory_id, temperature_k=float(w.temperature_k),
                         anchor_frame=int(w.anchor_frame), horizon=float(w.horizon),
                         actual_n_samples=int(ca.shape[1]),
                         geometry_vs_anchor=geometry,
                         contact_counts={k: float(metrics[k]) for k in ('weak_n_true', 'weak_n_pred', 'transient_n_true', 'transient_n_pred') if k in metrics},
                         lasp={m: float(metrics[m]) for m in METRICS},
                         persist={m: float(persist[m]) for m in METRICS}))
        if len(rows) % 10 == 0:
            print(f'scored {len(rows)}', flush=True)
    result = dict(metric_version='empirical_w2_v2_pca_projection',
                  noise_mix_reconstruction=0.0,
                  note='noise_mix=0.0 reconstruction matching training launch; checkpoint args.noise_mix=0.8 is stale',
                  checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                  checkpoint=a.checkpoint, args=vars(a), n_samples=a.n_samples, n_windows=len(rows),
                  n_proteins=len({r['protein'] for r in rows}), test_opened=False, per_window=rows)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(n_windows=len(rows), n_proteins=result['n_proteins'])), flush=True)


if __name__ == '__main__':
    main()
