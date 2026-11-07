"""Ordered-path (trajectory) eval for the S_0 arms with noise_mix=0.0 reconstruction.

Mirror of ``scripts/eval_ca_ordered_paths.py`` for the exact/prior/gru S_0-arm
checkpoints whose recorded ``args.noise_mix=0.8`` is stale metadata: the launch
scripts pass ``--noise-mix 0`` and the ``s0_mode='exact'`` Kalman posterior
requires diagonal transition noise (R = 2 h D). The committed ordered-path entry
re-reads ``args.noise_mix`` and reconstructs a spatially-correlated covariance
that the current ``model.py`` rejects.

This wrapper reuses the *exact same* ``sample_ca_trajectory`` adapter and
``ordered_path_summary`` metric code paths, but reconstructs ``CADynamics`` with
``noise_mix=0.0``. It emits the three E0.2 trajectory metrics per horizon:

  * ACF Error   = |forecast distance_acf_lag1 - reference distance_acf_lag1|  (lag-1 autocorrelation error vs MD)
  * Path Error  = |forecast path_length - reference path_length|               (internal path length error vs MD)
  * Latent Stable = fraction of continuous rollouts whose H/S latent stays finite
                    plus the max ||latent|| norm across surviving rollouts.

Only the val manifest is read; test.jsonl is never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from evaluation.ca_trajectory_adapter import sample_ca_trajectory
from evaluation.data import EvalProtocol, iter_windows
from evaluation.trajectory_metrics_v2 import ordered_path_summary
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed
from scripts.eval_ca_ordered_paths import (
    LAGS, ROLES, _reference, observed_time_grid, temporal_controls,
)


def load_model_noisemix0(checkpoint, checkpoint_path, device, component_path):
    if checkpoint.get('checkpoint_kind') != 'lasp_ca_dynamics' or 'component_checkpoint' not in checkpoint:
        raise ValueError('requires lasp_ca_dynamics with an embedded CA component')
    args = checkpoint['args']
    torch.save(checkpoint['component_checkpoint'], component_path)
    model = CADynamics(str(component_path), device, checkpoint_path,
        history_conditioning=args.get('history_conditioning', False),
        noise_mix=0.0, noise_length=args.get('noise_length', 8.),
        collective_mobility=args.get('collective_mobility', False),
        noise_kernel=args.get('noise_kernel', 'gaussian'),
        noise_cutoff=args.get('noise_cutoff', 10.),
        s0_mode=args.get('s0_mode', 'gru')).eval()
    model.prediction_kind = 'endpoint'
    return model


@torch.no_grad()
def score_window(model, window, *, device='cpu', n_samples=12, decode_chunk_size=64):
    observed_times, relative_times, step_ps = observed_time_grid(window, model)
    context = window.context
    observed = context['coords'].to(device)
    mask = context['residue_mask'].to(device).all(0).bool()
    aa, ri, ci = [context[key].to(device) for key in ('aatype', 'residue_index', 'chain_id')]
    seed = _window_seed(0, window)
    result = sample_ca_trajectory(model, observed[None], mask[None], aa[None], ri[None], ci[None],
        torch.tensor([window.temperature_k], dtype=observed.dtype, device=device),
        window.horizon, n_samples, torch.Generator(device=device).manual_seed(seed),
        decode_chunk_size=decode_chunk_size, context_dt_ps=step_ps)
    forecast = result.ca[0].detach().cpu().double()
    mask = mask.cpu()
    anchor = observed[-1, :, 1].detach().cpu().double()
    controls, _control_metadata = temporal_controls(forecast, seed)
    reference, aligned_reference, _raw_reference_mask, reference_times = _reference(
        window, observed_times, relative_times, mask, anchor)
    paths = {'forecast': forecast, **controls,
             'persistence': anchor[None, None].expand(n_samples, window.horizon + 1, -1, -1),
             'reference_MD': aligned_reference[None]}
    arms = {}
    for name, coordinates in paths.items():
        metrics = ordered_path_summary(coordinates, reference, anchor, mask,
            predicted_times_ps=relative_times, reference_times_ps=reference_times - observed_times[-1],
            reference_mask=mask[None].expand(window.horizon + 1, -1),
            residue_index=ri.cpu(), chain_id=ci.cpu(), lags=LAGS, max_pairs=512)
        arms[name] = dict(metrics=metrics)
    # Latent stability from the single continuous H/S rollout.
    latent = result.latent if result.latent is not None else None
    if latent is None:
        latent_finite_frac, latent_max_norm = None, None
    else:
        latent = latent.detach().cpu().double()
        flat = latent.reshape(latent.shape[0], -1)
        finite = torch.isfinite(flat).all(dim=1)
        latent_finite_frac = float(finite.float().mean())
        norms = flat[finite].norm(dim=1)
        latent_max_norm = float(norms.max()) if norms.numel() else None
    return dict(protein=window.protein_id, horizon=int(window.horizon),
                forecast=arms['forecast']['metrics']['scalars'],
                reference=arms['reference_MD']['metrics']['scalars'],
                persistence=arms['persistence']['metrics']['scalars'],
                latent=dict(finite_frac=latent_finite_frac, max_norm=latent_max_norm,
                            shape=list(result.latent.shape) if result.latent is not None else None))


def _gap(row, key):
    f = row['forecast'].get(key)
    r = row['reference'].get(key)
    if f is None or r is None or not math.isfinite(f) or not math.isfinite(r):
        return None
    return abs(f - r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--max-windows', type=int, default=200)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--n-samples', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--device', default='cuda')
    p.add_argument('--decode-chunk-size', type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    manifest = _development_manifest(a.manifest)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    component_path = Path(a.out).with_suffix('.component.pt')
    component_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_model_noisemix0(checkpoint, a.checkpoint, a.device, component_path)
    if float(model.kernel.cfg.step_ps) != 1000.:
        raise ValueError('ordered-path panels require 1000 ps per stored frame')

    protocol = EvalProtocol(manifest=manifest, context_frames=8,
        horizons_frames=tuple(a.horizons), max_residues=a.max_residues,
        max_anchors_per_trajectory=4, temperatures=(320., 348., 379., 413., 450.),
        seed=a.seed, round_robin_proteins=True, round_robin_conditions=True)
    rows = []
    for window in iter_windows(protocol):
        if len(rows) >= a.max_windows:
            break
        rows.append(score_window(model, window, device=a.device, n_samples=a.n_samples,
                                 decode_chunk_size=a.decode_chunk_size))
        if len(rows) % 10 == 0:
            print(f'ordered paths {len(rows)}/{a.max_windows}', flush=True)

    # Aggregate: mean over windows within horizon for forecast/reference/persistence scalars,
    # plus the three E0.2 trajectory metrics.
    by_horizon = {}
    for h in sorted(a.horizons):
        hrows = [r for r in rows if r['horizon'] == h]
        if not hrows:
            continue
        keys = tuple(hrows[0]['forecast'].keys())
        arms = {}
        for arm in ('forecast', 'reference', 'persistence'):
            scores = {}
            for key in keys:
                vals = [r[arm].get(key) for r in hrows if r[arm].get(key) is not None]
                scores[key] = dict(mean=sum(vals) / len(vals) if vals else None, n=len(vals))
            arms[arm] = scores
        acf = [v for v in (_gap(r, 'distance_acf_lag1') for r in hrows) if v is not None]
        plen = [v for v in (_gap(r, 'path_length') for r in hrows) if v is not None]
        ff = [r['latent']['finite_frac'] for r in hrows if r['latent']['finite_frac'] is not None]
        mn = [r['latent']['max_norm'] for r in hrows if r['latent']['max_norm'] is not None]
        by_horizon[str(h)] = dict(
            n_windows=len(hrows),
            acf_error=dict(mean=sum(acf) / len(acf) if acf else None, n=len(acf)),
            path_error=dict(mean=sum(plen) / len(plen) if plen else None, n=len(plen)),
            latent_stable=dict(
                finite_frac_mean=sum(ff) / len(ff) if ff else None,
                max_norm_mean=sum(mn) / len(mn) if mn else None, n=len(ff)),
            arms=arms)

    result = dict(noise_mix_reconstruction=0.0,
                  note='noise_mix=0.0 reconstruction matching training launch; checkpoint args.noise_mix=0.8 is stale',
                  checkpoint=a.checkpoint, manifest=manifest, args=vars(a),
                  n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}),
                  test_opened=False, by_horizon=by_horizon,
                  per_window=rows)
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(n_windows=len(rows), n_proteins=result['n_proteins'])), flush=True)


if __name__ == '__main__':
    main()
