"""v3 ordered CA trajectories (ACF / Path / Latent-Stable) evaluation.

The v3 model (``train_stage_b_hs`` -> ``HSEnsembleProducer``) has no
``CADynamics`` entry (its checkpoint carries ``checkpoint_kind=None`` and no
``component_checkpoint``), so ``eval_ca_ordered_paths.py`` cannot load it. This
script adds that missing trajectory entry point for v3 by reusing the SAME
metric / panel / summary machinery (``ordered_path_summary``,
``temporal_controls``, ``_reference``, ``summarize``, ``panel_spec``,
``validate_panel``) so the produced numbers are directly comparable to the
exact-arm ``eval_ca_ordered_paths.py`` output.

How v3 is driven continuously (mirrors ``sample_ca_trajectory``):
  * ``hs.sample_future_ca(..., n_samples)`` performs ONE GLE rollout to
    ``hs._rollout_steps = horizon`` and returns ``H_traj`` [BN, horizon+1, L, dH].
  * Each H step is decoded to CA in the observed anchor gauge via
    ``_decode_path`` (``decoder.decode_structure`` -> ``backbone_coords[:, :, 1, :]``).
  * Time zero is overwritten with the observed anchor, exactly like the
    exact-arm adapter; predictions are never re-encoded or re-anchored to truth.

Only the val manifest is read; test.jsonl is never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import kabsch_align
from evaluation.loaders import sha256_file
from evaluation.trajectory_metrics_v2 import ordered_path_summary
from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest, _window_seed

# Import the exact-arm eval so we can reuse its metric/summary helpers verbatim.
from scripts import eval_ca_ordered_paths as base

TASK_VERSION = 'ca_ordered_paths_v3_development_v1'


def build_hs(args, dev):
    """Construct v3's HSEnsembleProducer exactly as eval_lasp_final.build does."""
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
    k_cfg = kck.get("kernel_config") or {}
    k_args = kck.get("args", {})
    k_cfg = dict(k_cfg)
    k_cfg.setdefault("bath_coupling", k_args.get("bath_coupling", "force"))
    k_cfg["step_ps"] = k_args.get("step_ps", 1000.0)
    k_cfg["rollout_grad_steps"] = None
    kern = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
    kern.load_state_dict(kck["kernel"])
    kern.eval()
    te = bool(kck.get("train_encoder", False)) and kck.get("encoder") is not None
    td = bool(kck.get("train_decoder", False)) and kck.get("decoder") is not None
    hs = HSEnsembleProducer(encoder, decoder, kern, train_encoder=te,
                            train_decoder=td,
                            history_conditioning=bool(k_args.get("history_conditioning", False))).to(dev).eval()
    if te:
        hs.encoder.load_state_dict(kck["encoder"], strict=False)
    if td:
        hs.decoder.load_state_dict(kck["decoder"], strict=False)
    hs._rollout_steps = int(k_args.get("rollout_steps", 8))
    hs._eval_path_ensemble = False
    hs._rollout_grad_steps = None
    return hs, k_args


@torch.no_grad()
def sample_v3_ca_trajectory(hs, observed, mask, aa, ri, ci, temperature, steps,
                            n_samples, generator, *, context_dt_ps=None):
    """One continuous v3 CA path per member, in the observed anchor gauge.

    Returns (ca [B, n_samples, steps+1, L, 3], latent [B, n_samples, steps+1, L, dH],
    step_ps). ``observed`` is [B, K, L, 4, 3]; anchor = observed[:, -1].
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('use no_grad, not inference_mode: energy forces require autograd')
    if hs.training:
        raise ValueError('trajectory evaluation requires model.eval()')
    if hs.history_conditioning:
        raise NotImplementedError('v3 trajectory adapter does not yet support history conditioning')
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    from protein_jepa_frame.training.train_stage_b_hs import _decode_path
    b, k, length = observed.shape[:3]
    device = observed.device
    step_ps = float(hs.kernel.cfg.step_ps)
    if context_dt_ps is not None and abs(float(context_dt_ps) - step_ps) > 1e-9:
        raise ValueError('observed context spacing must equal kernel step_ps')

    anchor = observed[:, -1]                                    # [B, L, 4, 3]
    R, t = _extract_backbone_frames(anchor)                     # [B, L, 3, 3], [B, L, 3]
    # v3's rollout length is a module attribute (not a per-call argument); pin it
    # to this window's horizon so the continuous path spans exactly steps+1 frames.
    hs._rollout_steps = steps
    out = hs.sample_future_ca(
        observed, mask, aa, ri, ci, R, t,
        n_samples=n_samples, generator=generator,
        temperature_k=temperature)
    # H_traj: [B*n, steps+1, L, dH]. Decode every step to CA in anchor gauge.
    idx = torch.arange(steps + 1, device=device)
    ca_path = _decode_path(hs, out['H_traj'], R, t, aa, mask,
                           b, n_samples, length, None, idx)     # [B, n, steps+1, L, 3]
    ca_path[:, :, 0] = anchor[:, None, :, 1, :]                 # t0 = observed anchor CA
    latent = out['H_traj'].reshape(b, n_samples, steps + 1, length, -1).detach()
    return ca_path.detach(), latent, step_ps


@torch.no_grad()
def score_window(hs, window, *, device='cpu', n_samples=12, decode_chunk_size=64):
    if hs.training:
        raise ValueError('requires an eval-mode v3 producer')
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError('joint path U-scores require at least two independent forecast members')
    context = window.context
    observed = context['coords'].to(device)
    mask = context['residue_mask'].to(device).all(0).bool()
    if mask.shape != (observed.shape[1],) or int(mask.sum()) < 4:
        raise ValueError('requires at least four observed-valid residues')
    if not bool(torch.isfinite(observed[:, mask]).all()):
        raise ValueError('nonfinite valid observed backbone')
    aa, ri, ci = [context[key].to(device) for key in ('aatype', 'residue_index', 'chain_id')]
    seed = _window_seed(0, window)
    # The single sampling call; future data is not read until paths are formed.
    ca, latent, step_ps = sample_v3_ca_trajectory(
        hs, observed[None], mask[None], aa[None], ri[None], ci[None],
        torch.tensor([window.temperature_k], dtype=observed.dtype, device=device),
        window.horizon, n_samples, torch.Generator(device=device).manual_seed(seed),
        context_dt_ps=None)
    expected = (1, n_samples, window.horizon + 1, len(mask), 3)
    if ca.shape != expected:
        raise ValueError('v3 trajectory adapter returned unexpected CA shape')
    forecast = ca[0].detach().cpu().double()
    mask, ri, ci = mask.cpu(), ri.cpu(), ci.cpu()
    anchor = observed[-1, :, 1].detach().cpu().double()
    if not torch.equal(forecast[:, 0, mask], anchor[mask].expand(n_samples, -1, -1)):
        raise ValueError('forecast time zero differs from the observed anchor')

    observed_times = torch.as_tensor(context['time_ps'], dtype=torch.float64).cpu()
    relative_times = torch.arange(window.horizon + 1, dtype=torch.float64) * step_ps
    controls, control_metadata = base.temporal_controls(forecast, seed)
    reference, aligned_reference, raw_reference_mask, reference_times = base._reference(
        window, observed_times, relative_times, mask, anchor)
    paths = {'forecast': forecast, **controls,
        'persistence': anchor[None, None].expand(n_samples, window.horizon + 1, -1, -1),
        'reference_MD': aligned_reference[None]}
    arms = {}
    for name, coordinates in paths.items():
        metrics = ordered_path_summary(coordinates, reference, anchor, mask,
            predicted_times_ps=relative_times, reference_times_ps=reference_times - observed_times[-1],
            reference_mask=mask[None].expand(window.horizon + 1, -1),
            residue_index=ri, chain_id=ci, lags=base.LAGS, max_pairs=base.MAX_PAIRS)
        arms[name] = dict(role=base.ROLES[name], n_paths=len(coordinates), metrics=metrics)
    fingerprints = dict(
        observed=state_sha256({key: context[key] for key in (
            'coords', 'residue_mask', 'aatype', 'residue_index', 'chain_id', 'time_ps')}),
        reference=state_sha256(dict(ca=reference, residue_mask=raw_reference_mask, time_ps=reference_times)),
        forecast_CA_path=state_sha256({'ca': forecast}))
    return dict(protein=window.protein_id, trajectory=window.trajectory_id,
        temperature_k=float(window.temperature_k), anchor_frame=int(window.anchor_frame),
        target_frame=int(window.target_frame), horizon=int(window.horizon), horizon_ps=float(window.horizon_ps),
        context_frames=torch.as_tensor(window.context_frames).tolist(), valid_residues=int(mask.sum()),
        sampling_seed=seed, n_forecast_paths=n_samples, n_reference_paths=1,
        time_grid=dict(step_ps=step_ps, n_frames=window.horizon + 1,
                       anchor_time_ps=float(observed_times[-1]), endpoint_time_ps=float(reference_times[-1])),
        continuous_path=dict(n_sample_calls=1, future_reencodings=0, bath_resets=0,
                             includes_sampled_S0=True,
                             latent_shape=list(latent.shape),
                             bath_shape=None),
        input_fingerprints=fingerprints, temporal_controls=control_metadata, arms=arms)


def run(args):
    args.manifest = _development_manifest(args.manifest)
    panel = base.panel_spec(args.horizons, args.max_windows)
    if args.torch_threads < 1:
        raise ValueError('thread count must be positive')
    output = Path(args.out)
    if output.exists():
        raise FileExistsError('refusing to overwrite an ordered-path artifact')
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    hs, k_args = build_hs(args, args.device)
    if float(hs.kernel.cfg.step_ps) != 1000.:
        raise ValueError('these fixed development panels require 1000 ps per stored frame')
    hs._rollout_steps = max(panel['horizons'])
    protocol = EvalProtocol(manifest=args.manifest, context_frames=8,
        horizons_frames=panel['horizons'], max_residues=96, max_anchors_per_trajectory=4,
        temperatures=(320., 348., 379., 413., 450.), seed=0,
        round_robin_proteins=True, round_robin_conditions=True)
    report = dict(task_version=TASK_VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
        dataset_role='development_validation', args=vars(args), panel=panel,
        protocol=asdict(protocol), arm_roles=base.ROLES,
        model=dict(step_ps=float(hs.kernel.cfg.step_ps),
                   bath_dim=int(hs.kernel.cfg.bath_dim),
                   history_conditioning=bool(hs.history_conditioning),
                   rollout_steps=int(hs._rollout_steps)),
        test_opened=False, future_information_given_to_sampler=False,
        sampling=dict(seed=0, n_paths=12, prediction_kind='endpoint',
                      one_continuous_rollout_per_member=True, no_future_reencoding=True),
        metrics=dict(arithmetic='CPU float64', registered_lags=list(base.LAGS), max_physical_pairs=base.MAX_PAIRS))
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in iter_windows(protocol):
        if len(rows) == panel['max_windows']:
            break
        rows.append(score_window(hs, window, device=args.device, n_samples=12,
                                 decode_chunk_size=args.decode_chunk_size))
        if len(rows) % 10 == 0:
            print(f'Ordered paths v3 {len(rows)}/{panel["max_windows"]}', flush=True)
    base.validate_panel(rows, panel)
    report.update(completed_utc=datetime.now(timezone.utc).isoformat(),
        n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}),
        per_window=rows, by_horizon=base.summarize(rows))
    _write_json_exclusive(output, report)
    print(json.dumps(dict(output=str(output), panel=panel['name'], n_windows=len(rows),
                          n_proteins=report['n_proteins'])), flush=True)
    return report


def _write_json_exclusive(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-ckpt', required=True,
                        help='v3 kernel checkpoint (artifacts/lasp_v3_velocity/checkpoint_best.pt)')
    parser.add_argument('--stage-a-ckpt', required=True)
    parser.add_argument('--b1a-ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--horizons', nargs='+', type=int, default=list(base.DEFAULT_HORIZONS))
    parser.add_argument('--max-windows', type=int, default=None)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--decode-chunk-size', type=int, default=64)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
