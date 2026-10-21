"""From-none ensemble evaluation: six distributional metrics vs persistence.

Drives extended_ensemble_summary (rmsf_mae, root_mean_w2, variance_w2,
md_pca_w2, weak_contact_j, transient_contacts_j) directly from a from-none
checkpoint, without requiring the frozen-encoder Stage-A/B1a stack.

The "true" ensemble for a window is the slice of the real MD trajectory
that spans the horizon (future_path frames from iter_windows), giving a
genuine multi-frame MD ensemble to compare against.  Persistence is the
anchor CA repeated N times (zero-variance delta distribution).

Usage:
  python -m evaluation.run_eval_fromnone_ensemble \\
      --ckpt artifacts/runs/atlas_100ps_anchor_v3/checkpoint_best.pt \\
      --manifest artifacts/atlas_manifest_100ps/val.jsonl \\
      --horizons-ns 1 4 16 64 --n-samples 32 --max-proteins 10 \\
      --device cuda --out results/atlas_anchor_v3_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from protein_jepa_frame.training.train_stage_cb1b import load_models_for_eval
from protein_jepa_frame.models.encoder import _canonical_frame
from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary

METRICS = ["rmsf_corr", "rmsf_mae", "root_mean_w2", "variance_w2", "md_pca_w2",
           "weak_contact_j", "transient_contacts_j"]
# True = higher is better
METRIC_DIR = {
    "rmsf_corr": True,
    "rmsf_mae": False, "root_mean_w2": False, "variance_w2": False,
    "md_pca_w2": False, "weak_contact_j": True, "transient_contacts_j": True,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--horizons-ns", type=float, nargs="+", default=[1, 4, 16, 64],
                   help="Evaluation horizons in ns (converted to frames via checkpoint step_ps).")
    p.add_argument("--n-samples", type=int, default=32,
                   help="Rollout replicas per window (ensemble size for model).")
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument("--max-anchors-per-trajectory", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap total evaluated windows (fair comparison with the Stage-A suite, "
                        "which reports over 100 windows).  Windows are taken in manifest order.")
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--pca-k", type=int, default=3)
    p.add_argument("--weak-lo", type=float, default=0.05)
    p.add_argument("--weak-hi", type=float, default=0.50)
    p.add_argument("--trans-lo", type=float, default=0.01)
    p.add_argument("--trans-hi", type=float, default=0.99)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def _encode_history(encoder, normalizer, coords, rmask, aatype, ridx, cid, device):
    """Encode K history frames with optional anchor conditioning. [1,K,L,4,3] -> [1,K,L,d]."""
    B, K, L = coords.shape[:3]
    anc_centroid = anc_basis = None
    if getattr(encoder, "anchor_conditioned", False):
        anchor_coords = coords[:, -1]          # [B, L, 4, 3]
        anchor_mask   = rmask[:, -1].bool()    # [B, L]
        cent, basis   = _canonical_frame(anchor_coords, anchor_mask)
        anc_centroid  = cent[:, None].expand(B, K, 3).reshape(B * K, 3)
        anc_basis     = basis[:, None].expand(B, K, 3, 3).reshape(B * K, 3, 3)

    H = encoder(
        coords.reshape(B * K, 1, L, 4, 3),
        rmask.reshape(B * K, 1, L),
        aatype[:, None].expand(B, K, L).reshape(B * K, L),
        ridx[:, None].expand(B, K, L).reshape(B * K, L),
        cid[:, None].expand(B, K, L).reshape(B * K, L),
        token_mask=rmask.reshape(B * K, 1, L),
        anchor_centroid=anc_centroid,
        anchor_basis=anc_basis,
    )[:, 0].reshape(B, K, L, -1)
    if normalizer is not None:
        H = normalizer(H)
    return H


@torch.no_grad()
def _rollout_ensemble(encoder, kernel, decoder, normalizer, ctx, rmask, n_samples, step_ps, n_steps, device, seed):
    """Run N stochastic rollouts and decode CA for each. Returns [N, L, 3]."""
    from protein_jepa_frame.models.encoder import _canonical_frame

    coords  = ctx["coords"].to(device).unsqueeze(0)   # [1,K,L,4,3]
    rmask_d = rmask.to(device).unsqueeze(0)            # [1,L]
    # repeat mask to [1,K,L] for encode_history
    K = coords.shape[1]; L = coords.shape[2]
    rmask_K = rmask_d.unsqueeze(1).expand(1, K, L)    # [1,K,L]
    aat  = ctx["aatype"].to(device).unsqueeze(0)       # [1,L]
    ridx = ctx["residue_index"].to(device).unsqueeze(0)
    cid  = ctx["chain_id"].to(device).unsqueeze(0)
    temp = ctx.get("temperature_k")
    if torch.is_tensor(temp):
        temp = temp.to(device).unsqueeze(0)
    dt_hist = None
    if "time_ps" in ctx:
        tp = ctx["time_ps"].to(device).unsqueeze(0)   # [1,K]
        dt = tp.diff(dim=-1)
        dt_hist = torch.cat([dt[:, :1], dt], dim=-1)  # [1,K]

    H_hist = _encode_history(encoder, normalizer, coords, rmask_K, aat, ridx, cid, device)
    H0     = H_hist[:, -1]                             # [1,L,d]
    anchor_ca = coords[:, -1, :, 1, :]                # [1,L,3]

    # Build anchor rotation matrix from backbone
    from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
    anchor_bb = coords[:, -1]                          # [1,L,4,3]
    R_anc, t_anc = RigidBodyFrame.from_backbone(
        anchor_bb[..., 0, :], anchor_bb[..., 1, :], anchor_bb[..., 2, :])

    samples = []
    for rep in range(n_samples):
        gen = torch.Generator(device=device).manual_seed(seed + rep)
        traj = kernel.rollout(
            H0, rmask_d, temp, n_steps=n_steps,
            H_hist=H_hist, dt_hist=dt_hist,
            seq_feature=H0, generator=gen, detach=True, return_bath=True,
        )
        Hf = traj[0][:, -1]    # [1,L,d]
        Sf = traj[1][:, -1] if traj[1] is not None else None
        out = decoder(H_future=Hf, H_context=H0,
                      X_context=anchor_ca, residue_mask=rmask_d,
                      R_anchor=R_anc, S_future=Sf)
        samples.append(out["ca_pred"][0])              # [L,3]
    return torch.stack(samples, dim=0)                 # [N,L,3]


def _bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=0):
    if not deltas:
        return [float("nan"), float("nan")]
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    N = t.numel()
    means = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, N, (N,), generator=g)
        means[b] = t[idx].mean()
    return [torch.quantile(means, alpha / 2).item(),
            torch.quantile(means, 1 - alpha / 2).item()]


def main(argv=None):
    args = parse_args(argv)
    dev  = _device(args.device)

    encoder, kernel, decoder, normalizer, ck = load_models_for_eval(args.ckpt, dev)
    step_ps = float(kernel.cfg.step_ps)
    print(f"[eval] ckpt={args.ckpt}  step_ps={step_ps}  anchor_conditioned="
          f"{getattr(encoder,'anchor_conditioned',False)}  n_samples={args.n_samples}")

    # Convert ns horizons to frames
    horizons_frames = tuple(
        max(1, int(round(ns * 1000.0 / step_ps))) for ns in args.horizons_ns
    )
    print(f"[eval] horizons (ns): {args.horizons_ns}  ->  frames: {horizons_frames}")

    proto = EvalProtocol(
        manifest=args.manifest,
        horizons_frames=horizons_frames,
        context_frames=args.history_frames,
        context_stride=1, frame_stride=1,
        max_residues=args.max_residues,
        max_proteins=args.max_proteins,
        max_anchors_per_trajectory=args.max_anchors_per_trajectory,
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False,
        seed=args.seed,
    )

    # per-horizon accumulators.  Windows are the unit of analysis (same protocol as
    # the Stage-A suite: mean over windows + bootstrap CI over windows), NOT per-protein
    # means -- with per-protein means a 4-protein sample gives a meaningless CI.
    per_hz: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_windows = 0
    n_skipped = 0

    for w in iter_windows(proto):
        if args.max_windows is not None and n_windows >= args.max_windows:
            break
        rmask = w.context["residue_mask"].all(dim=0).bool()   # [L]
        if rmask.sum() < 4:
            n_skipped += 1
            continue

        # --- true MD ensemble: future_path CA frames [T, L, 3] ---
        fp = getattr(w, "future_path", None)
        if fp is None:
            continue
        true_ca_path = fp["coords"][:, :, 1, :]               # [T, L, 3]
        # restrict to valid residues only for the metrics
        L_val = int(rmask.sum())
        true_ens = true_ca_path[:, rmask.cpu()].to(dev)       # [T, L_val, 3]
        if true_ens.shape[0] < 2:
            continue

        n_steps = w.horizon
        # --- model ensemble ---
        try:
            pred_ens = _rollout_ensemble(
                encoder, kernel, decoder, normalizer,
                w.context, rmask, args.n_samples, step_ps, n_steps, dev, args.seed,
            )[:, rmask.cpu()]                                   # [N, L_val, 3]
        except Exception as e:
            print(f"  skip {w.protein_id} h={w.horizon}: {e}")
            continue

        # --- persistence ensemble: anchor CA repeated ---
        anchor_ca = w.context["coords"][-1, :, 1, :]          # [L, 3]
        anchor_ca = anchor_ca[rmask.cpu()].to(dev)             # [L_val, 3]
        pers_ens  = anchor_ca.unsqueeze(0).expand(true_ens.shape[0], -1, -1)  # [T, L_val, 3]

        mask_val = None   # metrics use valid residues directly (already masked)

        s_model = extended_ensemble_summary(
            pred_ens.float(), true_ens.float(), mask_val,
            cutoff=args.cutoff, pca_k=args.pca_k,
            weak_lo=args.weak_lo, weak_hi=args.weak_hi,
            trans_lo=args.trans_lo, trans_hi=args.trans_hi,
        )
        s_pers = extended_ensemble_summary(
            pers_ens.float(), true_ens.float(), mask_val,
            cutoff=args.cutoff, pca_k=args.pca_k,
            weak_lo=args.weak_lo, weak_hi=args.weak_hi,
            trans_lo=args.trans_lo, trans_hi=args.trans_hi,
        )

        hz = w.horizon
        for m in METRICS:
            per_hz[hz]["model_" + m].append(s_model[m])
            per_hz[hz]["pers_"  + m].append(s_pers[m])

        n_windows += 1
        if n_windows % 20 == 0:
            print(f"  processed {n_windows} windows...")

    print(f"[eval] total windows: {n_windows}  skipped: {n_skipped}")

    # Aggregate: mean + bootstrap CI over windows (same protocol as Stage-A suite).
    # This gives a meaningful CI even with few proteins (100 windows >> 4 proteins).
    result = {
        "ckpt": str(args.ckpt), "manifest": args.manifest,
        "step_ps": step_ps, "n_samples": args.n_samples, "n_windows": n_windows,
        "horizons_ns": args.horizons_ns, "horizons_frames": list(horizons_frames),
        "by_horizon": {}
    }

    summary_rows = []
    for hz in sorted(per_hz):
        hz_ns = hz * step_ps / 1000.0
        hz_data = per_hz[hz]
        n_hz = len(hz_data["model_" + METRICS[0]])

        hz_res = {"horizon_ns": hz_ns, "n_windows": n_hz, "metrics": {}}
        for m in METRICS:
            model_vals = hz_data["model_" + m]
            pers_vals  = hz_data["pers_"  + m]
            better_up  = METRIC_DIR[m]
            deltas     = [(mv - pv if better_up else pv - mv)
                          for mv, pv in zip(model_vals, pers_vals)]
            dmean      = float(np.nanmean(deltas))
            ci         = _bootstrap_ci([d for d in deltas if not math.isnan(d)])
            hz_res["metrics"][m] = {
                "model":        float(np.nanmean(model_vals)),
                "persistence":  float(np.nanmean(pers_vals)),
                "improvement":  dmean,
                "ci95":         ci,
                "model_better": bool(dmean > 0),
                "win_rate":     float(np.nanmean([1.0 if d > 0 else 0.0 for d in deltas])),
            }
            summary_rows.append((hz_ns, m, float(np.nanmean(model_vals)),
                                  float(np.nanmean(pers_vals)), dmean,
                                  hz_res["metrics"][m]["win_rate"]))

        result["by_horizon"][str(hz)] = hz_res

    # Pretty table
    print()
    print(f"{'horizon':>8}  {'metric':>22}  {'model':>9}  {'persist':>9}  {'impr':>9}  {'win':>5}  better?")
    print("-" * 80)
    for hz_ns, m, mv, pv, imp, wr in summary_rows:
        ok = "✓" if imp > 0 else "✗"
        print(f"{hz_ns:>7.1f}  {m:>22}  {mv:>9.4f}  {pv:>9.4f}  {imp:>+9.4f}  {wr*100:>4.0f}%  {ok}")

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"\n[eval] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
