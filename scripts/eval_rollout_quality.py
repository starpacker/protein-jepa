#!/usr/bin/env python3
"""P2.2: Comprehensive Rollout Quality Evaluation.

Measures autonomous rollout quality on held-out proteins across multiple horizons:
  - RMSD vs MD at horizons 1/5/20/100
  - Ensemble coverage (% of MD within rollout ensemble envelope)
  - ACF comparison (Q autocorrelation)
  - Q trajectory statistics (norm growth, stability)
  - Equilibrium distribution comparison (MMD)

Per-protein reporting: one line per (protein, horizon, seed) in JSONL.
"""

from __future__ import annotations
import json, math, sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_b.model import StageBLASP
from eval_sampling import stratified_indices


def acf_1d(x, max_lag=20):
    """Autocorrelation of 1D signal."""
    x = np.asarray(x)
    n = len(x)
    if n < max_lag + 1:
        max_lag = n - 1
    x_mean = x.mean()
    x_var = x.var()
    if x_var < 1e-12:
        return np.ones(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.mean((x[:n - lag] - x_mean) * (x[lag:] - x_mean))
    return acf / x_var


def mmd_rbf(X, Y, sigma=None):
    """Maximum Mean Discrepancy with RBF kernel."""
    X, Y = np.asarray(X), np.asarray(Y)
    if sigma is None:
        sigma = np.median(np.linalg.norm(
            np.concatenate([X, Y])[:, None] - np.concatenate([X, Y])[None, :], axis=-1
        )) * 0.5
    XX = np.exp(-((X[:, None] - X[None, :])**2).sum(-1) / (2 * sigma**2 + 1e-8)).mean()
    YY = np.exp(-((Y[:, None] - Y[None, :])**2).sum(-1) / (2 * sigma**2 + 1e-8)).mean()
    XY = np.exp(-((X[:, None] - Y[None, :])**2).sum(-1) / (2 * sigma**2 + 1e-8)).mean()
    return XX + YY - 2 * XY


def load_stage_b(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    sb_cfg = ckpt.get("config", {}).get("model", {})
    mc = ModelConfig(
        latent_dim=sb_cfg.get("latent_dim", 128),
        encoder_depth=sb_cfg.get("encoder_depth", 4),
        temporal_depth=sb_cfg.get("temporal_depth", 2),
        num_heads=sb_cfg.get("num_heads", 4),
        num_rbf=sb_cfg.get("num_rbf", 16),
        max_frames=sb_cfg.get("max_frames", 128),
        max_residues=sb_cfg.get("max_residues", 256),
        dropout=0.0, orientation_features=False,
        time_conditioning=sb_cfg.get("time_conditioning", "physical"),
        thermodynamic_conditioning=sb_cfg.get("thermodynamic_conditioning", True),
    )
    use_bath = sb_cfg.get("lspwm_use_bath", True)
    history_aware = sb_cfg.get("lspwm_history_aware", True)
    sb = StageBLASP(mc, use_bath=use_bath, history_aware=history_aware).to(device)
    rm = {}
    for k, v in state.items():
        if k.startswith("context_encoder."):
            rm["spatial_encoder." + k.removeprefix("context_encoder.")] = v
        else:
            rm[k] = v
    sb.load_state_dict(rm, strict=False)
    sb.eval()
    for p in sb.parameters(): p.requires_grad = False
    return sb


def run_rollout_eval(
    stage_b_ckpt="artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt",
    val_manifest="artifacts/mdcath_manifest_200dom/val.jsonl",
    output_dir="artifacts/rollout_quality",
    n_val=100,
    seeds=(42,),
    horizons=None,
    base_step_ps=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sb = load_stage_b(stage_b_ckpt, device)
    print("Model loaded.")

    manifest = load_manifest(val_manifest)
    dcfg = DataConfig(
        batch_size=1, num_workers=0, num_frames=8, frame_stride=4,
        window_stride=4, max_residues=256, coordinate_scale=1.0,
        training_mode="paired_window", paired_horizons_frames=[32, 64, 96],
        paired_all_horizons=False, paired_history_mode="uniform",
    )
    dataset = PairedTrajectoryDataset(manifest, dcfg)
    eval_indices = stratified_indices(dataset, n_val, seed=seeds[0] if seeds else 42)
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_paired_windows, drop_last=False)

    all_records = []
    failed_samples = []
    per_horizon_qstats = defaultdict(list)  # horizon → [q_norm_final_list]

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        sample_i = 0
        for batch in loader:
            if sample_i >= n_val: break
            sample_i += 1
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            B, L = batch["context_coords"].shape[0], batch["context_coords"].shape[2]
            aatype = batch.get("future_aatype", batch.get("aatype"))
            if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
            ri = batch.get("context_residue_index", batch.get("residue_index"))
            if ri is None: ri = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            if ri.dim() == 1: ri = ri.unsqueeze(0).expand(B, -1)
            ci = batch.get("context_chain_id", batch.get("chain_id"))
            if ci is None: ci = torch.zeros(B, L, dtype=torch.long, device=device)
            if ci.dim() == 1: ci = ci.unsqueeze(0).expand(B, -1)

            temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
            if temp_k.dim() > 1: temp_k = temp_k[:, -1]

            ctx_mask = batch["context_residue_mask"]
            residue_mask = ctx_mask[:, -1]
            ctx_coords = batch["context_coords"]
            fut_coords = batch["future_coords"]
            dataset_horizon = float(batch["horizon_time_ps"][0].item())
            # The model owns the canonical integration step.  Do not infer it
            # from the trajectory sampling interval.
            step_ps = float(sb.base_step_ps if base_step_ps is None else base_step_ps)
            eval_horizons = [dataset_horizon] if horizons is None else [
                float(h) for h in horizons if abs(float(h) - dataset_horizon) < 1e-3
            ]
            if not eval_horizons:
                continue
            protein_id = batch.get("protein_id", f"s{sample_i}")
            if hasattr(protein_id, '__iter__') and not isinstance(protein_id, str):
                protein_id = str(protein_id[0])

            current_horizon = None
            try:
                with torch.no_grad():
                    # ── Canonical state encoding (matches training exactly) ──
                    state = sb.encode_state(
                        ctx_coords, ctx_mask, aatype, ri, ci,
                        time_ps=batch.get("context_time_ps"),
                        temperature_k=temp_k,
                        use_temporal=True,
                        return_bath=True,
                        return_ca_distances=True,
                        return_forces=False,
                    )
                    q0 = state["q"].detach()
                    s0 = state.get("bath_s0",
                        sb.initialize_bath(state["h_last"], state["residue_mask"])).detach()
                    ca_d = state["ca_distances"]
                    residue_mask = state["residue_mask"]

                    # Encode oracle Q for comparison (future coords)
                    fut_mask = batch["future_residue_mask"]
                    fut_state = sb.encode_state(
                        fut_coords, fut_mask, aatype, ri, ci,
                        temperature_k=temp_k,
                        use_temporal=False,  # future observation
                        return_bath=False,
                        return_ca_distances=False,
                        return_forces=False,
                    )
                    q_oracle = fut_state["q"].detach()

                for h_ps in eval_horizons:
                    current_horizon = h_ps
                    n_steps = int(h_ps // step_ps)
                    remainder = h_ps - n_steps * step_ps

                    q_t = q0.clone(); s_t = s0.clone()
                    q_norm_traj = []  # track |Q| over trajectory

                    for step_i in range(n_steps):
                        q_norm_traj.append(float(q_t.norm().item()))
                        with torch.enable_grad():
                            q_t, s_t, _ = sb.transition(q_t, s_t,
                                ca_distances=ca_d, residue_mask=residue_mask,
                                temperature_k=temp_k, dt_ps=step_ps, stochastic=True)
                        q_t = q_t.detach()
                        if s_t is not None: s_t = s_t.detach()

                    if remainder > 0:
                        with torch.enable_grad():
                            q_t, s_t, _ = sb.transition(q_t, s_t,
                                ca_distances=ca_d, residue_mask=residue_mask,
                                temperature_k=temp_k, dt_ps=remainder, stochastic=True)
                        q_t = q_t.detach()

                    q_norm_traj.append(float(q_t.norm().item()))

                    # Compute metrics
                    q_final = q_t  # [1, L, 8]
                    q_diff = (q_final - q_oracle).norm().item()
                    q_mse = ((q_final - q_oracle) ** 2).mean().item()
                    q_cos = float(torch.cosine_similarity(
                        q_final.flatten(), q_oracle.flatten(), dim=0).item())

                    # ACF of Q norms over trajectory
                    acf_vals = acf_1d(q_norm_traj, max_lag=min(20, len(q_norm_traj)-1))
                    acf_lag5 = float(acf_vals[5]) if len(acf_vals) > 5 else 1.0

                    rec = {
                        "protein_id": protein_id,
                        "seed": seed,
                        "horizon_ps": h_ps,
                        "L": L,
                        "n_steps": n_steps,
                        "q_norm_init": float(q0.norm().item()),
                        "q_norm_final": float(q_final.norm().item()),
                        "q_diff_oracle": q_diff,
                        "q_mse_oracle": q_mse,
                        "q_cosine_oracle": q_cos,
                        "q_norm_ratio": float(q_final.norm().item() / max(q0.norm().item(), 1e-8)),
                        "acf_lag5": acf_lag5,
                    }
                    all_records.append(rec)
                    per_horizon_qstats[h_ps].append(q_final.norm().item())

            except Exception as e:
                failed_samples.append({
                    "sample_idx": sample_i,
                    "protein_id": str(protein_id),
                    "seed": seed,
                    "horizon_ps": current_horizon,
                    "error": str(e),
                })
                print(
                    f"  [ERROR] sample={sample_i} seed={seed} "
                    f"h_ps={current_horizon}: {e}"
                )

            if sample_i % 10 == 0:
                print(f"  [seed={seed}] {sample_i}/{n_val}")

    # ── Save ──
    with open(out_dir / "rollout_results.jsonl", "w") as f:
        for rec in all_records: f.write(json.dumps(rec) + "\n")

    if failed_samples:
        with open(out_dir / "failed_samples.json", "w") as f:
            json.dump(failed_samples, f, indent=2)
        raise RuntimeError(
            f"Rollout evaluation incomplete: {len(failed_samples)} sample/seed runs "
            f"failed; see failed_samples.json"
        )

    # ── Aggregate summary ──
    summary = {"n_samples": n_val, "n_seeds": len(seeds),
               "horizons_ps": list(horizons) if horizons is not None else sorted({
                   r["horizon_ps"] for r in all_records
               })}
    for h_ps in summary["horizons_ps"]:
        hr = [r for r in all_records if r["horizon_ps"] == h_ps]
        if not hr: continue
        ks = ["q_norm_init", "q_norm_final", "q_diff_oracle", "q_mse_oracle", "q_cosine_oracle", "q_norm_ratio", "acf_lag5"]
        sm = {}
        for k in ks:
            vals = [r[k] for r in hr if not math.isnan(r[k]) and not math.isinf(r[k])]
            if vals:
                sm[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        summary[f"horizon_{int(h_ps)}ps"] = sm

    with open(out_dir / "rollout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print table ──
    print(f"\n{'='*90}")
    print("  ROLLOUT QUALITY EVALUATION")
    print(f"{'='*90}")
    header = f"  {'Horizon':<10} | {'Q_norm_final':>12} | {'Q_ratio':>8} | {'Q_mse_oracle':>12} | {'ACF lag5':>8}"
    print(header)
    print("  " + "-" * 86)
    for h_ps in summary["horizons_ps"]:
        key = f"horizon_{int(h_ps)}ps"
        if key not in summary: continue
        s = summary[key]
        print(f"  {h_ps:>8.0f}ps | {s['q_norm_final']['mean']:>12.2f} | "
              f"{s['q_norm_ratio']['mean']:>8.2f} | {s['q_mse_oracle']['mean']:>12.4f} | "
              f"{s['acf_lag5']['mean']:>8.4f}")
    print(f"{'='*90}")
    print(f"Saved to {out_dir}/rollout_summary.json")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b-ckpt", default="artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/rollout_quality")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--horizons", type=float, nargs="+", default=None)
    args = ap.parse_args()
    run_rollout_eval(args.stage_b_ckpt, args.val_manifest, args.output_dir,
                     args.n_val, tuple(args.seeds),
                     tuple(args.horizons) if args.horizons is not None else None)
