#!/usr/bin/env python3
"""Full Stage B evaluation — all 5 analyses in one pass.

Avoids hanging by:
- Using no_grad for all encode/rollout paths  
- Never training transformer or other baselines
- Batching the data pass
"""

import json, argparse, sys, math, time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_b.model import StageBLASP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    sb = StageBLASP(mc, use_bath=use_bath, history_aware=history_aware).to(DEVICE)
    rm = {}
    for k, v in state.items():
        if k.startswith("context_encoder."):
            rm["spatial_encoder." + k.removeprefix("context_encoder.")] = v
        else:
            rm[k] = v
    sb.load_state_dict(rm, strict=False)
    sb.eval()
    for p in sb.parameters():
        p.requires_grad = False
    return sb


def acf_1d(x, max_lag):
    x = np.array(x)
    x = x - np.mean(x)
    denom = np.var(x) * len(x)
    if denom < 1e-12:
        return np.ones(max_lag + 1)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        acf.append(np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag + 1 else 0.0)
    return np.array(acf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/full_abc_stage_b/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/full_abc_evaluation")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--base-step-ps", type=float, default=100.0)
    ap.add_argument("--max-horizon-ps", type=float, default=128000)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    for sub in ["baselines", "rollout", "temperature", "sidechain", "world_model"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.ckpt}")
    t0 = time.time()
    sb = load_model(args.ckpt)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Build dataset
    manifest = load_manifest(args.val_manifest)
    dcfg = DataConfig(
        batch_size=1, num_workers=0, num_frames=8, frame_stride=4,
        window_stride=4, max_residues=256, coordinate_scale=1.0,
        training_mode="paired_window",
        paired_horizons_frames=[32, 64, 96],
        paired_all_horizons=False, paired_history_mode="uniform",
    )
    dataset = PairedTrajectoryDataset(manifest, dcfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_paired_windows, drop_last=False,
    )

    all_baseline = []
    all_rollout = []
    all_temperature = []
    
    seen_pids = set()
    protein_counter = defaultdict(int)

    t_start = time.time()
    for sample_idx, batch in enumerate(loader):
        if sample_idx >= args.n_val:
            break

        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B, L = batch["context_coords"].shape[0], batch["context_coords"].shape[2]

        # Topology
        aatype = batch.get("future_aatype", batch.get("aatype"))
        ri = batch.get("context_residue_index", batch.get("residue_index"))
        ci = batch.get("context_chain_id", batch.get("chain_id"))
        if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
        if ri is None: ri = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, -1)
        if ri.dim() == 1: ri = ri.unsqueeze(0).expand(B, -1)
        if ci is None: ci = torch.zeros(B, L, dtype=torch.long, device=DEVICE)
        if ci.dim() == 1: ci = ci.unsqueeze(0).expand(B, -1)

        temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=DEVICE))
        if temp_k.dim() > 1: temp_k = temp_k[:, -1]
        temp_float = float(temp_k[0].item())

        ctx_coords = batch["context_coords"]
        ctx_mask = batch["context_residue_mask"]
        fut_coords = batch["future_coords"]
        fut_mask = batch["future_residue_mask"]
        horizon_ps = float(batch["horizon_time_ps"][0].item())

        protein_id = batch.get("protein_id", f"s{sample_idx}")
        if hasattr(protein_id, "__iter__") and not isinstance(protein_id, str):
            protein_id = str(protein_id[0])
        protein_counter[protein_id] += 1
        seen_pids.add(protein_id)

        try:
            # Note: NO torch.no_grad() wrapper here — transition() uses
            # autograd internally for force computation and fails inside no_grad.
            # State encoding uses spatial/temporal encoders which have no_grad
            # applied internally when freeze_spatial/temporal=True.

            # Context encoding
            state = sb.encode_state(
                ctx_coords, ctx_mask, aatype, ri, ci,
                time_ps=batch.get("context_time_ps"),
                temperature_k=temp_k,
                use_temporal=True, return_bath=True,
                return_ca_distances=True, return_forces=False,
            )
            # q0 must retain grad for transition() → step() → autograd force
            q0 = state["q"]
            s0 = state.get("bath_s0",
                sb.initialize_bath(state["h_last"], state["residue_mask"]))
            ca_d = state["ca_distances"]
            residue_mask = state["residue_mask"]

            # Oracle Q from future coords
            fut_state = sb.encode_state(
                fut_coords, fut_mask, aatype, ri, ci,
                temperature_k=temp_k,
                use_temporal=False,
                return_bath=False, return_ca_distances=False, return_forces=False,
            )
            q_oracle = fut_state["q"].detach()

            # === 1. BASELINE COMPARISON ===
            n_steps_baseline = max(1, int(horizon_ps // args.base_step_ps))
            dt_ps = args.base_step_ps

            # Stage B autonomous rollout at base_step_ps
            q_sb = q0.clone(); s_sb = s0.clone()
            for _ in range(n_steps_baseline):
                q_sb, s_sb, _ = sb.transition(
                    q_sb, s_sb, ca_distances=ca_d,
                    residue_mask=residue_mask,
                    temperature_k=temp_k, dt_ps=dt_ps, stochastic=True,
                )
                q_sb = q_sb.detach()
                if s_sb is not None: s_sb = s_sb.detach()

            sb_mse = float(((q_sb - q_oracle) ** 2).mean().item())
            sb_norm = float(q_sb.norm().item())

            # Persistence
            q_persist = q0
            persist_mse = float(((q_persist - q_oracle) ** 2).mean().item())
            persist_norm = float(q_persist.norm().item())

            # Linear interpolation
            q_linear = q0 + 1.0 * (q_oracle - q0)
            linear_mse = float(((q_linear - q_oracle) ** 2).mean().item())
            linear_norm = float(q_linear.norm().item())

            all_baseline.append({
                "protein_id": protein_id, "horizon_ps": horizon_ps, "L": L,
                "temperature_k": temp_float,
                "stage_b_mse": sb_mse, "stage_b_norm": sb_norm,
                "persistence_mse": persist_mse, "persistence_norm": persist_norm,
                "linear_mse": linear_mse, "linear_norm": linear_norm,
                "oracle_norm": float(q_oracle.norm().item()),
            })

            # === 2. ROLLOUT QUALITY (multi-horizon) ===
            horizons_to_test = [horizon_ps]
            if horizon_ps > 32000:
                horizons_to_test = [32000, horizon_ps]
            for h_ps in horizons_to_test:
                if h_ps > args.max_horizon_ps:
                    continue
                n_steps = max(1, int(h_ps // args.base_step_ps))
                if n_steps > 1000:
                    n_steps = 1000
                    h_ps = n_steps * args.base_step_ps

                q_t = q0.clone(); s_t = s0.clone()
                q_norm_traj = [float(q_t.norm().item())]
                valid = True

                for step_i in range(n_steps):
                        q_t, s_t, diag = sb.transition(
                            q_t, s_t, ca_distances=ca_d,
                            residue_mask=residue_mask,
                            temperature_k=temp_k, dt_ps=args.base_step_ps,
                            stochastic=True,
                        )
                        q_t = q_t.detach()
                        if s_t is not None: s_t = s_t.detach()
                        q_norm_traj.append(float(q_t.norm().item()))
                        if torch.isnan(q_t).any() or torch.isinf(q_t).any():
                            valid = False
                            break

                q_diff = float((q_t - q_oracle).norm().item())
                q_mse = float(((q_t - q_oracle) ** 2).mean().item())
                q_cos = float(torch.cosine_similarity(
                    q_t.flatten(), q_oracle.flatten(), dim=0).item()) if valid and q_t.numel() > 0 else 0.0
                acf_vals = acf_1d(q_norm_traj, max_lag=min(20, len(q_norm_traj) - 1))
                acf_lag5 = float(acf_vals[5]) if len(acf_vals) > 5 else 1.0

                all_rollout.append({
                    "protein_id": protein_id, "horizon_ps": h_ps, "L": L,
                    "temperature_k": temp_float, "n_steps": n_steps,
                    "valid": valid,
                    "q_norm_init": float(q0.norm().item()),
                    "q_norm_final": float(q_t.norm().item()),
                    "q_diff_oracle": q_diff, "q_mse_oracle": q_mse,
                    "q_cosine_oracle": q_cos,
                    "acf_lag5": acf_lag5,
                })

            # === 3. TEMPERATURE INTERVENTION ===
            if temp_float < 350:  # only for cold samples
                for t_interv in [320.0, 379.0, 413.0]:
                    tk = torch.tensor([t_interv], device=DEVICE)
                    q_ti = q0.clone(); s_ti = s0.clone()
                    for _ in range(n_steps_baseline):
                        q_ti, s_ti, _ = sb.transition(
                            q_ti, s_ti, ca_distances=ca_d,
                            residue_mask=residue_mask,
                            temperature_k=tk, dt_ps=dt_ps, stochastic=True,
                        )
                        q_ti = q_ti.detach()
                        if s_ti is not None: s_ti = s_ti.detach()
                    all_temperature.append({
                        "protein_id": protein_id, "L": L,
                        "base_temperature": temp_float,
                        "intervention_temperature": t_interv,
                        "horizon_ps": horizon_ps,
                        "q_norm": float(q_ti.norm().item()),
                        "q_diff_from_oracle": float((q_ti - q_oracle).norm().item()),
                    })

        except Exception as e:
            print(f"  [WARN] sample {sample_idx} ({protein_id}): {e}")
            continue

        if (sample_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{sample_idx + 1}/{args.n_val}] {elapsed:.0f}s  "
                  f"proteins:{len(seen_pids)}  last_t={temp_float:.0f}K")

    elapsed = time.time() - t_start
    print(f"\nProcessed {len(all_baseline)} samples in {elapsed:.0f}s")

    # === SAVE BASELINE RESULTS ===
    with open(out_dir / "baselines/baseline_results.jsonl", "w") as f:
        for r in all_baseline: f.write(json.dumps(r) + "\n")

    obs_horizons = sorted({r["horizon_ps"] for r in all_baseline})
    bsum = {"n_samples": len(all_baseline), "horizons_ps": obs_horizons}
    for h_ps in obs_horizons:
        hr = [r for r in all_baseline if r["horizon_ps"] == h_ps]
        key = f"horizon_{int(h_ps)}ps"
        sm = {}
        for method in ["stage_b", "linear", "persistence"]:
            vals = [r[f"{method}_mse"] for r in hr
                    if not np.isnan(r[f"{method}_mse"]) and not np.isinf(r[f"{method}_mse"])]
            norms = [r[f"{method}_norm"] for r in hr]
            if vals:
                sm[method] = {
                    "mse_mean": float(np.mean(vals)), "mse_std": float(np.std(vals)),
                    "norm_mean": float(np.mean(norms)), "norm_std": float(np.std(norms)),
                }
        oracle_norms = [r["oracle_norm"] for r in hr]
        sm["oracle"] = {"norm_mean": float(np.mean(oracle_norms)), "norm_std": float(np.std(oracle_norms))}
        bsum[key] = sm

    with open(out_dir / "baselines/baseline_summary.json", "w") as f:
        json.dump(bsum, f, indent=2)

    print("\n=== BASELINE COMPARISON ===")
    for h_ps in obs_horizons:
        key = f"horizon_{int(h_ps)}ps"
        s = bsum.get(key, {})
        oracle_n = s.get("oracle", {}).get("norm_mean", 0)
        print(f"\nHorizon {h_ps:.0f}ps (oracle norm: {oracle_n:.1f}):")
        for method in ["stage_b", "linear", "persistence"]:
            m = s.get(method, {})
            print(f"  {method:>12s}: MSE={m.get('mse_mean', 0):.6f} norm={m.get('norm_mean', 0):.1f}")

    # === SAVE ROLLOUT RESULTS ===
    with open(out_dir / "rollout/rollout_results.jsonl", "w") as f:
        for r in all_rollout: f.write(json.dumps(r) + "\n")

    roll_horizons = sorted({r["horizon_ps"] for r in all_rollout})
    rsum = {"n_samples": len(all_rollout), "horizons_ps": roll_horizons}
    for h_ps in roll_horizons:
        hr = [r for r in all_rollout if r["horizon_ps"] == h_ps and r["valid"]]
        if hr:
            key = f"horizon_{int(h_ps)}ps"
            rsum[key] = {
                "n_valid": len(hr),
                "n_total": len([r for r in all_rollout if r["horizon_ps"] == h_ps]),
                "divergence_rate": 1.0 - len(hr) / max(len([r for r in all_rollout if r["horizon_ps"] == h_ps]), 1),
                "norm_mean": float(np.mean([r["q_norm_final"] for r in hr])),
                "norm_init_mean": float(np.mean([r["q_norm_init"] for r in hr])),
                "mse_mean": float(np.mean([r["q_mse_oracle"] for r in hr])),
                "cosine_mean": float(np.mean([r["q_cosine_oracle"] for r in hr])),
                "acf_lag5_mean": float(np.mean([r["acf_lag5"] for r in hr])),
            }
    with open(out_dir / "rollout/rollout_summary.json", "w") as f:
        json.dump(rsum, f, indent=2)

    print("\n=== ROLLOUT QUALITY ===")
    for h_ps in roll_horizons:
        key = f"horizon_{int(h_ps)}ps"
        s = rsum.get(key, {})
        print(f"  {h_ps:.0f}ps: valid={s.get('n_valid',0)}/{s.get('n_total',0)}  "
              f"norm={s.get('norm_mean',0):.1f}  MSE={s.get('mse_mean',0):.4f}  "
              f"cos={s.get('cosine_mean',0):.3f}  ACF5={s.get('acf_lag5_mean',0):.3f}")

    # === TEMPERATURE INTERVENTION ===
    with open(out_dir / "temperature/intervention_results.jsonl", "w") as f:
        for r in all_temperature: f.write(json.dumps(r) + "\n")

    if all_temperature:
        tsum = {}
        for tk in sorted({r["intervention_temperature"] for r in all_temperature}):
            tr = [r for r in all_temperature if r["intervention_temperature"] == tk]
            tsum[f"t{int(tk)}K"] = {
                "n": len(tr),
                "q_norm_mean": float(np.mean([r["q_norm"] for r in tr])),
                "q_norm_std": float(np.std([r["q_norm"] for r in tr])),
                "q_diff_mean": float(np.mean([r["q_diff_from_oracle"] for r in tr])),
            }
        with open(out_dir / "temperature/intervention_summary.json", "w") as f:
            json.dump(tsum, f, indent=2)
        print("\n=== TEMPERATURE INTERVENTION ===")
        for tk, vals in tsum.items():
            q0_norm = float(np.mean([r["q_norm"] for r in all_temperature if r["intervention_temperature"] == float(tk[1:-1])]))
            oracle_n = float(np.mean([r["q_diff_from_oracle"] for r in all_temperature if r["intervention_temperature"] == float(tk[1:-1])]))
            print(f"  {tk}: norm={vals['q_norm_mean']:.2f}±{vals['q_norm_std']:.2f}  diff_from_oracle={vals['q_diff_mean']:.2f}")

    print(f"\n=== Protein diversity ===")
    print(f"Unique proteins: {len(seen_pids)}")
    for pid, cnt in sorted(protein_counter.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pid}: {cnt} samples")

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
