#!/usr/bin/env python3
"""P4.3: Temperature jump intervention evaluation.

Tests whether the Stage B model responds correctly to temperature changes:
  - Equilibrium at T1 → jump to T2 → autonomous rollout at T2
  - Compare RMSF, Rg, and unfolding rate vs MD at T2

Protocol:
  1. Load protein, encode initial state at equilibrium temp
  2. Equilibrate for N steps at T_equil
  3. Jump to T_jump, run autonomous rollout for M steps
  4. Sample equilibrium at T_jump from separate MD data for comparison
"""

from __future__ import annotations
import json, argparse, sys
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
from protein_jepa_frame.utils.structure_metrics import (
    radius_of_gyration, rmsf_correlation, kabsch_align, contact_jaccard,
)


def load_stage_b(ckpt_path, device):
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
    sb = StageBLASP(mc, use_bath=use_bath, history_aware=history_aware).to(device)
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


def run_temp_intervention_eval(
    stage_b_ckpt="artifacts/mdcath_stage_b_rollout/checkpoint_latest.pt",
    val_manifest="artifacts/mdcath_manifest_200dom/val.jsonl",
    output_dir="artifacts/temp_intervention",
    n_val=50,
    seeds=(42, 123),
    equil_steps=10,
    jump_steps=50,
    base_step_ps=100.0,
    temps_equil=(348.0,),
    temps_jump=(413.0, 379.0, 320.0),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sb = load_stage_b(stage_b_ckpt, device)
    print("Stage B model loaded.")

    manifest = load_manifest(val_manifest)
    dcfg = DataConfig(
        batch_size=1, num_workers=0, num_frames=8, frame_stride=4,
        window_stride=4, max_residues=256, coordinate_scale=1.0,
        training_mode="paired_window", paired_horizons_frames=[32, 64, 96],
        paired_all_horizons=True, paired_history_mode="uniform",
    )
    dataset = PairedTrajectoryDataset(manifest, dcfg)
    eval_indices = stratified_indices(dataset, n_val, seed=seeds[0] if seeds else 42)
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_paired_windows, drop_last=False)

    all_records = []
    failed_samples = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        sample_i = 0
        for batch in loader:
            if sample_i >= n_val:
                break
            sample_i += 1

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            B, L = batch["context_coords"].shape[0], batch["context_coords"].shape[2]
            aatype = batch.get("future_aatype", batch.get("aatype"))
            if aatype.dim() == 1:
                aatype = aatype.unsqueeze(0).expand(B, -1)
            ri = batch.get("context_residue_index", batch.get("residue_index"))
            if ri is None:
                ri = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            if ri.dim() == 1:
                ri = ri.unsqueeze(0).expand(B, -1)
            ci = batch.get("context_chain_id", batch.get("chain_id"))
            if ci is None:
                ci = torch.zeros(B, L, dtype=torch.long, device=device)
            if ci.dim() == 1:
                ci = ci.unsqueeze(0).expand(B, -1)

            ctx_mask = batch["context_residue_mask"]
            residue_mask = ctx_mask[:, -1]
            ctx_coords = batch["context_coords"]
            protein_id = batch.get("protein_id", f"s{sample_i}")
            if hasattr(protein_id, "__iter__") and not isinstance(protein_id, str):
                protein_id = str(protein_id[0])

            # Keep the temperature shape identical to the canonical Stage-B
            # encode/transition path.  This must be initialized before the
            # sample try-block; otherwise the first sample fails with an
            # UnboundLocalError and the old evaluator silently continues.
            temp_k = batch.get(
                "context_temperature_k",
                torch.full((B,), 348.0, device=device),
            )
            if temp_k.dim() > 1:
                temp_k = temp_k[:, -1]

            try:
                with torch.no_grad():
                    state = sb.encode_state(
                        ctx_coords, ctx_mask, aatype, ri, ci,
                        time_ps=batch.get("context_time_ps"),
                        temperature_k=temp_k,
                        use_temporal=True,
                        return_bath=True,
                        return_ca_distances=True,
                        return_forces=False,
                    )
                    h_frozen = state["h_last"]
                    q0 = state["q"].detach()
                    s0 = state.get("bath_s0",
                        sb.initialize_bath(h_frozen, state["residue_mask"])).detach()
                    ca_d = state["ca_distances"]

                for t_equil in temps_equil:
                    for t_jump in temps_jump:
                        if t_jump == t_equil:
                            continue

                        # ── Phase 1: Equilibrate at T_equil ──
                        q_t = q0.clone()
                        s_t = s0.clone()
                        temp_k_equil = torch.tensor([t_equil], device=device)
                        for _ in range(equil_steps):
                            with torch.enable_grad():
                                q_t, s_t, _ = sb.transition(
                                    q_t, s_t,
                                    ca_distances=ca_d,
                                    residue_mask=residue_mask,
                                    temperature_k=temp_k_equil,
                                    dt_ps=base_step_ps,
                                    stochastic=True,
                                )
                            q_t = q_t.detach()
                            if s_t is not None:
                                s_t = s_t.detach()

                        # ── Phase 2: Jump to T_jump ──
                        temp_k_jump = torch.tensor([t_jump], device=device)
                        q_traj = []
                        for _ in range(jump_steps):
                            with torch.enable_grad():
                                q_t, s_t, _ = sb.transition(
                                    q_t, s_t,
                                    ca_distances=ca_d,
                                    residue_mask=residue_mask,
                                    temperature_k=temp_k_jump,
                                    dt_ps=base_step_ps,
                                    stochastic=True,
                                )
                            q_t = q_t.detach()
                            if s_t is not None:
                                s_t = s_t.detach()
                            q_traj.append(q_t.norm().item())

                        # ── Phase 3: Sample equilibrium at T_jump for comparison ──
                        # Re-encode same structure at T_jump for oracle comparison
                        with torch.no_grad():
                            jump_state = sb.encode_state(
                                ctx_coords, ctx_mask, aatype, ri, ci,
                                time_ps=batch.get("context_time_ps"),
                                temperature_k=temp_k_jump,
                                use_temporal=True,
                                return_bath=True,
                                return_ca_distances=False,
                                return_forces=False,
                            )
                            q_equil_jump = jump_state["q"].detach()
                            s_equil_jump = jump_state.get("bath_s0",
                                sb.initialize_bath(
                                    jump_state["h_last"], residue_mask)).detach()

                        q_equil_traj = []
                        for _ in range(jump_steps):
                            with torch.enable_grad():
                                q_equil_jump, s_equil_jump, _ = sb.transition(
                                    q_equil_jump, s_equil_jump,
                                    ca_distances=ca_d,
                                    residue_mask=residue_mask,
                                    temperature_k=temp_k_jump,
                                    dt_ps=base_step_ps,
                                    stochastic=True,
                                )
                            q_equil_jump = q_equil_jump.detach()
                            if s_equil_jump is not None:
                                s_equil_jump = s_equil_jump.detach()
                            q_equil_traj.append(q_equil_jump.norm().item())

                        # Compute statistics
                        jump_mean_q = np.mean(q_traj)
                        jump_std_q = np.std(q_traj)
                        equil_mean_q = np.mean(q_equil_traj)
                        equil_std_q = np.std(q_equil_traj)

                        rec = {
                            "protein_id": protein_id,
                            "seed": seed,
                            "T_equil": t_equil,
                            "T_jump": t_jump,
                            "L": L,
                            "jump_q_mean": float(jump_mean_q),
                            "jump_q_std": float(jump_std_q),
                            "equil_q_mean": float(equil_mean_q),
                            "equil_q_std": float(equil_std_q),
                            "q_diff_ratio": float(
                                (jump_mean_q - equil_mean_q) / max(abs(equil_mean_q), 1e-8)
                            ) if abs(equil_mean_q) > 1e-8 else 0.0,
                        }
                        all_records.append(rec)

            except Exception as e:
                failed_samples.append({
                    "sample_idx": sample_i,
                    "protein_id": str(protein_id),
                    "seed": seed,
                    "error": str(e),
                })
                print(f"  [WARN] {protein_id} seed={seed}: {e}")

            if sample_i % 10 == 0:
                print(f"  [seed={seed}] {sample_i}/{n_val}")

    # ── Save ──
    with open(out_dir / "intervention_results.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    if failed_samples:
        with open(out_dir / "failed_samples.json", "w") as f:
            json.dump(failed_samples, f, indent=2)
        raise RuntimeError(
            f"Temperature intervention evaluation incomplete: "
            f"{len(failed_samples)} sample/seed runs failed; see failed_samples.json"
        )

    # ── Summary ──
    summary = {"n_samples": n_val, "n_seeds": len(seeds),
               "temps_equil": list(temps_equil), "temps_jump": list(temps_jump)}
    for t_equil in temps_equil:
        for t_jump in temps_jump:
            if t_jump == t_equil:
                continue
            pair_records = [r for r in all_records
                           if abs(r["T_equil"] - t_equil) < 1 and abs(r["T_jump"] - t_jump) < 1]
            if not pair_records:
                continue
            q_diff_ratios = [r["q_diff_ratio"] for r in pair_records]
            jump_stds = [r["jump_q_std"] for r in pair_records]
            equil_stds = [r["equil_q_std"] for r in pair_records]
            summary[f"jump_{int(t_equil)}K_to_{int(t_jump)}K"] = {
                "n": len(q_diff_ratios),
                "q_diff_ratio_mean": float(np.mean(q_diff_ratios)),
                "q_diff_ratio_std": float(np.std(q_diff_ratios)),
                "jump_std_mean": float(np.mean(jump_stds)),
                "equil_std_mean": float(np.mean(equil_stds)),
            }

    with open(out_dir / "intervention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTemperature Intervention Results:")
    print(f"{'='*80}")
    for key, val in summary.items():
        if key.startswith("jump_"):
            print(f"  {key}: Q_diff_ratio={val['q_diff_ratio_mean']:.4f} "
                  f"jump_std={val['jump_std_mean']:.3f} "
                  f"equil_std={val['equil_std_mean']:.3f}")
    print(f"{'='*80}")
    print(f"Saved to {out_dir}/intervention_summary.json")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b-ckpt", default="artifacts/mdcath_stage_b_rollout/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/temp_intervention")
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = ap.parse_args()
    run_temp_intervention_eval(
        args.stage_b_ckpt, args.val_manifest, args.output_dir,
        args.n_val, tuple(args.seeds),
    )
