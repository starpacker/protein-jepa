#!/usr/bin/env python3
"""P6: Baseline comparisons for world model evaluation.

Compares Stage B autonomous rollout against:
  1. Linear interpolation: q_t = q_0 + t/(T * dt) * (q_T - q_0)
  2. Persistence: q_t = q_0 (no change)
  3. Transformer: temporal attention over history → predict future Q
  4. (RSSM placeholder — requires separate training)

Metrics: MSE vs oracle Q, stability (Q norm drift), ACF.
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


class SimpleTransformerPredictor(torch.nn.Module):
    """P6.1: Simple transformer baseline for Q prediction.

    Takes history Q sequence [B, T, L*8] and predicts future Q [B, L, 8].
    Uses 2-layer transformer encoder with learnable position encoding.
    """
    def __init__(self, dim=256, nhead=4, nlayers=2, max_len=32):
        super().__init__()
        self.pos_embed = torch.nn.Embedding(max_len, dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim*4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.linear_in = torch.nn.Linear(8, dim)
        self.linear_out = torch.nn.Linear(dim, 8)
        self.dim = dim

    def forward(self, q_history):
        B, T, LD = q_history.shape
        # q_history: [B, T, L*8] where L varies per sample
        L = LD // 8
        q_reshaped = q_history.reshape(B, T, L, 8)  # [B, T, L, 8]
        # Pool over residues: mean-pool to [B, T, 8]
        q_pooled = q_reshaped.mean(dim=2)  # [B, T, 8]
        h = self.linear_in(q_pooled)  # [B, T, dim]
        pos = torch.arange(T, device=q_history.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embed(pos)
        mask = torch.zeros(B, T, dtype=torch.bool, device=q_history.device)
        out = self.transformer(h, src_key_padding_mask=mask)  # [B, T, dim]
        # Predict L*8 output by expanding back to per-residue
        pred_pooled = self.linear_out(out[:, -1])  # [B, 8]
        # Broadcast to [B, L, 8]
        pred = pred_pooled.unsqueeze(1).expand(-1, L, -1).reshape(B, LD)
        return pred  # [B, LD]


def run_baseline_comparison(
    stage_b_ckpt="artifacts/mdcath_stage_b_rollout/checkpoint_latest.pt",
    val_manifest="artifacts/mdcath_manifest_200dom/val.jsonl",
    train_manifest="artifacts/mdcath_manifest_200dom/train.jsonl",
    output_dir="artifacts/baseline_comparison",
    n_val=100,
    seeds=(42,),
    horizons_ps=None,
    base_step_ps=100.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage B model ──
    sb = load_stage_b(stage_b_ckpt, device)
    print("Stage B model loaded.")

    # ── Transformer baseline ──
    transformer = SimpleTransformerPredictor().to(device)
    transformer.train()  # Will be trained online on the val set as a simple baseline estimate
    print("Transformer initialized.")

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

    train_dataset = PairedTrajectoryDataset(load_manifest(train_manifest), dcfg)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                              num_workers=0, collate_fn=collate_paired_windows,
                              drop_last=False)

    # ── Collect data for transformer training ──
    train_q_data = []
    failed_samples = []
    print("Collecting Q data for transformer training...")
    for sample_i, batch in enumerate(train_loader):
        if sample_i >= 200:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        try:
            with torch.no_grad():
                ctx = batch["context_coords"]
                ctx_mask = batch["context_residue_mask"]
                aatype = batch.get("future_aatype", batch.get("aatype"))
                ri = batch.get("context_residue_index", batch.get("residue_index"))
                ci = batch.get("context_chain_id", batch.get("chain_id"))
                if aatype.dim() == 1: aatype = aatype.unsqueeze(0)
                if ri is None: ri = torch.arange(ctx.shape[2], device=device).unsqueeze(0)
                if ri.dim() == 1: ri = ri.unsqueeze(0).expand(ctx.shape[0], -1)
                if ci is None: ci = torch.zeros(ctx.shape[0], ctx.shape[2], dtype=torch.long, device=device)
                if ci.dim() == 1: ci = ci.unsqueeze(0).expand(ctx.shape[0], -1)

                fut_coords = batch["future_coords"]
                fut_mask = batch["future_residue_mask"]
                temp_k = batch.get("context_temperature_k")
                if temp_k is not None and temp_k.dim() > 1:
                    temp_k = temp_k[:, -1]
                state = sb.encode_state(
                    ctx, ctx_mask, aatype, ri, ci,
                    time_ps=batch.get("context_time_ps"),
                    temperature_k=temp_k,
                    use_temporal=True,
                    return_bath=False,
                    return_ca_distances=False,
                    return_forces=False,
                )
                q0 = state["q"].detach()

                # The baseline target is the future endpoint.  Keep this
                # spatial-only, matching the oracle endpoint protocol below.
                fut_state = sb.encode_state(
                    fut_coords, fut_mask, aatype, ri, ci,
                    use_temporal=False,
                    return_bath=False,
                    return_ca_distances=False,
                    return_forces=False,
                )
                q_target = fut_state["q"].detach()

                B, L = q0.shape[:2]
                train_q_data.append((q0.reshape(B, L * 8), q_target.reshape(B, L * 8)))
        except Exception as e:
            failed_samples.append({
                "phase": "transformer_train_collection",
                "sample_idx": sample_i,
                "error": str(e),
            })
            print(f"  [ERROR] train sample={sample_i}: {e}")

    if not train_q_data:
        raise RuntimeError(
            "Baseline evaluation collected no training samples for the "
            "transformer baseline"
        )

    # Quick training of transformer baseline
    print(f"Training transformer on {len(train_q_data)} samples...")
    opt = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    for epoch in range(5):
        total_loss = 0.0
        for q_input, q_target in train_q_data:
            # q_input: [1, L*8], need temporal dim
            q_input = q_input.unsqueeze(1)  # [B, 1, LD]
            pred = transformer(q_input)
            loss = ((pred - q_target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_q_data):.6f}")

    transformer.eval()
    for p in transformer.parameters():
        p.requires_grad = False

    # ── Run baseline comparison ──
    all_records = []

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
            ri = batch.get("context_residue_index", batch.get("residue_index"))
            ci = batch.get("context_chain_id", batch.get("chain_id"))
            if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
            if ri is None: ri = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            if ri.dim() == 1: ri = ri.unsqueeze(0).expand(B, -1)
            if ci is None: ci = torch.zeros(B, L, dtype=torch.long, device=device)
            if ci.dim() == 1: ci = ci.unsqueeze(0).expand(B, -1)

            try:
                temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
                if temp_k.dim() > 1: temp_k = temp_k[:, -1]

                with torch.no_grad():
                    ctx = batch["context_coords"]
                    ctx_mask = batch["context_residue_mask"]

                    # ── Canonical state encoding ──
                    state = sb.encode_state(
                        ctx, ctx_mask, aatype, ri, ci,
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

                    fut_coords = batch["future_coords"]
                    fut_mask = batch["future_residue_mask"]
                    fut_state = sb.encode_state(
                        fut_coords, fut_mask, aatype, ri, ci,
                        temperature_k=temp_k,
                        use_temporal=False,
                        return_bath=False,
                        return_ca_distances=False,
                        return_forces=False,
                    )
                    q_oracle = fut_state["q"].detach()

                protein_id = batch.get("protein_id", f"s{sample_i}")
                if hasattr(protein_id, "__iter__") and not isinstance(protein_id, str):
                    protein_id = str(protein_id[0])

                dataset_horizon = float(batch["horizon_time_ps"][0].item())
                if horizons_ps is None:
                    eval_horizons = [dataset_horizon]
                else:
                    eval_horizons = [float(h) for h in horizons_ps
                                     if abs(float(h) - dataset_horizon) < 1e-3]
                for h_ps in eval_horizons:
                    n_steps = int(h_ps // base_step_ps)
                    remainder = max(0.0, h_ps - n_steps * base_step_ps)

                    # ── Method 1: Stage B autonomous rollout ──
                    q_sb = q0.clone()
                    s_sb = s0.clone()
                    for _ in range(n_steps):
                        with torch.enable_grad():
                            q_sb, s_sb, _ = sb.transition(
                                q_sb, s_sb, ca_distances=ca_d,
                                residue_mask=residue_mask,
                                temperature_k=temp_k, dt_ps=base_step_ps,
                                stochastic=True,
                            )
                        q_sb = q_sb.detach()
                        if s_sb is not None:
                            s_sb = s_sb.detach()
                    if remainder > 0:
                        with torch.enable_grad():
                            q_sb, s_sb, _ = sb.transition(
                                q_sb, s_sb, ca_distances=ca_d,
                                residue_mask=residue_mask,
                                temperature_k=temp_k, dt_ps=remainder,
                                stochastic=True,
                            )
                        q_sb = q_sb.detach()
                        if s_sb is not None:
                            s_sb = s_sb.detach()
                    sb_mse = float(((q_sb - q_oracle) ** 2).mean().item())
                    sb_norm = float(q_sb.norm().item())

                    # ── Method 2: Linear interpolation ──
                    q_linear = q0 + (n_steps / n_steps) * (q_oracle - q0)
                    linear_mse = float(((q_linear - q_oracle) ** 2).mean().item())
                    linear_norm = float(q_linear.norm().item())

                    # ── Method 3: Persistence ──
                    q_persist = q0
                    persist_mse = float(((q_persist - q_oracle) ** 2).mean().item())
                    persist_norm = float(q_persist.norm().item())

                    # ── Method 4: Transformer baseline ──
                    q_flat = q0.reshape(B, L * 8).unsqueeze(1)  # [B, T=1, LD]
                    q_transformer = transformer(q_flat)
                    q_transformer = q_transformer.reshape(B, L, 8)
                    transformer_mse = float(((q_transformer - q_oracle) ** 2).mean().item())
                    transformer_norm = float(q_transformer.norm().item())

                    rec = {
                        "protein_id": protein_id,
                        "seed": seed,
                        "horizon_ps": h_ps,
                        "L": L,
                        "stage_b_mse": sb_mse,
                        "stage_b_norm": sb_norm,
                        "linear_mse": linear_mse,
                        "linear_norm": linear_norm,
                        "persistence_mse": persist_mse,
                        "persistence_norm": persist_norm,
                        "transformer_mse": transformer_mse,
                        "transformer_norm": transformer_norm,
                        "oracle_norm": float(q_oracle.norm().item()),
                    }
                    all_records.append(rec)

            except Exception as e:
                failed_samples.append({
                    "phase": "evaluation",
                    "sample_idx": sample_i,
                    "seed": seed,
                    "error": str(e),
                })
                print(f"  [WARN] {sample_i}: {e}")

            if sample_i % 10 == 0:
                print(f"  [seed={seed}] {sample_i}/{n_val}")

    # ── Aggregate ──
    with open(out_dir / "baseline_results.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    if failed_samples:
        with open(out_dir / "failed_samples.json", "w") as f:
            json.dump(failed_samples, f, indent=2)
        raise RuntimeError(
            f"Baseline evaluation incomplete: {len(failed_samples)} samples failed; "
            f"see failed_samples.json"
        )

    observed_horizons = sorted({r["horizon_ps"] for r in all_records})
    summary = {"n_samples": n_val, "n_seeds": len(seeds),
               "horizons_ps": observed_horizons}
    for h_ps in observed_horizons:
        hr = [r for r in all_records if r["horizon_ps"] == h_ps]
        if not hr:
            continue
        key = f"horizon_{int(h_ps)}ps"
        sm = {}
        for method in ["stage_b", "linear", "persistence", "transformer"]:
            vals = [r[f"{method}_mse"] for r in hr
                    if not np.isnan(r[f"{method}_mse"]) and not np.isinf(r[f"{method}_mse"])]
            norms = [r[f"{method}_norm"] for r in hr]
            if vals:
                sm[method] = {
                    "mse_mean": float(np.mean(vals)),
                    "mse_std": float(np.std(vals)),
                    "norm_mean": float(np.mean(norms)),
                    "norm_std": float(np.std(norms)),
                }
        oracle_norms = [r["oracle_norm"] for r in hr]
        sm["oracle"] = {
            "norm_mean": float(np.mean(oracle_norms)),
            "norm_std": float(np.std(oracle_norms)),
        }
        summary[key] = sm

    with open(out_dir / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBaseline Comparison (n_samples={n_val}, seeds={len(seeds)}):")
    print(f"{'='*80}")
    for h_ps in observed_horizons:
        key = f"horizon_{int(h_ps)}ps"
        if key not in summary:
            continue
        s = summary[key]
        print(f"\nHorizon {h_ps}ps (oracle Q norm: {s['oracle']['norm_mean']:.1f}):")
        for method in ["stage_b", "linear", "persistence", "transformer"]:
            m = s[method]
            print(f"  {method:>12s}: MSE={m['mse_mean']:.6f} norm={m['norm_mean']:.1f}")
    print(f"{'='*80}")
    print(f"Saved to {out_dir}/baseline_summary.json")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-b-ckpt", default="artifacts/mdcath_stage_b_rollout/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/baseline_comparison")
    ap.add_argument("--train-manifest", default="artifacts/mdcath_manifest_200dom/train.jsonl")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()
    run_baseline_comparison(
        stage_b_ckpt=args.stage_b_ckpt, val_manifest=args.val_manifest,
        output_dir=args.output_dir, n_val=args.n_val, seeds=tuple(args.seeds),
        train_manifest=args.train_manifest,
    )
