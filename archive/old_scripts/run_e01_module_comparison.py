#!/usr/bin/env python3
"""E01: Module comparison — B0/B1/B2/B4/B5 on fixed Stage A representation.

Per CVPR Section 5.2 / 5.8:
  - B0: Persistence (Q_tau = Q_0, identity baseline)
  - B1: Causal Transformer world model (DINO-world/JEPA-style)
  - B2: Conditional flow matching (Gaussian NLL predictor)
  - B4: LaSP with C=0 (conservative force + diffusion, no bath)
  - B5: Full LaSP (conservative force + diffusion + exact-OU bath)

All baselines receive identical inputs from frozen Stage A:
  Q_0: [B,L,8] latent state at t=0
  H_last: [B,L,128] frozen context features
  tau_ps: horizon time in ps
  temperature_k: scalar condition
  residue_mask: [B,L]

All evaluated through shared PhysicalObservableReadout (M04).
Produces: metrics.jsonl, comparison_table.csv
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class E01Metrics:
    """Per-sample evaluation metrics."""
    model: str
    horizon_ps: float
    distance_mae: float   # CA-CA distance MAE (Å)
    contact_accuracy: float  # fraction correct contacts
    rmsf_mae: float        # per-residue RMSF MAE (Å)
    rg_error: float        # radius of gyration error (%)
    q_identity: float      # |Q_pred - Q_0| (normalized)
    n_samples: int         # number of test samples


def make_synthetic_batch(B=4, L=16, T=6):
    """Synthetic protein trajectories for smoke test."""
    device = torch.device("cpu")
    residue = torch.arange(L, dtype=torch.float32)
    base = torch.stack([1.5*residue, 4.0*torch.sin(residue/5.0), 4.0*torch.cos(residue/5.0)], dim=-1)
    offsets = torch.tensor([[-1.2,0.1,0.0],[0.0,0.0,0.0],[1.3,0.1,0.0],[1.8,1.0,0.0]])
    coords_list = []
    for b in range(B):
        phase = float(b % 17) / 17.0 * 2.0 * np.pi
        sample = []
        for t in range(T):
            time = phase + 0.4 * t
            bend = torch.zeros_like(base)
            bend[:,1] = torch.sin(residue/7.0 + time)*0.8
            bend[:,2] = torch.cos(residue/11.0 - time)*0.6
            sample.append(base[:,None,:] + offsets[None,:,:] + bend[:,None,:])
        coords_list.append(torch.stack(sample, dim=0))  # [T, L, 4, 3]
    coords = torch.stack(coords_list, dim=0)  # [B, T, L, 4, 3]
    return {
        "coords": coords,
        "residue_mask": torch.ones(B, T, L, dtype=torch.bool),
        "aatype": torch.full((B, L), 20, dtype=torch.long),
        "residue_index": torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1),
        "chain_id": torch.zeros(B, L, dtype=torch.long),
        "temperature_k": torch.full((B,), 348.0),
        "time_ps": torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1),
    }


def compute_target_observables(coords: torch.Tensor, residue_mask: torch.Tensor):
    """Compute ground-truth CA distances, contacts, RMSF from coordinates."""
    # coords: [B, T, L, 4, 3]
    # residue_mask: [B, T, L] or [B, L] — use last-frame mask for target
    ca = coords[..., 1, :]  # [B, T, L, 3]
    B, T, L = ca.shape[:3]

    # Use last frame as target
    if residue_mask.dim() == 3:
        last_mask = residue_mask[:, -1]  # [B, L]
    else:
        last_mask = residue_mask  # [B, L]

    target_ca = ca[:, -1]  # [B, L, 3]
    target_distances = torch.cdist(target_ca, target_ca)  # [B, L, L]
    target_contacts = (target_distances < 8.0).float()
    m2d = last_mask.unsqueeze(1).float() * last_mask.unsqueeze(2).float()
    target_rg = torch.sqrt(
        0.5 * (target_distances ** 2 * m2d).sum(dim=(1, 2)) / m2d.sum(dim=(1, 2)).clamp(min=1.0)
    )

    # RMSF from all frames: broadcast last_mask across time
    m_f_t = last_mask.unsqueeze(1).unsqueeze(-1).expand(B, T, L, 1).float()  # [B,T,L,1]
    mean_pos = (ca * m_f_t).sum(dim=1) / m_f_t.sum(dim=1).clamp(min=1.0)
    sq_dev = ((ca - mean_pos.unsqueeze(1)) ** 2).sum(dim=-1)
    msf = (sq_dev * m_f_t.squeeze(-1)).sum(dim=1) / m_f_t.squeeze(-1).sum(dim=1).clamp(min=1.0)
    target_rmsf = torch.sqrt(msf.clamp(min=1e-8))

    return {
        "ca_distances": target_distances,
        "contacts": target_contacts,
        "rmsf": target_rmsf,
        "global_rg": target_rg,
    }


def evaluate_model(
    model_name: str,
    pred_q: torch.Tensor,  # [B, L, 8]
    true_q: torch.Tensor,  # [B, L, 8] (ground-truth encoded Q)
    readout: torch.nn.Module,
    targets: dict,
    residue_mask: torch.Tensor,
    horizon_ps: float,
) -> E01Metrics:
    """Evaluate one model's prediction through readout."""
    with torch.no_grad():
        outputs = readout(pred_q, residue_mask)

        # Distance MAE
        d_mae = (outputs["ca_distances"] - targets["ca_distances"]).abs()
        m2 = residue_mask.unsqueeze(1).float() * residue_mask.unsqueeze(2).float()
        diag = torch.eye(residue_mask.shape[1], device=pred_q.device)
        off_diag = 1.0 - diag.unsqueeze(0)
        distance_mae_t = (d_mae * m2 * off_diag).sum() / (m2 * off_diag).sum().clamp(min=1.0)
        distance_mae = float(distance_mae_t.detach())

        # Contact accuracy
        pred_contacts = (outputs["ca_distances"] < 8.0).float()
        contact_acc_t = (
            (pred_contacts == targets["contacts"]).float() * m2 * off_diag
        ).sum() / (m2 * off_diag).sum().clamp(min=1.0)
        contact_acc = float(contact_acc_t.detach())

        # RMSF MAE
        rmsf_mae_t = (
            (outputs["rmsf"] - targets["rmsf"]).abs() * residue_mask.float()
        ).sum() / residue_mask.float().sum().clamp(min=1.0)
        rmsf_mae = float(rmsf_mae_t.detach())

        # RG error
        rg_err_t = ((outputs["global_rg"] - targets["global_rg"]).abs() / targets["global_rg"].clamp(min=1e-8))
        rg_err = float(rg_err_t.mean().detach())

        # Q identity (for B0, should be 0)
        q_norm = true_q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q_identity_tensor = (
            ((pred_q - true_q).norm(dim=-1) / q_norm.squeeze(-1)) * residue_mask.float()
        ).sum() / residue_mask.float().sum().clamp(min=1.0)
        q_identity = float(q_identity_tensor.detach())

        B = pred_q.shape[0]

    return E01Metrics(
        model=model_name,
        horizon_ps=horizon_ps,
        distance_mae=distance_mae,
        contact_accuracy=contact_acc,
        rmsf_mae=rmsf_mae,
        rg_error=rg_err,
        q_identity=q_identity,
        n_samples=B,
    )


def run_e01(
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
    horizon_ps_list: list[float] | None = None,
    seed: int = 42,
):
    """Run E01 module comparison.

    Args:
        checkpoint_path: frozen Stage A+B checkpoint (None = use smoke)
        output_dir: where to write results
        horizon_ps_list: list of prediction horizons to evaluate
        seed: RNG seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    print(f"E01: device={device}")

    out_dir = Path(output_dir or "artifacts/e01_module_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    if horizon_ps_list is None:
        horizon_ps_list = [100.0, 200.0, 500.0, 1000.0]

    # ── 1. Load frozen Stage A+B ──
    from protein_jepa_frame.config import ModelConfig
    from protein_jepa_frame.models.stage_b.model import StageBLASP

    cfg = ModelConfig(latent_dim=128, encoder_depth=2, temporal_depth=1,
        num_heads=2, num_rbf=8, max_frames=64, max_residues=128,
        dropout=0.0, orientation_features=False,
        time_conditioning="none", thermodynamic_conditioning=True)

    lasp = None
    if checkpoint_path and Path(checkpoint_path).exists():
        lasp = StageBLASP(cfg, state_dim=8, bath_dim=16, base_step_ps=100.0)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        remapped = {}
        for k, v in state.items():
            if k.startswith("context_encoder."):
                remapped["spatial_encoder." + k.removeprefix("context_encoder.")] = v
            elif k.startswith("temporal_encoder."):
                remapped[k] = v
        if remapped: state = remapped
        lasp.load_state_dict(state, strict=False)
        lasp.to(device)
        lasp.eval()
        for p in lasp.parameters(): p.requires_grad = False
        print(f"  Loaded checkpoint: {checkpoint_path}")

    # ── 2. Build readout (untrained, for demonstration) ──
    from protein_jepa_frame.models.stage_c.readout import (
        ReadoutConfig, PhysicalObservableReadout,
    )
    readout_cfg = ReadoutConfig(q_dim=8, hidden_dim=64, depth=2, num_rbf=16)
    readout = PhysicalObservableReadout(readout_cfg).to(device)
    print(f"  Readout params: {sum(p.numel() for p in readout.parameters()):,}")

    # ── 3. Build baselines ──
    from protein_jepa_frame.eval.baselines import (
        PersistenceBaseline,
        CausalTransformerWorldModel, TransformerWMConfig,
        ConditionalFlowMatchingPredictor, StochasticPredictorConfig,
    )
    b0 = PersistenceBaseline(state_dim=8)
    b1_cfg = TransformerWMConfig(state_dim=8, h_dim=128, d_model=64, n_heads=2, n_layers=2)
    b1 = CausalTransformerWorldModel(b1_cfg)
    b2_cfg = StochasticPredictorConfig(state_dim=8, h_dim=128, hidden_dim=64, n_layers=2)
    b2 = ConditionalFlowMatchingPredictor(b2_cfg)

    models = {
        "B0_Persistence": lambda q, h, tau, tk, rm: b0(q, h, tau, temperature_k=tk, residue_mask=rm),
        "B1_Transformer": lambda q, h, tau, tk, rm: b1(q, h, tau, temperature_k=tk, residue_mask=rm),
        "B2_FlowMatching": lambda q, h, tau, tk, rm: b2(q, h, tau, temperature_k=tk, residue_mask=rm)[0],
        # "B4" and "B5" require lasp.step() — only if checkpoint loaded
    }

    if lasp is not None:
        models["B5_FullLaSP"] = lambda q, h, tau, tk, rm: _b5_predict(lasp, q, h, tau, tk, rm, use_bath=True)
        models["B4_LaSP_noBath"] = lambda q, h, tau, tk, rm: _b4_predict(lasp, q, h, tau, tk, rm, use_bath=False)

    # ── 4. Generate synthetic data ──
    print("  Generating synthetic data...")
    batch = make_synthetic_batch(B=4, L=16, T=6)
    coords = batch["coords"].to(device)
    mask_t = batch["residue_mask"].to(device)
    aatype = batch["aatype"].to(device)
    res_idx = batch["residue_index"].to(device)
    chain_id = batch["chain_id"].to(device)
    temp_k = batch["temperature_k"].to(device)

    # Get frozen H and Q
    if lasp is not None:
        with torch.no_grad():
            h_ctx = lasp.forward_spatial(coords[:, :4], mask_t[:, :4], aatype, res_idx, chain_id)
            h_last = h_ctx[:, -1].detach()  # [B, L, 128]
            q_0 = lasp.project_to_q(h_last).detach()  # [B, L, 8]
            s_0 = lasp.initialize_bath(h_last, mask_t[:, -1]).detach()
            ca_pos = coords[:, -1, :, 1, :]
            ca_d = torch.cdist(ca_pos, ca_pos)
    else:
        # Random synthetic Q (no encoder available)
        B, L = 4, 16
        h_last = torch.randn(B, L, 128, device=device) * 0.5
        q_0 = torch.randn(B, L, 8, device=device) * 0.1
        s_0 = torch.zeros(B, L, 16, device=device)
        ca_d = torch.zeros(B, L, L, device=device)
        ca_pos = torch.randn(B, L, 3, device=device)

    rm = mask_t[:, -1]  # [B, L]
    targets = compute_target_observables(coords, rm)

    # ── 5. Evaluate all models at all horizons ──
    all_metrics = []

    for model_name, model_fn in models.items():
        print(f"\n  Evaluating {model_name}...")
        for h_ps in horizon_ps_list:
            tau_t = torch.full((q_0.shape[0],), h_ps, device=device)
            try:
                pred_q = model_fn(q_0, h_last, tau_t, temp_k, rm)
                metrics = evaluate_model(
                    model_name, pred_q, q_0, readout, targets, rm, h_ps,
                )
                all_metrics.append(metrics)
                print(f"    τ={h_ps:5.0f}ps: d_MAE={metrics.distance_mae:.3f}, "
                      f"contact={metrics.contact_accuracy:.3f}, "
                      f"rmsf={metrics.rmsf_mae:.3f}, rg_err={metrics.rg_error:.3f}, "
                      f"q_id={metrics.q_identity:.3f}")
            except Exception as e:
                print(f"    τ={h_ps:5.0f}ps: ERROR: {e}")

    # ── 6. Write results ──
    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for m in all_metrics:
            f.write(json.dumps({
                "model": m.model,
                "horizon_ps": m.horizon_ps,
                "distance_mae": m.distance_mae,
                "contact_accuracy": m.contact_accuracy,
                "rmsf_mae": m.rmsf_mae,
                "rg_error": m.rg_error,
                "q_identity": m.q_identity,
            }) + "\n")

    # CSV comparison table
    csv_path = out_dir / "comparison_table.csv"
    with open(csv_path, "w") as f:
        f.write("model,horizon_ps,distance_mae,contact_accuracy,rmsf_mae,rg_error,q_identity\n")
        for m in all_metrics:
            f.write(f"{m.model},{m.horizon_ps},{m.distance_mae:.6f},{m.contact_accuracy:.6f},"
                    f"{m.rmsf_mae:.6f},{m.rg_error:.6f},{m.q_identity:.6f}\n")

    print(f"\n  Metrics written to {metrics_path}")
    print(f"  CSV written to {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("E01 MODULE COMPARISON SUMMARY")
    print("=" * 70)
    for model_name in sorted(set(m.model for m in all_metrics)):
        model_metrics = [m for m in all_metrics if m.model == model_name]
        print(f"\n  {model_name}:")
        for m in sorted(model_metrics, key=lambda x: x.horizon_ps):
            print(f"    τ={m.horizon_ps:5.0f}ps: d_MAE={m.distance_mae:.3f}Å, "
                  f"contact_acc={m.contact_accuracy:.3f}")

    return all_metrics


def _b4_predict(lasp, q, h, tau, tk, rm, use_bath=False):
    """B4: Conservative force + diffusion, no bath (C=0)."""
    from protein_jepa_frame.models.stage_b.model import StageBLASP
    B, L, D = q.shape
    device = q.device
    if not hasattr(_b4_predict, "ca_d"):
        _b4_predict.ca_d = {}
    key = (B, L)
    if key not in _b4_predict.ca_d:
        _b4_predict.ca_d[key] = torch.zeros(B, L, L, device=device)

    dt_per_step = max(5.0, min(50.0, tau.mean().item() / 20))
    n_steps = max(1, int(tau.mean().item() / dt_per_step))
    dt = tau.mean().item() / n_steps

    q_cur = q.clone()
    s_cur = None  # No bath for B4
    for _ in range(n_steps):
        q_cur, s_cur = lasp.step(q_cur, s_cur, ca_distances=_b4_predict.ca_d[key],
                                 residue_mask=rm, temperature_k=tk,
                                 dt_ps=dt, stochastic=False)
    return q_cur


def _b5_predict(lasp, q, h, tau, tk, rm, use_bath=True):
    """B5: Full LaSP with bath."""
    from protein_jepa_frame.models.stage_b.model import StageBLASP
    B, L, D = q.shape
    device = q.device
    if not hasattr(_b5_predict, "ca_d"):
        _b5_predict.ca_d = {}
    key = (B, L)
    if key not in _b5_predict.ca_d:
        _b5_predict.ca_d[key] = torch.zeros(B, L, L, device=device)

    dt_per_step = max(5.0, min(50.0, tau.mean().item() / 20))
    n_steps = max(1, int(tau.mean().item() / dt_per_step))
    dt = tau.mean().item() / n_steps

    q_cur = q.clone()
    if hasattr(lasp, "initialize_bath"):
        s_cur = lasp.initialize_bath(h, rm).detach() if use_bath else None
    else:
        s_cur = torch.zeros(B, L, 16, device=device) if use_bath else None

    for _ in range(n_steps):
        q_cur, s_cur = lasp.step(q_cur, s_cur, ca_distances=_b5_predict.ca_d[key],
                                 residue_mask=rm, temperature_k=tk,
                                 dt_ps=dt, stochastic=False)
    return q_cur


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="E01: Module comparison")
    ap.add_argument("--checkpoint", default="artifacts/smoke_stage_b/checkpoint_latest.pt")
    ap.add_argument("--output-dir", default="artifacts/e01_module_comparison")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_e01(args.checkpoint, args.output_dir, seed=args.seed)
