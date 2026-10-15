"""iter_008: Stage B retraining with explicit kinematic routing into Q.

Root cause (artifacts/iter_007_phase0_audit, AB-1/AB-2 and probes):
  - The frozen H carries NO per-residue kinematic signal (val R^2 < 0 for
    displacement magnitude, ~0 for direction).
  - Q = proj(H) is therefore an empty per-residue latent: mag R^2 ~ 0.002,
    direction ~ -0.001.  The decoder consequently ignores Q entirely
    (Q-path 0.071 A vs total 1.54 A) and the Q-dependence gate shows
    direct == qshuffle == oracle (Delta = 0.0000 A).
  - The only demonstrable per-residue signal is RAW context kinematics
    (BA + VEL): mag R^2 = 0.267, dir R^2 = 0.151 on protein-disjoint val.

Fix: ROUTE that signal into Q explicitly (not regress it out of H).

    Q = proj_base(h_last) + proj_kin([h_last, kin_feat])

where kin_feat = SE(3)-invariant, context-only per-residue features
(BA, |BA|, VEL, |VEL|, SPAN, |SPAN|) expressed in each residue's anchor
frame.  The kinematic branch is zero-initialised, so loading v3_final
reproduces the canonical Q exactly and training grows the channel from
zero.  Only context frames are used -> legal for direct/autonomous.

Q-shaping supervision (training-time only auxiliary heads, discarded at
eval) additionally forces Q to organise the routed signal predictively:
  - disp_mag head on Q_i -> |future displacement|
  - disp_dir head on Q_i -> cos(BA_i, TL_i)
  - rg head on G = mean(Q) -> Rg delta
  - rmsf head on Q_i -> context displacement amplitude

All Gate-D objectives (NLL, force floor, force mag, energy reg, rollout
MSE, semigroup) are kept unchanged.  Stage A encoders stay frozen.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protein_jepa_frame.config import ModelConfig, DataConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_b.kinematic import StageBLASP_Kinematic
from protein_jepa_frame.models.stage_b.objectives import (
    stage_b_total_loss, rollout_mse_loss, semigroup_consistency_loss,
)
from protein_jepa_frame.training.train_utils import (
    _device, _set_reproducibility, _move_batch, _lr_scale, _save_checkpoint,
)


# ── Kinematic target computation ──────────────────────────────────────────────

def _kabsch_align_ca(mobile_ca: torch.Tensor, ref_ca: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Kabsch-align mobile_ca onto ref_ca. Returns aligned mobile_ca.
    mobile_ca, ref_ca: [L, 3], mask: [L] bool"""
    m = mask.float()
    n = m.sum().clamp(min=1.0)
    ref_c = (ref_ca * m.unsqueeze(-1)).sum(0) / n
    mob_c = (mobile_ca * m.unsqueeze(-1)).sum(0) / n
    ref_c_ca = ref_ca - ref_c
    mob_c_ca = mobile_ca - mob_c
    H = (mob_c_ca * m.unsqueeze(-1)).T @ (ref_c_ca * m.unsqueeze(-1))
    try:
        U, S, Vt = torch.linalg.svd(H)
        d = torch.det(Vt.T @ U.T)
        D = torch.diag(torch.tensor([1., 1., d], device=H.device, dtype=H.dtype))
        R = Vt.T @ D @ U.T
        aligned = (mobile_ca - mob_c) @ R.T + ref_c
    except Exception:
        aligned = mobile_ca
    return aligned


def compute_kinematic_targets(
    ctx_coords: torch.Tensor,   # [B, T_ctx, L, 4, 3]
    fut_coords: torch.Tensor,   # [B, T_fut, L, 4, 3]
    residue_mask: torch.Tensor, # [B, L]
) -> dict:
    """Compute per-residue kinematic supervision targets from raw coords.

    Returns:
        disp_mag:    [B, L]  per-residue future displacement magnitude (Å)
        disp_dir:    [B, L]  cos(basin_anchor_disp, future_disp) direction cosine
        rg_future:   [B]     Rg of future last frame CA
        q_norm_tgt:  [B, L]  RMSF proxy: mean |CA_t - CA_anchor| over context frames
    """
    B, T_ctx, L = ctx_coords.shape[:3]
    T_fut = fut_coords.shape[1]
    device = ctx_coords.device

    # Anchor = last context CA frame
    anchor_ca = ctx_coords[:, -1, :, 1, :]    # [B, L, 3]
    fut_last_ca = fut_coords[:, -1, :, 1, :]  # [B, L, 3]

    # --- per-sample Kabsch alignment & targets ---
    disp_mag = torch.zeros(B, L, device=device)
    disp_dir = torch.zeros(B, L, device=device)
    rg_delta = torch.zeros(B, device=device)
    q_norm_tgt = torch.zeros(B, L, device=device)

    for b in range(B):
        mask_b = residue_mask[b].bool()
        m = mask_b.float()
        n = m.sum().clamp(min=1.0)

        # Basin anchor displacement (BA): context mean CA - anchor CA
        ctx_ca_b = ctx_coords[b, :, :, 1, :]  # [T, L, 3]
        basin_mean_ca = ctx_ca_b.mean(0)       # [L, 3]
        ba = basin_mean_ca - anchor_ca[b]      # [L, 3]

        # Future displacement (TL): Kabsch-aligned future last CA - anchor CA
        fut_aligned = _kabsch_align_ca(fut_last_ca[b], anchor_ca[b], mask_b)
        tl = fut_aligned - anchor_ca[b]        # [L, 3]

        # Displacement magnitude (Å, typically 0-5)
        tl_norm = tl.norm(dim=-1)              # [L]
        disp_mag[b] = tl_norm

        # Direction cosine: cos(BA, TL)
        ba_norm = ba.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        tl_norm_v = tl.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        cos = ((ba / ba_norm) * (tl / tl_norm_v)).sum(-1)  # [L]
        disp_dir[b] = cos

        # Rg DELTA (future - anchor), O(1 Å) scale instead of ~16 Å absolute
        com_f = (fut_aligned * m.unsqueeze(-1)).sum(0) / n
        rg_fut = ((((fut_aligned - com_f) ** 2).sum(-1) * m).sum() / n).sqrt()
        com_a = (anchor_ca[b] * m.unsqueeze(-1)).sum(0) / n
        rg_anch = ((((anchor_ca[b] - com_a) ** 2).sum(-1) * m).sum() / n).sqrt()
        rg_delta[b] = rg_fut - rg_anch

        # RMSF proxy: mean CA displacement over context frames
        ctx_disp = (ctx_ca_b - anchor_ca[b].unsqueeze(0)).norm(dim=-1)  # [T, L]
        q_norm_tgt[b] = ctx_disp.mean(0)

    return {
        "disp_mag": disp_mag,
        "disp_dir": disp_dir,
        "rg_delta": rg_delta,
        "q_norm_tgt": q_norm_tgt,
    }


# ── Auxiliary head definition ─────────────────────────────────────────────────

class KinematicAuxHeads(nn.Module):
    """Four lightweight auxiliary prediction heads operating on Q."""

    def __init__(self, state_dim: int = 8):
        super().__init__()
        d = state_dim

        # Per-residue heads (input: q_i [D])
        self.disp_mag_head = nn.Sequential(
            nn.Linear(d, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 1),
        )
        self.disp_dir_head = nn.Sequential(
            nn.Linear(d, 32), nn.SiLU(),
            nn.Linear(32, 16), nn.SiLU(),
            nn.Linear(16, 1),
        )

        # Global heads (input: G = mean(Q) [D])
        self.rg_head = nn.Sequential(
            nn.Linear(d, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )

        # Per-residue RMSF proxy: predict |q_i| norm correlates with mobility
        self.rmsf_head = nn.Sequential(
            nn.Linear(d, 16), nn.SiLU(),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        q: torch.Tensor,        # [B, L, D]
        g: torch.Tensor,        # [B, D]
        residue_mask: torch.Tensor,  # [B, L]
        targets: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Compute auxiliary losses. Returns (total_aux_loss, loss_dict)."""
        B, L, D = q.shape
        mask_f = residue_mask.float()  # [B, L]
        n_res = mask_f.sum().clamp(min=1.0)
        total = q.new_zeros(1).squeeze()
        log = {}

        # 1. Displacement magnitude regression (MSE in log space for stability)
        disp_mag_tgt = targets["disp_mag"]    # [B, L]
        pred_mag = self.disp_mag_head(q).squeeze(-1)  # [B, L]
        # Predict log(1 + mag) to handle scale variation
        log_tgt = torch.log1p(disp_mag_tgt.clamp(min=0))
        log_pred = F.softplus(pred_mag)
        loss_mag = ((log_pred - log_tgt) ** 2 * mask_f).sum() / n_res
        total = total + loss_mag
        log["aux_disp_mag"] = float(loss_mag.detach().cpu())

        # 2. Displacement direction regression (cosine target in [-1, 1])
        disp_dir_tgt = targets["disp_dir"]    # [B, L]
        pred_dir = self.disp_dir_head(q).squeeze(-1).tanh()  # [B, L]
        loss_dir = ((pred_dir - disp_dir_tgt) ** 2 * mask_f).sum() / n_res
        total = total + loss_dir
        log["aux_disp_dir"] = float(loss_dir.detach().cpu())

        # 3. Rg delta regression on G (target is O(1) Å)
        rg_tgt = targets["rg_delta"].unsqueeze(-1)   # [B, 1]
        pred_rg = self.rg_head(g).squeeze(-1).tanh() * 5.0  # [B, 1] bounded ±5 Å
        loss_rg = ((pred_rg - rg_tgt) ** 2).mean()
        total = total + loss_rg
        log["aux_rg"] = float(loss_rg.detach().cpu())

        # 4. RMSF proxy: correlate |q_i| with context displacement amplitude
        q_norm = q.norm(dim=-1)  # [B, L]
        rmsf_tgt = targets["q_norm_tgt"]  # [B, L]
        pred_rmsf = self.rmsf_head(q).squeeze(-1)  # [B, L]
        rmsf_log_tgt = torch.log1p(rmsf_tgt.clamp(min=0))
        rmsf_log_pred = F.softplus(pred_rmsf)
        loss_rmsf = ((rmsf_log_pred - rmsf_log_tgt) ** 2 * mask_f).sum() / n_res
        total = total + loss_rmsf
        log["aux_rmsf"] = float(loss_rmsf.detach().cpu())

        log["aux_total"] = float(total.detach().cpu())
        return total, log


# ── Training loop ─────────────────────────────────────────────────────────────

def train_iter008_stage_b(
    init_checkpoint: str = "artifacts/full_abc_stage_b_v3_final/checkpoint_latest.pt",
    output_dir: str = "artifacts/iter_008_stage_b_kinematic",
    train_manifest: str = "artifacts/mdcath_manifest_200dom/train.jsonl",
    # Training budget
    total_steps: int = 3000,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 300,
    grad_clip: float = 5.0,
    # Auxiliary loss weights (tuned to not overwhelm Gate-D objectives)
    aux_weight: float = 0.5,
    # Gate-D loss weights (kept from v3_final)
    nll_weight: float = 1.0,
    force_floor_weight: float = 5.0,
    force_floor_target: float = 0.3,
    force_direction_weight: float = 0.1,
    energy_reg_weight: float = 0.01,
    rollout_weight: float = 0.1,
    semigroup_weight: float = 0.01,
    # Data
    batch_size: int = 1,
    num_frames: int = 8,
    frame_stride: int = 4,
    max_residues: int = 256,
    # Misc
    seed: int = 42,
    device_str: str = "cuda:0",
    log_every: int = 10,
    checkpoint_every: int = 500,
):
    _set_reproducibility(seed, False)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device(device_str)

    # ── Data ──
    dc = DataConfig(
        batch_size=batch_size, num_workers=0, num_frames=num_frames,
        frame_stride=frame_stride, window_stride=4, max_residues=max_residues,
        coordinate_scale=1.0, training_mode="paired_window",
        paired_horizons_frames=[32, 64, 96], paired_all_horizons=False,
        paired_history_mode="uniform",
    )
    dc.manifest_path = train_manifest
    dataset = build_dataset(dc)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", collate_fn=collate_paired_windows,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} windows, loader {len(loader)} batches/epoch", flush=True)

    # ── Model ──
    # Build a minimal config matching the v3_final checkpoint
    mc = ModelConfig(
        latent_dim=128, encoder_depth=4, temporal_depth=2, num_heads=4,
        num_rbf=16, max_frames=128, max_residues=max_residues, dropout=0.1,
        feature_norm=True, time_conditioning="physical",
        thermodynamic_conditioning=True,
        lspwm_per_residue_enabled=True, lspwm_use_bath=True,
        lspwm_history_aware=True, lspwm_rollout_enabled=True,
        lspwm_rollout_steps=32, lspwm_base_step_ps=100.0,
        lspwm_diffusion_scale=1e-6, lspwm_coupling_scale=1e-3,
    )
    model = StageBLASP_Kinematic(
        mc, kin_hidden=64, use_bath=True, history_aware=True,
    ).to(device)

    # Load v3_final checkpoint.  Infer the energy-graph condition/temperature
    # layout from the saved tensors (the checkpoint, not the constructor
    # default, is authoritative — see eval/stage_c_checkpoint.py).
    ckpt = torch.load(init_checkpoint, map_location=device)
    state = ckpt.get("model", ckpt)
    node_w = state.get("energy_graph.node_nets.0.0.weight")
    edge_w = state.get("energy_graph.edge_net.0.weight")
    if node_w is not None and edge_w is not None:
        condition_dim = int(edge_w.shape[1]) - 2 * 8 - 8
        has_temperature = int(node_w.shape[1]) == 8 + 128 + condition_dim + 1
        print(f"Checkpoint energy layout: condition_dim={condition_dim}, "
              f"has_temperature={has_temperature}", flush=True)
        if condition_dim != 0 or not has_temperature:
            model = StageBLASP_Kinematic(
                mc, kin_hidden=64, condition_dim=condition_dim,
                use_bath=True, history_aware=True,
                energy_temperature_feature=has_temperature,
            ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    # The only intentionally-missing keys are the zero-initialised kinematic
    # branch (grown from zero) — nothing else may be missing.
    kin_missing = [k for k in missing
                   if k.startswith(("kin_encoder.", "projection_kin."))]
    other_missing = [k for k in missing if k not in kin_missing]
    print(f"Loaded {init_checkpoint}: missing={len(missing)} "
          f"(kinematic={len(kin_missing)}, other={len(other_missing)}), "
          f"unexpected={len(unexpected)}", flush=True)
    if other_missing:
        raise RuntimeError(f"unexpected missing keys: {other_missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:8]}")

    # ── Auxiliary heads ──
    # These are training-time only — they shape Q to be predictive, but are
    # NOT saved into the Stage-B checkpoint consumed by Stage C / eval scripts.
    aux_heads = KinematicAuxHeads(state_dim=8).to(device)

    # ── Parameters to train ──
    # Frozen: spatial_encoder, temporal_encoder (same as v3_final)
    # Trainable: kin_encoder, projection_kin (kinematic branch),
    #            projection (base projection), energy_graph, diffusion, bath
    # NOT saved: aux_heads (training-side supervision only)
    trainable_params = []
    for name, param in model.named_parameters():
        if not name.startswith(("spatial_encoder.", "temporal_encoder.")):
            trainable_params.append(param)
    trainable_params.extend(aux_heads.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}", flush=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=weight_decay,
    )

    # ── Save config ──
    cfg_dict = {
        "init_checkpoint": init_checkpoint,
        "total_steps": total_steps, "lr": lr, "warmup_steps": warmup_steps,
        "aux_weight": aux_weight, "nll_weight": nll_weight,
        "force_floor_weight": force_floor_weight, "force_floor_target": force_floor_target,
        "force_direction_weight": force_direction_weight,
        "rollout_weight": rollout_weight, "semigroup_weight": semigroup_weight,
        "seed": seed,
        # Model layout — consumed by the Stage-B eval loader so it rebuilds
        # the identical (kinematic) class instead of a plain StageBLASP.
        "model": {
            "arch": "kinematic", "kin_hidden": model.kin_hidden,
            "latent_dim": mc.latent_dim, "encoder_depth": mc.encoder_depth,
            "temporal_depth": mc.temporal_depth, "num_heads": mc.num_heads,
            "num_rbf": mc.num_rbf, "max_frames": mc.max_frames,
            "max_residues": mc.max_residues,
            "time_conditioning": mc.time_conditioning,
            "thermodynamic_conditioning": mc.thermodynamic_conditioning,
            "lspwm_base_step_ps": mc.lspwm_base_step_ps,
            "lspwm_diffusion_scale": mc.lspwm_diffusion_scale,
            "lspwm_coupling_scale": mc.lspwm_coupling_scale,
            "lspwm_use_bath": mc.lspwm_use_bath,
            "lspwm_history_aware": mc.lspwm_history_aware,
            "condition_dim": getattr(model, "condition_dim", 0),
            "energy_temperature_feature": getattr(
                model, "energy_temperature_feature", True),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    log_path = out_dir / "train.jsonl"

    # ── Training ──
    global_step = 0
    iterator = iter(loader)

    while global_step < total_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = _move_batch(batch, device)

        # LR schedule
        lr_mult = _lr_scale(global_step, total_steps, warmup_steps)
        for grp in optimizer.param_groups:
            grp["lr"] = lr * lr_mult

        optimizer.zero_grad(set_to_none=True)

        ctx_coords = batch["context_coords"]
        fut_coords = batch.get("future_coords")
        B, T_ctx, L = ctx_coords.shape[:3]

        aatype = batch.get("context_aatype", batch.get("aatype"))
        ri = batch.get("context_residue_index", batch.get("residue_index"))
        ci = batch.get("context_chain_id", batch.get("chain_id"))
        if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
        if ri is not None and ri.dim() == 1: ri = ri.unsqueeze(0).expand(B, -1)
        if ci is not None and ci.dim() == 1: ci = ci.unsqueeze(0).expand(B, -1)

        temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
        if temp_k.dim() > 1: temp_k = temp_k[:, -1]

        ctx_mask = batch["context_residue_mask"]
        residue_mask_2d = ctx_mask[:, -1] if ctx_mask.dim() == 3 else ctx_mask

        # Stage B uses autograd through energy forces — AMP float16 produces
        # NaN in the NLL denominator on long-horizon batches. Run in float32.
        with torch.amp.autocast("cuda", enabled=False):
            # ── Stage B forward (Gate-D objectives) ──
            outputs = model.forward(batch)
            gate_d_loss, gate_d_log = stage_b_total_loss(
                outputs,
                nll_weight=nll_weight,
                force_floor_weight=force_floor_weight,
                force_floor_target=force_floor_target,
                force_direction_weight=force_direction_weight,
                energy_reg_weight=energy_reg_weight,
            )
            total_loss = gate_d_loss
            loss_log = dict(gate_d_log)

            # ── Rollout MSE ──
            if rollout_weight > 0 and fut_coords is not None:
                ctx_time = batch.get("context_time_ps")
                fut_time = batch.get("future_time_ps")
                target_dt = float(
                    (fut_time[0, -1] - ctx_time[0, -1]).item()
                ) if ctx_time is not None and fut_time is not None else 3200.0
                n_roll = max(1, min(32, int(round(target_dt / model.base_step_ps))))
                step_ps = max(target_dt / n_roll, 1e-6)
                q0 = outputs["q"].detach().clone()
                s0 = outputs.get("bath_s0",
                    torch.zeros(B, L, 16, device=device)).detach().clone()
                ca_d = torch.cdist(ctx_coords[:, -1, :, 1, :],
                                   ctx_coords[:, -1, :, 1, :])
                fut_mask = batch.get("future_residue_mask", ctx_mask)
                with torch.no_grad():
                    fut_state = model.encode_state(
                        fut_coords, fut_mask, aatype, ri, ci,
                        time_ps=batch.get("future_time_ps"),
                        temperature_k=temp_k,
                        use_temporal=True, return_bath=False,
                        return_ca_distances=False, return_forces=False,
                    )
                    q_tgt = fut_state["q"].detach()
                roll_loss = rollout_mse_loss(
                    model, q0, s0, n_roll, step_ps, q_tgt,
                    ca_d, residue_mask_2d, temp_k, stochastic=False,
                )
                if torch.isfinite(roll_loss):
                    total_loss = total_loss + rollout_weight * roll_loss
                    loss_log["rollout_mse"] = float(roll_loss.detach().cpu())

            # ── Semigroup ──
            if semigroup_weight > 0:
                z0 = outputs["q"].detach().clone()
                s0_sg = outputs.get("bath_s0",
                    torch.zeros(B, L, 16, device=device)).detach().clone()
                ca_d_sg = torch.cdist(ctx_coords[:, -1, :, 1, :],
                                      ctx_coords[:, -1, :, 1, :])
                semi_loss = semigroup_consistency_loss(
                    model, z0, s0_sg, float(model.base_step_ps),
                    ca_d_sg, residue_mask_2d, temp_k,
                )
                if torch.isfinite(semi_loss):
                    total_loss = total_loss + semigroup_weight * semi_loss
                    loss_log["semigroup"] = float(semi_loss.detach().cpu())

            # ── Kinematic auxiliary heads ──
            if aux_weight > 0 and fut_coords is not None:
                kin_targets = compute_kinematic_targets(
                    ctx_coords, fut_coords, residue_mask_2d,
                )
                q = outputs["q"]
                g = outputs["g"]
                aux_loss, aux_log = aux_heads(q, g, residue_mask_2d, kin_targets)
                if torch.isfinite(aux_loss):
                    total_loss = total_loss + aux_weight * aux_loss
                    loss_log.update(aux_log)

        # NaN/inf guard: a single pathological batch must not poison the run.
        if not torch.isfinite(total_loss):
            optimizer.zero_grad(set_to_none=True)
            print(json.dumps({"step": global_step, "skip": "nonfinite_loss"}), flush=True)
            global_step += 1
            continue

        total_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable_params if p.grad is not None], grad_clip,
            )
        optimizer.step()

        if global_step % log_every == 0:
            record = {
                "step": global_step,
                "loss": float(total_loss.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                **{k: v for k, v in loss_log.items()},
            }
            with log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if (global_step + 1) % checkpoint_every == 0:
            # Save Stage B model (kinematic branch included) + config.  The
            # config records the architecture so the eval loader rebuilds the
            # identical StageBLASP_Kinematic class.
            payload = {
                "model": model.state_dict(),
                "config": {"model": cfg_dict["model"], "stage_b": cfg_dict},
                "optimizer": optimizer.state_dict(),
                "step": global_step + 1,
            }
            ckpt_path = out_dir / f"checkpoint_step{global_step+1}.pt"
            torch.save(payload, ckpt_path)
            torch.save(payload, out_dir / "checkpoint_latest.pt")
            # Training-time auxiliary heads are advisory only; keep them in a
            # sidecar so training can be resumed without polluting the
            # Stage-B representation consumed by Stage C / evaluation.
            if aux_weight > 0:
                torch.save({"aux_heads": aux_heads.state_dict()},
                           out_dir / "aux_heads_latest.pt")
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

        global_step += 1

    # Final checkpoint
    payload = {
        "model": model.state_dict(),
        "config": {"model": cfg_dict["model"], "stage_b": cfg_dict},
        "optimizer": optimizer.state_dict(),
        "step": global_step,
    }
    torch.save(payload, out_dir / "checkpoint_latest.pt")
    torch.save(payload, out_dir / f"checkpoint_step{global_step}.pt")
    print(f"Training complete. Saved to {out_dir}/checkpoint_latest.pt", flush=True)
    return out_dir / "checkpoint_latest.pt"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--init_checkpoint", default="artifacts/full_abc_stage_b_v3_final/checkpoint_latest.pt")
    p.add_argument("--output_dir", default="artifacts/iter_008_stage_b_kinematic")
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    train_iter008_stage_b(
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        lr=args.lr,
        aux_weight=args.aux_weight,
        device_str=args.device,
    )
