"""Stage B training loop: freeze Stage A encoders, train canonical LaSP world model.

Gate 2 training protocol:
- Load frozen Stage A checkpoint (validated tensor-hash identical)
- Train: projection H→Q, energy graph, diffusion, bath, integrator
- Loss: transition NLL + force direction + energy regularization
- Synthetic tests (P0-2 through P0-7) must pass before protein training
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from protein_jepa_frame.config import TrainConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset, validate_real_paired_window_protocol
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_b.objectives import (
    stage_b_total_loss,
    rollout_mse_loss,
    semigroup_consistency_loss,
)
from protein_jepa_frame.training.train_utils import (
    _device,
    _set_reproducibility,
    _move_batch,
    _momentum,
    _lr_scale,
    _save_checkpoint,
    _load_init_checkpoint,
    _load_resume_checkpoint,
    _validate_init_freeze_contract,
)


class StageBCallback:
    """Callback registry for Stage B training."""
    def on_epoch_start(self, epoch: int, model: StageBLASP, optimizer, config):
        pass

    def on_step_end(self, step: int, loss: float, outputs: dict):
        pass

    def on_epoch_end(self, epoch: int, model: StageBLASP, optimizer, config) -> dict:
        return {}


def train_stage_b(config: TrainConfig) -> Path:
    """Train Stage B LaSP world model.

    Expected config fields (subset):
    - config.init_checkpoint: Path to frozen Stage A checkpoint
    - config.model: ModelConfig (with lspwm_* fields)
    - config.data.batch_size, config.optimization.*
    - config.data.training_mode: "paired_window"

    Returns:
        Path to output directory
    """
    if config.init_checkpoint and config.resume_checkpoint:
        raise ValueError("use either init_checkpoint or resume_checkpoint, not both")
    _validate_init_freeze_contract(config)
    validate_real_paired_window_protocol(config.data)
    _set_reproducibility(config.seed, config.deterministic)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config_to_dict(config), indent=2))

    device = _device(config.device)

    # Build dataset (paired windows with physical horizons)
    dataset = build_dataset(config.data, msm_labels_path=config.data.msm_labels_path)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=str(device).startswith("cuda"),
        collate_fn=collate_paired_windows,
        drop_last=True,
    )

    if len(loader) == 0:
        raise ValueError("dataloader is empty; reduce batch size")

    steps_per_epoch = config.optimization.steps_per_epoch or len(loader)
    # Support explicit max_steps (e.g., 2760 for exact reproducibility)
    max_steps = getattr(config.optimization, 'max_steps', None)
    if max_steps is not None and max_steps > 0:
        total_steps = max_steps
        # Adjust epochs to match (only needed for schedule)
        config.optimization.epochs = max(1, int(math.ceil(total_steps / steps_per_epoch)))
    else:
        total_steps = steps_per_epoch * config.optimization.epochs

    # Build Stage B model with frozen encoders
    model = StageBLASP(
        config.model,
        use_bath=getattr(config.model, 'lspwm_use_bath', True),
        history_aware=getattr(config.model, 'lspwm_history_aware', True),
    ).to(device)

    # Load frozen Stage A encoders (remap context_encoder → spatial_encoder)
    if config.init_checkpoint:
        ckpt = torch.load(config.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        # Remap Stage A context_encoder → Stage B spatial_encoder, keep all other keys
        remapped = {}
        for key, value in state.items():
            if key.startswith("context_encoder."):
                remapped["spatial_encoder." + key.removeprefix("context_encoder.")] = value
            else:
                remapped[key] = value
        state = remapped
        incompatible = model.load_state_dict(state, strict=False)
        init_info = {
            "event": "stage_b_init",
            "checkpoint": str(config.init_checkpoint),
            "loaded_keys": len(state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        print(json.dumps(init_info), flush=True)

    # Only train LaSP parameters
    lasp_params = []
    for name, param in model.named_parameters():
        if not name.startswith(("spatial_encoder.", "temporal_encoder.", "target_")):
            lasp_params.append(param)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    optimizer = torch.optim.AdamW(
        lasp_params,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
    )

    global_step = 0
    log_path = output_dir / "train.jsonl"
    start_epoch = 0
    grad_accum = max(1, int(getattr(config.optimization, 'gradient_accumulation_steps', 1)))

    if config.resume_checkpoint:
        global_step = _load_resume_checkpoint(
            model, optimizer, config.resume_checkpoint, device
        )

    if global_step >= total_steps:
        _save_checkpoint(
            output_dir, model, optimizer,
            config.optimization.epochs, global_step, config,
            write_step_file=False,
        )
        return output_dir / "checkpoint_latest.pt"

    start_epoch = global_step // steps_per_epoch

    for epoch in range(start_epoch, config.optimization.epochs):
        iterator = iter(loader)
        first_local_step = global_step % steps_per_epoch if epoch == start_epoch else 0
        for local_step in range(first_local_step, steps_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = _move_batch(batch, device)

            # LR schedule
            lr_mult = _lr_scale(global_step, total_steps, config.optimization.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.optimization.lr * lr_mult

            is_accum_step = (global_step % grad_accum != 0)
            if not is_accum_step:
                optimizer.zero_grad(set_to_none=True)

            # Forward: encode history to H_t, project to Q_t, compute forces/energy
            outputs = model.forward(batch)

            # Stage B loss (teacher-forced one-step NLL)
            total_loss, loss_components = stage_b_total_loss(
                outputs,
                nll_weight=config.optimization.lagged_state_jepa_weight,
                force_floor_weight=getattr(config.optimization, 'force_floor_weight', 0.5),
                force_floor_target=getattr(config.optimization, 'force_floor_target', 0.1),
                force_direction_weight=getattr(config.optimization, 'state_jepa_weight', 0.1),
                energy_reg_weight=getattr(config.optimization, 'variance_reg', 0.01),
                force_cosine_weight=getattr(
                    config.optimization, 'lspwm_force_direction_weight', 0.0
                ),
            )

            # ── P1.3: Rollout NLL (autonomous N-step) ──
            rollout_w = getattr(config.optimization, 'lspwm_rollout_trajectory_nll_weight', 0.0)
            semigroup_w = getattr(config.optimization, 'lspwm_semigroup_consistency_weight', 0.0)

            if rollout_w > 0 and "q" in outputs and "future_coords" in batch:
                # Match the endpoint target's physical time.
                ctx_time = batch.get("context_time_ps")
                fut_time = batch.get("future_time_ps")
                if ctx_time is not None and fut_time is not None:
                    target_dt = float((fut_time[0, -1] - ctx_time[0, -1]).item())
                else:
                    target_dt = float(model.base_step_ps)
                base_step_ps = float(model.base_step_ps)
                max_rollout = max(1, int(getattr(config.model, "lspwm_rollout_steps", 32)))
                n_rollout = max(1, min(max_rollout, int(round(target_dt / base_step_ps))))
                step_ps = max(float(target_dt) / n_rollout, 1e-6)
                q0 = outputs["q"].detach().clone()
                s0 = outputs.get("bath_s0", torch.zeros_like(q0[:, :, :1]).expand(-1, -1, 16)).detach().clone()
                ca_d = torch.cdist(
                    batch["context_coords"][:, -1, :, 1, :],
                    batch["context_coords"][:, -1, :, 1, :],
                )

                # Training rollout is deterministic.  The force_floor loss
                # guarantees |F| >= 0.3, which makes the LaSP dynamics
                # informative even without FDT noise.  FDT noise is used
                # only at evaluation time (stochastic=True) — it would add
                # uninformative variance to the training gradient since
                # 32-step SNR is only 0.6 even at |F|=0.3.
                use_stochastic = False
                noise_scale = 0.0
                temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=q0.device))
                if temp_k.dim() > 1: temp_k = temp_k[:, -1]

                fut_coords = batch.get("future_coords")
                fut_mask = batch.get("future_residue_mask")
                aatype = batch.get("context_aatype", batch.get("aatype"))
                ri = batch.get("context_residue_index", batch.get("residue_index"))
                ci = batch.get("context_chain_id", batch.get("chain_id"))
                B, L = q0.shape[:2]
                if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
                if ri.dim() == 1: ri = ri.unsqueeze(0).expand(B, -1)
                if ci is None or ci.dim() == 1: ci = torch.zeros(B, L, dtype=torch.long, device=q0.device).unsqueeze(0).expand(B, -1) if ci is None else ci.unsqueeze(0).expand(B,-1)

                with torch.no_grad():
                    # ── CANONICAL: q_target uses encode_state with use_temporal=True
                    # so it matches the same representation space as q0 (context Q).
                    fut_state = model.encode_state(
                        fut_coords, fut_mask, aatype, ri, ci,
                        time_ps=batch.get("future_time_ps"),
                        temperature_k=temp_k,
                        use_temporal=True,
                        return_bath=False,
                        return_ca_distances=False,
                        return_forces=False,
                    )
                    q_target = fut_state["q"].detach()
                residue_mask_2d = outputs.get("residue_mask")

                roll_loss = rollout_mse_loss(
                    model, q0, s0, n_rollout, step_ps,
                    q_target, ca_d, residue_mask_2d, temp_k,
                    stochastic=use_stochastic,
                    noise_scale=noise_scale,
                )
                # NaN guard: early-training stochastic rollouts diverge when
                # forces are still tiny (~3e-4).  The force_floor loss pushes
                # |F| up and the rollout becomes stable within a few epochs.
                if torch.isnan(roll_loss) or torch.isinf(roll_loss):
                    roll_loss = torch.zeros_like(total_loss)
                else:
                    total_loss = total_loss + rollout_w * roll_loss
                loss_components["rollout_nll"] = float(roll_loss.detach().cpu())

            # ── P1.3: Semigroup consistency ──
            if semigroup_w > 0 and "q" in outputs:
                z0 = outputs["q"].detach().clone()
                s0 = outputs.get("bath_s0", torch.zeros_like(z0[:, :, :1]).expand(-1, -1, 16)).detach().clone()
                ca_d = torch.cdist(
                    batch["context_coords"][:, -1, :, 1, :],
                    batch["context_coords"][:, -1, :, 1, :],
                )
                rm = outputs.get("residue_mask")
                tk = batch.get("context_temperature_k", torch.tensor([348.0], device=z0.device))
                if tk.dim() > 1: tk = tk[:, -1]

                semi_loss = semigroup_consistency_loss(
                    model, z0, s0, float(model.base_step_ps),
                    ca_d, rm, tk,
                )
                total_loss = total_loss + semigroup_w * semi_loss
                loss_components["semigroup"] = float(semi_loss.detach().cpu())

            # Clear cached allocator fragments before backward
            if device.type == "cuda":
                torch.cuda.empty_cache()
            scaled_loss = scaler.scale(total_loss / grad_accum)
            scaled_loss.backward()

            if not is_accum_step:
                if config.optimization.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in lasp_params if p.grad is not None],
                        config.optimization.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()

                # Logging
                true_step = global_step // grad_accum
                if true_step % config.log_every == 0:
                    record = {
                        "epoch": epoch,
                        "step": true_step,
                        "loss": float(total_loss.detach().cpu()),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    for name, loss_val in loss_components.items():
                        if torch.is_tensor(loss_val):
                            record[f"loss_{name}"] = float(loss_val.detach().cpu())
                        else:
                            record[f"loss_{name}"] = float(loss_val)
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                    print(json.dumps(record), flush=True)

                if config.checkpoint_every_steps > 0 and (true_step + 1) % config.checkpoint_every_steps == 0:
                    _save_checkpoint(output_dir, model, optimizer, epoch, true_step + 1, config, write_step_file=True)

            global_step += 1

            # ── Explicit max_steps guard ──
            # Force-save checkpoint_latest.pt at the exact target step so
            # the metadata always records the requested budget.
            if max_steps is not None and global_step >= max_steps:
                _save_checkpoint(
                    output_dir, model, optimizer,
                    epoch + 1, global_step, config,
                    write_step_file=False,
                )
                # Also write a sentinel file with the canonical state version
                (output_dir / "canonical_state_version.txt").write_text(
                    "v1\nstage_b_canonical\nuse_temporal=True for all encode_state calls\n"
                )
                return output_dir / "checkpoint_latest.pt"

        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.optimization.epochs:
            _save_checkpoint(output_dir, model, optimizer, epoch + 1, global_step, config, write_step_file=False)

    return output_dir / "checkpoint_latest.pt"


__all__ = ["train_stage_b", "StageBCallback"]
