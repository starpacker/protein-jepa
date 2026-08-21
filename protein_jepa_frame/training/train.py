from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from protein_jepa_frame.config import TrainConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.models.jepa import (
    lagged_state_jepa_loss,
    ProteinJEPA,
    residue_jepa_loss,
    state_jepa_loss,
    state_variance_regularizer,
    variance_regularizer,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _momentum(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(1.0, step / float(total_steps - 1))
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def _lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1.0e-8, (step + 1) / float(warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0, step - warmup_steps) / float(denom))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: TrainConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config_to_dict(config), indent=2))

    device = _device(config.device)
    dataset = build_dataset(config.data,
                            msm_labels_path=config.data.msm_labels_path)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clips,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("dataloader is empty; reduce batch size or clip length")
    steps_per_epoch = config.optimization.steps_per_epoch or len(loader)
    total_steps = steps_per_epoch * config.optimization.epochs

    model = ProteinJEPA(config.model).to(device)
    param_groups = [
        {
            "params": model.context_encoder.parameters(),
            "lr": config.optimization.lr * config.optimization.encoder_lr_scale,
            "base_lr_scale": config.optimization.encoder_lr_scale,
        },
        {"params": model.predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0},
    ]
    if model.temporal_encoder is not None:
        param_groups.append(
            {"params": model.temporal_encoder.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.state_predictor is not None:
        param_groups.append(
            {"params": model.state_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.lagged_state_predictor is not None:
        param_groups.append(
            {"params": model.lagged_state_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.optimization.weight_decay)
    # Pre-register MSM classifiers for all proteins in the training set
    # so they are part of the optimizer from the start
    if hasattr(dataset, 'records') and config.data.msm_labels_path:
        from collections import OrderedDict
        seen_ids = OrderedDict()
        for record in dataset.records:
            seen_ids[record.protein_id] = None
        for pid in seen_ids:
            if pid not in model.state_classifiers:
                model.state_classifiers[pid] = nn.Linear(
                    config.model.latent_dim, 6
                ).to(device)
    optimizer.add_param_group({
        "params": model.state_classifiers.parameters(),
        "lr": config.optimization.lr,
        "base_lr_scale": 1.0,
    })
    masker = SpatiotemporalBlockMasker(config.mask)
    log_path = output_dir / "train.jsonl"

    global_step = 0
    for epoch in range(config.optimization.epochs):
        iterator = iter(loader)
        for local_step in range(steps_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _move_batch(batch, device)
            context_mask, target_mask = masker(batch["residue_mask"])
            lr_mult = _lr_scale(global_step, total_steps, config.optimization.warmup_steps)
            for group in optimizer.param_groups:
                base = config.optimization.lr * float(group.get("base_lr_scale", 1.0))
                group["lr"] = base * lr_mult

            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, context_mask, target_mask)
            residue_loss = residue_jepa_loss(outputs)
            state_loss = state_jepa_loss(outputs)
            lagged_state_loss = lagged_state_jepa_loss(outputs)
            reg = variance_regularizer(outputs)
            state_reg = state_variance_regularizer(outputs)
            total_loss = (
                config.optimization.residue_jepa_weight * residue_loss
                + config.optimization.state_jepa_weight * state_loss
                + config.optimization.lagged_state_jepa_weight * lagged_state_loss
                + config.optimization.variance_reg * reg
                + config.optimization.state_variance_reg * state_reg
            )

            if config.optimization.msm_state_weight > 0:
                msm_state_loss = model.compute_msm_state_loss(outputs, batch)
                total_loss = total_loss + config.optimization.msm_state_weight * msm_state_loss
            else:
                msm_state_loss = total_loss.new_tensor(0.0)

            if config.optimization.vamp2_weight > 0:
                vamp2_loss = model.compute_vamp2_loss(outputs, batch, lag=config.optimization.vamp2_lag)
                total_loss = total_loss + config.optimization.vamp2_weight * vamp2_loss
            else:
                vamp2_loss = total_loss.new_tensor(0.0)

            total_loss.backward()
            if config.optimization.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.grad_clip)
            optimizer.step()
            momentum = _momentum(
                global_step,
                total_steps,
                config.optimization.ema_start,
                config.optimization.ema_end,
            )
            model.update_target_encoder(momentum)

            if global_step % config.log_every == 0:
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(total_loss.detach().cpu()),
                    "jepa_loss": float(residue_loss.detach().cpu()),
                    "residue_jepa_loss": float(residue_loss.detach().cpu()),
                    "state_jepa_loss": float(state_loss.detach().cpu()),
                    "lagged_state_jepa_loss": float(lagged_state_loss.detach().cpu()),
                    "variance_reg": float(reg.detach().cpu()),
                    "state_variance_reg": float(state_reg.detach().cpu()),
                    "msm_state_loss": float(msm_state_loss.detach().cpu()),
                    "vamp2_loss": float(vamp2_loss.detach().cpu()),
                    "lr": optimizer.param_groups[-1]["lr"],
                    "ema": momentum,
                    "target_tokens": int(target_mask.sum().detach().cpu()),
                    "context_tokens": int(context_mask.sum().detach().cpu()),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                print(json.dumps(record), flush=True)

            # Periodic checkpoint every 10k steps (in addition to epoch-level saves)
            if global_step > 0 and global_step % 10000 == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step": global_step,
                    "config": config_to_dict(config),
                }
                torch.save(ckpt, output_dir / "checkpoint_latest.pt")
                print(f"Checkpoint saved at step {global_step}", flush=True)
            global_step += 1

        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.optimization.epochs:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "config": config_to_dict(config),
            }
            torch.save(ckpt, output_dir / "checkpoint_latest.pt")
    return output_dir / "checkpoint_latest.pt"
