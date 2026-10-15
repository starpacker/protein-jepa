"""Stage A training: joint masked JEPA + multi-horizon endpoint forecasting.

Gate 1 requirements:
1. Joint dataloader: masked clip + paired physical-horizon batches at fixed ratio.
2. online branch receives mask; EMA branch always eval + no_grad + unmasked.
3. Multi-horizon targets use horizon_time_ps; frame indices only for data loading.
4. Persistence and horizon-shuffle baselines as negative controls.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protein_jepa_frame.config import TrainConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_clips, collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset, validate_real_paired_window_protocol
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.stage_a import StageAJEPA, stage_a_total_loss
from protein_jepa_frame.models.stage_a.objectives import (
    stage_a_displacement_loss,
    temporal_covariance_regularizer,
    temporal_variance_regularizer,
    temporal_velocity_loss,
)
from protein_jepa_frame.training.train_utils import (
    _device, _set_reproducibility, _move_batch,
    _momentum, _lr_scale, _warmup_scale,
    _save_checkpoint, _load_resume_checkpoint,
    _load_init_checkpoint,
)


class StageAJointBatchIterator:
    """Iterator that yields (masked_clip_batch, paired_batch) at a fixed ratio.
    
    Pairs are drawn from both the mask loader and paired loader,
    interleaving batch types with `paired_ratio` fraction of paired batches.
    All batches use horizon_time_ps (seconds) for time conditioning,
    and frame indices only internally for data loading.
    """
    
    def __init__(
        self,
        mask_loader: DataLoader,
        paired_loader: DataLoader,
        paired_ratio: float = 0.5,
        max_pairs_per_horizon: int = 4,
        seed: int = 42,
    ):
        self.mask_loader = mask_loader
        self.paired_loader = paired_loader
        self.paired_ratio = paired_ratio
        self.max_pairs_per_horizon = max_pairs_per_horizon
        self.rng = random.Random(seed)
        self.mask_iter = iter(mask_loader)
        self.paired_iter = iter(paired_loader)
        # Ensure paired loader uses horizon_time_ps properly
        self._verify_time_conditioning(paired_loader)
    
    def _verify_time_conditioning(self, loader: DataLoader):
        """Verify paired batches contain horizon_time_ps."""
        try:
            sample = next(iter(loader))
            if "horizon_time_ps" not in sample:
                raise ValueError(
                    "paired loader batch missing horizon_time_ps; "
                    "collate_paired_windows must export this field"
                )
        except StopIteration:
            pass  # empty loader
    
    def __iter__(self):
        self.mask_iter = iter(self.mask_loader)
        self.paired_iter = iter(self.paired_loader)
        return self
    
    def __next__(self) -> tuple[dict, dict | None, str]:
        """Returns (batch, None, "mask") or (paired_batch, None, "paired")."""
        if self.rng.random() < self.paired_ratio:
            try:
                paired = next(self.paired_iter)
            except StopIteration:
                self.paired_iter = iter(self.paired_loader)
                paired = next(self.paired_iter)
            return paired, None, "paired"
        else:
            try:
                mask_batch = next(self.mask_iter)
            except StopIteration:
                self.mask_iter = iter(self.mask_loader)
                mask_batch = next(self.mask_iter)
            return mask_batch, None, "mask"


def train_stage_a(config: TrainConfig) -> Path:
    """Train Stage A: joint masked JEPA + multi-horizon endpoint forecast.
    
    Returns path to latest checkpoint.
    """
    if config.init_checkpoint and config.resume_checkpoint:
        raise ValueError("use either init_checkpoint or resume_checkpoint, not both")

    validate_real_paired_window_protocol(config.data)

    _set_reproducibility(config.seed, config.deterministic)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config_to_dict(config), indent=2)
    )
    (output_dir / "stage.txt").write_text("stage_a\n")

    device = _device(config.device)

    # ── Build mask dataset (clip sampling) ──
    from dataclasses import replace as dc_replace
    mask_data_config = dc_replace(config.data, training_mode="clip")
    mask_dataset = build_dataset(
        mask_data_config,
        msm_labels_path=config.data.msm_labels_path,
    )
    mask_sampler = None
    if config.data.sampling_mode == "hierarchical":
        if not hasattr(mask_dataset, "entries"):
            raise ValueError("hierarchical sampling requires a trajectory clip dataset")
        mask_sampler = HierarchicalClipSampler(
            mask_dataset,
            num_samples=len(mask_dataset),
            seed=config.seed,
            dataset_weights=config.data.dataset_weights,
        )
    mask_loader = DataLoader(
        mask_dataset,
        batch_size=config.data.batch_size,
        shuffle=mask_sampler is None,
        sampler=mask_sampler,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_clips,
        drop_last=True,
    )

    # ── Build paired dataset ──
    paired_data_config = dc_replace(config.data, training_mode="paired_window")
    if not paired_data_config.paired_horizons_frames:
        raise ValueError("Stage A joint training requires paired_horizons_frames")
    
    paired_dataset = build_dataset(paired_data_config)
    paired_loader = DataLoader(
        paired_dataset,
        batch_size=max(1, config.data.batch_size // 2),
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_paired_windows,
        drop_last=True,
    )

    if len(mask_loader) == 0 or len(paired_loader) == 0:
        raise ValueError("empty dataloader; reduce batch size")

    # Joint iterator
    batch_iter = StageAJointBatchIterator(
        mask_loader, paired_loader,
        paired_ratio=getattr(config, 'stage_a_paired_ratio', 0.5),
        seed=config.seed + 1,
    )

    # ── Build model ──
    model = StageAJEPA(config.model).to(device)
    if config.init_checkpoint:
        _load_init_checkpoint(
            model, config.init_checkpoint, device,
            config.strict_init, config.init_exclude_prefixes,
            False,  # init_copy_temporal_to_offline
        )
        if config.sync_target_after_init:
            model.sync_target_encoders()

    # ── Optimizer ──
    param_groups = [
        {
            "params": model.context_encoder.parameters(),
            "lr": config.optimization.lr * config.optimization.encoder_lr_scale,
            "base_lr_scale": config.optimization.encoder_lr_scale,
        },
        {
            "params": model.predictor.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        },
    ]
    if model.temporal_encoder is not None:
        param_groups.append({
            "params": model.temporal_encoder.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        })
    if model.dynamic_encoder is not None:
        param_groups.append({
            "params": model.dynamic_encoder.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        })
    if model.dynamic_jepa_predictor is not None:
        param_groups.append({
            "params": model.dynamic_jepa_predictor.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        })
    if model.state_predictor is not None:
        param_groups.append({
            "params": model.state_predictor.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        })
    if model.paired_delta_predictor is not None:
        param_groups.append({
            "params": model.paired_delta_predictor.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        })
    if model.offline_temporal_encoder is not None:
        param_groups.append({
            "params": model.offline_temporal_encoder.parameters(),
            "lr": config.optimization.lr,
            "base_lr_scale": 1.0,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.optimization.weight_decay,
    )

    # ── Masker ──
    required_future_horizon = max(
        getattr(config.model, 'state_prediction_lags', ()) or (0,),
        default=0,
    )
    masker = SpatiotemporalBlockMasker(
        config.mask,
        required_future_horizon=required_future_horizon,
    )

    # ── Training loop ──
    steps_per_epoch = config.optimization.steps_per_epoch or len(mask_loader)
    total_steps = steps_per_epoch * config.optimization.epochs
    global_step = 0

    if config.resume_checkpoint:
        global_step = _load_resume_checkpoint(
            model, optimizer, config.resume_checkpoint, device
        )

    log_path = output_dir / "train.jsonl"
    
    if global_step >= total_steps:
        _save_checkpoint(
            output_dir, model, optimizer,
            config.optimization.epochs, global_step,
            config, write_step_file=False,
        )
        return output_dir / "checkpoint_latest.pt"

    skip_controls = getattr(config, 'stage_a_skip_negative_controls', False)
    start_epoch = global_step // steps_per_epoch

    for epoch in range(start_epoch, config.optimization.epochs):
        for local_step in range(
            global_step % steps_per_epoch if epoch == start_epoch else 0,
            steps_per_epoch,
        ):
            batch, _, task = next(batch_iter)
            batch = _move_batch(batch, device)

            # LR schedule
            lr_mult = _lr_scale(
                global_step, total_steps,
                config.optimization.warmup_steps,
            )
            for group in optimizer.param_groups:
                base = config.optimization.lr * float(
                    group.get("base_lr_scale", 1.0)
                )
                group["lr"] = base * lr_mult

            optimizer.zero_grad(set_to_none=True)

            if task == "mask":
                context_mask, target_mask = masker(batch["residue_mask"])
                outputs = model.forward_mask_prediction(
                    batch, context_mask, target_mask,
                )
                loss, log_dict = stage_a_total_loss(
                    mask_outputs=outputs,
                    paired_outputs=None,
                    mask_weight=config.optimization.residue_jepa_weight,
                    residue_weight=config.optimization.residue_jepa_weight,
                    state_weight=config.optimization.state_jepa_weight,
                    dynamic_activity_weight=config.optimization.stage_a_dynamic_activity_weight,
                    dynamic_covariance_weight=config.optimization.stage_a_dynamic_covariance_weight,
                    dynamic_variance_floor=config.optimization.stage_a_dynamic_variance_floor,
                )
            else:  # "paired"
                # The batch from collate_paired_windows is already flat:
                # {context_coords, future_coords, horizon_time_ps, ...}
                outputs = model.forward_paired(batch)
                
                # Persistence baseline (negative control)
                if not skip_controls:
                    with torch.no_grad():
                        persistence_loss = F.smooth_l1_loss(
                            outputs["context_endpoint"],
                            outputs["future_endpoint"],
                        )
                
                # Horizon-shuffle control (negative control)
                horizon_shuffle_loss = None
                if not skip_controls and "horizon_time_ps" in batch:
                    with torch.no_grad():
                        perm = torch.randperm(
                            batch["horizon_time_ps"].shape[0],
                            device=device,
                        )
                        shuffled_horizon = batch["horizon_time_ps"][perm]
                        if model.paired_delta_predictor is not None:
                            shuffle_pred = model.paired_delta_predictor(
                                outputs["context_endpoint"],
                                shuffled_horizon,
                                temperature_k=batch.get("temperature_k"),
                                dataset_index=batch.get("dataset_index"),
                            )
                            if hasattr(outputs, "__getitem__"):
                                anchor = outputs.get(
                                    "context_endpoint", outputs["context_endpoint"]
                                )
                            else:
                                anchor = outputs["context_endpoint"]
                            if model.config.paired_teacher_space_delta:
                                shuffle_state = outputs["future_endpoint"].detach() + shuffle_pred
                            else:
                                shuffle_state = anchor + shuffle_pred
                            horizon_shuffle_loss = F.smooth_l1_loss(
                                shuffle_state, outputs["future_endpoint"],
                            )
                
                loss, log_dict = stage_a_total_loss(
                    mask_outputs=None,
                    paired_outputs=outputs,
                    paired_weight=1.0,
                    mask_weight=0.0,
                    dynamic_jepa_weight=config.optimization.stage_a_dynamic_jepa_weight,
                    dynamic_activity_weight=config.optimization.stage_a_dynamic_activity_weight,
                    dynamic_covariance_weight=config.optimization.stage_a_dynamic_covariance_weight,
                    dynamic_variance_floor=config.optimization.stage_a_dynamic_variance_floor,
                )
                log_dict["task"] = "paired"
                if not skip_controls:
                    log_dict["persistence_baseline"] = float(
                        persistence_loss.detach().cpu()
                    )
                    if horizon_shuffle_loss is not None:
                        log_dict["horizon_shuffle_loss"] = float(
                            horizon_shuffle_loss.detach().cpu()
                        )

            loss.backward()

            if config.optimization.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optimization.grad_clip,
                )

            optimizer.step()

            # EMA update
            momentum = _momentum(
                global_step, total_steps,
                config.optimization.ema_start,
                config.optimization.ema_end,
            )
            model.update_target_encoder(momentum)

            # Logging
            if global_step % config.log_every == 0:
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "task": task,
                    "lr": optimizer.param_groups[-1]["lr"],
                    "ema": momentum,
                    **log_dict,
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                print(json.dumps(record), flush=True)

            completed_step = global_step + 1
            if (
                config.checkpoint_every_steps > 0
                and completed_step % config.checkpoint_every_steps == 0
            ):
                _save_checkpoint(
                    output_dir, model, optimizer,
                    epoch, completed_step, config,
                    write_step_file=True,
                )
            global_step += 1

        if (
            (epoch + 1) % config.save_every == 0
            or epoch + 1 == config.optimization.epochs
        ):
            _save_checkpoint(
                output_dir, model, optimizer,
                epoch + 1, global_step, config,
                write_step_file=False,
            )

    return output_dir / "checkpoint_latest.pt"
