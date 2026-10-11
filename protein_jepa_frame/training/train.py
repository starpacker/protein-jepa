"""Training entry points for Protein JEPA.

All implementations now live in:
  - train_utils.py: shared utilities (device, reproducibility, checkpointing)
  - train_paired.py: train_paired function (paired-window training)
  - train.py (this file): train function + backward-compat re-exports
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from protein_jepa_frame.config import TrainConfig, config_to_dict
from protein_jepa_frame.data.collate import collate_clips, collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import (
    kinetic_vampnet_loss,
    lagged_state_jepa_terms,
    lagged_state_semigroup_terms,
    linear_dynamics_jepa_loss,
    physchem_prediction_loss,
    ProteinJEPA,
    residue_jepa_loss,
    state_jepa_loss,
    state_transition_jepa_loss,
    state_covariance_regularizer,
    state_feature_variance_regularizer,
    state_variance_regularizer,
    variance_regularizer,
    paired_endpoint_variance_regularizer,
    paired_causal_alignment_loss,
    paired_forecast_terms,
    gaussian_transition_nll,
    observable_prediction_loss,
    offline_forecast_terms,
    slow_forecast_terms,
    reversible_kinetic_terms,
    metastable_vamp_terms,
    v4_slow_kinetic_terms,
    v4_physics_terms,
    v5_contextual_dynamics_terms,
    v51_sparse_motion_terms,
    tmjepa_terms,
    local_observable_prediction_loss,
    closed_local_state_terms,
    upgrade_tmjepa_v3_state_dict,
    upgrade_lspwm_state_dict,
)
from protein_jepa_frame.models.lspwm import vamp2_loss
from protein_jepa_frame.models.opf import opf_terms

from protein_jepa_frame.training.train_utils import (
    _device,
    _set_reproducibility,
    _move_batch,
    _momentum,
    _lr_scale,
    _warmup_scale,
    _module_gradient_norm,
    _tmjepa_optimizer_groups,
    _prefixes_overlap,
    _validate_init_freeze_contract,
    _load_init_checkpoint,
    _checkpoint_payload,
    _save_checkpoint,
    _load_resume_checkpoint,
    _reset_lspwm_conservative_correction,
    _lspwm_sequence_vamp_loss,
    _balanced_calibrated_endpoint_mse,
)


def train(config: TrainConfig) -> Path:
    # Imports at the checkpoint save path
    from protein_jepa_frame.training.train_paired import train_paired

    import json
    import torch
    from pathlib import Path

    training_mode = config.data.training_mode
    if training_mode in ("paired", "paired_window"):
        output_dir = train_paired(config)
    else:
        raise NotImplementedError(
            f"Training mode '{training_mode}' is not implemented. "
            f"Use 'paired' (paired-window JEPA training)."
        )

    # append train.jsonl if it exists to stage-agnostic log
    log_path = Path(output_dir) / "train.jsonl"
    if log_path.exists():
        with open(log_path) as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        manifest = os.path.join(output_dir, "run_manifest.json")
        payload = {"train_log_size": len(lines)}
        try:
            with open(manifest, "w") as fh:
                json.dump(payload, fh)
        except Exception:
            pass

    return output_dir


__all__ = [
    "train",
    "_device", "_set_reproducibility", "_move_batch",
    "_momentum", "_lr_scale", "_warmup_scale",
    "_module_gradient_norm", "_tmjepa_optimizer_groups",
    "_prefixes_overlap", "_validate_init_freeze_contract",
    "_load_init_checkpoint", "_checkpoint_payload",
    "_save_checkpoint", "_load_resume_checkpoint",
    "_reset_lspwm_conservative_correction",
    "_lspwm_sequence_vamp_loss",
    "_balanced_calibrated_endpoint_mse",
]


def train(config: TrainConfig) -> Path:
    # Stage C needs the frozen Stage-A/B init checkpoint plus a separate
    # decoder resume checkpoint. Other trainers retain mutual exclusion.
    if (config.init_checkpoint and config.resume_checkpoint
            and getattr(config.model, "model_type", "") != "stage_c"):
        raise ValueError("use either init_checkpoint or resume_checkpoint, not both")
    _validate_init_freeze_contract(config)
    if getattr(config.model, "model_type", "") == "stage_b":
        from protein_jepa_frame.training.stage_b import train_stage_b
        return train_stage_b(config)
    if getattr(config.model, "model_type", "") == "stage_c":
        from protein_jepa_frame.training.stage_c import train_stage_c
        return train_stage_c(config)
    if config.data.training_mode == "paired_window":
        from protein_jepa_frame.training.train_paired import train_paired
        return train_paired(config)

    _set_reproducibility(config.seed, config.deterministic)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config_to_dict(config), indent=2))

    device = _device(config.device)
    dataset = build_dataset(config.data,
                            msm_labels_path=config.data.msm_labels_path)
    sampler = None
    if config.data.sampling_mode == "hierarchical":
        if not hasattr(dataset, "entries"):
            raise ValueError("hierarchical sampling requires a trajectory clip dataset")
        sampler = HierarchicalClipSampler(
            dataset,
            num_samples=len(dataset),
            seed=config.seed,
            dataset_weights=config.data.dataset_weights,
        )
    elif config.data.sampling_mode != "clip_uniform":
        raise ValueError(f"unknown sampling mode {config.data.sampling_mode}")
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
    if config.init_checkpoint:
        _load_init_checkpoint(
            model,
            config.init_checkpoint,
            device,
            config.strict_init,
            config.init_exclude_prefixes,
            config.init_copy_temporal_to_offline,
        )
        if config.sync_target_after_init:
            model.sync_target_encoders()
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
    if model.linear_dynamics_predictor is not None:
        param_groups.append(
            {
                "params": model.linear_dynamics_predictor.parameters(),
                "lr": config.optimization.lr,
                "base_lr_scale": 1.0,
            }
        )
    if model.physchem_predictor is not None:
        param_groups.append(
            {"params": model.physchem_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.kinetic_readout is not None:
        param_groups.append(
            {"params": model.kinetic_readout.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.optimization.weight_decay)
    global_step = 0
    if config.resume_checkpoint:
        global_step = _load_resume_checkpoint(model, optimizer, config.resume_checkpoint, device)
    required_future_horizon = max(config.model.state_prediction_lags, default=0)
    masker = SpatiotemporalBlockMasker(config.mask, required_future_horizon=required_future_horizon)
    log_path = output_dir / "train.jsonl"
    if global_step >= total_steps:
        _save_checkpoint(output_dir, model, optimizer, config.optimization.epochs, global_step, config, write_step_file=False)
        return output_dir / "checkpoint_latest.pt"
    start_epoch = global_step // steps_per_epoch
    for epoch in range(start_epoch, config.optimization.epochs):
        iterator = iter(loader)
        local_start = global_step % steps_per_epoch if epoch == start_epoch else 0
        for local_step in range(local_start, steps_per_epoch):
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
            lagged_state_terms = lagged_state_jepa_terms(outputs)
            if lagged_state_terms:
                lagged_state_loss = torch.stack([term["loss"] for term in lagged_state_terms.values()]).mean()
            else:
                lagged_state_loss = outputs["predictions"].new_tensor(0.0)
            semigroup_terms = lagged_state_semigroup_terms(outputs, model.lagged_state_predictor)
            if semigroup_terms:
                semigroup_loss = torch.stack([term["loss"] for term in semigroup_terms.values()]).mean()
            else:
                semigroup_loss = outputs["predictions"].new_tensor(0.0)
            state_transition_loss = state_transition_jepa_loss(outputs)
            physchem_loss = physchem_prediction_loss(outputs)
            linear_dynamics_loss = linear_dynamics_jepa_loss(outputs)
            kinetic_vamp_loss = kinetic_vampnet_loss(outputs, lag=config.optimization.kinetic_vamp_lag)
            reg = variance_regularizer(outputs)
            state_reg = state_variance_regularizer(outputs)
            state_feature_reg = state_feature_variance_regularizer(outputs)
            state_covariance_reg = state_covariance_regularizer(outputs)
            aux_mult = _warmup_scale(global_step, config.optimization.auxiliary_warmup_steps)
            semigroup_weight = config.optimization.semigroup_weight * aux_mult
            state_transition_weight = config.optimization.state_transition_weight * aux_mult
            physchem_weight = config.optimization.physchem_weight * aux_mult
            linear_dynamics_weight = config.optimization.linear_dynamics_weight * aux_mult
            variance_weight = config.optimization.variance_reg * aux_mult
            state_variance_weight = config.optimization.state_variance_reg * aux_mult
            state_feature_variance_weight = config.optimization.state_feature_variance_reg * aux_mult
            state_covariance_weight = config.optimization.state_covariance_reg * aux_mult
            msm_state_weight = config.optimization.msm_state_weight * aux_mult
            vamp2_weight = config.optimization.vamp2_weight * aux_mult
            kinetic_vamp_weight = config.optimization.kinetic_vamp_weight * aux_mult
            total_loss = (
                config.optimization.residue_jepa_weight * residue_loss
                + config.optimization.state_jepa_weight * state_loss
                + config.optimization.lagged_state_jepa_weight * lagged_state_loss
                + semigroup_weight * semigroup_loss
                + state_transition_weight * state_transition_loss
                + physchem_weight * physchem_loss
                + linear_dynamics_weight * linear_dynamics_loss
                + variance_weight * reg
                + state_variance_weight * state_reg
                + state_feature_variance_weight * state_feature_reg
                + state_covariance_weight * state_covariance_reg
                + kinetic_vamp_weight * kinetic_vamp_loss
            )
            if msm_state_weight > 0:
                msm_state_loss = model.compute_msm_state_loss(outputs, batch)
                total_loss = total_loss + msm_state_weight * msm_state_loss
            msm_state_loss = msm_state_loss if 'msm_state_loss' in dir() else total_loss.new_tensor(0.0)
            if vamp2_weight > 0:
                vamp2_loss_value = model.compute_vamp2_loss(outputs, batch, lag=config.optimization.vamp2_lag)
                total_loss = total_loss + vamp2_weight * vamp2_loss_value
            vamp2_loss_value = vamp2_loss_value if 'vamp2_loss_value' in dir() else total_loss.new_tensor(0.0)
            total_loss.backward()
            if config.optimization.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.grad_clip)
            optimizer.step()
            momentum = _momentum(global_step, total_steps, config.optimization.ema_start, config.optimization.ema_end)
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
                    "state_transition_jepa_loss": float(state_transition_loss.detach().cpu()),
                    "physchem_loss": float(physchem_loss.detach().cpu()),
                    "linear_dynamics_jepa_loss": float(linear_dynamics_loss.detach().cpu()),
                    "kinetic_vampnet_loss": float(kinetic_vamp_loss.detach().cpu()),
                    "variance_reg": float(reg.detach().cpu()),
                    "state_variance_reg": float(state_reg.detach().cpu()),
                    "state_feature_variance_reg": float(state_feature_reg.detach().cpu()),
                    "state_covariance_reg": float(state_covariance_reg.detach().cpu()),
                    "msm_state_loss": float(msm_state_loss.detach().cpu()),
                    "vamp2_loss": float(vamp2_loss_value.detach().cpu()),
                    "auxiliary_warmup": aux_mult,
                    "lr": optimizer.param_groups[-1]["lr"],
                    "ema": momentum,
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                print(json.dumps(record), flush=True)
            completed_step = global_step + 1
            if config.checkpoint_every_steps > 0 and completed_step % config.checkpoint_every_steps == 0:
                _save_checkpoint(output_dir, model, optimizer, epoch, completed_step, config, write_step_file=True)
            global_step += 1
        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.optimization.epochs:
            _save_checkpoint(output_dir, model, optimizer, epoch + 1, global_step, config, write_step_file=False)
    return output_dir / "checkpoint_latest.pt"
