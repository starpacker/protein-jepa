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
    metastable_vamp_terms,
    v4_slow_kinetic_terms,
    v4_physics_terms,
    local_observable_prediction_loss,
    closed_local_state_terms,
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


def _warmup_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup_steps))


def _load_init_checkpoint(
    model: ProteinJEPA,
    checkpoint_path: str,
    device: torch.device,
    strict: bool,
    exclude_prefixes: tuple[str, ...] = (),
    copy_temporal_to_offline: bool = False,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if copy_temporal_to_offline:
        remapped = {}
        for key, value in state.items():
            if key.startswith("temporal_encoder."):
                remapped["offline_temporal_encoder." + key.removeprefix("temporal_encoder.")] = value
            elif key.startswith("target_temporal_encoder."):
                remapped["target_offline_temporal_encoder." + key.removeprefix("target_temporal_encoder.")] = value
        state = {**state, **remapped}
    excluded_keys = []
    if exclude_prefixes:
        if strict:
            raise ValueError("init checkpoint exclusions require strict_init=false")
        excluded_keys = [key for key in state if key.startswith(exclude_prefixes)]
        state = {key: value for key, value in state.items() if key not in excluded_keys}
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        print(
            json.dumps(
                {
                    "event": "loaded_init_checkpoint",
                    "checkpoint": checkpoint_path,
                    "excluded_keys": excluded_keys,
                    "missing_keys": list(incompatible.missing_keys),
                    "unexpected_keys": list(incompatible.unexpected_keys),
                }
            ),
            flush=True,
        )


def _checkpoint_payload(
    model: ProteinJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: TrainConfig,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config_to_dict(config),
    }


def _save_checkpoint(
    output_dir: Path,
    model: ProteinJEPA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: TrainConfig,
    *,
    write_step_file: bool,
) -> None:
    ckpt = _checkpoint_payload(model, optimizer, epoch, step, config)
    torch.save(ckpt, output_dir / "checkpoint_latest.pt")
    if write_step_file:
        torch.save(ckpt, output_dir / f"checkpoint_step{step}.pt")
        print(f"Checkpoint saved at step {step}", flush=True)


def _load_resume_checkpoint(
    model: ProteinJEPA,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "optimizer" not in ckpt:
        raise ValueError(f"resume checkpoint {checkpoint_path} must contain model and optimizer state")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    print(
        json.dumps(
            {
                "event": "resumed_checkpoint",
                "checkpoint": checkpoint_path,
                "epoch": ckpt.get("epoch"),
                "step": step,
            }
        ),
        flush=True,
    )
    return step


def train_paired(config: TrainConfig) -> Path:
    """Train the V2.2 paired endpoint and slow-state objectives."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config_to_dict(config), indent=2))
    device = _device(config.device)
    dataset = build_dataset(config.data, msm_labels_path=config.data.msm_labels_path)
    sampler = None
    if config.data.sampling_mode == "hierarchical":
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
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("paired dataloader is empty")
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
        {"params": model.context_encoder.parameters(), "lr": config.optimization.lr * config.optimization.encoder_lr_scale, "base_lr_scale": config.optimization.encoder_lr_scale},
        {"params": model.temporal_encoder.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0},
        {"params": model.paired_delta_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0},
    ]
    if model.offline_temporal_encoder is not None:
        param_groups.append(
            {
                "params": model.offline_temporal_encoder.parameters(),
                "lr": config.optimization.lr * config.optimization.offline_lr_scale,
                "base_lr_scale": config.optimization.offline_lr_scale,
            }
        )
    if model.offline_paired_delta_predictor is not None:
        param_groups.append(
            {"params": model.offline_paired_delta_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.gaussian_transition is not None:
        param_groups.append(
            {"params": model.gaussian_transition.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.observable_head is not None:
        param_groups.append(
            {"params": model.observable_head.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.slow_state_projector is not None:
        param_groups.extend(
            [
                {"params": model.slow_state_projector.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0},
                {"params": model.slow_delta_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0},
            ]
        )
    if model.metastable_state_head is not None:
        param_groups.append(
            {"params": model.metastable_state_head.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.local_dynamics_head is not None:
        param_groups.append(
            {"params": model.local_dynamics_head.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.closed_local_operator is not None:
        param_groups.append(
            {"params": model.closed_local_operator.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.v4_slow_kinetic_model is not None:
        param_groups.append(
            {
                "params": model.v4_slow_kinetic_model.parameters(),
                "lr": config.optimization.lr,
                "base_lr_scale": 1.0,
            }
        )
    if model.v4_physics_slow_model is not None:
        param_groups.append(
            {
                "params": model.v4_physics_slow_model.parameters(),
                "lr": config.optimization.lr,
                "base_lr_scale": 1.0,
            }
        )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.optimization.weight_decay)
    global_step = 0
    if config.resume_checkpoint:
        global_step = _load_resume_checkpoint(model, optimizer, config.resume_checkpoint, device)
    log_path = output_dir / "train.jsonl"
    if global_step >= total_steps:
        _save_checkpoint(output_dir, model, optimizer, config.optimization.epochs, global_step, config, write_step_file=False)
        return output_dir / "checkpoint_latest.pt"

    start_epoch = global_step // steps_per_epoch
    for epoch in range(start_epoch, config.optimization.epochs):
        iterator = iter(loader)
        local_start = global_step % steps_per_epoch if epoch == start_epoch else 0
        for _ in range(local_start, steps_per_epoch):
            try:
                batch = _move_batch(next(iterator), device)
            except StopIteration:
                iterator = iter(loader)
                batch = _move_batch(next(iterator), device)
            lr_mult = _lr_scale(global_step, total_steps, config.optimization.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.optimization.lr * float(group.get("base_lr_scale", 1.0)) * lr_mult
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_paired(batch)
            terms = paired_forecast_terms(outputs)
            gaussian_loss = gaussian_transition_nll(outputs)
            observable_loss = observable_prediction_loss(outputs)
            offline_terms = offline_forecast_terms(outputs)
            slow_terms = slow_forecast_terms(outputs) if "slow_delta_prediction" in outputs else None
            metastable_terms = metastable_vamp_terms(outputs)
            local_observable_loss = local_observable_prediction_loss(outputs)
            closed_terms = closed_local_state_terms(
                outputs,
                margin=config.optimization.horizon_ranking_margin,
                difference_min_gap=config.optimization.horizon_difference_min_gap,
                robust_beta=config.optimization.closed_local_robust_beta,
                student_t_df=config.optimization.closed_local_student_t_df,
            )
            v4_slow_terms = v4_slow_kinetic_terms(
                outputs, lags=config.model.v4_slow_vamp_lags
            )
            v4_physics = v4_physics_terms(
                outputs,
                lags=config.model.v4_physics_vamp_lags,
                force_scale=config.model.v4_physics_force_scale,
                per_trajectory_vamp=config.model.v4_physics_per_trajectory_vamp,
                horizon_ranking_margin=config.optimization.v4_physics_horizon_ranking_margin,
            )
            endpoint_reg = paired_endpoint_variance_regularizer(outputs)
            alignment_loss = paired_causal_alignment_loss(outputs)
            total_loss = (
                config.optimization.paired_delta_weight * terms["delta_loss"]
                + config.optimization.paired_state_weight * terms["state_loss"]
                + config.optimization.paired_endpoint_variance_weight * endpoint_reg
                + config.optimization.gaussian_nll_weight * gaussian_loss
                + config.optimization.observable_weight * observable_loss
                + config.optimization.offline_forecast_weight * (
                    offline_terms["delta_loss"] + 0.25 * offline_terms["state_loss"]
                )
                + config.optimization.causal_alignment_weight * alignment_loss
                + config.optimization.metastable_vamp_weight * metastable_terms["vamp_loss"]
                + config.optimization.metastable_balance_weight * metastable_terms["balance_loss"]
                + config.optimization.metastable_entropy_weight * metastable_terms["assignment_entropy"]
                + config.optimization.local_observable_weight * local_observable_loss
                + config.optimization.closed_local_state_weight * closed_terms["state_loss"]
                + config.optimization.horizon_ranking_weight * closed_terms["ranking_loss"]
                + config.optimization.horizon_difference_weight * closed_terms["difference_loss"]
                + config.optimization.closed_local_robust_state_weight * closed_terms["robust_state_loss"]
                + config.optimization.horizon_robust_difference_weight * closed_terms["robust_difference_loss"]
                + config.optimization.closed_local_student_t_weight * closed_terms["student_t_nll"]
                + config.optimization.closed_local_base_step_weight * closed_terms["base_state_loss"]
                + config.optimization.closed_local_robust_base_step_weight * closed_terms["base_robust_loss"]
                + config.optimization.v4_slow_reconstruction_weight * v4_slow_terms["reconstruction_loss"]
                + config.optimization.v4_slow_vamp_weight * v4_slow_terms["vamp_loss"]
                + config.optimization.v4_slow_balance_weight * v4_slow_terms["balance_loss"]
                + config.optimization.v4_slow_entropy_weight * v4_slow_terms["assignment_entropy"]
                + config.optimization.v4_slow_base_transition_weight
                * v4_slow_terms["base_transition_loss"]
                + config.optimization.v4_slow_rollout_transition_weight
                * v4_slow_terms["rollout_transition_loss"]
                + config.optimization.v4_physics_reconstruction_weight
                * v4_physics["reconstruction_loss"]
                + config.optimization.v4_physics_vamp_weight * v4_physics["vamp_loss"]
                + config.optimization.v4_physics_balance_weight * v4_physics["balance_loss"]
                + config.optimization.v4_physics_entropy_weight
                * v4_physics["assignment_entropy"]
                + config.optimization.v4_physics_cross_covariance_weight
                * v4_physics["cross_covariance_loss"]
                + config.optimization.v4_physics_increment_covariance_weight
                * v4_physics["increment_covariance_loss"]
                + config.optimization.v4_physics_timescale_weight
                * v4_physics["timescale_separation_loss"]
                + config.optimization.v4_physics_trajectory_balance_weight
                * v4_physics["trajectory_balance_loss"]
                + config.optimization.v4_physics_activity_weight
                * v4_physics["activity_loss"]
                + config.optimization.v4_physics_latent_variance_weight
                * v4_physics["latent_variance_loss"]
                + config.optimization.v4_physics_drift_weight * v4_physics["drift_loss"]
                + config.optimization.v4_physics_force_weight * v4_physics["force_loss"]
                + config.optimization.v4_physics_stationary_weight
                * v4_physics["stationary_loss"]
                + config.optimization.v4_physics_base_transition_weight
                * v4_physics["base_transition_loss"]
                + config.optimization.v4_physics_rollout_transition_weight
                * v4_physics["rollout_transition_loss"]
                + config.optimization.v4_physics_transition_nll_weight
                * v4_physics["transition_nll"]
                + config.optimization.v4_physics_horizon_ranking_weight
                * v4_physics["horizon_ranking_loss"]
                + config.optimization.v4_physics_horizon_difference_weight
                * v4_physics["horizon_difference_loss"]
            )
            if slow_terms is not None:
                total_loss = total_loss + (
                    config.optimization.slow_delta_weight * slow_terms["delta_loss"]
                    + config.optimization.slow_state_weight * slow_terms["state_loss"]
                    + config.optimization.slow_contrastive_weight * slow_terms["contrastive_loss"]
                    + config.optimization.slow_variance_weight * slow_terms["variance_loss"]
                    + config.optimization.slow_covariance_weight * slow_terms["covariance_loss"]
                )
            total_loss.backward()
            if config.optimization.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.grad_clip)
            optimizer.step()
            momentum = _momentum(global_step, total_steps, config.optimization.ema_start, config.optimization.ema_end)
            model.update_target_encoder(momentum)
            if global_step % config.log_every == 0:
                horizon_log = {}
                for horizon in sorted(set(int(x) for x in outputs["horizon_frames"].detach().cpu().tolist())):
                    mask = outputs["horizon_frames"] == horizon
                    if not mask.any():
                        continue
                    subset = {
                        key: value[mask]
                        for key, value in outputs.items()
                        if torch.is_tensor(value) and value.shape[0] == mask.shape[0]
                    }
                    subterms = paired_forecast_terms(subset)
                    horizon_log[str(horizon)] = {
                        "pairs": int(mask.sum().detach().cpu()),
                        "delta_loss": float(subterms["delta_loss"].detach().cpu()),
                        "state_loss": float(subterms["state_loss"].detach().cpu()),
                        "persistence_delta_loss": float(subterms["persistence_delta_loss"].detach().cpu()),
                        "skill": float(subterms["skill"].detach().cpu()),
                    }
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(total_loss.detach().cpu()),
                    "paired_delta_loss": float(terms["delta_loss"].detach().cpu()),
                    "paired_state_loss": float(terms["state_loss"].detach().cpu()),
                    "paired_persistence_delta_loss": float(terms["persistence_delta_loss"].detach().cpu()),
                    "paired_skill": float(terms["skill"].detach().cpu()),
                    "paired_endpoint_variance": float(endpoint_reg.detach().cpu()),
                    "causal_alignment_loss": float(alignment_loss.detach().cpu()),
                    "gaussian_nll": float(gaussian_loss.detach().cpu()),
                    "observable_loss": float(observable_loss.detach().cpu()),
                    "offline_delta_loss": float(offline_terms["delta_loss"].detach().cpu()),
                    "offline_state_loss": float(offline_terms["state_loss"].detach().cpu()),
                    "offline_skill": float(offline_terms["skill"].detach().cpu()),
                    "metastable_vamp_loss": float(metastable_terms["vamp_loss"].detach().cpu()),
                    "metastable_balance_loss": float(metastable_terms["balance_loss"].detach().cpu()),
                    "metastable_assignment_entropy": float(metastable_terms["assignment_entropy"].detach().cpu()),
                    "local_observable_loss": float(local_observable_loss.detach().cpu()),
                    "closed_local_state_loss": float(closed_terms["state_loss"].detach().cpu()),
                    "closed_local_persistence_loss": float(closed_terms["persistence_loss"].detach().cpu()),
                    "closed_local_skill": float(closed_terms["skill"].detach().cpu()),
                    "closed_local_ranking_loss": float(closed_terms["ranking_loss"].detach().cpu()),
                    "closed_local_wrong_horizon_loss": float(closed_terms["wrong_horizon_loss"].detach().cpu()),
                    "closed_local_difference_loss": float(closed_terms["difference_loss"].detach().cpu()),
                    "closed_local_target_difference_mse": float(closed_terms["target_difference_mse"].detach().cpu()),
                    "closed_local_prediction_difference_mse": float(closed_terms["prediction_difference_mse"].detach().cpu()),
                    "closed_local_robust_state_loss": float(closed_terms["robust_state_loss"].detach().cpu()),
                    "closed_local_robust_difference_loss": float(closed_terms["robust_difference_loss"].detach().cpu()),
                    "closed_local_student_t_nll": float(closed_terms["student_t_nll"].detach().cpu()),
                    "closed_local_base_state_loss": float(closed_terms["base_state_loss"].detach().cpu()),
                    "closed_local_base_robust_loss": float(closed_terms["base_robust_loss"].detach().cpu()),
                    "closed_local_base_skill": float(closed_terms["base_skill"].detach().cpu()),
                    "v4_slow_reconstruction_loss": float(v4_slow_terms["reconstruction_loss"].detach().cpu()),
                    "v4_slow_vamp_loss": float(v4_slow_terms["vamp_loss"].detach().cpu()),
                    "v4_slow_balance_loss": float(v4_slow_terms["balance_loss"].detach().cpu()),
                    "v4_slow_assignment_entropy": float(v4_slow_terms["assignment_entropy"].detach().cpu()),
                    "v4_slow_base_transition_loss": float(v4_slow_terms["base_transition_loss"].detach().cpu()),
                    "v4_slow_base_transition_skill": float(v4_slow_terms["base_transition_skill"].detach().cpu()),
                    "v4_slow_rollout_transition_loss": float(v4_slow_terms["rollout_transition_loss"].detach().cpu()),
                    "v4_slow_rollout_transition_skill": float(v4_slow_terms["rollout_transition_skill"].detach().cpu()),
                    "v4_slow_mean_lambda2": float(v4_slow_terms["mean_lambda2"].detach().cpu()),
                    "v4c_reconstruction_loss": float(v4_physics["reconstruction_loss"].detach().cpu()),
                    "v4c_vamp_loss": float(v4_physics["vamp_loss"].detach().cpu()),
                    "v4c_balance_loss": float(v4_physics["balance_loss"].detach().cpu()),
                    "v4c_assignment_entropy": float(v4_physics["assignment_entropy"].detach().cpu()),
                    "v4c_cross_covariance_loss": float(v4_physics["cross_covariance_loss"].detach().cpu()),
                    "v4c_increment_covariance_loss": float(v4_physics["increment_covariance_loss"].detach().cpu()),
                    "v4c_timescale_separation_loss": float(v4_physics["timescale_separation_loss"].detach().cpu()),
                    "v4c_trajectory_balance_loss": float(v4_physics["trajectory_balance_loss"].detach().cpu()),
                    "v4c_activity_loss": float(v4_physics["activity_loss"].detach().cpu()),
                    "v4c_latent_variance_loss": float(v4_physics["latent_variance_loss"].detach().cpu()),
                    "v4c_assignment_switching": float(v4_physics["assignment_switching"].detach().cpu()),
                    "v4c_physical_activity": float(v4_physics["physical_activity"].detach().cpu()),
                    "v4c_within_assignment_variance": float(v4_physics["within_assignment_variance"].detach().cpu()),
                    "v4c_between_assignment_variance": float(v4_physics["between_assignment_variance"].detach().cpu()),
                    "v4c_identity_leakage_ratio": float(v4_physics["identity_leakage_ratio"].detach().cpu()),
                    "v4c_drift_loss": float(v4_physics["drift_loss"].detach().cpu()),
                    "v4c_force_loss": float(v4_physics["force_loss"].detach().cpu()),
                    "v4c_force_cosine": float(v4_physics["force_cosine"].detach().cpu()),
                    "v4c_stationary_loss": float(v4_physics["stationary_loss"].detach().cpu()),
                    "v4c_base_transition_loss": float(v4_physics["base_transition_loss"].detach().cpu()),
                    "v4c_base_transition_skill": float(v4_physics["base_transition_skill"].detach().cpu()),
                    "v4c_rollout_transition_loss": float(v4_physics["rollout_transition_loss"].detach().cpu()),
                    "v4c_rollout_transition_skill": float(v4_physics["rollout_transition_skill"].detach().cpu()),
                    "v4c_transition_nll": float(v4_physics["transition_nll"].detach().cpu()),
                    "v4c_horizon_ranking_loss": float(v4_physics["horizon_ranking_loss"].detach().cpu()),
                    "v4c_correct_vs_best_fixed_skill": float(v4_physics["correct_vs_best_fixed_skill"].detach().cpu()),
                    "v4c_horizon_difference_loss": float(v4_physics["horizon_difference_loss"].detach().cpu()),
                    "v4c_target_horizon_difference": float(v4_physics["target_horizon_difference"].detach().cpu()),
                    "v4c_detailed_balance_residual": float(v4_physics["detailed_balance_residual"].detach().cpu()),
                    "v4c_mean_lambda2": float(v4_physics["mean_lambda2"].detach().cpu()),
                    "v4c_latent_autocorrelation": float(v4_physics["latent_autocorrelation"].detach().cpu()),
                    "v4c_residual_autocorrelation": float(v4_physics["residual_autocorrelation"].detach().cpu()),
                    "paired_by_horizon": horizon_log,
                    "horizon_frames": [int(x) for x in outputs["horizon_frames"].detach().cpu().tolist()],
                    "lr": optimizer.param_groups[-1]["lr"],
                    "ema": momentum,
                }
                if slow_terms is not None:
                    record.update(
                        {
                            "slow_delta_loss": float(slow_terms["delta_loss"].detach().cpu()),
                            "slow_state_loss": float(slow_terms["state_loss"].detach().cpu()),
                            "slow_contrastive_loss": float(slow_terms["contrastive_loss"].detach().cpu()),
                            "slow_variance_loss": float(slow_terms["variance_loss"].detach().cpu()),
                            "slow_covariance_loss": float(slow_terms["covariance_loss"].detach().cpu()),
                            "slow_skill": float(slow_terms["skill"].detach().cpu()),
                        }
                    )
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


def train(config: TrainConfig) -> Path:
    if config.data.training_mode == "paired_window":
        return train_paired(config)
    if config.init_checkpoint and config.resume_checkpoint:
        raise ValueError("use either init_checkpoint or resume_checkpoint, not both")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

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
            else:
                msm_state_loss = total_loss.new_tensor(0.0)

            if vamp2_weight > 0:
                vamp2_loss = model.compute_vamp2_loss(outputs, batch, lag=config.optimization.vamp2_lag)
                total_loss = total_loss + vamp2_weight * vamp2_loss
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
                    "lagged_state_loss_by_lag": {
                        str(lag): float(term["loss"].detach().cpu()) for lag, term in lagged_state_terms.items()
                    },
                    "lagged_state_pairs_by_lag": {
                        str(lag): int(term["pairs"]) for lag, term in lagged_state_terms.items()
                    },
                    "lagged_state_persistence_by_lag": {
                        str(lag): float(term["persistence_loss"].detach().cpu())
                        for lag, term in lagged_state_terms.items()
                    },
                    "lagged_state_intrinsic_persistence_by_lag": {
                        str(lag): float(term["intrinsic_persistence_loss"].detach().cpu())
                        for lag, term in lagged_state_terms.items()
                    },
                    "lagged_state_skill_by_lag": {
                        str(lag): float(
                            1.0
                            - term["loss"].detach().cpu()
                            / term["persistence_loss"].detach().cpu().clamp(min=1.0e-8)
                        )
                        for lag, term in lagged_state_terms.items()
                    },
                    "transition_source_alignment_loss": float(
                        torch.stack(
                            [term["source_alignment_loss"] for term in lagged_state_terms.values()]
                        ).mean().detach().cpu()
                    ) if lagged_state_terms else 0.0,
                    "semigroup_loss": float(semigroup_loss.detach().cpu()),
                    "semigroup_loss_by_composition": {
                        key: float(term["loss"].detach().cpu()) for key, term in semigroup_terms.items()
                    },
                    "semigroup_pairs_by_composition": {
                        key: int(term["pairs"]) for key, term in semigroup_terms.items()
                    },
                    "state_transition_jepa_loss": float(state_transition_loss.detach().cpu()),
                    "physchem_loss": float(physchem_loss.detach().cpu()),
                    "linear_dynamics_jepa_loss": float(linear_dynamics_loss.detach().cpu()),
                    "kinetic_vampnet_loss": float(kinetic_vamp_loss.detach().cpu()),
                    "variance_reg": float(reg.detach().cpu()),
                    "state_variance_reg": float(state_reg.detach().cpu()),
                    "state_feature_variance_reg": float(state_feature_reg.detach().cpu()),
                    "state_covariance_reg": float(state_covariance_reg.detach().cpu()),
                    "msm_state_loss": float(msm_state_loss.detach().cpu()),
                    "vamp2_loss": float(vamp2_loss.detach().cpu()),
                    "auxiliary_warmup": aux_mult,
                    "lr": optimizer.param_groups[-1]["lr"],
                    "ema": momentum,
                    "target_tokens": int(target_mask.sum().detach().cpu()),
                    "context_tokens": int(context_mask.sum().detach().cpu()),
                }
                with log_path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                print(json.dumps(record), flush=True)

            completed_step = global_step + 1
            if config.checkpoint_every_steps > 0 and completed_step % config.checkpoint_every_steps == 0:
                _save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    epoch,
                    completed_step,
                    config,
                    write_step_file=True,
                )
            global_step += 1

        if (epoch + 1) % config.save_every == 0 or epoch + 1 == config.optimization.epochs:
            _save_checkpoint(
                output_dir,
                model,
                optimizer,
                epoch + 1,
                global_step,
                config,
                write_step_file=False,
            )
    return output_dir / "checkpoint_latest.pt"
