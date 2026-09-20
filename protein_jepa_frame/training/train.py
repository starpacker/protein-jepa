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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _lspwm_sequence_vamp_loss(
    outputs: dict, lag: int = 1
) -> torch.Tensor:
    """VAMP-2 on actual causal frame sequences, not only endpoints."""
    sequence = outputs.get("lspwm_context_sequence")
    mask = outputs.get("lspwm_context_mask")
    if sequence is None or mask is None or lag <= 0 or lag >= sequence.shape[1]:
        reference = sequence if sequence is not None else outputs["delta_target"]
        return reference.new_zeros(())
    pair_mask = mask[:, :-lag] & mask[:, lag:]
    if int(pair_mask.sum()) < 2:
        return sequence.new_zeros(())
    # The clip is short relative to the 256D state. A low-dimensional fixed
    # slice keeps the covariance rank identifiable instead of asking VAMP to
    # invert a rank-deficient 256x256 minibatch covariance.
    sequence = sequence[..., : min(16, sequence.shape[-1])]
    source = sequence[:, :-lag][pair_mask]
    target = sequence[:, lag:][pair_mask]
    return vamp2_loss(source, target, ridge=1.0e-3)


def _balanced_calibrated_endpoint_mse(outputs: dict) -> torch.Tensor:
    """Match the endpoint evaluator's deterministic mean, equally by horizon."""
    prediction = outputs.get("lspwm_calibrated_transition_mean")
    if prediction is None:
        reference = outputs["lspwm_target_state"]
        return reference.new_zeros(())
    target = outputs["lspwm_target_state"].detach()
    per_row = (prediction - target).square().mean(dim=-1)
    horizons = outputs["horizon_frames"]
    horizon_losses = []
    for horizon in torch.unique(horizons, sorted=True):
        mask = horizons == horizon
        if mask.any():
            horizon_losses.append(per_row[mask].mean())
    if not horizon_losses:
        return per_row.new_zeros(())
    return torch.stack(horizon_losses).mean()


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


def _module_gradient_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = None
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().square().sum()
        total = squared if total is None else total + squared
    return float(total.sqrt().cpu()) if total is not None else 0.0


def _tmjepa_optimizer_groups(
    model: ProteinJEPA,
    base_lr: float,
    diffusion_lr_scale: float,
    observation_noise_lr_scale: float,
    memory_lr_scale: float = 1.0,
    projected_force_lr_scale: float = 1.0,
) -> list[dict]:
    """Split Q/R parameters only when independent rates are requested."""
    trainable = [
        (name, parameter)
        for name, parameter in model.tmjepa.named_parameters()
        if parameter.requires_grad
    ]
    if (
        diffusion_lr_scale == 1.0
        and observation_noise_lr_scale == 1.0
        and memory_lr_scale == 1.0
        and projected_force_lr_scale == 1.0
    ):
        return [
            {
                "params": [parameter for _, parameter in trainable],
                "lr": base_lr,
                "base_lr_scale": 1.0,
                "group_name": "tmjepa",
            }
        ]

    diffusion_prefixes = (
        "dynamics.kinetics.mobility_head.",
        "dynamics.kinetics.diffusion_basis",
        "dynamics.kinetics.activation_energy_raw",
        "dynamics.kinetics.diagonal_activation_deviation",
        "dynamics.kinetics.mode_activation_deviation",
    )
    memory_prefixes = (
        "dynamics.kinetics.memory_head.",
        "dynamics.kinetics.coupling_vectors",
        "dynamics.kinetics.memory_activation_energy_raw",
        "dynamics.kinetics.memory_activation_deviation",
        "dynamics.kinetics.memory_rotation_raw",
    )
    observation_prefix = "dynamics.kinetics.observation_noise_"
    projected_force_prefix = "projected_force_corrector."
    diffusion = []
    memory = []
    observation = []
    projected_force = []
    remainder = []
    for name, parameter in trainable:
        if name.startswith(observation_prefix):
            observation.append(parameter)
        elif name.startswith(projected_force_prefix):
            projected_force.append(parameter)
        elif name.startswith(diffusion_prefixes):
            diffusion.append(parameter)
        elif name.startswith(memory_prefixes):
            memory.append(parameter)
        else:
            remainder.append(parameter)

    groups = []
    for name, parameters, scale in (
        ("tmjepa_remainder", remainder, 1.0),
        ("tmjepa_diffusion", diffusion, diffusion_lr_scale),
        ("tmjepa_memory", memory, memory_lr_scale),
        ("tmjepa_observation_noise", observation, observation_noise_lr_scale),
        ("tmjepa_projected_force", projected_force, projected_force_lr_scale),
    ):
        if parameters:
            groups.append(
                {
                    "params": parameters,
                    "lr": base_lr * scale,
                    "base_lr_scale": scale,
                    "group_name": name,
                }
            )
    return groups


def _prefixes_overlap(left: str, right: str) -> bool:
    return left.startswith(right) or right.startswith(left)


def _validate_init_freeze_contract(config: TrainConfig) -> None:
    """Reject checkpoints that reinitialize parameters and then freeze them."""
    rollout_weights = (
        config.optimization.lspwm_rollout_energy_weight,
        config.optimization.lspwm_rollout_endpoint_weight,
        config.optimization.lspwm_rollout_gaussian_nll_weight,
        config.optimization.lspwm_rollout_distribution_weight,
        config.optimization.lspwm_rollout_koopman_weight,
        config.optimization.lspwm_rollout_force_weight,
        config.optimization.lspwm_rollout_trajectory_nll_weight,
    )
    if not config.model.lspwm_rollout_enabled and any(weight > 0.0 for weight in rollout_weights):
        raise ValueError(
            "LSP-WM rollout weights must be zero when lspwm_rollout_enabled=false"
        )
    if config.model.freeze_target_encoders:
        if not (config.model.freeze_context_encoder and config.model.freeze_temporal_encoder):
            raise ValueError(
                "freeze_target_encoders requires frozen context and temporal encoders"
            )
        if config.sync_target_after_init:
            raise ValueError(
                "freeze_target_encoders cannot be combined with sync_target_after_init"
            )
    if (
        config.optimization.lspwm_dynamic_weight > 0.0
        and not config.model.lspwm_dynamic_nll_enabled
    ):
        raise ValueError(
            "lspwm_dynamic_weight requires lspwm_dynamic_nll_enabled=true"
        )
    if (
        config.optimization.lspwm_calibrated_dynamic_weight > 0.0
        and not config.model.lspwm_physical_calibration_enabled
    ):
        raise ValueError(
            "lspwm_calibrated_dynamic_weight requires physical calibration"
        )
    if (
        config.optimization.lspwm_calibrated_endpoint_weight > 0.0
        and not config.model.lspwm_physical_calibration_enabled
    ):
        raise ValueError(
            "lspwm_calibrated_endpoint_weight requires physical calibration"
        )
    if (
        config.optimization.lspwm_calibrated_force_weight > 0.0
        and not config.model.lspwm_physical_calibration_enabled
    ):
        raise ValueError(
            "lspwm_calibrated_force_weight requires physical calibration"
        )
    if (
        config.optimization.lspwm_calibration_regularization_weight > 0.0
        and not config.model.lspwm_physical_calibration_enabled
    ):
        raise ValueError(
            "lspwm_calibration_regularization_weight requires physical calibration"
        )
    if config.model.lspwm_freeze_base_dynamics:
        if not config.model.lspwm_physical_calibration_enabled:
            raise ValueError(
                "lspwm_freeze_base_dynamics requires physical calibration"
            )
        if not config.init_checkpoint and not config.resume_checkpoint:
            raise ValueError(
                "freezing base dynamics requires an initialized checkpoint"
            )
    if config.model.lspwm_freeze_calibration_scales:
        if not config.model.lspwm_physical_calibration_enabled:
            raise ValueError(
                "lspwm_freeze_calibration_scales requires physical calibration"
            )
        if not config.model.lspwm_freeze_base_dynamics:
            raise ValueError(
                "lspwm_freeze_calibration_scales requires frozen base dynamics"
            )
    if config.model.lspwm_reset_conservative_correction_on_init:
        if not config.model.lspwm_conservative_correction_enabled:
            raise ValueError(
                "lspwm_reset_conservative_correction_on_init requires a correction head"
            )
        if not config.init_checkpoint:
            raise ValueError(
                "lspwm_reset_conservative_correction_on_init requires init_checkpoint"
            )
    if config.model.learned_state_training:
        if config.optimization.encoder_lr_scale <= 0.0:
            raise ValueError("learned-state training requires a positive encoder LR scale")
        if config.optimization.paired_delta_weight <= 0.0:
            raise ValueError("learned-state training requires an active JEPA predictor loss")
        if config.model.ls2_reversible_enabled:
            if not config.data.paired_all_horizons:
                raise ValueError("LS2 reversible training requires paired_all_horizons")
            if config.optimization.ls2_state_weight <= 0.0:
                raise ValueError("LS2 reversible training requires an active state loss")
            if config.optimization.ls2_assignment_weight <= 0.0:
                raise ValueError("LS2 reversible training requires an active assignment loss")
        elif config.optimization.slow_delta_weight <= 0.0:
            raise ValueError("learned-state training requires an active slow predictor loss")
    learned_v3 = (
        config.model.tmjepa_enabled
        and config.model.tmjepa_version == 3
        and config.model.tmjepa_learned_cv_dim > 0
    )
    representation_frozen = (
        config.model.freeze_context_encoder
        and config.model.freeze_temporal_encoder
        and config.model.tmjepa_freeze_learned_projection
    )
    physics_objective_active = any(
        weight > 0.0
        for weight in (
            config.optimization.tmjepa_energy_score_weight,
            config.optimization.tmjepa_forecast_relative_mse_weight,
            config.optimization.tmjepa_transition_nll_weight,
            config.optimization.tmjepa_path_nll_weight,
            config.optimization.tmjepa_equilibrium_score_weight,
            config.optimization.tmjepa_innovation_mean_weight,
            config.optimization.tmjepa_innovation_covariance_weight,
            config.optimization.tmjepa_innovation_autocorrelation_weight,
            config.optimization.tmjepa_innovation_covariance_spectral_weight,
            config.optimization.tmjepa_innovation_worst_group_weight,
            config.optimization.tmjepa_diffusion_condition_weight,
            config.optimization.tmjepa_memory_intervention_weight,
            config.optimization.tmjepa_memory_relative_intervention_weight,
            config.optimization.tmjepa_memory_ratio_band_weight,
            config.optimization.tmjepa_memory_log_ratio_weight,
            config.optimization.tmjepa_memory_long_horizon_intervention_weight,
            config.optimization.tmjepa_memory_long_horizon_relative_intervention_weight,
            config.optimization.tmjepa_memory_residual_weight,
            config.optimization.tmjepa_memory_long_horizon_residual_weight,
            config.optimization.tmjepa_long_horizon_energy_score_weight,
            config.optimization.tmjepa_long_horizon_forecast_relative_mse_weight,
            config.optimization.tmjepa_long_horizon_transition_nll_weight,
            config.optimization.tmjepa_long_horizon_path_nll_weight,
            config.optimization.tmjepa_projected_force_regularization_weight,
            config.optimization.tmjepa_auxiliary_kl_weight,
            config.optimization.tmjepa_auxiliary_transition_kl_weight,
        )
    )
    if config.model.tmjepa_representation_only and physics_objective_active:
        raise ValueError("TM-JEPA representation-only mode cannot optimize physical dynamics")
    if learned_v3 and representation_frozen and not (
        config.init_checkpoint or config.resume_checkpoint
    ):
        raise ValueError(
            "frozen TM-JEPA learned representation requires an init or resume checkpoint"
        )
    if not config.init_checkpoint or not config.init_exclude_prefixes:
        return
    frozen_prefixes: list[str] = []
    if config.model.freeze_context_encoder:
        frozen_prefixes.append("context_encoder.")
    if config.model.freeze_temporal_encoder:
        frozen_prefixes.append("temporal_encoder.")
    if config.model.tmjepa_freeze_learned_projection:
        frozen_prefixes.append("tmjepa.learned_projection.")
    if config.model.tmjepa_freeze_potential:
        frozen_prefixes.extend(
            [
                "tmjepa.condition_encoder.",
                "tmjepa.dynamics.potential.",
            ]
        )
    kinetic_objective_active = any(
        weight > 0.0
        for weight in (
            config.optimization.tmjepa_energy_score_weight,
            config.optimization.tmjepa_forecast_relative_mse_weight,
            config.optimization.tmjepa_transition_nll_weight,
            config.optimization.tmjepa_path_nll_weight,
            config.optimization.tmjepa_innovation_mean_weight,
            config.optimization.tmjepa_innovation_covariance_weight,
            config.optimization.tmjepa_innovation_autocorrelation_weight,
            config.optimization.tmjepa_innovation_covariance_spectral_weight,
            config.optimization.tmjepa_innovation_worst_group_weight,
            config.optimization.tmjepa_diffusion_condition_weight,
            config.optimization.tmjepa_memory_intervention_weight,
            config.optimization.tmjepa_memory_relative_intervention_weight,
            config.optimization.tmjepa_memory_ratio_band_weight,
            config.optimization.tmjepa_memory_log_ratio_weight,
            config.optimization.tmjepa_memory_long_horizon_intervention_weight,
            config.optimization.tmjepa_memory_long_horizon_relative_intervention_weight,
            config.optimization.tmjepa_memory_residual_weight,
            config.optimization.tmjepa_memory_long_horizon_residual_weight,
            config.optimization.tmjepa_long_horizon_energy_score_weight,
            config.optimization.tmjepa_long_horizon_forecast_relative_mse_weight,
            config.optimization.tmjepa_long_horizon_transition_nll_weight,
            config.optimization.tmjepa_long_horizon_path_nll_weight,
            config.optimization.tmjepa_projected_force_regularization_weight,
            config.optimization.tmjepa_auxiliary_kl_weight,
            config.optimization.tmjepa_auxiliary_transition_kl_weight,
        )
    )
    if config.model.tmjepa_freeze_kinetics and kinetic_objective_active:
        frozen_prefixes.append("tmjepa.dynamics.kinetics.")
    if config.model.tmjepa_freeze_diffusion and kinetic_objective_active:
        frozen_prefixes.extend(
            [
                "tmjepa.dynamics.kinetics.mobility_head.",
                "tmjepa.dynamics.kinetics.diffusion_basis",
                "tmjepa.dynamics.kinetics.activation_energy_raw",
                "tmjepa.dynamics.kinetics.diagonal_activation_deviation",
                "tmjepa.dynamics.kinetics.mode_activation_deviation",
            ]
        )
    if config.model.tmjepa_freeze_memory and kinetic_objective_active:
        frozen_prefixes.extend(
            [
                "tmjepa.dynamics.kinetics.memory_head.",
                "tmjepa.dynamics.kinetics.coupling_vectors",
                "tmjepa.dynamics.kinetics.memory_activation_energy_raw",
                "tmjepa.dynamics.kinetics.memory_activation_deviation",
                "tmjepa.dynamics.kinetics.memory_rotation_raw",
            ]
        )
    if config.model.tmjepa_freeze_observation_noise and kinetic_objective_active:
        frozen_prefixes.extend(
            [
                "tmjepa.dynamics.kinetics.observation_noise_",
            ]
        )
    conflicts = sorted(
        {
            excluded
            for excluded in config.init_exclude_prefixes
            for frozen in frozen_prefixes
            if _prefixes_overlap(excluded, frozen)
        }
    )
    if conflicts:
        raise ValueError(
            "init checkpoint exclusions overlap frozen TM-JEPA parameters: "
            f"{conflicts}. Load pretrained frozen parameters or do not freeze them."
        )


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
    state = upgrade_tmjepa_v3_state_dict(model, state)
    state = upgrade_lspwm_state_dict(model, state)
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


def _reset_lspwm_conservative_correction(model: ProteinJEPA) -> None:
    if model.lspwm is None or not model.lspwm.physical_calibration_enabled:
        raise ValueError("resetting the LSP-WM correction requires physical calibration")
    correction = (
        model.lspwm.world_model.physical_calibration.conservative_correction
    )
    if correction is None:
        raise ValueError("resetting the LSP-WM correction requires a correction head")
    with torch.no_grad():
        correction.raw_state_basis.normal_(mean=0.0, std=0.02)
        correction.bias.zero_()
        correction.condition_basis.zero_()
    print(
        json.dumps(
            {
                "event": "reset_lspwm_conservative_correction",
                "raw_state_basis_std": 0.02,
                "bias": "zero",
                "condition_basis": "zero",
            }
        ),
        flush=True,
    )


def train_paired(config: TrainConfig) -> Path:
    """Train the V2.2 paired endpoint and slow-state objectives."""
    _set_reproducibility(config.seed, config.deterministic)
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
        if config.model.lspwm_reset_conservative_correction_on_init:
            _reset_lspwm_conservative_correction(model)
        if config.sync_target_after_init:
            model.sync_target_encoders()
    fixed_cv_tmjepa = model.tmjepa is not None and model.tmjepa.learned_cv_dim == 0
    param_groups = [] if fixed_cv_tmjepa else [
        {"params": model.context_encoder.parameters(), "lr": config.optimization.lr * config.optimization.encoder_lr_scale, "base_lr_scale": config.optimization.encoder_lr_scale},
        {"params": model.temporal_encoder.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0},
    ]
    if not fixed_cv_tmjepa and (
        config.optimization.paired_delta_weight > 0.0
        or config.optimization.paired_state_weight > 0.0
        or config.optimization.paired_relative_delta_weight > 0.0
    ) and model.gaussian_transition is None:
        param_groups.append(
            {"params": model.paired_delta_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
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
    if model.observable_head is not None and config.optimization.observable_weight > 0.0:
        param_groups.append(
            {"params": model.observable_head.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
    if model.slow_state_projector is not None:
        param_groups.append(
            {"params": model.slow_state_projector.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
        )
        if model.slow_delta_predictor is not None:
            param_groups.append(
                {"params": model.slow_delta_predictor.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
            )
    if model.ls2_reversible_dynamics is not None:
        param_groups.append(
            {"params": model.ls2_reversible_dynamics.parameters(), "lr": config.optimization.lr, "base_lr_scale": 1.0}
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
    if model.v5_contextual_dynamics is not None:
        param_groups.append(
            {
                "params": model.v5_contextual_dynamics.parameters(),
                "lr": config.optimization.lr,
                "base_lr_scale": 1.0,
            }
        )
    if model.v51_contextual_dynamics is not None:
        param_groups.append(
            {
                "params": (
                    parameter
                    for parameter in model.v51_contextual_dynamics.parameters()
                    if parameter.requires_grad
                ),
                "lr": config.optimization.lr,
                "base_lr_scale": 1.0,
            }
        )
    if model.tmjepa is not None:
        tmjepa_groups = _tmjepa_optimizer_groups(
            model,
            config.optimization.lr,
            config.optimization.tmjepa_diffusion_lr_scale,
            config.optimization.tmjepa_observation_noise_lr_scale,
            memory_lr_scale=config.model.tmjepa_kinetics_lr_scale,
            projected_force_lr_scale=(
                config.optimization.tmjepa_projected_force_lr_scale
            ),
        )
        if fixed_cv_tmjepa:
            param_groups = tmjepa_groups
        else:
            param_groups.extend(tmjepa_groups)
    if model.lspwm is not None:
        lspwm_parameters = (
            model.lspwm.calibration_parameters()
            if config.model.lspwm_freeze_base_dynamics
            else model.lspwm.parameters()
        )
        param_groups.append(
            {
                "params": lspwm_parameters,
                "lr": config.optimization.lr,
                "base_lr_scale": 1.0,
                "group_name": (
                    "lspwm_physical_calibration"
                    if config.model.lspwm_freeze_base_dynamics
                    else "lspwm"
                ),
            }
        )
    param_groups = [
        {**group, "params": [parameter for parameter in group["params"] if parameter.requires_grad]}
        for group in param_groups
    ]
    param_groups = [group for group in param_groups if group["params"]]
    if not param_groups:
        raise ValueError("training configuration has no trainable parameters")
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
            if model.tmjepa is not None:
                warmup = config.model.tmjepa_memory_coupling_warmup_steps
                if warmup > 0 and model.tmjepa.dynamics.kinetics.memory_enabled:
                    ramp = min(1.0, float(global_step + 1) / float(warmup))
                    model.tmjepa.dynamics.kinetics._memory_coupling_ramp.fill_(ramp)
            model._curriculum_step = global_step
            optimizer.zero_grad(set_to_none=True)
            outputs = (
                model.forward_tmjepa(batch)
                if model.tmjepa is not None and model.tmjepa.learned_cv_dim == 0
                else model.forward_paired(batch)
            )
            terms = paired_forecast_terms(outputs)
            gaussian_loss = gaussian_transition_nll(outputs)
            observable_loss = observable_prediction_loss(outputs)
            offline_terms = offline_forecast_terms(outputs)
            slow_terms = (
                slow_forecast_terms(
                    outputs,
                    vamp_lags=config.model.slow_vamp_lags,
                    vamp_burn_in_frames=config.model.slow_vamp_burn_in_frames,
                )
                if "slow_delta_prediction" in outputs
                else None
            )
            ls2_terms = reversible_kinetic_terms(
                outputs,
                vamp_lags=config.model.slow_vamp_lags,
            )
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
            v5_terms = v5_contextual_dynamics_terms(
                outputs,
                horizon_ranking_margin=config.optimization.v5_horizon_ranking_margin,
            )
            v51_terms = v51_sparse_motion_terms(
                outputs,
                ranking_margin=config.optimization.v51_ranking_margin,
            )
            tm_terms = tmjepa_terms(
                outputs,
                memory_intervention_margin=(
                    config.optimization.tmjepa_memory_intervention_margin
                ),
                memory_relative_intervention_margin=(
                    config.optimization.tmjepa_memory_relative_intervention_margin
                ),
                memory_ratio_min=config.optimization.tmjepa_memory_ratio_min,
                memory_ratio_max=config.optimization.tmjepa_memory_ratio_max,
                memory_ratio_target=config.optimization.tmjepa_memory_ratio_target,
                long_horizon_min_frames=(
                    config.optimization.tmjepa_long_horizon_min_frames
                ),
                long_horizon_power=config.optimization.tmjepa_long_horizon_power,
                vamp_lags=config.optimization.tmjepa_vamp_lags,
                innovation_lags=config.optimization.tmjepa_innovation_lags,
                diffusion_condition_target=(
                    config.optimization.tmjepa_diffusion_condition_target
                ),
            )
            history_ramp = min(
                1.0,
                float(global_step + 1)
                / max(1, config.optimization.v53_history_warmup_steps),
            ) if config.optimization.v53_history_warmup_steps > 0 else 1.0
            evolution_ramp = min(
                1.0,
                float(global_step + 1)
                / max(1, config.optimization.v53_evolution_warmup_steps),
            ) if config.optimization.v53_evolution_warmup_steps > 0 else 1.0
            endpoint_reg = paired_endpoint_variance_regularizer(outputs)
            alignment_loss = paired_causal_alignment_loss(outputs)
            lspwm_dynamic_loss = outputs.get(
                "lspwm_dynamic_nll", terms["delta_loss"].new_zeros(())
            )
            lspwm_calibrated_dynamic_loss = outputs.get(
                "lspwm_calibrated_dynamic_nll", terms["delta_loss"].new_zeros(())
            )
            lspwm_calibrated_endpoint_loss = _balanced_calibrated_endpoint_mse(
                outputs
            )
            lspwm_calibrated_force_loss = outputs.get(
                "lspwm_calibrated_force_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_calibration_regularization = outputs.get(
                "lspwm_calibration_regularization", terms["delta_loss"].new_zeros(())
            )
            lspwm_force_loss = outputs.get(
                "lspwm_force_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_koopman_loss = outputs.get(
                "lspwm_koopman_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_msm_loss = outputs.get(
                "lspwm_msm_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_sequence_vamp_loss = _lspwm_sequence_vamp_loss(
                outputs, lag=config.optimization.lspwm_sequence_vamp_lag
            )
            lspwm_geometric_loss = outputs.get(
                "lspwm_geometric_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_auxiliary_loss = outputs.get(
                "lspwm_auxiliary_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_observation_loss = outputs.get(
                "lspwm_observation_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_energy_loss = outputs.get(
                "lspwm_rollout_energy_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_endpoint_loss = outputs.get(
                "lspwm_rollout_endpoint_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_gaussian_nll = outputs.get(
                "lspwm_rollout_gaussian_nll", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_distribution_loss = outputs.get(
                "lspwm_rollout_distribution_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_koopman_loss = outputs.get(
                "lspwm_rollout_koopman_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_force_loss = outputs.get(
                "lspwm_rollout_force_loss", terms["delta_loss"].new_zeros(())
            )
            lspwm_rollout_trajectory_nll = outputs.get(
                "lspwm_rollout_trajectory_nll", terms["delta_loss"].new_zeros(())
            )
            lspwm_endpoint_energy_skill = outputs.get(
                "lspwm_endpoint_energy_skill", terms["delta_loss"].new_zeros(())
            )
            lspwm_deterministic_skill = outputs.get(
                "lspwm_deterministic_skill", terms["delta_loss"].new_zeros(())
            )
            lspwm_active_phase = outputs.get("lspwm_active_phase", "unknown")
            # v21 THM-JEPA reversible diagnostics
            lspwm_dissipative_force_norm = outputs.get(
                "lspwm_dissipative_force_norm", None
            )
            lspwm_reversible_force_norm = outputs.get(
                "lspwm_reversible_force_norm", None
            )
            lspwm_bath_drift_norm = outputs.get("lspwm_bath_drift_norm", None)
            lspwm_total_drift_norm = outputs.get("lspwm_total_drift_norm", None)
            lspwm_drift_incr_cosine = outputs.get("lspwm_drift_incr_cosine", None)
            lspwm_reversible_drift_fraction = outputs.get(
                "lspwm_reversible_drift_fraction", None
            )
            # Phase-gated weights: five canonical curriculum phases (Ch.7.2)
            lspwm_phase_koopman_weight = 1.0
            lspwm_phase_force_weight = 0.0
            lspwm_phase_dynamic_weight = 0.0
            lspwm_phase_rollout_weight = 0.0
            lspwm_phase_observation_weight = 0.0
            phase = lspwm_active_phase
            if phase == "representation":
                lspwm_phase_koopman_weight = 1.0
            elif phase == "thermodynamic_identification":
                lspwm_phase_force_weight = 1.0
                lspwm_phase_dynamic_weight = 1.0
            elif phase == "dynamics":
                lspwm_phase_force_weight = 1.0
                lspwm_phase_dynamic_weight = 1.0
            elif phase == "rollout":
                lspwm_phase_force_weight = 1.0
                lspwm_phase_dynamic_weight = 1.0
                lspwm_phase_rollout_weight = 1.0
            total_loss = (
                config.optimization.paired_delta_weight * terms["delta_loss"]
                + config.optimization.paired_state_weight * terms["state_loss"]
                + config.optimization.paired_relative_delta_weight
                * terms["relative_delta_loss"]
                + config.optimization.paired_endpoint_variance_weight * endpoint_reg
                + config.optimization.lspwm_dynamic_weight * lspwm_dynamic_loss * lspwm_phase_dynamic_weight
                + config.optimization.lspwm_calibrated_dynamic_weight
                * lspwm_calibrated_dynamic_loss
                * lspwm_phase_dynamic_weight
                + config.optimization.lspwm_calibrated_endpoint_weight
                * lspwm_calibrated_endpoint_loss
                * lspwm_phase_dynamic_weight
                + config.optimization.lspwm_calibrated_force_weight
                * lspwm_calibrated_force_loss
                * lspwm_phase_force_weight
                + config.optimization.lspwm_calibration_regularization_weight
                * lspwm_calibration_regularization
                + config.optimization.lspwm_force_weight * lspwm_force_loss * lspwm_phase_force_weight
                + config.optimization.lspwm_koopman_weight * lspwm_koopman_loss * lspwm_phase_koopman_weight
                + config.optimization.lspwm_msm_weight * lspwm_msm_loss * lspwm_phase_koopman_weight
                + config.optimization.lspwm_sequence_vamp_weight
                * lspwm_sequence_vamp_loss
                * lspwm_phase_koopman_weight
                + config.optimization.lspwm_geometric_weight * lspwm_geometric_loss * lspwm_phase_koopman_weight
                + config.optimization.lspwm_auxiliary_weight * lspwm_auxiliary_loss * lspwm_phase_koopman_weight
                + config.optimization.lspwm_observation_weight * lspwm_observation_loss * lspwm_phase_observation_weight
                + config.optimization.lspwm_rollout_energy_weight * lspwm_rollout_energy_loss * lspwm_phase_rollout_weight
                + config.optimization.lspwm_rollout_endpoint_weight * lspwm_rollout_endpoint_loss * lspwm_phase_rollout_weight
                + config.optimization.lspwm_rollout_gaussian_nll_weight * lspwm_rollout_gaussian_nll * lspwm_phase_rollout_weight
                + config.optimization.lspwm_rollout_distribution_weight * lspwm_rollout_distribution_loss * lspwm_phase_rollout_weight
                + config.optimization.lspwm_rollout_koopman_weight * lspwm_rollout_koopman_loss * lspwm_phase_rollout_weight
                + config.optimization.lspwm_rollout_force_weight * lspwm_rollout_force_loss * lspwm_phase_rollout_weight
                + config.optimization.lspwm_rollout_trajectory_nll_weight * lspwm_rollout_trajectory_nll * lspwm_phase_rollout_weight
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
                + config.optimization.v5_transition_weight * v5_terms["transition_loss"]
                + config.optimization.v5_state_reconstruction_weight
                * v5_terms["state_reconstruction_loss"]
                + config.optimization.v5_feature_forecast_weight
                * v5_terms["feature_forecast_loss"]
                + config.optimization.v5_observable_weight * v5_terms["observable_loss"]
                + config.optimization.v5_transition_nll_weight * v5_terms["transition_nll"]
                + config.optimization.v5_horizon_ranking_weight
                * v5_terms["horizon_ranking_loss"]
                + config.optimization.v5_horizon_difference_weight
                * v5_terms["horizon_difference_loss"]
                + config.optimization.v5_observable_horizon_difference_weight
                * v5_terms["observable_horizon_difference_loss"]
                + config.optimization.v5_slot_diversity_weight
                * v5_terms["slot_diversity_loss"]
                + config.optimization.v5_gate_complexity_weight
                * v5_terms["gate_complexity_loss"]
                + config.optimization.v51_delta_weight * v51_terms["delta_loss"]
                + config.optimization.v51_normalized_delta_weight
                * v51_terms["normalized_delta_loss"]
                + config.optimization.v51_horizon_calibration_weight
                * v51_terms["horizon_calibration_loss"]
                + config.optimization.v51_online_delta_weight
                * v51_terms["online_delta_loss"]
                + config.optimization.v51_online_normalized_delta_weight
                * v51_terms["online_normalized_delta_loss"]
                + config.optimization.v51_displacement_magnitude_weight
                * v51_terms["displacement_magnitude_loss"]
                + config.optimization.v51_nll_weight * v51_terms["nll_loss"]
                + config.optimization.v51_current_alignment_weight
                * v51_terms["current_alignment_loss"]
                + config.optimization.v51_observable_weight
                * v51_terms["observable_loss"]
                + config.optimization.v51_oracle_observable_weight
                * v51_terms["oracle_observable_loss"]
                + config.optimization.v51_online_oracle_observable_weight
                * v51_terms["online_oracle_observable_loss"]
                + config.optimization.v51_horizon_ranking_weight
                * v51_terms["horizon_ranking_loss"]
                + config.optimization.v51_history_ranking_weight
                * v51_terms["history_ranking_loss"]
                + config.optimization.v51_history_truncation_weight
                * history_ramp * v51_terms["history_truncation_loss"]
                + config.optimization.v51_causal_prediction_weight
                * history_ramp * v51_terms["causal_next_loss"]
                + config.optimization.v51_horizon_consistency_weight
                * evolution_ramp * v51_terms["horizon_consistency_loss"]
                + config.optimization.v51_physical_horizon_consistency_weight
                * evolution_ramp * v51_terms["physical_horizon_consistency_loss"]
                + config.optimization.v51_semigroup_weight
                * evolution_ramp * v51_terms["semigroup_loss"]
                + config.optimization.v51_variance_weight
                * v51_terms["variance_loss"]
                + config.optimization.v51_covariance_weight
                * v51_terms["covariance_loss"]
                + config.optimization.v51_load_balance_weight
                * v51_terms["load_balance_loss"]
                + config.optimization.v51_active_load_balance_weight
                * v51_terms["active_load_balance_loss"]
                + config.optimization.v51_mode_diversity_weight
                * v51_terms["mode_diversity_loss"]
                + config.optimization.v51_mode_orthogonality_weight
                * v51_terms["mode_orthogonality_loss"]
                + config.optimization.tmjepa_energy_score_weight
                * tm_terms["energy_score"]
                + config.optimization.tmjepa_forecast_relative_mse_weight
                * tm_terms["forecast_relative_mse_loss"]
                + config.optimization.tmjepa_transition_nll_weight
                * tm_terms["transition_nll"]
                + config.optimization.tmjepa_path_nll_weight
                * tm_terms["path_nll"]
                + config.optimization.tmjepa_equilibrium_score_weight
                * tm_terms["equilibrium_score_loss"]
                + config.optimization.tmjepa_innovation_mean_weight
                * tm_terms["innovation_mean_loss"]
                + config.optimization.tmjepa_innovation_covariance_weight
                * tm_terms["innovation_covariance_loss"]
                + config.optimization.tmjepa_innovation_autocorrelation_weight
                * tm_terms["innovation_autocorrelation_loss"]
                + config.optimization.tmjepa_innovation_covariance_spectral_weight
                * tm_terms["innovation_covariance_spectral_loss"]
                + config.optimization.tmjepa_innovation_worst_group_weight
                * (
                    tm_terms["innovation_covariance_worst_loss"]
                    + tm_terms["innovation_autocorrelation_worst_loss"]
                )
                + config.optimization.tmjepa_diffusion_condition_weight
                * tm_terms["diffusion_condition_loss"]
                + config.optimization.tmjepa_memory_intervention_weight
                * tm_terms["memory_intervention_loss"]
                + config.optimization.tmjepa_memory_relative_intervention_weight
                * tm_terms["memory_relative_intervention_loss"]
                + config.optimization.tmjepa_memory_ratio_band_weight
                * tm_terms["memory_ratio_band_loss"]
                + config.optimization.tmjepa_memory_log_ratio_weight
                * tm_terms["memory_log_ratio_loss"]
                + config.optimization.tmjepa_memory_long_horizon_intervention_weight
                * tm_terms["memory_long_horizon_intervention_loss"]
                + config.optimization.tmjepa_memory_long_horizon_relative_intervention_weight
                * tm_terms["memory_long_horizon_relative_intervention_loss"]
                + config.optimization.tmjepa_memory_residual_weight
                * tm_terms["memory_residual_loss"]
                + config.optimization.tmjepa_memory_long_horizon_residual_weight
                * tm_terms["memory_long_horizon_residual_loss"]
                + config.optimization.tmjepa_long_horizon_energy_score_weight
                * tm_terms["long_horizon_energy_score"]
                + config.optimization.tmjepa_long_horizon_forecast_relative_mse_weight
                * tm_terms["long_horizon_forecast_relative_mse_loss"]
                + config.optimization.tmjepa_long_horizon_transition_nll_weight
                * tm_terms["long_horizon_transition_nll"]
                + config.optimization.tmjepa_long_horizon_path_nll_weight
                * tm_terms["long_horizon_path_nll"]
                + config.optimization.tmjepa_projected_force_regularization_weight
                * tm_terms["projected_force_regularization_loss"]
                + config.optimization.tmjepa_auxiliary_kl_weight
                * tm_terms["auxiliary_kl_loss"]
                + config.optimization.tmjepa_auxiliary_transition_kl_weight
                * tm_terms["auxiliary_transition_kl_loss"]
                + config.optimization.tmjepa_learned_cv_variance_weight
                * tm_terms["learned_cv_variance_loss"]
                + config.optimization.tmjepa_learned_cv_covariance_weight
                * tm_terms["learned_cv_covariance_loss"]
                + config.optimization.tmjepa_vamp_weight * tm_terms["vamp_loss"]
                + config.optimization.tmjepa_coverage_calibration_weight
                * tm_terms["coverage_calibration_loss"]
            )
            if slow_terms is not None:
                total_loss = total_loss + (
                    config.optimization.slow_delta_weight * slow_terms["delta_loss"]
                    + config.optimization.slow_state_weight * slow_terms["state_loss"]
                    + config.optimization.slow_relative_delta_weight
                    * slow_terms["relative_delta_loss"]
                    + config.optimization.slow_alignment_weight
                    * slow_terms["alignment_loss"]
                    + config.optimization.slow_vamp_weight * slow_terms["vamp_loss"]
                    + config.optimization.slow_contrastive_weight * slow_terms["contrastive_loss"]
                    + config.optimization.slow_variance_weight * slow_terms["variance_loss"]
                    + config.optimization.slow_covariance_weight * slow_terms["covariance_loss"]
                )
            if model.ls2_reversible_dynamics is not None:
                total_loss = total_loss + (
                    config.optimization.ls2_state_weight * ls2_terms["state_loss"]
                    + config.optimization.ls2_assignment_weight
                    * ls2_terms["assignment_loss"]
                    + config.optimization.ls2_nll_weight * ls2_terms["nll"]
                    + config.optimization.ls2_balance_weight * ls2_terms["balance_loss"]
                    + config.optimization.ls2_entropy_weight * ls2_terms["entropy_loss"]
                    + config.optimization.ls2_vamp_weight * ls2_terms["vamp_loss"]
                )
            total_loss.backward()
            if config.model.learned_state_training:
                full_state_predictor = model.ls2_reversible_dynamics or (
                    model.gaussian_transition
                    if model.gaussian_transition is not None
                    else model.paired_delta_predictor
                )
                slow_state_predictor = (
                    model.ls2_reversible_dynamics
                    if model.ls2_reversible_dynamics is not None
                    else model.slow_delta_predictor
                )
                required_gradient_norms = {
                    "context encoder": _module_gradient_norm(model.context_encoder),
                    "temporal encoder": _module_gradient_norm(model.temporal_encoder),
                    "full-state predictor": _module_gradient_norm(full_state_predictor),
                    "slow-state projector": _module_gradient_norm(model.slow_state_projector),
                    "slow-state predictor": _module_gradient_norm(slow_state_predictor),
                }
                invalid = {
                    name: value
                    for name, value in required_gradient_norms.items()
                    if not math.isfinite(value) or value <= 0.0
                }
                if invalid:
                    raise RuntimeError(
                        "learned-state V-JEPA joint-gradient contract failed: "
                        + ", ".join(f"{name}={value}" for name, value in invalid.items())
                    )
            gradient_diagnostics = {}
            if global_step % config.log_every == 0:
                gradient_diagnostics = {
                    "grad_norm_context_encoder": _module_gradient_norm(model.context_encoder),
                    "grad_norm_temporal_encoder": _module_gradient_norm(model.temporal_encoder),
                    "grad_norm_paired_predictor": _module_gradient_norm(model.paired_delta_predictor),
                    "grad_norm_gaussian_predictor": _module_gradient_norm(model.gaussian_transition),
                    "grad_norm_slow_state_projector": _module_gradient_norm(model.slow_state_projector),
                    "grad_norm_slow_predictor": _module_gradient_norm(model.slow_delta_predictor),
                    "grad_norm_ls2_reversible": _module_gradient_norm(
                        model.ls2_reversible_dynamics
                    ),
                    "grad_norm_tmjepa_learned_projection": _module_gradient_norm(
                        getattr(model.tmjepa, "learned_projection", None)
                        if model.tmjepa is not None
                        else None
                    ),
                    "grad_norm_tmjepa_potential": _module_gradient_norm(
                        getattr(model.tmjepa.dynamics, "potential", None)
                        if model.tmjepa is not None
                        else None
                    ),
                    "grad_norm_tmjepa_kinetics": _module_gradient_norm(
                        getattr(model.tmjepa.dynamics, "kinetics", None)
                        if model.tmjepa is not None
                        else None
                    ),
                    "grad_norm_lspwm_calibration": _module_gradient_norm(
                        getattr(
                            getattr(model.lspwm, "world_model", None),
                            "physical_calibration",
                            None,
                        )
                        if model.lspwm is not None
                        else None
                    ),
                }
            if config.optimization.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.grad_clip)
            optimizer.step()
            momentum = _momentum(global_step, total_steps, config.optimization.ema_start, config.optimization.ema_end)
            model.update_target_encoder(momentum)
            if (
                model.v51_contextual_dynamics is not None
                and global_step < config.optimization.v51_target_hard_sync_steps
            ):
                model.v51_contextual_dynamics.sync_target_encoder()
            if global_step % config.log_every == 0:
                horizon_log = {}
                for horizon in sorted(set(int(x) for x in outputs["horizon_frames"].detach().cpu().tolist())):
                    mask = outputs["horizon_frames"] == horizon
                    if not mask.any():
                        continue
                    subset = {
                        key: value[mask]
                        for key, value in outputs.items()
                        if torch.is_tensor(value)
                        and value.ndim > 0
                        and value.shape[0] == mask.shape[0]
                    }
                    subterms = paired_forecast_terms(subset)
                    horizon_log[str(horizon)] = {
                        "pairs": int(mask.sum().detach().cpu()),
                        "delta_loss": float(subterms["delta_loss"].detach().cpu()),
                        "state_loss": float(subterms["state_loss"].detach().cpu()),
                        "relative_delta_loss": float(
                            subterms["relative_delta_loss"].detach().cpu()
                        ),
                        "persistence_delta_loss": float(subterms["persistence_delta_loss"].detach().cpu()),
                        "skill": float(subterms["skill"].detach().cpu()),
                    }
                record = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(total_loss.detach().cpu()),
                    "paired_delta_loss": float(terms["delta_loss"].detach().cpu()),
                    "paired_state_loss": float(terms["state_loss"].detach().cpu()),
                    "paired_relative_delta_loss": float(
                        terms["relative_delta_loss"].detach().cpu()
                    ),
                    "paired_persistence_delta_loss": float(terms["persistence_delta_loss"].detach().cpu()),
                    "paired_skill": float(terms["skill"].detach().cpu()),
                    "paired_endpoint_variance": float(endpoint_reg.detach().cpu()),
                    "lspwm_dynamic_nll": float(lspwm_dynamic_loss.detach().cpu()),
                    "lspwm_calibrated_dynamic_nll": float(
                        lspwm_calibrated_dynamic_loss.detach().cpu()
                    ),
                    "lspwm_calibrated_endpoint_mse": float(
                        lspwm_calibrated_endpoint_loss.detach().cpu()
                    ),
                    "lspwm_calibrated_force_loss": float(
                        lspwm_calibrated_force_loss.detach().cpu()
                    ),
                    "lspwm_calibration_regularization": float(
                        lspwm_calibration_regularization.detach().cpu()
                    ),
                    "lspwm_force_loss": float(lspwm_force_loss.detach().cpu()),
                    "lspwm_koopman_loss": float(lspwm_koopman_loss.detach().cpu()),
                    "lspwm_msm_loss": float(lspwm_msm_loss.detach().cpu()),
                    "lspwm_sequence_vamp_loss": float(
                        lspwm_sequence_vamp_loss.detach().cpu()
                    ),
                    "lspwm_geometric_loss": float(lspwm_geometric_loss.detach().cpu()),
                    "lspwm_auxiliary_loss": float(lspwm_auxiliary_loss.detach().cpu()),
                    "lspwm_observation_loss": float(lspwm_observation_loss.detach().cpu()),
                    "lspwm_rollout_energy_loss": float(lspwm_rollout_energy_loss.detach().cpu()),
                    "lspwm_rollout_endpoint_loss": float(lspwm_rollout_endpoint_loss.detach().cpu()),
                    "lspwm_rollout_gaussian_nll": float(lspwm_rollout_gaussian_nll.detach().cpu()),
                    "lspwm_rollout_distribution_loss": float(lspwm_rollout_distribution_loss.detach().cpu()),
                    "lspwm_rollout_koopman_loss": float(lspwm_rollout_koopman_loss.detach().cpu()),
                    "lspwm_rollout_force_loss": float(lspwm_rollout_force_loss.detach().cpu()),
                    "lspwm_rollout_trajectory_nll": float(lspwm_rollout_trajectory_nll.detach().cpu()),
                    "lspwm_endpoint_energy_skill": float(lspwm_endpoint_energy_skill.detach().cpu()),
                    "lspwm_deterministic_skill": float(lspwm_deterministic_skill.detach().cpu()),
                    "lspwm_active_phase": lspwm_active_phase,
                    # v21 THM-JEPA reversible diagnostics
                    "lspwm_dissipative_force_norm": (
                        float(lspwm_dissipative_force_norm.detach().cpu())
                        if lspwm_dissipative_force_norm is not None else 0.0
                    ),
                    "lspwm_reversible_force_norm": (
                        float(lspwm_reversible_force_norm.detach().cpu())
                        if lspwm_reversible_force_norm is not None else 0.0
                    ),
                    "lspwm_bath_drift_norm": (
                        float(lspwm_bath_drift_norm.detach().cpu())
                        if lspwm_bath_drift_norm is not None else 0.0
                    ),
                    "lspwm_total_drift_norm": (
                        float(lspwm_total_drift_norm.detach().cpu())
                        if lspwm_total_drift_norm is not None else 0.0
                    ),
                    "lspwm_drift_incr_cosine": (
                        float(lspwm_drift_incr_cosine.detach().cpu())
                        if lspwm_drift_incr_cosine is not None else 0.0
                    ),
                    "lspwm_reversible_drift_fraction": (
                        float(lspwm_reversible_drift_fraction.detach().cpu())
                        if lspwm_reversible_drift_fraction is not None else 0.0
                    ),
                    "lspwm_physical_mobility_scale": float(
                        outputs.get(
                            "lspwm_physical_mobility_scale",
                            terms["delta_loss"].new_zeros(()),
                        ).detach().cpu()
                    ),
                    "lspwm_physical_free_energy_scale": float(
                        outputs.get(
                            "lspwm_physical_free_energy_scale",
                            terms["delta_loss"].new_zeros(()),
                        ).detach().cpu()
                    ),
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
                    "v5_transition_loss": float(v5_terms["transition_loss"].detach().cpu()),
                    "v5_persistence_loss": float(v5_terms["persistence_loss"].detach().cpu()),
                    "v5_transition_skill": float(v5_terms["transition_skill"].detach().cpu()),
                    "v5_state_reconstruction_loss": float(v5_terms["state_reconstruction_loss"].detach().cpu()),
                    "v5_feature_forecast_loss": float(v5_terms["feature_forecast_loss"].detach().cpu()),
                    "v5_observable_loss": float(v5_terms["observable_loss"].detach().cpu()),
                    "v5_observable_zero_motion_loss": float(v5_terms["observable_zero_motion_loss"].detach().cpu()),
                    "v5_observable_zero_motion_skill": float(v5_terms["observable_zero_motion_skill"].detach().cpu()),
                    "v5_transition_nll": float(v5_terms["transition_nll"].detach().cpu()),
                    "v5_horizon_ranking_loss": float(v5_terms["horizon_ranking_loss"].detach().cpu()),
                    "v5_correct_vs_best_fixed_skill": float(v5_terms["correct_vs_best_fixed_skill"].detach().cpu()),
                    "v5_horizon_difference_loss": float(v5_terms["horizon_difference_loss"].detach().cpu()),
                    "v5_target_horizon_difference": float(v5_terms["target_horizon_difference"].detach().cpu()),
                    "v5_observable_horizon_difference_loss": float(v5_terms["observable_horizon_difference_loss"].detach().cpu()),
                    "v5_observable_target_horizon_difference": float(v5_terms["observable_target_horizon_difference"].detach().cpu()),
                    "v5_slot_diversity_loss": float(v5_terms["slot_diversity_loss"].detach().cpu()),
                    "v5_gate_complexity_loss": float(v5_terms["gate_complexity_loss"].detach().cpu()),
                    "v5_mean_active_slots": float(v5_terms["mean_active_slots"].detach().cpu()),
                    "v5_effective_slots": float(v5_terms["effective_slots"].detach().cpu()),
                    "v5_detailed_balance_residual": float(v5_terms["detailed_balance_residual"].detach().cpu()),
                    "v51_delta_loss": float(v51_terms["delta_loss"].detach().cpu()),
                    "v51_persistence_loss": float(v51_terms["persistence_loss"].detach().cpu()),
                    "v51_delta_skill": float(v51_terms["delta_skill"].detach().cpu()),
                    "v51_normalized_delta_loss": float(v51_terms["normalized_delta_loss"].detach().cpu()),
                    "v51_horizon_calibration_loss": float(v51_terms["horizon_calibration_loss"].detach().cpu()),
                    "v51_online_delta_loss": float(v51_terms["online_delta_loss"].detach().cpu()),
                    "v51_online_normalized_delta_loss": float(v51_terms["online_normalized_delta_loss"].detach().cpu()),
                    "v51_displacement_magnitude_loss": float(v51_terms["displacement_magnitude_loss"].detach().cpu()),
                    "v51_nll_loss": float(v51_terms["nll_loss"].detach().cpu()),
                    "v51_current_alignment_loss": float(v51_terms["current_alignment_loss"].detach().cpu()),
                    "v51_observable_loss": float(v51_terms["observable_loss"].detach().cpu()),
                    "v51_oracle_observable_loss": float(v51_terms["oracle_observable_loss"].detach().cpu()),
                    "v51_online_oracle_observable_loss": float(v51_terms["online_oracle_observable_loss"].detach().cpu()),
                    "v51_observable_zero_motion_loss": float(v51_terms["observable_zero_motion_loss"].detach().cpu()),
                    "v51_observable_skill": float(v51_terms["observable_skill"].detach().cpu()),
                    "v51_oracle_observable_skill": float(v51_terms["oracle_observable_skill"].detach().cpu()),
                    "v51_online_oracle_observable_skill": float(v51_terms["online_oracle_observable_skill"].detach().cpu()),
                    "v51_horizon_ranking_loss": float(v51_terms["horizon_ranking_loss"].detach().cpu()),
                    "v51_correct_vs_best_wrong_skill": float(v51_terms["correct_vs_best_wrong_skill"].detach().cpu()),
                    "v51_history_ranking_loss": float(v51_terms["history_ranking_loss"].detach().cpu()),
                    "v51_history_advantage": float(v51_terms["history_advantage"].detach().cpu()),
                    "v51_history_truncation_loss": float(v51_terms["history_truncation_loss"].detach().cpu()),
                    "v51_history_truncation_advantage": float(v51_terms["history_truncation_advantage"].detach().cpu()),
                    "v51_causal_next_loss": float(v51_terms["causal_next_loss"].detach().cpu()),
                    "v51_causal_next_skill": float(v51_terms["causal_next_skill"].detach().cpu()),
                    "v51_horizon_consistency_loss": float(v51_terms["horizon_consistency_loss"].detach().cpu()),
                    "v51_horizon_consistency_skill": float(v51_terms["horizon_consistency_skill"].detach().cpu()),
                    "v51_physical_horizon_consistency_loss": float(v51_terms["physical_horizon_consistency_loss"].detach().cpu()),
                    "v51_physical_horizon_consistency_skill": float(v51_terms["physical_horizon_consistency_skill"].detach().cpu()),
                    "v51_semigroup_loss": float(v51_terms["semigroup_loss"].detach().cpu()),
                    "v51_semigroup_skill": float(v51_terms["semigroup_skill"].detach().cpu()),
                    "v51_variance_loss": float(v51_terms["variance_loss"].detach().cpu()),
                    "v51_covariance_loss": float(v51_terms["covariance_loss"].detach().cpu()),
                    "v51_load_balance_loss": float(v51_terms["load_balance_loss"].detach().cpu()),
                    "v51_active_load_balance_loss": float(v51_terms["active_load_balance_loss"].detach().cpu()),
                    "v51_mode_diversity_loss": float(v51_terms["mode_diversity_loss"].detach().cpu()),
                    "v51_mode_orthogonality_loss": float(v51_terms["mode_orthogonality_loss"].detach().cpu()),
                    "v51_mean_active_modes": float(v51_terms["mean_active_modes"].detach().cpu()),
                    "v51_target_displacement": float(v51_terms["target_displacement"].detach().cpu()),
                    "v51_predicted_displacement": float(v51_terms["predicted_displacement"].detach().cpu()),
                    "v51_displacement_ratio": float(v51_terms["displacement_ratio"].detach().cpu()),
                    "tmjepa_energy_score": float(tm_terms["energy_score"].detach().cpu()),
                    "tmjepa_forecast_relative_mse_loss": float(
                        tm_terms["forecast_relative_mse_loss"].detach().cpu()
                    ),
                    "tmjepa_persistence_energy_score": float(tm_terms["persistence_energy_score"].detach().cpu()),
                    "tmjepa_energy_skill": float(tm_terms["energy_skill"].detach().cpu()),
                    "tmjepa_transition_nll": float(tm_terms["transition_nll"].detach().cpu()),
                    "tmjepa_path_nll": float(tm_terms["path_nll"].detach().cpu()),
                    "tmjepa_equilibrium_score_loss": float(tm_terms["equilibrium_score_loss"].detach().cpu()),
                    "tmjepa_equilibrium_score_baseline": float(
                        tm_terms["equilibrium_score_baseline"].detach().cpu()
                    ),
                    "tmjepa_equilibrium_score_skill": float(
                        tm_terms["equilibrium_score_skill"].detach().cpu()
                    ),
                    "tmjepa_innovation_mean_loss": float(tm_terms["innovation_mean_loss"].detach().cpu()),
                    "tmjepa_innovation_covariance_loss": float(tm_terms["innovation_covariance_loss"].detach().cpu()),
                    "tmjepa_innovation_autocorrelation_loss": float(tm_terms["innovation_autocorrelation_loss"].detach().cpu()),
                    "tmjepa_innovation_covariance_spectral_loss": float(
                        tm_terms[
                            "innovation_covariance_spectral_loss"
                        ].detach().cpu()
                    ),
                    "tmjepa_innovation_covariance_worst_loss": float(
                        tm_terms["innovation_covariance_worst_loss"].detach().cpu()
                    ),
                    "tmjepa_innovation_autocorrelation_worst_loss": float(
                        tm_terms[
                            "innovation_autocorrelation_worst_loss"
                        ].detach().cpu()
                    ),
                    "tmjepa_memory_intervention_loss": float(tm_terms["memory_intervention_loss"].detach().cpu()),
                    "tmjepa_memory_intervention_advantage": float(tm_terms["memory_intervention_advantage"].detach().cpu()),
                    "tmjepa_memory_intervention_mse_advantage": float(
                        tm_terms["memory_intervention_mse_advantage"].detach().cpu()
                    ),
                    "tmjepa_memory_relative_intervention_loss": float(
                        tm_terms["memory_relative_intervention_loss"].detach().cpu()
                    ),
                    "tmjepa_memory_relative_intervention_advantage": float(
                        tm_terms[
                            "memory_relative_intervention_advantage"
                        ].detach().cpu()
                    ),
                    "tmjepa_memory_ratio_band_loss": float(
                        tm_terms["memory_ratio_band_loss"].detach().cpu()
                    ),
                    "tmjepa_memory_log_ratio_loss": float(
                        tm_terms["memory_log_ratio_loss"].detach().cpu()
                    ),
                    "tmjepa_memory_long_horizon_intervention_loss": float(
                        tm_terms[
                            "memory_long_horizon_intervention_loss"
                        ].detach().cpu()
                    ),
                    "tmjepa_memory_long_horizon_intervention_advantage": float(
                        tm_terms[
                            "memory_long_horizon_intervention_advantage"
                        ].detach().cpu()
                    ),
                    "tmjepa_memory_long_horizon_relative_intervention_loss": float(
                        tm_terms[
                            "memory_long_horizon_relative_intervention_loss"
                        ].detach().cpu()
                    ),
                    "tmjepa_memory_long_horizon_relative_intervention_advantage": float(
                        tm_terms[
                            "memory_long_horizon_relative_intervention_advantage"
                        ].detach().cpu()
                    ),
                    "tmjepa_memory_residual_loss": float(
                        tm_terms["memory_residual_loss"].detach().cpu()
                    ),
                    "tmjepa_memory_long_horizon_residual_loss": float(
                        tm_terms["memory_long_horizon_residual_loss"].detach().cpu()
                    ),
                    "tmjepa_long_horizon_energy_score": float(
                        tm_terms["long_horizon_energy_score"].detach().cpu()
                    ),
                    "tmjepa_long_horizon_forecast_relative_mse_loss": float(
                        tm_terms[
                            "long_horizon_forecast_relative_mse_loss"
                        ].detach().cpu()
                    ),
                    "tmjepa_long_horizon_transition_nll": float(
                        tm_terms["long_horizon_transition_nll"].detach().cpu()
                    ),
                    "tmjepa_long_horizon_path_nll": float(
                        tm_terms["long_horizon_path_nll"].detach().cpu()
                    ),
                    "tmjepa_projected_force_regularization_loss": float(
                        tm_terms[
                            "projected_force_regularization_loss"
                        ].detach().cpu()
                    ),
                    "tmjepa_auxiliary_kl_loss": float(tm_terms["auxiliary_kl_loss"].detach().cpu()),
                    "tmjepa_auxiliary_transition_kl_loss": float(tm_terms["auxiliary_transition_kl_loss"].detach().cpu()),
                    "tmjepa_learned_cv_variance_loss": float(tm_terms["learned_cv_variance_loss"].detach().cpu()),
                    "tmjepa_learned_cv_covariance_loss": float(tm_terms["learned_cv_covariance_loss"].detach().cpu()),
                    "tmjepa_vamp_loss": float(tm_terms["vamp_loss"].detach().cpu()),
                    "tmjepa_vamp2_score": float(tm_terms["vamp2_score"].detach().cpu()),
                    "tmjepa_mean_coverage_90": float(tm_terms["mean_coverage_90"].detach().cpu()),
                    "tmjepa_soft_coverage_90": float(
                        tm_terms["soft_coverage_90"].detach().cpu()
                    ),
                    "tmjepa_coverage_calibration_loss": float(
                        tm_terms["coverage_calibration_loss"].detach().cpu()
                    ),
                    "tmjepa_minimum_diffusion_eigenvalue": float(tm_terms["minimum_diffusion_eigenvalue"].detach().cpu()),
                    "tmjepa_diffusion_condition_number": float(
                        tm_terms["diffusion_condition_number"].detach().cpu()
                    ),
                    "tmjepa_diffusion_condition_loss": float(
                        tm_terms["diffusion_condition_loss"].detach().cpu()
                    ),
                    "tmjepa_mean_observation_noise_variance": float(
                        tm_terms["mean_observation_noise_variance"].detach().cpu()
                    ),
                    "tmjepa_observation_to_diffusion_increment_variance_ratio": float(
                        tm_terms[
                            "observation_to_diffusion_increment_variance_ratio"
                        ].detach().cpu()
                    ),
                    "tmjepa_correlated_observation_variance_fraction": float(
                        tm_terms[
                            "correlated_observation_variance_fraction"
                        ].detach().cpu()
                    ),
                    "tmjepa_mean_endpoint_calibration_log_scale": float(
                        tm_terms["mean_endpoint_calibration_log_scale"].detach().cpu()
                    ),
                    "tmjepa_mean_endpoint_calibration_variance_multiplier": float(
                        tm_terms[
                            "mean_endpoint_calibration_variance_multiplier"
                        ].detach().cpu()
                    ),
                    "tmjepa_integrated_memory_to_diffusion_ratio": float(
                        tm_terms["integrated_memory_to_diffusion_ratio"].detach().cpu()
                    ),
                    "tmjepa_bath_information_fraction": float(
                        tm_terms["bath_information_fraction"].detach().cpu()
                    ),
                    "tmjepa_mean_relaxation_time_ps": float(tm_terms["mean_relaxation_time_ps"].detach().cpu()),
                    "paired_by_horizon": horizon_log,
                    "horizon_frames": [int(x) for x in outputs["horizon_frames"].detach().cpu().tolist()],
                    "lr": optimizer.param_groups[-1]["lr"],
                    "learning_rates": {
                        str(group.get("group_name", index)): group["lr"]
                        for index, group in enumerate(optimizer.param_groups)
                    },
                    "ema": momentum,
                }
                record.update(gradient_diagnostics)
                if slow_terms is not None:
                    record.update(
                        {
                            "slow_delta_loss": float(slow_terms["delta_loss"].detach().cpu()),
                            "slow_state_loss": float(slow_terms["state_loss"].detach().cpu()),
                            "slow_relative_delta_loss": float(slow_terms["relative_delta_loss"].detach().cpu()),
                            "slow_alignment_loss": float(slow_terms["alignment_loss"].detach().cpu()),
                            "slow_vamp_loss": float(slow_terms["vamp_loss"].detach().cpu()),
                            "slow_vamp2_score": float(slow_terms["vamp2_score"].detach().cpu()),
                            "slow_contrastive_loss": float(slow_terms["contrastive_loss"].detach().cpu()),
                            "slow_variance_loss": float(slow_terms["variance_loss"].detach().cpu()),
                            "slow_covariance_loss": float(slow_terms["covariance_loss"].detach().cpu()),
                            "slow_skill": float(slow_terms["skill"].detach().cpu()),
                        }
                    )
                if model.ls2_reversible_dynamics is not None:
                    record.update(
                        {
                            "ls2_state_loss": float(ls2_terms["state_loss"].detach().cpu()),
                            "ls2_persistence_loss": float(
                                ls2_terms["persistence_loss"].detach().cpu()
                            ),
                            "ls2_state_skill": float(ls2_terms["state_skill"].detach().cpu()),
                            "ls2_assignment_loss": float(
                                ls2_terms["assignment_loss"].detach().cpu()
                            ),
                            "ls2_assignment_skill": float(
                                ls2_terms["assignment_skill"].detach().cpu()
                            ),
                            "ls2_nll": float(ls2_terms["nll"].detach().cpu()),
                            "ls2_balance_loss": float(
                                ls2_terms["balance_loss"].detach().cpu()
                            ),
                            "ls2_entropy_loss": float(
                                ls2_terms["entropy_loss"].detach().cpu()
                            ),
                            "ls2_vamp2_score": float(
                                ls2_terms["vamp2_score"].detach().cpu()
                            ),
                            "ls2_detailed_balance_max": float(
                                ls2_terms["detailed_balance_max"].detach().cpu()
                            ),
                            "ls2_stationarity_error": float(
                                ls2_terms["stationarity_error"].detach().cpu()
                            ),
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
    if config.init_checkpoint and config.resume_checkpoint:
        raise ValueError("use either init_checkpoint or resume_checkpoint, not both")
    _validate_init_freeze_contract(config)
    if config.data.training_mode == "paired_window":
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
