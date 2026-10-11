from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
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
        reference = outputs.get("lspwm_target_state")
        if reference is None:
            # LSPWM not active — return zero
            return outputs.get("coords", torch.zeros(1)).new_zeros(())
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


def _write_run_manifest(
    output_dir: Path,
    *,
    config_dict: dict,
    init_checkpoint_path: str | None = None,
    previous_manifest_path: str | None = None,
    data_paths: list[str] | None = None,
) -> Path:
    """Write run manifest capturing full provenance per P0-11 audit.

    Includes: git SHA, dirty diff hash, full initial config, resume lineage,
    upstream checkpoint SHA256, dependency lock, RNG state.
    """
    manifest: dict = {}

    # ── Git state ──
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=output_dir.parent, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=output_dir.parent, text=True
        )
        manifest["git_sha"] = sha
        manifest["git_dirty_hash"] = hashlib.sha256(dirty.encode()).hexdigest() if dirty else "clean"
    except (subprocess.CalledProcessError, FileNotFoundError):
        manifest["git_sha"] = "unknown"
        manifest["git_dirty_hash"] = "unknown"

    # ── Full config ──
    manifest["config"] = config_dict

    # ── Resume lineage ──
    lineage: list[dict] = []
    if previous_manifest_path is not None:
        try:
            with open(previous_manifest_path) as f:
                prev = json.load(f)
            lineage.extend(prev.get("lineage", []))
            lineage.append({"manifest": str(previous_manifest_path)})
        except Exception:
            pass
    manifest["lineage"] = lineage

    # ── Upstream checkpoint hash ──
    if init_checkpoint_path is not None and os.path.exists(init_checkpoint_path):
        sha256 = hashlib.sha256()
        with open(init_checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        manifest["init_checkpoint_sha256"] = sha256.hexdigest()
    else:
        manifest["init_checkpoint_sha256"] = None

    # ── Data manifest ──
    if data_paths:
        data_hashes = {}
        for dp in data_paths:
            if os.path.isdir(dp):
                data_hashes[dp] = "directory"
            elif os.path.isfile(dp):
                sha256 = hashlib.sha256()
                with open(dp, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                data_hashes[dp] = sha256.hexdigest()
        manifest["data_manifest"] = data_hashes

    # ── Dependency lock ──
    try:
        deps = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL,
        )
        manifest["dependency_lock"] = deps.strip().split("\n")
    except Exception:
        manifest["dependency_lock"] = "unavailable"

    # ── RNG state ──
    manifest["rng_state"] = {
        "torch_rng": torch.get_rng_state().tolist() if hasattr(torch.get_rng_state(), "tolist") else str(torch.get_rng_state()),
        "numpy_rng": str(np.random.get_state()),
        "python_rng": str(random.getstate()),
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


