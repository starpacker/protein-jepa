from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest_path: str | None = None
    batch_size: int = 4
    num_workers: int = 0
    num_frames: int = 8
    frame_stride: int = 1
    window_stride: int = 1
    max_residues: int | None = None
    coordinate_scale: float = 1.0
    synthetic_size: int = 128
    synthetic_residues: int = 64
    msm_labels_path: str | None = None
    sampling_mode: str = "clip_uniform"
    dataset_weights: dict[str, float] = field(default_factory=dict)
    training_mode: str = "clip"
    paired_horizons_frames: tuple[int, ...] = (8, 16, 24, 48)
    paired_nonoverlap_only: bool = False
    paired_all_horizons: bool = False
    load_forces: bool = False


@dataclass
class MaskConfig:
    mode: str = "block"
    num_blocks: int = 2
    temporal_scale: tuple[float, float] = (0.25, 0.5)
    residue_scale: tuple[float, float] = (0.15, 0.4)
    min_context_ratio: float = 0.2
    future_context_ratio: float = 0.5
    future_gap_frames: int = 0


@dataclass
class ModelConfig:
    latent_dim: int = 128
    encoder_depth: int = 4
    temporal_depth: int = 0
    predictor_depth: int = 4
    state_predictor_depth: int = 2
    state_prediction_lags: tuple[int, ...] = ()
    frame_pooling: str = "mean"
    frame_pool_queries: int = 4
    lag_predictor_depth: int = 2
    lag_predictor_dropout: float | None = None
    lag_conditioning: str = "none"
    lag_prediction_mode: str = "direct"
    time_conditioning: str = "none"
    thermodynamic_conditioning: bool = False
    orientation_features: bool = False
    temporal_causal: bool = False
    temporal_position_encoding: str = "learned"
    temporal_delta_features: bool = False
    motion_depth: int = 0
    motion_stopgrad_input: bool = False
    transition_source: str = "state"
    transition_thermodynamic_conditioning: bool = False
    linear_dynamics_lags: tuple[int, ...] = ()
    physchem_prediction_lags: tuple[int, ...] = ()
    physchem_predictor_depth: int = 2
    physchem_contact_cutoff: float = 8.0
    kinetic_readout_dim: int = 0
    kinetic_readout_depth: int = 2
    kinetic_readout_stopgrad: bool = False
    num_heads: int = 4
    num_rbf: int = 16
    dropout: float = 0.1
    max_frames: int = 256
    max_residues: int = 4096
    feature_norm: bool = True
    paired_forecast: bool = False
    paired_predictor_depth: int = 3
    slow_state_enabled: bool = False
    slow_state_dim: int = 32
    slow_predictor_depth: int = 2
    dual_temporal_towers: bool = False
    offline_temporal_depth: int = 0
    causal_temporal_depth: int = 0
    probabilistic_transition: bool = False
    transition_log_sigma_min: float = -5.0
    transition_log_sigma_max: float = 1.5
    observable_head: bool = False
    observable_dim: int = 4
    causal_encoder_gradient_scale: float = 1.0
    paired_teacher_space_delta: bool = False
    paired_target_tower: str = "causal"
    freeze_context_encoder: bool = False
    freeze_offline_temporal_encoder: bool = False
    history_conditioned_transition: bool = False
    history_transition_strides: tuple[int, ...] = (1, 2, 4, 8)
    metastable_state_dim: int = 0
    observable_target_version: str = "v2"
    local_dynamics_enabled: bool = False
    local_observable_targets: bool = True
    local_dynamics_tokens: int = 4
    local_dynamics_dim: int = 32
    closed_local_state_enabled: bool = False
    closed_local_state_dim: int = 12
    closed_local_horizons: tuple[int, ...] = (48, 96, 192)
    closed_local_state_scale: tuple[float, ...] = (1.0,) * 12
    closed_local_state_mean: tuple[float, ...] = (0.0,) * 12
    closed_local_operator_mode: str = "experts"
    closed_local_relaxation_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    closed_local_velocity_strides: tuple[int, ...] = (1, 2, 4, 8)
    closed_local_log_sigma_min: float = -4.0
    closed_local_log_sigma_max: float = 2.0
    closed_local_base_step_frames: int = 16
    closed_local_history_frame_stride: int = 1
    closed_local_cross_segment_heads: int = 4
    freeze_closed_local_operator: bool = False
    v4_slow_state_enabled: bool = False
    v4_slow_latent_dim: int = 16
    v4_slow_num_states: int = 8
    v4_slow_vamp_lags: tuple[int, ...] = (4, 8, 16)
    v4_physics_state_enabled: bool = False
    v4_physics_latent_dim: int = 16
    v4_physics_num_states: int = 8
    v4_physics_vamp_lags: tuple[int, ...] = (4, 8, 16)
    v4_physics_lowpass_width: int = 5
    v4_physics_force_scale: float = 50.0
    v4_physics_dynamic_only: bool = False
    v4_physics_condition_on_equilibrium: bool = False
    v4_physics_per_trajectory_vamp: bool = False
    freeze_v4_physics_representation: bool = False
    v4_physics_training_fixed_controls: bool = False


@dataclass
class OptimizationConfig:
    epochs: int = 10
    steps_per_epoch: int | None = None
    lr: float = 3e-4
    encoder_lr_scale: float = 0.2
    weight_decay: float = 0.05
    warmup_steps: int = 500
    ema_start: float = 0.996
    ema_end: float = 1.0
    grad_clip: float | None = 1.0
    precision: str = "float32"
    auxiliary_warmup_steps: int = 0
    residue_jepa_weight: float = 1.0
    state_jepa_weight: float = 0.0
    lagged_state_jepa_weight: float = 0.0
    semigroup_weight: float = 0.0
    state_transition_weight: float = 0.0
    physchem_weight: float = 0.0
    linear_dynamics_weight: float = 0.0
    variance_reg: float = 0.0
    state_variance_reg: float = 0.0
    state_feature_variance_reg: float = 0.0
    state_covariance_reg: float = 0.0
    msm_state_weight: float = 0.0
    vamp2_weight: float = 0.0
    vamp2_lag: int = 1
    kinetic_vamp_weight: float = 0.0
    kinetic_vamp_lag: int = 1
    paired_delta_weight: float = 1.0
    paired_state_weight: float = 0.25
    paired_endpoint_variance_weight: float = 0.02
    slow_delta_weight: float = 0.5
    slow_state_weight: float = 0.25
    slow_contrastive_weight: float = 0.1
    slow_variance_weight: float = 0.05
    slow_covariance_weight: float = 0.01
    gaussian_nll_weight: float = 0.0
    observable_weight: float = 0.0
    offline_forecast_weight: float = 0.0
    offline_lr_scale: float = 1.0
    causal_alignment_weight: float = 0.0
    metastable_vamp_weight: float = 0.0
    metastable_balance_weight: float = 0.0
    metastable_entropy_weight: float = 0.0
    local_observable_weight: float = 0.0
    closed_local_state_weight: float = 0.0
    horizon_ranking_weight: float = 0.0
    horizon_ranking_margin: float = 0.002
    horizon_difference_weight: float = 0.0
    horizon_difference_min_gap: int = 0
    closed_local_robust_state_weight: float = 0.0
    horizon_robust_difference_weight: float = 0.0
    closed_local_student_t_weight: float = 0.0
    closed_local_robust_beta: float = 0.5
    closed_local_student_t_df: float = 3.0
    closed_local_base_step_weight: float = 0.0
    closed_local_robust_base_step_weight: float = 0.0
    v4_slow_reconstruction_weight: float = 0.0
    v4_slow_vamp_weight: float = 0.0
    v4_slow_balance_weight: float = 0.0
    v4_slow_entropy_weight: float = 0.0
    v4_slow_base_transition_weight: float = 0.0
    v4_slow_rollout_transition_weight: float = 0.0
    v4_physics_reconstruction_weight: float = 0.0
    v4_physics_vamp_weight: float = 0.0
    v4_physics_balance_weight: float = 0.0
    v4_physics_entropy_weight: float = 0.0
    v4_physics_cross_covariance_weight: float = 0.0
    v4_physics_increment_covariance_weight: float = 0.0
    v4_physics_timescale_weight: float = 0.0
    v4_physics_trajectory_balance_weight: float = 0.0
    v4_physics_activity_weight: float = 0.0
    v4_physics_latent_variance_weight: float = 0.0
    v4_physics_drift_weight: float = 0.0
    v4_physics_force_weight: float = 0.0
    v4_physics_stationary_weight: float = 0.0
    v4_physics_base_transition_weight: float = 0.0
    v4_physics_rollout_transition_weight: float = 0.0
    v4_physics_transition_nll_weight: float = 0.0
    v4_physics_horizon_ranking_weight: float = 0.0
    v4_physics_horizon_ranking_margin: float = 0.002
    v4_physics_horizon_difference_weight: float = 0.0


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    seed: int = 0
    device: str = "auto"
    output_dir: str = "artifacts/run"
    init_checkpoint: str | None = None
    init_exclude_prefixes: tuple[str, ...] = ()
    sync_target_after_init: bool = False
    resume_checkpoint: str | None = None
    strict_init: bool = True
    init_copy_temporal_to_offline: bool = False
    log_every: int = 20
    save_every: int = 1
    checkpoint_every_steps: int = 10000


def _coerce_dataclass(cls: type, values: dict[str, Any]) -> Any:
    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    for name, spec in field_map.items():
        if name not in values:
            continue
        value = values[name]
        default_value = getattr(cls(), name)
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[name] = _coerce_dataclass(type(default_value), value)
        elif isinstance(default_value, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must decode to a mapping")
    return _coerce_dataclass(TrainConfig, raw)


def model_config_from_dict(values: dict[str, Any], state_dict: dict[str, Any] | None = None) -> ModelConfig:
    """Build a model config while recovering flags absent from old artifacts."""
    resolved = dict(values)
    keys = tuple(state_dict or ())
    if any(key.startswith("paired_delta_predictor.") for key in keys):
        resolved.setdefault("paired_forecast", True)
    if any(key.startswith("slow_state_projector.") for key in keys):
        resolved["slow_state_enabled"] = True
    if any(key.startswith("offline_temporal_encoder.") for key in keys):
        resolved["dual_temporal_towers"] = True
    if any(key.startswith("gaussian_transition.") for key in keys):
        resolved["probabilistic_transition"] = True
    if any(key.startswith("observable_head.") for key in keys):
        resolved["observable_head"] = True
    if any(key.startswith("paired_delta_predictor.summary.") for key in keys):
        resolved["history_conditioned_transition"] = True
    metastable_weights = []
    for key, value in (state_dict or {}).items():
        parts = key.split(".")
        if len(parts) == 4 and parts[:2] == ["metastable_state_head", "net"] and parts[3] == "weight":
            metastable_weights.append((int(parts[2]), value))
    if metastable_weights:
        resolved["metastable_state_dim"] = int(max(metastable_weights, key=lambda item: item[0])[1].shape[0])
    if any(key.startswith("local_dynamics_head.") for key in keys):
        resolved["local_dynamics_enabled"] = True
    local_token_weight = (state_dict or {}).get("local_dynamics_head.token_proj.1.weight")
    if local_token_weight is not None:
        resolved["local_dynamics_dim"] = int(local_token_weight.shape[0])
    local_segment_weight = (state_dict or {}).get("local_dynamics_head.segment_embed.weight")
    if local_segment_weight is not None:
        resolved["local_dynamics_tokens"] = int(local_segment_weight.shape[0])
    if any(key.startswith("closed_local_operator.") for key in keys):
        resolved["closed_local_state_enabled"] = True
    if any(key.startswith("v4_slow_kinetic_model.") for key in keys):
        resolved["v4_slow_state_enabled"] = True
    if any(key.startswith("v4_physics_slow_model.") for key in keys):
        resolved["v4_physics_state_enabled"] = True
    known = {item.name for item in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in resolved.items() if key in known})


def config_to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return {f.name: config_to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, tuple):
        return list(config)
    return config
