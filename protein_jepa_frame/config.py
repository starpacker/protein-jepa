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
    lag_predictor_depth: int = 2
    linear_dynamics_lags: tuple[int, ...] = ()
    physchem_prediction_lags: tuple[int, ...] = ()
    physchem_predictor_depth: int = 2
    physchem_contact_cutoff: float = 8.0
    num_heads: int = 4
    num_rbf: int = 16
    dropout: float = 0.1
    max_frames: int = 256
    max_residues: int = 4096
    feature_norm: bool = True


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
    msm_state_weight: float = 0.0
    vamp2_weight: float = 0.0
    vamp2_lag: int = 1


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
    strict_init: bool = True
    log_every: int = 20
    save_every: int = 1


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


def config_to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return {f.name: config_to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, tuple):
        return list(config)
    return config
