"""Horizon selection protocol (Gate 3).

Rules:
- short:  fastest observable decorrelation time, > 1 frame interval
- medium: characteristic time of slow observables
- long:   maximum candidate with sufficient independent pairs

All horizons defined in physical time (ps), converted to frames per dataset.
Output: read-only protocol/horizons.yaml, locked before test evaluation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class HorizonProtocol:
    """Physical-time horizon protocol for training and evaluation.
    
    Immutable after creation — test split must not influence these values.
    All times in picoseconds.
    """
    short_ps: float       # shortest horizon (ps)
    medium_ps: float      # medium horizon (ps)
    long_ps: float        # longest horizon (ps)
    frame_interval_ps: float = 1000.0  # base frame interval
    selection_split: str = "train"     # which split used for selection

    def as_dict(self) -> dict:
        return {
            "short_ps": self.short_ps,
            "medium_ps": self.medium_ps,
            "long_ps": self.long_ps,
            "frame_interval_ps": self.frame_interval_ps,
            "selection_split": self.selection_split,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HorizonProtocol":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.as_dict(), f)

    @classmethod
    def load(cls, path: str | Path) -> "HorizonProtocol":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


def compute_acf(
    values: torch.Tensor,
    max_lag: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Empirical autocorrelation function (bias-corrected, per-sample mean).
    
    Args:
        values: [N, T, d] or [N, T] trajectories
        max_lag: maximum lag in frames
        mask: [N, T] valid frame mask (1=valid)
        
    Returns:
        acf: [max_lag] mean ACF over samples and dims
    """
    if values.dim() == 2:
        values = values.unsqueeze(-1)  # [N, T, 1]
    N, T, d = values.shape
    if mask is None:
        mask = torch.ones(N, T, device=values.device)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # [N, T, 1]
    
    # Center each trajectory
    values_centered = values - values.mean(dim=1, keepdim=True)
    
    acf = torch.zeros(max_lag, device=values.device)
    for lag in range(max_lag):
        # Forward pairs: t, t+lag
        forward = values_centered[:, :T - lag] * values_centered[:, lag:]
        f_mask = mask[:, :T - lag] * mask[:, lag:]
        # Estimate: mean over valid pairs
        acf_val = (forward * f_mask).sum() / f_mask.sum().clamp(min=1.0)
        acf[lag] = acf_val
    
    # Normalize to correlation
    acf = acf / acf[0].clamp(min=1e-10)
    return acf


def compute_integrated_timescale(
    acf: torch.Tensor,
    max_lag: int | None = None,
) -> float:
    """Integrated autocorrelation time from ACF.
    
    tau_int = 1 + 2 * sum_{t=1}^{M} acf[t]
    with automatic window via Sokal's criterion.
    """
    if max_lag is None:
        max_lag = len(acf) - 1
    
    tau = 1.0
    for t in range(1, min(len(acf), max_lag + 1)):
        acf_t = acf[t].item()
        if acf_t <= 0:
            break
        tau += 2.0 * acf_t
    
    return tau


def select_horizons(
    acf_data: torch.Tensor,
    frame_interval_ps: float,
    min_separation_ps: float = 2000.0,
    max_horizon_ps: float | None = None,
) -> HorizonProtocol:
    """Select short/medium/long horizons from ACF data.
    
    Strategy:
    - short:  where ACF first drops below 1/e (effective decorrelation)
    - medium: 2 * tau_int (characteristic relaxation)
    - long:   4 * tau_int or max_horizon_ps, whichever is smaller
    
    Args:
        acf_data: ACF values [max_lag] or [N, T] raw data
        frame_interval_ps: time between frames in ps
        min_separation_ps: minimum gap between horizons
        max_horizon_ps: cap for long horizon
        
    Returns:
        HorizonProtocol with physical time horizons
    """
    if acf_data.dim() > 1:
        # Raw data — compute ACF first
        max_lag = min(acf_data.shape[1] // 2, 200)
        acf = compute_acf(acf_data, max_lag)
    else:
        acf = acf_data
        max_lag = len(acf) - 1
    
    tau_int = compute_integrated_timescale(acf)
    
    # Short: first crossing of 1/e
    short_frames = 1
    for t in range(1, len(acf)):
        if acf[t] < 1.0 / math.e:
            short_frames = t
            break
    else:
        short_frames = max(1, len(acf) // 4)
    
    # Convert to physical time and ensure minimum separation
    short_ps = max(short_frames * frame_interval_ps, min_separation_ps)
    medium_ps = max(2.0 * tau_int * frame_interval_ps, short_ps + min_separation_ps)
    
    if max_horizon_ps is not None:
        long_ps = min(max_horizon_ps, 4.0 * tau_int * frame_interval_ps)
    else:
        long_ps = min(max_lag * frame_interval_ps * 0.8, 4.0 * tau_int * frame_interval_ps)
    
    long_ps = max(long_ps, medium_ps + min_separation_ps)
    
    return HorizonProtocol(
        short_ps=round(short_ps),
        medium_ps=round(medium_ps),
        long_ps=round(long_ps),
        frame_interval_ps=frame_interval_ps,
    )
