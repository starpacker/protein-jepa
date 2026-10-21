"""Shared training helpers for the alive Stage A v2 / B1a / B1b trainers.

Only the helpers actually imported by the alive entry points are kept:
  _device, _set_reproducibility, _move_batch, _momentum, _lr_scale, _warmup_scale
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch


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
    # warn_only=True: keep deterministic algorithms where the backend provides
    # them, but do not hard-crash on ops that lack a deterministic CUDA kernel
    # (e.g. cumsum in the ensemble-RMSF surrogate).  A plain True aborts the run
    # mid-step; warn_only degrades gracefully while preserving reproducibility
    # for every op that does support it.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _momentum(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine schedule for the EMA momentum, from `start` to `end`."""
    if total_steps <= 1:
        return end
    progress = min(1.0, step / float(total_steps - 1))
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def _lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up followed by cosine decay, as a multiplier in [0, 1]."""
    if warmup_steps > 0 and step < warmup_steps:
        return max(1.0e-8, (step + 1) / float(warmup_steps))
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0, step - warmup_steps) / float(denom))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _warmup_scale(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / float(warmup_steps))
