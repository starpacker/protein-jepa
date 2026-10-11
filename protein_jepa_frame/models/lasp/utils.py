"""Shared utilities for the LaSP subpackage."""

from __future__ import annotations

import torch


def _as_batch_temperature(
    temperature_k: torch.Tensor | float | None,
    batch: int,
    reference: torch.Tensor,
    reference_temperature_k: float,
) -> torch.Tensor:
    if temperature_k is None:
        temperature = reference.new_full((batch,), reference_temperature_k)
    elif torch.is_tensor(temperature_k):
        temperature = temperature_k.to(device=reference.device, dtype=reference.dtype)
        if temperature.ndim == 0:
            temperature = temperature.expand(batch)
        if temperature.shape != (batch,):
            raise ValueError("temperature_k must be scalar or have shape [batch]")
    else:
        temperature = reference.new_full((batch,), float(temperature_k))
    if not bool(torch.isfinite(temperature).all()) or torch.any(temperature <= 0.0):
        raise ValueError("temperature_k must be finite and positive")
    return temperature.clamp(max=5000.0)
