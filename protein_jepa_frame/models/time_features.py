from __future__ import annotations

import math

import torch
import torch.nn as nn


class LogTimeEmbedding(nn.Module):
    """Trainable embedding for physical time values measured in ps."""

    def __init__(self, latent_dim: int, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(64.0), self.num_frequencies))
        self.register_buffer("frequencies", frequencies)
        input_dim = 2 + 2 * self.num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, time_ps: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(time_ps):
            time_ps = time_ps.float()
        log_t = torch.log1p(time_ps.clamp(min=0.0)) / math.log(10.0)
        scaled = log_t.unsqueeze(-1) * self.frequencies.to(device=time_ps.device, dtype=time_ps.dtype)
        features = torch.cat(
            [
                log_t.unsqueeze(-1),
                torch.sqrt(time_ps.clamp(min=0.0) + 1.0).log10().unsqueeze(-1),
                torch.sin(scaled),
                torch.cos(scaled),
            ],
            dim=-1,
        )
        return self.proj(features)


def relative_clip_time(time_ps: torch.Tensor | None) -> torch.Tensor | None:
    if time_ps is None:
        return None
    return (time_ps - time_ps[:, :1]).clamp(min=0.0)


def lag_delta_time(time_ps: torch.Tensor | None, lag: int, length: int, device: torch.device) -> torch.Tensor:
    if time_ps is None:
        return torch.full((1, length), float(lag), device=device)
    if lag >= time_ps.shape[1]:
        return torch.full((time_ps.shape[0], length), float(lag), device=time_ps.device, dtype=time_ps.dtype)
    delta = (time_ps[:, lag:] - time_ps[:, :-lag]).clamp(min=0.0)
    if delta.shape[1] < length:
        pad = delta[:, -1:].expand(-1, length - delta.shape[1])
        delta = torch.cat([delta, pad], dim=1)
    return delta[:, :length]
