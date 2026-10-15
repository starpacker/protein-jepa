"""Diffusion tensors: constant, state-dependent, and diagonal.

Contains:
  - ConstantDiffusion: global PSD diffusion with temperature clock
  - StateDependentDiffusion: heteroscedastic diffusion head
  - DiagonalDiffusion: diagonal diffusion for per-residue dynamics
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature


class ConstantDiffusion(nn.Module):
    """Global PSD diffusion tensor with the required temperature clock."""

    def __init__(
        self,
        state_dim: int,
        *,
        reference_temperature_k: float = 348.0,
        diffusion_scale: float = 1.0e-3,
        diagonal_floor: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if state_dim < 1 or reference_temperature_k <= 0.0:
            raise ValueError("invalid constant-diffusion dimensions")
        if diffusion_scale <= 0.0 or diagonal_floor <= 0.0:
            raise ValueError("diffusion scales must be positive")
        self.state_dim = int(state_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        self.diffusion_scale = float(diffusion_scale)
        self.diagonal_floor = float(diagonal_floor)
        raw = torch.eye(state_dim)
        raw.diagonal().copy_(torch.log(torch.expm1(raw.diagonal())))
        self.raw_factor = nn.Parameter(raw)

    def factor(self) -> torch.Tensor:
        lower = torch.tril(self.raw_factor, diagonal=-1)
        diagonal = F.softplus(torch.diagonal(self.raw_factor)) + self.diagonal_floor
        return lower + torch.diag(diagonal)

    def matrix(
        self,
        batch: int | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference is None:
            reference = self.raw_factor
        if batch is None:
            if torch.is_tensor(temperature_k) and temperature_k.ndim > 0:
                batch = int(temperature_k.shape[0])
            else:
                batch = 1
        temperature = _as_batch_temperature(
            temperature_k, batch, reference, self.reference_temperature_k,
        )
        factor = self.factor().to(dtype=reference.dtype, device=reference.device)
        base = self.diffusion_scale * (factor @ factor.T)
        return base.unsqueeze(0) * (
            temperature / self.reference_temperature_k
        ).view(-1, 1, 1)

    def forward(
        self,
        batch: int,
        temperature_k: torch.Tensor | float | None = None,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.matrix(batch, temperature_k, reference=reference)


class StateDependentDiffusion(nn.Module):
    """Heteroscedastic diffusion head: predicts per-sample diagonal from state + condition."""

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        *,
        reference_temperature_k: float = 348.0,
        diffusion_scale: float = 1.0e-3,
        diagonal_floor: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if state_dim < 1 or reference_temperature_k <= 0.0:
            raise ValueError("invalid state-dependent diffusion dimensions")
        if diffusion_scale <= 0.0 or diagonal_floor <= 0.0:
            raise ValueError("diffusion scales must be positive")
        self.state_dim = int(state_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        self.diffusion_scale = float(diffusion_scale)
        self.diagonal_floor = float(diagonal_floor)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.log_amplitude = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        batch: int | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference is None or reference.ndim == 1 or reference.shape[-1] != self.state_dim:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            diag = self.diffusion_scale * torch.exp(self.log_amplitude) * torch.ones(
                int(batch or 1), self.state_dim, device=device, dtype=dtype
            )
        else:
            q = reference[..., :self.state_dim].contiguous()
            raw = self.net(q)
            diag = self.diffusion_scale * torch.exp(self.log_amplitude) * (
                F.softplus(raw) + self.diagonal_floor
            )
        temperature = _as_batch_temperature(
            temperature_k, int(batch or diag.shape[0]), diag, self.reference_temperature_k,
        )
        diag = diag * (temperature / self.reference_temperature_k).unsqueeze(-1)
        return torch.diag_embed(diag.clamp(min=1e-12))


class DiagonalDiffusion(nn.Module):
    """State/condition dependent diagonal diffusion with temperature scaling."""

    def __init__(self, state_dim: int, condition_dim: int = 0, hidden_dim: int = 128,
                 reference_temperature_k: float = 348.0, diffusion_scale: float = 1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.reference_temperature_k = reference_temperature_k
        self.diffusion_scale = diffusion_scale
        self.net = nn.Sequential(
            nn.Linear(state_dim + condition_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, G, condition=None, temperature_k=None):
        if G.ndim == 2:
            c = G.new_zeros(G.shape[0], self.condition_dim) if condition is None else condition
            raw = self.net(torch.cat([G, c], -1))
            d = F.softplus(raw) * self.diffusion_scale
            t = _as_batch_temperature(temperature_k, G.shape[0], G, self.reference_temperature_k)
            return torch.diag_embed(d * (t / self.reference_temperature_k).unsqueeze(-1))
        return self.forward(
            G.reshape(-1, self.state_dim), condition, temperature_k
        ).reshape(*G.shape[:-1], self.state_dim, self.state_dim)
