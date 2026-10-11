"""Mori bath structures: multi-timescale, gauge-fixed, and projected force closure.

Contains:
  - MultiTimescaleBath: fast/mid/slow auxiliary OU bath
  - GaugeProjector: gauge condition projector
  - ResidueConditionalDecoder: condition on G_t, G_next
  - GaugeFixedBath: orthogonal, spectrally ordered bath realization
  - ProjectedForceClosure: bounded unresolved-force correction
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature


class MultiTimescaleBath(nn.Module):
    """FDT-consistent fast/mid/slow auxiliary OU bath."""

    def __init__(self, state_dim=8, num_fast=4, num_mid=8, num_slow=4,
                 init_taus=(1000., 8000., 64000.)):
        super().__init__()
        self.state_dim = state_dim
        self.group_sizes = (num_fast, num_mid, num_slow)
        self.log_tau_fast = nn.Parameter(torch.full((num_fast,), math.log(init_taus[0])))
        self.log_tau_mid = nn.Parameter(torch.full((num_mid,), math.log(init_taus[1])))
        self.log_tau_slow = nn.Parameter(torch.full((num_slow,), math.log(init_taus[2])))
        self.coupling_fast = nn.Parameter(torch.randn(state_dim, num_fast) * .01)
        self.coupling_mid = nn.Parameter(torch.randn(state_dim, num_mid) * .01)
        self.coupling_slow = nn.Parameter(torch.randn(state_dim, num_slow) * .01)

    @property
    def auxiliary_dim(self):
        return sum(self.group_sizes)

    def parameters_at(self, dtype=None, device=None):
        taus = torch.cat([
            self.log_tau_fast.exp(), self.log_tau_mid.exp(), self.log_tau_slow.exp()
        ]).to(dtype=dtype, device=device)
        C = torch.cat([
            self.coupling_fast, self.coupling_mid, self.coupling_slow
        ], 1).to(dtype=dtype, device=device)
        return taus, C

    def forward(self, s, dt=1.0):
        taus, C = self.parameters_at(s.dtype, s.device)
        gamma = taus.reciprocal()
        return s * torch.exp(-gamma * dt), C


class GaugeProjector(nn.Module):
    """Projects latents to a gauge-condition vector for symmetry breaking."""

    def __init__(self, latent_dim=256, gauge_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, gauge_dim),
        )

    def forward(self, H, ca_coords=None, residue_mask=None):
        g = self.net(H)
        mask = (torch.ones(g.shape[:-1], device=g.device, dtype=g.dtype)
                if residue_mask is None else residue_mask.to(g.dtype))
        pooled = (g * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return pooled, ca_coords


class ResidueConditionalDecoder(nn.Module):
    """Decoder predicting q_next from q, G_t, G_next."""

    def __init__(self, cv_dim=8, global_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cv_dim + 2 * global_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 2 * cv_dim),
        )
        self.cv_dim = cv_dim

    def forward(self, q, G_t, G_next):
        p = self.net(torch.cat([
            q,
            G_t[:, None, :].expand(-1, q.shape[1], -1),
            G_next[:, None, :].expand(-1, q.shape[1], -1),
        ], -1))
        return p[..., :self.cv_dim], F.softplus(p[..., self.cv_dim:]) + 1e-4


class GaugeFixedBath(nn.Module):
    """Minimal, orthogonal and spectrally ordered bath realization."""

    def __init__(
        self,
        state_dim: int,
        auxiliary_dim: int,
        *,
        relaxation_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0),
        coupling_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if state_dim < 1 or auxiliary_dim < 1 or auxiliary_dim > state_dim:
            raise ValueError("bath dimension must be between 1 and state_dim")
        relaxation_times_ps = tuple(float(value) for value in relaxation_times_ps)
        if len(relaxation_times_ps) != auxiliary_dim or any(
            value <= 0.0 for value in relaxation_times_ps
        ):
            raise ValueError("relaxation_times_ps must be positive and complete")
        if tuple(sorted(relaxation_times_ps)) != relaxation_times_ps:
            raise ValueError("relaxation_times_ps must be in ascending order")
        if coupling_scale < 0.0:
            raise ValueError("coupling_scale must be non-negative")
        self.state_dim = int(state_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.coupling_scale = float(coupling_scale)
        self.raw_basis = nn.Parameter(torch.randn(state_dim, auxiliary_dim) * 0.02)
        self.raw_amplitude = nn.Parameter(torch.full((auxiliary_dim,), -2.0))
        self.register_buffer(
            "reference_relaxation_rates",
            torch.tensor(relaxation_times_ps, dtype=torch.float32).reciprocal(),
        )

    def basis(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.raw_basis, mode="reduced")
        sign = torch.sign(torch.diagonal(r)).masked_fill(
            torch.diagonal(r).abs() < 1.0e-8, 1.0,
        )
        return q * sign.unsqueeze(0)

    def parameters_at(
        self, *, dtype: torch.dtype, device: torch.device,
    ) -> dict[str, torch.Tensor]:
        basis = self.basis().to(dtype=dtype, device=device)
        rates = self.reference_relaxation_rates.to(dtype=dtype, device=device)
        amplitudes = self.coupling_scale * F.softplus(self.raw_amplitude).to(
            dtype=dtype, device=device,
        )
        coupling = basis * amplitudes.unsqueeze(0)
        return {"basis": basis, "rates": rates, "amplitudes": amplitudes, "coupling": coupling}

    def forward(
        self,
        time_ps: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dtype = dtype or self.raw_basis.dtype
        device = device or self.raw_basis.device
        params = self.parameters_at(dtype=dtype, device=device)
        time = time_ps.to(dtype=dtype, device=device).reshape(-1)
        decay = torch.exp(-time[:, None] * params["rates"][None, :])
        return torch.einsum("qa,ta,ra->tqr", params["coupling"], decay, params["coupling"])


class ProjectedForceClosure(nn.Module):
    """Bounded unresolved-force correction with an O(dt^2) short-time gate."""

    def __init__(
        self,
        state_dim: int,
        history_feature_dim: int,
        *,
        hidden_dim: int = 128,
        max_shift: float = 0.55,
        horizon_gate_steps: float = 2.0,
        base_step_ps: float = 1000.0,
        reference_temperature_k: float = 348.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or history_feature_dim < 1 or hidden_dim < 4:
            raise ValueError("invalid projected-force closure dimensions")
        if max_shift < 0.0 or horizon_gate_steps <= 0.0 or base_step_ps <= 0.0:
            raise ValueError("projected-force closure scales are invalid")
        self.state_dim = int(state_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.max_shift = float(max_shift)
        self.horizon_gate_steps = float(horizon_gate_steps)
        self.base_step_ps = float(base_step_ps)
        self.reference_temperature_k = float(reference_temperature_k)
        input_dim = state_dim + history_feature_dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        q: torch.Tensor,
        history_features: torch.Tensor,
        dt_ps: torch.Tensor | float,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if q.ndim != 2 or q.shape[-1] != self.state_dim:
            raise ValueError("q has the wrong projected-force state shape")
        if history_features.shape != (q.shape[0], self.history_feature_dim):
            raise ValueError("history_features has the wrong projected-force shape")
        dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
        if dt.ndim == 0:
            dt = dt.expand(q.shape[0])
        if dt.shape != (q.shape[0],):
            raise ValueError("dt_ps must be scalar or have shape [batch]")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.reference_temperature_k,
        )
        horizon = (dt / self.base_step_ps).clamp(min=0.0)
        features = torch.cat(
            [
                q, history_features,
                torch.log1p(horizon).unsqueeze(-1),
                torch.sqrt(horizon + 1.0e-6).unsqueeze(-1),
                (temperature / self.reference_temperature_k - 1.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        shift = self.max_shift * torch.tanh(self.net(features))
        gate = -torch.expm1(-horizon / self.horizon_gate_steps)
        return shift * gate.square().unsqueeze(-1)
