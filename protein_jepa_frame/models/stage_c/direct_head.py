"""Direct endpoint prediction head for Stage C.

Gate 6 - direct_endpoint mode:
- New per-residue distribution head p(Q_{t+tau}|history,c,T,tau)
- Explicitly receives tau_ps (horizon time in picoseconds)
- Produces multiple independent latent state samples from same context
- All samples decoded by shared StructureDecoder
- Training: only target time for supervision (no future-window frames)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DirectHeadConfig:
    """Configuration for the direct endpoint prediction head."""
    latent_dim: int = 8          # Q dimension (matches Stage B)
    hidden_dim: int = 128         # hidden dimension
    hidden_depth: int = 2          # depth of prediction MLP
    context_dim: int = 256         # H dimension from Stage A encoder
    condition_dim: int = 64        # static condition dimension
    num_samples: int = 1           # number of samples per context during inference
    sample_temperature: float = 1.0  # temperature for sampling log-variance


class DirectEndpointHead(nn.Module):
    """Per-residue distribution head for p(Q_{t+tau} | history, c, T, tau).

    Produces Gaussian parameters (mean, log_sigma) per residue at target horizon.
    Input: frozen H_history, static_condition, temperature, horizon_time_ps.
    Output: [B, L, latent_dim] mean + [B, L, latent_dim] log_sigma.

    Architecture invariants:
    - Receives tau_ps explicitly (not frame index)
    - No access to target time or future window frames during training
    - Shared across all residue positions
    """

    def __init__(self, config: DirectHeadConfig | None = None):
        super().__init__()
        self.config = config or DirectHeadConfig()
        c = self.config

        # Time encoding: tau_ps -> sinusoidal embedding -> time_dim=32
        self.time_dim = 32
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        # Temperature encoding
        self.temp_dim = 16
        self.temp_proj = nn.Sequential(
            nn.Linear(1, self.temp_dim),
            nn.SiLU(),
            nn.Linear(self.temp_dim, self.temp_dim),
        )

        # Residue-level predictor: (H[L,256] + condition[L,64] + time[32] + temp[16]) -> mean, log_sigma
        per_residue_input = c.context_dim + c.condition_dim + self.time_dim + self.temp_dim

        layers = []
        in_dim = per_residue_input
        for _ in range(c.hidden_depth):
            layers.extend([
                nn.Linear(in_dim, c.hidden_dim),
                nn.SiLU(),
            ])
            in_dim = c.hidden_dim
        self.shared_net = nn.Sequential(*layers)

        # Separate heads for mean and log_sigma
        self.mean_head = nn.Linear(in_dim, c.latent_dim)
        self.log_sigma_head = nn.Linear(in_dim, c.latent_dim)

        # Initialize mean_head to near-zero output so that initial Q_pred ≈ 0,
        # which produces FrameUpdate ≈ identity (decoder outputs ≈ anchor coords).
        # The model then learns non-zero Q deltas from a reasonable starting point.
        # Without this, random-initialized weights push Q to extreme values
        # (norm ~168 vs oracle ~15), FrameUpdate blows up coordinates, and
        # FAPE saturates at the 10.0 clamp with zero effective gradient.
        nn.init.normal_(self.mean_head.weight, std=1e-4)
        self.mean_head.bias.data.zero_()

        # Initialize log_sigma to small values
        self.log_sigma_head.weight.data.zero_()
        self.log_sigma_head.bias.data.fill_(-2.0)  # log(sigma) = -2 => sigma ~ 0.14

    def _encode_time(self, tau_ps: torch.Tensor) -> torch.Tensor:
        """Encode horizon time into sinusoidal features.

        Args:
            tau_ps: [B] or [B, 1] horizon time in picoseconds

        Returns:
            [B, time_dim] time encoding
        """
        if tau_ps.dim() == 1:
            tau_ps = tau_ps.unsqueeze(-1)
        # Log-scale encoding for wide range of timescales
        tau_safe = tau_ps.clamp(min=1.0)  # prevent log(0)
        tau_log = torch.log(tau_safe)
        return self.time_proj(tau_log)

    def _encode_temperature(self, temperature_k: torch.Tensor) -> torch.Tensor:
        """Encode temperature into features.

        Args:
            temperature_k: [B] or [B, 1]

        Returns:
            [B, temp_dim]
        """
        if temperature_k.dim() == 1:
            temperature_k = temperature_k.unsqueeze(-1)
        return self.temp_proj(temperature_k)

    def forward(
        self,
        h_history: torch.Tensor,
        static_condition: torch.Tensor,
        tau_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-residue latent state distribution at horizon tau.

        Args:
            h_history: [B, L, context_dim] frozen H features from last observed frame
            static_condition: [B, L, condition_dim] or None
            tau_ps: [B] horizon time in picoseconds
            temperature_k: [B] simulation temperature
            residue_mask: [B, L] valid residue mask

        Returns:
            mean: [B, L, latent_dim] predicted Q at tau
            log_sigma: [B, L, latent_dim] per-dimension log standard deviation
        """
        B, L = h_history.shape[:2]
        device = h_history.device

        # Encode time and temperature
        time_enc = self._encode_time(tau_ps).unsqueeze(1).expand(B, L, self.time_dim)
        temp_enc = self._encode_temperature(temperature_k).unsqueeze(1).expand(B, L, self.temp_dim)

        # Concatenate per-residue features
        if static_condition is None:
            static_condition = torch.zeros(B, L, self.config.condition_dim, device=device)
        x = torch.cat([h_history, static_condition, time_enc, temp_enc], dim=-1)

        # Shared MLP (applied per-residue via batched linear)
        h = self.shared_net(x)

        # Mean and log_sigma
        mean = self.mean_head(h)
        log_sigma = self.log_sigma_head(h)

        # Clamp log_sigma for stability
        log_sigma = torch.clamp(log_sigma, min=-5.0, max=2.0)

        if residue_mask is not None:
            mean = mean * residue_mask.unsqueeze(-1).float()
            log_sigma = log_sigma * residue_mask.unsqueeze(-1).float()

        return mean, log_sigma

    def sample(
        self,
        h_history: torch.Tensor,
        static_condition: torch.Tensor,
        tau_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """Sample Q predictions at horizon tau.

        Args:
            Same as forward plus:
            num_samples: number of independent samples (defaults to config)

        Returns:
            [B, S, L, latent_dim] sampled Q values (S=samples)
        """
        mean, log_sigma = self.forward(
            h_history, static_condition, tau_ps, temperature_k, residue_mask
        )

        n_samples = num_samples or self.config.num_samples
        sigma = torch.exp(log_sigma) * self.config.sample_temperature

        # Sample multiple times
        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mean)
            q_sample = mean + sigma * eps
            samples.append(q_sample)

        return torch.stack(samples, dim=1)  # [B, S, L, D]


__all__ = [
    "DirectHeadConfig",
    "DirectEndpointHead",
]