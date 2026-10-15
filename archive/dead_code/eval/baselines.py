"""CVPR baseline models: B0 persistence, B1 causal transformer, B2 stochastic predictor.

Follows Section 5.8 of the CVPR-Nature plan:
  B0: Persistence — current latent state held constant
  B1: Causal Transformer latent world model (DINO-world/JEPA adaptation)
  B2: Conditional stochastic predictor (flow matching)

All baselines receive identical inputs: frozen H_last [B,L,128], Q_init [B,L,8],
static condition, temperature, horizon time. They predict Q at target horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── B0: Persistence baseline ───────────────────────────────────────

class PersistenceBaseline(nn.Module):
    """Predicts Q_{t+tau} = Q_t (identity, no change).

    The simplest possible baseline: the latent state does not evolve.
    Used to establish the floor that any learned dynamics must beat.
    """

    def __init__(self, state_dim: int = 8):
        super().__init__()
        self.state_dim = state_dim

    def forward(
        self, q: torch.Tensor,
        h_frozen: torch.Tensor | None = None,
        tau_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return q unchanged regardless of horizon or conditions."""
        return q.clone()

    def rollout(
        self, q_init: torch.Tensor,
        h_frozen: torch.Tensor | None = None,
        n_steps: int = 1,
        step_ps: float = 100.0,
        temperature_k: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return identical q for all steps."""
        B, L, D = q_init.shape
        return q_init.unsqueeze(1).expand(B, n_steps + 1, L, D).contiguous()


# ── B1: Causal Transformer world model ────────────────────────────

@dataclass(frozen=True)
class TransformerWMConfig:
    """Configuration for B1 causal transformer world model."""
    state_dim: int = 8               # Q dimension per residue
    h_dim: int = 128                 # frozen H dimension
    d_model: int = 128               # transformer hidden dim
    n_heads: int = 4                 # attention heads
    n_layers: int = 3                # transformer layers
    n_history: int = 4               # context frames (if multi-step)
    dropout: float = 0.1
    causal: bool = True              # causal masking
    time_conditioning: bool = True   # embed horizon time


class CausalTransformerWorldModel(nn.Module):
    """B1: Causal transformer for latent-state world modeling.

    Adapts DINO-world/JEPA-style latent prediction to protein residues.
    Input: Q_init + H_frozen + time_condition → Q_future.
    Operates per-residue with shared transformer across residues.

    Two modes:
      1. Direct: predict Q at horizon tau from current state
      2. Recursive: autoregressively predict step-by-step
    """

    def __init__(self, config: TransformerWMConfig | None = None):
        super().__init__()
        self.cfg = config or TransformerWMConfig()
        c = self.cfg

        # Input projection: Q [B,L,8] + H [B,L,128] + time_enc [B,L,32] → d_model
        self.time_enc = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32),
        )
        input_dim = c.state_dim + c.h_dim + 32
        self.input_proj = nn.Linear(input_dim, c.d_model)

        # Positional encoding (shared per-residue, not learned per-position)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, c.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.n_heads, dropout=c.dropout,
            dim_feedforward=c.d_model * 4, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.n_layers)

        # Output head: d_model → state_dim
        self.output_head = nn.Sequential(
            nn.Linear(c.d_model, c.d_model), nn.SiLU(),
            nn.Linear(c.d_model, c.state_dim),
        )

    def _encode_time(self, tau_ps: torch.Tensor, B: int, L: int) -> torch.Tensor:
        """Encode horizon time to [B, L, 32]."""
        if tau_ps.dim() == 1:
            tau_ps = tau_ps.unsqueeze(-1)
        tau_log = torch.log(tau_ps.clamp(min=1.0).float())
        return self.time_enc(tau_log).unsqueeze(1).expand(B, L, -1)

    def forward(
        self,
        q: torch.Tensor,
        h_frozen: torch.Tensor | None = None,
        tau_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict Q at horizon tau.

        Args:
            q: [B, L, state_dim] current Q
            h_frozen: [B, L, h_dim] frozen H features
            tau_ps: [B] horizon in picoseconds
            residue_mask: [B, L]

        Returns:
            [B, L, state_dim] predicted Q at tau
        """
        B, L = q.shape[:2]
        device = q.device

        if h_frozen is None:
            h_frozen = torch.zeros(B, L, self.cfg.h_dim, device=device)

        if tau_ps is None:
            tau_ps = torch.ones(B, device=device) * 1000.0

        time_enc = self._encode_time(tau_ps, B, L)
        x = torch.cat([q, h_frozen, time_enc], dim=-1)
        x = self.input_proj(x) + self.pos_encoding[:, :, :]
        x = x.expand(B, L, self.cfg.d_model)

        if residue_mask is not None:
            mask_1d = ~residue_mask
            # Per-residue masking: use float mask
            x = x * residue_mask.unsqueeze(-1).float()

        h = self.encoder(x)
        q_pred = self.output_head(h)

        if residue_mask is not None:
            q_pred = q_pred * residue_mask.unsqueeze(-1).float()

        return q_pred

    def rollout(
        self,
        q_init: torch.Tensor,
        h_frozen: torch.Tensor | None = None,
        n_steps: int = 1,
        step_ps: float = 100.0,
        temperature_k: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive rollout for n_steps.

        Returns [B, n_steps+1, L, state_dim]: states at t0, t1, ..., t_n.
        """
        B, L = q_init.shape[:2]
        device = q_init.device
        D = q_init.shape[-1]

        traj = [q_init.clone()]
        q = q_init.clone()

        for step in range(n_steps):
            tau = torch.full((B,), step_ps, device=device)
            q = self.forward(q, h_frozen, tau, temperature_k, residue_mask)
            traj.append(q)

        return torch.stack(traj, dim=1)


# ── B2: Conditional stochastic predictor ─────────────────────────

@dataclass(frozen=True)
class StochasticPredictorConfig:
    """Configuration for B2 conditional flow matching predictor."""
    state_dim: int = 8
    h_dim: int = 128
    hidden_dim: int = 128
    n_layers: int = 4
    time_embed_dim: int = 32


class ConditionalFlowMatchingPredictor(nn.Module):
    """B2: Conditional flow matching for latent state prediction.

    Learns p(Q_{t+tau} | Q_t, H, c, T, tau) as a continuous normalizing flow.
    Uses the DirectEndpointHead architecture augmented with sample-based NLL.

    For speed, we implement the simplest effective form: a Gaussian distribution
    with learned mean and diagonal covariance conditioned on all inputs.
    """

    def __init__(self, config: StochasticPredictorConfig | None = None):
        super().__init__()
        self.cfg = config or StochasticPredictorConfig()
        c = self.cfg

        self.time_enc = nn.Sequential(
            nn.Linear(1, c.time_embed_dim), nn.SiLU(),
            nn.Linear(c.time_embed_dim, c.time_embed_dim),
        )

        input_dim = c.state_dim + c.h_dim + c.time_embed_dim
        layers = []
        in_dim = input_dim
        for _ in range(c.n_layers):
            layers.extend([nn.Linear(in_dim, c.hidden_dim), nn.SiLU()])
            in_dim = c.hidden_dim
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, c.state_dim)
        self.logvar_head = nn.Linear(in_dim, c.state_dim)
        self.logvar_head.weight.data.zero_()
        self.logvar_head.bias.data.fill_(-2.0)

    def forward(
        self,
        q: torch.Tensor,
        h_frozen: torch.Tensor | None = None,
        tau_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, logvar) for predicted Q distribution."""
        B, L = q.shape[:2]
        device = q.device

        if h_frozen is None:
            h_frozen = torch.zeros(B, L, self.cfg.h_dim, device=device)
        if tau_ps is None:
            tau_ps = torch.ones(B, device=device) * 1000.0
        if tau_ps.dim() == 1:
            tau_ps = tau_ps.unsqueeze(-1)

        t_enc = self.time_enc(torch.log(tau_ps.clamp(min=1.0)))
        t_enc = t_enc.unsqueeze(1).expand(B, L, -1)

        x = torch.cat([q, h_frozen, t_enc], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-5.0, max=2.0)

        if residue_mask is not None:
            m = residue_mask.unsqueeze(-1).float()
            mean = mean * m
            logvar = logvar * m

        return mean, logvar

    def sample(
        self, q: torch.Tensor, h_frozen: torch.Tensor, tau_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Sample Q predictions. Returns [B, S, L, D] for S samples."""
        mean, logvar = self.forward(q, h_frozen, tau_ps, temperature_k, residue_mask)
        samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mean)
            s = mean + torch.exp(logvar * 0.5) * eps
            samples.append(s)
        return torch.stack(samples, dim=1)

    def rollout(
        self, q_init: torch.Tensor, n_steps: int = 1,
        step_ps: float = 100.0, **kwargs,
    ) -> torch.Tensor:
        """Autoregressive rollout using mean prediction."""
        traj = [q_init.clone()]
        q = q_init.clone()
        for _ in range(n_steps):
            tau = torch.full((q_init.shape[0],), step_ps, device=q_init.device)
            q, _ = self.forward(q, kwargs.get('h_frozen'), tau,
                                kwargs.get('temperature_k'), kwargs.get('residue_mask'))
            traj.append(q)
        return torch.stack(traj, dim=1)


__all__ = [
    "PersistenceBaseline",
    "TransformerWMConfig",
    "CausalTransformerWorldModel",
    "StochasticPredictorConfig",
    "ConditionalFlowMatchingPredictor",
]
