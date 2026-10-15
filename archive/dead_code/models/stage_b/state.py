"""P1.1: Canonical latent state z_t = (q_t, m_t, e_t).

Defines the world-model state as a typed container:
  q_t: [B, L, 8]  — internal conformational state (SE(3)-invariant)
  m_t: [B, L, 16] — Mori/finite-memory bath state
  e_t: [B, 8]      — environment state (temperature-dependent equilibrium info)
"""

from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class LatentState:
    """Canonical latent state for protein dynamics world model.
    z_t = (q_t, m_t, e_t)"""
    q: torch.Tensor   # [B, L, 8]
    m: torch.Tensor   # [B, L, 16]
    e: torch.Tensor   # [B, 8]

    @property
    def batch_size(self) -> int: return self.q.shape[0]
    @property
    def n_residues(self) -> int: return self.q.shape[1]
    @property
    def device(self): return self.q.device

    def clone(self): return LatentState(self.q.clone(), self.m.clone(), self.e.clone())
    def detach(self): return LatentState(self.q.detach(), self.m.detach(), self.e.detach())
    def to(self, d): return LatentState(self.q.to(d), self.m.to(d), self.e.to(d))

    def apply_mask(self, mask):
        mf = mask.unsqueeze(-1).float()
        return LatentState(self.q * mf, self.m * mf, self.e)

    @staticmethod
    def zeros(B, L, state_dim=8, bath_dim=16, env_dim=8, **kw):
        return LatentState(
            torch.zeros(B, L, state_dim, **kw),
            torch.zeros(B, L, bath_dim, **kw),
            torch.zeros(B, env_dim, **kw),
        )


@dataclass
class TransitionResult:
    """Result of a single transition step."""
    next_state: LatentState
    forces: torch.Tensor | None = None
    diffusion: torch.Tensor | None = None
    energy_total: torch.Tensor | None = None
    diagnostics: dict | None = None


__all__ = ["LatentState", "TransitionResult"]
