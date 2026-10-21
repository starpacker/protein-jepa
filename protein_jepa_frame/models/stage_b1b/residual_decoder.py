"""Optional local-frame residual CA readout.

The main B1b trainer shares the frozen B1a oracle. This auxiliary decoder emits
invariant local displacements and rotates them through observed anchor frames.
A zero residual reproduces persistence; this does not guarantee improvement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ResidualDecoderConfig:
    latent_dim: int = 128
    hidden: int = 256
    depth: int = 3
    n_neighbors: int = 0          # 0 = pure per-residue; >0 adds local context pooling
    max_update: float = 12.0      # clamp on |dx| (A) to keep it a bounded residual


class ResidualCADecoder(nn.Module):
    """Per-residue displacement head with optional sequence-local context.

    forward(H_future, H_context, X_context, residue_mask) -> dict with
      'ca_pred'  : [B, L, 3]  predicted future CA (context Kabsch frame)
      'dx'       : [B, L, 3]  the residual displacement
    X_context is assumed already centered (mean over valid residues removed) and
    expressed in the frame we score against.
    """

    def __init__(self, cfg: ResidualDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        in_dim = 3 * d  # [H_future, H_context, delta]
        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden), nn.GELU()]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden, 3)
        # Small-norm init: lets gradients flow through the trunk from step 1,
        # while keeping initial dx small (~0.01 A) so we start near persistence.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        H_future: torch.Tensor,     # [B, L, dH]
        H_context: torch.Tensor,    # [B, L, dH]
        X_context: torch.Tensor,    # [B, L, 3]  centered CA of the anchor frame
        residue_mask: torch.Tensor, # [B, L]
        R_anchor: torch.Tensor | None = None, # [B,L,3,3], required observed basis
    ) -> dict[str, torch.Tensor]:
        if R_anchor is None:
            raise ValueError("ResidualCADecoder requires observed R_anchor for SE(3) equivariance")
        if R_anchor.shape != (*X_context.shape[:-1], 3, 3):
            raise ValueError("R_anchor must have shape [B,L,3,3]")
        delta = H_future - H_context
        feats = torch.cat([H_future, H_context, delta], dim=-1)
        x = self.trunk(feats)
        dx = self.head(x)
        # bounded residual
        m = self.cfg.max_update
        dx = m * torch.tanh(dx / max(m, 1e-6))
        mask = residue_mask.unsqueeze(-1).to(dx.dtype)
        dx_local = dx * mask
        dx = (R_anchor @ dx_local.unsqueeze(-1)).squeeze(-1)
        ca_pred = (X_context + dx) * mask
        return {"ca_pred": ca_pred, "dx": dx, "dx_local": dx_local}
