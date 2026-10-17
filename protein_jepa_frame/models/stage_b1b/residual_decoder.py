"""Residual CA decoder for Stage B1b end-to-end RMSD training.

Motivation
----------
The Stage-A encoder H is SE(3)-invariant, so it cannot encode the absolute
context->target rigid-body pose.  Any decoder that tries to place atoms in an
absolute frame from H alone is doomed (this is why the B1a oracle decoder with
anchor_mode=context stalled at ca_mse ~150 A^2).

Instead we decode a *residual displacement* relative to the context structure:

    dx_i = g( H_future_i , H_context_i , (H_future_i - H_context_i) )
    X_hat = X_context_centered + dx          (in the context Kabsch frame)

Because the prediction is anchored on the real context coordinates, a
zero-output decoder reproduces persistence exactly.  The decoder can therefore
only *improve* on persistence; it is structurally incapable of being worse once
dx is regularised toward 0 at init.  The learning signal is the Kabsch RMSD to
the true future structure.

This is deliberately CA-only (one point per residue): it targets the metric we
care about (CA RMSD vs persistence) without the full backbone-frame machinery
that made B1a brittle.
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
        # Zero-init blocks gradients entirely because dL/d(trunk_out) = head_W^T * dL/d(head_out),
        # and zero head_W means zero gradient into the trunk forever.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        H_future: torch.Tensor,     # [B, L, dH]
        H_context: torch.Tensor,    # [B, L, dH]
        X_context: torch.Tensor,    # [B, L, 3]  centered CA of the anchor frame
        residue_mask: torch.Tensor, # [B, L]
    ) -> dict[str, torch.Tensor]:
        delta = H_future - H_context
        feats = torch.cat([H_future, H_context, delta], dim=-1)
        x = self.trunk(feats)
        dx = self.head(x)
        # bounded residual
        m = self.cfg.max_update
        dx = m * torch.tanh(dx / max(m, 1e-6))
        mask = residue_mask.unsqueeze(-1).to(dx.dtype)
        dx = dx * mask
        ca_pred = (X_context + dx) * mask
        return {"ca_pred": ca_pred, "dx": dx}
