"""Lightweight CA-displacement decoder for the from-none Stage B pipeline.

Contract
--------
The decoder turns a *state* (the dynamics H, optionally plus the bath S) into a
per-residue CA update expressed in the OBSERVED anchor rigid frame.  It never
sees future coordinates, and a zero update reproduces persistence exactly:

    X_hat_tau = X_anchor + R_anchor @ dx_local(H_tau, H_anchor [, S_tau])

where ``R_anchor`` is the per-residue backbone frame of the last observed frame
(``RigidBodyFrame.from_backbone``) and ``dx_local`` is predicted from the
invariant H/S state alone.  Predicting the update in the *local* frame keeps the
whole map SE(3)-equivariant: a global rotation of the input rotates both the
anchor frame and the output displacement consistently.

Why a point loss on dx (not just an energy score)
-------------------------------------------------
The diagnosis (B1B_DIAGNOSIS_AND_PLAN.md, DIAGNOSIS_H_READOUT_20261018.md) is
that a same-frame invariant H has no *aligned* difference: scalar energy scores
only constrain the distribution of H, not the geometry of ``H_future - H_anchor``.
A direct differentiable CA-displacement regression is what forces that
difference to be geometrically decodable.  This decoder therefore reports both a
bounded ``dx`` and the assembled ``ca_pred``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FromNoneConfig:
    """Sizing for the from-none CA decoder head."""

    latent_dim: int = 128          # dH
    bath_dim: int = 0              # dS appended to the state when > 0
    hidden: int = 256
    depth: int = 3
    n_neighbors: int = 0           # reserved: >0 would add sequence-local pooling
    max_update: float = 12.0       # clamp on |dx_local| (A): bounded residual
    predict_log_scale: bool = False

    @property
    def state_dim(self) -> int:
        return self.latent_dim * 3 + self.bath_dim


class SimpleCADecoder(nn.Module):
    """Per-residue MLP: (H_tau, H_anchor, dH, [S_tau]) -> local CA displacement.

    ``forward`` returns a dict with:
      'dx_local' : [B, L, 3]  update in the anchor local frame
      'dx'       : [B, L, 3]  the same update rotated to the global frame
      'ca_pred'  : [B, L, 3]  X_anchor + dx  (global, in the anchor gauge)
    """

    def __init__(self, cfg: FromNoneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.state_dim
        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden), nn.GELU()]
        for _ in range(max(cfg.depth - 1, 0)):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        out_dim = 4 if cfg.predict_log_scale else 3
        self.head = nn.Linear(cfg.hidden, out_dim)
        # Small-norm init: dx ~ 0 at step 0, so training starts at persistence and
        # the trunk still receives gradient (see residual_decoder.py rationale).
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        H_future: torch.Tensor,        # [B, L, dH]
        H_context: torch.Tensor,       # [B, L, dH]
        X_context: torch.Tensor,       # [B, L, 3]  anchor CA (global gauge)
        residue_mask: torch.Tensor,    # [B, L]
        R_anchor: torch.Tensor,        # [B, L, 3, 3]  observed anchor frame
        S_future: torch.Tensor | None = None,  # [B, L, dS] optional bath
    ) -> dict[str, torch.Tensor]:
        if R_anchor is None:
            raise ValueError("SimpleCADecoder requires the observed R_anchor")
        expected = (*X_context.shape[:-1], 3, 3)
        if tuple(R_anchor.shape) != tuple(expected):
            raise ValueError(f"R_anchor must have shape {expected}, got {tuple(R_anchor.shape)}")

        feats = [H_future, H_context, H_future - H_context]
        if self.cfg.bath_dim > 0:
            if S_future is None:
                raise ValueError("bath_dim > 0 but S_future was not provided")
            feats.append(S_future)
        x = self.trunk(torch.cat(feats, dim=-1))
        raw = self.head(x)

        mask = residue_mask.unsqueeze(-1).to(raw.dtype)
        if self.cfg.predict_log_scale:
            dx_local, log_s = raw[..., :3], raw[..., 3:]
            # bounded residual keeps |dx| <= max_update so a bad step cannot
            # teleport a residue; log_s is a per-residue per-axis uncertainty
            # (unused by the point loss, exposed for later probabilistic use).
            m = self.cfg.max_update
            dx_local = m * torch.tanh(dx_local / max(m, 1e-6))
        else:
            m = self.cfg.max_update
            dx_local = m * torch.tanh(raw / max(m, 1e-6))
            log_s = None

        dx_local = dx_local * mask
        dx = (R_anchor @ dx_local.unsqueeze(-1)).squeeze(-1) * mask
        ca_pred = (X_context + dx) * mask
        out = {"dx_local": dx_local, "dx": dx, "ca_pred": ca_pred}
        if log_s is not None:
            out["log_scale"] = log_s * mask
        return out
