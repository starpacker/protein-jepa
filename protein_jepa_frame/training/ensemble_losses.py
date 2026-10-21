"""T4 — Ensemble-level losses (NEW, 2026-10-18).

Matches the MODEL ensemble statistics (from `models/ensemble_producer.py`) to the
TRUE future-window statistics (from `training/ensemble_targets.py`).  All terms
are differentiable w.r.t. the model's per-residue Gaussian H samples, hence flow
gradient into the trainable `GaussianHHead` while the encoder/decoder stay frozen.

    L_rmsf    : per-residue flexibility-profile mismatch (masked L1 on aligned RMSF)
    L_contact : contact-PROBABILITY map mismatch (masked L1 on P(d<cutoff))
    L_dist    : per-pair internal-distance distribution mismatch (soft-EMD)

This file does NOT touch the precise-coordinate objectives
(`stage_b1a/objectives.py` ca_mse/fape); it is an additive, ensemble-level family.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from evaluation.ensemble_metrics import align_ensemble, rmsf_profile, contact_probability_map

__all__ = ["EnsembleLossWeights", "EnsembleLossComponents", "compute_ensemble_loss"]


@dataclass
class EnsembleLossWeights:
    rmsf_weight: float = 1.0
    contact_weight: float = 1.0
    dist_weight: float = 1.0
    pos_weight: float = 0.0          # D2 (2026-10-19): atom-position guide
    eps: float = 1e-6

    def validate(self) -> None:
        for name, val in self.__dict__.items():
            if isinstance(val, float) and not (val >= 0.0 and val < float("inf")):
                raise ValueError(f"EnsembleLossWeights.{name} must be finite non-negative, got {val}")


@dataclass
class EnsembleLossComponents:
    total: torch.Tensor
    rmsf: torch.Tensor
    contact: torch.Tensor
    dist: torch.Tensor
    pos: torch.Tensor | None = None

    def as_loggable_dict(self) -> dict[str, float]:
        d = {
            "ens/total": float(self.total.detach()),
            "ens/rmsf": float(self.rmsf.detach()),
            "ens/contact": float(self.contact.detach()),
            "ens/dist": float(self.dist.detach()),
        }
        if self.pos is not None:
            d["ens/pos"] = float(self.pos.detach())
        return d


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dims) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _rmsf_loss(pred_ca: torch.Tensor, true_rmsf: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """[B,n,L,3] pred ensemble vs [B,L] true RMSF.  Masked L1; per-protein L2 norm."""
    B, n, L, _ = pred_ca.shape
    pred_r = torch.stack([rmsf_profile(align_ensemble(pred_ca[b], mask[b]), mask[b])
                          for b in range(B)], dim=0)                # [B,L]
    m = mask.to(pred_r.dtype)
    diff = (pred_r - true_rmsf).abs() * m
    return diff.sum() / m.sum().clamp_min(1.0)


def _contact_loss(pred_ca: torch.Tensor, true_contact: torch.Tensor,
                  mask: torch.Tensor, cutoff: float) -> torch.Tensor:
    """[B,n,L,3] pred vs [B,L,L] true contact probability map."""
    B, n, L, _ = pred_ca.shape
    pred_c = torch.stack([contact_probability_map(pred_ca[b], mask[b], cutoff)
                          for b in range(B)], dim=0)                # [B,L,L]
    m = mask.to(pred_c.dtype)[:, :, None] * mask.to(pred_c.dtype)[:, None, :]
    diff = (pred_c - true_contact).abs() * m
    return diff.sum() / m.sum().clamp_min(1.0)


def _dist_loss(pred_hist: torch.Tensor, true_hist: torch.Tensor,
               bin_w: float) -> torch.Tensor:
    """[B,P,bins] vs [B,P,bins] soft-EMD (cumulative difference) averaged."""
    emd = (pred_hist - true_hist).cumsum(dim=-1).abs().sum(dim=-1) * bin_w  # [B,P]
    return emd.mean()


def _position_loss(pred_ca: torch.Tensor, true_ca: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """D2 atom-position guide: Kabsch-aligned CA MSE (Å²) of a deterministic
    decode against the TRUE future frame.

    This is the AUXILIARY metric the user asked for: it anchors the ensemble to
    a real absolute position so the Gaussian head cannot win the ensemble
    statistics by emitting a diffuse cloud that ignores the true fold.  It reuses
    the same masked Kabsch convention as `stage_b1a/objectives.py` but is
    implemented here (read-only w.r.t. the precise-coordinate pipeline).
    """
    if pred_ca.shape != true_ca.shape or pred_ca.shape[-1] != 3:
        raise ValueError("pred_ca and true_ca must both be [B, L, 3]")
    B, L, _ = pred_ca.shape
    m = mask.to(pred_ca.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)          # [B,1]
    pc = (pred_ca * m.unsqueeze(-1)).sum(1, keepdim=True) / denom.unsqueeze(-1)
    tc = (true_ca * m.unsqueeze(-1)).sum(1, keepdim=True) / denom.unsqueeze(-1)
    p0 = (pred_ca - pc) * m.unsqueeze(-1)
    t0 = (true_ca - tc) * m.unsqueeze(-1)
    total = pred_ca.new_zeros(())
    for b in range(B):
        H = p0[b].T @ t0[b]                                    # [3,3]
        U, S, Vh = torch.linalg.svd(H)
        d = torch.det(U @ Vh).sign()
        D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
        R = U @ D @ Vh
        diff = p0[b] @ R - t0[b]
        total = total + (diff.square().sum(-1) * m[b]).sum() / denom[b, 0].clamp(min=1.0)
    return total / max(B, 1)


def compute_ensemble_loss(
    pred_ca_samples: torch.Tensor,       # [B, n, L, 3]
    true_rmsf: torch.Tensor,             # [B, L]
    true_contact: torch.Tensor,          # [B, L, L]
    pred_dist_hist: torch.Tensor,        # [B, P, bins]
    true_dist_hist: torch.Tensor,        # [B, P, bins]
    mask: torch.Tensor,                  # [B, L]
    dist_bin_width: float,
    weights: EnsembleLossWeights,
    cutoff: float = 8.0,
    true_ca_pos: torch.Tensor | None = None,      # [B, L, 3] future CA (D2 guide)
    pred_ca_pos: torch.Tensor | None = None,      # [B, L, 3] deterministic decode
) -> EnsembleLossComponents:
    l_rmsf = _rmsf_loss(pred_ca_samples, true_rmsf, mask)
    l_contact = _contact_loss(pred_ca_samples, true_contact, mask, cutoff)
    l_dist = _dist_loss(pred_dist_hist, true_dist_hist, dist_bin_width)
    l_pos = None
    if weights.pos_weight > 0.0:
        if true_ca_pos is None or pred_ca_pos is None:
            raise ValueError("pos_weight>0 requires true_ca_pos and pred_ca_pos")
        l_pos = _position_loss(pred_ca_pos, true_ca_pos, mask)
    total = (weights.rmsf_weight * l_rmsf
             + weights.contact_weight * l_contact
             + weights.dist_weight * l_dist)
    if l_pos is not None:
        total = total + weights.pos_weight * l_pos
    return EnsembleLossComponents(total=total, rmsf=l_rmsf, contact=l_contact,
                                  dist=l_dist, pos=l_pos)
