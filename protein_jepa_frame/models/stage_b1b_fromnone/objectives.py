"""Objectives for the from-none Stage B pipeline.

The career of these losses is to force a *from-scratch* encoder to produce an H
whose inter-frame change is geometrically decodable, while the H/S kernel learns
an autonomous multi-lag law.  Three terms:

  energy_score        : distributional term on the rollout H path (proper ES).
  ca_displacement     : differentiable Kabsch-aligned CA point loss (decodability).
  physical_energy     : energy score on the decoded backbone physical features.

The CA term is the one Stage A never had; without it a same-frame invariant H
optimises the ES by moving in a direction that is not aligned with real motion
(measured R^2 < 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Differentiable Kabsch alignment (row-vector convention: x @ R + t)
# -----------------------------------------------------------------------------

def kabsch_rotation(pred: Tensor, true: Tensor) -> tuple[Tensor, Tensor]:
    """Least-squares rotation/translation mapping ``pred`` onto ``true``.

    ``pred`` / ``true``: [..., N, 3] (last two dims are residues and xyz; the
    leading dims are batch).  Returns ``(R, t)`` with ``x @ R + t`` convention,
    differentiable through the SVD.  Fully batched (no python loop).
    """
    pmean = pred.mean(-2, keepdim=True)
    qmean = true.mean(-2, keepdim=True)
    pc = pred - pmean
    qc = true - qmean
    H = pc.transpose(-1, -2) @ qc                     # [..., 3, 3]
    U, _, Vh = torch.linalg.svd(H)
    sgn = torch.sign(torch.det(U @ Vh))               # [...,] proper-rotation sign
    D = torch.diag_embed(torch.stack([
        torch.ones_like(sgn), torch.ones_like(sgn), sgn], dim=-1))
    R = U @ D @ Vh                                    # maps centered pred -> centered true
    # Full affine map so pred @ R + t == true (translation absorbs both means).
    t = qmean - pmean @ R
    return R, t


def aligned_ca_rmsd(pred_ca: Tensor, true_ca: Tensor, mask: Tensor) -> Tensor:
    """Batched Kabsch CA RMSD (A); differentiable through the rotation.

    ``pred_ca``/``true_ca``: [B, L, 3]; ``mask``: [B, L].  Uses a masked
    Kabsch: invalid residues are replaced by the masked centroid so they
    contribute exactly zero error and do not bias the rotation.
    """
    m = mask.unsqueeze(-1).to(pred_ca.dtype)                 # [B, L, 1]
    valid = mask.sum(-1)                                     # [B]
    if int((valid < 3).sum()) == pred_ca.shape[0]:
        return pred_ca.new_zeros(())
    # Fill invalid residues with the per-sample valid centroid so they cancel.
    def fill(x):
        c = (x * m).sum(-2, keepdim=True) / m.sum(-2, keepdim=True).clamp_min(1.0)
        return x * m + c * (1.0 - m)
    p, q = fill(pred_ca), fill(true_ca)
    R, t = kabsch_rotation(p, q)
    aligned = p @ R + t
    diff = (aligned - q) * m
    # squared error is already summed over xyz; the mean is over valid residues.
    per_b = diff.square().sum(-1).sum(-1) / m.sum(-2).squeeze(-1).clamp_min(1e-8)
    good = (valid >= 3)
    if not bool(good.any()):
        return pred_ca.new_zeros(())
    return per_b[good].clamp_min(1e-8).sqrt().mean()


def persistence_ca_rmsd(true_ca: Tensor, anchor_ca: Tensor, mask: Tensor) -> Tensor:
    """Kabsch CA RMSD of the no-op forecast (pred = anchor), for skill logging."""
    return aligned_ca_rmsd(anchor_ca, true_ca, mask)


# -----------------------------------------------------------------------------
# Energy score (proper scoring rule, as in stage_b1b.objectives)
# -----------------------------------------------------------------------------

def energy_score(samples: Tensor, target: Tensor) -> Tensor:
    """ES_hat = (1/M) sum_m ||v_m - v*|| - (1/(2M(M-1))) sum_{m!=m'} ||v_m - v_m'||.

    ``samples``: [M, D]; ``target``: [D].
    """
    if samples.dim() != 2:
        raise ValueError(f"samples must be [M, D], got {tuple(samples.shape)}")
    M = samples.shape[0]
    if M < 2:
        raise ValueError("energy_score needs M >= 2 samples")
    d_target = (samples - target.unsqueeze(0)).norm(dim=-1)
    if M < 2:
        return d_target.mean()
    diff = samples.unsqueeze(0) - samples.unsqueeze(1)
    dist = diff.norm(dim=-1)
    off = dist.sum() / (M * (M - 1)) if M > 1 else d_target.new_zeros(())
    return d_target.mean() - 0.5 * off


# -----------------------------------------------------------------------------
# Composite
# -----------------------------------------------------------------------------

@dataclass
class FromNoneWeights:
    """Loss weights for the from-none Stage B objective.

    ``ca_weight`` dominates: it is the decodability term that shapes the encoder.
    ``es_weight`` keeps the rollout distributional; ``phys_weight`` ties the
    decoded backbone to physical features.
    """
    es_weight: float = 0.1
    ca_weight: float = 1.0
    phys_weight: float = 0.1
    anchor_weight: float = 0.0    # optional same-frame H->anchor-CA probe (0 = off)
    variance_weight: float = 0.0  # optional temporal-variance floor (0 = off)

    def validate(self) -> None:
        for k, v in self.__dict__.items():
            if not (v >= 0.0 and v < float("inf")):
                raise ValueError(f"FromNoneWeights.{k} must be finite and non-negative, got {v}")


def temporal_variance_floor(H_seq: Tensor, residue_mask: Tensor, floor: float) -> Tensor:
    """Penalty pushing per-(residue,channel) temporal std toward ``floor``.

    ``H_seq``: [B, T, L, dH].  Returns a scalar hinge penalty
    ``mean_relu(floor - std_t)``.  Prevents H from collapsing to a constant,
    which is the trivial way to make the ES term small.
    """
    if floor <= 0:
        return H_seq.new_zeros(())
    std = H_seq.std(dim=1)                          # [B, L, dH]
    mask = residue_mask.unsqueeze(-1).float()        # [B, L, 1]
    pen = F.relu(floor - std) * mask
    return pen.sum() / mask.sum().clamp_min(1.0) / H_seq.shape[-1]


@dataclass
class FromNoneLossComponents:
    total: Tensor
    es: Tensor
    ca: Tensor
    phys: Tensor
    anchor: Tensor
    variance: Tensor
    persist: Tensor

    def as_loggable_dict(self) -> dict[str, float]:
        d = {
            "loss/total": float(self.total.detach()),
            "loss/es": float(self.es.detach()),
            "loss/ca": float(self.ca.detach()),
            "loss/phys": float(self.phys.detach()),
            "loss/anchor": float(self.anchor.detach()),
            "loss/variance": float(self.variance.detach()),
        }
        if torch.isfinite(self.persist):
            d["loss/persist_ref"] = float(self.persist.detach())
        return d
