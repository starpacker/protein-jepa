"""Stage B1b objectives: Kalman innovation NLL + multi-step energy score.

Wiring per STAGE_ABC_REDESIGN_PLAN.md 2.4-2.5:

    L_B1_full = mean_valid_NLL + lambda_ES * ES

  NLL: StageB1b.kalman_nll(...) summed over steps and batch, normalised here by
       (valid_residues * dH * n_steps) so that it is a per-dimension mean.
  ES:  autonomous multi-step conditional energy score (2.5)

       ES_hat = (1/M) sum_m ||v^(m) - v*||_2
                - (1/(2M(M-1))) sum_{m!=m'} ||v^(m) - v^(m')||_2

       v = Phi(H_{1:N}) is a fixed-standardised multi-time path feature vector
       extracted by the (frozen) B1a decoder/probe.  ES is used only as a small
       regulariser after an NLL warm-up; lambda_ES ramps from 0.

Forbidden (plan 8.2) and NOT used here:
  - force floor / force normalisation
  - softplus(NLL) truncation  (raw continuous NLL may be negative)
  - per-step detach inside the rollout
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------------

@dataclass
class StageB1bLossWeights:
    """Scalar loss weights for Stage B1b.

    warmup_steps > 0 keeps the run NLL-only (lambda_ES effectively 0) until the
    Kalman recursion and innovation statistics are stable (2.5).
    """
    nll_weight: float = 1.0
    es_weight: float = 0.1          # lambda_ES; ramped in after warm-up
    es_warmup_steps: int = 2000     # steps before ES is enabled
    es_num_samples: int = 4         # M, independent rollout samples
    es_horizon_steps: int = 8       # N steps scored by ES

    def lambda_es_at(self, step: int) -> float:
        """Ramp lambda_ES linearly over the warm-up, then hold."""
        if self.es_warmup_steps <= 0:
            return self.es_weight
        frac = min(1.0, max(0.0, step / float(self.es_warmup_steps)))
        return self.es_weight * frac

    def validate(self) -> None:
        for name, val in self.__dict__.items():
            if isinstance(val, float) and not (val >= 0.0 and val < float("inf")):
                raise ValueError(f"StageB1bLossWeights.{name} must be finite and non-negative, got {val}")


# -----------------------------------------------------------------------------
# Energy score
# -----------------------------------------------------------------------------

def kabsch_rmsd_loss(
    pred_ca: Tensor, true_ca: Tensor, residue_mask: Tensor,
) -> Tensor:
    """Mean per-sample Kabsch CA RMSD (A), differentiable through the alignment.

    Aligns pred onto true per batch element using masked centroids and the SVD
    rotation, then returns the mean RMSD over the batch.  Gradients flow through
    the rotation (SVD backward is supported by torch.linalg).

    Args:
        pred_ca: [B, L, 3]  predicted CA coordinates.
        true_ca: [B, L, 3]  ground-truth CA coordinates.
        residue_mask: [B, L]  valid-residue mask.
    """
    B = pred_ca.shape[0]
    rmsds = []
    for b in range(B):
        m = residue_mask[b].bool()
        n = int(m.sum())
        if n < 3:
            continue
        P = pred_ca[b][m]              # [n,3]
        Q = true_ca[b][m]
        Pc = P - P.mean(0, keepdim=True)
        Qc = Q - Q.mean(0, keepdim=True)
        H = Pc.transpose(0, 1) @ Qc    # [3,3]
        U, S, Vh = torch.linalg.svd(H)
        d = torch.sign(torch.det(U @ Vh))
        D = torch.diag(torch.stack([
            torch.ones((), device=P.device, dtype=P.dtype),
            torch.ones((), device=P.device, dtype=P.dtype),
            d,
        ]))
        # Optimal rotation for min ||Pc @ R - Qc|| with H = Pc^T Qc = U S Vh
        # is R = U diag(1,1,d) Vh (verified against scipy align_vectors).
        R = U @ D @ Vh
        aligned = Pc @ R
        rmsd = ((aligned - Qc) ** 2).sum(-1).mean().clamp_min(1e-8).sqrt()
        rmsds.append(rmsd)
    if not rmsds:
        return pred_ca.new_zeros(())
    return torch.stack(rmsds).mean()


def energy_score(samples: Tensor, target: Tensor) -> Tensor:
    """Proper energy score for a set of M forecast samples (2.5).

    Args:
        samples: [M, D]  M independent feature-vector samples v^(m).
        target:  [D]     observed feature vector v*.

    Returns:
        Scalar ES_hat = (1/M) sum_m ||v^m - v*|| - (1/(2M(M-1))) sum_{m!=m'} ||v^m - v^m'||.
    """
    if samples.dim() != 2:
        raise ValueError(f"samples must be [M, D], got {tuple(samples.shape)}")
    M = samples.shape[0]
    if M < 2:
        raise ValueError("energy_score needs M >= 2 samples for the spread term")

    # Term 1: mean distance to the target
    d_target = torch.linalg.vector_norm(samples - target.unsqueeze(0), dim=-1)  # [M]
    term1 = d_target.mean()

    # Term 2: pairwise spread (mean over ordered pairs m != m')
    # ||v^m - v^m'|| for all pairs, excluding the diagonal (zeros).
    diff = samples.unsqueeze(0) - samples.unsqueeze(1)          # [M, M, D]
    dist = torch.linalg.vector_norm(diff, dim=-1)               # [M, M]
    off_diag = dist.sum() / (M * (M - 1))                       # mean over ordered pairs
    term2 = 0.5 * off_diag

    return term1 - term2


# -----------------------------------------------------------------------------
# Composite loss
# -----------------------------------------------------------------------------

@dataclass
class StageB1bLossComponents:
    total: Tensor
    nll: Tensor
    es: Tensor
    lambda_es: float

    def as_loggable_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total.item(),
            "loss/nll": self.nll.item(),
            "loss/es": self.es.item(),
            "loss/lambda_es": float(self.lambda_es),
        }


def compute_stage_b1b_loss(
    nll_sum: Tensor,
    n_valid_dims: int,
    weights: StageB1bLossWeights,
    step: int,
    es_value: Tensor | None = None,
) -> StageB1bLossComponents:
    """Combine the mean-normalised NLL with the (optional) energy score.

    Args:
        nll_sum:       scalar NLL summed over steps/batch (from kalman_nll).
        n_valid_dims:  number of valid scalar dimensions the NLL was summed over
                       (valid_residues * dH * n_steps).  NLL is divided by this.
        weights:       StageB1bLossWeights.
        step:          global step, used for the ES warm-up ramp.
        es_value:      scalar energy score (None while still in NLL-only mode).
    """
    weights.validate()
    if n_valid_dims <= 0:
        raise ValueError("n_valid_dims must be positive")
    nll_mean = nll_sum / float(n_valid_dims)
    lam = weights.lambda_es_at(step)

    if es_value is None or lam == 0.0:
        es_used = torch.zeros((), device=nll_sum.device, dtype=nll_sum.dtype)
        total = weights.nll_weight * nll_mean
    else:
        es_used = es_value
        total = weights.nll_weight * nll_mean + lam * es_value

    return StageB1bLossComponents(
        total=total, nll=nll_mean.detach(), es=es_used.detach(), lambda_es=lam,
    )
