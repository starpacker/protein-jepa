"""Supplementary (non-canonical) contact metrics for Stage-A/B auditing.

These metrics are SUPPLEMENTARY DIAGNOSTICS ONLY. They are intentionally kept
in a separate module so the canonical hard contact-Jaccard gate in
``structure_metrics.compute_structure_metrics`` is never altered or shadowed.

Motivation (from the Phase-2 canonical gate audit): the hard contact-Jaccard
gate is non-monotone in prediction accuracy — a partially-correct displacement
field can score *worse* than persistence because it flips individual binary
contacts. That is the correct behaviour for a hard gate, but it makes the gate
useless as a *training-signal diagnostic*. These soft/probabilistic metrics
measure whether a prediction is moving contact geometry in the right direction,
even when it has not yet reached the near-oracle accuracy the hard gate needs.

Provided metrics:
  - soft_contact_error: MAE between sigmoid-soft contact occupancies
    (a smooth relaxation of the 8 Å step), monotone in distance error.
  - contact_brier: Brier score of predicted contact probabilities vs the binary
    target contact map (proper scoring rule, rewards calibrated partial belief).
  - contact_auprc_proxy: rank-correlation proxy (Spearman) between predicted
    closeness and target contact, insensitive to a global distance offset.

All three use the same nonlocal pair mask convention as the canonical metric
(min_seq_sep within chain, cross-chain always eligible) so they are directly
comparable to the hard gate's pair set.
"""

from __future__ import annotations

import torch

from protein_jepa_frame.eval.structure_metrics import _pair_mask


def _soft_contact(ca: torch.Tensor, *, cutoff: float, temp: float) -> torch.Tensor:
    """Smooth contact occupancy in [0,1]: sigmoid((cutoff - d)/temp)."""
    d = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0))[0]
    return torch.sigmoid((cutoff - d) / temp)


def soft_contact_error(
    pred_ca: torch.Tensor,
    target_ca: torch.Tensor,
    *,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
    cutoff: float = 8.0,
    temp: float = 1.0,
) -> float:
    """Mean absolute error of soft contact occupancy over nonlocal pairs.

    Monotone in distance error, so unlike the hard Jaccard it does NOT penalise
    a prediction for being partially-correct. Lower is better; 0 = identical.
    """
    n = pred_ca.shape[0]
    pm = _pair_mask(
        n, residue_index=residue_index, chain_id=chain_id,
        min_seq_sep=min_seq_sep, device=pred_ca.device,
    )
    if int(pm.sum()) == 0:
        return 0.0
    sp = _soft_contact(pred_ca, cutoff=cutoff, temp=temp)
    st = _soft_contact(target_ca, cutoff=cutoff, temp=temp)
    return float((sp - st).abs()[pm].mean().item())


def contact_brier(
    pred_ca: torch.Tensor,
    target_ca: torch.Tensor,
    *,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
    cutoff: float = 8.0,
    temp: float = 1.0,
) -> float:
    """Brier score: mean squared error between predicted contact probability and
    the binary target contact. Proper scoring rule; lower is better.
    """
    n = pred_ca.shape[0]
    pm = _pair_mask(
        n, residue_index=residue_index, chain_id=chain_id,
        min_seq_sep=min_seq_sep, device=pred_ca.device,
    )
    if int(pm.sum()) == 0:
        return 0.0
    p = _soft_contact(pred_ca, cutoff=cutoff, temp=temp)[pm]
    dt = torch.cdist(target_ca.unsqueeze(0), target_ca.unsqueeze(0))[0]
    y = (dt < cutoff).float()[pm]
    return float(((p - y) ** 2).mean().item())


def contact_rank_corr(
    pred_ca: torch.Tensor,
    target_ca: torch.Tensor,
    *,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
) -> float:
    """Spearman rank correlation between predicted and target pair closeness
    (-distance) over nonlocal pairs. Offset/scale-insensitive direction check;
    higher is better, 1 = perfect ordering of which pairs are closest.
    """
    n = pred_ca.shape[0]
    pm = _pair_mask(
        n, residue_index=residue_index, chain_id=chain_id,
        min_seq_sep=min_seq_sep, device=pred_ca.device,
    )
    if int(pm.sum()) < 3:
        return float("nan")
    dp = torch.cdist(pred_ca.unsqueeze(0), pred_ca.unsqueeze(0))[0][pm]
    dt = torch.cdist(target_ca.unsqueeze(0), target_ca.unsqueeze(0))[0][pm]
    # Spearman = Pearson on ranks
    def _rank(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(x.numel(), device=x.device, dtype=x.dtype)
        return ranks
    rp = _rank(-dp)  # closeness
    rt = _rank(-dt)
    rp = rp - rp.mean()
    rt = rt - rt.mean()
    denom = (rp.norm() * rt.norm()).clamp(min=1e-9)
    return float((rp @ rt / denom).item())


__all__ = ["soft_contact_error", "contact_brier", "contact_rank_corr"]
