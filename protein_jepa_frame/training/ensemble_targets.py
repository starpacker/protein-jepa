"""T2 — Ensemble targets from mdCATH future windows (NEW, 2026-10-18).

This is the DATA side of the ensemble-level objective (see
`PLAN_ENSEMBLE_20261018.md`).  It does NOT touch the precise-atom-coordinate
decoding path; it only derives distribution-level statistics from a window of
*future* frames that the paired dataset already provides.

A paired batch carries `future_coords` of shape [B, T_fut, L, 4, 3]: the T_fut
frames at horizons (h, h+stride, ..., h+(T_fut-1)*stride) after the context
window.  From this window we build the TRUE ensemble statistics:

    rmsf_true       : [B, L]
    contact_true    : [B, L, L]     (probability over the window)
    dist_hist_true  : [B, P, bins]  (per-pair internal-distance distribution)

The same builders are applied to the *model* ensemble (multiple decoded samples)
and to the persistence/oracle anchors, so every quantity is produced by one
code path (`evaluation.ensemble_metrics`).
"""
from __future__ import annotations

import torch

from evaluation.ensemble_metrics import (
    align_ensemble,
    rmsf_profile,
    contact_probability_map,
    pair_distance_hist,
)

__all__ = [
    "EnsembleTargets",
    "extract_ca_window",
    "build_ensemble_targets",
    "persistence_ensemble",
    "oracle_ensemble",
    "pair_indices",
]


def extract_ca_window(
    future_coords: torch.Tensor, residue_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, T_fut, L, 4, 3] -> (ca [B, T_fut, L, 3], mask [B, L])."""
    ca = future_coords[:, :, :, 1, :]                       # [B, T, L, 3]
    if residue_mask is None:
        mask = torch.ones(ca.shape[0], ca.shape[2], dtype=torch.bool, device=ca.device)
    else:
        mask = residue_mask[:, -1].bool() if residue_mask.dim() == 3 else residue_mask.bool()
    return ca, mask


def pair_indices(
    L: int, mask: torch.Tensor, min_seq_sep: int = 3, max_pairs: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared residue-pair index set for distance-distribution targets.

    Uses a MASK-INDEPENDENT, deterministic subsample of the upper triangle so
    that the target and the model ensemble are always evaluated on the SAME
    pairs (essential for a meaningful EMD).
    """
    i, j = torch.triu_indices(L, L, offset=min_seq_sep, device=mask.device)
    if len(i) > max_pairs:
        sel = torch.linspace(0, len(i) - 1, max_pairs, device=i.device).long()
        i, j = i[sel], j[sel]
    return i, j


class EnsembleTargets:
    """Container for per-batch ensemble truth + the shared pair index set."""

    def __init__(self, rmsf, contact, ca_window, mask, pair_i, pair_j,
                 dist_lo, dist_hi, dist_bins):
        self.rmsf = rmsf                 # [B, L]
        self.contact = contact           # [B, L, L]
        self.ca_window = ca_window       # [B, T, L, 3] (aligned per sample)
        self.mask = mask                 # [B, L]
        self.pair_i = pair_i             # [P]
        self.pair_j = pair_j             # [P]
        self.dist_lo = dist_lo
        self.dist_hi = dist_hi
        self.dist_bins = dist_bins

    def dist_hist_from_ca(self, ca: torch.Tensor) -> torch.Tensor:
        """Per-pair distance histograms [B, P, bins] for an ensemble [B, N, L, 3]."""
        B = ca.shape[0]
        outs = []
        for b in range(B):
            outs.append(pair_distance_hist(
                ca[b], self.pair_i, self.pair_j,
                self.dist_lo, self.dist_hi, self.dist_bins, self.mask[b]))
        return torch.stack(outs, dim=0)

    def dist_hist_true(self) -> torch.Tensor:
        return self.dist_hist_from_ca(self.ca_window)


def build_ensemble_targets(
    future_coords: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    cutoff: float = 8.0,
    dist_lo: float = 0.0,
    dist_hi: float = 40.0,
    dist_bins: int = 40,
    min_seq_sep: int = 3,
    max_pairs: int = 512,
    min_frames: int = 2,
) -> EnsembleTargets:
    """Build ensemble truth from a future-frame window.

    Args:
        future_coords: [B, T_fut, L, 4, 3] (T_fut >= min_frames).
        residue_mask:  [B, L] or [B, T_fut, L]; defaults to all-valid.
    """
    ca, mask = extract_ca_window(future_coords, residue_mask)
    if ca.shape[1] < min_frames:
        raise ValueError(
            f"ensemble targets need >= {min_frames} future frames, got {ca.shape[1]}"
        )
    B, T, L, _ = ca.shape
    # Per-sample alignment: remove each sample's own global rigid motion.
    aligned = torch.stack([align_ensemble(ca[b], mask[b]) for b in range(B)], dim=0)
    rmsf = torch.stack([rmsf_profile(aligned[b], mask[b]) for b in range(B)], dim=0)
    contact = torch.stack(
        [contact_probability_map(aligned[b], mask[b], cutoff) for b in range(B)], dim=0)
    pi, pj = pair_indices(L, mask[0], min_seq_sep, max_pairs)
    return EnsembleTargets(rmsf, contact, aligned, mask, pi, pj,
                           dist_lo, dist_hi, dist_bins)


def persistence_ensemble(
    context_coords: torch.Tensor, n: int, residue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persistence ensemble [B, n, L, 3]: the anchor CA repeated n times.

    A forecast that assumes the structure never moves.  Its ensemble statistics
    are those of a delta distribution (RMSF=0, contact = anchor contacts).
    """
    ctx_ca, mask = extract_ca_window(context_coords, residue_mask)
    anchor = ctx_ca[:, -1]                                  # [B, L, 3]
    return anchor.unsqueeze(1).expand(-1, n, -1, -1).contiguous()


def oracle_ensemble(
    future_coords: torch.Tensor, residue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Oracle ensemble = the true future window itself (upper bound)."""
    ca, _ = extract_ca_window(future_coords, residue_mask)
    return ca
