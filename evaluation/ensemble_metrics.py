"""Ensemble-level metrics for mdCATH structure dynamics (NEW, 2026-10-18).

This module is deliberately SEPARATE from the precise-atom-coordinate decoding
path (`evaluation/geometry.py`, `protein_jepa_frame/models/stage_c/structure_decoder.py`,
`protein_jepa_frame/models/stage_b1a/*`), which is retained unchanged.  Nothing
here decodes absolute coordinates; every quantity is a *distribution / ensemble
statistic* over a set of frames or samples:

    - RMSF profile              : per-residue flexibility
    - contact probability map   : P(||CA_i - CA_j|| < cutoff) over the ensemble
    - internal distance EMD     : per-pair distance distribution distance
    - covariance slow modes     : top-k eigenvalues/vectors of the CA covariance

All functions accept batched [N, L, 3] clouds (N = frames or samples) plus a
[L] valid-residue mask, and are written so the *model* side can be made
differentiable (smooth/detached-upstream as needed) while the *target* side is
identical code applied to true future frames.

Metric-vs-baseline helpers (`rmsf_profile_error`, `contact_prob_error`,
`distance_distribution_emd`) return a scalar "error" (lower is better) so that a
forecast can be compared directly against the `persist` and `oracle` anchors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "rmsf_profile",
    "align_ensemble",
    "contact_probability_map",
    "internal_distance_matrix",
    "covariance_slow_modes",
    "rmsf_profile_error",
    "contact_probability_error",
    "distance_distribution_emd",
    "ensemble_summary",
    "pair_distance_hist",
]


# ---------------------------------------------------------------------------
# Core ensemble statistics
# ---------------------------------------------------------------------------

def rmsf_profile(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-residue RMS fluctuation of an ensemble of CA clouds.

    Args:
        x:    [N, L, 3] ensemble of N frames/samples in a COMMON frame
              (no per-frame alignment; use `align_ensemble` first if the inputs
              are absolute coordinates ordered by time).
        mask: [L] valid-residue mask (optional).
    Returns:
        r:    [L] RMSF in Å, zero on invalid residues.
    """
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [N, L, 3]")
    mean = x.mean(dim=0, keepdim=True)                       # [1, L, 3]
    sq = (x - mean).square().sum(dim=-1).mean(dim=0)         # [L]
    # Safe sqrt: d/dx sqrt(x) = 1/(2 sqrt(x)) -> +inf at x=0 (masked residues
    # have exactly zero variance).  Clamping the radicand away from zero keeps
    # the FORWARD value exact for any real fluctuation and makes the backward
    # gradient finite (and zero) wherever the variance is (numerically) zero.
    r = torch.sqrt(sq.clamp_min(1e-12))
    if mask is not None:
        r = r * mask.to(r.dtype)
    return r


def align_ensemble(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Kabsch-align every frame in [N, L, 3] onto the ensemble mean cloud.

    Absolute MD frames carry global rigid motion; RMSF/covariance are meaningful
    only after removing it.  The reference is the FIRST frame (standard MD
    convention); this is deterministic and, for pure translation or a single
    global rotation, recovers each frame exactly.
    """
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [N, L, 3]")
    if mask is None:
        m = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
    else:
        m = mask.to(x.device, x.dtype)
    mm = m[:, None]
    denom = m.sum().clamp_min(1.0)
    # Fixed reference = the FIRST frame, valid-residue centroid removed.  This
    # is the standard MD convention (align all frames to a common reference);
    # for pure translation / a global rotation it recovers each frame exactly.
    ref_cloud = x[0]
    ref_centroid = (ref_cloud * mm).sum(0, keepdim=True) / denom  # [1, 3]
    refc = (ref_cloud - ref_centroid) * mm                        # [L, 3]
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        p = (x[i] - (x[i] * mm).sum(0, keepdim=True) / denom) * mm   # [L,3]
        Hm = p.T @ refc
        U, S, Vh = torch.linalg.svd(Hm.to(torch.float64))
        det_uvh = torch.det((U @ Vh).detach())
        d = 1.0 if float(det_uvh) >= 0 else -1.0
        D = torch.diag(torch.tensor([1.0, 1.0, d], device=x.device, dtype=torch.float64))
        R = (U @ D @ Vh).to(x.dtype)
        out[i] = p @ R + ref_centroid
    return out


def internal_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Pairwise CA distance matrix [L, L] averaged over the ensemble [N, L, 3]."""
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [N, L, 3]")
    d = torch.cdist(x, x)                                        # [N, L, L]
    return d.mean(dim=0)


def contact_probability_map(
    x: torch.Tensor, mask: torch.Tensor | None = None, cutoff: float = 8.0,
) -> torch.Tensor:
    """Ensemble contact probability P(||CA_i - CA_j|| < cutoff) in [0, 1].

    Args:
        x: [N, L, 3] ensemble. Returns [L, L]; diagonal and invalid pairs = 0.
    """
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [N, L, 3]")
    L = x.shape[1]
    d = torch.cdist(x, x)                                        # [N, L, L]
    contact = (d < cutoff).to(x.dtype).mean(dim=0)               # [L, L]
    contact.fill_diagonal_(0.0)
    if mask is not None:
        m = mask.to(x.device, x.dtype)
        contact = contact * m[:, None] * m[None, :]
    return contact


def covariance_slow_modes(
    x: torch.Tensor, mask: torch.Tensor | None = None, k: int = 5,
) -> torch.Tensor:
    """Top-k eigenvalues (descending) of the CA covariance.

    Args:
        x: [N, L, 3] ensemble (should be pre-aligned via `align_ensemble`).
    Returns:
        evals: [k] eigenvalues of the 3L x 3L covariance (variance, Å²).
    """
    if x.dim() != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [N, L, 3]")
    N = x.shape[0]
    if mask is not None:
        x = x[:, mask.bool(), :]
    flat = x.reshape(N, -1)                                       # [N, 3L]
    flat = flat - flat.mean(dim=0, keepdim=True)
    cov = (flat.T @ flat) / max(N - 1, 1)                         # [3L, 3L]
    evals = torch.linalg.eigvalsh(cov)                            # ascending
    evals = torch.flip(evals, dims=[-1])
    kk = min(k, evals.shape[0])
    return evals[:kk].clamp_min(0.0)


# ---------------------------------------------------------------------------
# Forecast-vs-target error helpers (scalar, lower = better)
# ---------------------------------------------------------------------------

def rmsf_profile_error(
    pred_x: torch.Tensor, true_x: torch.Tensor, mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compare predicted vs true ensemble flexibility profiles.

    Returns L1/L2 MAE of RMSF (Å) and Pearson correlation over valid residues.
    """
    rp = rmsf_profile(align_ensemble(pred_x, mask), mask)
    rt = rmsf_profile(align_ensemble(true_x, mask), mask)
    if mask is not None:
        m = mask.bool()
        if m.sum() < 2:
            return {"rmsf_mae": float("nan"), "rmsf_corr": float("nan")}
        rp, rt = rp[m], rt[m]
    mae = (rp - rt).abs().mean()
    a = rp - rp.mean()
    b = rt - rt.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    corr = (a * b).sum() / denom
    return {"rmsf_mae": float(mae), "rmsf_corr": float(corr)}


def contact_probability_error(
    pred_x: torch.Tensor, true_x: torch.Tensor, mask: torch.Tensor | None = None,
    cutoff: float = 8.0,
) -> dict[str, float]:
    """Compare predicted vs true ensemble contact-probability maps.

    Returns mean absolute probability error and the Jaccard of the thresholded
    (p >= 0.5) maps.
    """
    cp = contact_probability_map(pred_x, mask, cutoff)
    ct = contact_probability_map(true_x, mask, cutoff)
    L = cp.shape[0]
    i, j = torch.triu_indices(L, L, offset=1, device=cp.device)
    if mask is not None:
        m = mask.bool()
        keep = m[i] & m[j]
        i, j = i[keep], j[keep]
    if not len(i):
        return {"contact_prob_mae": float("nan"), "contact_prob_jaccard": float("nan")}
    mae = (cp[i, j] - ct[i, j]).abs().mean()
    bp = cp[i, j] >= 0.5
    bt = ct[i, j] >= 0.5
    union = (bp | bt).sum()
    jac = (bp & bt).sum().to(torch.float32) / union.clamp_min(1)
    return {"contact_prob_mae": float(mae), "contact_prob_jaccard": float(jac)}


def pair_distance_hist(
    x: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor,
    lo: float, hi: float, bins: int, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized per-pair distance histogram [n_pairs, bins] over ensemble [N,L,3]."""
    # Diffuse (soft) histogram so the map is differentiable w.r.t. x.
    # sqrt(d^2 + eps) keeps the gradient finite even if two points coincide.
    diff = x[:, pair_i, :] - x[:, pair_j, :]
    d = (diff.square().sum(dim=-1) + 1e-12).sqrt()                # [N, P]
    centers = torch.linspace(lo, hi, bins, device=x.device, dtype=x.dtype)
    width = (hi - lo) / (bins - 1)
    # triangular soft bins
    tri = (1.0 - (d.unsqueeze(-1) - centers).abs() / width).clamp_min(0.0)  # [N,P,bins]
    hist = tri.mean(dim=0)                                        # [P, bins]
    hist = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return hist


def distance_distribution_emd(
    pred_x: torch.Tensor, true_x: torch.Tensor, mask: torch.Tensor | None = None,
    lo: float = 0.0, hi: float = 40.0, bins: int = 40, min_seq_sep: int = 3,
    max_pairs: int = 512,
) -> dict[str, float]:
    """Earth-mover (1-Wasserstein) distance between pred and true per-pair
    internal-distance distributions, averaged over residue pairs.

    Lower = better.  Uses the cumulative-difference (exact 1-Wasserstein on a
    shared bin grid) so it is cheap and stable.
    """
    L = pred_x.shape[1]
    i, j = torch.triu_indices(L, L, offset=min_seq_sep, device=pred_x.device)
    if mask is not None:
        m = mask.bool()
        keep = m[i] & m[j]
        i, j = i[keep], j[keep]
    if len(i) == 0:
        return {"dist_emd": float("nan")}
    if len(i) > max_pairs:
        sel = torch.linspace(0, len(i) - 1, max_pairs, device=i.device).long()
        i, j = i[sel], j[sel]
    hp = pair_distance_hist(pred_x, i, j, lo, hi, bins, mask)    # [P, bins]
    ht = pair_distance_hist(true_x, i, j, lo, hi, bins, mask)
    bin_w = (hi - lo) / (bins - 1)
    # exact 1-Wasserstein between histograms on a common grid
    emd = (hp - ht).cumsum(dim=-1).abs().sum(dim=-1) * bin_w      # [P]
    return {"dist_emd": float(emd.mean())}


def ensemble_summary(
    pred_x: torch.Tensor, true_x: torch.Tensor, mask: torch.Tensor | None = None,
    cutoff: float = 8.0,
) -> dict[str, float]:
    """All ensemble-level forecast errors in one dict."""
    out: dict[str, float] = {}
    out.update(rmsf_profile_error(pred_x, true_x, mask))
    out.update(contact_probability_error(pred_x, true_x, mask, cutoff))
    out.update(distance_distribution_emd(pred_x, true_x, mask))
    return out
