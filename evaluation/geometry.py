"""Single source of truth for geometry in evaluation.

Nothing in `evaluation/` re-implements Kabsch or CA extraction.  All metrics go
through these functions so that the number we report means exactly one thing.

Convention (LOCKED — matches the corrected Stage-B1b pipeline, see
`B1B_B2_FULL_PIPELINE_RESULTS.md` §0):

    Given point clouds P, Q with a valid-residue mask,
        pc = P - mean(P),  qc = Q - mean(Q)          (means over valid residues)
        Hm = pc.T @ qc
        U, S, Vh = svd(Hm)
        d  = sign(det(U @ Vh))
        R  = U @ diag(1, 1, d) @ Vh                  # rotation mapping P onto Q
        aligned_P = pc @ R
    Then aligned_rmsd(P, Q) = sqrt(mean(|| aligned_P - qc ||^2)).

Verified against an independently constructed ground-truth rotation
(tests/test_geometry.py) to < 1e-6 Å.
"""
from __future__ import annotations

import torch

CA_INDEX = 1  # backbone tensor atom order is (N, CA, C, O)


def ca_from_backbone(coords: torch.Tensor) -> torch.Tensor:
    """Extract CA coordinates.

    Args:
        coords: [..., L, 4, 3] backbone (N, CA, C, O).
    Returns:
        [..., L, 3] CA positions.
    """
    if coords.shape[-2] != 4:
        raise ValueError(f"expected backbone dim of 4 (N,CA,C,O), got {coords.shape[-2]}")
    return coords[..., CA_INDEX, :]


def center(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract the masked mean. x:[...,L,3], mask:[...,L] bool -> centered x.

    Zeroes out masked-out residues so they cannot contribute to later reductions.
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    mean = (x * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + 1e-6)
    return (x - mean) * m


def _kabsch_rotation(pc: torch.Tensor, qc: torch.Tensor) -> torch.Tensor:
    """Optimal rotation R (mapping pc onto qc) for already-centered clouds [L,3]."""
    Hm = pc.T @ qc
    # SVD in float64 for numerical stability; promote if needed.
    dtype = torch.float64 if pc.dtype in (torch.float32, torch.float64) else pc.dtype
    U, _, Vh = torch.linalg.svd(Hm.to(dtype))
    d = torch.sign(torch.det(U @ Vh))
    diag = torch.tensor([1.0, 1.0, float(d)], dtype=dtype, device=pc.device)
    return (U @ torch.diag(diag) @ Vh).to(pc.dtype)


def kabsch_align(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rigid-body superpose `pred` onto `target` (mask over valid residues).

    Args:
        pred, target: [L, 3]
        mask: [L] bool
    Returns:
        aligned pred: [L, 3]  (= pred_centered @ R + target_mean)
    """
    p, q = pred[mask], target[mask]
    if p.shape[0] < 3:
        raise ValueError("need >= 3 valid residues for Kabsch")
    pc = p - p.mean(0, keepdim=True)
    qc = q - q.mean(0, keepdim=True)
    R = _kabsch_rotation(pc, qc)
    aligned = pc @ R + q.mean(0, keepdim=True)
    out = torch.zeros_like(pred)
    if mask.dtype == torch.bool:
        out[mask] = aligned
    else:
        out[mask.bool()] = aligned
    return out


def aligned_rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Anchor-aligned CA RMSD in Angstrom.

    Superposes `pred` onto `target` over valid residues, returns RMSD.  This is
    THE metric.  Persistence = aligned_rmsd(X_anchor, X_true, mask).
    """
    if int(mask.sum()) < 3:
        return float("nan")
    p, q = pred[mask], target[mask]
    if p.shape[0] < 3:
        return float("nan")
    pc = p - p.mean(0, keepdim=True)
    qc = q - q.mean(0, keepdim=True)
    R = _kabsch_rotation(pc, qc)
    aligned = pc @ R
    msq = ((aligned - qc) ** 2).sum(-1).mean()
    return float(msq.clamp_min(0.0).sqrt())


def anchor_align_rotation(anchor: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Return (R, t) mapping `anchor` onto `target`.  R:[3,3], t:[3].

    Used when we must apply the SAME alignment to a model prediction that lives
    in the anchor frame (so the prediction and anchor are put into the target
    frame identically).  Computed from REAL coordinates only (no leakage).
    """
    p, q = anchor[mask], target[mask]
    pc = p - p.mean(0, keepdim=True)
    qc = q - q.mean(0, keepdim=True)
    R = _kabsch_rotation(pc, qc)
    t = q.mean(0, keepdim=True) - (p.mean(0, keepdim=True) @ R)
    return R, t.squeeze(0)


def apply_rigid(x: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply (R,t): x:[...,L,3] -> x @ R + t.  Matches _kabsch_rotation convention."""
    return x @ R + t


def contact_map(ca: torch.Tensor, mask: torch.Tensor, cutoff: float = 8.0) -> torch.Tensor:
    """Binary residue-residue contact map from CA positions.

    Args: ca:[L,3], mask:[L]
    Returns: [L,L] bool, True if i!=j, both valid, and ||ca_i - ca_j|| < cutoff.
    """
    d = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0))[0]
    valid = mask.bool()
    m = (d < cutoff) & valid[:, None] & valid[None, :]
    m.fill_diagonal_(False)
    return m


def contact_jaccard(pred_ca: torch.Tensor, true_ca: torch.Tensor,
                    mask: torch.Tensor, cutoff: float = 8.0) -> float:
    """Jaccard similarity of contact maps (secondary signal; not gating)."""
    a = contact_map(pred_ca, mask, cutoff)
    b = contact_map(true_ca, mask, cutoff)
    union = (a | b).sum().item()
    if union == 0:
        return float("nan")
    return float((a & b).sum().item()) / union
