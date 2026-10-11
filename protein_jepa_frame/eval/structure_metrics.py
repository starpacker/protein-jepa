"""Audited single-structure metrics used by the Stage-C evaluations.

The evaluation scripts used to carry several copies of a small Kabsch/contact
implementation.  That made it very easy for one script to silently use a
different mask or normalization.  This module is the one metric adapter for
single predicted structures versus a target structure.

Kabsch fitting is delegated to the same official Biopython SVD implementation
used by OpenFold (``openfold.utils.superimposition``).  Contacts and pairwise
distances are simple tensor calculations, but their masks and denominators are
defined here once and tested explicitly.

This module intentionally does *not* call a single-structure quantity RMWD.
RMWD is an ensemble metric produced by the AlphaFlow protocol; a Kabsch RMSD
is not an RMWD and must not be reported under that name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from protein_jepa_frame import vendored as _vendored  # noqa: F401
from openfold.utils.superimposition import superimpose as _openfold_superimpose
from Bio.SVDSuperimposer import SVDSuperimposer


def _as_numpy(points: torch.Tensor) -> np.ndarray:
    return np.asarray(points.detach().cpu(), dtype=np.float64)


def _validate_pair(source: torch.Tensor, target: torch.Tensor) -> None:
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError(
            f"Kabsch inputs must both be [N, 3], got {tuple(source.shape)} "
            f"and {tuple(target.shape)}"
        )
    if source.shape[0] < 3:
        raise ValueError("Kabsch evaluation requires at least 3 valid points")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Kabsch inputs contain non-finite coordinates")


def official_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align ``source`` onto ``target`` using OpenFold's official path.

    Returns the transformed source and its RMSD in the input coordinate unit
    (Angstrom for this project).  OpenFold's public ``superimpose`` is used for
    the RMSD.  Biopython's public ``SVDSuperimposer`` is the same primitive
    OpenFold calls internally and supplies the rigid transform needed to apply
    the fit to additional atoms (N/C/O).
    """
    _validate_pair(source, target)
    ref_np = _as_numpy(target)
    src_np = _as_numpy(source)
    sup = SVDSuperimposer()
    sup.set(ref_np, src_np)
    sup.run()
    aligned_np = sup.get_transformed()

    # Use OpenFold's own adapter for the reported RMSD, rather than a second
    # hand-written SVD/RMSD formula.  It expects [*, N, 3] and a point mask.
    ref_t = target.detach().cpu().to(torch.float32).unsqueeze(0)
    src_t = source.detach().cpu().to(torch.float32).unsqueeze(0)
    mask = torch.ones(1, source.shape[0], dtype=torch.float32)
    _, rmsd = _openfold_superimpose(ref_t, src_t, mask)
    aligned = torch.as_tensor(aligned_np, dtype=source.dtype, device=source.device)
    return aligned, rmsd[0].to(device=source.device, dtype=source.dtype)


def align_backbone_by_ca(
    source_bb: torch.Tensor,
    target_bb: torch.Tensor,
) -> torch.Tensor:
    """Apply the official CA Kabsch fit to every atom in a backbone.

    Both inputs are ``[N, 4, 3]`` in ``[N, CA, C, O]`` order.  The fit is
    computed on CA only, matching the Stage-C RMSD convention.
    """
    if source_bb.ndim != 3 or source_bb.shape[-2:] != (4, 3):
        raise ValueError("backbone must have shape [N, 4, 3]")
    if source_bb.shape != target_bb.shape:
        raise ValueError("source and target backbone shapes differ")
    _validate_pair(source_bb[:, 1, :], target_bb[:, 1, :])
    sup = SVDSuperimposer()
    sup.set(_as_numpy(target_bb[:, 1, :]), _as_numpy(source_bb[:, 1, :]))
    sup.run()
    rot_np, tran_np = sup.get_rotran()
    rot = torch.as_tensor(rot_np, dtype=source_bb.dtype, device=source_bb.device)
    tran = torch.as_tensor(tran_np, dtype=source_bb.dtype, device=source_bb.device)
    return torch.matmul(source_bb, rot) + tran


def _valid_structure_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor | None,
) -> torch.Tensor:
    mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
    if residue_mask is not None:
        mask &= residue_mask.to(device=pred.device, dtype=torch.bool)
    pred_finite = torch.isfinite(pred).all(dim=(-1, -2))
    target_finite = torch.isfinite(target).all(dim=(-1, -2))
    # A finite check is a validity gate, not a silent failure filter.  Missing
    # residues may contain padded zeros/non-finite sentinels and are excluded
    # by residue_mask; a non-finite coordinate on a residue we intend to score
    # must abort the sample so it cannot become a deceptively complete report.
    if (~pred_finite & mask).any() or (~target_finite & mask).any():
        raise ValueError("Kabsch inputs contain non-finite coordinates")
    return mask & pred_finite & target_finite


def _pair_mask(
    n: int,
    *,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
    device: torch.device,
) -> torch.Tensor:
    """Return valid nonlocal residue pairs.

    Sequence separation is applied only within a chain.  Residues from two
    different chains are eligible regardless of their numeric residue index.
    """
    idx = torch.arange(n, device=device)
    if residue_index is None:
        residue_index = idx
    else:
        residue_index = residue_index.to(device=device)
    if chain_id is None:
        chain_id = torch.zeros(n, dtype=torch.long, device=device)
    else:
        chain_id = chain_id.to(device=device)
    same_chain = chain_id[:, None] == chain_id[None, :]
    separation = (residue_index[:, None] - residue_index[None, :]).abs()
    return (~torch.eye(n, dtype=torch.bool, device=device)) & (
        (~same_chain) | (separation > min_seq_sep)
    )


def _contact_map(
    ca: torch.Tensor,
    *,
    lo: float,
    hi: float,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0))[0]
    return (distances >= lo) & (distances < hi) & pair_mask


def _jaccard(pred: torch.Tensor, target: torch.Tensor) -> float:
    union = pred | target
    union_count = int(union.sum().item())
    if union_count == 0:
        # Two empty sets are identical.  Returning 0 here makes proteins with
        # no contacts look artificially bad and is not the Jaccard convention.
        return 1.0
    return float((pred & target).sum().item() / union_count)


def compute_structure_metrics(
    pred_bb: torch.Tensor,
    target_bb: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
) -> dict[str, float]:
    """Compute the canonical Stage-C single-target metrics.

    ``pred_bb`` and ``target_bb`` are ``[L, 4, 3]`` in ``[N, CA, C, O]``
    order.  The model and target masks are intersected with finite-coordinate
    checks before any metric is computed.  The global fit uses CA atoms and is
    then applied to all four backbone atoms.
    """
    if pred_bb.ndim != 3 or pred_bb.shape[-2:] != (4, 3):
        raise ValueError(f"expected pred_bb [L,4,3], got {tuple(pred_bb.shape)}")
    if target_bb.shape != pred_bb.shape:
        raise ValueError("prediction and target backbone shapes differ")

    valid = _valid_structure_mask(pred_bb, target_bb, residue_mask)
    n_valid = int(valid.sum().item())
    if n_valid < 3:
        raise ValueError(f"fewer than 3 valid residues ({n_valid})")

    p = pred_bb[valid]
    t = target_bb[valid]
    p_ca, t_ca = p[:, 1, :], t[:, 1, :]
    aligned_ca, rmsd = official_kabsch(p_ca, t_ca)

    # Fit once on CA and apply the same rigid transform to N/CA/C/O.  This is
    # exactly the alignment convention used for the backbone point metric.
    sup = SVDSuperimposer()
    sup.set(_as_numpy(t_ca), _as_numpy(p_ca))
    sup.run()
    rot_np, tran_np = sup.get_rotran()
    rot = torch.as_tensor(rot_np, dtype=p.dtype, device=p.device)
    tran = torch.as_tensor(tran_np, dtype=p.dtype, device=p.device)
    aligned_bb = torch.matmul(p, rot) + tran

    point_error = (aligned_bb - t).norm(dim=-1).mean()
    pair_mask = _pair_mask(
        n_valid,
        residue_index=residue_index[valid] if residue_index is not None else None,
        chain_id=chain_id[valid] if chain_id is not None else None,
        min_seq_sep=min_seq_sep,
        device=p.device,
    )
    pred_dist = torch.cdist(p_ca, p_ca)
    target_dist = torch.cdist(t_ca, t_ca)
    n_pairs = int(pair_mask.sum().item())
    if n_pairs:
        dist_mae = (pred_dist - target_dist).abs()[pair_mask].mean()
        contact_values = {
            "contact_jaccard": _jaccard(
                _contact_map(p_ca, lo=0.0, hi=8.0, pair_mask=pair_mask),
                _contact_map(t_ca, lo=0.0, hi=8.0, pair_mask=pair_mask),
            ),
            "weak_contact_j": _jaccard(
                _contact_map(p_ca, lo=8.0, hi=15.0, pair_mask=pair_mask),
                _contact_map(t_ca, lo=8.0, hi=15.0, pair_mask=pair_mask),
            ),
            "transient_contact_j": _jaccard(
                _contact_map(p_ca, lo=15.0, hi=25.0, pair_mask=pair_mask),
                _contact_map(t_ca, lo=15.0, hi=25.0, pair_mask=pair_mask),
            ),
        }
    else:
        dist_mae = torch.zeros((), dtype=p.dtype, device=p.device)
        contact_values = {
            "contact_jaccard": 1.0,
            "weak_contact_j": 1.0,
            "transient_contact_j": 1.0,
        }

    return {
        "rmsd_kabsch": float(rmsd.item()),
        "kabsch_ca_point_error": float(point_error.item()),
        "dist_mae": float(dist_mae.item()),
        **contact_values,
        "n_valid_residues": n_valid,
        "n_nonlocal_pairs": n_pairs,
    }


__all__ = ["official_kabsch", "align_backbone_by_ca", "compute_structure_metrics"]
