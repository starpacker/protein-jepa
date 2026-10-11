"""P3.1: Canonical structure observation metrics.

Replaces ad-hoc FAPE usage across eval scripts with a single, well-documented
set of metrics.  All metrics operate on backbone coordinates [..., N, 4, 3]
where dim=-2 indexes [N, CA, C, O] atoms.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from collections.abc import Callable

from protein_jepa_frame import vendored as _vendored  # noqa: F401
from Bio.SVDSuperimposer import SVDSuperimposer


# ── Core geometry utilities ────────────────────────────────────────────────

def kabsch_rotation(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Optimal rotation from P to Q using OpenFold's official SVD primitive.

    Args:
        P: [B, N, 3] source points
        Q: [B, N, 3] target points
    Returns:
        R: [B, 3, 3] rotation matrices
    """
    if P.ndim != 3 or Q.shape != P.shape or P.shape[-1] != 3:
        raise ValueError("Kabsch inputs must have shape [B,N,3]")
    rotations = []
    for p, q in zip(P, Q):
        sup = SVDSuperimposer()
        sup.set(q.detach().cpu().numpy(), p.detach().cpu().numpy())
        sup.run()
        rot, _ = sup.get_rotran()
        rotations.append(torch.as_tensor(rot, dtype=P.dtype, device=P.device))
    return torch.stack(rotations, dim=0)


def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align source points P to target points Q via optimal rotation.

    Args:
        P: [B, N, 3] source
        Q: [B, N, 3] target
    Returns:
        P_aligned, R  where P_aligned = (P - mean_P) @ R^T + mean_Q
    """
    if P.ndim != 3 or Q.shape != P.shape or P.shape[-1] != 3:
        raise ValueError("Kabsch inputs must have shape [B,N,3]")
    aligned = []
    rotations = []
    for p, q in zip(P, Q):
        sup = SVDSuperimposer()
        sup.set(q.detach().cpu().numpy(), p.detach().cpu().numpy())
        sup.run()
        transformed = sup.get_transformed()
        rot, _ = sup.get_rotran()
        aligned.append(torch.as_tensor(transformed, dtype=P.dtype, device=P.device))
        rotations.append(torch.as_tensor(rot, dtype=P.dtype, device=P.device))
    return torch.stack(aligned, dim=0), torch.stack(rotations, dim=0)


def kabsch_align_backbone(pred_bb: torch.Tensor, targ_bb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align predicted Nx4x3 backbone to target, using CA (atom 1) for alignment.

    Args:
        pred_bb: [B, N, 4, 3]
        targ_bb: [B, N, 4, 3]
    Returns:
        pred_aligned [B,N,4,3], R [B,3,3]
    """
    B, N = pred_bb.shape[0], pred_bb.shape[1]
    pred_ca = pred_bb[:, :, 1, :]  # [B, N, 3]
    targ_ca = targ_bb[:, :, 1, :]  # [B, N, 3]
    aligned = []
    rotations = []
    for p, q in zip(pred_bb, targ_bb):
        sup = SVDSuperimposer()
        sup.set(q[:, 1, :].detach().cpu().numpy(), p[:, 1, :].detach().cpu().numpy())
        sup.run()
        rot, tran = sup.get_rotran()
        aligned.append(
            torch.as_tensor(p.detach().cpu().numpy() @ rot + tran,
                            dtype=pred_bb.dtype, device=pred_bb.device)
        )
        rotations.append(torch.as_tensor(rot, dtype=pred_bb.dtype, device=pred_bb.device))
    return torch.stack(aligned, dim=0), torch.stack(rotations, dim=0)


def per_residue_frames(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-residue local frames from CA, C, N atoms.

    Args:
        coords: [B, N, 4, 3] backbone, atoms = [N, CA, C, O]

    Returns:
        R: [B, N, 3, 3] rotation matrices
        t: [B, N, 3] translation vectors
    """
    N_atom = coords[:, :, 0, :]  # N
    CA = coords[:, :, 1, :]      # CA
    C_atom = coords[:, :, 2, :]  # C
    v1 = CA - N_atom
    v2 = C_atom - CA
    e1 = F.normalize(v1, dim=-1)
    e2 = F.normalize(v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)  # [B, N, 3, 3]
    return R, CA


def radius_of_gyration(ca_coords: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute radius of gyration.

    Args:
        ca_coords: [B, N, 3]
        mask: [B, N] bool
    Returns:
        [B] Rg values
    """
    if mask is not None:
        mask_f = mask.float().unsqueeze(-1)
        ca_coords = ca_coords * mask_f
        n = mask_f.sum(dim=1).clamp(min=1)
    else:
        n = torch.tensor(ca_coords.shape[-2], device=ca_coords.device)
    com = ca_coords.sum(dim=-2, keepdim=True) / n.unsqueeze(-1)
    sq_diff = ((ca_coords - com) ** 2).sum(dim=-1)
    if mask is not None:
        sq_diff = sq_diff * mask.float()
    return torch.sqrt(sq_diff.sum(dim=-1) / n)


def contact_matrix(ca_coords: torch.Tensor, cutoff: float = 8.0) -> torch.Tensor:
    """Compute binary contact map.

    Args:
        ca_coords: [B, N, 3]
        cutoff: distance threshold in Å
    Returns:
        [B, N, N] boolean contact matrix
    """
    dists = torch.cdist(ca_coords, ca_coords)  # [B,N,N]
    return (dists < cutoff) & ~torch.eye(dists.shape[1], device=dists.device, dtype=torch.bool).unsqueeze(0)


def contact_jaccard(pred_ca: torch.Tensor, targ_ca: torch.Tensor,
                    cutoff: float = 8.0, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Jaccard index between predicted and target contact maps.

    Args:
        pred_ca, targ_ca: [B, N, 3]
        cutoff: distance threshold
        mask: [B, N]
    Returns:
        [B] Jaccard index per sample
    """
    pred_contacts = contact_matrix(pred_ca, cutoff)
    targ_contacts = contact_matrix(targ_ca, cutoff)
    if mask is not None:
        pair_mask = mask.bool().unsqueeze(1) & mask.bool().unsqueeze(2)
        pred_contacts = pred_contacts & pair_mask
        targ_contacts = targ_contacts & pair_mask
    intersect = (pred_contacts & targ_contacts).float().sum(dim=(1, 2))
    union = (pred_contacts | targ_contacts).float().sum(dim=(1, 2))
    return intersect / union.clamp(min=1)


def pairwise_distance_mae(pred_ca: torch.Tensor, targ_ca: torch.Tensor,
                          mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean absolute error of pairwise distances.

    Args:
        pred_ca, targ_ca: [B, N, 3]
        mask: [B, N]
    Returns:
        [B] MAE per sample
    """
    pred_dists = torch.cdist(pred_ca, pred_ca)
    targ_dists = torch.cdist(targ_ca, targ_ca)
    diff = (pred_dists - targ_dists).abs()
    off_diagonal = ~torch.eye(
        diff.shape[-1], device=diff.device, dtype=torch.bool
    ).unsqueeze(0)
    diff = diff * off_diagonal.float()
    if mask is not None:
        mask2d = mask.float().unsqueeze(-1) * mask.float().unsqueeze(-2)
        diff = diff * mask2d
        denom = (mask2d * off_diagonal.float()).sum(dim=(1, 2)).clamp(min=1)
        return diff.sum(dim=(1, 2)) / denom
    return diff.sum(dim=(1, 2)) / off_diagonal.float().sum(dim=(1, 2)).clamp(min=1)


# ── Canonical structure metrics ────────────────────────────────────────────

def structure_metrics(
    pred_bb: torch.Tensor,
    target_bb: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all canonical structure metrics.

    Args:
        pred_bb: [B, N, 4, 3] predicted backbone
        target_bb: [B, N, 4, 3] target backbone
        mask: [B, N] residue mask

    Returns:
        dict with keys: rmsd_kabsch, kabsch_point_error_ca, dist_mae,
                        contact_jaccard_8A, rg_error
    """
    B = pred_bb.shape[0]
    results = {}

    # 1. Kabsch-aligned CA RMSD (global rotation)
    pred_ca = pred_bb[:, :, 1, :].reshape(B, -1, 3)
    targ_ca = target_bb[:, :, 1, :].reshape(B, -1, 3)
    pred_aligned, _ = kabsch_align(pred_ca, targ_ca)
    diff = (pred_aligned - targ_ca).pow(2).sum(dim=-1)
    if mask is not None:
        diff = diff * mask.float()
        n_res = mask.float().sum(dim=1).clamp(min=1)
    else:
        n_res = torch.tensor(pred_ca.shape[1], device=pred_ca.device)
    results["rmsd_kabsch"] = torch.sqrt(diff.sum(dim=1) / n_res)

    # 2. Kabsch-aligned CA point error (per-atom mean using global alignment)
    pred_aligned_bb, _ = kabsch_align_backbone(pred_bb, target_bb)
    bb_diff = (pred_aligned_bb - target_bb).pow(2).sum(dim=-1)  # [B,N,4]
    if mask is not None:
        bb_diff = bb_diff * mask.unsqueeze(-1).float()
        n_res = mask.float().sum(dim=1).clamp(min=1)  # [B]
        n_atoms = n_res * 4
    else:
        n_atoms = pred_bb.shape[1] * 4
    results["kabsch_point_error_ca"] = torch.sqrt(bb_diff.sum(dim=(1, 2)) / n_atoms)

    # 3. Pairwise CA distance MAE
    pred_ca_aligned, _ = kabsch_align(pred_ca, targ_ca)
    results["dist_mae"] = pairwise_distance_mae(pred_ca_aligned, targ_ca, mask)

    # 4. Contact Jaccard at 8Å
    results["contact_jaccard_8A"] = contact_jaccard(pred_ca_aligned, targ_ca, cutoff=8.0, mask=mask)

    # 5. Radius of gyration error
    pred_rg = radius_of_gyration(pred_ca_aligned, mask)
    targ_rg = radius_of_gyration(targ_ca, mask)
    results["rg_error"] = (pred_rg - targ_rg).abs()

    return results


def true_fape_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    target_R: torch.Tensor | None = None,
    target_t: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    clamp: float = 10.0,
) -> torch.Tensor:
    """P3.1: True Frame Aligned Point Error.

    For each residue i, measure the error of all points in frame i's local
    coordinates.  This is the correct FAPE, not Kabsch-aligned error.

    Args:
        pred_coords: [B, N, 4, 3] predicted backbone
        target_coords: [B, N, 4, 3] target backbone
        target_R: [B, N, 3, 3] target residue frames (computed from target if None)
        target_t: [B, N, 3] target residue origins (CA positions)
        mask: [B, N]
        clamp: max error per atom in Å

    Returns:
        scalar loss
    """
    B, N = pred_coords.shape[:2]

    pred_R, pred_t = per_residue_frames(pred_coords)
    if target_R is None or target_t is None:
        target_R, target_t = per_residue_frames(target_coords)

    # Frame i against every point j, using the corresponding predicted and
    # target local frames.  The previous implementation accidentally broadcast
    # only the matching residue and was therefore not FAPE.
    pred_centered = pred_coords[:, None] - pred_t[:, :, None, None, :]
    targ_centered = target_coords[:, None] - target_t[:, :, None, None, :]
    pred_local = torch.einsum("bnjpd,bnid->bnjpi", pred_centered, pred_R)
    targ_local = torch.einsum("bnjpd,bnid->bnjpi", targ_centered, target_R)

    clamped = (pred_local - targ_local).norm(dim=-1).clamp(max=clamp)
    if mask is not None:
        pair_mask = mask[:, :, None] * mask[:, None, :]
        clamped = clamped * pair_mask[:, :, :, None].float()
        denom = pair_mask.float().sum(dim=(1, 2)).clamp(min=1.0) * pred_coords.shape[2]
        loss = clamped.sum(dim=(1, 2, 3)) / denom
        return loss.mean()
    return clamped.mean()

    return loss


def rmsf_correlation(
    pred_ensemble: torch.Tensor,
    target_ensemble: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pearson correlation of per-residue RMSF vectors.

    Args:
        pred_ensemble: [S, N, 3] S samples of CA positions
        target_ensemble: [S, N, 3] S samples of CA positions
        mask: [N]

    Returns:
        scalar Pearson r
    """
    pred_rmsf = pred_ensemble.std(dim=0).norm(dim=-1)  # [N]
    targ_rmsf = target_ensemble.std(dim=0).norm(dim=-1)  # [N]

    if mask is not None:
        pred_rmsf = pred_rmsf * mask.float()
        targ_rmsf = targ_rmsf * mask.float()
        valid = mask.float().sum() > 2
        if not valid:
            return torch.tensor(0.0, device=pred_rmsf.device)

    pred_centered = pred_rmsf - pred_rmsf.mean()
    targ_centered = targ_rmsf - targ_rmsf.mean()
    numerator = (pred_centered * targ_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (targ_centered ** 2).sum())
    return numerator / denominator.clamp(min=1e-8)


__all__ = [
    "kabsch_rotation", "kabsch_align", "kabsch_align_backbone",
    "per_residue_frames", "radius_of_gyration", "contact_matrix",
    "contact_jaccard", "pairwise_distance_mae", "structure_metrics",
    "true_fape_loss", "rmsf_correlation",
]
