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
    """Independently superposed shape RMSD in Angstrom (secondary metric).

    Superposes `pred` onto `target` over valid residues, returns RMSD.  This is
    It is not the common-anchor displacement error. Persistence has the same
    value under both definitions, but a prediction generally does not.
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
    mask = mask.bool()
    if int(mask.sum()) < 3:
        raise ValueError("need >= 3 valid residues for anchor alignment")
    p, q = anchor[mask], target[mask]
    pc = p - p.mean(0, keepdim=True)
    qc = q - q.mean(0, keepdim=True)
    R = _kabsch_rotation(pc, qc)
    t = q.mean(0, keepdim=True) - (p.mean(0, keepdim=True) @ R)
    return R, t.squeeze(0)


def apply_rigid(x: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply (R,t): x:[...,L,3] -> x @ R + t.  Matches _kabsch_rotation convention."""
    return x @ R + t


def common_anchor_rmsd(pred: torch.Tensor, anchor: torch.Tensor,
                       target: torch.Tensor, mask: torch.Tensor) -> float:
    """Primary displacement error: apply the real anchor->target transform.

    The fitted transform never depends on a prediction. All samples, the oracle,
    and persistence for a window therefore use the same coordinate gauge.
    """
    mask = mask.bool()
    if int(mask.sum()) < 3:
        return float("nan")
    R, t = anchor_align_rotation(anchor, target, mask)
    err = apply_rigid(pred, R, t)[mask] - target[mask]
    return float(err.square().sum(-1).mean().sqrt())


def displacement_msd(pred: torch.Tensor, anchor: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean squared sample displacement; excludes invalid residues."""
    mask = mask.bool()
    if not bool(mask.any()):
        return float("nan")
    return float((pred[mask] - anchor[mask]).square().sum(-1).mean())


def calibrate_observed_anchor_pose(pred: torch.Tensor, decoded_anchor: torch.Tensor,
                                    observed_anchor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fix decoder's arbitrary canonical pose using history only.

    Inputs pred:[B,...,3], anchor clouds:[B,L,3], mask:[B,L]. Fit one
    decoded-current-H -> real-last-observed-anchor transform per protein, then
    apply it unchanged to every predicted point. No future coordinates enter.
    Transform fitting is detached; gradients can flow through forecast points.
    """
    if pred.shape[0] != decoded_anchor.shape[0] or observed_anchor.shape != decoded_anchor.shape:
        raise ValueError("pose calibration requires matching batch and anchor clouds")
    from protein_jepa_frame.models.stage_c.geometry import fit_history_pose
    rotation, translation = fit_history_pose(decoded_anchor,observed_anchor,mask)
    outputs = []
    for b in range(pred.shape[0]):
        outputs.append(pred[b] @ rotation[b] + translation[b])
    return torch.stack(outputs)


def physical_distance_energy_score(samples: torch.Tensor, target: torch.Tensor,
                                   mask: torch.Tensor, *, scale_angstrom: float = 1.0,
                                   max_pairs: int = 512, return_tensor: bool = False):
    """Energy score of the empirical CA-distance forecast (lower is better).

    Uses nonlocal pairs (sequence separation >=3), fixed Angstrom scaling, and
    RMS Euclidean norm. No target-derived normalization or best-of-M selection.
    The second term is the V-statistic for the actual empirical distribution,
    including its diagonal; M=1 reduces to observable-space prediction error.
    This measures distance marginals jointly, not full path dynamics.
    """
    if samples.ndim != 3 or samples.shape[1:] != target.shape:
        raise ValueError("samples must be [M,L,3] and target [L,3]")
    if samples.shape[0] < 1 or scale_angstrom <= 0 or max_pairs < 1:
        raise ValueError("positive sample count, scale and pair budget required")
    valid = mask.bool()
    i, j = torch.triu_indices(target.shape[0], target.shape[0], offset=3, device=target.device)
    keep = valid[i] & valid[j]
    i, j = i[keep], j[keep]
    if not i.numel():
        return float("nan")
    if i.numel() > max_pairs:
        selected = torch.linspace(0, i.numel()-1, max_pairs, device=i.device).long()
        i, j = i[selected], j[selected]
    forecast = (samples[:, i] - samples[:, j]).norm(dim=-1) / scale_angstrom
    observed = (target[i] - target[j]).norm(dim=-1) / scale_angstrom
    norm = forecast.shape[-1] ** 0.5
    accuracy = (forecast - observed).norm(dim=-1).mean() / norm
    spread = torch.cdist(forecast, forecast).mean() / norm
    score = accuracy - 0.5 * spread
    return score if return_tensor else float(score)


def ca_geometry_summary(ca, mask, *, chain_id=None, residue_index=None):
    """CA-only validity diagnostics; coordinate means need not be conformations.

    Bond range 3.3--4.3 A and nonlocal clash cutoff 2.5 A are fixed diagnostic
    conventions, not a substitute for full backbone stereochemistry.
    """
    mask = mask.bool()
    L = len(ca)
    idx = torch.arange(L,device=ca.device) if residue_index is None else residue_index
    chain = torch.zeros(L,device=ca.device,dtype=torch.long) if chain_id is None else chain_id
    adjacent = mask[1:] & mask[:-1] & (chain[1:]==chain[:-1]) & (idx[1:]-idx[:-1]==1)
    lengths = (ca[1:]-ca[:-1]).norm(dim=-1)[adjacent]
    i,j = torch.triu_indices(L,L,offset=1,device=ca.device)
    nonlocal_pair = mask[i]&mask[j]&((chain[i]!=chain[j])|((idx[i]-idx[j]).abs()>=3))
    distances = (ca[i[nonlocal_pair]]-ca[j[nonlocal_pair]]).norm(dim=-1)
    return dict(ca_bond_mean_angstrom=float(lengths.mean()) if len(lengths) else None,
                ca_bond_invalid_fraction=float(((lengths<3.3)|(lengths>4.3)).float().mean()) if len(lengths) else None,
                ca_nonlocal_clash_fraction=float((distances<2.5).float().mean()) if len(distances) else None,
                ca_radius_gyration_angstrom=float((ca[mask]-ca[mask].mean(0)).square().sum(-1).mean().sqrt()))


def ca_observable_errors(pred, target, mask, *, chain_id=None, residue_index=None):
    """Internal distance/contact and signed CA pseudo-torsion endpoint errors."""
    L = len(target)
    i,j = torch.triu_indices(L,L,offset=3,device=target.device)
    good = mask[i]&mask[j]
    i,j = i[good],j[good]
    if not len(i):
        raise ValueError('no valid nonlocal residue pairs')
    pd=(pred[i]-pred[j]).norm(dim=-1)
    td=(target[i]-target[j]).norm(dim=-1)
    keep=mask[:-3]&mask[1:-2]&mask[2:-1]&mask[3:]
    if chain_id is not None:
        same=chain_id[1:]==chain_id[:-1]
        keep &= same[:-2]&same[1:-1]&same[2:]
    if residue_index is not None:
        contiguous=residue_index[1:]-residue_index[:-1]==1
        keep &= contiguous[:-2]&contiguous[1:-1]&contiguous[2:]
    def torsion(x):
        a,b,c=x[1:-2]-x[:-3],x[2:-1]-x[1:-2],x[3:]-x[2:-1]
        unit=b/b.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        u=a-(a*unit).sum(-1,keepdim=True)*unit
        v=c-(c*unit).sum(-1,keepdim=True)*unit
        return torch.atan2((torch.cross(unit,u,dim=-1)*v).sum(-1),(u*v).sum(-1))
    angle=torsion(pred)-torsion(target)
    return dict(distance_rmse_angstrom=float((pd-td).square().mean().sqrt()),
                contact_error_fraction=float(((pd<8)!=(td<8)).float().mean()),
                ca_pseudotorsion_mae_rad=float(torch.atan2(angle.sin(),angle.cos())[keep].abs().mean()) if keep.any() else None)


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
