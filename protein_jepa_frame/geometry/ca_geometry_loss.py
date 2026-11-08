"""Differentiable Cα local-geometry losses on DECODED coordinates.

Purpose (spec §7.5.2): the v3 ensemble objective scores only terminal-cloud
statistics (RMSD / Rg / contact / distance histograms), so the decoder is free
to map widely-different H onto the same average CA shape, and the GLE rollout is
free to move H off the manifold the decoder can invert.  Both defects manifest
as collapsed local geometry (adjacent CA-CA distances << 3.8 Å) that the global
metrics cannot see.

These losses act directly on the *decoded* CA coordinates ``[..., L, 3]``, are
fully differentiable, and mask out chain breaks using the true residue topology
(``residue_index`` / ``chain_id``), never "array adjacency".  They are meant to be
added to both

  * the static decoder round-trip ``Enc(X_t) -> Dec -> X_t`` (fixes decoder
    variance / posterior-mean collapse), and
  * the decoded free-rollout frames at t = 1, 2, 4, 8, ... (fixes the GLE
    train/inference mismatch, keeps H on the decoder manifold).

Reference values are *calibrated* to real MD geometry (bond ≈ 3.8 Å, pseudo-angle
≈ a broad distribution centred near ~90-120°, clash threshold ≈ 3.5 Å), NOT
single ideal angles, so a well-formed but flexible chain is not forced onto one
rigid pose.  See the spec §7.5.2 "不要把所有残基强迫到同一参考角/扭转角".
"""

from __future__ import annotations

import torch


def _chain_bond_mask(
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
) -> torch.Tensor:
    """Boolean [.., L-1] mask of TRUE consecutive peptide bonds.

    A bond (i, i+1) is real iff both residues are valid, in the same chain, and
    their residue indices differ by exactly 1 (so a missing residue / chain
    break at a gap never forms a spurious bond).  Broadcasting-safe over any
    leading sample dims.
    """
    valid = residue_mask.bool()
    same_chain = chain_id[..., :-1] == chain_id[..., 1:]
    consecutive = (residue_index[..., 1:] - residue_index[..., :-1]) == 1
    return valid[..., :-1] & valid[..., 1:] & same_chain & consecutive


def _bond_vectors(ca: torch.Tensor) -> torch.Tensor:
    """CA-CA bond vectors ``ca[..., 1:, :] - ca[..., :-1, :]`` -> ``[..., L-1, 3]``."""
    return ca[..., 1:, :] - ca[..., :-1, :]


def ca_bond_loss(
    ca: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    d_ref: float = 3.8,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Soft penalty on the deviation of CA-CA bond lengths from ``d_ref``.

    ``ca``: [..., L, 3] decoded CA coordinates (Å).
    Uses a Huber (smooth-L1) profile so well-formed bonds cost ~0 while a
    collapsed 2.4 Å bond contributes a strong, well-scaled gradient.
    """
    v = _bond_vectors(ca)                              # [..., L-1, 3]
    # Squared-distance penalty is differentiable everywhere (||v|| at v=0 has an
    # undefined gradient, which would make a fully-coincident collapsed pair --
    # exactly the failure mode at t=32, resid 28-29 at 0.06 Å -- invisible to the
    # loss).  Penalise (||v||^2 - d_ref^2) so the gradient stays finite as d -> 0.
    d2 = (v * v).sum(dim=-1)                           # [..., L-1]
    bond_mask = _chain_bond_mask(residue_mask, residue_index, chain_id)
    err = d2 - (d_ref * d_ref)
    loss = _huber(err, huber_delta * (2.0 * d_ref))
    n = bond_mask.sum().clamp_min(1)
    return (loss * bond_mask.float()).sum() / n


def ca_pseudo_angle_loss(
    ca: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    angle_ref_deg: float = 110.0,
    kappa: float = 0.5,
) -> torch.Tensor:
    """Soft penalty on CA pseudo-bond angles (i-1, i, i+1).

    The reference is a *single broad* centre; the ``kappa`` weight is small so a
    genuinely flexible loop / secondary-structure transition is not forced flat.
    We penalise only severe deviations via ``1 - cos(angle - angle_ref)``, which
    is smooth, periodic, and saturates (never explodes) for large angles.
    """
    v1 = ca[..., :-2, :] - ca[..., 1:-1, :]            # r_{i-1} - r_i
    v2 = ca[..., 2:, :] - ca[..., 1:-1, :]             # r_{i+1} - r_i
    n1 = v1.norm(dim=-1).clamp_min(1e-6)
    n2 = v2.norm(dim=-1).clamp_min(1e-6)
    cosang = (v1 * v2).sum(-1) / (n1 * n2)             # [..., L-2]
    ang = torch.acos(cosang.clamp(-1.0, 1.0))
    ref = torch.tensor(angle_ref_deg, dtype=ca.dtype, device=ca.device).deg2rad()
    dtheta = ang - ref
    loss = 1.0 - torch.cos(dtheta)                     # bounded [0, 2]
    # angle valid iff its three residues form two real consecutive bonds
    valid = (
        residue_mask.bool()[..., :-2]
        & residue_mask.bool()[..., 1:-1]
        & residue_mask.bool()[..., 2:]
        & (chain_id[..., :-2] == chain_id[..., 1:-1])
        & (chain_id[..., 1:-1] == chain_id[..., 2:])
        & ((residue_index[..., 1:-1] - residue_index[..., :-2]) == 1)
        & ((residue_index[..., 2:] - residue_index[..., 1:-1]) == 1)
    )
    n = valid.sum().clamp_min(1)
    return kappa * (loss * valid.float()).sum() / n


def ca_clash_loss(
    ca: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    clash_cutoff: float = 3.5,
) -> torch.Tensor:
    """Penalty on nonlocal CA-CA pairs closer than ``clash_cutoff`` (Å).

    Only pairs separated by >= 4 along the true chain (or in different chains)
    are considered, matching the spec's "exclude normal local sequence atoms".
    A differentiable softplus barrier turns on below the cutoff.
    """
    D = torch.cdist(ca, ca)                             # [..., L, L]
    valid = residue_mask.bool()
    pair = valid[..., :, None] & valid[..., None, :]
    sep = (residue_index[..., :, None] - residue_index[..., None, :]).abs()
    same = chain_id[..., :, None] == chain_id[..., None, :]
    nonlocal_mask = pair & ((sep >= 4) | ~same)
    triu = torch.triu(torch.ones_like(nonlocal_mask), diagonal=1).bool()
    mask = nonlocal_mask & triu
    barrier = torch.nn.functional.softplus(clash_cutoff - D, beta=10.0)
    n = mask.sum().clamp_min(1)
    return (barrier * mask.float()).sum() / n


def ca_geometry_loss(
    ca: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    bond_weight: float = 1.0,
    angle_weight: float = 0.2,
    clash_weight: float = 0.1,
    d_ref: float = 3.8,
) -> dict[str, torch.Tensor]:
    """Aggregate local-geometry penalty on decoded CA, returning per-term dict.

    Returns a dict with ``bond``, ``angle``, ``clash`` and their weighted
    ``total`` so callers can log each term separately and choose weights on the
    validation set (spec §7.5.3.3).
    """
    bond = ca_bond_loss(ca, residue_mask, residue_index, chain_id, d_ref=d_ref)
    angle = ca_pseudo_angle_loss(ca, residue_mask, residue_index, chain_id)
    clash = ca_clash_loss(ca, residue_mask, residue_index, chain_id)
    total = bond_weight * bond + angle_weight * angle + clash_weight * clash
    return {"bond": bond, "angle": angle, "clash": clash, "total": total}


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    absx = x.abs()
    quad = 0.5 * x * x
    lin = delta * (absx - 0.5 * delta)
    return torch.where(absx <= delta, quad, lin)
