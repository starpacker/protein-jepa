"""Backbone (N, CA, C, O) atom-level metrics (NEW, 2026-10-21).

Complements the Cα-only ensemble metrics in ``ensemble_metrics*.py``.  Given a
predicted/true *mainchain* tensor ``bb [.., L, 4, 3]`` in the (N, CA, C, O) order
(the same order the Stage C decoder / PDB exporter use), this module computes:

    * backbone RMSD (N/CA/C/O) under a common anchor gauge,
    * per-atom-type RMSD (N, CA, C, O),
    * backbone bond-length statistics and validity fractions,
    * backbone bond-angle statistics,
    * φ/ψ backbone dihedrals and Ramachandran region fractions,
    * ω (peptide-bond) trans fraction,
    * peptide C(i)-N(i+1) bond validity,
    * nonlocal backbone clash fraction.

All are mask / chain / residue-index aware and return plain floats plus
``n_valid``/``n_total`` counts so a table can be traced.  None of these replace
full stereochemistry (sidechains are excluded by construction); they are the
Cα-adjacent backbone validity checks the report's §2.2 asks for.

Design mirrors ``evaluation/ensemble_metrics.py``: per-window scalars, no
binning for the distributional quantities, exact tensors on the input device.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# Ideal backbone geometry (Å / rad); OpenFold literature values.
IDEAL_BOND = {"N_CA": 1.458, "CA_C": 1.525, "C_N": 1.329, "C_O": 1.231}
BOND_TOL = {"N_CA": 0.05, "CA_C": 0.05, "C_N": 0.05, "C_O": 0.03}
IDEAL_ANGLE = {"N_CA_C": math.radians(111.0), "CA_C_N": math.radians(116.2),
               "C_N_CA": math.radians(121.7)}
ANGLE_TOL = 0.20  # rad (~11 deg)

# Ramachandran "generously allowed" box (deg) — a hard, documented convention,
# not a substitute for a per-structure reference distribution.
_RA_ALPHA = (-160.0, -20.0, -90.0, 60.0)   # (phi_lo, phi_hi, psi_lo, psi_hi) α
_RA_BETA = (-180.0, -40.0, 90.0, 180.0)    # β / PPII
_RA_LEFT = (0.0, 180.0, -180.0, 180.0)     # left-handed (broad)


def _nmask(mask, *shape_leading):
    return mask if mask is not None else torch.ones(*shape_leading, dtype=torch.bool)


def _bond(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=-1)


def _angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Angle at ``b`` (radians), shape [.., N]."""
    ba = a - b
    bc = c - b
    ba = ba / ba.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    bc = bc / bc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cosv = (ba * bc).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cosv)


def _dihedral(p0, p1, p2, p3) -> torch.Tensor:
    """Signed dihedral (radians) about p1-p2, shape [.., N] (standard IUPAC)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    x = (v * w).sum(-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


def _valid_chain(mask: torch.Tensor, chain_id: Optional[torch.Tensor],
                 residue_index: Optional[torch.Tensor], sep: int = 1):
    """Adjacent-residue validity for step ``sep`` within the same chain."""
    L = mask.shape[-1]
    ok = mask[..., sep:] & mask[..., :-sep]
    if chain_id is not None:
        ok = ok & (chain_id[..., sep:] == chain_id[..., :-sep])
    if residue_index is not None:
        ok = ok & ((residue_index[..., sep:] - residue_index[..., :-sep]) == sep)
    return ok


def _ramachandran_frac(phi_deg: torch.Tensor, psi_deg: torch.Tensor) -> dict:
    def inbox(box):
        plo, phi_, slo, shi = box
        return ((phi_deg >= plo) & (phi_deg <= phi_) &
                (psi_deg >= slo) & (psi_deg <= shi))
    allowed = inbox(_RA_ALPHA) | inbox(_RA_BETA) | inbox(_RA_LEFT)
    return {"ra_allowed_frac": float(allowed.double().mean()) if allowed.numel() else None,
            "n_residues": int(allowed.numel())}


def backbone_geometry_summary(
    bb: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    chain_id: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
    clash_cutoff: float = 3.0,
) -> dict:
    """Stereochemistry validity of a backbone tensor ``bb [.., L, 4, 3]``.

    ``bb[..., 0]`` = N, ``[...,1]`` = CA, ``[...,2]`` = C, ``[...,3]`` = O.
    Returns a flat dict of floats (+ ``n_*`` counts).
    """
    if bb.shape[-2] != 4:
        raise ValueError("backbone tensor must be [.., L, 4, 3] (N, CA, C, O)")
    mask = _nmask(mask, *bb.shape[:-2])
    N, CA, C, O = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :], bb[..., 3, :]
    out: dict[str, float] = {}

    # Bonds
    b_nca = _bond(N, CA)
    b_cac = _bond(CA, C)
    b_co = _bond(C, O)
    ok = mask & torch.roll(mask, -1, dims=-1)
    out["bond_N_CA_mean"] = float(b_nca[mask].mean()) if mask.any() else None
    out["bond_CA_C_mean"] = float(b_cac[mask].mean()) if mask.any() else None
    out["bond_C_O_mean"] = float(b_co[mask].mean()) if mask.any() else None
    out["bond_N_CA_valid_frac"] = float(((b_nca - IDEAL_BOND["N_CA"]).abs() < BOND_TOL["N_CA"])[mask].double().mean()) if mask.any() else None
    out["bond_CA_C_valid_frac"] = float(((b_cac - IDEAL_BOND["CA_C"]).abs() < BOND_TOL["CA_C"])[mask].double().mean()) if mask.any() else None
    out["bond_C_O_valid_frac"] = float(((b_co - IDEAL_BOND["C_O"]).abs() < BOND_TOL["C_O"])[mask].double().mean()) if mask.any() else None

    # Peptide bond C(i)-N(i+1)
    adj = _valid_chain(mask, chain_id, residue_index, 1)
    if adj.any():
        cn = _bond(C[..., :-1, :], N[..., 1:, :])
        out["bond_C_N_mean"] = float(cn[adj].mean())
        out["bond_C_N_valid_frac"] = float(((cn - IDEAL_BOND["C_N"]).abs() < BOND_TOL["C_N"])[adj].double().mean())
    else:
        out["bond_C_N_mean"] = None
        out["bond_C_N_valid_frac"] = None

    # Angles (N-CA-C, CA-C-N, C-N-CA)
    ang_ncac = _angle(N, CA, C)
    out["angle_N_CA_C_mean_deg"] = float(torch.rad2deg(ang_ncac)[mask].mean()) if mask.any() else None
    out["angle_N_CA_C_valid_frac"] = float(((ang_ncac - IDEAL_ANGLE["N_CA_C"]).abs() < ANGLE_TOL)[mask].double().mean()) if mask.any() else None
    adj3 = _valid_chain(mask, chain_id, residue_index, 2)
    if adj3.any():
        # CA-C-N: needs residues i and i+1 for (CA_i, C_i, N_{i+1})
        ok1 = _valid_chain(mask, chain_id, residue_index, 1)
        acn = _angle(CA[..., :-1, :], C[..., :-1, :], N[..., 1:, :])
        out["angle_CA_C_N_mean_deg"] = float(torch.rad2deg(acn)[ok1].mean())
        out["angle_CA_C_N_valid_frac"] = float(((acn - IDEAL_ANGLE["CA_C_N"]).abs() < ANGLE_TOL)[ok1].double().mean())
        # C-N-CA: (C_{i-1}, N_i, CA_i)
        cnc = _angle(C[..., :-1, :], N[..., 1:, :], CA[..., 1:, :])
        out["angle_C_N_CA_mean_deg"] = float(torch.rad2deg(cnc)[ok1].mean())
        out["angle_C_N_CA_valid_frac"] = float(((cnc - IDEAL_ANGLE["C_N_CA"]).abs() < ANGLE_TOL)[ok1].double().mean())
    else:
        for k in ("angle_CA_C_N_mean_deg", "angle_CA_C_N_valid_frac",
                  "angle_C_N_CA_mean_deg", "angle_C_N_CA_valid_frac"):
            out[k] = None

    # phi (C_{i-1}-N_i-CA_i-C_i), psi (N_i-CA_i-C_i-N_{i+1}), omega (CA_{i-1}-C_{i-1}-N_i-CA_i)
    phis, psis, omegas = [], [], []
    if mask.shape[-1] > 3:
        # phi for residues 1..L-1 (needs i-1..i) -> use valid-chain sep 1 shifted
        valid_phi = _valid_chain(mask, chain_id, residue_index, 1)  # between i-1,i
        phi = _dihedral(C[..., :-1, :], N[..., 1:, :], CA[..., 1:, :], C[..., 1:, :])
        phis = phi[valid_phi]
        if mask.shape[-1] > 2:
            valid_psi = _valid_chain(mask, chain_id, residue_index, 1)  # between i,i+1
            psi = _dihedral(N[..., :-1, :], CA[..., :-1, :], C[..., :-1, :], N[..., 1:, :])
            psis = psi[valid_psi]
            omega = _dihedral(CA[..., :-1, :], C[..., :-1, :], N[..., 1:, :], CA[..., 1:, :])
            omegas = omega[valid_psi]
    if len(phis):
        phid = torch.rad2deg(phis)
        psid = torch.rad2deg(psis)
        out.update(_ramachandran_frac(phid, psid))
        out["n_dihedral_pairs"] = int(phid.numel())
    else:
        out.update({"ra_allowed_frac": None, "n_residues": 0, "n_dihedral_pairs": 0})
    if len(omegas):
        # trans = |omega| within 30 deg of 180
        tr = (omegas.abs() > math.radians(150.0))
        out["omega_trans_frac"] = float(tr.double().mean())
        out["omega_n"] = int(omegas.numel())
    else:
        out["omega_trans_frac"] = None
        out["omega_n"] = 0

    # Nonlocal backbone clash (CA-CA nonlocal, same convention as ca_geometry_summary)
    L = mask.shape[-1]
    i, j = torch.triu_indices(L, L, offset=1, device=bb.device)
    nonlocal_pair = mask[..., i] & mask[..., j]
    if chain_id is not None:
        nonlocal_pair = nonlocal_pair & (chain_id[..., i] != chain_id[..., j])
    if residue_index is not None:
        nonlocal_pair = nonlocal_pair & ((residue_index[..., i] - residue_index[..., j]).abs() >= 3)
    ca_d = (CA[..., i, :] - CA[..., j, :]).norm(dim=-1)
    if nonlocal_pair.any():
        out["ca_nonlocal_clash_frac"] = float((ca_d[nonlocal_pair] < 2.5).double().mean())
    else:
        out["ca_nonlocal_clash_frac"] = None
    return out


def _kabsch_align(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor):
    """Kabsch-superpose pred onto ref using masked atoms.

    Returns ``(aligned_pred, centered_ref)``: both are centroid-free, so a
    residual RMSD is ``(aligned_pred - centered_ref).norm()``.  Computed in
    float64 for numerical stability.
    """
    p = pred.to(torch.float64)
    r = ref.to(torch.float64)
    m = mask[..., None].to(torch.float64)
    pc = p - (p * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp_min(1.0)
    qc = r - (r * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True).clamp_min(1.0)
    H = (pc * m).mT @ qc
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.mT @ U.mT))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=-1))
    R = Vt.mT @ D @ U.mT
    return pc @ R.mT, qc


def backbone_rmsd(pred_bb: torch.Tensor, ref_bb: torch.Tensor,
                  mask: torch.Tensor | None = None,
                  atom_sel: torch.Tensor | None = None) -> float:
    """Kabsch backbone RMSD between pred and ref ``[..., L, 4, 3]``.

    ``atom_sel`` is an optional [4] bool mask over (N, CA, C, O); default all.
    Returns a float (Å).  The alignment uses exactly the selected atoms.
    """
    if pred_bb.shape[-2] != 4 or ref_bb.shape[-2] != 4:
        raise ValueError("expected [..., L, 4, 3] backbones")
    L = pred_bb.shape[-3]
    if mask is None:
        mask = torch.ones(pred_bb.shape[:-2], dtype=torch.bool, device=pred_bb.device)
    if atom_sel is None:
        atom_sel = torch.ones(4, dtype=torch.bool, device=pred_bb.device)
    # Flatten (L, 4) atoms into a single point set for Kabsch.
    sel = atom_sel.bool()
    p = pred_bb[..., sel, :].reshape(*pred_bb.shape[:-3], -1, 3)
    r = ref_bb[..., sel, :].reshape(*ref_bb.shape[:-3], -1, 3)
    m = mask[..., :, None].expand(*mask.shape, 4)[..., sel].reshape(*mask.shape[:-1], -1)
    aligned, ref_c = _kabsch_align(p, r, m)
    d = (aligned - ref_c).norm(dim=-1)
    mm = m
    return float((d * mm).sum() / mm.sum().clamp_min(1).item())


def backbone_per_atom_rmsd(pred_bb: torch.Tensor, ref_bb: torch.Tensor,
                           mask: torch.Tensor | None = None) -> dict:
    """Per-atom-type N/CA/C/O RMSD (Å) under a shared all-atom Kabsch alignment."""
    if mask is None:
        mask = torch.ones(pred_bb.shape[:-2], dtype=torch.bool, device=pred_bb.device)
    L = pred_bb.shape[-3]
    p = pred_bb.reshape(*pred_bb.shape[:-3], -1, 3)
    r = ref_bb.reshape(*ref_bb.shape[:-3], -1, 3)
    m = mask[..., :, None].expand(*mask.shape, 4).reshape(*mask.shape[:-1], -1)
    aligned, ref_c = _kabsch_align(p, r, m)
    d = (aligned - ref_c).norm(dim=-1).reshape(*mask.shape, 4)
    names = ("N", "CA", "C", "O")
    out = {}
    for k, nm in enumerate(names):
        mk = mask
        if mk.any():
            out[f"rmsd_{nm}"] = float(d[..., k][mk].mean())
        else:
            out[f"rmsd_{nm}"] = None
    return out


__all__ = [
    "backbone_geometry_summary",
    "backbone_rmsd",
    "backbone_per_atom_rmsd",
    "IDEAL_BOND",
    "IDEAL_ANGLE",
]
