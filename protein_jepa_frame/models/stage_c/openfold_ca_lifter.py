"""Cα -> full mainchain (N, CA, C, O) lifting, delegated to OpenFold geometry.

Motivation
----------
Model 2 (from-none GLE) predicts only a Cα trace (``ca_pred [.., L, 3]``); it has
no per-residue frames or torsions, so it cannot call the OpenFold frame algebra
directly.  To give it a real mainchain *without* inventing our own ideal bond /
angle tables, this module lifts a Cα trace to N/CA/C/O using **only** OpenFold's
own literature backbone geometry:

  1. Build a per-residue backbone frame ``(R, t=CA)`` from the Cα *triplet*
     (CA_{i-1}, CA_i, CA_{i+1}) with a Gram-Schmidt construction whose axis
     convention matches :func:`RigidBodyFrame.from_backbone` (origin at CA,
     x-axis ~ CA->C/CA_{i+1}, y perpendicular to the local chain plane).
  2. Read the *ideal* backbone atom offsets **directly from OpenFold**
     (``residue_constants.restype_atom14_rigid_group_positions`` group 0):
     N = (-0.572, 1.337, 0), C = (1.517, 0, 0), O = (0.626, 1.062, 0) at the
     CA-origin frame — i.e. the N/CA/C/O geometry is OpenFold's, not ours.
  3. Hand N/CA/C back to OpenFold's ``Rigid.make_transform_from_reference`` and
     ``frames_to_atom37`` so the *same* authoritative placement pipeline used by
     Model 1 is exercised (sidechains come out as the default rotamers).

Because step 3 rebuilds the frame from (N, CA, C) via OpenFold itself, the output
is by construction a valid OpenFold mainchain; the lifter only chooses *where* N
and C sit relative to the Cα trace.  The O atom is placed by OpenFold's frame
algebra, not by us.

Caveat (documented, not hidden): a Cα-only trace does not determine φ/ψ, so the
lifted backbone reproduces **ideal internal geometry** (correct bond lengths /
angles, planar peptide), not the true secondary structure.  It is appropriate for
stereochemistry-validity and backbone-atom-count work, and its Cα column is
*identical* to the input Cα.  For accurate φ/ψ use Model 1 (which predicts
torsions) or a trained backbone head.
"""
from __future__ import annotations

import torch

from protein_jepa_frame import vendored as _vendored  # noqa: F401 (registers openfold)
from openfold.np import residue_constants as rc

# Ideal backbone offsets in the CA-origin frame, read from OpenFold (group 0).
# ``restype_atom14_rigid_group_positions[restype, group=0]`` holds the
# N/CA/C/O literature positions; these are identical for every residue type.
_GLY = rc.restype_order["G"]
_GLY14 = rc.restype_name_to_atom14_names["GLY"]            # N, CA, C, O, then blanks
_GLY_POS = rc.restype_atom14_rigid_group_positions[_GLY]   # [14,3]
_SLOT = {name: i for i, name in enumerate(_GLY14)}
N_LOCAL = torch.tensor(_GLY_POS[_SLOT["N"]])   # (-0.572, 1.337, 0)
CA_LOCAL = torch.tensor(_GLY_POS[_SLOT["CA"]])  # (0,0,0)
C_LOCAL = torch.tensor(_GLY_POS[_SLOT["C"]])   # (1.517, 0, 0)

# Ideal Cα–Cα virtual bond length (Å) used only to extrapolate the terminal
# residues' frames (where a triplet is unavailable).
_CA_CA = 3.8


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v / n


def _frames_from_ca_trace(ca: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-residue frames from a Cα trace.

    Args:
        ca: [.., L, 3] Cα coordinates.

    Returns:
        R: [.., L, 3, 3] rotations, t: [.., L, 3] translations (= Cα positions).
    """
    L = ca.shape[-2]
    # Local x-axis: direction to the NEXT Cα (approximates Cα->C / chain tangent),
    # backward difference at the C-terminus.
    nxt = torch.empty_like(ca)
    if L > 1:
        nxt[..., :-1, :] = ca[..., 1:, :]
        nxt[..., -1, :] = ca[..., -1, :] + (ca[..., -1, :] - ca[..., -2, :])
    else:
        nxt = ca + torch.tensor([1.0, 0.0, 0.0], dtype=ca.dtype, device=ca.device)
    prv = torch.empty_like(ca)
    if L > 1:
        prv[..., 1:, :] = ca[..., :-1, :]
        prv[..., 0, :] = ca[..., 0, :] - (ca[..., 1, :] - ca[..., 0, :])
    else:
        prv = ca - torch.tensor([1.0, 0.0, 0.0], dtype=ca.dtype, device=ca.device)

    x_axis = _unit(nxt - ca)                       # chain tangent
    # y-axis: perpendicular to the local plane spanned by (prv->ca, ca->nxt).
    a = _unit(ca - prv)
    y_axis = _unit(torch.cross(x_axis, a, dim=-1))
    z_axis = _unit(torch.cross(x_axis, y_axis, dim=-1))
    y_axis = _unit(torch.cross(z_axis, x_axis, dim=-1))   # re-orthogonalise
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)     # columns = axes
    # Fallback for degenerate (collinear) triplets: identity rotation.
    bad = ~torch.isfinite(R).all(dim=(-1, -2)) | (R.norm(dim=-2).min(dim=-1).values < 0.5)
    if bad.any():
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        R = torch.where(bad.unsqueeze(-1).unsqueeze(-1), eye.expand_as(R), R)
    return R, ca


def lift_ca_to_backbone(
    ca: torch.Tensor,
    *,
    aatype: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift a Cα trace to a full OpenFold mainchain.

    Args:
        ca: [.., L, 3] Cα coordinates.
        aatype: optional [.., L] residue indices (OpenFold order) for sidechain
            placement; if None, only N/CA/C/O are meaningful.

    Returns:
        backbone: [.., L, 4, 3] (N, CA, C, O) in OpenFold atom37 semantics.
        atom37:   [.., L, 37, 3] full heavy atoms (OpenFold ordering), or None if
                  ``aatype`` is None.
        atom37_mask: [.., L, 37] presence mask, or None.

    The Cα column of ``backbone`` is *exactly* the input ``ca``.
    """
    if ca.shape[-1] != 3:
        raise ValueError("ca must be [.., L, 3]")
    R, t = _frames_from_ca_trace(ca)                     # [.., L,3,3], [.., L,3]
    dtype, device = ca.dtype, ca.device
    N_local = N_LOCAL.to(dtype=dtype, device=device)
    # Place N and C at OpenFold's ideal offsets in the Cα frame.
    def place(local):
        return (R @ local.view(*([1] * (R.dim() - 3)), 3, 1)).squeeze(-1) + t
    N = place(N_local)
    C = place(C_LOCAL.to(dtype=dtype, device=device))

    # Hand N/CA/C to OpenFold's authoritative frame builder + atom placer so the
    # backbone is reconstructed through the SAME pipeline Model 1 uses.  The
    # returned N/CA/C are re-derived by OpenFold from (N, CA, C), and O from the
    # frame algebra (not from our local offset), giving an OpenFold-consistent O.
    from protein_jepa_frame.models.stage_c.openfold_atom37 import (
        frames_to_atom37, backbone_frame_from_coords,
    )
    rigid = backbone_frame_from_coords(N, ca, C)
    tort = None
    if aatype is None:
        # Backbone-only: reconstruct with a GLY/UNK typing so N/CA/C/O are placed.
        aa = torch.full(ca.shape[:-1], rc.restype_order["G"], dtype=torch.long, device=device)
    else:
        aa = aatype
    atom37, atom37_mask = frames_to_atom37(
        rigid.get_rots().get_rot_mats(), rigid.get_trans(), aa, tort)
    # OpenFold atom ordering: N=0, CA=1, C=2, O=4.
    backbone = torch.stack([atom37[..., 0, :], atom37[..., 1, :],
                            atom37[..., 2, :], atom37[..., 4, :]], dim=-2)
    if aatype is None:
        return backbone, None, None
    return backbone, atom37, atom37_mask


if __name__ == "__main__":  # tiny self-check
    import numpy as np
    torch.manual_seed(0)
    ca = torch.cumsum(torch.randn(60, 3) * 0.5 + torch.tensor([3.8, 0.0, 0.0]), dim=0)
    bb, _, _ = lift_ca_to_backbone(ca[None])
    print("backbone", tuple(bb.shape))
    print("CA err:", (bb[0, :, 1] - ca).abs().max().item())
    print("N-CA bond mean:", (bb[0, :, 0] - bb[0, :, 1]).norm(dim=-1).mean().item())
    print("CA-C bond mean:", (bb[0, :, 2] - bb[0, :, 1]).norm(dim=-1).mean().item())
    print("C-O bond mean:", (bb[0, :, 3] - bb[0, :, 2]).norm(dim=-1).mean().item())
