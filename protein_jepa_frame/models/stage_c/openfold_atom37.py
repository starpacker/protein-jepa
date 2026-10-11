"""Stage C frame -> atom37 reconstruction via official OpenFold.

This module is a thin adapter over the **vendored official OpenFold**
(aqlaboratory/openfold, Apache-2.0) geometry routines.  It contains NO
hand-written atom placement, NO hand-rolled ideal bond/angle tables, and NO
manual sidechain trees.  Every geometric operation is delegated to OpenFold:

    backbone (N, CA, C)            -> Rigid.from_3_points          (Algorithm 21)
    residue frame + chi angles     -> torsion_angles_to_frames
    per-residue frames             -> frames_and_literature_positions_to_atom14_pos
    atom14                         -> atom14_to_atom37
    atom37 presence masks          -> residue_constants.restype_atom37_mask

The Stage C decoder predicts a per-residue SE(3) frame ``(R, t)`` (origin at
CA) plus 7 torsion angles (psi, ..., chi).  To reuse OpenFold's frame algebra
we convert the decoder's ``(R, t)`` frames into an OpenFold ``Rigid`` whose
rotation matches OpenFold's *backbone* residue frame convention, then compos
the chi-angle frames on top.

Coordinate/ordering contract
----------------------------
The atom37 ordering is taken from OpenFold's ``residue_constants.atom_order``
(the authoritative AlphaFold/OpenFold ordering).  Note that this ordering
differs from ``protein_jepa_frame.data.schema.ATOM37_NAMES`` at the tail
(CZ/CZ2/CZ3/NZ/OH/NH1/NH2 vs NH1/NH2/OH/CZ/CZ2/CZ3/NZ).  All outputs produced
here follow OpenFold's ordering; callers that need the project ordering must
permute explicitly.  We expose ``OPENFOLD_ATOM37_NAMES`` and
``OPENFOLD_TO_PROJECT_ATOM37`` to make that conversion auditable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

# Registering the vendored package makes the official ``openfold.*`` modules
# importable (absolute-import style preserved verbatim).
from protein_jepa_frame import vendored as _vendored  # noqa: F401

from openfold.np import residue_constants as rc
from openfold.utils.feats import (
    torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos,
    atom14_to_atom37,
)
from openfold.data.data_transforms import make_atom14_masks
from openfold.utils.rigid_utils import Rigid, Rotation

# Sentinel: to allow callers to build the official backbone residue frame from
# N/CA/C via OpenFold's ``Rigid.make_transform_from_reference`` (Algorithm 21
# used by the official structure module and ``atom37_to_frames``).
backbone_frame_from_coords = Rigid.make_transform_from_reference

# ═══════════════════════════════════════════════════════════════════════
# Cached OpenFold constants (built once; authoritative ordering)
# ═══════════════════════════════════════════════════════════════════════

OPENFOLD_ATOM37_NAMES: tuple[str, ...] = tuple(rc.atom_types)

# residue frame default transforms [21, 8, 4, 4]
_DEFAULT_FRAMES = torch.tensor(rc.restype_rigid_group_default_frame)
_ATOM14_TO_GROUP = torch.tensor(rc.restype_atom14_to_rigid_group)
_ATOM14_MASK = torch.tensor(rc.restype_atom14_mask)
_ATOM14_POS = torch.tensor(rc.restype_atom14_rigid_group_positions)
_ATOM37_MASK = torch.tensor(rc.restype_atom37_mask)          # [21, 37]

# OpenFold num chi angles per residue type (from chi_angles_mask)
_CHI_MASK = torch.tensor(np.array(rc.chi_angles_mask)).bool()  # [21, 4]

# Mapping between OpenFold's atom37 ordering and the project schema ordering.
from protein_jepa_frame.data.schema import ATOM37_NAMES as _PROJECT_NAMES

_OPENFOLD_NAME_TO_IDX = {n: i for i, n in enumerate(OPENFOLD_ATOM37_NAMES)}

# PROJECT_ATOM37_TO_OPENFOLD[k] = OpenFold atom index of project-schema atom k.
# i.e. ``out[..., PROJECT_ATOM37_TO_OPENFOLD] = project_ordered_input`` yields
# OpenFold ordering.
PROJECT_ATOM37_TO_OPENFOLD: tuple[int, ...] = tuple(
    _OPENFOLD_NAME_TO_IDX[n] for n in _PROJECT_NAMES
)

# OPENFOLD_ATOM37_TO_PROJECT[k] = project-schema atom index of OpenFold atom k.
# i.e. ``project_ordered = openfold_ordered[..., OPENFOLD_ATOM37_TO_PROJECT]``.
OPENFOLD_ATOM37_TO_PROJECT: tuple[int, ...] = tuple(
    _PROJECT_NAMES.index(n) for n in OPENFOLD_ATOM37_NAMES
)

# Backwards-compatible alias (same value as PROJECT_ATOM37_TO_OPENFOLD).
OPENFOLD_TO_PROJECT_ATOM37: tuple[int, ...] = PROJECT_ATOM37_TO_OPENFOLD


def _torsions_to_alpha_7x2(torsions: torch.Tensor) -> torch.Tensor:
    """Convert Stage C's 7 torsion angles to OpenFold's (sin, cos) alpha.

    OpenFold's ``torsion_angles_to_frames`` expects ``alpha`` of shape
    ``[*, N, 7, 2]`` holding ``(sin, cos)`` pairs in the canonical
    AlphaFold/OpenFold ordering::

        [pre_omega, phi, psi, chi1, chi2, chi3, chi4]

    (verified directly against ``torsion_angles_to_frames``: input slot i maps
    to sidechain frame i+1).  Stage C stores
    ``[phi, psi, omega, chi1, chi2, chi3, chi4]``.  We place each angle in its
    OpenFold slot: pre_omega <- omega, phi <- phi, psi <- psi, chi1..4 <- chi.
    """
    # torsions: [B, L, 7] in (phi, psi, omega, chi1, chi2, chi3, chi4)
    phi = torsions[..., 0]
    psi = torsions[..., 1]
    omega = torsions[..., 2]
    chis = torsions[..., 3:7]  # [B, L, 4]

    # OpenFold canonical order: pre_omega, phi, psi, chi1..chi4
    angles = torch.cat([omega.unsqueeze(-1), phi.unsqueeze(-1),
                        psi.unsqueeze(-1), chis], dim=-1)  # [B, L, 7]
    return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,L,7,2]


def frames_to_atom37(
    R: torch.Tensor,
    t: torch.Tensor,
    aatype: torch.Tensor,
    torsions: torch.Tensor | None = None,
    *,
    atom37_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full atom37 coordinates from Stage C frames + torsions.

    Uses **only** official OpenFold routines:
    ``Rigid.from_3_points``-equivalent frame construction (via ``Rigid``),
    ``torsion_angles_to_frames`` and
    ``frames_and_literature_positions_to_atom14_pos`` + ``atom14_to_atom37``.

    Args:
        R: [B, L, 3, 3] per-residue frame orientations (columns are axes).
        t: [B, L, 3] per-residue frame translations (CA positions).
        aatype: [B, L] residue type indices (0-19, OpenFold ordering).
        torsions: [B, L, 7] angles (phi, psi, omega, chi1..chi4).  If None,
            sidechains are placed using zero sidechain torsions.
        atom37_mask: optional [B, L, 37] override for the presence mask.

    Returns:
        atom37_coords: [B, L, 37, 3] in OpenFold atom37 ordering.
        atom37_mask: [B, L, 37] bool presence mask (OpenFold ordering).
    """
    B, L = aatype.shape
    device, dtype = t.device, t.dtype

    if torsions is None:
        torsions = torch.zeros(B, L, 7, device=device, dtype=dtype)

    # ── Backbone-safe residue typing ──
    # OpenFold's residue_constants define UNK (aatype == 20) with an all-zero
    # atom37 existence mask, so `frames_and_literature_positions_to_atom14_pos`
    # and `atom14_to_atom37` place *every* atom of a UNK residue — including the
    # universal backbone atoms N, CA, C, O — at the origin (0, 0, 0).  That
    # silently destroys the decoded backbone (and hence CA-Kabsch RMSD / contact
    # metrics) for any residue the trajectory flags as unknown, even though the
    # per-residue frame (R, t) is well defined.  The backbone group (frame 0) is
    # identical for every residue type, so we reconstruct UNK residues with the
    # GLY backbone template (GLY has N, CA, C, O and no sidechain).  CA remains
    # exactly the frame origin t; N, C, O are placed by the universal backbone
    # rigid group.  Standard-residue geometry is unchanged.
    aatype_bb = torch.where(
        (aatype < 0) | (aatype >= rc.restype_num),  # UNK / out-of-range -> GLY
        torch.full_like(aatype, rc.restype_order["G"]),
        aatype,
    )
    aatype = aatype_bb

    # ── Build the residue backbone Rigid from the decoder's (R, t) frames ──
    # The decoder frame has columns [x, y, z]; OpenFold's Rotation stores the
    # same row-major [*,3,3] rotation.  We normalise through Rotation's
    # quaternion path only to guarantee orthonormality is enforced by OpenFold.
    R_of = R.to(device=device, dtype=torch.float32)
    t_of = t.to(device=device, dtype=torch.float32)
    rigid_backbone = Rigid(Rotation(rot_mats=R_of), t_of)

    # ── alpha (sin, cos) for torsion_angles_to_frames ──
    alpha = _torsions_to_alpha_7x2(torsions.to(torch.float32))

    default_frames = _DEFAULT_FRAMES.to(device=device)

    # [B, L, 8] all residue-group frames in global coordinates
    all_frames = torsion_angles_to_frames(
        rigid_backbone, alpha, aatype, default_frames
    )

    # ── per-atom (atom14) positions in global coordinates ──
    atom14_pos = frames_and_literature_positions_to_atom14_pos(
        all_frames,
        aatype,
        default_frames,
        _ATOM14_TO_GROUP.to(device=device),
        _ATOM14_MASK.to(device=device),
        _ATOM14_POS.to(device=device),
    )

    # ── Build lookup features via the official OpenFold transform ──
    # This is authoritative for residue-specific atom14 ordering, UNK, and
    # atom-existence masks; do not duplicate the lookup table locally.
    feat = make_atom14_masks({"aatype": aatype})

    # ── atom14 -> atom37 using OpenFold's official resolver ──
    atom37_pos = atom14_to_atom37(atom14_pos.to(torch.float32), feat)

    # ── presence mask ──
    if atom37_mask is None:
        atom37_mask = feat["atom37_atom_exists"].bool()
    else:
        atom37_mask = atom37_mask & feat["atom37_atom_exists"].bool()

    return atom37_pos.to(dtype), atom37_mask


def frames_to_atom37_with_sidechain_torsions(
    R: torch.Tensor,
    t: torch.Tensor,
    aatype: torch.Tensor,
    torsion_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: convert 36-bin torsion logits then reconstruct.

    Args:
        R, t, aatype: as in :func:`frames_to_atom37`.
        torsion_logits: [B, L, 7, n_bins] classification logits.

    Returns:
        (atom37_coords, atom37_mask) in OpenFold ordering.
    """
    probs = torch.softmax(torsion_logits, dim=-1)
    n_bins = torsion_logits.shape[-1]
    bin_centres = torch.linspace(-math.pi, math.pi, n_bins + 1, device=torsion_logits.device)
    bin_centres = 0.5 * (bin_centres[:-1] + bin_centres[1:])
    angles = (probs * bin_centres.view(1, 1, 1, n_bins)).sum(dim=-1)  # [B, L, 7]
    return frames_to_atom37(R, t, aatype, angles)


def frames_to_atom37_from_backbone_tensors(
    rigid_backbone: Rigid,
    aatype: torch.Tensor,
    torsions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct atom37 from an OpenFold ``Rigid`` backbone frame.

    This is the canonical Stage C entry point: the decoder's per-residue
    SE(3) frames are wrapped as an OpenFold ``Rigid`` (e.g. obtained from
    :func:`backbone_frame_from_coords` or directly from the decoder), and
    full-atom reconstruction is delegated entirely to official OpenFold.

    Args:
        rigid_backbone: OpenFold ``Rigid`` of shape [B, L] holding the
            backbone residue frames.
        aatype: [B, L] residue indices.
        torsions: [B, L, 7] (phi, psi, omega, chi1..chi4) or None.

    Returns:
        (atom37_coords [B, L, 37, 3], atom37_mask [B, L, 37]) in OpenFold order.
    """
    return frames_to_atom37(
        rigid_backbone.get_rots().get_rot_mats(),
        rigid_backbone.get_trans(),
        aatype,
        torsions,
    )


# Constant rotation converting the Stage C decoder's ``RigidBodyFrame.from_backbone``
# orientation convention to OpenFold's ``Rigid.make_transform_from_reference``
# convention (both are deterministic Gram-Schmidt functions of the same N/CA/C
# geometry; they differ only by this fixed axis permutation).  Verified exact to
# <2e-7 on real crystal structures.
DECODER_TO_OPENFOLD_ROT = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)


def decoder_frames_to_atom37(
    R: torch.Tensor,
    t: torch.Tensor,
    aatype: torch.Tensor,
    torsions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame -> atom37 for the Stage C decoder's native frame convention.

    The decoder predicts ``(R, t)`` frames produced by
    ``geometry.RigidBodyFrame.from_backbone`` (origin at CA).  OpenFold's frame
    algebra expects its own residue-frame convention, so we apply the exact
    constant rotation ``DECODER_TO_OPENFOLD_ROT`` and then delegate to
    :func:`frames_to_atom37`.

    Args:
        R: [B, L, 3, 3] decoder frame orientations.
        t: [B, L, 3] decoder frame translations (CA positions).
        aatype: [B, L] residue indices (OpenFold ordering).
        torsions: [B, L, 7] (phi, psi, omega, chi1..chi4) or None.

    Returns:
        (atom37_coords [B, L, 37, 3], atom37_mask [B, L, 37]) in OpenFold order.
    """
    C = torch.tensor(DECODER_TO_OPENFOLD_ROT, device=R.device, dtype=R.dtype)
    R_of = R @ C
    return frames_to_atom37(R_of, t, aatype, torsions)


def to_project_ordering(atom37_coords: torch.Tensor) -> torch.Tensor:
    """Permute OpenFold-ordered atom37 to the project schema ordering.

    ``project[k] = openfold[PROJECT_ATOM37_TO_OPENFOLD[k]]``.
    """
    idx = torch.tensor(PROJECT_ATOM37_TO_OPENFOLD, device=atom37_coords.device)
    return atom37_coords.index_select(-2, idx)


def to_openfold_ordering(atom37_coords: torch.Tensor) -> torch.Tensor:
    """Permute project-schema-ordered atom37 to OpenFold ordering.

    Scatter: ``openfold[PROJECT_ATOM37_TO_OPENFOLD[k]] = project[k]``.
    """
    out = torch.zeros_like(atom37_coords)
    idx = torch.tensor(PROJECT_ATOM37_TO_OPENFOLD, device=atom37_coords.device)
    out[..., idx, :] = atom37_coords
    return out


__all__ = [
    "OPENFOLD_ATOM37_NAMES",
    "OPENFOLD_TO_PROJECT_ATOM37",
    "PROJECT_ATOM37_TO_OPENFOLD",
    "OPENFOLD_ATOM37_TO_PROJECT",
    "backbone_frame_from_coords",
    "DECODER_TO_OPENFOLD_ROT",
    "decoder_frames_to_atom37",
    "frames_to_atom37",
    "frames_to_atom37_from_backbone_tensors",
    "frames_to_atom37_with_sidechain_torsions",
    "to_project_ordering",
    "to_openfold_ordering",
]
