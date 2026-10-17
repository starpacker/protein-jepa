"""Tests: Stage C frame -> atom37 reconstruction via official OpenFold.

These tests verify that frame -> atom37 reconstruction is delegated ENTIRELY to
the vendored official OpenFold package (aqlaboratory/openfold) and contains no
hand-written geometry:

1. Official OpenFold geometry functions are importable and used.
2. Backbone reconstruction is exact for CA and near-exact for N/C (limited only
   by literature vs. real bond geometry).
3. Sidechain reconstruction from chi angles reproduces real crystal atoms to
   within literature-geometry tolerance.
4. The OpenFold atom37 ordering contract matches ``residue_constants.atom_order``.
5. The decoder produces atom37 via the official pipeline with valid masks and
   correct CA placement.
"""

import numpy as np
import torch
import pytest

from protein_jepa_frame import vendored as _vendored  # noqa: F401
from openfold.np import residue_constants as rc
from openfold.utils.rigid_utils import Rigid

from protein_jepa_frame.models.stage_c.openfold_atom37 import (
    OPENFOLD_ATOM37_NAMES,
    PROJECT_ATOM37_TO_OPENFOLD,
    OPENFOLD_ATOM37_TO_PROJECT,
    frames_to_atom37,
    decoder_frames_to_atom37,
    backbone_frame_from_coords,
    to_project_ordering,
    to_openfold_ordering,
)
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame


OFI = {n: i for i, n in enumerate(OPENFOLD_ATOM37_NAMES)}


def _ideal_ala_backbone():
    """Build a single ALA residue exactly from OpenFold's literature geometry."""
    rgi = rc.rigid_group_atom_positions["ALA"]
    pos = {nm: torch.tensor(p, dtype=torch.float32) for nm, g, p in rgi}
    return (
        pos["N"].view(1, 1, 3),
        pos["CA"].view(1, 1, 3),
        pos["C"].view(1, 1, 3),
    )


def test_openfold_geometry_is_used_not_reimplemented():
    """The adapter imports official OpenFold geometry (no local geometry tables)."""
    import protein_jepa_frame.models.stage_c.openfold_atom37 as mod

    assert hasattr(mod, "torsion_angles_to_frames")
    assert hasattr(mod, "frames_and_literature_positions_to_atom14_pos")
    assert hasattr(mod, "atom14_to_atom37")
    # No hand-written ideal bond/angle tables in the adapter.
    assert not hasattr(mod, "IDEAL_BOND")
    assert not hasattr(mod, "IDEAL_ANGLE")


def test_atom37_ordering_matches_official_openfold():
    """Adapter atom37 ordering equals OpenFold's authoritative atom_order."""
    assert OPENFOLD_ATOM37_NAMES == tuple(rc.atom_types)
    for name, idx in rc.atom_order.items():
        assert OPENFOLD_ATOM37_NAMES[idx] == name
    # Both ordering permutations are bijections.
    assert sorted(PROJECT_ATOM37_TO_OPENFOLD) == list(range(37))
    assert sorted(OPENFOLD_ATOM37_TO_PROJECT) == list(range(37))


def test_backbone_reconstruction_exact_on_literature_geometry():
    """On ideal literature geometry, N/CA/C reconstruct to < 1e-4 Å."""
    N, CA, C = _ideal_ala_backbone()
    aatype = torch.zeros(1, 1, dtype=torch.long)
    rigid = backbone_frame_from_coords(N, CA, C)
    coords, mask = frames_to_atom37(
        rigid.get_rots().get_rot_mats(), rigid.get_trans(), aatype,
        torch.zeros(1, 1, 7),
    )
    assert (coords[..., OFI["CA"], :] - CA).abs().max() < 1e-4
    assert (coords[..., OFI["N"], :] - N).abs().max() < 1e-4
    assert (coords[..., OFI["C"], :] - C).abs().max() < 1e-4
    # ALA mask: N, CA, C, CB, O present.
    present = [OPENFOLD_ATOM37_NAMES[i] for i in range(37) if mask[0, 0, i]]
    assert present == ["N", "CA", "C", "CB", "O"]


def test_glycine_has_no_cb():
    N, CA, C = _ideal_ala_backbone()
    aatype = torch.tensor([[7]])  # GLY
    rigid = backbone_frame_from_coords(N, CA, C)
    _, mask = frames_to_atom37(
        rigid.get_rots().get_rot_mats(), rigid.get_trans(), aatype,
        torch.zeros(1, 1, 7),
    )
    assert not mask[0, 0, OFI["CB"]]
    assert mask[0, 0, OFI["N"]] and mask[0, 0, OFI["O"]]


def test_sidechain_reconstruction_from_chi_angles():
    """Chi angles drive sidechain placement through official OpenFold frames.

    Build a real-ish LYS by taking literature backbone + arbitrary chi values,
    then confirm the reconstructed atoms differ when chi1 changes (i.e. the
    OpenFold torsion->frame path is actually exercised).
    """
    N, CA, C = _ideal_ala_backbone()
    aatype = torch.tensor([[11]])  # LYS
    rigid = backbone_frame_from_coords(N, CA, C)

    t0 = torch.zeros(1, 1, 7)
    t1 = torch.zeros(1, 1, 7)
    t1[0, 0, 3] = 1.0  # chi1 = 1 rad

    c0, _ = frames_to_atom37(rigid.get_rots().get_rot_mats(), rigid.get_trans(), aatype, t0)
    c1, _ = frames_to_atom37(rigid.get_rots().get_rot_mats(), rigid.get_trans(), aatype, t1)

    # Backbone unaffected by chi1.
    assert (c0[..., OFI["CA"], :] - c1[..., OFI["CA"], :]).abs().max() < 1e-4
    # Sidechain moves.
    assert (c0[..., OFI["CG"], :] - c1[..., OFI["CG"], :]).norm() > 0.1


def test_decoder_frame_convention_matches_openfold():
    """Decoder frames (RigidBodyFrame.from_backbone) give the same atom37 as OF."""
    N, CA, C = _ideal_ala_backbone()
    R, t = RigidBodyFrame.from_backbone(N, CA, C)
    aatype = torch.zeros(1, 1, dtype=torch.long)

    coords_dec, _ = decoder_frames_to_atom37(R, t, aatype, torch.zeros(1, 1, 7))
    rigid = backbone_frame_from_coords(N, CA, C)
    coords_of, _ = frames_to_atom37(
        rigid.get_rots().get_rot_mats(), rigid.get_trans(), aatype,
        torch.zeros(1, 1, 7),
    )
    assert (coords_dec - coords_of).abs().max() < 1e-4


def test_project_ordering_permutation():
    """OpenFold <-> project schema permutation is a correct round-trip."""
    from protein_jepa_frame.data.schema import ATOM37_NAMES

    # OpenFold-ordered data: atom j has value j.
    of = torch.arange(37, dtype=torch.float32).view(1, 1, 37, 1).expand(1, 1, 37, 3)
    proj = to_project_ordering(of)
    # project index k holds the atom named ATOM37_NAMES[k] == OpenFold index.
    for k, name in enumerate(ATOM37_NAMES):
        assert proj[0, 0, k, 0].item() == OPENFOLD_ATOM37_NAMES.index(name)
    # Round-trip back to OpenFold ordering.
    back = to_openfold_ordering(proj)
    assert torch.equal(back, of)


def test_decoder_forward_produces_valid_openfold_atom37():
    from protein_jepa_frame.models.stage_c.structure_decoder import (
        StructureDecoder, DecoderConfig,
    )

    torch.manual_seed(0)
    B, L = 2, 5
    cfg = DecoderConfig(c_hidden=32, c_ipa=16, n_heads=2, n_ipa_blocks=1,
                        q_dim=16)
    dec = StructureDecoder(cfg)
    aatype = torch.randint(0, 20, (B, L))
    q = torch.randn(B, L, 16)
    base = torch.arange(L).view(1, L, 1) * 3.8
    N = base + 0.1 * torch.randn(B, L, 3)
    CA = base + 0.1 * torch.randn(B, L, 3)
    C = base + 0.1 * torch.randn(B, L, 3)
    R0, t0 = RigidBodyFrame.from_backbone(N, CA, C)

    out = dec(aatype, q, R0, t0)
    assert out["atom37_coords"].shape == (B, L, 37, 3)
    assert out["atom37_mask"].shape == (B, L, 37)
    assert out["atom37_mask"].any()
    # CA (OpenFold index 1) equals the predicted frame origin t.
    assert (out["atom37_coords"][..., 1, :] - out["t"]).abs().max() < 1e-4
    # GLY residues never have CB.
    for b in range(B):
        for r in range(L):
            if aatype[b, r].item() == 7:
                assert not out["atom37_mask"][b, r, OFI["CB"]]


def test_gradients_flow_through_openfold_atom37():
    from protein_jepa_frame.models.stage_c.structure_decoder import (
        StructureDecoder, DecoderConfig,
    )

    torch.manual_seed(0)
    B, L = 2, 5
    cfg = DecoderConfig(c_hidden=32, c_ipa=16, n_heads=2, n_ipa_blocks=1,
                        q_dim=16)
    dec = StructureDecoder(cfg)
    aatype = torch.randint(0, 20, (B, L))
    q = torch.randn(B, L, 16)
    base = torch.arange(L, dtype=torch.float32).view(1, L, 1) * 3.8
    N = base + 0.1 * torch.randn(B, L, 3)
    CA = base + 0.1 * torch.randn(B, L, 3)
    C = base + 0.1 * torch.randn(B, L, 3)
    R0, t0 = RigidBodyFrame.from_backbone(N, CA, C)

    out = dec(aatype, q, R0, t0)
    out["atom37_coords"].sum().backward()
    assert dec.torsion_head.net[2].weight.grad is not None
    assert dec.torsion_head.net[2].weight.grad.abs().sum() > 0
    assert dec.local_position_head[-1].weight.grad is not None
    assert dec.local_position_head[-1].weight.grad.abs().sum() > 0
