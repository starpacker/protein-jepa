"""Gate 3 tests: unified atom37 data layer (round-trip, units, atom names, residue/chain IDs).

Gate 3 requirements:
- mdCATH: all-atom trajectory → atom37/mask
- ATLAS: XTC+PDB → atom37 with PBC/altloc/missing-atom policies
- FastFolding: per-frame PDB → unified topology, scaffold grouping
- atom37 round-trip, unit, atom-name, residue/chain-ID tests
"""

import numpy as np
import torch
import pytest

from protein_jepa_frame.data.atom37_layer import (
    Atom37Frames,
    Atom37Policies,
    aatype_from_sequence,
    all_atom_to_atom37,
    atom37_roundtrip_backbone,
    backbone_to_atom37,
    build_atom37_frames,
    AA_ONE_LETTER,
    BACKBONE_ATOM_NAMES,
)
from protein_jepa_frame.data.schema import (
    ATOM37_NAMES,
    ATOM37_NAME_TO_INDEX,
    ATOM37_NUM,
    BACKBONE_ATOM37_INDICES,
    PBCUnwrapPolicy,
    AltlocPolicy,
    MissingResiduePolicy,
)


# ── Fixtures ─────────────────────────────────────────────────────

def _make_backbone(T=4, L=6, seed=0):
    """Make synthetic backbone [T, L, 4, 3] with a helix-like shape."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.zeros(T, L, 4, 3)
    for t in range(T):
        for i in range(L):
            ca = torch.tensor([float(i) * 3.8, float(t) * 0.1, 0.0])
            coords[t, i, 0] = ca + torch.tensor([-0.5, 0.0, -0.5])  # N
            coords[t, i, 1] = ca                                    # CA
            coords[t, i, 2] = ca + torch.tensor([0.5, 0.0, 0.5])    # C
            coords[t, i, 3] = ca + torch.tensor([0.5, 0.5, 0.5])    # O
    return coords


# ── Atom-name correctness ────────────────────────────────────────

def test_atom37_names_and_indices_consistent():
    """Atom37 names map to unique indices; backbone indices are correct."""
    assert len(ATOM37_NAMES) == ATOM37_NUM == 37
    # Unique names
    assert len(set(ATOM37_NAMES)) == 37
    # Backbone indices correspond to N, CA, C, O
    bb_names = tuple(ATOM37_NAMES[i] for i in BACKBONE_ATOM37_INDICES)
    assert bb_names == BACKBONE_ATOM_NAMES == ("N", "CA", "C", "O")
    # Name→index round trip
    for i, name in enumerate(ATOM37_NAMES):
        assert ATOM37_NAME_TO_INDEX[name] == i


def test_backbone_atom_names_placed_correctly():
    """backbone_to_atom37 places N/CA/C/O in correct atom37 slots."""
    T, L = 2, 4
    bb = _make_backbone(T, L)
    aatype = aatype_from_sequence("ACDE")
    coords, mask = backbone_to_atom37(bb, aatype)

    assert coords.shape == (T, L, 37, 3)
    assert mask.shape == (T, L, 37)

    for bb_pos, a37_idx in enumerate(BACKBONE_ATOM37_INDICES):
        assert torch.allclose(coords[:, :, a37_idx, :], bb[:, :, bb_pos, :])
        assert mask[:, :, a37_idx].all(), f"atom {ATOM37_NAMES[a37_idx]} not marked present"


# ── Round-trip ───────────────────────────────────────────────────

def test_backbone_roundtrip_identity():
    """atom37 → backbone → atom37 preserves backbone coordinates."""
    bb = _make_backbone(3, 5)
    aatype = aatype_from_sequence("ACDEF")
    frames = build_atom37_frames(backbone_coords=bb, aatype=aatype)
    assert atom37_roundtrip_backbone(frames)

    # Direct check: extracted backbone equals input
    bb_rt = frames.backbone()
    assert torch.allclose(bb_rt, bb, atol=1e-6)


def test_all_atom_roundtrip_backbone():
    """All-atom → atom37 → backbone matches the input N/CA/C/O atoms."""
    T, L = 2, 3
    atom_names = ["N", "CA", "C", "O", "CB"]
    A = len(atom_names)
    coords = torch.randn(T, L, A, 3) * 5.0
    aatype = aatype_from_sequence("AAA")  # ALA has CB

    a37, mask = all_atom_to_atom37(coords, atom_names, aatype)
    frames = Atom37Frames(
        coords=a37, atom37_mask=mask,
        residue_mask=torch.ones(L, dtype=torch.bool),
        aatype=aatype,
        residue_index=torch.arange(L),
        chain_id=torch.zeros(L, dtype=torch.long),
        time_ps=torch.arange(T, dtype=torch.float32),
    )
    bb = frames.backbone()
    # N, CA, C, O are atom_names[0:4]
    for bb_pos in range(4):
        assert torch.allclose(bb[:, :, bb_pos, :], coords[:, :, bb_pos, :], atol=1e-6)


# ── Units ────────────────────────────────────────────────────────

def test_time_units_picoseconds():
    """time_ps increments by frame_interval_ps starting at first_frame_time_ps."""
    bb = _make_backbone(5, 3)
    aatype = aatype_from_sequence("ACD")
    frames = build_atom37_frames(
        backbone_coords=bb, aatype=aatype,
        frame_interval_ps=10.0, first_frame_time_ps=100.0,
    )
    expected = torch.tensor([100.0, 110.0, 120.0, 130.0, 140.0])
    assert torch.allclose(frames.time_ps, expected)


def test_coordinate_units_angstrom_policy():
    """Policy records Angstrom coordinate unit."""
    policies = Atom37Policies()
    rec = policies.manifest_record()
    assert rec["coordinate_unit"] == "Angstrom"
    assert rec["time_unit"] == "picosecond"


# ── Residue / chain IDs ──────────────────────────────────────────

def test_residue_and_chain_ids_preserved():
    """Custom residue_index and chain_id survive the atom37 build."""
    bb = _make_backbone(2, 6)
    aatype = aatype_from_sequence("ACDEFG")
    residue_index = torch.tensor([10, 11, 12, 20, 21, 22])
    chain_id = torch.tensor([0, 0, 0, 1, 1, 1])
    frames = build_atom37_frames(
        backbone_coords=bb, aatype=aatype,
        residue_index=residue_index, chain_id=chain_id,
    )
    assert torch.equal(frames.residue_index, residue_index)
    assert torch.equal(frames.chain_id, chain_id)


def test_aatype_from_sequence_roundtrip():
    """Sequence → aatype → sequence is identity for canonical residues.

    The canonical order is 3-letter alphabetical (OpenFold/atom37): ARNDCQEGHILKMFPSTWYV
    """
    seq = AA_ONE_LETTER  # the canonical one-letter order
    aatype = aatype_from_sequence(seq)
    assert aatype.tolist() == list(range(20))
    recovered = "".join(AA_ONE_LETTER[i] for i in aatype.tolist())
    assert recovered == seq


def test_unknown_residue_maps_to_20():
    """Unknown residue letter maps to UNK index 20."""
    aatype = aatype_from_sequence("AXZ")
    assert aatype[0].item() == 0   # A
    assert aatype[1].item() == 20  # X unknown
    assert aatype[2].item() == 20  # Z unknown


# ── Mask correctness ─────────────────────────────────────────────

def test_residue_mask_from_backbone_presence():
    """Residue mask derives from CA presence across frames."""
    bb = _make_backbone(3, 4)
    aatype = aatype_from_sequence("ACDE")
    frames = build_atom37_frames(backbone_coords=bb, aatype=aatype)
    # All CAs present → all residues masked present
    assert frames.residue_mask.all()


def test_all_atom_mask_drops_stray_atoms():
    """All-atom conversion drops atoms not expected for a residue type (GLY has no CB)."""
    T, L = 1, 1
    atom_names = ["N", "CA", "C", "O", "CB"]
    coords = torch.randn(T, L, len(atom_names), 3)
    aatype = aatype_from_sequence("G")  # GLY: no CB

    a37, mask = all_atom_to_atom37(coords, atom_names, aatype)
    cb_idx = ATOM37_NAME_TO_INDEX["CB"]
    # CB present in input but not expected for GLY → masked out
    assert not mask[0, 0, cb_idx].item(), "GLY should not have a CB atom in atom37"
    # Backbone still present
    for a37_idx in BACKBONE_ATOM37_INDICES:
        assert mask[0, 0, a37_idx].item()


# ── PBC unwrap policy ────────────────────────────────────────────

def test_pbc_unwrap_policy_applied():
    """PBC unwrap policy runs when enabled (records in manifest)."""
    policies = Atom37Policies(pbc=PBCUnwrapPolicy(unwrap=True, max_displacement=5.0))
    bb = _make_backbone(4, 3)
    aatype = aatype_from_sequence("ACD")
    frames = build_atom37_frames(backbone_coords=bb, aatype=aatype, policies=policies)
    assert frames.num_frames == 4
    rec = policies.manifest_record()
    assert rec["pbc_unwrap"] is True
    assert rec["pbc_method"] == "sequential"


def test_policies_manifest_complete():
    """All policy fields recorded for reproducibility."""
    policies = Atom37Policies(
        pbc=PBCUnwrapPolicy(unwrap=True),
        altloc=AltlocPolicy(keep_altloc="B"),
        missing_residue=MissingResiduePolicy(policy="strict"),
    )
    rec = policies.manifest_record()
    assert rec["altloc_policy"] == "B"
    assert rec["missing_residue_policy"] == "strict"
    assert rec["coordinate_unit"] == "Angstrom"


# ── Cross-dataset uniformity ─────────────────────────────────────

def test_mdcath_style_all_atom_build():
    """mdCATH: all-atom trajectory → atom37 (uniform entry point)."""
    T, L = 3, 5
    # mdCATH provides full heavy atoms; emulate with backbone + CB
    atom_names = ["N", "CA", "C", "O", "CB"]
    coords = torch.randn(T, L, len(atom_names), 3) * 3.0
    aatype = aatype_from_sequence("ACDEF")
    a37, mask = all_atom_to_atom37(coords, atom_names, aatype)
    frames = Atom37Frames(
        coords=a37, atom37_mask=mask,
        residue_mask=torch.ones(L, dtype=torch.bool),
        aatype=aatype, residue_index=torch.arange(L),
        chain_id=torch.zeros(L, dtype=torch.long),
        time_ps=torch.arange(T, dtype=torch.float32) * 1000.0,  # mdCATH 1ns interval
    )
    assert frames.coords.shape == (T, L, 37, 3)
    assert frames.num_frames == 3


def test_atlas_style_backbone_build_with_chains():
    """ATLAS: backbone → atom37 with multi-chain IDs and 10ps interval."""
    bb = _make_backbone(4, 8)
    aatype = aatype_from_sequence("ACDEFGHI")
    chain_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    frames = build_atom37_frames(
        backbone_coords=bb, aatype=aatype, chain_id=chain_id,
        frame_interval_ps=10.0,  # ATLAS interval
    )
    assert frames.chain_id.unique().tolist() == [0, 1]
    assert torch.allclose(frames.time_ps, torch.tensor([0.0, 10.0, 20.0, 30.0]))


def test_fastfolding_style_single_frame_build():
    """FastFolding: per-frame PDB → unified topology (single-frame build)."""
    bb = _make_backbone(1, 6)  # one frame per PDB
    aatype = aatype_from_sequence("ACDEFG")
    frames = build_atom37_frames(backbone_coords=bb, aatype=aatype)
    assert frames.num_frames == 1
    assert atom37_roundtrip_backbone(frames)


def test_shape_validation_rejects_bad_input():
    """Atom37Frames rejects inconsistent shapes."""
    with pytest.raises(ValueError):
        Atom37Frames(
            coords=torch.zeros(2, 3, 37, 3),
            atom37_mask=torch.zeros(2, 3, 37, dtype=torch.bool),
            residue_mask=torch.zeros(4, dtype=torch.bool),  # wrong L
            aatype=torch.zeros(3, dtype=torch.long),
            residue_index=torch.arange(3),
            chain_id=torch.zeros(3, dtype=torch.long),
            time_ps=torch.zeros(2),
        )
