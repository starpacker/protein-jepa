"""Independent numerical checks for versioned, sample-wise backbone metrics.

The synthetic chain has unit bonds and right angles so its dihedrals are
analytical. It is a metric fixture, not an assertion of physical geometry.
No trajectories, learned models, or OpenFold reconstruction are used.
"""
from __future__ import annotations

import math

import pytest
import torch

from evaluation.backbone_metrics_v2 import (
    backbone_error_summary,
    backbone_geometry_summary,
)


def _torsion_backbone() -> torch.Tensor:
    """Four residues with phi=[-60,-170,-60], psi=[0,75,0] degrees."""
    points = [
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
    ]
    # Appended N, CA, C triplets specify psi, omega, phi in that order.
    for degrees in (0, 180, -60, 75, 180, -170, 0, 180, -60):
        a, b, c = points[-3:]
        axis = (c - b) / (c - b).norm()
        radial = a - b - torch.dot(a - b, axis) * axis
        radial = radial / radial.norm()
        tangent = torch.cross(axis, radial, dim=0)
        theta = math.radians(degrees)
        points.append(c + math.cos(theta) * radial + math.sin(theta) * tangent)
    n_ca_c = torch.stack(points).reshape(4, 3, 3)
    oxygen = n_ca_c[:, 2:3] + n_ca_c.new_tensor([0.0, 0.0, 1.0])
    return torch.cat((n_ca_c, oxygen), dim=1)


def _tetrahedron() -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64,
    )


def _two_residue_backbone() -> torch.Tensor:
    return torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [1.0, 1.0, 0.0], [1.0, 1.0, 1.231]],
         [[2.329, 1.0, 0.0], [2.329, 2.458, 0.0],
          [3.854, 2.458, 0.0], [3.854, 2.458, 1.231]]],
        dtype=torch.float64,
    )


def test_coordinate_error_is_root_mean_square_of_atom_distances():
    ref = _tetrahedron()
    pred = ref.clone()
    pred[0, 3, 0] += 2.0
    result = backbone_error_summary(pred, ref)
    row = result["per_sample"][0]
    assert result["schema_version"] == "backbone_metrics_v2"
    assert row["n_atoms"] == 4
    # Distances [0,0,0,2]: RMS=1; arithmetic mean distance would be 0.5.
    assert row["coordinate_rmsd_angstrom"] == pytest.approx(1.0)
    assert row["coordinate_rmsd_O_angstrom"] == pytest.approx(2.0)
    for atom in ("N", "CA", "C"):
        assert row[f"coordinate_rmsd_{atom}_angstrom"] == pytest.approx(0.0)


def test_coordinate_gauge_preserves_rigid_pose_error_while_shape_removes_it():
    ref = _tetrahedron()
    pred = ref + ref.new_tensor([3.0, 4.0, 0.0])
    row = backbone_error_summary(pred, ref)["per_sample"][0]
    assert row["coordinate_rmsd_angstrom"] == pytest.approx(5.0)
    assert row["shape_kabsch_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-10)
    for atom in ("N", "CA", "C", "O"):
        assert row[f"coordinate_rmsd_{atom}_angstrom"] == pytest.approx(5.0)
        assert row[f"shape_rmsd_{atom}_angstrom"] == pytest.approx(0.0, abs=1e-10)


def test_per_atom_type_errors_use_rms_and_one_shared_shape_alignment():
    ref = _torsion_backbone()
    pred = ref.clone()
    pred[-1, 0, 0] += 2.0
    row = backbone_error_summary(pred, ref)["per_sample"][0]
    # One of sixteen atoms moves 2 A; one of four N atoms moves 2 A.
    assert row["coordinate_rmsd_angstrom"] == pytest.approx(0.5)
    assert row["coordinate_rmsd_N_angstrom"] == pytest.approx(1.0)
    for atom in ("CA", "C", "O"):
        assert row[f"coordinate_rmsd_{atom}_angstrom"] == pytest.approx(0.0)
    # Equal atom-type counts imply this exact decomposition only when the
    # same whole-backbone fit defines all per-type shape residuals.
    per_type_mse = sum(row[f"shape_rmsd_{atom}_angstrom"] ** 2
                       for atom in ("N", "CA", "C", "O")) / 4
    assert row["shape_kabsch_rmsd_angstrom"] ** 2 == pytest.approx(per_type_mse, abs=1e-12)


def test_error_samples_are_equally_weighted_after_individual_rms():
    ref = _tetrahedron().expand(2, -1, -1, -1).clone()
    pred = ref.clone()
    pred[1] += pred.new_tensor([2.0, 0.0, 0.0])
    result = backbone_error_summary(pred, ref)
    assert [r["coordinate_rmsd_angstrom"] for r in result["per_sample"]] == [0.0, 2.0]
    assert result["mean"]["coordinate_rmsd_angstrom"] == pytest.approx(1.0)
    assert result["totals"]["n_atoms"] == 8
    assert "n_atoms" not in result["mean"]


def test_two_residues_have_peptide_angles_and_omega():
    row = backbone_geometry_summary(_two_residue_backbone())["per_sample"][0]
    assert row["bond_C_N_n"] == 1
    assert row["bond_C_N_mean_angstrom"] == pytest.approx(1.329)
    assert row["bond_C_N_legacy_valid_frac"] == pytest.approx(1.0)
    assert row["angle_CA_C_N_n"] == 1
    assert row["angle_C_N_CA_n"] == 1
    assert row["angle_CA_C_N_mean_deg"] == pytest.approx(90.0)
    assert row["angle_C_N_CA_mean_deg"] == pytest.approx(90.0)
    assert row["omega_n"] == 1
    assert row["omega_trans_diagnostic_frac"] == pytest.approx(1.0)
    assert row["ramachandran_box_n"] == 0
    assert row["ramachandran_box_diagnostic_frac"] is None


def test_ramachandran_pairs_phi_and_psi_at_the_same_residue():
    bb = _torsion_backbone()
    row = backbone_geometry_summary(bb)["per_sample"][0]
    # Correct pairs (-60,75),(-170,0) are both outside the declared boxes.
    # Shifted pairing instead gives two allowed pairs out of three.
    assert row["ramachandran_box_n"] == 2
    assert row["ramachandran_box_diagnostic_frac"] == pytest.approx(0.0)
    assert row["omega_n"] == 3
    assert row["omega_trans_diagnostic_frac"] == pytest.approx(1.0)
    short = backbone_geometry_summary(bb[:3])["per_sample"][0]
    assert short["ramachandran_box_n"] == 1
    assert short["ramachandran_box_diagnostic_frac"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("chain", "indices", "mask", "expected_adjacent", "expected_rama"),
    [
        ([0, 0, 0, 0], [0, 1, 3, 4], [True] * 4, 2, 0),
        ([0, 0, 1, 1], [0, 1, 2, 3], [True] * 4, 2, 0),
        ([0, 0, 0, 0], [0, 1, 2, 3], [True, False, True, True], 1, 0),
        ([0, 1, 0, 0], [0, 1, 2, 3], [True] * 4, 1, 0),
    ],
)
def test_chain_gaps_and_missing_middle_residues_break_torsions(
    chain, indices, mask, expected_adjacent, expected_rama,
):
    row = backbone_geometry_summary(
        _torsion_backbone(), torch.tensor(mask),
        chain_id=torch.tensor(chain), residue_index=torch.tensor(indices),
    )["per_sample"][0]
    assert row["bond_C_N_n"] == expected_adjacent
    assert row["omega_n"] == expected_adjacent
    assert row["angle_CA_C_N_n"] == expected_adjacent
    assert row["angle_C_N_CA_n"] == expected_adjacent
    assert row["ramachandran_box_n"] == expected_rama


@pytest.mark.parametrize(("cutoff", "clash_count"), [(2.5, 3), (1.1, 2), (0.9, 0)])
def test_ca_clashes_include_same_chain_nonlocal_or_any_cross_chain_pair(cutoff, clash_count):
    ca = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
         [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64,
    )
    bb = _tetrahedron().expand(5, -1, -1).clone()
    bb += (ca - bb[:, 1]).unsqueeze(-2)
    metadata = {"chain_id": torch.tensor([0, 0, 0, 0, 1]),
                "residue_index": torch.tensor([10, 11, 12, 13, 10])}
    row = backbone_geometry_summary(bb, clash_cutoff=cutoff, **metadata)["per_sample"][0]
    # Eligible pairs: (0,3), (0,4), (1,4), (2,4), (3,4).
    assert row["ca_nonlocal_pair_n"] == 5
    assert row["ca_nonlocal_clash_n"] == clash_count
    assert row["ca_nonlocal_clash_frac"] == pytest.approx(clash_count / 5)
    same_chain = backbone_geometry_summary(
        bb, torch.tensor([True, True, True, True, False]),
        clash_cutoff=2.5, **metadata,
    )["per_sample"][0]
    assert same_chain["ca_nonlocal_pair_n"] == 1
    assert same_chain["ca_nonlocal_clash_n"] == 1


def test_geometry_uses_every_sample_and_reports_counts_separately():
    bb = _two_residue_backbone().expand(2, -1, -1, -1).clone()
    bb[1, 1] += bb.new_tensor([1.0, 0.0, 0.0])
    result = backbone_geometry_summary(bb)
    assert result["schema_version"] == "backbone_metrics_v2"
    assert len(result["per_sample"]) == 2
    assert set(result["per_sample"][0]) == set(result["per_sample"][1])
    assert result["per_sample"][0]["bond_C_N_mean_angstrom"] == pytest.approx(1.329)
    assert result["per_sample"][1]["bond_C_N_mean_angstrom"] == pytest.approx(2.329)
    assert result["mean"]["bond_C_N_mean_angstrom"] == pytest.approx(1.829)
    assert result["mean"]["bond_C_N_legacy_valid_frac"] == pytest.approx(0.5)
    assert result["totals"]["bond_C_N_n"] == 2
    assert "bond_C_N_n" not in result["mean"]


def test_batch_residue_mask_and_topology_broadcast_across_sample_axis():
    ref = _torsion_backbone().expand(2, 3, -1, -1, -1).clone()
    pred = ref.clone()
    pred[0] += pred.new_tensor([1.0, 0.0, 0.0])
    pred[1] += pred.new_tensor([2.0, 0.0, 0.0])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    chain = torch.zeros(2, 4, dtype=torch.long)
    indices = torch.arange(4).expand(2, -1)
    errors = backbone_error_summary(pred, ref, mask)
    assert len(errors["per_sample"]) == 6
    assert [r["n_atoms"] for r in errors["per_sample"]] == [12, 12, 12, 8, 8, 8]
    assert errors["totals"]["n_atoms"] == 60
    assert errors["mean"]["coordinate_rmsd_angstrom"] == pytest.approx(1.5)
    assert errors == backbone_error_summary(pred, ref, mask[:, None, :])
    geom = backbone_geometry_summary(pred, mask, chain_id=chain, residue_index=indices)
    assert [r["bond_C_N_n"] for r in geom["per_sample"]] == [2, 2, 2, 1, 1, 1]
    assert geom["totals"]["bond_C_N_n"] == 9
    assert geom == backbone_geometry_summary(
        pred, mask[:, None, :], chain_id=chain[:, None, :],
        residue_index=indices[:, None, :],
    )


def test_missing_oxygen_does_not_remove_backbone_dihedrals_and_masked_nan_is_ignored():
    ref = _torsion_backbone()
    bb = ref.clone()
    atom_mask = torch.ones(4, 4, dtype=torch.bool)
    atom_mask[:, 3] = False
    bb[:, 3] = float("nan")
    row = backbone_geometry_summary(bb, atom_mask=atom_mask)["per_sample"][0]
    assert row["ramachandran_box_n"] == 2
    assert row["ramachandran_box_diagnostic_frac"] == pytest.approx(0.0)
    assert row["omega_n"] == 3
    assert row["bond_C_N_n"] == 3
    err = backbone_error_summary(bb, ref, atom_mask=atom_mask)["per_sample"][0]
    assert err["n_atoms"] == 12
    assert err["coordinate_rmsd_angstrom"] == pytest.approx(0.0)
    assert err["coordinate_rmsd_O_angstrom"] is None
    assert err["shape_kabsch_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_unmasked_nonfinite_atoms_are_rejected(bad_value):
    ref = _torsion_backbone()
    bb = ref.clone()
    bb[1, 0, 0] = bad_value
    with pytest.raises(ValueError):
        backbone_geometry_summary(bb)
    with pytest.raises(ValueError):
        backbone_error_summary(bb, ref)
    with pytest.raises(ValueError):
        backbone_error_summary(ref, bb)


def test_empty_valid_set_has_zero_counts_and_undefined_metrics():
    bb = _torsion_backbone()
    mask = torch.zeros(4, dtype=torch.bool)
    invalid = torch.full_like(bb, float("nan"))
    err = backbone_error_summary(invalid, bb, mask)["per_sample"][0]
    assert err["n_atoms"] == 0
    assert err["coordinate_rmsd_angstrom"] is None
    assert err["shape_kabsch_rmsd_angstrom"] is None
    result = backbone_geometry_summary(invalid, mask)
    row = result["per_sample"][0]
    assert row["bond_C_N_n"] == 0
    assert row["bond_C_N_mean_angstrom"] is None
    assert row["ramachandran_box_n"] == 0
    assert row["ramachandran_box_diagnostic_frac"] is None
    assert row["ca_nonlocal_pair_n"] == 0
    assert row["ca_nonlocal_clash_frac"] is None
    assert all(value == 0 for value in result["totals"].values())
    nonempty = backbone_geometry_summary(bb)["per_sample"][0]
    assert set(row) == set(nonempty)


def test_zero_length_backbones_are_rejected():
    empty = torch.empty(0, 4, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        backbone_geometry_summary(empty)
    with pytest.raises(ValueError):
        backbone_error_summary(empty, empty)


def test_residue_dimension_is_not_broadcast_as_a_sample_axis():
    with pytest.raises(ValueError, match="same residue count"):
        backbone_error_summary(_torsion_backbone(), _tetrahedron())


def test_known_rotation_and_translation_have_zero_shape_error():
    ref = _tetrahedron()
    rotation = ref.new_tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = ref @ rotation + ref.new_tensor([3.0, 4.0, 0.0])
    row = backbone_error_summary(pred, ref)["per_sample"][0]
    # Squared coordinate errors of N,CA,C,O are 25,29,13,25.
    assert row["coordinate_rmsd_angstrom"] == pytest.approx(math.sqrt(23.0))
    assert row["shape_kabsch_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-10)
    for atom in ("N", "CA", "C", "O"):
        assert row[f"shape_rmsd_{atom}_angstrom"] == pytest.approx(0.0, abs=1e-10)


def test_equal_batch_and_sample_sizes_require_explicit_mask_axis():
    bb = _torsion_backbone().expand(2, 2, -1, -1, -1).clone()
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    with pytest.raises(ValueError):
        backbone_error_summary(bb, bb, mask)
    with pytest.raises(ValueError):
        backbone_geometry_summary(bb, mask)
    per_batch = backbone_error_summary(bb, bb, mask[:, None, :])
    per_sample = backbone_error_summary(bb, bb, mask[None, :, :])
    assert [r["n_atoms"] for r in per_batch["per_sample"]] == [16, 16, 8, 8]
    assert [r["n_atoms"] for r in per_sample["per_sample"]] == [16, 8, 16, 8]
    # Full [B,S,L] topology/masks are always explicit and standard-broadcastable.
    indices = torch.tensor([0, 1, 2, 3]).expand(2, 2, -1).clone()
    indices[:, 1] = torch.tensor([0, 1, 3, 5])
    geometry = backbone_geometry_summary(
        bb, mask[None].expand(2, -1, -1), residue_index=indices,
        chain_id=torch.zeros(2, 2, 4, dtype=torch.long),
    )
    assert [r["bond_C_N_n"] for r in geometry["per_sample"]] == [3, 1, 3, 1]


def test_shared_residue_atom_mask_remains_unambiguous_when_batch_equals_length():
    bb = _torsion_backbone().expand(4, 2, -1, -1, -1).clone()
    atom_mask = torch.ones(4, 4, dtype=torch.bool)
    atom_mask[:, 3] = False
    result = backbone_error_summary(bb, bb, atom_mask=atom_mask)
    assert len(result["per_sample"]) == 8
    assert [r["n_atoms"] for r in result["per_sample"]] == [12] * 8
    geometry = backbone_geometry_summary(bb, atom_mask=atom_mask)
    assert [r["ramachandran_box_n"] for r in geometry["per_sample"]] == [2] * 8


def test_angle_is_scale_invariant_above_the_declared_degeneracy_threshold():
    bb = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
          [0.5, math.sqrt(3) / 2, 0.0], [0.5, math.sqrt(3) / 2, 1.0]]],
        dtype=torch.float64,
    )
    for scale in (1.0, 1e-6):
        row = backbone_geometry_summary(bb * scale)["per_sample"][0]
        assert row["angle_N_CA_C_n"] == 1
        assert row["angle_N_CA_C_degenerate_n"] == 0
        assert row["angle_N_CA_C_mean_deg"] == pytest.approx(60.0)


@pytest.mark.parametrize(
    ("first_aa", "next_aa", "bond_length", "expected_z"),
    [(14, 0, 1.329, 0.0), (0, 14, 1.341, 0.0),
     (0, 14, 1.357, 1.0), (14, 0, 1.343, 1.0)],
)
def test_openfold_peptide_bond_reference_depends_on_next_residue_proline(
    first_aa, next_aa, bond_length, expected_z,
):
    # OpenFold restype indices: ALA=0, PRO=14. The bond distribution is
    # (1.329,0.014) A for non-PRO and (1.341,0.016) A for next-residue PRO.
    bb = _two_residue_backbone()
    bb[1, :, 0] += bond_length - 1.329
    row = backbone_geometry_summary(bb, aatype=torch.tensor([first_aa, next_aa]))["per_sample"][0]
    assert row["openfold_bond_C_N_n"] == 1
    assert row["openfold_bond_C_N_mean_abs_z"] == pytest.approx(expected_z, abs=1e-12)
    # These right angles have cosine zero. The source distributions are in
    # cosine space, with stddevs 0.0311 and 0.0353 (not C-N length stddev).
    assert row["openfold_angle_CA_C_N_mean_abs_z"] == pytest.approx(0.4473 / 0.0311)
    assert row["openfold_angle_C_N_CA_mean_abs_z"] == pytest.approx(0.5203 / 0.0353)


@pytest.mark.parametrize(
    ("aa", "nca", "nca_sd", "cac", "co", "angle", "angle_sd"),
    [(0, 1.459, 0.020, 1.525, 1.229, 111.0, 2.7),
     (7, 1.456, 0.015, 1.514, 1.232, 113.1, 2.5),
     (14, 1.468, 0.017, 1.524, 1.228, 112.1, 2.6)],
)
def test_openfold_intrarezidue_z_uses_residue_specific_stereo_table(
    aa, nca, nca_sd, cac, co, angle, angle_sd,
):
    # Independent literals from vendored stereo_chemical_props.txt, including
    # GLY and PRO differences. N-CA and N-CA-C are exactly one sigma high.
    theta = math.radians(angle + angle_sd)
    carbon = [cac * math.cos(theta), cac * math.sin(theta), 0.0]
    bb = torch.tensor(
        [[[nca + nca_sd, 0.0, 0.0], [0.0, 0.0, 0.0], carbon,
          [carbon[0], carbon[1], co]]], dtype=torch.float64,
    )
    row = backbone_geometry_summary(bb, aatype=torch.tensor([aa]))["per_sample"][0]
    assert row["openfold_bond_N_CA_mean_abs_z"] == pytest.approx(1.0)
    assert row["openfold_bond_CA_C_mean_abs_z"] == pytest.approx(0.0, abs=1e-12)
    assert row["openfold_bond_C_O_mean_abs_z"] == pytest.approx(0.0, abs=1e-12)
    assert row["openfold_angle_N_CA_C_mean_abs_z"] == pytest.approx(1.0)


def test_missing_and_unknown_residue_identity_leave_class_specific_z_undefined():
    bb = _tetrahedron()
    absent = backbone_geometry_summary(bb)["per_sample"][0]
    unknown = backbone_geometry_summary(bb, aatype=torch.tensor([20]))["per_sample"][0]
    for row in (absent, unknown):
        assert row["openfold_bond_N_CA_mean_abs_z"] is None
        assert row["openfold_bond_N_CA_n"] == 0
        assert row["openfold_angle_N_CA_C_mean_abs_z"] is None
        assert row["openfold_angle_N_CA_C_n"] == 0
        assert row["bond_N_CA_n"] == 1
    assert set(absent) == set(unknown)


@pytest.mark.parametrize("aa", [-1, 21])
def test_out_of_range_residue_identity_is_rejected(aa):
    with pytest.raises(ValueError):
        backbone_geometry_summary(_tetrahedron(), aatype=torch.tensor([aa]))


def test_zero_sample_axis_is_rejected_even_when_residue_length_is_positive():
    bb = torch.empty(0, 4, 4, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        backbone_geometry_summary(bb)
    with pytest.raises(ValueError):
        backbone_error_summary(bb, bb)
