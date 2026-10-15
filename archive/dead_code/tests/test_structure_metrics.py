import pytest
import torch

from protein_jepa_frame.eval.structure_metrics import (
    align_backbone_by_ca,
    compute_structure_metrics,
    official_kabsch,
)


def _backbone(n=8):
    ca = torch.stack(
        [torch.arange(n, dtype=torch.float32) * 3.8,
         torch.zeros(n), torch.zeros(n)], dim=-1
    )
    return torch.stack(
        [ca + torch.tensor([-0.4, 0.1, 0.0]), ca,
         ca + torch.tensor([0.5, 0.2, -0.1]),
         ca + torch.tensor([0.2, -0.3, 0.2])], dim=1
    )


def test_official_kabsch_and_backbone_alignment_are_rigid_invariant():
    target = _backbone()
    theta = torch.tensor(0.41)
    rot = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta), torch.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    source = target @ rot + torch.tensor([5.0, -2.0, 1.0])

    aligned, rmsd = official_kabsch(source[:, 1], target[:, 1])
    assert rmsd.item() < 1e-5
    assert torch.max(torch.abs(aligned - target[:, 1])) < 1e-5
    aligned_bb = align_backbone_by_ca(source, target)
    assert torch.max(torch.abs(aligned_bb - target)) < 1e-5

    metrics = compute_structure_metrics(source, target)
    assert metrics["rmsd_kabsch"] < 1e-5
    assert metrics["dist_mae"] < 1e-5
    assert metrics["contact_jaccard"] == 1.0


def test_masks_and_chain_boundaries_are_respected():
    target = _backbone(8)
    source = target.clone()
    source[2, 1] += torch.tensor([0.0, 10.0, 0.0])
    mask = torch.ones(8, dtype=torch.bool)
    mask[2] = False
    # A new chain at residue 4 makes the pair (3, 4) eligible despite adjacent
    # array positions; missing residue 2 must not enter any denominator.
    chain = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    residue_index = torch.arange(8)
    metrics = compute_structure_metrics(
        source, target, mask,
        chain_id=chain, residue_index=residue_index,
    )
    assert metrics["n_valid_residues"] == 7
    assert metrics["n_nonlocal_pairs"] > 0
    assert metrics["rmsd_kabsch"] < 1e-5


def test_nonfinite_prediction_is_a_hard_evaluation_failure():
    target = _backbone(5)
    source = target.clone()
    source[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        compute_structure_metrics(source, target)


def test_empty_contact_sets_follow_jaccard_convention():
    target = _backbone(4)
    metrics = compute_structure_metrics(target, target, min_seq_sep=99)
    assert metrics["contact_jaccard"] == 1.0
    assert metrics["weak_contact_j"] == 1.0
    assert metrics["transient_contact_j"] == 1.0
