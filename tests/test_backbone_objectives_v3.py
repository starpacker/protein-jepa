"""Synthetic CPU contracts for directed explicit-topology backbone losses."""
import pytest
import torch

from protein_jepa_frame.models.stage_c.backbone_objectives_v2 import backbone_geometry_losses_v2
from protein_jepa_frame.models.stage_c.backbone_objectives_v3 import (
    backbone_connection_loss, backbone_geometry_losses_v3, backbone_nonbonded_clash_loss,
)


def inputs(length=4):
    generator = torch.Generator().manual_seed(33)
    bb = torch.randn(length, 4, 3, generator=generator, dtype=torch.float64)
    bb += torch.arange(length, dtype=bb.dtype)[:, None, None] * bb.new_tensor([1.7, .3, .2])
    return bb, dict(chemical_aatype=torch.zeros(length, dtype=torch.long),
        atom_mask=torch.ones(length, 4, dtype=torch.bool),
        peptide_edges=torch.stack((torch.arange(length-1), torch.arange(1, length)), dim=-1),
        geometry_row_ids=torch.arange(100, 100+length))


def remap_subset(metadata, selection):
    """Test-only explicit crop/permutation, mirroring the topology API contract."""
    inverse = {int(old): new for new, old in enumerate(selection)}
    edges = [(inverse[int(i)], inverse[int(j)]) for i, j in metadata['peptide_edges']
             if int(i) in inverse and int(j) in inverse]
    return {**{key: value[selection] for key, value in metadata.items() if key != 'peptide_edges'},
            'peptide_edges': torch.tensor(edges, dtype=torch.long).reshape(-1, 2)}


@pytest.mark.parametrize('sample_axes', [(), (2,), (2, 3)])
def test_v2_equivalent_clear_topology_values_counts_and_gradients(sample_axes):
    bb, kw = inputs()
    kw['chemical_aatype'][2] = 14  # OpenFold PRO.
    bb = bb.expand(*sample_axes, *bb.shape).clone().requires_grad_()
    old = backbone_geometry_losses_v2(bb, chemical_aatype=kw['chemical_aatype'],
        atom_mask=kw['atom_mask'], residue_index=torch.arange(4),
        chain_id=torch.zeros(4, dtype=torch.long), tolerance_factor=1.)
    new = backbone_geometry_losses_v3(bb, **kw, tolerance_factor=1.)
    for key, value in old['losses'].items():
        assert torch.allclose(new['losses'][key], value, atol=1e-12, rtol=1e-12)
        a = torch.autograd.grad(value, bb, retain_graph=True)[0]
        b = torch.autograd.grad(new['losses'][key], bb, retain_graph=True)[0]
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    for key, value in old['counts'].items():
        assert torch.equal(new['counts'][key], value)
    assert new['schema_version'] != old['schema_version']
    assert new['conventions']['formula_version'] == old['schema_version']


def test_insertion_code_rows_do_not_require_distinct_legacy_residue_numbers():
    bb, kw = inputs(3)
    # Author residues 1A, 1B, 1C can all have legacy residue_index=1. Their
    # immutable topology row IDs remain unique; bonds come only from PSF edges.
    kw['geometry_row_ids'] = torch.tensor([81, 82, 83])
    result = backbone_geometry_losses_v3(bb, **kw)
    assert int(result['counts']['n_connection_candidates']) == 2
    assert int(result['counts']['n_excluded_peptide_atom_pairs']) == 2
    kw['peptide_edges'] = torch.empty(0, 2, dtype=torch.long)
    no_edges = backbone_geometry_losses_v3(bb, **kw)
    assert int(no_edges['counts']['n_connection_candidates']) == 0
    assert int(no_edges['counts']['n_excluded_peptide_atom_pairs']) == 0
    assert int(no_edges['counts']['n_nonbonded_atom_pairs']) == 48


def test_crop_drops_outside_edges_without_bridging_and_preserves_row_identity():
    bb, kw = inputs(4)
    selected = torch.tensor([0, 2, 3])
    crop = remap_subset(kw, selected)
    assert torch.equal(crop['peptide_edges'], torch.tensor([[1, 2]]))
    assert torch.equal(crop['geometry_row_ids'], torch.tensor([100, 102, 103]))
    result = backbone_geometry_losses_v3(bb[selected], **crop)
    assert int(result['counts']['n_connection_candidates']) == 1
    assert int(result['counts']['n_excluded_peptide_atom_pairs']) == 1
    assert int(result['counts']['n_nonbonded_atom_pairs']) == 47


def test_storage_and_edge_order_do_not_change_values_or_gradients():
    bb, kw = inputs()
    bb.requires_grad_()
    original = backbone_geometry_losses_v3(bb, **kw, tolerance_factor=0.)
    permutation = torch.tensor([3, 1, 0, 2])
    metadata = remap_subset(kw, permutation)
    metadata['peptide_edges'] = metadata['peptide_edges'].flip(0)
    permuted = backbone_geometry_losses_v3(bb[permutation], **metadata, tolerance_factor=0.)
    for key, value in original['losses'].items():
        assert torch.allclose(permuted['losses'][key], value, atol=1e-12, rtol=1e-12)
        a = torch.autograd.grad(value, bb, retain_graph=True)[0]
        b = torch.autograd.grad(permuted['losses'][key], bb, retain_graph=True)[0]
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)


def test_explicit_cross_chain_and_reverse_bonds_exclude_only_actual_C_N_pairs():
    bb, kw = inputs(2)
    # No chain argument can incorrectly delete an authoritative PSF edge.
    kw['geometry_row_ids'] = torch.tensor([9, 703])
    one = backbone_geometry_losses_v3(bb, **kw)
    assert int(one['counts']['n_connection_candidates']) == 1
    assert int(one['counts']['n_nonbonded_atom_pairs']) == 15
    assert int(one['counts']['n_excluded_peptide_atom_pairs']) == 1
    kw['peptide_edges'] = torch.tensor([[0, 1], [1, 0]])
    both = backbone_geometry_losses_v3(bb, **kw)
    assert int(both['counts']['n_connection_candidates']) == 2
    assert int(both['counts']['n_nonbonded_atom_pairs']) == 14
    assert int(both['counts']['n_excluded_peptide_atom_pairs']) == 2


def test_missing_atoms_masked_nan_unknown_and_no_mask_gap_bridge():
    bb, kw = inputs(4)
    bb[1, 0] = torch.nan
    kw['atom_mask'][1, 0] = False
    kw['chemical_aatype'][3] = 20
    bb.requires_grad_()
    result = backbone_geometry_losses_v3(bb, **kw, tolerance_factor=0.)
    assert int(result['counts']['n_c_n_atom_candidates']) == 2
    assert int(result['counts']['n_c_n_unknown_next_excluded']) == 1
    assert int(result['counts']['n_c_n']) == 1
    assert int(result['counts']['n_ca_c_n']) == 2
    gradient = torch.autograd.grad(sum(result['losses'].values()), bb)[0]
    assert torch.isfinite(gradient).all() and torch.equal(gradient[1, 0], torch.zeros(3))
    result = backbone_geometry_losses_v3(bb, **kw, residue_mask=torch.tensor([True, False, True, True]))
    assert int(result['counts']['n_connection_candidates']) == 1
    assert int(result['counts']['n_excluded_peptide_atom_pairs']) == 1


def test_all_four_terms_pass_full_coordinate_finite_difference():
    bb, kw = inputs(3)
    bb.requires_grad_()
    def loss(coordinates):
        result = backbone_geometry_losses_v3(coordinates, **kw, tolerance_factor=1.)
        return torch.stack(list(result['losses'].values()))
    assert torch.autograd.gradcheck(loss, (bb,), eps=1e-6, atol=2e-5, rtol=2e-4)
    for value in loss(bb):
        assert torch.autograd.grad(value, bb, retain_graph=True)[0].abs().max() > 1e-7


def test_empty_edges_single_residue_and_empty_samples_have_differentiable_zero():
    bb, kw = inputs(1)
    bb.requires_grad_()
    result = backbone_geometry_losses_v3(bb, **kw)
    assert all(value == 0 and value.requires_grad for value in result['losses'].values())
    assert torch.equal(torch.autograd.grad(sum(result['losses'].values()), bb)[0], torch.zeros_like(bb))
    bb, kw = inputs(3)
    kw['atom_mask'].zero_()
    bb[:] = torch.nan
    bb.requires_grad_()
    result = backbone_geometry_losses_v3(bb, **kw)
    assert all(value == 0 for value in result['losses'].values())
    assert torch.equal(torch.autograd.grad(sum(result['losses'].values()), bb)[0], torch.zeros_like(bb))


@pytest.mark.parametrize('edges', [torch.tensor([[0, 0]]), torch.tensor([[0, 1], [0, 1]]),
    torch.tensor([[-1, 1]]), torch.tensor([[0, 4]]), torch.tensor([[0., 1.]]),
    torch.tensor([[False, True]]), torch.tensor([0, 1]), torch.empty(0, dtype=torch.long),
    torch.tensor([[[0, 1]]])])
def test_invalid_edges_fail_closed(edges):
    bb, kw = inputs()
    kw['peptide_edges'] = edges
    with pytest.raises(ValueError):
        backbone_geometry_losses_v3(bb, **kw)


@pytest.mark.parametrize('ids', [torch.tensor([1, 1, 2, 3]), torch.arange(3),
    torch.arange(4).float(), torch.ones(4, dtype=torch.bool), torch.arange(4)[None]])
def test_invalid_row_identities_fail_closed(ids):
    bb, kw = inputs()
    kw['geometry_row_ids'] = ids
    with pytest.raises(ValueError):
        backbone_geometry_losses_v3(bb, **kw)


def test_standalone_terms_and_shared_batch_topology():
    bb, kw = inputs(3)
    bb = bb.expand(2, 3, *bb.shape).clone()
    kw['atom_mask'] = kw['atom_mask'].expand(2, 1, 3, 4).clone()
    kw['atom_mask'][1] = False
    full = backbone_geometry_losses_v3(bb, **kw)
    assert full['per_sample']['c_n'].shape == (2, 3)
    assert full['counts']['n_c_n'].tolist() == [[2, 2, 2], [0, 0, 0]]
    for standalone in (backbone_connection_loss(bb, **kw), backbone_nonbonded_clash_loss(bb, **kw)):
        for key, value in standalone['losses'].items():
            assert torch.equal(value, full['losses'][key])
