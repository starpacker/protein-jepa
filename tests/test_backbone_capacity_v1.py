"""Synthetic CPU tests; labels enter only the explicitly oracle capacity fit."""
import math

import pytest
import torch

from protein_jepa_frame.models.stage_c.backbone_capacity_v1 import (
    _proper_fit, fixed_ca_nco_capacity, ideal_nco_template, psi_domain_bounds,
)


def rotation():
    axis = torch.tensor([.2, -.5, .3], dtype=torch.float64)
    axis /= axis.norm()
    x, y, z = axis
    cross = torch.tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(.7)*cross + (1-math.cos(.7))*(cross@cross)


def example(length=3, grid=32):
    aa = torch.arange(length) % 21
    ca = torch.arange(length, dtype=torch.float64)[:, None]*torch.tensor([[2.3, -.7, .5]], dtype=torch.float64)
    lo, hi = psi_domain_bounds()
    psi = lo+(torch.arange(length, dtype=torch.float64)+.5)*(hi-lo)/grid
    nco = ideal_nco_template(aa, psi)@rotation().T + ca[:, None]
    bb = torch.stack((nco[:, 0], ca, nco[:, 1], nco[:, 2]), -2)
    return bb, ca, dict(chemical_aatype=aa, atom_mask=torch.ones(length, 4, dtype=torch.bool))


def test_proper_procrustes_no_translation_or_reflection():
    x = torch.eye(3, dtype=torch.float64)
    q, sse = _proper_fit(x, x@rotation(), torch.ones(3, dtype=torch.float64))
    assert sse < 1e-28
    assert torch.allclose(q, rotation(), atol=1e-14, rtol=0)
    reflected = x.clone(); reflected[0, 0] = -1
    q, sse = _proper_fit(x, reflected, torch.ones(3, dtype=torch.float64))
    assert q.det() == pytest.approx(1.)
    assert sse == pytest.approx(4.)
    _, shifted = _proper_fit(x, x+10, torch.ones(3, dtype=torch.float64))
    assert shifted > 100


def test_all_chemical_templates_and_unknown_match_actual_openfold():
    from protein_jepa_frame.models.stage_c.openfold_atom37 import frames_to_atom37
    aa = torch.arange(21)
    for psi in (-2.4, .0, 1.9):
        torsion = torch.zeros(1, 21, 7, dtype=torch.float64); torsion[..., 1] = psi
        ca = torch.tensor([1.2, -2.7, 3.5], dtype=torch.float64).expand(1, 21, 3)
        atoms, mask = frames_to_atom37(rotation()[None, None].expand(1, 21, 3, 3), ca, aa[None], torsion)
        expected = ideal_nco_template(aa, psi)@rotation().T + ca[0, :, None]
        assert mask[..., [0, 2, 4]].all()
        assert torch.allclose(atoms[0, :, [0, 2, 4]], expected, atol=8e-7, rtol=0)
    assert torch.equal(ideal_nco_template(torch.tensor(20), .7), ideal_nco_template(torch.tensor(7), .7))


def test_on_grid_target_zero_lower_and_achievable_upper_exact_CA():
    bb, ca, kw = example()
    result = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=32, grid_chunk_size=7)
    assert result['per_sample']['ideal_rmsd_lower'] == 0
    assert result['per_sample']['ideal_rmsd_upper'] < 1e-6
    assert result['per_sample']['actual_witness_rmsd'] < 1e-6
    assert torch.equal(result['witness']['actual_backbone'][:, 1], ca)
    assert torch.allclose(result['witness']['ideal_backbone'], bb, atol=1e-13, rtol=0)
    r = result['witness']['R_openfold']
    assert torch.allclose(r@r.transpose(-1, -2), torch.eye(3, dtype=torch.float64), atol=2e-14, rtol=0)
    assert torch.allclose(torch.linalg.det(r), torch.ones(3, dtype=torch.float64), atol=2e-14, rtol=0)
    assert result['conventions']['actual_float32_global_bound'] is False


def test_global_interval_contains_independent_nested_dense_grid_minimum():
    bb, ca, kw = example(4)
    noise = torch.randn(bb.shape, generator=torch.Generator().manual_seed(19), dtype=bb.dtype)*.4
    bb = bb+noise
    coarse = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=24, grid_chunk_size=5)
    lo, hi = psi_domain_bounds()
    dense_psi = lo+(torch.arange(24*31, dtype=torch.float64)+.5)*(hi-lo)/(24*31)
    template = ideal_nco_template(kw['chemical_aatype'][None], dense_psi[:, None])
    y = bb[:, [0, 2, 3]]-ca[:, None]
    # Independent analytic SVD objective (sum norms minus proper singular sum),
    # avoiding the module's fitted-residual expression and grid loop.
    u, singular, vh = torch.linalg.svd(template.transpose(-1, -2)@y[None])
    correction = torch.linalg.det(u@vh)
    optimum = template.square().sum((-1, -2))+y.square().sum((-1, -2))[None] - 2*(singular[..., :2].sum(-1)+correction*singular[..., 2])
    dense_min = optimum.min(0).values
    values = coarse['per_residue']
    assert (values['ideal_sse_lower'] <= dense_min+1e-12).all()
    assert (dense_min <= values['ideal_sse_upper']+1e-12).all()
    assert (values['ideal_sse_upper']-values['ideal_sse_lower'] <= values['grid_gap_bound_sse']+2*values['numerical_guard_sse']+1e-12).all()


def test_closed_form_radial_lower_is_tight_for_zero_relative_targets():
    bb, ca, kw = example()
    bb = ca[:, None].expand_as(bb).clone()
    result = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=8)
    expected = ideal_nco_template(kw['chemical_aatype'], .0).square().sum((-1, -2))
    assert torch.allclose(result['per_residue']['ideal_sse_lower'], expected, atol=1e-11, rtol=0)
    assert torch.allclose(result['per_residue']['ideal_sse_upper'], expected, atol=1e-11, rtol=0)


def test_masks_broadcast_batch_empty_unknown_and_nonfinite_policy():
    bb, ca, kw = example()
    bb = bb.expand(2, -1, -1, -1).clone(); ca = ca.expand(2, -1, -1).clone()
    bb[:, -1] = float('nan'); ca[:, -1] = float('nan')
    kw['chemical_aatype'][-1] = -123
    kw['atom_mask'][-1] = False; kw['atom_mask'][0, 3] = False
    result = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=8)
    assert result['per_sample']['n_nco_atoms'].tolist() == [5, 5]
    assert result['per_residue']['n_nco_atoms'].tolist() == [[2, 3, 0], [2, 3, 0]]
    assert torch.isfinite(result['witness']['actual_backbone']).all()
    assert (result['witness']['actual_backbone'][:, -1] == 0).all()
    kw['chemical_aatype'][1] = 20
    result = fixed_ca_nco_capacity(bb, ca, **kw, residue_mask=torch.zeros(3, dtype=torch.bool), grid_size=4)
    assert all((value == 0).all() for value in result['per_sample'].values())
    bb[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite valid'):
        fixed_ca_nco_capacity(bb, ca, **kw, grid_size=4)


def test_default_psi_domain_matches_actual_arithmetic_centers_and_witness_interior():
    from protein_jepa_frame.models.stage_c.structure_decoder import _torsion_logits_to_angles
    logits = torch.full((2, 1, 36), -1000.); logits[0, 0, 0] = logits[1, 0, -1] = 1000.
    endpoints = _torsion_logits_to_angles(logits).flatten().tolist()
    assert list(psi_domain_bounds()) == endpoints
    assert endpoints[0] > -math.pi and endpoints[1] < math.pi
    bb, ca, kw = example()
    result = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=4)
    assert ((result['witness']['psi']>endpoints[0]) & (result['witness']['psi']<endpoints[1])).all()
    assert psi_domain_bounds(psi_domain='full_circle_relaxation') == (-math.pi, math.pi)


def test_rigid_gauge_equivariance_and_chunk_invariance():
    bb, ca, kw = example()
    bb += torch.tensor([.1, -.3, .2], dtype=torch.float64)
    one = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=20, grid_chunk_size=1)
    chunked = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=20, grid_chunk_size=13)
    moved = fixed_ca_nco_capacity(bb@rotation()+2, ca@rotation()+2, **kw, grid_size=20)
    for key in ('ideal_sse_lower', 'ideal_sse_upper', 'ideal_nc_radial_sse_lower'):
        assert torch.allclose(one['per_residue'][key], chunked['per_residue'][key], atol=1e-13, rtol=0)
        assert torch.allclose(one['per_residue'][key], moved['per_residue'][key], atol=2e-13, rtol=0)


def test_actual_witness_reconstructs_frozen_producer_and_scoring():
    from protein_jepa_frame.models.stage_c.structure_decoder import reconstruct_atom37_from_frames
    bb, ca, kw = example()
    bb += .15
    result = fixed_ca_nco_capacity(bb, ca, **kw, grid_size=12)
    witness = result['witness']; torsions = torch.zeros(1, 3, 7, dtype=torch.float64)
    torsions[0, :, 1] = witness['psi']
    atoms, _ = reconstruct_atom37_from_frames(witness['R_decoder'][None], ca[None], kw['chemical_aatype'][None], torsions)
    actual = torch.stack((atoms[0, :, 0], ca, atoms[0, :, 2], atoms[0, :, 4]), -2)
    assert torch.equal(actual, witness['actual_backbone'])
    expected_sse = (actual[:, [0, 2, 3]]-bb[:, [0, 2, 3]]).square().sum((-1, -2))
    assert torch.allclose(result['per_residue']['actual_witness_sse'], expected_sse, atol=1e-15, rtol=0)


@pytest.mark.parametrize('bad', [dict(grid_size=True), dict(grid_size=1), dict(grid_chunk_size=0), dict(torsion_bins=1), dict(psi_domain='circle')])
def test_invalid_grid_domain_arguments_fail(bad):
    bb, ca, kw = example()
    with pytest.raises(ValueError): fixed_ca_nco_capacity(bb, ca, **kw, **bad)


def test_metadata_dtype_and_valid_ID_fail_closed():
    bb, ca, kw = example()
    for aa in (torch.zeros(3), torch.zeros(3, dtype=torch.bool), torch.tensor([0, 1, 21])):
        with pytest.raises(ValueError):
            fixed_ca_nco_capacity(bb, ca, atom_mask=kw['atom_mask'], chemical_aatype=aa, grid_size=4)
    with pytest.raises(ValueError, match='boolean'):
        fixed_ca_nco_capacity(bb, ca, chemical_aatype=kw['chemical_aatype'], atom_mask=kw['atom_mask'].int(), grid_size=4)
