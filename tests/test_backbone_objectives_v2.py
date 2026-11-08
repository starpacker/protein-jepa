"""CPU synthetic tests of actual-coordinate backbone losses; no dataset access."""
import math

import pytest
import torch

from protein_jepa_frame.models.stage_c.backbone_objectives_v2 import (
    backbone_connection_loss, backbone_geometry_losses_v2, backbone_nonbonded_clash_loss,
)
from protein_jepa_frame.vendored.openfold.np import residue_constants as rc


def inputs(length=3):
    generator = torch.Generator().manual_seed(27)
    bb = torch.randn(length,4,3,generator=generator,dtype=torch.float64)
    bb += torch.arange(length,dtype=bb.dtype)[:,None,None]*bb.new_tensor([1.7,.3,.2])
    return bb, dict(chemical_aatype=torch.zeros(length,dtype=torch.long),
        residue_index=torch.arange(length), chain_id=torch.zeros(length,dtype=torch.long),
        atom_mask=torch.ones(length,4,dtype=torch.bool))


def pair_only(distance=.3, names=(2,0)):
    bb, kw = inputs(2)
    bb.zero_()
    bb[1,names[1],0] = distance
    kw['atom_mask'].zero_()
    kw['atom_mask'][0,names[0]] = kw['atom_mask'][1,names[1]] = True
    return bb,kw


def test_CN_uses_next_proline_chemical_ID_and_unknown_is_not_guessed():
    bb,kw = pair_only(1.6)
    standard = backbone_connection_loss(bb,**kw,tolerance_factor=0.)
    kw['chemical_aatype'][1] = rc.restype_order['P']
    proline = backbone_connection_loss(bb,**kw,tolerance_factor=0.)
    length = math.sqrt(1.6**2+1e-6)
    for result,index in ((standard,0),(proline,1)):
        expected = math.sqrt((length-rc.between_res_bond_length_c_n[index])**2+1e-6)
        assert float(result['losses']['c_n']) == pytest.approx(expected,abs=1e-12)
        assert int(result['counts']['n_c_n']) == 1
        assert int(result['counts']['n_c_n_proline']) == index
    kw['chemical_aatype'][0] = 20
    previous_unknown = backbone_connection_loss(bb,**kw,tolerance_factor=0.)
    assert int(previous_unknown['counts']['n_c_n_proline']) == 1
    kw['chemical_aatype'][1] = 20
    unknown = backbone_connection_loss(bb.requires_grad_(),**kw)
    assert unknown['losses']['c_n'] == 0
    assert int(unknown['counts']['n_c_n']) == 0
    assert int(unknown['counts']['n_c_n_unknown_next_excluded']) == 1
    assert int(unknown['counts']['n_unknown_residues']) == 2
    assert torch.equal(torch.autograd.grad(unknown['losses']['c_n'],bb)[0],torch.zeros_like(bb))


def test_ca_c_n_uses_cosine_standard_deviation_not_CN_length_std():
    bb,kw = inputs(2)
    bb.zero_(); kw['atom_mask'].zero_()
    cosine = rc.between_res_cos_angles_ca_c_n[0]+.2
    bb[0,1] = torch.tensor([1.,0.,0.])
    bb[1,0] = bb.new_tensor([cosine,math.sqrt(1-cosine*cosine),0.])
    kw['atom_mask'][0,1:3] = True; kw['atom_mask'][1,0] = True
    result = backbone_connection_loss(bb,**kw,tolerance_factor=1.)
    actual = cosine/(1+1e-6)
    expected = math.sqrt((actual-rc.between_res_cos_angles_ca_c_n[0])**2+1e-6)-.0311
    assert float(result['losses']['ca_c_n']) == pytest.approx(expected,abs=1e-12)
    assert result['conventions']['ca_c_n_cos_mean_std'] == [-.4473,.0311]
    assert backbone_connection_loss(bb,**kw)['losses']['ca_c_n'] == 0


@pytest.mark.parametrize('key', ['c_n','ca_c_n','c_n_ca','backbone_interresidue_clash'])
def test_actual_coordinate_gradient_is_nonzero_and_matches_central_difference(key):
    bb,kw = inputs()
    bb.requires_grad_()
    result = backbone_geometry_losses_v2(bb,**kw,tolerance_factor=0.)
    value = result['losses'][key]
    gradient = torch.autograd.grad(value,bb)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-5
    # Compare a deterministic nonstationary coordinate, away from hinge knots.
    index = int(gradient.abs().flatten().argmax())
    plus,minus = bb.detach().clone(),bb.detach().clone()
    plus.flatten()[index] += 1e-5; minus.flatten()[index] -= 1e-5
    vp = backbone_geometry_losses_v2(plus,**kw,tolerance_factor=0.)['losses'][key]
    vm = backbone_geometry_losses_v2(minus,**kw,tolerance_factor=0.)['losses'][key]
    assert float(gradient.flatten()[index]) == pytest.approx(float((vp-vm)/2e-5),rel=2e-5,abs=2e-7)


def test_real_frame_rotation_receives_geometry_gradient_without_moving_frozen_CA():
    from protein_jepa_frame.models.stage_c.openfold_atom37 import decoder_frames_to_atom37
    from protein_jepa_frame.models.stage_c.geometry import rotation_matrix_from_quaternion
    bb,kw = inputs(3)
    quaternion = torch.tensor([[1.,.1,.2,.3],[1.,.3,-.2,.1],[1.,-.1,.2,-.3]],requires_grad=True)
    ca = torch.tensor([[0.,0.,0.],[3.8,.3,.1],[7.3,1.6,.2]])
    R = rotation_matrix_from_quaternion(quaternion)[None]
    torsions = torch.zeros(1,3,7,requires_grad=True)
    atoms,_ = decoder_frames_to_atom37(R,ca[None],kw['chemical_aatype'][None],torsions)
    actual = atoms[0,:, [0,1,2,4]]
    assert torch.equal(actual[:,1],ca)
    result = backbone_geometry_losses_v2(actual,**kw,tolerance_factor=0.)
    grad = torch.autograd.grad(sum(result['losses'].values()),quaternion)[0]
    assert torch.isfinite(grad).all() and grad.abs().max() > 1e-5
    assert not ca.requires_grad


@pytest.mark.parametrize('change', ['different_chain','sequence_gap','missing_atom'])
def test_connection_masks_never_bridge_chain_gap_or_missing_atoms(change):
    bb,kw = inputs(2)
    if change == 'different_chain': kw['chain_id'][1] = 1
    elif change == 'sequence_gap': kw['residue_index'][1] = 2
    else: kw['atom_mask'][1,0] = False
    result = backbone_connection_loss(bb.requires_grad_(),**kw,tolerance_factor=0.)
    assert all(int(result['counts']['n_'+key]) == 0 for key in ('c_n','ca_c_n','c_n_ca'))
    assert all(value == 0 and value.requires_grad for value in result['losses'].values())


def test_only_actual_peptide_bond_is_excluded_other_adjacent_atom_pairs_remain():
    bb,kw = pair_only()
    bonded = backbone_nonbonded_clash_loss(bb,**kw)
    assert int(bonded['counts']['n_nonbonded_atom_pairs']) == 0
    assert int(bonded['counts']['n_excluded_peptide_atom_pairs']) == 1
    assert bonded['losses']['backbone_interresidue_clash'] == 0
    for change in ('chain','gap'):
        values = {key:value.clone() for key,value in kw.items()}
        values['chain_id' if change == 'chain' else 'residue_index'][1] = 3
        clash = backbone_nonbonded_clash_loss(bb,**values)
        assert int(clash['counts']['n_nonbonded_atom_pairs']) == 1
        assert int(clash['counts']['n_excluded_peptide_atom_pairs']) == 0
        assert clash['losses']['backbone_interresidue_clash'] > 1
    other,metadata = pair_only(names=(0,0))
    assert backbone_nonbonded_clash_loss(other,**metadata)['losses']['backbone_interresidue_clash'] > 1


def test_cross_chain_equal_numbers_and_unknown_atoms_still_clash():
    bb,kw = pair_only(names=(3,0))
    kw['residue_index'].zero_(); kw['chain_id'][1] = 1; kw['chemical_aatype'][:] = 20
    result = backbone_geometry_losses_v2(bb,**kw)
    assert int(result['counts']['n_c_n']) == 0
    assert int(result['counts']['n_nonbonded_atom_pairs']) == 1
    assert int(result['counts']['n_clashing_atom_pairs']) == 1
    expected = rc.van_der_waals_radius['O']+rc.van_der_waals_radius['N']-1.5-math.sqrt(.3**2+1e-10)
    assert float(result['losses']['backbone_interresidue_clash']) == pytest.approx(expected,abs=1e-12)


def test_permutation_and_uint8_metadata_preserve_physical_topology_and_values():
    bb,kw = inputs(4)
    kw['chemical_aatype'][2] = rc.restype_order['P']
    original = backbone_geometry_losses_v2(bb,**kw,tolerance_factor=0.)
    order = torch.tensor([3,1,0,2])
    for dtype in (torch.long,torch.uint8,torch.int16):
        metadata = {key:(value[order] if value.dtype == torch.bool else value[order].to(dtype))
                    for key,value in kw.items()}
        result = backbone_geometry_losses_v2(bb[order],**metadata,tolerance_factor=0.)
        for key,value in original['losses'].items():
            assert torch.allclose(result['losses'][key],value,atol=1e-12,rtol=1e-12)
        for key,value in original['counts'].items(): assert torch.equal(result['counts'][key],value)


def test_masked_padding_nan_inf_are_excluded_before_arithmetic_and_have_zero_gradient():
    bb,kw = inputs(2)
    original = backbone_geometry_losses_v2(bb,**kw,tolerance_factor=0.)
    padded = torch.cat([bb,torch.full((1,4,3),float('nan'),dtype=bb.dtype)]).requires_grad_()
    padded.data[-1,0,0] = float('inf')
    kwargs = {key:torch.cat([value,torch.zeros_like(value[:1])]) for key,value in kw.items()}
    kwargs['chemical_aatype'][-1] = -1
    result = backbone_geometry_losses_v2(padded,**kwargs,residue_mask=torch.tensor([True,True,False]),tolerance_factor=0.)
    for key,value in original['losses'].items(): assert torch.equal(result['losses'][key],value)
    gradient = torch.autograd.grad(sum(result['losses'].values()),padded)[0]
    assert torch.isfinite(gradient).all() and torch.equal(gradient[-1],torch.zeros_like(gradient[-1]))
    with pytest.raises(ValueError,match='nonfinite'):
        kwargs['atom_mask'][-1,1] = True
        kwargs['chemical_aatype'][-1] = 0
        backbone_geometry_losses_v2(padded,**kwargs)


def test_unknown_keeps_generic_angles_and_missing_O_does_not_hide_CN():
    bb,kw = inputs(2)
    kw['chemical_aatype'][1] = 20; kw['atom_mask'][:,3] = False
    result = backbone_geometry_losses_v2(bb,**kw)
    assert int(result['counts']['n_c_n']) == 0
    assert int(result['counts']['n_ca_c_n']) == int(result['counts']['n_c_n_ca']) == 1
    assert int(result['counts']['n_nonbonded_atom_pairs']) == 8


def test_broadcast_equal_sample_aggregation_and_explicit_ambiguous_axis():
    bb,kw = inputs()
    coordinates = bb.expand(2,3,*bb.shape).clone()
    metadata = {key:value.expand(2,*value.shape).clone() for key,value in kw.items()}
    result = backbone_geometry_losses_v2(coordinates,**metadata)
    assert result['per_sample']['c_n'].shape == (2,3)
    assert result['counts']['n_valid_atoms'].shape == (2,3)
    with pytest.raises(ValueError,match='ambiguous'):
        backbone_geometry_losses_v2(coordinates[:,:2],**metadata)
    explicit = {key:value[:,None] for key,value in metadata.items()}
    backbone_geometry_losses_v2(coordinates[:,:2],**explicit)
    # Empty members do not dilute the loss or masquerade as valid geometry.
    mask = kw['atom_mask'].expand(2,*kw['atom_mask'].shape).clone(); mask[1] = False
    one = backbone_geometry_losses_v2(bb,**kw,tolerance_factor=0.)
    batched = backbone_geometry_losses_v2(bb.expand(2,*bb.shape),**{**kw,'atom_mask':mask},tolerance_factor=0.)
    for key,value in one['losses'].items(): assert torch.equal(batched['losses'][key],value)


def test_single_residue_empty_terms_and_half_precision_policy():
    bb,kw = inputs(1)
    bb = bb.float().requires_grad_()
    result = backbone_geometry_losses_v2(bb,**kw)
    assert all(value == 0 for value in result['losses'].values())
    assert torch.equal(torch.autograd.grad(sum(result['losses'].values()),bb)[0],torch.zeros_like(bb))
    half,kw = inputs(2)
    promoted = backbone_geometry_losses_v2(half.to(torch.bfloat16),**kw)
    assert all(value.dtype == torch.float32 for value in promoted['losses'].values())


def test_rigid_transform_invariance_and_input_contract_failures():
    bb,kw = inputs()
    base = backbone_geometry_losses_v2(bb,**kw,tolerance_factor=0.)
    rotation = bb.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    transformed = backbone_geometry_losses_v2(bb@rotation+bb.new_tensor([8.,-4.,2.]),**kw,tolerance_factor=0.)
    for key,value in base['losses'].items(): assert torch.allclose(transformed['losses'][key],value,atol=1e-12)
    for bad in (-1.,float('nan'),float('inf'),True):
        with pytest.raises(ValueError): backbone_geometry_losses_v2(bb,**kw,tolerance_factor=bad)
        with pytest.raises(ValueError): backbone_geometry_losses_v2(bb,**kw,overlap_tolerance_angstrom=bad)
    for key,bad in (('chemical_aatype',torch.tensor([0,21,0])),
                    ('chemical_aatype',kw['chemical_aatype'].float()),
                    ('residue_index',torch.tensor([0,0,2])),
                    ('atom_mask',kw['atom_mask'].float())):
        with pytest.raises(ValueError): backbone_geometry_losses_v2(bb,**{**kw,key:bad})
