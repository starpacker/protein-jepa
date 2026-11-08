"""Synthetic float64 geometry/gradient tests; no cache, checkpoint or model IO."""
import math

import pytest
import torch

from evaluation.backbone_metrics_v2 import _dihedral
from protein_jepa_frame.models.stage_c import joint_frame_oracle_v1 as oracle


def quartet(angle=.4,central=1.,left=1.,right=1.):
    return torch.tensor([[left,0.,0.],[0.,0.,0.],[0.,0.,central],
                         [right*math.cos(angle),right*math.sin(angle),central]],dtype=torch.float64)


@pytest.mark.parametrize('reflection',[False,True])
def test_strict_valid_components_values_and_gradients_match_frozen_dihedral(reflection):
    gen=torch.Generator().manual_seed(917)
    points=torch.randn(80,4,3,generator=gen,dtype=torch.float64)
    if reflection:points[...,0]*=-1
    points.requires_grad_()
    angle,valid=_dihedral(*points.unbind(-2))
    assert valid.all()
    c,r=oracle.confidence_dihedral_components(points)
    expected=torch.stack((angle.sin(),angle.cos()),-1)
    torch.testing.assert_close(c,expected,atol=3e-14,rtol=3e-14)
    torch.testing.assert_close(r,torch.ones_like(r),atol=3e-14,rtol=3e-14)
    target_angle=torch.linspace(-2.,2.,80,dtype=torch.float64)
    target=torch.stack((target_angle.sin(),target_angle.cos()),-1)
    actual=oracle.confidence_circular_loss(points,target).sum()
    reference=(1-torch.cos(angle-target_angle)).sum()
    torch.testing.assert_close(actual,reference,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(actual,points,retain_graph=True)[0]
    gb=torch.autograd.grad(reference,points)[0]
    torch.testing.assert_close(ga,gb,atol=2e-12,rtol=2e-12)


@pytest.mark.parametrize('outer',[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.]])
def test_zero_central_axis_never_becomes_perfect_cosine(outer):
    points=quartet(central=0.)
    points[3]=torch.tensor(outer,dtype=torch.float64)
    points.requires_grad_()
    c,r=oracle.confidence_dihedral_components(points)
    assert torch.equal(c,torch.zeros(2,dtype=torch.float64)) and float(r)==0.
    loss=oracle.confidence_circular_loss(points,torch.tensor([0.,1.],dtype=torch.float64))
    assert float(loss)==2.
    assert torch.isfinite(torch.autograd.grad(loss,points)[0]).all()


@pytest.mark.parametrize('kind',['all_zero','left_zero','right_zero','left_parallel','right_parallel',
                                  'tiny_central','tiny_left','tiny_right','tiny_central_left_parallel'])
def test_each_necessary_geometry_degeneracy_has_finite_gradients(kind):
    points=quartet()
    if kind=='all_zero':points.zero_()
    elif kind=='left_zero':points[0]=points[1]
    elif kind=='right_zero':points[3]=points[2]
    elif kind=='left_parallel':points[0]=torch.tensor([0.,0.,2.])
    elif kind=='right_parallel':points[3]=torch.tensor([0.,0.,2.])
    elif kind=='tiny_central':points=quartet(central=1e-9)
    elif kind=='tiny_left':points=quartet(left=1e-9)
    elif kind=='tiny_right':points=quartet(right=1e-9)
    else:
        points=quartet(central=1e-9);points[0]=torch.tensor([0.,0.,2.])
    points.requires_grad_()
    target=torch.tensor([0.,1.],dtype=torch.float64)
    c,r=oracle.confidence_dihedral_components(points)
    loss=oracle.confidence_circular_loss(points,target)
    assert -1e-14 <= float(r) <= 1.+1e-14 and float(loss)>1.99
    gradient=torch.autograd.grad(loss,points)[0]
    assert torch.isfinite(gradient).all()
    if not kind.startswith('tiny_') or kind.endswith('parallel'):
        assert float(loss)==2. and torch.equal(c,torch.zeros_like(c))
    if kind=='all_zero':assert torch.equal(gradient,torch.zeros_like(gradient))


@pytest.mark.parametrize('left',[1.,1e-7,0.])
def test_exact_opposite_direction_is_flat_in_confidence(left):
    points=quartet(left=left)
    points[3]=torch.tensor([-1.,0.,1.],dtype=torch.float64)
    points.requires_grad_()
    loss=oracle.confidence_circular_loss(points,torch.tensor([0.,1.],dtype=torch.float64))
    assert float(loss)==2.
    assert torch.isfinite(torch.autograd.grad(loss,points)[0]).all()


def test_target_validity_uses_frozen_threshold_not_confidence_epsilon():
    points=torch.stack([quartet(left=0.),quartet(left=5e-11),quartet(left=2e-10),quartet(left=2e-6)])
    units,valid=oracle.dihedral_target_units(points)
    angle,oldvalid=_dihedral(*points.unbind(-2))
    assert torch.equal(valid,oldvalid) and valid.tolist()==[False,False,True,True]
    torch.testing.assert_close(units[valid],torch.stack((angle.sin(),angle.cos()),-1)[valid])
    assert oracle.confidence_dihedral_components(points)[1][2] < 1.


def test_nonzero_strict_invalid_prediction_uses_continuous_penalty_below_two():
    points=quartet(angle=0.,left=5e-11).requires_grad_()
    _,valid=oracle.dihedral_target_units(points)
    assert not bool(valid)
    loss=oracle.confidence_circular_loss(points,torch.tensor([0.,1.],dtype=torch.float64))
    torch.testing.assert_close(loss,torch.tensor(2.-1e-4,dtype=torch.float64))
    assert 1.99<float(loss.detach())<2.
    gradient=torch.autograd.grad(loss,points)[0]
    assert torch.isfinite(gradient).all() and float(gradient.norm())>0.


def test_zero_correction_exact_parity_proper_rotation_ca_and_local_distances():
    gen=torch.Generator().manual_seed(11)
    bb=torch.randn(1,5,4,3,generator=gen,dtype=torch.float64)
    vectors=torch.zeros(1,5,3,dtype=torch.float64,requires_grad=True)
    moved,rotation=oracle.rotate_atom_family(bb,vectors)
    assert torch.equal(moved,bb)
    assert torch.equal(rotation,torch.eye(3,dtype=torch.float64).expand(1,5,3,3))
    loss=moved.square().sum();assert torch.isfinite(torch.autograd.grad(loss,vectors)[0]).all()
    vectors=torch.randn(1,5,3,generator=gen,dtype=torch.float64)*.1
    moved,rotation=oracle.rotate_atom_family(bb,vectors)
    assert torch.equal(moved[:,:,1],bb[:,:,1])
    torch.testing.assert_close(torch.cdist(moved[0],moved[0]),torch.cdist(bb[0],bb[0]),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(torch.linalg.det(rotation),torch.ones(1,5,dtype=torch.float64),atol=1e-12,rtol=0)


def fixture():
    gen=torch.Generator().manual_seed(94)
    initial=torch.randn(1,4,4,3,generator=gen,dtype=torch.float64)
    initial+=torch.arange(4,dtype=torch.float64)[None,:,None,None]*torch.tensor([2.,.2,.1])
    target=initial.clone();target[:,:,[0,2,3]]+=torch.tensor([.3,-.2,.15])
    kwargs=dict(residue_mask=torch.ones(4,dtype=torch.bool),atom_mask=torch.ones(4,4,dtype=torch.bool),
        chemical_aatype=torch.zeros(4,dtype=torch.long),geometry_row_ids=torch.tensor([9,2,81,41]),
        peptide_edges=torch.tensor([[0,1],[1,2],[2,3]]),
        initial_R=torch.eye(3,dtype=torch.float64).expand(1,4,3,3).clone(),
        psi_logits=torch.randn(1,4,36,generator=gen,dtype=torch.float64))
    return initial,target,kwargs


@pytest.mark.parametrize('torsion_index',[0,1],ids=['phi','psi'])
def test_psf_neighbor_frames_both_receive_frozen_signed_torsion_gradients(torsion_index):
    initial,_,_=fixture()
    edges=torch.tensor([[0,1]])
    vectors=torch.zeros(1,4,3,dtype=torch.float64,requires_grad=True)
    moved,_=oracle.rotate_atom_family(initial,vectors)
    points=oracle._edge_quartets(moved,edges)[0,torsion_index]
    angle,valid=_dihedral(*points.unbind(-2))
    assert bool(valid)
    target_angle=.53
    target=torch.tensor([math.sin(target_angle),math.cos(target_angle)],dtype=torch.float64)
    actual=oracle.confidence_circular_loss(points,target)
    reference=1-torch.cos(angle-target_angle)
    ga=torch.autograd.grad(actual,vectors,retain_graph=True)[0]
    gb=torch.autograd.grad(reference,vectors)[0]
    torch.testing.assert_close(ga,gb,atol=2e-12,rtol=2e-12)
    assert bool((torch.linalg.vector_norm(ga[0,:2],dim=-1)>1e-8).all())
    assert torch.equal(ga[0,2:],torch.zeros_like(ga[0,2:]))


def test_actual_fixed200_optimizer_preserves_witnesses_bounds_and_static_counts():
    initial,target,kw=fixture()
    before=[initial.clone(),target.clone(),kw['initial_R'].clone(),kw['psi_logits'].clone()]
    result=oracle.optimize_joint_frames(initial,target,**kw)
    assert result['status']=='completed' and len(result['trace'])==201
    assert [r['step'] for r in result['trace']]==list(range(201))
    assert result['conventions']['selection']=='final iteration only; no best iterate'
    assert result['trace'][-1]['gradient_norm_before_clip'] is None
    assert all(r['gradient_norm_before_clip'] is not None for r in result['trace'][:-1])
    assert torch.equal(result['final_backbone'][:,:,1],initial[:,:,1])
    assert torch.equal(result['psi_logits'],kw['psi_logits']) and torch.equal(result['initial_R'],kw['initial_R'])
    assert float(torch.linalg.vector_norm(result['rotation_vectors'],dim=-1).max())<=math.pi/6+1e-14
    torch.testing.assert_close(result['final_R'],result['delta_rotation']@kw['initial_R'])
    assert result['trace'][-1]['total']==result['final_losses']['total']
    for field in ['n_c_n','n_ca_c_n','n_c_n_ca','n_nonbonded_atom_pairs']:
        counts=[r['geometry_counts'][field] for r in result['trace']]
        assert all(torch.equal(counts[0],x) for x in counts)
    for old,new in zip(before,[initial,target,kw['initial_R'],kw['psi_logits']]):assert torch.equal(old,new)


def test_predicted_degeneracy_cannot_leave_fixed_torsion_denominator():
    initial,target,kw=fixture()
    fixed=oracle._fixed_support(initial,target,kw['residue_mask'],kw['atom_mask'],kw['peptide_edges'])
    degenerate=torch.zeros_like(initial,requires_grad=True)
    terms,audit=oracle._objective(degenerate,target,fixed,kw['chemical_aatype'],kw['peptide_edges'],kw['geometry_row_ids'],kw['residue_mask'])
    assert audit['torsion_count']==6 and audit['predicted_torsion_invalid_on_fixed_support']==6
    assert float(terms['torsion_circular'])==2.
    assert torch.isfinite(torch.autograd.grad(terms['total'],degenerate)[0]).all()


def test_static_missing_atoms_masked_nan_and_empty_torsion_keep_fixed_support():
    initial,target,kw=fixture();kw['peptide_edges']=torch.empty(0,2,dtype=torch.long)
    kw['atom_mask'][1,3]=False;initial[:,1,3]=torch.nan;target[:,1,3]=torch.nan
    fixed=oracle._fixed_support(initial,target,kw['residue_mask'],kw['atom_mask'],kw['peptide_edges'])
    assert fixed['nco_atom_count']==11 and fixed['phi_count']==fixed['psi_count']==0
    safe=torch.where(kw['atom_mask'][None,:,:,None],initial,0.).requires_grad_()
    target=torch.where(kw['atom_mask'][None,:,:,None],target,0.)
    terms,audit=oracle._objective(safe,target,fixed,kw['chemical_aatype'],kw['peptide_edges'],kw['geometry_row_ids'],kw['residue_mask'])
    assert float(terms['torsion_circular'])==0. and audit['torsion_count']==0
    assert torch.isfinite(torch.autograd.grad(terms['total'],safe)[0]).all()


def test_optimization_failure_exposes_partial_attempt_without_best_selection(monkeypatch):
    initial,target,kw=fixture();original=oracle._objective;calls=[]
    def failing(*args,**kwargs):
        calls.append(True)
        if len(calls)==3:raise FloatingPointError('synthetic stop')
        return original(*args,**kwargs)
    monkeypatch.setattr(oracle,'_objective',failing)
    with pytest.raises(oracle.JointFrameOptimizationError) as caught:
        oracle.optimize_joint_frames(initial,target,**kw)
    partial=caught.value.partial
    assert partial['status']=='failed' and len(partial['trace'])==2
    assert torch.equal(partial['final_backbone'][:,:,1],initial[:,:,1])
    assert 'delta_rotation' in partial and 'rotation_vectors' in partial


@pytest.mark.parametrize('bad',['duplicate_ids','self_edge','nonfinite_target','bad_R','no_labels'])
def test_invalid_static_contract_fails_before_optimizer(bad):
    initial,target,kw=fixture()
    if bad=='duplicate_ids':kw['geometry_row_ids'][1]=kw['geometry_row_ids'][0]
    elif bad=='self_edge':kw['peptide_edges'][0]=torch.tensor([0,0])
    elif bad=='nonfinite_target':target[:,0,0]=torch.nan
    elif bad=='bad_R':kw['initial_R'][0,0,0,0]=-1
    else:kw['atom_mask'][:,[0,2,3]]=False
    with pytest.raises(ValueError):oracle.optimize_joint_frames(initial,target,**kw)
