"""Synthetic production-decoder contracts; no datasets, checkpoints or sampling."""
import copy
import inspect

import pytest
import torch
from torch import nn

from protein_jepa_frame.models.geometric_delta_decoder import (
    BoundedEdgeCADecoder, GeometricDeltaCADecoder,
)
from protein_jepa_frame.models.geometric_local_residual_v1 import GeometricLocalResidualCADecoderV1
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames


def fixture(dtype=torch.float64, *, planar=True):
    torch.manual_seed(1701)
    base=GeometricDeltaCADecoder().to(dtype=dtype).eval()
    b,l,d=2,6,128
    anchor=torch.zeros(b,l,3,dtype=dtype)
    anchor[:,:,0]=torch.arange(l,dtype=dtype)*3.8
    if planar:
        anchor[:,:,1]=torch.tensor([0.,.5,-.25,.4,-.3,.2],dtype=dtype)
    anchor[1,:,:2]+=torch.tensor([1.,2.],dtype=dtype)
    backbone=anchor[:,:,None].expand(b,l,4,3).clone()
    backbone[:,:,0]+=torch.tensor([-.5,1.,0.],dtype=dtype)
    backbone[:,:,2]+=torch.tensor([1.5,0.,0.],dtype=dtype)
    backbone[:,:,3]+=torch.tensor([1.8,.8,0.],dtype=dtype)
    rotation,anchor=_extract_backbone_frames(backbone)
    h=torch.randn(b,l,d,dtype=dtype)
    inputs=[h+.3*torch.randn_like(h),h,torch.randint(0,21,(b,l)),
            torch.arange(l)[None].expand(b,l),torch.zeros(b,l,dtype=torch.long),
            rotation,anchor,torch.ones(b,l,dtype=torch.bool)]
    return base,inputs,backbone


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('training',[False,True])
def test_zero_residual_is_bitwise_production_output_input_and_old_parameter_gradient(dtype,training):
    base,inputs,_=fixture(dtype)
    base.train(training)
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    left=[value.clone() for value in inputs];right=[value.clone() for value in inputs]
    for values in (left,right):
        values[0].requires_grad_();values[1].requires_grad_()
    original=base(*left);actual=new(*right)
    assert torch.equal(original,actual)
    target=torch.linspace(-.2,.4,original.numel(),dtype=dtype).reshape_as(original)
    (original-target).square().sum().backward()
    (actual-target).square().sum().backward()
    for name,parameter in base.named_parameters():
        other=dict(new.named_parameters())[name]
        assert parameter.grad is not None and other.grad is not None,name
        assert torch.equal(parameter.grad,other.grad),name
    assert torch.equal(left[0].grad,right[0].grad)
    assert torch.equal(left[1].grad,right[1].grad)
    assert torch.isfinite(new.local_residual.weight.grad).all()
    assert new.local_residual.weight.grad.abs().sum()>0


@pytest.mark.parametrize('training',[False,True])
def test_zero_latent_change_is_exact_anchor_even_with_nonzero_residual(training):
    base,inputs,_=fixture()
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base).train(training)
    with torch.no_grad():new.local_residual.weight.normal_()
    inputs[0]=inputs[1].clone()
    inputs[-1][:,-1]=False
    assert torch.equal(new(*inputs),inputs[-2])


@pytest.mark.parametrize('planar',[False,True])
def test_nonzero_branch_can_leave_synthetic_anchor_plane(planar):
    base,inputs,_=fixture(planar=planar)
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    with torch.no_grad():new.local_residual.weight[1,0]=.5
    old=base(*inputs);actual=new(*inputs)
    # These particular CA anchors all have z=0, while each observed native
    # frame's local y axis is the backbone-plane normal (global +z).
    assert torch.equal(old[...,2],torch.zeros_like(old[...,2]))
    assert actual[...,2].abs().max()>.001
    assert torch.equal(actual[...,:2],old[...,:2])


def test_matching_observed_frames_give_proper_rigid_motion_equivariance():
    base,inputs,backbone=fixture()
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    with torch.no_grad():new.local_residual.weight.normal_(std=.2)
    q,_=torch.linalg.qr(torch.tensor([[.3,.9,-.2],[-.8,.1,.7],[.4,-.6,.5]],dtype=torch.float64))
    if torch.linalg.det(q)<0:q[:,-1]*=-1
    shift=torch.tensor([5.,-2.,3.],dtype=torch.float64)
    transformed=backbone@q.T+shift
    rotated=copy.copy(inputs)
    rotated[5],rotated[6]=_extract_backbone_frames(transformed)
    torch.testing.assert_close(rotated[5],q@inputs[5],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(new(*rotated),new(*inputs)@q.T+shift,rtol=2e-12,atol=2e-12)


def test_padding_is_ignored_and_nan_padding_rotations_are_safe():
    base,inputs,_=fixture()
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    with torch.no_grad():new.local_residual.weight.normal_(std=.1)
    inputs[-1][:,-2:]=False
    original=new(*inputs)
    changed=[value.clone() for value in inputs]
    changed[0][:,-2:]+=100.
    changed[1][:,-2:]-=90.
    changed[2][:,-2:]=20
    changed[5][:,-2:]=float('nan')
    changed[6][:,-2:]+=1000.
    actual=new(*changed)
    torch.testing.assert_close(actual[:,:-2],original[:,:-2],rtol=0.,atol=0.)
    assert torch.equal(actual[:,-2:],changed[6][:,-2:])
    actual[:,:-2].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())


def test_nonzero_branch_has_finite_gradients_into_shared_trunk_and_W():
    base,inputs,_=fixture()
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    with torch.no_grad():new.local_residual.weight.normal_(std=.1)
    inputs[0].requires_grad_()
    new(*inputs).square().mean().backward()
    for name,p in new.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    for name in ('input.weight','query.weight','key.weight','gate.weight',
                 'pair.0.weight','layers.0.linear1.weight','local_residual.weight'):
        assert dict(new.named_parameters())[name].grad.abs().sum()>0,name
    assert torch.isfinite(inputs[0].grad).all() and inputs[0].grad.abs().sum()>0


def test_base_bridge_is_explicit_strict_nonsharing_and_preserves_rng(monkeypatch):
    base,_,_=fixture()
    before=torch.get_rng_state().clone()
    def forbidden(*a,**k):raise AssertionError('no hidden checkpoint loading allowed')
    monkeypatch.setattr(torch,'load',forbidden)
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    assert torch.equal(torch.get_rng_state(),before)
    old_state=base.state_dict();new_state=new.state_dict()
    assert set(new_state)==set(old_state)|{'local_residual.weight'}
    for name,value in old_state.items():
        assert torch.equal(value,new_state[name]) and value.data_ptr()!=new_state[name].data_ptr()
    assert torch.equal(new.local_residual.weight,torch.zeros(3,128,dtype=torch.float64))
    assert new.local_residual.bias is None
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in base.parameters())==384
    with pytest.raises(RuntimeError,match='local_residual.weight'):
        new.load_state_dict(old_state)
    with pytest.raises(ValueError,match='state keys'):
        GeometricLocalResidualCADecoderV1.from_base_state_dict(new_state)
    with pytest.raises(TypeError,match='exact'):
        GeometricLocalResidualCADecoderV1.from_base_decoder(BoundedEdgeCADecoder())


@pytest.mark.parametrize('change',['missing','extra','shape','nonfinite','dtype','hidden','depth'])
def test_explicit_bridge_rejects_ambiguous_or_incompatible_state(change):
    base,_,_=fixture();state=copy.deepcopy(base.state_dict());kwargs={}
    if change=='missing':del state['gate.bias']
    elif change=='extra':state['foreign_weight']=torch.ones(1)
    elif change=='shape':state['query.weight']=state['query.weight'][:1]
    elif change=='nonfinite':state['input.weight'][0,0]=float('nan')
    elif change=='dtype':state['centers']=state['centers'].float()
    elif change=='hidden':kwargs['hidden']=64
    else:kwargs['depth']=2
    with pytest.raises(ValueError):GeometricLocalResidualCADecoderV1.from_base_state_dict(state,**kwargs)


def test_new_state_roundtrip_requires_and_restores_the_residual():
    base,inputs,_=fixture()
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    with torch.no_grad():new.local_residual.weight.normal_(std=.1)
    restored=GeometricLocalResidualCADecoderV1().double().eval()
    restored.load_state_dict(copy.deepcopy(new.state_dict()),strict=True)
    assert torch.equal(restored(*inputs),new(*inputs))


def test_original_external_32_step_projection_remains_bitwise_equal_at_initialization():
    from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
    base,inputs,_=fixture(torch.float32)
    new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    bundles=[]
    for decoder in (base,new):
        bundle=CAComponentBundle.__new__(CAComponentBundle);nn.Module.__init__(bundle)
        bundle.decoder=decoder;bundle.projection_steps=32;bundles.append(bundle)
    assert torch.equal(bundles[0].decode_ca(*inputs),bundles[1].decode_ca(*inputs))
    assert not hasattr(new,'projection_steps')


@pytest.mark.parametrize('invalid',['shape','nonfinite','empty_mask'])
def test_invalid_observed_frame_fails_without_future_frame_fallback(invalid):
    base,inputs,_=fixture();new=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    if invalid=='shape':inputs[5]=inputs[5][...,:2,:]
    elif invalid=='nonfinite':inputs[5][0,0,0,0]=float('nan')
    else:inputs[-1][0]=False
    with pytest.raises(ValueError):new(*inputs)


def test_public_forward_has_only_latents_and_observed_geometry():
    assert tuple(inspect.signature(GeometricLocalResidualCADecoderV1.forward).parameters)==(
        'self','future_h','anchor_h','aatype','residue_index','chain_id','rotation','anchor','mask')
