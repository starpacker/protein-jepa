"""Synthetic explicit physical-support values/gradients; no actual data reads."""
import copy
import itertools

import numpy as np
import pytest
import torch

from protein_jepa_frame.training import physical_distance_path_objective_v1 as old
from protein_jepa_frame.training import physical_distance_path_objective_v2 as core
from test_physical_distance_path_objective_v1 import example,parameterized_path


def explicit(args):
    result=dict(args)
    ri,ci=result.pop('residue_index'),result.pop('chain_id')
    result['geometry_row_ids']=torch.arange(ri.shape[1])[None].expand_as(ri).clone()
    result['physical_pair_eligibility']=(ci[:,:,None]!=ci[:,None,:])|((ri[:,:,None]-ri[:,None,:]).abs()>=3)
    return result


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('horizon',[1,3])
def test_same_pairset_v1_values_full_path_gradients_bitwise(dtype,horizon):
    legacy=example(horizon=horizon,dtype=dtype)
    legacy['predicted'].requires_grad_();args=explicit(legacy)
    a=old.physical_distance_path_scores(**legacy)
    b=core.physical_distance_path_scores(**args)
    for name in a:
        if name=='pair_indices':assert all(torch.equal(x,y) for x,y in zip(a[name],b[name]))
        else:assert torch.equal(a[name],b[name])
    for loss in ('loss_absolute','loss_increment'):
        ga=torch.autograd.grad(a[loss],legacy['predicted'],retain_graph=True)[0]
        gb=torch.autograd.grad(b[loss],legacy['predicted'],retain_graph=True)[0]
        assert torch.equal(ga,gb)


def test_arbitrary_declared_pairs_all_used_independent_numpy_no_numeric_id_distance():
    args=explicit(example(batch=1));matrix=args['physical_pair_eligibility'];matrix.zero_()
    pairs=[(0,1),(0,5),(1,2),(2,4)]
    for i,j in pairs:matrix[0,i,j]=matrix[0,j,i]=True
    args['geometry_row_ids']=torch.tensor([[500,9,88,-7,17,2]])
    result=core.physical_distance_path_scores(**args)
    assert result['pair_indices'][0].tolist()==[list(p) for p in pairs]
    p=args['predicted'][0].numpy();r=args['reference'][0].numpy()
    dp=np.stack([np.linalg.norm(p[:,:,i]-p[:,:,j],axis=-1) for i,j in pairs],-1)
    dr=np.stack([np.linalg.norm(r[:,i]-r[:,j],axis=-1) for i,j in pairs],-1)
    for kind,x,y in [('absolute',dp[:,1:],dr[1:]),('increment',np.diff(dp,axis=1),np.diff(dr,axis=0))]:
        expected=[]
        for t in range(x.shape[1]):
            cross=np.mean([np.linalg.norm(a-y[t]) for a in x[:,t]])
            spread=np.mean([np.linalg.norm(x[i,t]-x[j,t]) for i,j in itertools.combinations(range(len(x)),2)])
            expected.append((cross-.5*spread)/np.sqrt(len(pairs)))
        np.testing.assert_allclose(result[kind+'_per_window_time'].numpy(),[expected],rtol=2e-14,atol=2e-14)
    renamed={**args,'geometry_row_ids':torch.arange(6)[None]}
    assert torch.equal(result['loss_increment'],core.physical_distance_path_scores(**renamed)['loss_increment'])


def test_h1_full_parameter_chain_identity_and_adjacent_frame_gradients():
    args=explicit(example(horizon=1))
    prediction,parameters,_=parameterized_path(args,1);args['predicted']=prediction
    result=core.physical_distance_path_scores(**args)
    a=torch.autograd.grad(result['loss_absolute'],parameters,retain_graph=True)
    b=torch.autograd.grad(result['loss_increment'],parameters)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-12,atol=2e-14)
    args=explicit(example(batch=1,horizon=3));args['predicted'].requires_grad_()
    result=core.physical_distance_path_scores(**args)
    grad=torch.autograd.grad(result['increment_per_window_time'][0,1],args['predicted'])[0]
    assert grad[:,:,1].norm()>0 and grad[:,:,2].norm()>0
    assert torch.equal(grad[:,:,0],torch.zeros_like(grad[:,:,0]))
    assert torch.equal(grad[:,:,3],torch.zeros_like(grad[:,:,3]))


def test_padding_nan_duplicate_masked_ids_and_extra_reference_support_irrelevant():
    args=explicit(example(batch=1));args['residue_mask'][0,-1]=False
    args['geometry_row_ids'][0,-1]=args['geometry_row_ids'][0,0]
    for name in ('predicted','reference','observed_anchor'):args[name][...,-1,:]=float('nan')
    args['reference_mask'][:,:,-1]=True;args['predicted'].requires_grad_()
    result=core.physical_distance_path_scores(**args)
    assert all(5 not in pair for pair in result['pair_indices'][0].tolist())
    grad=torch.autograd.grad(result['loss_increment'],args['predicted'])[0]
    assert torch.isfinite(grad).all() and torch.equal(grad[...,-1,:],torch.zeros_like(grad[...,-1,:]))


@pytest.mark.parametrize('bad',['one_member','nonfinite_pred','nonfinite_ref','missing_support','pred_anchor','ref_anchor',
    'nonuniform_time','mismatched_time','mask_dtype','ids_dtype','ids_shape','duplicate_ids','eligibility_dtype',
    'eligibility_shape','asymmetric','diagonal','no_pairs'])
def test_invalid_metadata_coordinates_time_and_support_fail(bad):
    args=explicit(example(batch=1))
    if bad=='one_member':args['predicted']=args['predicted'][:,:1]
    elif bad=='nonfinite_pred':args['predicted'][0,0,1,0,0]=float('nan')
    elif bad=='nonfinite_ref':args['reference'][0,1,0,0]=float('inf')
    elif bad=='missing_support':args['reference_mask'][0,-1,0]=False
    elif bad=='pred_anchor':args['predicted'][0,0,0,0,0]+=1
    elif bad=='ref_anchor':args['reference'][0,0,0,0]+=1
    elif bad=='nonuniform_time':args['predicted_times_ps'][0,2]+=1
    elif bad=='mismatched_time':args['reference_times_ps']+=1000
    elif bad=='mask_dtype':args['residue_mask']=args['residue_mask'].long()
    elif bad=='ids_dtype':args['geometry_row_ids']=args['geometry_row_ids'].float()
    elif bad=='ids_shape':args['geometry_row_ids']=args['geometry_row_ids'][0]
    elif bad=='duplicate_ids':args['geometry_row_ids'][0,1]=0
    elif bad=='eligibility_dtype':args['physical_pair_eligibility']=args['physical_pair_eligibility'].float()
    elif bad=='eligibility_shape':args['physical_pair_eligibility']=args['physical_pair_eligibility'][0]
    elif bad=='asymmetric':args['physical_pair_eligibility'][0,0,1]=True
    elif bad=='diagonal':args['physical_pair_eligibility'][0,0,0]=True
    else:args['physical_pair_eligibility'].zero_()
    with pytest.raises(ValueError):core.physical_distance_path_scores(**args)


def test_no_mutation_metadata_and_finite_zero_duplicate_gradients():
    args=explicit(example(batch=1))
    for name in ('predicted','reference','observed_anchor'):args[name].zero_()
    before=copy.deepcopy(args);args['predicted'].requires_grad_()
    result=core.physical_distance_path_scores(**args)
    grad=torch.autograd.grad(result['loss_absolute']+result['loss_increment'],args['predicted'])[0]
    assert result['loss_absolute']==result['loss_increment']==0 and torch.equal(grad,torch.zeros_like(grad))
    assert all(torch.equal(args[k],value) for k,value in before.items())
