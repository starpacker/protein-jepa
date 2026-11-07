"""Independent synthetic physical-distance path values, symmetries and gradients."""
import itertools

import numpy as np
import pytest
import torch

from protein_jepa_frame.training.physical_distance_path_objective_v1 import physical_distance_path_scores


def example(batch=2,members=3,horizon=3,length=6,dtype=torch.float64):
    rng=np.random.default_rng(771)
    anchor=torch.tensor(rng.normal(size=(batch,length,3)),dtype=dtype)
    predicted=torch.tensor(rng.normal(size=(batch,members,horizon+1,length,3)),dtype=dtype)
    reference=torch.tensor(rng.normal(size=(batch,horizon+1,length,3)),dtype=dtype)
    predicted[:,:,0]=anchor[:,None];reference[:,0]=anchor
    mask=torch.ones(batch,length,dtype=torch.bool)
    ri=torch.arange(length,dtype=torch.long)[None].expand(batch,-1).clone()
    ci=torch.zeros_like(ri)
    if batch>1:mask[1,-2:]=False;ci[1,2:]=1
    times=4000+torch.arange(horizon+1,dtype=dtype)[None].expand(batch,-1)*1000
    return dict(predicted=predicted,reference=reference,observed_anchor=anchor,residue_mask=mask,
        residue_index=ri,chain_id=ci,predicted_times_ps=times.clone(),reference_times_ps=times.clone(),
        reference_mask=mask[:,None].expand(batch,horizon+1,length).clone())


def numpy_scores(arguments):
    """Enumerate physical support and both pairwise U scores without Torch helpers."""
    arrays={key:value.detach().numpy() for key,value in arguments.items()}
    pred,ref=arrays['predicted'],arrays['reference']
    all_abs,all_inc,all_pairs=[],[],[]
    for b in range(len(pred)):
        valid=arrays['residue_mask'][b];ri=arrays['residue_index'][b];ci=arrays['chain_id'][b]
        pairs=[(i,j) for i in range(len(valid)) for j in range(i+1,len(valid))
               if valid[i] and valid[j] and (ci[i]!=ci[j] or abs(int(ri[i])-int(ri[j]))>=3)]
        dp=np.stack([np.linalg.norm(pred[b,:,:,i]-pred[b,:,:,j],axis=-1) for i,j in pairs],-1)
        dr=np.stack([np.linalg.norm(ref[b,:,i]-ref[b,:,j],axis=-1) for i,j in pairs],-1)
        def score(x,y):
            cross=sum(np.linalg.norm(member-y) for member in x)/len(x)
            spread=sum(np.linalg.norm(x[i]-x[j]) for i,j in itertools.combinations(range(len(x)),2))
            spread/=len(x)*(len(x)-1)/2
            return (cross-.5*spread)/np.sqrt(len(pairs))
        all_abs.append([score(dp[:,t],dr[t]) for t in range(1,ref.shape[1])])
        all_inc.append([score(dp[:,t]-dp[:,t-1],dr[t]-dr[t-1]) for t in range(1,ref.shape[1])])
        all_pairs.append(pairs)
    return np.array(all_abs),np.array(all_inc),all_pairs


def test_independent_numpy_values_all_pairs_equal_time_and_window_weighting():
    args=example();result=physical_distance_path_scores(**args)
    absolute,increment,pairs=numpy_scores(args)
    np.testing.assert_allclose(result['absolute_per_window_time'].detach(),absolute,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(result['increment_per_window_time'].detach(),increment,rtol=2e-14,atol=2e-14)
    assert result['loss_absolute']==pytest.approx(absolute.mean(),rel=2e-14)
    assert result['loss_increment']==pytest.approx(increment.mean(),rel=2e-14)
    assert result['pair_counts'].tolist()==[len(p) for p in pairs]
    assert [p.tolist() for p in result['pair_indices']]==[[list(pair) for pair in p] for p in pairs]
    assert torch.equal(result['residue_mask'],args['residue_mask'])
    assert torch.equal(result['time_ps'],args['predicted_times_ps'])
    # Different support sizes do not make a larger pair set a larger window weight.
    assert len(pairs[0])!=len(pairs[1])


def parameterized_path(args,horizon):
    generator=torch.Generator().manual_seed(889)
    b,m,_,l,_=args['predicted'].shape
    features=torch.randn(b,m,horizon,l,4,generator=generator,dtype=torch.float64)
    weight=(torch.randn(4,3,generator=generator,dtype=torch.float64)*.15).requires_grad_()
    feedback=(torch.randn(3,3,generator=generator,dtype=torch.float64)*.1).requires_grad_()
    bias=torch.randn(3,generator=generator,dtype=torch.float64).requires_grad_()
    current=args['observed_anchor'][:,None].expand(b,m,l,3)
    frames=[current]
    for t in range(horizon):
        current=current+.2*torch.tanh(features[:,:,t]@weight+current@feedback+bias)
        frames.append(current)
    return torch.stack(frames,2),(weight,feedback,bias),features


@pytest.mark.parametrize('seed',[0,1,2])
def test_h1_values_and_full_tiny_parameter_chain_gradients_equal(seed):
    args=example(horizon=1)
    args['observed_anchor']=args['observed_anchor']+(seed*.17)
    args['reference'][:,0]=args['observed_anchor']
    prediction,parameters,_=parameterized_path(args,1);args['predicted']=prediction
    result=physical_distance_path_scores(**args)
    torch.testing.assert_close(result['absolute_per_window_time'],result['increment_per_window_time'],rtol=2e-14,atol=2e-14)
    ga=torch.autograd.grad(result['loss_absolute'],parameters,retain_graph=True)
    gi=torch.autograd.grad(result['loss_increment'],parameters)
    for left,right in zip(ga,gi):
        torch.testing.assert_close(left,right,rtol=2e-12,atol=2e-14)
        assert torch.isfinite(left).all() and left.norm()>0


def test_member_permutation_and_gradients_equivariant():
    args=example();args['predicted'].requires_grad_()
    base=physical_distance_path_scores(**args)
    grad=torch.autograd.grad(base['loss_increment'],args['predicted'])[0]
    order=torch.tensor([2,0,1]);permuted={**args,'predicted':args['predicted'].detach()[:,order].requires_grad_()}
    moved=physical_distance_path_scores(**permuted)
    moved_grad=torch.autograd.grad(moved['loss_increment'],permuted['predicted'])[0]
    for key in ('loss_absolute','loss_increment'):torch.testing.assert_close(base[key],moved[key],rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(moved_grad,grad[:,order],rtol=2e-13,atol=2e-14)


def test_independent_per_frame_rigid_and_reflection_invariance_t0_retained():
    args=example();base=physical_distance_path_scores(**args)
    transformed={k:v.clone() for k,v in args.items()};rng=np.random.default_rng(66)
    for b in range(len(args['predicted'])):
        for member in range(args['predicted'].shape[1]):
            for t in range(1,args['predicted'].shape[2]):
                q,_=np.linalg.qr(rng.normal(size=(3,3)));q[:,0]*=-1
                transformed['predicted'][b,member,t]=args['predicted'][b,member,t]@torch.tensor(q)+torch.tensor(rng.normal(size=3))
        for t in range(1,args['reference'].shape[1]):
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            transformed['reference'][b,t]=args['reference'][b,t]@torch.tensor(q)+torch.tensor(rng.normal(size=3))
    result=physical_distance_path_scores(**transformed)
    for key in ('absolute_per_window_time','increment_per_window_time'):
        torch.testing.assert_close(base[key],result[key],rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('scale',[.1,7.3])
def test_coordinate_unit_scaling_no_target_scale(scale):
    args=example();base=physical_distance_path_scores(**args)
    transformed={**args,**{key:args[key]*scale for key in ('predicted','reference','observed_anchor')}}
    result=physical_distance_path_scores(**transformed)
    for key in ('loss_absolute','loss_increment'):
        torch.testing.assert_close(result[key],base[key]*scale,rtol=1e-13,atol=1e-13)


def two_residue_path(predicted_distances,reference_distances):
    pred=torch.zeros(1,len(predicted_distances),len(reference_distances),2,3,dtype=torch.float64)
    pred[0,:,:,1,0]=torch.tensor(predicted_distances,dtype=torch.float64)
    ref=torch.zeros(1,len(reference_distances),2,3,dtype=torch.float64)
    ref[0,:,1,0]=torch.tensor(reference_distances,dtype=torch.float64)
    times=torch.arange(len(reference_distances))[None].double()*1000
    return dict(predicted=pred,reference=ref,observed_anchor=ref[:,0].clone(),residue_mask=torch.ones(1,2,dtype=torch.bool),
        residue_index=torch.tensor([[0,0]]),chain_id=torch.tensor([[0,1]]),predicted_times_ps=times,
        reference_times_ps=times.clone(),reference_mask=torch.ones(1,len(reference_distances),2,dtype=torch.bool))


def test_signed_contraction_vs_expansion_not_magnitude_or_endpoint_only():
    args=two_residue_path([[2.,3.,2.],[2.,3.,2.]],[2.,1.,2.])
    result=physical_distance_path_scores(**args)
    assert result['absolute_per_window_time'].tolist()==[[2.,0.]]
    assert result['increment_per_window_time'].tolist()==[[2.,2.]]
    assert result['loss_absolute']==1 and result['loss_increment']==2


def test_joint_vector_norm_not_average_of_pairwise_scalar_scores():
    args=example(batch=1,members=2,horizon=2)
    result=physical_distance_path_scores(**args)
    absolute,increment,_=numpy_scores(args)
    assert float(result['loss_absolute'])==pytest.approx(absolute.mean())
    # L2 score depends jointly on all pair components; it is not an L1 average.
    pairs=result['pair_indices'][0];p=args['predicted'][0,:,1];r=args['reference'][0,1]
    x=(p[:,pairs[:,0]]-p[:,pairs[:,1]]).norm(dim=-1);y=(r[pairs[:,0]]-r[pairs[:,1]]).norm(dim=-1)
    separate=(x-y).abs().mean(0)-.5*(x[0]-x[1]).abs()
    assert not torch.isclose(result['absolute_per_window_time'][0,0],separate.mean(),rtol=1e-4,atol=1e-4)


@pytest.mark.parametrize('duplicate',['all_zero','duplicate_members','reference_match'])
def test_coincident_vectors_and_duplicate_members_have_finite_gradients(duplicate):
    args=example(batch=1)
    if duplicate=='all_zero':
        for key in ('predicted','reference','observed_anchor'):args[key].zero_()
    elif duplicate=='duplicate_members':args['predicted'][:,1:]=args['predicted'][:,:1]
    else:args['predicted']=args['reference'][:,None].expand_as(args['predicted']).clone()
    args['predicted'].requires_grad_();result=physical_distance_path_scores(**args)
    gradient=torch.autograd.grad(result['loss_absolute']+result['loss_increment'],args['predicted'])[0]
    assert torch.isfinite(gradient).all()
    if duplicate in ('all_zero','reference_match'):
        assert result['loss_absolute']==result['loss_increment']==0 and torch.equal(gradient,torch.zeros_like(gradient))


def test_increment_backpropagates_into_both_adjacent_generated_frames():
    args=example(batch=1,horizon=3);args['predicted'].requires_grad_()
    result=physical_distance_path_scores(**args)
    gradient=torch.autograd.grad(result['increment_per_window_time'][0,1],args['predicted'])[0]
    assert gradient[:,:,1].norm()>0 and gradient[:,:,2].norm()>0
    assert torch.equal(gradient[:,:,0],torch.zeros_like(gradient[:,:,0]))
    assert torch.equal(gradient[:,:,3],torch.zeros_like(gradient[:,:,3]))


def test_horizon3_predictor_gradient_matches_independent_numpy_finite_difference():
    args=example(batch=1,horizon=3);prediction,parameters,features=parameterized_path(args,3)
    args['predicted']=prediction;result=physical_distance_path_scores(**args)
    gradient=torch.autograd.grad(result['loss_increment'],parameters)[0]
    def score_at(delta):
        w,f,b=[p.detach().clone() for p in parameters];w[0,0]+=delta
        current=args['observed_anchor'][:,None].expand(1,3,6,3);frames=[current]
        for t in range(3):
            current=current+.2*torch.tanh(features[:,:,t]@w+current@f+b);frames.append(current)
        return numpy_scores({**args,'predicted':torch.stack(frames,2)})[1].mean()
    estimate=(score_at(1e-6)-score_at(-1e-6))/(2e-6)
    assert float(gradient[0,0])==pytest.approx(estimate,rel=2e-5,abs=2e-8)
    assert abs(float(gradient[0,0]))>1e-8


def test_padding_nans_extra_reference_validity_and_physical_gap_chain_support():
    args=example(batch=1,length=6)
    args['residue_mask'][0,-1]=False
    args['reference_mask'][:,:,-1]=True  # Outside observed support remains irrelevant.
    args['residue_index']=torch.tensor([[0,1,4,5,5,0]])
    args['chain_id']=torch.tensor([[0,0,0,0,1,0]])
    for key in ('predicted','reference','observed_anchor'):args[key][...,-1,:]=float('nan')
    args['predicted'].requires_grad_()
    result=physical_distance_path_scores(**args);_,_,pairs=numpy_scores(args)
    assert result['pair_indices'][0].tolist()==[list(p) for p in pairs[0]]
    assert [0,2] in result['pair_indices'][0].tolist() and [3,4] in result['pair_indices'][0].tolist()
    assert [0,1] not in result['pair_indices'][0].tolist()
    gradient=torch.autograd.grad(result['loss_increment'],args['predicted'])[0]
    assert torch.isfinite(gradient).all() and torch.equal(gradient[...,-1,:],torch.zeros_like(gradient[...,-1,:]))


def test_uint8_physical_ids_are_promoted_before_subtraction():
    args=example(batch=1,length=4)
    args['residue_index']=torch.tensor([[0,1,254,255]],dtype=torch.uint8)
    result=physical_distance_path_scores(**args)
    assert result['pair_indices'][0].tolist()==[[0,2],[0,3],[1,2],[1,3]]


@pytest.mark.parametrize('bad',['one_member','no_future','nonfinite_pred','nonfinite_ref','nonfinite_anchor',
    'missing_support','duplicate_identity','no_pairs','pred_anchor','ref_anchor','nonuniform_time',
    'mismatched_times','nonfinite_times','mask_dtype','reference_mask_shape','ri_dtype','reference_dtype'])
def test_invalid_inputs_fail(bad):
    args=example(batch=1)
    if bad=='one_member':args['predicted']=args['predicted'][:,:1]
    elif bad=='no_future':args['predicted']=args['predicted'][:,:,:1]
    elif bad=='nonfinite_pred':args['predicted'][0,0,1,0,0]=float('nan')
    elif bad=='nonfinite_ref':args['reference'][0,1,0,0]=float('inf')
    elif bad=='nonfinite_anchor':args['observed_anchor'][0,0,0]=float('nan')
    elif bad=='missing_support':args['reference_mask'][0,-1,0]=False
    elif bad=='duplicate_identity':args['residue_index'][0,1]=0
    elif bad=='no_pairs':args['residue_mask'][:]=False;args['residue_mask'][0,:2]=True
    elif bad=='pred_anchor':args['predicted'][0,0,0,0,0]+=1.
    elif bad=='ref_anchor':args['reference'][0,0,0,0]+=1.
    elif bad=='nonuniform_time':args['predicted_times_ps'][0,2]+=1.
    elif bad=='mismatched_times':args['reference_times_ps']+=1000.
    elif bad=='nonfinite_times':args['reference_times_ps'][0,1]=float('nan')
    elif bad=='mask_dtype':args['residue_mask']=args['residue_mask'].long()
    elif bad=='reference_mask_shape':args['reference_mask']=args['reference_mask'][:,0]
    elif bad=='ri_dtype':args['residue_index']=args['residue_index'].float()
    elif bad=='reference_dtype':args['reference']=args['reference'].float()
    with pytest.raises(ValueError):physical_distance_path_scores(**args)


def test_float32_path_gradients_and_no_mutation_of_inputs():
    args=example(dtype=torch.float32);before={k:v.clone() for k,v in args.items()}
    args['predicted'].requires_grad_();result=physical_distance_path_scores(**args)
    grad=torch.autograd.grad(result['loss_increment'],args['predicted'])[0]
    assert result['loss_increment'].dtype==torch.float32 and torch.isfinite(grad).all()
    assert all(torch.equal(args[key],value) for key,value in before.items())
