"""Synthetic-only tests for the independent mean-control audit."""
import copy

import numpy as np
import pytest
import torch

from scripts import verify_hs_history_mean_v1 as audit


def test_numpy_whitening_uses_valid_block_and_exact_double_subtraction():
    mean=torch.tensor([[[2.],[float('nan')],[4.]]])
    reference=torch.tensor([[[0.],[float('nan')],[0.]]])
    factor=torch.tensor([[[2.,0.,0.],[0.,0.,0.],[1.,0.,3.]]])
    result=audit.whitened_difference(mean,reference,factor,np.array([[True,False,True]]))
    assert result['n_valid_bath_dimensions']==2
    assert result['mahalanobis_squared']==pytest.approx(2.)
    assert result['raw_l2']==pytest.approx(np.sqrt(20.))
    assert result['standardized_rms']==pytest.approx(1.)
    mean=torch.tensor([[[2.**24]]]);reference=torch.tensor([[[-1.]]])
    result=audit.whitened_difference(mean,reference,torch.ones(1,1,1),np.ones((1,1),bool))
    assert result['raw_l2']==2.**24+1


def test_invalid_whitening_factor_is_rejected():
    x=torch.ones(1,2,1);mask=np.ones((1,2),bool)
    for factor in (torch.tensor([[[1.,.1],[0.,1.]]]),torch.zeros(1,2,2)):
        with pytest.raises(AssertionError,match='lower triangular'):
            audit.whitened_difference(x,torch.zeros_like(x),factor,mask)


def initial_fixture():
    g=torch.Generator().manual_seed(722);m,l,d,s=12,3,4,2
    mask=torch.tensor([[True,False,True]]).expand(m,-1).clone()
    anchor_h=torch.randn(1,l,d,generator=g)*mask[:1,:,None]
    h0=anchor_h.expand(m,-1,-1).clone()
    chol=torch.zeros(1,l*s,l*s);indices=torch.tensor([0,1,4,5])
    packed=torch.eye(4)*.5;packed[2,0]=.2
    chol[0,indices[:,None],indices[None,:]]=packed
    mean=torch.randn(1,l,s,generator=g)*mask[:1,:,None]
    eps=torch.randn(1,m,l*s,generator=g)
    centered=torch.einsum('bij,bmj->bmi',chol,eps).reshape(1,m,l,s)
    means={'real_history':mean,'repeated_anchor_history':mean+.1*mask[:1,:,None],
           'zero_mean':torch.zeros_like(mean)}
    states={name:(value[:,None]+centered)[0] for name,value in means.items()}
    metadata={'step':1,'protein':'p','trajectory':'p-r0','temperature_k':320.,'anchor_frame':7,'horizon':1}
    state=dict(H0=h0,S0=states['real_history'],posterior_cholesky=chol,posterior_standard_normal=eps,
        posterior_mean=mean,mask=mask,temp=torch.full((m,),320.),D=torch.ones_like(h0),
        C=torch.ones(m,l,d,s),Gamma=torch.ones(m,l,s),factor=torch.eye(l)[None].expand(m,-1,-1).clone(),
        anchor_H=anchor_h,rotation=torch.eye(3)[None,None].expand(1,l,-1,-1).clone(),
        anchor=torch.randn(1,l,3,generator=g),metadata=[torch.arange(l)[None]],
        history=anchor_h[:,None].expand(-1,4,-1,-1).clone(),dt_history=torch.full((1,4),1000.),
        fine_H=torch.randn(m,1,8,l,d,generator=g),fine_S=torch.randn(m,1,8,l,s,generator=g))
    original=dict(metadata=metadata,state=state,observed={'residue_mask':mask[:1].clone()},
        target={'coords':torch.randn(1,l,4,3,generator=g)},branches={})
    control=dict(metadata=metadata,means=means,repeated_chol_bitwise_equal=True,
        repeated_history_sha256=audit.state_sha256({'H':state['history']}),
        repeated_chol_sha256=audit.state_sha256({'chol':chol}))
    differences={name:audit.whitened_difference(value,mean,chol,mask[:1].numpy()) for name,value in means.items()}
    differences['real_minus_zero']=audit.whitened_difference(mean,means['zero_mean'],chol,mask[:1].numpy())
    control['mean_differences']=differences
    arms={name:dict(S0=s0,H=torch.stack((h0,h0+.1),1),S=torch.stack((s0,s0+.2),1),
                   CA=torch.randn(m,2,l,3,generator=g)) for name,s0 in states.items()}
    original['branches']['1']={key:arms['real_history'][key].clone() for key in ('H','S','CA')}
    names=('mask','temp','D','C','Gamma','factor','anchor_H','rotation','anchor','metadata','history','dt_history')
    raw=dict(metadata=metadata,controls=copy.deepcopy(control),H0=h0.clone(),original_full_chol=chol.clone(),
        posterior_epsilon=eps.clone(),observation=copy.deepcopy(original['observed']),target=copy.deepcopy(original['target']),
        condition={name:copy.deepcopy(state[name]) for name in names},shared_centered_bath=centered.clone(),arms=arms,
        brownian_H=state['fine_H'].sum(2,keepdim=True)/np.sqrt(8),
        brownian_S=state['fine_S'].sum(2,keepdim=True)/np.sqrt(8))
    return raw,original,control


def test_all_shared_inputs_and_full_real_replay_checked_without_producer_helpers():
    raw,original,control=initial_fixture();tracker=audit.independent.Audit()
    values=audit.check_initial_inputs(raw,original,control,tracker)
    assert set(values)==set(audit.ARMS)|{'real_minus_zero'}
    assert tracker.counts['real_path_bitwise_elements']>0


@pytest.mark.parametrize('where',['covariance','epsilon','brownian','mean','initial_H','initial_S','real_H','real_S','real_CA'])
def test_one_changed_source_or_path_component_is_rejected(where):
    raw,original,control=initial_fixture()
    if where=='covariance':target=raw['original_full_chol']
    elif where=='epsilon':target=raw['posterior_epsilon']
    elif where=='brownian':target=raw['brownian_H']
    elif where=='mean':target=raw['controls']['means']['repeated_anchor_history']
    elif where=='initial_H':target=raw['arms']['zero_mean']['H'][:,0]
    elif where=='initial_S':target=raw['arms']['zero_mean']['S'][:,0]
    else:target=raw['arms']['real_history'][where.split('_')[1]][:,1]
    index=(0,)*target.ndim
    target[index]=torch.nextafter(target[index],torch.tensor(float('inf')))
    with pytest.raises(AssertionError):audit.check_initial_inputs(raw,original,control,audit.independent.Audit())


def test_failed_member_evidence_exactly_preserves_nan_bytes():
    x=torch.tensor([float('nan'),1.]);audit.exact(x,x.clone(),'failure evidence')
    audit.exact(x.numpy(),x.numpy().copy(),'array evidence')
    with pytest.raises(AssertionError):audit.exact(x,torch.tensor([0.,1.]),'failure evidence')
    with pytest.raises(AssertionError):audit.exact(x.numpy(),x.numpy().astype(np.float64),'array dtype')


def test_independent_aggregation_preserves_missing_panel_and_protein_weight():
    rows=[dict(metadata={'protein':'a'},scalars={'delta':1.}),
          dict(metadata={'protein':'a'},scalars={'delta':3.}),
          dict(metadata={'protein':'b'},scalars={'delta':5.})]
    complete=audit.independent.aggregate(rows)['delta']
    assert complete['equal_protein_mean']==3.5
    rows[1]['scalars']['delta']=None
    missing=audit.independent.aggregate(rows)['delta']
    assert missing['equal_protein_mean'] is None and missing['available_subset_mean']==3.
    assert missing['n_windows']==2 and missing['expected_windows']==3


def test_exclusive_evidence_outputs(tmp_path):
    path=tmp_path/'evidence.json';audit.write(path,{'first':True})
    with pytest.raises(FileExistsError):audit.write(path,{'second':True})
