"""Synthetic dense NumPy references; no actual models, trajectories or sampling."""
import copy
import numpy as np
import pytest
import torch
from protein_jepa_frame.models.stage_b1b.h_moment_budget_v1 import one_step_h_moments


def case(seed=3,L=3,d=5,s=2):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(L,d))+1.7;f=rng.normal(size=(L,d))*.03
    D=np.exp(rng.normal(size=(L,d)))*.02;C=rng.normal(size=(L,d,s))*.1
    k=rng.normal(size=(L,L));K=k@k.T+.3*np.eye(L)
    p=rng.normal(size=(L*s,L*s));P=p@p.T+.2*np.eye(L*s)
    m=rng.normal(size=(L,s));S=rng.normal(size=(L,s))
    return {key:torch.from_numpy(value) for key,value in dict(H=x,force=f,diffusion=D,coupling=C,
        spatial_covariance=K,bath_mean=m,bath_covariance=P,realized_bath=S).items()}


def dense_reference(a,h,mode):
    x,f,D,C,K,m,P,S=[a[k].numpy() for k in ('H','force','diffusion','coupling','spatial_covariance','bath_mean','bath_covariance','realized_bath')]
    L,d=x.shape;s=m.shape[-1];root=np.diag(np.sqrt(D.ravel()));M=root@np.kron(K,np.eye(d))@root
    cb=np.zeros((L*d,L*s))
    for i in range(L):cb[i*d:(i+1)*d,i*s:(i+1)*s]=C[i]
    mean=m if mode=='posterior' else S
    b=(h*M@(f.ravel()-(cb@mean.ravel() if mode!='C_zero' else 0))).reshape(L,d)
    therm=2*h*M;bath=h*h*M@cb@P@cb.T@M if mode=='posterior' else np.zeros_like(M)
    Q=np.eye(d)-np.ones((d,d))/d;values=[]
    for i in range(L):
        t=therm[i*d:(i+1)*d,i*d:(i+1)*d];p=bath[i*d:(i+1)*d,i*d:(i+1)*d];sig=t+p
        z=Q@x[i];u=z/np.linalg.norm(z);T=Q-np.outer(u,u);v=b[i];row={}
        for name,projection in [('raw',np.eye(d)),('centered',Q)]:
            xx=projection@x[i];bb=projection@v
            row.update({name+'_current_norm2':xx@xx,name+'_cross':2*xx@bb,name+'_drift2':bb@bb,
                name+'_thermal_trace':np.trace(projection@t@projection),name+'_bath_trace':np.trace(projection@p@projection),
                name+'_expected_change':2*xx@bb+bb@bb+np.trace(projection@sig@projection),
                name+'_expected_next_norm2':(xx+bb)@(xx+bb)+np.trace(projection@sig@projection)})
        mh=x[i].mean();mb=v.mean();mv=sig.sum()/d**2
        row.update(channel_mean_current=mh,channel_mean_increment=mb,channel_mean_noise_variance=mv,
            channel_expected_change=d*(2*mh*mb+mb*mb+mv),channel_increment_second_moment=d*(mb*mb+mv),
            raw_centered_channel_closure=0.,tangent_drift2=(T@v)@(T@v),tangent_thermal_trace=np.trace(T@t@T),
            tangent_bath_trace=np.trace(T@p@T),radial_mean_increment=u@v,radial_drift2=(u@v)**2,
            radial_thermal_variance=u@t@u,radial_bath_variance=u@p@u,
            radial_increment_second_moment=(u@v)**2+u@sig@u,
            tangent_increment_second_moment=(T@v)@(T@v)+np.trace(T@sig@T),
            increment_decomposition_closure=0.,centered_radial_closure=0.)
        values.append(row)
    return b,values


@pytest.mark.parametrize('seed,L,d,s,h',[(1,3,5,2,2.),(2,2,7,3,.5),(3,4,4,2,1000.),(4,2,128,8,1.),(5,1,3,1,3.)])
def test_full_correlated_dense_gaussian_reference(seed,L,d,s,h):
    a=case(seed,L,d,s);out=one_step_h_moments(**a,dt=h)
    for mode in ('posterior','realized_S','C_zero'):
        b,expected=dense_reference(a,h,mode);actual=out[mode]
        np.testing.assert_allclose(actual['mean_increment'],b,atol=2e-10,rtol=1e-12)
        np.testing.assert_allclose(actual['next_H_mean'],a['H'].numpy()+b,atol=2e-10,rtol=1e-12)
        assert set(actual['per_residue'])==set(expected[0])
        for key in expected[0]:
            wanted=np.asarray([r[key] for r in expected]);got=actual['per_residue'][key].numpy()
            np.testing.assert_allclose(got,wanted,atol=2e-8,rtol=2e-12,err_msg=key)
            assert actual['summary'][key]==pytest.approx(wanted.mean(),abs=2e-8,rel=2e-12)
    assert out['audit']['largest_response_shape']==[L,d,L*s]
    assert out['audit']['full_H_covariance_materialized'] is False


def test_cross_residue_covariance_cannot_be_replaced_by_blocks():
    a=case();full=one_step_h_moments(**a,dt=7.)
    diagonal=copy.deepcopy(a);P=diagonal['bath_covariance'];s=diagonal['bath_mean'].shape[-1]
    for i in range(3):
        for j in range(3):
            if i!=j:P[i*s:(i+1)*s,j*s:(j+1)*s]=0
    block=one_step_h_moments(**diagonal,dt=7.)
    assert not torch.allclose(full['posterior']['per_residue']['raw_bath_trace'],block['posterior']['per_residue']['raw_bath_trace'])
    assert torch.equal(full['posterior']['mean_increment'],block['posterior']['mean_increment'])


def test_conditional_realized_S_has_only_thermal_noise_and_C_zero_is_exact():
    a=case();a['realized_bath']=a['bath_mean'].clone();out=one_step_h_moments(**a,dt=1000.)
    assert torch.equal(out['posterior']['mean_increment'],out['realized_S']['mean_increment'])
    assert torch.equal(out['realized_S']['per_residue']['raw_bath_trace'],torch.zeros(3,dtype=torch.float64))
    assert bool((out['posterior']['per_residue']['raw_bath_trace']>0).all())
    a['coupling'].zero_();zero=one_step_h_moments(**a,dt=1000.)
    for mode in ('posterior','realized_S'):
        assert torch.equal(zero[mode]['mean_increment'],out['C_zero']['mean_increment'])
        for key,value in zero[mode]['per_residue'].items():torch.testing.assert_close(value,out['C_zero']['per_residue'][key])


def test_masks_pack_principal_full_K_and_P_ignore_only_masked_values():
    a=case(L=4);mask=torch.tensor([True,False,True,False]);valid=torch.tensor([0,2]);bath=torch.tensor([0,1,4,5])
    packed={k:v[valid] for k,v in a.items() if k not in ('spatial_covariance','bath_covariance')}
    packed['spatial_covariance']=a['spatial_covariance'][valid][:,valid]
    packed['bath_covariance']=a['bath_covariance'][bath][:,bath]
    wanted=one_step_h_moments(**packed,dt=3.)
    for k,v in a.items():
        if k=='spatial_covariance':v[~mask,:]=float('nan');v[:,~mask]=float('nan')
        elif k=='bath_covariance':bm=mask.repeat_interleave(2);v[~bm,:]=float('nan');v[:,~bm]=float('nan')
        else:v[~mask]=float('nan')
    got=one_step_h_moments(**a,dt=3.,residue_mask=mask)
    assert torch.equal(got['valid_indices'],valid)
    for mode in ('posterior','realized_S','C_zero'):
        for key in wanted[mode]['per_residue']:torch.testing.assert_close(got[mode]['per_residue'][key],wanted[mode]['per_residue'][key])


def test_permutation_preserves_full_bath_residue_order():
    a=case();order=torch.tensor([2,0,1]);bath=(order[:,None]*2+torch.arange(2)).flatten()
    p={k:v[order] for k,v in a.items() if k not in ('spatial_covariance','bath_covariance')}
    p.update(spatial_covariance=a['spatial_covariance'][order][:,order],bath_covariance=a['bath_covariance'][bath][:,bath])
    x=one_step_h_moments(**a,dt=2.);y=one_step_h_moments(**p,dt=2.)
    for mode in ('posterior','realized_S','C_zero'):
        for key in x[mode]['per_residue']:torch.testing.assert_close(y[mode]['per_residue'][key],x[mode]['per_residue'][key][order])


@pytest.mark.parametrize('all_constant',[False,True])
def test_degenerate_centered_H_keeps_raw_centered_support_and_null_tangent(all_constant):
    a=case();a['H'][0]=2.
    if all_constant:a['H'][:]=2.
    out=one_step_h_moments(**a,dt=1.)
    assert not out['tangent_valid'][0]
    for mode in ('posterior','realized_S','C_zero'):
        p=out[mode]['per_residue'];summary=out[mode]['summary']
        assert torch.isfinite(p['raw_expected_change']).all() and torch.isfinite(p['centered_expected_change']).all()
        assert torch.isnan(p['tangent_increment_second_moment'][0]) and summary['n_residues']==3
        if all_constant:assert summary['tangent_increment_second_moment'] is None and summary['n_tangent_valid']==0
        else:assert summary['tangent_increment_second_moment']==float(p['tangent_increment_second_moment'][1:].mean())


def test_nonzero_channel_mean_and_anisotropic_diffusion_are_not_discarded():
    a=case();out=one_step_h_moments(**a,dt=2.)['posterior']['per_residue']
    assert bool((out['channel_mean_current'].abs()>.1).all())
    assert bool((out['channel_increment_second_moment']>0).all())
    assert not torch.allclose(out['raw_expected_change'],out['centered_expected_change'])
    torch.testing.assert_close(out['raw_centered_channel_closure'],torch.zeros(3,dtype=torch.float64),atol=1e-12,rtol=0)


@pytest.mark.parametrize('bad',['dtype','shape','nonfinite_H','nonfinite_P','Dzero','Dnegative','K_asymmetric','P_asymmetric','K_nonPSD','P_nonPSD','dtzero','dtinf','empty_mask','mask_dtype','epsilon'])
def test_invalid_inputs_fail_closed(bad):
    a=case();kwargs=dict(dt=1.)
    if bad=='dtype':a['force']=a['force'].float()
    elif bad=='shape':a['coupling']=a['coupling'][:1]
    elif bad=='nonfinite_H':a['H'][0,0]=float('nan')
    elif bad=='nonfinite_P':a['bath_covariance'][0,0]=float('inf')
    elif bad=='Dzero':a['diffusion'][0,0]=0
    elif bad=='Dnegative':a['diffusion'][0,0]=-1
    elif bad.endswith('asymmetric'):a['spatial_covariance' if bad[0]=='K' else 'bath_covariance'][0,1]+=1
    elif bad.endswith('nonPSD'):a['spatial_covariance' if bad[0]=='K' else 'bath_covariance'][0,0]=-100
    elif bad=='dtzero':kwargs['dt']=0
    elif bad=='dtinf':kwargs['dt']=float('inf')
    elif bad=='empty_mask':kwargs['residue_mask']=torch.zeros(3,dtype=torch.bool)
    elif bad=='mask_dtype':kwargs['residue_mask']=torch.ones(3)
    else:kwargs['centered_norm_epsilon']=0
    with pytest.raises(ValueError):one_step_h_moments(**a,**kwargs)


def test_semidefinite_zero_covariance_allowed_no_mutation_rng_or_grad():
    a=case();a['bath_covariance'].zero_();a['H'].requires_grad_();before={k:v.clone() for k,v in a.items()};rng=torch.get_rng_state().clone()
    out=one_step_h_moments(**a,dt=1.)
    assert torch.equal(torch.get_rng_state(),rng)
    for key in a:assert torch.equal(a[key],before[key])
    assert not out['posterior']['mean_increment'].requires_grad
    assert out['posterior']['summary']['raw_bath_trace']==0.
