"""CPU-only contracts; no checkpoint, manifest, or molecular data reads."""
from types import SimpleNamespace

import pytest
import torch

from protein_jepa_frame.models.stage_b1b.model import StageB1b
from run_scripts import diagnose_hs_microsteps_v1 as diag


class LinearKernel:
    sample_step = StageB1b.sample_step
    rollout = StageB1b.rollout
    rollout_microstep_v1 = StageB1b.rollout_microstep_v1
    _collective_velocity_step_fixed = StageB1b._collective_velocity_step_fixed

    def __init__(self, coupling=.1, constant=False):
        self.cfg=SimpleNamespace(use_bath=True,bath_coupling='velocity',bath_dim=2,step_ps=.2)
        self.s0_mode='exact_spatial'
        self.coupling=coupling
        self.constant=constant
        self.posterior_calls=0

    def bath(self,anchor,temperature):
        d=torch.arange(1,anchor.shape[-1]+1,dtype=anchor.dtype)[None,None]*.2
        return (d.expand_as(anchor),torch.full((*anchor.shape,2),self.coupling,dtype=anchor.dtype),
                torch.full((*anchor.shape[:2],2),.4,dtype=anchor.dtype))

    def compute_forces(self,h,mask,temperature,create_graph=False):
        return (torch.ones_like(h)*.3 if self.constant else -.3*h)*mask[...,None]

    def initial_bath_state(self,history,dt,temp,mask,seq_feature,noise_factor,create_graph):
        self.posterior_calls+=1
        assert history.shape[1]==4 and dt.shape==(1,4)
        assert torch.equal(history[:,-1],seq_feature)
        assert not create_graph
        mean=torch.ones(1,history.shape[2],2,dtype=history.dtype)*.2
        chol=torch.eye(mean.numel(),dtype=history.dtype)[None]*.7
        chol[:,1,0]=.15
        return mean,chol


def state(kernel,members=3,horizon=3,dtype=torch.float64):
    gen=torch.Generator().manual_seed(77)
    h=torch.randn(members,4,4,dtype=dtype,generator=gen)
    s=torch.randn(members,4,2,dtype=dtype,generator=gen)
    mask=torch.tensor([[True,True,True,False]]).expand(members,-1).clone()
    h=h*mask[...,None];s=s*mask[...,None]
    temp=torch.ones(members,dtype=dtype)*320
    fac=torch.tensor([[1.,0.,0.,0.],[.2,.9,0.,0.],[.1,.2,1.1,0.],[0.,0.,0.,1.]],dtype=dtype)[None].expand(members,-1,-1)
    d,c,gamma=kernel.bath(h,temp)
    return dict(H0=h,S0=s,mask=mask,temp=temp,factor=fac,D=d,C=c,Gamma=gamma,
        fine_H=torch.randn(members,horizon,8,4,4,dtype=dtype,generator=gen),
        fine_S=torch.randn(members,horizon,8,4,2,dtype=dtype,generator=gen))


@pytest.mark.parametrize('channels',[2,4])
def test_brownian_coarsening_preserves_macro_physical_increment(channels):
    fine=torch.randn(3,5,8,4,channels,dtype=torch.float64,generator=torch.Generator().manual_seed(41))
    expected=(.2/8)**.5*fine.sum(2)
    for count in (1,4,8):
        actual=(.2/count)**.5*diag.coarsen(fine,count).sum(2)
        torch.testing.assert_close(actual,expected,rtol=2e-14,atol=2e-14)
    with pytest.raises(ValueError):diag.coarsen(fine,3)


def test_dense_covariance_trace_and_exact_second_moment():
    k=LinearKernel();s=state(k,members=1,horizon=1)
    h,bath,mask,d,c,fac=[s[n] for n in ('H0','S0','mask','D','C','factor')]
    f=k.compute_forces(h,mask,None);dt=.2
    # Explicit dense matrix independently verifies channel/residue flattening.
    root=torch.diag(d.flatten().sqrt())
    m=root@torch.kron((fac[0]@fac[0].T).contiguous(),torch.eye(4,dtype=h.dtype))@root
    valid=torch.diag(mask.repeat_interleave(4,dim=1).flatten().double())
    q=torch.kron(torch.diag(mask[0].double()),torch.eye(4,dtype=h.dtype)-torch.ones(4,4,dtype=h.dtype)/4)
    zero=torch.zeros_like(h)
    budget=diag.norm_budget(h,h,f,bath,d,c,fac,dt,zero,mask)
    assert float(budget['raw']['noise_trace'])==pytest.approx(2*dt*torch.trace(valid@m@valid).item(),abs=1e-12)
    assert float(budget['centered']['noise_trace'])==pytest.approx(2*dt*torch.trace(q@m@q).item(),abs=1e-12)
    drift=diag.mobility(f-torch.einsum('blij,blj->bli',c,bath),d,fac)
    # +/-sqrt(n)e_i have exactly Gaussian first/second moments, which suffice.
    n=h.numel();points=torch.cat((torch.eye(n),-torch.eye(n))).double()*n**.5
    increments=[]
    for z in points.reshape(2*n,*h.shape[1:]):
        eta=torch.sqrt(2*dt*d)*(fac@z[None])
        after=(h+dt*drift+eta)*mask[...,None]
        terms=diag.norm_budget(h,after,f,bath,d,c,fac,dt,z[None],mask)
        increments.append(terms)
        for space in ('raw','centered'):
            assert terms[space]['rounding_closure'].abs().max()<1e-12
    for space in ('raw','centered'):
        actual=torch.stack([x[space]['actual'] for x in increments]).mean(0)
        torch.testing.assert_close(actual,budget[space]['expected'],rtol=1e-13,atol=1e-13)
    assert increments[0]['raw']['stochastic_residual'].abs().max()>1e-3


def test_monte_carlo_conditional_norm_expectation():
    k=LinearKernel();s=state(k,members=1,horizon=1)
    h,bath,mask,d,c,fac=[s[n] for n in ('H0','S0','mask','D','C','factor')]
    n=12000;dt=.2
    repeat=lambda x:x.expand(n,*x.shape[1:])
    h,bath,mask,d,c,fac=[repeat(x) for x in (h,bath,mask,d,c,fac)]
    z=torch.randn(h.shape,dtype=h.dtype,generator=torch.Generator().manual_seed(23))
    f=k.compute_forces(h,mask,None)
    b=diag.mobility(f-torch.einsum('blij,blj->bli',c,bath),d,fac)
    after=(h+dt*b+torch.sqrt(2*dt*d)*(fac@z))*mask[...,None]
    terms=diag.norm_budget(h,after,f,bath,d,c,fac,dt,z,mask)
    for space in ('raw','centered'):
        residual=terms[space]['stochastic_residual']
        assert abs(float(residual.mean()))<5*float(residual.std()/n**.5)


def test_mean_direction_noise_has_zero_centered_effect():
    h=torch.zeros(1,2,4,dtype=torch.float64);d=torch.ones_like(h)
    c=torch.zeros(1,2,4,2,dtype=h.dtype);s=torch.zeros(1,2,2,dtype=h.dtype)
    fac=torch.eye(2,dtype=h.dtype)[None];mask=torch.ones(1,2,dtype=torch.bool)
    z=torch.ones_like(h);dt=.2;after=(2*dt)**.5*z
    terms=diag.norm_budget(h,after,h,s,d,c,fac,dt,z,mask)
    assert terms['raw']['actual'].item()>0
    assert terms['centered']['actual'].item()==0
    # Conditional centered noise trace is positive even though this particular
    # realization points wholly in the mean direction; residual is not zero.
    assert terms['centered']['stochastic_residual'].item()<0
    assert terms['centered']['rounding_closure'].item()==pytest.approx(0.,abs=1e-12)


def test_m1_equals_actual_production_rollout_with_identical_explicit_noise(monkeypatch):
    k=LinearKernel();s=state(k,dtype=torch.float32)
    zh,zs=[diag.coarsen(s[key],1) for key in ('fine_H','fine_S')]
    queue=[x for i in range(3) for x in (zh[:,i,0],zs[:,i,0])]
    def supplied(shape,**kwargs):
        result=queue.pop(0)
        assert tuple(shape)==tuple(result.shape)
        return result.clone()
    monkeypatch.setattr(torch,'randn',supplied)
    with torch.no_grad():
        old=k.rollout(s['H0'],s['mask'],s['temp'],3,S0=s['S0'],seq_feature=s['H0'],
            detach=True,return_bath=True,noise_factor=s['factor'],collective_mobility=True)
    assert not queue
    branch,new=diag.run_branch(SimpleNamespace(kernel=k),s,3,1)
    assert branch['status']=='completed'
    assert torch.equal(old[0],new['H']) and torch.equal(old[1],new['S'])
    assert k.posterior_calls==0


def test_additive_constant_force_has_same_endpoint_and_fixed_trace_on_all_grids():
    k=LinearKernel(coupling=0.,constant=True);s=state(k)
    output=[diag.run_branch(SimpleNamespace(kernel=k),s,3,count) for count in (1,4,8)]
    for (branch,raw) in output:
        assert branch['status']=='completed'
        for space in ('raw','centered'):
            torch.testing.assert_close(raw['budgets'][space]['noise_trace'].sum(0),
                output[0][1]['budgets'][space]['noise_trace'].sum(0),rtol=1e-13,atol=1e-13)
    for _,raw in output[1:]:
        torch.testing.assert_close(raw['H'][:,-1],output[0][1]['H'][:,-1],rtol=2e-14,atol=2e-14)


def test_initial_state_computes_one_full_posterior_and_all_branches_share_it():
    class Bundle:
        def encode(self,coords,mask,*metadata):
            ca=coords[:,:,1]
            return torch.cat((ca,ca.mean(-1,keepdim=True)),-1)*mask[...,None]
    class Model:
        bundle=Bundle()
        kernel=LinearKernel()
        anchor_calls=0
        def anchor_kernel(self,ca,mask):
            self.anchor_calls+=1
            factor=torch.eye(ca.shape[1],dtype=ca.dtype)[None]
            factor[:,1,0]=.23
            return factor
    model=Model();length=4
    context=dict(coords=torch.randn(4,length,4,3,dtype=torch.float64,generator=torch.Generator().manual_seed(3)),
        residue_mask=torch.ones(4,length,dtype=torch.bool),aatype=torch.zeros(length,dtype=torch.long),
        residue_index=torch.arange(length),chain_id=torch.zeros(length,dtype=torch.long),temperature_k=320.)
    result=diag.initial_state(model,context,2,19,members=3)
    assert model.anchor_calls==1 and model.kernel.posterior_calls==1
    s0=result['S0'].clone();h0=result['H0'].clone()
    for count in (1,4,8):
        branch,_=diag.run_branch(model,result,2,count)
        assert branch['status']=='completed'
        assert torch.equal(result['S0'],s0) and torch.equal(result['H0'],h0)
    assert model.kernel.posterior_calls==1
    assert result['posterior_cholesky'].shape==(1,8,8)
    assert result['factor'][0,1,0]==.23


def test_numerical_failure_keeps_location_original_tensor_and_all_members():
    class BadKernel(LinearKernel):
        def _collective_velocity_step_fixed(self,*args,**kwargs):
            h,s=super()._collective_velocity_step_fixed(*args,**kwargs)
            h[1,0,0]=float('nan')
            return h,s
    k=BadKernel();st=state(k)
    branch,raw=diag.run_branch(SimpleNamespace(kernel=k),st,3,4)
    assert branch['status']=='numerical_failure' and branch['completed_macro_steps']==0
    assert branch['failure']['macro_step']==0 and branch['failure']['micro_step']==0
    assert branch['failure']['nonfinite_per_member']['H_after']==[0,1,0]
    assert torch.isnan(raw['failure_tensors']['H_after'][1,0,0])
    assert raw['H'].shape[0]==3 and raw['H'].shape[1]==1


def test_interface_error_is_not_swallowed_as_numerical_failure():
    class WrongKernel(LinearKernel):
        def _collective_velocity_step_fixed(self,*args,**kwargs):raise ValueError('wrong contract')
    k=WrongKernel()
    with pytest.raises(ValueError,match='wrong contract'):
        diag.run_branch(SimpleNamespace(kernel=k),state(k),3,4)


def test_incomplete_metric_support_is_null_not_silent_nanmean():
    rows=[dict(metadata=dict(protein=f'p{i}'),scalars={'m1/error':float(i)}) for i in range(32)]
    rows[2]['scalars']['m1/error']=None
    result=diag.summarize(rows)['m1/error']
    assert result['equal_protein_mean'] is None and result['n_windows']==31 and result['expected_windows']==32
    assert not result['complete'] and result['available_subset_mean'] is not None
    assert len(rows)==32


def test_exclusive_files_and_explicit_cpu_device_guard(tmp_path,monkeypatch):
    p=tmp_path/'evidence.json';diag.write_new(p,{'original':True})
    with pytest.raises(FileExistsError):diag.write_new(p,{'changed':True})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):diag.cpu_setup()


def test_registered_protocol_cannot_change_midrun(tmp_path,monkeypatch):
    monkeypatch.setattr(diag,'OUT',tmp_path)
    protocol=dict(input_hashes={},source_hashes={},hdf5_file_stats={})
    path=tmp_path/'protocol.json';diag.write_new(path,protocol)
    digest=diag.sha(path)
    diag.verify_registered(protocol,digest)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='registration protocol changed'):
        diag.verify_registered(protocol,digest)
