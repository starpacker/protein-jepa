"""CPU synthetic contracts; no checkpoint, dataset or stored diagnostic reads."""
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest
import torch

from protein_jepa_frame.models.stage_b1b.model import StageB1b
from run_scripts import diagnose_hs_history_mean_v1 as diag


class Kernel:
    rollout_microstep_v1=StageB1b.rollout_microstep_v1
    _collective_velocity_step_fixed=StageB1b._collective_velocity_step_fixed

    def __init__(self):
        self.cfg=SimpleNamespace(use_bath=True,bath_coupling='velocity',bath_dim=2,step_ps=.2)
        self.posterior_calls=[]
        self.chol=None

    def bath(self,h,temp):
        return torch.full_like(h,.2),torch.full((*h.shape,2),.1,dtype=h.dtype),torch.full((*h.shape[:2],2),.4,dtype=h.dtype)

    def compute_forces(self,h,mask,temp,create_graph=False):return -.3*h*mask[...,None]

    def initial_bath_state(self,history,dt,temp,mask,seq_feature,noise_factor,create_graph):
        self.posterior_calls.append((history.clone(),dt.clone(),temp.clone(),mask.clone(),seq_feature.clone(),noise_factor.clone()))
        assert not create_graph
        return (.2+.1*history.mean(1)[...,:2])*mask[...,None],self.chol.clone()


class Bundle:
    def decode_ca(self,h,h0,aa,ri,ci,rotation,anchor,mask):
        return (anchor+.1*(h-h0)[...,:3])*mask[...,None]


def fixture_state(members=12,horizon=3,kernel=None):
    k=kernel or Kernel();g=torch.Generator().manual_seed(12)
    mask=torch.tensor([[True,True,True,False]])
    h0=torch.randn(1,4,4,generator=g)*mask[...,None]
    history=h0[:,None].expand(-1,4,-1,-1).clone()
    history[:,:3]+=torch.tensor([.15,.1,-.1,.0])[None,None,None,:]*mask[:,None,:,None]
    chol=torch.zeros(1,8,8);chol[0,:6,:6]=torch.eye(6)*.7;chol[0,1,0]=.15
    k.chol=chol.clone()
    mean=(.2+.1*history.mean(1)[...,:2])*mask[...,None]
    eps=torch.randn(1,members,8,generator=g)
    centered=torch.einsum('bij,bmj->bmi',chol,eps).reshape(1,members,4,2)
    s0=(mean[:,None]+centered)[0]
    h=h0.expand(members,-1,-1).clone();temp=torch.full((members,),320.)
    factor=torch.eye(4)[None].expand(members,-1,-1).clone();factor[:,1,0]=.2
    d,c,gamma=k.bath(h,temp)
    state=dict(H0=h,S0=s0,mask=mask.expand(members,-1).clone(),temp=temp,factor=factor,D=d,C=c,Gamma=gamma,
        anchor_H=h0,history=history,dt_history=torch.full((1,4),.2),posterior_mean=mean,
        posterior_cholesky=chol,posterior_standard_normal=eps,
        fine_H=torch.randn(members,horizon,8,4,4,generator=g),fine_S=torch.randn(members,horizon,8,4,2,generator=g),
        metadata=[torch.zeros(1,4,dtype=torch.long),torch.arange(4)[None],torch.zeros(1,4,dtype=torch.long)],
        rotation=torch.eye(3)[None,None].expand(1,4,-1,-1),anchor=torch.randn(1,4,3,generator=g))
    return SimpleNamespace(kernel=k,bundle=Bundle()),state


def test_repeated_control_changes_only_history_with_exact_original_condition():
    model,state=fixture_state();control=diag.posterior_controls(model,state)
    assert len(model.kernel.posterior_calls)==1
    hist,dt,temp,mask,h0,fac=model.kernel.posterior_calls[0]
    assert torch.equal(hist,state['anchor_H'][:,None].expand(-1,4,-1,-1))
    for actual,expected in ((dt,state['dt_history']),(temp,state['temp'][:1]),(mask,state['mask'][:1]),
                            (h0,state['anchor_H']),(fac,state['factor'][:1])):
        assert torch.equal(actual,expected)
    assert not torch.equal(control['means']['repeated_anchor_history'],control['means']['real_history'])
    assert control['means']['repeated_anchor_history'].abs().sum()>0
    assert control['means']['zero_mean'].count_nonzero()==0


def test_covariance_one_ulp_change_fails_closed():
    model,state=fixture_state()
    model.kernel.chol[0,0,0]=torch.nextafter(model.kernel.chol[0,0,0],torch.tensor(float('inf')))
    with pytest.raises(diag.CovarianceMismatch,match='bitwise'):
        diag.posterior_controls(model,state)


def test_initial_states_share_one_full_covariance_draw_and_real_is_bitwise():
    model,state=fixture_state();control=diag.posterior_controls(model,state)
    centered,s0=diag.shared_initial_states(state,control)
    assert torch.equal(s0['real_history'],state['S0'])
    assert torch.equal(s0['zero_mean'],centered[0])
    assert not torch.equal(centered.flatten(),state['posterior_standard_normal'].flatten())
    for arm in diag.ARMS:
        assert torch.equal(s0[arm],(control['means'][arm][:,None]+centered)[0])
        assert s0[arm][:,3].count_nonzero()==0
    state['S0'][0,0,0]=torch.nextafter(state['S0'][0,0,0],torch.tensor(float('inf')))
    with pytest.raises(ValueError,match='posterior draw differs'):diag.shared_initial_states(state,control)


def test_whitening_packs_valid_residues_and_subtracts_float64_values():
    mean=torch.tensor([[[2.**24,1.],[float('nan'),float('nan')],[-3.,.5]]],dtype=torch.float32)
    ref=torch.tensor([[[-1.,-.25],[float('nan'),float('nan')],[1.,-.125]]],dtype=torch.float32)
    mask=torch.tensor([[True,False,True]]);idx=torch.tensor([0,1,4,5])
    lower=torch.tensor([[.1,0,0,0],[.03,.2,0,0],[-.1,.2,.5,0],[.2,-.1,.3,.4]],dtype=torch.float32)
    chol=torch.zeros(1,6,6);chol[0,idx[:,None],idx[None,:]]=lower
    result=diag.mean_difference(mean,ref,chol,mask)
    delta=mean.double().numpy().reshape(-1)[idx]-ref.double().numpy().reshape(-1)[idx]
    expected=np.linalg.solve(lower.double().numpy(),delta)
    assert result['n_valid_bath_dimensions']==4
    assert result['raw_l2']==pytest.approx(np.linalg.norm(delta),rel=1e-14)
    assert result['mahalanobis_squared']==pytest.approx(expected@expected,rel=1e-14)
    # 2**24 - (-1) is not representable in float32: this rejects pre-cast subtraction.
    assert float((mean-ref)[0,0,0])!=float(delta[0])


def test_actual_primitive_real_replay_and_only_s0_changes_paths():
    model,state=fixture_state();before=copy.deepcopy(state)
    controls=diag.posterior_controls(model,state);_,s0=diag.shared_initial_states(state,controls)
    zh,zs=[diag.coarsen(state[k],1) for k in ('fine_H','fine_S')]
    original_h,original_s=model.kernel.rollout_microstep_v1(state['H0'],state['mask'],state['temp'],3,
        S0=state['S0'],seq_feature=state['H0'],detach=True,return_bath=True,noise_factor=state['factor'],
        collective_mobility=True,microsteps=1,brownian_H=zh,brownian_S=zs,integrator='explicit')
    original=dict(H=original_h,S=original_s,CA=diag.decode_macro(model,state,original_h))
    outputs={arm:diag.sample_arm(model,state,s0[arm],zh,zs,3) for arm in diag.ARMS}
    diag.verify_real_replay(*outputs['real_history'],original)
    assert model.kernel.posterior_calls and len(model.kernel.posterior_calls)==1
    assert not torch.equal(outputs['zero_mean'][1]['H'],original_h)
    for key,value in state.items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,before[key]),key


@pytest.mark.parametrize('key',['H','S','CA'])
def test_real_replay_rejects_single_ulp_in_any_saved_path(key):
    original={k:torch.ones(2,3,4,3) for k in ('H','S','CA')};changed=copy.deepcopy(original)
    changed[key][0,1,2,0]=torch.nextafter(changed[key][0,1,2,0],torch.tensor(float('inf')))
    with pytest.raises(ValueError,match=key):diag.verify_real_replay({'status':'completed'},changed,original)


def test_failed_transition_preserves_all_members_and_actual_nonfinite_state():
    class BadKernel(Kernel):
        def _collective_velocity_step_fixed(self,*args,**kwargs):
            h,s=super()._collective_velocity_step_fixed(*args,**kwargs);h[1,0,0]=float('nan');return h,s
    model,state=fixture_state(kernel=BadKernel());zh,zs=[diag.coarsen(state[k],1) for k in ('fine_H','fine_S')]
    result,raw=diag.sample_arm(model,state,state['S0'],zh,zs,3)
    assert result['status']=='numerical_failure' and result['completed_macro_steps']==0
    assert raw['H'].shape==(12,1,4,4) and raw['S'].shape==(12,1,4,2)
    assert result['failure']['nonfinite_per_member']['H_after']==[0,1]+[0]*10
    assert torch.isnan(raw['failure_tensors']['H_after'][1,0,0])


def test_precise_metric_failure_is_retained_but_interface_errors_propagate(monkeypatch):
    def numerical(*args):raise diag.MetricNumericalFailure('nonfinite score')
    monkeypatch.setattr(diag,'coordinate_scores',numerical)
    result={'status':'completed'};diag.attach_scores(result,*([None]*6))
    assert result['status']=='numerical_failure' and result['endpoint_scores'] is None
    assert result['failure']['stage']=='endpoint_metric'
    def interface(*args):raise ValueError('wrong API')
    monkeypatch.setattr(diag,'coordinate_scores',interface)
    with pytest.raises(ValueError,match='wrong API'):diag.attach_scores({'status':'completed'},*([None]*6))


def test_later_covariance_mismatch_aborts_global_preflight_before_sampling(tmp_path,monkeypatch):
    model,state=fixture_state();items=[]
    for step in range(3):
        path=tmp_path/f'raw{step}.pt';s=copy.deepcopy(state)
        if step==1:s['posterior_cholesky'][0,0,0]+=1e-5
        torch.save(dict(metadata={'step':step},state=s),path);items.append(dict(path=str(path),metadata={'step':step}))
    monkeypatch.setattr(diag,'BASE',tmp_path)
    monkeypatch.setattr(diag,'sample_arm',lambda *a:pytest.fail('sampling before preflight'))
    with pytest.raises(diag.CovarianceMismatch):diag.preflight_all(model,{'input_rows':items})
    report=json.loads((tmp_path/'covariance_preflight_failure.json').read_text())
    assert report['already_checked_windows']==1 and report['sampling_started'] is False
    assert not (tmp_path/'mean_controls.pt').exists()


def test_incomplete_fixed_panel_retains_null_full_mean_and_explicit_support():
    rows=[dict(metadata={'protein':f'p{i}'},scalars={'zero_minus_real/error':float(i)}) for i in range(32)]
    rows[3]['scalars']['zero_minus_real/error']=None
    value=diag.summarize(rows)['zero_minus_real/error']
    assert value['equal_protein_mean'] is None and value['n_windows']==31 and value['expected_windows']==32
    assert value['available_subset_mean'] is not None and value['complete'] is False


def test_protocol_lock_and_exclusive_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(diag,'BASE',tmp_path);protocol={'input_hashes':{},'source_hashes':{}}
    path=tmp_path/'protocol.json';diag.write_new(path,protocol);digest=diag.sha(path)
    diag.verify_files(protocol,digest)
    with pytest.raises(FileExistsError):diag.write_new(path,protocol)
    path.write_text('{}')
    with pytest.raises(ValueError,match='protocol changed'):diag.verify_files(protocol,digest)
