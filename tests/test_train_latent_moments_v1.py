"""Synthetic only: no saved scientific inputs, real checkpoints or trajectories."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from run_scripts import diagnose_train_latent_moments_v1 as run
from protein_jepa_frame.models.stage_b1b.h_moment_budget_v1 import one_step_h_moments


class TinyKernel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('coefficient',torch.tensor(.03))
    def compute_forces(self,H,mask,temperature,create_graph=False):
        return -H*self.coefficient*mask[...,None]


def window(h=4):
    n=4;d=128;s=8
    base=torch.arange(n*d,dtype=torch.float32).reshape(n,d)/500
    history=torch.stack([base+.01*k for k in range(4)])
    true=torch.stack([history[-1]+.02*k for k in range(h+1)])
    c=dict(D=torch.full((n,d),1e-4,dtype=torch.float64),C=torch.full((n,d,s),.001,dtype=torch.float64),
        Gamma=torch.full((n,s),1e-4,dtype=torch.float64),K=torch.eye(n,dtype=torch.float64)*.8+.2,
        tail_coefficients=torch.full((d,),.01,dtype=torch.float64))
    return dict(coefficients=c,indices=torch.arange(n),mask=torch.ones(1,n,dtype=torch.bool),
        true=true,history=history,temperature=torch.tensor([348.]),anchor_H=history[-1],anchor_batch_max_abs=0.)


def patch_imports(monkeypatch):
    from protein_jepa_frame.models.correlated_bath_posterior import correlated_bath_filter_step
    monkeypatch.setattr(run,'imports',lambda:(None,None,one_step_h_moments,None,correlated_bath_filter_step,None))


def read_npz(file):
    with np.load(file,allow_pickle=False) as archive:return {key:archive[key].copy() for key in archive.files}


def test_causal_predictions_saved_before_target_and_full_filter(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);patch_imports(monkeypatch)
    row=dict(step=1,horizon=4,batch=dict(protein='synthetic'));w=window();events=[]
    actual_save,actual_filter=run.save_npz,run.assimilate
    def save(file,values):
        if '.prediction.' in str(file):
            assert 'next_H' not in values;events.append('prediction')
        if '.observed.' in str(file):events.append('observed')
        actual_save(file,values)
    def update(*args):
        if len(events)>0:assert events[-1]=='prediction'
        return actual_filter(*args)
    monkeypatch.setattr(run,'save_npz',save);monkeypatch.setattr(run,'assimilate',update)
    before=run.tree_sha(run.cpu_rng());model=TinyKernel()
    result=run.diagnose_window(model,w,row,tmp_path,run.Budget(tmp_path))
    assert events==['prediction','observed']*4 and result['transitions']==4
    assert run.tree_sha(run.cpu_rng())==before
    for i,r in enumerate(result['per_transition']):
        p=read_npz(tmp_path/r['predictive']['path']);o=read_npz(tmp_path/r['observed']['path'])
        assert p['bath_covariance'].shape==(32,32)
        assert np.array_equal(p['force_tail']+p['force_residual'],p['force'])
        assert np.allclose(o['raw_norm2_change'],np.sum(o['next_H']**2-p['H']**2,axis=-1),atol=1e-13)
        if i<3:
            nxt=read_npz(tmp_path/result['per_transition'][i+1]['predictive']['path'])
            assert np.array_equal(nxt['bath_mean'],o['bath_mean_after'])
            assert np.allclose(nxt['bath_covariance'],o['bath_factor_after']@o['bath_factor_after'].T,atol=1e-14)
        assert all(r['moments'][role]['n_residues']==4 for role in ('posterior','C_zero'))


def test_changed_future_cannot_change_earlier_prediction(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);patch_imports(monkeypatch)
    row=dict(step=1,horizon=4,batch=dict(protein='synthetic'));a=window();b=deepcopy(a)
    b['true'][2:]+=.2
    results=[]
    for name,w in [('a',a),('b',b)]:
        out=tmp_path/name;out.mkdir();results.append(run.diagnose_window(TinyKernel(),w,row,out,run.Budget(out)))
    for i in (0,1):
        left=read_npz(tmp_path/results[0]['per_transition'][i]['predictive']['path'])
        right=read_npz(tmp_path/results[1]['per_transition'][i]['predictive']['path'])
        assert all(np.array_equal(left[k],right[k],equal_nan=True) for k in left)
    p=read_npz(tmp_path/results[0]['per_transition'][2]['predictive']['path'])
    q=read_npz(tmp_path/results[1]['per_transition'][2]['predictive']['path'])
    assert not np.array_equal(p['bath_mean'],q['bath_mean'])


def test_failure_preserves_completed_prefix_and_predictive_witness(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);patch_imports(monkeypatch)
    original=run.assimilate;calls=[]
    def fail(*args):
        calls.append(1)
        if len(calls)==5:raise ValueError('synthetic assimilation failure')
        return original(*args)
    monkeypatch.setattr(run,'assimilate',fail)
    with pytest.raises(ValueError,match='synthetic'):
        run.diagnose_window(TinyKernel(),window(),dict(step=1,horizon=4,batch=dict(protein='p')),tmp_path,run.Budget(tmp_path))
    assert (tmp_path/'step001/transition000.json').exists()
    assert (tmp_path/'step001/transition001.prediction.npz').exists()
    assert not (tmp_path/'step001/transition001.json').exists()
    with pytest.raises(FileExistsError):run.diagnose_window(TinyKernel(),window(),dict(step=1,horizon=4,batch=dict(protein='p')),tmp_path,run.Budget(tmp_path))


def saved_fixture(h=4,n=4):
    x=torch.arange((4+h)*n*4*3,dtype=torch.float32).reshape(1,4+h,n,4,3)/100
    b=dict(coords=x,residue_mask=torch.ones(1,4+h,n,dtype=torch.bool),atom_mask=torch.ones(1,4+h,n,4,dtype=torch.bool),
        time_ps=torch.arange(4+h,dtype=torch.float64)[None]*1000,aatype=torch.zeros(1,n,dtype=torch.long),
        residue_index=torch.arange(n)[None],chain_id=torch.zeros(1,n,dtype=torch.long),temperature_k=torch.tensor([348.]))
    saved=dict(batch=b,mask=b['residue_mask'][0,0],anchor=x[0,3,:,1],absolute_time_ps=b['time_ps'][0],
        time_ps=b['time_ps'][0,3:]-3000,true_H=torch.zeros(1,h+1,n,128),
        **{k:b[k][0] for k in ('aatype','residue_index','chain_id')})
    row=dict(horizon=h,replay_tensor_sha256=run.tree_sha(saved),batch=dict(temperature_k=348.))
    return saved,row


@pytest.mark.parametrize('fault',['time','mask','future_mask','atom','nan','Hshape','Hnan','anchor','temperature','metadata','hash'])
def test_saved_boundary_rejects_drift(fault):
    saved,row=saved_fixture();run.validate_saved(saved,row)
    if fault=='time':saved['time_ps'][1]+=.1
    elif fault=='mask':saved['mask']=~saved['mask']
    elif fault=='future_mask':saved['batch']['residue_mask'][0,-1,0]=False
    elif fault=='atom':saved['batch']['atom_mask'][0,-1,0,0]=False
    elif fault=='nan':saved['batch']['coords'][0,0,0,0,0]=float('nan')
    elif fault=='Hshape':saved['true_H']=saved['true_H'][...,:127]
    elif fault=='Hnan':saved['true_H'][0,1,0,0]=float('nan')
    elif fault=='anchor':saved['anchor']=saved['anchor']+1
    elif fault=='temperature':saved['batch']['temperature_k']+=1
    elif fault=='metadata':saved['aatype']=saved['aatype']+1
    else:row['replay_tensor_sha256']='wrong'
    if fault!='hash':row['replay_tensor_sha256']=run.tree_sha(saved)
    with pytest.raises(ValueError):run.validate_saved(saved,row)


def test_observed_raw_centered_identity_and_undefined_tangent():
    H=torch.tensor([[1.,2.,3.],[2.,2.,2.]],dtype=torch.float64);n=H+torch.tensor([[.1,.4,.8],[.2,-.1,.5]])
    values,summary=run.observed_budgets(H,n,torch.tensor([True,False]))
    assert torch.allclose(values['raw_norm2_change'],values['centered_norm2_change']+3*values['channel_mean2_change'],atol=1e-14)
    assert torch.isnan(values['tangent_increment_norm2'][1]) and summary['tangent_increment_norm2'] is not None
    _,none=run.observed_budgets(H,n,torch.tensor([False,False]));assert none['tangent_increment_norm2'] is None


@pytest.mark.parametrize('history_offset',[0.,1e-7,.1])
def test_native_masked_history_batching_and_full_factor_before_mask(monkeypatch,history_offset):
    saved,row=saved_fixture(n=5)
    saved['batch']['residue_mask'][:,:,1]=False
    saved['batch']['coords'][:,:,1]=float('nan')
    saved['anchor']=torch.where(saved['mask'][:,None],saved['batch']['coords'][0,3,:,1],0.)
    row['replay_tensor_sha256']=run.tree_sha(saved)
    calls=[]
    class Encoder:
        def __call__(self,x,mask,*metadata,token_mask):
            calls.append(tuple(x.shape));assert bool(torch.isfinite(x).all())
            value=torch.zeros(x.shape[:3]+(128,),dtype=x.dtype)
            if len(x)==4:value+=history_offset
            return value
    class Kernel:
        energy=SimpleNamespace(log_tail=torch.zeros(128))
        def bath(self,anchor,temp):
            assert torch.equal(anchor,saved['true_H'][:,0])
            return torch.full((1,5,128),1e-4),torch.zeros(1,5,128,8),torch.full((1,5,8),1e-4)
    F=torch.eye(5)[None];F[0,0,1]=.3
    monkeypatch.setattr(run,'imports',lambda:(None,None,None,None,None,lambda *a:F))
    if history_offset==.1:
        with pytest.raises(ValueError,match='batch versus separate'):run.prepare_window(Kernel(),Encoder(),saved,row)
    else:
        result=run.prepare_window(Kernel(),Encoder(),saved,row)
        assert calls==[(1,1,5,4,3),(4,1,5,4,3)]
        assert result['indices'].tolist()==[0,2,3,4]
        expected=(F@F.transpose(-1,-2))[0][[0,2,3,4]][:,[0,2,3,4]].double()
        assert torch.equal(result['coefficients']['K'],expected) and expected[0,0]>1.08
        assert result['anchor_batch_max_abs']==pytest.approx(history_offset,abs=1e-12)
        assert bool(torch.isnan(saved['batch']['coords'][:,:,1]).all())


@pytest.mark.parametrize('limit',['wall','rss','disk'])
def test_resource_limits_fail_without_fallback(tmp_path,monkeypatch,limit):
    budget=run.Budget(tmp_path)
    if limit=='wall':monkeypatch.setattr(run,'MAX_SECONDS',-1)
    elif limit=='rss':monkeypatch.setattr(run,'MAX_RSS_BYTES',0)
    else:(tmp_path/'x').write_bytes(b'1');monkeypatch.setattr(run,'MAX_OUTPUT_BYTES',0)
    with pytest.raises(ValueError,match='budget'):budget.check()


def test_cpu_guard_precedes_output_or_cuda_query(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):run.register()
    assert not list(tmp_path.iterdir())


def test_registration_hashes_only_and_execution_rejects_drift_before_load(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(run,'OUTPUT','output');monkeypatch.setattr(run,'PARENT','parent')
    (tmp_path/'parent').mkdir();(tmp_path/'parent/started.json').write_text('{}')
    (tmp_path/'kernel.pt').write_bytes(b'synthetic');(tmp_path/'config.pt').write_bytes(b'synthetic')
    monkeypatch.setattr(run,'CHECKPOINT','kernel.pt');monkeypatch.setattr(run,'CONFIG_CHECKPOINT','config.pt')
    monkeypatch.setattr(run,'CHECKPOINT_SHA',run.sha('kernel.pt'));monkeypatch.setattr(run,'CONFIG_SHA',run.sha('config.pt'))
    monkeypatch.setattr(run,'PARENT_HASHES',{})
    parent=dict(source_hashes={},source_snapshot='parent',checkpoint_audit={},training_arguments={})
    monkeypatch.setattr(run,'parent_metadata',lambda:(parent,[]));monkeypatch.setattr(run,'sources',lambda:[])
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('binary deserialized'))
    p=run.register();assert p['binary_deserialized_at_registration'] is False
    assert p['future_transitions']==85 and p['bath_dim']==8 and p['initial_prior'].startswith('N(0,I)')
    assert p['new_rollouts']==p['optimizer_updates']==p['readouts']==0
    (tmp_path/'kernel.pt').write_bytes(b'drift')
    with pytest.raises(ValueError,match='input changed'):run.run()
    failed=json.loads((tmp_path/'output/execution.json').read_text())
    assert failed['status']=='failed' and failed['failed_phase']=='verify'
    with pytest.raises(ValueError,match='already attempted'):run.run()


def test_npz_no_pickle_and_nooverwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='pickle'):run.save_npz(tmp_path/'bad.npz',{'x':np.asarray([{}],dtype=object)})
    run.save_npz(tmp_path/'ok.npz',{'x':torch.ones(2)})
    with pytest.raises(FileExistsError):run.save_npz(tmp_path/'ok.npz',{'x':torch.ones(2)})
