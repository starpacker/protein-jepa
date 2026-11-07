"""Synthetic selective-I/O and complete-panel integration; no real inputs."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from run_scripts import diagnose_train_local_relpos_continuity_v1 as run


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run.common,'ROOT',tmp_path)
    monkeypatch.setattr(run,'OUTPUT','trial');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    return tmp_path


def make_inputs(root,horizons=(1,4,16,64)):
    rows=[];g=torch.Generator().manual_seed(135)
    for step,h in enumerate(horizons,1):
        folder=root/f'input{step}';folder.mkdir()
        mask=torch.tensor([1,1,1,1,1,0],dtype=torch.bool)
        fixed=dict(mask=mask,aatype=torch.tensor([2,3,4,5,6,20]),residue_index=torch.tensor([49,49,51,52,53,-1]),
            chain_id=torch.tensor([0,0,0,1,1,-1]),time_ps=torch.arange(h+1,dtype=torch.float64)*1000)
        file=folder/'fixed.npz';run.save_npz(file,fixed)
        row=dict(step=step,horizon=h,protein='synthetic'+str(step),temperature_k=320,fixed_input=run.ref(file),frames=[],transitions=[])
        for i in range(h+1):
            x=torch.randn(6,4,3,generator=g);x[~mask]=0
            ca=x[None,:,1];m=mask[None,:,None].float();centroid=(ca*m).sum(1,keepdim=True)/m.sum(1,keepdim=True)
            centered=(ca-centroid)*m
            values=dict(source_native_backbone=x,mask=mask,centered=centered[0],centroid=centroid[0,0],canonical=centered[0].clone(),basis=torch.eye(3))
            file=folder/f'frame{i}.npz'
            # A forbidden object member witnesses true lazy selected-key access.
            with file.open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in values.items()},saved_true_H=np.asarray([object()],dtype=object))
            row['frames'].append(dict(index=i,time_ps=i*1000.,**run.ref(file)))
        for i in range(h):
            file=folder/f'pair{i}.npz'
            with file.open('xb') as f:np.savez(f,R=np.eye(3),H_increment=np.asarray([object()],dtype=object))
            row['transitions'].append(dict(index=i,time_ps=i*1000.,next_time_ps=(i+1)*1000.,**run.ref(file)))
        rows.append(row)
    return rows


def protocol(root,rows):
    out=root/'trial';out.mkdir();p=dict(run.contract(),selected_windows=rows,input_hashes={},source_hashes={},source_snapshot='unused')
    run.write(out/'protocol.json',p)
    run.write(out/'registration.execution.json',dict(status='completed',returncode=0,protocol_sha256=run.sha(out/'protocol.json')))
    return p,out


def test_full89_85_coverage_lazy_keys_metadata_and_no_model_or_fits(workspace,monkeypatch):
    rows=make_inputs(workspace);p,out=protocol(workspace,rows)
    accessed=[];original=np.lib.npyio.NpzFile.__getitem__
    def getitem(self,key):
        accessed.append(key)
        assert key not in ('saved_true_H','H_increment')
        return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',getitem)
    def forbidden(*args,**kwargs):raise AssertionError('new model/PCA/Kabsch prohibited')
    monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(torch.linalg,'eigh',forbidden);monkeypatch.setattr(torch.linalg,'svd',forbidden)
    before=run.tree_sha(run.cpu_rng());status=run.run()
    assert status['returncode']==0 and status['completed_windows']==4 and before==run.tree_sha(run.cpu_rng())
    result=json.loads((out/'result.json').read_text())
    assert result['frames']==89 and result['transitions']==85
    assert result['H_arrays_loaded']==result['encoder_calls']==result['model_checkpoint_loads']==result['new_PCA_fits']==result['new_Kabsch_fits']==0
    assert set(accessed)==set(run.FRAME_KEYS+run.FIXED_KEYS+run.TRANSITION_KEYS)
    assert result['input_tensor_sha256_before']==result['input_tensor_sha256_after']
    for source,window in zip(rows,result['windows']):
        assert len(window['frames'])==source['horizon']+1 and len(window['transitions'])==source['horizon']
        old=run.load_keys(source['fixed_input'],run.FIXED_KEYS);new=run.load_keys(window['fixed_input'],run.FIXED_KEYS)
        assert all(torch.equal(old[k],new[k]) for k in old)
        for frame in window['frames']:
            assert all(mode in frame['summary'] for mode in ('pca','zero','local'))
        for transition in window['transitions']:
            assert transition['summary']['zero']['source_jump_mean_squared']==0
    assert (out/'inputs.initial.json').exists()


def test_bad_last_rotation_preflight_fails_before_any_frame_and_preserves_attempt(workspace):
    rows=make_inputs(workspace);last=rows[-1]['transitions'][-1];file=run.local(last['path'])
    file.unlink();run.save_npz(file,dict(R=torch.zeros(3,3,dtype=torch.float64)));last['sha256']=run.sha(file)
    _,out=protocol(workspace,rows)
    with pytest.raises(ValueError,match='proper R'):run.run()
    e=json.loads((out/'execution.json').read_text())
    assert e['status']=='failed' and e['completed_windows']==0 and e['failed_phase']=='selective_NPZ_load'
    assert not list(out.glob('step*'))
    with pytest.raises(ValueError,match='already attempted'):run.run()


def test_pair_failure_retains_complete_earlier_window_and_frame_prefix(workspace,monkeypatch):
    rows=make_inputs(workspace);_,out=protocol(workspace,rows);frame,pair=run.imports();calls=[0]
    def failing_pair(*args):
        calls[0]+=1
        if calls[0]==3:raise ValueError('synthetic pair failure')
        return pair(*args)
    monkeypatch.setattr(run,'imports',lambda:(frame,failing_pair))
    with pytest.raises(ValueError,match='synthetic pair'):run.run()
    e=json.loads((out/'execution.json').read_text())
    assert e['completed_windows']==1 and e['failed_phase']=='step002/transition001'
    assert (out/'step001/result.json').exists() and (out/'step002/frame004.json').exists()
    assert (out/'step002/transition000.npz').exists() and not (out/'step002/transition001.npz').exists()
    assert not (out/'result.json').exists()


def test_input_mutation_is_rejected_before_saving_changed_frame(workspace,monkeypatch):
    rows=make_inputs(workspace,(1,));w=run.load_windows(dict(selected_windows=rows))[0];out=workspace/'out';out.mkdir()
    frame,pair=run.imports()
    def mutating(*args):
        value=frame(*args);args[0][0,0,0]+=1;return value
    monkeypatch.setattr(run,'imports',lambda:(mutating,pair))
    with pytest.raises(ValueError,match='input mutated'):run.diagnose_window(w,out,run.Budget(out),['start'])
    assert not (out/'step001/frame000.npz').exists()


def test_frame_output_mutation_detected_during_pair(workspace,monkeypatch):
    rows=make_inputs(workspace,(1,));w=run.load_windows(dict(selected_windows=rows))[0];out=workspace/'out';out.mkdir()
    frame,pair=run.imports()
    def mutating(a,b,R,mask):
        value=pair(a,b,R,mask);a['arrays']['local_canonical'][0,0]+=1;return value
    monkeypatch.setattr(run,'imports',lambda:(frame,mutating))
    with pytest.raises(ValueError,match='frame output mutated'):run.diagnose_window(w,out,run.Budget(out),['start'])
    assert (out/'step001/frame001.npz').exists() and not (out/'step001/transition000.npz').exists()


def test_unregistered_key_rejected_before_numpy_open(workspace,monkeypatch):
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('opened NPZ'))
    with pytest.raises(ValueError,match='key selection'):run.load_keys(dict(path='unused',sha256='unused'),('saved_true_H',))


def test_cpu_guard_before_protocol_or_input_access(workspace,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):run.run()


def test_contract_no_h_no_fits_and_unfiltered_support():
    c=run.contract();assert c['branches']==['pca','zero','local'] and c['frames']==89 and c['transitions']==85
    assert c['H_arrays_loaded']==c['encoder_calls']==c['model_checkpoint_loads']==c['new_PCA_fits']==c['new_Kabsch_fits']==0
    assert 'degenerate' in c['support'] and 'no pooled' in c['reporting']
    assert not any('H' in key for key in c['frame_npz_keys'])
