"""Synthetic integration of the fixed three-phase diagnostic; no real inputs."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from run_scripts import diagnose_train_relpos_counterfactual_v1 as run
from protein_jepa_frame.models.encoder import _within_frame_canonical_positions


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.relpos_proj=nn.Linear(3,128,bias=False)
        self.register_buffer('rbf_centers',torch.arange(3,dtype=torch.float32),persistent=False)
        self.mutate_buffer=False
        self.seen=[]
        with torch.no_grad():
            self.relpos_proj.weight.copy_(torch.arange(384).reshape(128,3).float()/384)
        self.eval().requires_grad_(False)

    def forward(self,x,mask,aa,ri,ci,token_mask=None):
        assert x.shape[0:2]==(1,1) and torch.equal(mask,token_mask)
        self.seen.append((aa.clone(),ri.clone(),ci.clone()))
        z=_within_frame_canonical_positions(x[:,0],mask[:,0])
        H=torch.tanh(self.relpos_proj(z)+aa[...,None].float()*.01+ri[...,None].float()*.001+ci[...,None].float()*.002)
        if self.mutate_buffer:self.rbf_centers.add_(1)
        return (H*mask[:,0,:,None])[:,None]


@pytest.fixture
def panel(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run.common,'ROOT',tmp_path)
    torch.manual_seed(17)
    model=ToyEncoder()
    windows=[]
    for step,h in ((1,1),(2,2)):
        mask=torch.tensor([1,1,1,1,1,0],dtype=torch.bool)
        # Repeated author indices and mixed/padded IDs are deliberately unchanged.
        fixed=dict(mask=mask,aatype=torch.tensor([1,2,3,4,5,20]),residue_index=torch.tensor([49,49,51,52,53,-1]),
            chain_id=torch.tensor([0,0,0,1,1,-1]),time_ps=torch.arange(h+1,dtype=torch.float64)*1000)
        frames=[]
        with torch.no_grad():
            for i in range(h+1):
                x=torch.randn(6,4,3)*(1+step*.1);x[~mask]=0
                H=model(x[None,None],mask[None,None],*[fixed[k][None] for k in ('aatype','residue_index','chain_id')],token_mask=mask[None,None])[:,0]
                # Saved padded H is irrelevant to valid-support replay.
                saved=H.clone();saved[:,~mask]=123.
                frames.append(dict(coords=x,H_saved=saved,canonical=_within_frame_canonical_positions(x[None],mask[None]),source=dict(index=i,time_ps=i*1000)))
        Ps=[torch.eye(3,dtype=torch.float64) if i==0 else torch.diag(torch.tensor([1.,-1.,1.],dtype=torch.float64)) for i in range(h)]
        row=dict(step=step,horizon=h,protein='synthetic'+str(step),temperature_k=320,transitions=[dict(index=i) for i in range(h)])
        windows.append(dict(source=row,fixed=fixed,frames=frames,permutations=Ps))
    model.seen=[]
    out=tmp_path/'out';out.mkdir()
    return model,windows,out


def execute(panel):
    model,windows,out=panel;trace=[];phase=['start']
    results,audits=run.execute_panel(model,windows,out,run.Budget(out),trace,phase)
    return results,audits,trace


def test_global_prepass_then_pairs_then_post_and_saved_witnesses(panel):
    model,windows,out=panel
    before=run.tree_sha(windows);state=run.model_snapshot(model);rng=run.tree_sha(run.cpu_rng())
    results,audits,trace=execute(panel)
    assert len(trace)==21  # 5 frames*3 + 3 pairs*2
    assert [r['phase'] for r in trace[:10]]==['baseline_pre','identity_sham']*5
    assert [r['phase'] for r in trace[10:16]]==['next_P','current_P']*3
    assert [r['phase'] for r in trace[16:]]==['baseline_post']*5
    assert [a['successful_forwards'] for a in audits]==[10,16,21]
    gate=json.loads((out/'prepass_gate.json').read_text())
    assert gate['frames']==5 and gate['successful_forwards']==10
    assert json.loads((out/'model.initial.json').read_text())['successful_forwards']==0
    assert before==run.tree_sha(windows) and state==run.model_snapshot(model) and rng==run.tree_sha(run.cpu_rng())
    for call,seen in zip(trace,model.seen):
        fixed=windows[call['step']-1]['fixed']
        assert all(torch.equal(a,fixed[k][None]) for a,k in zip(seen,('aatype','residue_index','chain_id')))
    for result in results:
        for frame in result['frames']:
            b=run.load_npz(frame['baseline']);s=run.load_npz(frame['sham']);p=run.load_npz(frame['post'])
            assert run.equal_bytes(b['H_baseline'],s['H_sham']) and run.equal_bytes(s['H_sham'],p['H_post'])
            assert frame['sham_witness']==dict(call_count=1,is_identity=True,identity_input_object_preserved=True,hook_removed=True)
        for pair in result['pairs']:
            a=run.load_npz(pair['sidecar'])
            assert run.equal_bytes(a['H1_next_only'],a['H1_shared'])
            assert torch.equal(a['next_hook_input'],a['canonical1'])
    assert not model.relpos_proj._forward_pre_hooks


def test_last_window_bad_baseline_prevents_all_interventions_preserves_prefix(panel):
    model,windows,out=panel
    windows[-1]['frames'][-1]['H_saved'][0,0,0]+=1
    trace=[]
    with pytest.raises(ValueError,match='baseline pre'):
        run.execute_panel(model,windows,out,run.Budget(out),trace,['start'])
    assert all(r['phase'] in ('baseline_pre','identity_sham') for r in trace)
    assert (out/'model.initial.json').exists() and (out/'step001/frame000.baseline.npz').exists()
    assert not (out/'prepass_gate.json').exists() and not list(out.rglob('pair*.npz'))


def test_nonpersistent_buffer_change_detected_before_gate(panel):
    model,windows,out=panel
    initial=run.model_snapshot(model);model.mutate_buffer=True
    with pytest.raises(ValueError,match='parameters/buffers'):
        execute(panel)
    final=run.model_snapshot(model)
    assert initial['state_dict_sha256']==final['state_dict_sha256']
    assert initial['all_parameters_and_buffers_sha256']!=final['all_parameters_and_buffers_sha256']
    assert not (out/'prepass_gate.json').exists()


def test_wrong_canonical_removes_hook_and_keeps_original_witness(panel):
    model,windows,out=panel;windows[0]['frames'][0]['canonical'][0,0,0]+=.01
    with pytest.raises(ValueError,match='canonical'):
        execute(panel)
    assert (out/'step001/frame000.baseline.npz').exists()
    assert not model.relpos_proj._forward_pre_hooks and not (out/'prepass_gate.json').exists()


def test_exclusive_prefix_not_overwritten(panel):
    execute(panel);out=panel[2];first=(out/'model.initial.json').read_bytes()
    with pytest.raises(FileExistsError):execute(panel)
    assert (out/'model.initial.json').read_bytes()==first


def test_duplicate_keys_fail_before_forward(panel):
    model,windows,out=panel;windows[1]['source']['step']=1
    with pytest.raises(ValueError,match='duplicate'):execute(panel)
    assert model.seen==[] and not (out/'prepass_gate.json').exists()


def serialized_panel(panel):
    _,windows,out=panel;selected=[]
    for w in windows:
        row=dict(w['source']);folder=out/f"input{row['step']}";folder.mkdir()
        file=folder/'fixed.npz';run.save_npz(file,w['fixed']);row['fixed_input']=run.ref(file)
        row['frames']=[];row['transitions']=[]
        for i,f in enumerate(w['frames']):
            file=folder/f'f{i}.npz';run.save_npz(file,dict(source_native_backbone=f['coords'],saved_true_H=f['H_saved'][0],canonical=f['canonical'][0],mask=w['fixed']['mask']))
            row['frames'].append(dict(index=i,**run.ref(file)))
        for i,P in enumerate(w['permutations']):
            file=folder/f'p{i}.npz';run.save_npz(file,dict(selected_permutation=P));row['transitions'].append(dict(index=i,**run.ref(file)))
        selected.append(row)
    return dict(selected_windows=selected)


def test_npz_loading_preserves_ids_mask_native_gauge_and_valid_saved_H(panel):
    p=serialized_panel(panel);loaded=run.load_windows(p)
    for old,new in zip(panel[1],loaded):
        for k,v in old['fixed'].items():assert run.equal_bytes(v,new['fixed'][k])
        for a,b in zip(old['frames'],new['frames']):
            for k in ('coords','H_saved','canonical'):assert run.equal_bytes(a[k],b[k])


@pytest.mark.parametrize('field',['time_ps','residue_index','mask'])
def test_malformed_fixed_npz_rejected(panel,field):
    p=serialized_panel(panel);ref=p['selected_windows'][0]['fixed_input'];values=run.load_npz(ref)
    if field=='time_ps':values[field][1]=2000
    elif field=='mask':values[field]=values[field].float()
    else:values[field]=values[field].float()
    file=run.local(ref['path']);file.unlink();run.save_npz(file,values);ref['sha256']=run.sha(file)
    with pytest.raises(ValueError):run.load_windows(p)
    assert panel[0].seen==[]


def test_cpu_guard_precedes_protocol_or_model_loading(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):run.run()


def test_contract_fixed_counts_boundary_and_no_selection():
    c=run.contract()
    assert c['encoder_forwards']==437 and c['prepass_forwards']==178 and c['pair_forwards']==170 and c['postpass_forwards']==89
    assert c['checkpoint_deserializations']==2 and c['stage_b_model_constructions']==c['decoder_model_constructions']==0
    assert 'no SDPA' in c['attention'] and 'no aggregation' in c['reporting']
    assert c['configuration_checkpoint_sha256']=='b2de4fed497c62cb196d908aa68a3a0d64a59e8c8179b9a91558797384b78689'
