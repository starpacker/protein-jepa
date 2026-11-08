"""Synthetic driver checks; never open saved scientific inputs or checkpoints."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from run_scripts import diagnose_train_pca_continuity_v1 as run


def saved_fixture(h=4,n=6):
    axis=torch.arange(n,dtype=torch.float32)
    ca=torch.stack((axis,axis.sin(),(axis*.7).cos()),-1)
    atoms=torch.tensor([[.1,.2,0.],[0.,0.,0.],[.3,-.1,.2],[.4,.1,.3]])
    x=torch.stack([ca[:,None]+atoms+t*.002 for t in range(4+h)])[None]
    b=dict(coords=x,residue_mask=torch.ones(1,4+h,n,dtype=torch.bool),
        atom_mask=torch.ones(1,4+h,n,4,dtype=torch.bool),time_ps=torch.arange(4+h,dtype=torch.float64)[None]*1000,
        aatype=torch.zeros(1,n,dtype=torch.long),residue_index=torch.arange(n)[None],
        chain_id=torch.zeros(1,n,dtype=torch.long),temperature_k=torch.tensor([348.]))
    H=torch.stack([torch.arange(n*128,dtype=torch.float32).reshape(n,128)/300+t*.01 for t in range(h+1)])[None]
    saved=dict(batch=b,mask=b['residue_mask'][0,0],anchor=x[0,3,:,1],absolute_time_ps=b['time_ps'][0],
        time_ps=b['time_ps'][0,3:]-3000,true_H=H,
        **{k:b[k][0] for k in ('aatype','residue_index','chain_id')})
    row=dict(step=1,horizon=h,replay_tensor_sha256=run.lineage.tree_sha(saved),
        batch=dict(temperature_k=348.,protein='synthetic'))
    return saved,row


def fake_helpers():
    def frame(coords,mask):
        from protein_jepa_frame.models.encoder import _within_frame_canonical_positions
        return dict(ca=coords[:,1].clone(),mask=mask.clone(),eigvals_desc=torch.tensor([3.,2.,1.]),
            canonical=_within_frame_canonical_positions(coords[None],mask[None])[0],basis_determinant=torch.tensor(1.),
            skew_before_sign=torch.tensor([.1,.2,.3]),sign=torch.ones(3),
            abs_standardized_skew=torch.tensor([.1,.2,float('nan')],dtype=torch.float64),
            skew_defined=torch.tensor([True,True,False]),normalized_eigengaps=torch.tensor([.2,.3]))
    def transition(f0,f1,H0,H1,mask):
        return dict(arrays=dict(H_increment=(H1-H0).double(),mask=mask.clone(),full_overlap=torch.eye(3,dtype=torch.float64)),
            summary=dict(H_increment_mean=float((H1-H0).double().square().sum(-1)[mask].mean()),undefined=None))
    return frame,transition


def prepared_run(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUTPUT','diagnostic')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(run,'imports',fake_helpers)
    out=tmp_path/'diagnostic';out.mkdir();rows=[]
    for i,h in enumerate(run.HORIZONS):
        saved,row=saved_fixture(h);row['step']=i+1;sidecar=tmp_path/f'input{i}.pt';torch.save(saved,sidecar)
        row.update(sidecar=str(sidecar.relative_to(tmp_path)),sidecar_sha256=run.sha(sidecar));rows.append(row)
    p=dict(run.contract(),selected_paths=rows,input_hashes={r['sidecar']:r['sidecar_sha256'] for r in rows},
        source_hashes={},source_snapshot='snapshot',parent_source_hashes={},parent_source_snapshot='parent_snapshot')
    run.write_json(out/'protocol.json',p)
    run.write_json(out/'registration.execution.json',dict(status='completed',returncode=0,protocol_sha256=run.sha(out/'protocol.json')))
    return out,p


def test_contract_predeclares_geometry_only_matching_precision_and_no_models():
    c=run.contract();assert c['frames']==89 and c['transitions']==85 and c['horizons']==[1,4,16,64]
    assert c['encoder_calls']==c['model_checkpoint_loads']==c['new_rollouts']==c['optimizer_updates']==0
    assert 'all48' in c['matching'] and 'geometry only' in c['matching'] and 'first exact minimizer' in c['matching']
    assert 'never replace eigendecomposition' in c['precision'] and 'not explained percentages' in c['decomposition']
    assert 'not frozen' in c['pair_support'] and 'not a unique causal partition' in c['limits']


def test_source_collection_includes_frozen_helper_and_loaded_dependencies():
    files=run.sources()
    assert 'protein_jepa_frame/models/encoder.py' in files
    assert 'protein_jepa_frame/models/stage_c/canonical_continuity_v1.py' in files
    assert 'run_scripts/diagnose_train_latent_moments_v1.py' in files


def test_full85_run_loads_only_four_sidecars_and_keeps_H_rng_sources(tmp_path,monkeypatch):
    out,p=prepared_run(tmp_path,monkeypatch);loaded=[];original=torch.load
    def load(path,**kwargs):
        assert str(Path(path).relative_to(tmp_path)) in p['input_hashes'];loaded.append(str(path))
        return original(path,**kwargs)
    monkeypatch.setattr(torch,'load',load)
    monkeypatch.setattr(run.lineage,'load_models',lambda *_:pytest.fail('model load'))
    before=run.lineage.tree_sha(run.lineage.cpu_rng());status=run.run()
    assert status['returncode']==0 and len(loaded)==4
    result=json.loads((out/'result.json').read_text())
    assert result['transitions']==85 and result['frames']==89
    assert result['cpu_rng_sha256_before']==result['cpu_rng_sha256_after']==before
    assert [len(w['transitions']) for w in result['windows']]==[1,4,16,64]
    for row,w in zip(p['selected_paths'],result['windows']):
        source=original(tmp_path/row['sidecar'],weights_only=False)
        for f in w['frames']:
            assert f['summary']['basis_determinant']==1. and f['summary']['source_function_equal'] is True
            assert f['summary']['source_function_max_abs_error']==0.
            with np.load(tmp_path/f['path'],allow_pickle=False) as a:
                assert np.array_equal(a['saved_true_H'],source['true_H'][0,f['index']].numpy())
                assert np.array_equal(a['source_native_backbone'],source['batch']['coords'][0,3+f['index']].numpy())
        assert run.sha(row['sidecar'])==row['sidecar_sha256']
    with pytest.raises(ValueError,match='already attempted'):run.run()


def test_failure_preserves_completed_window_and_transition_prefix(tmp_path,monkeypatch):
    out,_=prepared_run(tmp_path,monkeypatch);frame,transition=fake_helpers();calls=[]
    def fail(*args):
        calls.append(1)
        if len(calls)==3:raise ValueError('synthetic geometry failure')
        return transition(*args)
    monkeypatch.setattr(run,'imports',lambda:(frame,fail))
    with pytest.raises(ValueError,match='synthetic geometry'):run.run()
    e=json.loads((out/'execution.json').read_text())
    assert e['status']=='failed' and e['returncode']==1 and e['completed_windows']==1
    assert e['failed_phase']=='step002/geometry' and e['pid']>0 and e['completed_utc']
    assert (out/'step001/result.json').exists() and (out/'step002/transition000.json').exists()
    assert not (out/'step002/transition001.json').exists() and not (out/'result.json').exists()


@pytest.mark.parametrize('fault',['input','contract','registration','source'])
def test_preload_failures_never_deserialize(tmp_path,monkeypatch,fault):
    out,p=prepared_run(tmp_path,monkeypatch)
    if fault=='input':(tmp_path/p['selected_paths'][0]['sidecar']).write_bytes(b'changed')
    elif fault=='contract':
        p['encoder_calls']=1;(out/'protocol.json').write_text(json.dumps(p))
    elif fault=='registration':
        (out/'registration.execution.json').write_text(json.dumps(dict(status='failed',returncode=1,protocol_sha256='wrong')))
    else:
        p['source_hashes']={'source.py':'0'*64};(out/'protocol.json').write_text(json.dumps(p))
        (tmp_path/'source.py').write_text('drift')
        (out/'registration.execution.json').write_text(json.dumps(dict(status='completed',returncode=0,protocol_sha256=run.sha(out/'protocol.json'))))
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:pytest.fail('preflight must precede deserialization'))
    with pytest.raises(ValueError):run.run()


def test_cpu_isolation_guard_precedes_input_and_cuda_queries(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:pytest.fail('load'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA query'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):run.run()


def test_masked_padding_is_sanitized_and_null_is_not_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'imports',fake_helpers)
    saved,row=saved_fixture(h=1)
    saved['batch']['residue_mask'][:,:,2]=False;saved['batch']['coords'][:,:,2]=float('nan')
    saved['true_H'][:,:,2]=float('nan');saved['anchor'][2]=0.
    row['replay_tensor_sha256']=run.lineage.tree_sha(saved)
    result=run.diagnose_window(saved,row,tmp_path,run.Budget(tmp_path))
    assert result['n_valid_residues']==5
    f=result['frames'][0];assert f['summary']['abs_standardized_skew'][-1] is None
    with np.load(tmp_path/f['path'],allow_pickle=False) as a:
        assert np.array_equal(a['source_native_backbone'][2],np.zeros((4,3)))
        assert np.isnan(a['saved_true_H'][2]).all()


def test_input_mutation_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);saved,row=saved_fixture(h=1);frame,transition=fake_helpers()
    def mutate(f0,f1,H0,H1,mask):
        H1[0,0]+=1
        return transition(f0,f1,H0,H1,mask)
    monkeypatch.setattr(run,'imports',lambda:(frame,mutate))
    with pytest.raises(ValueError,match='input mutated'):run.diagnose_window(saved,row,tmp_path,run.Budget(tmp_path))


def test_actual_frame_source_parity_must_pass_before_frame_saved(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);saved,row=saved_fixture(h=1);frame,transition=fake_helpers()
    def wrong(coords,mask):
        result=frame(coords,mask);result['canonical']=result['canonical']+1
        return result
    monkeypatch.setattr(run,'imports',lambda:(wrong,transition))
    with pytest.raises(ValueError,match='source function parity'):run.diagnose_window(saved,row,tmp_path,run.Budget(tmp_path))
    assert not (tmp_path/'step001/frame000.npz').exists()


def test_register_hashes_and_snapshots_without_deserializing(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);monkeypatch.setattr(run,'OUTPUT','new')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(run,'MOTIVATION_HASHES',{})
    monkeypatch.setattr(run.lineage,'PARENT_HASHES',{})
    for folder,name,data in [(run.MOMENT,'result.json',dict(status='completed',windows=[dict(horizon=h) for h in run.HORIZONS])),
        (run.VERIFIER,'verification.json',dict(status='passed',future_transitions=85)),(run.lineage.PARENT,'started.json',{})]:
        p=tmp_path/folder;p.mkdir(parents=True,exist_ok=True);(p/name).write_text(json.dumps(data))
    sidecar=tmp_path/'synthetic.pt';sidecar.write_bytes(b'only hash, never deserialize')
    rows=[dict(step=i+1,horizon=h,sidecar='synthetic.pt',sidecar_sha256=run.sha(sidecar)) for i,h in enumerate(run.HORIZONS)]
    parent=dict(checkpoint='metadata_only.pt',checkpoint_sha256='a'*64,source_hashes={},source_snapshot='old_snapshot')
    monkeypatch.setattr(run.lineage,'parent_metadata',lambda:(parent,rows))
    (tmp_path/'source.py').write_text('synthetic source\n');monkeypatch.setattr(run,'sources',lambda:['source.py'])
    monkeypatch.setattr(torch,'load',lambda *_a,**_k:pytest.fail('registration deserialization'))
    p=run.register();assert p['binary_deserialized_at_registration'] is False and p['checkpoint_bytes_read'] is False
    assert (tmp_path/'new/source_snapshot/source.py').read_bytes()==(tmp_path/'source.py').read_bytes()
    assert p['input_hashes']['synthetic.pt']==run.sha(sidecar)
    with pytest.raises(FileExistsError):run.register()


def test_real_pure_helper_driver_roundtrip_without_encoder(tmp_path,monkeypatch):
    pytest.importorskip('protein_jepa_frame.models.stage_c.canonical_continuity_v1')
    monkeypatch.setattr(run,'ROOT',tmp_path);saved,row=saved_fixture(h=3)
    before=deepcopy(saved);result=run.diagnose_window(saved,row,tmp_path,run.Budget(tmp_path))
    assert len(result['frames'])==4 and len(result['transitions'])==3
    assert run.lineage.tree_sha(saved)==run.lineage.tree_sha(before)
    for t in result['transitions']:
        with np.load(tmp_path/t['path'],allow_pickle=False) as a:
            assert a['candidate_costs'].shape==(48,) and a['full_overlap'].shape==(3,3)
