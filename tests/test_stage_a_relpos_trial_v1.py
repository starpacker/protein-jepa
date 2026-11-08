"""Driver boundary tests with synthetic files/metadata; no actual inputs."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from run_scripts import run_stage_a_relpos_adaptation_v1 as trial


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(trial,'ROOT',tmp_path)
    out=tmp_path/'trial';out.mkdir()
    monkeypatch.setattr(trial,'OUTPUT',out)
    return tmp_path,out


def fake_phase(out,name,protocol_sha,index,*,status='completed'):
    folder=out/name;folder.mkdir()
    t=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(seconds=2*index)
    started=dict(pid=100+index,command=['synthetic',name],started_utc=t.isoformat(),
        protocol_sha256=protocol_sha,phase=name)
    trial.write_json(folder/'started.json',started)
    result=dict(phase=name,protocol_sha256=protocol_sha,status=status,artifacts={})
    if name.startswith('train_'):
        draws=folder/'draws';draws.mkdir();manifest=[]
        for i in range(1,501):
            path=draws/f'{i:04d}.pt';path.write_bytes(str(i).encode());manifest.append(trial.ref(path))
        path=folder/'draws.manifest.json';trial.write_json(path,manifest)
        result['artifacts']['draw_manifest']=trial.ref(path)
    trial.write_json(folder/'result.json',result)
    trial.write_json(folder/'execution.json',dict(started,status=status,returncode=0 if status=='completed' else 1,
        completed_utc=(t+timedelta(seconds=1)).isoformat(),seconds=1.,result=trial.ref(folder/'result.json')))
    return folder


def test_registration_contract_exact_schedule_and_no_scope_expansion():
    p=trial.contract()
    assert p['adaptation']['row_order']==list(range(400))*5
    assert p['readout']['order']==list(range(400))+list(range(100))
    assert p['score']['indices']==list(range(400,500))
    assert p['phase_order']==['smoke','prepare','train_pca','train_zero','train_local','fit_pca','fit_zero','fit_local','score']
    assert not p['test_opened'] and not p['bioemu2_rerun'] and not p['stage_b_or_decoder_constructed']
    assert p['score']['bootstrap']['seed']==20261102


def test_all_completed_and_ordered_prerequisites(workspace):
    _,out=workspace
    for i,name in enumerate(trial.contract()['phase_order'][:-1]):fake_phase(out,name,'p',i)
    actual=trial.prerequisites('score','p',phase_started='2026-01-01T00:00:20+00:00')
    assert list(actual)==trial.contract()['phase_order'][:-1]
    with pytest.raises(ValueError,match='before prerequisites'):
        trial.prerequisites('score','p',phase_started='2026-01-01T00:00:00+00:00')


def test_nested_draw_corruption_is_detected(workspace):
    _,out=workspace;folder=fake_phase(out,'train_pca','p',0)
    (folder/'draws/0250.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact bytes changed'):trial.completed('train_pca','p')


@pytest.mark.parametrize('missing',range(8))
def test_score_cannot_reach_data_if_any_prior_phase_is_absent(workspace,monkeypatch,missing):
    _,out=workspace
    protocol=trial.contract();trial.write_json(out/'protocol.json',protocol);digest=trial.sha(out/'protocol.json')
    trial.write_json(out/'registration.execution.json',dict(status='completed',returncode=0,protocol_sha256=digest))
    for i,name in enumerate(protocol['phase_order'][:-1]):
        if i!=missing:fake_phase(out,name,digest,i)
    monkeypatch.setattr(trial,'verify',lambda p:None)
    monkeypatch.setattr(trial,'backend',lambda **kw:pytest.fail('backend reached before prerequisite gate'))
    monkeypatch.setattr(trial,'read_native',lambda *a,**kw:pytest.fail('trajectory read before all fits'))
    with pytest.raises(FileNotFoundError):trial.run_phase('score')
    failure=json.loads((out/'score/execution.json').read_text())
    assert failure['status']=='failed' and failure['phase']=='score'
    before=(out/'score/execution.json').read_bytes()
    with pytest.raises(FileExistsError):trial.run_phase('score')
    assert (out/'score/execution.json').read_bytes()==before


def test_failed_fit_cannot_be_used(workspace):
    _,out=workspace;fake_phase(out,'fit_local','p',0,status='failed')
    with pytest.raises(ValueError,match='did not complete'):trial.completed('fit_local','p')


def test_phase_identity_status_and_time_guards(workspace):
    _,out=workspace;folder=fake_phase(out,'prepare','p',0)
    execution=json.loads((folder/'execution.json').read_text())
    execution['completed_utc']='2025-01-01T00:00:00+00:00'
    (folder/'execution.json').write_text(json.dumps(execution))
    with pytest.raises(ValueError,match='execution time'):trial.completed('prepare','p')


def test_observed_history_reads_no_endpoint_and_preserves_native_geometry(monkeypatch):
    n=5;coords=torch.arange(8*n*12,dtype=torch.float32).reshape(8,n,4,3)
    context=dict(coords=coords,atom_mask=torch.ones(8,n,4,dtype=torch.bool),residue_mask=torch.ones(8,n,dtype=torch.bool),
        aatype=torch.arange(n),residue_index=torch.tensor([47,48,49,49,50]),chain_id=torch.zeros(n,dtype=torch.long),
        time_ps=123.5+1000*torch.arange(2,10,dtype=torch.float32),temperature_k=torch.tensor(320.))
    metadata={'synthetic':True}
    topology=SimpleNamespace(as_metadata=lambda:metadata,topology_sha256='t')
    spec=SimpleNamespace(record=SimpleNamespace(first_frame_time_ps=123.5),target_frame=13,horizon=4,
        context_frames=torch.arange(2,10).numpy())
    calls=[]
    def native(window,*,endpoint):
        calls.append(endpoint);return context,None,topology,spec
    monkeypatch.setattr(trial,'read_native',native)
    protocol={'windows':[None]*401};protocol['windows'][400]=dict(cache_row_index=400,split='readout_heldout')
    plan=dict(split='readout_heldout',cache_topology=metadata)
    cached=dict(observed_mask=torch.ones(n,dtype=torch.bool),fingerprints=dict(observation=trial.clip_fingerprint(context)))
    before=trial.adapter.fingerprint(context)
    inputs,audit=trial.observed_history(400,protocol,plan,cached)
    assert calls==[False] and not audit['endpoint_coordinates_read']
    assert torch.equal(inputs['history_coords'][0],coords)
    assert torch.equal(inputs['residue_index'][0],context['residue_index'])
    assert float(inputs['endpoint_time_ps'][0])==13123.5
    assert before==trial.adapter.fingerprint(context)
    cached['fingerprints']['observation']='changed'
    with pytest.raises(ValueError,match='differs from fixed'):trial.observed_history(400,protocol,plan,cached)


def test_failed_phase_gets_no_result_and_no_retry(workspace,monkeypatch):
    _,out=workspace;protocol=trial.contract()
    trial.write_json(out/'protocol.json',protocol);digest=trial.sha(out/'protocol.json')
    trial.write_json(out/'registration.execution.json',dict(status='completed',returncode=0,protocol_sha256=digest))
    monkeypatch.setattr(trial,'verify',lambda p:None);monkeypatch.setattr(trial,'backend',lambda **kw:None)
    def fail(*args):raise RuntimeError('synthetic smoke failure')
    monkeypatch.setattr(trial,'smoke',fail)
    with pytest.raises(RuntimeError,match='synthetic smoke'):trial.run_phase('smoke')
    assert not (out/'smoke/result.json').exists()
    assert json.loads((out/'smoke/execution.json').read_text())['returncode']==1
    with pytest.raises(FileExistsError):trial.run_phase('smoke')
