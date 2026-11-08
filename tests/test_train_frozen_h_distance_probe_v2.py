"""Synthetic runner tests; no actual cache/checkpoint/MD metadata reads."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from run_scripts import probe_train_frozen_h_distances_v2 as runner
from protein_jepa_frame.models.stage_c import frozen_h_distance_probe_v2 as pure


@pytest.fixture
def metadata_fixture(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    for key,value in [('OUTPUT','output'),('CACHE','cache.pt'),('SELECTION','cache.selection.json'),
        ('CACHE_PROTOCOL','cache.protocol.json'),('SUPPORT','support/support.json')]:monkeypatch.setattr(runner,key,value)
    monkeypatch.setattr(runner,'CHECKPOINTS',dict(original='original.pt',current='current.pt'))
    (tmp_path/'support').mkdir()
    for file in ('cache.pt','original.pt','current.pt','synthetic.py'):(tmp_path/file).write_bytes(b'not a torch checkpoint')
    for name in ('support','protocol','started','execution'):(tmp_path/'support'/f'{name}.json').write_text('{}')
    monkeypatch.setattr(runner,'CACHE_SHA',runner.sha('cache.pt'))
    monkeypatch.setattr(runner,'CHECKPOINT_SHAS',{arm:runner.sha(file) for arm,file in runner.CHECKPOINTS.items()})
    monkeypatch.setattr(runner,'SUPPORT_SHA',runner.sha(runner.SUPPORT))
    (tmp_path/'prior').mkdir()
    runner.write_json('prior/protocol.json',dict(previous='synthetic'))
    previous=dict(pid=123,command=['synthetic'],started_utc='recorded',protocol_sha256=runner.sha('prior/protocol.json'))
    runner.write_json('prior/started.json',previous)
    runner.write_json('prior/execution.json',dict(previous,status='failed',returncode=1,failed_phase='current/encoding',error="KeyError: 'stage_a_v2_config'"))
    monkeypatch.setattr(runner,'PRIOR_FAILURE_INPUTS',{p:runner.sha(p) for p in ('prior/protocol.json','prior/started.json','prior/execution.json')})
    rows=[dict(protein=f'p{i}',horizon=runner.HORIZONS[i%4]) for i in range(500)]
    selection=dict(schema_version='backbone_readout_cache_v1',mode='train',seed=42,n_windows=500,
        selected_prefix=500,full_selection=[dict(metadata=row) for row in rows])
    runner.write_json(runner.SELECTION,selection)
    p=dict(mode='train',dataset_role='training_readout_calibration',seed=42,n_windows=500,
        context_frames=8,max_residues=96,horizons=list(runner.HORIZONS),n_forecast_samples=0,
        optimizer_steps=0,test_opened=False,bioemu2_rerun=False,model_tensor_sha256='model',
        selection_file=runner.SELECTION,selection_sha256=runner.sha(runner.SELECTION))
    runner.write_json(runner.CACHE_PROTOCOL,dict(protocol=p))
    support=dict(plans=[dict(cache_row_index=i,split='fit' if i<400 else 'readout_heldout',metadata=row)
                        for i,row in enumerate(rows)])
    monkeypatch.setattr(runner,'support_module',lambda:SimpleNamespace(load_support=lambda *a:support))
    monkeypatch.setattr(runner,'source_files',lambda:['synthetic.py'])
    return selection,dict(protocol=p),support,rows


def test_registration_only_metadata_hashes_and_fixed_fit_order(metadata_fixture,monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('registration deserialized binary'))
    monkeypatch.setattr(runner,'probe_module',lambda:pytest.fail('registered model'))
    protocol=runner.register()
    assert protocol['training']['row_order']==list(range(400))+list(range(100))
    assert protocol['training']['lr']==.001 and protocol['training']['updates']==500
    assert protocol['split']['readout_heldout_rows']==list(range(400,500))
    assert protocol['evaluation']['aggregation_dtype']=='float64'
    reconstruction=protocol['encoder_reconstruction']
    assert reconstruction['strict_load'] is True and reconstruction['latent_dim']==128
    assert reconstruction['state_sources']==dict(original='model/context_encoder.*',current='component_checkpoint.encoder')
    assert 'width12 enables' in reconstruction['orientation_rule'] and 'width6 disables' in reconstruction['orientation_rule']
    assert 'relpos_proj.weight presence enables' in reconstruction['relative_position_rule']
    assert 'saved flags and actual inferred flags are both reported' in reconstruction['rationale']
    assert 'Reject anchor_pos_proj.*' in reconstruction['anchor_rule']
    assert protocol['checkpoint_deserialized_at_registration'] is False
    assert protocol['cache_deserialized_at_registration'] is False
    assert set(protocol['input_hashes'])=={'cache.pt','cache.selection.json','cache.protocol.json','original.pt','current.pt',
        'support/support.json','support/protocol.json','support/started.json','support/execution.json',
        'prior/protocol.json','prior/started.json','prior/execution.json'}
    runner.verify(protocol)
    with pytest.raises(FileExistsError):runner.register()


@pytest.mark.parametrize('mutation',['duplicate','horizon','prefix','link','support_order','support_split','support_missing'])
def test_split_rejects_all_selection_drift_without_replacement(metadata_fixture,mutation):
    selection,cached,support,_=deepcopy(metadata_fixture)
    if mutation=='duplicate':selection['full_selection'][400]['metadata']['protein']='p0'
    elif mutation=='horizon':selection['full_selection'][400]['metadata']['horizon']=4
    elif mutation=='prefix':selection['selected_prefix']=400
    elif mutation=='link':cached['protocol']['selection_sha256']='changed'
    elif mutation=='support_order':support['plans'][0]['cache_row_index']=1
    elif mutation=='support_split':support['plans'][400]['split']='fit'
    else:support['plans'].pop()
    with pytest.raises(ValueError):runner.split_metadata(selection,cached,support)


@pytest.mark.parametrize('stage',['register','run'])
def test_support_failure_precedes_any_binary_deserialization(metadata_fixture,monkeypatch,stage):
    if stage=='run':runner.register()
    def fail(*args):raise ValueError('synthetic incomplete support')
    monkeypatch.setattr(runner,'support_module',lambda:SimpleNamespace(load_support=fail))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('deserialized before support'))
    with pytest.raises(ValueError,match='incomplete support'):getattr(runner,stage)()
    file='registration.execution.json' if stage=='register' else 'execution.json'
    result=json.loads((runner.ROOT/runner.OUTPUT/file).read_text())
    assert result['status']=='failed' and result['returncode']==1 and result['pid']>0


@pytest.mark.parametrize('field',['input','source','snapshot'])
def test_run_provenance_drift_fails_before_deserialization(metadata_fixture,monkeypatch,field):
    runner.register()
    file={'input':'cache.pt','source':'synthetic.py','snapshot':'output/source_snapshot/synthetic.py'}[field]
    (runner.ROOT/file).write_bytes(b'changed')
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('deserialized before provenance'))
    with pytest.raises(ValueError,match='changed'):runner.run()
    record=json.loads((runner.ROOT/'output/execution.json').read_text())
    assert record['status']=='failed' and record['failed_phase']=='verify'
    with pytest.raises(ValueError,match='already attempted'):runner.run()


def test_cpu_guard_precedes_writes_or_cuda_queries(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('queried CUDA'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):runner.register()
    assert not list(tmp_path.iterdir())


def test_actual_cache_schema_and_all500_prefix(metadata_fixture):
    _,saved,_,rows=metadata_fixture
    cache=dict(schema_version='backbone_readout_cache_v1',protocol=saved['protocol'],
        completion=dict(frozen_inputs_verified=True,model_sha256_before='model',model_sha256_after='model'),
        rows=[dict(metadata=row,oracle_H='MUST NOT USE',anchor_H='MUST NOT USE') for row in rows])
    protocol=dict(selected_metadata=rows)
    assert runner.validate_cache(cache,protocol) is cache['rows']
    cache['rows'][-1]['metadata']={'protein':'changed'}
    with pytest.raises(ValueError,match='identities'):runner.validate_cache(cache,protocol)


def synthetic_prepared():
    pairs=torch.tensor([[0,3]],dtype=torch.long)
    labels=dict(target_distance=torch.tensor([2.]),target_log=torch.log1p(torch.tensor([2.])),observed_distance=torch.tensor([1.]))
    return [dict(row=dict(metadata=dict(protein=f'p{i}',horizon=runner.HORIZONS[i%4]),
        oracle_H='not consulted',anchor_H='not consulted'),mask=torch.ones(4,dtype=torch.bool),pairs=pairs,
        labels=labels,labels_sha256='label',physical_support_audit={'plan':'fixed'}) for i in range(500)]


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.05))
    def forward(self,H,mask,pairs):
        return self.weight*H[0,pairs[:,0],0]


def tiny_api(calls=None,fail_at=None):
    calls=[] if calls is None else calls
    def loss(head,H,mask,pairs,target):
        calls.append(float(H[0,0,0]))
        if fail_at is not None and len(calls)==fail_at:raise FloatingPointError('synthetic training failure')
        prediction=head(H,mask,pairs)
        return (prediction-target).square().mean(),prediction
    return SimpleNamespace(new_probe=lambda seed:TinyHead(),probe_loss=loss,
        score_predictions=pure.score_predictions,distance_metrics=pure.distance_metrics)


def features(scale=1.):
    return [dict(target_H=torch.full((1,4,128),scale*(i+1)/500),
        observed_H=torch.full((1,4,128),scale*.01)) for i in range(500)]


def test_fixed500_training_uses_only_target_fit_features_and_same_initial_state(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path);calls=[]
    monkeypatch.setattr(runner,'probe_module',lambda:tiny_api(calls))
    prepared=synthetic_prepared();before=torch.get_rng_state().clone()
    heads={};audits={}
    for arm,scale in [('original',1.),('current',2.)]:
        heads[arm],audits[arm]=runner.fit_readout(arm,prepared,features(scale),tmp_path)
    assert len(calls)==1000 and np.allclose(calls[:500],[(i+1)/500 for i in runner.TRAIN_ORDER])
    assert np.allclose(calls[500:],[2*(i+1)/500 for i in runner.TRAIN_ORDER])
    assert max(runner.TRAIN_ORDER)==399
    assert audits['original']['initial_head_sha256']==audits['current']['initial_head_sha256']
    assert audits['original']['final_head_sha256']!=audits['current']['final_head_sha256']
    assert torch.equal(before,torch.get_rng_state())
    checkpoint=torch.load(tmp_path/'original.readout_latest500.pt',weights_only=False)
    assert checkpoint['step']==500 and checkpoint['training_order']==list(runner.TRAIN_ORDER)
    assert list(checkpoint['optimizer']['state'].values())[0]['step']==500
    logs=[json.loads(line) for line in (tmp_path/'original.training.jsonl').read_text().splitlines()]
    assert len(logs)==500 and all(row['lr']==.001 for row in logs)
    assert all(row['n_pairs']==1 for row in logs)


def test_failure_retains_completed_updates_and_partial_head_no_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path);calls=[]
    monkeypatch.setattr(runner,'probe_module',lambda:tiny_api(calls,fail_at=4))
    with pytest.raises(FloatingPointError,match='synthetic'):runner.fit_readout('original',synthetic_prepared(),features(),tmp_path)
    saved=torch.load(tmp_path/'original.readout_failed.pt',weights_only=False)
    assert saved['completed_updates']==3 and 'head' in saved and 'optimizer' in saved
    assert len((tmp_path/'original.training.jsonl').read_text().splitlines())==3
    with pytest.raises(FileExistsError):runner.fit_readout('original',synthetic_prepared(),features(),tmp_path)


def test_heldout_pair_replay_and_same_head_observed_substitution(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'probe_module',lambda:tiny_api())
    heads={arm:TinyHead().eval() for arm in runner.ARMS}
    with torch.no_grad():heads['current'].weight.fill_(-.1)
    encoded={arm:features(i+1.) for i,arm in enumerate(runner.ARMS)}
    before={arm:runner.state_sha256(head) for arm,head in heads.items()}
    records=runner.score_heldout(heads,encoded,synthetic_prepared(),tmp_path)
    assert len(records)==100 and [row['cache_row_index'] for row in records]==list(range(400,500))
    row=records[0]
    with np.load(tmp_path/row['pair_predictions']['path'],allow_pickle=False) as arrays:
        assert arrays['target_log'].dtype==arrays['target_distance'].dtype==np.float32
        assert arrays['pair_indices'].dtype==np.int64
        for arm in runner.ARMS:
            for role in runner.INPUT_ROLES:
                prediction=arrays[arm+'__'+role+'__predicted_log'].astype(np.float64)
                error=np.expm1(prediction)-arrays['target_distance'].astype(np.float64)
                score=row['scores'][arm+'/'+role]
                assert score['log_mse']==pytest.approx(np.mean((prediction-arrays['target_log'].astype(np.float64))**2),abs=1e-14)
                assert score['mse_angstrom2']==pytest.approx(np.mean(error**2),abs=1e-14)
                assert score['negative_count']==int((prediction<0).sum())
    assert row['scores']['current/target_H']['negative_count']==1
    assert row['scores']['original/target_H']!=row['scores']['original/observed_H_substitution']
    assert row['scores']['observed_CA_persistence']['mae_angstrom']==1.
    assert all(runner.state_sha256(heads[arm])==before[arm] for arm in heads)
    summary=runner.aggregate(records)
    assert summary['overall']['current/target_H']['negative_count']==100
    assert all(summary['h'+str(h)]['original/target_H']['n_proteins']==25 for h in runner.HORIZONS)


def test_aggregation_equal_proteins_and_sqrt_after_averaging_squared_errors():
    roles=[a+'/'+r for a in runner.ARMS for r in runner.INPUT_ROLES]+['observed_CA_persistence']
    records=[]
    for i in range(100):
        squared=1. if i<50 else 9.
        scores={role:dict(log_mse=squared,mae_angstrom=math.sqrt(squared),mse_angstrom2=squared,
            rmse_angstrom=math.sqrt(squared),negative_count=0,n_pairs=1 if i<50 else 100) for role in roles}
        records.append(dict(metadata=dict(protein=f'p{i}',horizon=runner.HORIZONS[i%4]),scores=scores))
    result=runner.aggregate(records)['overall'][roles[0]]
    assert result['log_mse']==5. and result['mae_angstrom']==2.
    assert result['rmse_angstrom']==pytest.approx(np.sqrt(5.)) and result['rmse_angstrom']!=2.
    records[-1]['metadata']['protein']='p0'
    with pytest.raises(ValueError,match='support'):runner.aggregate(records)


def test_npz_excludes_object_pickle_and_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='pickle'):runner.save_npz(tmp_path/'bad.npz',dict(x=np.asarray([{}],dtype=object)))
    runner.save_npz(tmp_path/'good.npz',dict(x=np.asarray([1.],dtype=np.float32)))
    with pytest.raises(FileExistsError):runner.save_npz(tmp_path/'good.npz',dict(x=np.asarray([2.])))


@pytest.mark.parametrize('fail_current',[False,True])
def test_complete_synthetic_run_preserves_encoder_and_rng(metadata_fixture,monkeypatch,fail_current):
    _,saved,support,rows=metadata_fixture
    runner.register();events=[];prepared=synthetic_prepared()
    cache=dict(schema_version='backbone_readout_cache_v1',protocol=saved['protocol'],
        completion=dict(frozen_inputs_verified=True,model_sha256_before='model',model_sha256_after='model'),
        rows=[dict(metadata=row) for row in rows])
    monkeypatch.setattr(torch,'load',lambda *a,**k:(events.append('cache') or cache))
    def prepare(actual,s):
        assert len(actual)==500 and s==support
        events.append('all500');return prepared
    monkeypatch.setattr(runner,'prepare_rows',prepare)
    api=tiny_api()
    def encoder(path,*,kind,expected_sha256,**configuration):
        assert events.index('all500')>events.index('cache')
        events.append(kind+'/encoder')
        if kind=='current':
            assert configuration==dict(configuration_checkpoint_path=runner.local(runner.CHECKPOINTS['original']),
                expected_configuration_sha256=runner.CHECKPOINT_SHAS['original'])
            if fail_current:raise ValueError('synthetic current configuration failure')
        else:assert not configuration
        model=torch.nn.Linear(1,1,bias=False)
        # Constructor RNG is isolated by the real encoder loader; use fork_rng below.
        model.eval().requires_grad_(False)
        return model,dict(kind=kind,checkpoint_sha256=expected_sha256)
    def isolated_encoder(*a,**k):
        with torch.random.fork_rng(devices=[]):return encoder(*a,**k)
    api.load_frozen_encoder=isolated_encoder
    monkeypatch.setattr(runner,'probe_module',lambda:api)
    monkeypatch.setattr(runner,'encode_rows',lambda e,p:features())
    actual_fit=runner.fit_readout
    def fit(arm,*args):
        assert 'original/encoder' in events and 'current/encoder' in events
        events.append(arm+'/fit');return actual_fit(arm,*args)
    monkeypatch.setattr(runner,'fit_readout',fit)
    before=torch.get_rng_state().clone()
    if fail_current:
        with pytest.raises(ValueError,match='current configuration failure'):runner.run()
        assert not any(event.endswith('/fit') for event in events)
        assert not list(runner.local(runner.OUTPUT).glob('*.training.jsonl'))
        assert torch.equal(before,torch.get_rng_state())
        return
    status=runner.run()
    assert status['status']=='completed' and status['returncode']==0
    result=json.loads(runner.local(runner.OUTPUT+'/result.json').read_text())
    assert len(result['per_protein'])==100 and len(json.loads(runner.local(runner.OUTPUT+'/all500_support_audit.json').read_text()))==500
    assert result['tensor_identity_evidence']['cpu_rng_sha256_before']==result['tensor_identity_evidence']['cpu_rng_sha256_after']
    assert torch.equal(before,torch.get_rng_state())
    assert all(result['arm_audits'][arm]['readout']['optimizer_steps']==500 for arm in runner.ARMS)
    assert events.index('current/encoder')<events.index('original/fit')<events.index('current/fit')
    with pytest.raises(ValueError,match='already attempted'):runner.run()


def test_cpu_autocast_rejected_before_registration(tmp_path,monkeypatch):
    assert runner.SUPPORT_MODULE=='protein_jepa_frame.data.frozen_h_distance_support_v2'
    assert runner.SUPPORT=='results/hs_joint_20261101/frozen_h_distance_support_v2/support.json'
    assert 'protein_jepa_frame/data/frozen_h_distance_support_v2.py' in runner.SOURCES
    assert 'run_scripts/prepare_frozen_h_distance_support_v2.py' in runner.SOURCES
    assert 'tests/test_frozen_h_distance_support_v2.py' in runner.SOURCES
    assert 'docs/FROZEN_H_DISTANCE_SUPPORT_V2_20261102.md' in runner.SOURCES
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with torch.autocast('cpu'):
        with pytest.raises(ValueError,match='autocast'):runner.register()
    assert not list(tmp_path.iterdir())


def test_prior_failed_execution_drift_precedes_model_or_cache(metadata_fixture,monkeypatch):
    runner.register()
    runner.local('prior/execution.json').write_text('{}')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('loaded before frozen failed-lineage check'))
    with pytest.raises(ValueError,match='input changed'):runner.run()
    failure=json.loads(runner.local(runner.OUTPUT+'/execution.json').read_text())
    assert failure['status']=='failed' and failure['failed_phase']=='verify'


def test_v2_contract_declares_exact_upstream_and_fresh_restart(metadata_fixture):
    protocol=runner.register();r=protocol['encoder_reconstruction'];repair=protocol['versioned_repair']
    assert r['configuration_sources']['current']=='registered original checkpoint stage_a_v2_config'
    assert r['current_configuration_checkpoint']==runner.CHECKPOINTS['original']
    assert r['current_configuration_sha256']==runner.CHECKPOINT_SHAS['original']
    assert repair['both_arms_fresh'] is True and repair['prior_head_reused'] is False
    assert repair['original_v1_completed_updates']==500 and repair['current_v1_completed_updates']==0
    assert repair['v1_heldout_scored'] is False and repair['prior_failure_inputs']==runner.PRIOR_FAILURE_INPUTS
