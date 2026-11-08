"""Synthetic provenance/support contracts for the CA decoder probe summary."""
from copy import deepcopy
import json
import math

import pytest
import torch

from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
from protein_jepa_frame.models.geometric_local_residual_v1 import GeometricLocalResidualCADecoderV1
from protein_jepa_frame.utils.contracts import state_sha256
from scripts import summarize_ca_local_residual_probe_v1 as s


def leaf(role='forecast',persistence=2.):
    count=12 if role=='forecast' else 1
    members=[]
    for i in range(count):
        rmsd=1. if i%2==0 else 3.
        values=dict(coordinate_mse=rmsd**2/3,ca_rmsd=rmsd,
                    rmsd_to_persistence_ratio=rmsd/persistence if persistence else None)
        values.update({m:.1 if m.endswith('fraction') else 3. for m in s.numeric.GEOMETRY})
        members.append(values)
    scalars={m:s.complete_mean([r[m] for r in members]) for m in s.numeric.SCALARS}
    if role=='forecast':scalars.update(coordinate_energy_score=-.1,physical_distance_energy_score=.5,contact_brier=.2)
    return dict(n_samples=count,n_members=count,per_member=members,scalars=scalars)


def test_forecast_audit_preserves_mean_member_rmsd_and_all12_members():
    data=leaf();assert s.audit_leaf(data,'forecast',2.)==12
    assert data['scalars']['ca_rmsd']==2.
    data['scalars']['ca_rmsd']=math.sqrt(3*data['scalars']['coordinate_mse'])
    with pytest.raises(ValueError,match='arithmetic'):s.audit_leaf(data,'forecast',2.)


@pytest.mark.parametrize('role',['oracle','anchor_identity','wrong_horizon_H'])
def test_deterministic_roles_are_one_member_not_replicated(role):
    data=leaf(role);assert s.audit_leaf(data,role,2.)==1
    data['n_samples']=12
    with pytest.raises(ValueError,match='budget'):s.audit_leaf(data,role,2.)


def test_zero_persistence_denominator_is_null_and_never_zero():
    data=leaf(persistence=0.)
    assert s.audit_leaf(data,'forecast',0.)==12
    assert data['scalars']['rmsd_to_persistence_ratio'] is None
    data['per_member'][0]['rmsd_to_persistence_ratio']=0.
    with pytest.raises(ValueError,match='undefined'):s.audit_leaf(data,'forecast',0.)


def test_null_geometry_member_makes_whole_window_null_without_member_deletion():
    data=leaf();metric='psf_ca_bond_mean_angstrom'
    data['per_member'][0][metric]=None;data['scalars'][metric]=None
    s.audit_leaf(data,'forecast',2.)
    data['scalars'][metric]=3.
    with pytest.raises(ValueError,match='undefined'):s.audit_leaf(data,'forecast',2.)


@pytest.mark.parametrize('mutation',['member_missing','count_bool','negative','nonfinite','fraction','field','proper'])
def test_bad_leaf_counts_or_values_rejected(mutation):
    data=leaf()
    if mutation=='member_missing':data['per_member'].pop()
    elif mutation=='count_bool':data['n_samples']=True
    elif mutation=='negative':data['per_member'][0]['coordinate_mse']=-1.
    elif mutation=='nonfinite':data['per_member'][0]['ca_rmsd']=float('nan')
    elif mutation=='fraction':data['per_member'][0]['psf_ca_nonlocal_clash_fraction']=1.1
    elif mutation=='field':del data['scalars']['coordinate_mse']
    else:data['scalars']['contact_brier']=2.
    with pytest.raises(ValueError):s.audit_leaf(data,'forecast',2.)


def states():
    torch.manual_seed(4)
    base=GeometricDeltaCADecoder(depth=1)
    residual=GeometricLocalResidualCADecoderV1.from_base_decoder(base)
    original=deepcopy(base.state_dict())
    return {'baseline':deepcopy(original),'base_tuned':deepcopy(original),
            'local_residual':deepcopy(residual.state_dict())},original


def test_production_head_initial_bridge_and_final_updates_are_audited():
    initial,original=states()
    report=s.audit_initial_states(initial,original)
    assert report['extra_parameters']==384 and report['extra_initially_zero']
    for arm in ('base_tuned','local_residual'):
        final=deepcopy(initial[arm]);final['gate.weight']+=.1
        if arm=='local_residual':final[s.EXTRA][0,0]=.2
        assert 'gate.weight' in s.audit_final_state(initial[arm],final,arm)['changed_tensor_names']


@pytest.mark.parametrize('mutation',['old_weight','nonzero_extra','extra_shape','hidden_extra','missing'])
def test_initial_nested_state_drift_is_rejected(mutation):
    initial,original=states()
    if mutation=='old_weight':initial['base_tuned']['gate.bias']+=.1
    elif mutation=='nonzero_extra':initial['local_residual'][s.EXTRA][0,0]=.1
    elif mutation=='extra_shape':initial['local_residual'][s.EXTRA]=torch.zeros(3,127)
    elif mutation=='hidden_extra':initial['local_residual']['unused.weight']=torch.ones(2)
    else:del initial['baseline']['centers']
    with pytest.raises(ValueError):s.audit_initial_states(initial,original)


@pytest.mark.parametrize('mutation',['unchanged','zero_extra','nonfinite','shape','buffer'])
def test_false_or_invalid_final_update_is_rejected(mutation):
    initial,_=states();start=initial['local_residual'];final=deepcopy(start)
    if mutation!='unchanged':final['gate.weight']+=.1
    if mutation not in ('zero_extra','unchanged'):final[s.EXTRA][0,0]=.1
    if mutation=='nonfinite':final['gate.weight'][0,0]=float('inf')
    if mutation=='shape':final['centers']=final['centers'][:2]
    if mutation=='buffer':final['centers']+=1.
    with pytest.raises(ValueError):s.audit_final_state(start,final,'local_residual')


def optimizer_fixture():
    state={'centers':torch.ones(16),'input.weight':torch.ones(2,2),'local_residual.weight':torch.zeros(3,128)}
    names=['input.weight','local_residual.weight']
    optimizer=dict(param_groups=[dict(params=[0,1],lr=1e-5,betas=(.9,.999),weight_decay=1e-4)],
        state={i:dict(step=torch.tensor(500.),exp_avg=torch.zeros_like(state[name]),
            exp_avg_sq=torch.ones_like(state[name])) for i,name in enumerate(names)})
    return optimizer,names,state,dict(lr=1e-5,betas=[.9,.999],weight_decay=1e-4)


def test_optimizer_checks_real500_moments_and_json_hyperparameters():
    assert s.audit_optimizer(*optimizer_fixture())['active_parameter_tensors']==2


@pytest.mark.parametrize('mutation',['step','missing','order','shape','nan','negative_moment','hyper'])
def test_optimizer_state_and_hyperparameter_drift_rejected(mutation):
    optimizer,names,state,hyper=optimizer_fixture()
    if mutation=='step':optimizer['state'][1]['step']=torch.tensor(499.)
    elif mutation=='missing':del optimizer['state'][1]
    elif mutation=='order':names.reverse()
    elif mutation=='shape':optimizer['state'][0]['exp_avg']=torch.zeros(1)
    elif mutation=='nan':optimizer['state'][0]['exp_avg'][0,0]=float('nan')
    elif mutation=='negative_moment':optimizer['state'][0]['exp_avg_sq'][0,0]=-1.
    else:hyper['lr']=2e-5
    with pytest.raises(ValueError):s.audit_optimizer(optimizer,names,state,hyper)


def cache(split='val'):
    count=500 if split=='train' else 200;length=4
    p=dict(mode=split,dataset_role='training_readout_calibration' if split=='train' else 'development_validation',
        n_windows=count,complete_panel_windows=count,full_panel=True,seed=42 if split=='train' else 0,
        horizons=[1,4,16,64],max_residues=96,context_frames=8,n_forecast_samples=0 if split=='train' else 12,
        forecast_observed_only=True,before_target_read=True,optimizer_steps=0,test_opened=False,bioemu2_rerun=False,
        frame_interval_ps=1000.,model_tensor_sha256='model')
    rows=[]
    for i in range(count):
        h=s.numeric.HORIZONS[i%4];protein=i if split=='train' else i//4
        temperature=s.numeric.TEMPERATURES[i%5 if split=='train' else protein//10]
        row=dict(metadata=dict(protein=f'{split}{protein}',trajectory=f'{split}{protein}_traj',temperature_k=temperature,
            anchor_frame=7,horizon=h,target_frame=7+h,horizon_ps=h*1000.,context_frames=list(range(8))),
            observed_mask=torch.ones(length,dtype=torch.bool))
        for name in ('embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype','geometry_row_ids'):
            row[name]=torch.arange(length,dtype=torch.long)
        row['peptide_edges']=torch.tensor([[0,1],[1,2],[2,3]])
        row['atom_mask']=torch.ones(length,4,dtype=torch.bool);row['target_atom_mask']=row['atom_mask'].clone()
        row['topology']={name:row[name].tolist() for name in (*('embedding_aatype','embedding_residue_index',
            'embedding_chain_id','chemical_aatype','geometry_row_ids'),'peptide_edges','atom_mask')}
        shapes=dict(observed_R=(1,length,3,3),observed_backbone=(1,length,4,3),target_backbone=(1,length,4,3),
            anchor_H=(1,length,128),oracle_H=(1,length,128),oracle_CA=(1,length,3))
        if split=='val':shapes.update(forecast_H=(12,length,128),forecast_CA=(12,length,3))
        row.update({name:torch.zeros(shape) for name,shape in shapes.items()})
        row['boundary']=dict(forecast_before_target_read=True,future_reencoding_for_forecast=0,
            observed_frames=list(range(8)),label_frames=[7+h])
        row['fingerprints']=dict(aligned_label=state_sha256({'backbone':row['target_backbone']}),
            oracle_features=state_sha256({'H':row['oracle_H'],'CA':row['oracle_CA']}),
            forecast=None if split=='train' else state_sha256({name:row[name] for name in ('forecast_H','forecast_CA')}))
        rows.append(row)
    return dict(schema_version='backbone_readout_cache_v1',protocol=p,rows=rows,
        completion=dict(frozen_inputs_verified=True,completed_utc='done',model_sha256_before='model',model_sha256_after='model'))


@pytest.mark.parametrize('split',['train','val'])
def test_complete_cache_contract_and_fingerprint_reconstruction(split):
    data=cache(split)
    assert len(s.audit_cache(data,split))==(500 if split=='train' else 200)


@pytest.mark.parametrize('mutation',['count','future_boundary','fingerprint','topology','tensor_shape','atom_support','feature_grad'])
def test_corrupt_cache_contract_rejected(mutation):
    data=cache();row=data['rows'][0]
    if mutation=='count':data['rows'].pop()
    elif mutation=='future_boundary':row['boundary']['forecast_before_target_read']=False
    elif mutation=='fingerprint':row['oracle_H'][0,0,0]=.1
    elif mutation=='topology':row['topology']['geometry_row_ids'][0]=123
    elif mutation=='tensor_shape':row['forecast_H']=row['forecast_H'][:11]
    elif mutation=='atom_support':row['target_atom_mask'][0,0]=False
    else:row['oracle_H'].requires_grad_()
    with pytest.raises(ValueError):s.audit_cache(data,'val')


def test_sealed_inputs_and_exclusive_output(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'REPO',tmp_path)
    for name in ('test.jsonl','bioemu.json'):
        with pytest.raises(ValueError):s.local(name)
    out=tmp_path/'result.json';s.write_new(out,{'value':1})
    with pytest.raises(FileExistsError):s.write_new(out,{'value':2})


def execution(phase='run'):
    value=dict(phase=phase,status='completed',exit_code=0,pid=123,started_utc='2026-11-02T01:00:00+00:00',
        completed_utc='2026-11-02T01:04:00+00:00',seconds=240.,protocol_sha256='protocol',source_sha256='source',
        frozen_inputs_sources_verified=True,cpu_threads=2,process_kind='single_actual_process_with_sequential_stages',
        command=[s.producer.SCRIPT,'--mode',phase],output=str(s.BASE/('raw.json' if phase=='run' else 'smoke.json')),
        output_sha256='result')
    if phase=='run':
        stages=[]
        for index,arm in enumerate(s.producer.ARMS):
            stages.append(dict(arm=arm,steps=500,all_optimizer_steps=500,pid=123,status='completed',exit_code=0,
                process_kind='in_process_sequential_stage',started_utc=f'2026-11-02T01:0{index}:00+00:00',
                completed_utc=f'2026-11-02T01:0{index+1}:00+00:00',
                checkpoint=str(s.ARTIFACTS/arm/'checkpoint_latest.pt')))
        stages.append(dict(stage='evaluate',pid=123,status='completed',exit_code=0,
            process_kind='in_process_sequential_stage',started_utc='2026-11-02T01:02:00+00:00',
            completed_utc='2026-11-02T01:03:00+00:00',raw=str(s.BASE/'raw.json'),raw_sha256='result',
            path=str(s.BASE/'score_coordinates.pt'),sha256='coords'))
        value['stages']=stages
    return value


def test_actual_two_processes_and_three_in_process_stages_json_roundtrip():
    for phase in ('smoke','run'):
        record=json.loads(json.dumps(execution(phase)))
        assert s.audit_execution(record,phase,'protocol','source') is record


@pytest.mark.parametrize('mutation',['running','failed','missing_stage','failed_stage','foreign_pid','steps','source','parallel'])
def test_incomplete_or_invented_execution_rejected(mutation):
    record=execution()
    if mutation=='running':record['status']='running'
    elif mutation=='failed':record['exit_code']=1
    elif mutation=='missing_stage':record['stages'].pop()
    elif mutation=='failed_stage':record['stages'][1]['exit_code']=2
    elif mutation=='foreign_pid':record['stages'][1]['pid']=999
    elif mutation=='steps':record['stages'][1]['all_optimizer_steps']=499
    elif mutation=='source':record['source_sha256']='changed'
    else:record['stages'][1]['started_utc']=record['stages'][0]['started_utc']
    with pytest.raises(ValueError):s.audit_execution(record,'run','protocol','source')


def test_build_rejects_incomplete_before_reading_scores_or_tensor_files(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'REPO',tmp_path)
    monkeypatch.setattr(s,'audit_implementation',lambda inputs:({'producer_protocol_sha256':'protocol'},{}))
    def rejected(*args):raise ValueError('run is incomplete or failed')
    monkeypatch.setattr(s,'execution_gate',rejected)
    monkeypatch.setattr(s.torch,'load',lambda *a,**kw:pytest.fail('tensor read before successful execution'))
    monkeypatch.setattr(s.numeric,'analysis',lambda *a:pytest.fail('statistics before successful execution'))
    with pytest.raises(ValueError,match='incomplete'):s.build_summary()
    assert not (tmp_path/s.BASE/'summary.json').exists()


def test_actual_training_stream_order_schedule_and_input_hash_are_audited():
    rows=cache('train')['rows'];metrics=[];batches=[]
    for step,row in enumerate(rows,1):
        metrics.append(dict(step=step,metadata=row['metadata'],loss=1.,gradient_norm=2.,lr=s.producer.learning_rate(step)))
        batches.append(dict(step=step,key=list(s.producer.key(row)),input_tensor_sha256=s.producer.row_fingerprint(row)))
    assert s.audit_training_records(metrics,batches,rows)['n_steps']==500
    assert metrics[-1]['lr']==0 and sum(x['lr']>0 for x in metrics)==499
    batches[2]['input_tensor_sha256']='other'
    with pytest.raises(ValueError,match='stream'):s.audit_training_records(metrics,batches,rows)


@pytest.fixture(scope='module')
def complete_raw_fixture(tmp_path_factory):
    """Real producer scoring and serialization on synthetic tensors, no model."""
    mp=pytest.MonkeyPatch();tmp=tmp_path_factory.mktemp('local-residual-raw')
    mp.setattr(s.producer,'ROOT',tmp)
    datasets={split:cache(split) for split in ('train','val')}
    for split,dataset in datasets.items():
        for index,row in enumerate(dataset['rows']):
            row['embedding_chain_id'].zero_()
            if split=='train' and index<7:row['peptide_edges']=row['peptide_edges'][:2]
            row['topology']={name:row[name].tolist() for name in ('embedding_aatype','embedding_residue_index',
                'embedding_chain_id','chemical_aatype','geometry_row_ids','peptide_edges','atom_mask')}
            row['topology']['sha256']='synthetic-topology'
            # Every horizon for one validation protein has the same observation.
    def decode(model,row,H):
        ca=row['observed_backbone'][:,:,1,:].expand(len(H),-1,-1).clone()
        return {stage:ca.clone() for stage in s.numeric.STAGES}
    mp.setattr(s.producer,'decode',decode)
    protocol=dict(source_hashes={},input_hashes={},precision={'synthetic':True},topology='synthetic',controls={})
    evidence=s.producer.evaluate({arm:None for arm in s.numeric.ARMS},datasets,{}, {'identity':'fixed'},protocol,'protocol')
    raw=json.loads((tmp/s.BASE/'raw.json').read_text())
    sidecar=torch.load(tmp/s.BASE/'score_coordinates.pt',weights_only=False)
    mp.undo()
    return raw,sidecar,datasets,{'identity':'fixed'},protocol,'protocol'


def test_full700_raw_json_roundtrip_member_coordinates_counts_and_cache_reference(complete_raw_fixture):
    report=s.audit_raw(*complete_raw_fixture)
    assert report['leaf_count']==7800 and report['member_coordinate_sets']==21000
    assert len(report['legacy_psf_mismatch_windows']['train'])==7


@pytest.mark.parametrize('mutation',['reference','output_hash','member','geometry_count','extra_train_role','wrong_donor'])
def test_corrupt_raw_or_sidecar_is_rejected(complete_raw_fixture,mutation):
    # Shallow copy unaffected large trees; mutate just the earliest tested leaf.
    original,coords,caches,identity,protocol,digest=complete_raw_fixture
    raw={**original,'per_window':{k:list(v) for k,v in original['per_window'].items()}}
    sidecar={**coords,'per_window':{k:list(v) for k,v in coords['per_window'].items()}}
    if mutation=='reference':
        first=deepcopy(sidecar['per_window']['train'][0]);first['reference_tensors']['target_ca'][0,0]=.1
        sidecar['per_window']['train'][0]=first
    else:
        split='val' if mutation=='wrong_donor' else 'train'
        row=deepcopy(raw['per_window'][split][0]);raw['per_window'][split][0]=row
        leaf=row['routes']['baseline']['oracle']['raw']
        if mutation=='output_hash':leaf['output_tensor_sha256']='wrong'
        elif mutation=='member':leaf['per_member'][0]['ca_rmsd']=1.
        elif mutation=='geometry_count':leaf['geometry_counts']['n_psf_edges']+=1
        elif mutation=='extra_train_role':row['routes']['baseline']['forecast']={}
        else:row['wrong_horizon_donor']['row_index']=100
    with pytest.raises(ValueError):s.audit_raw(raw,sidecar,caches,identity,protocol,digest)


def test_sidecar_reference_tensor_equality_is_dtype_and_shape_strict():
    a={'x':torch.ones(2),'y':[1,2]};b=deepcopy(a)
    assert s.tree_equal(a,b)
    b['x']=b['x'].double();assert not s.tree_equal(a,b)
    assert not s.tree_equal({'x':True},{'x':1})


def test_metadata_registration_records_existing_unread_scores_and_refuses_overwrite(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'REPO',tmp_path);monkeypatch.setattr(s.producer,'ROOT',tmp_path)
    monkeypatch.setattr(s,'verify_statistics',lambda inputs:None)
    names=(s.SCRIPT,s.TEST,s.producer.SCRIPT,*s.statistics_registration.SOURCES)
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic source\n')
    sources={s.producer.SCRIPT:s.sha(s.producer.SCRIPT)}
    snapshot=s.ARTIFACTS/'source_snapshot';s.producer.snapshot(snapshot,sources)
    protocol=dict(schema_version=s.producer.VERSION,hyperparameters=s.producer.HYPER,
        source_hashes=sources,input_hashes={},source_snapshot=str(snapshot))
    s.write_new(s.BASE/'protocol.json',protocol)
    raw=tmp_path/s.BASE/'raw.json';raw.write_text('Not JSON: registration must never parse this performance file')
    monkeypatch.setattr(s.torch,'load',lambda *a,**kw:pytest.fail('registration deserialized tensors'))
    registration=s.register()
    assert registration['output_existence_at_registration'][str(s.BASE/'raw.json')]
    assert not registration['output_existence_at_registration'][str(s.BASE/'summary.json')]
    assert json.loads((tmp_path/s.IMPLEMENTATION/'protocol.json').read_text())==registration
    for name,digest in registration['source_hashes'].items():
        assert s.sha(s.IMPLEMENTATION/'source_snapshot'/name)==digest
    with pytest.raises(ValueError,match='already exists'):s.register()


def test_snapshot_and_tracked_inputs_rechecked_after_audit(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'REPO',tmp_path)
    name='source.py';(tmp_path/name).write_text('fixed')
    source_hash=s.sha(name)
    snap=tmp_path/'snapshot';snap.mkdir();(snap/name).write_text('fixed')
    p=dict(input_hashes={},source_hashes={name:source_hash},source_snapshot='snapshot')
    inputs=s.Inputs();s.audit_frozen(p,inputs);inputs.verify()
    (snap/name).write_text('changed')
    with pytest.raises(ValueError,match='changed'):inputs.verify()
    with pytest.raises(ValueError,match='SHA'):s.audit_frozen(p,s.Inputs())


@pytest.mark.parametrize('corrupt',[False,True])
def test_execution_gate_checks_every_complete_before_after_hash_map(monkeypatch,tmp_path,corrupt):
    monkeypatch.setattr(s,'REPO',tmp_path)
    protocol=dict(source_hashes={s.producer.SCRIPT:'source'},input_hashes={'cache.pt':'cache'})
    immutable={**protocol['input_hashes'],**protocol['source_hashes']}
    for phase in ('smoke','run'):
        record=execution(phase)
        if phase=='smoke':
            record['started_utc']='2026-11-02T00:55:00+00:00';record['completed_utc']='2026-11-02T00:59:00+00:00'
        for item in [record]+record.get('stages',[]):
            item['frozen_before']=dict(immutable);item['frozen_after']=dict(immutable)
        if phase=='run' and corrupt:del record['stages'][1]['frozen_after']['cache.pt']
        s.write_new(s.BASE/f'{phase}.execution.json',record)
        s.write_new(s.BASE/f'{phase}.started.json',{k:record[k] for k in ('phase','pid','command','started_utc')})
    if corrupt:
        with pytest.raises(ValueError,match='identities'):s.execution_gate(s.Inputs(),protocol,'protocol')
    else:assert set(s.execution_gate(s.Inputs(),protocol,'protocol'))=={'smoke','run'}
