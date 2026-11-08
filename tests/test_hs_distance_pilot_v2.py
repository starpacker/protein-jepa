"""CLI and provenance contracts for the matched distance-path experiment."""
import copy
import json

import pytest
import torch

from protein_jepa_frame.training import train_lasp_ca_distance_v2 as training
from run_scripts import run_hs_distance_pilot_v2 as pilot


def test_all_twelve_real_cli_parsers_keep_evaluation_budgets():
    parsed = pilot.preflight_commands()
    for arm in pilot.ARMS:
        assert tuple(parsed[arm]) == pilot.JOBS
        assert parsed[arm]['train']['distance_path_objective'] == arm
        assert parsed[arm]['train']['run_kind'] == 'train'
        assert parsed[arm]['ensemble']['max_windows'] == 200
        assert parsed[arm]['endpoint']['n_samples'] == 12
        assert '--max-residues' not in pilot.commands(arm)['endpoint']
        assert parsed[arm]['ordered']['max_windows'] == 200
        assert parsed[arm]['suite']['modes'] == ['correct', 'wrong', 'none']
        assert parsed[arm]['matched8']['modes'] == ['correct']
    assert not torch.cuda.is_initialized()


def test_two_arms_only_change_distance_path_objective_and_output():
    commands = [pilot.commands(arm)['train'][:] for arm in pilot.ARMS]
    arguments = [vars(training.arguments(pilot.ARTIFACTS/arm,arm)) for arm in pilot.ARMS]
    for flag in ('--output','--distance-path-objective'):
        for command in commands:
            command[command.index(flag)+1] = '<varies>'
    assert commands[0] == commands[1]
    normalized = [{k:v for k,v in a.items() if k not in ('output','distance_path_objective')} for a in arguments]
    assert normalized[0] == normalized[1]
    assert normalized[0]['coord_objective'] == 'marginal_path'
    assert normalized[0]['distance_path_weight'] == 1.
    assert normalized[0]['contact_estimator'] == 'plugin'
    assert normalized[0]['steps'] == 500
    assert not normalized[0]['validation_during_training']


def synthetic_checkpoint(arm='increment'):
    component = dict(checkpoint_kind='ca_component_probe',decoder_mode='geometric_ca',
        calibration_args=dict(bond_projection_steps=32,joint=True),
        encoder={'weight':torch.tensor([1.])},delta_ca_decoder={'weight':torch.tensor([2.])})
    args = vars(training.arguments(pilot.ARTIFACTS/arm,arm))
    names = [f'active.weight{i}' for i in range(64)] + ['energy.energy_head.2.bias'] + [f'history_posterior.weight{i}' for i in range(8)]
    kernel = {name:torch.tensor([3.]) for name in names}
    checkpoint = dict(checkpoint_kind='lasp_ca_dynamics',training_version=training.VERSION,
        step=500,args=args,kernel_config={'latent_dim':128},kernel=kernel,
        component_checkpoint=component,target_encoder=None,validation=None,
        optimizer={'state':{i:{'step':torch.tensor(500.)} for i in range(64)},
                   'param_groups':[{'params':list(range(73))}]},
        sampling_rng=torch.tensor([1,2],dtype=torch.uint8))
    protocol = dict(training_arguments={arm:copy.deepcopy(args)},initial_kernel_config=copy.deepcopy(checkpoint['kernel_config']))
    return checkpoint,protocol,dict(component_checkpoint=copy.deepcopy(component),kernel=copy.deepcopy(kernel))


@pytest.mark.parametrize('field',['step','estimator','weight','contact','coordinate','encoder','decoder','projection','config','teacher','validation','optimizer','inventory','order','active_support','unused_decay'])
def test_checkpoint_audit_rejects_off_protocol_changes(field,monkeypatch):
    ck,protocol,initial = synthetic_checkpoint()
    monkeypatch.setattr(pilot.torch,'load',lambda *a,**k:initial)
    assert pilot.checkpoint_audit(ck,'increment',protocol)['complete_frozen_component_verified']
    ck = copy.deepcopy(ck)
    if field == 'step': ck['step'] = 499
    elif field == 'estimator': ck['args']['distance_path_objective'] = 'absolute'
    elif field == 'weight': ck['args']['distance_path_weight'] = 2.
    elif field == 'contact': ck['args']['contact_estimator'] = 'population_u'
    elif field == 'coordinate': ck['args']['coord_objective'] = 'joint_path'
    elif field == 'encoder': ck['component_checkpoint']['encoder']['weight'] += 1
    elif field == 'decoder': ck['component_checkpoint']['delta_ca_decoder']['weight'] += 1
    elif field == 'projection': ck['component_checkpoint']['calibration_args']['bond_projection_steps'] = 0
    elif field == 'config': ck['kernel_config']['latent_dim'] = 64
    elif field == 'teacher': ck['target_encoder'] = {'weight':torch.ones(1)}
    elif field == 'validation': ck['validation'] = {'loss':0.}
    elif field == 'inventory': ck['optimizer']['param_groups'][0]['params'].pop()
    elif field == 'order': ck['optimizer']['param_groups'][0]['params'].reverse()
    elif field == 'active_support': ck['optimizer']['state'][64] = {'step':torch.tensor(500.)}
    elif field == 'unused_decay': ck['kernel']['energy.energy_head.2.bias'] *= .99
    else: ck['optimizer']['state'][0]['step'] = torch.tensor(499.)
    with pytest.raises(ValueError): pilot.checkpoint_audit(ck,'increment',protocol)


@pytest.mark.parametrize('suffix',['.json','.component.pt','.protocol.json','.source_snapshot'])
def test_existing_ordered_outputs_cannot_be_overwritten(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path)
    occupied = (tmp_path/'increment.ordered.json').with_suffix(suffix)
    if suffix == '.source_snapshot': occupied.mkdir()
    else: occupied.write_text('preserved')
    with pytest.raises(FileExistsError): pilot.absent_outputs('increment','ordered')


def test_failed_child_preserves_real_pid_exit_and_prevents_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path)
    monkeypatch.setattr(pilot,'ARTIFACTS',tmp_path/'artifacts')
    monkeypatch.setattr(pilot,'verify',lambda *a,**k:{})
    calls = []
    class Failure:
        pid = 187345
        def __init__(self,command,**kwargs): calls.append(command)
        def wait(self): return 9
    monkeypatch.setattr(pilot.subprocess,'Popen',Failure)
    protocol = dict(arms=pilot.ARMS,per_arm_commands={'increment':{'train':['synthetic-failure']}})
    with pytest.raises(RuntimeError,match='exited 9'):
        pilot.run_arm('increment',protocol,'digest')
    result = json.loads((tmp_path/'increment.execution.json').read_text())
    assert result['status'] == 'failed' and len(calls) == 1
    assert result['jobs'][0]['pid'] == 187345 and result['jobs'][0]['returncode'] == 9
    with pytest.raises(FileExistsError): pilot.run_arm('increment',protocol,'digest')


def test_actual_batch_log_requires_complete_registered_metadata(tmp_path,monkeypatch):
    path = tmp_path/'windows.jsonl'
    expected = [dict(step=1,protein='p',horizon=1)]
    physical = dict(record_plan_sha256='a'*64,pair_count=5,record_key='b'*64,physical_metadata_sha256='c'*64)
    support = {'records':{'b'*64:{'hdf_stat':{'size':1,'mtime_ns':2}}}}
    monkeypatch.setattr(pilot,'expected_physical_audit',lambda *a:physical)
    row = {**expected[0],'actual_batch_sha256':'a'*64,'file_stat':{'size':1,'mtime_ns':2},'physical_support_audit':physical}
    path.write_text(json.dumps(row)+'\n')
    assert pilot.audit_training_windows(path,expected,support)['windows'] == 1
    row['horizon'] = 4
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='metadata differs'):
        pilot.audit_training_windows(path,expected,support)


def smoke_fixture():
    metadata = dict(step=4,horizon=64,record_path='/synthetic.h5')
    stat = dict(size=100,mtime_ns=200)
    rng = dict(before_forward={'global_sha256':'a'*64,'sampling_sha256':'b'*64},
        after_forward={'global_sha256':'a'*64,'sampling_sha256':'c'*64},
        after_backward={'global_sha256':'a'*64,'sampling_sha256':'c'*64})
    row = dict(kernel_parameter_tensors=73,gradient_parameters=64,component_gradients_present=False,
        gradient_norm=1.,rng_audit=rng,sampling_rng_sha256='c'*64)
    result = dict(status='completed',optimizer_steps=0,selected_training_step=4,horizon=64,
        n_samples=12,n_residues=96,model_before_sha256='d'*64,model_after_sha256='d'*64,
        batch=dict(metadata,actual_batch_sha256='e'*64,file_stat=stat),
        arms={a:copy.deepcopy(row) for a in pilot.ARMS})
    physical = dict(record_plan_sha256='a'*64,pair_count=5,record_key='b'*64,physical_metadata_sha256='c'*64)
    result['batch']['physical_support_audit'] = physical
    protocol = dict(selected_training_window=metadata,selected_hdf5_stats={'/synthetic.h5':stat})
    return result,protocol


@pytest.mark.parametrize('mutation',['status','step','horizon','members','residues','updates','model',
    'batch','fingerprint','stat','inventory','connected','component','nonfinite','backward_rng','paired_rng','rng_hash',
    'physical_missing','physical_hash','physical_count','physical_plan','physical_key','physical_extra'])
def test_smoke_registration_rejects_incomplete_budget_support_or_rng(mutation,monkeypatch):
    result,protocol = smoke_fixture()
    expected_physical = copy.deepcopy(result['batch']['physical_support_audit'])
    monkeypatch.setattr(pilot,'expected_physical_audit',lambda *a:expected_physical)
    assert pilot.audit_smoke_result(result,protocol,{})['backward_rng_preserved']
    row = result['arms']['increment']
    if mutation == 'status':result['status']='failed'
    elif mutation == 'step':result['selected_training_step']=3
    elif mutation == 'horizon':result['horizon']=16
    elif mutation == 'members':result['n_samples']=8
    elif mutation == 'residues':result['n_residues']=95
    elif mutation == 'updates':result['optimizer_steps']=1
    elif mutation == 'model':result['model_after_sha256']='f'*64
    elif mutation == 'batch':result['batch']['step']=3
    elif mutation == 'fingerprint':result['batch']['actual_batch_sha256']='z'*64
    elif mutation == 'stat':result['batch']['file_stat']={'size':99,'mtime_ns':200}
    elif mutation == 'inventory':row['kernel_parameter_tensors']=64
    elif mutation == 'connected':row['gradient_parameters']=63
    elif mutation == 'component':row['component_gradients_present']=True
    elif mutation == 'nonfinite':row['gradient_norm']=float('nan')
    elif mutation == 'backward_rng':row['rng_audit']['after_backward']['global_sha256']='f'*64
    elif mutation == 'paired_rng':row['rng_audit']['before_forward']['global_sha256']='f'*64
    elif mutation == 'physical_missing':del result['batch']['physical_support_audit']
    elif mutation == 'physical_hash':result['batch']['physical_support_audit']['physical_metadata_sha256']='f'*64
    elif mutation == 'physical_count':result['batch']['physical_support_audit']['pair_count']=6
    elif mutation == 'physical_plan':result['batch']['physical_support_audit']['record_plan_sha256']='f'*64
    elif mutation == 'physical_key':result['batch']['physical_support_audit']['record_key']='f'*64
    elif mutation == 'physical_extra':result['batch']['physical_support_audit']['unregistered']=True
    else:row['sampling_rng_sha256']='f'*64
    with pytest.raises(ValueError):pilot.audit_smoke_result(result,protocol,{})


def test_full500_preflight_failure_precedes_checkpoint_tensor_loading(monkeypatch):
    def fail(*args):
        raise ValueError('synthetic incomplete full500 support')
    monkeypatch.setattr(training,'load_support',fail)
    monkeypatch.setattr(pilot.torch,'load',lambda *a,**k:pytest.fail('checkpoint read before support acceptance'))
    with pytest.raises(ValueError,match='incomplete full500'):
        pilot.fixed_inputs()


@pytest.mark.parametrize('mutation',['missing','hash','count','plan','key','extra','stat'])
def test_training_log_rejects_physical_metadata_change(tmp_path,monkeypatch,mutation):
    metadata=dict(step=1,protein='p',horizon=1)
    audit=dict(record_plan_sha256='a'*64,pair_count=5,record_key='b'*64,physical_metadata_sha256='c'*64)
    stat=dict(size=1,mtime_ns=2)
    support={'records':{'b'*64:{'hdf_stat':stat}}}
    monkeypatch.setattr(pilot,'expected_physical_audit',lambda *a:copy.deepcopy(audit))
    row=dict(metadata,actual_batch_sha256='d'*64,file_stat=copy.deepcopy(stat),physical_support_audit=copy.deepcopy(audit))
    if mutation=='missing':del row['physical_support_audit']
    elif mutation=='hash':row['physical_support_audit']['physical_metadata_sha256']='f'*64
    elif mutation=='count':row['physical_support_audit']['pair_count']=6
    elif mutation=='plan':row['physical_support_audit']['record_plan_sha256']='f'*64
    elif mutation=='key':row['physical_support_audit']['record_key']='f'*64
    elif mutation=='extra':row['physical_support_audit']['new']=True
    else:row['file_stat']['size']=2
    path=tmp_path/'windows.jsonl'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='physical support|HDF stat'):
        pilot.audit_training_windows(path,[metadata],support)


def test_expected_physical_audit_reconstructs_pinned_graph_and_hash():
    from protein_jepa_frame.training.physical_distance_pair_support_v2 import canonical_sha256,record_key
    full=dict(residue_uids=[{'ordinal':i} for i in range(4)],geometry_row_ids=list(range(4)),
        peptide_edges=[[0,1],[1,2],[2,3]],embedding_aatype=[0]*4,
        embedding_residue_index=[1,1,2,3],embedding_chain_id=[0]*4,
        atom_indices=[[4*i+j for j in range(4)] for i in range(4)],atom_mask=[[True]*4 for i in range(4)])
    key=record_key('/synthetic.h5','p')
    plan=dict(record_key=key,record_path='/synthetic.h5',protein='p',max_residues=96,
        full_topology=full,full_topology_sha256=canonical_sha256(full),selected_geometry_row_ids=list(range(4)),
        legacy_mapping={k:full[k] for k in ('embedding_aatype','embedding_residue_index','embedding_chain_id','atom_indices','atom_mask')},
        pair_indices=[[0,3]],pair_count=1)
    plan['record_plan_sha256']=canonical_sha256(plan)
    support={'records':{key:plan}}
    identity=dict(record_path='/synthetic.h5',protein='p')
    result=pilot.expected_physical_audit(identity,support)
    assert result['pair_count']==1 and result['record_key']==key
    assert result['record_plan_sha256']==plan['record_plan_sha256']
    assert len(result['physical_metadata_sha256'])==64
    plan['pair_indices']=[[0,2]]
    with pytest.raises(ValueError,match='record plan changed'):
        pilot.expected_physical_audit(identity,support)
