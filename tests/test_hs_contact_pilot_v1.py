"""CLI and provenance contracts for the matched contact-estimator experiment."""
import copy
import json

import pytest
import torch

from protein_jepa_frame.training import train_lasp_ca_contact_v1 as training
from run_scripts import run_hs_contact_pilot_v1 as pilot


def test_all_twelve_real_cli_parsers_keep_evaluation_budgets():
    parsed = pilot.preflight_commands()
    for arm in pilot.ARMS:
        assert tuple(parsed[arm]) == pilot.JOBS
        assert parsed[arm]['train']['contact_estimator'] == arm
        assert parsed[arm]['train']['run_kind'] == 'train'
        assert parsed[arm]['ensemble']['max_windows'] == 200
        assert parsed[arm]['endpoint']['n_samples'] == 12
        assert '--max-residues' not in pilot.commands(arm)['endpoint']
        assert parsed[arm]['ordered']['max_windows'] == 200
        assert parsed[arm]['suite']['modes'] == ['correct', 'wrong', 'none']
        assert parsed[arm]['matched8']['modes'] == ['correct']
    assert not torch.cuda.is_initialized()


def test_two_arms_only_change_contact_estimator_and_output():
    commands = [pilot.commands(arm)['train'][:] for arm in pilot.ARMS]
    arguments = [vars(training.arguments(pilot.ARTIFACTS/arm,arm)) for arm in pilot.ARMS]
    for flag in ('--output','--contact-estimator'):
        for command in commands:
            command[command.index(flag)+1] = '<varies>'
    assert commands[0] == commands[1]
    normalized = [{k:v for k,v in a.items() if k not in ('output','contact_estimator')} for a in arguments]
    assert normalized[0] == normalized[1]
    assert normalized[0]['coord_objective'] == 'marginal_path'
    assert normalized[0]['steps'] == 500
    assert not normalized[0]['validation_during_training']


def synthetic_checkpoint(arm='population_u'):
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


@pytest.mark.parametrize('field',['step','estimator','coordinate','encoder','decoder','projection','config','teacher','validation','optimizer','inventory','order','active_support','unused_decay'])
def test_checkpoint_audit_rejects_off_protocol_changes(field,monkeypatch):
    ck,protocol,initial = synthetic_checkpoint()
    monkeypatch.setattr(pilot.torch,'load',lambda *a,**k:initial)
    assert pilot.checkpoint_audit(ck,'population_u',protocol)['complete_frozen_component_verified']
    ck = copy.deepcopy(ck)
    if field == 'step': ck['step'] = 499
    elif field == 'estimator': ck['args']['contact_estimator'] = 'plugin'
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
    with pytest.raises(ValueError): pilot.checkpoint_audit(ck,'population_u',protocol)


@pytest.mark.parametrize('suffix',['.json','.component.pt','.protocol.json','.source_snapshot'])
def test_existing_ordered_outputs_cannot_be_overwritten(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path)
    occupied = (tmp_path/'population_u.ordered.json').with_suffix(suffix)
    if suffix == '.source_snapshot': occupied.mkdir()
    else: occupied.write_text('preserved')
    with pytest.raises(FileExistsError): pilot.absent_outputs('population_u','ordered')


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
    protocol = dict(arms=pilot.ARMS,per_arm_commands={'population_u':{'train':['synthetic-failure']}})
    with pytest.raises(RuntimeError,match='exited 9'):
        pilot.run_arm('population_u',protocol,'digest')
    result = json.loads((tmp_path/'population_u.execution.json').read_text())
    assert result['status'] == 'failed' and len(calls) == 1
    assert result['jobs'][0]['pid'] == 187345 and result['jobs'][0]['returncode'] == 9
    with pytest.raises(FileExistsError): pilot.run_arm('population_u',protocol,'digest')


def test_actual_batch_log_requires_complete_registered_metadata(tmp_path):
    path = tmp_path/'windows.jsonl'
    expected = [dict(step=1,protein='p',horizon=1)]
    row = {**expected[0],'actual_batch_sha256':'a'*64,'file_stat':{'size':1,'mtime_ns':2}}
    path.write_text(json.dumps(row)+'\n')
    assert pilot.audit_training_windows(path,expected)['windows'] == 1
    row['horizon'] = 4
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='metadata differs'):
        pilot.audit_training_windows(path,expected)
