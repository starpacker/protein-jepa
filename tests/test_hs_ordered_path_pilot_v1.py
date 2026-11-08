"""CPU contracts for the independently versioned ordered training and runner."""
import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as training
from run_scripts import run_hs_ordered_path_pilot_v1 as pilot


def test_all_twelve_actual_cli_parsers_and_fixed_budgets():
    parsed = pilot.preflight_commands()
    for arm in pilot.ARMS:
        assert tuple(parsed[arm]) == pilot.JOBS
        assert parsed[arm]['train']['coord_objective'] == arm
        assert parsed[arm]['train']['run_kind'] == 'train'
        assert parsed[arm]['endpoint']['n_samples'] == 12
        assert '--max-residues' not in pilot.commands(arm)['endpoint']
        assert parsed[arm]['ensemble']['max_windows'] == 200
        assert parsed[arm]['ordered']['max_windows'] == 200
        assert parsed[arm]['suite']['modes'] == ['correct', 'wrong', 'none']
        assert parsed[arm]['matched8']['modes'] == ['correct']
    assert not torch.cuda.is_initialized()


def test_two_training_commands_only_change_coordinate_objective_and_output():
    left, right = [pilot.commands(arm)['train'][:] for arm in pilot.ARMS]
    for flag in ('--output', '--coord-objective'):
        left[left.index(flag)+1] = right[right.index(flag)+1] = '<varies>'
    assert left == right
    args = training.arguments('new', 'joint_path')
    assert args.steps == 500 and args.n_samples == 12 and args.grad_steps == 64
    assert args.horizons == [1,4,16,64] and args.path_weight == .5
    assert not args.validation_during_training and not args.train_encoder and not args.resume
    assert args.target == 'endpoint' and args.checkpoint_decoder
    assert (args.latent_weight, args.contact_weight, args.clash_weight) == (.1,20.,40.)
    assert args.distance_weight == args.rmsf_weight == args.mode_weight == args.mode_distribution_weight == 0


class MetadataDataset:
    def __init__(self, path):
        self.path = path
        self.loaded = []
    def __len__(self):
        return 1007
    def _locate(self, index):
        import numpy as np
        record = SimpleNamespace(path=self.path, protein_id='train_protein',
                                 trajectory_id='train_T320_R0', temperature_k=320.)
        return record, index + np.arange(68)
    def __getitem__(self, index):
        self.loaded.append(index)
        return dict(coords=torch.full((68,6,4,3),float(index)),
            atom_mask=torch.ones(68,6,4,dtype=torch.bool),
            residue_mask=torch.ones(68,6,dtype=torch.bool),
            aatype=torch.zeros(6,dtype=torch.long), residue_index=torch.arange(6),
            chain_id=torch.zeros(6,dtype=torch.long), time_ps=torch.arange(index,index+68)*1000.,
            temperature_k=torch.tensor(320.), protein_id='train_protein', trajectory_id='train_T320_R0')


def test_metadata_selection_is_exact_original_step_batch_index_without_validation(tmp_path):
    path = tmp_path/'trajectory.h5'; path.write_bytes(b'metadata-only synthetic fixture')
    dataset = MetadataDataset(path); loader = SimpleNamespace(dataset=dataset,batch_size=1)
    first = training.training_identity(loader,4,42)
    assert dataset.loaded == []
    assert first['dataset_index'] == random.Random(42+1000003*4).randrange(len(dataset))
    random.seed(99)
    for _ in range(13): random.random()
    torch.rand(9)  # Stand-ins for unrelated global RNG use: stateless data is unchanged.
    assert training.training_identity(loader,4,42) == first
    batch, actual = training.load_training_batch(loader,4,42)
    assert dataset.loaded == [first['dataset_index']]
    assert actual['horizon'] == 64 and actual['anchor_frame']+64 == actual['target_frame']
    assert batch['coords'].shape == (1,68,6,4,3)
    assert len(actual['actual_batch_sha256']) == 64


def test_training_batch_rejects_data_mutation_during_read(tmp_path,monkeypatch):
    path=tmp_path/'data';path.write_bytes(b'fixed')
    loader=SimpleNamespace(dataset=MetadataDataset(path),batch_size=1)
    original=training.step_batch
    def changed(*args):
        value=original(*args);path.write_bytes(b'changed-length');return value
    monkeypatch.setattr(training,'step_batch',changed)
    with pytest.raises(ValueError,match='changed during read'):
        training.load_training_batch(loader,4,42)


def synthetic_checkpoint(arm='joint_path'):
    component=dict(checkpoint_kind='ca_component_probe',decoder_mode='geometric_ca',
        calibration_args=dict(bond_projection_steps=32,joint=True),
        encoder={'weight':torch.tensor([1.])},delta_ca_decoder={'weight':torch.tensor([2.])})
    args=vars(training.arguments(pilot.ARTIFACTS/arm,arm))
    ck=dict(checkpoint_kind='lasp_ca_dynamics',training_version=training.VERSION,
        step=500,args=args,kernel_config={'latent_dim':128},kernel={'weight':torch.tensor([3.])},
        component_checkpoint=component,target_encoder=None,validation=None,
        optimizer={'state':{0:{'step':torch.tensor(500.)}}},
        sampling_rng=torch.tensor([1,2],dtype=torch.uint8))
    protocol=dict(training_arguments={arm:args},initial_kernel_config=ck['kernel_config'])
    return ck,protocol,dict(component_checkpoint=copy.deepcopy(component))


@pytest.mark.parametrize('field',['step','objective','encoder','decoder','projection','config','teacher','validation','optimizer'])
def test_checkpoint_audit_rejects_any_off_protocol_training_change(field,monkeypatch):
    ck,protocol,initial=synthetic_checkpoint()
    monkeypatch.setattr(pilot.torch,'load',lambda *a,**k:initial)
    assert pilot.checkpoint_audit(ck,'joint_path',protocol)['complete_frozen_component_verified']
    ck=copy.deepcopy(ck)
    if field=='step':ck['step']=499
    elif field=='objective':ck['args']['coord_objective']='marginal_path'
    elif field=='encoder':ck['component_checkpoint']['encoder']['weight']+=1
    elif field=='decoder':ck['component_checkpoint']['delta_ca_decoder']['weight']+=1
    elif field=='projection':ck['component_checkpoint']['calibration_args']['bond_projection_steps']=0
    elif field=='config':ck['kernel_config']['latent_dim']=64
    elif field=='teacher':ck['target_encoder']={'weight':torch.ones(1)}
    elif field=='validation':ck['validation']={'loss':0.}
    else:ck['optimizer']['state'][0]['step']=torch.tensor(499.)
    with pytest.raises(ValueError):pilot.checkpoint_audit(ck,'joint_path',protocol)


@pytest.mark.parametrize('suffix',['.json','.component.pt','.protocol.json','.source_snapshot'])
def test_ordered_outputs_and_registration_sidecars_cannot_be_overwritten(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path)
    output=tmp_path/'joint_path.ordered.json'
    occupied=output.with_suffix(suffix)
    if suffix=='.source_snapshot':occupied.mkdir()
    else:occupied.write_text('original')
    with pytest.raises(FileExistsError):pilot.absent_outputs('joint_path','ordered')


def test_failed_child_preserves_real_attempt_record_without_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path)
    monkeypatch.setattr(pilot,'ARTIFACTS',tmp_path/'artifacts')
    monkeypatch.setattr(pilot,'verify',lambda *a,**k:{})
    calls=[]
    class Failure:
        pid=187345
        def __init__(self,command,**kwargs):calls.append(command)
        def wait(self):return 9
    monkeypatch.setattr(pilot.subprocess,'Popen',Failure)
    protocol=dict(arms=pilot.ARMS,per_arm_commands={'joint_path':{'train':['synthetic-failure']}})
    with pytest.raises(RuntimeError,match='exited 9'):
        pilot.run_arm('joint_path',protocol,'digest')
    result=json.loads((tmp_path/'joint_path.execution.json').read_text())
    assert result['status']=='failed' and len(calls)==1
    assert result['jobs'][0]['pid']==187345 and result['jobs'][0]['returncode']==9
    with pytest.raises(FileExistsError):pilot.run_arm('joint_path',protocol,'digest')


def test_real_tiny_loss_backward_updates_only_kernel_gradients():
    from test_hs_target_pilot import tiny
    model,batch=tiny()
    batch['time_ps']=torch.arange(6,dtype=torch.float64)[None]*model.kernel.cfg.step_ps
    args=training.arguments('unused','joint_path',device='cpu')
    args.n_samples=3;args.grad_steps=2
    before=training.state_sha256(model.bundle)
    terms,norm=training.loss_backward(model,batch,2,args,torch.Generator().manual_seed(42))
    assert torch.isfinite(norm) and norm>0 and terms['loss'].requires_grad
    assert training.state_sha256(model.bundle)==before
    assert all(p.grad is None for p in model.bundle.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.kernel.parameters())


def test_actual_training_window_log_requires_full_metadata_and_fingerprints(tmp_path):
    path=tmp_path/'windows.jsonl';expected=[dict(step=1,protein='p',horizon=1)]
    row={**expected[0],'actual_batch_sha256':'a'*64,'file_stat':{'size':1,'mtime_ns':2}}
    path.write_text(json.dumps(row)+'\n')
    assert pilot.audit_training_windows(path,expected)['windows']==1
    row['horizon']=4;path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='metadata differs'):pilot.audit_training_windows(path,expected)
