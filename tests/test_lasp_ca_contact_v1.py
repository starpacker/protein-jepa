"""Synthetic training/CLI/optimizer contracts; no real data or checkpoint loads."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from protein_jepa_frame.training import train_lasp_ca_contact_v1 as training
from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as ordered


def test_fixed_arguments_only_change_estimator_and_version_from_frozen_control():
    old=vars(ordered.arguments('out','marginal_path'))
    arms=[vars(training.arguments('out',arm)) for arm in training.ARMS]
    for values in arms:
        assert {k:v for k,v in values.items() if k not in ('contact_estimator','training_version')} == {
            k:v for k,v in old.items() if k!='training_version'}
        assert values['training_version']==training.VERSION
    assert {k:v for k,v in arms[0].items() if k!='contact_estimator'} == {
        k:v for k,v in arms[1].items() if k!='contact_estimator'}
    assert old['steps']==500 and old['horizons']==[1,4,16,64] and old['seed']==42
    assert old['n_samples']==12 and old['grad_steps']==64 and old['checkpoint_decoder']
    assert not old['validation_during_training'] and not old['resume'] and not old['train_encoder']


@pytest.mark.parametrize('arm',training.ARMS)
def test_actual_cli_parse_fixed_scope_without_execution(arm):
    args=training.parse_args(['--output','unused','--contact-estimator',arm,'--run-kind','smoke'])
    assert args.contact_estimator==arm and args.coord_objective=='marginal_path' and args.run_kind=='smoke'
    assert args.device=='cuda:0' and args.torch_threads==4


@pytest.mark.parametrize('extra',[['--steps','1'],['--resume'],['--best'],['--val-every','1'],['--contact-estimator','other']])
def test_cli_rejects_unregistered_knobs(extra):
    with pytest.raises(SystemExit):training.parse_args(['--output','unused','--contact-estimator','plugin',*extra])


def test_real_synthetic_plugin_backward_and_adamw_match_frozen_training():
    from test_contact_corrected_path_objective_v1 import fixture
    left,batch,args=fixture(3)
    right=copy.deepcopy(left)
    ga,gb=[torch.Generator().manual_seed(42) for _ in range(2)]
    old_terms,old_norm=ordered.loss_backward(left,batch,3,args,ga)
    new_terms,new_norm=training.loss_backward(right,batch,3,args,gb)
    assert torch.equal(old_terms['loss'],new_terms['loss']) and torch.equal(old_norm,new_norm)
    for p,q in zip(left.kernel.parameters(),right.kernel.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.equal(p.grad,q.grad)
    a=torch.optim.AdamW(left.kernel.parameters(),lr=1e-4,weight_decay=.01)
    b=torch.optim.AdamW(right.kernel.parameters(),lr=1e-4,weight_decay=.01)
    a.step();b.step()
    assert all(torch.equal(p,q) for p,q in zip(left.kernel.parameters(),right.kernel.parameters()))
    assert torch.equal(ga.get_state(),gb.get_state())
    assert all(p.grad is None and not p.requires_grad for p in right.bundle.parameters())


@dataclass
class FakeConfig:
    latent_dim:int=1


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel=nn.Module()
        self.kernel.values=nn.ParameterList([nn.Parameter(torch.tensor([.1+i*.01])) for i in range(73)])
        self.kernel.cfg=FakeConfig()
        self.bundle=nn.Linear(1,1).requires_grad_(False)
        self.component_source={'checkpoint_kind':'synthetic_component'}

    def component_checkpoint(self,source):
        return dict(source,weights=copy.deepcopy(self.bundle.state_dict()))


def fake_losses(model,batch,horizon,args,generator):
    torch.rand(1,generator=generator)
    plugin=sum(p.square().sum() for p in list(model.kernel.parameters())[:64])
    variance=.001*model.kernel.values[0].square().sum()
    return dict(loss_plugin=plugin,contact_variance=variance,
                loss=plugin if args.contact_estimator=='plugin' else plugin-20*variance)


@pytest.mark.parametrize('arm',training.ARMS)
def test_full500_optimizer_has73_params64_active_and_preserves_unused_none(tmp_path,monkeypatch,arm):
    model=FakeModel();initial=copy.deepcopy(model.state_dict())
    args=training.arguments(tmp_path,arm,device='cpu')
    calls=[]
    def loader(args,manifest,horizon,train):
        assert manifest==args.manifest and train is True
        return horizon
    def batch(horizon,step,seed):
        calls.append((step,horizon,seed));return {},dict(step=step,horizon=horizon,synthetic=True)
    monkeypatch.setattr(training,'make_loader',loader)
    monkeypatch.setattr(training,'load_training_batch',batch)
    monkeypatch.setattr(training,'contact_corrected_path_losses',fake_losses)
    result=training.train(model,args,tmp_path)
    assert result['optimizer_steps']==500 and result['kernel_parameter_tensors']==73
    assert result['active_optimizer_states']==64
    assert calls==[(s,[1,4,16,64][(s-1)%4],42) for s in range(1,501)]
    for i,p in enumerate(model.kernel.values):
        if i>=64:
            assert p.grad is None and torch.equal(p,initial[f'kernel.values.{i}'])
    assert all(torch.equal(v,initial[f'bundle.{k}']) for k,v in model.bundle.state_dict().items())
    checkpoint=torch.load(tmp_path/'checkpoint_latest.pt',weights_only=False)
    assert checkpoint['checkpoint_kind']=='lasp_ca_dynamics' and checkpoint['training_version']==training.VERSION
    assert checkpoint['args']['contact_estimator']==arm and checkpoint['step']==500
    assert checkpoint['validation'] is checkpoint['target_encoder'] is None
    assert checkpoint['selection']=='latest exactly500; no validation or best selection'
    assert len(checkpoint['optimizer']['param_groups'][0]['params'])==73
    assert len(checkpoint['optimizer']['state'])==64
    assert all(int(x['step'])==500 for x in checkpoint['optimizer']['state'].values())
    rows=[json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert len(rows)==500 and rows[0]['lr']==pytest.approx(2e-6) and all(r['lr']>0 for r in rows)
    assert all(r['gradient_parameters']==64 and r['kernel_parameters']==73 for r in rows)
    assert [r['lr'] for r in rows]==[1e-4*training._lr_scale(s-1,500,50) for s in range(1,501)]
    assert not list(tmp_path.glob('*best*')) and not list(tmp_path.glob('*val*'))
    with pytest.raises(FileExistsError):training.train(model,args,tmp_path)


def synthetic_smoke_environment(monkeypatch,model):
    monkeypatch.setattr(training,'make_loader',lambda *a:None)
    monkeypatch.setattr(training,'load_training_batch',lambda *a:({'coords':torch.zeros(1,68,96,4,3)},dict(step=4,horizon=64)))
    monkeypatch.setattr(training,'_move_batch',lambda batch,*a:batch)
    monkeypatch.setattr(training,'contact_corrected_path_losses',fake_losses)
    generator=torch.Generator
    monkeypatch.setattr(training.torch,'Generator',lambda **kw:generator(device='cpu'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    for name in ('synchronize','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'synthetic GPU')
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer instantiated during smoke'))


def test_smoke_both_arms_zero_updates_same_rng_and_unchanged_weights(tmp_path,monkeypatch):
    model=FakeModel();synthetic_smoke_environment(monkeypatch,model)
    args=training.arguments(tmp_path,'plugin',run_kind='smoke')
    report=training.smoke(model,args,tmp_path)
    assert report['optimizer_steps']==0 and report['selected_training_step']==4
    assert report['horizon']==64 and report['n_residues']==96 and report['n_samples']==12
    assert report['model_before_sha256']==report['model_after_sha256']
    assert set(report['arms'])==set(training.ARMS)
    a,b=[report['arms'][arm] for arm in training.ARMS]
    assert a['sampling_rng_sha256']==b['sampling_rng_sha256']
    assert a['gradient_parameters']==b['gradient_parameters']==64
    assert a['kernel_parameter_tensors']==b['kernel_parameter_tensors']==73
    assert all(p.grad is None for p in model.kernel.values[64:])


def test_smoke_rejects_estimator_rng_drift(tmp_path,monkeypatch):
    model=FakeModel();synthetic_smoke_environment(monkeypatch,model)
    def changed(model,batch,horizon,args,generator):
        result=fake_losses(model,batch,horizon,args,generator)
        if args.contact_estimator=='population_u':torch.rand(1,generator=generator)
        return result
    monkeypatch.setattr(training,'contact_corrected_path_losses',changed)
    with pytest.raises(ValueError,match='sampling RNG'):
        training.smoke(model,training.arguments(tmp_path,'plugin',run_kind='smoke'),tmp_path)
    assert not (tmp_path/'smoke.json').exists()


def test_actual_entry_failure_records_and_exclusive_retry(tmp_path,monkeypatch):
    out=tmp_path/'new';calls=[]
    monkeypatch.setattr(training,'_set_reproducibility',lambda *a:None)
    def fail(args):calls.append(args);raise RuntimeError('synthetic model failure')
    monkeypatch.setattr(training,'build_model',fail)
    argv=['--output',str(out),'--contact-estimator','plugin','--device','cpu']
    with pytest.raises(RuntimeError,match='synthetic'):training.main(argv)
    failed=json.loads((out/'execution.json').read_text())
    assert failed['status']=='failed' and failed['exit_code']==1 and len(calls)==1
    assert failed['error']=='RuntimeError: synthetic model failure'
    before=(out/'execution.json').read_bytes()
    with pytest.raises(FileExistsError):training.main(argv)
    assert (out/'execution.json').read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('bad',['bundle_trainable','nonfinite_loss','nonfinite_gradient'])
def test_bad_losses_and_component_fail_without_step(bad,monkeypatch):
    model=FakeModel();args=training.arguments('unused','plugin',device='cpu')
    monkeypatch.setattr(training,'contact_corrected_path_losses',fake_losses)
    if bad=='bundle_trainable':model.bundle.requires_grad_(True)
    elif bad=='nonfinite_loss':
        monkeypatch.setattr(training,'contact_corrected_path_losses',lambda *a:dict(loss=torch.tensor(float('nan'))))
    else:
        model.kernel.values[0].register_hook(lambda g:g*float('nan'))
    with pytest.raises((ValueError,FloatingPointError)):
        training.loss_backward(model,{},1,args,torch.Generator().manual_seed(42))
