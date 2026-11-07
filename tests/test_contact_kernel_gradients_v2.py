"""Synthetic graph/provenance tests; never read real training data or checkpoints."""
import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from run_scripts import diagnose_contact_kernel_gradients_v2 as d


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = nn.Module()
        self.kernel.drift = nn.Linear(2,1,bias=False,dtype=torch.float64)
        self.kernel.unused = nn.Parameter(torch.tensor([7.],dtype=torch.float64))
        self.kernel.bath = nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.bundle = nn.Linear(2,2,dtype=torch.float64).requires_grad_(False)
        with torch.no_grad():
            self.kernel.drift.weight.copy_(torch.tensor([[2.,3.]]))
            self.kernel.bath.weight.fill_(4.)


def graph(model, generator=None, checkpointed=False):
    x = torch.tensor([[1.,2.]],dtype=torch.float64)
    if generator is not None:
        x += torch.randn(x.shape,generator=generator,dtype=x.dtype)
    forward = lambda z: model.kernel.bath(model.kernel.drift(z)).sum()
    y = checkpoint(forward,x,use_reentrant=False) if checkpointed else forward(x)
    plugin = y.square() + .1*y
    variance = .002*y.square()
    return dict(loss_plugin=plugin,contact_variance=variance,loss=plugin-20*variance)


@pytest.mark.parametrize('checkpointed',[False,True])
def test_actual_three_gradients_unused_support_identity_and_no_optimizer(checkpointed,monkeypatch):
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer instantiated'))
    model = Tiny()
    before = copy.deepcopy(model.state_dict())
    result = d.collect_gradients(model,graph(model,checkpointed=checkpointed))
    summary = d.summarize_gradients(result)
    assert summary['identity']['passed'] and summary['identity']['max_absolute_error'] < 1e-12
    assert summary['parameters']['plugin']['unused_names'] == ['unused']
    assert summary['parameters']['plugin']['nonzero'] == 2
    assert set(summary['module_groups']) == {'drift','bath','unused'}
    # y=32, dy/d(drift)=[4,8], dy/d(bath)=8.
    torch.testing.assert_close(result['gradients']['plugin']['drift.weight'],torch.tensor([[256.4,512.8]],dtype=torch.float64))
    torch.testing.assert_close(result['gradients']['variance_penalty_20']['bath.weight'],torch.tensor([[20.48]],dtype=torch.float64))
    assert all(torch.equal(before[n],v) for n,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_gradient_statistics_independent_dot_sign_clip_and_zero_null():
    vectors=dict(plugin=torch.tensor([3.,4.]),variance_penalty_20=torch.tensor([1.,-2.]),
                 corrected=torch.tensor([2.,6.]))
    result=d.vector_statistics(vectors)
    assert result['norms']['plugin']==5.
    assert result['penalty_vs_plugin']['dot']==-5.
    assert result['penalty_vs_plugin']['change_right_along_minus_left']==5.
    assert result['penalty_vs_plugin']['left_over_right_norm']==pytest.approx(5**.5/5)
    assert result['unit_clipped_norms']['plugin']==pytest.approx(5/(5+1e-6))
    assert result['unit_clipped_plugin_vs_corrected']['cosine']==pytest.approx(result['plugin_vs_corrected']['cosine'])
    zero=d.vector_statistics(dict.fromkeys(d.TERMS,torch.zeros(3)))
    assert zero['penalty_vs_plugin']['cosine'] is None
    assert zero['penalty_vs_plugin']['left_over_right_norm'] is None
    assert zero['plugin_vs_corrected']['change_right_along_unit_minus_left'] is None
    assert zero['unit_clip_scale']['plugin']==1.
    json.dumps(zero,allow_nan=False)


@pytest.mark.parametrize('bad',['shape','nan','empty'])
def test_gradient_vector_corruption_rejected(bad):
    vectors={k:torch.ones(3) for k in d.TERMS}
    if bad=='shape': vectors['corrected']=torch.ones(4)
    elif bad=='nan': vectors['corrected'][0]=float('nan')
    else: vectors={k:torch.empty(0) for k in d.TERMS}
    with pytest.raises(ValueError): d.vector_statistics(vectors)


def test_saved_tensor_roundtrip_identity_mismatch_and_shape(tmp_path):
    model=Tiny();saved=d.collect_gradients(model,graph(model))
    p=tmp_path/'gradients.pt';torch.save(saved,p)
    loaded=torch.load(p,weights_only=True)
    assert d.summarize_gradients(loaded)==d.summarize_gradients(saved)
    loaded['gradients']['corrected']['bath.weight']+=1.
    assert not d.summarize_gradients(loaded)['identity']['passed']
    loaded['gradients']['corrected']['bath.weight']=torch.ones(2)
    with pytest.raises(ValueError,match='shape/dtype'): d.summarize_gradients(loaded)


def test_successive_sampling_stream_and_backward_rng_preservation():
    model=Tiny();generator=torch.Generator().manual_seed(42)
    reference=torch.Generator().manual_seed(42)
    initial=d.rng_fingerprint(generator)
    previous=initial
    for _ in range(4):
        terms=graph(model,generator,checkpointed=True)
        torch.randn((1,2),generator=reference,dtype=torch.float64)
        forward=d.rng_fingerprint(generator)
        d.collect_gradients(model,terms)
        assert d.rng_fingerprint(generator)==forward
        assert torch.equal(generator.get_state(),reference.get_state())
        assert forward['sampling_sha256']!=previous['sampling_sha256']
        assert forward['global_sha256']==initial['global_sha256']
        previous=forward


@pytest.mark.parametrize('bad',['component_trainable','stale_grad','nonscalar','nonfinite'])
def test_invalid_graph_fails_before_grad(bad):
    model=Tiny();terms=graph(model)
    if bad=='component_trainable': model.bundle.requires_grad_(True)
    elif bad=='stale_grad': model.kernel.unused.grad=torch.ones_like(model.kernel.unused)
    elif bad=='nonscalar': terms['loss_plugin']=terms['loss_plugin'].expand(2)
    else: terms['loss_plugin']=terms['loss_plugin']*float('nan')
    with pytest.raises(ValueError): d.collect_gradients(model,terms)


def test_unregistered_or_mutated_run_stops_before_loader_and_retains_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(d,'prior_failure',lambda:{})
    monkeypatch.setattr(d,'execute',lambda *a:pytest.fail('model/data execution before hash validation'))
    with pytest.raises(ValueError,match='register before run'): d.run()
    d.OUT.mkdir();source=tmp_path/'source.py';source.write_text('new')
    d.write_new(d.OUT/'protocol.json',dict(schema_version=d.VERSION,versioned_repair={},source_hashes={'source.py':'0'*64},input_hashes={}))
    with pytest.raises(ValueError,match='frozen file changed'): d.run()
    failed=json.loads((d.OUT/'execution.json').read_text())
    assert failed['status']=='failed' and failed['exit_code']==1
    assert not (d.OUT/'result.json').exists()
    with pytest.raises(ValueError,match='refusing rerun'): d.run()


def test_registration_metadata_only_and_full_source_pinning(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(d,'prior_failure',lambda:{})
    monkeypatch.setattr(d,'PRIOR_FILES',{})
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('tensor load during registration'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('GPU required at registration'))
    monkeypatch.setattr(d,'SOURCES',('new.py',));(tmp_path/'new.py').write_text('new')
    (tmp_path/'old.py').write_text('old');(tmp_path/'old_snapshot').mkdir()
    (tmp_path/'old_snapshot/old.py').write_text('old')
    inputs={}
    for name in (d.INITIAL,'artifacts/mdcath_manifest_full/train.jsonl',
                 'artifacts/stage_a_disp_relpos/checkpoint_latest.pt',
                 'artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic dependency')
        inputs[name]=d.sha(p)
    monkeypatch.setattr(d,'INITIAL_SHA',inputs[d.INITIAL])
    hdf=tmp_path/'fake.h5';hdf.write_text('metadata only')
    selected=[dict(step=i,horizon=h,record_path=str(hdf)) for i,h in enumerate((1,4,16,64),1)]
    historical=[dict(r,actual_batch_sha256='a'*64) for r in selected]
    p=tmp_path/d.WINDOW_LOG;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r)+'\n' for r in historical))
    monkeypatch.setattr(d,'WINDOW_LOG_SHA',d.sha(p))
    old=dict(training_windows=selected,input_hashes=inputs,source_hashes={'old.py':d.sha('old.py')},
             source_snapshot='old_snapshot',selected_hdf5_stats={str(hdf):d.file_stat(hdf)},
             training_arguments={'marginal_path':{}},initial_kernel_sha256='k',
             initial_component_identity={},initial_kernel_config={})
    p=tmp_path/d.HISTORY;p.parent.mkdir(parents=True,exist_ok=True);d.write_new(p,old)
    monkeypatch.setattr(d,'HISTORY_SHA',d.sha(p))
    correct_log_sha=d.WINDOW_LOG_SHA
    monkeypatch.setattr(d,'WINDOW_LOG_SHA','0'*64)
    with pytest.raises(ValueError,match='training-window log changed'):d.register()
    assert not d.OUT.exists()
    monkeypatch.setattr(d,'WINDOW_LOG_SHA',correct_log_sha)
    d.register()
    protocol=json.loads((d.OUT/'protocol.json').read_text())
    assert protocol['historical_actual_batches']==historical
    assert set(protocol['source_hashes'])=={'old.py','new.py'}
    d.verify_files(protocol)
    with pytest.raises(ValueError,match='existing'): d.register()
    hdf.write_text('changed metadata')
    with pytest.raises(ValueError,match='HDF identity'): d.verify_files(protocol)


def test_exclusive_and_sealed_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path)
    with pytest.raises(ValueError):d.path('test.jsonl')
    with pytest.raises(ValueError):d.path(tmp_path.parent/'outside')
    p=tmp_path/'result.json'
    with pytest.raises(ValueError):d.write_new(p,{'x':float('nan')})
    assert not p.exists()
    d.write_new(p,{'x':1})
    with pytest.raises(FileExistsError):d.write_new(p,{})


def test_production_kernel_inventory_has73_including_history8_and_energy_bias(monkeypatch):
    from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint load in inventory regression'))
    # Exact frozen protocol configuration, constructed from metadata values.
    config=StageB1bConfig(latent_dim=128,bath_dim=8,energy_depth=3,energy_heads=4,energy_hidden=128,
        reference_temperature_k=348.,diffusion_scale=.0001,coupling_scale=.1,c_init_std=.1,
        gamma_scale=2e-5,gamma_floor=2e-6,history_hidden=128,step_ps=1000.,jitter=1e-6,
        max_gamma_step=1.,force_clip=None,tail_init=.01,bath_coupling='velocity',
        max_step_norm=10.,rollout_grad_steps=None)
    with torch.random.fork_rng(devices=[]):
        kernel=StageB1b(config)
    inventory=d.kernel_inventory(kernel)
    names=[row['name'] for row in inventory]
    assert len(inventory)==73 and len(set(names))==73 and all(row['requires_grad'] for row in inventory)
    history=[name for name in names if name.startswith('history_posterior.')]
    assert len(history)==8
    assert 'energy.energy_head.2.bias' in names
    assert next(row for row in inventory if row['name']=='energy.energy_head.2.bias')['shape']==[1]
    assert all(row['dtype']=='torch.float32' for row in inventory)


def test_prior_failure_pin_is_checked_before_execution_and_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    prior=tmp_path/'failed.json'
    d.write_new(prior,dict(status='failed',exit_code=1,error='ValueError: kernel tensor count differs',
                          completed_utc='2026-11-02T05:29:20.042491+00:00'))
    monkeypatch.setattr(d,'PRIOR_EXECUTION','failed.json')
    monkeypatch.setattr(d,'PRIOR_FILES',{'failed.json':d.sha(prior)})
    repair=d.prior_failure()
    assert repair['prior_rollouts']==repair['prior_optimizer_steps']==0
    assert repair['numerical_definitions_changed'] is False and repair['gradient_tolerances_changed'] is False
    d.OUT.mkdir()
    d.write_new(d.OUT/'protocol.json',dict(schema_version=d.VERSION,versioned_repair=repair))
    prior.write_text('{}')
    monkeypatch.setattr(d,'execute',lambda *a:pytest.fail('model executed before prior failure hash guard'))
    with pytest.raises(ValueError,match='prior failure evidence changed'):d.run()
    assert json.loads((d.OUT/'execution.json').read_text())['exit_code']==1
