"""Synthetic five-gradient and provenance audit tests; no actual inputs."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from run_scripts import diagnose_distance_path_kernel_gradients_v1 as d


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.kernel=nn.Module()
        self.kernel.linear=nn.Linear(2,1,bias=False,dtype=torch.float64)
        self.kernel.unused=nn.Parameter(torch.ones(1,dtype=torch.float64))
        self.bundle=nn.Linear(2,2,dtype=torch.float64).requires_grad_(False).eval()
        with torch.no_grad():self.kernel.linear.weight.copy_(torch.tensor([[2.,3.]]))


def terms(model,generator,checkpointed=False,horizon=3):
    x=torch.randn(2,2,generator=generator,dtype=torch.float64)
    call=lambda value:model.kernel.linear(value).square().mean()
    value=checkpoint(call,x,use_reentrant=False) if checkpointed else call(x)
    base=value+.1*value.square();absolute=.2*value
    increment=absolute if horizon==1 else .03*value.square()
    return dict(loss_base=base,distance_absolute_es=absolute,distance_increment_es=increment,
        loss_absolute=base+absolute,loss_increment=base+increment)


@pytest.mark.parametrize('checkpointed',[False,True])
@pytest.mark.parametrize('horizon',[1,3])
def test_five_shared_graph_gradients_identity_unused_no_updates_rng(checkpointed,horizon,monkeypatch):
    model=Tiny();before=copy.deepcopy(model.state_dict());generator=torch.Generator().manual_seed(42)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer created'))
    losses=terms(model,generator,checkpointed,horizon);after=d.rng_state(generator)
    calls=[];original=torch.autograd.grad
    def counted(*args,**kwargs):calls.append(kwargs['retain_graph']);return original(*args,**kwargs)
    monkeypatch.setattr(torch.autograd,'grad',counted)
    saved=d.collect_gradients(model,losses,generator)
    report=d.summarize_gradients(saved,{k:float(v.detach()) for k,v in losses.items()},horizon)
    d.check_audit(report,saved,after)
    assert calls==[True,True,True,True,False]
    assert all(saved['unused'][term]==['unused'] for term in d.TERMS)
    assert all(report['parameters'][term]['total']==2 for term in d.TERMS)
    assert report['h1_identity'] is None if horizon!=1 else all(v['passed'] for v in report['h1_identity'].values())
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    assert all(p.grad is None for p in model.parameters())
    assert saved['memory_after_gradients']['process_peak_rss_bytes']>=saved['memory_before_gradients']['process_peak_rss_bytes']>0


def test_statistics_independent_dots_norms_ratios_zero_null_and_fixed_identity():
    vectors=dict(base=torch.tensor([3.,4.]),absolute_aux=torch.tensor([1.,-2.]),increment_aux=torch.tensor([-1.,2.]),
        absolute_total=torch.tensor([4.,2.]),increment_total=torch.tensor([2.,6.]))
    report=d.vector_statistics(vectors)
    assert report['norms']['base']==5.
    assert report['comparisons']['absolute_aux_vs_base']['dot']==-5.
    assert report['comparisons']['increment_aux_vs_base']['dot']==5.
    assert report['comparisons']['absolute_aux_vs_increment_aux']['cosine']==pytest.approx(-1.)
    assert report['comparisons']['absolute_aux_vs_base']['left_over_right_norm']==pytest.approx(5**.5/5)
    zero=d.vector_statistics({term:torch.zeros(2) for term in d.TERMS})
    assert zero['comparisons']['absolute_aux_vs_base']['cosine'] is None
    assert zero['comparisons']['absolute_aux_vs_base']['left_over_right_norm'] is None
    assert d.ATOL==2e-6 and d.RTOL==5e-4
    assert d.identity(torch.tensor([1e-6]),torch.zeros(1))['passed']
    assert not d.identity(torch.tensor([3e-6]),torch.zeros(1))['passed']
    json.dumps(zero,allow_nan=False)


@pytest.mark.parametrize('horizon',[1,3])
def test_actual_synthetic_augmented_adapter_one_rollout_five_gradients(monkeypatch,horizon):
    from test_physical_distance_augmented_objective_v1 import fixture,geometric_p32
    from protein_jepa_frame.training.physical_distance_augmented_objective_v1 import physical_distance_augmented_path_losses
    model,batch,args=fixture(horizon);geometric_p32(model,monkeypatch);args.checkpoint_decoder=True
    count=[];original=model.kernel.rollout
    def rollout(*a,**kw):count.append(1);return original(*a,**kw)
    monkeypatch.setattr(model.kernel,'rollout',rollout)
    generator=torch.Generator().manual_seed(42)
    value=physical_distance_augmented_path_losses(model,batch,horizon,args,generator);after=d.rng_state(generator)
    saved=d.collect_gradients(model,value,generator)
    report=d.summarize_gradients(saved,{k:float(v.detach()) for k,v in value.items()},horizon)
    d.check_audit(report,saved,after)
    assert count==[1]
    assert report['norms']['absolute_aux']>0 and report['norms']['increment_aux']>0


def test_successive_generator_stream_never_reset_and_checkpoint_backward_preserves_it():
    model=Tiny();generator=torch.Generator().manual_seed(42);reference=torch.Generator().manual_seed(42)
    for horizon in (1,4,16,64):
        value=terms(model,generator,True,horizon);after=d.rng_state(generator)
        torch.randn(2,2,generator=reference,dtype=torch.float64)
        saved=d.collect_gradients(model,value,generator)
        assert torch.equal(generator.get_state(),reference.get_state())
        assert all(d.rng_sha(state)==d.rng_sha(after) for state in saved['rng_after_gradients'].values())


def test_replay_roundtrip_additive_h1_and_rng_failure_guards(tmp_path):
    model=Tiny();generator=torch.Generator().manual_seed(42);value=terms(model,generator,horizon=1)
    after=d.rng_state(generator);saved=d.collect_gradients(model,value,generator)
    losses={k:float(v.detach()) for k,v in value.items()}
    path=tmp_path/'gradients.pt';torch.save(saved,path);loaded=torch.load(path,weights_only=False)
    report=d.summarize_gradients(loaded,losses,1);d.check_audit(report,loaded,after)
    assert report==d.summarize_gradients(saved,losses,1)
    changed=copy.deepcopy(loaded);changed['rng_after_gradients']['increment_aux']['sampling'][0]^=1
    with pytest.raises(ValueError,match='consumed RNG'):d.check_audit(report,changed,after)
    changed=copy.deepcopy(loaded);changed['parameter_gradients_present']=True
    with pytest.raises(ValueError,match='populated'):d.check_audit(report,changed,after)
    changed=copy.deepcopy(loaded);changed['gradients']['absolute_total']['linear.weight']+=1.
    report_bad=d.summarize_gradients(changed,losses,1)
    with pytest.raises(ValueError,match='additive'):d.check_audit(report_bad,changed,after)
    changed=copy.deepcopy(loaded)
    for term in ('increment_aux','increment_total'):changed['gradients'][term]['linear.weight']+=1.
    report_bad=d.summarize_gradients(changed,losses,1)
    with pytest.raises(ValueError,match='h1'):d.check_audit(report_bad,changed,after)


@pytest.mark.parametrize('bad',['trainable_component','stale_grad','nonscalar','nan'])
def test_gradient_preconditions_fail_before_autograd(bad,monkeypatch):
    model=Tiny();generator=torch.Generator().manual_seed(42);value=terms(model,generator)
    if bad=='trainable_component':model.bundle.requires_grad_(True)
    elif bad=='stale_grad':model.kernel.unused.grad=torch.ones_like(model.kernel.unused)
    elif bad=='nonscalar':value['loss_base']=value['loss_base'].expand(2)
    else:value['loss_base']=value['loss_base']*float('nan')
    monkeypatch.setattr(torch.autograd,'grad',lambda *a,**k:pytest.fail('autograd before validation'))
    with pytest.raises(ValueError):d.collect_gradients(model,value,generator)


def test_saved_gradient_shape_unused_and_nonfinite_corruption_rejected():
    model=Tiny();generator=torch.Generator().manual_seed(42);value=terms(model,generator)
    saved=d.collect_gradients(model,value,generator);losses={k:float(v.detach()) for k,v in value.items()}
    wrong=copy.deepcopy(saved);wrong['gradients']['base']['linear.weight']=torch.ones(4)
    with pytest.raises(ValueError,match='shape/dtype'):d.summarize_gradients(wrong,losses,3)
    wrong=copy.deepcopy(saved);wrong['gradients']['base']['unused']+=1
    with pytest.raises(ValueError,match='unused audit zero'):d.summarize_gradients(wrong,losses,3)
    wrong=copy.deepcopy(saved);wrong['gradients']['base']['linear.weight'][0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):d.summarize_gradients(wrong,losses,3)


def test_production_kernel73_tensors669429_elements_without_checkpoint(monkeypatch):
    from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint read'))
    config=StageB1bConfig(latent_dim=128,bath_dim=8,energy_depth=3,energy_heads=4,energy_hidden=128,
        reference_temperature_k=348.,diffusion_scale=.0001,coupling_scale=.1,c_init_std=.1,
        gamma_scale=2e-5,gamma_floor=2e-6,history_hidden=128,step_ps=1000.,jitter=1e-6,
        max_gamma_step=1.,force_clip=None,tail_init=.01,bath_coupling='velocity',max_step_norm=10.,rollout_grad_steps=None)
    with torch.random.fork_rng(devices=[]):kernel=StageB1b(config)
    rows=d.inventory(kernel);names=[r['name'] for r in rows]
    assert len(rows)==d.KERNEL_TENSORS==73 and sum(r['numel'] for r in rows)==d.KERNEL_ELEMENTS==669429
    assert all(r['requires_grad'] for r in rows) and len(set(names))==73
    assert len([n for n in names if n.startswith('history_posterior.')])==8 and 'energy.energy_head.2.bias' in names


def test_cpu_guard_before_model_checkpoint_or_cuda_query(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint before guard'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES empty'):d.execute({})


def test_unregistered_changed_source_fail_before_load_preserve_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(d,'execute',lambda *a:pytest.fail('model/data before hash verification'))
    with pytest.raises(ValueError,match='register before'):d.run()
    d.OUT.mkdir();(tmp_path/'source.py').write_text('changed')
    d.write_new(d.OUT/'protocol.json',dict(schema_version=d.VERSION,source_hashes={'source.py':'0'*64},input_hashes={}))
    with pytest.raises(ValueError,match='frozen file changed'):d.run()
    failure=json.loads((d.OUT/'execution.json').read_text());assert failure['exit_code']==1 and failure['status']=='failed'
    with pytest.raises(ValueError,match='refusing rerun'):d.run()


def test_registration_metadata_only_pins_first4_sources_inputs_and_hdf_stat(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('model checkpoint during register'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried'))
    monkeypatch.setattr(d,'SOURCES',('new.py',));(tmp_path/'new.py').write_text('new')
    (tmp_path/'old.py').write_text('old');(tmp_path/'old_snapshot').mkdir();(tmp_path/'old_snapshot/old.py').write_text('old')
    inputs={}
    for name in d.TRAIN_INPUTS:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic bytes');inputs[name]=d.sha(p)
    monkeypatch.setattr(d,'INITIAL_SHA',inputs[d.INITIAL])
    hdf=tmp_path/'fake.h5';hdf.write_text('stat only')
    selected=[dict(step=i,horizon=h,record_path=str(hdf)) for i,h in enumerate((1,4,16,64),1)]
    historical=[dict(r,actual_batch_sha256='a'*64) for r in selected]
    p=tmp_path/d.WINDOW_LOG;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in historical))
    monkeypatch.setattr(d,'WINDOW_LOG_SHA',d.sha(p))
    old=dict(training_windows=selected,input_hashes=inputs,source_hashes={'old.py':d.sha('old.py')},source_snapshot='old_snapshot',
        selected_hdf5_stats={str(hdf):d.file_stat(hdf)},training_arguments={'marginal_path':{}},initial_kernel_sha256='k',
        initial_kernel_config={},initial_component_identity={})
    p=tmp_path/d.HISTORY;p.parent.mkdir(parents=True,exist_ok=True);d.write_new(p,old)
    monkeypatch.setattr(d,'HISTORY_SHA',d.sha(p))
    d.register();protocol=json.loads((d.OUT/'protocol.json').read_text());d.verify_files(protocol)
    assert protocol['historical_actual_batches']==historical and protocol['kernel_parameter_elements']==669429
    assert protocol['contact_estimator']=='plugin' and protocol['distance_path_weight']==1.
    assert protocol['torch_threads']==2 and protocol['device']=='cpu' and protocol['optimizer_steps']==0
    assert protocol['gradient_identity_tolerance']==dict(atol=2e-6,rtol=5e-4)
    with pytest.raises(ValueError,match='existing'):d.register()
    hdf.write_text('changed metadata')
    with pytest.raises(ValueError,match='HDF identity'):d.verify_files(protocol)
