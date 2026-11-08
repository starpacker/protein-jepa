"""Random-initialized synthetic models/batches only; no checkpoint/data/GPU IO."""
import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1
from protein_jepa_frame.models.stage_a_v2 import StageAV2Config
from protein_jepa_frame.training import stage_a_relpos_adaptation_v1 as core


def config():
    return StageAV2Config(latent_dim=16,encoder_depth=1,num_heads=4,num_rbf=5,dropout=.1,
        max_residues=96,predictor_depth=1,predictor_spatial_depth=1,predictor_heads=4,max_frames=8,
        struct_recon_dim=24,struct_recon_pairs=6,ema_warmup_steps=1000)


def model(mode='pca'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(36);value=StageARelposV1(config(),feature_mode=mode)
    return value.train()


def batch():
    large=core.synthetic_smoke_batch()
    for k,v in large.items():
        if k=='context_coords':large[k]=v[:2,:3,:8].clone()
        elif k=='future_coords':large[k]=v[:2,:,:8].clone()
        elif k=='context_residue_mask':large[k]=v[:2,:3,:8].clone()
        elif k=='future_residue_mask':large[k]=v[:2,:,:8].clone()
        elif k in ('context_aatype','context_residue_index','context_chain_id'):large[k]=v[:2,:8].clone()
        elif k=='context_time_ps':large[k]=v[:2,:3].clone()
        else:large[k]=v[:2].clone()
    return large


def source_backward(m,b):
    t=m.forward_temporal_multilag(b,(0,),recent_frac=.5);t['context_coords']=b['context_coords']
    s=m.forward_spatial(core.build_clip_batch_from_pair(b,torch.device('cpu')))
    c=core.compute_stage_a_v2_loss(t,s,m.opf.analysis_basis(),core.loss_weights(),dH=m.config.latent_dim)
    r=m.struct_recon_loss(t['context_H_full'],b['context_coords'],b['context_residue_mask'],n_pairs=m.config.struct_recon_pairs)
    d=m.displacement_readout_loss(pred_factors=t['pred_factors'],target_delta_local=t['target_delta_local'],
        residue_mask=t['valid_mask'],synth_delta_H=t['synth_delta_H'],direction_weight=1.)
    total=c.total+.3*r+1.*d;total.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    return total.detach(),norm.detach(),s['spatial_mask'].detach(),t['lag_slot'].detach()


def test_all500_schedule_exact400_five_passes_and_balanced_batches():
    schedule=[core.fixed_batch_indices(i) for i in range(500)]
    assert [j for row in schedule for j in row]==list(range(400))*5
    assert all([j%4 for j in row]==[0,1,2,3] for row in schedule)
    for invalid in (-1,500,True,1.):
        with pytest.raises(ValueError):core.fixed_batch_indices(invalid)


def test_pinned_weights_optimizer_inventory_and_schedule():
    w=asdict(core.loss_weights());assert w==dict(future_weight=1.,masked_weight=.5,orth_weight=.01,
        factor_activity_weight=.01,encoder_variance_weight=.01,temporal_variance_weight=.05,
        temporal_min_std=.1,temporal_contrast_weight=2.,temporal_contrast_rmsd_scale=2.,
        factor_min_std=.1,encoder_min_std=.1,lag_balanced=False,eps=1e-6)
    m=model();opt=core.fresh_optimizer(m)
    assert opt.param_groups[0]['lr']==3e-4*.3 and opt.param_groups[1]['lr']==3e-4
    assert all(g['weight_decay']==.02 for g in opt.param_groups) and not opt.state
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in m.parameters() if p.requires_grad}
    assert core._lr_scale(0,500,100)==.01 and core._lr_scale(99,500,100)==1.
    assert core._lr_scale(100,500,100)==1. and core._lr_scale(499,500,100)>0.


@pytest.mark.parametrize('mode',core.MODES)
def test_observer_preserves_full_loss_gradient_and_rng_bitwise(mode):
    m=model(mode);old=copy.deepcopy(m);b=batch();rng=core.rng_snapshot('cpu')
    expected,norm,mask,lag=source_backward(old,b);after=core.rng_snapshot('cpu')
    core.restore_rng(rng);record=core.one_step(m,b)
    assert record['losses']['loss/total']==float(expected)
    assert record['global_grad_norm_before_clip']==float(norm)
    assert core.rng_snapshot('cpu')['sha256']==after['sha256']
    assert torch.equal(record['tensor_audit']['draws']['spatial_mask'],mask)
    assert torch.equal(record['tensor_audit']['draws']['lag_slot'],lag)
    for (name,p),(oldname,q) in zip(m.named_parameters(),old.named_parameters()):
        assert name==oldname and ((p.grad is None)==(q.grad is None))
        if p.grad is not None:assert torch.equal(p.grad,q.grad),name
    draws=record['tensor_audit']['draws']['structural'];rnd=[x['value'] for x in draws['random']]
    sign=torch.where(rnd[0]<.5,1,-1)
    assert torch.equal(draws['pair_i'],torch.cat([rnd[1],rnd[3],rnd[5]],1))
    expected_j=torch.cat([(rnd[i]+rnd[i+1]*sign).clamp(0,7) for i in (1,3,5)],1)
    assert torch.equal(draws['pair_j'],expected_j)
    assert record['ema_before']==record['ema_after']==0 and not record['optimizer_updated']
    def check(value):
        if torch.is_tensor(value):assert value.device.type=='cpu' and not value.requires_grad and value.grad_fn is None
        elif isinstance(value,dict):
            for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
    check(record)


def test_update_order_single_ema_all_buffers_and_adam_counts(monkeypatch):
    m=model();opt=core.fresh_optimizer(m);events=[];b=batch()
    original_step=opt.step;original_after=m.after_optimizer_step;original_ema=m.update_ema
    def step(*args,**kwargs):events.append(('adam',int(m._ema_step)));return original_step(*args,**kwargs)
    def after():events.append(('opf',int(m._ema_step)));return original_after()
    def ema():events.append(('ema',int(m._ema_step)));return original_ema()
    monkeypatch.setattr(opt,'step',step);monkeypatch.setattr(m,'after_optimizer_step',after);monkeypatch.setattr(m,'update_ema',ema)
    oldteacher={n:p.detach().clone() for n,p in m.teacher_encoder.named_parameters()}
    record=core.one_step(m,b,optimizer=opt,step=0)
    assert events==[('adam',0),('opf',0),('ema',0)]
    assert record['ema_momentum']==.998 and record['ema_after']==1
    assert record['lr']==[3e-4*.3*.01,3e-4*.01]
    assert all(p.grad is None for p in m.parameters())
    assert all(int(v['step'])==1 for v in opt.state.values())
    for n,p in m.context_encoder.named_parameters():
        # Production uses alpha=1-.998, retain its exact binary value.
        expected=oldteacher[n].mul(.998).add(p.detach(),alpha=1.-.998)
        assert torch.equal(dict(m.teacher_encoder.named_parameters())[n],expected)
    assert all(not q.training for q in m.teacher_encoder.modules())


def test_structural_observer_scope_removed_on_exception():
    before=core.rng_snapshot('cpu')
    capture=core.StructuralDrawCapture()
    with pytest.raises(RuntimeError,match='injected'):
        with capture:
            torch.rand(2);raise RuntimeError('injected')
    assert len(capture.random)==1
    torch.rand(3)
    assert len(capture.random)==1
    core.restore_rng(before)


def test_ranking_audit_keeps_exact_source_margin_and_each_history_time():
    b=batch();valid=b['context_residue_mask'][:,0]
    result=core.ranking_support(b['context_coords'],valid)
    expected=torch.stack([core._kabsch_rmsd(b['context_coords'][:,k,:,1],b['context_coords'][:,k+1,:,1],valid) for k in range(2)])
    assert torch.equal(result['rmsd'],expected)
    assert torch.equal(result['eligible'],expected[:,:,None]-expected[:,None,:]>.5)
    assert result['regression_count']==4 and result['eligible_counts'].shape==(2,)


def test_nonfinite_loss_stops_before_backward_and_optimizer(monkeypatch):
    m=model();opt=core.fresh_optimizer(m);old=m.displacement_readout_loss
    monkeypatch.setattr(m,'displacement_readout_loss',lambda **kw:old(**kw)*float('nan'))
    with pytest.raises(ValueError,match='nonfinite Stage A loss'):core.one_step(m,batch(),optimizer=opt)
    assert not opt.state and int(m._ema_step)==0 and all(p.grad is None for p in m.parameters())


def test_nonfinite_backward_fails_without_optimizer_update():
    m=model();opt=core.fresh_optimizer(m)
    handle=next(m.context_encoder.parameters()).register_hook(lambda g:g*float('nan'))
    try:
        with pytest.raises(ValueError,match='nonfinite/empty gradients'):core.one_step(m,batch(),optimizer=opt)
    finally:handle.remove()
    assert not opt.state and int(m._ema_step)==0


def test_backward_rng_drift_fails():
    m=model()
    def drift(g):torch.rand(1);return g
    handle=next(m.context_encoder.parameters()).register_hook(drift)
    try:
        with pytest.raises(ValueError,match='backward consumed RNG'):core.one_step(m,batch())
    finally:handle.remove()


def test_synthetic_worst_shape_all_arms_no_updates_input_rng_or_state_mutation():
    before=core.rng_snapshot('cpu');models=[]
    def factory(mode):
        m=model(mode);models.append(m);return m
    batch_before=core.fingerprint(core.synthetic_smoke_batch())
    results=core.smoke_all_arms(factory)
    assert list(results)==list(core.MODES) and core.rng_snapshot('cpu')['sha256']==before['sha256']
    assert core.fingerprint(core.synthetic_smoke_batch())==batch_before
    for record,m in zip(results.values(),models):
        assert record['tensor_audit']['draws']['spatial_mask'].shape==(4,96)
        assert record['ranking_regression_count']==28 and len(record['ranking_eligible_counts'])==7
        assert record['model_initial_sha256']==record['model_final_sha256']
        assert record['ema_after']==0 and all(p.grad is None for p in m.parameters())


def test_registered_runner_rejects_cpu_before_loading_batch_or_emitting():
    called=[]
    with pytest.raises(ValueError,match='CUDA:0'):
        core.run_fixed_training(model(),lambda indices:called.append(indices),lambda record:called.append(record))
    assert called==[]


@pytest.mark.parametrize('failure_at',['reader','emit','interrupt','partial_update'])
def test_fixed_loop_failure_preserves_detached_cpu_optimizer_and_original_error(monkeypatch,failure_at):
    # Device is metadata only in this loop test; no CUDA API or model is used.
    fake=SimpleNamespace(config=SimpleNamespace(ema_start=.998,ema_warmup_steps=1000,struct_recon_pairs=512),
        _ema_step=torch.tensor(0),parameters=lambda:iter([SimpleNamespace(device=torch.device('cuda:0'))]),
        train=lambda:None)
    p=torch.nn.Parameter(torch.tensor(1.));optimizer=torch.optim.AdamW([p],lr=.01)
    monkeypatch.setattr(core,'fresh_optimizer',lambda _:optimizer)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    visits=[];emitted=[]
    error=KeyboardInterrupt('synthetic stop') if failure_at=='interrupt' else RuntimeError('synthetic stop')
    def reader(indices):
        visits.append(indices)
        if failure_at=='reader' and len(visits)==2:raise error
        return dict(context_coords=torch.zeros(4,8,4,4,3),future_coords=torch.zeros(4,1,4,4,3),
            adaptation_metadata=[dict(cache_row_index=i,split='fit') for i in indices],
            horizon_frames=torch.tensor([1,4,16,64]))
    def step(*args,**kwargs):
        p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
        if failure_at=='partial_update':raise error
        fake._ema_step+=1
        return dict(ema_momentum=.998)
    def emit(record):
        emitted.append(record)
        if failure_at in ('emit','interrupt'):raise error
    monkeypatch.setattr(core,'one_step',step)
    with pytest.raises(BaseException) as caught:core.run_fixed_training(fake,reader,emit)
    assert caught.value is error
    payload=error.adaptation_failure_payload
    assert payload['completed_updates']==(0 if failure_at=='partial_update' else 1)
    assert payload['completed_updates_source']=='model._ema_step'
    def digest(state):
        return core.fingerprint(dict(state={str(k):v for k,v in state['state'].items()},param_groups=state['param_groups']))
    snapshot=payload['optimizer_state_dict'];assert digest(snapshot)==digest(optimizer.state_dict())
    before=digest(snapshot)
    state=next(iter(snapshot['state'].values()))
    assert int(state['step'])==1  # Retains even a step that failed before EMA.
    assert all(v.device.type=='cpu' and v.grad_fn is None and not v.requires_grad for v in state.values())
    live=next(iter(optimizer.state.values()));live['exp_avg'].add_(3)
    optimizer.param_groups[0]['lr']=9.
    assert digest(snapshot)==before
    assert len(visits)==(2 if failure_at=='reader' else 1)
    assert len(emitted)==(0 if failure_at=='partial_update' else 1)
