"""Fixed Stage A relpos adaptation arithmetic; caller owns IO/provenance.

No data/checkpoint readers, resume, validation or arm ranking. The fixed runner
requires CUDA:0 FP32 and exactly500 B4 updates. Only one_step permits small CPU
synthetic fixtures. Caller serializes tensor_audit separately from scalar logs.
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from protein_jepa_frame.data.stage_a_relpos_support_v1 import fingerprint
from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1
from protein_jepa_frame.models.stage_a_v2 import StageAV2LossWeights, compute_stage_a_v2_loss
from protein_jepa_frame.models.stage_a_v2.objectives import _kabsch_rmsd
from protein_jepa_frame.training.train_stage_a_v2 import build_optimizer_groups, build_clip_batch_from_pair
from protein_jepa_frame.training.train_utils import _lr_scale
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'stage_a_relpos_adaptation_v1'
MODES = ('pca','zero','local')
STEPS, BATCH_SIZE, WARMUP = 500, 4, 100


def require(ok, message):
    if not ok: raise ValueError(message)


def loss_weights():
    return StageAV2LossWeights(future_weight=1.,masked_weight=.5,orth_weight=.01,
        factor_activity_weight=.01,encoder_variance_weight=.01,temporal_variance_weight=.05,
        temporal_min_std=.1,temporal_contrast_weight=2.,temporal_contrast_rmsd_scale=2.,
        factor_min_std=.1,encoder_min_std=.1,lag_balanced=False,eps=1e-6)


def contract():
    return dict(schema_version=VERSION,steps=500,batch_size=4,fit_rows=400,passes=5,
        lr=3e-4,encoder_lr_scale=.3,weight_decay=.02,warmup_steps=100,grad_clip=1.,
        lags=[0],recent_frac=.5,loss_weights=asdict(loss_weights()),struct_recon_weight=.3,
        displacement_weight=1.,displacement_direction_weight=1.,ema_start=.998,
        ema_warmup_steps=1000,struct_recon_pairs=512,device='cuda:0',dtype='float32',
        amp=False,tf32=False,optimizer_resume=False,validation=False,
        loss_comparison='arm-local teacher losses do not rank representations')


def fixed_batch_indices(step):
    require(type(step) is int and 0 <= step < STEPS, 'step must be zero-based0..499')
    return [(4*step+j)%400 for j in range(4)]


def rng_snapshot(device):
    states = {'cpu':torch.get_rng_state().clone()}
    if torch.device(device).type == 'cuda':
        states['cuda:0'] = torch.cuda.get_rng_state(0).cpu().clone()
    return dict(states=states,sha256=state_sha256(states))


def restore_rng(snapshot):
    torch.set_rng_state(snapshot['states']['cpu'])
    if 'cuda:0' in snapshot['states']:
        torch.cuda.set_rng_state(snapshot['states']['cuda:0'],0)


def model_identity(model):
    # Includes nonpersistent buffers such as encoder RBF centers.
    return state_sha256({**{'parameter/'+n:p for n,p in model.named_parameters()},
                         **{'buffer/'+n:b for n,b in model.named_buffers()}})


def fresh_optimizer(model):
    groups=build_optimizer_groups(model,3e-4,.3,.02)
    optimizer=torch.optim.AdamW(groups,lr=3e-4,weight_decay=.02)
    for group in optimizer.param_groups: group['base_lr']=group['lr']
    require(not optimizer.state, 'fresh optimizer required')
    return optimizer


class StructuralDrawCapture(TorchDispatchMode):
    """Observe real random outputs/index operands; redispatch each op once.

    Thread-local dispatcher scope is removed by its context manager, including
    exceptions. No global function/method replacement and no extra RNG draws.
    """
    def __init__(self):
        super().__init__();self.random=[];self.indices=[]

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        out=func(*args,**(kwargs or {}))
        name=str(func)
        if name.startswith('aten.rand.') or name.startswith('aten.randint.'):
            self.random.append(dict(op=name,value=out.detach().cpu().clone()))
        if name=='aten.index.Tensor':
            indices=args[1]
            require(len(indices)==2 and all(torch.is_tensor(x) for x in indices),
                    'structural indexing schema changed')
            self.indices.append([x.detach().cpu().clone() for x in indices])
        return out

    def audit(self,mask):
        require(len(self.random)==7 and len(self.indices)==6, 'structural draw/index count changed')
        require(self.random[0]['op']=='aten.rand.default'
                and all(x['op'].startswith('aten.randint.') for x in self.random[1:]),
                'structural random operation order changed')
        a,i=self.indices[0];b,j=self.indices[1]
        require(torch.equal(a,b), 'structural frame index differs')
        for offset in (2,4):
            require(all(torch.equal(x,y) for x,y in zip(self.indices[0],self.indices[offset]))
                    and all(torch.equal(x,y) for x,y in zip(self.indices[1],self.indices[offset+1])),
                    'H/CA/mask structural pairs differ')
        mask=mask.detach().cpu().reshape(-1,mask.shape[-1])
        return dict(random=self.random,frame_index=a,pair_i=i,pair_j=j,
                    pair_valid=mask[a,i]&mask[a,j],actual_pairs_per_frame=i.shape[1])


@torch.no_grad()
def ranking_support(coords,valid):
    rmsd=torch.stack([_kabsch_rmsd(coords[:,k,:,1],coords[:,k+1,:,1],valid)
                      for k in range(coords.shape[1]-1)])
    require(bool(torch.isfinite(rmsd).all()), 'nonfinite contrast RMSD audit')
    eligible=rmsd[:,:,None]-rmsd[:,None,:] > .5
    return dict(rmsd=rmsd.cpu(),eligible=eligible.cpu(),
                eligible_counts=eligible.sum((1,2)).cpu(),regression_count=int(rmsd.numel()))


def _precision(model,paired):
    require(isinstance(model,StageARelposV1), 'versioned Stage A model required')
    model._runtime_contract()
    require(model.training, 'online training mode required')
    device=next(model.parameters()).device
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('cuda'), 'AMP forbidden')
    if device.type=='cuda':
        require(device==torch.device('cuda:0') and not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32,'CUDA:0 and disabled TF32 required')
    require(all(p.device==device and p.dtype==torch.float32 for p in model.parameters()),
            'single-device FP32 model required')
    require(all(not b.is_floating_point() or b.dtype==torch.float32 for b in model.buffers()),
            'FP32 floating buffers required')
    for prefix in ('context','future'):
        x=paired[prefix+'_coords'];mask=paired[prefix+'_residue_mask']
        require(x.dtype==torch.float32 and x.device==device and bool(torch.isfinite(x).all())
                and mask.device==device and mask.dtype==torch.bool,'finite native FP32 batch required')
    require(paired['future_coords'].shape[1]==1 and paired['context_coords'].shape[1]>=2,
            'history plus single endpoint required')
    return device


def one_step(model,paired,*,optimizer=None,step=0):
    """Unchanged source-order loss/backward; optional single optimizer update.

    CPU/tiny shapes are supported only here for synthetic tests. No graphs are
    returned. With optimizer=None, parameter gradients remain for inspection;
    smoke clears them afterward. All audit tensors are detached CPU copies.
    """
    fixed_batch_indices(step)
    device=_precision(model,paired)
    require(all(p.grad is None for p in model.parameters()), 'clear gradients required at step entry')
    before_input=fingerprint(paired);ema_before=int(model._ema_step)
    require(optimizer is None or ema_before==step, 'EMA/update counter differs')
    lrs=[]
    if optimizer is not None:
        for group in optimizer.param_groups:
            group['lr']=group['base_lr']*_lr_scale(step,500,100);lrs.append(group['lr'])
    rng={'before':rng_snapshot(device)}
    temporal=model.forward_temporal_multilag(paired,(0,),recent_frac=.5)
    temporal['context_coords']=paired['context_coords']
    spatial=model.forward_spatial(build_clip_batch_from_pair(paired,device))
    components=compute_stage_a_v2_loss(temporal,spatial,model.opf.analysis_basis(),loss_weights(),
                                     dH=model.config.latent_dim)
    with StructuralDrawCapture() as capture:
        recon=model.struct_recon_loss(temporal['context_H_full'],paired['context_coords'],
                                      paired['context_residue_mask'],n_pairs=model.config.struct_recon_pairs)
    displacement=model.displacement_readout_loss(pred_factors=temporal['pred_factors'],
        target_delta_local=temporal['target_delta_local'],residue_mask=temporal['valid_mask'],
        synth_delta_H=temporal['synth_delta_H'],direction_weight=1.)
    total=components.total+.3*recon+1.*displacement
    terms={**components.as_loggable_dict(),'loss/base':float(components.total.detach()),
           'loss/struct_recon':float(recon.detach()),'loss/displacement':float(displacement.detach()),
           'loss/total':float(total.detach())}
    require(all(torch.isfinite(torch.tensor(x,dtype=torch.float64)) for x in terms.values()),
            'nonfinite Stage A loss')
    draws=dict(spatial_mask=spatial['spatial_mask'].detach().cpu().clone(),
               spatial_valid=spatial['valid_mask'].detach().cpu().clone(),
               lag_slot=temporal['lag_slot'].detach().cpu().clone(),
               lag_sel=temporal['lag_sel'].detach().cpu().clone(),
               structural=capture.audit(paired['context_residue_mask']))
    ranking=ranking_support(paired['context_coords'],temporal['valid_mask'])
    rng['after_forward']=rng_snapshot(device)
    total.backward()
    rng['after_backward']=rng_snapshot(device)
    require(rng['after_forward']['sha256']==rng['after_backward']['sha256'], 'backward consumed RNG')
    active={n:p for n,p in model.named_parameters() if p.grad is not None}
    require(active and all(bool(torch.isfinite(p.grad).all()) for p in active.values()), 'nonfinite/empty gradients')
    require(all(p.grad is None for p in model.teacher_encoder.parameters()), 'teacher acquired gradient')
    encoder_grad_norm=float(torch.linalg.vector_norm(torch.stack([
        p.grad.detach().float().norm() for n,p in active.items() if n.startswith('context_encoder.')])) )
    require(encoder_grad_norm>0., 'online encoder gradient is zero')
    # clip_grad_norm_ returns the pre-clipping global norm used by the source.
    grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    momentum=None
    if optimizer is not None:
        optimizer.step();model.after_optimizer_step();optimizer.zero_grad(set_to_none=True)
        momentum=model.update_ema()
        require(int(model._ema_step)==ema_before+1, 'EMA failed single increment')
        require(all(bool(torch.isfinite(p).all()) for p in model.parameters())
                and all(not b.is_floating_point() or bool(torch.isfinite(b).all()) for b in model.buffers()),
                'nonfinite post-update model')
    rng['after_update']=rng_snapshot(device)
    require(rng['after_backward']['sha256']==rng['after_update']['sha256'], 'update consumed RNG')
    require(fingerprint(paired)==before_input, 'training batch mutated')
    record=dict(step=step+1,feature_mode=model.feature_mode,losses=terms,lr=lrs,
        global_grad_norm_before_clip=grad_norm,encoder_grad_norm_before_clip=encoder_grad_norm,
        grad_active_names=list(active),grad_none_names=[n for n,p in model.named_parameters() if n not in active],
        ema_before=ema_before,ema_after=int(model._ema_step),ema_momentum=momentum,
        optimizer_updated=optimizer is not None,input_sha256=before_input,
        draws_sha256=fingerprint(draws),ranking_eligible_counts=ranking['eligible_counts'].tolist(),
        ranking_regression_count=ranking['regression_count'],
        rng_sha256={k:v['sha256'] for k,v in rng.items()},
        tensor_audit=dict(rng=rng,draws=draws,ranking=ranking))
    return record


def run_fixed_training(model,batch_for_indices,emit):
    """Run exactly500 updates; callback returns an already CUDA:0-moved batch.

    Callback/emit own data and exclusive artifact writes. A raised exception
    stops immediately; no skip, replacement, resume or partial success. Returns
    the final optimizer on success. On failure, the original exception carries
    adaptation_failure_payload with the EMA count and a detached CPU optimizer
    state snapshot for the caller's exclusive failure artifacts. EMA count is
    not evidence that a partially failed optimizer/OPF/EMA sequence was atomic.
    """
    require(next(model.parameters()).device==torch.device('cuda:0'), 'fixed run requires CUDA:0')
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'TF32 forbidden')
    require(model.config.ema_start==.998 and model.config.ema_warmup_steps==1000
            and model.config.struct_recon_pairs==512 and int(model._ema_step)==0,
            'original configuration/fresh EMA counter differs')
    model.train();optimizer=fresh_optimizer(model)
    try:
        for step in range(500):
            indices=fixed_batch_indices(step);batch=batch_for_indices(indices)
            require(batch['context_coords'].shape[:2]==(4,8)
                    and batch['context_coords'].shape[2]<=96 and batch['future_coords'].shape[1]==1,
                    'fixed B4K8 prefix shape differs')
            require([x['cache_row_index'] for x in batch['adaptation_metadata']]==indices
                    and all(x['split']=='fit' for x in batch['adaptation_metadata'])
                    and batch['horizon_frames'].tolist()==[1,4,16,64], 'fixed TRAIN400 row/horizon schedule differs')
            record=one_step(model,batch,optimizer=optimizer,step=step)
            require(record['ema_momentum']==.998, 'fixed500 EMA momentum differs')
            record['cache_row_indices']=indices;emit(record)
            del record,batch
    except BaseException as error:
        error.adaptation_failure_payload = dict(completed_updates=int(model._ema_step),
            completed_updates_source='model._ema_step',
            optimizer_state_dict=_cpu_snapshot(optimizer.state_dict()))
        raise
    return optimizer


def _cpu_snapshot(value):
    """Independent recursive optimizer witness; never move live optimizer state."""
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {copy.deepcopy(k):_cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def synthetic_smoke_batch(device='cpu'):
    """Deterministic native-like B4K8L96 geometry; local generator only."""
    g=torch.Generator(device='cpu').manual_seed(90210)
    i=torch.arange(96,dtype=torch.float32)
    ca=torch.stack((3.8*i,2*torch.sin(i/3),2*torch.cos(i/3)),-1)
    offsets=torch.tensor([[-1.2,.8,0.],[0.,0.,0.],[1.5,.1,.1],[2.,1.,.2]])
    x=(ca[:,None,:]+offsets)[None,None].expand(4,9,-1,-1,-1).clone()
    for b in range(4):
        for t in range(9):x[b,t,:,:,1]+=(b+1)*.3*torch.sin(i[:,None]*.2+t*.8)
    x+=.02*torch.randn(x.shape,generator=g)
    aa=(torch.arange(96)%20)[None].expand(4,-1).clone();ri=torch.arange(96)[None].expand(4,-1).clone()
    paired=dict(context_coords=x[:,:8].clone(),future_coords=x[:,8:].clone(),
        context_residue_mask=torch.ones(4,8,96,dtype=torch.bool),future_residue_mask=torch.ones(4,1,96,dtype=torch.bool),
        context_aatype=aa,context_residue_index=ri,context_chain_id=torch.zeros_like(ri),
        context_time_ps=(1000*torch.arange(8,dtype=torch.float32))[None].expand(4,-1).clone(),
        future_time_ps=7000+1000*torch.tensor([[1.],[4.],[16.],[64.]]),
        temperature_k=torch.tensor([320.,348.,413.,450.]))
    return {k:v.to(device) for k,v in paired.items()}


def smoke_all_arms(model_factory):
    """Construct arms sequentially; restore common RNG after each construction.

    Factory(mode) supplies a fresh identically initialized model already on the
    intended device. No optimizer is constructed. Caller measures time/memory
    and must create fresh actual training models after this synthetic smoke.
    """
    results={};common_rng=None;initial_identity=None
    try:
        for mode in MODES:
            model=model_factory(mode);model.train();device=next(model.parameters()).device
            require(model.feature_mode==mode and int(model._ema_step)==0,'fresh matching smoke arm required')
            require(all(p.grad is None for p in model.parameters()),'fresh smoke gradients required')
            if common_rng is None:common_rng=rng_snapshot(device)
            restore_rng(common_rng)
            initial=model_identity(model)
            if initial_identity is None:initial_identity=initial
            require(initial==initial_identity,'smoke arm parameter/buffer initialization differs')
            batch=synthetic_smoke_batch(device)
            try:
                record=one_step(model,batch)
                require(model_identity(model)==initial and int(model._ema_step)==0,'zero-update smoke mutated model')
                model._runtime_contract()
                record['model_initial_sha256']=initial;record['model_final_sha256']=model_identity(model)
                results[mode]=record
            finally:
                model.zero_grad(set_to_none=True)
            del model,batch
        require(len({r['draws_sha256'] for r in results.values()})==1,'smoke arm stochastic draws differ')
        require(len({fingerprint(r['tensor_audit']['rng']) for r in results.values()})==1,
                'smoke arm RNG streams differ')
        return results
    finally:
        if common_rng is not None:restore_rng(common_rng)
