"""Prospective causal one-step H moments on four frozen TRAIN paths.

No trajectories are generated; no decoder, optimizer, HDF or evaluation loader
is used. Native saved true H is the teacher state, never aligned CA readouts.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.utils.contracts import state_sha256
from run_scripts.diagnose_train_path_readout_v1 import tree_sha

ROOT=Path(__file__).resolve().parents[1]
OUTPUT='results/hs_joint_20261101/train_latent_moments_v1'
VERSION='train_latent_moments_v1'
PARENT='results/hs_joint_20261101/train_path_readout_v1'
PARENT_HASHES={PARENT+'/protocol.json':'8f2b3c36a5342f0a6d3fc9736f504d2be3dee98def9a6db0e971a7665abb630d',
    PARENT+'/execution.json':'563d120c14fd512bbd1df75e042e8ad34ecc561a5552ec74caadcb85f7634dab',
    PARENT+'/result.json':'1c4509f82a5aa1ac81271bfca3bef7d813bc0c620ed640e62368d776792295a5'}
CHECKPOINT='artifacts/hs_joint_20261101/ordered_path_v1/marginal_path/checkpoint_latest.pt'
CHECKPOINT_SHA='0f45200e331df6dc50c75cd0aec870dc965948fbbd4a5b430bd5d7ef2147886f'
CONFIG_CHECKPOINT='artifacts/stage_a_disp_relpos/checkpoint_latest.pt'
CONFIG_SHA='b2de4fed497c62cb196d908aa68a3a0d64a59e8c8179b9a91558797384b78689'
SOURCES=('run_scripts/diagnose_train_latent_moments_v1.py','tests/test_train_latent_moments_v1.py',
    'docs/TRAIN_LATENT_MOMENTS_V1_20261102.md',
    'protein_jepa_frame/models/stage_b1b/h_moment_budget_v1.py','tests/test_h_moment_budget_v1.py',
    'protein_jepa_frame/models/stage_c/frozen_h_distance_probe_v2.py',
    'protein_jepa_frame/models/correlated_bath_posterior.py',
    'protein_jepa_frame/training/train_lasp_ca_dynamics.py')
HORIZONS=(1,4,16,64)
DT=1000.
HISTORY_ATOL=1e-5
HISTORY_RTOL=1e-5
MAX_SECONDS=1800
MAX_RSS_BYTES=8*1024**3
MAX_OUTPUT_BYTES=2*1024**3


def require(condition,message):
    if not condition:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()


def local(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    require(p.is_relative_to(ROOT) and p.name!='test.jsonl','outside repository or sealed test input')
    return p


def sha(value):
    h=hashlib.sha256()
    with local(value).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write_json(file,value):
    with local(file).open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_npz(file,values):
    arrays={key:(value.detach().cpu().numpy().copy() if torch.is_tensor(value) else value)
            for key,value in values.items()}
    require(all(isinstance(v,np.ndarray) and not v.dtype.hasobject for v in arrays.values()),'non-array or pickle payload')
    with local(file).open('xb') as f:np.savez(f,**arrays)


def cpu_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','requires CUDA_VISIBLE_DEVICES empty')
    require(not torch.is_autocast_enabled('cpu'),'CPU autocast must be disabled')


def cpu_rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy_metadata=[n[0],*n[2:]],
        numpy_keys=torch.from_numpy(n[1].copy()),torch=torch.get_rng_state().clone())


def imports():
    from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
    from protein_jepa_frame.models.stage_b1b.h_moment_budget_v1 import one_step_h_moments
    from protein_jepa_frame.models.stage_c.frozen_h_distance_probe_v2 import load_frozen_encoder
    from protein_jepa_frame.models.correlated_bath_posterior import correlated_bath_filter_step
    from protein_jepa_frame.training.train_lasp_ca_dynamics import anchor_noise_factor
    return StageB1b,StageB1bConfig,one_step_h_moments,load_frozen_encoder,correlated_bath_filter_step,anchor_noise_factor


def sources():
    imports();result=set(SOURCES)
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT) and p.is_file():result.add(str(p.relative_to(ROOT)))
    return sorted(result)


def parent_metadata():
    for file,digest in PARENT_HASHES.items():require(sha(file)==digest,'parent evidence changed: '+file)
    p=json.loads(local(PARENT+'/protocol.json').read_text())
    r=json.loads(local(PARENT+'/result.json').read_text())
    e=json.loads(local(PARENT+'/execution.json').read_text())
    started=json.loads(local(PARENT+'/started.json').read_text())
    require(e['status']=='completed' and e['exit_code']==0 and e['output_sha256']==PARENT_HASHES[PARENT+'/result.json']
        and e['protocol_sha256']==PARENT_HASHES[PARENT+'/protocol.json']
        and all(e[k]==started[k] for k in ('pid','command','started_utc','protocol_sha256')),'parent execution linkage differs')
    require(r['initial_model_sha256']==r['final_model_sha256'] and r['optimizer_steps']==0,'parent model identity differs')
    require(p['checkpoint']==CHECKPOINT and p['checkpoint_sha256']==CHECKPOINT_SHA,'old marginal500 checkpoint differs')
    a=p['checkpoint_audit'];cfg=a['kernel_config'];args=p['training_arguments']
    require(a['step']==500 and cfg['latent_dim']==128 and cfg['bath_dim']==8 and cfg['step_ps']==DT
        and cfg['bath_coupling']=='velocity' and cfg['force_clip'] is None,'active kernel contract differs')
    require(args['coord_objective']=='marginal_path' and args['s0_mode']=='exact_spatial'
        and args['collective_mobility'] is True and args['history_conditioning'] is True
        and args['noise_kernel']=='gaussian' and args['noise_mix']==.8 and args['noise_length']==8.,'sampler convention differs')
    rows=[]
    require(len(r['rows'])==4,'requires exactly four saved paths')
    for index,row in enumerate(r['rows']):
        require(row['step']==index+1 and row['horizon']==HORIZONS[index]
            and row['batch']==p['historical_actual_batches'][index],'saved TRAIN identity differs')
        require(sha(row['sidecar'])==row['sidecar_sha256'],'saved path bytes differ')
        rows.append({key:row[key] for key in ('step','horizon','batch','sidecar','sidecar_sha256','replay_tensor_sha256')})
    for file,digest in p['source_hashes'].items():
        require(sha(file)==digest and sha(Path(p['source_snapshot'])/file)==digest,'parent source/snapshot differs')
    return p,rows


def contract():
    return dict(schema_version=VERSION,horizons=list(HORIZONS),future_transitions=85,observed_history_frames=4,
        dt_ps=DT,latent_dim=128,bath_dim=8,max_residues=96,device='cpu',torch_threads=2,
        model_dtype='float32',analytic_filter_dtype='float64',amp=False,tf32=False,
        initial_prior='N(0,I) at oldest observed frame; assimilate three observed increments',
        coefficients='D,C,Gamma and full F@F.T fixed at separately encoded original anchor, then valid principal K',
        force='frozen model float32 force evaluated at current native H; positive quadratic tail retained',
        teacher='saved native unaligned true_H; no H projection, normalization, target-dependent mask or CA re-encoding',
        filtration='score and save before assimilating each next teacher H; no reset/reanchoring',
        controls=['posterior','C_zero'],control_meaning='C_zero is local algebraic removal, no separately filtered model or autonomous trajectory',
        metrics=['raw norm-square budget','centered norm-square budget','channel-mean-square closure','tangent increment second moment'],
        history_anchor_check=dict(atol=HISTORY_ATOL,rtol=HISTORY_RTOL,replacement=False,
            saved_single_frame_anchor='bitwise equal; batched history anchor may differ numerically'),
        numerical_policy='no adaptive jitter, covariance repair, force clipping or undefined-tangent support replacement; no float32 bitwise sampler claim',
        reporting='per-transition/per-window only; temporal means within each fixed window; no across-window headline or CI',
        limits='Teacher-forced one-step calibration on four TRAIN windows, not observed-anchor-only forecast skill, Mori closure, thermodynamic FDT, information bound or adoption evidence.',
        resource_limits=dict(wall_seconds=MAX_SECONDS,peak_rss_bytes=MAX_RSS_BYTES,output_bytes=MAX_OUTPUT_BYTES,
            max_full_bath_dimension=768,full_H_covariance_materialized=False),
        new_rollouts=0,optimizer_updates=0,readouts=0,hdf_loaded=False,test_opened=False,validation_loaded=False)


def verify(p):
    for file,digest in p['input_hashes'].items():require(sha(file)==digest,'input changed: '+file)
    for field,snapshot in (('source_hashes','source_snapshot'),('parent_source_hashes','parent_source_snapshot')):
        for file,digest in p[field].items():
            require(sha(file)==digest and sha(Path(p[snapshot])/file)==digest,'source/snapshot changed: '+file)


def register():
    cpu_guard();out=local(OUTPUT);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(pid=os.getpid(),command=sys.argv,started_utc=now())
    try:
        parent,rows=parent_metadata()
        require(sha(CHECKPOINT)==CHECKPOINT_SHA and sha(CONFIG_CHECKPOINT)==CONFIG_SHA,'checkpoint bytes differ')
        inputs=dict(PARENT_HASHES)
        inputs[PARENT+'/started.json']=sha(PARENT+'/started.json')
        inputs.update({CHECKPOINT:CHECKPOINT_SHA,CONFIG_CHECKPOINT:CONFIG_SHA})
        inputs.update({r['sidecar']:r['sidecar_sha256'] for r in rows})
        hashes={file:sha(file) for file in sources()};snapshot=Path(OUTPUT)/'source_snapshot'
        for file,digest in hashes.items():
            target=local(snapshot/file);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(local(file).read_bytes())
            require(sha(target)==digest,'source changed during snapshot')
        p=dict(contract(),registered_utc=now(),input_hashes=inputs,source_hashes=hashes,source_snapshot=str(snapshot),
            parent_source_hashes=parent['source_hashes'],parent_source_snapshot=parent['source_snapshot'],
            selected_paths=rows,checkpoint_audit=parent['checkpoint_audit'],training_arguments=parent['training_arguments'],
            binary_deserialized_at_registration=False)
        verify(p);write_json(out/'protocol.json',p)
        status.update(status='completed',returncode=0,protocol_sha256=sha(out/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-start);write_json(out/'registration.execution.json',status)
    return p


class Budget:
    def __init__(self,out):self.start=time.monotonic();self.out=Path(out)
    def check(self):
        require(time.monotonic()-self.start<=MAX_SECONDS,'wall-clock budget exceeded')
        # Linux ru_maxrss is KiB, unlike macOS. This diagnostic requires Linux.
        require(sys.platform.startswith('linux'),'RSS convention requires Linux')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        require(rss<=MAX_RSS_BYTES,'peak RSS budget exceeded')
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        require(size<=MAX_OUTPUT_BYTES,'output disk budget exceeded')
        return dict(seconds=time.monotonic()-self.start,peak_rss_bytes=rss,output_bytes=size)


def validate_saved(saved,row):
    require(tree_sha(saved)==row['replay_tensor_sha256'],'saved replay tensor identity differs')
    b=saved['batch'];h=row['horizon'];x=b['coords'];rm=b['residue_mask']
    require(x.dtype==torch.float32 and x.device.type=='cpu' and x.ndim==5
        and x.shape[:2]==(1,4+h) and x.shape[-2:]==(4,3) and x.shape[2]<=96,'saved coordinate dimensions differ')
    n=x.shape[2];require(rm.shape==(1,4+h,n) and rm.dtype==torch.bool,'saved mask differs')
    mask=rm[:,:4].all(1)
    require(int(mask.sum())>=4 and torch.equal(mask[0],saved['mask']),'fixed observed support differs')
    require(bool(rm[:,4:][mask[:,None].expand(1,h,n)].all()),'future loses fixed observed support')
    selected=mask[:,None,:,None,None].expand_as(x)
    require(bool(torch.isfinite(x[selected]).all()),'nonfinite coordinate on fixed support')
    if 'atom_mask' in b:
        am=b['atom_mask'];require(am.shape==(1,4+h,n,4) and am.dtype==torch.bool
            and bool(am[mask[:,None,:,None].expand_as(am)].all()),'missing atom on fixed support')
    times=b['time_ps'].double();require(times.shape==(1,4+h) and bool(torch.isfinite(times).all())
        and torch.allclose(times.diff(),torch.full((1,3+h),DT,dtype=torch.float64),atol=1e-6,rtol=0),'physical time grid differs')
    require(torch.equal(saved['absolute_time_ps'].double(),times[0])
        and torch.equal(saved['time_ps'],times[0,3:]-times[0,3]),'saved path time axis differs')
    true=saved['true_H'];require(true.shape==(1,h+1,n,128) and true.dtype==torch.float32
        and true.device.type=='cpu' and bool(torch.isfinite(true[:,:,mask[0]]).all()),'saved true-H shape/finiteness differs')
    for name,key in [('aatype','aatype'),('residue_index','residue_index'),('chain_id','chain_id')]:
        require(torch.equal(b[name][0],saved[key]) and b[name].shape==(1,n) and b[name].dtype==torch.long,'embedding metadata differs')
    sanitized_anchor=torch.where(mask[0,:,None],x[0,3,:,1],torch.zeros_like(x[0,3,:,1]))
    require(torch.equal(saved['anchor'],sanitized_anchor) and b['temperature_k'].numel()==1
        and float(b['temperature_k'][0])==row['batch']['temperature_k'],'anchor/temperature differs')
    safe=dict(b);safe['coords']=torch.where(selected,x,torch.zeros_like(x));safe['residue_mask']=mask[:,None].expand_as(rm)
    return safe,mask,true[0]


def load_models(p):
    Kernel,Config,_,load_encoder,_,_=imports()
    ck=torch.load(local(CHECKPOINT),map_location='cpu',weights_only=False)
    audit=p['checkpoint_audit'];cfg=audit['kernel_config']
    require(ck['step']==500 and ck['kernel_config']==cfg and state_sha256(ck['kernel'])==audit['kernel_tensor_sha256'],
        'checkpoint kernel state/config differs')
    require(ck['args']['coord_objective']=='marginal_path' and len(ck['kernel'])==73,'checkpoint selection/inventory differs')
    with torch.random.fork_rng(devices=[]):kernel=Kernel(Config(**cfg))
    kernel.load_state_dict(ck['kernel'],strict=True);kernel.eval().requires_grad_(False)
    require(state_sha256(kernel)==audit['kernel_tensor_sha256'],'strict kernel tensor mismatch')
    encoder,ea=load_encoder(local(CHECKPOINT),kind='current',expected_sha256=CHECKPOINT_SHA,
        configuration_checkpoint_path=local(CONFIG_CHECKPOINT),expected_configuration_sha256=CONFIG_SHA)
    require(ea['encoder_tensor_sha256']==audit['component_identity']['encoder'],'teacher encoder identity differs')
    return kernel,encoder,ea


@torch.no_grad()
def prepare_window(kernel,encoder,saved,row):
    safe,mask,true=validate_saved(saved,row)
    x=safe['coords'];n=x.shape[2];metadata=[safe[k] for k in ('aatype','residue_index','chain_id')]
    anchor=encoder(x[:,3:4],mask[:,None],*metadata,token_mask=mask[:,None])[:,0]
    require(torch.equal(anchor[0],true[0]),'single-frame teacher anchor differs bitwise from saved H')
    expand=lambda a:a[:,None].expand(1,4,n).reshape(4,n)
    histmask=mask[:,None].expand(1,4,n).reshape(4,n)
    history=encoder(x[:,:4].reshape(4,1,n,4,3),histmask[:,None],*[expand(a) for a in metadata],
        token_mask=histmask[:,None])[:,0].reshape(1,4,n,128)
    difference=(history[0,-1]-anchor[0]).abs()
    require(torch.allclose(history[0,-1,mask[0]],anchor[0,mask[0]],atol=HISTORY_ATOL,rtol=HISTORY_RTOL),
        'history batch versus separate anchor exceeds declared numerical tolerance')
    temperature=safe['temperature_k']
    D,C,G=kernel.bath(anchor,temperature)
    factor=imports()[-1](x[:,3,:,1],mask,.8,8.)
    K=factor@factor.transpose(-1,-2) # FULL product before selecting valid principal matrix.
    indices=mask[0].nonzero().flatten()
    tail=torch.nn.functional.softplus(kernel.energy.log_tail).detach()
    require(tail.shape==(128,) and bool(torch.isfinite(tail).all()) and bool((tail>0).all()),'invalid quadratic tail')
    coefficients=dict(D=D[0,indices].double(),C=C[0,indices].double(),Gamma=G[0,indices].double(),
        K=K[0,indices][:,indices].double(),tail_coefficients=tail.double())
    require(coefficients['C'].shape==(len(indices),128,8),'actual bath shape differs')
    return dict(safe=safe,mask=mask,indices=indices,true=true,history=history[0],temperature=temperature,
        coefficients=coefficients,anchor_H=anchor[0],anchor_batch_max_abs=float(difference[mask[0]].max()))


def force_at(kernel,H,mask,temperature):
    with sdpa_kernel(SDPBackend.MATH):
        f=kernel.compute_forces(H[None],mask,temperature,create_graph=False)[0].detach()
    require(f.dtype==torch.float32 and bool(torch.isfinite(f[mask[0]]).all()),'invalid frozen float32 force')
    return f


def assimilate(mean,factor,delta,force,c):
    filter_step=imports()[4]
    m,l=filter_step(mean[None],factor[None],delta[None],force[None],c['D'][None],c['C'][None],
        c['Gamma'][None],c['K'][None],DT,validate=True)
    return m[0],l[0]


def observed_budgets(H,next_H,tangent_valid):
    delta=next_H-H;z=H-H.mean(-1,keepdim=True);dz=delta-delta.mean(-1,keepdim=True)
    u=z/z.norm(dim=-1).clamp_min(1e-12)[:,None]
    radial=(u*dz).sum(-1);tangent=dz-u*radial[:,None]
    values=dict(raw_norm2_change=next_H.square().sum(-1)-H.square().sum(-1),
        centered_norm2_change=(next_H-next_H.mean(-1,keepdim=True)).square().sum(-1)-z.square().sum(-1),
        channel_mean2_change=next_H.mean(-1).square()-H.mean(-1).square(),
        raw_increment_norm2=delta.square().sum(-1),centered_increment_norm2=dz.square().sum(-1),
        tangent_increment_norm2=torch.where(tangent_valid,tangent.square().sum(-1),torch.full_like(radial,float('nan'))),
        radial_increment_norm2=torch.where(tangent_valid,radial.square(),torch.full_like(radial,float('nan'))))
    summary={k:(float(v[tangent_valid].mean()) if bool(tangent_valid.any()) else None)
        if k.startswith(('tangent_','radial_')) else float(v.mean()) for k,v in values.items()}
    return values,summary


def mode_arrays(moments):
    arrays=dict(tangent_valid=moments['tangent_valid'],valid_indices=moments['valid_indices'])
    for mode in ('posterior','C_zero'):
        for key,value in moments[mode]['per_residue'].items():arrays[mode+'__'+key]=value
        arrays[mode+'__mean_increment']=moments[mode]['mean_increment']
    return arrays


def diagnose_window(kernel,window,row,out,budget):
    """Next H cannot enter predictive moments/posterior until after their save.

    The complete immutable teacher tensor is already loaded and validated; this
    boundary concerns conditional computation, not whether future bytes exist.
    """
    _,_,moment,_,_,_=imports();c=window['coefficients'];indices=window['indices'];n=len(indices)
    sub=out/f"step{row['step']:03d}";sub.mkdir(exist_ok=False)
    save_npz(sub/'fixed.npz',dict(**c,valid_indices=indices,anchor_H=window['anchor_H'],
        observed_history_H=window['history'],fixed_mask=window['mask'][0],temperature=window['temperature']))
    mean=torch.zeros(n,8,dtype=torch.float64);factor=torch.eye(n*8,dtype=torch.float64)
    observed=[]
    for i in range(3):
        budget.check();h=window['history'][i]
        force=force_at(kernel,h,window['mask'],window['temperature'])[indices].double()
        delta=(window['history'][i+1,indices].double()-h[indices].double())
        previous=tree_sha(dict(mean=mean,factor=factor))
        mean,factor=assimilate(mean,factor,delta,force,c)
        history_file=sub/f'observed{i}.npz'
        save_npz(history_file,dict(H=h[indices].double(),next_H=window['history'][i+1,indices].double(),
            force=force,bath_mean_after=mean,bath_factor_after=factor))
        observed.append(dict(index=i,before_sha256=previous,after_sha256=tree_sha(dict(mean=mean,factor=factor)),
            path=str(history_file.relative_to(ROOT)),sha256=sha(history_file)))
    records=[]
    for i in range(row['horizon']):
        budget.check();H=window['true'][i];packed=H[indices].double()
        force=force_at(kernel,H,window['mask'],window['temperature'])[indices].double()
        covariance=factor@factor.T;before=tree_sha(dict(mean=mean,factor=factor))
        moments=moment(packed,force,c['D'],c['C'],c['K'],mean,covariance,dt=DT)
        force_tail=-packed*c['tail_coefficients'];force_residual=force-force_tail
        predictive=sub/f'transition{i:03d}.prediction.npz'
        save_npz(predictive,dict(H=packed,force=force,force_tail=force_tail,force_residual=force_residual,
            bath_mean=mean,bath_covariance=covariance,**mode_arrays(moments)))
        prediction_sha=sha(predictive)
        # No current target appears in moment construction or the predictive file.
        next_H=window['true'][i+1,indices].double()
        values,observed_summary=observed_budgets(packed,next_H,moments['tangent_valid'])
        mean,factor=assimilate(mean,factor,next_H-packed,force,c)
        actual=sub/f'transition{i:03d}.observed.npz'
        save_npz(actual,dict(next_H=next_H,bath_mean_after=mean,bath_factor_after=factor,**values))
        record=dict(index=i,time_ps=i*DT,next_time_ps=(i+1)*DT,
            predictive=dict(path=str(predictive.relative_to(ROOT)),sha256=prediction_sha),
            observed=dict(path=str(actual.relative_to(ROOT)),sha256=sha(actual)),
            moments={m:moments[m]['summary'] for m in ('posterior','C_zero')},observed_budgets=observed_summary,
            moment_audit=moments['audit'],bath_before_sha256=before,bath_after_sha256=tree_sha(dict(mean=mean,factor=factor)),
            prediction_saved_before_target_assimilation=True)
        write_json(sub/f'transition{i:03d}.json',record);records.append(record)
        del moments,covariance
    # Means retain all transitions; undefined tangent fields alone carry null support.
    summaries={}
    for role in ('posterior','C_zero','observed'):
        values=[r['observed_budgets'] if role=='observed' else r['moments'][role] for r in records]
        summaries[role]={key:dict(mean=None if any(v[key] is None for v in values) else float(np.mean([v[key] for v in values])),
            defined_transitions=sum(v[key] is not None for v in values),total_transitions=len(values)) for key in values[0]}
    result=dict(step=row['step'],horizon=row['horizon'],protein=row['batch']['protein'],transitions=len(records),
        fixed_input=dict(path=str((sub/'fixed.npz').relative_to(ROOT)),sha256=sha(sub/'fixed.npz')),
        observed_updates=observed,anchor_batch_max_abs=window['anchor_batch_max_abs'],per_transition=records,
        temporal_means=summaries,resources=budget.check())
    write_json(sub/'result.json',result);return result


def run():
    cpu_guard();out=local(OUTPUT);file=out/'protocol.json'
    require(not any((out/n).exists() for n in ('started.json','execution.json','result.json')),'execution already attempted')
    digest=sha(file);start=time.monotonic();identity=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest)
    write_json(out/'started.json',identity);status=dict(identity);phase='verify';completed=[]
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('registered wall-clock budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        p=json.loads(file.read_text());require(all(p.get(k)==v for k,v in contract().items()),'registered contract differs')
        registration=json.loads((out/'registration.execution.json').read_text())
        require(registration['status']=='completed' and registration['returncode']==0 and registration['protocol_sha256']==digest,'registration incomplete')
        verify(p);budget=Budget(out);budget.check();torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        before_rng=tree_sha(cpu_rng());phase='model_load';kernel,encoder,encoder_audit=load_models(p)
        initial=dict(kernel=state_sha256(kernel),encoder=state_sha256(encoder));phase='windows'
        for row in p['selected_paths']:
            phase=f"step{row['step']:03d}/load";budget.check()
            saved=torch.load(local(row['sidecar']),map_location='cpu',weights_only=False)
            window=prepare_window(kernel,encoder,saved,row)
            phase=f"step{row['step']:03d}/moments";result=diagnose_window(kernel,window,row,out,budget)
            completed.append(result)
            require(initial==dict(kernel=state_sha256(kernel),encoder=state_sha256(encoder))
                and all(v.grad is None for model in (kernel,encoder) for v in model.parameters()),'model state/gradient changed')
            require(tree_sha(cpu_rng())==before_rng,'diagnostic consumed process RNG')
            del saved,window
        require(sum(r['transitions'] for r in completed)==85,'incomplete fixed transition panel')
        verify(p);require(sha(file)==digest,'protocol changed during execution')
        result=dict(schema_version=VERSION,status='completed',protocol_sha256=digest,windows=completed,
            initial_model_hashes=initial,final_model_hashes=dict(kernel=state_sha256(kernel),encoder=state_sha256(encoder)),
            encoder_audit=encoder_audit,cpu_rng_sha256_before=before_rng,cpu_rng_sha256_after=tree_sha(cpu_rng()),
            resources=budget.check(),new_rollouts=0,optimizer_updates=0,readouts=0,hdf_loaded=False,test_opened=False,
            numerical_policy=p['numerical_policy'],scientific_limits=p['limits'])
        write_json(out/'result.json',result);status.update(status='completed',returncode=0,result_sha256=sha(out/'result.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,failed_phase=phase,completed_windows=len(completed),
            error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        status.update(completed_utc=now(),seconds=time.monotonic()-start);write_json(out/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args();value=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=value.get('status','registered')),allow_nan=False),flush=True)


if __name__=='__main__':main()
