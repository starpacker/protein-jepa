"""Offline observed-radius map of saved TRAIN H; no trajectories or dynamics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from run_scripts import diagnose_train_path_readout_v1 as parent

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'results/hs_joint_20261101/saved_h_support_readout_v1'
VERSION = 'saved_h_support_readout_v1'
PARENT_ROOT = 'results/hs_joint_20261101/train_path_readout_v1'
PARENT_FILES = {
    PARENT_ROOT+'/protocol.json':'8f2b3c36a5342f0a6d3fc9736f504d2be3dee98def9a6db0e971a7665abb630d',
    PARENT_ROOT+'/execution.json':'563d120c14fd512bbd1df75e042e8ad34ecc561a5552ec74caadcb85f7634dab',
    PARENT_ROOT+'/result.json':'1c4509f82a5aa1ac81271bfca3bef7d813bc0c620ed640e62368d776792295a5',
}
SOURCES = ('run_scripts/diagnose_saved_h_support_readout_v1.py',
    'tests/test_saved_h_support_readout_v1.py','docs/SAVED_H_SUPPORT_READOUT_DIAGNOSTIC_V1_20261102.md')
ROLES = ('support_raw','support_P32')
RATIONALE = ('Prior fixed32 TRAIN readout-only LayerNorm worsened endpoint UES from 2.2033 to 2.5667 Angstrom '
    'and was not adopted. This fixed old marginal latest500 checkpoint, four saved continuous TRAIN paths and '
    'observed-radius map test temporal/readout geometry despite that negative; no expected benefit, extra arms or adoption.')
require,write_new,now,tree_sha = parent.require,parent.write_new,parent.now,parent.tree_sha


def path(value):
    p=Path(value)
    p=(p if p.is_absolute() else REPO/p).resolve()
    require(p.is_relative_to(REPO) and p.name!='test.jsonl','outside repository or sealed test input')
    return p


def sha(value):
    import hashlib
    digest=hashlib.sha256()
    with path(value).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU diagnostic requires CUDA_VISIBLE_DEVICES empty')


def rng_snapshot():
    """CPU process streams only; no CUDA query and no sampling generator."""
    import random
    import numpy as np
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy_metadata=[state[0],*state[2:]],
                numpy_keys=torch.from_numpy(state[1].copy()),torch=torch.get_rng_state().clone())


def bitwise_equal(left,right):
    return (left.shape==right.shape and left.dtype==right.dtype and left.device==right.device
        and torch.equal(left.detach().contiguous().view(torch.uint8),right.detach().contiguous().view(torch.uint8)))


def verify_files(protocol):
    for name,digest in {**protocol['source_hashes'],**protocol['input_hashes']}.items():
        require(sha(name)==digest,'frozen file changed: '+name)
    for field,files in (('source_snapshot',protocol['source_hashes']),
                        ('metadata_snapshot',protocol['metadata_snapshots'])):
        for name,digest in files.items():
            require(sha(path(protocol[field])/name)==digest,'snapshot changed: '+name)
    # Deliberately no HDF stat/read and no parent.verify_files(), which would stat HDF.


def register():
    require(not OUT.exists(),'refusing existing diagnostic directory')
    for name,digest in PARENT_FILES.items():
        require(sha(name)==digest,'parent metadata changed: '+name)
    old=json.loads(path(PARENT_ROOT+'/protocol.json').read_text())
    execution=json.loads(path(PARENT_ROOT+'/execution.json').read_text())
    result=json.loads(path(PARENT_ROOT+'/result.json').read_text())
    require(execution['status']=='completed' and execution['exit_code']==0
        and execution['protocol_sha256']==PARENT_FILES[PARENT_ROOT+'/protocol.json']
        and execution['output_sha256']==PARENT_FILES[PARENT_ROOT+'/result.json'],'parent execution differs')
    require(old['checkpoint']==parent.CHECKPOINT and old['checkpoint_sha256']==parent.CHECKPOINT_SHA
        and result['initial_model_sha256']==result['final_model_sha256'],'parent frozen model differs')
    rows=result['rows']
    require([r['step'] for r in rows]==[1,2,3,4] and [r['horizon'] for r in rows]==[1,4,16,64],
            'requires exactly four historical windows')
    require([r['batch'] for r in rows]==old['historical_actual_batches'],'parent batch identities differ')
    windows=[{k:r[k] for k in ('step','horizon','batch','sidecar','sidecar_sha256','replay_tensor_sha256')}
             for r in rows]
    metadata={**PARENT_FILES,**old['metadata_snapshots']}
    inputs={**metadata,parent.CHECKPOINT:parent.CHECKPOINT_SHA,
            **{r['sidecar']:r['sidecar_sha256'] for r in windows}}
    for name,digest in inputs.items():
        require(sha(name)==digest,'input identity differs: '+name)
    sources=dict(old['source_hashes'])
    for name,digest in sources.items():
        require(sha(name)==digest and sha(path(old['source_snapshot'])/name)==digest,
                'parent source changed: '+name)
    sources.update({name:sha(name) for name in SOURCES})
    protocol=dict(schema_version=VERSION,registered_utc=now(),source_hashes=sources,input_hashes=inputs,
        source_snapshot=str((OUT/'source_snapshot').relative_to(REPO)),metadata_snapshots=metadata,
        metadata_snapshot=str((OUT/'metadata_snapshot').relative_to(REPO)),windows=windows,
        checkpoint=parent.CHECKPOINT,checkpoint_sha256=parent.CHECKPOINT_SHA,
        checkpoint_audit=old['checkpoint_audit'],parent_protocol=PARENT_ROOT+'/protocol.json',
        parent_result=PARENT_ROOT+'/result.json',roles=list(ROLES),device='cpu',torch_threads=2,
        map='t>0 valid tokens: (H-mean_channels(H))*centered_norm(observed_H0)/centered_norm(H); t0 and masked bytes unchanged',
        map_arithmetic='float64 center/norm/scale, cast to original H dtype; no epsilon, floor or clamp',
        true_H_control='same map with same observed H0; descriptive H-only control, never map fitting or decoder target',
        rationale=RATIONALE,new_rollouts=0,optimizer_steps=0,trajectory_files_opened=False,
        validation_loaded=False,test_opened=False,bioemu2_rerun=False,
        inference='four individual TRAIN cases; no pooled headline/CI; cannot separate mean from radius or establish FDT/kinetics/rollout fix')
    OUT.mkdir(parents=True,exist_ok=False)
    for directory,files in (('source_snapshot',sources),('metadata_snapshot',metadata)):
        for name,digest in files.items():
            target=OUT/directory/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(path(name).read_bytes())
            require(sha(target)==digest,'snapshot source changed during copy')
    verify_files(protocol)
    write_new(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(OUT/'protocol.json'),sha256=sha(OUT/'protocol.json'))),flush=True)


@torch.no_grad()
def support_map(latent,observed_h0,mask):
    require(latent.ndim==4 and latent.shape[1]>=2 and latent.is_floating_point()
        and latent.device.type=='cpu' and observed_h0.shape==latent.shape[2:]
        and observed_h0.dtype==latent.dtype and observed_h0.device==latent.device,
        'H/anchor shape dtype or CPU device differs')
    require(mask.shape==(latent.shape[2],) and mask.dtype==torch.bool and mask.device==latent.device
        and bool(mask.any()),'invalid fixed support')
    require(bitwise_equal(latent[:,0],observed_h0[None].expand_as(latent[:,0])),'H0 differs from observed anchor')
    require(bool(torch.isfinite(latent[...,mask,:]).all()),'nonfinite valid H')
    h0=observed_h0[mask].double()
    h0_centered=h0-h0.mean(-1,keepdim=True)
    radius=h0_centered.norm(dim=-1)
    future=latent[:,1:,mask].double()
    centered=future-future.mean(-1,keepdim=True)
    norms=centered.norm(dim=-1)
    require(bool(torch.isfinite(radius).all() and (radius>0).all()
        and torch.isfinite(norms).all() and (norms>0).all()),'nonfinite or degenerate centered norm')
    mapped=latent.clone()
    mapped[:,1:,mask]=(centered*(radius[None,None,:,None]/norms[...,None])).to(latent.dtype)
    require(torch.isfinite(mapped[...,mask,:]).all(),'nonfinite mapped H')
    require(bitwise_equal(mapped[:,0],latent[:,0]) and bitwise_equal(mapped[...,~mask,:],latent[...,~mask,:]),
            'map changed time zero or masked bytes')
    value=mapped[:,1:,mask].double()
    means=value.mean(-1)
    norm=(value-means[...,None]).norm(dim=-1)
    details=dict(observed_centered_radius=radius,mapped_future_channel_mean=means,
        mapped_future_centered_norm=norm,mapped_future_radius_residual=norm-radius[None,None],
        H_change_norm=(mapped.double()-latent.double()).norm(dim=-1))
    report=dict(t0_bitwise_unchanged=True,masked_tokens_bitwise_unchanged=True,
        future_valid_tokens=means.numel(),max_absolute_channel_mean_residual=float(means.abs().max()),
        max_absolute_centered_radius_residual=float((norm-radius[None,None]).abs().max()),
        H_change_mean=float(details['H_change_norm'][:,1:,mask].mean()),
        H_change_max=float(details['H_change_norm'][:,1:,mask].max()),
        convention='joint centering and observed-radius scaling; floating cast residuals retained')
    return mapped,details,report


def validate_saved(saved,horizon):
    mask=saved['mask'];h=saved['forecast_H'];true=saved['true_H']
    require(mask.ndim==1 and mask.dtype==torch.bool and int(mask.sum())>=4,'saved support differs')
    require(h.shape==(12,horizon+1,len(mask),128) and true.shape==(1,horizon+1,len(mask),128)
        and h.dtype==true.dtype==torch.float32 and h.device.type==true.device.type=='cpu','saved H inventory differs')
    require(tuple(saved['roles'])==parent.ROLES,'saved role inventory differs')
    for name,value in saved['roles'].items():
        members=12 if name.startswith('forecast') else 1
        require(value.shape==(members,horizon+1,len(mask),3) and value.device.type=='cpu'
            and torch.isfinite(value[...,mask,:]).all(),'saved coordinate support differs: '+name)
        require(bitwise_equal(value[:,0],saved['anchor'].to(value.dtype)[None].expand_as(value[:,0])),
                'saved coordinate anchor differs: '+name)
    times=saved['time_ps'];absolute=saved['absolute_time_ps']
    require(times.shape==(horizon+1,) and absolute.shape==(horizon+4,) and torch.isfinite(times).all()
        and torch.equal(times,absolute[3:].double()-absolute[3].double())
        and torch.equal(times.diff(),torch.full_like(times[1:],1000.)),'saved time grid differs')
    require(bitwise_equal(h[:,0],true[:,0].expand_as(h[:,0])),'saved H0 differs')


@torch.no_grad()
def score_new_roles(roles,saved):
    from evaluation.geometry import ca_geometry_summary
    from evaluation.trajectory_metrics_v2 import ordered_path_summary
    require(tuple(roles)==ROLES,'two support roles required')
    mask,ri,ci,times=[saved[k] for k in ('mask','residue_index','chain_id','time_ps')]
    result={}
    for name,value in roles.items():
        metrics=ordered_path_summary(value,saved['reference_unaligned'],saved['anchor'].double(),mask,
            predicted_times_ps=times,reference_times_ps=times,reference_mask=mask[None].expand(len(times),-1),
            residue_index=ri,chain_id=ci,lags=parent.LAGS,max_pairs=parent.MAX_PAIRS)
        increments=value.double().diff(dim=1)[...,mask,:]
        length=increments.norm(dim=-1).mean(-1).sum(-1)
        rmsd=increments.square().sum(-1).mean(-1).sqrt()
        geometry=[[ca_geometry_summary(frame.double(),mask,residue_index=ri,chain_id=ci)
                   for frame in member] for member in value]
        result[name]=dict(n_paths=len(value),coordinate_dtype=str(value.dtype),ordered_metrics=metrics,
            anchor_gauge=dict(path_length_per_member=length.tolist(),increment_rmsd_per_member=rmsd.tolist(),
                increment_rmsd_mean_curve=rmsd.mean(0).tolist(),
                convention='no prediction alignment; t0 included; length=sum time mean residue displacement norm'),
            geometry_per_member_per_time=geometry,
            geometry_support='finite invalid geometry retained; no frame/residue removal; includes t0')
    return result


def displacement(left,right,mask):
    require(left.shape==right.shape and left.ndim==4,'coordinate displacement shape differs')
    value=(left.double()-right.double()).norm(dim=-1)
    require(torch.isfinite(value[...,mask]).all(),'nonfinite coordinate displacement')
    valid=value[...,mask]
    return value,dict(rms_per_member_time=valid.square().mean(-1).sqrt().tolist(),
        mean_norm_per_member_time=valid.mean(-1).tolist(),max_norm_per_member_time=valid.amax(-1).tolist(),
        convention='matched member/time/residue, unchanged anchor gauge, includes t0')


@torch.no_grad()
def diagnose_saved(bundle,saved,horizon):
    validate_saved(saved,horizon)
    require(not bundle.decoder.training and all(not p.requires_grad and p.grad is None
        and p.device.type=='cpu' for p in bundle.decoder.parameters()),'frozen eval CPU decoder required')
    before=rng_snapshot()
    args=[saved[k] for k in ('aatype','residue_index','chain_id','rotation','anchor','mask')]
    raw,p32=parent.decode_latents(bundle,saved['forecast_H'],*args)
    require(bitwise_equal(raw,saved['roles']['forecast_raw']) and
            bitwise_equal(p32,saved['roles']['forecast_P32']),'untreated saved forecast baseline mismatch')
    h0=saved['forecast_H'][0,0]
    mapped,details,map_report=support_map(saved['forecast_H'],h0,saved['mask'])
    true_mapped,true_details,true_report=support_map(saved['true_H'],h0,saved['mask'])
    support_raw,support_p32=parent.decode_latents(bundle,mapped,*args)
    roles=dict(zip(ROLES,(support_raw,support_p32)))
    tensors,reports={},{}
    for name,value in (('baseline_forecast_H',saved['forecast_H']),('support_H',mapped),
                       ('true_H',saved['true_H']),('true_H_mapped_control',true_mapped)):
        tensors[name],reports[name]=parent.latent_descriptives(value,saved['mask'])
    delta,delta_reports={},{}
    for name,left,right in (('baseline_P32_minus_raw',p32,raw),('support_P32_minus_raw',support_p32,support_raw),
        ('support_raw_minus_baseline_raw',support_raw,raw),('support_P32_minus_baseline_P32',support_p32,p32)):
        delta[name],delta_reports[name]=displacement(left,right,saved['mask'])
    scores=score_new_roles(roles,saved)
    after=rng_snapshot()
    require(tree_sha(before)==tree_sha(after),'offline readout consumed RNG')
    replay=dict(schema_version=VERSION,support_H=mapped,true_H_mapped_control=true_mapped,roles=roles,
        baseline_redecoded=dict(forecast_raw=raw,forecast_P32=p32),map_details=details,true_control_details=true_details,
        latent_descriptives=tensors,coordinate_displacement_norm=delta,
        fixed={k:saved[k] for k in ('mask','aatype','residue_index','chain_id','rotation','anchor','time_ps',
                                  'absolute_time_ps','reference_unaligned')},observed_H0=h0,
        rng_before=before,rng_after=after)
    return replay,dict(roles=scores,map=map_report,true_H_control=true_report,latent=reports,
        coordinate_displacement=delta_reports,baseline_raw_bitwise_match=True,baseline_P32_bitwise_match=True,
        rng_before_sha256=tree_sha(before),rng_after_sha256=tree_sha(after))


def load_decoder(checkpoint,audit):
    """Only D is constructed; random initialization is isolated then overwritten."""
    from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
    from protein_jepa_frame.utils.contracts import state_sha256
    require(checkpoint['step']==500 and checkpoint['args']['coord_objective']=='marginal_path',
            'requires old marginal latest500')
    component=checkpoint['component_checkpoint']
    require(component['decoder_mode']=='geometric_ca' and
        component['calibration_args']['bond_projection_steps']==32,'decoder configuration differs')
    state=component['delta_ca_decoder']
    require(state_sha256(state)==audit['component_identity']['decoder'],'checkpoint decoder identity differs')
    before=rng_snapshot()
    with torch.random.fork_rng(devices=[]):
        decoder=GeometricDeltaCADecoder().cpu()
        decoder.load_state_dict(state,strict=True)
    decoder.requires_grad_(False).eval()
    require(state_sha256(decoder)==audit['component_identity']['decoder'],'loaded decoder differs')
    require(tree_sha(before)==tree_sha(rng_snapshot()),'decoder construction changed process RNG')
    return SimpleNamespace(decoder=decoder,projection_steps=32)


def execute(protocol):
    cpu_guard()
    from protein_jepa_frame.utils.contracts import state_sha256
    torch.set_num_threads(2)
    process_before=rng_snapshot()
    checkpoint=torch.load(path(protocol['checkpoint']),map_location='cpu',weights_only=False)
    bundle=load_decoder(checkpoint,protocol['checkpoint_audit'])
    del checkpoint
    initial=state_sha256(bundle.decoder)
    write_new(OUT/'initial.json',dict(decoder_sha256=initial,device='cpu',torch_threads=2,
        process_rng_sha256=tree_sha(process_before),checkpoint_sha256=protocol['checkpoint_sha256'],kernel_constructed=False))
    old_result=json.loads(path(protocol['parent_result']).read_text())
    old_rows={r['step']:r for r in old_result['rows']}
    rows=[]
    for window in protocol['windows']:
        verify_files(protocol)
        step,horizon=window['step'],window['horizon']
        write_new(OUT/f'step{step:03d}.started.json',dict(step=step,horizon=horizon,input=window,started_utc=now()))
        started=time.monotonic()
        saved=torch.load(path(window['sidecar']),map_location='cpu',weights_only=False)
        require(tree_sha(saved)==window['replay_tensor_sha256'],'saved replay tensor identity differs')
        replay,report=diagnose_saved(bundle,saved,horizon)
        replay['input_identity']=window
        file=OUT/f'step{step:03d}.support.pt'
        with file.open('xb') as stream:torch.save(replay,stream)
        row=dict(step=step,horizon=horizon,input_identity=window,**report,
            baseline_roles=old_rows[step]['roles'],seconds=time.monotonic()-started,
            sidecar=str(file.relative_to(REPO)),sidecar_sha256=sha(file),replay_tensor_sha256=tree_sha(replay),
            decoder_sha256=state_sha256(bundle.decoder),new_rollouts=0,optimizer_steps=0)
        write_new(OUT/f'step{step:03d}.json',row)
        require(row['decoder_sha256']==initial and all(p.grad is None for p in bundle.decoder.parameters()),
                'decoder tensor or gradient changed')
        verify_files(protocol)
        rows.append(row)
        print(json.dumps(dict(step=step,horizon=horizon,status='completed',seconds=row['seconds'])),flush=True)
    process_after=rng_snapshot()
    require(tree_sha(process_before)==tree_sha(process_after),'process RNG changed')
    return dict(schema_version=VERSION,completed_utc=now(),rows=rows,rationale=protocol['rationale'],
        inference=protocol['inference'],initial_decoder_sha256=initial,final_decoder_sha256=state_sha256(bundle.decoder),
        process_rng_before_sha256=tree_sha(process_before),process_rng_after_sha256=tree_sha(process_after),
        device='cpu',torch_threads=2,new_rollouts=0,optimizer_steps=0,trajectory_files_opened=False,
        validation_loaded=False,test_opened=False,bioemu2_rerun=False,
        source_hashes=protocol['source_hashes'],input_hashes=protocol['input_hashes'])


def run():
    require((OUT/'protocol.json').exists(),'register before run')
    require(not any((OUT/name).exists() for name in ('started.json','execution.json','result.json')),'refusing rerun')
    started=time.monotonic()
    identity=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=sha(OUT/'protocol.json'))
    write_new(OUT/'started.json',identity)
    try:
        protocol=json.loads((OUT/'protocol.json').read_text())
        require(protocol['schema_version']==VERSION,'protocol version differs')
        verify_files(protocol)
        result=execute(protocol)
        verify_files(protocol)
        require(sha(OUT/'protocol.json')==identity['protocol_sha256'],'protocol changed during run')
        write_new(OUT/'result.json',result)
    except BaseException as error:
        write_new(OUT/'execution.json',dict(identity,status='failed',exit_code=1,seconds=time.monotonic()-started,
            completed_utc=now(),error=f'{type(error).__name__}: {error}'))
        raise
    write_new(OUT/'execution.json',dict(identity,status='completed',exit_code=0,seconds=time.monotonic()-started,
        completed_utc=now(),output_sha256=sha(OUT/'result.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('register','run'),required=True)
    args=parser.parse_args()
    register() if args.mode=='register' else run()
