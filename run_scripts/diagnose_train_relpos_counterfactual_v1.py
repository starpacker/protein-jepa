"""Frozen-encoder internal-branch counterfactual on verified TRAIN NPZ frames.

Two complete checkpoint dictionaries are deserialized; only the encoder is
constructed/executed. All original/sham replays precede pair interventions.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import torch

from run_scripts import diagnose_train_pca_continuity_v1 as common
from protein_jepa_frame.utils.contracts import state_sha256

ROOT=common.ROOT
OUTPUT='results/hs_joint_20261101/train_relpos_counterfactual_v1'
VERSION='train_relpos_counterfactual_v1'
PARENT='results/hs_joint_20261101/train_pca_continuity_v1'
VERIFIER='results/hs_joint_20261101/pca_continuity_verification_v1'
PINNED={PARENT+'/protocol.json':'7074ab4e3583261027f2d8411be2c3d9511d7c072acb31a11bb69cf60bc7ea6a',
    PARENT+'/result.json':'4e8341d120a1d2fdf695dc96b9f14bf7ea5877485484c1963fdf9a16d7655f2f',
    VERIFIER+'/protocol.json':'47d3fb4d5b77f6ad722375714220626798afd7182d7e855edb613bec89d326d1',
    VERIFIER+'/verification.json':'27a57b733a3edd30ebb00c9dc83be344b174ee79cbfaeb6bd1d8627c07f509f9'}
CHECKPOINT=common.lineage.CHECKPOINT
CHECKPOINT_SHA=common.lineage.CHECKPOINT_SHA
CONFIG_CHECKPOINT=common.lineage.CONFIG_CHECKPOINT
CONFIG_SHA=common.lineage.CONFIG_SHA
ENCODER_SHA='741d8fbf712981d8aefd4a4841d75ea201663c247dca92396e69f4bc43bbcc27'
MAX_SECONDS=1200
MAX_RSS_BYTES=8*1024**3
MAX_OUTPUT_BYTES=512*1024**2
SOURCES=('run_scripts/diagnose_train_relpos_counterfactual_v1.py','tests/test_train_relpos_counterfactual_v1.py',
    'docs/TRAIN_RELPOS_COUNTERFACTUAL_V1_20261102.md','protein_jepa_frame/models/stage_c/relpos_counterfactual_v1.py',
    'tests/test_relpos_counterfactual_v1.py','protein_jepa_frame/models/stage_c/frozen_h_distance_probe_v2.py')
require=common.require
local=common.local
sha=common.sha
write=common.write_json
save_npz=common.save_npz
now=common.now
tree_sha=common.lineage.tree_sha
cpu_rng=common.lineage.cpu_rng


def imports():
    from protein_jepa_frame.models.stage_c.relpos_counterfactual_v1 import relpos_input_override,pair_counterfactual_audit
    from protein_jepa_frame.models.stage_c.frozen_h_distance_probe_v2 import load_frozen_encoder
    return relpos_input_override,pair_counterfactual_audit,load_frozen_encoder


def sources():
    imports();return sorted(set(SOURCES)|set(common.sources()))


def contract():
    return dict(schema_version=VERSION,horizons=[1,4,16,64],frames=89,pairs=85,encoder_forwards=437,
        phases=['all89 baseline_pre plus identity_sham','all85 next_P and current_P pairs','all89 baseline_post'],
        prepass_forwards=178,pair_forwards=170,postpass_forwards=89,device='cpu',torch_threads=2,dtype='float32',
        no_grad=True,amp=False,latent_dim=128,max_residues=96,dt_ps=1000.,
        checkpoint_deserializations=2,stage_b_model_constructions=0,decoder_model_constructions=0,
        encoder_state_sha256=ENCODER_SHA,encoder_checkpoint=CHECKPOINT,encoder_checkpoint_sha256=CHECKPOINT_SHA,
        configuration_checkpoint=CONFIG_CHECKPOINT,configuration_checkpoint_sha256=CONFIG_SHA,
        configuration_rule='explicit registered upstream StageA config; strict current component.encoder weights; saved/actual orientation and relpos flags audited; anchor conditioning rejected',
        inputs='verified continuity frame/fixed/transition NPZ only; original Torch path sidecars remain hash lineage, never deserialized',
        metadata='native coordinates and fixed original embedding IDs/mask unchanged; no ID padding rewrite',
        hook='exactly one relpos_proj pre-hook call; captured input bitwise equals saved source FP32 canonical; signed-column gather/sign; identity returns input object; removal in finally',
        baseline='all89 pre and identity shams must pass before any pair; valid H bytes equal saved H, masked output zero; all89 post equals pre',
        pair_control='next-only changes next relpos using fixed geometry P; shared-P changes current with same P and reuses identical next output; retains all P including identity/ties/degeneracy',
        attention='frozen encoder explicit matmul/softmax; no SDPA backend override',
        reporting='all frames/pairs/per-residue arrays; no aggregation, thresholds, correlations, CI, model selection or adoption',
        limits='Controlled internal branch intervention can depart from the learned canonical convention. Shared-P measures label sensitivity, not expected equivariance. Pairwise future-aware mappings need not form a consistent trajectory; no forecast or causal physical claim. Original/next-only tangent uses original H0, shared-P tangent uses changed H0.',
        resource_limits=dict(wall_seconds=MAX_SECONDS,peak_rss_bytes=MAX_RSS_BYTES,output_bytes=MAX_OUTPUT_BYTES),
        new_rollouts=0,optimizer_updates=0,hdf_loaded=False,test_opened=False,validation_loaded=False)


def verify(p):
    for file,digest in p['input_hashes'].items():require(sha(file)==digest,'input changed: '+file)
    for file,digest in p['source_hashes'].items():
        require(sha(file)==digest and sha(Path(p['source_snapshot'])/file)==digest,'source/snapshot changed: '+file)


def evidence():
    for file,digest in PINNED.items():require(sha(file)==digest,'verified continuity evidence changed')
    parent=json.loads(local(PARENT+'/protocol.json').read_text());result=json.loads(local(PARENT+'/result.json').read_text())
    vp=json.loads(local(VERIFIER+'/protocol.json').read_text());verified=json.loads(local(VERIFIER+'/verification.json').read_text())
    require(vp['main_protocol_sha256']==PINNED[PARENT+'/protocol.json'] and verified['status']=='passed'
        and verified['frames']==89 and verified['transitions']==85,'independent continuity verification incomplete')
    require(result['status']=='completed' and result['frames']==89 and result['transitions']==85
        and [w['horizon'] for w in result['windows']]==[1,4,16,64],'continuity panel differs')
    inputs=dict(verified['checked_hashes']);inputs.update(PINNED)
    require(inputs.get(str(local(PARENT+'/result.json')))==PINNED[PARENT+'/result.json'],'result absent from independent hashes')
    for directory,output in ((PARENT,'result'),(VERIFIER,'verification')):
        e=json.loads(local(directory+'/execution.json').read_text());s=json.loads(local(directory+'/started.json').read_text())
        require(e['status']=='completed' and e['returncode']==0 and e['pid']>0 and e.get('completed_utc')
            and all(e[k]==s[k] for k in ('pid','command','started_utc','protocol_sha256'))
            and e['protocol_sha256']==PINNED[directory+'/protocol.json']
            and e[output+'_sha256']==PINNED[directory+'/'+output+'.json'],'execution linkage differs')
        for name in ('execution.json','started.json'):inputs[directory+'/'+name]=sha(directory+'/'+name)
    common.verify(parent)
    inputs.update(parent['input_hashes'])
    for hashes,snapshot in ((parent['source_hashes'],parent['source_snapshot']),
        (parent['parent_source_hashes'],parent['parent_source_snapshot']),(vp['source_hashes'],vp['source_snapshot'])):
        for file,digest in hashes.items():inputs[file]=digest;inputs[str(Path(snapshot)/file)]=digest
    inputs.update({CHECKPOINT:CHECKPOINT_SHA,CONFIG_CHECKPOINT:CONFIG_SHA})
    return result['windows'],inputs


def register():
    common.lineage.cpu_guard();out=local(OUTPUT);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(pid=os.getpid(),command=sys.argv,started_utc=now())
    try:
        windows,inputs=evidence();hashes={file:sha(file) for file in sources()};snapshot=Path(OUTPUT)/'source_snapshot'
        for file,digest in hashes.items():
            target=local(snapshot/file);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(local(file).read_bytes())
            require(sha(target)==digest,'source changed during snapshot')
        p=dict(contract(),registered_utc=now(),input_hashes=inputs,source_hashes=hashes,source_snapshot=str(snapshot),
            selected_windows=windows,binary_deserialized_at_registration=False)
        verify(p);write(out/'protocol.json',p);status.update(status='completed',returncode=0,protocol_sha256=sha(out/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-start);write(out/'registration.execution.json',status)
    return p


def load_npz(reference):
    file=local(reference['path']);require(sha(file)==reference['sha256'],'NPZ input hash differs')
    with np.load(file,allow_pickle=False) as a:
        values={k:torch.from_numpy(a[k].copy()) for k in a.files}
    return values


def load_windows(p):
    windows=[]
    for row in p['selected_windows']:
        fixed=load_npz(row['fixed_input']);mask=fixed['mask'];h=row['horizon'];n=len(mask)
        require(mask.dtype==torch.bool and mask.shape==(n,) and 4<=int(mask.sum())<=n<=96,'invalid fixed support')
        for key in ('aatype','residue_index','chain_id'):require(fixed[key].dtype==torch.int64 and fixed[key].shape==(n,),'legacy IDs differ')
        require(fixed['time_ps'].dtype==torch.float64 and torch.equal(fixed['time_ps'],torch.arange(h+1,dtype=torch.float64)*1000.),'time grid differs')
        require([f['index'] for f in row['frames']]==list(range(h+1)) and [t['index'] for t in row['transitions']]==list(range(h)),'input coverage differs')
        frames=[]
        for r in row['frames']:
            v=load_npz(r);x=v['source_native_backbone'];H=v['saved_true_H'];z=v['canonical']
            require(x.dtype==torch.float32 and x.shape==(n,4,3) and bool(torch.isfinite(x).all())
                and bool((x[~mask]==0).all()),'native sanitized coordinates differ')
            require(H.dtype==torch.float32 and H.shape==(n,128) and bool(torch.isfinite(H[mask]).all()),'saved H differs')
            require(z.dtype==torch.float32 and z.shape==(n,3) and bool(torch.isfinite(z).all())
                and bool((z[~mask]==0).all()) and torch.equal(v['mask'],mask),'source canonical support differs')
            frames.append(dict(coords=x,H_saved=H[None],canonical=z[None],source=r))
        permutations=[load_npz(r)['selected_permutation'] for r in row['transitions']]
        windows.append(dict(source=row,fixed=fixed,frames=frames,permutations=permutations))
    return windows


def model_snapshot(encoder):
    values=dict(parameters={k:v for k,v in encoder.named_parameters()},buffers={k:v for k,v in encoder.named_buffers()})
    require(all(not m.training for m in encoder.modules()) and all(not v.requires_grad and v.grad is None for v in encoder.parameters()),'encoder freeze/eval changed')
    return dict(state_dict_sha256=state_sha256(encoder),all_parameters_and_buffers_sha256=tree_sha(values),
        inventory={kind:[dict(name=k,shape=list(v.shape),dtype=str(v.dtype),device=str(v.device)) for k,v in part.items()] for kind,part in values.items()})


def equal_bytes(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def witness_meta(w):return {k:w[k] for k in ('call_count','is_identity','identity_input_object_preserved','hook_removed')}


def ref(file):return dict(path=str(Path(file).relative_to(ROOT)),sha256=sha(file))


class Budget:
    def __init__(self,out):self.out=Path(out);self.start=time.monotonic()
    def check(self):
        require(sys.platform.startswith('linux'),'Linux RSS units required')
        s=time.monotonic()-self.start;r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        b=sum(f.stat().st_size for f in self.out.rglob('*') if f.is_file())
        require(s<=MAX_SECONDS and r<=MAX_RSS_BYTES and b<=MAX_OUTPUT_BYTES,'resource budget exceeded')
        return dict(seconds=s,peak_rss_bytes=r,output_bytes=b)


@torch.no_grad()
def execute_panel(encoder,windows,out,budget,trace,phase):
    override,audit,_=imports();initial=model_snapshot(encoder);rng=tree_sha(cpu_rng());input_sha=tree_sha(windows)
    keys=[(w['source']['step'],f['source']['index']) for w in windows for f in w['frames']]
    require(len(set(keys))==len(keys),'duplicate step/frame input key')
    write(Path(out)/'model.initial.json',dict(model=initial,cpu_rng_sha256=rng,input_tensor_sha256=input_sha,successful_forwards=0))
    results=[];cache={};phase_audits=[]
    def check_phase(name):
        require(model_snapshot(encoder)==initial,'model parameters/buffers or inventory changed')
        require(tree_sha(cpu_rng())==rng and tree_sha(windows)==input_sha,'RNG or input tensors changed')
        record=dict(phase=name,successful_forwards=len(trace),model=initial,cpu_rng_sha256=rng,input_tensor_sha256=input_sha)
        phase_audits.append(record);return record
    def forward(w,frame,label,pair=None,P=None):
        budget.check();step=w['source']['step'];index=frame['source']['index'];phase[0]=f'step{step:03d}/{label}/frame{index:03d}'
        mask=w['fixed']['mask'];x=frame['coords'][None,None];ids=[w['fixed'][k][None] for k in ('aatype','residue_index','chain_id')]
        if P is None:H=encoder(x,mask[None,None],*ids,token_mask=mask[None,None])[:,0];witness=None
        else:
            with override(encoder,frame['canonical'],P,mask) as witness:
                H=encoder(x,mask[None,None],*ids,token_mask=mask[None,None])[:,0]
            require(witness['hook_removed'] and witness['call_count']==1,'hook lifecycle differs')
        require(H.dtype==torch.float32 and H.shape==frame['H_saved'].shape and bool(torch.isfinite(H).all())
            and bool((H[:,~mask]==0).all()),'invalid encoder output or masked padding')
        trace.append(dict(forward_index=len(trace),phase=label,step=step,frame_index=index,pair_index=pair))
        return H.detach().clone(),witness
    # GLOBAL PHASE 1: all windows before any pair intervention.
    for w in windows:
        row=w['source'];folder=Path(out)/f"step{row['step']:03d}";folder.mkdir(exist_ok=False)
        fixed=folder/'fixed.npz';save_npz(fixed,w['fixed'])
        result=dict(step=row['step'],protein=row['protein'],horizon=row['horizon'],temperature_k=row['temperature_k'],
            fixed=ref(fixed),frames=[],pairs=[])
        for frame in w['frames']:
            i=frame['source']['index'];H,unused=forward(w,frame,'baseline_pre');mask=w['fixed']['mask']
            require(equal_bytes(H[:,mask],frame['H_saved'][:,mask]),'baseline pre differs from saved H valid bits')
            file=folder/f'frame{i:03d}.baseline.npz';save_npz(file,dict(coords=frame['coords'],canonical=frame['canonical'],H_saved=frame['H_saved'],H_baseline=H))
            record=dict(index=i,time_ps=float(w['fixed']['time_ps'][i]),source=frame['source'],baseline=ref(file))
            cache[(row['step'],i)]=H
            sham,sw=forward(w,frame,'identity_sham',P=torch.eye(3,dtype=torch.float64))
            require(equal_bytes(sham,H) and sw['identity_input_object_preserved'],'identity sham differs')
            file=folder/f'frame{i:03d}.sham.npz';save_npz(file,dict(H_sham=sham,hook_input=sw['input'],hook_output=sw['output'],permutation=sw['permutation']))
            record.update(sham=ref(file),sham_witness=witness_meta(sw));result['frames'].append(record)
            write(folder/f'frame{i:03d}.pre.json',record)
        results.append(result)
    require(len(cache)==sum(len(w['frames']) for w in windows) and len(trace)==2*len(cache),'prepass coverage differs')
    pre=check_phase('prepass');write(Path(out)/'calls.prepass.json',trace);write(Path(out)/'prepass_gate.json',dict(status='passed',frames=len(cache),**pre))
    # GLOBAL PHASE 2: shared control reuses the same next-frame output.
    for w,result in zip(windows,results):
        step=w['source']['step'];folder=Path(out)/f'step{step:03d}'
        for t,P in enumerate(w['permutations']):
            f0,f1=w['frames'][t:t+2];H0,H1=cache[(step,t)],cache[(step,t+1)]
            H1P,nw=forward(w,f1,'next_P',t,P);H0P,cw=forward(w,f0,'current_P',t,P)
            if nw['is_identity']:require(equal_bytes(H1P,H1) and equal_bytes(H0P,H0),'identity selected P changed H')
            checked=audit(H0,H1,H1P,H0P,H1P,w['fixed']['mask'])
            file=folder/f'pair{t:03d}.npz';save_npz(file,dict(H0=H0,H1=H1,H1_next_only=H1P,H0_shared=H0P,H1_shared=H1P,
                coords0=f0['coords'],coords1=f1['coords'],canonical0=f0['canonical'],canonical1=f1['canonical'],P=P,
                next_hook_input=nw['input'],next_hook_output=nw['output'],current_hook_input=cw['input'],current_hook_output=cw['output'],**checked['arrays']))
            record=dict(index=t,time_ps=float(w['fixed']['time_ps'][t]),next_time_ps=float(w['fixed']['time_ps'][t+1]),
                source=w['source']['transitions'][t],sidecar=ref(file),audit=checked['summary'],next_witness=witness_meta(nw),current_witness=witness_meta(cw))
            write(folder/f'pair{t:03d}.json',record);result['pairs'].append(record)
    check_phase('pairs');write(Path(out)/'calls.pairs.json',trace)
    # GLOBAL PHASE 3: all original outputs replay after every intervention.
    for w,result in zip(windows,results):
        step=w['source']['step'];folder=Path(out)/f'step{step:03d}'
        for frame,record in zip(w['frames'],result['frames']):
            i=frame['source']['index'];H,_=forward(w,frame,'baseline_post')
            require(equal_bytes(H,cache[(step,i)]),'baseline post differs')
            file=folder/f'frame{i:03d}.post.npz';save_npz(file,dict(H_post=H));record['post']=ref(file)
            write(folder/f'frame{i:03d}.json',record)
        write(folder/'result.json',result)
    check_phase('postpass');write(Path(out)/'calls.final.json',trace)
    return results,phase_audits


def run():
    common.lineage.cpu_guard();out=local(OUTPUT);file=out/'protocol.json';require(file.is_file(),'registration required')
    require(not any((out/n).exists() for n in ('started.json','execution.json','result.json')),'execution already attempted')
    digest=sha(file);start=time.monotonic();status=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest)
    write(out/'started.json',status);phase=['verify'];trace=[];encoder=None;windows=None
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('registered wall-clock budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        p=json.loads(file.read_text());require(all(p.get(k)==v for k,v in contract().items()),'registered contract differs')
        registration=json.loads((out/'registration.execution.json').read_text())
        require(registration['status']=='completed' and registration['returncode']==0 and registration['protocol_sha256']==digest,'registration incomplete')
        verify(p);torch.set_num_threads(2);budget=Budget(out);budget.check();before_rng=tree_sha(cpu_rng())
        phase[0]='verified_NPZ_load';windows=load_windows(p)
        require([w['source']['step'] for w in windows]==[1,2,3,4] and [w['source']['horizon'] for w in windows]==[1,4,16,64]
            and sum(len(w['frames']) for w in windows)==89 and sum(len(w['permutations']) for w in windows)==85,'fixed panel differs')
        phase[0]='encoder_load';_,_,loader=imports()
        encoder,ea=loader(local(CHECKPOINT),kind='current',expected_sha256=CHECKPOINT_SHA,
            configuration_checkpoint_path=local(CONFIG_CHECKPOINT),expected_configuration_sha256=CONFIG_SHA)
        require(state_sha256(encoder)==ENCODER_SHA and ea['actual_flags']['relative_position_features'] is True,'encoder identity/channel differs')
        require(tree_sha(cpu_rng())==before_rng,'encoder loading changed process RNG')
        results,phases=execute_panel(encoder,windows,out,budget,trace,phase)
        require(len(trace)==437 and sum(len(w['frames']) for w in results)==89 and sum(len(w['pairs']) for w in results)==85,'forward/panel count differs')
        verify(p);require(sha(file)==digest,'protocol changed')
        result=dict(schema_version=VERSION,status='completed',protocol_sha256=digest,windows=results,frames=89,pairs=85,
            successful_forwards=len(trace),encoder_audit=ea,phase_audits=phases,cpu_rng_sha256_before=before_rng,
            cpu_rng_sha256_after=tree_sha(cpu_rng()),resources=budget.check(),checkpoint_deserializations=2,
            stage_b_model_constructions=0,decoder_model_constructions=0,new_rollouts=0,optimizer_updates=0,scientific_limits=p['limits'])
        write(out/'result.json',result);status.update(status='completed',returncode=0,result_sha256=sha(out/'result.json'))
    except BaseException as error:
        write(out/'calls.failure.json',trace)
        status.update(status='failed',returncode=1,failed_phase=phase[0],error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        status.update(completed_utc=now(),seconds=time.monotonic()-start,successful_forwards=len(trace),cpu_rng_sha256_final=tree_sha(cpu_rng()))
        if encoder is not None:
            try:status['model_final']=model_snapshot(encoder)
            except BaseException as e:status['model_final_audit_error']=str(e)
        if windows is not None:status['input_tensor_sha256_final']=tree_sha(windows)
        write(out/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args();result=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=result.get('status','registered'))),flush=True)


if __name__=='__main__':main()
