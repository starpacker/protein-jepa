"""Three raw coordinate features on verified TRAIN NPZ geometry only.

Selective NPZ access never materializes H. No checkpoint/model, encoder forward,
new PCA or Kabsch fit; complete fixed support and failure prefixes are retained.
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

ROOT=common.ROOT
OUTPUT='results/hs_joint_20261101/train_local_relpos_continuity_v1'
VERSION='train_local_relpos_continuity_v1'
PARENT='results/hs_joint_20261101/train_pca_continuity_v1'
VERIFIER='results/hs_joint_20261101/pca_continuity_verification_v1'
MOTIVATION='results/hs_joint_20261101/train_relpos_counterfactual_v1'
MOTIVATION_VERIFIER='results/hs_joint_20261101/relpos_counterfactual_verification_v1'
PINNED={PARENT+'/protocol.json':'7074ab4e3583261027f2d8411be2c3d9511d7c072acb31a11bb69cf60bc7ea6a',
    PARENT+'/result.json':'4e8341d120a1d2fdf695dc96b9f14bf7ea5877485484c1963fdf9a16d7655f2f',
    VERIFIER+'/protocol.json':'47d3fb4d5b77f6ad722375714220626798afd7182d7e855edb613bec89d326d1',
    VERIFIER+'/verification.json':'27a57b733a3edd30ebb00c9dc83be344b174ee79cbfaeb6bd1d8627c07f509f9',
    MOTIVATION+'/protocol.json':'43cd6e7e951ce8976bfb30e9639050dc154eb4962ff0881fdc7f32de2984033f',
    MOTIVATION+'/result.json':'f50b9b94c0cd5e59c61f484f385059cd560554277402ac37c2d08fe22dbb7ec0',
    MOTIVATION_VERIFIER+'/protocol.json':'bc42f8df2e72911f0b14158106cd97edfda55786b5cc1cb39470ef7c7dbe9eca',
    MOTIVATION_VERIFIER+'/verification.json':'b65166d4256047dd751923b3b3d5019e74a33a5d28487438b9e6f0f3531cd58c'}
FRAME_KEYS=('source_native_backbone','mask','centered','centroid','canonical','basis')
FIXED_KEYS=('mask','aatype','residue_index','chain_id','time_ps')
TRANSITION_KEYS=('R',)
MAX_SECONDS=600
MAX_RSS_BYTES=4*1024**3
MAX_OUTPUT_BYTES=256*1024**2
SOURCES=('run_scripts/diagnose_train_local_relpos_continuity_v1.py','tests/test_train_local_relpos_continuity_v1.py',
    'docs/TRAIN_LOCAL_RELPOS_CONTINUITY_V1_20261102.md','protein_jepa_frame/models/stage_c/local_relpos_continuity_v1.py',
    'tests/test_local_relpos_continuity_v1.py','protein_jepa_frame/models/encoder.py')
require=common.require
local=common.local
sha=common.sha
write=common.write_json
save_npz=common.save_npz
now=common.now
tree_sha=common.lineage.tree_sha
cpu_rng=common.lineage.cpu_rng


def imports():
    from protein_jepa_frame.models.stage_c.local_relpos_continuity_v1 import frame_audit,pair_audit
    return frame_audit,pair_audit


def sources():
    imports();found=set(SOURCES)
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT) and p.is_file():found.add(str(p.relative_to(ROOT)))
    return sorted(found)


def contract():
    return dict(schema_version=VERSION,horizons=[1,4,16,64],frames=89,transitions=85,dt_ps=1000.,
        branches=['pca','zero','local'],device='cpu',torch_threads=2,source_dtype='float32',analysis_dtype='float64',
        max_residues=96,frame_npz_keys=list(FRAME_KEYS),fixed_npz_keys=list(FIXED_KEYS),transition_npz_keys=list(TRANSITION_KEYS),
        input_rule='select only registered geometry keys; never materialize saved H; all original metadata and fixed valid support unchanged',
        source_frame_rule='direct frozen _local_backbone_frames on sanitized native CPU FP32 B1 backbone; recorded PCA basis/canonical and source centering reused without eigendecomposition',
        local_feature='F_i @ (CA_i-visible_centroid), raw float32 local coordinates; no fitted scaling, normalization of feature amplitudes, learned projection or feature replacement in an encoder',
        analysis='promote recorded FP32 centered CA and source local frames to float64; reuse saved proper row-Kabsch R; retain source FP32 versus promoted-analysis distinction and complete local shape/basis/signed-cross decomposition',
        support='all fixed valid residues retained including finite degenerate local frames; epsilon/angle/orthogonality diagnostics are descriptions, not filters or chemical validity guarantees',
        reporting='every frame/pair array and within-frame/pair residue summary; no pooled statistics, event threshold, correlation, filtering, CI or selection',
        limits='Raw feature continuity and amplitude only. Zero has trivial continuity and no positional information. Local-frame degeneracy and centroid lever arms can amplify changes. No H, downstream attention, readout, dynamics or forecasting comparison; no feature adoption.',
        motivation='completed PCA continuity and relpos branch intervention; latter JSON identities only, not its checkpoint-containing evidence collector',
        resource_limits=dict(wall_seconds=MAX_SECONDS,peak_rss_bytes=MAX_RSS_BYTES,output_bytes=MAX_OUTPUT_BYTES),
        encoder_calls=0,model_checkpoint_loads=0,H_arrays_loaded=0,new_PCA_fits=0,new_Kabsch_fits=0,
        new_rollouts=0,optimizer_updates=0,hdf_loaded=False,test_opened=False,validation_loaded=False)


def verify(p):
    for file,digest in p['input_hashes'].items():require(sha(file)==digest,'input changed: '+file)
    for file,digest in p['source_hashes'].items():
        require(sha(file)==digest and sha(Path(p['source_snapshot'])/file)==digest,'source/snapshot changed: '+file)


def linked_execution(directory,output,inputs):
    s=json.loads(local(directory+'/started.json').read_text());e=json.loads(local(directory+'/execution.json').read_text())
    require(e['status']=='completed' and e['returncode']==0 and e['pid']>0 and e.get('completed_utc')
        and all(s[k]==e[k] for k in ('pid','command','started_utc','protocol_sha256'))
        and e['protocol_sha256']==PINNED[directory+'/protocol.json']
        and e[output+'_sha256']==PINNED[directory+'/'+output+'.json'],'execution linkage differs')
    for name in ('started.json','execution.json'):inputs[directory+'/'+name]=sha(directory+'/'+name)


def evidence():
    for file,digest in PINNED.items():require(sha(file)==digest,'fixed evidence changed')
    parent=json.loads(local(PARENT+'/protocol.json').read_text());result=json.loads(local(PARENT+'/result.json').read_text())
    vp=json.loads(local(VERIFIER+'/protocol.json').read_text());verified=json.loads(local(VERIFIER+'/verification.json').read_text())
    require(vp['main_protocol_sha256']==PINNED[PARENT+'/protocol.json'] and verified['status']=='passed'
        and verified['frames']==89 and verified['transitions']==85,'PCA verification incomplete')
    require(result['status']=='completed' and result['frames']==89 and result['transitions']==85
        and [w['horizon'] for w in result['windows']]==[1,4,16,64],'fixed PCA panel differs')
    inputs=dict(verified['checked_hashes']);inputs.update(PINNED)
    require(inputs.get(str(local(PARENT+'/result.json')))==PINNED[PARENT+'/result.json'],'PCA result missing from independent hashes')
    common.verify(parent);inputs.update(parent['input_hashes'])
    for hashes,snapshot in ((parent['source_hashes'],parent['source_snapshot']),
        (parent['parent_source_hashes'],parent['parent_source_snapshot']),(vp['source_hashes'],vp['source_snapshot'])):
        for file,digest in hashes.items():inputs[file]=digest;inputs[str(Path(snapshot)/file)]=digest
    # Only completed JSON identities from the later motivation: do not import
    # its evidence collector or copy its checkpoint-containing input/hash map.
    mp=json.loads(local(MOTIVATION_VERIFIER+'/protocol.json').read_text())
    mv=json.loads(local(MOTIVATION_VERIFIER+'/verification.json').read_text())
    mr=json.loads(local(MOTIVATION+'/result.json').read_text())
    require(mp['main_protocol_sha256']==PINNED[MOTIVATION+'/protocol.json'] and mv['status']=='passed'
        and mv['frames']==89 and mv['pairs']==85 and mv['successful_forwards']==437,'motivation verification incomplete')
    require(mr['status']=='completed' and mr['frames']==89 and mr['pairs']==85,'motivation incomplete')
    for directory,output in ((PARENT,'result'),(VERIFIER,'verification'),(MOTIVATION,'result'),(MOTIVATION_VERIFIER,'verification')):
        linked_execution(directory,output,inputs)
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
        p=dict(contract(),registered_utc=now(),selected_windows=windows,input_hashes=inputs,source_hashes=hashes,
            source_snapshot=str(snapshot),binary_deserialized_at_registration=False)
        verify(p);write(out/'protocol.json',p);status.update(status='completed',returncode=0,protocol_sha256=sha(out/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:status.update(completed_utc=now(),seconds=time.monotonic()-start);write(out/'registration.execution.json',status)
    return p


def load_keys(reference,keys):
    require(tuple(keys) in (FRAME_KEYS,FIXED_KEYS,TRANSITION_KEYS),'unregistered NPZ key selection')
    file=local(reference['path']);require(sha(file)==reference['sha256'],'NPZ input hash differs')
    with np.load(file,allow_pickle=False) as a:
        # NpzFile is lazy. Never iterate items or deserialize any H member.
        return {key:torch.from_numpy(a[key].copy()) for key in keys}


def load_windows(p):
    windows=[]
    for row in p['selected_windows']:
        fixed=load_keys(row['fixed_input'],FIXED_KEYS);mask=fixed['mask'];h=row['horizon'];n=len(mask)
        require(mask.dtype==torch.bool and mask.shape==(n,) and 4<=int(mask.sum())<=n<=96,'invalid fixed support')
        for key in ('aatype','residue_index','chain_id'):require(fixed[key].dtype==torch.int64 and fixed[key].shape==(n,),'legacy ID shape/dtype differs')
        require(fixed['time_ps'].dtype==torch.float64 and torch.equal(fixed['time_ps'],torch.arange(h+1,dtype=torch.float64)*1000.),'time grid differs')
        require([r['index'] for r in row['frames']]==list(range(h+1)) and [r['index'] for r in row['transitions']]==list(range(h)),'input coverage differs')
        frames=[]
        for r in row['frames']:
            a=load_keys(r,FRAME_KEYS)
            require(torch.equal(a['mask'],mask),'frame mask differs')
            for k,shape in (('source_native_backbone',(n,4,3)),('centered',(n,3)),('centroid',(3,)),('canonical',(n,3)),('basis',(3,3))):
                require(a[k].dtype==torch.float32 and a[k].shape==shape and bool(torch.isfinite(a[k]).all()),'source geometry shape/dtype/finite differs: '+k)
            require(bool((a['source_native_backbone'][~mask]==0).all()) and bool((a['canonical'][~mask]==0).all()),'source padding differs')
            frames.append(a)
        rotations=[load_keys(r,TRANSITION_KEYS)['R'] for r in row['transitions']]
        for R in rotations:
            require(R.dtype==torch.float64 and R.shape==(3,3) and bool(torch.isfinite(R).all())
                and torch.allclose(R.T@R,torch.eye(3,dtype=torch.float64),atol=1e-10,rtol=0)
                and abs(float(torch.linalg.det(R))-1.)<=1e-10,'saved proper R differs')
        windows.append(dict(source=row,fixed=fixed,frames=frames,rotations=rotations))
    return windows


class Budget:
    def __init__(self,out):self.out=Path(out);self.start=time.monotonic()
    def check(self):
        require(sys.platform.startswith('linux'),'Linux RSS units required')
        s=time.monotonic()-self.start;r=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        b=sum(f.stat().st_size for f in self.out.rglob('*') if f.is_file())
        require(s<=MAX_SECONDS and r<=MAX_RSS_BYTES and b<=MAX_OUTPUT_BYTES,'resource budget exceeded')
        return dict(seconds=s,peak_rss_bytes=r,output_bytes=b)


def ref(file):return dict(path=str(Path(file).relative_to(ROOT)),sha256=sha(file))


@torch.no_grad()
def diagnose_window(window,out,budget,phase):
    frame_audit,pair_audit=imports();row=window['source'];mask=window['fixed']['mask'];original=tree_sha(window)
    folder=Path(out)/f"step{row['step']:03d}";folder.mkdir(exist_ok=False)
    file=folder/'fixed.npz';save_npz(file,window['fixed'])
    result=dict(step=row['step'],protein=row['protein'],horizon=row['horizon'],temperature_k=row['temperature_k'],
        n_valid_residues=int(mask.sum()),fixed_input=ref(file),frames=[],transitions=[])
    frames=[]
    for t,a in enumerate(window['frames']):
        phase[0]=f"step{row['step']:03d}/frame{t:03d}";budget.check()
        f=frame_audit(a['source_native_backbone'],mask,a['centered'],a['canonical'],a['basis'])
        require(torch.equal(f['arrays']['centroid'].view(torch.uint8),a['centroid'].view(torch.uint8)),'source centroid differs')
        require(tree_sha(window)==original,'input mutated during frame audit')
        file=folder/f'frame{t:03d}.npz';save_npz(file,f['arrays'])
        record=dict(index=t,time_ps=float(window['fixed']['time_ps'][t]),source=row['frames'][t],**ref(file),summary=f['summary'])
        write(folder/f'frame{t:03d}.json',record);result['frames'].append(record);frames.append(f)
    frame_sha=tree_sha(frames)
    for t,R in enumerate(window['rotations']):
        phase[0]=f"step{row['step']:03d}/transition{t:03d}";budget.check()
        value=pair_audit(frames[t],frames[t+1],R,mask)
        require(tree_sha(window)==original and tree_sha(frames)==frame_sha,'input or frame output mutated during pair audit')
        file=folder/f'transition{t:03d}.npz';save_npz(file,value['arrays'])
        record=dict(index=t,time_ps=float(window['fixed']['time_ps'][t]),next_time_ps=float(window['fixed']['time_ps'][t+1]),
            source=row['transitions'][t],**ref(file),summary=value['summary'])
        write(folder/f'transition{t:03d}.json',record);result['transitions'].append(record)
    require(len(result['frames'])==row['horizon']+1 and len(result['transitions'])==row['horizon'],'output coverage differs')
    result['input_tensor_sha256']=original;write(folder/'result.json',result);return result


def run():
    common.lineage.cpu_guard();out=local(OUTPUT);file=out/'protocol.json';require(file.is_file(),'registration required')
    require(not any((out/n).exists() for n in ('started.json','execution.json','result.json')),'execution already attempted')
    digest=sha(file);start=time.monotonic();status=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest)
    write(out/'started.json',status);phase=['verify'];windows=None;completed=[]
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('registered wall-clock budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        p=json.loads(file.read_text());require(all(p.get(k)==v for k,v in contract().items()),'registered contract differs')
        registration=json.loads((out/'registration.execution.json').read_text())
        require(registration['status']=='completed' and registration['returncode']==0 and registration['protocol_sha256']==digest,'registration incomplete')
        verify(p);torch.set_num_threads(2);budget=Budget(out);budget.check();before_rng=tree_sha(cpu_rng())
        phase[0]='selective_NPZ_load';windows=load_windows(p)
        require([w['source']['step'] for w in windows]==[1,2,3,4] and [w['source']['horizon'] for w in windows]==[1,4,16,64]
            and sum(len(w['frames']) for w in windows)==89 and sum(len(w['rotations']) for w in windows)==85,'fixed panel differs')
        initial=tree_sha(windows);write(out/'inputs.initial.json',dict(input_tensor_sha256=initial,cpu_rng_sha256=before_rng))
        for w in windows:
            completed.append(diagnose_window(w,out,budget,phase))
            require(tree_sha(windows)==initial and tree_sha(cpu_rng())==before_rng,'input tensors or RNG changed')
        verify(p);require(sha(file)==digest,'protocol changed')
        result=dict(schema_version=VERSION,status='completed',protocol_sha256=digest,windows=completed,frames=89,transitions=85,
            input_tensor_sha256_before=initial,input_tensor_sha256_after=tree_sha(windows),cpu_rng_sha256_before=before_rng,
            cpu_rng_sha256_after=tree_sha(cpu_rng()),resources=budget.check(),H_arrays_loaded=0,encoder_calls=0,model_checkpoint_loads=0,
            new_PCA_fits=0,new_Kabsch_fits=0,new_rollouts=0,optimizer_updates=0,scientific_limits=p['limits'])
        write(out/'result.json',result);status.update(status='completed',returncode=0,result_sha256=sha(out/'result.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,failed_phase=phase[0],error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        status.update(completed_utc=now(),seconds=time.monotonic()-start,completed_windows=len(completed),cpu_rng_sha256_final=tree_sha(cpu_rng()))
        if windows is not None:status['input_tensor_sha256_final']=tree_sha(windows)
        write(out/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args();result=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=result.get('status','registered'))),flush=True)


if __name__=='__main__':main()
