"""Descriptive canonical-coordinate continuity on four existing TRAIN paths.

Only saved path sidecars are deserialized. No encoder/model/checkpoint is loaded,
no H is recomputed, and no canonicalization is changed in the frozen encoder.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import torch

from run_scripts import diagnose_train_latent_moments_v1 as lineage

ROOT=Path(__file__).resolve().parents[1]
OUTPUT='results/hs_joint_20261101/train_pca_continuity_v1'
VERSION='train_pca_continuity_v1'
HORIZONS=(1,4,16,64)
MAX_SECONDS=600
MAX_RSS_BYTES=4*1024**3
MAX_OUTPUT_BYTES=256*1024**2
MOMENT='results/hs_joint_20261101/train_latent_moments_v1'
VERIFIER='results/hs_joint_20261101/latent_moment_verification_v1'
MOTIVATION_HASHES={MOMENT+'/protocol.json':'29d37b53f6c17b5f7dbcc7242a3869b0b3c5dc6b3747a7ffac584e53af2272f2',
    MOMENT+'/result.json':'fac5e892603d8a0914ba98eaf6cbaacf6f316f83a9ff9677bbd7fc99a3bcc0cb',
    MOMENT+'/execution.json':'6d13d125ab7f351b79ecb1400d2ec69a69f9d96e5532492f3644f510be9cc001',
    VERIFIER+'/protocol.json':'185000ba968952d29e05e9aef7a31b791dbb66227c2a907a30c2001a4ca9f4c3',
    VERIFIER+'/verification.json':'f8647a25c5e9e464c10c93006c676e4692225e737622d63dd344a252fb6d23c2',
    VERIFIER+'/execution.json':'a0102d3479e694ed98db0fa35d63a66cfc110b1c9a14a2b2e58b00f603f2e316'}
SOURCES=('run_scripts/diagnose_train_pca_continuity_v1.py','tests/test_train_pca_continuity_v1.py',
    'docs/TRAIN_PCA_CONTINUITY_V1_20261102.md','protein_jepa_frame/models/stage_c/canonical_continuity_v1.py',
    'tests/test_canonical_continuity_v1.py','protein_jepa_frame/models/encoder.py',
    'run_scripts/diagnose_train_latent_moments_v1.py','tests/test_train_latent_moments_v1.py')


def require(ok,message):
    if not ok:raise ValueError(message)


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
    arrays={k:v.detach().cpu().numpy().copy() if torch.is_tensor(v) else np.asarray(v) for k,v in values.items()}
    require(all(not v.dtype.hasobject for v in arrays.values()),'object arrays forbidden')
    with local(file).open('xb') as f:np.savez(f,**arrays)


def imports():
    # Make the frozen source helper's transitive repository dependencies visible
    # to sources() before registration; importing does not construct an encoder.
    import protein_jepa_frame.models.encoder
    from protein_jepa_frame.models.stage_c.canonical_continuity_v1 import canonical_frame_audit,transition_audit
    return canonical_frame_audit,transition_audit


def sources():
    imports();found=set(SOURCES)
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT) and p.is_file():found.add(str(p.relative_to(ROOT)))
    return sorted(found)


def contract():
    return dict(schema_version=VERSION,horizons=list(HORIZONS),frames=89,transitions=85,dt_ps=1000.,
        device='cpu',torch_threads=2,frame_dtype='float32',analysis_dtype='float64',max_residues=96,latent_dim=128,
        source='exact per-frame B1 frozen canonical helper on sanitized native backbone, fixed observed mask',
        frame_rule='descending eigh of centered CA scatter +1e-6 I, then nonnegative projected third-moment sign',
        precision='retain source FP32 centered CA, signed basis and local features; promote recorded centered CA/basis and recompute analysis Z in FP64; save analysis minus source roundoff; never replace eigendecomposition with FP64',
        alignment='proper row-vector Kabsch next native CA to current in FP64; physical RMSD uses FP64 centering; final next basis transported as R.T @ Bnext',
        matching='always all48 O(3) signed permutations: lexicographic permutations then signs(-1,+1); P[perm[j],j]=sign[j]; minimize ||Bcurrent-R.T@Bnext@P||F^2; first exact minimizer; geometry only',
        decomposition='analysisZnext-analysisZcurrent = (Ysource_centered@R-Xsource_centered)@Bcurrent + (analysisZnext-Ysource_centered@R@Bcurrent); retain both squared terms, signed cross term and closure, not explained percentages',
        pair_support='all lexicographic i<j pairs on identical fixed valid CA support; descriptive shape distance change, not frozen physical-nonlocal evaluation metric',
        reporting='every frame and every transition within each of four separate windows; full48 costs and full basis overlaps; no event threshold, filtering, correlations, CI, pooled headline or selection',
        degeneracy='record eigengaps, standardized skew support, determinants and alignment singular values; isotropic jitter cannot split repeated eigenspaces; signed permutation matching cannot resolve general continuous degenerate-subspace rotations',
        limits='Association-only prospective diagnostic motivated by prior moment results. Basis/shape allocation is a fixed algebraic convention, not a unique causal partition. Matching/transport sees both adjacent true frames and is not an inference-time repair. No H re-encoding or channel intervention; no causal attribution of H jumps or forecast failure.',
        resource_limits=dict(wall_seconds=MAX_SECONDS,peak_rss_bytes=MAX_RSS_BYTES,output_bytes=MAX_OUTPUT_BYTES),
        new_rollouts=0,optimizer_updates=0,encoder_calls=0,model_checkpoint_loads=0,hdf_loaded=False,test_opened=False,validation_loaded=False)


def verify(p):
    for file,digest in p['input_hashes'].items():require(sha(file)==digest,'input changed: '+file)
    for key,snapshot in (('source_hashes','source_snapshot'),('parent_source_hashes','parent_source_snapshot')):
        for file,digest in p[key].items():
            require(sha(file)==digest and sha(Path(p[snapshot])/file)==digest,'source/snapshot changed: '+file)


def register():
    lineage.cpu_guard();out=local(OUTPUT);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(pid=os.getpid(),command=sys.argv,started_utc=now())
    try:
        parent,rows=lineage.parent_metadata()
        inputs=dict(lineage.PARENT_HASHES);inputs.update(MOTIVATION_HASHES)
        inputs[lineage.PARENT+'/started.json']=sha(lineage.PARENT+'/started.json')
        inputs.update({r['sidecar']:r['sidecar_sha256'] for r in rows})
        for file,digest in inputs.items():require(sha(file)==digest,'input evidence differs: '+file)
        motivation=json.loads(local(MOMENT+'/result.json').read_text())
        require(motivation['status']=='completed' and [w['horizon'] for w in motivation['windows']]==list(HORIZONS),
            'moment motivation panel incomplete')
        checked=json.loads(local(VERIFIER+'/verification.json').read_text())
        require(checked['status']=='passed' and checked['future_transitions']==85,'moment verification incomplete')
        hashes={file:sha(file) for file in sources()};snapshot=Path(OUTPUT)/'source_snapshot'
        for file,digest in hashes.items():
            target=local(snapshot/file);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(local(file).read_bytes())
            require(sha(target)==digest,'source changed during snapshot')
        p=dict(contract(),registered_utc=now(),selected_paths=rows,input_hashes=inputs,source_hashes=hashes,
            source_snapshot=str(snapshot),parent_source_hashes=parent['source_hashes'],parent_source_snapshot=parent['source_snapshot'],
            parent_checkpoint_identity=dict(path=parent['checkpoint'],sha256=parent['checkpoint_sha256']),
            binary_deserialized_at_registration=False,checkpoint_bytes_read=False)
        verify(p);write_json(out/'protocol.json',p);status.update(status='completed',returncode=0,protocol_sha256=sha(out/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-start);write_json(out/'registration.execution.json',status)
    return p


class Budget:
    def __init__(self,out):self.start=time.monotonic();self.out=Path(out)
    def check(self):
        require(sys.platform.startswith('linux'),'RSS convention requires Linux')
        elapsed=time.monotonic()-self.start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        require(elapsed<=MAX_SECONDS and rss<=MAX_RSS_BYTES and size<=MAX_OUTPUT_BYTES,'registered resource budget exceeded')
        return dict(seconds=elapsed,peak_rss_bytes=rss,output_bytes=size)


def frame_summary(frame):
    # Every source/intermediate tensor is retained in NPZ. JSON carries the
    # compact quantities needed to inspect sign conventions and degeneracy.
    keys=('eigvals_desc','skew_before_sign','sign','abs_standardized_skew','skew_defined','normalized_eigengaps',
        'basis_determinant','source_function_equal','source_function_max_abs_error')
    require(all(k in frame for k in keys),'missing frame summary descriptor')
    def clean(v):
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    return {k:clean(frame[k].detach().cpu().tolist() if torch.is_tensor(frame[k]) else np.asarray(frame[k]).tolist())
            for k in keys}


def source_frame_parity(coords,mask,frame):
    from protein_jepa_frame.models.encoder import _within_frame_canonical_positions
    # This pure coordinate helper has no parameters and is not encoder.forward.
    frozen=_within_frame_canonical_positions(coords[None],mask[None])[0]
    require(torch.equal(frame['canonical'],frozen),'per-frame frozen source function parity failed')
    return dict(frame,source_function_equal=torch.tensor(True),
        source_function_max_abs_error=(frame['canonical'].double()-frozen.double()).abs().max())


@torch.no_grad()
def diagnose_window(saved,row,out,budget):
    frame_audit,transition_audit=imports()
    safe,mask_batch,H=lineage.validate_saved(saved,row);mask=mask_batch[0]
    coords=safe['coords'][0,3:];h=row['horizon'];times=saved['time_ps']
    folder=local(Path(out)/f"step{row['step']:03d}");folder.mkdir(exist_ok=False)
    fixed=folder/'fixed.npz';save_npz(fixed,dict(mask=mask,time_ps=times,
        **{k:safe[k][0] for k in ('aatype','residue_index','chain_id')}))
    frames=[];frame_records=[];records=[]
    # No encoding: H entries below are the existing native true-H bytes.
    for t in range(h+1):
        budget.check();frame=source_frame_parity(coords[t],mask,frame_audit(coords[t],mask))
        file=folder/f'frame{t:03d}.npz'
        save_npz(file,dict(frame,source_native_backbone=coords[t],saved_true_H=H[t]))
        record=dict(index=t,time_ps=float(times[t]),path=str(file.relative_to(ROOT)),sha256=sha(file),summary=frame_summary(frame))
        write_json(folder/f'frame{t:03d}.json',record);frames.append(frame);frame_records.append(record)
    for t in range(h):
        budget.check();value=transition_audit(frames[t],frames[t+1],H[t],H[t+1],mask)
        file=folder/f'transition{t:03d}.npz';save_npz(file,value['arrays'])
        record=dict(index=t,time_ps=float(times[t]),next_time_ps=float(times[t+1]),
            path=str(file.relative_to(ROOT)),sha256=sha(file),summary=value['summary'])
        write_json(folder/f'transition{t:03d}.json',record);records.append(record)
    require(len(records)==h and len(frame_records)==h+1,'window coverage differs')
    require(lineage.tree_sha(saved)==row['replay_tensor_sha256'],'saved input mutated')
    result=dict(step=row['step'],horizon=h,protein=row['batch']['protein'],temperature_k=row['batch']['temperature_k'],
        n_valid_residues=int(mask.sum()),frames=frame_records,transitions=records,
        fixed_input=dict(path=str(fixed.relative_to(ROOT)),sha256=sha(fixed)),resources=budget.check())
    write_json(folder/'result.json',result);return result


def run():
    lineage.cpu_guard();out=local(OUTPUT);file=out/'protocol.json'
    require(file.is_file(),'registration required')
    require(not any((out/n).exists() for n in ('started.json','execution.json','result.json')),'execution already attempted')
    digest=sha(file);start=time.monotonic();identity=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest)
    write_json(out/'started.json',identity);status=dict(identity);completed=[];phase='verify'
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*_):raise TimeoutError('registered wall-clock budget exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        p=json.loads(file.read_text());require(all(p.get(k)==v for k,v in contract().items()),'registered contract differs')
        registration=json.loads((out/'registration.execution.json').read_text())
        require(registration['status']=='completed' and registration['returncode']==0 and registration['protocol_sha256']==digest,'registration incomplete')
        require([r['horizon'] for r in p['selected_paths']]==list(HORIZONS)
            and [r['step'] for r in p['selected_paths']]==[1,2,3,4],'fixed four-window order differs')
        verify(p);torch.set_num_threads(2);budget=Budget(out);budget.check()
        before=lineage.tree_sha(lineage.cpu_rng())
        for row in p['selected_paths']:
            phase=f"step{row['step']:03d}/sidecar_load";budget.check()
            saved=torch.load(local(row['sidecar']),map_location='cpu',weights_only=False)
            phase=f"step{row['step']:03d}/geometry";completed.append(diagnose_window(saved,row,out,budget))
            require(lineage.tree_sha(lineage.cpu_rng())==before,'geometry diagnostic consumed process RNG')
            del saved
        require(sum(len(w['transitions']) for w in completed)==85 and sum(len(w['frames']) for w in completed)==89,'panel coverage differs')
        verify(p);require(sha(file)==digest,'protocol changed during execution')
        result=dict(schema_version=VERSION,status='completed',protocol_sha256=digest,windows=completed,
            transitions=85,frames=89,cpu_rng_sha256_before=before,cpu_rng_sha256_after=lineage.tree_sha(lineage.cpu_rng()),
            resources=budget.check(),new_rollouts=0,optimizer_updates=0,encoder_calls=0,model_checkpoint_loads=0,
            hdf_loaded=False,test_opened=False,validation_loaded=False,scientific_limits=p['limits'])
        write_json(out/'result.json',result);status.update(status='completed',returncode=0,result_sha256=sha(out/'result.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,failed_phase=phase,completed_windows=len(completed),error=f'{type(error).__name__}: {error}');raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        status.update(completed_utc=now(),seconds=time.monotonic()-start);write_json(out/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args();result=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=result.get('status','registered'))),flush=True)


if __name__=='__main__':main()
