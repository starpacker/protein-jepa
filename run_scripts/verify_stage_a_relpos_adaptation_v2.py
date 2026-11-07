"""Independent NumPy score replay and recorded lifecycle audit; no Torch imports.

V2 repairs only v1 registration JSON scalar serialization; all numerical
functions and tolerance values are unchanged. Failed v1 evidence is pinned.
Checkpoint/draw artifacts are stream-hashed only. Encoder, optimizer, masks and
RNG execution are NOT numerically replayed. Registered source and recorded
identities supply those claims. Saved FP32 labels/logs are promoted to FP64.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PRODUCER=ROOT/'results/hs_joint_20261101/stage_a_relpos_adaptation_v1'
OUTPUT=ROOT/'results/hs_joint_20261101/stage_a_relpos_verification_v2'
MAIN_PROTOCOL_SHA='148c197e6198f149e16bad205a6c68172d7231404fd112d766ea838c0a1aeb49'
VERSION='stage_a_relpos_independent_verification_v2'
PRIOR=Path('results/hs_joint_20261101/stage_a_relpos_verification_v1')
PRIOR_FAILURE_HASHES={
    'run_scripts/verify_stage_a_relpos_adaptation_v1.py':'6bc185a6be9709715026bbbc172e72f26e6556f65a7c703a4cb35a8bb78b477c',
    'tests/test_verify_stage_a_relpos_adaptation_v1.py':'72ad253d8827a3b1220aa0094afe7b05e6d55141f77570d394d72d9d7dc2de27',
    str(PRIOR/'protocol.json'):'a1b956aff2ace6c488d74ffeccf78dde9cf26cd91200007c4760cf18e4f63348',
    str(PRIOR/'registration.execution.json'):'d97e588fd068b1acdceb518b1cf171ec94970c49514ad4577f9ff646dc064d80',
    str(PRIOR/'source_snapshot/run_scripts/verify_stage_a_relpos_adaptation_v1.py'):'6bc185a6be9709715026bbbc172e72f26e6556f65a7c703a4cb35a8bb78b477c',
    str(PRIOR/'source_snapshot/tests/test_verify_stage_a_relpos_adaptation_v1.py'):'72ad253d8827a3b1220aa0094afe7b05e6d55141f77570d394d72d9d7dc2de27',
    str(PRIOR)+'_register.execution.json':'7eb3c6e2ef3f0f5bbf47d7b379d3666f3dafd9a1ac8b54baaef8070bad4c8ccb',
    str(PRIOR)+'_register.launch.json':'9d40297ed0586543a893fc0b61c810c822730e3410f77b507a2cb390fde86df8',
    str(PRIOR)+'_register.log':'b8a3e36c66944dd99e5231224ff843b36ec5cd8e45e5fa4bd4628f73a2d4ff34',
}
ARMS=('pca','zero','local');ROLES=('future_H','observed_H_substitution','predicted_H')
HORIZONS=(1,4,16,64)
PHASES=('smoke','prepare','train_pca','train_zero','train_local','fit_pca','fit_zero','fit_local','score')
METRICS=('log_mse','mae_angstrom','mse_angstrom2','rmse_angstrom')
ATOL=1e-10;RTOL=2e-10
LOG_FP32_BOUND=float(8*np.finfo(np.float32).eps)
LIMITS=('Independent saved-array scoring and descriptive bootstrap replay, plus hash-linked recorded lifecycle. '
        'No model, checkpoint, optimizer, draw-tensor or encoder numerical replay; no trajectory/HDF reads. '
        'Matching initialization/RNG/draw identities are producer records, not independently recomputed tensors. '
        'One seed and previously inspected TRAIN panel; no multiplicity-adjusted significance or adoption claim.')


def need(ok,message):
    if not ok:raise ValueError(message)


def path(value):
    p=Path(value);p=(ROOT/p if not p.is_absolute() else p).resolve()
    need(p.is_relative_to(ROOT) and p.name!='test.jsonl','unsafe path')
    return p


def sha(file):
    h=hashlib.sha256()
    with path(file).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def timestamp(value):
    t=datetime.fromisoformat(value);need(t.tzinfo is not None,'timezone required');return t

def write(file,value):
    with path(file).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def digest(value):
    need(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'invalid digest')
    return value


class Audit:
    def __init__(self):self.hashes={};self.comparisons=0;self.max_difference=0.
    def track(self,file,expected=None):
        p=path(file);d=sha(p)
        need(expected is None or d==expected,'hash mismatch: '+str(p))
        need(str(p) not in self.hashes or self.hashes[str(p)]==d,'artifact changed')
        self.hashes[str(p)]=d;return p
    def read(self,file,expected=None):return json.loads(self.track(file,expected).read_text())
    def ref(self,value):return self.track(value['path'],value['sha256'])
    def compare(self,actual,expected,label=''):
        if isinstance(expected,dict):
            need(isinstance(actual,dict) and set(actual)==set(expected),'keys differ: '+label)
            for k,v in expected.items():self.compare(actual[k],v,label+'/'+k)
        elif isinstance(expected,list):
            need(isinstance(actual,list) and len(actual)==len(expected),'list differs: '+label)
            for i,v in enumerate(expected):self.compare(actual[i],v,label+'/'+str(i))
        elif isinstance(expected,float):
            need(type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected),'nonfinite: '+label)
            delta=abs(actual-expected);self.comparisons+=1;self.max_difference=max(self.max_difference,delta)
            need(delta<=ATOL+RTOL*abs(expected),'numeric mismatch: '+label)
        else:need(type(actual) is type(expected) and actual==expected,'value differs: '+label)
    def finish(self):
        for file,d in self.hashes.items():need(sha(file)==d,'input changed during audit')


def float_vector(value,name):
    x=np.asarray(value)
    need(x.dtype==np.float32 and x.ndim==1 and x.size>0 and np.isfinite(x).all(),'invalid FP32 '+name)
    return x.astype(np.float64)


def metrics(pred_log,target_log,target_distance,*,pred_distance=None):
    p=float_vector(pred_log,'prediction log');y=float_vector(target_log,'target log');d=float_vector(target_distance,'target distance')
    need(p.shape==y.shape==d.shape and (d>=0).all(),'pair shape/distance differs')
    with np.errstate(over='ignore',invalid='ignore'):a=np.expm1(p) if pred_distance is None else float_vector(pred_distance,'persistence distance')
    need(a.shape==d.shape and np.isfinite(a).all(),'nonfinite distance prediction')
    error=a-d;mse=float(np.dot(error,error)/len(error))
    need(math.isfinite(mse),'nonfinite squared distance error')
    return dict(log_mse=float(np.mean((p-y)**2)),mae_angstrom=float(np.mean(np.abs(error))),
        mse_angstrom2=mse,rmse_angstrom=math.sqrt(mse),n_pairs=len(d),negative_count=int(np.count_nonzero(a<0)))


def log_link(log,raw):
    x=float_vector(log,'saved log');y=float_vector(raw,'raw distance')
    need(x.shape==y.shape and (y>=0).all(),'invalid raw label')
    expected=np.log1p(y)
    need(np.all(np.abs(x-expected)<=LOG_FP32_BOUND*np.maximum(1.,np.abs(expected))), 'FP32 log label linkage differs')


def score_arrays(arrays):
    scores={};reference=None
    for arm in ARMS:
        def a(key):return arrays[arm+'__'+key]
        pair=np.asarray(a('pair_indices'));n=len(pair)
        need(pair.dtype==np.int64 and pair.shape==(n,2) and n>0 and (pair>=0).all()
             and (pair[:,0]<pair[:,1]).all(),'invalid unique upper-triangle pairs')
        need(np.array_equal(np.lexsort((pair[:,1],pair[:,0])),np.arange(n))
             and len(np.unique(pair,axis=0))==n,'pair ordering/duplicates differ')
        labels=[a(k) for k in ('target_distance','target_log','observed_distance','persistence_log')]
        need(all(np.asarray(v).shape==(n,) for v in labels),'label support differs')
        current=[pair,*labels]
        if reference is not None:
            need(all(x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                     for x,y in zip(current,reference)),'arm pair/label bytes differ')
        reference=current
        target,target_log,observed,persistent_log=labels
        log_link(target_log,target);log_link(persistent_log,observed)
        for role in ROLES:scores[arm+'/'+role]=metrics(a(role+'__predicted_log'),target_log,target)
        persistence=metrics(persistent_log,target_log,target,pred_distance=observed)
        if 'CA_persistence' in scores:need(scores['CA_persistence']==persistence,'persistence differs')
        scores['CA_persistence']=persistence
    return scores


def selected_arrays(file):
    # Deliberately never access H_observed/H_predicted/pred_factors members.
    keys=('pair_indices','target_distance','target_log','observed_distance','persistence_log',
          *(role+'__predicted_log' for role in ROLES))
    with np.load(file,allow_pickle=False) as archive:
        return {arm+'__'+key:archive[arm+'__'+key].copy() for arm in ARMS for key in keys}


def panel(records):
    need(len(records)==100 and [r['cache_row_index'] for r in records]==list(range(400,500)),'heldout row order differs')
    need(len({r['metadata']['protein'] for r in records})==100
         and [r['metadata']['horizon'] for r in records]==list(HORIZONS)*25,'heldout protein/horizon panel differs')
    keys={arm+'/'+role for arm in ARMS for role in ROLES}|{'CA_persistence'}
    for r in records:
        need(set(r['scores'])==keys,'score role inventory differs')
        for s in r['scores'].values():
            need(type(s['n_pairs']) is int and s['n_pairs']==r['n_pairs']>0,'pair count differs')
            need(type(s['negative_count']) is int and 0<=s['negative_count']<=s['n_pairs'],'negative count invalid')
            need(all(math.isfinite(s[k]) and s[k]>=0 for k in METRICS),'invalid error score')
    return sorted(keys)


def aggregate(records):
    keys=panel(records);out={}
    for group in ('overall',*(f'h{h}' for h in HORIZONS)):
        rows=records if group=='overall' else [r for r in records if r['metadata']['horizon']==int(group[1:])]
        out[group]={}
        for key in keys:
            values=[r['scores'][key] for r in rows]
            means={m:float(np.mean([v[m] for v in values],dtype=np.float64)) for m in METRICS[:3]}
            out[group][key]=dict(**means,rmse_angstrom=math.sqrt(means['mse_angstrom2']),n_proteins=len(rows),
                n_pairs=sum(v['n_pairs'] for v in values),negative_count=sum(v['negative_count'] for v in values),
                mean_protein_negative_fraction=float(np.mean([v['negative_count']/v['n_pairs'] for v in values])))
    return out


def bootstrap(records):
    panel(records);rng=np.random.default_rng(20261102)
    samples=[]
    for h in HORIZONS:
        rows=np.flatnonzero([r['metadata']['horizon']==h for r in records])
        need(len(rows)==25,'bootstrap stratum differs')
        samples.append(rows[rng.integers(25,size=(10000,25))])
    indices=np.concatenate(samples,axis=1);out={}
    for arm in ARMS[1:]:
        out[arm+'-pca']={}
        for role in ROLES:
            result={}
            for metric in METRICS:
                field='mse_angstrom2' if metric=='rmse_angstrom' else metric
                x=np.array([r['scores'][arm+'/'+role][field] for r in records],dtype=np.float64)
                y=np.array([r['scores']['pca/'+role][field] for r in records],dtype=np.float64)
                a=x[indices].mean(axis=1);b=y[indices].mean(axis=1)
                point=float(x.mean()-y.mean())
                if metric=='rmse_angstrom':a=np.sqrt(a);b=np.sqrt(b);point=math.sqrt(x.mean())-math.sqrt(y.mean())
                lo,hi=np.percentile(a-b,[2.5,97.5],method='linear')
                result[metric]=dict(difference=point,interval95=[float(lo),float(hi)])
            out[arm+'-pca'][role]=result
    return out


def provenance(main,audit):
    for field in ('source_hashes','input_hashes'):
        for file,d in main[field].items():
            audit.track(file,d)
            if field=='source_hashes':audit.track(Path(main['source_snapshot'])/file,d)
    for file,expected in main['selected_hdf5_stats'].items():
        s=Path(file).stat()  # stat only, never open HDF content
        actual=dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev)
        need(actual==expected,'HDF stat changed')


def phase_results(main,audit,registration_end):
    need(main['phase_order']==list(PHASES),'phase schema differs')
    prior_end=timestamp(main['registered_utc']);results={}
    for phase in PHASES:
        folder=PRODUCER/phase;st=audit.read(folder/'started.json');ex=audit.read(folder/'execution.json')
        need(ex['status']=='completed' and ex['returncode']==0 and ex['pid']>0,'phase failed')
        need(all(st[k]==ex[k] for k in ('pid','command','started_utc','protocol_sha256','phase'))
             and ex['protocol_sha256']==MAIN_PROTOCOL_SHA and ex['phase']==phase,'phase linkage differs')
        start=timestamp(ex['started_utc']);end=timestamp(ex['completed_utc'])
        need(prior_end<=start<=end and math.isfinite(ex['seconds']) and ex['seconds']>=0,'phase order/time differs')
        if phase=='score':need(registration_end<=start,'verifier registered after score started')
        prior_end=end
        need(audit.ref(ex['result'])==folder/'result.json','phase result path differs')
        r=audit.read(folder/'result.json')
        need(r['phase']==phase and r['protocol_sha256']==MAIN_PROTOCOL_SHA and r['status']=='completed'
             and r['test_opened'] is False and r['bioemu2_rerun'] is False,'result contract differs')
        for ref in r['artifacts'].values():audit.ref(ref)
        results[phase]=r
    return results


def jsonlines(audit,ref):
    return [json.loads(line) for line in audit.ref(ref).read_text().splitlines()]


def training_logs(log,arm,audit,manifest):
    need(len(log)==len(manifest)==500,'500 adaptation updates required')
    for step,r in enumerate(log):
        need(r['step']==step+1 and r['feature_mode']==arm and r['cache_row_indices']==[(4*step+j)%400 for j in range(4)],'adaptation row cycle differs')
        need(r['ema_before']==step and r['ema_after']==step+1 and r['ema_momentum']==.998 and r['optimizer_updated'] is True,'EMA/update sequence differs')
        scale=(step+1)/100 if step<100 else .5*(1+math.cos(math.pi*(step-100)/400))
        audit.compare(r['lr'],[3e-4*.3*scale,3e-4*scale],'learning rates')
        need(all(math.isfinite(v) for v in r['losses'].values()) and math.isfinite(r['global_grad_norm_before_clip'])
             and r['global_grad_norm_before_clip']>=0 and math.isfinite(r['encoder_grad_norm_before_clip'])
             and r['encoder_grad_norm_before_clip']>0,'nonfinite losses/gradients')
        # FP32 loss sum is not independently reevaluated as FP64 arithmetic.
        need(r['tensor_audit']==manifest[step] and audit.ref(manifest[step])==PRODUCER/('train_'+arm)/'draws'/f'{step+1:04d}.pt','draw artifact order differs')
        digest(r['tensor_audit_sha256']);digest(r['draws_sha256']);digest(r['input_sha256'])
        rng=r['rng_sha256'];need(set(rng)=={'before','after_forward','after_backward','after_update'},'RNG phases differ')
        for value in rng.values():digest(value)
        need(rng['after_forward']==rng['after_backward']==rng['after_update'],'backward/update RNG differs')
        if step:need(log[step-1]['rng_sha256']['after_update']==rng['before'],'successive RNG stream differs')
        need(len(r['ranking_eligible_counts'])==7 and all(type(n)is int and 0<=n<=6 for n in r['ranking_eligible_counts'])
             and r['ranking_regression_count']==28,'ranking support counts invalid')
    return log


def readout_logs(log,metadata):
    need(len(log)==500,'500 readout steps required')
    for i,r in enumerate(log):
        index=i%400;md=metadata[index]
        need(r['step']==i+1 and r['cache_row_index']==index and r['protein']==md['protein']
             and r['horizon']==md['horizon'] and r['lr']==.001 and type(r['n_pairs'])is int and r['n_pairs']>0,'readout schedule differs')
        need(math.isfinite(r['log_mse']) and r['log_mse']>=0 and math.isfinite(r['gradient_norm_before_clip'])
             and r['gradient_norm_before_clip']>=0,'invalid readout loss/gradient')
        digest(r['labels_sha256'])
        if i>=400:need(r['labels_sha256']==log[index]['labels_sha256'] and r['n_pairs']==log[index]['n_pairs'],'repeated fit row labels differ')


def check_training(results,metadata,audit):
    first=None;first_fit=None
    need(results['smoke']['updates']==0 and results['smoke']['actual_training_rows_read'] is False
         and results['smoke']['smoke_state_reused'] is False,'smoke scope differs')
    need(results['prepare']['n_fit_rows']==400 and results['prepare']['heldout_history_loaded'] is False,'prepare split differs')
    for arm in ARMS:
        r=results['train_'+arm];t=r['training'];need(t['optimizer']=='AdamW' and t['updates']==t['ema_updates']==500,'training incomplete')
        manifest=audit.read(audit.ref(r['artifacts']['draw_manifest']))
        log=training_logs(jsonlines(audit,r['artifacts']['updates']),arm,audit,manifest)
        need(r['draw_sha256_by_step']==[x['tensor_audit_sha256'] for x in log],'draw manifest semantic hashes differ')
        digest(r['initial_state_sha256']);digest(r['initial_buffers_sha256'])
        need(r['warm_start']['model_state_sha256']==r['initial_state_sha256']
             and r['warm_start']['named_buffers_sha256']==r['initial_buffers_sha256'],'warm start recorded identity differs')
        matched=(r['initial_state_sha256'],r['initial_buffers_sha256'],r['draw_sha256_by_step'],
                 [{k:x[k] for k in ('input_sha256','draws_sha256','rng_sha256','ranking_eligible_counts')} for x in log])
        if first is not None:need(matched==first,'arms initialization/draw tapes differ')
        first=matched
        fit=results['fit_'+arm];flog=jsonlines(audit,fit['artifacts']['updates']);readout_logs(flog,metadata)
        fa=fit['readout'];need(fa['optimizer_steps']==500 and fit['heldout_history_loaded'] is False
            and all(fa[k] is True for k in ('inputs_unchanged','features_unchanged','cpu_rng_unchanged')),'readout completion differs')
        digest(fa['initial_head_sha256']);digest(fa['final_head_sha256'])
        current=(fa['initial_head_sha256'],[(x['cache_row_index'],x['labels_sha256'],x['n_pairs']) for x in flog])
        if first_fit is not None:need(current==first_fit,'head initialization/labels differ across arms')
        first_fit=current
        model=fit['model'];need(model['checkpoint_sha256']==r['artifacts']['checkpoint']['sha256'] and model['ema_step']==500
            and model['model_state_sha256']==t['terminal_state_sha256'] and model['teacher_resynced'] is False,'fit terminal model linkage differs')
        enc=fit['encoding'];need(enc['rows']==400 and enc['feature_mode']==arm and enc['readout_encoder']=='online'
            and enc['model_state_sha256']==t['terminal_state_sha256'],'fit encoder identity differs')
        for key in ('model_unchanged','buffers_unchanged','cpu_rng_unchanged'):need(enc[key] is True,'encoding mutated state')
        score=results['score']['identities'][arm]
        need(score['model_state']==t['terminal_state_sha256'] and score['head_state']==fa['final_head_sha256']
             and score['model']['checkpoint_sha256']==r['artifacts']['checkpoint']['sha256']
             and score['model_buffers']==model['named_buffers_sha256'],'score model/head identity differs')
        need(score['encoding']['rows']==100 and score['encoding']['feature_mode']==arm
             and score['encoding']['model_state_sha256']==score['model_state'],'score encoding differs')


def check_scores(result,main,audit):
    records=result['per_protein'];panel(records)
    need(result['heldout_history_read_after_all_fits'] is True and result['no_cached_H_used'] is True
         and result['no_stage_b_or_decoder_execution'] is True,'score scope differs')
    recomputed=[]
    for r in records:
        i=r['cache_row_index'];need(r['metadata']==main['selected_metadata'][i],'selected heldout identity differs')
        file=audit.ref(r['predictions']);need(file==PRODUCER/'score'/f'heldout_{i:03d}.npz','pair artifact path differs')
        need(audit.read(file.with_suffix('.json'))==r,'per-row JSON differs')
        values=score_arrays(selected_arrays(file));audit.compare(r['scores'],values,f'row{i}')
        recomputed.append(dict(r,scores=values))
        h=r['history'];need(h['cache_row_index']==i and h['endpoint_coordinates_read'] is False
                            and len(h['context_frames'])==8,'observed-only history differs')
        for key in ('observation_sha256','input_sha256','topology_sha256'):digest(h[key])
        need(set(r['prediction_audits'])==set(ARMS),'prediction arm inventory differs')
        for arm,p in r['prediction_audits'].items():
            need(p['feature_mode']==arm and p['encoder']=='online/context_encoder' and p['history_frames']==8
                 and p['history_interval_ps']==1000 and p['horizon_frames']==[r['metadata']['horizon']]
                 and p['teacher_called'] is False and p['future_coordinates_used'] is False
                 and p['normalization']=='none' and p['rebasing']=='none' and p['cpu_rng_unchanged'] is True,'prediction scope differs')
    need(result['history_audit_sha256']==canonical([r['history'] for r in records]),'history audit hash differs')
    audit.compare(result['aggregates'],aggregate(recomputed),'aggregates')
    b=result['paired_bootstrap'];need(b['draws']==10000 and b['seed']==20261102 and b['percentile_method']=='linear'
        and b['primary']=='future_H/log_mse' and b['schema_version']=='stage_a_relpos_readout_v1','bootstrap contract differs')
    audit.compare(b['contrasts'],bootstrap(recomputed),'paired bootstrap')


def main_contract(main):
    need(main['schema_version']=='stage_a_relpos_adaptation_trial_v1' and main['arms']==list(ARMS) and main['seed']==42,'main contract differs')
    metadata=main['selected_metadata'];need(len(metadata)==500 and len({m['protein'] for m in metadata})==500
        and [m['horizon'] for m in metadata]==list(HORIZONS)*125,'registered500 panel differs')
    a=main['adaptation'];need(a['updates']==500 and a['batch_size']==4 and a['row_order']==list(range(400))*5
        and a['lr']==.0003 and a['warmup_steps']==100 and a['encoder_lr_scale']==.3 and a['weight_decay']==.02
        and a['gradient_clip']==1 and a['history']==8 and a['future_frames']==1,'adaptation contract differs')
    f=main['readout'];need(f['updates']==500 and f['order']==list(range(400))+list(range(100)) and f['hidden_dim']==256
        and f['lr']==.001 and f['seed']==42 and f['weight_decay']==0 and f['gradient_clip']==1,'readout contract differs')
    need(main['score']['indices']==list(range(400,500)) and main['checkpoint_deserialized_at_registration'] is False
        and main['coordinate_arrays_read_at_registration'] is False,'registration scope differs')


def registration_record(audit,folder,protocol_sha):
    r=audit.read(folder/'registration.execution.json')
    need(r['status']=='completed' and r['returncode']==0 and r['protocol_sha256']==protocol_sha and r['pid']>0
         and r['seconds']>=0 and timestamp(r['started_utc'])<=timestamp(r['completed_utc']),'registration failed')
    return r


def prior_failure(audit):
    for file,d in PRIOR_FAILURE_HASHES.items():audit.track(file,d)
    failed=audit.read(PRIOR/'registration.execution.json')
    need(failed['status']=='failed' and failed['returncode']==1 and failed['pid']>0
         and failed['error']=="TypeError('Object of type float32 is not JSON serializable')",
         'prior registration failure differs')
    outer=audit.read(str(PRIOR)+'_register.execution.json')
    launch=audit.read(str(PRIOR)+'_register.launch.json')
    need(outer['exit_code']==1 and all(outer[k]==launch[k] for k in ('command','started_utc','cwd','log'))
         and path(outer['log'])==path(str(PRIOR)+'_register.log'),'prior outer failure differs')
    need(timestamp(outer['started_utc'])<=timestamp(failed['started_utc'])
         <=timestamp(failed['completed_utc'])<=timestamp(outer['completed_utc']),'prior failure times differ')
    need('Object of type float32 is not JSON serializable' in path(outer['log']).read_text(),
         'prior failure log differs')
    # Partial protocol is deliberately byte-pinned only; JSON writing failed.
    return dict(version='v1',stage='registration',status='failed',pid=failed['pid'],
        started_utc=failed['started_utc'],completed_utc=failed['completed_utc'],error=failed['error'],
        repair='Convert NumPy float32 bound to Python float; exact binary value unchanged; no metric or tolerance change',
        prior_failure_hashes=PRIOR_FAILURE_HASHES)


def register():
    OUTPUT.mkdir(parents=True,exist_ok=False);start=dict(pid=os.getpid(),command=sys.argv,started_utc=now());status=dict(start);begin=time.monotonic()
    try:
        need(not (PRODUCER/'score/started.json').exists() and not (PRODUCER/'score/result.json').exists(),'registration must precede scoring')
        audit=Audit();failure=prior_failure(audit);main=audit.read(PRODUCER/'protocol.json',MAIN_PROTOCOL_SHA);main_contract(main)
        registration_record(audit,PRODUCER,MAIN_PROTOCOL_SHA);provenance(main,audit)
        sources={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__),
                 'tests/test_verify_stage_a_relpos_adaptation_v2.py':sha('tests/test_verify_stage_a_relpos_adaptation_v2.py')}
        snapshot=OUTPUT/'source_snapshot'
        for file,d in sources.items():
            target=snapshot/file;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(path(file).read_bytes())
            need(sha(target)==d,'snapshot source changed')
        protocol=dict(schema_version=VERSION,registered_utc=now(),main_protocol_sha256=MAIN_PROTOCOL_SHA,
            main_protocol_path=str((PRODUCER/'protocol.json').relative_to(ROOT)),prior_failure=failure,
            declared_result_path=str((PRODUCER/'score/result.json').relative_to(ROOT)),source_hashes=sources,
            source_snapshot=str(snapshot.relative_to(ROOT)),input_hashes={str(path(k).relative_to(ROOT)):v for k,v in audit.hashes.items()},
            numerical_tolerances=dict(atol=ATOL,rtol=RTOL,FP32_log_bound=LOG_FP32_BOUND),
            timing='registered before score, potentially after adaptation started; not pre-training registration',limits=LIMITS)
        audit.finish();need(not (PRODUCER/'score/started.json').exists(),'scoring began during registration')
        write(OUTPUT/'protocol.json',protocol);status.update(status='completed',returncode=0,protocol_sha256=sha(OUTPUT/'protocol.json'))
    except BaseException as error:status.update(status='failed',returncode=1,error=repr(error));raise
    finally:status.update(completed_utc=now(),seconds=time.monotonic()-begin);write(OUTPUT/'registration.execution.json',status)


def run():
    protocol_path=OUTPUT/'protocol.json';protocol_sha=sha(protocol_path)
    started=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=protocol_sha)
    write(OUTPUT/'started.json',started);status=dict(started);begin=time.monotonic()
    try:
        audit=Audit();protocol=audit.read(protocol_path,protocol_sha)
        need(protocol['main_protocol_sha256']==MAIN_PROTOCOL_SHA and protocol['schema_version']==VERSION,'verifier protocol differs')
        own=registration_record(audit,OUTPUT,protocol_sha)
        need(protocol['prior_failure']==prior_failure(audit),'repair provenance differs')
        for file,d in protocol['input_hashes'].items():audit.track(file,d)
        for file,d in protocol['source_hashes'].items():audit.track(file,d);audit.track(Path(protocol['source_snapshot'])/file,d)
        main=audit.read(PRODUCER/'protocol.json',MAIN_PROTOCOL_SHA);main_contract(main);provenance(main,audit)
        registration_record(audit,PRODUCER,MAIN_PROTOCOL_SHA)
        results=phase_results(main,audit,timestamp(own['completed_utc']))
        check_training(results,main['selected_metadata'],audit);check_scores(results['score'],main,audit)
        need(timestamp(audit.read(PRODUCER/'score/execution.json')['completed_utc'])<=timestamp(started['started_utc']),'verification precedes scoring completion')
        audit.finish();provenance(main,audit)
        result=dict(status='passed',rows=100,groups=5,arms=3,readout_roles=3,bootstrap_draws=10000,
            adaptation_updates=1500,readout_updates=1500,main_protocol_sha256=MAIN_PROTOCOL_SHA,
            checked_hashes=audit.hashes,comparisons=audit.comparisons,max_abs_difference=audit.max_difference,limits=LIMITS)
        write(OUTPUT/'verification.json',result);status.update(status='completed',returncode=0,verification_sha256=sha(OUTPUT/'verification.json'))
    except BaseException as error:status.update(status='failed',returncode=1,error=repr(error));raise
    finally:status.update(completed_utc=now(),seconds=time.monotonic()-begin);write(OUTPUT/'execution.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('register','run'),required=True)
    globals()[parser.parse_args().mode]()
