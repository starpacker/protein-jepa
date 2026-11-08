"""Render verified Stage A endpoint-readout JSON, never models or tensor files.

Descriptive presentation only: no new bootstrap, selection or cross-arm latent
loss ranking. Source-only until explicitly invoked after independent success.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT/'results/hs_joint_20261101/stage_a_relpos_adaptation_v1'
VERIFIER = ROOT/'results/hs_joint_20261101/stage_a_relpos_verification_v2'
OUTPUT = ROOT/'results/hs_joint_20261101/stage_a_relpos_adaptation_report_v1'
VERSION = 'stage_a_relpos_adaptation_report_v1'
ARMS = ('pca','zero','local')
ROLES = ('future_H','observed_H_substitution','predicted_H')
GROUPS = ('overall','h1','h4','h16','h64')
METRICS = ('log_mse','mae_angstrom','mse_angstrom2','rmse_angstrom')
PHASES = ('smoke','prepare','train_pca','train_zero','train_local','fit_pca','fit_zero','fit_local','score')
SCORE_KEYS = tuple(a+'/'+r for r in ROLES for a in ARMS)+('CA_persistence',)
LIMITS = ('One primary: equal-protein future_H log-MSE, using true target-geometry H, not a forecast. '
    'predicted_H is the observed-only Stage A endpoint predictor and a fixed secondary; '
    'observed-H substitution uses the same fitted head, and raw observed-CA persistence is a descriptive baseline. '
    'Single paired seed and PCA-pretrained projection/predictor/heads privilege the PCA warm start. '
    'The 100 inspected TRAIN proteins were held out only from this adaptation/readout fitting and may have been seen in pretraining. '
    'Arm-local EMA teacher losses cannot rank representations. Endpoint readability does not prove latent stability, '
    'latent geometry, Stage B kinetics, downstream compatibility or superiority to BioEmu2. No adoption decision is made here.')


def need(ok,message):
    if not ok:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()


def local(value):
    p=Path(value);p=(p if p.is_absolute() else ROOT/p).resolve()
    need(p.is_relative_to(ROOT) and p.name!='test.jsonl','path outside repository or sealed test input')
    return p


def sha(path):
    h=hashlib.sha256()
    with local(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with local(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


class Inputs:
    def __init__(self):self.hashes={}
    def track(self,path,expected=None):
        p=local(path);digest=sha(p)
        need(expected is None or digest==expected,'input hash differs: '+str(p))
        need(str(p) not in self.hashes or self.hashes[str(p)]==digest,'input changed during rendering')
        self.hashes[str(p)]=digest;return p
    def read(self,path,expected=None):return json.loads(self.track(path,expected).read_text())
    def finish(self):
        for path,digest in self.hashes.items():need(sha(path)==digest,'input changed after rendering')


def normalized_hashes(values):
    result={}
    for path,digest in values.items():
        key=str(local(path));need(key not in result or result[key]==digest,'contradictory verified path aliases')
        result[key]=digest
    return result


def checked_read(inputs,path,checked):
    key=str(local(path));need(key in checked,'JSON not covered by independent verification: '+key)
    return inputs.read(path,checked[key])


def completed_identity(started,execution,protocol_sha):
    need(execution['status']=='completed' and execution['returncode']==0 and execution['pid']>0
        and execution['protocol_sha256']==protocol_sha
        and all(execution[k]==started[k] for k in ('pid','command','started_utc','protocol_sha256')),
        'incomplete/unlinked execution')
    need(datetime.fromisoformat(execution['completed_utc'])>=datetime.fromisoformat(execution['started_utc'])
        and math.isfinite(execution['seconds']) and execution['seconds']>=0,'invalid execution timing')


def verified_inputs(inputs,main_sha,verifier_sha):
    need(len(main_sha)==len(verifier_sha)==64,'explicit protocol hashes required')
    protocol=inputs.read(PRODUCER/'protocol.json',main_sha)
    vp=inputs.read(VERIFIER/'protocol.json',verifier_sha)
    need(vp['main_protocol_sha256']==main_sha,'different producer verified')
    vs=inputs.read(VERIFIER/'started.json');ve=inputs.read(VERIFIER/'execution.json')
    completed_identity(vs,ve,verifier_sha)
    verification=inputs.read(VERIFIER/'verification.json',ve['verification_sha256'])
    need(verification['status']=='passed' and verification['rows']==100 and verification['groups']==5
        and verification['arms']==3 and verification['readout_roles']==3
        and verification['bootstrap_draws']==10000,'verification incomplete')
    checked=normalized_hashes(verification['checked_hashes'])
    need(checked.get(str(local(PRODUCER/'protocol.json')))==main_sha,'main protocol not independently covered')
    results={};executions={};logs={}
    for phase in PHASES:
        folder=PRODUCER/phase
        started=checked_read(inputs,folder/'started.json',checked)
        execution=checked_read(inputs,folder/'execution.json',checked)
        completed_identity(started,execution,main_sha)
        need(started['phase']==execution['phase']==phase,'phase identity differs')
        ref=execution['result'];need(local(ref['path'])==local(folder/'result.json'),'phase result path differs')
        result=checked_read(inputs,folder/'result.json',checked)
        need(sha(folder/'result.json')==ref['sha256'] and result['status']=='completed'
            and result['phase']==phase and result['protocol_sha256']==main_sha,'phase result linkage differs')
        results[phase]=result;executions[phase]=execution
        if phase.startswith(('train_','fit_')):
            ref=result['artifacts']['updates'];path=local(ref['path'])
            need(path==local(folder/'updates.jsonl') and checked.get(str(path))==ref['sha256'],
                 'learning log not independently covered')
            logs[phase]=[json.loads(line) for line in inputs.track(path,ref['sha256']).read_text().splitlines()]
            need([r['step'] for r in logs[phase]]==list(range(1,501)),'incomplete learning trajectory')
    validate_scores(results['score'])
    return protocol,results,executions,logs,verification


def validate_scores(result):
    rows=result['per_protein']
    need(len(rows)==100 and [r['cache_row_index'] for r in rows]==list(range(400,500))
        and len({r['metadata']['protein'] for r in rows})==100
        and [r['metadata']['horizon'] for r in rows]==[1,4,16,64]*25,'complete fixed heldout100 required')
    for row in rows:
        need(set(row['scores'])==set(SCORE_KEYS),'missing score role')
        for score in row['scores'].values():
            need(score['n_pairs']==row['n_pairs'] and score['n_pairs']>0
                and all(math.isfinite(score[k]) and score[k]>=0 for k in METRICS),'invalid metric support')
    need(set(result['aggregates'])==set(GROUPS),'missing horizon groups')
    for group,values in result['aggregates'].items():
        need(set(values)==set(SCORE_KEYS),'aggregate role coverage differs')
        need(all(v['n_proteins']==(100 if group=='overall' else 25) for v in values.values()),'aggregate protein coverage differs')
    bootstrap=result['paired_bootstrap']
    need(bootstrap['primary']=='future_H/log_mse' and bootstrap['draws']==10000
         and set(bootstrap['contrasts'])=={'zero-pca','local-pca'},'registered contrast coverage differs')


def tables(score,results,executions):
    proteins=[]
    for row in score['per_protein']:
        value=dict(cache_row_index=row['cache_row_index'],**row['metadata'],n_pairs=row['n_pairs'])
        # context_frames is metadata, represented as JSON rather than a Python repr.
        value={k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in value.items()}
        for key in SCORE_KEYS:
            for metric in METRICS+('negative_count',):value[key+'__'+metric]=row['scores'][key][metric]
        proteins.append(value)
    aggregates=[]
    for group in GROUPS:
        for key in SCORE_KEYS:aggregates.append(dict(group=group,role=key,**score['aggregates'][group][key]))
    resources=[]
    for phase in PHASES:
        e=executions[phase];r=results[phase]['resources']
        resources.append(dict(phase=phase,pid=e['pid'],started_utc=e['started_utc'],completed_utc=e['completed_utc'],
            seconds=e['seconds'],host_peak_rss_bytes=r['host_peak_rss_bytes'],gpu_peak_allocated_bytes=r['gpu_peak_allocated_bytes']))
    return proteins,aggregates,resources


def csv_write(path,rows):
    with local(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figures(score,logs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    def save(fig,name):
        try:
            for ext in ('png','pdf'):
                path=output/(name+'.'+ext)
                with path.open('xb') as f:fig.savefig(f,format=ext,dpi=160)
                files.append(path)
        finally:plt.close(fig)
    colors=dict(pca='#1764ab',zero='#d28615',local='#26947c')
    horizons=(1,4,16,64);hc=('#1764ab','#d28615','#26947c','#9759a4')
    fig,axes=plt.subplots(3,2,figsize=(10,12),layout='constrained')
    for i,role in enumerate(ROLES):
        for j,arm in enumerate(ARMS[1:]):
            ax=axes[i,j];maximum=0.
            for h,color in zip(horizons,hc):
                rows=[r for r in score['per_protein'] if r['metadata']['horizon']==h]
                x=[r['scores']['pca/'+role]['log_mse'] for r in rows]
                y=[r['scores'][arm+'/'+role]['log_mse'] for r in rows]
                maximum=max(maximum,*x,*y);ax.scatter(x,y,s=20,alpha=.65,color=color,label='h'+str(h)+' (25)')
            end=maximum*1.04 if maximum else 1.
            ax.plot([0,end],[0,end],color='#777777',linestyle='--',linewidth=.8)
            ax.set(xlim=(0,end),ylim=(0,end),xlabel='PCA protein log-MSE',ylabel=arm+' protein log-MSE',
                   title=role+' | all 100 paired proteins')
            ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    fig.suptitle('Endpoint readout: future_H primary; observed/predicted roles secondary\n'
                 'Same fitted head per arm; all proteins retained, no event selection',fontsize=12)
    save(fig,'paired_log_mse')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for i,metric in enumerate(('mae_angstrom','rmse_angstrom')):
        for j,role in enumerate(ROLES):
            ax=axes[i,j]
            for arm in ARMS:
                ax.plot(range(4),[score['aggregates']['h'+str(h)][arm+'/'+role][metric] for h in horizons],
                        marker='o',label=arm,color=colors[arm])
            ax.plot(range(4),[score['aggregates']['h'+str(h)]['CA_persistence'][metric] for h in horizons],
                    '--o',color='#555555',label='observed CA persistence')
            ax.set(xticks=range(4),xticklabels=[str(h) for h in horizons],xlabel='Endpoint lag (ns)',
                   ylabel=('MAE' if i==0 else 'RMSE')+' (Å)',title=role)
            ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Physical endpoint errors: 25 equally weighted proteins per horizon\n'
                 'RMSE = sqrt(mean protein pair-MSE); secondary descriptors',fontsize=12)
    save(fig,'physical_errors_by_horizon')
    fig,axes=plt.subplots(3,4,figsize=(17,10),layout='constrained')
    for i,arm in enumerate(ARMS):
        train=logs['train_'+arm];fit=logs['fit_'+arm];x=[r['step'] for r in train]
        for key,label in [('loss/total','total'),('loss/base','JEPA base')]:
            axes[i,0].plot(x,[r['losses'][key] for r in train],label=label,linewidth=.7)
        axes[i,0].set_title(arm+': arm-local teacher objective')
        axes[i,0].legend(fontsize=8,frameon=False)
        for key,label in [('global_grad_norm_before_clip','global'),('encoder_grad_norm_before_clip','encoder')]:
            axes[i,1].plot(x,[r[key] for r in train],label=label,linewidth=.7)
        axes[i,1].set_title(arm+': gradient norm before clipping');axes[i,1].legend(fontsize=8,frameon=False)
        for k in range(7):axes[i,2].plot(x,[r['ranking_eligible_counts'][k] for r in train],linewidth=.6,label=str(k)+'→'+str(k+1))
        axes[i,2].set_title(arm+': eligible ranking pairs / history interval')
        axes[i,2].legend(fontsize=6,ncol=4,frameon=False)
        axes[i,3].plot([r['step'] for r in fit],[r['log_mse'] for r in fit],linewidth=.7,color=colors[arm])
        axes[i,3].set_title(arm+': fresh readout fit loss (in-sample)')
        for ax in axes[i]:ax.set_xlabel('Optimizer update');ax.grid(alpha=.15)
    fig.suptitle('All 500 updates, no smoothing or selected endpoint\n'
                 'Arm-local EMA targets differ: latent learning-loss curves do not rank representation quality',fontsize=12)
    save(fig,'learning_diagnostics')
    return files


def markdown(score,aggregates,resources,main_sha,verifier_sha):
    lines=['# Stage A relpos adaptation: verified endpoint readout','',LIMITS,'',
        '## Prespecified contrasts','',
        'Lower error is better. Intervals below are the already verified, fixed 10,000-resample paired '
        'horizon-stratified percentile intervals, not a new analysis. Both primary contrasts are retained; '
        'there is no multiplicity-adjusted or architecture-selection claim.','',
        '| Contrast | Role | Log-MSE difference | Descriptive 95% interval |','|---|---|---:|---|']
    for contrast in ('zero-pca','local-pca'):
        for role in ('future_H','predicted_H'):
            v=score['paired_bootstrap']['contrasts'][contrast][role]['log_mse']
            lines.append(f"| {contrast} | {role} ({'primary' if role=='future_H' else 'fixed secondary'}) | {v['difference']:.8g} | [{v['interval95'][0]:.8g}, {v['interval95'][1]:.8g}] |")
    lines+=['','## All roles and horizon groups','',
        'Each protein contributes its own mean pair error with equal protein weight. RMSE is the square '
        'root of the equal-protein mean pair-MSE, not the mean of per-protein roots. '
        'Negative distance predictions are retained without clamping and counted.','',
        '| Group | Arm / role | N proteins | Log-MSE | MAE / Å | RMSE / Å | Negative pairs |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['group']} | {r['role']} | {r['n_proteins']} | {r['log_mse']:.8g} | {r['mae_angstrom']:.8g} | {r['rmse_angstrom']:.8g} | {r['negative_count']} |")
    lines+=['','## Complete artifacts','',
        '- [All 100 proteins, paired scores](per_protein.csv); [all aggregates](aggregates.csv); '
        '[fixed bootstrap and full summary](metrics.json).',
        '- [All-protein paired log-MSE scatter](paired_log_mse.png) ([PDF](paired_log_mse.pdf)).',
        '- [Physical errors by horizon](physical_errors_by_horizon.png) ([PDF](physical_errors_by_horizon.pdf)).',
        '- [All learning, gradient and ranking-support traces](learning_diagnostics.png) ([PDF](learning_diagnostics.pdf)); '
        '[all 1,500 adaptation updates](adaptation_updates.csv); [all 1,500 readout updates](readout_updates.csv).','',
        'Training plots are diagnostics with separate arm-local teacher targets, not a comparison of latent quality. '
        'The readout-fit curves are in-sample. No temporal stability, complete structural reconstruction, '
        'Mori/FDT property, downstream compatibility or new Stage B forecast result is tested here.','',
        '## Actual execution resources','',
        '| Phase | PID | Seconds | Peak host RSS / GiB | Peak GPU allocation / GiB |',
        '|---|---:|---:|---:|---:|']
    for r in resources:lines.append(f"| {r['phase']} | {r['pid']} | {r['seconds']:.6f} | {r['host_peak_rss_bytes']/2**30:.4f} | {r['gpu_peak_allocated_bytes']/2**30:.4f} |")
    lines+=['','[Exact phase timestamps/resources](resources.csv). Memory maxima are per process, '
        'not additive across phases; elapsed phase seconds include that phase’s IO and checks.','',
        'The original checker registration failed on NumPy float32 JSON serialization before scoring; '
        '[the v1 registration failure is preserved](../stage_a_relpos_verification_v1/registration.execution.json). '
        'Checker v2 converts that scalar with float(...), preserving the exact same bound and numerical formulas. '
        'Its registration is before SCORE but after training, not a pre-training checker registration.','',
        f'Producer protocol SHA `{main_sha}`; verifier v2 protocol SHA `{verifier_sha}`. '
        '[Rendering manifest](report.json) and [actual execution](execution.json) record input/source/output hashes and timing. '
        'This renderer only parses verified JSON/JSONL, never NPZ, PyTorch tensors or checkpoints; '
        'the independent checker does not numerically replay encoder training. No rerun or new bootstrap is performed.','']
    return '\n'.join(lines)


def run(main_sha,verifier_sha):
    need(not OUTPUT.exists(),'report directory already exists; no retry/overwrite')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    sources={str(p.relative_to(ROOT)):sha(p) for p in
             (ROOT/'run_scripts/render_stage_a_relpos_adaptation_v1.py',ROOT/'tests/test_render_stage_a_relpos_adaptation_v1.py')}
    start=time.monotonic();identity=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),
        main_protocol_sha256=main_sha,verifier_protocol_sha256=verifier_sha,source_hashes=sources)
    write(OUTPUT/'started.json',identity);status=dict(identity);inputs=Inputs();phase='verify'
    try:
        protocol,results,executions,logs,verification=verified_inputs(inputs,main_sha,verifier_sha)
        score=results['score'];proteins,aggregates,resources=tables(score,results,executions);phase='render'
        artifacts=[]
        for name,rows in [('per_protein',proteins),('aggregates',aggregates),('resources',resources)]:
            path=OUTPUT/(name+'.csv');csv_write(path,rows);artifacts.append(path)
        trainrows=[];fitrows=[]
        for arm in ARMS:
            for r in logs['train_'+arm]:
                trainrows.append(dict(arm=arm,step=r['step'],**r['losses'],
                    global_grad_norm_before_clip=r['global_grad_norm_before_clip'],encoder_grad_norm_before_clip=r['encoder_grad_norm_before_clip'],
                    **{'ranking_history_'+str(k):r['ranking_eligible_counts'][k] for k in range(7)}))
            for r in logs['fit_'+arm]:fitrows.append(dict(arm=arm,**r))
        for name,rows in [('adaptation_updates',trainrows),('readout_updates',fitrows)]:
            path=OUTPUT/(name+'.csv');csv_write(path,rows);artifacts.append(path)
        path=OUTPUT/'metrics.json';write(path,dict(aggregates=score['aggregates'],paired_bootstrap=score['paired_bootstrap'],limits=LIMITS));artifacts.append(path)
        artifacts+=figures(score,logs,OUTPUT)
        path=OUTPUT/'REPORT.md'
        with path.open('x') as f:f.write(markdown(score,aggregates,resources,main_sha,verifier_sha))
        artifacts.append(path);inputs.finish()
        for path,digest in sources.items():need(sha(path)==digest,'renderer source changed')
        import matplotlib
        manifest=dict(schema_version=VERSION,status='completed',source_hashes=sources,input_hashes=inputs.hashes,
            outputs={str(p.relative_to(ROOT)):sha(p) for p in artifacts},rows=100,groups=5,arms=3,readout_roles=3,
            main_protocol_sha256=main_sha,verifier_protocol_sha256=verifier_sha,limits=LIMITS,
            environment=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__))
        write(OUTPUT/'report.json',manifest);status.update(status='completed',returncode=0,report_sha256=sha(OUTPUT/'report.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,failed_phase=phase,error=f'{type(error).__name__}: {error}');raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-start,input_hashes=inputs.hashes)
        write(OUTPUT/'execution.json',status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-protocol-sha',required=True);parser.add_argument('--verifier-protocol-sha',required=True)
    args=parser.parse_args();print(json.dumps(run(args.main_protocol_sha,args.verifier_protocol_sha)),flush=True)
