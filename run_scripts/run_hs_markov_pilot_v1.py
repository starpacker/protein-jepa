"""Independent C=0 training, reusing the previously fixed endpoint H+S arm.

Registration and execution are separate. Existing producers stay unchanged;
this runner never reruns or rewrites the reused H+S arm.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from run_scripts import run_hs_target_pilot as old
from run_scripts.continue_hs_target_pilot_v2 import parse_cli, assert_endpoint_fixed96
from scripts.eval_ca_endpoint_scores import load_model
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics, make_loader
from protein_jepa_frame.utils.contracts import state_sha256


RESULTS=Path('results/hs_joint_20261101/markov_v1')
ARTIFACTS=Path('artifacts/hs_joint_20261101/markov_v1')
DOCUMENT=Path('docs/HS_MARKOV_PILOT_V1_20261101.md')
TESTS=Path('tests/test_hs_markov_pilot_v1.py')
DECISION_SHA='3bdfca0bebaa37c21867cd8d3f5b00f6833ed13fbe125c1043377e85ba92c4fa'
S_CHECKPOINT=old.ARTIFACTS/'endpoint/checkpoint_latest.pt'
JOBS=('train','ensemble','endpoint')
VERSION='hs_functional_markov_c0_pilot_v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def exact_tree(actual,expected,label):
    if isinstance(expected,torch.Tensor):
        require(isinstance(actual,torch.Tensor) and actual.dtype==expected.dtype and
                actual.shape==expected.shape and torch.equal(actual,expected),label+' tensor differs')
    elif isinstance(expected,np.ndarray):
        require(isinstance(actual,np.ndarray) and actual.dtype==expected.dtype and
                actual.shape==expected.shape and np.array_equal(actual,expected),label+' array differs')
    elif isinstance(expected,dict):
        require(isinstance(actual,dict) and actual.keys()==expected.keys(),label+' keys differ')
        for key in expected:exact_tree(actual[key],expected[key],label+'/'+str(key))
    elif isinstance(expected,(list,tuple)):
        require(type(actual) is type(expected) and len(actual)==len(expected),label+' sequence differs')
        for index,(a,b) in enumerate(zip(actual,expected)):exact_tree(a,b,label+'/'+str(index))
    else:require(type(actual) is type(expected) and actual==expected,label+' value differs')


def commands():
    result={kind:list(old.commands('endpoint')[kind]) for kind in JOBS}
    train=result['train']
    train[train.index('--coupling-scale')+1]='0'
    train[train.index('--output')+1]=str(ARTIFACTS)
    for kind in JOBS[1:]:
        command=result[kind]
        command[command.index('--checkpoint')+1]=str(ARTIFACTS/'checkpoint_latest.pt')
        command[command.index('--out')+1]=str(RESULTS/f'markov.{kind}.json')
    endpoint=result['endpoint'];index=endpoint.index('--max-residues')
    require(endpoint[index+1]=='96','unexpected old endpoint CLI')
    del endpoint[index:index+2]
    return result


def effective_identity(model):
    return dict(kernel_tensor_sha256=state_sha256(model.kernel),kernel_config=asdict(model.kernel.cfg),
        bath_coupling_scale=model.kernel.bath.coupling_scale,
        s0_mode=model.kernel.s0_mode,history_conditioning=model.history_conditioning,
        collective_mobility=model.collective_mobility,noise_kernel=model.noise_kernel,
        noise_mix=model.noise_mix,noise_length=model.noise_length,noise_cutoff=model.noise_cutoff)


def selected_training_stream(args):
    """Resolve the existing stateless stream through metadata only, no __getitem__."""
    namespace=argparse.Namespace(**args)
    loaders={h:make_loader(namespace,args['manifest'],h,True) for h in args['horizons']}
    rows=[];stats={}
    for step in range(1,501):
        h=args['horizons'][(step-1)%len(args['horizons'])];dataset=loaders[h].dataset
        index=random.Random(args['seed']+1000003*step).randrange(len(dataset))
        record,frames=dataset._locate(index)
        rows.append(dict(step=step,horizon=h,dataset_index=index,protein=record.protein_id,
            trajectory=record.trajectory_id,record_path=str(record.path),temperature_k=record.temperature_k,
            context_frames=frames[:4].tolist(),target_frame=int(frames[-1])))
        stat=record.path.stat();stats[str(record.path)]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    return rows,stats


def initialization_audit():
    """No coordinates: recreate the exact two warm starts and compare all tensors."""
    s=CADynamics(str(old.INITIAL),'cpu',str(old.INITIAL),history_conditioning=True,
        noise_mix=.8,noise_length=8.,collective_mobility=True,noise_kernel='gaussian',noise_cutoff=10.,s0_mode='exact_spatial')
    m=copy.deepcopy(s)
    # Exact same override convention as the unchanged trainer.
    m.kernel.cfg.coupling_scale=0.;m.kernel.bath.coupling_scale=0.
    require(s.kernel.cfg.coupling_scale==.1 and s.kernel.bath.coupling_scale==.1,'source coupling differs')
    require(s.state_dict().keys()==m.state_dict().keys(),'initial parameter set differs')
    for name,value in s.state_dict().items():require(torch.equal(value,m.state_dict()[name]),'initial tensor differs: '+name)
    config_s,config_m=asdict(s.kernel.cfg),asdict(m.kernel.cfg)
    require({k:v for k,v in config_s.items() if k!='coupling_scale'}==
            {k:v for k,v in config_m.items() if k!='coupling_scale'},'unexpected configuration change')
    h=torch.zeros(1,4,s.kernel.cfg.latent_dim);temp=torch.tensor([320.])
    with torch.no_grad():ds,cs,gs=s.kernel.bath(h,temp);dm,cm,gm=m.kernel.bath(h,temp)
    require(torch.equal(ds,dm) and torch.equal(gs,gm) and torch.isfinite(cm).all() and torch.count_nonzero(cm)==0,
            'initial D/Gamma or zero C probe differs')
    names=[dict(name=n,shape=list(p.shape),numel=p.numel(),
        functional_c0_inactive=n.startswith(('bath.c_net.','bath.gamma_net.','history_posterior.')))
        for n,p in m.kernel.named_parameters()]
    return dict(all_initial_model_tensors_bitwise_equal=True,shared_initial_energy_D=True,
        s_identity=effective_identity(s),markov_identity=effective_identity(m),parameter_manifest=names,
        stored_kernel_parameters=sum(p.numel() for p in m.kernel.parameters()),
        caveat='equal stored parameter shapes, not equal effective capacity; unused bath parameters may receive zero gradients and AdamW decay')


def checkpoint_audit(markov,s_checkpoint,initial_identity):
    require(markov.get('checkpoint_kind')=='lasp_ca_dynamics' and markov.get('step')==500,'requires latest exactly500')
    args=markov['args'];expected=dict(s_checkpoint['args'],coupling_scale=0.,output=str(ARTIFACTS))
    require(old.canonical(args)==old.canonical(expected),'training differs beyond coupling_scale/output')
    expected_config=dict(s_checkpoint['kernel_config'],coupling_scale=0.)
    require(markov['kernel_config']==expected_config,'kernel configuration differs')
    require(old.component_identity(markov['component_checkpoint'])==initial_identity and markov.get('target_encoder') is None,
            'frozen component differs')
    exact_tree(markov['component_checkpoint'],s_checkpoint['component_checkpoint'],'complete embedded component')
    a,b=markov['kernel'],s_checkpoint['kernel']
    require(a.keys()==b.keys() and all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype for k in a),'parameter schema differs')
    require(all(torch.isfinite(v).all() for v in a.values()),'nonfinite saved kernel')
    require(torch.equal(markov['sampling_rng'],s_checkpoint['sampling_rng']),'final training sampling RNG differs')
    exact_tree(markov['rng_state'],s_checkpoint['rng_state'],'final global RNG')
    ma,sa=markov['optimizer'],s_checkpoint['optimizer']
    require(ma['param_groups']==sa['param_groups'],'optimizer groups/budget/hyperparameters differ')
    require(ma['state'].keys()==sa['state'].keys(),'optimizer state support differs')
    steps={}
    for key in ma['state']:
        ms,ss=ma['state'][key]['step'],sa['state'][key]['step']
        require(torch.equal(torch.as_tensor(ms),torch.as_tensor(ss)),'optimizer step budget differs')
        steps[str(key)]=int(ms)
    # The tensor hash alone excludes the scale/config; reload and verify both.
    restored=load_model(str(ARTIFACTS/'checkpoint_latest.pt'),markov,'cpu')
    require(restored.kernel.cfg.coupling_scale==0. and restored.kernel.bath.coupling_scale==0.,'reloaded C scale is nonzero')
    require(restored.kernel.cfg.bath_dim==8 and restored.kernel.s0_mode=='exact_spatial' and restored.history_conditioning,
            'reloaded dummy-bath/history/RNG contract differs')
    with torch.no_grad():
        _,c,_=restored.kernel.bath(torch.zeros(1,4,restored.kernel.cfg.latent_dim),torch.tensor([320.]))
    require(torch.isfinite(c).all() and torch.count_nonzero(c)==0,'reloaded C is not exactly zero')
    return dict(effective_identity=effective_identity(restored),component_identity=initial_identity,
        optimizer_steps_by_state=steps,optimizer_moments_not_compared=True,
        final_sampling_rng_bitwise_equal=True,final_global_rng_bitwise_equal=True,parameters_expected_to_diverge=True,
        kernel_tensor_sha256=state_sha256(a))


def verify(protocol,protocol_sha=None):
    if protocol_sha is not None:require(old.sha256_file(RESULTS/'protocol.json')==protocol_sha,'protocol changed')
    old.verify_files(dict(protocol['input_hashes'],**protocol['source_hashes']))
    for path,digest in protocol['source_hashes'].items():
        require(old.sha256_file(Path(protocol['source_snapshot'])/path)==digest,'source snapshot changed: '+path)
    for path,expected in protocol.get('selected_hdf5_stats',{}).items():
        actual=Path(path).stat()
        require(dict(size=actual.st_size,mtime_ns=actual.st_mtime_ns)==expected,'selected HDF5 changed: '+path)


def register(gpu):
    require(not ARTIFACTS.exists(),'artifact directory already exists')
    require(RESULTS.exists() and {p.name for p in RESULTS.iterdir()}=={'decision.json'},'requires only exclusive prior decision')
    require(old.sha256_file(RESULTS/'decision.json')==DECISION_SHA,'reuse decision changed')
    decision=json.loads((RESULTS/'decision.json').read_text());old.verify_files(decision['frozen_file_hashes'])
    original=json.loads((old.RESULTS/'protocol.json').read_text())
    old.verify_files(dict(original['input_hashes'],**original['source_hashes']))
    inputs=dict(original['input_hashes'],**decision['frozen_file_hashes'])
    for path,digest in original['source_hashes'].items():
        snapshot_path=Path(original['source_snapshot'])/path
        require(old.sha256_file(snapshot_path)==digest,'old source snapshot changed: '+path)
        inputs[str(snapshot_path)]=digest
    inputs[str(RESULTS/'decision.json')]=DECISION_SHA
    evidence={}
    continuation=json.loads((old.RESULTS/'continuation_v2/protocol.json').read_text())
    old.verify_files(dict(continuation['input_hashes'],**continuation['source_hashes']))
    inputs.update(continuation['input_hashes'])
    for path,digest in continuation['source_hashes'].items():
        snap=Path(continuation['source_snapshot'])/path
        require(old.sha256_file(snap)==digest,'old continuation snapshot changed: '+path)
        inputs[str(snap)]=digest
    # Wait for the existing complete10 logical jobs. Execution records can then
    # be pinned as immutable files without a dynamic-status exception.
    for arm in old.ARMS:
        first_path=old.RESULTS/f'{arm}.execution.json'
        last_path=old.RESULTS/'continuation_v2'/f'{arm}.execution.json'
        first=json.loads(first_path.read_text());last=json.loads(last_path.read_text())
        require(first['status']=='failed' and first.get('failed_utc') and last.get('completed_utc') and
                last.get('frozen_inputs_verified') is True,'old completion evidence differs')
        require([(j['kind'],j.get('returncode')) for j in first['jobs']]==[('train',0),('ensemble',0),('endpoint',2)],'old execution differs')
        require(last['status']=='completed' and [(j['kind'],j.get('returncode')) for j in last['jobs']]==
                [('endpoint',0),('suite',0),('matched8',0)],'existing10 logical jobs must finish before registration')
        for path in (first_path,last_path):inputs[str(path)]=old.sha256_file(path)
        for job in first['jobs']+last['jobs']:
            registration=original if job in first['jobs'] else continuation
            immutable=dict(registration['input_hashes'],**registration['source_hashes'])
            require(job['command']==registration['per_arm_commands'][arm][job['kind']],'old registered command differs')
            require(job.get('frozen_before')==immutable and job.get('frozen_after')==immutable,'old immutable audit differs')
            require(isinstance(job.get('pid'),int) and job['pid']>0 and job.get('completed_utc'),'missing real job PID/completion')
            require(old.sha256_file(job['log'])==job['log_sha256'],'old execution log changed')
            inputs[job['log']]=job['log_sha256']
        evidence[arm]=dict(original_execution=str(first_path),continuation_execution=str(last_path),logical_jobs_completed=5)
    for path in (old.RESULTS/'continuation_v2/protocol.json',old.STATISTICS,old.DOCUMENT):inputs[str(path)]=old.sha256_file(path)
    s_checkpoint=torch.load(S_CHECKPOINT,map_location='cpu',weights_only=False)
    identity=original['initial_component_identity']
    s_audit=old.audit_checkpoint(s_checkpoint,'endpoint',identity)
    original_run=json.loads((old.RESULTS/'endpoint.execution.json').read_text())
    require(old.sha256_file(S_CHECKPOINT)==original_run['latest_sha256'],'reused checkpoint differs from execution')
    require(old.sha256_file(old.RESULTS/'endpoint.ensemble.json')==original_run['jobs'][1]['output_sha256'],'reused ensemble differs')
    continued=json.loads((old.RESULTS/'continuation_v2/endpoint.execution.json').read_text())
    require(old.sha256_file(old.RESULTS/'endpoint.endpoint.json')==continued['jobs'][0]['output_sha256'],'reused endpoint differs')
    initial=initialization_audit();selection,stats=selected_training_stream(s_checkpoint['args'])
    old.assert_endpoint_default();assert_endpoint_fixed96()
    argv=commands();parsed={kind:parse_cli(command) for kind,command in argv.items()}
    require(parsed['train']['coupling_scale']==0. and parsed['train']['target']=='endpoint','bad train CLI')
    source_hashes=dict(continuation['source_hashes'])
    for path in (Path(__file__).resolve().relative_to(Path.cwd().resolve()),
                 DOCUMENT,TESTS,Path('run_scripts/continue_hs_target_pilot_v2.py')):
        source_hashes[str(path)]=old.sha256_file(path)
    require(gpu>=0 and torch.cuda.device_count()>gpu,'explicit GPU is unavailable')
    ARTIFACTS.mkdir(parents=True,exist_ok=False);snapshot=ARTIFACTS/'source_snapshot';snapshot.mkdir()
    for path,digest in source_hashes.items():
        data=Path(path).read_bytes();require(hashlib.sha256(data).hexdigest()==digest,'source changed before snapshot')
        dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(data)
    protocol=dict(schema_version=VERSION,registered_utc=old.now(),gpu=gpu,decision=decision,decision_sha256=DECISION_SHA,
        source_hashes=source_hashes,input_hashes=inputs,source_snapshot=str(snapshot),
        reused_checkpoint=str(S_CHECKPOINT),reused_checkpoint_audit=s_audit,original_execution_evidence=evidence,
        initial_component_identity=identity,initialization_audit=initial,commands=argv,parser_contracts=parsed,
        new_jobs=list(JOBS),optimizer_steps=500,n_samples=12,n_windows=200,n_proteins=50,
        statistics=decision['primary'],secondary=decision['secondary'],complete_suite_in_new_budget=False,
        interpretation='independently trained functional H-Markov conditional on fixed anchor/mask/T; retains unused S/history computations',
        inactive_parameters='keep original optimizer/parameter schema; C/gamma have zero predictive gradients and may undergo AdamW decay; history GRU unused in exact_spatial both arms',
        selected_train500=selection,selected_hdf5_stats=stats,
        historical_training_identity_limit='old S500 did not serialize training-row identities or HDF5 content hashes; replay proves same config/seed/frozen loader and current selected-file stability, not historical coordinate byte identity',
        rng='same unchanged exact_spatial S0 draw and each-step H/S draws; test actual H draws and generator state; require final training RNG equality',
        train_stream='same frozen stateless step_batch(seed42,step1..500); h=(1,4,16,64)[(step-1)%4]; each horizon125 updates',
        scoring='endpoint scores keep their U/V/Brier definitions; historical seven metrics remain future-path-mixture diagnostics',
        source_scores_read_by_registration=False,root_scores_unread_at_decision='attestation recorded in prior decision, not inferred from file timestamps',
        automatic_restart=False,test_opened=False,bioemu2_rerun=False)
    verify(protocol);old.write_new(RESULTS/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(RESULTS/'protocol.json'),sha256=old.sha256_file(RESULTS/'protocol.json'))),flush=True)


def run():
    path=RESULTS/'protocol.json';protocol=json.loads(path.read_text());protocol_sha=old.sha256_file(path)
    require(protocol['schema_version']==VERSION,'wrong protocol');verify(protocol,protocol_sha)
    execution=RESULTS/'execution.json'
    status=dict(status='running',runner_pid=os.getpid(),argv=sys.argv,started_utc=old.now(),protocol_sha256=protocol_sha,jobs=[])
    old.write_new(execution,status)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(protocol['gpu']),PYTHONPATH='/tmp/lasp_stability_deps:.',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    latest=ARTIFACTS/'checkpoint_latest.pt'
    try:
        for kind in JOBS:
            verify(protocol,protocol_sha);command=protocol['commands'][kind]
            if kind=='train':require(not latest.exists(),'training output already exists')
            else:
                require(old.sha256_file(latest)==status['latest_sha256'],'new checkpoint changed')
                output=RESULTS/f'markov.{kind}.json'
                require(not output.exists() and not output.with_suffix('.component.pt').exists(),'evaluation output exists')
            entry=dict(kind=kind,command=command,log=str(RESULTS/f'markov.{kind}.log'),started_utc=old.now())
            status['jobs'].append(entry);started=time.monotonic()
            with Path(entry['log']).open('x') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment)
                entry['pid']=process.pid;old.update_status(execution,status);code=process.wait()
            entry.update(returncode=code,completed_utc=old.now(),seconds=time.monotonic()-started,log_sha256=old.sha256_file(entry['log']))
            old.update_status(execution,status);verify(protocol,protocol_sha)
            if code:raise RuntimeError(f'{kind} exited{code}; no restart')
            if kind=='train':
                saved=torch.load(latest,map_location='cpu',weights_only=False)
                reused=torch.load(S_CHECKPOINT,map_location='cpu',weights_only=False)
                status['checkpoint_audit']=checkpoint_audit(saved,reused,protocol['initial_component_identity'])
                status['latest_sha256']=old.sha256_file(latest)
            else:
                require(old.sha256_file(latest)==status['latest_sha256'],'checkpoint changed during evaluation')
                entry['output_sha256']=old.sha256_file(output)
                if output.with_suffix('.component.pt').exists():entry['component_sha256']=old.sha256_file(output.with_suffix('.component.pt'))
            old.update_status(execution,status);print(kind+' completed exit0',flush=True)
        status.update(status='completed',completed_utc=old.now(),frozen_files_verified=True)
    except BaseException as error:
        status.update(status='failed',failed_utc=old.now(),error=f'{type(error).__name__}: {error}');raise
    finally:old.update_status(execution,status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['register','run'],required=True)
    p.add_argument('--gpu',type=int,default=0);args=p.parse_args()
    register(args.gpu) if args.mode=='register' else run()


if __name__=='__main__':main()
