"""Train32 fixed-covariance initial-history mean controls, CPU only.

Register before running. Only immutable microstep raw tensors supply observation
and target data. No training, new windows, identity-prior replacement, or tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.utils.contracts import state_sha256
from scripts.eval_ca_endpoint_scores import load_model
from run_scripts.diagnose_hs_microsteps_v1 import (
    REPO, CHECKPOINT, CHECKPOINT_SHA, coarsen, coordinate_scores, cpu_setup,
    decode_macro, macro_statistics, plain, require, save_new, sha, summarize, write_new,
    NumericalBranchFailure as MetricNumericalFailure,
)

BASE=REPO/'results/hs_joint_20261101/history_mean_train32_v1'
MICRO=REPO/'results/hs_joint_20261101/microstep_train32_v1'
MICRO_PROTOCOL_SHA='12d033a62a21046cadda39ba7f3035753d09922ee50fdb2b8db3f245d4297c34'
MICRO_RESULT_SHA='0bce069c70ac415b88f261e7710e4d78ad26fd1f3bc3855969c6c2e7ad11eee7'
DOCUMENT='docs/HS_HISTORY_MEAN_DIAGNOSTIC_V1_20261101.md'
TEST_SOURCE='tests/test_hs_history_mean_v1.py'
VERSION='hs_history_mean_fixed_covariance_train32_v1'
ARMS=('real_history','repeated_anchor_history','zero_mean')


def now():return datetime.now(timezone.utc).isoformat()


class CovarianceMismatch(RuntimeError):
    pass


class NumericalArmFailure(RuntimeError):
    pass


def mean_difference(mean,reference,chol,mask):
    """Whiten only the valid bath block: padded full-Cholesky rows are zero."""
    require(mean.shape==reference.shape and mean.shape[0]==1,'mean shape differs')
    ds=mean.shape[-1]
    residues=mask[0].nonzero(as_tuple=True)[0]
    index=(residues[:,None]*ds+torch.arange(ds)).flatten()
    delta=(mean.double()-reference.double())[0].reshape(-1)[index]
    lower=chol[0][index][:,index].double()
    standardized=torch.linalg.solve_triangular(lower,delta[:,None],upper=False).squeeze(-1)
    require(bool(torch.isfinite(standardized).all()),'nonfinite standardized mean difference')
    return dict(n_valid_bath_dimensions=len(index),raw_l2=float(delta.norm()),
        raw_rms=float(delta.square().mean().sqrt()),mahalanobis_squared=float(standardized.square().sum()),
        standardized_rms=float(standardized.square().mean().sqrt()),
        per_valid_dimension_mahalanobis_squared=float(standardized.square().mean()))


@torch.no_grad()
def posterior_controls(model,state):
    """No coordinate/target arguments: only frozen observed-H state inputs."""
    h0=state['anchor_H'];mask=state['mask'][0:1];temp=state['temp'][0:1]
    repeated=h0[:,None].expand(-1,4,-1,-1).contiguous()
    with sdpa_kernel(SDPBackend.MATH):
        repeated_mean,repeated_chol=model.kernel.initial_bath_state(
            repeated,state['dt_history'],temp,mask,seq_feature=h0,
            noise_factor=state['factor'][0:1],create_graph=False)
    original_chol=state['posterior_cholesky']
    if not torch.equal(repeated_chol,original_chol):
        raise CovarianceMismatch('repeated-anchor posterior Cholesky is not bitwise identical to real-history Cholesky')
    require(bool(torch.isfinite(repeated_mean).all()),'nonfinite repeated-anchor posterior mean')
    real=state['posterior_mean']
    means=dict(real_history=real,repeated_anchor_history=repeated_mean,zero_mean=torch.zeros_like(real))
    mean_stats={arm:mean_difference(value,real,original_chol,mask) for arm,value in means.items()}
    mean_stats['real_minus_zero']=mean_difference(real,torch.zeros_like(real),original_chol,mask)
    return dict(means=means,mean_differences=mean_stats,repeated_chol_bitwise_equal=True,
                repeated_history_sha256=state_sha256({'H':repeated}),
                repeated_chol_sha256=state_sha256({'chol':repeated_chol}))


@torch.no_grad()
def shared_initial_states(state,controls):
    chol=state['posterior_cholesky'];eps=state['posterior_standard_normal']
    mean=state['posterior_mean'];members=eps.shape[1]
    centered=torch.einsum('bij,bmj->bmi',chol,eps).reshape(1,members,*mean.shape[1:])
    states={arm:(value[:,None]+centered)[0] for arm,value in controls['means'].items()}
    require(torch.equal(states['real_history'],state['S0']),'real posterior draw differs from frozen microstep input')
    return centered,states


class TraceSink:
    def __init__(self,h,s):
        self.h=[h.clone()];self.s=[s.clone()];self.failure=None;self.failure_tensors=None

    def append(self,event):
        require(event['micro_step']==0 and event['macro_step']==len(self.h)-1,'unexpected m1 trace order')
        fields=('H_before','S_before','H_after','S_after')
        bad={key:(~torch.isfinite(event[key])).flatten(1).sum(1).tolist() for key in fields}
        if any(any(values) for values in bad.values()):
            self.failure=dict(stage='transition',macro_step=event['macro_step'],micro_step=0,
                              nonfinite_per_member=bad)
            self.failure_tensors={key:event[key].clone() for key in fields}
            raise NumericalArmFailure('nonfinite H/S transition')
        self.h.append(event['H_after'].clone());self.s.append(event['S_after'].clone())


@torch.no_grad()
def sample_arm(model,state,s0,zh,zs,horizon):
    d,c,gamma=model.kernel.bath(state['H0'],state['temp'])
    require(all(torch.equal(x,state[k]) for x,k in ((d,'D'),(c,'C'),(gamma,'Gamma'))),'frozen coefficients differ')
    sink=TraceSink(state['H0'],s0)
    try:
        with sdpa_kernel(SDPBackend.MATH):
            h,s=model.kernel.rollout_microstep_v1(state['H0'],state['mask'],state['temp'],horizon,
                S0=s0,seq_feature=state['H0'],detach=True,return_bath=True,
                noise_factor=state['factor'],collective_mobility=True,microsteps=1,
                brownian_H=zh,brownian_S=zs,integrator='explicit',microstate_trace=sink)
        require(torch.equal(h,torch.stack(sink.h,1)) and torch.equal(s,torch.stack(sink.s,1)),
                'macro trace differs from sampler output')
        status='completed'
    except NumericalArmFailure:
        h,s=torch.stack(sink.h,1),torch.stack(sink.s,1);status='numerical_failure'
    raw=dict(H=h,S=s,S0=s0,failure_tensors=sink.failure_tensors)
    result=dict(status=status,failure=sink.failure,completed_macro_steps=h.shape[1]-1,
        macro_times_ps=[1000*i for i in range(h.shape[1])],macro_statistics=macro_statistics(h,s,state['mask'][0]))
    if status=='completed':
        ca=decode_macro(model,state,h);raw['CA']=ca
        if not bool(torch.isfinite(ca[:,:,state['mask'][0]]).all()):
            result.update(status='numerical_failure',failure=dict(stage='CA_decode',
                nonfinite_per_member_macro=(~torch.isfinite(ca[:,:,state['mask'][0]])).flatten(2).sum(-1).tolist()))
    return result,raw


def verify_real_replay(result,raw,original):
    require(result['status']=='completed','real-history replay unexpectedly failed numerically')
    for key in ('H','S','CA'):
        require(torch.equal(raw[key],original[key]),f'real-history {key} does not reproduce frozen m1 bitwise')


def attach_scores(result,samples,target,anchor,mask,ri,ci):
    result['endpoint_scores']=None
    if result['status']=='completed':
        try:result['endpoint_scores']=coordinate_scores(samples,target,anchor,mask,ri,ci)
        except MetricNumericalFailure as error:
            result.update(status='numerical_failure',failure=dict(stage='endpoint_metric',error=str(error)))


def registered_sources(micro_protocol):
    paths=set(micro_protocol['source_hashes'])|{str(Path(__file__).resolve().relative_to(REPO)),DOCUMENT,TEST_SOURCE}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_relative_to(REPO) and path.suffix=='.py' and path.is_file():paths.add(str(path.relative_to(REPO)))
    return {path:sha(REPO/path) for path in sorted(paths)}


def register():
    cpu_setup();require(not BASE.exists(),'refuse existing history-mean experiment')
    require(sha(MICRO/'protocol.json')==MICRO_PROTOCOL_SHA and sha(MICRO/'diagnostic.json')==MICRO_RESULT_SHA,
            'registered microstep provenance differs')
    old=json.loads((MICRO/'protocol.json').read_text());report=json.loads((MICRO/'diagnostic.json').read_text())
    execution=json.loads((MICRO/'execution.json').read_text())
    require(execution['returncode']==0 and execution['numerical_failure_branches']==0,'original microstep experiment incomplete')
    require(len(report['per_window'])==32 and len(old['selection'])==32,'fixed32 count differs')
    require(Counter(r['horizon'] for r in old['selection'])==Counter({1:8,4:8,16:8,64:8}), 'horizon panel differs')
    require(len({r['protein'] for r in old['selection']})==32,'protein panel differs')
    inputs=dict(old['input_hashes']);inputs.update({str((MICRO/name).relative_to(REPO)):sha(MICRO/name)
        for name in ('protocol.json','diagnostic.json','execution.json','independent_verification.json',
                     'verification.execution.json','initial_state_and_verifier_hash_check.json')})
    input_rows=[]
    for selection,row in zip(old['selection'],report['per_window']):
        require(selection==row['metadata'],'selected row differs')
        path=row['raw_tensors']['path'];digest=row['raw_tensors']['sha256']
        require(sha(REPO/path)==digest,'original raw tensor changed')
        inputs[path]=digest;input_rows.append(dict(metadata=selection,path=path,sha256=digest))
    for path,digest in old['source_hashes'].items():require(sha(REPO/path)==digest,'frozen source changed: '+path)
    for path,digest in inputs.items():require(sha(REPO/path)==digest,'input changed: '+path)
    sources=registered_sources(old)
    protocol=dict(schema_version=VERSION,registered_utc=now(),device='cpu',torch_threads=2,
        checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,model_tensor_sha256=old['model_tensor_sha256'],
        source_hashes=sources,input_hashes=inputs,input_rows=input_rows,selection=old['selection'],
        n_windows=32,n_proteins=32,n_samples=12,horizons=[1,4,16,64],microsteps=1,arms=list(ARMS),
        observation='exactly original four observed H frames; repeated control replaces all four by same anchor H0',
        covariance='all three arms keep original real-history full P0 Cholesky; strict bitwise equality of repeated-history factor must pass all32 preflight before any rollout',
        epsilon_and_brownian='reuse original posterior epsilon and fine8 H/S; coarsen by sum/sqrt(8) to original m1; no new RNG draws',
        initial_states='real m0, repeated-anchor posterior mean, zero mean; all plus one identically computed original Chol*eps',
        zero_control='matched original full covariance; NOT legacy prior with identity covariance',
        replay='real arm must reproduce original complete m1 H/S/CA tensors bitwise, otherwise abort experiment',
        mean_difference='relative to real mean; valid-bath principal Cholesky triangular solve; raw L2/RMS, Mahalanobis squared, per-valid-bath-dimension mean and RMS',
        failure='retain32 rows and3 arms; any failed member makes whole arm score null; no partial ensemble score; covariance/replay/source/interface errors abort',
        scores='same frozen microstep helpers: float64 endpoint target-to-anchor Kabsch, coordinate U-ES, member RMSD, CA geometry; fixed original persistence',
        aggregation='within-window member means, then within-protein then equal-protein, all horizons and overall; explicit full/support counts; paired control-minus-real deltas; no CI',
        label_use='original saved future endpoint only for scoring after all arm sampling; never posterior, covariance, initial state or rollout input',
        adaptive_status='same previously examined train32 windows, follow-up registered before new mean-control sampling; not unseen-data validation',
        limitations=['tests initialization mean/history information only, not separately trained S-versus-Markov benefit',
                    'repeated-anchor history is artificial and may be off-distribution',
                    'all horizons use different eight proteins; not a matched temporal curve',
                    'no best arm/step/checkpoint selection, no full-goal or physical-memory claim'],
        optimizer_steps=0,new_molecular_coordinates_read=False,test_opened=False,validation_opened=False,bioemu2_rerun=False,
        versions=dict(torch=torch.__version__,python=sys.version))
    BASE.mkdir(parents=True,exist_ok=False);snapshot=BASE/'source_snapshot';snapshot.mkdir()
    for path,digest in sources.items():
        data=(REPO/path).read_bytes();require(hashlib.sha256(data).hexdigest()==digest,'source changed before snapshot')
        dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(data)
    write_new(BASE/'protocol.json',protocol)
    print(json.dumps(dict(registered=str(BASE/'protocol.json'),sha256=sha(BASE/'protocol.json'))),flush=True)


def verify_files(protocol,protocol_sha):
    require(sha(BASE/'protocol.json')==protocol_sha,'protocol changed')
    for field in ('input_hashes','source_hashes'):
        for path,digest in protocol[field].items():
            require(sha(REPO/path)==digest,'registered file changed: '+path)
            if field=='source_hashes':require(sha(BASE/'source_snapshot'/path)==digest,'snapshot changed: '+path)


def preflight_all(model,protocol):
    controls=[]
    for item in protocol['input_rows']:
        raw=torch.load(REPO/item['path'],map_location='cpu',weights_only=False)
        require(raw['metadata']==item['metadata'],'raw metadata differs')
        try:control=posterior_controls(model,raw['state'])
        except CovarianceMismatch as error:
            write_new(BASE/'covariance_preflight_failure.json',dict(metadata=item['metadata'],error=str(error),
                already_checked_windows=len(controls),sampling_started=False))
            raise
        control['metadata']=item['metadata'];controls.append(control)
        del raw
    save_new(BASE/'mean_controls.pt',controls)
    write_new(BASE/'covariance_preflight.json',dict(all32_bitwise_equal=True,n_windows=32,sampling_started=False,
        means_tensor_sha256=sha(BASE/'mean_controls.pt'),per_window=[dict(metadata=c['metadata'],
            repeated_chol_sha256=c['repeated_chol_sha256'],mean_differences=c['mean_differences']) for c in controls]))
    print('All32 repeated-history Cholesky factors exactly equal; starting three-arm sampling',flush=True)
    return controls


def execute(protocol,protocol_sha):
    checkpoint=torch.load(REPO/CHECKPOINT,map_location='cpu',weights_only=False)
    require(sha(REPO/CHECKPOINT)==CHECKPOINT_SHA,'checkpoint hash differs')
    model=load_model(str(REPO/CHECKPOINT),checkpoint,'cpu').eval().requires_grad_(False)
    require(state_sha256(model)==protocol['model_tensor_sha256'],'model state differs')
    require(model.kernel.s0_mode=='exact_spatial' and model.kernel.cfg.step_ps==1000.,'sampler contract differs')
    controls=preflight_all(model,protocol);rows=[];started=time.monotonic()
    for item,control in zip(protocol['input_rows'],controls):
        old=torch.load(REPO/item['path'],map_location='cpu',weights_only=False)
        state=old['state'];mask=state['mask'][0];metadata=item['metadata']
        centered,s0=shared_initial_states(state,control)
        zh,zs=coarsen(state['fine_H'],1),coarsen(state['fine_S'],1)
        raw=dict(metadata=metadata,source_raw=item,controls=control,shared_centered_bath=centered,
            H0=state['H0'],original_full_chol=state['posterior_cholesky'],posterior_epsilon=state['posterior_standard_normal'],
            observation=old['observed'],brownian_H=zh,brownian_S=zs,
            condition={key:state[key] for key in ('mask','temp','D','C','Gamma','factor','anchor_H',
                'rotation','anchor','metadata','history','dt_history')},arms={})
        row=dict(metadata=metadata,mean_differences=control['mean_differences'],repeated_chol_bitwise_equal=True,
            shared_input_sha256=state_sha256({'H0':state['H0'],'chol':state['posterior_cholesky'],
                'eps':state['posterior_standard_normal'],'brownian_H':zh,'brownian_S':zs,'factor':state['factor'],
                'D':state['D'],'C':state['C'],'Gamma':state['Gamma'],'temp':state['temp']}),arms={},scalars={})
        for arm in ARMS:
            result,tensors=sample_arm(model,state,s0[arm],zh,zs,metadata['horizon'])
            if arm=='real_history':verify_real_replay(result,tensors,old['branches']['1'])
            row['arms'][arm]=result;raw['arms'][arm]=tensors
        # Only now are the saved target tensors used for any computation.
        target=old['target']['coords'][0,:,1];context=old['observed']
        ri,ci=context['residue_index'],context['chain_id'];anchor=state['anchor'][0]
        raw['target']=old['target']
        for arm in ARMS:
            result=row['arms'][arm]
            samples=raw['arms'][arm]['CA'][:,-1] if result['status']=='completed' else None
            attach_scores(result,samples,target,anchor,mask,ri,ci)
            for key in ('coordinate_energy_score','member_rmsd'):
                row['scalars'][f'{arm}/{key}']=result['endpoint_scores'][key] if result['endpoint_scores'] else None
            for key in ('channel_mean_squared','channel_variance_mean','norm_squared_per_dim','bath_norm_squared_per_dim'):
                row['scalars'][f'{arm}/{key}']=float(result['macro_statistics'][key][:,-1].mean()) if result['status']=='completed' else None
            if result['endpoint_scores']:
                for key,value in result['endpoint_scores']['geometry_mean'].items():row['scalars'][f'{arm}/geometry/{key}']=value
            for key,value in control['mean_differences'][arm].items():row['scalars'][f'{arm}/initial_mean/{key}']=value
        row['persistence']=coordinate_scores(state['anchor'].expand(12,-1,-1),target,anchor,mask,ri,ci)
        for key in ('coordinate_energy_score','member_rmsd'):row['scalars'][f'persistence/{key}']=row['persistence'][key]
        for key,value in row['persistence']['geometry_mean'].items():
            row['scalars'][f'persistence/geometry/{key}']=value
            for arm in ARMS:row['scalars'].setdefault(f'{arm}/geometry/{key}',None)
        metric_suffixes=[k.removeprefix('real_history/') for k in row['scalars'] if k.startswith('real_history/') and '/initial_mean/' not in k]
        for arm in ARMS[1:]:
            for key in metric_suffixes:
                a,b=row['scalars'][f'{arm}/{key}'],row['scalars'][f'real_history/{key}']
                row['scalars'][f'{arm}_minus_real/{key}']=a-b if a is not None and b is not None else None
        row['real_full_path_bitwise_replay']=True
        path=BASE/f"window_{metadata['step']:02d}.pt";save_new(path,raw)
        row['raw_tensors']=dict(path=str(path.relative_to(REPO)),sha256=sha(path))
        write_new(BASE/f"window_{metadata['step']:02d}.json",row);rows.append(plain(row))
        print(json.dumps(dict(completed=len(rows),total=32,statuses={a:r['status'] for a,r in row['arms'].items()},
            seconds=time.monotonic()-started)),flush=True)
        del old,raw,state
    require(len(rows)==32,'fixed panel incomplete')
    require(state_sha256(model)==protocol['model_tensor_sha256'] and all(p.grad is None for p in model.parameters()),'model changed')
    require(not torch.cuda.is_initialized(),'CUDA initialized')
    verify_files(protocol,protocol_sha)
    result=dict(schema_version=VERSION,protocol_sha256=protocol_sha,n_windows=32,n_proteins=32,
        covariance_preflight_all32_bitwise_equal=True,real_all32_H_S_CA_bitwise_replay=True,
        source_input_snapshot_model_before_after_verified=True,
        numerical_failure_arms=sum(a['status']!='completed' for r in rows for a in r['arms'].values()),
        per_window=rows,overall=summarize(rows),by_horizon={str(h):summarize([r for r in rows if r['metadata']['horizon']==h]) for h in (1,4,16,64)},
        completed_utc=now())
    write_new(BASE/'diagnostic.json',result)
    return result


def run():
    cpu_setup();data=(BASE/'protocol.json').read_bytes();digest=hashlib.sha256(data).hexdigest();protocol=json.loads(data)
    require(protocol['schema_version']==VERSION,'wrong protocol version')
    require(not (BASE/'execution.started.json').exists(),'no automatic restart')
    verify_files(protocol,digest)
    write_new(BASE/'execution.started.json',dict(pid=os.getpid(),started_utc=now(),argv=sys.argv,protocol_sha256=digest))
    started=time.monotonic()
    try:result=execute(protocol,digest)
    except BaseException as error:
        code=130 if isinstance(error,KeyboardInterrupt) else 1
        write_new(BASE/'execution.json',dict(pid=os.getpid(),returncode=code,seconds=time.monotonic()-started,
            error=f'{type(error).__name__}: {error}',completed_utc=now()))
        raise
    write_new(BASE/'execution.json',dict(pid=os.getpid(),returncode=0,seconds=time.monotonic()-started,
        output_sha256=sha(BASE/'diagnostic.json'),numerical_failure_arms=result['numerical_failure_arms'],completed_utc=now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['register','run'],required=True)
    args=parser.parse_args();register() if args.mode=='register' else run()


if __name__=='__main__':main()
