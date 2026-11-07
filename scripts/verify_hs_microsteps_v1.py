"""Independent NumPy verification of the registered train32 microstep diagnostic.

Does not import the producer or its statistics/budget/coarsening functions.
Register before reading any generated rows; run only after producer exit0.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.utils.contracts import state_sha256
from scripts.eval_ca_endpoint_scores import load_model

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'results/hs_joint_20261101/microstep_train32_v1'
PROTOCOL_SHA='12d033a62a21046cadda39ba7f3035753d09922ee50fdb2b8db3f245d4297c34'
VERSION='independent_numpy_hs_microsteps_verification_v1'
KEYS=('force','bath','drift_squared','noise_trace','expected','actual','stochastic_residual',
      'noise_cross','noise_squared_minus_trace','rounding_closure')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1<<20),b''):h.update(data)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise AssertionError(message)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def array(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


class Audit:
    def __init__(self):self.counts=defaultdict(int);self.max_abs=defaultdict(float)

    def numeric(self,actual,expected,category,*,atol=2e-9,rtol=3e-10):
        a=np.asarray(actual,dtype=np.float64);e=np.asarray(expected,dtype=np.float64)
        require(a.shape==e.shape,f'{category}: shape mismatch {a.shape} vs {e.shape}')
        require(np.isfinite(a).all() and np.isfinite(e).all(),f'{category}: nonfinite compared quantity')
        np.testing.assert_allclose(a,e,atol=atol,rtol=rtol,err_msg=category)
        self.counts[category]+=int(a.size)
        self.max_abs[category]=max(self.max_abs[category],float(np.max(np.abs(a-e),initial=0.)))

    def nested(self,actual,expected,category):
        if isinstance(expected,dict):
            require(set(actual)==set(expected),f'{category}: keys differ')
            for key in expected:self.nested(actual[key],expected[key],category)
        elif isinstance(expected,(str,bool)) or expected is None:
            require(actual==expected,f'{category}: exact field differs')
        elif isinstance(expected,(list,tuple)) and expected and isinstance(expected[0],dict):
            require(len(actual)==len(expected),f'{category}: list count differs')
            for a,e in zip(actual,expected):self.nested(a,e,category)
        else:self.numeric(actual,expected,category)


def macro_statistics(h,s,mask):
    h=h[:,:,mask];s=s[:,:,mask]
    mean=np.mean(h,axis=-1);var=np.var(h,axis=-1,ddof=0)
    return dict(channel_mean_abs=np.mean(np.abs(mean),axis=-1),channel_mean_squared=np.mean(mean**2,axis=-1),
        channel_variance_mean=np.mean(var,axis=-1),centered_radius_rms=np.sqrt(np.mean(var,axis=-1)),
        norm_squared_per_dim=np.mean(h*h,axis=(-1,-2)),bath_norm_squared_per_dim=np.mean(s*s,axis=(-1,-2)),
        bath_l2_norm=np.sqrt(np.sum(s*s,axis=(-1,-2))),n_valid_residues=int(mask.sum()))


def difference(a,b,mask,coordinates):
    d=(a[:,:,mask]-b[:,:,mask])**2
    values=np.sqrt(np.mean(np.sum(d,axis=-1),axis=-1) if coordinates else np.mean(d,axis=(-1,-2)))
    return dict(per_member_per_macro=values,mean_per_macro=values.mean(0),endpoint_mean=float(values[:,-1].mean()),
        units='angstrom RMSD per residue' if coordinates else 'raw H RMS per component')


def geometry(ca,mask,ri,ci):
    adjacent=mask[1:]&mask[:-1]&(ci[1:]==ci[:-1])&(ri[1:]-ri[:-1]==1)
    lengths=np.linalg.norm(ca[1:]-ca[:-1],axis=-1)[adjacent]
    i,j=np.triu_indices(len(ca),k=1)
    keep=mask[i]&mask[j]&((ci[i]!=ci[j])|(np.abs(ri[i]-ri[j])>=3))
    distances=np.linalg.norm(ca[i[keep]]-ca[j[keep]],axis=-1)
    # Frozen geometry computes these Boolean fraction reductions in float32.
    return dict(ca_bond_mean_angstrom=float(lengths.mean()) if len(lengths) else None,
        ca_bond_invalid_fraction=float(((lengths<3.3)|(lengths>4.3)).astype(np.float32).mean()) if len(lengths) else None,
        ca_nonlocal_clash_fraction=float((distances<2.5).astype(np.float32).mean()) if len(distances) else None,
        ca_radius_gyration_angstrom=float(np.sqrt(np.mean(np.sum((ca[mask]-ca[mask].mean(0))**2,axis=-1)))))


def scores(samples,target,anchor,mask,ri,ci):
    p=target[mask];q=anchor[mask]
    pc=p-p.mean(0);qc=q-q.mean(0)
    u,_,vh=np.linalg.svd(pc.T@qc)
    rotation=u@np.diag([1.,1.,np.sign(np.linalg.det(u@vh))])@vh
    truth=pc@rotation+q.mean(0)
    x=samples[:,mask]
    mse=np.mean(np.sum((x-truth)**2,axis=-1),axis=-1)
    flat=x.reshape(len(x),-1);y=truth.reshape(-1)
    i,j=np.triu_indices(len(x),k=1)
    cross=np.linalg.norm(flat-y,axis=-1).mean()/math.sqrt(flat.shape[1])
    diversity=np.linalg.norm(flat[i]-flat[j],axis=-1).mean()/math.sqrt(flat.shape[1])
    geom=[geometry(ca,mask,ri,ci) for ca in samples]
    return dict(coordinate_energy_score=float(cross-.5*diversity),member_rmsd=float(np.sqrt(mse).mean()),
        per_sample_rmsd=np.sqrt(mse),geometry_per_sample=geom,
        geometry_mean={k:float(np.mean([g[k] for g in geom])) if all(g[k] is not None for g in geom) else None for k in geom[0]})


def aggregate(rows):
    result={}
    for key in sorted(set().union(*(row['scalars'] for row in rows))):
        groups=defaultdict(list)
        for row in rows:
            value=row['scalars'].get(key)
            if value is not None:groups[row['metadata']['protein']].append(value)
        per_protein={name:float(np.mean(values)) for name,values in groups.items()}
        count=sum(len(v) for v in groups.values());complete=count==len(rows)
        mean=float(np.mean(list(per_protein.values()))) if per_protein else None
        result[key]=dict(equal_protein_mean=mean if complete else None,available_subset_mean=mean,
            complete=complete,n_windows=count,expected_windows=len(rows),n_proteins=len(per_protein),per_protein=per_protein)
    return result


def numpy_budget(h,after,f,s,d,c,factor,dt,z,mask):
    root=np.sqrt(d);k=factor@np.swapaxes(factor,-1,-2)
    apply=lambda v:root*np.einsum('bij,bjc->bic',k,root*v)
    valid=mask[...,None]
    mf=apply(f)*valid;mcs=apply(np.einsum('blhs,bls->blh',c,s))*valid
    b=mf-mcs;eta=np.sqrt(2*dt*d)*np.einsum('bij,bjc->bic',factor,z)*valid
    trace=2*dt*np.sum(d*np.diagonal(k,axis1=-2,axis2=-1)[...,None]*valid,axis=(-1,-2))
    result={}
    for name in ('raw','centered'):
        project=(lambda x:x-x.mean(-1,keepdims=True)) if name=='centered' else (lambda x:x)
        hp,ap,fp,cp,bp,ep=[project(x) for x in (h*valid,after*valid,mf,mcs,b,eta)]
        inner=lambda x,y:np.sum(x*y,axis=(-1,-2))
        force=2*dt*inner(hp,fp);bath=-2*dt*inner(hp,cp);drift=dt**2*inner(bp,bp)
        nt=trace*(1-1/h.shape[-1]) if name=='centered' else trace
        expected=force+bath+drift+nt;actual=inner(ap,ap)-inner(hp,hp)
        cross=2*inner(hp+dt*bp,ep);sq=inner(ep,ep)-nt
        result[name]=dict(force=force,bath=bath,drift_squared=drift,noise_trace=nt,expected=expected,
            actual=actual,stochastic_residual=actual-expected,noise_cross=cross,
            noise_squared_minus_trace=sq,rounding_closure=actual-expected-cross-sq)
    return result


def replay_selected(model,raw,audit):
    st=raw['state'];steps=2;result={}
    d,c,gamma=model.kernel.bath(st['H0'],st['temp'])
    require(all(torch.equal(value,st[key]) for value,key in ((d,'D'),(c,'C'),(gamma,'Gamma'))),'actual coefficients differ')
    for count in (1,4,8):
        h=st['H0'].clone();s=st['S0'].clone();group=8//count
        # Independent reconstruction of the registered explicit-noise contract;
        # all NumPy moment checks below additionally verify coarse increments.
        zs=[]
        for key in ('fine_H','fine_S'):
            fine=st[key]
            zs.append(fine.reshape(*fine.shape[:2],count,group,*fine.shape[3:]).sum(3)/math.sqrt(group))
        checked=0
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            for macro in range(steps):
                for micro in range(count):
                    index=macro*count+micro
                    saved=raw['branches'][str(count)]
                    if index>=saved['budgets']['raw']['actual'].shape[0]:break
                    force=model.kernel.compute_forces(h,st['mask'],st['temp'],create_graph=False).detach()
                    hn,sn=model.kernel._collective_velocity_step_fixed(h,s,st['mask'],st['temp'],d,c,gamma,
                        1000/count,noise_H=zs[0][:,macro,micro],noise_S=zs[1][:,macro,micro],noise_factor=st['factor'])
                    expected=numpy_budget(*[array(x) for x in (h,hn,force,s,d,c,st['factor'])],1000/count,
                        array(zs[0][:,macro,micro]),st['mask'].numpy())
                    for space in expected:
                        for key,value in expected[space].items():
                            audit.numeric(saved['budgets'][space][key][index].numpy(),value,'actual_force_budget_replay')
                    h,s=hn,sn;checked+=1
                else:
                    require(torch.equal(h,saved['H'][:,macro+1]) and torch.equal(s,saved['S'][:,macro+1]),
                            'actual frozen primitive macro replay differs bitwise')
                    audit.counts['actual_primitive_bitwise_replay']+=h.numel()+s.numel()
                    continue
                break
        result[str(count)]=dict(microsteps_checked=checked,requested_macro_steps=steps,
            complete=checked==steps*count,actual_primitive_macro_equal=True)
    return result


def register():
    require(sha(BASE/'protocol.json')==PROTOCOL_SHA,'producer protocol hash differs')
    protocol=json.loads((BASE/'protocol.json').read_text())
    selected=next(row for row in protocol['selection'] if row['horizon']==64)
    require(selected['step']==4,'first h64 selection differs')
    write(BASE/'verification.protocol.json',dict(schema_version=VERSION,registered_utc=datetime.now(timezone.utc).isoformat(),
        verifier_source=str(Path(__file__).resolve().relative_to(REPO)),verifier_sha256=sha(__file__),
        producer_protocol_sha256=PROTOCOL_SHA,replay_selection=selected,replay_macro_steps=2,replay_microsteps=[1,4,8],
        selection_rule='first h64 in already fixed train32 schedule, step4; fixed before reading generated rows/scores',
        verification_scope='all32 macro statistics, path differences, per-step budget identities/totals, full aggregation; Brownian coupling/RNG and S0 posterior draw; hashes; first2macro actual force replay',
        arithmetic='independent NumPy float64; canonical Boolean CA fractions retain float32 rounding',
        tolerances=dict(general_atol=2e-9,general_rtol=3e-10,posterior_draw_atol=2e-6,posterior_draw_rtol=3e-6,
                        brownian_roundoff_multiplier=16),
        raw_results_read_before_registration=False,no_target_arm_scores_read=True,
        sampling=False,optimizer_steps=0,test_opened=False,device='cpu',torch_threads=2))
    print('Verification registered: fixed step4 / first2macro / m1,4,8',flush=True)


def verify_window(raw,row,audit):
    st=raw['state'];mask=st['mask'][0].numpy();length=int(mask.sum());dim=st['H0'].shape[-1]
    require(raw['metadata']==row['metadata'],'row metadata differs')
    require(set(row['branches'])=={'1','4','8'} and set(raw['branches'])=={'1','4','8'},'branch set differs')
    require(st['H0'].shape==(12,len(mask),128) and st['S0'].shape==(12,len(mask),8),'member/latent budget differs')
    require(torch.equal(st['mask'],st['mask'][0:1].expand_as(st['mask'])),'member masks differ')
    require(torch.equal(st['H0'],st['anchor_H'].expand_as(st['H0'])),'member H0 differs')
    mean=array(st['posterior_mean'])[0];chol=array(st['posterior_cholesky'])[0];eps=array(st['posterior_standard_normal'])[0]
    expected=(eps@chol.T).reshape(st['S0'].shape)+mean
    audit.numeric(array(st['S0']),expected,'full_posterior_draw',atol=2e-6,rtol=3e-6)
    generator=torch.Generator(device='cpu').manual_seed(row['metadata']['sampling_seed'])
    require(torch.equal(generator.get_state(),st['rng_initial']),'initial RNG state differs')
    for key in ('posterior_standard_normal','fine_H','fine_S'):
        expected_tensor=torch.randn(st[key].shape,dtype=st[key].dtype,generator=generator)
        require(torch.equal(st[key],expected_tensor),f'explicit RNG replay differs: {key}')
        audit.counts['rng_bitwise_replay']+=expected_tensor.numel()
    require(torch.equal(generator.get_state(),st['rng_final']),'final RNG state differs')
    for key in ('fine_H','fine_S'):
        fine=array(st[key]);reference=fine.sum(2)*math.sqrt(1000/8)
        for count in (1,4,8):
            group=8//count
            coarse=fine.reshape(*fine.shape[:2],count,group,*fine.shape[3:]).sum(3)/math.sqrt(group)
            audit.numeric(coarse.sum(2)*math.sqrt(1000/count),reference,'brownian_physical_increment',atol=2e-10,rtol=2e-12)
            # Float32 coarse increments are the actual arithmetic used by the sampler.
            ft=st[key];ct=ft.reshape(*ft.shape[:2],count,group,*ft.shape[3:]).sum(3)/math.sqrt(group)
            actual=array(ct).sum(2)*math.sqrt(1000/count)
            bound=16*np.finfo(np.float32).eps*np.maximum(1.,np.abs(fine).sum(2)*math.sqrt(1000/8))
            require(np.all(np.abs(actual-reference)<=bound),'coarse float32 Brownian error exceeds fixed rounding bound')
    context=raw['observed'];target=raw['target']
    require(context['coords'].shape[0]==4 and target['coords'].shape[0]==1,'observation/label frame count differs')
    require(float(target['time_ps'][0]-context['time_ps'][-1])==1000*row['metadata']['horizon'],'endpoint physical duration differs')
    require(state_sha256({k:context[k] for k in ('coords','atom_mask','residue_mask','time_ps','aatype','residue_index','chain_id')})==row['observed_sha256'],'observation fingerprint differs')
    require(state_sha256({k:target[k] for k in ('coords','atom_mask','time_ps')})==row['label_sha256'],'label fingerprint differs')
    ri=context['residue_index'].numpy();ci=context['chain_id'].numpy()
    anchor=array(st['anchor'])[0];truth=array(target['coords'])[0,:,1]
    scalars={};computed_stats={}
    for count in (1,4,8):
        key=str(count);branch=row['branches'][key];tensor=raw['branches'][key]
        require(branch['status'] in ('completed','numerical_failure'),'unknown branch status')
        h,s=array(tensor['H']),array(tensor['S'])
        stats=macro_statistics(h,s,mask);computed_stats[key]=stats
        audit.nested(branch['macro_statistics'],stats,'all_macro_statistics')
        audit.numeric(branch['macro_times_ps'],np.arange(h.shape[1])*1000,'macro_physical_times')
        totals={}
        for space in ('raw','centered'):
            terms={k:array(v) for k,v in tensor['budgets'][space].items()}
            require(set(terms)==set(KEYS),'budget keys differ')
            totals[space]={k:float(v.sum(0).mean()/(length*dim)) for k,v in terms.items()}
            for actual,expected in ((terms['expected'],terms['force']+terms['bath']+terms['drift_squared']+terms['noise_trace']),
                (terms['stochastic_residual'],terms['actual']-terms['expected']),
                (terms['rounding_closure'],terms['actual']-terms['expected']-terms['noise_cross']-terms['noise_squared_minus_trace'])):
                audit.numeric(actual,expected,'all_micro_budget_identities')
            first=h[:,0];last=h[:,-1]
            if len(terms['actual'])%count:
                require(tensor['failure_tensors'] is not None,'partial macro lacks failure-state evidence')
                last=array(tensor['failure_tensors']['H_before'])
            if space=='centered':first=first-first.mean(-1,keepdims=True);last=last-last.mean(-1,keepdims=True)
            change=np.sum((last[:,mask]**2-first[:,mask]**2),axis=(-1,-2))
            audit.numeric(terms['actual'].sum(0),change,'budget_telescope',atol=1e-8,rtol=5e-11)
            kii=np.sum(array(st['factor'])**2,axis=-1)
            expected_trace=2*(1000/count)*len(terms['actual'])*np.sum(array(st['D'])*kii[...,None]*mask[None,:,None],axis=(-1,-2))
            if space=='centered':expected_trace*=1-1/dim
            audit.numeric(terms['noise_trace'].sum(0),expected_trace,'budget_physical_trace')
        audit.nested(branch['budget_totals'],totals,'all_budget_totals')
        score=None
        if branch['status']=='completed':
            require(branch['completed_macro_steps']==row['metadata']['horizon'],'completed branch duration differs')
            score=scores(array(tensor['CA'])[:,-1],truth,anchor,mask,ri,ci)
            audit.nested(branch['endpoint_scores'],score,'numpy_endpoint_scores_geometry')
        else:
            require(branch['endpoint_scores'] is None and branch['failure'] is not None,'failed branch has score or no failure')
        for name in ('coordinate_energy_score','member_rmsd'):scalars[f'm{count}/{name}']=score[name] if score else None
        for name in ('channel_mean_squared','channel_variance_mean','norm_squared_per_dim','bath_norm_squared_per_dim'):
            scalars[f'm{count}/{name}']=float(stats[name][:,-1].mean()) if score else None
        for space in ('raw','centered'):
            for name in KEYS:scalars[f'm{count}/{space}/{name}']=totals[space][name] if score else None
        if score:
            for name,value in score['geometry_mean'].items():scalars[f'm{count}/geometry/{name}']=value
    persistence=scores(np.repeat(anchor[None],12,axis=0),truth,anchor,mask,ri,ci)
    audit.nested(row['persistence'],persistence,'numpy_persistence_scores_geometry')
    for name in ('coordinate_energy_score','member_rmsd'):scalars[f'persistence/{name}']=persistence[name]
    for name,value in persistence['geometry_mean'].items():
        scalars[f'persistence/geometry/{name}']=value
        for count in (1,4,8):scalars.setdefault(f'm{count}/geometry/{name}',None)
    for left,right in ((1,4),(4,8)):
        name=f'{left}_vs_{right}'
        expected=None
        if all(row['branches'][str(k)]['status']=='completed' for k in (left,right)):
            expected={kind:difference(array(raw['branches'][str(left)][kind]),array(raw['branches'][str(right)][kind]),mask,kind=='CA') for kind in ('H','CA')}
        audit.nested(row['differences'][name],expected,'all_coupled_path_differences')
        for kind in ('H','CA'):scalars[f'{name}/{kind}_endpoint']=expected[kind]['endpoint_mean'] if expected else None
    audit.nested(row['scalars'],scalars,'all_reconstructed_window_scalars')
    return dict(metadata=row['metadata'],scalars=scalars)


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='',"require CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(2)
    vp=BASE/'verification.protocol.json';registered_bytes=vp.read_bytes();registered_sha=hashlib.sha256(registered_bytes).hexdigest()
    registration=json.loads(registered_bytes)
    require(sha(__file__)==registration['verifier_sha256'],'verifier source changed after registration')
    require(sha(BASE/'protocol.json')==PROTOCOL_SHA,'producer protocol changed')
    protocol=json.loads((BASE/'protocol.json').read_text())
    execution=json.loads((BASE/'execution.json').read_text())
    require(execution['returncode']==0,'producer did not complete exit0')
    result_path=BASE/'diagnostic.json';result_sha=sha(result_path)
    require(execution['result_sha256']==result_sha,'producer result hash differs')
    report=json.loads(result_path.read_text());audit=Audit();started=time.monotonic()
    write(BASE/'verification.started.json',dict(pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=registered_sha,result_sha256=result_sha,argv=sys.argv))
    inputs={str(result_path.relative_to(REPO)):result_sha,str(vp.relative_to(REPO)):registered_sha}
    for section in ('input_hashes','source_hashes'):
        for path,digest in protocol[section].items():
            require(sha(REPO/path)==digest,'registered producer file differs: '+path);inputs[path]=digest
            if section=='source_hashes':
                snapshot=BASE/'source_snapshot'/path
                require(sha(snapshot)==digest,'snapshot differs: '+path)
                inputs[str(snapshot.relative_to(REPO))]=digest
    for path,expected in protocol['hdf5_file_stats'].items():
        stat=Path(path).stat();actual=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino,device=stat.st_dev)
        require(actual==expected,'HDF5 stat differs')
    checkpoint=torch.load(REPO/protocol['checkpoint'],map_location='cpu',weights_only=False)
    model=load_model(str(REPO/protocol['checkpoint']),checkpoint,'cpu').eval().requires_grad_(False)
    require(state_sha256(model)==protocol['model_tensor_sha256'],'actual loaded model hash differs')
    require(len(report['per_window'])==32,'incomplete report panel')
    reconstructed=[];replay=None
    for index,row in enumerate(report['per_window'],1):
        require(row['metadata']==protocol['selection'][index-1],'selection order differs')
        json_path=BASE/f'window_{index:02d}.json';saved=json.loads(json_path.read_text())
        require(saved==row,'per-window JSON differs from final report');inputs[str(json_path.relative_to(REPO))]=sha(json_path)
        raw_path=REPO/row['raw_tensors']['path'];require(sha(raw_path)==row['raw_tensors']['sha256'],'raw tensor hash differs')
        inputs[str(raw_path.relative_to(REPO))]=row['raw_tensors']['sha256']
        raw=torch.load(raw_path,map_location='cpu',weights_only=False)
        reconstructed.append(verify_window(raw,row,audit))
        if row['metadata']==registration['replay_selection']:
            replay=replay_selected(model,raw,audit)
        print(json.dumps(dict(verified_windows=index,total=32,seconds=time.monotonic()-started)),flush=True)
        del raw
    require(replay is not None,'preselected replay window missing')
    audit.nested(report['overall'],aggregate(reconstructed),'all_aggregate_overall')
    for horizon in (1,4,16,64):
        audit.nested(report['by_horizon'][str(horizon)],aggregate([r for r in reconstructed if r['metadata']['horizon']==horizon]),'all_aggregate_horizons')
    failures=sum(b['status']!='completed' for r in report['per_window'] for b in r['branches'].values())
    require(failures==report['numerical_failure_branches']==execution['numerical_failure_branches'],'failure counts differ')
    require(state_sha256(model)==protocol['model_tensor_sha256'],'verification changed model state')
    require(all(p.grad is None for p in model.parameters()),'verification created parameter gradients')
    require(not torch.cuda.is_initialized(),'verification initialized CUDA')
    for path,digest in inputs.items():require(sha(REPO/path)==digest,'input changed during independent verification: '+path)
    for path,expected in protocol['hdf5_file_stats'].items():
        stat=Path(path).stat();actual=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,inode=stat.st_ino,device=stat.st_dev)
        require(actual==expected,'HDF5 stat changed during verification')
    require(sha(BASE/'protocol.json')==PROTOCOL_SHA and sha(vp)==registered_sha,'protocol changed during verification')
    output=dict(schema_version=VERSION,passed=True,registered_sha256=registered_sha,
        verifier_sha256=registration['verifier_sha256'],producer_result_sha256=result_sha,
        n_windows=32,actual_force_replay_selection=registration['replay_selection'],actual_force_replay=replay,
        numeric_values_checked=dict(audit.counts),maximum_absolute_discrepancy=dict(audit.max_abs),
        source_input_snapshot_model_hashes_verified=True,numerical_failure_branches=failures,
        completed_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,pid=os.getpid(),
        limitations='Numerical/provenance verification, not a stability or forecast improvement claim; actual force replay limited to preselected first2macro of step4')
    write(BASE/'independent_verification.json',output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['register','run'],required=True)
    args=parser.parse_args()
    if args.mode=='register':register();return
    start=time.monotonic()
    try:
        result=run()
    except BaseException as error:
        code=130 if isinstance(error,KeyboardInterrupt) else 1
        write(BASE/'verification.execution.json',dict(pid=os.getpid(),returncode=code,error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-start))
        raise
    write(BASE/'verification.execution.json',dict(pid=os.getpid(),returncode=0,seconds=time.monotonic()-start,
        output_sha256=sha(BASE/'independent_verification.json'),n_windows=result['n_windows']))


if __name__=='__main__':main()
