"""Independent NumPy audit of fixed-covariance train32 mean controls.

No producer statistics/helpers are imported. Register before opening generated
mean-control rows/scores; run only after the producer's recorded exit zero.
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
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts import verify_hs_microsteps_v1 as independent
from scripts.eval_ca_endpoint_scores import load_model
from protein_jepa_frame.utils.contracts import state_sha256


REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'results/hs_joint_20261101/history_mean_train32_v1'
PRODUCER_PROTOCOL_SHA = 'b0de71715a87c42047f2814d20d5cd73c87430696eee7e7fcfa013695c7806b0'
INDEPENDENT_SOURCE = 'scripts/verify_hs_microsteps_v1.py'
INDEPENDENT_SHA = 'ee3745bf4ef418bbd1c1f12c0cd961877217464ab4516ca054f0580fdce04df4'
TEST_SOURCE = 'tests/test_hs_history_mean_verification_v1.py'
VERSION = 'independent_numpy_history_mean_train32_verification_v1'
ARMS = ('real_history', 'repeated_anchor_history', 'zero_mean')
require, array = independent.require, independent.array


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def read(path): return json.loads(Path(path).read_text())


def exact(actual, expected, name):
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype
                and actual.shape == expected.shape, name+': tensor schema differs')
        # Equal NaN bit patterns in retained failure evidence are permitted.
        a = actual.detach().cpu().contiguous().numpy()
        e = expected.detach().cpu().contiguous().numpy()
        require(a.tobytes() == e.tobytes(), name+': tensor bytes differ')
    elif isinstance(expected, np.ndarray):
        require(isinstance(actual,np.ndarray) and actual.dtype==expected.dtype
                and actual.shape==expected.shape and actual.tobytes()==expected.tobytes(),
                name+': array schema/bytes differ')
    elif isinstance(expected, dict):
        require(set(actual) == set(expected), name+': keys differ')
        for key in expected: exact(actual[key], expected[key], name+'/'+str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), name+': sequence differs')
        for index, (a, e) in enumerate(zip(actual, expected)): exact(a, e, name+'/'+str(index))
    else:
        require(actual == expected, name+': field differs')


def whitened_difference(mean, reference, chol, mask):
    """NumPy double arithmetic; only packed valid bath coordinates are solved."""
    mean, reference, chol = map(array, (mean, reference, chol))
    mask = np.asarray(mask, dtype=bool)
    ds = mean.shape[-1]
    indices = (np.flatnonzero(mask[0])[:, None]*ds+np.arange(ds)[None]).reshape(-1)
    delta = (mean-reference)[0].reshape(-1)[indices]
    lower = chol[0][np.ix_(indices, indices)]
    require(len(indices) > 0 and np.isfinite(delta).all() and np.isfinite(lower).all(),
            'invalid packed whitening inputs')
    require(np.array_equal(lower, np.tril(lower)) and (np.diag(lower) > 0).all(),
            'packed factor is not positive-diagonal lower triangular')
    white = np.linalg.solve(lower, delta)
    require(np.isfinite(white).all(), 'nonfinite independent mean whitening')
    squared = float(white@white)
    return dict(n_valid_bath_dimensions=len(indices), raw_l2=float(np.linalg.norm(delta)),
        raw_rms=float(np.sqrt(np.mean(delta*delta))), mahalanobis_squared=squared,
        standardized_rms=float(np.sqrt(squared/len(indices))),
        per_valid_dimension_mahalanobis_squared=squared/len(indices))


def check_initial_inputs(raw, original, control, audit):
    state = original['state']
    exact(raw['metadata'], original['metadata'], 'source metadata')
    exact(raw['controls'], control, 'preflight control artifact')
    exact(raw['H0'], state['H0'], 'shared H0')
    exact(raw['original_full_chol'], state['posterior_cholesky'], 'original full covariance')
    exact(raw['posterior_epsilon'], state['posterior_standard_normal'], 'original epsilon')
    exact(raw['observation'], original['observed'], 'observed source')
    exact(raw['target'], original['target'], 'target source')
    for name, value in raw['condition'].items(): exact(value, state[name], 'condition/'+name)
    require(set(raw['condition']) == {'mask','temp','D','C','Gamma','factor','anchor_H',
            'rotation','anchor','metadata','history','dt_history'}, 'condition set differs')
    means = control['means']
    require(set(means) == set(ARMS), 'mean control set differs')
    exact(means['real_history'], state['posterior_mean'], 'original posterior mean')
    exact(means['zero_mean'], torch.zeros_like(state['posterior_mean']), 'zero mean control')
    repeated = state['anchor_H'][:, None].expand(-1,4,-1,-1).contiguous()
    require(control['repeated_history_sha256'] == state_sha256({'H':repeated}), 'repeated history fingerprint differs')
    require(control['repeated_chol_bitwise_equal'] is True
            and control['repeated_chol_sha256'] == state_sha256({'chol':state['posterior_cholesky']}),
            'preflight covariance evidence differs')
    expected_stats = {arm:whitened_difference(mean, means['real_history'],
        state['posterior_cholesky'], state['mask'][:1].numpy()) for arm, mean in means.items()}
    expected_stats['real_minus_zero'] = whitened_difference(means['real_history'], means['zero_mean'],
        state['posterior_cholesky'], state['mask'][:1].numpy())
    audit.nested(control['mean_differences'], expected_stats, 'all_mean_whitening')
    # Independently spell out the original float32 matrix product and grouping;
    # double NumPy additionally checks the mathematical draw, with declared tolerance.
    chol, eps = state['posterior_cholesky'], state['posterior_standard_normal']
    shape = (1, eps.shape[1], *state['posterior_mean'].shape[1:])
    centered = torch.einsum('bij,bmj->bmi', chol, eps).reshape(shape)
    exact(raw['shared_centered_bath'], centered, 'single shared centered draw')
    expected_double = np.einsum('bij,bmj->bmi', array(chol), array(eps)).reshape(shape)
    audit.numeric(array(centered), expected_double, 'numpy_full_covariance_draw', atol=2e-6, rtol=3e-6)
    for arm in ARMS:
        expected_s0 = (means[arm][:,None]+centered)[0]
        exact(raw['arms'][arm]['S0'], expected_s0, arm+'/shared-noise S0')
        exact(raw['arms'][arm]['H'][:,0], state['H0'], arm+'/initial H')
        exact(raw['arms'][arm]['S'][:,0], expected_s0, arm+'/initial S')
    exact(raw['arms']['real_history']['S0'], state['S0'], 'real original S0')
    for source, destination in (('fine_H','brownian_H'),('fine_S','brownian_S')):
        fine = state[source]
        coarse = fine.reshape(*fine.shape[:2],1,8,*fine.shape[3:]).sum(3)/math.sqrt(8)
        exact(raw[destination], coarse, 'original m1 '+destination)
    for name in ('H','S','CA'):
        exact(raw['arms']['real_history'][name], original['branches']['1'][name], 'real full '+name)
        audit.counts['real_path_bitwise_elements'] += raw['arms']['real_history'][name].numel()
    return expected_stats


def verify_window(raw, original, control, row, audit):
    exact(row['metadata'], original['metadata'], 'reported metadata')
    expected_means = check_initial_inputs(raw, original, control, audit)
    audit.nested(row['mean_differences'], expected_means, 'reported_mean_whitening')
    require(row['repeated_chol_bitwise_equal'] is True and row['real_full_path_bitwise_replay'] is True,
            'reported replay evidence missing')
    require(set(raw['arms']) == set(row['arms']) == set(ARMS), 'arm set differs')
    state = original['state']; mask = state['mask'][0].numpy()
    target = array(original['target']['coords'])[0,:,1]
    anchor = array(state['anchor'])[0]
    ri, ci = [original['observed'][key].numpy() for key in ('residue_index','chain_id')]
    shared = {'H0':state['H0'],'chol':state['posterior_cholesky'],'eps':state['posterior_standard_normal'],
        'brownian_H':raw['brownian_H'],'brownian_S':raw['brownian_S'],'factor':state['factor'],
        'D':state['D'],'C':state['C'],'Gamma':state['Gamma'],'temp':state['temp']}
    require(row['shared_input_sha256'] == state_sha256(shared), 'shared input fingerprint differs')
    scalars = {}
    for arm in ARMS:
        result, values = row['arms'][arm], raw['arms'][arm]
        require(result['status'] in ('completed','numerical_failure'), 'unknown arm status')
        h, s = array(values['H']), array(values['S'])
        require(h.shape[0] == s.shape[0] == 12 and h.shape[1] == s.shape[1], 'member/time budget differs')
        require(result['completed_macro_steps'] == h.shape[1]-1, 'completed macro count differs')
        audit.numeric(result['macro_times_ps'], np.arange(h.shape[1])*1000, 'all_macro_times')
        statistics = independent.macro_statistics(h,s,mask)
        audit.nested(result['macro_statistics'], statistics, 'all_macro_statistics')
        score = None
        if result['status'] == 'completed':
            require(h.shape[1] == row['metadata']['horizon']+1, 'complete arm duration differs')
            require(result['failure'] is None and values['failure_tensors'] is None, 'completed arm has failure evidence')
            require(values['CA'].shape[:3] == values['H'].shape[:3], 'CA path time/member axes differ')
            score = independent.scores(array(values['CA'])[:,-1],target,anchor,mask,ri,ci)
            audit.nested(result['endpoint_scores'],score,'all_numpy_endpoint_scores_geometry')
        else:
            require(result['endpoint_scores'] is None and result['failure'] is not None, 'failed arm lacks null/evidence')
            if result['failure']['stage'] == 'transition':
                require(values['failure_tensors'] is not None, 'transition failure tensors missing')
                counts = {k:(~np.isfinite(array(v))).reshape(12,-1).sum(1).tolist()
                          for k,v in values['failure_tensors'].items()}
                exact(result['failure']['nonfinite_per_member'], counts, 'failure member counts')
                require(any(any(v) for v in counts.values()), 'transition failure has no nonfinite tensor')
            elif result['failure']['stage'] == 'CA_decode':
                counts=(~np.isfinite(array(values['CA'])[:,:,mask])).reshape(12,values['CA'].shape[1],-1).sum(-1).tolist()
                exact(result['failure']['nonfinite_per_member_macro'],counts,'CA failure member counts')
                require(any(any(v) for v in counts),'CA failure has no nonfinite coordinate')
            else:
                require(result['failure']['stage']=='endpoint_metric' and isinstance(result['failure']['error'],str),
                        'unknown numerical failure stage')
        for name in ('coordinate_energy_score','member_rmsd'):
            scalars[arm+'/'+name] = score[name] if score else None
        for name in ('channel_mean_squared','channel_variance_mean','norm_squared_per_dim','bath_norm_squared_per_dim'):
            scalars[arm+'/'+name] = float(statistics[name][:,-1].mean()) if score else None
        if score:
            for name,value in score['geometry_mean'].items(): scalars[arm+'/geometry/'+name] = value
        for name,value in expected_means[arm].items(): scalars[arm+'/initial_mean/'+name] = value
    persistence = independent.scores(np.repeat(anchor[None],12,axis=0),target,anchor,mask,ri,ci)
    audit.nested(row['persistence'],persistence,'numpy_persistence')
    for name in ('coordinate_energy_score','member_rmsd'): scalars['persistence/'+name] = persistence[name]
    for name,value in persistence['geometry_mean'].items():
        scalars['persistence/geometry/'+name] = value
        for arm in ARMS: scalars.setdefault(arm+'/geometry/'+name,None)
    suffixes = [k[len('real_history/'):] for k in scalars if k.startswith('real_history/') and '/initial_mean/' not in k]
    for arm in ARMS[1:]:
        for suffix in suffixes:
            a,b = scalars[arm+'/'+suffix],scalars['real_history/'+suffix]
            scalars[arm+'_minus_real/'+suffix] = a-b if a is not None and b is not None else None
    audit.nested(row['scalars'],scalars,'all_reconstructed_scalars')
    return dict(metadata=row['metadata'],scalars=scalars)


def replay_selected(model, raw, original, audit):
    """Preselected step4: actual repeated posterior and first two macro steps."""
    state=original['state'];h0=state['anchor_H']
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        mean,chol=model.kernel.initial_bath_state(h0[:,None].expand(-1,4,-1,-1).contiguous(),
            state['dt_history'],state['temp'][:1],state['mask'][:1],seq_feature=h0,
            noise_factor=state['factor'][:1],create_graph=False)
        exact(chol,state['posterior_cholesky'],'preselected actual repeated posterior Cholesky')
        exact(mean,raw['controls']['means']['repeated_anchor_history'],'preselected actual repeated mean')
        d,c,gamma=model.kernel.bath(state['H0'],state['temp'])
        for value,key in ((d,'D'),(c,'C'),(gamma,'Gamma')):exact(value,state[key],'actual coefficient '+key)
        result={}
        for arm in ARMS:
            h,s=state['H0'].clone(),raw['arms'][arm]['S0'].clone();checked=0
            requested=min(2,raw['arms'][arm]['H'].shape[1]-1)
            for step in range(requested):
                h,s=model.kernel._collective_velocity_step_fixed(h,s,state['mask'],state['temp'],d,c,gamma,1000.,
                    noise_H=raw['brownian_H'][:,step,0],noise_S=raw['brownian_S'][:,step,0],noise_factor=state['factor'])
                exact(h,raw['arms'][arm]['H'][:,step+1],arm+'/actual macro H')
                exact(s,raw['arms'][arm]['S'][:,step+1],arm+'/actual macro S')
                audit.counts['actual_primitive_bitwise_elements'] += h.numel()+s.numel();checked+=1
            result[arm]=dict(requested_macro_steps=2,checked_macro_steps=checked,complete=checked==2)
    return dict(repeated_posterior_mean_and_chol_bitwise=True,arms=result)


def register():
    require(sha(BASE/'protocol.json') == PRODUCER_PROTOCOL_SHA, 'producer protocol hash differs')
    require(sha(REPO/INDEPENDENT_SOURCE) == INDEPENDENT_SHA, 'independent NumPy implementation changed')
    producer=read(BASE/'protocol.json')
    require(producer['schema_version']=='hs_history_mean_fixed_covariance_train32_v1'
            and producer['n_windows']==producer['n_proteins']==32 and producer['n_samples']==12,
            'producer panel differs')
    selected=next(row for row in producer['selection'] if row['horizon']==64)
    require(selected['step']==4,'fixed first h64 selection differs')
    sources={str(Path(__file__).relative_to(REPO)):sha(__file__),TEST_SOURCE:sha(REPO/TEST_SOURCE),
             INDEPENDENT_SOURCE:INDEPENDENT_SHA}
    protocol=dict(schema_version=VERSION,registered_utc=now(),producer_protocol_sha256=PRODUCER_PROTOCOL_SHA,
        source_hashes=sources,replay_selection=selected,replay_macro_steps=2,replay_arms=list(ARMS),
        audit_scope='all32 NumPy mean whitening, shared covariance/epsilon/S0/Brownian, real full H/S/CA bitwise, macro statistics, endpoint U-ES/RMSD/geometry, scalars, full aggregation, input/source/snapshot hashes',
        selected_actual_replay='fixed first h64/step4; actual repeated posterior mean and Cholesky, first two macro steps of each arm',
        covariance_scope='all32 saved repeated-Cholesky fingerprints equal original; actual repeated posterior rerun only preselected step4',
        generated_results_read_before_registration=False,selection_rule='from already fixed producer metadata only',
        producer_result_existed_at_registration=(BASE/'diagnostic.json').exists(),
        tolerances=dict(general_absolute=2e-9,general_relative=3e-10,full_covariance_draw_absolute=2e-6,full_covariance_draw_relative=3e-6),
        device='cpu',torch_threads=2,new_random_draws=0,deterministic_kernel_replay=True,optimizer_steps=0,test_opened=False,
        versions=dict(numpy=np.__version__,torch=torch.__version__,python=sys.version))
    write(BASE/'verification.protocol.json',protocol)
    snapshot=BASE/'verification.source_snapshot';snapshot.mkdir(exist_ok=False)
    for path,digest in sources.items():
        data=(REPO/path).read_bytes();require(hashlib.sha256(data).hexdigest()==digest,'verifier source changed during registration')
        dest=snapshot/path;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:stream.write(data)
    print(json.dumps(dict(registered=True,sha256=sha(BASE/'verification.protocol.json'))),flush=True)


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'verifier requires CPU-only environment')
    torch.set_num_threads(2)
    vp=BASE/'verification.protocol.json';registered_sha=sha(vp);registration=read(vp)
    require(registration['producer_protocol_sha256']==sha(BASE/'protocol.json')==PRODUCER_PROTOCOL_SHA,
            'producer protocol differs')
    producer=read(BASE/'protocol.json');execution=read(BASE/'execution.json')
    require(execution['returncode']==0,'producer must finish exit zero')
    require(execution['output_sha256']==sha(BASE/'diagnostic.json'),'producer result fingerprint differs')
    inputs={str(vp.relative_to(REPO)):registered_sha,str((BASE/'protocol.json').relative_to(REPO)):PRODUCER_PROTOCOL_SHA,
        str((BASE/'execution.json').relative_to(REPO)):sha(BASE/'execution.json'),
        str((BASE/'diagnostic.json').relative_to(REPO)):sha(BASE/'diagnostic.json')}
    for path,digest in registration['source_hashes'].items():
        require(sha(REPO/path)==digest and sha(BASE/'verification.source_snapshot'/path)==digest,'verifier source/snapshot differs')
        inputs[path]=digest;inputs[str((BASE/'verification.source_snapshot'/path).relative_to(REPO))]=digest
    for section in ('input_hashes','source_hashes'):
        for path,digest in producer[section].items():
            require(sha(REPO/path)==digest,'producer source/input changed: '+path);inputs[path]=digest
            if section=='source_hashes':
                snapshot=BASE/'source_snapshot'/path;require(sha(snapshot)==digest,'producer snapshot differs')
                inputs[str(snapshot.relative_to(REPO))]=digest
    write(BASE/'verification.started.json',dict(pid=os.getpid(),started_utc=now(),argv=sys.argv,registered_sha256=registered_sha))
    started=time.monotonic();report=read(BASE/'diagnostic.json');audit=independent.Audit()
    require(len(report['per_window'])==32,'incomplete reported panel')
    preflight=read(BASE/'covariance_preflight.json')
    require(preflight['all32_bitwise_equal'] is True and preflight['n_windows']==32
            and preflight['sampling_started'] is False,'covariance preflight incomplete')
    require(preflight['means_tensor_sha256']==sha(BASE/'mean_controls.pt'),'mean-control artifact differs')
    controls=torch.load(BASE/'mean_controls.pt',map_location='cpu',weights_only=False)
    require(len(controls)==32 and len(preflight['per_window'])==32,'mean-control panel incomplete')
    for name in ('covariance_preflight.json','mean_controls.pt'):inputs[str((BASE/name).relative_to(REPO))]=sha(BASE/name)
    checkpoint=torch.load(REPO/producer['checkpoint'],map_location='cpu',weights_only=False)
    model=load_model(str(REPO/producer['checkpoint']),checkpoint,'cpu').eval().requires_grad_(False)
    require(state_sha256(model)==producer['model_tensor_sha256'],'loaded model differs')
    reconstructed=[];replay=None
    for index,(row,item,control,pre) in enumerate(zip(report['per_window'],producer['input_rows'],controls,preflight['per_window']),1):
        exact(row['metadata'],item['metadata'],'selection order')
        exact(control['metadata'],item['metadata'],'control selection')
        exact(pre['metadata'],item['metadata'],'preflight selection')
        exact(pre['mean_differences'],control['mean_differences'],'preflight means')
        require(pre['repeated_chol_sha256']==control['repeated_chol_sha256'],'preflight factor hash differs')
        raw_path=REPO/row['raw_tensors']['path'];require(sha(raw_path)==row['raw_tensors']['sha256'],'new raw hash differs')
        inputs[str(raw_path.relative_to(REPO))]=row['raw_tensors']['sha256']
        json_path=BASE/f"window_{row['metadata']['step']:02d}.json"
        exact(read(json_path),row,'window result');inputs[str(json_path.relative_to(REPO))]=sha(json_path)
        raw=torch.load(raw_path,map_location='cpu',weights_only=False)
        original=torch.load(REPO/item['path'],map_location='cpu',weights_only=False)
        exact(raw['source_raw'],item,'source raw identity')
        reconstructed.append(verify_window(raw,original,control,row,audit))
        if item['metadata']==registration['replay_selection']:replay=replay_selected(model,raw,original,audit)
        print(json.dumps(dict(verified_windows=index,total=32,seconds=time.monotonic()-started)),flush=True)
        del raw,original
    require(replay is not None,'fixed replay selection missing')
    audit.nested(report['overall'],independent.aggregate(reconstructed),'all_aggregate_overall')
    for horizon in (1,4,16,64):
        audit.nested(report['by_horizon'][str(horizon)],independent.aggregate([r for r in reconstructed if r['metadata']['horizon']==horizon]),'all_aggregate_horizons')
    failures=sum(a['status']!='completed' for r in report['per_window'] for a in r['arms'].values())
    require(failures==report['numerical_failure_arms']==execution['numerical_failure_arms'],'failure counts differ')
    require(state_sha256(model)==producer['model_tensor_sha256'] and all(p.grad is None for p in model.parameters()),'verification modified model')
    require(not torch.cuda.is_initialized(),'verification initialized CUDA')
    for path,digest in inputs.items():require(sha(REPO/path)==digest,'input changed during verification: '+path)
    result=dict(schema_version=VERSION,passed=True,completed_utc=now(),pid=os.getpid(),seconds=time.monotonic()-started,
        registration_sha256=registered_sha,producer_result_sha256=execution['output_sha256'],n_windows=32,numerical_failure_arms=failures,
        numeric_values_checked=dict(audit.counts),maximum_absolute_discrepancy=dict(audit.max_abs),
        actual_replay_selection=registration['replay_selection'],actual_replay=replay,
        source_input_snapshot_model_hashes_verified_before_after=True,
        limitations='Numerical/provenance audit, not a benefit/stability claim; actual repeated-posterior and primitive replay only the prespecified step4.')
    write(BASE/'independent_verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['register','run'],required=True)
    args=parser.parse_args()
    if args.mode=='register':register();return
    started=time.monotonic()
    try:result=run()
    except BaseException as error:
        write(BASE/'verification.execution.json',dict(pid=os.getpid(),returncode=130 if isinstance(error,KeyboardInterrupt) else 1,
            error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-started,completed_utc=now()))
        raise
    write(BASE/'verification.execution.json',dict(pid=os.getpid(),returncode=0,seconds=time.monotonic()-started,
        output_sha256=sha(BASE/'independent_verification.json'),n_windows=result['n_windows'],completed_utc=now()))


if __name__=='__main__':main()
