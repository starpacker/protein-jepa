"""CPU audit/summary of the two registered continuous CA path panels.

No trajectory reads, model construction, sampling, training, test or BioEmu2.
Only the two joint path U-scores receive paired protein-bootstrap intervals.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/hs_joint_20261101/ordered_paths'
REGISTRATION_SHA = '4107df30c13337b151fd9de58e4b73faba1b25fa4e7a8f9cfbb5734570aa130e'
AMENDMENT_SHA = 'cdfc2e545fc43ed16ae0afa92531a5a337b0f0b9415b0f186b3d1fba9e822a1d'
SCORES = ('coordinate_path_energy_score', 'physical_distance_path_energy_score')
ARMS = ('forecast', 'reversed_forecast', 'shuffled_forecast', 'persistence', 'reference_MD')
CONTRASTS = (('forecast', 'persistence'), ('reversed_forecast', 'forecast'),
             ('shuffled_forecast', 'forecast'))
LAGS = (1, 2, 4, 8, 16, 32)
DESCRIPTIVE = ('per_step_rmsd_mean', 'path_length', 'end_to_end', 'tortuosity', 'rg_mean',
    'walk_rmsf_mae', 'walk_rmsf_corr', 'distance_acf_lag1', 'distance_msd_lag1',
    'reference_distance_acf_lag1', 'reference_distance_msd_lag1', 'distance_acf_corr')
REPS, SEED = 10000, 20261101


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(value):
    file = Path(value)
    file = (REPO / file).resolve() if not file.is_absolute() else file.resolve()
    require(file.is_relative_to(REPO), 'input must be inside repository')
    require(file.name != 'test.jsonl' and 'bioemu' not in str(file).lower(), 'sealed/excluded input')
    return file


def sha(file):
    digest = hashlib.sha256()
    with local(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, file, expected=None):
        file = local(file)
        actual = sha(file)
        require(expected is None or actual == expected, f'hash mismatch: {file}')
        require(file not in self.hashes or self.hashes[file] == actual, f'changed input: {file}')
        self.hashes[file] = actual
        return file

    def read(self, file, expected=None):
        return json.loads(self.track(file, expected).read_text())

    def verify(self):
        for file, expected in self.hashes.items():
            require(sha(file) == expected, f'input changed during summary: {file}')


def integer(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and int(value) == value, 'invalid integer metadata')
    return int(value)


def finite_or_none(value):
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value))


def key(row):
    require(all(isinstance(row[k], str) and row[k] for k in ('protein', 'trajectory')), 'invalid window identity')
    return (row['protein'], row['trajectory'], float(row['temperature_k']),
            integer(row['anchor_frame']), integer(row['horizon']))


def audit_rows(rows, panel):
    from scripts.eval_ca_ordered_paths import validate_panel, temporal_controls, ROLES
    import torch
    validate_panel(rows, panel)
    seen, observed = set(), {}
    for row in rows:
        k = key(row)
        require(k not in seen, 'duplicate window key')
        seen.add(k)
        h, anchor, valid = k[4], k[3], integer(row['valid_residues'])
        require(h in panel['horizons'] and anchor >= 7 and 4 <= valid <= 96, 'window budget differs')
        require(row['target_frame'] == anchor+h and row['horizon_ps'] == h*1000
                and row['context_frames'] == list(range(anchor-7, anchor+1)), 'nonconsecutive frame grid')
        grid = row['time_grid']
        require(grid['step_ps'] == 1000 and grid['n_frames'] == h+1
                and np.isfinite(grid['anchor_time_ps'])
                and grid['endpoint_time_ps']-grid['anchor_time_ps'] == h*1000, 'physical time grid differs')
        require(row['n_forecast_paths'] == 12 and row['n_reference_paths'] == 1, 'path/reference budget differs')
        seed = int(hashlib.sha256(f'{k[0]}|{k[1]}|{anchor}|{h}'.encode()).hexdigest()[:12],16) % (2**31-1)
        require(row['sampling_seed'] == seed, 'sampling seed differs')
        expected_controls = temporal_controls(torch.zeros(1,h+1,1,3), seed)[1]
        require(row['temporal_controls'] == expected_controls, 'temporal-control permutation differs')
        continuity = row['continuous_path']
        require(continuity['n_sample_calls'] == 1 and continuity['future_reencodings'] == 0
                and continuity['bath_resets'] == 0 and continuity['includes_sampled_S0'] is True,
                'not one continuous sampled H/S path')
        latent, bath = continuity['latent_shape'], continuity['bath_shape']
        require(len(latent) == 5 and latent[:3] == [1,12,h+1]
                and valid <= latent[3] <= 96 and integer(latent[4]) > 0
                and isinstance(bath,list) and len(bath) == 5 and bath[:4] == latent[:4]
                and integer(bath[4]) > 0, 'latent/bath trajectory shape differs')
        fps = row['input_fingerprints']
        require(set(fps) == {'observed','reference','forecast_CA_path'}
                and all(re.fullmatch('[0-9a-f]{64}', v) for v in fps.values()), 'invalid tensor fingerprints')
        old = observed.setdefault(k[0], fps['observed'])
        require(old == fps['observed'], 'shared-anchor horizons have different observed inputs')
        require(set(row['arms']) == set(ARMS), 'arm schema differs')
        common_pairs = None
        reference_lags = None
        for arm in ARMS:
            a = row['arms'][arm]
            m = 1 if arm == 'reference_MD' else 12
            require(a['role'] == ROLES[arm] and a['n_paths'] == m, 'arm role/path count differs')
            metrics, values = a['metrics'], a['metrics']['scalars']
            require(metrics['task_version'] == 'ca_ordered_path_metrics_v2'
                    and set(values) == set((*SCORES,*DESCRIPTIVE,'distance_acf_corr_n_lags'))
                    and all(finite_or_none(v) for v in values.values()), 'metric version/scalar schema differs')
            require(all((values[s] is None) == (arm == 'reference_MD') for s in SCORES),
                    'joint U-scores require M>=2; reference M1 must remain undefined')
            c = metrics['conventions']
            require(c['n_samples'] == m and c['n_reference_paths'] == 1
                    and c['n_future_frames'] == h and c['n_valid_residues'] == valid
                    and c['arithmetic_dtype'] == 'float64' and c['requested_lags'] == list(LAGS),
                    'metric sample/time/arithmetic contract differs')
            pairs = c['pair_indices']
            require(0 < len(pairs) == c['n_nonlocal_pairs'] <= 512
                    and len({tuple(p) for p in pairs}) == len(pairs)
                    and all(len(p)==2 and 0 <= integer(p[0]) < integer(p[1]) < latent[3] for p in pairs),
                    'physical pair identity/count differs')
            common_pairs = pairs if common_pairs is None else common_pairs
            require(pairs == common_pairs, 'arms use different physical pairs')
            require(len(metrics['per_path']) == m
                    and [p['path_index'] for p in metrics['per_path']] == list(range(m)), 'per-path diagnostics incomplete')
            curves = metrics['curves']
            require(curves['times_ps'] == [1000.*i for i in range(1,h+1)]
                    and len(curves['per_step_rmsd']) == len(curves['rg_mean']) == h
                    and all(v is not None and finite_or_none(v) for q in ('per_step_rmsd','rg_mean') for v in curves[q]),
                    'continuous metric curves incomplete')
            for role, members in (('prediction',m),('reference',1)):
                lag = metrics['lag_statistics'][role]
                require(len(lag['per_path']) == members and len(lag['aggregate']) == len(LAGS),
                        'reference replication or joined-path lag statistics')
                for i, per in enumerate(lag['per_path']):
                    require(per['path_index'] == i and per['n_frames'] == h+1
                            and len(per['by_lag']) == len(LAGS), 'per-path lag frame budget differs')
                    for offset, item in zip(LAGS, per['by_lag']):
                        count = max(h+1-offset,0)
                        require(item['lag'] == offset and item['lag_ps'] == offset*1000
                                and item['n_pairs'] == count
                                and all(finite_or_none(item[q]) for q in ('acf','msd'))
                                and (item['msd'] is None) == (count == 0)
                                and (count > 0 or item['acf'] is None), 'illegal within-path lag pair count')
                for offset,item in zip(LAGS,lag['aggregate']):
                    pos=LAGS.index(offset)
                    acf_count=sum(p['by_lag'][pos]['n_pairs'] for p in lag['per_path']
                                  if p['by_lag'][pos]['acf'] is not None)
                    count=members*max(h+1-offset,0)
                    require(item['lag']==offset and item['lag_ps']==offset*1000
                            and item['n_pairs']==count and item['n_acf_pairs']==acf_count
                            and all(finite_or_none(item[q]) for q in ('acf','msd'))
                            and (item['msd'] is None)==(count==0)
                            and (item['acf'] is None)==(acf_count==0), 'aggregate lag counts/undefinedness differ')
            ref = metrics['lag_statistics']['reference']
            reference_lags = ref if reference_lags is None else reference_lags
            require(ref == reference_lags, 'arms use different reference diagnostics')
    return rows


def descriptive(rows, getter):
    by_protein = defaultdict(list)
    for row in rows:
        value = getter(row)
        require(finite_or_none(value), 'nonfinite diagnostic')
        if value is not None:
            by_protein[row['protein']].append(value)
    means = [np.mean(v) for _,v in sorted(by_protein.items())]
    return dict(mean=float(np.mean(means)) if means else None,
                n_valid_proteins=len(means), n_valid_windows=sum(map(len,by_protein.values())))


@lru_cache(maxsize=8)
def draws(n):
    return np.random.default_rng(SEED).integers(n,size=(REPS,n))


def paired(rows, candidate, reference, metric):
    require(metric in SCORES, 'only joint path U-scores receive paired intervals')
    grouped = defaultdict(list)
    for row in rows:
        a,b = (row['arms'][x]['metrics']['scalars'][metric] for x in (candidate,reference))
        require(a is not None and b is not None and finite_or_none(a) and finite_or_none(b), 'incomplete score pairing')
        grouped[row['protein']].append((a,b))
    values = np.array([np.mean(v,axis=0) for _,v in sorted(grouped.items())])
    require(len(values)>0, 'empty paired group')
    difference=values[:,0]-values[:,1]
    ci=np.quantile(difference[draws(len(values))].mean(1),[.025,.975])
    return dict(candidate=float(values[:,0].mean()),reference=float(values[:,1].mean()),
        candidate_minus_reference=float(difference.mean()),ci95=ci.tolist(),
        n_proteins=len(values),n_windows=len(rows))


def summarize_panel(rows):
    result={}
    for horizon in sorted({r['horizon'] for r in rows}):
        selected=[r for r in rows if r['horizon']==horizon]
        scalar=lambda r,a,m:r['arms'][a]['metrics']['scalars'][m]
        diagnostics={a:{m:descriptive(selected,lambda r,a=a,m=m:
            None if (a,m)==('persistence','walk_rmsf_corr') else scalar(r,a,m))
            for m in DESCRIPTIVE} for a in ARMS}
        lags={a:{str(l):dict(
            acf=descriptive(selected,lambda r,a=a,i=i:r['arms'][a]['metrics']['lag_statistics']['prediction']['aggregate'][i]['acf']),
            msd=descriptive(selected,lambda r,a=a,i=i:r['arms'][a]['metrics']['lag_statistics']['prediction']['aggregate'][i]['msd']),
            n_time_pairs=sum(r['arms'][a]['metrics']['lag_statistics']['prediction']['aggregate'][i]['n_pairs'] for r in selected),
            n_acf_time_pairs=sum(r['arms'][a]['metrics']['lag_statistics']['prediction']['aggregate'][i]['n_acf_pairs'] for r in selected))
            for i,l in enumerate(LAGS)} for a in ARMS}
        result[str(horizon)]=dict(n_windows=len(selected),n_proteins=len({r['protein'] for r in selected}),
            comparisons={f'{a}_minus_{b}':{m:paired(selected,a,b,m) for m in SCORES} for a,b in CONTRASTS},
            descriptive=diagnostics,lag_diagnostics=lags,
            raw_persistence_walk_rmsf_corr=dict(
                aggregate=descriptive(selected,lambda r:scalar(r,'persistence','walk_rmsf_corr')),
                per_window=[dict(key=list(key(r)),value=scalar(r,'persistence','walk_rmsf_corr'))
                            for r in sorted(selected,key=key)],
                interpretation='Uninterpretable raw floating-centering artifact for a constant path; excluded from descriptive estimates.'))
    return result


def audit_panel(spec, inputs, registration):
    from scripts.eval_ca_ordered_paths import summarize as raw_summarize
    protocol=inputs.read(spec['registration'],spec['registration_sha256'])
    file=local(spec['result'])
    execution=inputs.read(file.with_suffix('.execution.json'))
    require(execution['status']=='completed' and execution['exit_code']==0
            and execution['frozen_inputs_verified'] is True and execution['source_and_input_hashes_rechecked'] is True
            and local(execution['result'])==file, 'panel execution incomplete')
    data=inputs.read(file,execution['result_sha256'])
    require(all(data.get(k)==v for k,v in protocol.items()), 'result changed registered fields')
    require(data.get('completed_utc') and data.get('frozen_inputs_verified') is True
            and data['n_windows']==execution['n_windows']==spec['n_windows']
            and data['n_proteins']==execution['n_proteins']==50, 'incomplete formal panel')
    require(data['task_version']=='ca_ordered_paths_development_v1'
            and data['optimizer_steps']==0 and data['model_selection'] is False
            and data['test_opened'] is False and data['bioemu2_rerun'] is False
            and data['future_information_given_to_sampler'] is False, 'task/provenance differs')
    require(data['panel']['full_panel'] is True and data['panel']['horizons']==spec['horizons']
            and data['panel']['max_windows']==spec['n_windows'], 'smoke/partial panel is not formal evidence')
    require(local(data['args']['checkpoint'])==local(registration['checkpoint'])
            and data['input_hashes'][data['args']['checkpoint']]==registration['checkpoint_sha256']
            and data['sampling']==dict(seed=0,n_paths=12,model_history_frames=4,prediction_kind='endpoint',
                s0_mode='exact_spatial',one_continuous_rollout_per_member=True,no_future_reencoding=True,no_bath_reset=True),
            'checkpoint/sampling differs')
    for file,digest in data['input_hashes'].items():
        inputs.track(file,digest)
    for file,digest in data['source_hashes'].items():
        inputs.track(file,digest)
        inputs.track(local(spec['source_snapshot'])/file,digest)
    for split in data['provenance']['training_splits']:
        require(all(split[k]==0 for k in ('protein_overlap','domain_overlap','source_file_overlap')), 'training/development overlap')
        inputs.track(split['manifest'],split['train_manifest_sha256'])
        inputs.track(data['args']['manifest'],split['val_manifest_sha256'])
    rows=audit_rows(data['per_window'],data['panel'])
    require(raw_summarize(rows)==data['by_horizon'], 'raw equal-protein summary does not reproduce')
    require(re.fullmatch('[0-9a-f]{64}',data['model_tensor_sha256']), 'missing loaded model fingerprint')
    return data


def build_summary(output=ROOT/'summary.json'):
    output=local(output)
    require(output==ROOT/'summary.json', 'output must be the registered summary.json')
    require(not output.exists(), 'refusing to overwrite ordered summary')
    inputs=Inputs()
    inputs.track(__file__)
    registration=inputs.read(ROOT/'summary.protocol.json',REGISTRATION_SHA)
    amendment=inputs.read(ROOT/'summary.amendment.constant_persistence.json',AMENDMENT_SHA)
    data={name:audit_panel(spec,inputs,registration) for name,spec in registration['panels'].items()}
    standard,long=data['standard200'],data['long192_separate50']
    require(standard['model_tensor_sha256']==long['model_tensor_sha256']
            and standard['input_hashes']==long['input_hashes']
            and standard['source_hashes']==long['source_hashes'], 'panel weights/input/source identities differ')
    report=dict(schema_version=1,task=registration['task'],registration=registration,
        known_issue_amendment=amendment,model_tensor_sha256=standard['model_tensor_sha256'],
        test_opened=False,bioemu2_rerun=False,optimizer_steps=0,
        audit=dict(full_panels=True,execution_exit0=True,input_source_snapshot_hashes_verified=True,
            same_checkpoint_and_loaded_model=True,continuous_path_and_single_reference_contract=True,
            raw_horizon_summaries_reproduced=True,
            independent192_panel=True,no_cross_horizon_pairing=True,
            no_trajectory_arrays_reopened=True),
        panels={name:dict(result=registration['panels'][name]['result'],
            result_sha256=inputs.hashes[local(registration['panels'][name]['result'])],
            by_horizon=summarize_panel(d['per_window'])) for name,d in data.items()})
    inputs.verify()
    report['input_sha256']={str(p.relative_to(REPO)):digest for p,digest in sorted(inputs.hashes.items())}
    serialized=json.dumps(report,indent=2,allow_nan=False)+'\n'
    with output.open('x') as handle:
        handle.write(serialized)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(ROOT/'summary.json'))
    args=parser.parse_args()
    report=build_summary(args.out)
    print(json.dumps(dict(output=args.out,panels=list(report['panels']),bootstrap_reps=REPS,bootstrap_seed=SEED)))


if __name__=='__main__':
    main()
