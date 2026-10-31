"""Protein-cluster paired endpoint vs time-mixture semantics comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_lasp_bounded_paired import HIGHER, SHARED_METRICS, identity


def paired_summary(candidate, baseline, draws=10000, seed=20261024,
                   expected_windows=None, diagnostic_only=False):
    left={identity(r):r for r in candidate['per_window']}
    right={identity(r):r for r in baseline['per_window']}
    if len(left)!=len(candidate['per_window']) or len(right)!=len(baseline['per_window']):
        raise ValueError('duplicate window identities')
    if not left or set(left)!=set(right):
        raise ValueError('window identities differ')
    if expected_windows is not None and (len(left)!=expected_windows or
        candidate.get('n_windows')!=expected_windows or baseline.get('n_windows')!=expected_windows):
        raise ValueError('incomplete expected window population')
    n_samples=candidate.get('n_samples')
    if not isinstance(n_samples,int) or n_samples<2 or n_samples!=baseline.get('n_samples'):
        raise ValueError('sample budgets differ or absent')
    version=candidate.get('metric_version')
    if not version or version!=baseline.get('metric_version'):
        raise ValueError('metric versions differ or absent')
    if (not candidate.get('checkpoint_sha256') or
            candidate['checkpoint_sha256']!=baseline.get('checkpoint_sha256')):
        raise ValueError('paired semantics must use the same checkpoint')
    for key,row in left.items():
        other=right[key]
        for item in (row,other):
            if item.get('actual_n_samples')!=n_samples:
                raise ValueError('actual sample count absent or mismatched')
            times=item.get('observation_steps')
            if times is None or len(times)!=n_samples or any(t<1 or t>key[-2] for t in times):
                raise ValueError('invalid observation times')
        a,b=row.get('persist',{}),other.get('persist',{})
        if not a or set(a)!=set(b) or any(
                not np.isfinite([a[k],b[k]]).all() or
                not np.isclose(a[k],b[k],rtol=1e-5,atol=1e-5) for k in a):
            raise ValueError('persistence reference differs')
    proteins=sorted({k[0] for k in left})
    temps=sorted({k[-1] for k in left})
    horizons=sorted({k[-2] for k in left})
    if not diagnostic_only:
        per_temp={t:sum(1 for k in left if k[-1]==t) for t in temps}
        if (len(proteins)!=50 or temps!=[320.,348.,379.,413.,450.] or
                horizons!=[1.,4.,16.,64.] or any(v!=40 for v in per_temp.values())):
            raise ValueError('unexpected frozen standard200 population')
    rng=np.random.default_rng(seed)
    resamples=rng.integers(0,len(proteins),(draws,len(proteins)))
    results={}
    for metric in sorted(SHARED_METRICS):
        cells={}
        for key in left:
            a=left[key]['lasp'][metric];b=right[key]['lasp'][metric]
            if not np.isfinite([a,b]).all():raise ValueError('nonfinite paired metric')
            # Positive gap means the mixture arm is worse, respecting direction.
            gap=(b-a) if metric in HIGHER else (a-b)
            cells.setdefault((key[0],key[-1]),[]).append(float(gap))
        mat=np.full((len(proteins),len(temps)),np.nan)
        for i,protein in enumerate(proteins):
            for j,temp in enumerate(temps):
                if (protein,temp) in cells:mat[i,j]=np.mean(cells[protein,temp])
        sampled=mat[resamples]; counts=np.isfinite(sampled).sum(axis=1)
        valid=(counts>0).all(axis=1)
        if not valid.any():raise ValueError('no bootstrap samples cover all temperatures')
        boot=(np.nansum(sampled[valid],axis=1)/counts[valid]).mean(axis=1)
        results[metric]={'signed_error_gap_mixture_minus_endpoint':float(np.nanmean(mat,axis=0).mean()),
            'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'one_sided_upper_983333':float(np.quantile(boot,.9833333333)),
            'bootstrap_valid_draws':int(valid.sum()),
            'per_horizon':{str(h):float(np.mean([((right[k]['lasp'][metric]-left[k]['lasp'][metric]) if metric in HIGHER else (left[k]['lasp'][metric]-right[k]['lasp'][metric])) for k in left if k[-2]==h])) for h in horizons},
            'per_temperature':{str(t):float(v) for t,v in zip(temps,np.nanmean(mat,axis=0))},
            'per_protein_per_temperature':{protein:{str(t):float(mat[i,j]) for j,t in enumerate(temps) if np.isfinite(mat[i,j])}
                for i,protein in enumerate(proteins)}}
    return {'n_windows':len(left),'n_proteins':len(proteins),'temperatures':temps,'horizons':horizons,
        'aggregation':'equal represented proteins within temperature, then equal temperatures',
        'bootstrap_draws':draws,'bootstrap_seed':seed,'metrics':results,
        'diagnostic_only':diagnostic_only,'protocol_success_assessed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',nargs='+',required=True);p.add_argument('--mixture',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--expected-windows',type=int,default=200)
    p.add_argument('--diagnostic-only',action='store_true',help='permit sub-200-window diagnostics only')
    a=p.parse_args()
    if len(a.endpoint)!=len(a.mixture):raise ValueError('endpoint/mixture seed counts differ')
    by_seed=[]
    for left,right in zip(a.endpoint,a.mixture):
        e,m=(json.loads(Path(x).read_text()) for x in (left,right))
        if not a.diagnostic_only and (e.get('n_windows')!=a.expected_windows or m.get('n_windows')!=a.expected_windows):
            raise ValueError('incomplete protocol panel; pass --diagnostic-only only for development diagnostics')
        for x,kind in ((e,'endpoint'),(m,'time_mixture')):
            if x.get('lasp_sampling')!=kind or x.get('optimizer_steps')!=0 or x.get('test_opened') is not False:
                raise ValueError(f'invalid semantics/provenance for {kind}')
        n=e.get('n_windows')
        result=paired_summary(m,e,expected_windows=a.expected_windows if not a.diagnostic_only else n,
                              diagnostic_only=a.diagnostic_only)
        by_seed.append({'evaluation_seed':e['args'].get('seed'),'training_seed':int(Path(e['checkpoint']).parent.name.removeprefix('dynamics_seed')),'endpoint_checkpoint':e['checkpoint'],
                        'mixture_checkpoint':m['checkpoint'],'comparison':result})
    summary={}
    for metric in SHARED_METRICS:
        vals=[r['comparison']['metrics'][metric]['signed_error_gap_mixture_minus_endpoint'] for r in by_seed]
        summary[metric]={'mean_over_seeds':float(np.mean(vals)),'per_seed':vals}
    output={'scope':'paired validation development comparison; not promotion or holdout result',
        'protocol_success_assessed':False,'prediction_semantics':'one random observation from each rollout; not a temporally ordered path forecast',
        'seeds':by_seed,'summary':summary}
    Path(a.out).write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
