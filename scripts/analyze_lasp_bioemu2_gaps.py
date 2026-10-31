"""Analyze strictly paired completed forecasts by horizon, condition and protein."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.compare_paired_ensembles import index_rows
from scripts.eval_lasp_final import METRICS,HIGHER_BETTER
from scripts.eval_lasp_ca_probe import bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate');p.add_argument('reference');p.add_argument('--out',required=True)
    a=p.parse_args();c,r=index_rows(a.candidate),index_rows(a.reference)
    if c.keys()!=r.keys():raise ValueError('exact window coverage required')
    rows=[]
    for k,x in c.items():
        y=r[k];v=dict(protein=k[0],temperature=k[2],horizon=k[4])
        for m in METRICS:
            cv,rv=x['lasp'][m],y['bioemu2'][m]
            if not np.isfinite([cv,rv]).all():raise ValueError('nonfinite metric')
            v[m]=(1 if m in HIGHER_BETTER else -1)*(cv-rv)
            v['candidate_'+m]=cv;v['reference_'+m]=rv
        rows.append(v)
    def summarize(selected):
        return {m:dict(improvement=bootstrap(selected,m),
            candidate=bootstrap(selected,'candidate_'+m),reference=bootstrap(selected,'reference_'+m)) for m in METRICS}
    result=dict(scope='Development diagnostic; positive improvement favors LASP',n_windows=len(rows),
        aggregate=summarize(rows),by_horizon={str(h):summarize([x for x in rows if x['horizon']==h]) for h in sorted({x['horizon'] for x in rows})},
        by_temperature={str(t):summarize([x for x in rows if x['temperature']==t]) for t in sorted({x['temperature'] for x in rows})},
        by_protein={p:{m:float(np.mean([x[m] for x in rows if x['protein']==p])) for m in METRICS} for p in sorted({x['protein'] for x in rows})})
    Path(a.out).write_text(json.dumps(result,indent=2))
    for m,v in result['aggregate'].items():print(m,v['candidate']['mean'],v['reference']['mean'],v['improvement']['ci95'])


if __name__=='__main__':main()
