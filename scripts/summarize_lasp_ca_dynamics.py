"""Summarize fixed-window autonomous development forecasts, clustered by protein."""
import argparse
import json
from pathlib import Path
from scripts.eval_lasp_ca_probe import bootstrap
from scripts.eval_lasp_final import METRICS, HIGHER_BETTER


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args();summary={}
    for path in a.inputs:
        d=json.loads(Path(path).read_text());rows=d['per_window']
        aggregate={}
        for metric in METRICS:
            direction=1 if metric in HIGHER_BETTER else -1
            values=[dict(protein=r['protein'],candidate=r['lasp'][metric],
                persist=r['persist'][metric],improvement=direction*(r['lasp'][metric]-r['persist'][metric])) for r in rows]
            aggregate[metric]={key:bootstrap(values,key) for key in ('candidate','persist','improvement')}
        geometry={}
        if all('geometry_vs_anchor' in r for r in rows):
            for key in rows[0]['geometry_vs_anchor']:
                geometry[key]=bootstrap([dict(protein=r['protein'],value=r['geometry_vs_anchor'][key]) for r in rows],'value')
        contacts={}
        if all('contact_counts' in r for r in rows):
            for metric,prefix in [('weak_contact_j','weak'),('transient_contacts_j','transient')]:
                eligible=[r for r in rows if r['contact_counts'].get(prefix+'_n_true',0)>0]
                if eligible:
                    contacts[metric]=dict(n_windows=len(eligible),
                        candidate=bootstrap([dict(protein=r['protein'],value=r['lasp'][metric]) for r in eligible],'value'),
                        persist=bootstrap([dict(protein=r['protein'],value=r['persist'][metric]) for r in eligible],'value'))
        summary[Path(path).stem]=dict(n_windows=len(rows),metrics=aggregate,geometry_vs_anchor=geometry,
            nonempty_true_contacts=contacts)
    Path(a.out).write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
