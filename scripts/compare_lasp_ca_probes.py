"""Strict paired protein-level comparisons of oracle CA probe records."""
import argparse
import json
from pathlib import Path
from scripts.eval_lasp_ca_probe import bootstrap


def key(row):
    return (row['protein'],row['trajectory'],row['stride'],tuple(row['time_ps'][0]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate');p.add_argument('reference');p.add_argument('--out',required=True)
    a=p.parse_args()
    candidate=json.loads(Path(a.candidate).read_text());reference=json.loads(Path(a.reference).read_text())
    def indexed(data):
        rows={key(row):row for row in data['rows']}
        if len(rows)!=len(data['rows']):raise ValueError('duplicate window')
        return rows
    c,r=indexed(candidate),indexed(reference)
    if c.keys()!=r.keys():raise ValueError('window keys differ')
    rows=[]
    for k,x in c.items():
        y=r[k]
        row=dict(protein=x['protein'])
        for metric in ['ca_rmsd','rmsf_mae','ca_bond','contact','ca_angle_cos_mae','ca_clash_fraction']:
            row[metric]=y[metric]-x[metric]
        row['rmsf_corr']=x['rmsf_corr']-y['rmsf_corr']
        rows.append(row)
    result=dict(scope='Oracle component only; positive improvement is better',
        candidate=candidate['checkpoint'],reference=reference['checkpoint'],n_windows=len(rows),
        improvement={m:bootstrap(rows,m) for m in rows[0] if m!='protein'})
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
