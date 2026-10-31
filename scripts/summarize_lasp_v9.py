"""Summarize completed or ongoing decoder development trials without promoting them."""
import json
from pathlib import Path


def main():
    result={}
    for directory in sorted(Path('artifacts').glob('lasp_v9_*')):
        metrics=directory/'metrics.jsonl'
        if not metrics.exists():continue
        rows=[json.loads(line) for line in metrics.read_text().splitlines()]
        vals=[row for row in rows if 'val' in row]
        finite=[row for row in vals if all(__import__('math').isfinite(x) for x in row['val']['mean'].values())]
        if not finite:
            result[directory.name]=dict(status='invalid_validation_discarded');continue
        feasible=[row for row in finite if row['val']['mean']['ca_bond']<=.1]
        result[directory.name]=dict(last_step=rows[-1]['step'],initial=finite[0]['val']['mean'],
            latest=finite[-1]['val']['mean'],
            feasible_best=min(feasible,key=lambda row:row['val']['mean']['ca_rmsd']) if feasible else None)
    out=Path('results/lasp_v9/development_summary.json');out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2))
    for name,row in result.items():
        if 'latest' in row:print(name,row['last_step'],row['latest']['ca_rmsd'],row['latest']['ca_bond'])
        else:print(name,row['status'])


if __name__=='__main__':main()
