"""Single-factor geometry-weight continuation of the completed CA probe."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    jobs=[]
    for weight,gpu in [(1,0),(10,1)]:
        command=[sys.executable,'-u','-m','protein_jepa_frame.training.train_lasp_calibration',
            '--init-from','artifacts/lasp_v8_delta_ca_pilot/checkpoint_latest.pt',
            '--output',f'artifacts/lasp_v8_r2_bond{weight}','--decoder-mode','delta_ca',
            '--steps','300','--max-residues','96','--max-train-samples','1024',
            '--val-batches','4','--val-every','50','--warmup','30',
            '--bond-weight',str(weight),'--contact-weight','2','--change-weight','1']
        with open(f'.tmpwork/logs/lasp_v8_r2_bond{weight}.log','w') as log:
            job=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
        jobs.append((weight,job))
    failures=[weight for weight,job in jobs if job.wait()!=0]
    if failures:raise RuntimeError(f'Failed geometry pilots: {failures}')
    result={}
    for weight,_ in jobs:
        rows=[json.loads(s) for s in Path(f'artifacts/lasp_v8_r2_bond{weight}/metrics.jsonl').read_text().splitlines()]
        result[str(weight)]=[r for r in rows if 'val' in r]
    Path('results/lasp_v8/geometry_summary.json').write_text(json.dumps(result,indent=2))
    with Path('LASP_OPTIMIZATION_JOURNAL.md').open('a') as stream:
        stream.write('\n### R2 完成（自动记录，尚待研究判断）\n\n两组继续训练各 300 updates 完成。详见 `results/lasp_v8/geometry_summary.json`。\n')
        for weight,rows in result.items():
            m=rows[-1]['val']['mean']
            stream.write(f"- bond={weight}: final RMSD={m['ca_rmsd']:.6f}, fixed={m['fixed_ca_rmsd']:.6f}, shuffled={m['shuffled_ca_rmsd']:.6f}, CA bond={m['ca_bond']:.6f}, RMSF ratio={m['rmsf_ratio']:.6f}.\n")
    print('Geometry round completed.',flush=True)


if __name__=='__main__':main()
