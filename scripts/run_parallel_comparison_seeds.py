"""Run independent remaining seeds in isolated directories, then publish locally.

The original sequential runner can reuse completed directories. No active or
completed canonical run is overwritten; all data, weights and budgets agree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def job(root,seed,variant,budget):
    final=root/'trained'/f'seed{seed}'/variant
    if final.exists():
        return dict(seed=seed,variant=variant,status='canonical_run_already_exists')
    out=root/'parallel_work'/f'seed{seed}'/variant
    out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,'-u','-m','protein_jepa_frame.training.train_cached_comparison',
        '--root',str(root),'--output',str(out),'--variant',variant,'--seed',str(seed),
        '--steps',str(budget['dynamics_updates_per_variant_seed']),
        '--validate-every',str(budget['validation_every'])]
    meta=dict(command=cmd,started_unix=time.time(),isolated_work_directory=str(out),final_directory=str(final))
    (out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with (out/'run.log').open('w') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
    meta.update(exit_code=result.returncode,elapsed_seconds=time.time()-meta['started_unix'])
    (out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
    if result.returncode:
        return dict(seed=seed,variant=variant,status='failed',exit_code=result.returncode,directory=str(out))
    final.parent.mkdir(parents=True,exist_ok=True)
    if final.exists():
        return dict(seed=seed,variant=variant,status='isolated_complete_canonical_exists',directory=str(out))
    out.rename(final)
    return dict(seed=seed,variant=variant,status='completed',elapsed_seconds=meta['elapsed_seconds'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    root=Path(args.root).resolve()
    protocol=yaml.safe_load((root/'experiment.yaml').read_text())
    variants=['direct_jepa','direct_last','h_markov','h_plus_s','h_plus_s_no_history']
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(job,root,seed,v,protocol['training_budget']) for seed in args.seeds for v in variants]
        rows=[]
        for f in as_completed(jobs):
            row=f.result();rows.append(row)
            print(json.dumps(row),flush=True)
    (root/'parallel_training_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    if any(r['status']=='failed' for r in rows):
        raise SystemExit(1)


if __name__=='__main__':
    main()
