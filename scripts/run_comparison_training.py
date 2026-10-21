"""Development-only H gates followed by the preregistered diagnostic runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def launch(command, output, allowed=(0,)):
    output.mkdir(parents=True,exist_ok=True)
    meta_path=output/'launch.json'
    if meta_path.exists():
        prior=json.loads(meta_path.read_text())
        if prior.get('exit_code') in allowed:
            print(f'already completed {output}',flush=True)
            return
        raise RuntimeError(f'incomplete prior run: {output}; inspect before restarting')
    meta=dict(command=command,started_unix=time.time())
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with (output/'run.log').open('w') as f:
        p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    meta.update(exit_code=p.returncode,elapsed_seconds=time.time()-meta['started_unix'])
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),exit_code=p.returncode,elapsed_seconds=meta['elapsed_seconds'])),flush=True)
    if p.returncode not in allowed:
        raise SystemExit(p.returncode)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    args=p.parse_args()
    root=Path(args.root).resolve()
    if 'exit_code' not in json.loads((root/'b1a/launch.json').read_text()):
        raise RuntimeError('wait for the representation calibration to finish')
    if json.loads((root/'b1a/launch.json').read_text())['exit_code']!=0:
        raise RuntimeError('representation calibration failed')
    protocol=yaml.safe_load((root/'experiment.yaml').read_text())
    for role,scope,stride,proteins in [('cross_dev','heldout_protein',53,4),('chrono_dev','diagnostic_fit',9,8),('train','diagnostic_fit',30,8)]:
        out=root/('h_gate_'+role)
        command=[sys.executable,'-u','-m','evaluation.run_h_gate','--b1a-ckpt',str(root/'b1a/checkpoint_best.pt'),
            '--manifest',str(root/(role+'.jsonl')),'--train-manifest',str(root/'train.jsonl'),
            '--scope',scope,'--history-frames','8','--horizons-frames','1','2','4','8',
            '--anchor-stride',str(stride),'--max-proteins',str(proteins),'--max-anchors','8',
            '--torch-threads','2','--output',str(out)]
        launch(command,out,allowed=(0,2))
    budget=protocol['training_budget']
    for seed in protocol['training_seeds']:
        # History usefulness is diagnosed first, then the independently trained
        # matched no-bath and bath kernels. No threshold is relaxed on failure.
        for variant in ['direct_jepa','direct_last','h_markov','h_plus_s','h_plus_s_no_history']:
            out=root/'trained'/f'seed{seed}'/variant
            launch([sys.executable,'-u','-m','protein_jepa_frame.training.train_cached_comparison',
                    '--root',str(root),'--output',str(out),'--variant',variant,'--seed',str(seed),
                    '--steps',str(budget['dynamics_updates_per_variant_seed']),
                    '--validate-every',str(budget['validation_every'])],out)


if __name__=='__main__':
    main()
