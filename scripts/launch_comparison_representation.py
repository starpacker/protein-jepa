"""Run a bounded fresh A/B1a calibration; never touch final test scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    common = ['--manifest', str(root/'train.jsonl'), '--val-manifest', str(root/'cross_dev.jsonl'),
              '--device','cpu','--torch-threads','2','--history-frames','8',
              '--paired-horizons-frames','1','2','4','8','--batch-size','2','--num-workers','0',
              '--max-residues','128','--max-train-samples','1024','--seed','42',
              '--val-every','100','--val-batches','16','--save-every','100','--log-every','50']
    stages = [
        ('stage_a', ['-m','protein_jepa_frame.training.train_stage_a_v2',
                     '--latent-dim','32','--num-factors','4','--num-heads','4',
                     '--encoder-depth','2','--predictor-depth','2','--predictor-spatial-depth','1',
                     '--temporal-lags','0','--struct-recon-dim','64','--struct-recon-pairs','64',
                     '--steps','600','--warmup-steps','50']),
        ('b1a', ['-m','protein_jepa_frame.training.train_stage_b1a',
                 '--stage-a-ckpt', str(root/'stage_a/checkpoint_best.pt'),
                 '--mode','oracle','--unfreeze-top-n-blocks','0',
                 '--decoder-hidden','64','--decoder-blocks','2','--decoder-heads','4','--decoder-dropout','0',
                 '--num-distance-pairs','16','--fit-normalizer-batches','32',
                 '--steps','2000','--warmup-steps','50','--lr','0.001',
                 '--fape-weight','0','--torsion-weight','0','--distance-weight','0.1',
                 '--phys-future-weight','0','--anchor-weight','0','--ca-mse-weight','1']),
    ]
    for name, options in stages:
        out = root/name
        out.mkdir(exist_ok=True)
        if (out/'launch.json').exists():
            raise FileExistsError(f'{out} already launched')
        command = [sys.executable,'-u'] + options + common + ['--output',str(out)]
        meta = dict(command=command, started_unix=time.time(),scope='development; fresh chronological-train representation')
        (out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(f'launch {name}', flush=True)
        with (out/'train.log').open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        meta.update(exit_code=result.returncode, elapsed_seconds=time.time()-meta['started_unix'])
        (out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps(meta), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
