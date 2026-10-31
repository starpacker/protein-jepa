"""Run verified v7 pilots, then component and frozen-GLE compatibility evaluations."""
import json
import os
from pathlib import Path
import subprocess
import time
import torch


def checked_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    for component in ('encoder','decoder','kernel'):
        if ck.get(component) is None:raise RuntimeError(f'missing {component}')
        if not all(torch.isfinite(v).all() for v in ck[component].values() if v.is_floating_point()):
            raise RuntimeError(f'nonfinite {component}')
    return ck


def run(command,log,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with Path(log).open('w') as f:
        return subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT)


def main():
    # Smoke jobs were already launched; require their success before spending pilot budget.
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        logs=[Path(f'.tmpwork/logs/lasp_v7_{m}_smoke.log').read_text() for m in ('decoder','joint')]
        if any('Traceback (most recent call last)' in x for x in logs):
            raise RuntimeError('smoke failure; pilots not started')
        if all('done: 2 updates' in x for x in logs):break
        time.sleep(10)
    else:raise RuntimeError('smoke timeout; pilots not started')
    for mode in ('decoder','joint'):
        ck=checked_checkpoint(f'artifacts/lasp_v7_{mode}_smoke/checkpoint_latest.pt')
        if ck['step']!=2:raise RuntimeError('smoke checkpoint step mismatch')
    source=checked_checkpoint('artifacts/lasp_v6_pilot_input/v3_best.pt')
    dec=checked_checkpoint('artifacts/lasp_v7_decoder_smoke/checkpoint_latest.pt')
    assert all(torch.equal(v,dec['encoder'][k]) for k,v in source['encoder'].items())
    for mode in ('decoder','joint'):
        ck=checked_checkpoint(f'artifacts/lasp_v7_{mode}_smoke/checkpoint_latest.pt')
        assert all(torch.equal(v,ck['kernel'][k]) for k,v in source['kernel'].items())
        assert any(not torch.equal(v,ck['decoder'][k]) for k,v in source['decoder'].items())
    print('Smoke checkpoint checks passed: finite weights, frozen GLE, frozen decoder-only encoder, updated decoders',flush=True)
    jobs=[]
    for mode,gpu in [('decoder',1),('joint',2)]:
        env=dict(os.environ,MODE=mode,GPU=str(gpu))
        jobs.append(subprocess.Popen(['bash','run_scripts/run_lasp_v7_calibration.sh'],env=env))
    for job in jobs:
        if job.wait()!=0:raise RuntimeError('pilot failed; inspect per-mode log')
    results=Path('results/lasp_v7');results.mkdir(exist_ok=True,parents=True)
    eval_jobs=[]
    for mode,gpu in [('decoder',1),('joint',2)]:
        ck=f'artifacts/lasp_v7_{mode}_pilot/checkpoint_best.pt'
        checked_checkpoint(ck)
        command=['python','-u','scripts/diag_lasp_components.py','--kernel-ckpt',ck,
                 '--out',str(results/f'{mode}_components.json')]
        eval_jobs.append(run(command,f'.tmpwork/logs/lasp_v7_{mode}_components.log',gpu))
    for job in eval_jobs:
        if job.wait()!=0:raise RuntimeError('component evaluation failed')
    subprocess.run(['python','-m','scripts.summarize_lasp_components',
                    'results/lasp_components/v3.json',str(results/'decoder_components.json'),
                    str(results/'joint_components.json'),'--out',str(results/'component_summary.json')],check=True)
    # An unchanged GLE may be incompatible with a changed encoder/decoder. Measure it explicitly.
    for mode,gpu in [('decoder',1),('joint',2)]:
        command=['python','-u','scripts/eval_lasp_final.py',
            '--kernel-ckpt',f'artifacts/lasp_v7_{mode}_pilot/checkpoint_best.pt',
            '--ensemble-ckpt','artifacts/stage_ensemble_1000dom/checkpoint_best.pt',
            '--stage-a-ckpt','artifacts/stage_a_disp_relpos/checkpoint_latest.pt',
            '--b1a-ckpt','artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt',
            '--manifest','artifacts/mdcath_manifest_full/val.jsonl','--device','cuda',
            '--history-frames','4','--max-residues','96','--n-samples','6','--max-windows','40',
            '--max-windows-per-protein','4','--seed','0','--out',str(results/f'{mode}_endpoint.json')]
        job=run(command,f'.tmpwork/logs/lasp_v7_{mode}_endpoint.log',gpu)
        if job.wait()!=0:raise RuntimeError('frozen-GLE compatibility evaluation failed')
        subprocess.run(['python','-m','scripts.compare_paired_ensembles',
                        str(results/f'{mode}_endpoint.per_window.json'),
                        'results/lasp_v6/v3_endpoint.per_window.json','--reference-field','lasp',
                        '--out',str(results/f'{mode}_endpoint_vs_v3.json')],check=True)
    print('Round completed: both pilots, component diagnostics and frozen-GLE endpoint compatibility.',flush=True)


if __name__=='__main__':main()
