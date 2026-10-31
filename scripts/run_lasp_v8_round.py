"""Run bounded decoder pilots after checking the increment-decoder smoke run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def main():
    log = Path('.tmpwork/logs/lasp_v8_delta_smoke.log')
    for _ in range(120):
        text = log.read_text() if log.exists() else ''
        if 'Traceback (most recent call last)' in text:
            raise RuntimeError('smoke failed')
        if 'done: 2 updates' in text:
            break
        time.sleep(5)
    else:
        raise RuntimeError('smoke timed out')
    smoke = torch.load('artifacts/lasp_v8_delta_smoke/checkpoint_latest.pt',map_location='cpu',weights_only=False)
    source = torch.load('artifacts/lasp_v6_pilot_input/v3_best.pt',map_location='cpu',weights_only=False)
    assert smoke['checkpoint_kind'] == 'ca_component_probe'
    assert all(torch.equal(v,smoke['encoder'][k]) for k,v in source['encoder'].items())
    assert all(torch.isfinite(v).all() for v in smoke['delta_ca_decoder'].values())
    jobs = []
    for mode, gpu in [('legacy', 1), ('delta_ca', 0)]:
        output = f'artifacts/lasp_v8_{mode}_pilot'
        command = [sys.executable,'-u','-m','protein_jepa_frame.training.train_lasp_calibration',
                   '--output',output,'--decoder-mode',mode,'--steps','300',
                   '--max-residues','96','--max-train-samples','1024',
                   '--val-batches','4','--val-every','50','--warmup','30',
                   '--bond-weight','1','--contact-weight','2','--change-weight','1']
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with open(f'.tmpwork/logs/lasp_v8_{mode}_pilot.log','w') as stream:
            jobs.append((mode,subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)))
    failures = []
    for mode, job in jobs:
        if job.wait() != 0:
            failures.append(mode)
    if failures:
        raise RuntimeError(f'pilot failures: {failures}')
    summary = {}
    for mode, _ in jobs:
        rows = [json.loads(line) for line in Path(f'artifacts/lasp_v8_{mode}_pilot/metrics.jsonl').read_text().splitlines()]
        validation = [row for row in rows if 'val' in row]
        summary[mode] = dict(initial=validation[0],final=validation[-1],
                             best_rmsd=min(validation,key=lambda row:row['val']['mean']['ca_rmsd']))
    dest = Path('results/lasp_v8');dest.mkdir(parents=True,exist_ok=True)
    (dest/'round_summary.json').write_text(json.dumps(summary,indent=2))
    with Path('LASP_OPTIMIZATION_JOURNAL.md').open('a') as stream:
        stream.write('\n### R1 训练完成（自动汇总，尚待研究判断）\n\n')
        stream.write('两组各 300 updates 完成。结果：`results/lasp_v8/round_summary.json`。\n\n')
        for mode, item in summary.items():
            initial=item['initial']['val']['mean'];best=item['best_rmsd']['val']['mean']
            stream.write(f"- {mode}: initial RMSD={initial['ca_rmsd']:.6f}; best RMSD={best['ca_rmsd']:.6f}; "
                         f"RMSF ratio={best['rmsf_ratio']:.6f}; CA bond MAE={best['ca_bond']:.6f}.\n")
        stream.write('\n自动完成不代表晋级；需检查逐 stride 指标、fixed/shuffled 对照与几何退化。\n')
    print('Both pilots completed; summary and journal updated.',flush=True)


if __name__ == '__main__':
    main()
