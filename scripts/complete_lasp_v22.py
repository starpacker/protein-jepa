"""Finish fixed v22 development comparisons after the three training runs end."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',choices=['v22','v23'],default='v22')
    args=parser.parse_args();version=args.version
    steps=600 if version=='v22' else 400
    root=Path('results')/('lasp_'+version)
    jobs=([('control',3),('distribution_1',4),('distribution_4',6)] if version=='v22' else
        [('legacy_es',2),('collective_es',3),('legacy_multi',4),('collective_multi',6)])
    running=[]
    pending=list(jobs)
    while pending:
        for label,gpu in pending[:]:
            artifact=Path('artifacts')/f'lasp_{version}_{label}'
            log=root/f'{label}.log'
            if not log.exists() or not log.read_text().rstrip().endswith('done'):
                continue
            rows=[json.loads(x) for x in (artifact/'metrics.jsonl').read_text().splitlines()]
            if rows[-1].get('step')!=steps or 'val' not in rows[-1]:
                raise RuntimeError(f'incomplete training: {label}')
            output=root/f'{label}_selected.json'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            handle=open(root/f'{label}_eval.log','w')
            proc=subprocess.Popen(['python','-m','scripts.eval_lasp_ca_dynamics',
                '--checkpoint',str(artifact/'checkpoint_best.pt'),'--out',str(output)],
                env=env,stdout=handle,stderr=subprocess.STDOUT)
            running.append((label,proc,handle,output));pending.remove((label,gpu))
            print(f'evaluation started: {label}, pid={proc.pid}',flush=True)
        if pending:time.sleep(15)
    for label,proc,handle,output in running:
        if proc.wait()!=0:raise RuntimeError(f'evaluation failed: {label}')
        handle.close()
        if json.loads(output.read_text())['n_windows']!=200:raise RuntimeError('incomplete evaluation')
    def run(module,*args):
        subprocess.run(['python','-m',module,*map(str,args)],check=True)
    paths=[output for _,_,_,output in running]
    run('scripts.summarize_lasp_ca_dynamics',*paths,'--out',root/'summary.json')
    for label,_,_,output in running:
        for refname,ref,field in [('bioemu2','results/lasp_v19/bioemu2_complete.json','bioemu2'),
            ('parent','results/lasp_v20/v20_spatial_0.8_final.json','lasp'),
            ('control',str(root/('control_selected.json' if version=='v22' else 'legacy_es_selected.json')),'lasp')]:
            run('scripts.compare_paired_ensembles',output,ref,'--reference-field',field,
                '--strict-finite','--out',root/f'{label}_vs_{refname}.json')
        run('scripts.analyze_lasp_bioemu2_gaps',output,'results/lasp_v19/bioemu2_complete.json',
            '--out',root/f'{label}_gaps.json')
    with Path('LASP_OPTIMIZATION_JOURNAL.md').open('a') as f:
        f.write(f'\nAutomated {version} completion: all {steps}-step runs and selected-checkpoint\n'
            f'200-window evaluations finished; strict comparisons and geometry summary in {root}.\n'
            'These outputs require scientific interpretation; completion is not a superiority claim.\n')
    (root/'completion.json').write_text(json.dumps(dict(status='complete',evaluations=list(map(str,paths))),indent=2))
    print(f'{version} comparisons complete',flush=True)


if __name__=='__main__':main()
