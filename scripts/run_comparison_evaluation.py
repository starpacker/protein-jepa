"""Score all locked checkpoints on both final-test scopes, including ablations."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from evaluation.loaders import sha256_file
from evaluation.run_eval import run, write_reports
from evaluation.preregister import create_registration


VARIANTS=('direct_jepa','direct_last','h_markov','h_plus_s','h_plus_s_no_history')


def lock_selection(root, protocol):
    """Lock all weights before the first test score; no favorable seed selection."""
    paths={}
    for seed in protocol['training_seeds']:
        for variant in VARIANTS:
            p=root/'trained'/f'seed{seed}'/variant/'checkpoint_best.pt'
            ck=torch.load(p,map_location='cpu',weights_only=False)
            if ck['training_budget']['steps']!=protocol['training_budget']['dynamics_updates_per_variant_seed']:
                raise ValueError('checkpoint training budget differs from registration')
            # Best checkpoint may precede final update, but the full training
            # budget must have completed before opening the test set.
            summary=json.loads(p.with_name('summary.json').read_text())
            latest=torch.load(p.with_name('checkpoint_latest.pt'),map_location='cpu',weights_only=False)
            if latest['step']!=protocol['training_budget']['dynamics_updates_per_variant_seed']:
                raise ValueError('incomplete training budget')
            paths[f'{seed}/{variant}']=dict(path=str(p),sha256=sha256_file(str(p)),
                selected_step=ck['step'],best_development_score=ck['best_score'],
                parameter_counts=summary['parameter_counts'],data_schedule_sha256=summary['data_schedule_sha256'])
    for seed in protocol['training_seeds']:
        if len({paths[f'{seed}/{v}']['data_schedule_sha256'] for v in VARIANTS})!=1:
            raise ValueError('variants trained on different sampled windows/horizons')
    locked=dict(preregistration_sha256=sha256_file(str(root/'preregistration.json')),
        selection='checkpoint_best by preregistered development score, all training seeds retained',checkpoints=paths)
    dest=root/'selection_lock.json'
    if dest.exists():
        if json.loads(dest.read_text())!=locked:
            raise ValueError('a selected weight or protocol changed after opening test')
    else:
        with dest.open('x') as f: json.dump(locked,f,indent=2)
    return locked


def score_config(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text())
    result=run(cfg)
    write_reports(result,Path(config_path).parent)
    print(f'completed {result["n_windows"]} windows -> {Path(config_path).parent}',flush=True)


def worker(path):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with path.with_name('run.log').open('w') as f:
        result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),
            '--worker-config',str(path)],stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        raise RuntimeError(f'evaluation failed ({result.returncode}); inspect {path.with_name("run.log")}')
    return str(path.parent)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root')
    p.add_argument('--worker-config')
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.worker_config:
        score_config(args.worker_config)
        return
    if not args.root:
        p.error('--root or --worker-config required')
    root=Path(args.root).resolve()
    protocol=yaml.safe_load((root/'experiment.yaml').read_text())
    locked=lock_selection(root,protocol)
    # The ablation uses the same scoring protocol and independently trained,
    # same-budget last-frame direct/no-history posterior models. Register its
    # interpretation before any test score as a separate, secondary comparison.
    ablation_path=root/'ablation_preregistration.json'
    if not ablation_path.exists():
        ablation=dict(protocol,limitations=protocol['limitations']+[
            'Secondary ablation: direct_jepa row loads direct_last; h_plus_s row loads h_plus_s_no_history. '
            'These models were independently retrained with repeated last H; same parameters and budget.'])
        with ablation_path.open('x') as f: json.dump(create_registration(ablation),f,indent=2)
    jobs=[]
    for group in ('main','no_history_models'):
        for seed in protocol['training_seeds']:
            base=root/'trained'/f'seed{seed}'
            for scope,spec in protocol['scopes'].items():
                dest=root/'evaluation'/group/f'seed{seed}'/scope
                if (dest/'results.json').exists():
                    print(f'already scored {dest}',flush=True)
                    continue
                direct='direct_jepa' if group=='main' else 'direct_last'
                bath='h_plus_s' if group=='main' else 'h_plus_s_no_history'
                cfg={k:v for k,v in spec.items() if k!='manifests'}
                cfg.update(name=f'{group}_seed{seed}_{scope}',manifest=spec['manifests']['test'],
                    train_manifests=[str(root/'train.jsonl')],device='cpu',
                    preregistration=str(root/('preregistration.json' if group=='main' else 'ablation_preregistration.json')),
                    comparison_scope=scope,dataset_role='final_test',diagnostic_comparison=True,
                    models=dict(b1a_ckpt=str(root/'b1a/checkpoint_best.pt')),
                    variant_models=dict(h_markov=dict(b1b_ckpt=str(base/'h_markov/checkpoint_best.pt')),
                        h_plus_s=dict(b1b_ckpt=str(base/bath/'checkpoint_best.pt')),
                        direct_jepa=dict(direct_ckpt=str(base/direct/'checkpoint_best.pt'))),
                    limitations=protocol['limitations']+[
                        f'Group={group}; direct row training_variant={direct}; bath row training_variant={bath}',
                        'No oracle-normalized skill is interpretable when the oracle error exceeds persistence.',
                        'Training seeds share one frozen encoder/readout; uncertainty excludes representation retraining.'])
                dest.mkdir(parents=True,exist_ok=True)
                (dest/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
                jobs.append(dest/'config.yaml')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,p) for p in jobs]
        for f in as_completed(futures):
            print(f'completed {f.result()}',flush=True)


if __name__=='__main__':
    main()
