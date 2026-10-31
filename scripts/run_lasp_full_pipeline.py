"""Execute the frozen LASP collective-mobility training recipe and evaluation."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('artifacts/lasp_full_v1')
RESULT=Path('results/lasp_full_v1')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(command,gpu,log):
    print(json.dumps(dict(command=command,gpu=gpu)),flush=True)
    with open(log,'a') as stream:
        subprocess.run([sys.executable,'-u','-m',*command],check=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
            stdout=stream,stderr=subprocess.STDOUT)


def train(command,gpu,name,steps):
    import torch
    directory=ROOT/name
    latest=directory/'checkpoint_latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['step']==steps:return
        command=command+['--resume']
    execute(command,gpu,RESULT/(name+'.log'))
    ck=torch.load(latest,map_location='cpu',weights_only=False)
    if ck['step']!=steps:raise RuntimeError(f'incomplete stage {name}')


def recipe():
    c=['protein_jepa_frame.training.train_lasp_calibration',
       '--init-from','artifacts/lasp_v10_geometry/checkpoint_latest.pt',
       '--output',str(ROOT/'representation'),'--decoder-mode','geometric_ca','--joint',
       '--steps','8000','--max-train-samples','0','--stateless-data','--cosine',
       '--warmup','400','--val-batches','64','--val-every','500','--max-residues','96',
       '--bond-projection-steps','32','--selection-bond-limit','.1',
       '--contact-weight','10','--bond-weight','1','--change-weight','1',
       '--angle-weight','2','--clash-weight','50','--teacher-weight','.1',
       '--teacher-delta-weight','.01','--lr','.0001','--seed','42']
    d=['protein_jepa_frame.training.train_lasp_ca_dynamics',
       '--component',str(ROOT/'representation/checkpoint_best.pt'),
       '--collective-mobility','--noise-mix','.8','--noise-length','8',
       '--steps','30000','--train-windows','0','--stateless-data','--cosine',
       '--warmup','1500','--val-windows','64','--val-every','1000',
       '--n-samples','12','--grad-steps','64','--target','path_mixture',
       '--max-residues','96','--lr','.0001','--latent-weight','.1',
       '--contact-objective','brier','--contact-weight','20','--rmsf-weight','1',
       '--clash-weight','40','--mode-weight','2']
    return c,d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
    component,dynamics=recipe()
    manifest=dict(recipe_version='lasp_full_v1',representation_command=component,
        dynamics_command=dynamics,seeds=[42,43,44],
        initialization='Pretrained v10 representation; fresh dynamics per seed, no pilot kernel.',
        sampling='Training target is a future path mixture; sampler emits endpoints.',
        selection='Minimum fixed validation loss per dynamics seed; geometry-gated reconstruction for representation.',
        manifests={p:digest(p) for p in ['artifacts/mdcath_manifest_full/train.jsonl','artifacts/mdcath_manifest_full/val.jsonl']},
        source_sha256={p:digest(p) for p in ['scripts/run_lasp_full_pipeline.py',
            'protein_jepa_frame/training/train_lasp_ca_dynamics.py',
            'protein_jepa_frame/training/train_lasp_calibration.py',
            'protein_jepa_frame/training/lasp_run_state.py',
            'protein_jepa_frame/models/stage_b1b/model.py']})
    frozen=ROOT/'recipe.json'
    if frozen.exists() and json.loads(frozen.read_text())!=manifest:
        raise RuntimeError('frozen recipe/source changed; create a new recipe version')
    frozen.write_text(json.dumps(manifest,indent=2))
    if not args.execute:print(frozen);return
    train(component,2,'representation',8000)
    if not (ROOT/'representation/checkpoint_best.pt').exists():raise RuntimeError('no valid representation')
    def run_seed(seed,gpu):
        name=f'dynamics_seed{seed}'
        train(dynamics+['--output',str(ROOT/name),'--seed',str(seed)],gpu,name,30000)
        out=RESULT/f'{name}_development.json'
        if not out.exists():execute(['scripts.eval_lasp_ca_dynamics','--checkpoint',
            str(ROOT/name/'checkpoint_best.pt'),'--out',str(out)],gpu,RESULT/(name+'_eval.log'))
        if json.loads(out.read_text())['n_windows']!=200:raise RuntimeError('incomplete evaluation')
        execute(['scripts.compare_paired_ensembles',str(out),'results/lasp_v19/bioemu2_complete.json',
            '--strict-finite','--out',str(RESULT/(name+'_vs_bioemu2.json'))],gpu,RESULT/(name+'_compare.log'))
        execute(['scripts.analyze_lasp_bioemu2_gaps',str(out),'results/lasp_v19/bioemu2_complete.json',
            '--out',str(RESULT/(name+'_gaps.json'))],gpu,RESULT/(name+'_compare.log'))
        return str(out)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_seed,seed,gpu) for seed,gpu in [(42,3),(43,4),(44,6)]]
        outputs=[future.result() for future in futures]
    execute(['scripts.summarize_lasp_ca_dynamics',*outputs,'--out',str(RESULT/'summary.json')],2,RESULT/'summary.log')
    (RESULT/'completion.json').write_text(json.dumps(dict(status='complete',outputs=outputs),indent=2))
    print('Full training and development comparison complete',flush=True)


if __name__=='__main__':main()
