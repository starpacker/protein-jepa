"""Observer-conditioned Kalman diagnostics, kept separate from forecast scores."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
import yaml

from evaluation.run_gate import run
from evaluation.loaders import sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(args.root).resolve()
    protocol=yaml.safe_load((root/'experiment.yaml').read_text())
    rows=[]
    for seed in protocol['training_seeds']:
        for scope,spec in protocol['scopes'].items():
            for variant in ('h_markov','h_plus_s'):
                out=root/'kalman_diagnostics'/f'{scope}_seed{seed}_{variant}.json'
                a=SimpleNamespace(seed=seed,device='cpu',variant=variant,diagnostic_comparison=True,
                    ckpt=str(root/'trained'/f'seed{seed}'/variant/'checkpoint_best.pt'),
                    b1a_ckpt=str(root/'b1a/checkpoint_best.pt'),manifest=spec['manifests']['test'],
                    train_manifest=[str(root/'train.jsonl')],split=spec['split'],
                    horizon=8,context_frames=8,n_prot=8 if scope=='chronological' else 4,
                    anchor_stride=spec['anchor_stride'],max_anchors_per_trajectory=2,out=str(out))
                with sdpa_kernel([SDPBackend.MATH]):
                    result=run(a)
                rows.append(dict(seed=seed,scope=scope,variant=variant,
                    per_protein=result['per_protein'],paired_protein_delta_ci=result['paired_protein_delta_ci'],
                    jitter_hits=sum(r['jitter_hits'] for r in result['rows']),windows=len(result['rows'])))
                print(f'Kalman diagnostic {seed} {scope} {variant}',flush=True)
    payload=dict(diagnostic_only=True,training_nll_weight=0.,
        caveat='teacher-forced latent observer score; never an optimized loss or autonomous prediction metric',
        selection_lock_sha256=sha256_file(str(root/'selection_lock.json')),rows=rows)
    (root/'kalman_diagnostics/summary.json').write_text(json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    main()
