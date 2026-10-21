"""Diagnostic teacher-forced Kalman checks; not a training loss or forecast gate.

Uses observed consecutive states with the true physical step, normalized once.
A likelihood improvement here is not evidence of autonomous trajectory quality.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import types
import torch
from evaluation.data import EvalProtocol,iter_windows
from evaluation.loaders import build_bundle
from evaluation.metrics import paired_bootstrap_ci
from evaluation.run_eval import _batch,_jsonable,audit_split,set_determinism,source_provenance
from protein_jepa_frame.data.protocol import split_contiguous_window


def persist_nll(H_seq,rmask,D,h):
    count = int(rmask.sum())*H_seq.shape[-1]*(H_seq.shape[1]-1)
    if count < 1: raise ValueError('empty NLL scoring path')
    variance = (2*h*D).clamp_min(1e-6)
    error = H_seq[:,1:]-H_seq[:,:-1]
    nll = .5*(error.square()/variance[:,None]+torch.log(variance[:,None])+math.log(2*math.pi))
    return float((nll*rmask[:,None,:,None]).sum())/count


def normalized_path(bundle,window,device):
    """Audit physical timestamps with the same continuous-window splitter as training."""
    ctx = _batch(window.context,device)
    path = _batch(window.future_path,device)
    merged = {}
    for key,value in ctx.items():
        if key in {'coords','residue_mask','atom_mask','time_ps','forces','msm_state_labels'} and torch.is_tensor(value):
            merged[key] = torch.cat([value,path[key][:,1:]],dim=1)
        else: merged[key] = value
    K = ctx['coords'].shape[1]
    N = path['coords'].shape[1]-1
    split = split_contiguous_window(merged,K,N,bundle.b1b.cfg.step_ps)
    hist = bundle.normalize(bundle.encode(split['context_coords'],split['context_residue_mask'],
        split['context_aatype'],split['context_residue_index'],split['context_chain_id']))
    sequence = bundle.normalize(bundle.encode(split['future_coords'],split['future_residue_mask'],
        split['future_aatype'],split['future_residue_index'],split['future_chain_id']))
    # Same physical anchor, shared exact tensor despite independent batch encoding.
    sequence = torch.cat([hist[:,-1:],sequence[:,1:]],dim=1)
    return hist,sequence,split


def run(args):
    set_determinism(args.seed,True)
    device = torch.device(args.device)
    bundle = build_bundle(variant=getattr(args,'variant','h_plus_s'),b1b_ckpt=args.ckpt,
        b1a_ckpt=args.b1a_ckpt,device=device,
        diagnostic_comparison=getattr(args,'diagnostic_comparison',False))
    model = bundle.b1b
    protocol = EvalProtocol(args.manifest,horizons_frames=(args.horizon,),context_frames=args.context_frames,
        max_proteins=args.n_prot,anchor_stride=args.anchor_stride,
        max_anchors_per_trajectory=args.max_anchors_per_trajectory)
    cfg = dict(manifest=args.manifest,split=args.split,train_manifests=args.train_manifest)
    split_audit = audit_split(cfg)
    rows = []
    for window in iter_windows(protocol):
        with torch.no_grad():
            history,seq,split = normalized_path(bundle,window,device)
            mask = split['valid_mask'].bool()
            count = int(mask.sum())*(seq.shape[1]-1)*seq.shape[-1]
            if count < 1: raise ValueError('no valid diagnostic state dimensions')
            temp = split.get('temperature_k')
            full,diag = model.kalman_nll(history,split['context_dt_ps'],seq,mask,temp,return_diagnostics=True)
            values = {'full':float(full)/count}
            if model.cfg.use_bath:
                posterior = model.history_posterior.forward
                def prior(*a,**kwargs):
                    b,_,l,_ = a[0].shape; d=model.cfg.bath_dim
                    mean=a[0].new_zeros(b,l,d)
                    chol=torch.eye(d,device=device,dtype=a[0].dtype).expand(b,l,d,d).contiguous()
                    return mean,chol
                try:
                    model.history_posterior.forward = prior
                    values['nohist'] = float(model.kalman_nll(history,split['context_dt_ps'],seq,mask,temp))/count
                finally:
                    model.history_posterior.forward = posterior
                old_dim=model.cfg.bath_dim
                try:
                    model.cfg.bath_dim=0
                    values['nobath'] = float(model.kalman_nll(history,split['context_dt_ps'],seq,mask,temp))/count
                finally:
                    model.cfg.bath_dim=old_dim
            else:
                values['nohist']=values['nobath']=values['full']
            D,_,_=model.bath(seq[:,0],temp)
            values['persist']=persist_nll(seq,mask,D,model.cfg.step_ps)
        rows.append(dict(protein_id=window.protein_id,trajectory_id=window.trajectory_id,
            anchor_frame=window.anchor_frame,target_frame=window.target_frame,temperature_k=window.temperature_k,
            steps=seq.shape[1]-1,step_ps=model.cfg.step_ps,n_scored_dimensions=count,**values,
            jitter_hits=int(diag.get('jitter_hits',0)),bath_ratio=diag.get('bath_ratio')))
    if not rows: raise ValueError('no diagnostic windows')
    proteins={}
    for row in rows:
        proteins.setdefault(row['protein_id'],[]).append(row)
    per_protein=[]
    for protein,rs in sorted(proteins.items()):
        per_protein.append(dict(protein_id=protein,**{key:sum(r[key] for r in rs)/len(rs)
            for key in ['full','nohist','nobath','persist']}))
    differences={other:list(paired_bootstrap_ci([r['full'] for r in per_protein],
        [r[other] for r in per_protein],seed=args.seed)) for other in ['nohist','nobath','persist']}
    out=dict(diagnostic_only=True,headline_eligible=False,
        caveat='teacher-forced latent Kalman likelihood only; no autonomous forecast conclusion',
        meta={**source_provenance(),'bundle':bundle.provenance,'split_audit':split_audit,
              'window_protocol':asdict(protocol),'seed':args.seed},
        rows=rows,per_protein=per_protein,paired_protein_delta_ci=differences)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(_jsonable(out),indent=2,sort_keys=True,allow_nan=False)+'\n')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt',required=True)
    parser.add_argument('--b1a-ckpt')
    parser.add_argument('--variant',choices=['h_markov','h_plus_s'],default='h_plus_s')
    parser.add_argument('--diagnostic-comparison',action='store_true',
                        help='Explicitly score a failed-gate diagnostic checkpoint; never certify it.')
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--train-manifest',action='append',default=[])
    parser.add_argument('--split',choices=['diagnostic','protein_holdout','temperature_subset','temperature_holdout','chronological'],default='diagnostic')
    parser.add_argument('--n-prot',type=int,default=2)
    parser.add_argument('--horizon',type=int,default=4,help='raw frames; every intermediate observation is scored')
    parser.add_argument('--context-frames',type=int,default=8)
    parser.add_argument('--anchor-stride',type=int,default=80)
    parser.add_argument('--max-anchors-per-trajectory',type=int,default=2)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--out',default='artifacts/eval/kalman_diagnostic.json')
    args=parser.parse_args()
    result=run(args)
    print(json.dumps(result['paired_protein_delta_ci'],indent=2))
    print(f'Diagnostic only: wrote {args.out}')


if __name__=='__main__': main()
