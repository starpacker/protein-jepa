"""Bounded diagnostic comparison on a frozen representation.

This entry point cannot certify a failed oracle gate. All outputs remain
diagnostic_comparison. It reuses the production history-only sampler and raw-H
decoder, while caching only deterministic frozen encoder outputs. Test labels
are not cached or accessed during training/model selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.models.stage_a_v2.model import CausalTransformerPredictor
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_stage_b1b import (
    load_frozen_representation, encode_frames, autonomous_samples, decode_paths,
    normalized_path_energy_score, displacement_point_loss,
)
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from evaluation.geometry import physical_distance_energy_score
from evaluation.loaders import sha256_file


class DirectJEPAPredictor(nn.Module):
    """JEPA temporal+spatial endpoint architecture in normalized H coordinates.

One full-width output head is a no-OPF direct baseline. The wrapper returns a
delta, never an encoded future. Inputs are H history, physical lag, temperature,
and the observed validity mask; sequence is not a separate predictor input.
"""
    def __init__(self, latent_dim=32, depth=1, spatial_depth=1, num_heads=4, dropout=0.0):
        super().__init__()
        self.network = CausalTransformerPredictor(latent_dim=latent_dim,
            num_factors=1, factor_dim=latent_dim, depth=depth, num_heads=num_heads,
            spatial_depth=spatial_depth, dropout=dropout)
        # These embeddings serve masked spatial queries during A; this cached
        # temporal-only comparison never uses them. Do not count as trainable.
        for name in ('query_aatype','query_residue','query_chain'):
            getattr(self.network, name).requires_grad_(False)
        self.network.mask_query.requires_grad_(False)
        nn.init.zeros_(self.network.out_head.weight)
        nn.init.zeros_(self.network.out_head.bias)

    def forward(self, H, lag_ps, temperature_k, valid_mask=None):
        return self.network(H,lag_ps,temperature_k,valid_mask=valid_mask,task_id=1).squeeze(-2)


@torch.no_grad()
def make_cache(root, b1a_path):
    root = Path(root)
    encoder, oracle, bundle = load_frozen_representation(str(b1a_path), 'cpu')
    rid = bundle['representation_contract']['representation_id']
    dest = root/'comparison_cache.pt'
    roles = ('train','chrono_dev','cross_dev')
    expected = {k:sha256_file(str(root/(k+'.jsonl'))) for k in roles}
    if dest.exists():
        cache = torch.load(dest,map_location='cpu',weights_only=False)
        if (cache['representation_id'] != rid or cache['manifests'] != expected
                or cache['b1a_checkpoint_sha256'] != sha256_file(str(b1a_path))):
            raise ValueError('stale frozen-representation cache')
        return cache, oracle, bundle
    ds = TrajectoryClipDataset.__new__(TrajectoryClipDataset)
    ds.config=DataConfig(max_residues=None)
    ds._init_auxiliary_state(None)
    splits={}
    for role in roles:
        trajectories=[]
        for record in load_manifest(root/(role+'.jsonl')):
            m=record.metadata or {}
            if sha256_file(str(record.path)) != m['extracted_npz_sha256']:
                raise ValueError('materialized trajectory content changed')
            sample=ds._read_clip(record,np.arange(record.num_frames))
            h=[]
            for start in range(0,record.num_frames,32):
                end=min(start+32,record.num_frames)
                value=encode_frames(encoder,sample['coords'][None,start:end],
                    sample['residue_mask'][None,start:end],sample['aatype'][None],
                    sample['residue_index'][None],sample['chain_id'][None],oracle.h_normalizer)
                h.append(value[0])
            trajectories.append(dict(sample, H=torch.cat(h),
                original_frame_start=m['original_frame_start'],npz_sha256=m['extracted_npz_sha256']))
        splits[role]=trajectories
    cache=dict(schema_version=2,representation_id=rid,
        b1a_checkpoint_sha256=sha256_file(str(b1a_path)),manifests=expected,
        H_units='train-only normalized, frozen eval encoder, detached labels',splits=splits)
    torch.save(cache,dest)
    return cache,oracle,bundle


def window(record, anchors, K=8, N=8):
    """Construct contiguous, fixed-topology history plus future supervision."""
    anchors=torch.as_tensor(anchors,dtype=torch.long)
    ix=anchors[:,None]+torch.arange(-K+1,N+1)[None]
    if ix.min()<0 or ix.max()>=len(record['H']):
        raise ValueError('complete window must remain within its partition')
    coords=record['coords'][ix]
    masks=record['residue_mask'][ix].bool()
    mask=masks[:,K-1]
    if not torch.equal(masks,mask[:,None].expand_as(masks)):
        raise ValueError('varying future topology cannot enter forecast conditioning')
    times=record['time_ps'][ix]
    if not torch.allclose(times.diff(dim=1),torch.full_like(times[:,1:],1000.)):
        raise ValueError('comparison requires actual uniform 1000ps sampling')
    B=len(anchors)
    batch=dict(coords=coords,residue_mask=masks,time_ps=times,
        **{k:record[k][None].expand(B,-1) for k in ('aatype','residue_index','chain_id')})
    return dict(batch=batch,history=record['H'][ix[:,:K]].detach(),
        target=record['H'][ix[:,K:]].detach(),mask=mask,
        dt=torch.full((B,K),1000.),temperature=torch.full((B,),320.),horizon=N,
        protein_id=record['protein_id'],anchor_indices=anchors.tolist())


def predict(model, variant, w, *, generator, training, samples=2):
    """Causal producer API has no future label or future coordinates argument."""
    return forecast_from_history(model,variant,w['history'],w['dt'],w['mask'],
        w['temperature'],w['horizon'],generator=generator,training=training,samples=samples)


def forecast_from_history(model,variant,history,dt,mask,temperature,horizon,*,generator,training,samples=2):
    h=history
    if variant in {'direct_last','h_plus_s_no_history'}:
        h=h[:,-1:].expand_as(h)
    if variant.startswith('direct'):
        pred=h[:,-1]+model(h,temperature.new_full((len(h),),horizon*1000.),temperature,valid_mask=mask)
        # Point-mass direct prediction; duplicate draws do not add uncertainty.
        return pred[None,:,None].expand(samples,-1,-1,-1,-1)
    paths,_=autonomous_samples(model,h,dt,mask,temperature,horizon,samples,
                               generator=generator,training=training)
    return paths[:,:,-1:]


def endpoint_loss(samples, w, oracle, *, physical_weight=0.1, point_weight=0.0):
    # Future normalized H are detached labels only. Decode ONLY predicted H.
    latent=normalized_path_energy_score(samples,w['target'][:,-1:],w['mask'])
    bb=decode_paths(oracle,samples,w['batch'],w['mask'],8,w['history'][:,-1])
    truth=w['batch']['coords'][:,-1:]
    physical=[]
    for b in range(len(w['history'])):
        physical.append(physical_distance_energy_score(bb[:,b,0,:,1],truth[b,0,:,1].detach(),
            w['mask'][b],scale_angstrom=10.,max_pairs=128,return_tensor=True))
    physical=torch.stack(physical).mean()
    point=displacement_point_loss(bb,truth,w['batch']['coords'][:,7],w['mask'])
    total=latent+physical_weight*physical+point_weight*point
    return total,dict(loss=float(total.detach()),latent_es=float(latent.detach()),
                      physical_es=float(physical.detach()),primary_rmsd=float(point.detach()))


@torch.no_grad()
def validate(model,variant,cache,oracle,seed):
    model.eval()
    rows=[]
    with sdpa_kernel([SDPBackend.MATH]):
        for role in ('chrono_dev','cross_dev'):
            for record in cache['splits'][role]:
                anchors=torch.linspace(7,len(record['H'])-9,2).long()
                for horizon in (1,2,4,8):
                    w=window(record,anchors,N=horizon)
                    gen=torch.Generator().manual_seed(seed+1234567+horizon)
                    samples=predict(model,variant,w,generator=gen,training=False,samples=4)
                    _,logs=endpoint_loss(samples,w,oracle)
                    rows.append(dict(role=role,protein_id=record['protein_id'],horizon=horizon,**logs))
    # Each role, protein and horizon has equal selection weight. No test data.
    score=sum(np.mean([r['loss'] for r in rows if r['role']==role])
              for role in ('chrono_dev','cross_dev'))/2
    return dict(selection_metric='mean of role-equal protein/horizon-balanced latent_ES + .1*distance_ES',
                score=float(score),rows=rows)


def train(args):
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    root=Path(args.root).resolve()
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint_latest.pt').exists():
        raise FileExistsError('never overwrite a completed comparison run')
    b1a_path=(root/'b1a/checkpoint_best.pt').resolve()
    cache,oracle,bundle=make_cache(root,b1a_path)
    report_path=root/'h_gate_cross_dev/report.json'
    gate=json.loads(report_path.read_text())
    if gate['representation_id'] != cache['representation_id']:
        raise ValueError('gate and representation mismatch')
    variant=args.variant
    cfg=StageB1bConfig(latent_dim=32,bath_dim=0 if variant=='h_markov' else 8,
        energy_depth=1,energy_heads=4,energy_hidden=64,history_hidden=64,max_gamma_step=1.9)
    direct_config=dict(latent_dim=32,depth=1,spatial_depth=1,num_heads=4,dropout=0.0)
    model=DirectJEPAPredictor(**direct_config) if variant.startswith('direct') else StageB1b(cfg)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr,weight_decay=0.01)
    # The data schedule is independent of model RNG/noise and identical across variants.
    data_rng=np.random.default_rng(args.seed)
    noise=torch.Generator().manual_seed(args.seed+2000003)
    best=float('inf')
    start=time.time()
    history=[]
    schedule_hash=hashlib.sha256()
    for step in range(1,args.steps+1):
        model.train()
        record=cache['splits']['train'][int(data_rng.integers(len(cache['splits']['train'])))]
        N=(1,2,4,8)[(step-1)%4]
        anchors=data_rng.integers(7,len(record['H'])-N,size=2).tolist()
        schedule_hash.update(json.dumps([step,record['protein_id'],anchors,N]).encode())
        w=window(record,anchors,N=N)
        with sdpa_kernel([SDPBackend.MATH]):
            smp=predict(model,variant,w,generator=noise,training=True,samples=2)
            loss,logs=endpoint_loss(smp,w,oracle,physical_weight=.1,point_weight=0.)
            if not torch.isfinite(loss) or not torch.isfinite(smp).all():
                raise FloatingPointError(f'nonfinite forecast/loss at {step}; no omission allowed')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms={}
            if step==1:
                for name,prefix in [('energy','energy.'),('bath','bath.'),('posterior','history_posterior.'),('direct','network.')]:
                    grads=[p.grad.detach().square().sum() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    norms[name]=float(torch.stack(grads).sum().sqrt()) if grads else 0.
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        if step==1 or step%50==0:
            row=dict(step=step,elapsed_seconds=time.time()-start,**logs)
            if norms: row['gradient_norms']=norms
            print(json.dumps(row),flush=True)
            history.append(row)
        if step%args.validate_every==0 or step==args.steps:
            validation=validate(model,variant,cache,oracle,args.seed)
            is_best=validation['score']<best
            if is_best: best=validation['score']
            payload={**{k:bundle[k] for k in ('encoder','stage_a_v2_config','oracle','stage_b1a_config','representation_contract')},
                'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                'comparison_model': 'direct_jepa' if variant.startswith('direct') else variant,
                'training_variant':variant,'training_contract':'autonomous_history_only_v2',
                'run_scope':'diagnostic_comparison','stage_b1b_config':asdict(cfg),
                'train_args':dict(manifest=str(root/'train.jsonl'),b1a_ckpt=str(b1a_path),**vars(args)),
                'train_manifest_sha256':cache['manifests']['train'],
                'oracle_gate':dict(passed=gate['passed'],report=str(report_path),report_sha256=sha256_file(str(report_path)),
                    checkpoint_sha256=cache['b1a_checkpoint_sha256'],representation_id=cache['representation_id']),
                'validation':validation,'best_score':best,'data_schedule_sha256':schedule_hash.hexdigest(),
                'parameter_counts':dict(total=sum(p.numel() for p in model.parameters()),
                    optimized=sum(p.numel() for p in model.parameters() if p.requires_grad)),
                'training_budget':dict(steps=args.steps,batch_size=2,training_draws=2,lr=args.lr,
                    horizon_cycle=[1,2,4,8],latent_weight=1.,distance_weight=.1,distance_scale=10.,
                    point_weight=0.,max_distance_pairs=128),
                'elapsed_seconds':time.time()-start,
                'prediction_contract':dict(schema_version=2,representation_id=cache['representation_id'],
                    predictor_input='normalized_H_history',predictor_output='normalized_H_delta')}
            if variant.startswith('direct'):
                payload.update(direct_predictor=model.state_dict(),direct_predictor_config=direct_config)
            torch.save(payload,out/'checkpoint_latest.pt')
            if is_best: torch.save(payload,out/'checkpoint_best.pt')
            print(json.dumps(dict(step=step,validation_score=validation['score'],best_score=best)),flush=True)
    (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    (out/'summary.json').write_text(json.dumps(dict(variant=variant,seed=args.seed,
        elapsed_seconds=time.time()-start,best_score=best,parameter_counts=payload['parameter_counts'],
        data_schedule_sha256=schedule_hash.hexdigest(),run_scope='diagnostic_comparison'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=['h_markov','h_plus_s','direct_jepa','direct_last','h_plus_s_no_history'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--validate-every',type=int,default=200)
    p.add_argument('--torch-threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4)
    train(p.parse_args())


if __name__=='__main__':
    main()
