"""Locked history interventions and 64-step path diagnostics; never fit on test."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
import yaml

from evaluation import variants as V
from evaluation.geometry import (aligned_rmsd,common_anchor_rmsd,kabsch_align,
    physical_distance_energy_score,ca_geometry_summary,ca_observable_errors)
from evaluation.loaders import build_bundle,sha256_file
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
from protein_jepa_frame.training.train_cached_comparison import (
    DirectJEPAPredictor,make_cache,window,predict,endpoint_loss,
)
from protein_jepa_frame.training.train_stage_b1b import autonomous_samples,decode_paths


def load_net(path,variant):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if variant.startswith('direct'):
        net=DirectJEPAPredictor(**ck['direct_predictor_config'])
        net.load_state_dict(ck['direct_predictor'])
    else:
        net=StageB1b(StageB1bConfig(**ck['stage_b1b_config']))
        net.load_state_dict(ck['model'])
    return net.eval().requires_grad_(False),ck


@torch.no_grad()
def history_diagnostic(root,protocol):
    cache,oracle,bundle=make_cache(root,root/'b1a/checkpoint_best.pt')
    rows=[]
    for seed in protocol['training_seeds']:
        for variant in ('direct_jepa','direct_last','h_plus_s','h_plus_s_no_history','h_markov'):
            path=root/'trained'/f'seed{seed}'/variant/'checkpoint_best.pt'
            net,ck=load_net(path,variant)
            conditions=('full_history','last_H_repeated','reverse_past_keep_anchor') if variant in {'direct_jepa','h_plus_s'} else ('as_trained',)
            for role in ('chrono_dev','cross_dev'):
                for record in cache['splits'][role]:
                    anchors=torch.linspace(7,len(record['H'])-9,2).long()
                    for horizon in (1,2,4,8):
                        base=window(record,anchors,N=horizon)
                        for condition in conditions:
                            w=dict(base)
                            if condition=='last_H_repeated':
                                w['history']=base['history'][:,-1:].expand_as(base['history'])
                            elif condition=='reverse_past_keep_anchor':
                                h=base['history']
                                w['history']=torch.cat([h[:,:-1].flip(1),h[:,-1:]],1)
                            # Same generator per intervention; targets stay fixed.
                            gen=torch.Generator().manual_seed(seed+1234567+horizon)
                            with sdpa_kernel([SDPBackend.MATH]):
                                smp=predict(net,variant,w,generator=gen,training=False,samples=4)
                                _,metrics=endpoint_loss(smp,base,oracle)
                            rows.append(dict(seed=seed,role=role,protein_id=record['protein_id'],
                                variant=variant,condition=condition,horizon_ns=horizon,
                                checkpoint_sha256=sha256_file(str(path)),**metrics))
        for role in ('chrono_dev','cross_dev'):
            for record in cache['splits'][role]:
                anchors=torch.linspace(7,len(record['H'])-9,2).long()
                for horizon in (1,2,4,8):
                    w=window(record,anchors,N=horizon)
                    latent_anchor=w['history'][:,-1][None,:,None].expand(4,-1,-1,-1,-1)
                    _,metrics=endpoint_loss(latent_anchor,w,oracle)
                    rows.append(dict(seed=seed,role=role,protein_id=record['protein_id'],
                        variant='normalized_H_persistence',condition='last_H_without_evolution',
                        horizon_ns=horizon,**metrics))
    out=root/'history_diagnostics.json'
    out.write_text(json.dumps(dict(scope='development-only interventions, not independently retrained controls except explicitly named variants',
        protocol_sha256=sha256_file(str(root/'supplementary_protocol.json')),rows=rows),indent=2)+'\n')
    print(f'completed history diagnostics: {len(rows)} rows',flush=True)


def distance_features(path,mask,max_pairs=128):
    i,j=torch.triu_indices(path.shape[-2],path.shape[-2],offset=3)
    keep=mask[i]&mask[j]
    i,j=i[keep],j[keep]
    if len(i)>max_pairs:
        take=torch.linspace(0,len(i)-1,max_pairs).long()
        i,j=i[take],j[take]
    return (path[...,i,:]-path[...,j,:]).norm(dim=-1)


def acf(features,lag):
    centered=features-features.mean(-2,keepdim=True)
    x,y=centered[...,:-lag,:],centered[...,lag:,:]
    den=(x.square().sum((-1,-2))*y.square().sum((-1,-2))).sqrt()
    vals=(x*y).sum((-1,-2))/den.clamp_min(1e-12)
    # A constant path has no measurable fluctuations; correlation is undefined.
    return [float(v) if float(d)>1e-10 else None for v,d in zip(vals.reshape(-1),den.reshape(-1))]


@torch.no_grad()
def path_metrics(paths,anchor,truth,mask,aa,ri,ci):
    M,N,L,_=paths.shape
    aligned_truth=torch.stack([kabsch_align(t,anchor,mask) for t in truth])
    point=paths.mean(0)
    primary=[common_anchor_rmsd(point[t],anchor,truth[t],mask) for t in range(N)]
    shape=[aligned_rmsd(point[t],truth[t],mask) for t in range(N)]
    samples=[[common_anchor_rmsd(paths[m,t],anchor,truth[t],mask) for t in range(N)] for m in range(M)]
    es=[physical_distance_energy_score(paths[:,t],truth[t],mask) for t in range(N)]
    msd=(paths[:,:,mask]-anchor[mask]).square().sum(-1).mean(-1)
    mean_msd=(point[:,mask]-anchor[mask]).square().sum(-1).mean(-1)
    md_msd=(aligned_truth[:,mask]-anchor[mask]).square().sum(-1).mean(-1)
    def rmsf(x):
        return (x-x.mean(-3,keepdim=True)).square().sum(-1).mean(-2).sqrt()
    real_rmsf=rmsf(aligned_truth[:,mask])
    predicted_rmsf=rmsf(paths[:,:,mask])
    rmsf_error=(predicted_rmsf-real_rmsf).square().mean(-1).sqrt()
    geometries=[ca_geometry_summary(p,mask,chain_id=ci,residue_index=ri) for p in paths.reshape(M*N,L,3)]
    target_geometries=[ca_geometry_summary(p,mask,chain_id=ci,residue_index=ri) for p in truth]
    observations=[ca_observable_errors(p,q,mask,chain_id=ci,residue_index=ri)
                  for p,q in zip(point,truth)]
    avg=lambda rows: {k:float(np.mean([r[k] for r in rows if r[k] is not None]))
                      if any(r[k] is not None for r in rows) else None for k in rows[0]}
    pf=distance_features(paths,mask)
    tf=distance_features(aligned_truth[None],mask)
    return dict(primary_rmsd_curve=primary,shape_rmsd_curve=shape,
        per_sample_primary_rmsd_curves=samples,physical_distance_ES_curve=es,
        sample_mean_MSD_curve=msd.mean(0).tolist(),ensemble_mean_MSD_curve=mean_msd.tolist(),
        observed_MSD_curve=md_msd.tolist(),sample_RMSF_rmse_angstrom=rmsf_error.tolist(),
        observed_RMSF_rms_angstrom=float(real_rmsf.square().mean().sqrt()),
        sample_geometry_mean=avg(geometries),target_geometry_mean=avg(target_geometries),
        ensemble_observable_errors_mean=avg(observations),
        distance_ACF={str(lag):dict(predicted_samples=acf(pf,lag),observed=acf(tf,lag)[0]) for lag in (1,4,8,16)})


@torch.no_grad()
def path_diagnostic(root,protocol):
    # Scoring weights must have been locked before any supplementary test use.
    locked=json.loads((root/'selection_lock.json').read_text())
    for entry in locked['checkpoints'].values():
        if sha256_file(entry['path'])!=entry['sha256']:
            raise ValueError('checkpoint changed after test selection lock')
    ds=TrajectoryClipDataset.__new__(TrajectoryClipDataset)
    ds.config=DataConfig(max_residues=None)
    ds._init_auxiliary_state(None)
    rows=[]
    out=root/'path_diagnostics'
    out.mkdir(exist_ok=True)
    for seed in protocol['training_seeds']:
        base=root/'trained'/f'seed{seed}'
        bundles={v:build_bundle(variant=v,b1a_ckpt=str(root/'b1a/checkpoint_best.pt'),
            b1b_ckpt=str(base/v/'checkpoint_best.pt') if v!='direct_jepa' else None,
            direct_ckpt=str(base/v/'checkpoint_best.pt') if v=='direct_jepa' else None,
            diagnostic_comparison=True,device='cpu') for v in ('h_markov','h_plus_s','direct_jepa')}
        for role in ('chrono_test','cross_test'):
            for record in load_manifest(root/(role+'.jsonl')):
                sample=ds._read_clip(record,np.arange(72))
                ctx={k:v[None,:8] if k in {'coords','residue_mask','atom_mask','time_ps'} else
                    v[None] if torch.is_tensor(v) else v for k,v in sample.items()}
                anchor=sample['coords'][7,:,1]
                mask=sample['residue_mask'][:8].all(0).bool()
                if not torch.equal(sample['residue_mask'].bool(),mask[None].expand_as(sample['residue_mask'])):
                    raise ValueError('changing future topology cannot enter path conditioning')
                aa,ri,ci=(sample[k] for k in ('aatype','residue_index','chain_id'))
                reference=bundles['h_markov']
                raw=V.encode_clip(reference,ctx)
                h=reference.normalize(raw)
                batch=dict(coords=sample['coords'][None,:8],aatype=aa[None],residue_index=ri[None],chain_id=ci[None])
                for variant in ('h_markov','h_plus_s','direct_jepa','persist'):
                    row=dict(seed=seed,role=role,protein_id=record.protein_id,variant=variant,
                        original_anchor_frame=(record.metadata or {})['original_frame_start']+7,
                        steps=64,step_ps=1000.,diagnostic_only=True)
                    try:
                        with sdpa_kernel([SDPBackend.MATH]):
                            if variant=='persist':
                                ca=anchor[None,None].expand(1,64,-1,-1)
                            elif variant=='direct_jepa':
                                bundle=bundles[variant]
                                future=[]
                                for start in range(1,65,8):
                                    lags=torch.arange(start,min(start+8,65))*1000.
                                    future.append(bundle.normalize(bundle.predict_future(raw.expand(len(lags),-1,-1,-1),
                                        lags,torch.full((len(lags),),320.),mask[None].expand(len(lags),-1))))
                                paths=torch.cat(future)[None,None]
                                ca=decode_paths(bundle.decoder,paths,batch,mask[None],8,h[:,-1])[:,0,:,:,1]
                                row['normalized_H_abs_max']=float(paths.abs().max())
                            else:
                                bundle=bundles[variant]
                                gen=torch.Generator().manual_seed(seed+99881)
                                traj,bath=autonomous_samples(bundle.b1b,h,torch.full((1,8),1000.),mask[None],
                                    torch.tensor([320.]),64,4,generator=gen,training=False)
                                if not torch.isfinite(traj).all():
                                    raise FloatingPointError('nonfinite long autonomous H; no filtering')
                                ca=decode_paths(bundle.decoder,traj[:,:,1:],batch,mask[None],8,h[:,-1])[:,0,:,:,1]
                                row['normalized_H_abs_max']=float(traj.abs().max())
                                row['bath_abs_max']=float(bath.abs().max()) if bath is not None else None
                                row['kernel_parameters']=V.dynamics_parameters(bundle,h,ctx,mask[None])
                        if not torch.isfinite(ca).all():
                            raise FloatingPointError('nonfinite decoded path; no filtering')
                        # Forecast is complete before any future coordinate is scored.
                        row.update(status='finite',**path_metrics(ca,anchor,sample['coords'][8:72,:,1],mask,aa,ri,ci))
                        # Retain actual paths so continuity and geometry are reviewable.
                        torch.save(dict(predicted_CA=ca,observed_backbone=sample['coords'][8:72],anchor_backbone=sample['coords'][7],
                            time_ps=sample['time_ps'][8:72],residue_mask=mask,aatype=aa,residue_index=ri,chain_id=ci),
                            out/f'{role}_seed{seed}_{record.protein_id}_{variant}.pt')
                    except (FloatingPointError,RuntimeError) as exc:
                        row.update(status='failed',failure=str(exc))
                    rows.append(row)
                print(f'path diagnostics {role} seed{seed} {record.protein_id}',flush=True)
    payload=dict(scope='fixed-history 64ns path diagnostic; >8ns is extrapolation',
        protocol_sha256=sha256_file(str(root/'supplementary_protocol.json')),
        selection_lock_sha256=sha256_file(str(root/'selection_lock.json')),
        rows=rows)
    (out/'report.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--mode',choices=['history','paths'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(args.root).resolve()
    protocol=yaml.safe_load((root/'experiment.yaml').read_text())
    (history_diagnostic if args.mode=='history' else path_diagnostic)(root,protocol)


if __name__=='__main__':
    main()
