"""Score autonomous CA forecasts on the shared BioEmu2 ensemble protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.training.train_lasp_calibration import calibration_terms
from scripts.eval_lasp_final import METRICS


def anchor_target(context,n_samples):
    """Observed-anchor geometry reference, never a future input to the sampler."""
    return context['coords'][-1,:,1][None,None].cuda().expand(1,n_samples,-1,-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out',required=True)
    p.add_argument('--max-windows',type=int,default=200)
    p.add_argument('--max-residues',type=int,default=96)
    p.add_argument('--n-samples',type=int,default=12)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--prediction-kind',choices=['endpoint','time_mixture'],default='endpoint')
    p.add_argument('--temperatures',type=float,nargs='+',default=[320.,348.,379.,413.,450.])
    p.add_argument('--round-robin-proteins',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--round-robin-conditions',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--anchors-per-trajectory',type=int,default=4)
    p.add_argument('--horizons',type=int,nargs='+',default=[1,4,16,64])
    # Override the checkpoint's S_0 source for a no-update probe.  Default None
    # keeps the trained source, so a normal evaluation never silently changes it.
    p.add_argument('--s0-mode',choices=['gru','exact','exact_spatial','prior'],default=None)
    # Override the bath scale to separate "the scale hurt" from "the retraining
    # hurt".  Default None keeps the checkpoint's own scale.
    p.add_argument('--coupling-scale',type=float,default=None)
    p.add_argument('--gamma-scale',type=float,default=None)
    p.add_argument('--gamma-floor',type=float,default=None)
    a=p.parse_args();torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('checkpoint_kind') not in ('lasp_ca_dynamics','lasp_bounded_endpoint_v1'):raise ValueError('incorrect checkpoint kind')
    # Materialize the embedded component, avoiding mutable training input files.
    component=Path(a.out).with_suffix('.component.pt')
    component.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck['component_checkpoint'],component)
    if ck.get('checkpoint_kind')=='lasp_bounded_endpoint_v1':
        from protein_jepa_frame.training.train_lasp_bounded_endpoint import load_bounded_checkpoint
        model,_=load_bounded_checkpoint(a.checkpoint,component_path=component,device='cuda')
        model.eval()
    else:
        model=CADynamics(str(component),'cuda',a.checkpoint,
            history_conditioning=ck['args'].get('history_conditioning',False),
            noise_mix=ck['args'].get('noise_mix',0.),noise_length=ck['args'].get('noise_length',8.),
            collective_mobility=ck['args'].get('collective_mobility',False),
            noise_kernel=ck['args'].get('noise_kernel','gaussian'),
            noise_cutoff=ck['args'].get('noise_cutoff',10.),
            s0_mode=a.s0_mode).eval()
    if a.s0_mode is not None:
        model.kernel.s0_mode=a.s0_mode
    _scale=vars(a)
    for attr in ('coupling_scale','gamma_scale','gamma_floor'):
        val=_scale[attr.replace('-','_')]
        if val is None:continue
        for obj in (getattr(model.kernel,'cfg',None),getattr(model.kernel,'bath',None)):
            if obj is not None and hasattr(obj,attr):setattr(obj,attr,float(val))
        print(f'scale override {attr}={val}',flush=True)
    if ck.get('checkpoint_kind')=='lasp_bounded_endpoint_v1':
        model._sample_time_mixture=a.prediction_kind=='time_mixture'
    else:
        model.prediction_kind=a.prediction_kind
    protocol=EvalProtocol(manifest=a.manifest,horizons_frames=tuple(a.horizons),
        context_frames=8,frame_stride=1,max_residues=a.max_residues,
        max_anchors_per_trajectory=a.anchors_per_trajectory,temperatures=tuple(a.temperatures),
        seed=a.seed,round_robin_proteins=a.round_robin_proteins,
        round_robin_conditions=a.round_robin_conditions)
    rows=[]
    for w in iter_windows(protocol):
        if len(rows)>=a.max_windows:break
        mask=w.context['residue_mask'].all(0).bool().cuda()
        if int(mask.sum())<4:continue
        key=f'{w.protein_id}|{w.trajectory_id}|{w.anchor_frame}|{w.horizon}'
        seed=(a.seed*1000003+int(hashlib.sha256(key.encode()).hexdigest()[:12],16))%(2**31-1)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        with torch.no_grad():
            ca,_=model.sample(w.context['coords'][None].cuda(),mask[None],
                *[w.context[k][None].cuda() for k in ('aatype','residue_index','chain_id')],
                torch.tensor([w.temperature_k],device='cuda'),w.horizon,a.n_samples,generator)
            geometry=calibration_terms(ca,anchor_target(w.context,ca.shape[1]),mask[None],
                w.context['residue_index'][None].cuda(),w.context['chain_id'][None].cuda())
            geometry={k:float(geometry[k]) for k in ('ca_bond','ca_clash_fraction','ca_angle_cos_mae')}
            pred=ca[0,:,mask]
            truth=w.future_path['coords'][:,:,1].cuda()[:,mask]
            anchor=w.context['coords'][-1,:,1].cuda()[mask]
            metrics=extended_ensemble_summary(pred.float(),truth.float(),None,cutoff=8.,pca_k=3)
            persist=extended_ensemble_summary(anchor[None].expand(len(truth),-1,-1),truth.float(),None,cutoff=8.,pca_k=3)
        rows.append(dict(protein=w.protein_id,traj=w.trajectory_id,temperature_k=float(w.temperature_k),
            anchor_frame=int(w.anchor_frame),horizon=float(w.horizon),
            prediction_kind=a.prediction_kind,actual_n_samples=int(ca.shape[1]),
            observation_steps=getattr(model,'last_observation_steps',
                getattr(model,'last_sample_diagnostics',{}).get('observation_steps')),
            geometry_vs_anchor=geometry,
            contact_counts={k:float(metrics[k]) for k in ('weak_n_true','weak_n_pred','transient_n_true','transient_n_pred') if k in metrics},
            lasp={m:float(metrics[m]) for m in METRICS},persist={m:float(persist[m]) for m in METRICS}))
        if len(rows)%10==0:print(f'scored {len(rows)}',flush=True)
    result=dict(metric_version='empirical_w2_v2_pca_projection',lasp_sampling=a.prediction_kind,
        reference_kind='future_path_mixture',optimizer_steps=0,test_opened=False,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__),Path('protein_jepa_frame/training/train_lasp_ca_dynamics.py'),
            Path('protein_jepa_frame/models/stage_b1b/model.py'),
            Path('protein_jepa_frame/models/correlated_bath_posterior.py'))},
        checkpoint=a.checkpoint,args=vars(a),n_samples=a.n_samples,n_windows=len(rows),
        n_proteins=len({r['protein'] for r in rows}),per_window=rows)
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(n_windows=len(rows),n_proteins=result['n_proteins'])),flush=True)


if __name__=='__main__':main()
