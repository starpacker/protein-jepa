"""Component calibration with a frozen GLE and optional frozen JEPA encoder teacher.

This fits true train-split conformations, not stochastic forecasts. A calibrated
checkpoint requires a new dynamics compatibility evaluation before deployment.
"""
from __future__ import annotations
from protein_jepa_frame.training.lasp_run_state import step_batch,rng_state,restore_rng,budget_subset
from protein_jepa_frame.training.train_utils import _lr_scale
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import validate_manifest_split
from protein_jepa_frame.training.train_utils import _move_batch, _set_reproducibility
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack, _extract_backbone_frames
from protein_jepa_frame.training.ensemble_losses import _position_loss
from evaluation.ensemble_metrics_ext import _align_to
from evaluation.ensemble_metrics import align_ensemble, rmsf_profile
from protein_jepa_frame.models.latent_delta_decoder import LatentDeltaCADecoder
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder, BoundedEdgeCADecoder
from protein_jepa_frame.models.ca_geometry import project_ca_bonds


def masked_mean(value, mask):
    mask = mask.to(value.dtype).expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def calibration_terms(pred, target, mask, residue_index, chain_id):
    """[B,T,L,3] coordinates in a common anchor gauge; [B,L] mask.

    Nonlocal pair distances and their changes prevent identity reconstruction
    from being rewarded as dynamic reconstruction. Adjacent-CA error is a metric.
    """
    coord = masked_mean((pred-target).square().sum(-1), mask[:,None])
    length = pred.shape[2]
    i,j = torch.triu_indices(length,length,offset=1,device=pred.device)
    same_chain = chain_id[:,i] == chain_id[:,j]
    sep = (residue_index[:,i]-residue_index[:,j]).abs()
    valid = mask[:,i] & mask[:,j]
    nonlocal_pair = valid & (~same_chain | (sep >= 3))
    pd = (pred[:,:,i]-pred[:,:,j]).square().sum(-1).clamp_min(1e-12).sqrt()
    td = (target[:,:,i]-target[:,:,j]).square().sum(-1).clamp_min(1e-12).sqrt()
    distance = masked_mean((pd-td).abs(),nonlocal_pair[:,None])
    delta = masked_mean(((pd[:,1:]-pd[:,:1])-(td[:,1:]-td[:,:1])).abs(),nonlocal_pair[:,None])
    contact = masked_mean((torch.sigmoid((8-pd)/0.5)-torch.sigmoid((8-td)/0.5)).abs(),nonlocal_pair[:,None])
    adjacent = valid & same_chain & (sep==1)
    ca_bond = masked_mean((pd-td).abs(),adjacent[:,None])
    change = masked_mean(((pred[:,1:]-pred[:,:1])-(target[:,1:]-target[:,:1])).square().sum(-1), mask[:,None])
    pred_edges = pred[:,:,1:]-pred[:,:,:-1]
    true_edges = target[:,:,1:]-target[:,:,:-1]
    consecutive = (mask[:,1:] & mask[:,:-1] & (chain_id[:,1:]==chain_id[:,:-1])
                   & ((residue_index[:,1:]-residue_index[:,:-1])==1))
    angle_valid = consecutive[:,1:] & consecutive[:,:-1]
    def cosines(edges):
        unit = torch.nn.functional.normalize(edges,dim=-1,eps=1e-8)
        return (unit[:,:,1:]*unit[:,:,:-1]).sum(-1)
    angle = masked_mean((cosines(pred_edges)-cosines(true_edges)).abs(),angle_valid[:,None])
    clash = masked_mean((pd<3.).float(),nonlocal_pair[:,None])
    true_clash = masked_mean((td<3.).float(),nonlocal_pair[:,None])
    clash_penalty = masked_mean(torch.relu(3.5-pd).square(),nonlocal_pair[:,None])
    return dict(coord=coord,distance=distance,delta_distance=delta,contact=contact,ca_bond=ca_bond,
                change=change,ca_angle_cos_mae=angle,ca_clash_fraction=clash,true_ca_clash_fraction=true_clash,clash_penalty=clash_penalty)


def teacher_terms(student, teacher, mask):
    """Keep absolute coordinates AND conformational differences in the teacher basis."""
    absolute = masked_mean((student-teacher).square().mean(-1),mask[:,None])
    ds,dt = student[:,1:]-student[:,:1],teacher[:,1:]-teacher[:,:1]
    scale = masked_mean(dt.square().mean(-1),mask[:,None]).detach().clamp_min(1e-4)
    delta = masked_mean((ds-dt).square().mean(-1),mask[:,None]) / scale
    return absolute,delta


class Calibration(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.encoder,self.decoder,_ = load_frozen_stack(args.stage_a_ckpt,args.b1a_ckpt,device)
        self.source = torch.load(args.init_from,map_location='cpu',weights_only=False)
        for name in ('encoder','decoder'):
            if self.source.get(name) is not None:
                getattr(self,name).load_state_dict(self.source[name],strict=True)
        self.decoder_mode = getattr(args, 'decoder_mode', 'legacy')
        if self.decoder_mode in ('delta_ca','geometric_ca','bounded_edge_ca'):
            decoder_class = {'geometric_ca': GeometricDeltaCADecoder,
                             'bounded_edge_ca': BoundedEdgeCADecoder,
                             'delta_ca': LatentDeltaCADecoder}[self.decoder_mode]
            self.decoder = decoder_class().to(device)
            compatible_mode = (self.source.get('decoder_mode') == self.decoder_mode
                               or (self.decoder_mode == 'bounded_edge_ca'
                                   and self.source.get('decoder_mode') == 'geometric_ca'))
            if self.source.get('delta_ca_decoder') is not None and compatible_mode:
                self.decoder.load_state_dict(self.source['delta_ca_decoder'], strict=True)
        self.teacher = copy.deepcopy(self.encoder).eval() if args.joint else None
        for p in self.encoder.parameters():
            p.requires_grad_(args.joint)
        for p in self.decoder.parameters():
            p.requires_grad_(True)
        if self.teacher is not None:
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self.joint = args.joint
        self.bond_projection_steps = getattr(args, "bond_projection_steps", 0)

    def forward(self,batch,controls=False):
        coords = batch['coords']
        b,t,l = coords.shape[:3]
        mask = batch['residue_mask'].bool().all(1)
        aa,ri,ci = [batch[k] for k in ('aatype','residue_index','chain_id')]
        rm = mask[:,None].expand(b,t,l).reshape(b*t,l)
        expand = lambda x:x[:,None].expand(b,t,l).reshape(b*t,l)
        def encode(encoder):
            return encoder(coords.reshape(b*t,1,l,4,3),rm[:,None],expand(aa),expand(ri),expand(ci),token_mask=rm[:,None])[:,0].reshape(b,t,l,-1)
        if self.joint:
            latent = encode(self.encoder)
        else:
            with torch.no_grad():
                latent = encode(self.encoder)
        R,anchor = _extract_backbone_frames(coords[:,0])
        def decode(h):
            if self.decoder_mode in ('delta_ca','geometric_ca','bounded_edge_ca'):
                decoded = self.decoder(h.reshape(b*t,l,-1),
                    latent[:,:1].expand_as(latent).reshape(b*t,l,-1),
                    expand(aa),expand(ri),expand(ci),
                    R[:,None].expand(b,t,l,3,3).reshape(b*t,l,3,3),
                    anchor[:,None].expand(b,t,l,3).reshape(b*t,l,3),rm)
                if self.bond_projection_steps:
                    decoded = project_ca_bonds(decoded,anchor[:,None].expand(b,t,l,3).reshape(b*t,l,3),
                        rm,expand(ri),expand(ci),self.bond_projection_steps)
                return decoded.reshape(b,t,l,3)
            out = self.decoder.decode_structure(h.reshape(b*t,l,-1),expand(aa),
                    R[:,None].expand(b,t,l,3,3).reshape(b*t,l,3,3),
                    anchor[:,None].expand(b,t,l,3).reshape(b*t,l,3),rm)
            return out['backbone_coords'][:,:,1].reshape(b,t,l,3)
        pred = decode(latent)
        with torch.no_grad():
            target = torch.stack([_align_to(coords[k,:,:,1],coords[k,0,:,1],mask[k]) for k in range(b)])
        terms = calibration_terms(pred,target,mask,ri,ci)
        if self.teacher is not None:
            with torch.no_grad():
                teacher_h = encode(self.teacher)
            terms['teacher'],terms['teacher_delta'] = teacher_terms(latent,teacher_h,mask)
        else:
            terms['teacher'] = terms['teacher_delta'] = pred.new_zeros(())
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise RuntimeError('nonfinite calibration terms')
        if controls:
            fixed = decode(latent[:,:1].expand_as(latent))
            shuffled = decode(latent.flip(1))
            terms['shuffled_ca_rmsd'] = _position_loss(shuffled.reshape(b*t,l,3),target.reshape(b*t,l,3),rm).sqrt()
            terms['ca_rmsd'] = _position_loss(pred.reshape(b*t,l,3),target.reshape(b*t,l,3),rm).sqrt()
            terms['fixed_ca_rmsd'] = _position_loss(fixed.reshape(b*t,l,3),target.reshape(b*t,l,3),rm).sqrt()
            profiles = [(rmsf_profile(align_ensemble(pred[k],mask[k]),mask[k]),
                         rmsf_profile(align_ensemble(target[k],mask[k]),mask[k])) for k in range(b)]
            terms['rmsf_ratio'] = torch.stack([p[m].mean()/q[m].mean().clamp_min(1e-6) for (p,q),m in zip(profiles,mask)]).mean()
            correlations = []
            magnitude_errors = []
            for (p, q), m in zip(profiles, mask):
                pv, qv = p[m], q[m]
                magnitude_errors.append((pv-qv).abs().mean())
                pc, qc = pv-pv.mean(), qv-qv.mean()
                correlations.append((pc*qc).sum() / (pc.norm()*qc.norm()).clamp_min(1e-8))
            terms['rmsf_mae'] = torch.stack(magnitude_errors).mean()
            terms['rmsf_corr'] = torch.stack(correlations).mean()
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise RuntimeError('nonfinite calibration validation terms')
        return terms


def selection_score(metrics, bond_limit):
    """Reject geometrically invalid checkpoints before selecting reconstruction quality."""
    mean = metrics['mean']
    if mean['ca_bond'] > bond_limit:
        return float('inf')
    return mean['ca_rmsd']


def objective(terms,args):
    return (terms['coord'] + args.distance_weight*terms['distance']
            + args.delta_weight*terms['delta_distance']
            + args.teacher_weight*terms['teacher'] + args.teacher_delta_weight*terms['teacher_delta']
            + getattr(args,'contact_weight',0.)*terms['contact']
            + getattr(args,'bond_weight',0.)*terms['ca_bond']
            + getattr(args,'change_weight',0.)*terms['change']
            + getattr(args,'angle_weight',0.)*terms['ca_angle_cos_mae']
            + getattr(args,'clash_weight',0.)*terms['clash_penalty'])


def loaders(args,manifest,training):
    result = {}
    for stride in (1,4,16):
        cfg = DataConfig(manifest_path=manifest,training_mode='clip',num_frames=5,
                         frame_stride=stride,max_residues=args.max_residues)
        ds = build_dataset(cfg)
        budget = args.max_train_samples if training else args.val_batches*args.batch_size
        n = len(ds) if budget==0 else min(len(ds),budget)
        if n < args.batch_size:
            raise ValueError('empty calibration loader')
        if getattr(args,'stateless_data',False):ds=budget_subset(ds,budget,training,args.seed+stride)
        else:
            idx = torch.linspace(0,len(ds)-1,n).long().tolist()
            ds = Subset(ds,idx)
        generator = torch.Generator().manual_seed(args.seed+stride)
        result[stride] = DataLoader(ds,batch_size=args.batch_size,shuffle=training and not getattr(args,'stateless_data',False),
                num_workers=0,collate_fn=collate_clips,generator=generator,drop_last=training)
    return result


@torch.no_grad()
def validate(model,ls,args,device):
    model.eval()
    by_stride = {}
    by_window = []
    for stride,loader in ls.items():
        rows = []
        for batch in loader:
            terms=model(_move_batch(batch,device),controls=True)
            row = {k:float(v) for k,v in terms.items()}
            rows.append(row)
            by_window.append(dict(proteins=batch["protein_id"],trajectories=batch["trajectory_id"],
                time_ps=batch["time_ps"].tolist(),stride=stride,metrics=row))
        by_stride[str(stride)] = {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    mean = {k:sum(r[k] for r in by_stride.values())/len(by_stride) for k in next(iter(by_stride.values()))}
    return dict(mean=mean,by_stride=by_stride,by_window=by_window)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stateless-data',action='store_true')
    p.add_argument('--cosine',action='store_true')
    p.add_argument('--init-from',default='artifacts/lasp_v6_pilot_input/v3_best.pt')
    p.add_argument('--stage-a-ckpt',default='artifacts/stage_a_disp_relpos/checkpoint_latest.pt')
    p.add_argument('--b1a-ckpt',default='artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt')
    p.add_argument('--manifest',default='artifacts/mdcath_manifest_full/train.jsonl')
    p.add_argument('--val-manifest',default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--decoder-mode',choices=['legacy','delta_ca','geometric_ca','bounded_edge_ca'],default='legacy')
    p.add_argument('--contact-weight',type=float,default=0.)
    p.add_argument('--bond-weight',type=float,default=0.)
    p.add_argument('--change-weight',type=float,default=0.)
    p.add_argument('--selection-bond-limit',type=float,default=float('inf'))
    p.add_argument('--bond-projection-steps',type=int,default=0)
    p.add_argument('--angle-weight',type=float,default=0.)
    p.add_argument('--clash-weight',type=float,default=0.)
    p.add_argument('--output',required=True);p.add_argument('--joint',action='store_true')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--max-residues',type=int,default=96)
    p.add_argument('--max-train-samples',type=int,default=1024);p.add_argument('--val-batches',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--encoder-lr-scale',type=float,default=0.1);p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--distance-weight',type=float,default=0.2);p.add_argument('--delta-weight',type=float,default=0.5)
    p.add_argument('--teacher-weight',type=float,default=1.);p.add_argument('--teacher-delta-weight',type=float,default=0.1)
    p.add_argument('--val-every',type=int,default=50);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.decoder_mode == 'bounded_edge_ca' and a.bond_projection_steps:
        raise ValueError('bounded edge decoder must train and evaluate without post-projection')
    if a.steps<1 or a.val_batches<1 or a.batch_size<1:
        raise ValueError('steps, val batches and batch size must be positive')
    audit = validate_manifest_split(a.manifest,a.val_manifest)
    torch.set_num_threads(4);_set_reproducibility(a.seed,False)
    dev=torch.device(a.device);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.resume and not a.stateless_data:raise ValueError('resume requires stateless data')
    if (out/'metrics.jsonl').exists() and not a.resume:
        raise FileExistsError('use a fresh output directory for a new calibration run')
    model=Calibration(a,dev).to(dev).eval()
    train=loaders(a,a.manifest,True);val=loaders(a,a.val_manifest,False)
    groups=[dict(params=list(model.decoder.parameters()),lr=a.lr,base_lr=a.lr)]
    if a.joint:
        groups.append(dict(params=list(model.encoder.parameters()),lr=a.lr*a.encoder_lr_scale,base_lr=a.lr*a.encoder_lr_scale))
    opt=torch.optim.AdamW(groups,weight_decay=0.01)
    params=[p for p in model.parameters() if p.requires_grad]
    if not a.resume:(out/'run.json').write_text(json.dumps(dict(args=vars(a),split_audit=audit),indent=2))
    def record(row):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def save(tag,step,metrics):
        # Preserve original kernel semantics and producer arguments for evaluation.
        ck={k:v for k,v in model.source.items() if k!='optimizer'}
        ck.update(encoder=model.encoder.state_dict(),decoder=model.decoder.state_dict(),
                  train_encoder=True,train_decoder=True,step=step,best_val=metrics['mean']['ca_rmsd'],
                  calibration_args=vars(a),calibration_metrics=metrics,
                  calibration_optimizer=opt.state_dict(),gle_requires_revalidation=True,
                  checkpoint_kind=('ca_component_probe' if a.decoder_mode in ('delta_ca','geometric_ca','bounded_edge_ca') else 'lasp_calibration'),
                  decoder_mode=a.decoder_mode)
        if a.decoder_mode in ('delta_ca','geometric_ca','bounded_edge_ca'):
            # Fail closed: existing GLE producers must not accept a CA-only probe.
            ck['delta_ca_decoder']=ck.pop('decoder')
            ck.pop('kernel',None)
        ck.update(rng_state=rng_state(),best_score=best)
        if model.teacher is not None:ck['calibration_teacher']=model.teacher.state_dict()
        torch.save(ck,out/f'checkpoint_{tag}.tmp')
        (out/f'checkpoint_{tag}.tmp').replace(out/f'checkpoint_{tag}.pt')
    start_step=0
    if a.resume:
        ck=torch.load(out/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        for key,value in ck['calibration_args'].items():
            if key!='resume' and getattr(a,key)!=value:raise ValueError(f'resume argument differs: {key}')
        model.encoder.load_state_dict(ck['encoder'],strict=True)
        model.decoder.load_state_dict(ck.get('delta_ca_decoder',ck.get('decoder')),strict=True)
        if model.teacher is not None:model.teacher.load_state_dict(ck['calibration_teacher'],strict=True)
        opt.load_state_dict(ck['calibration_optimizer']);best=ck['best_score'];start_step=ck['step']
        restore_rng(ck['rng_state']);record(dict(resumed_from=start_step))
    else:
        initial=validate(model,val,a,dev);record(dict(step=0,val=initial))
        best=selection_score(initial,a.selection_bond_limit)
        save('initial',0,initial);save('latest',0,initial)
        if best < float('inf'):save('best',0,initial)
    iters={h:iter(dl) for h,dl in train.items()};cycle=list(train);start=time.time()
    for step in range(start_step+1,a.steps+1):
        h=cycle[(step-1)%len(cycle)]
        if a.stateless_data:batch=step_batch(train[h],step,a.seed)
        else:
            try:batch=next(iters[h])
            except StopIteration:
                iters[h]=iter(train[h]);batch=next(iters[h])
        opt.zero_grad(set_to_none=True)
        terms=model(_move_batch(batch,dev));loss=objective(terms,a)
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss step {step}')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(gn):raise RuntimeError(f'nonfinite gradient step {step}')
        scale=_lr_scale(step-1,a.steps,a.warmup) if a.cosine else min(step/max(a.warmup,1),1.)
        for group in opt.param_groups:group['lr']=group['base_lr']*scale
        opt.step()
        if step%10==0 or step==1:
            record(dict(step=step,stride=h,seconds=time.time()-start,loss=float(loss.detach()),
                        grad_norm=float(gn),train={k:float(v.detach()) for k,v in terms.items()}))
        if step%a.val_every==0 or step==a.steps:
            metrics=validate(model,val,a,dev);record(dict(step=step,val=metrics))
            score=selection_score(metrics,a.selection_bond_limit)
            if score<best:
                best=score;save('best',step,metrics)
            save('latest',step,metrics)
    print(f'done: {a.steps} updates; best reconstruction RMSD={best:.6f}',flush=True)


if __name__=='__main__':main()
