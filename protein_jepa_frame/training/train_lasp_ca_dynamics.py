"""Train autonomous GLE against a fixed geometric CA representation.

True futures supply training targets only. Rollout starts from the observed
anchor and optionally its observed history. Checkpoints embed the component
weights and kernel configuration; no external mutable component is needed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from protein_jepa_frame.training.lasp_run_state import step_batch,rng_state,restore_rng,budget_subset
from protein_jepa_frame.training.train_utils import _lr_scale
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader,Subset
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.protocol import validate_manifest_split
from protein_jepa_frame.training.train_utils import _move_batch,_set_reproducibility
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
from evaluation.ensemble_metrics_ext import _align_to


def energy_score(samples,target):
    """Unbiased sample energy score, Euclidean distance normalized per dimension."""
    # [M,D] samples, [T,D] empirical observations; all target frames equal weight.
    scale=samples.shape[-1]**.5
    cross=torch.cdist(samples[None],target[None]).mean()/scale
    if len(samples)<2:raise ValueError('energy score requires >=2 samples')
    pair=torch.pdist(samples).mean()/scale
    return cross-.5*pair


def displacement_scales(anchor_h, target_h, anchor_x, target_x, mask):
    """Return fixed per-dimension displacement scales for paired score terms."""
    valid = mask.bool()
    h_delta = (target_h - anchor_h)[valid].flatten()
    x_delta = (target_x - anchor_x)[valid].flatten()
    if h_delta.numel() == 0 or x_delta.numel() == 0:
        raise ValueError('displacement scales require valid residues')
    h_scale = h_delta.norm() / h_delta.numel()**.5
    x_scale = x_delta.norm() / x_delta.numel()**.5
    if not all(bool(torch.isfinite(v)) for v in (h_scale, x_scale)):
        raise FloatingPointError('nonfinite training displacement scale')
    return h_scale.detach(), x_scale.detach()


def anchor_noise_factor(anchor,mask,mix,length):
    """PSD Gaussian anchor kernel, unit marginal variance; no future information."""
    if not 0<=mix<1 or length<=0:raise ValueError('invalid spatial noise parameters')
    distance=torch.cdist(anchor,anchor)
    correlation=torch.exp(-distance.square()/(2*length**2))
    valid=mask[:,:,None]&mask[:,None,:]
    correlation=correlation*valid
    eye=torch.eye(anchor.shape[1],device=anchor.device,dtype=anchor.dtype)[None]
    covariance=mix*correlation+(1-mix)*eye
    covariance=covariance+torch.diag_embed((~mask).to(anchor.dtype)*mix)
    return torch.linalg.cholesky(covariance)


def elastic_network_kernel(anchor,mask,cutoff=10.,mix=.8,temperature=1.,jitter=1e-5):
    """Anchor-derived ENM covariance, blended with identity; no future information.

    The current Gaussian kernel is normalized to UNIT marginal variance at every
    residue, so it cannot express *which* residues are soft: its top collective
    subspace overlaps the true future slow-mode subspace at the random-baseline
    rate. An elastic network built from the observed anchor contact graph
    recovers the true future flexibility profile (correlation ~.5) and the true
    slow subspace (~8x the random baseline) using the anchor alone.

    Covariance is the pseudo-inverse of the Kirchhoff (contact-graph) matrix,
    which is the Gaussian-network-model mean-square-fluctuation operator. A
    constant is added along the diagonal so every marginal variance is strictly
    positive and the blend with the identity is well conditioned. The result is
    PSD by construction (eigenvalues of a pseudo-inverse of a PSD matrix are
    non-negative) and is returned as a Cholesky factor with the same contract as
    ``anchor_noise_factor``: unit-scale standard normals enter, correlated noise
    leaves.

    ``temperature`` scales the overall fluctuation amplitude as 1/T around the
    fluctuation scale, which is the equilibrium relation for a fixed elastic
    network; it is a diagnostic knob, not a calibrated thermodynamic model.
    """
    if not 0<=mix<1:raise ValueError('elastic mix must lie in [0,1)')
    if not cutoff>0:raise ValueError('elastic cutoff must be positive')
    if not temperature>0:raise ValueError('elastic temperature must be positive')
    length=anchor.shape[1]
    device,dtype=anchor.device,anchor.dtype
    distance=torch.cdist(anchor,anchor)
    valid=mask[:,:,None]&mask[:,None,:]
    contact=(distance<cutoff)&valid
    contact=contact&~torch.eye(length,device=device,dtype=torch.bool)[None]
    # Kirchhoff matrix: -1 on contacts, degree on the diagonal.
    gamma=torch.zeros_like(distance)
    gamma=torch.where(contact,-torch.ones_like(gamma),gamma)
    gamma=gamma+torch.diag_embed((contact.to(dtype).sum(-1)).clamp_min(1.))
    # Isolated residues (no contact inside the cutoff) keep a self term only.
    isolated=(contact.to(dtype).sum(-1)==0)
    gamma=gamma+torch.diag_embed(isolated.to(dtype))
    eigenvalues,eigenvectors=torch.linalg.eigh(gamma.double())
    keep=eigenvalues>1e-8
    if not bool(keep.all()):
        # Any batch with too few nonzero modes cannot define a pseudo-inverse.
        counts=keep.sum(-1)
        if bool((counts<2).any()):
            raise ValueError('elastic network has no nonzero Kirchhoff modes')
    reciprocal=torch.where(keep,1.0/eigenvalues.clamp_min(1e-30),
                           torch.zeros_like(eigenvalues))
    inverse=(eigenvectors*reciprocal[:,None,:])@eigenvectors.transpose(-1,-2)
    inverse=inverse.to(dtype)
    # Zero-mean modes leave an arbitrary additive constant; pin it with the
    # trace so the diagonal is strictly positive and scale-free across proteins.
    scale=inverse.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
    inverse=inverse/scale[:,None,None]*(1./temperature)
    inverse=inverse+jitter*torch.eye(length,device=device,dtype=dtype)[None]
    eye=torch.eye(length,device=device,dtype=dtype)[None]
    covariance=mix*inverse+(1-mix)*eye
    covariance=covariance+torch.diag_embed((~mask).to(dtype)*mix)
    return torch.linalg.cholesky(covariance)


def slow_mode_scale_loss(samples,target,k=3):
    """Match spread along target slow modes, fitted only to training observations."""
    with torch.no_grad():
        centered=target.flatten(1)-target.flatten(1).mean(0)
        if len(target)<2:return samples.sum()*0
        _,_,vectors=torch.linalg.svd(centered,full_matrices=False)
        vectors=vectors[:min(k,len(target)-1)]
        desired=(centered@vectors.T).square().mean(0).clamp_min(1e-8).sqrt()
    projected=(samples.flatten(1)-samples.flatten(1).mean(0))@vectors.T
    predicted=projected.square().mean(0).clamp_min(1e-8).sqrt()
    return (predicted-desired).abs().mean()/(samples.shape[1]**.5)


def slow_mode_distribution_loss(samples,target,k=3):
    """Centered projected squared W2; target modes exist only during training."""
    with torch.no_grad():
        centered=target.flatten(1)-target.flatten(1).mean(0)
        if len(target)<2:return samples.sum()*0
        _,_,vectors=torch.linalg.svd(centered,full_matrices=False)
        vectors=vectors[:min(k,len(target)-1)]
        desired=centered@vectors.T
    projected=(samples.flatten(1)-samples.flatten(1).mean(0))@vectors.T
    # Exact empirical quantile integral supports unequal ensemble sizes. Squared
    # W2 avoids the square-root singularity at a perfect match.
    n,t=len(samples),len(target)
    boundaries=torch.cat((torch.arange(n+1,device=samples.device,dtype=samples.dtype)/n,
        torch.arange(t+1,device=samples.device,dtype=samples.dtype)/t)).unique(sorted=True)
    widths=boundaries.diff();mid=(boundaries[:-1]+boundaries[1:])*.5
    pi=(mid*n).long().clamp_max(n-1);ti=(mid*t).long().clamp_max(t-1)
    delta=projected.sort(dim=0).values[pi]-desired.sort(dim=0).values[ti]
    return (delta.square()*widths[:,None]).sum(0).mean()/samples.shape[1]


def contact_probability_loss(predicted,target,objective='mae'):
    error=predicted-target
    if objective=='brier':return error.square().mean()
    if objective=='mae':return error.abs().mean()
    raise ValueError(f'unknown contact objective: {objective}')


class CADynamics(torch.nn.Module):
    def __init__(self,component,device,init_kernel=None,bath_dim=8,history_conditioning=False,noise_mix=0.,noise_length=8.,collective_mobility=False,noise_kernel=None,noise_cutoff=None,s0_mode=None):
        super().__init__()
        self.bundle=CAComponentBundle(component,device)
        self.history_conditioning=history_conditioning
        self.noise_mix=noise_mix;self.noise_length=noise_length
        self.collective_mobility=collective_mobility
        if init_kernel:
            ck=torch.load(init_kernel,map_location='cpu',weights_only=False)
            config=StageB1bConfig(**ck['kernel_config'])
            # A checkpoint must be evaluated with the kernel it was trained under.
            # Leaving the flag unset inherits the recorded kernel instead of
            # silently reverting a trained arm to the default.
            recorded=ck.get('args',{})
            if noise_kernel is None:noise_kernel=recorded.get('noise_kernel','gaussian')
            if noise_cutoff is None:noise_cutoff=recorded.get('noise_cutoff',10.)
        else:
            config=StageB1bConfig(bath_dim=bath_dim,tail_init=.01,max_step_norm=10.,max_gamma_step=1.)
        if noise_kernel is None:noise_kernel='gaussian'
        if noise_cutoff is None:noise_cutoff=10.
        if noise_kernel not in ('gaussian','elastic'):
            raise ValueError(f'unknown noise kernel {noise_kernel!r}')
        if not noise_cutoff>0:raise ValueError('noise cutoff must be positive')
        self.noise_kernel=noise_kernel;self.noise_cutoff=noise_cutoff
        self.kernel=StageB1b(config).to(device)
        if init_kernel:self.kernel.load_state_dict(ck['kernel'],strict=True)
        # Restore the trained S_0 source so evaluation cannot silently use a
        # different prior than the checkpoint saw.
        if s0_mode is None:s0_mode=ck.get('args',{}).get('s0_mode','gru') if init_kernel else 'gru'
        if s0_mode not in ('gru','exact','prior'):
            raise ValueError(f'unknown s0_mode {s0_mode!r}')
        self.kernel.s0_mode=s0_mode

    def anchor_kernel(self,anchor_ca,mask,temperature_k=None):
        """Cholesky factor of the anchor-conditioned spatial noise kernel."""
        if self.noise_kernel=='elastic':
            return elastic_network_kernel(anchor_ca,mask,self.noise_cutoff,self.noise_mix)
        if self.noise_kernel=='gaussian':
            return anchor_noise_factor(anchor_ca,mask,self.noise_mix,self.noise_length)
        raise ValueError(f'unknown noise kernel {self.noise_kernel!r}')

    def encode_clip(self,batch):
        coords=batch['coords'];b,t,l=coords.shape[:3]
        mask=batch['residue_mask'].all(1)
        expand=lambda x:x[:,None].expand(b,t,l).reshape(b*t,l)
        with torch.no_grad():
            h=self.bundle.encode(coords.reshape(b*t,l,4,3),mask[:,None].expand(b,t,l).reshape(b*t,l),
                *[expand(batch[k]) for k in ('aatype','residue_index','chain_id')])
        return h.reshape(b,t,l,-1),mask

    def sample(self,observed,mask,aa,ri,ci,temp,steps,n_samples,generator=None,grad_steps=None):
        # observed [B,K,L,4,3]; only observed input, no future or target argument.
        b,k,l=observed.shape[:3]
        with torch.no_grad():h=self.bundle.encode(observed[:,-1],mask,aa,ri,ci)
        repeat=lambda x:x[:,None].expand(b,n_samples,*x.shape[1:]).reshape(b*n_samples,*x.shape[1:])
        start=repeat(h);rm=repeat(mask)
        history=None;dt=None
        if self.history_conditioning:
            # Use the same final four observed frames at train and evaluation time.
            frames=observed[:,-4:];n=frames.shape[1]
            expand=lambda x:x[:,None].expand(b,n,*x.shape[1:]).reshape(b*n,*x.shape[1:])
            with torch.no_grad():
                history=self.bundle.encode(frames.reshape(b*n,l,4,3),expand(mask),
                    expand(aa),expand(ri),expand(ci)).reshape(b,n,l,-1)
            history=repeat(history)
            dt=torch.full((b*n_samples,n),self.kernel.cfg.step_ps,device=start.device)
        factor=None
        if getattr(self,'noise_mix',0.):
            factor=repeat(self.anchor_kernel(observed[:,-1,:,1],mask))
        # Training differentiates through -grad U, requiring attention double backward.
        # Fused CUDA SDPA kernels do not implement that derivative.
        # temperature=None omits the explicit temperature input (T3 'none' ablation).
        temp_tensor=None if temp is None else repeat(temp[:,None]).flatten()
        with sdpa_kernel(SDPBackend.MATH):
            traj=self.kernel.rollout(start,rm,temp_tensor,steps,
                 H_hist=history,dt_hist=dt,seq_feature=start,generator=generator,
                 detach=not self.training,grad_steps=grad_steps,noise_factor=factor,
                 collective_mobility=getattr(self,'collective_mobility',False))
        prediction_kind=getattr(self,'prediction_kind','endpoint')
        if prediction_kind not in ('endpoint','time_mixture'):
            raise ValueError(f'unknown prediction_kind: {prediction_kind}')
        if prediction_kind=='time_mixture':
            # Independent observation times give one iid mixture draw per path.
            # Clone the path RNG so selecting times never changes later paths.
            time_rng=torch.Generator(device=start.device)
            if generator is None:
                time_rng.set_state(torch.cuda.get_rng_state(start.device) if start.is_cuda else torch.get_rng_state())
            else:
                time_rng.set_state(generator.get_state())
            times=torch.randint(1,steps+1,(b*n_samples,),device=start.device,generator=time_rng)
            endpoint=traj[torch.arange(b*n_samples,device=start.device),times]
        else:
            times=torch.full((b*n_samples,),steps,device=start.device,dtype=torch.long)
            endpoint=traj[:,-1]
        self.last_observation_steps=times.detach().cpu().tolist()
        R,anchor=_extract_backbone_frames(observed[:,-1])
        ca=self.bundle.decode_ca(endpoint,start,repeat(aa),repeat(ri),repeat(ci),repeat(R),repeat(anchor),rm)
        return ca.reshape(b,n_samples,l,3),endpoint.reshape(b,n_samples,l,-1)

    def losses(self,batch,horizon,args,generator):
        # 4 history frames + horizon true future, all at 1ns spacing.
        observed=batch['coords'][:,:4]
        h,mask=self.encode_clip(batch)
        ca,generated=self.sample(observed,mask,*[batch[k] for k in ('aatype','residue_index','chain_id')],
             batch['temperature_k'],horizon,args.n_samples,generator,args.grad_steps)
        latent_loss=ca.new_zeros(());coord_loss=latent_loss;contact_loss=latent_loss;rmsf_loss=latent_loss;distance_loss=latent_loss;clash_loss=latent_loss;mode_loss=latent_loss;mode_distribution_loss=latent_loss
        for b in range(len(ca)):
            m=mask[b];anchor=observed[b,-1,:,1]
            truth=batch['coords'][b,3:,:,1]
            with torch.no_grad():truth=_align_to(truth,anchor,m)
            # Endpoint vs last frame, or explicitly labelled time-mixture targets.
            target_h=h[b,4:] if args.target=='path_mixture' else h[b,-1:]
            target_x=truth[1:] if args.target=='path_mixture' else truth[-1:]
            latent_loss=latent_loss+energy_score(generated[b,:,m].flatten(1),target_h[:,m].flatten(1))
            coord_loss=coord_loss+energy_score(ca[b,:,m].flatten(1),target_x[:,m].flatten(1))
            pred_dist=torch.cdist(ca[b:b+1],ca[b:b+1])[0]
            true_dist=torch.cdist(target_x[None],target_x[None])[0]
            pairs=m[:,None]&m[None,:]
            sep=(batch['residue_index'][b,:,None]-batch['residue_index'][b,None,:]).abs()
            same=batch['chain_id'][b,:,None]==batch['chain_id'][b,None,:]
            pairs=pairs&((sep>=3)|~same)
            pairs=pairs&torch.triu(torch.ones_like(pairs),diagonal=1)
            if not pairs.any():raise ValueError('no valid nonlocal residue pairs')
            distance_loss=distance_loss+energy_score(pred_dist[:,pairs],true_dist[:,pairs])
            clash_loss=clash_loss+torch.relu(3.5-pred_dist[:,pairs]).square().mean()
            # Squared probability error targets conditional mean occupancy;
            # absolute error against single endpoints instead targets a median.
            contact_loss=contact_loss+contact_probability_loss(
                torch.sigmoid((8-pred_dist)/.5).mean(0)[pairs],
                torch.sigmoid((8-true_dist)/.5).mean(0)[pairs],args.contact_objective)
            # Common anchor gauge avoids independently moving the predicted fold.
            pr=(ca[b,:,m]-ca[b,:,m].mean(0)).square().sum(-1).mean(0).clamp_min(1e-8).sqrt()
            # A single observed endpoint cannot estimate conditional ensemble RMSF.
            # With one frame the spread target is identically zero, which would
            # push every prediction to a point mass; that term is undefined, not
            # zero, so skip it and leave the timescale terms to carry the signal.
            if args.target=='path_mixture' and target_x.shape[0]>=2:
                tr=(target_x[:,m]-target_x[:,m].mean(0)).square().sum(-1).mean(0).clamp_min(1e-8).sqrt()
                rmsf_loss=rmsf_loss+(pr-tr).abs().mean()
                if args.mode_weight:
                    mode_loss=mode_loss+slow_mode_scale_loss(ca[b,:,m],target_x[:,m])
                if getattr(args,'mode_distribution_weight',0.):
                    mode_distribution_loss=mode_distribution_loss+slow_mode_distribution_loss(ca[b,:,m],target_x[:,m])
        b=len(ca)
        terms=dict(latent_es=latent_loss/b,coord_es=coord_loss/b,contact=contact_loss/b,rmsf=rmsf_loss/b,distance_es=distance_loss/b,clash=clash_loss/b,slow_mode_scale=mode_loss/b,slow_mode_distribution=mode_distribution_loss/b)
        terms['loss']=args.latent_weight*terms['latent_es']+terms['coord_es']+args.contact_weight*terms['contact']+args.rmsf_weight*terms['rmsf']+args.distance_weight*terms['distance_es']+args.clash_weight*terms['clash']+args.mode_weight*terms['slow_mode_scale']+getattr(args,'mode_distribution_weight',0.)*terms['slow_mode_distribution']
        return terms


def make_loader(args,manifest,horizon,training):
    ds=build_dataset(DataConfig(manifest_path=manifest,training_mode='clip',num_frames=4+horizon,
              frame_stride=1,max_residues=args.max_residues))
    budget=args.train_windows if training else args.val_windows
    if getattr(args,'stateless_data',False):
        ds=budget_subset(ds,budget,training,args.seed+horizon)
    else:
        n=min(len(ds),budget)
        ds=Subset(ds,torch.linspace(0,len(ds)-1,n).long().tolist())
    return DataLoader(ds,batch_size=1,shuffle=training and not getattr(args,'stateless_data',False),
        generator=torch.Generator().manual_seed(args.seed+horizon),collate_fn=collate_clips,num_workers=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stateless-data',action='store_true')
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--cosine',action='store_true')
    p.add_argument('--component',required=True);p.add_argument('--output',required=True)
    p.add_argument('--init-kernel');p.add_argument('--bath-dim',type=int,default=8)
    p.add_argument('--history-conditioning',action='store_true')
    p.add_argument('--noise-mix',type=float,default=0.)
    p.add_argument('--noise-length',type=float,default=8.)
    p.add_argument('--noise-kernel',choices=['gaussian','elastic'],default=None)
    p.add_argument('--noise-cutoff',type=float,default=None)
    p.add_argument('--collective-mobility',action='store_true')
    # Bath physical scales.  The defaults reproduce the historical kernel; the
    # audit showed that at these values C^T M C << Gamma, so S is driven by its
    # own thermostat noise and carries no information about H.  Exposing them
    # lets a run sit where S is a genuine memory instead of an inert OU process.
    p.add_argument('--coupling-scale',type=float,default=None)
    p.add_argument('--gamma-scale',type=float,default=None)
    p.add_argument('--gamma-floor',type=float,default=None)
    # Source of the initial bath state when history conditioning is on.
    p.add_argument('--s0-mode',choices=['gru','exact','prior'],default='gru')
    p.add_argument('--manifest',default='artifacts/mdcath_manifest_full/train.jsonl')
    p.add_argument('--val-manifest',default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--horizons',type=int,nargs='+',default=[1,4,16,64])
    p.add_argument('--n-samples',type=int,default=4);p.add_argument('--grad-steps',type=int,default=8)
    p.add_argument('--max-residues',type=int,default=96);p.add_argument('--train-windows',type=int,default=1024)
    p.add_argument('--val-windows',type=int,default=4);p.add_argument('--val-every',type=int,default=50)
    p.add_argument('--latent-weight',type=float,default=.1);p.add_argument('--contact-weight',type=float,default=10.)
    p.add_argument('--rmsf-weight',type=float,default=1.);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--contact-objective',choices=['mae','brier'],default='mae')
    p.add_argument('--distance-weight',type=float,default=0.)
    p.add_argument('--clash-weight',type=float,default=0.)
    p.add_argument('--mode-weight',type=float,default=0.)
    p.add_argument('--mode-distribution-weight',type=float,default=0.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--target',choices=['endpoint','path_mixture'],default='endpoint')
    a=p.parse_args();torch.set_num_threads(4);_set_reproducibility(a.seed,False)
    audit=validate_manifest_split(a.manifest,a.val_manifest)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not a.resume:raise FileExistsError(out)
    if a.resume and not a.stateless_data:raise ValueError('resume requires stateless data')
    model=CADynamics(a.component,'cuda',a.init_kernel,a.bath_dim,a.history_conditioning,a.noise_mix,a.noise_length,a.collective_mobility,a.noise_kernel,a.noise_cutoff).to('cuda')
    # Record any physical-scale override in the kernel config so the saved
    # checkpoint stays self-describing and eval cannot silently use defaults.
    for name in ('coupling_scale','gamma_scale','gamma_floor'):
        value=getattr(a,name,None)
        if value is not None:
            setattr(model.kernel.cfg,name,float(value))
            setattr(model.kernel.bath,name,float(value))
    # S_0 source recorded on the module (and in run.json via vars(a)).
    model.kernel.s0_mode=a.s0_mode
    opt=torch.optim.AdamW(model.kernel.parameters(),lr=a.lr,weight_decay=.01)
    train={h:make_loader(a,a.manifest,h,True) for h in a.horizons}
    val={h:make_loader(a,a.val_manifest,h,False) for h in a.horizons}
    iters={h:iter(dl) for h,dl in train.items()}
    if not a.resume:(out/'run.json').write_text(json.dumps(dict(args=vars(a),split_audit=audit),indent=2))
    def record(row):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    def validate():
        model.eval();generator=torch.Generator(device='cuda').manual_seed(123)
        rows={}
        # Kernel forces enable autograd internally; all parameters frozen for eval graph.
        with torch.no_grad():
            for h,loader in val.items():
                batch_rows=[{k:float(v) for k,v in model.losses(_move_batch(b,'cuda'),h,a,generator).items()} for b in loader]
                if not batch_rows or any(not all(torch.isfinite(torch.tensor(v)) for v in row.values()) for row in batch_rows):
                    raise RuntimeError('empty or nonfinite dynamics validation')
                rows[str(h)]={k:sum(r[k] for r in batch_rows)/len(batch_rows) for k in batch_rows[0]}
        return dict(by_horizon=rows,loss=sum(r['loss'] for r in rows.values())/len(rows))
    def save(tag,step,metrics):
        torch.save(dict(checkpoint_kind='lasp_ca_dynamics',kernel=model.kernel.state_dict(),
            kernel_config=asdict(model.kernel.cfg),component_checkpoint=torch.load(a.component,map_location='cpu',weights_only=False),
            args=vars(a),step=step,validation=metrics,optimizer=opt.state_dict(),
            rng_state=rng_state(),sampling_rng=generator.get_state(),best_loss=best),out/f'checkpoint_{tag}.tmp')
        (out/f'checkpoint_{tag}.tmp').replace(out/f'checkpoint_{tag}.pt')
    generator=torch.Generator(device='cuda').manual_seed(a.seed)
    start_step=0
    if a.resume:
        ck=torch.load(out/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        for key,value in ck['args'].items():
            if key!='resume' and getattr(a,key)!=value:raise ValueError(f'resume argument differs: {key}')
        model.kernel.load_state_dict(ck['kernel'],strict=True);opt.load_state_dict(ck['optimizer'])
        best=ck['best_loss'];start_step=ck['step'];restore_rng(ck['rng_state'])
        generator.set_state(ck['sampling_rng'].cpu())
        record(dict(resumed_from=start_step))
    else:
        initial=validate();record(dict(step=0,val=initial));best=initial['loss']
        save('best',0,initial);save('latest',0,initial)
    start=time.time()
    for step in range(start_step+1,a.steps+1):
        h=a.horizons[(step-1)%len(a.horizons)]
        if a.stateless_data:batch=step_batch(train[h],step,a.seed)
        else:
            try:batch=next(iters[h])
            except StopIteration:iters[h]=iter(train[h]);batch=next(iters[h])
        scale=_lr_scale(step-1,a.steps,a.warmup) if a.cosine else min(1.,step/max(1,a.warmup))
        for group in opt.param_groups:group['lr']=a.lr*scale
        model.train();model.bundle.eval();opt.zero_grad(set_to_none=True)
        terms=model.losses(_move_batch(batch,'cuda'),h,a,generator)
        if not torch.isfinite(terms['loss']):raise RuntimeError('nonfinite training loss')
        terms['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(model.kernel.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step==1 or step%10==0:record(dict(step=step,horizon=h,seconds=time.time()-start,grad_norm=float(norm),train={k:float(v.detach()) for k,v in terms.items()}))
        if step%a.val_every==0 or step==a.steps:
            metrics=validate();record(dict(step=step,val=metrics))
            if metrics['loss']<best:best=metrics['loss'];save('best',step,metrics)
            save('latest',step,metrics)
    print('done',flush=True)


if __name__=='__main__':main()
