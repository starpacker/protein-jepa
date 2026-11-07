"""Train autonomous GLE with a fixed or teacher-constrained CA representation.

True futures supply training targets only. Rollout starts from the observed
anchor and optionally its observed history. Checkpoints embed the component
weights and kernel configuration; no external mutable component is needed.
"""
import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import tempfile
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
from protein_jepa_frame.training.train_lasp_calibration import teacher_terms
from evaluation.ensemble_metrics_ext import _align_to
from evaluation.geometry import kabsch_align


def _observed_readout_weight(args,train_encoder):
    weight=float(getattr(args,'observed_readout_weight',0.))
    if not math.isfinite(weight) or weight<0:
        raise ValueError('--observed-readout-weight must be finite and nonnegative')
    if weight and not train_encoder:
        raise ValueError('--observed-readout-weight > 0 requires --train-encoder')
    return weight


def _validate_resume_arguments(args,recorded):
    recorded=dict(recorded)
    # Absence in a legacy checkpoint means the term was disabled; it cannot
    # silently acquire a positive weight on resume.
    recorded.setdefault('observed_readout_weight',0.)
    for key,value in recorded.items():
        if key=='noise_kernel' and value is None:value='gaussian'
        if key=='noise_cutoff' and value is None:value=10.
        if key!='resume' and getattr(args,key)!=value:
            raise ValueError(f'resume argument differs: {key}')


def energy_score(samples,target):
    """Unbiased sample energy score, Euclidean distance normalized per dimension."""
    # [M,D] samples, [T,D] empirical observations; all target frames equal weight.
    scale=samples.shape[-1]**.5
    cross=torch.cdist(samples[None],target[None]).mean()/scale
    if len(samples)<2:raise ValueError('energy score requires >=2 samples')
    pair=torch.pdist(samples).mean()/scale
    return cross-.5*pair


def temporal_observable(ca, mask, pairs, cutoff=8.0):
    """Fixed differentiable geometry features for the multi-lag joint target.

    ``ca`` is [M, L, 3] CA (one time point, full padded length), ``mask`` the
    [L] boolean validity mask, ``pairs`` a [P,2] tensor of nonlocal valid-pair
    indices.  Returns [M, 2P+1] with
      * internal nonlocal pair distances (upper-triangular, |i-j|>=3 or cross-chain),
      * nonlocal soft-contact occupancies sigmoid((cutoff - d)/0.5),
      * global conformational scale (radius of gyration over valid residues, scalar).
    The observable is the same for every lag, so concatenating lags gives the
    joint vector Phi = [phi(X_1), phi(X_4), phi(X_16)].
    """
    dist = torch.cdist(ca, ca)                       # [M, L, L]
    d = dist[:, pairs[:, 0], pairs[:, 1]]            # [M, P]
    occ = torch.sigmoid((cutoff - d) / 0.5)          # [M, P]
    v = ca[:, mask, :]
    rg = (v - v.mean(dim=1, keepdim=True)).square().sum(-1).mean(-1).sqrt()  # [M]
    return torch.cat((d, occ, rg[:, None]), dim=-1)  # [M, 2P+1]


def joint_temporal_score(generated_cas, target_cas, mask, pairs):
    """Multi-lag joint Energy Score.

    ``generated_cas`` and ``target_cas`` are lists (one per lag) of [M,L,3] and
    [T,L,3] CA tensors.  Each lag's observable is concatenated into one joint
    vector, then a single energy score is taken over the joint vectors -- so the
    three time points of one trajectory are scored jointly, not as independent
    per-lag ensemble errors.
    """
    gen = torch.cat([temporal_observable(ca, mask, pairs) for ca in generated_cas], dim=-1)
    tgt = torch.cat([temporal_observable(ca, mask, pairs) for ca in target_cas], dim=-1)
    return energy_score(gen, tgt)


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
    def __init__(self,component,device,init_kernel=None,bath_dim=8,history_conditioning=False,noise_mix=0.,noise_length=8.,collective_mobility=None,noise_kernel=None,noise_cutoff=None,s0_mode=None):
        super().__init__()
        self.history_conditioning=history_conditioning
        self.noise_mix=noise_mix;self.noise_length=noise_length
        if init_kernel:
            ck=torch.load(init_kernel,map_location='cpu',weights_only=False)
            config=StageB1bConfig(**ck['kernel_config'])
            # Preserve checkpoint transition semantics. Older checkpoints without
            # this field used the non-collective sampler.
            recorded=ck.get('args',{})
            if collective_mobility is None:
                collective_mobility=recorded.get('collective_mobility',False)
            if noise_kernel is None:noise_kernel=recorded.get('noise_kernel','gaussian')
            if noise_cutoff is None:noise_cutoff=recorded.get('noise_cutoff',10.)
        else:
            config=StageB1bConfig(bath_dim=bath_dim,tail_init=.01,max_step_norm=10.,max_gamma_step=1.)
            if collective_mobility is None:
                collective_mobility=True
        if noise_kernel is None:noise_kernel='gaussian'
        if noise_cutoff is None:noise_cutoff=10.
        # Restore the entire representation paired with the dynamics, including
        # decoder weights, projection settings and calibration metadata. This
        # also makes resumption independent of a mutable --component file.
        embedded=ck.get('component_checkpoint') if init_kernel else None
        if embedded is not None:
            self.component_source=embedded
            with tempfile.TemporaryDirectory(prefix='lasp_component_') as directory:
                path=Path(directory)/'component.pt'
                torch.save(embedded,path)
                self.bundle=CAComponentBundle(str(path),device)
        else:
            # The bundle owns loading its component. Training reads its source
            # metadata only when saving; lightweight alternative bundle classes
            # need not implement the checkpoint file interface themselves.
            self.component_source=None
            self.bundle=CAComponentBundle(component,device)
        if noise_kernel not in ('gaussian','elastic'):
            raise ValueError(f'unknown noise kernel {noise_kernel!r}')
        if not noise_cutoff>0:raise ValueError('noise cutoff must be positive')
        self.noise_kernel=noise_kernel;self.noise_cutoff=noise_cutoff
        self.collective_mobility=collective_mobility
        self.kernel=StageB1b(config).to(device)
        if init_kernel:self.kernel.load_state_dict(ck['kernel'],strict=True)
        # Restore the trained S_0 source so evaluation cannot silently use a
        # different prior than the checkpoint saw.
        if s0_mode is None:s0_mode=ck.get('args',{}).get('s0_mode','gru') if init_kernel else 'gru'
        if s0_mode not in ('gru','exact','exact_spatial','prior'):
            raise ValueError(f'unknown s0_mode {s0_mode!r}')
        self.kernel.s0_mode=s0_mode
        self.train_encoder=False
        self.target_encoder=None

    def enable_encoder_training(self):
        """Keep a fixed target basis while learning from autonomous forecasts.

        The teacher is the calibrated starting encoder, not future input to
        the sampler. The geometric decoder remains fixed so an apparent gain
        cannot be obtained by changing the readout at the same time.
        """
        self.train_encoder=True
        self.target_encoder=copy.deepcopy(self.bundle.encoder).eval()
        self.target_encoder.requires_grad_(False)
        self.bundle.encoder.requires_grad_(True)
        self.kernel.bath.detach_conditioning=False

    def train(self,mode=True):
        super().train(mode)
        # Preserve the deterministic frame representation used at deployment.
        # eval() controls dropout, not whether encoder gradients are computed.
        self.bundle.eval()
        if getattr(self,'target_encoder',None) is not None:
            self.target_encoder.eval()
        return self

    def component_checkpoint(self,source):
        """Snapshot the live encoder along with the existing decoder contract."""
        checkpoint=copy.deepcopy(source)
        checkpoint['encoder']={k:v.detach().cpu().clone() for k,v in self.bundle.encoder.state_dict().items()}
        return checkpoint

    def representation_retention(self,batch):
        # Retention sees only observed frames. Future supervision stays in the
        # detached teacher target and the coordinate forecast losses.
        observed=batch['coords'][:,:4]
        b,t,l=observed.shape[:3]
        mask=batch['residue_mask'][:,:4].all(1)
        expand=lambda x:x[:,None].expand(b,t,l).reshape(b*t,l)
        rm=mask[:,None].expand(b,t,l).reshape(b*t,l)
        inputs=(observed.reshape(b*t,1,l,4,3),rm[:,None],
                *[expand(batch[k]) for k in ('aatype','residue_index','chain_id')])
        student=self.bundle.encoder(*inputs,token_mask=rm[:,None])[:,0].reshape(b,t,l,-1)
        with torch.no_grad():
            teacher=self.target_encoder(*inputs,token_mask=rm[:,None])[:,0].reshape(b,t,l,-1)
        return teacher_terms(student,teacher,mask)

    def observed_readout_retention(self,batch):
        """Observed CA vector MSE (Angstrom^2), matching the readout diagnostic.

        Decode the first three observed H states relative to the fourth anchor
        through the fixed decoder and its existing bond projection. Both H
        paths retain encoder gradients; labels use only the observed anchor
        gauge. Each clip averages residues/frames before the batch average.
        """
        observed=batch['coords'][:,:4]
        mask=batch['residue_mask'][:,:4].all(1)
        b,t,l=observed.shape[:3]
        if t!=4 or bool((mask.sum(-1)<4).any()):
            raise ValueError('observed readout retention needs four frames and >=4 valid residues')
        expand=lambda value,n:value[:,None].expand(b,n,*value.shape[1:]).reshape(b*n,*value.shape[1:])
        metadata=[batch[k] for k in ('aatype','residue_index','chain_id')]
        h=self.bundle.encode(observed.reshape(b*t,l,4,3),expand(mask,t),
                             *[expand(value,t) for value in metadata]).reshape(b,t,l,-1)
        rotation,anchor=_extract_backbone_frames(observed[:,-1])
        prediction=self.bundle.decode_ca(h[:,:3].reshape(b*3,l,-1),expand(h[:,3],3),
            *[expand(value,3) for value in metadata],expand(rotation,3),expand(anchor,3),
            expand(mask,3)).reshape(b,3,l,3)
        values=[]
        for index in range(b):
            m=mask[index]
            with torch.no_grad():
                truth=torch.stack([kabsch_align(frame[:,1],anchor[index],m)
                                   for frame in observed[index,:3]])
            values.append((prediction[index,:,m]-truth[:,m]).square().sum(-1).mean())
        return torch.stack(values).mean()

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
            if getattr(self,'target_encoder',None) is None:
                h=self.bundle.encode(coords.reshape(b*t,l,4,3),mask[:,None].expand(b,t,l).reshape(b*t,l),
                    *[expand(batch[k]) for k in ('aatype','residue_index','chain_id')])
            else:
                rm=mask[:,None].expand(b,t,l).reshape(b*t,l)
                h=self.target_encoder(coords.reshape(b*t,1,l,4,3),rm[:,None],
                    *[expand(batch[k]) for k in ('aatype','residue_index','chain_id')],token_mask=rm[:,None])[:,0]
        return h.reshape(b,t,l,-1),mask

    def sample(self,observed,mask,aa,ri,ci,temp,steps,n_samples,generator=None,grad_steps=None):
        # observed [B,K,L,4,3]; only observed input, no future or target argument.
        b,k,l=observed.shape[:3]
        encoder_grad=torch.is_grad_enabled() and self.training and getattr(self,'train_encoder',False)
        with torch.set_grad_enabled(encoder_grad):
            h=self.bundle.encode(observed[:,-1],mask,aa,ri,ci)
        repeat=lambda x:x[:,None].expand(b,n_samples,*x.shape[1:]).reshape(b*n_samples,*x.shape[1:])
        start=repeat(h);rm=repeat(mask)
        history=None;dt=None;history_once=None;dt_once=None
        if self.history_conditioning:
            # Use the same final four observed frames at train and evaluation time.
            frames=observed[:,-4:];n=frames.shape[1]
            expand=lambda x:x[:,None].expand(b,n,*x.shape[1:]).reshape(b*n,*x.shape[1:])
            with torch.set_grad_enabled(encoder_grad):
                history_once=self.bundle.encode(frames.reshape(b*n,l,4,3),expand(mask),
                    expand(aa),expand(ri),expand(ci)).reshape(b,n,l,-1)
            history=repeat(history_once)
            dt_once=torch.full((b,n),self.kernel.cfg.step_ps,device=start.device)
            dt=repeat(dt_once)
        factor=None;factor_once=None
        if getattr(self,'noise_mix',0.):
            factor_once=self.anchor_kernel(observed[:,-1,:,1],mask)
            factor=repeat(factor_once)
        # Training differentiates through -grad U, requiring attention double backward.
        # Fused CUDA SDPA kernels do not implement that derivative.
        # temperature=None omits the explicit temperature input (T3 'none' ablation).
        temp_tensor=None if temp is None else repeat(temp[:,None]).flatten()
        with sdpa_kernel(SDPBackend.MATH):
            initial_bath=None
            if getattr(self.kernel,'s0_mode','gru')=='exact_spatial':
                if history_once is None or not self.collective_mobility:
                    raise ValueError('exact_spatial requires observed history and collective mobility')
                # Ensemble members share the same observed window. Filter once
                # per window, then draw independent samples with the FULL bath
                # factor. Repeating the filter would multiply its dense bath
                # covariance memory and backward cost by n_samples.
                mean,chol=self.kernel.initial_bath_state(
                    history_once,dt_once,temp,mask,seq_feature=h,
                    noise_factor=factor_once,create_graph=self.training)
                eps=torch.randn((b,n_samples,mean.shape[1]*mean.shape[2]),
                    device=start.device,dtype=start.dtype,generator=generator)
                centered=torch.einsum('bij,bmj->bmi',chol,eps).reshape(
                    b,n_samples,*mean.shape[1:])
                initial_bath=(mean[:,None]+centered).reshape(b*n_samples,*mean.shape[1:])
            traj=self.kernel.rollout(start,rm,temp_tensor,steps,
                 H_hist=history,dt_hist=dt,seq_feature=start,generator=generator,
                 S0=initial_bath,
                 detach=not self.training,grad_steps=grad_steps,noise_factor=factor,
                 collective_mobility=self.collective_mobility)
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

    def sample_temporal(self,observed,mask,aa,ri,ci,temp,lags,n_samples,generator=None,grad_steps=None):
        """Roll out one continuous trajectory to max(lags) and decode CA at each lag.

        Mirrors ``sample`` (same encoder/history/bath-initialisation), but instead
        of a single endpoint it returns the decoded CA at every requested lag as a
        list of [b, n_samples, L, 3] tensors sharing one H/S rollout.  This is the
        sampling backbone for the multi-lag joint Energy Score (target='temporal').
        """
        b,k,l=observed.shape[:3]
        encoder_grad=torch.is_grad_enabled() and self.training and getattr(self,'train_encoder',False)
        with torch.set_grad_enabled(encoder_grad):
            h=self.bundle.encode(observed[:,-1],mask,aa,ri,ci)
        repeat=lambda x:x[:,None].expand(b,n_samples,*x.shape[1:]).reshape(b*n_samples,*x.shape[1:])
        start=repeat(h);rm=repeat(mask)
        history=None;dt=None;history_once=None;dt_once=None
        if self.history_conditioning:
            frames=observed[:,-4:];n=frames.shape[1]
            expand=lambda x:x[:,None].expand(b,n,*x.shape[1:]).reshape(b*n,*x.shape[1:])
            with torch.set_grad_enabled(encoder_grad):
                history_once=self.bundle.encode(frames.reshape(b*n,l,4,3),expand(mask),
                    expand(aa),expand(ri),expand(ci)).reshape(b,n,l,-1)
            history=repeat(history_once)
            dt_once=torch.full((b,n),self.kernel.cfg.step_ps,device=start.device)
            dt=repeat(dt_once)
        factor=None;factor_once=None
        if getattr(self,'noise_mix',0.):
            factor_once=self.anchor_kernel(observed[:,-1,:,1],mask)
            factor=repeat(factor_once)
        temp_tensor=None if temp is None else repeat(temp[:,None]).flatten()
        steps=max(lags)
        with sdpa_kernel(SDPBackend.MATH):
            initial_bath=None
            if getattr(self.kernel,'s0_mode','gru')=='exact_spatial':
                if history_once is None or not self.collective_mobility:
                    raise ValueError('exact_spatial requires observed history and collective mobility')
                mean,chol=self.kernel.initial_bath_state(
                    history_once,dt_once,temp,mask,seq_feature=h,
                    noise_factor=factor_once,create_graph=self.training)
                eps=torch.randn((b,n_samples,mean.shape[1]*mean.shape[2]),
                    device=start.device,dtype=start.dtype,generator=generator)
                centered=torch.einsum('bij,bmj->bmi',chol,eps).reshape(
                    b,n_samples,*mean.shape[1:])
                initial_bath=(mean[:,None]+centered).reshape(b*n_samples,*mean.shape[1:])
            traj=self.kernel.rollout(start,rm,temp_tensor,steps,
                 H_hist=history,dt_hist=dt,seq_feature=start,generator=generator,
                 S0=initial_bath,
                 detach=not self.training,grad_steps=grad_steps,noise_factor=factor,
                 collective_mobility=self.collective_mobility)
        R,anchor=_extract_backbone_frames(observed[:,-1])
        cas=[]
        for lag in lags:
            endpoint=traj[:,lag]
            ca=self.bundle.decode_ca(endpoint,start,repeat(aa),repeat(ri),repeat(ci),
                                     repeat(R),repeat(anchor),rm)
            cas.append(ca.reshape(b,n_samples,l,3))
        return cas

    def losses(self,batch,horizon,args,generator):
        observed_readout_weight=_observed_readout_weight(args,getattr(self,'train_encoder',False))
        # 4 history frames + horizon true future, all at 1ns spacing.
        observed=batch['coords'][:,:4]
        h,mask=self.encode_clip(batch)
        # Future validity affects scoring only; forecasts receive the mask
        # available from the observed history, exactly as at evaluation.
        observed_mask=batch['residue_mask'][:,:4].all(1)
        if args.target=='temporal':
            # Multi-lag joint Energy Score: one continuous rollout, decode at each
            # requested lag, and score the concatenated observable vector against
            # the same-lag true MD observables jointly.
            lags=tuple(int(v) for v in args.temporal_lags)
            if horizon<max(lags):
                # Roll out to max(lags) regardless of the horizon slot.
                horizon=max(lags)
            cas=self.sample_temporal(observed,observed_mask,*[batch[k] for k in ('aatype','residue_index','chain_id')],
                 batch['temperature_k'],lags,args.n_samples,generator,args.grad_steps)
            temporal_loss=cas[0].new_zeros(())
            n_batches=0
            for b in range(len(cas[0])):
                m=mask[b];anchor=observed[b,-1,:,1]
                truth=batch['coords'][b,3:,:,1]
                with torch.no_grad():truth=_align_to(truth,anchor,m)
                # True CA at each lag: frame index 3+lag (0-based, after 4 history frames).
                target_cas=[truth[lag-1:lag] for lag in lags]      # each [1,L,3]
                gen_cas=[ca[b] for ca in cas]                       # each [n_samples,L,3]
                # Build nonlocal pair index for the temporal observable.
                pairs=m[:,None]&m[None,:]
                sep=(batch['residue_index'][b,:,None]-batch['residue_index'][b,None,:]).abs()
                same=batch['chain_id'][b,:,None]==batch['chain_id'][b,None,:]
                pairs=pairs&((sep>=3)|~same)
                pairs=pairs&torch.triu(torch.ones_like(pairs),diagonal=1)
                if not pairs.any():raise ValueError('no valid nonlocal residue pairs')
                pair_idx=pairs.nonzero(as_tuple=False)
                temporal_loss=temporal_loss+joint_temporal_score(gen_cas,target_cas,m,pair_idx)
                n_batches+=1
            temporal_loss=temporal_loss/max(1,n_batches)
            return dict(temporal_es=temporal_loss,loss=temporal_loss)
        ca,generated=self.sample(observed,observed_mask,*[batch[k] for k in ('aatype','residue_index','chain_id')],
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
        if getattr(self,'train_encoder',False):
            terms['teacher'],terms['teacher_delta']=self.representation_retention(batch)
            terms['loss']=terms['loss']+args.teacher_weight*terms['teacher']+args.teacher_delta_weight*terms['teacher_delta']
            if observed_readout_weight:
                terms['observed_readout_mse']=self.observed_readout_retention(batch)
                terms['loss']=terms['loss']+observed_readout_weight*terms['observed_readout_mse']
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
    p.add_argument('--train-encoder',action='store_true',
                   help='feed autonomous forecast gradients into H, with a frozen target encoder and decoder')
    p.add_argument('--encoder-lr-scale',type=float,default=.1)
    p.add_argument('--teacher-weight',type=float,default=1.)
    p.add_argument('--teacher-delta-weight',type=float,default=.1)
    p.add_argument('--observed-readout-weight',type=float,default=0.,
                   help='joint-only observed CA vector MSE weight; keeps the fixed decoder readout of the first four frames')
    p.add_argument('--noise-mix',type=float,default=0.)
    p.add_argument('--noise-length',type=float,default=8.)
    p.add_argument('--noise-kernel',choices=['gaussian','elastic'],default=None)
    p.add_argument('--noise-cutoff',type=float,default=None)
    p.add_argument('--collective-mobility',action=argparse.BooleanOptionalAction,default=None,
                   help='use the shared-displacement transition required by exact S_0 inference (default: on for new runs)')
    # Bath physical scales.  The defaults reproduce the historical kernel; the
    # audit showed that at these values C^T M C << Gamma, so S is driven by its
    # own thermostat noise and carries no information about H.  Exposing them
    # lets a run sit where S is a genuine memory instead of an inert OU process.
    p.add_argument('--coupling-scale',type=float,default=None)
    p.add_argument('--gamma-scale',type=float,default=None)
    p.add_argument('--gamma-floor',type=float,default=None)
    # Source of the initial bath state when history conditioning is on.
    p.add_argument('--s0-mode',choices=['gru','exact','exact_spatial','prior'],default=None,
                   help='initial bath inference (default: inherit warm start/resume, otherwise gru)')
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
    p.add_argument('--target',choices=['endpoint','path_mixture','temporal'],default='endpoint')
    p.add_argument('--temporal-lags',type=int,nargs='+',default=[1,4,16],
                   help='lags (ns) scored jointly by the temporal target')
    a=p.parse_args()
    _observed_readout_weight(a,a.train_encoder)
    if a.s0_mode is None:
        if a.resume:
            source_args=json.loads((Path(a.output)/'run.json').read_text())['args']
        elif a.init_kernel:
            source_args=torch.load(a.init_kernel,map_location='cpu',weights_only=False).get('args',{})
        else:
            source_args={}
        a.s0_mode=source_args.get('s0_mode','gru')
    if a.collective_mobility is None:
        if a.resume:
            prior_args=json.loads((Path(a.output)/'run.json').read_text())['args']
            a.collective_mobility=prior_args.get('collective_mobility',False)
        elif a.init_kernel:
            init_ck=torch.load(a.init_kernel,map_location='cpu',weights_only=False)
            a.collective_mobility=init_ck.get('args',{}).get('collective_mobility',False)
        else:
            a.collective_mobility=True
    if a.target=='temporal':
        # The temporal target needs the true MD path to cover every requested lag,
        # so the loader horizon is pinned to the longest lag and the schedule uses
        # that single horizon.
        a.temporal_lags=sorted(set(int(v) for v in a.temporal_lags))
        if any(v<1 for v in a.temporal_lags):
            raise ValueError('--temporal-lags must be positive integer lags')
        a.horizons=[max(a.temporal_lags)]
    torch.set_num_threads(4);_set_reproducibility(a.seed,False)
    if a.train_encoder:
        if not a.history_conditioning:
            raise ValueError('--train-encoder requires --history-conditioning for H+S feedback')
        if a.grad_steps<max(a.horizons):
            raise ValueError('--train-encoder requires --grad-steps >= max(horizons) to retain the H/S initial-state gradient')
        if a.teacher_weight<=0 or a.teacher_delta_weight<=0 or a.encoder_lr_scale<=0:
            raise ValueError('joint encoder training requires positive teacher and encoder learning-rate weights')
    if a.s0_mode in ('exact','exact_spatial'):
        if not a.history_conditioning:
            raise ValueError(f"s0_mode={a.s0_mode!r} requires --history-conditioning")
        if not a.collective_mobility:
            raise ValueError(f"s0_mode={a.s0_mode!r} requires collective mobility")
        if a.s0_mode=='exact' and a.noise_mix != 0.:
            raise ValueError("s0_mode='exact' requires --noise-mix 0 for the diagonal-noise Kalman posterior")
    audit=validate_manifest_split(a.manifest,a.val_manifest)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not a.resume:raise FileExistsError(out)
    if a.resume and not a.stateless_data:raise ValueError('resume requires stateless data')
    kernel_source=str(out/'checkpoint_latest.pt') if a.resume else a.init_kernel
    model=CADynamics(a.component,'cuda',kernel_source,a.bath_dim,a.history_conditioning,a.noise_mix,a.noise_length,a.collective_mobility,a.noise_kernel,a.noise_cutoff).to('cuda')
    # Persist the effective inherited kernel, not nullable CLI placeholders.
    # Evaluation must reconstruct the same anchor-conditioned covariance.
    a.noise_kernel=model.noise_kernel
    a.noise_cutoff=model.noise_cutoff
    # Record any physical-scale override in the kernel config so the saved
    # checkpoint stays self-describing and eval cannot silently use defaults.
    for name in ('coupling_scale','gamma_scale','gamma_floor'):
        value=getattr(a,name,None)
        if value is not None:
            setattr(model.kernel.cfg,name,float(value))
            setattr(model.kernel.bath,name,float(value))
    # S_0 source recorded on the module (and in run.json via vars(a)).
    model.kernel.s0_mode=a.s0_mode
    if a.train_encoder:model.enable_encoder_training()
    groups=[dict(params=list(model.kernel.parameters()),lr=a.lr,lr_scale=1.)]
    if a.train_encoder:
        groups.append(dict(params=list(model.bundle.encoder.parameters()),lr=a.lr*a.encoder_lr_scale,lr_scale=a.encoder_lr_scale))
    parameters=[p for group in groups for p in group['params']]
    opt=torch.optim.AdamW(groups,lr=a.lr,weight_decay=.01)
    component_source=model.component_source
    if component_source is None:
        component_source=torch.load(a.component,map_location='cpu',weights_only=False)
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
            kernel_config=asdict(model.kernel.cfg),component_checkpoint=model.component_checkpoint(component_source),
            args=vars(a),step=step,validation=metrics,optimizer=opt.state_dict(),
            target_encoder=model.target_encoder.state_dict() if a.train_encoder else None,
            rng_state=rng_state(),sampling_rng=generator.get_state(),best_loss=best),out/f'checkpoint_{tag}.tmp')
        (out/f'checkpoint_{tag}.tmp').replace(out/f'checkpoint_{tag}.pt')
    generator=torch.Generator(device='cuda').manual_seed(a.seed)
    start_step=0
    if a.resume:
        ck=torch.load(out/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
        _validate_resume_arguments(a,ck['args'])
        model.kernel.load_state_dict(ck['kernel'],strict=True);opt.load_state_dict(ck['optimizer'])
        if a.train_encoder:
            model.bundle.encoder.load_state_dict(ck['component_checkpoint']['encoder'],strict=True)
            model.target_encoder.load_state_dict(ck['target_encoder'],strict=True)
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
        for group in opt.param_groups:group['lr']=a.lr*scale*group.get('lr_scale',1.)
        model.train();model.bundle.eval();opt.zero_grad(set_to_none=True)
        terms=model.losses(_move_batch(batch,'cuda'),h,a,generator)
        if not torch.isfinite(terms['loss']):raise RuntimeError('nonfinite training loss')
        terms['loss'].backward()
        encoder_norm=None
        if a.train_encoder and (step==1 or step%10==0):
            gradients=[p.grad.detach().double().norm() for p in model.bundle.encoder.parameters() if p.grad is not None]
            encoder_norm=float(torch.stack(gradients).norm()) if gradients else 0.
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
        opt.step()
        if step==1 or step%10==0:
            row=dict(step=step,horizon=h,seconds=time.time()-start,grad_norm=float(norm),train={k:float(v.detach()) for k,v in terms.items()})
            if encoder_norm is not None:row['encoder_grad_norm']=encoder_norm
            record(row)
        if step%a.val_every==0 or step==a.steps:
            metrics=validate();record(dict(step=step,val=metrics))
            if metrics['loss']<best:best=metrics['loss'];save('best',step,metrics)
            save('latest',step,metrics)
    print('done',flush=True)


if __name__=='__main__':main()
