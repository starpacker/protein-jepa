"""Continuous-path supervision using the frozen CADynamics sampler and losses.

Only new orchestration lives here. A read-only proxy captures the one production
rollout; the original sampler still draws its posterior/noise and decodes the
endpoint. Intermediate states are decoded with that same B*M batch shape.
"""
from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics, energy_score, _extract_backbone_frames, _align_to

VERSION = 'continuous_ca_ordered_path_objective_v1'
TIME_ATOL_PS = 1e-6
OBJECTIVES = ('joint_path', 'marginal_path')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _times(times, batch, frames, step_ps):
    _require(isinstance(times, torch.Tensor) and times.shape == (batch, frames)
             and times.is_floating_point() and torch.isfinite(times).all(), 'actual time_ps must be finite [B,T]')
    _require(math.isfinite(step_ps) and step_ps > 0, 'invalid kernel step_ps')
    differences = times.detach().double().diff(dim=1)
    _require(torch.allclose(differences, torch.full_like(differences, step_ps), rtol=0., atol=TIME_ATOL_PS),
             'actual time_ps spacing differs from kernel step_ps (absolute tolerance 1e-6 ps)')


def _frozen(model):
    _require(isinstance(model, CADynamics), 'requires the frozen CADynamics implementation')
    _require(not getattr(model, 'train_encoder', False) and getattr(model, 'target_encoder', None) is None,
             'v1 requires a frozen encoder without teacher updates')
    _require(not any(p.requires_grad for p in model.bundle.parameters()), 'encoder/decoder parameters must be frozen')
    _require(not any(module.training for module in model.bundle.modules()),
             'frozen encoder/decoder and all submodules must remain in eval mode')
    _require(getattr(model, 'prediction_kind', 'endpoint') == 'endpoint', 'path sampling requires endpoint sampler semantics')
    _require(model.history_conditioning and model.kernel.s0_mode == 'exact_spatial'
             and model.collective_mobility, 'v1 requires the registered exact_spatial history/collective sampler')


class _KernelCapture:
    def __init__(self, kernel):
        self.source = kernel
        self.calls = 0
        self.trajectory = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    def rollout(self, *args, **kwargs):
        self.calls += 1
        _require(self.calls == 1, 'continuous sampling must call rollout exactly once')
        self.trajectory = self.source.rollout(*args, **kwargs)
        return self.trajectory


class _SamplingView:
    def __init__(self, model):
        self.source = model
        self.kernel = _KernelCapture(model.kernel)

    def __getattr__(self, name):
        return getattr(self.source, name)


def sample_paths(model, observed, mask, aa, ri, ci, temp, steps, n_samples, generator=None,
                 grad_steps=None, *, observed_time_ps, checkpoint_decoder=False):
    """Return CA/H paths [B,M,steps+1,L,3/D], including the observed anchor.

    No future argument is accepted. Parameter modules are never modified.
    Terminal decoding occurs inside the original sample() before any additional
    decode. Full BPTT is mandatory in training; optional checkpointing affects
    only deterministic intermediate-frame decoder activations.
    """
    _frozen(model)
    _require(type(steps) is int and steps >= 1 and type(n_samples) is int and n_samples >= 2,
             'positive integer horizon and at least two members required')
    _require(grad_steps is None or (type(grad_steps) is int and grad_steps >= steps), 'path objective requires full BPTT')
    _require(type(checkpoint_decoder) is bool, 'checkpoint_decoder must be bool')
    _require(observed.ndim == 5 and observed.shape[1] == 4 and observed.shape[-2:] == (4,3),
             'observed coordinates must be [B,4,L,4,3]')
    b, _, length = observed.shape[:3]
    _require(mask.shape == (b,length) and mask.dtype == torch.bool and (mask.sum(-1) >= 4).all(),
             'fixed observed mask needs >=4 residues per window')
    for metadata in (aa,ri,ci):
        _require(metadata.shape == (b,length) and metadata.dtype == torch.long and metadata.device == observed.device,
                 'integer residue metadata must be [B,L] on coordinate device')
    _require(mask.device == observed.device and observed.is_floating_point(), 'coordinate/mask device or dtype differs')
    _require(temp is not None and temp.shape == (b,) and temp.device == observed.device
             and torch.isfinite(temp).all() and (temp > 0).all(), 'finite positive per-window temperature required')
    _times(observed_time_ps,b,4,float(model.kernel.cfg.step_ps))
    selected = mask[:,None,:,None,None].expand_as(observed)
    _require(torch.isfinite(observed[selected]).all(), 'nonfinite observed backbone on valid residues')
    safe_observed = torch.where(selected,observed,torch.zeros_like(observed))
    view = _SamplingView(model)
    endpoint_ca,endpoint_h = CADynamics.sample(view,safe_observed,mask,aa,ri,ci,temp,steps,n_samples,generator,grad_steps)
    _require(view.kernel.calls == 1, 'missing continuous production rollout')
    trajectory = view.kernel.trajectory
    _require(trajectory.shape[:3] == (b*n_samples,steps+1,length), 'unexpected production path shape')
    h_paths = trajectory.reshape(b,n_samples,steps+1,length,-1)
    _require(torch.equal(endpoint_h,h_paths[:,:,-1]), 'endpoint H differs from captured trajectory')
    _require(torch.isfinite(h_paths).all() and torch.isfinite(endpoint_ca[mask[:,None].expand(b,n_samples,length)]).all(),
             'nonfinite generated path')
    repeat = lambda value: value[:,None].expand(b,n_samples,*value.shape[1:]).reshape(b*n_samples,*value.shape[1:])
    rotation,anchor = _extract_backbone_frames(safe_observed[:,-1])
    fixed = (repeat(aa),repeat(ri),repeat(ci),repeat(rotation),repeat(anchor),repeat(mask))
    start = trajectory[:,0]
    def decode(state):
        return model.bundle.decode_ca(state,start,*fixed)
    frames = [repeat(anchor).reshape(b,n_samples,length,3)]
    for t in range(1,steps):
        state = trajectory[:,t]
        value = checkpoint(decode,state,use_reentrant=False,preserve_rng_state=True) if (
            checkpoint_decoder and torch.is_grad_enabled() and state.requires_grad) else decode(state)
        _require(torch.isfinite(value[fixed[-1]]).all(), 'nonfinite decoded intermediate CA')
        frames.append(value.reshape(b,n_samples,length,3))
    frames.append(endpoint_ca)
    ca_paths = torch.stack(frames,dim=2)
    return dict(ca_paths=ca_paths,h_paths=h_paths,endpoint_ca=endpoint_ca,endpoint_h=endpoint_h,
        step_ps=float(model.kernel.cfg.step_ps),rollout_calls=view.kernel.calls,
        observation_steps=list(view.last_observation_steps),checkpoint_decoder=checkpoint_decoder)


def coordinate_path_scores(ca_paths, aligned_reference, residue_mask):
    """Differentiable U-ES, excluding time zero and normalizing by sqrt(dim).

    Joint flattens time/residue/coordinate into one observation. Marginal takes
    an equal-time mean of separate U-ES values, discarding cross-time coupling.
    Inputs are already in the observed anchor gauge; no prediction alignment.
    """
    _require(ca_paths.ndim == 5 and ca_paths.shape[-1] == 3, 'CA paths must be [B,M,T,L,3]')
    b,m,t,l,_ = ca_paths.shape
    _require(m >= 2 and t >= 2 and aligned_reference.shape == (b,t,l,3)
             and residue_mask.shape == (b,l) and residue_mask.dtype == torch.bool,
             'path target/member/mask shape differs')
    endpoint,joint,marginal = [],[],[]
    for i in range(b):
        mask = residue_mask[i]
        _require(mask.any(), 'empty path support')
        predicted = ca_paths[i,:,1:,mask]
        truth = aligned_reference[i,1:,mask]
        _require(torch.isfinite(predicted).all() and torch.isfinite(truth).all(), 'nonfinite scored path')
        end = energy_score(predicted[:,-1].flatten(1),truth[-1:].flatten(1))
        endpoint.append(end)
        if t == 2:
            joint.append(end);marginal.append(end)
        else:
            joint.append(energy_score(predicted.flatten(1),truth.flatten()[None]))
            marginal.append(torch.stack([energy_score(predicted[:,j].flatten(1),truth[j:j+1].flatten(1))
                                         for j in range(t-1)]).mean())
    # Match the frozen endpoint loss's accumulation order, including B>1.
    def average(values):
        total = ca_paths.new_zeros(())
        for value in values:
            total = total + value
        return total / b
    return dict(endpoint=average(endpoint),joint=average(joint),marginal=average(marginal))


class _CachedSampleView:
    def __init__(self, model, paths):
        self.source = model
        self.paths = paths
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.source, name)

    def sample(self, observed,mask,aa,ri,ci,temp,steps,n_samples,generator=None,grad_steps=None):
        self.calls += 1
        _require(self.calls == 1, 'endpoint objective requested more than one cached sample')
        _require(self.paths['ca_paths'].shape[:3] == (observed.shape[0],n_samples,steps+1), 'cached sample budget differs')
        return self.paths['endpoint_ca'],self.paths['endpoint_h']


def _validate_batch(model,batch,horizon,args):
    _frozen(model)
    _require(args.coord_objective in OBJECTIVES and args.path_weight == .5, 'v1 fixes joint/marginal path weight to .5')
    weights = dict(latent_weight=.1,contact_weight=20.,clash_weight=40.,distance_weight=0.,rmsf_weight=0.,
                   mode_weight=0.,mode_distribution_weight=0.,observed_readout_weight=0.)
    _require(all(getattr(args,k,0.) == v for k,v in weights.items()), 'unregistered endpoint/regularizer weights')
    _require(args.target == 'endpoint' and args.contact_objective == 'brier', 'only the endpoint Brier baseline is supported')
    coords = batch['coords']
    _require(type(horizon) is int and horizon >= 1 and coords.ndim == 5 and coords.shape[1] == 4+horizon
             and coords.shape[-2:] == (4,3), 'clip must contain exactly4 observed plus horizon future frames')
    b,t,l = coords.shape[:3]
    rm = batch['residue_mask']
    _require(rm.shape == (b,t,l) and rm.dtype == torch.bool, 'clip residue mask shape/type differs')
    mask = rm[:,:4].all(1)
    _require((mask.sum(-1) >= 4).all(), 'insufficient observed support')
    _require(rm[:,4:][mask[:,None].expand(b,horizon,l)].all(),
             'every future frame must contain the complete fixed observed support')
    _times(batch['time_ps'],b,t,float(model.kernel.cfg.step_ps))
    selected = mask[:,None,:,None,None].expand_as(coords)
    _require(torch.isfinite(coords[selected]).all(), 'nonfinite label or observation on fixed support')
    if 'atom_mask' in batch:
        am = batch['atom_mask']
        _require(am.shape == (b,t,l,4) and am.dtype == torch.bool
                 and am[mask[:,None,:,None].expand_as(am)].all(), 'missing backbone atom on fixed support')
    safe = dict(batch)
    safe['coords'] = torch.where(selected,coords,torch.zeros_like(coords))
    # The fixed support is selected exclusively from observations; neither
    # encoding nor scoring may introduce a future-dependent intersection.
    safe['residue_mask'] = mask[:,None].expand(b,t,l)
    return safe,mask


def ordered_path_losses(model,batch,horizon,args,generator):
    """One observation-only path draw, shared by matched joint/marginal losses.

    Required args: coord_objective, path_weight=.5, checkpoint_decoder (bool),
    n_samples, grad_steps and the frozen endpoint-loss arguments. Targets are
    read only for validation/scoring; they never reach sample_paths().
    """
    safe,mask = _validate_batch(model,batch,horizon,args)
    paths = sample_paths(model,safe['coords'][:,:4],mask,*[safe[k] for k in ('aatype','residue_index','chain_id')],
        safe['temperature_k'],horizon,args.n_samples,generator,args.grad_steps,
        observed_time_ps=safe['time_ps'][:,:4],checkpoint_decoder=getattr(args,'checkpoint_decoder',False))
    cached = _CachedSampleView(model,paths)
    rng_before = None if generator is None else generator.get_state().clone()
    terms = CADynamics.losses(cached,safe,horizon,args,generator)
    _require(cached.calls == 1, 'endpoint losses did not consume exactly one cached sample')
    if generator is not None:
        _require(torch.equal(generator.get_state(),rng_before), 'cached endpoint scoring consumed sampling RNG')
    with torch.no_grad():
        aligned = torch.stack([_align_to(safe['coords'][i,3:,:,1],safe['coords'][i,3,:,1],mask[i])
                               for i in range(len(mask))])
    scores = coordinate_path_scores(paths['ca_paths'],aligned,mask)
    _require(torch.equal(scores['endpoint'],terms['coord_es']), 'path endpoint scoring differs from frozen endpoint objective')
    if horizon == 1:
        # Preserve the exact old tensor graph and arithmetic at h1.
        joint = marginal = objective = terms['coord_es']
    else:
        joint,marginal = scores['joint'],scores['marginal']
        selected = joint if args.coord_objective == 'joint_path' else marginal
        objective = (1-args.path_weight)*terms['coord_es'] + args.path_weight*selected
        terms['loss'] = terms['loss'] + args.path_weight*(selected-terms['coord_es'])
    terms.update(coordinate_path_es=joint,coordinate_marginal_es=marginal,coordinate_objective=objective)
    _require(all(torch.isfinite(value).all() for value in terms.values()), 'nonfinite ordered path loss')
    return terms
