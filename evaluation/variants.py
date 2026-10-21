"""Causal future producers. Raw H at encoder/predictor/decoder boundaries.

Only the dynamics kernel sees normalized H. No true future state is accepted by
produce(); future states are encoded separately by the runner for oracle controls.
"""
from __future__ import annotations
import math
import torch
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from .geometry import ca_from_backbone, displacement_msd, calibrate_observed_anchor_pose


def encode_clip(bundle, clip):
    return bundle.encode(clip['coords'], clip['residue_mask'], clip['aatype'],
                         clip['residue_index'], clip['chain_id'], normalize=False)


def anchor_frames(ctx):
    bb = ctx['coords'][:, -1]
    return RigidBodyFrame.from_backbone(bb[...,0,:], bb[...,1,:], bb[...,2,:])


def decode(bundle, h_raw, h_anchor_raw, ctx, R_anchor, t_anchor, rmask):
    if bundle.decoder_kind == 'oracle':
        # Canonical-shape decoders have no supervised global pose. Calibrate it
        # once from decoded CURRENT H to the real observed anchor, never target.
        observed_mask = ctx['residue_mask'].all(dim=1).bool()
        kwargs = dict(aatype=ctx['aatype'],R_anchor=R_anchor,t_anchor=t_anchor,
                      residue_mask=observed_mask)
        predicted = bundle.decoder.decode_structure(H=h_raw,**kwargs)['CA']
        current = bundle.decoder.decode_structure(H=h_anchor_raw,**kwargs)['CA']
        observed = ca_from_backbone(ctx['coords'][:,-1])
        return calibrate_observed_anchor_pose(predicted,current,observed,observed_mask)
    if bundle.decoder_kind == 'residual' and bundle.diagnostic_only:
        anchor = ca_from_backbone(ctx['coords'][:, -1])
        return bundle.decoder(h_raw, h_anchor_raw, anchor, rmask,R_anchor=R_anchor)['ca_pred']
    if bundle.decoder_kind == 'identity':
        return ca_from_backbone(ctx['coords'][:, -1])
    raise ValueError('headline evaluation requires the shared raw-H oracle decoder')


def history_intervals(ctx):
    times = ctx['time_ps']
    if times.shape[-1] < 2:
        raise ValueError('history requires >=2 observations to infer physical dt')
    dt = times.diff(dim=-1)
    if not torch.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError('invalid history physical timestamps')
    return torch.cat([dt[:, :1], dt], dim=-1)


def dynamics_parameters(bundle, h_hist, ctx, mask):
    """Observed-anchor parameter diagnostics; never alter the learned kernel."""
    model = bundle.b1b
    D,C,Gamma = model.bath(h_hist[:,-1],ctx.get('temperature_k'))
    h = model.cfg.step_ps
    good = D[mask]
    stats = dict(diffusion_min=float(good.min()),diffusion_mean=float(good.mean()),
                 diffusion_max=float(good.max()),H_noise_variance_mean=float((2*h*good).mean()),
                 history_normalized_rms=float(h_hist[:,:,mask[0]].square().mean().sqrt()))
    if C is not None:
        gamma = h*Gamma[mask]
        stats.update(coupling_frobenius_mean=float(C[mask].square().sum((-1,-2)).sqrt().mean()),
                     gamma_step_min=float(gamma.min()),gamma_step_max=float(gamma.max()),
                     isolated_bath_euler_abs_max=float((1-gamma).abs().max()))
    return stats


@torch.no_grad()
def produce(variant, bundle, ctx, H_ctx, horizon, *, horizon_ps=None,
            seed=0, n_rollout_reps=1):
    rmask = ctx['residue_mask'].all(dim=1).bool()
    anchor = ca_from_backbone(ctx['coords'][:, -1])
    if variant == 'persist':
        return {'ca_pred':anchor[0], 'ca_samples':anchor[0,None], 'H_future':None,
                'meta':{'msd':0., 'n_reps':1}}
    R, t = anchor_frames(ctx)
    physical_gap = float(horizon_ps if horizon_ps is not None else horizon*1000.)
    if variant in {'h_only','h_ft','direct_jepa'}:
        lag = H_ctx.new_full((H_ctx.shape[0],), physical_gap)
        temp = ctx.get('temperature_k', H_ctx.new_full((H_ctx.shape[0],),348.))
        temp = torch.as_tensor(temp, device=H_ctx.device, dtype=H_ctx.dtype).reshape(-1)
        direct_history = H_ctx
        if bundle.provenance.get('training_variant') == 'direct_last':
            direct_history = H_ctx[:,-1:].expand_as(H_ctx)
        h_future = bundle.predict_future(direct_history, lag, temp, rmask)
        ca = decode(bundle, h_future, H_ctx[:,-1], ctx, R,t,rmask)
        return {'ca_pred':ca[0], 'ca_samples':ca[0,None], 'H_future':h_future[0],
                'meta':{'msd':displacement_msd(ca[0],anchor[0],rmask[0]),'n_reps':1}}
    if variant in {'h_markov','h_plus_s'}:
        if bundle.b1b is None or n_rollout_reps < 1:
            raise ValueError('h_plus_s needs a trained kernel and positive rollout reps')
        dt = history_intervals(ctx)
        cfg = bundle.b1b.cfg
        if not torch.allclose(dt, torch.full_like(dt,cfg.step_ps),rtol=1e-5,atol=1e-3):
            raise ValueError('history dt differs from kernel.step_ps')
        steps = physical_gap/cfg.step_ps
        if not math.isclose(steps,round(steps),rel_tol=1e-6,abs_tol=1e-6) or steps < 1:
            raise ValueError('requested horizon must be an integer number of physical kernel steps')
        n_steps = round(steps)
        # Normalization happens exactly once, and inverse exactly once.
        h_hist = bundle.normalize(H_ctx)
        if bundle.provenance.get('training_variant') == 'h_plus_s_no_history':
            h_hist = h_hist[:,-1:].expand_as(h_hist)
        reps, latent_msds, latent_rms, latent_max = [], [], [], []
        parameter_stats = dynamics_parameters(bundle,h_hist,ctx,rmask) if hasattr(bundle.b1b,'bath') else {}
        for rep in range(n_rollout_reps):
            gen = torch.Generator(device=H_ctx.device).manual_seed(seed+rep)
            traj = bundle.b1b.rollout(h_hist[:,-1], rmask,ctx.get('temperature_k'),
                    n_steps=n_steps,H_hist=h_hist,dt_hist=dt,generator=gen,detach=True)
            if not torch.isfinite(traj).all():
                raise FloatingPointError('nonfinite autonomous rollout; no sample may be silently omitted')
            h_future = bundle.inverse(traj[:,-1])
            ca = decode(bundle,h_future,H_ctx[:,-1],ctx,R,t,rmask)[0]
            reps.append(ca)
            delta = traj[:,-1]-h_hist[:,-1]
            latent_msds.append(float(delta[0,rmask[0]].square().mean()))
            latent_rms.append(float(traj[0,:,rmask[0]].square().mean().sqrt()))
            latent_max.append(float(traj[0,:,rmask[0]].abs().max()))
        samples = torch.stack(reps)
        ensemble = samples[0]+(samples-samples[0]).mean(0)
        return {'ca_pred':ensemble,'ca_samples':samples,'H_future':None,
                'meta':{'n_reps':n_rollout_reps,'latent_msd_reps':latent_msds,
                        'dynamics_stats':{**parameter_stats,'rollout_normalized_rms':sum(latent_rms)/len(latent_rms),
                                          'rollout_normalized_abs_max':max(latent_max)},
                        'msd':displacement_msd(ensemble,anchor[0],rmask[0])}}
    raise ValueError(f'unknown variant: {variant}')
