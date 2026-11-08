"""Matched absolute-distance and signed-distance-increment marginal path U-ES.

Both objectives use every fixed observed-valid physical nonlocal residue pair.
Each future time is one joint pair vector in Angstrom; member spread uses the
mean over unordered distinct member pairs. Scores divide by sqrt(pair_count),
then average equally over future times and windows. No pair/time weighting,
coordinate alignment, sampling, target scaling or kinetics interpretation is
introduced. Values are returned as computed, without a nonnegativity clamp.
"""
from __future__ import annotations

import math

import torch

VERSION = 'physical_distance_path_objective_v1'
STEP_PS = 1000.
TIME_ATOL_PS = 1e-6
INTEGER_DTYPES = (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _time_grid(value, batch, frames, device, name):
    _require(isinstance(value,torch.Tensor) and value.shape==(batch,frames)
        and value.is_floating_point() and value.device==device
        and bool(torch.isfinite(value).all()), name+' must be finite floating [B,T+1] on coordinate device')
    differences=value.double().diff(dim=1)
    _require(torch.allclose(differences,torch.full_like(differences,STEP_PS),rtol=0.,atol=TIME_ATOL_PS),
             name+' must advance uniformly by 1000 ps')


def _marginal_u_score(samples, reference):
    """[M,h,P], [h,P] -> [h], differentiable at coincident vectors via norm."""
    members,_,pairs=samples.shape
    i,j=torch.triu_indices(members,members,offset=1,device=samples.device)
    cross=torch.linalg.vector_norm(samples-reference[None],dim=-1).mean(0)
    spread=torch.linalg.vector_norm(samples[i]-samples[j],dim=-1).mean(0)
    return (cross-.5*spread)/math.sqrt(pairs)


def physical_distance_path_scores(predicted,reference,observed_anchor,residue_mask,*,
        residue_index,chain_id,predicted_times_ps,reference_times_ps,reference_mask):
    """Score paired absolute distances and signed increments on identical paths.

    Coordinates: predicted [B,M,h+1,L,3], reference [B,h+1,L,3],
    observed_anchor [B,L,3], all same floating dtype/device; M>=2, h>=1.
    Fixed observed residue_mask is bool [B,L]; physical IDs are integer [B,L].
    Both actual time arrays are floating [B,h+1], identical, uniform 1000 ps.
    reference_mask is bool [B,h+1,L] and must retain every observed-valid
    residue at every frame. Extra reference validity outside fixed support is
    ignored. Every predicted and reference t0 must equal the observed anchor.

    Return loss_absolute/loss_increment (scalar equal-window/time means),
    absolute_per_window_time/increment_per_window_time [B,h], pair_counts [B],
    pair_indices (tuple of [P_b,2] index tensors), and copied mask/time support.
    The absolute vector at t is d(t); the signed increment vector is
    d(t)-d(t-1). Both adjacent predicted frames remain in the graph. At h=1,
    values and parameter gradients coincide when t0 is the fixed observation;
    an independently variable predicted t0 is a different derivative problem.
    """
    _require(isinstance(predicted,torch.Tensor) and predicted.ndim==5
        and predicted.shape[-1]==3 and predicted.is_floating_point(),
        'predicted must be floating [B,M,h+1,L,3]')
    batch,members,frames,length,_=predicted.shape
    _require(batch>=1 and members>=2 and frames>=2 and length>=2,'requires B>=1, M>=2, h>=1, L>=2')
    for value,shape,name in ((reference,(batch,frames,length,3),'reference'),
                             (observed_anchor,(batch,length,3),'observed_anchor')):
        _require(isinstance(value,torch.Tensor) and value.shape==shape and value.dtype==predicted.dtype
            and value.device==predicted.device,name+' shape/dtype/device differs')
    _require(isinstance(residue_mask,torch.Tensor) and residue_mask.shape==(batch,length)
        and residue_mask.dtype==torch.bool and residue_mask.device==predicted.device,'invalid fixed observed mask')
    for value,name in ((residue_index,'residue_index'),(chain_id,'chain_id')):
        _require(isinstance(value,torch.Tensor) and value.shape==(batch,length)
            and value.dtype in INTEGER_DTYPES and value.device==predicted.device,name+' must be integer [B,L]')
    _require(isinstance(reference_mask,torch.Tensor) and reference_mask.shape==(batch,frames,length)
        and reference_mask.dtype==torch.bool and reference_mask.device==predicted.device,'invalid reference mask')
    fixed_reference_support=residue_mask[:,None].expand(batch,frames,length)
    _require(bool(reference_mask[fixed_reference_support].all()),'missing fixed observed support in reference')
    _time_grid(predicted_times_ps,batch,frames,predicted.device,'predicted_times_ps')
    _time_grid(reference_times_ps,batch,frames,predicted.device,'reference_times_ps')
    _require(torch.equal(predicted_times_ps.double(),reference_times_ps.double()),'actual time grids differ')
    _require(bool(torch.isfinite(predicted[residue_mask[:,None,None,:,None].expand_as(predicted)]).all())
        and bool(torch.isfinite(reference[residue_mask[:,None,:,None].expand_as(reference)]).all())
        and bool(torch.isfinite(observed_anchor[residue_mask]).all()),'nonfinite coordinates on fixed support')
    absolute,increment,support=[],[],[]
    upper_i,upper_j=torch.triu_indices(length,length,offset=1,device=predicted.device)
    for b in range(batch):
        mask=residue_mask[b]
        _require(torch.equal(reference[b,0,mask],observed_anchor[b,mask])
            and torch.equal(predicted[b,:,0,mask],observed_anchor[b,mask][None].expand(members,-1,-1)),
            'every path must start at the real observed anchor')
        # Promote IDs before subtraction to avoid uint8/int8 wraparound.
        ri,ci=residue_index[b].long(),chain_id[b].long()
        identities=torch.stack((ci[mask],ri[mask]),-1)
        _require(len(torch.unique(identities,dim=0))==int(mask.sum()),'duplicate physical residue identity')
        keep=mask[upper_i]&mask[upper_j]&((ci[upper_i]!=ci[upper_j])|((ri[upper_i]-ri[upper_j]).abs()>=3))
        i,j=upper_i[keep],upper_j[keep]
        _require(len(i)>0,'no valid physical nonlocal pairs')
        pairs=torch.stack((i,j),-1)
        # Only selected coordinates enter arithmetic: padded NaNs cannot leak.
        dp=torch.linalg.vector_norm(predicted[b,:,:,i]-predicted[b,:,:,j],dim=-1)
        dr=torch.linalg.vector_norm(reference[b,:,i]-reference[b,:,j],dim=-1)
        abs_score=_marginal_u_score(dp[:,1:],dr[1:])
        inc_score=_marginal_u_score(dp[:,1:]-dp[:,:-1],dr[1:]-dr[:-1])
        _require(bool(torch.isfinite(abs_score).all() and torch.isfinite(inc_score).all()),'nonfinite distance score')
        absolute.append(abs_score);increment.append(inc_score);support.append(pairs)
    absolute=torch.stack(absolute);increment=torch.stack(increment)
    return dict(loss_absolute=absolute.mean(),loss_increment=increment.mean(),
        absolute_per_window_time=absolute,increment_per_window_time=increment,
        pair_counts=torch.tensor([len(pairs) for pairs in support],device=predicted.device,dtype=torch.long),
        pair_indices=tuple(support),residue_mask=residue_mask.clone(),time_ps=predicted_times_ps.clone())
