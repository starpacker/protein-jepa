"""Analytic next-H Euler moments for anchor-fixed collective mobility.

M = sqrt(D) (K kron I_d) sqrt(D), b = h M(f-Cm), and
Sigma = 2h M + h² M C P C.T M. P is the FULL residue-major bath
covariance. Only MC and its product with P are formed, never an (Ld)²
matrix. The realized-S conditional mode has covariance 2hM; C_zero keeps
D, K, f and H unchanged. Gamma affects the subsequent S update, not H_next.

Channel centering is Q=I-11.T/d; tangent projection is Q-u u.T for
u=QH/||QH||. Undefined tangent/radial entries are NaN with an explicit mask;
raw/centered budgets retain every valid residue. No fixed radius, zero mean,
exact continuous-time dynamics, or forecast improvement is assumed.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor

PSD_ATOL=1e-10
PSD_RTOL=1e-10


def _need(ok,message):
    if not ok:raise ValueError(message)


def _tensor(value,name,shape):
    _need(isinstance(value,Tensor) and value.device.type=='cpu' and value.dtype==torch.float64,
          name+' must be a CPU float64 tensor')
    _need(tuple(value.shape)==tuple(shape),name+' shape differs')


def _psd(value,name):
    scale=max(1.,float(value.abs().max()))
    tolerance=PSD_ATOL+PSD_RTOL*scale
    _need(float((value-value.T).abs().max())<=tolerance,name+' is not symmetric')
    # No jitter, clipping, symmetrization, or fallback changes the input matrix.
    minimum=float(torch.linalg.eigvalsh(value).min())
    _need(minimum>=-tolerance,name+' is not positive semidefinite')
    return dict(min_eigenvalue=minimum,tolerance=tolerance)


def _budget(H,b,thermal,A,AP,tangent_valid,u):
    d=H.shape[1];z=H-H.mean(-1,keepdim=True);qb=b-b.mean(-1,keepdim=True)
    bath=(A*AP).sum((-2,-1))
    sum_a=A.sum(1);sum_ap=AP.sum(1)
    bath_mean_direction=(sum_a*sum_ap).sum(-1)/d
    thermal_raw=thermal.sum(-1)
    thermal_center=thermal_raw*(1.-1./d)
    bath_center=bath-bath_mean_direction
    p={}
    for prefix,x,drift,noise_t,noise_p in (
        ('raw',H,b,thermal_raw,bath),('centered',z,qb,thermal_center,bath_center)):
        p[prefix+'_current_norm2']=x.square().sum(-1)
        p[prefix+'_cross']=2*(x*drift).sum(-1)
        p[prefix+'_drift2']=drift.square().sum(-1)
        p[prefix+'_thermal_trace']=noise_t
        p[prefix+'_bath_trace']=noise_p
        p[prefix+'_expected_change']=p[prefix+'_cross']+p[prefix+'_drift2']+noise_t+noise_p
        p[prefix+'_expected_next_norm2']=(x+drift).square().sum(-1)+noise_t+noise_p
    mean_h=H.mean(-1);mean_b=b.mean(-1)
    mean_variance=(thermal_raw/d+bath_mean_direction)/d
    p.update(channel_mean_current=mean_h,channel_mean_increment=mean_b,
        channel_mean_noise_variance=mean_variance,
        channel_expected_change=d*(2*mean_h*mean_b+mean_b.square()+mean_variance),
        channel_increment_second_moment=d*(mean_b.square()+mean_variance))
    p['raw_centered_channel_closure']=p['raw_expected_change']-p['centered_expected_change']-p['channel_expected_change']
    radial_mean=(u*b).sum(-1);tangent_b=qb-u*radial_mean[:,None]
    radial_thermal=(u.square()*thermal).sum(-1)
    ua=torch.einsum('id,idr->ir',u,A);uap=torch.einsum('id,idr->ir',u,AP)
    radial_bath=(ua*uap).sum(-1)
    tangent=dict(tangent_drift2=tangent_b.square().sum(-1),
        tangent_thermal_trace=thermal_center-radial_thermal,tangent_bath_trace=bath_center-radial_bath,
        radial_mean_increment=radial_mean,radial_drift2=radial_mean.square(),
        radial_thermal_variance=radial_thermal,radial_bath_variance=radial_bath)
    tangent['radial_increment_second_moment']=tangent['radial_drift2']+radial_thermal+radial_bath
    tangent['tangent_increment_second_moment']=tangent['tangent_drift2']+tangent['tangent_thermal_trace']+tangent['tangent_bath_trace']
    tangent['increment_decomposition_closure']=(p['raw_drift2']+thermal_raw+bath
        -tangent['tangent_increment_second_moment']-tangent['radial_increment_second_moment']-p['channel_increment_second_moment'])
    tangent['centered_radial_closure']=(p['centered_expected_change']-2*z.norm(dim=-1)*radial_mean
        -tangent['radial_increment_second_moment']-tangent['tangent_increment_second_moment'])
    for key,value in tangent.items():p[key]=torch.where(tangent_valid,value,torch.full_like(value,float('nan')))
    for key,value in p.items():
        valid=tangent_valid if key in tangent else torch.ones_like(tangent_valid)
        _need(bool(torch.isfinite(value[valid]).all()),'nonfinite derived budget: '+key)
    summary={key:(float(value[tangent_valid].mean()) if bool(tangent_valid.any()) else None)
             if key in tangent else float(value.mean()) for key,value in p.items()}
    summary.update(n_residues=len(H),n_tangent_valid=int(tangent_valid.sum()))
    return dict(per_residue=p,summary=summary,mean_increment=b,next_H_mean=H+b)


@torch.no_grad()
def one_step_h_moments(H:Tensor,force:Tensor,diffusion:Tensor,coupling:Tensor,
                       spatial_covariance:Tensor,bath_mean:Tensor,bath_covariance:Tensor,*,
                       dt:float,residue_mask:Tensor|None=None,realized_bath:Tensor|None=None,
                       centered_norm_epsilon:float=1e-12):
    """Return posterior, C_zero and optional realized_S per-residue norm budgets.

    Inputs: H/f/D[L,d], C[L,d,s], K[L,L], m[L,s], P[L*s,L*s].
    Optional bool mask packs principal K and full P blocks in residue-major
    order; K must already equal the FULL spatial factor product before masking.
    P describes S conditional on the fixed H and history. New H noise is
    independent of that S posterior. Caller retains inputs to reconstruct the
    covariance operator; no full H covariance is emitted. All computations are
    CPU float64. A PSD tolerance accepts roundoff without changing any matrix.
    """
    _need(isinstance(H,Tensor) and H.ndim==2 and min(H.shape)>0,'H must be nonempty [L,d]')
    length,d=H.shape
    _need(isinstance(coupling,Tensor) and coupling.ndim==3 and coupling.shape[-1]>0,'C must be [L,d,s]')
    s=coupling.shape[-1]
    shapes={'H':(H,(length,d)),'force':(force,(length,d)),'diffusion':(diffusion,(length,d)),
        'coupling':(coupling,(length,d,s)),'spatial_covariance':(spatial_covariance,(length,length)),
        'bath_mean':(bath_mean,(length,s)),'bath_covariance':(bath_covariance,(length*s,length*s))}
    if realized_bath is not None:shapes['realized_bath']=(realized_bath,(length,s))
    for name,(value,shape) in shapes.items():_tensor(value,name,shape)
    _need(type(dt) in (float,int) and math.isfinite(dt) and dt>0,'dt must be finite and positive')
    _need(type(centered_norm_epsilon) in (float,int) and math.isfinite(centered_norm_epsilon)
          and centered_norm_epsilon>0,'centered norm epsilon must be finite and positive')
    if residue_mask is None:residue_mask=torch.ones(length,dtype=torch.bool)
    _need(isinstance(residue_mask,Tensor) and residue_mask.device.type=='cpu' and residue_mask.dtype==torch.bool
          and residue_mask.shape==(length,) and bool(residue_mask.any()),'mask must be nonempty CPU bool[L]')
    indices=residue_mask.nonzero().flatten();bath_indices=(indices[:,None]*s+torch.arange(s)[None,:]).flatten()
    H=H[indices];force=force[indices];D=diffusion[indices];C=coupling[indices];m=bath_mean[indices]
    K=spatial_covariance[indices][:,indices];P=bath_covariance[bath_indices][:,bath_indices]
    packed=[H,force,D,C,m,K,P]
    realized=None if realized_bath is None else realized_bath[indices]
    if realized is not None:packed.append(realized)
    _need(all(bool(torch.isfinite(x).all()) for x in packed),'nonfinite on fixed valid support')
    _need(bool((D>0).all()),'diffusion must be positive')
    k_audit=_psd(K,'K');p_audit=_psd(P,'P')
    root=D.sqrt()
    def mobility(value):return root*(K@(root*value))
    # A[i,a,j,b] = h sqrt(D_ia) K_ij sqrt(D_ja) C[j,a,b].
    A=(dt*root[:,:,None,None]*K[:,None,:,None]*(root[:,:,None]*C).permute(1,0,2)[None]).reshape(len(H),d,-1)
    AP=(A.reshape(-1,A.shape[-1])@P).reshape_as(A)
    thermal=2*dt*K.diag()[:,None]*D
    z=H-H.mean(-1,keepdim=True);norm=z.norm(dim=-1);tangent_valid=norm>centered_norm_epsilon
    u=z/norm.clamp_min(centered_norm_epsilon)[:,None]
    no_bath=A[:,:,:0];no_ap=AP[:,:,:0]
    outputs=dict(valid_indices=indices,tangent_valid=tangent_valid)
    outputs['posterior']=_budget(H,dt*mobility(force-torch.einsum('ids,is->id',C,m)),thermal,A,AP,tangent_valid,u)
    outputs['C_zero']=_budget(H,dt*mobility(force),thermal,no_bath,no_ap,tangent_valid,u)
    if realized is not None:
        outputs['realized_S']=_budget(H,dt*mobility(force-torch.einsum('ids,is->id',C,realized)),thermal,no_bath,no_ap,tangent_valid,u)
    outputs['audit']=dict(input_residues=length,valid_residues=len(H),latent_dim=d,bath_dim=s,dt=float(dt),
        centered_norm_epsilon=float(centered_norm_epsilon),K=k_audit,P=p_audit,
        largest_response_shape=list(A.shape),full_H_covariance_materialized=False,
        boundary='Frozen-H conditional Euler moments; Gamma only affects subsequent S. No fixed-radius, tangent-only or forecast claim.')
    return outputs
