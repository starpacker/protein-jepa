"""Bounded target-informed local-frame oracle, with fixed CA and PSF support.

This rotates an existing atom family; it does not re-decode OpenFold atoms.
Degenerate predicted torsions retain their fixed target-valid denominators.
The strict 1e-10 validity threshold is diagnostic only for predictions: a tiny
nonzero invalid quartet can have continuous loss slightly below 2. Exact zero
confidence gives 2; no prediction-dependent validity gate changes the loss.
The confidence objective is finite but has no guarantee of escaping collapse
or the flat exactly-antipodal confidence direction.
"""
from __future__ import annotations

import math
import torch

from .backbone_objectives_v3 import backbone_geometry_losses_v3

VERSION = 'joint_frame_oracle_v1'
TORSION_EPS_ANGSTROM = 1e-6
VALIDITY_EPS_ANGSTROM = 1e-10
MAX_ROTATION_RADIANS = math.pi / 6
STEPS = 200
LEARNING_RATE = .02
WEIGHTS = dict(c_n=1., ca_c_n=4., c_n_ca=4., backbone_interresidue_clash=100.)


def require(value, message):
    if not value:
        raise ValueError(message)


def _quartets(points):
    require(isinstance(points, torch.Tensor) and points.shape[-2:] == (4,3)
            and points.dtype == torch.float64 and bool(torch.isfinite(points).all()),
            'torsion quartets must be finite float64 [...,4,3]')
    a,b,c,d = points.unbind(-2)
    return c-b, a-b, d-c


def dihedral_target_units(points):
    """Frozen signed-dihedral convention and fixed 1e-10 geometric validity."""
    central,left,right = _quartets(points)
    length = torch.linalg.vector_norm(central,dim=-1)
    q = central / length.clamp_min(VALIDITY_EPS_ANGSTROM)[...,None]
    v = left - (left*q).sum(-1,keepdim=True)*q
    w = right - (right*q).sum(-1,keepdim=True)*q
    vn,wn = torch.linalg.vector_norm(v,dim=-1),torch.linalg.vector_norm(w,dim=-1)
    valid = (length>VALIDITY_EPS_ANGSTROM)&(vn>VALIDITY_EPS_ANGSTROM)&(wn>VALIDITY_EPS_ANGSTROM)
    v = v/vn.clamp_min(VALIDITY_EPS_ANGSTROM)[...,None]
    w = w/wn.clamp_min(VALIDITY_EPS_ANGSTROM)[...,None]
    components = torch.stack(((torch.cross(q,v,dim=-1)*w).sum(-1),(v*w).sum(-1)),-1)
    norm = torch.linalg.vector_norm(components,dim=-1)
    unit = components/norm.clamp_min(VALIDITY_EPS_ANGSTROM)[...,None]
    return unit,valid


def confidence_dihedral_components(points):
    """Return bounded (sin,cos) confidence vector without atan2 singularity.

    q=u/max(|u|,eps); n1=q cross(a-b), n2=q cross(d-c). Normalize each plane
    by max(|n|,eps), and multiply both components by rho=|u|/max(|u|,eps).
    Exact unit-axis/plane normalization is recovered above eps. A zero central
    axis or either zero plane gives c=0; collinear projections cannot masquerade
    as a valid cosine. r=|c|<=1 in exact arithmetic (roundoff is not clamped).
    """
    central,left,right = _quartets(points)
    length = torch.linalg.vector_norm(central,dim=-1)
    denominator = length.clamp_min(TORSION_EPS_ANGSTROM)
    q = central/denominator[...,None]
    rho = length/denominator
    n1,n2 = torch.cross(q,left,dim=-1),torch.cross(q,right,dim=-1)
    n1 = n1/torch.linalg.vector_norm(n1,dim=-1).clamp_min(TORSION_EPS_ANGSTROM)[...,None]
    n2 = n2/torch.linalg.vector_norm(n2,dim=-1).clamp_min(TORSION_EPS_ANGSTROM)[...,None]
    c = rho[...,None]*torch.stack(((torch.cross(q,n1,dim=-1)*n2).sum(-1),(n1*n2).sum(-1)),-1)
    return c,torch.linalg.vector_norm(c,dim=-1)


def confidence_circular_loss(points,target_unit):
    """Continuous 2-r-c·target; fixed membership, no strict-invalid hard gate."""
    c,r = confidence_dihedral_components(points)
    require(target_unit.shape == c.shape and target_unit.dtype == c.dtype
            and target_unit.device == c.device and bool(torch.isfinite(target_unit).all()),
            'target unit shape/dtype/device differs')
    require(bool(torch.allclose(torch.linalg.vector_norm(target_unit,dim=-1),
                               torch.ones_like(r),atol=1e-12,rtol=1e-12)), 'target must be unit sin/cos')
    return 2-r-(c*target_unit).sum(-1)


def rotation_from_vectors(vectors):
    require(vectors.dtype == torch.float64 and vectors.shape[-1] == 3
            and bool(torch.isfinite(vectors).all()), 'rotation vectors must be finite float64 [...,3]')
    x,y,z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*vectors.shape[:-1],3,3)
    return torch.matrix_exp(skew)


def rotate_atom_family(initial_backbone,vectors):
    """X0+(deltaR-I)(X0-CA), with CA copied explicitly and zero exact parity."""
    require(initial_backbone.shape[:-2] == vectors.shape[:-1]
            and initial_backbone.shape[-2:] == (4,3), 'atom/rotation shapes differ')
    rotation = rotation_from_vectors(vectors)
    ca = initial_backbone[...,1,:]
    local = initial_backbone-ca[...,None,:]
    delta = rotation-torch.eye(3,dtype=rotation.dtype,device=rotation.device)
    moved = initial_backbone+torch.einsum('...ij,...aj->...ai',delta,local)
    moved = torch.stack((moved[...,0,:],ca,moved[...,2,:],moved[...,3,:]),-2)
    return moved,rotation


def _edge_quartets(backbone,edges):
    i,j=edges.unbind(-1)
    phi=torch.stack((backbone[:,i,2],backbone[:,j,0],backbone[:,j,1],backbone[:,j,2]),-2)
    psi=torch.stack((backbone[:,i,0],backbone[:,i,1],backbone[:,i,2],backbone[:,j,0]),-2)
    return torch.cat((phi,psi),dim=1)


def _fixed_support(initial,target,residue_mask,atom_mask,edges):
    atoms=atom_mask & residue_mask[:,None]
    nco=atoms.clone();nco[:,1]=False
    require(bool(nco.any()), 'no fixed N/C/O labels')
    require(bool(torch.isfinite(initial[:,atoms]).all()) and bool(torch.isfinite(target[:,atoms]).all()),
            'nonfinite coordinate on fixed observed/target atom support')
    # CA is the fixed rotation center even when that residue has no target atom.
    require(bool(torch.isfinite(initial[:,:,1]).all()), 'all fixed CA centers must be finite')
    safe_target=torch.where(atoms[None,:,:,None],target,0.)
    target_quartets=_edge_quartets(safe_target,edges)
    units,valid=dihedral_target_units(target_quartets)
    i,j=edges.unbind(-1)
    phi=atoms[i,2]&atoms[j,0]&atoms[j,1]&atoms[j,2]
    psi=atoms[i,0]&atoms[i,1]&atoms[i,2]&atoms[j,0]
    candidates=torch.cat((phi,psi))[None]
    fixed=candidates & valid
    return dict(atoms=atoms,nco=nco,target_units=units.detach(),torsion_mask=fixed.detach(),
        torsion_atom_candidates=candidates, target_torsion_geometric_valid=valid.detach(),
        phi_count=int(fixed[:,:len(edges)].sum()),psi_count=int(fixed[:,len(edges):].sum()),
        nco_atom_count=int(nco.sum()))


def _objective(backbone,target,support,chemical_aatype,edges,ids,residue_mask):
    scored=support['nco']
    residual=torch.where(scored[None,:,:,None],backbone-target,0.)
    nco=residual.square().sum()/support['nco_atom_count']
    quartets=_edge_quartets(backbone,edges)
    keep=support['torsion_mask']
    if bool(keep.any()):
        torsion=confidence_circular_loss(quartets[keep],support['target_units'][keep]).mean()
    else:
        torsion=backbone.sum()*0.
    _,valid=dihedral_target_units(quartets)
    geometry=backbone_geometry_losses_v3(backbone,chemical_aatype=chemical_aatype,
        atom_mask=support['atoms'],residue_mask=residue_mask,peptide_edges=edges,geometry_row_ids=ids)
    terms=dict(nco_mse_angstrom2=nco,torsion_circular=torsion,**geometry['losses'])
    total=nco+torsion+sum(WEIGHTS[k]*terms[k] for k in WEIGHTS)
    terms['total']=total
    require(all(bool(torch.isfinite(value)) for value in terms.values()), 'nonfinite oracle objective')
    return terms,dict(predicted_torsion_invalid_on_fixed_support=int((keep & ~valid).sum()),
        torsion_count=int(keep.sum()), geometry_counts={k:v.detach().clone() for k,v in geometry['counts'].items()})


class JointFrameOptimizationError(RuntimeError):
    def __init__(self,message,partial):
        super().__init__(message)
        self.partial=partial


def optimize_joint_frames(initial_backbone,target_backbone,*,residue_mask,atom_mask,
        chemical_aatype,geometry_row_ids,peptide_edges,initial_R,psi_logits):
    """Fixed200 Adam(.02,wd0), clip1, projected30-degree rotations; final only.

    The caller supplies fixed observed/target intersection masks. Missing head
    atoms must fail before calling, never shrink this support. Inputs and witnesses
    are detached/cloned; only the zero-initialized rotation3-vectors are optimized.
    On optimization failure, exception.partial retains the last attempted state
    and completed trace. No target-informed output is a forecast or fitted head.
    """
    require(initial_backbone.ndim==4 and initial_backbone.shape[0]==1
            and initial_backbone.shape[-2:]==(4,3) and initial_backbone.dtype==torch.float64,
            'initial backbone must be float64 [1,L,4,3]')
    require(target_backbone.shape==initial_backbone.shape and target_backbone.dtype==torch.float64
            and target_backbone.device==initial_backbone.device, 'target shape/dtype/device differs')
    length=initial_backbone.shape[1];device=initial_backbone.device
    require(residue_mask.shape==(length,) and residue_mask.dtype==torch.bool
            and atom_mask.shape==(length,4) and atom_mask.dtype==torch.bool
            and residue_mask.device==device and atom_mask.device==device, 'fixed mask shape/dtype/device differs')
    require(initial_R.shape==(1,length,3,3) and initial_R.dtype==torch.float64
            and initial_R.device==device and bool(torch.isfinite(initial_R).all()), 'invalid initial frame witness')
    eye=torch.eye(3,dtype=torch.float64,device=device)
    require(torch.allclose(initial_R.transpose(-2,-1)@initial_R,eye.expand_as(initial_R),atol=1e-5,rtol=0)
            and bool((torch.linalg.det(initial_R)>0).all()), 'initial frames must be proper rotations')
    require(psi_logits.ndim==3 and psi_logits.shape[:2]==(1,length)
            and bool(torch.isfinite(psi_logits).all()), 'invalid psi-logit witness')
    initial=initial_backbone.detach().clone();target=target_backbone.detach().clone()
    r0=initial_R.detach().clone();psi=psi_logits.detach().clone()
    residue=residue_mask.detach().clone();atoms=atom_mask.detach().clone()
    # Frozen v3 performs integer-ID/edge and chemistry validation without fitting.
    backbone_geometry_losses_v3(initial,chemical_aatype=chemical_aatype,atom_mask=atoms,
        residue_mask=residue,peptide_edges=peptide_edges,geometry_row_ids=geometry_row_ids)
    edges=peptide_edges.detach().clone().long();ids=geometry_row_ids.detach().clone().long()
    chemical=chemical_aatype.detach().clone()
    fixed=_fixed_support(initial,target,residue,atoms,edges)
    # Zero only masked non-CA atoms for arithmetic; output restores their bytes.
    arithmetic=torch.where(fixed['atoms'][None,:,:,None],initial,initial[:,:,1:2].expand_as(initial))
    arithmetic[:,:,1]=initial[:,:,1]
    safe_target=torch.where(fixed['atoms'][None,:,:,None],target,0.)
    vectors=torch.zeros((1,length,3),dtype=torch.float64,device=device,requires_grad=True)
    optimizer=torch.optim.Adam([vectors],lr=LEARNING_RATE,weight_decay=0.)
    trace=[];last={}
    def witness(status):
        with torch.no_grad():
            moved,delta=rotate_atom_family(arithmetic,vectors)
            moved=torch.where(fixed['atoms'][None,:,:,None],moved,initial)
            moved[:,:,1]=initial[:,:,1]
        return dict(schema_version=VERSION,status=status,initial_backbone=initial.clone(),
            final_backbone=moved.detach().clone(),rotation_vectors=vectors.detach().clone(),
            delta_rotation=delta.detach().clone(),initial_R=r0.clone(),final_R=(delta@r0).detach(),
            psi_logits=psi.clone(),trace=list(trace),final_losses=dict(last),fixed_support={
                k:v.detach().clone() if isinstance(v,torch.Tensor) else v for k,v in fixed.items()},
            conventions=dict(steps=STEPS,optimizer='Adam',learning_rate=LEARNING_RATE,weight_decay=0.,
                gradient_clip_norm=1.,maximum_rotation_radians=MAX_ROTATION_RADIANS,
                torsion_eps_angstrom=TORSION_EPS_ANGSTROM,validity_eps_angstrom=VALIDITY_EPS_ANGSTROM,
                nco='mean squared Euclidean error per fixed N/C/O atom / 1 angstrom^2',
                torsion='mean over fixed target-valid phi/psi quartets; 2-r-dot(c,target_unit)',
                predicted_validity='1e-10 diagnostic only; nonzero invalid confidence may give loss below 2; no hard gate',
                degeneracy_limit='zero confidence loss 2; collapse and exact antipodes can have zero gradient; no escape guarantee',
                geometry_weights=WEIGHTS,selection='final iteration only; no best iterate',
                reconstruction='local rotation of stored atom family; not fresh OpenFold decode'))
    try:
        for step in range(STEPS+1):
            moved,delta=rotate_atom_family(arithmetic,vectors)
            terms,audit=_objective(moved,safe_target,fixed,chemical,edges,ids,residue)
            last={k:float(v.detach()) for k,v in terms.items()}
            trace.append(dict(step=step,**last,**audit,gradient_norm_before_clip=None,
                maximum_rotation_radians=float(torch.linalg.vector_norm(vectors.detach(),dim=-1).max())))
            require(torch.equal(moved[:,:,1],initial[:,:,1]), 'fixed CA changed')
            if step==STEPS:break
            optimizer.zero_grad(set_to_none=True)
            terms['total'].backward()
            require(vectors.grad is not None and bool(torch.isfinite(vectors.grad).all()), 'nonfinite rotation gradient')
            norm=torch.nn.utils.clip_grad_norm_([vectors],1.)
            require(bool(torch.isfinite(norm)), 'nonfinite clipped gradient norm')
            trace[-1]['gradient_norm_before_clip']=float(norm)
            optimizer.step()
            require(bool(torch.isfinite(vectors).all()), 'nonfinite optimizer rotation')
            with torch.no_grad():
                norms=torch.linalg.vector_norm(vectors,dim=-1,keepdim=True)
                vectors.mul_((MAX_ROTATION_RADIANS/norms.clamp_min(1e-30)).clamp_max(1.))
                vectors[:,~residue]=0.
        result=witness('completed')
        require(torch.allclose(result['delta_rotation'].transpose(-2,-1)@result['delta_rotation'],
                               eye.expand_as(result['delta_rotation']),atol=1e-12,rtol=0)
                and bool((torch.linalg.det(result['delta_rotation'])>0).all()), 'invalid proper rotation witness')
        return result
    except Exception as error:
        try:
            partial=witness('failed')
        except Exception:
            partial=dict(schema_version=VERSION,status='failed',initial_backbone=initial,
                rotation_vectors=vectors.detach().clone(),initial_R=r0,psi_logits=psi,trace=list(trace),final_losses=dict(last))
        raise JointFrameOptimizationError(f'{type(error).__name__}: {error}',partial) from error
