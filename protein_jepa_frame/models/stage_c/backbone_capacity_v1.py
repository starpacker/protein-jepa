"""Label-oracle bounds for independent fixed-CA N/C/O kinematic fitting.

This diagnostic is never an inference readout. It consumes target coordinates
to fit a separate proper rotation and psi for EACH residue, without shared
neural weights or inter-residue constraints. Its witness does not establish
that an H-conditioned neural network can produce those parameters.

For fixed psi, translation-free proper SO(3) Procrustes solves N/C/O SSE
globally. A midpoint grid in the registered psi domain supplies a constructive
upper bound. O moves on a circle of radius rho around the CA--C axis and its
distance to CA is constant. Consequently, for any fixed rotation,

  |E(psi)-E(phi)| <= 4*rho*||target_O-fixed_CA||*sin(|psi-phi|/2).

The same modulus holds for F(psi)=min_R E(R,psi), by comparing the same R at
both angles. For K midpoint cells across width W, every angle is within W/2K
of a grid point. Thus grid_min - 4*rho*||y_O||*sin(W/4K) is a GLOBAL lower
bound, never a claim that the grid/local minimum itself is a lower bound.
An independent closed-form bound is exact NC Procrustes SSE plus O's radial
relaxation (||y_O||-||template_O||)^2. We take the maximum of the two and zero.

The default psi domain is the closure of the current arithmetic softmax-bin
mean, using its float32 bin centers (36 bins: about -175 to +175 degrees).
Grid witnesses are strict interior points. A full-circle option is explicitly
a relaxation of that head's psi domain. These are ideal float64 mathematical
kinematic bounds with a stated floating-point engineering guard, NOT formal
interval-arithmetic certificates or bounds on every numerical quirk of the
float32 decoder/quaternion implementation. Actual OpenFold reconstruction of
the witness is separately returned and scored. No local numerical optimizer,
data-dependent threshold, training, H encoder or forecast input is involved.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math

import numpy as np
import torch

from protein_jepa_frame.vendored.openfold.np import residue_constants as rc


SCHEMA_VERSION = 'fixed_ca_nco_kinematic_capacity_v1'
ATOM_ORDER = ('N','CA','C','O')
NCO_INDICES = (0,2,3)
FLOAT64_GUARD_MULTIPLIER = 128.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _metadata(value, shape, name, *, boolean=False):
    tensor = torch.as_tensor(value,device='cpu')
    if boolean:
        _require(tensor.dtype == torch.bool, f'{name} must be boolean')
    else:
        _require(tensor.dtype != torch.bool and not tensor.is_floating_point() and not tensor.is_complex(),
                 f'{name} must contain integer chemical IDs')
        tensor = tensor.long()
    try:
        return torch.broadcast_to(tensor,shape).clone()
    except RuntimeError as error:
        raise ValueError(f'{name} cannot broadcast to {shape}') from error


@lru_cache(maxsize=1)
def _templates():
    """Widen the actual pinned float32 placement constants, never retype IDs."""
    atoms = rc.restype_atom37_rigid_group_positions
    ids = [rc.atom_order[name] for name in ('N','C','O')]
    local = np.asarray(atoms[:20,ids],dtype=np.float32)
    frames = np.asarray(rc.restype_rigid_group_default_frame[:20,3],dtype=np.float32)
    _require(np.array_equal(frames[:,:3,:3],np.broadcast_to(np.diag([1.,-1.,-1.]),(20,3,3))),
             'pinned psi frame is no longer the canonical backbone circle')
    _require(np.array_equal(local[:,1,1:],np.zeros((20,2))) and np.array_equal(local[:,0,2],np.zeros(20))
             and np.array_equal(local[:,2,2],np.zeros(20)), 'noncanonical pinned backbone template')
    _require(np.array_equal(frames[:,:3,3],local[:,1]), 'psi translation differs from carbonyl C')
    _require(np.array_equal(rc.restype_atom37_to_rigid_group[:20,ids],np.broadcast_to([0,0,3],(20,3))),
             'N/C/O rigid groups differ from the registered family')
    # Unknown chemistry retains identity 20 externally and uses exactly the
    # existing OpenFold adapter's GLY backbone template, not a guessed type.
    table = np.concatenate([local,local[rc.restype_order['G']:rc.restype_order['G']+1]],axis=0)
    table = torch.from_numpy(table.copy()).double()
    nc = table[:,:2]
    axial = table[:,1,0]+table[:,2,0]
    radius = table[:,2,1]
    _require(bool((radius>0).all()),'nonpositive oxygen circle radius')
    digest = hashlib.sha256(local.tobytes()+frames.tobytes()).hexdigest()
    return nc,axial,radius,digest


def psi_domain_bounds(torsion_bins=36, psi_domain='head_linear_bins'):
    """Return the current float32 arithmetic-mean range, or named relaxation."""
    _require(type(torsion_bins) is int and torsion_bins>=2,'torsion_bins must be an integer >=2')
    _require(psi_domain in ('head_linear_bins','full_circle_relaxation'),'unknown psi domain')
    if psi_domain == 'full_circle_relaxation':
        return -math.pi,math.pi
    edges = torch.linspace(-math.pi,math.pi,torsion_bins+1,dtype=torch.float32)
    centers = (edges[:-1]+edges[1:])/2
    return float(centers[0]),float(centers[-1])


def ideal_nco_template(chemical_aatype, psi):
    """Independent ideal template in OPENFOLD frame, with broadcast IDs/angles."""
    aa = torch.as_tensor(chemical_aatype,device='cpu')
    _require(aa.dtype!=torch.bool and not aa.is_floating_point() and not aa.is_complex(),'chemical IDs must be integers')
    aa,angle = torch.broadcast_tensors(aa.long(),torch.as_tensor(psi,device='cpu',dtype=torch.float64))
    _require(bool(((aa>=0)&(aa<=20)).all()) and bool(torch.isfinite(angle).all()),'invalid chemical ID/psi')
    nc,axial,radius,_ = _templates()
    oxygen = torch.stack((axial[aa],-radius[aa]*angle.cos(),-radius[aa]*angle.sin()),dim=-1)
    return torch.cat((nc[aa],oxygen.unsqueeze(-2)),dim=-2)


def _proper_fit(template, target, weights):
    """Global no-translation SO(3) fit; row vectors transform as template @ Q."""
    covariance = (template*weights[...,None]).transpose(-1,-2) @ target
    left,_,right_t = torch.linalg.svd(covariance,full_matrices=False)
    sign = torch.where(torch.linalg.det(left@right_t)<0,-1.,1.)
    diagonal = torch.ones_like(left[...,0,:]);diagonal[...,-1]=sign
    q = (left*diagonal[...,None,:]) @ right_t
    residual = template@q-target
    sse = (residual.square().sum(-1)*weights).sum(-1)
    return q,sse


@torch.no_grad()
def fixed_ca_nco_capacity(
    target_backbone, fixed_ca, *, chemical_aatype, atom_mask,
    residue_mask=None, torsion_bins=36, grid_size=2048, grid_chunk_size=128,
    psi_domain='head_linear_bins',
):
    """Return CPU float64 ideal brackets and an actual OpenFold witness.

    Shapes: target [...,L,4,3], fixed_ca [...,L,3]; metadata broadcasts to
    [...,L] / [...,L,4]. Coordinates must already share the OBSERVED anchor
    gauge. No further target/prediction alignment or translation is performed.
    atom_mask is the caller's explicit common scoring mask. CA never enters the
    NCO objective. Metadata IDs remain chemical IDs, with 20=unknown.

    Per-sample MSE divides summed per-residue SSE by the actual NCO atom count;
    RMSD is its square root. Empty support returns zero PLUS count zero: a
    report must convert that sample to null, not call it perfect reconstruction.
    Arrays under witness are label-oracle diagnostic products, not model input.
    """
    _require(type(grid_size) is int and grid_size>=2,'grid_size must be an integer >=2')
    _require(type(grid_chunk_size) is int and grid_chunk_size>=1,'grid_chunk_size must be positive')
    low,high = psi_domain_bounds(torsion_bins,psi_domain)
    target = torch.as_tensor(target_backbone)
    ca = torch.as_tensor(fixed_ca)
    _require(target.ndim>=3 and target.shape[-2:]==(4,3) and target.numel()>0,'target must be nonempty [...,L,4,3]')
    _require(ca.shape==target.shape[:-2]+(3,) and target.is_floating_point() and ca.is_floating_point(),
             'fixed_ca shape/dtype differs')
    _require(target.device.type==ca.device.type=='cpu','capacity diagnostics require explicit CPU inputs')
    shape = target.shape[:-2];leading=shape[:-1];length=shape[-1]
    residue = torch.ones(shape,dtype=torch.bool) if residue_mask is None else _metadata(residue_mask,shape,'residue_mask',boolean=True)
    masks = _metadata(atom_mask,shape+(4,),'atom_mask',boolean=True)&residue[...,None]
    aa = _metadata(chemical_aatype,shape,'chemical_aatype')
    active = masks.any(-1)
    _require(bool((((aa>=0)&(aa<=20))|~active).all()),'valid chemical IDs must be 0..20')
    _require(bool(torch.isfinite(target[masks]).all()) and bool(torch.isfinite(ca[active]).all()),
             'nonfinite valid target/fixed CA')
    target = torch.where(masks[...,None],target.double(),0.).reshape(-1,4,3)
    ca = torch.where(active[...,None],ca.double(),0.).reshape(-1,3)
    masks = masks.reshape(-1,4);active=active.reshape(-1)
    aa = torch.where(active,aa.reshape(-1),20)
    weights = masks[:,NCO_INDICES].double()
    y = torch.where(weights[...,None].bool(),target[:,NCO_INDICES]-ca[:,None],0.)
    counts = weights.sum(-1).long()
    nc,axial,radius,template_hash = _templates()
    _,nc_sse = _proper_fit(nc[aa],y[:,:2],weights[:,:2])
    o_radius = torch.sqrt(axial[aa].square()+radius[aa].square())
    target_o_radius = torch.linalg.vector_norm(y[:,2],dim=-1)
    radial_o_sse = (o_radius-target_o_radius).square()*weights[:,2]
    best = torch.full_like(radial_o_sse,float('inf'))
    best_q = torch.eye(3,dtype=torch.float64).expand(len(aa),3,3).clone()
    best_psi = torch.zeros(len(aa),dtype=torch.float64)
    grid = low+(torch.arange(grid_size,dtype=torch.float64)+.5)*(high-low)/grid_size
    for start in range(0,grid_size,grid_chunk_size):
        angles = grid[start:start+grid_chunk_size]
        templates = ideal_nco_template(aa[None],angles[:,None])
        q,sse = _proper_fit(templates,y[None],weights[None])
        values,indices = sse.min(dim=0)
        choose = values<best
        residues = torch.arange(len(aa))
        best = torch.where(choose,values,best)
        best_q = torch.where(choose[:,None,None],q[indices,residues],best_q)
        best_psi = torch.where(choose,angles[indices],best_psi)
    # Constant-radius chord modulus applies to every rotation, hence also its
    # exact Procrustes minimum. A finite grid is an upper, never a lower bound.
    gap_bound = 4*radius[aa]*target_o_radius*math.sin((high-low)/(4*grid_size))*weights[:,2]
    template_norm_sq = torch.cat((nc[aa].square().sum(-1),o_radius[:,None].square()),dim=-1)
    guard = FLOAT64_GUARD_MULTIPLIER*torch.finfo(torch.float64).eps*(
        (y.square().sum(-1)*weights).sum(-1)+(template_norm_sq*weights).sum(-1)+1.)
    guard = torch.where(counts>0,guard,0.)
    nc_lower = (nc_sse+radial_o_sse-guard).clamp_min(0)
    grid_lower = (best-gap_bound-guard).clamp_min(0)
    lower = torch.maximum(nc_lower,grid_lower)
    upper = best+guard
    _require(bool((lower<=upper).all()),'internal ideal bracket inconsistency')
    fitted = ideal_nco_template(aa,best_psi)@best_q+ca[:,None]
    ideal_bb = torch.stack((fitted[:,0],ca,fitted[:,1],fitted[:,2]),dim=-2)
    # The current head's observed-frame product and quaternion eps need not be
    # mathematically exact SO(3). This witness bypasses that neural map and
    # exercises only the frozen actual atom producer, without claiming a
    # uniform float32 error bound for every possible head output.
    from protein_jepa_frame.models.stage_c.openfold_atom37 import DECODER_TO_OPENFOLD_ROT
    from protein_jepa_frame.models.stage_c.structure_decoder import reconstruct_atom37_from_frames
    rotation_of = best_q.transpose(-1,-2)
    basis = torch.tensor(DECODER_TO_OPENFOLD_ROT,dtype=torch.float64)
    rotation_decoder = rotation_of@basis.T
    torsions = torch.zeros(1,len(aa),7,dtype=torch.float64)
    torsions[0,:,1]=best_psi
    atom37,exists = reconstruct_atom37_from_frames(rotation_decoder[None],ca[None],aa[None],torsions)
    _require(bool(exists[..., [0,1,2,4]].all()),'official witness omits backbone atoms')
    actual_bb = torch.stack((atom37[0,:,0],ca,atom37[0,:,2],atom37[0,:,4]),dim=-2)
    actual_sse = ((actual_bb[:,NCO_INDICES]-target[:,NCO_INDICES]).square().sum(-1)*weights).sum(-1)
    roundoff = (actual_bb[:,NCO_INDICES]-ideal_bb[:,NCO_INDICES]).abs().amax((-2,-1))
    _require(bool(torch.isfinite(actual_bb[active]).all()),'nonfinite actual float32 witness')
    res_values = dict(n_nco_atoms=counts,ideal_sse_lower=lower,ideal_sse_upper=upper,
        ideal_nc_radial_sse_lower=nc_lower,ideal_grid_sse_lower=grid_lower,
        ideal_grid_witness_sse=best,grid_gap_bound_sse=gap_bound,numerical_guard_sse=guard,
        actual_witness_sse=actual_sse,actual_minus_ideal_sse=actual_sse-best,
        actual_vs_ideal_max_coordinate_difference=roundoff)
    per_residue = {name:value.reshape(shape) for name,value in res_values.items()}
    total = counts.reshape(shape).sum(-1)
    totals = {name:value.reshape(shape).sum(-1) for name,value in
              [('ideal_sse_lower',lower),('ideal_sse_upper',upper),('actual_witness_sse',actual_sse)]}
    per_sample = dict(n_nco_atoms=total,**totals)
    for prefix,sse in [('ideal_lower',totals['ideal_sse_lower']),('ideal_upper',totals['ideal_sse_upper']),
                       ('actual_witness',totals['actual_witness_sse'])]:
        mse = sse/total.clamp_min(1)
        if prefix.startswith('ideal_'):
            suffix=prefix.removeprefix('ideal_')
            per_sample['ideal_mse_'+suffix]=mse
            per_sample['ideal_rmsd_'+suffix]=mse.sqrt()
        else:
            per_sample['actual_witness_mse']=mse
            per_sample['actual_witness_rmsd']=mse.sqrt()
    witness = dict(R_openfold=rotation_of.reshape(shape+(3,3)),
        R_decoder=rotation_decoder.reshape(shape+(3,3)),psi=best_psi.reshape(shape),
        ideal_backbone=torch.where(active[:,None,None],ideal_bb,0.).reshape(shape+(4,3)),
        actual_backbone=torch.where(active[:,None,None],actual_bb,0.).reshape(shape+(4,3)),
        atom_mask=active.reshape(shape)[...,None].expand(shape+(4,)).clone(),
        scoring_atom_mask=masks.reshape(shape+(4,)))
    return dict(schema_version=SCHEMA_VERSION,per_residue=per_residue,per_sample=per_sample,witness=witness,
        conventions=dict(role='label-oracle kinematic capacity diagnostic; never forecast/head input',
            family='independent per-residue proper SO(3), fixed supplied CA and psi; no shared neural weight constraint',
            objective='sum squared Euclidean N/C/O coordinate errors; no translation/alignment fit',
            normalization='sum per-residue SSE / number of valid NCO atoms per sample; then sqrt for RMSD',
            psi_domain=psi_domain,psi_bounds_rad=[low,high],torsion_bins=torsion_bins,
            psi_bounds_kind='closure of current float32 arithmetic-bin mean' if psi_domain=='head_linear_bins' else 'full-circle relaxation of current head',
            grid_size=grid_size,grid_rule='uniform cell midpoints; strict interior witnesses; covering radius width/(2K)',
            global_lower='max(0, exact_NC_SO3_plus_independent_O_radial, grid_min_minus_circle_chord_bound), minus float64 guard',
            global_upper='explicit grid witness plus float64 guard; upper only for independent kinematic family, not neural weights',
            float64_guard_multiplier=FLOAT64_GUARD_MULTIPLIER,
            numerical_scope='ideal float64 mathematical bounds with engineering rounding guard; not formal interval certification',
            actual_witness='frozen OpenFold float32 atom producer; exact CA restored as in current head; scored separately',
            actual_float32_global_bound=False,
            empty_support='zero plus count zero; report as null, never perfect fit',
            chemical_order='ARNDCQEGHILKMFPSTWYV',unknown_id=20,unknown_backbone='existing GLY template fallback; identity remains unknown',
            constants_source='protein_jepa_frame/vendored/openfold/np/residue_constants.py',template_constants_sha256=template_hash,
            not_constrained=['inter-residue peptide closure','bond angles across residues','clashes','neural mapping capacity','H information content']))
