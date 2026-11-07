"""Differentiable, versioned actual-backbone connection and steric objectives.

Inputs are actual N/CA/C/O coordinates in Angstrom, never a detached diagnostic
reconstruction. Chemical IDs MUST use pinned OpenFold ARNDCQEGHILKMFPSTWYV order,
20=unknown; dataset embedding IDs must be converted by the caller. This module
never changes embedding semantics, coordinates, masks, or a frozen CA decoder.

The connection flat-bottom formula follows AlphaFold all_atom.py (Jumper et al.
2021 supplement 1.9.11, eqs 44/45). Unlike the old vendored OpenFold loss, the
CA-C-N cosine standard deviation comes from the ANGLE table (0.0311), not the
C-N bond table (0.014). Default tolerance is 12 reference standard deviations.
The inter-residue clash formula follows eq 46: relu(r_a+r_b-1.5A-distance).
Only real same-chain consecutive peptide C--N bonds are excluded. Other
inter-residue atom pairs, including cross-chain equal residue numbers, remain.
Same-residue pairs and sidechains/hydrogens are outside this objective's scope.

Each term averages valid features within each sample, then equally averages
nonempty samples. Empty terms return differentiable zero AND zero counts.
Terms retain native units (C-N/clash Angstrom, angle cosines dimensionless);
there is deliberately no implicitly weighted total or performance/pass gate.

All leading coordinate axes are sample axes. Masks/metadata support standard
broadcast and [B,L] -> [B,S,L]; ambiguous B==S requires explicit [B,1,L].
Masked NaN/Inf are removed before arithmetic; nonfinite VALID atoms fail.
Float16/bfloat16 arithmetic is promoted to float32; float64 stays float64.

References (formulas; constants are always the LOCAL vendored module):
https://github.com/google-deepmind/alphafold/blob/main/alphafold/model/all_atom.py
protein_jepa_frame/vendored/PROVENANCE.json
"""
from __future__ import annotations

import math

import torch

from protein_jepa_frame.vendored.openfold.np import residue_constants as rc


SCHEMA_VERSION = "backbone_objectives_v2.0"
ATOM_NAMES = ("N", "CA", "C", "O")
DEFAULT_CONNECTION_TOLERANCE = 12.0
DEFAULT_CLASH_OVERLAP_ANGSTROM = 1.5
CONNECTION_EPS = 1e-6
CLASH_EPS = 1e-10


def _broadcast(value, shape, *, device, name, atom=False, boolean=False):
    value = torch.as_tensor(value, device=device)
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean")
    elif value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must contain integer IDs")
    else:
        # Avoid uint8 underflow (0-1 -> 255) or narrow signed overflow when
        # detecting reverse-order peptide links. Chemical IDs stay unchanged.
        value = value.to(torch.long)
    event_ndim = 2 if atom else 1
    expanded = None
    if len(shape) > value.ndim > event_ndim and value.shape[0] == shape[0]:
        expanded = value.reshape(value.shape[:1] + (1,) * (len(shape)-value.ndim) + value.shape[1:])
    try:
        regular = torch.broadcast_to(value, shape)
    except RuntimeError:
        if expanded is not None:
            try:
                return torch.broadcast_to(expanded, shape)
            except RuntimeError:
                pass
        raise ValueError(f"{name} cannot broadcast to {shape}")
    if expanded is not None and value.shape[0] > 1:
        try:
            torch.broadcast_to(expanded, shape)
        except RuntimeError:
            pass
        else:
            raise ValueError(f"ambiguous batch/sample {name}; insert a singleton axis")
    return regular


def _prepare(backbone, atom_mask, chemical_aatype, residue_index, chain_id, residue_mask):
    if (not torch.is_tensor(backbone) or backbone.ndim < 3
            or backbone.shape[-2:] != (4, 3) or backbone.numel() == 0
            or not backbone.is_floating_point()):
        raise ValueError("backbone must be nonempty floating [...,L,4,3] N/CA/C/O")
    shape, device = backbone.shape[:-2], backbone.device
    residue = (torch.ones(shape, dtype=torch.bool, device=device) if residue_mask is None
               else _broadcast(residue_mask, shape, device=device, name="residue_mask", boolean=True))
    atoms = _broadcast(atom_mask, (*shape, 4), device=device, name="atom_mask", atom=True, boolean=True)
    atoms = atoms & residue[..., None]
    if bool((~torch.isfinite(backbone).all(-1) & atoms).any()):
        raise ValueError("nonfinite coordinate at a valid backbone atom")
    aa, ri, ci = [_broadcast(v, shape, device=device, name=n)
                  for v,n in ((chemical_aatype,"chemical_aatype"),
                              (residue_index,"residue_index"),(chain_id,"chain_id"))]
    if bool((((aa < 0) | (aa > 20)) & residue).any()):
        raise ValueError("chemical_aatype must use OpenFold IDs 0..19 or UNK=20")
    i,j = torch.triu_indices(shape[-1], shape[-1], offset=1, device=device)
    same = ci[..., i] == ci[..., j]
    present = residue[..., i] & residue[..., j]
    difference = ri[..., j]-ri[..., i]
    if bool((same & present & (difference == 0)).any()):
        raise ValueError("duplicate residue_index in a valid chain")
    # Works with explicit padding, multiple chains, gaps, or permuted storage.
    forward = same & present & (difference == 1)
    reverse = same & present & (difference == -1)
    x = torch.where(atoms[..., None], backbone, 0)
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return x, atoms, residue, aa, i, j, forward, reverse


def _reduce(value, valid):
    counts = valid.sum(-1)
    per_sample = torch.where(valid, value, 0).sum(-1) / counts.clamp_min(1)
    nonempty = counts > 0
    mean = (per_sample * nonempty).sum() / nonempty.sum().clamp_min(1)
    return mean, per_sample, counts


def _connections(prepared, tolerance):
    x, atoms, residue, aa, i, j, forward, reverse = prepared
    previous = torch.where(forward[..., None, None], x[..., i, :, :], x[..., j, :, :])
    following = torch.where(forward[..., None, None], x[..., j, :, :], x[..., i, :, :])
    pm = torch.where(forward[..., None], atoms[..., i, :], atoms[..., j, :])
    nm = torch.where(forward[..., None], atoms[..., j, :], atoms[..., i, :])
    next_type = torch.where(forward, aa[..., j], aa[..., i])
    connected = forward | reverse
    ca, c = previous[..., 1, :], previous[..., 2, :]
    n, next_ca = following[..., 0, :], following[..., 1, :]
    c_n = n-c
    cn_length = torch.sqrt(c_n.square().sum(-1)+CONNECTION_EPS)
    ca_c_length = torch.sqrt((ca-c).square().sum(-1)+CONNECTION_EPS)
    n_ca_length = torch.sqrt((next_ca-n).square().sum(-1)+CONNECTION_EPS)
    ca_c_n_cos = (((ca-c)/ca_c_length[..., None])*(c_n/cn_length[..., None])).sum(-1)
    c_n_ca_cos = ((-c_n/cn_length[..., None])*((next_ca-n)/n_ca_length[..., None])).sum(-1)
    cn_candidates = connected & pm[..., 2] & nm[..., 0]
    proline = next_type == rc.restype_order['P']
    unknown = next_type == 20
    known_cn = cn_candidates & ~unknown
    means = x.new_tensor(rc.between_res_bond_length_c_n)
    deviations = x.new_tensor(rc.between_res_bond_length_stddev_c_n)
    cn_mean, cn_std = means[proline.long()], deviations[proline.long()]
    features = {
        'c_n': (cn_length, cn_mean, cn_std, known_cn),
        'ca_c_n': (ca_c_n_cos, rc.between_res_cos_angles_ca_c_n[0],
                   rc.between_res_cos_angles_ca_c_n[1], cn_candidates & pm[..., 1]),
        'c_n_ca': (c_n_ca_cos, rc.between_res_cos_angles_c_n_ca[0],
                   rc.between_res_cos_angles_c_n_ca[1], cn_candidates & nm[..., 1]),
    }
    losses, per_sample, counts = {}, {}, {
        'n_connection_candidates': connected.sum(-1),
        'n_c_n_atom_candidates': cn_candidates.sum(-1),
        'n_c_n_proline': (known_cn & proline).sum(-1),
        'n_c_n_nonproline': (known_cn & ~proline).sum(-1),
        'n_c_n_unknown_next_excluded': (cn_candidates & unknown).sum(-1),
    }
    for key, (actual, mean, std, valid) in features.items():
        error = torch.sqrt((actual-mean).square()+CONNECTION_EPS)
        loss = torch.relu(error-tolerance*std)
        losses[key], per_sample[key], counts['n_'+key] = _reduce(loss, valid)
        counts['n_'+key+'_outside_flat_bottom'] = ((error > tolerance*std) & valid).sum(-1)
    return losses, per_sample, counts


def _clashes(prepared, overlap):
    x, atoms, residue, aa, i, j, forward, reverse = prepared
    delta = x[..., i, :, None, :]-x[..., j, None, :, :]
    distances = torch.sqrt(delta.square().sum(-1)+CLASH_EPS)
    pairs = atoms[..., i, :, None] & atoms[..., j, None, :]
    cn = torch.zeros((4,4), dtype=torch.bool, device=x.device)
    cn[2,0] = True
    peptide = forward[..., None, None] & cn
    peptide = peptide | (reverse[..., None, None] & cn.transpose(0,1))
    excluded = pairs & peptide
    pairs = pairs & ~peptide
    radii = x.new_tensor([rc.van_der_waals_radius[a[0]] for a in ATOM_NAMES])
    boundary = radii[:,None]+radii[None,:]-overlap
    penalty = torch.relu(boundary-distances)
    loss, per_sample, count = _reduce(penalty.flatten(-3), pairs.flatten(-3))
    return {'backbone_interresidue_clash': loss}, {'backbone_interresidue_clash': per_sample}, {
        'n_nonbonded_atom_pairs': count,
        'n_excluded_peptide_atom_pairs': excluded.flatten(-3).sum(-1),
        'n_clashing_atom_pairs': ((distances < boundary) & pairs).flatten(-3).sum(-1),
    }


def _parameter(value, name):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _result(prepared, parts, tolerance, overlap):
    x, atoms, residue, aa, *_ = prepared
    losses, per_sample, counts = {}, {}, {
        'n_valid_atoms': atoms.sum((-2,-1)),
        'n_unknown_residues': ((aa == 20) & residue & atoms.any(-1)).sum(-1),
    }
    for part in parts:
        losses.update(part[0]); per_sample.update(part[1]); counts.update(part[2])
    return dict(schema_version=SCHEMA_VERSION, losses=losses, per_sample=per_sample, counts=counts,
        conventions=dict(atom_order=ATOM_NAMES, chemical_order='ARNDCQEGHILKMFPSTWYV', unknown_id=20,
            constants_source='protein_jepa_frame/vendored/openfold/np/residue_constants.py',
            reference_provenance='protein_jepa_frame/vendored/PROVENANCE.json',
            connection_tolerance_factor=tolerance, clash_overlap_angstrom=overlap,
            c_n_means=list(rc.between_res_bond_length_c_n),
            c_n_stddevs=list(rc.between_res_bond_length_stddev_c_n),
            ca_c_n_cos_mean_std=list(rc.between_res_cos_angles_ca_c_n),
            c_n_ca_cos_mean_std=list(rc.between_res_cos_angles_c_n_ca),
            van_der_waals_radii={name:rc.van_der_waals_radius[name[0]] for name in ATOM_NAMES},
            connection_eps=CONNECTION_EPS, clash_eps=CLASH_EPS, arithmetic=str(x.dtype),
            aggregation='equal nonempty samples; feature mean within sample; empty terms zero with zero count',
            units={'c_n':'angstrom','ca_c_n':'cosine','c_n_ca':'cosine','backbone_interresidue_clash':'angstrom'},
            unknown_policy='exclude next-UNK C-N prior only; retain generic angle and existing-atom clash terms',
            topology='same chain and consecutive residue IDs; storage-order independent; no gap bridging',
            clash_scope='unique inter-residue N/CA/C/O atom pairs; exclude only actual peptide C-N bonds',
            not_covered=['same-residue violations','sidechains','hydrogens','MolProbity','Ramachandran','omega'],
            total='caller must explicitly specify unit-aware weights; no implicit total'))


def backbone_connection_loss(backbone, *, chemical_aatype, residue_index, chain_id,
                             atom_mask, residue_mask=None,
                             tolerance_factor=DEFAULT_CONNECTION_TOLERANCE):
    tolerance = _parameter(tolerance_factor, 'tolerance_factor')
    prepared = _prepare(backbone, atom_mask, chemical_aatype, residue_index, chain_id, residue_mask)
    return _result(prepared, [_connections(prepared, tolerance)], tolerance, None)


def backbone_nonbonded_clash_loss(backbone, *, chemical_aatype, residue_index, chain_id,
                                atom_mask, residue_mask=None,
                                overlap_tolerance_angstrom=DEFAULT_CLASH_OVERLAP_ANGSTROM):
    overlap = _parameter(overlap_tolerance_angstrom, 'overlap_tolerance_angstrom')
    prepared = _prepare(backbone, atom_mask, chemical_aatype, residue_index, chain_id, residue_mask)
    return _result(prepared, [_clashes(prepared, overlap)], None, overlap)


def backbone_geometry_losses_v2(backbone, *, chemical_aatype, residue_index, chain_id,
                                atom_mask, residue_mask=None,
                                tolerance_factor=DEFAULT_CONNECTION_TOLERANCE,
                                overlap_tolerance_angstrom=DEFAULT_CLASH_OVERLAP_ANGSTROM):
    """Return four separately weighted-able actual-coordinate losses and counts."""
    tolerance = _parameter(tolerance_factor, 'tolerance_factor')
    overlap = _parameter(overlap_tolerance_angstrom, 'overlap_tolerance_angstrom')
    prepared = _prepare(backbone, atom_mask, chemical_aatype, residue_index, chain_id, residue_mask)
    return _result(prepared, [_connections(prepared,tolerance), _clashes(prepared,overlap)], tolerance, overlap)
