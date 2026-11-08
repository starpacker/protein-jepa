"""Versioned N/CA/C/O diagnostics; v1 and historical results are unchanged.

Coordinates are in Angstrom, angles in degrees in reported summaries. The
primary error is sqrt(mean(sum_xyz(error**2))) in an ALREADY SHARED coordinate
gauge. The caller fits real future CA to observed anchor CA once and applies
that transform to every real future atom; predictions remain in the anchor
gauge. No prediction-dependent fit enters this primary error. Shape RMSD is a
separate, independently superposed all-backbone-atom diagnostic.

Geometry is computed for EVERY leading sample. ``mean`` gives equal weight to
nonempty samples, never a pooled atom mean. ``totals`` sums counts; retain
``per_sample`` for protein-level downstream aggregation/uncertainty. Missing
values are None. Masks may be broadcast, including [B,L] -> [B,S,L]. Explicit
[B,1,L] is required if B == S makes a [B,L] mask ambiguous.

Threshold provenance: *_legacy_valid_frac retains v1's explicitly documented
bond/angle windows, omega uses its 30-degree trans window, and CA clashes use
its effective 2.5-Angstrom cutoff (not its unused 3.0 argument). These are
diagnostics, not newly registered pass gates. The legacy rectangular phi/psi
boxes are named ``ramachandran_box_diagnostic_frac``: this is NOT a MolProbity
allowed fraction. A residue-class-specific statistical Ramachandran reference
and full-backbone/all-atom steric validation remain absent.

Optional aatype is OpenFold's restypes ordering, with 20=unknown. Additional
continuous absolute-z deviations use vendored OpenFold stereo_chemical_props
and residue_constants (including the C-N bond to PRO). No new z cutoff is
introduced. Source provenance: protein_jepa_frame/vendored/PROVENANCE.json.
"""
from __future__ import annotations

import math
from functools import lru_cache
from importlib import resources
from pathlib import Path

import torch

from evaluation.geometry import kabsch_align

SCHEMA_VERSION = "backbone_metrics_v2"
ATOM_NAMES = ("N", "CA", "C", "O")
# Exact historical v1 diagnostic conventions, not chemical reference tables.
_LEGACY_BONDS = {"N_CA": (1.458, .05), "CA_C": (1.525, .05),
                 "C_O": (1.231, .03), "C_N": (1.329, .05)}
_LEGACY_ANGLES = {"N_CA_C": 111.0, "CA_C_N": 116.2, "C_N_CA": 121.7}
_ANGLE_TOL_RAD = .20
_RAMA_BOXES = ((-160., -20., -90., 60.), (-180., -40., 90., 180.),
               (0., 180., -180., 180.))
_EPS = 1e-10  # numerical degeneracy, not a stereochemical pass threshold


def _check_bb(bb: torch.Tensor) -> None:
    if bb.ndim < 3 or bb.shape[-2:] != (4, 3) or bb.numel() == 0:
        raise ValueError("expected nonempty [...,L,4,3] N/CA/C/O backbone")
    if not bb.is_floating_point():
        raise ValueError("backbone coordinates must be floating point")


def _broadcast(value, shape, *, device, dtype, name):
    value = torch.as_tensor(value, device=device, dtype=dtype)
    batch_expanded = None
    event_ndim = 2 if name == "atom_mask" else 1
    if len(shape) > value.ndim and value.ndim > event_ndim and value.shape[0] == shape[0]:
        batch_expanded = value.reshape(value.shape[:1] + (1,) * (len(shape) - value.ndim)
                                       + value.shape[1:])
    try:
        result = torch.broadcast_to(value, shape)
    except RuntimeError:
        # Common batch metadata [B,L] / [B,L,4] across sample axis [B,S,...].
        if batch_expanded is not None:
            try:
                return torch.broadcast_to(batch_expanded, shape)
            except RuntimeError:
                pass
        raise ValueError(f"{name} with shape {tuple(value.shape)} cannot broadcast to {shape}")
    if batch_expanded is not None and value.shape[0] > 1:
        try:
            torch.broadcast_to(batch_expanded, shape)
        except RuntimeError:
            pass
        else:
            raise ValueError(f"ambiguous batch/sample {name}: insert an explicit singleton sample or batch axis")
    return result


def _mask(bb, mask=None, atom_mask=None):
    shape = bb.shape[:-2]
    m = torch.ones(shape, device=bb.device, dtype=torch.bool) if mask is None else _broadcast(
        mask, shape, device=bb.device, dtype=torch.bool, name="mask")
    am = m[..., None].expand(*shape, 4)
    if atom_mask is not None:
        am = am & _broadcast(atom_mask, (*shape, 4), device=bb.device,
                             dtype=torch.bool, name="atom_mask")
    if bool((~torch.isfinite(bb).all(-1) & am).any()):
        raise ValueError("nonfinite coordinate at a valid backbone atom")
    return m, am


def _pack(rows, sample_shape, conventions):
    keys = rows[0].keys() if rows else ()
    counts = {k for k in keys if k.startswith("n_") or k.endswith("_n")}
    means = {}
    for k in keys:
        if k in counts:
            continue
        vals = [row[k] for row in rows if row[k] is not None]
        means[k] = sum(vals) / len(vals) if vals else None
    return {"schema_version": SCHEMA_VERSION, "sample_shape": list(sample_shape),
            "n_samples": len(rows), "conventions": conventions,
            "per_sample": rows, "mean": means,
            "totals": {k: sum(row[k] for row in rows) for k in sorted(counts)}}


def _rms(error, mask):
    if not bool(mask.any()):
        return None
    return float(error[mask].square().sum(-1).mean().sqrt())


def backbone_error_summary(pred_bb, ref_bb, mask=None, *, atom_mask=None):
    """True RMS errors per sample, in a caller-established shared anchor gauge.

    Reference and predictions follow standard Torch broadcasting, e.g.
    pred [S,L,4,3] versus ref [L,4,3], or pred [B,S,L,4,3] versus ref
    [B,1,L,4,3]. ``atom_mask`` is the shared scoring/fit mask. The shape fit
    uses all valid N/CA/C/O atoms once; atom-type errors share that same fit.
    With fewer than three valid atoms shape quantities are None.
    """
    _check_bb(pred_bb)
    _check_bb(ref_bb)
    if pred_bb.shape[-3] != ref_bb.shape[-3]:
        raise ValueError("prediction and reference must have the same residue count")
    pred, ref = torch.broadcast_tensors(pred_bb, ref_bb)
    _, am = _mask(pred, mask, atom_mask)
    _mask(ref, mask, atom_mask)
    pred = torch.where(am[..., None], pred, 0).double()
    ref = torch.where(am[..., None], ref, 0).double()
    leading, L = pred.shape[:-3], pred.shape[-3]
    rows = []
    for p, r, m in zip(pred.reshape(-1, L, 4, 3), ref.reshape(-1, L, 4, 3),
                       am.reshape(-1, L, 4)):
        err = p - r
        row = {"coordinate_rmsd_angstrom": _rms(err, m),
               "shape_kabsch_rmsd_angstrom": None, "n_atoms": int(m.sum())}
        shape_err = None
        if int(m.sum()) >= 3:
            aligned = kabsch_align(p.reshape(-1, 3), r.reshape(-1, 3), m.flatten())
            shape_err = aligned.reshape(L, 4, 3) - r
            row["shape_kabsch_rmsd_angstrom"] = _rms(shape_err, m)
        for a, name in enumerate(ATOM_NAMES):
            row[f"coordinate_rmsd_{name}_angstrom"] = _rms(err[:, a], m[:, a])
            row[f"shape_rmsd_{name}_angstrom"] = (
                _rms(shape_err[:, a], m[:, a]) if shape_err is not None else None)
            row[f"n_atoms_{name}"] = int(m[:, a].sum())
        rows.append(row)
    return _pack(rows, leading, {"coordinate_alignment": "caller_shared_anchor_gauge_no_fit",
                                 "shape_alignment": "independent_all_valid_backbone_atom_kabsch",
                                 "error_unit": "angstrom", "aggregation": "equal_sample_mean"})


def _angle(a, b, c):
    u, v = a - b, c - b
    un, vn = u.norm(dim=-1), v.norm(dim=-1)
    valid = (un > _EPS) & (vn > _EPS)
    cos = ((u / un.clamp_min(_EPS)[..., None]) *
           (v / vn.clamp_min(_EPS)[..., None])).sum(-1)
    return torch.acos(cos.clamp(-1, 1)), valid


def _dihedral(a, b, c, d):
    u = c - b
    un = u.norm(dim=-1)
    unit = u / un.clamp_min(_EPS)[..., None]
    v, w = a - b, d - c
    v = v - (v * unit).sum(-1, keepdim=True) * unit
    w = w - (w * unit).sum(-1, keepdim=True) * unit
    valid = (un > _EPS) & (v.norm(dim=-1) > _EPS) & (w.norm(dim=-1) > _EPS)
    angle = torch.atan2((torch.cross(unit, v, dim=-1) * w).sum(-1), (v * w).sum(-1))
    return angle, valid


@lru_cache(maxsize=1)
def _openfold_tables():
    # Import the pinned module, not a potentially installed external version.
    from protein_jepa_frame.vendored.openfold.np import residue_constants as rc
    # Its resource loader uses the top-level package name even when imported
    # through the vendored namespace. Fail rather than silently use an external
    # installation's stereochemistry while claiming the repository reference.
    expected = Path(rc.__file__).resolve().parents[1] / "resources" / "stereo_chemical_props.txt"
    actual = resources.files("openfold.resources").joinpath("stereo_chemical_props.txt")
    if Path(str(actual)).resolve() != expected:
        raise RuntimeError("OpenFold stereo_chemical_props resource is not the pinned vendored reference")
    bonds, _, angles = rc.load_stereo_chemical_props()
    bond_tables, angle_tables = {}, {}
    for name in ("N_CA", "CA_C", "C_O"):
        pair = set(name.split("_"))
        bond_tables[name] = [(next((b.length for b in bonds[rc.restype_1to3[aa]]
                                   if {b.atom1_name, b.atom2_name} == pair), math.nan),
                              next((b.stddev for b in bonds[rc.restype_1to3[aa]]
                                    if {b.atom1_name, b.atom2_name} == pair), math.nan))
                             for aa in rc.restypes] + [(math.nan, math.nan)]
    angle_tables["N_CA_C"] = [next(((a.angle_rad, a.stddev) for a in angles[rc.restype_1to3[aa]]
                                                  if (a.atom1_name, a.atom2_name, a.atom3name)
                                                  == ("N", "CA", "C")), (math.nan, math.nan))
                                   for aa in rc.restypes] + [(math.nan, math.nan)]
    return rc, bond_tables, angle_tables


def _mean(x, keep):
    return float(x[keep].mean()) if bool(keep.any()) else None


def _one_geometry(bb, am, chain, indices, aa, clash_cutoff):
    N, CA, C, O = bb.unbind(-2)
    n, ca, c, o = am.unbind(-1)
    L = len(bb)
    adjacent = (chain[1:] == chain[:-1]) & (indices[1:] - indices[:-1] == 1)
    row = {"n_atoms": int(am.sum()), "n_residues": int(am.any(-1).sum())}
    bonds = {"N_CA": ((N-CA).norm(dim=-1), n & ca),
             "CA_C": ((CA-C).norm(dim=-1), ca & c),
             "C_O": ((C-O).norm(dim=-1), c & o),
             "C_N": ((C[:-1]-N[1:]).norm(dim=-1), adjacent & c[:-1] & n[1:])}
    angles = {}
    for name, points, keep in (
        ("N_CA_C", (N, CA, C), n & ca & c),
        ("CA_C_N", (CA[:-1], C[:-1], N[1:]), adjacent & ca[:-1] & c[:-1] & n[1:]),
        ("C_N_CA", (C[:-1], N[1:], CA[1:]), adjacent & c[:-1] & n[1:] & ca[1:])):
        values, nondegenerate = _angle(*points)
        angles[name] = (values, keep & nondegenerate)
        row[f"angle_{name}_degenerate_n"] = int((keep & ~nondegenerate).sum())
    for name, (values, keep) in bonds.items():
        ideal, tol = _LEGACY_BONDS[name]
        row[f"bond_{name}_mean_angstrom"] = _mean(values, keep)
        row[f"bond_{name}_legacy_valid_frac"] = _mean(((values-ideal).abs() < tol).double(), keep)
        row[f"bond_{name}_n"] = int(keep.sum())
    for name, (values, keep) in angles.items():
        row[f"angle_{name}_mean_deg"] = _mean(torch.rad2deg(values), keep)
        row[f"angle_{name}_legacy_valid_frac"] = _mean(
            ((values-math.radians(_LEGACY_ANGLES[name])).abs() < _ANGLE_TOL_RAD).double(), keep)
        row[f"angle_{name}_n"] = int(keep.sum())

    phi, phi_ok = _dihedral(C[:-1], N[1:], CA[1:], C[1:])
    psi, psi_ok = _dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
    omega, omega_ok = _dihedral(CA[:-1], C[:-1], N[1:], CA[1:])
    phi_req = adjacent & c[:-1] & n[1:] & ca[1:] & c[1:]
    psi_req = adjacent & n[:-1] & ca[:-1] & c[:-1] & n[1:]
    omega_req = adjacent & ca[:-1] & c[:-1] & n[1:] & ca[1:]
    phi_mask, psi_mask, omega_mask = phi_req & phi_ok, psi_req & psi_ok, omega_req & omega_ok
    # phi array's entry i belongs to residue i+1; psi's entry i belongs to i.
    pair_mask = phi_mask[:-1] & psi_mask[1:]
    phid, psid = torch.rad2deg(phi[:-1]), torch.rad2deg(psi[1:])
    inbox = torch.zeros_like(pair_mask)
    for plo, phi_hi, slo, shi in _RAMA_BOXES:
        inbox |= (phid >= plo) & (phid <= phi_hi) & (psid >= slo) & (psid <= shi)
    row.update({"phi_n": int(phi_mask.sum()), "psi_n": int(psi_mask.sum()),
                "phi_degenerate_n": int((phi_req & ~phi_ok).sum()),
                "psi_degenerate_n": int((psi_req & ~psi_ok).sum()),
                "omega_degenerate_n": int((omega_req & ~omega_ok).sum()),
                "ramachandran_box_diagnostic_frac": _mean(inbox.double(), pair_mask),
                "ramachandran_box_n": int(pair_mask.sum()),
                "omega_trans_diagnostic_frac": _mean((omega.abs() > math.radians(150)).double(), omega_mask),
                "omega_abs_deviation_from_trans_mean_deg": _mean(180-torch.rad2deg(omega.abs()), omega_mask),
                "omega_n": int(omega_mask.sum())})
    i, j = torch.triu_indices(L, L, offset=1, device=bb.device)
    nonlocal_pair = ca[i] & ca[j] & ((chain[i] != chain[j]) | ((indices[i]-indices[j]).abs() >= 3))
    distances = (CA[i]-CA[j]).norm(dim=-1)
    clash = distances < clash_cutoff
    row.update({"ca_nonlocal_clash_frac": _mean(clash.double(), nonlocal_pair),
                "ca_nonlocal_pair_n": int(nonlocal_pair.sum()),
                "ca_nonlocal_clash_n": int((clash & nonlocal_pair).sum())})

    # Fixed keys even when amino-acid identities are unavailable or unknown.
    for kind, names in (("bond", bonds), ("angle", angles)):
        for name in names:
            row[f"openfold_{kind}_{name}_mean_abs_z"] = None
            row[f"openfold_{kind}_{name}_n"] = 0
    if aa is not None:
        rc, btable, atable = _openfold_tables()
        known = aa < 20
        for name, (values, keep) in bonds.items():
            if name == "C_N":
                pro = (aa[1:] == rc.resname_to_idx["PRO"]).long()
                means = bb.new_tensor(rc.between_res_bond_length_c_n)[pro]
                stds = bb.new_tensor(rc.between_res_bond_length_stddev_c_n)[pro]
                keep = keep & known[1:]
            else:
                means, stds = bb.new_tensor(btable[name])[aa].unbind(-1)
                keep = keep & known
            row[f"openfold_bond_{name}_mean_abs_z"] = _mean((values-means).abs()/stds, keep)
            row[f"openfold_bond_{name}_n"] = int(keep.sum())
        for name, (values, keep) in angles.items():
            if name == "N_CA_C":
                means, stds = bb.new_tensor(atable[name])[aa].unbind(-1)
                keep = keep & known
                deviation = (values-means).abs()/stds
            else:
                means, stds = (rc.between_res_cos_angles_ca_c_n if name == "CA_C_N"
                               else rc.between_res_cos_angles_c_n_ca)
                # OpenFold reference here is a distribution of cosine, not radians.
                deviation = (values.cos()-means).abs()/stds
            row[f"openfold_angle_{name}_mean_abs_z"] = _mean(deviation, keep)
            row[f"openfold_angle_{name}_n"] = int(keep.sum())
    return row


def backbone_geometry_summary(bb, mask=None, *, atom_mask=None, chain_id=None,
                              residue_index=None, aatype=None, clash_cutoff=2.5):
    """Evaluate each sample, excluding only atoms required by each observable.

    Missing topology means one chain and consecutive array indices; this
    assumption is explicitly recorded. Chain boundaries and index gaps break
    peptide bonds/torsions. Nonlocal CA pairs are unique unordered pairs:
    different chains OR sequence-index separation >=3 within a chain. This is
    a CA-distance diagnostic, never a full-backbone/all-atom clash score.
    """
    _check_bb(bb)
    if not math.isfinite(clash_cutoff) or clash_cutoff <= 0:
        raise ValueError("clash_cutoff must be positive and finite")
    _, am = _mask(bb, mask, atom_mask)
    bb = torch.where(am[..., None], bb, 0).double()
    shape, leading, L = bb.shape[:-2], bb.shape[:-3], bb.shape[-3]
    metadata = {}
    for name, value, default in (
        ("chain_id", chain_id, torch.zeros(L, dtype=torch.long, device=bb.device)),
        ("residue_index", residue_index, torch.arange(L, device=bb.device)),
        ("aatype", aatype, None)):
        metadata[name] = (None if value is None and default is None else _broadcast(
            default if value is None else value, shape, device=bb.device, dtype=torch.long, name=name)
            .reshape(-1, L))
    aas = metadata["aatype"]
    if aas is not None and bool(((aas < 0) | (aas > 20)).any()):
        raise ValueError("aatype must use OpenFold indices 0..20")
    rows = []
    for i, (b, m) in enumerate(zip(bb.reshape(-1, L, 4, 3), am.reshape(-1, L, 4))):
        rows.append(_one_geometry(b, m, metadata["chain_id"][i], metadata["residue_index"][i],
                                  aas[i] if aas is not None else None, clash_cutoff))
    return _pack(rows, leading, {
        "coordinate_unit": "angstrom", "angle_unit": "degree", "aggregation": "equal_sample_mean",
        "chain_id_assumed_single_chain": chain_id is None,
        "residue_index_assumed_consecutive": residue_index is None,
        "legacy_bond_reference_and_tolerance_angstrom": _LEGACY_BONDS,
        "legacy_angle_reference_deg": _LEGACY_ANGLES, "legacy_angle_tolerance_rad": _ANGLE_TOL_RAD,
        "omega_trans_diagnostic_abs_degrees_gt": 150.0,
        "ramachandran_box_degrees": _RAMA_BOXES,
        "ramachandran_statistical_reference": None,
        "ramachandran_box_is_molprobity_allowed": False,
        "ca_clash_cutoff_angstrom": float(clash_cutoff), "ca_nonlocal_min_sequence_separation": 3,
        "full_backbone_atom_clashes_evaluated": False,
        "openfold_reference": "protein_jepa_frame/vendored/PROVENANCE.json",
        "openfold_z_threshold": None})


__all__ = ["SCHEMA_VERSION", "backbone_error_summary", "backbone_geometry_summary"]
