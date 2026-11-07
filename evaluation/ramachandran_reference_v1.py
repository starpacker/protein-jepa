"""Top8000 Ramachandran scores with explicit topology and input completeness.

This is this project's statistical backbone diagnostic, not the full MolProbity
pipeline. Scores, periodic interpolation, six classes, and thresholds reproduce
the pinned cctbx compiled reference. ``allowed_only`` excludes ``favored``;
``non_outlier`` includes both. Unknown chemical types are never GLY fallbacks.

Unlike upstream ramalyze's missing-omega fallback to trans-Pro, this wrapper
excludes PRO with missing/degenerate preceding omega and reports that reason.
Explicit directed LOCAL-row peptide edges determine predecessors/successors;
geometry_row_ids validate identity only. No distance-based edge inference occurs. Valid
nonfinite atoms fail; masked atoms can contain NaN. All evaluation is CPU float64
and non-differentiable. O atoms do not enter the torsions.

Data attribution: Richardson Lab at Duke University; Williams et al. (2018),
doi:10.1002/pro.3330. Upstream code/data licenses and unmodified source files are
in reference_data/ramachandran_top8000_v1/ (BSD-style cctbx and CC BY 4.0 data).
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re

import numpy as np
import torch


VERSION = 'explicit_topology_top8000_compiled_ramachandran_v1'
REFERENCE = Path(__file__).with_name('reference_data')/'ramachandran_top8000_v1'
TABLE_PATH = REFERENCE/'cctbx/mmtbx/validation/ramachandran/rama8000_tables.h'
TABLE_SHA256 = '314621665c0be2801f12a5a34288e6f1cb52a79ecee23e9d93b9508c42ec1acd'
CLASSES = ('general', 'glycine', 'cis-proline', 'trans-proline', 'pre-proline', 'isoleucine or valine')
TABLE_ARRAY_NAMES = ('general', 'glycine', 'cis_pro', 'trans_pro', 'pre_pro', 'ile_val')
CHEMICAL_ALPHABET = 'ARNDCQEGHILKMFPSTWYV'  # OpenFold; 20 is unknown.
GLY, PRO, ILE, VAL = (CHEMICAL_ALPHABET.index(a) for a in 'GPIV')
ALLOWED_THRESHOLDS = (.0005, .001, .002, .001, .001, .001)
FAVORED_THRESHOLD = .02
EPS_ANGSTROM = 1e-10  # Numerical degeneracy only, not a stereochemical cutoff.
EXCLUSION_ORDER = ('masked', 'topology_incomplete', 'neighbor_masked', 'unknown_current',
                   'unknown_next', 'missing_atoms', 'degenerate')
ISSUES = ('missing_previous', 'missing_next', 'phi_atoms_missing', 'psi_atoms_missing',
          'pro_omega_atoms_missing', 'phi_degenerate', 'psi_degenerate', 'pro_omega_degenerate')


def _integer(value, name):
    value = torch.as_tensor(value, device='cpu')
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f'{name} must contain integers')
    return value.long()


def _class_tensor(value):
    if isinstance(value, str):
        if value not in CLASSES:
            raise ValueError(f'unknown Ramachandran class: {value}')
        value = CLASSES.index(value)
    result = _integer(value, 'rama_class')
    if bool(((result < 0) | (result >= 6)).any()):
        raise ValueError('rama_class must be in 0..5')
    return result


@lru_cache(maxsize=1)
def _tables():
    data = TABLE_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != TABLE_SHA256:
        raise ValueError('pinned official Ramachandran table hash mismatch')
    text = data.decode('utf-8')
    arrays = []
    for name in TABLE_ARRAY_NAMES:
        matches = re.findall(r'const double linear_table_'+name+r'\[\]\s*=\s*\{([^}]+)\};', text)
        if len(matches) != 1:
            raise ValueError(f'missing or ambiguous official table: {name}')
        values = np.fromstring(matches[0].strip().rstrip(','), sep=',', dtype=np.float64)
        if values.size != 180*180 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'invalid compiled reference table: {name}')
        arrays.append(values.reshape(180,180))
    return torch.from_numpy(np.stack(arrays))


@torch.no_grad()
def _classify_scores(score_tensor, class_tensor):
    scores = torch.as_tensor(score_tensor, dtype=torch.float64, device='cpu')
    classes = _class_tensor(class_tensor)
    scores, classes = torch.broadcast_tensors(scores, classes)
    if not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError('reference scores must be finite in [0,1]')
    allowed = torch.tensor(ALLOWED_THRESHOLDS, dtype=torch.float64)[classes]
    return torch.where(scores >= FAVORED_THRESHOLD, 2,
                       torch.where(scores >= allowed, 1, 0)).long()


@torch.no_grad()
def score_angles(phi_degrees, psi_degrees, rama_class):
    """Return CPU tensors ``score`` and ``category`` (0 outlier, 1 allowed-only, 2 favored).

    Angles and classes follow normal Torch broadcasting. Class accepts an exact
    CLASSES string or an integer tensor 0..5. Phi is the first table dimension.
    Table nodes are -179,-177,...,179 degrees. The 360-degree periodic seam is
    interpolated, without clipping the reference contour score. The score is
    not a normalized angle PDF or a probability of a particular conformation.
    """
    phi = torch.as_tensor(phi_degrees, dtype=torch.float64, device='cpu')
    psi = torch.as_tensor(psi_degrees, dtype=torch.float64, device='cpu')
    classes = _class_tensor(rama_class)
    phi, psi, classes = torch.broadcast_tensors(phi, psi, classes)
    if not bool(torch.isfinite(phi).all() and torch.isfinite(psi).all()):
        raise ValueError('phi/psi must be finite')
    x, y = torch.remainder(phi+180.,360.)-180., torch.remainder(psi+180.,360.)-180.
    x0 = torch.floor((x+179.)/2.)*2.-179.
    y0 = torch.floor((y+179.)/2.)*2.-179.
    ix = ((x0+179.)/2.).long().remainder(180)
    iy = ((y0+179.)/2.).long().remainder(180)
    jx, jy = (ix+1).remainder(180), (iy+1).remainder(180)
    tables = _tables()
    v1, v4 = tables[classes,ix,iy], tables[classes,jx,iy]
    v3, v2 = tables[classes,ix,jy], tables[classes,jx,jy]
    # Preserve upstream scitbx arithmetic order: two x interpolations, then y.
    v14 = v1+(v4-v1)*(x-x0)/2.
    v32 = v3+(v2-v3)*(x-x0)/2.
    score = v14+(v32-v14)*(y-y0)/2.
    return {'score':score, 'category':_classify_scores(score,classes)}


def _broadcast(value, shape, name, dtype=None):
    tensor = torch.as_tensor(value, device='cpu')
    if dtype == torch.bool and tensor.dtype != torch.bool:
        raise ValueError(f'{name} must be boolean')
    event_ndim = 2 if name == 'atom_mask' else 1
    expanded = None
    if len(shape) > tensor.ndim > event_ndim and tensor.shape[0] == shape[0]:
        expanded = tensor.reshape(tensor.shape[:1]+(1,)*(len(shape)-tensor.ndim)+tensor.shape[1:])
    try:
        regular = torch.broadcast_to(tensor, shape)
    except RuntimeError as error:
        if expanded is not None:
            try:
                return torch.broadcast_to(expanded, shape)
            except RuntimeError:
                pass
        raise ValueError(f'{name} shape {tuple(tensor.shape)} cannot broadcast to {shape}; '
                         'insert explicit sample/batch singleton axes') from error
    if expanded is not None and tensor.shape[0] > 1:
        try:
            torch.broadcast_to(expanded,shape)
        except RuntimeError:
            pass
        else:
            raise ValueError(f'ambiguous batch/sample {name}; insert explicit singleton axes')
    return regular


def _topology(peptide_edges, geometry_row_ids, n_residues):
    ids = _integer(geometry_row_ids, 'geometry_row_ids')
    if ids.shape != (n_residues,) or len(set(ids.tolist())) != n_residues:
        raise ValueError('geometry_row_ids must be a unique [L] vector')
    edges = _integer(peptide_edges, 'peptide_edges')
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError('peptide_edges must be [E,2] LOCAL-row C_i to N_j indices, including empty [0,2]')
    if bool(((edges < 0) | (edges >= n_residues)).any()):
        raise ValueError('peptide edge index outside local coordinate rows')
    previous, following = [-1]*n_residues, [-1]*n_residues
    seen = set()
    for i,j in edges.tolist():
        if i == j or (i,j) in seen:
            raise ValueError('self or duplicate peptide edge')
        seen.add((i,j))
        if following[i] != -1 or previous[j] != -1:
            raise ValueError('ambiguous topology: multiple incoming/outgoing peptide edges')
        following[i],previous[j] = j,i
    return ids, previous, following, len(seen)


def _dihedral(points):
    a,b,c,d = points
    axis = c-b
    norm = torch.linalg.vector_norm(axis)
    if float(norm) <= EPS_ANGSTROM:
        return None
    unit = axis/norm
    v,w = a-b,d-c
    v,w = v-(v*unit).sum()*unit,w-(w*unit).sum()*unit
    if float(torch.linalg.vector_norm(v)) <= EPS_ANGSTROM or float(torch.linalg.vector_norm(w)) <= EPS_ANGSTROM:
        return None
    return float(torch.rad2deg(torch.atan2((torch.linalg.cross(unit,v)*w).sum(),(v*w).sum())))


def _class_of(current, following, previous_omega):
    if current == GLY:
        return 1
    if current == PRO:
        if previous_omega is None:
            raise ValueError('PRO requires a valid preceding peptide omega')
        return 2 if -90. < previous_omega < 90. else 3
    if following == PRO:
        return 4
    return 5 if current in (ILE,VAL) else 0


def _counts(categories):
    n = len(categories)
    counts = dict(n_scored=n, favored_count=categories.count(2), allowed_only_count=categories.count(1),
                  outlier_count=categories.count(0), non_outlier_count=sum(c != 0 for c in categories))
    return {**counts, **{key.replace('_count','_fraction'):value/n if n else None
                        for key,value in counts.items() if key.endswith('_count')}}


@torch.no_grad()
def ramachandran_summary(bb, chemical_aatype, *, residue_mask=None, atom_mask=None,
                         peptide_edges, geometry_row_ids, include_per_residue=True):
    """Score [...,L,4,3] N/CA/C/O using actual directed peptide connectivity.

    Chemical indices are OpenFold ARNDCQEGHILKMFPSTWYV (20=unknown), never
    dataset embedding indices. Edges use LOCAL coordinate-row indices, while
    unique geometry_row_ids carry identity only. Crop/permutation must remap
    edges explicitly. Metadata supports unambiguous batch broadcasting; for
    ambiguous [B,S,L] use [B,1,L].
    Fractions use only scored residues. Across samples, ``mean`` is equal weight
    over nonempty samples; retain ``per_sample`` for protein-level aggregation.
    Exclusion counts are mutually exclusive in EXCLUSION_ORDER; issue counts
    describe missing torsion inputs/degeneracy within the reached category and
    may overlap. Unknown next identity matters only outside GLY/PRO priority.
    """
    coords = torch.as_tensor(bb, device='cpu')
    if not coords.is_floating_point() or coords.ndim < 3 or coords.shape[-2:] != (4,3) or coords.numel()==0:
        raise ValueError('expected nonempty floating [...,L,4,3] N/CA/C/O backbone')
    shape,n_residues = coords.shape[:-2],coords.shape[-3]
    sample_shape = shape[:-1]
    coords = coords.double()
    chemical = _integer(chemical_aatype,'chemical_aatype')
    chemical = _broadcast(chemical,shape,'chemical_aatype')
    valid = torch.ones(shape,dtype=torch.bool) if residue_mask is None else _broadcast(residue_mask,shape,'residue_mask',torch.bool)
    if bool((((chemical < 0) | (chemical > 20)) & valid).any()):
        raise ValueError('valid chemical_aatype must be OpenFold 0..20')
    chemical = torch.where(valid,chemical,20)
    atoms = valid[...,None].expand(*shape,4)
    if atom_mask is not None:
        atoms = atoms & _broadcast(atom_mask,(*shape,4),'atom_mask',torch.bool)
    if bool((~torch.isfinite(coords).all(-1) & atoms).any()):
        raise ValueError('nonfinite coordinate at a valid backbone atom')
    coords = torch.where(atoms[...,None],coords,0.)
    ids,previous,following,n_edges = _topology(peptide_edges,geometry_row_ids,n_residues)
    rows=[]
    for points,aa,mask,am in zip(coords.reshape(-1,n_residues,4,3),chemical.reshape(-1,n_residues),
                               valid.reshape(-1,n_residues),atoms.reshape(-1,n_residues,4)):
        exclusions={k:0 for k in EXCLUSION_ORDER}
        issues={k:0 for k in ISSUES}
        records=[]
        def exclude(record,reason):
            exclusions[reason]+=1
            record['excluded_reason']=reason
            records.append(record)
        for i in range(n_residues):
            p,n=previous[i],following[i]
            record=dict(row=i,geometry_row_id=int(ids[i]),chemical_aatype=int(aa[i]),
                previous_row=p if p>=0 else None,next_row=n if n>=0 else None,
                phi_degrees=None,psi_degrees=None,previous_omega_degrees=None,
                rama_class=None,score=None,category=None,excluded_reason=None)
            if not bool(mask[i]):
                exclude(record,'masked'); continue
            if p<0 or n<0:
                issues['missing_previous']+=int(p<0); issues['missing_next']+=int(n<0)
                exclude(record,'topology_incomplete'); continue
            if not bool(mask[p] and mask[n]):
                exclude(record,'neighbor_masked'); continue
            current,next_aa=int(aa[i]),int(aa[n])
            if current==20:
                exclude(record,'unknown_current'); continue
            if next_aa==20 and current not in (GLY,PRO):
                exclude(record,'unknown_next'); continue
            phi_atoms=((p,2),(i,0),(i,1),(i,2))
            psi_atoms=((i,0),(i,1),(i,2),(n,0))
            omega_atoms=((p,1),(p,2),(i,0),(i,1))
            needs={'phi':phi_atoms,'psi':psi_atoms}
            if current==PRO:
                needs['pro_omega']=omega_atoms
            missing=[name for name,quad in needs.items() if not all(bool(am[r,a]) for r,a in quad)]
            if missing:
                for name in missing: issues[name+'_atoms_missing']+=1
                exclude(record,'missing_atoms'); continue
            angles={name:_dihedral(torch.stack([points[r,a] for r,a in quad])) for name,quad in needs.items()}
            record.update(phi_degrees=angles['phi'],psi_degrees=angles['psi'],previous_omega_degrees=angles.get('pro_omega'))
            degenerate=[name for name,value in angles.items() if value is None]
            if degenerate:
                for name in degenerate: issues[name+'_degenerate']+=1
                exclude(record,'degenerate'); continue
            cls=_class_of(current,next_aa,angles.get('pro_omega'))
            value=score_angles(angles['phi'],angles['psi'],cls)
            record.update(rama_class=CLASSES[cls],score=float(value['score']),category=int(value['category']))
            records.append(record)
        categories=[r['category'] for r in records if r['category'] is not None]
        counts=_counts(categories)
        row={**counts,'n_residues':n_residues,'n_excluded':sum(exclusions.values()),
             'exclusion_counts':exclusions,'issue_counts':issues,
             'per_class':{c:_counts([r['category'] for r in records if r['rama_class']==c]) for c in CLASSES}}
        if include_per_residue: row['per_residue']=records
        rows.append(row)
    fractions=('favored_fraction','allowed_only_fraction','outlier_fraction','non_outlier_fraction')
    mean={k:sum(values)/len(values) if values else None
          for k in fractions for values in [[r[k] for r in rows if r[k] is not None]]}
    return dict(schema_version=VERSION,sample_shape=list(sample_shape),n_samples=len(rows),
        n_edges=n_edges,per_sample=rows,mean=mean,
        totals={k:sum(r[k] for r in rows) for k in ('n_residues','n_scored','n_excluded',
                    'favored_count','allowed_only_count','outlier_count','non_outlier_count')},
        conventions=dict(reference='pinned cctbx Top8000 compiled 6-significant-digit tables',
            table_sha256=TABLE_SHA256,classes=list(CLASSES),chemical_alphabet=CHEMICAL_ALPHABET,
            unknown_chemical_index=20,angle_unit='degrees',coordinate_unit='angstrom',
            categories={'0':'outlier','1':'allowed_only','2':'favored'},
            non_outlier='favored plus allowed_only',favored_score_min=FAVORED_THRESHOLD,
            allowed_only_score_min=dict(zip(CLASSES,ALLOWED_THRESHOLDS)),
            classification_priority='GLY; PRO via preceding omega; next PRO; ILE/VAL; general',
            cis_pro_rule='-90 < preceding CA-C-N-CA omega < 90 degrees; endpoints are trans',
            pro_missing_omega='excluded, unlike upstream missing-omega fallback to trans',
            topology='explicit LOCAL-row C_i to N_j edges; no row adjacency, chain-label, or distance inference',
            geometry_row_ids='unique caller-provided identities only; no connectivity semantics',
            score_kind='official reference contour score, not normalized angle PDF/conformation probability',
            exclusion_priority=list(EXCLUSION_ORDER),degeneracy_eps_angstrom=EPS_ANGSTROM,
            aggregation='equal nonempty sample fractions; retain per_sample for protein weighting',
            full_molprobity_pipeline=False))
