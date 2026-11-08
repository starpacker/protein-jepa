"""Actual-backbone losses with explicit directed peptide topology (v3).

Coordinates, chemical IDs, masks, reductions, flat-bottom constants and units
follow frozen ``backbone_objectives_v2``. Only topology changes: callers supply
``peptide_edges[E,2]`` in LOCAL coordinate-row indices, directed C_i -> N_j.
No connectivity is inferred from residue numbers, insertion codes, chain IDs,
storage order, distances, or geometry_row_ids. A genuine PSF edge across author
chains is retained. Cropping/permutation must explicitly remap the edge list.

``geometry_row_ids[L]`` are caller-provided unique identities for these rows,
e.g. original topology row IDs after a crop. They validate identity, not bonds;
we never silently renumber. One topology is shared by all leading sample axes
of ``backbone[...,L,4,3]``. Distinct topologies require separate calls.

Clashes enumerate every distinct residue-row pair and exclude exactly the
specified covalent C-N atom pairs, including both directed bonds if supplied.
Missing atoms/residues remove terms, never create a connection across the gap.
UNK policy, native units, differentiable empty terms, precision policy and
soft tolerance interpretation are unchanged from v2. This is not a geometry
pass/fail test or a complete all-atom/Ramachandran objective.
"""
from __future__ import annotations

import torch

from . import backbone_objectives_v2 as _v2


SCHEMA_VERSION = "backbone_objectives_v3.0_explicit_peptide_edges"
DEFAULT_CONNECTION_TOLERANCE = _v2.DEFAULT_CONNECTION_TOLERANCE
DEFAULT_CLASH_OVERLAP_ANGSTROM = _v2.DEFAULT_CLASH_OVERLAP_ANGSTROM


def _integers(value, *, device, name):
    value = torch.as_tensor(value, device=device)
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must contain integer IDs")
    return value.to(torch.long)


def _prepare(backbone, atom_mask, chemical_aatype, peptide_edges, geometry_row_ids,
             residue_mask):
    if (not torch.is_tensor(backbone) or backbone.ndim < 3
            or backbone.shape[-2:] != (4, 3) or backbone.numel() == 0
            or not backbone.is_floating_point()):
        raise ValueError("backbone must be nonempty floating [...,L,4,3] N/CA/C/O")
    shape, device = backbone.shape[:-2], backbone.device
    length = shape[-1]
    identities = _integers(geometry_row_ids, device=device, name="geometry_row_ids")
    if identities.shape != (length,) or identities.unique().numel() != length:
        raise ValueError("geometry_row_ids must be [L] unique row identities")
    edges = _integers(peptide_edges, device=device, name="peptide_edges")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("peptide_edges must have shape [E,2], including empty [0,2]")
    if bool(((edges < 0) | (edges >= length)).any()):
        raise ValueError("peptide edge index outside local coordinate rows")
    if bool((edges[:, 0] == edges[:, 1]).any()):
        raise ValueError("peptide self-edge is invalid")
    if edges.unique(dim=0).shape[0] != edges.shape[0]:
        raise ValueError("duplicate directed peptide edge")
    residue = (torch.ones(shape, dtype=torch.bool, device=device) if residue_mask is None
               else _v2._broadcast(residue_mask, shape, device=device,
                                   name="residue_mask", boolean=True))
    atoms = _v2._broadcast(atom_mask, (*shape, 4), device=device,
                            name="atom_mask", atom=True, boolean=True)
    atoms = atoms & residue[..., None]
    if bool((~torch.isfinite(backbone).all(-1) & atoms).any()):
        raise ValueError("nonfinite coordinate at a valid backbone atom")
    aa = _v2._broadcast(chemical_aatype, shape, device=device, name="chemical_aatype")
    if bool((((aa < 0) | (aa > 20)) & residue).any()):
        raise ValueError("chemical_aatype must use OpenFold IDs 0..19 or UNK=20")
    x = torch.where(atoms[..., None], backbone, 0)
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    # Each directed edge is one connection feature. The frozen coordinate
    # formula receives explicit source/target indices, with no inferred links.
    source, target = edges.unbind(-1)
    connected = residue[..., source] & residue[..., target]
    connection = (x, atoms, residue, aa, source, target, connected,
                  torch.zeros_like(connected))

    # Keep unique unordered residue pairs for clash enumeration, but distinguish
    # the two directions of their possible covalent C-N atom exclusions.
    i, j = torch.triu_indices(length, length, offset=1, device=device)
    adjacency = torch.zeros((length, length), device=device, dtype=torch.bool)
    adjacency[source, target] = True
    present = residue[..., i] & residue[..., j]
    clash = (x, atoms, residue, aa, i, j,
             present & adjacency[i, j], present & adjacency[j, i])
    return connection, clash, edges


def _result(prepared, parts, tolerance, overlap):
    connection, _, edges = prepared
    # Reuse the frozen chemistry formulas and their explicit unit/count contract.
    result = _v2._result(connection, parts, tolerance, overlap)
    result['schema_version'] = SCHEMA_VERSION
    result['conventions'].update(
        formula_version=_v2.SCHEMA_VERSION,
        topology='explicit local-row peptide_edges[E,2], directed C_i->N_j; no residue/chain/identity-number inference',
        geometry_row_ids='required unique caller-provided [L] integer row identities; no renumbering or connectivity semantics',
        shared_topology='one edge list and row identity list shared by all leading sample axes',
        n_topology_edges=edges.shape[0],
        clash_scope='all unique inter-residue N/CA/C/O atom pairs; exclude only explicit directed peptide C-N bonds',
    )
    return result


def backbone_connection_loss(backbone, *, chemical_aatype, atom_mask,
                             peptide_edges, geometry_row_ids, residue_mask=None,
                             tolerance_factor=DEFAULT_CONNECTION_TOLERANCE):
    tolerance = _v2._parameter(tolerance_factor, 'tolerance_factor')
    prepared = _prepare(backbone, atom_mask, chemical_aatype, peptide_edges,
                        geometry_row_ids, residue_mask)
    return _result(prepared, [_v2._connections(prepared[0], tolerance)], tolerance, None)


def backbone_nonbonded_clash_loss(backbone, *, chemical_aatype, atom_mask,
                                peptide_edges, geometry_row_ids, residue_mask=None,
                                overlap_tolerance_angstrom=DEFAULT_CLASH_OVERLAP_ANGSTROM):
    overlap = _v2._parameter(overlap_tolerance_angstrom, 'overlap_tolerance_angstrom')
    prepared = _prepare(backbone, atom_mask, chemical_aatype, peptide_edges,
                        geometry_row_ids, residue_mask)
    return _result(prepared, [_v2._clashes(prepared[1], overlap)], None, overlap)


def backbone_geometry_losses_v3(backbone, *, chemical_aatype, atom_mask,
                                peptide_edges, geometry_row_ids, residue_mask=None,
                                tolerance_factor=DEFAULT_CONNECTION_TOLERANCE,
                                overlap_tolerance_angstrom=DEFAULT_CLASH_OVERLAP_ANGSTROM):
    """Return four differentiable native-unit losses with explicit PSF topology."""
    tolerance = _v2._parameter(tolerance_factor, 'tolerance_factor')
    overlap = _v2._parameter(overlap_tolerance_angstrom, 'overlap_tolerance_angstrom')
    prepared = _prepare(backbone, atom_mask, chemical_aatype, peptide_edges,
                        geometry_row_ids, residue_mask)
    return _result(prepared, [_v2._connections(prepared[0], tolerance),
                             _v2._clashes(prepared[1], overlap)], tolerance, overlap)
