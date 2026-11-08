"""Metadata-only PSF pair support; legacy model IDs and coordinates stay intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'physical_distance_pair_support_v2'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns)


def record_key(path, protein):
    return canonical_sha256(dict(path=str(Path(path).resolve()), protein=str(protein)))


def build_pair_eligibility(geometry_row_ids, full_residue_uids, full_peptide_edges):
    """Symmetric eligibility on selected rows, using the entire peptide graph.

    IDs index the full UID list. One/two undirected peptide hops are excluded;
    disconnected residues remain eligible. No crop or observed mask enters the
    graph, and author numbering never defines adjacency.
    """
    n = len(full_residue_uids)
    require(n > 0, 'empty full residue topology')
    fingerprints = [canonical_sha256(uid) for uid in full_residue_uids]
    require(len(set(fingerprints)) == n, 'duplicate full residue UID')
    ids = torch.as_tensor(geometry_row_ids)
    require(ids.ndim == 1 and ids.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
            'selected geometry IDs must be one-dimensional integers')
    ids = ids.long().cpu()
    require(bool(((ids >= 0) & (ids < n)).all()) and len(ids.unique()) == len(ids),
            'invalid or duplicate selected geometry IDs')
    edges = torch.as_tensor(full_peptide_edges)
    if edges.numel() == 0:
        edges = torch.empty((0, 2), dtype=torch.long)
    require(edges.ndim == 2 and edges.shape[1] == 2
            and edges.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
            'peptide edges must be integer [E,2]')
    adjacency = [set() for _ in range(n)]
    seen = set()
    for a, b in edges.long().cpu().tolist():
        require(0 <= a < n and 0 <= b < n and a != b, 'invalid peptide edge')
        pair = tuple(sorted((a, b)))
        require(pair not in seen, 'duplicate undirected peptide edge')
        seen.add(pair)
        adjacency[a].add(b)
        adjacency[b].add(a)
    eligible = torch.zeros((len(ids), len(ids)), dtype=torch.bool)
    for i, node in enumerate(ids.tolist()):
        near = {node} | adjacency[node]
        for middle in adjacency[node]:
            near = near | adjacency[middle]
        eligible[i] = torch.tensor([other not in near for other in ids.tolist()], dtype=torch.bool)
    require(torch.equal(eligible, eligible.T) and not bool(eligible.diag().any()),
            'invalid graph eligibility')
    return eligible


def validate_raw_pdb_identity(pdb_text):
    """Reject residue/atom collisions before the frozen parser can merge them.

    Distinct alternate-location labels are allowed. The legacy blank/A atom
    selection must still be unique: blank plus A for the same atom is ambiguous
    and fails rather than selecting whichever record appears first.
    """
    if isinstance(pdb_text, bytes):
        pdb_text = pdb_text.decode('utf-8', errors='strict')
    require(isinstance(pdb_text, str), 'PDB metadata must be UTF-8 text')
    blocks, atoms, effective, names = set(), set(), set(), {}
    previous = None
    for line in pdb_text.splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        require(len(line) >= 76, 'PDB record lacks full identity columns')
        uid = (line[72:76].strip(), line[21:22].strip() or '_',
               int(line[22:26]), line[26:27].strip())
        if uid != previous:
            require(uid not in blocks, 'repeated noncontiguous full residue UID')
            blocks.add(uid)
            previous = uid
        name = line[17:20].strip().upper()
        require(uid not in names or names[uid] == name, 'conflicting residue name for full UID')
        names[uid] = name
        atom, altloc = line[12:16].strip(), line[16:17].strip()
        key = (uid, atom, altloc)
        require(key not in atoms, 'duplicate raw atom identity for full residue UID')
        atoms.add(key)
        if atom in ('N','CA','C','O') and altloc in ('','A'):
            key = (uid, atom)
            require(key not in effective, 'ambiguous duplicate selected backbone atom for full residue UID')
            effective.add(key)
    require(bool(blocks), 'empty protein PDB metadata')
    return dict(full_uid_count=len(blocks), raw_atom_count=len(atoms),
                selected_backbone_atom_count=len(effective), raw_identity_guard=VERSION)


def support_record(record, max_residues=96):
    """Read PDB/PSF metadata through the frozen, legacy-parity-checked parser."""
    import h5py
    from protein_jepa_frame.data.backbone_topology_v2 import parse_mdcath_backbone_topology
    require(record.format.lower() == 'mdcath-hdf5', 'requires mdCATH topology metadata')
    path = record.path.resolve()
    before = file_stat(path)
    domain = record.domain_id or record.protein_id
    metadata = record.metadata or {}
    pdb_path = str(metadata.get('protein_pdb_dataset') or f'{domain}/pdbProteinAtoms').lstrip('/')
    psf_path = str(metadata.get('protein_psf_dataset') or f'{domain}/psf').lstrip('/')
    with h5py.File(path, 'r') as handle:
        pdb, psf = handle[pdb_path][()], handle[psf_path][()]
        count = metadata.get('atom_count')
        if count is None:
            require(len(handle[f'{domain}/resid'].shape) == 1, 'resid metadata must be one-dimensional')
            count = len(handle[f'{domain}/resid'])
    raw_audit = validate_raw_pdb_identity(pdb)
    topology = parse_mdcath_backbone_topology(pdb, psf, source_path=str(path),
        domain_id=domain, expected_atom_count=count)
    require(topology.n_residues == raw_audit['full_uid_count'], 'raw/parser full UID counts differ')
    require(file_stat(path) == before, 'HDF metadata changed during parsing')
    topology.source_identity.update(protein_id=record.protein_id, pdb_dataset=pdb_path,
                                    psf_dataset=psf_path, raw_identity_audit=raw_audit)
    full = topology.as_metadata()
    ids = topology.geometry_row_ids[:max_residues]
    require(torch.equal(topology.geometry_row_ids, torch.arange(topology.n_residues)),
            'full geometry ID axis differs from UID order')
    eligibility = build_pair_eligibility(ids, full['residue_uids'], topology.peptide_edges)
    pairs = torch.nonzero(torch.triu(eligibility, diagonal=1), as_tuple=False)
    require(len(pairs) > 0, 'no eligible pairs before observed masking')
    result = dict(record_key=record_key(record.path, record.protein_id),
        record_path=str(record.path.resolve()), protein=record.protein_id,
        domain_id=record.domain_id or record.protein_id, max_residues=max_residues,
        hdf_stat=file_stat(record.path),
        full_topology=full, full_topology_sha256=topology.topology_sha256,
        selected_geometry_row_ids=ids.tolist(), pair_indices=pairs.tolist(), pair_count=len(pairs),
        legacy_mapping={key: full[key][:len(ids)] for key in
            ('embedding_aatype', 'embedding_residue_index', 'embedding_chain_id', 'atom_indices', 'atom_mask')})
    result['record_plan_sha256'] = canonical_sha256(result)
    return result


def validate_record_plan(plan):
    content = {k: v for k, v in plan.items() if k != 'record_plan_sha256'}
    require(canonical_sha256(content) == plan['record_plan_sha256'], 'record plan changed')
    full = plan['full_topology']
    require(canonical_sha256(full) == plan['full_topology_sha256'], 'full topology changed')
    require(plan['record_key'] == record_key(plan['record_path'], plan['protein']), 'record key differs')
    ids = plan['selected_geometry_row_ids']
    require(ids == full['geometry_row_ids'][:plan['max_residues']]
            and full['geometry_row_ids'] == list(range(len(full['residue_uids']))), 'prefix geometry IDs differ')
    require(set(plan['legacy_mapping']) == {'embedding_aatype','embedding_residue_index',
            'embedding_chain_id','atom_indices','atom_mask'}, 'incomplete legacy mapping')
    for key, value in plan['legacy_mapping'].items():
        require(value == full[key][:len(ids)], 'legacy mapping differs from full topology')
    eligibility = build_pair_eligibility(ids, full['residue_uids'], full['peptide_edges'])
    pairs = torch.nonzero(torch.triu(eligibility, diagonal=1), as_tuple=False).tolist()
    require(pairs == plan['pair_indices'] and len(pairs) == plan['pair_count'] and len(pairs) > 0,
            'stored pair support differs from full graph')
    return eligibility


def verify_provenance(protocol):
    root = Path.cwd().resolve()
    for path in protocol['source_hashes']:
        source = Path(path)
        require(not source.is_absolute() and source.resolve().is_relative_to(root),
                'source path outside repository')
        require((Path(protocol['source_snapshot'])/path).resolve().is_relative_to(
                Path(protocol['source_snapshot']).resolve()), 'snapshot path escapes root')
    for path, digest in {**protocol['input_hashes'], **protocol['source_hashes']}.items():
        require(sha256_file(path) == digest, 'frozen file changed: ' + path)
    for path, digest in protocol['source_hashes'].items():
        require(sha256_file(Path(protocol['source_snapshot']) / path) == digest, 'source snapshot changed: ' + path)
    for path, expected in protocol['selected_hdf5_stats'].items():
        require(file_stat(path) == expected, 'HDF stat changed: ' + path)


def load_support(path, expected_sha):
    """Verify successful all500 metadata support before constructing any model."""
    require(sha256_file(path) == expected_sha, 'support file hash differs')
    document = json.loads(Path(path).read_text())
    require(document['schema_version'] == VERSION and document['status'] == 'completed'
            and not document['errors'], 'support preflight did not complete')
    execution = json.loads((Path(path).parent/'execution.json').read_text())
    started = json.loads((Path(path).parent/'started.json').read_text())
    require(execution.get('status') == 'completed' and execution.get('returncode') == 0
            and execution.get('pid', 0) > 0 and execution.get('support_sha256') == expected_sha
            and all(execution.get(k) == started.get(k) for k in ('pid','command','started_utc','protocol_sha256'))
            and execution.get('protocol_sha256') == document['protocol_sha256']
            and isinstance(execution.get('command'), list) and bool(execution['command'])
            and bool(execution.get('started_utc')) and bool(execution.get('completed_utc'))
            and isinstance(execution.get('seconds'), (int,float)) and execution['seconds'] >= 0,
            'support execution lacks actual successful completion')
    protocol_path = Path(document['protocol_path'])
    require(sha256_file(protocol_path) == document['protocol_sha256'], 'support protocol changed')
    protocol = json.loads(protocol_path.read_text())
    verify_provenance(protocol)
    require(document['training_windows'] == protocol['training_windows']
            and len(document['training_windows']) == 500
            and [r['step'] for r in document['training_windows']] == list(range(1,501))
            and canonical_sha256(document['training_windows']) == protocol['training_windows_sha256'],
            'incomplete fixed500 support')
    expected = {p['record_key'] for p in protocol['record_plans']}
    require(set(document['records']) == expected, 'record coverage differs')
    require(document['registered_record_count'] == document['completed_record_count'] == len(expected),
            'record completion counts differ')
    for registered in protocol['record_plans']:
        plan = document['records'][registered['record_key']]
        require(all(plan[key] == registered[key] for key in ('record_key','record_path','protein','domain_id'))
                and plan['max_residues'] == protocol['max_residues']
                and plan['hdf_stat'] == protocol['selected_hdf5_stats'][plan['record_path']],
                'support record differs from registered metadata')
        validate_record_plan(plan)
    return document


def attach_physical_pair_support(batch, support_document, *, identities):
    """Attach separate support tensors; preserve all original batch objects.

    For padded batches, sentinel geometry ID -1 and false pair eligibility stay
    outside each real prefix. Masks affect only the later loss, never the graph.
    """
    require(support_document.get('status') == 'completed' and not support_document.get('errors'),
            'successful support preflight required')
    require('geometry_row_ids' not in batch and 'physical_pair_eligibility' not in batch,
            'physical support already attached')
    identities = [identities] if isinstance(identities, dict) else list(identities)
    aa = batch['aatype']
    require(aa.ndim == 2 and len(identities) == aa.shape[0], 'batch/identity dimensions differ')
    b, length = aa.shape
    require(batch['coords'].ndim == 5 and batch['coords'].shape[0] == b
            and batch['coords'].shape[2:] == (length,4,3)
            and batch['atom_mask'].shape == batch['coords'].shape[:-1]
            and batch['residue_mask'].shape == batch['coords'].shape[:3], 'batch prefix dimensions differ')
    ids = torch.full((b, length), -1, dtype=torch.long, device=aa.device)
    eligible = torch.zeros((b, length, length), dtype=torch.bool, device=aa.device)
    audits = []
    for i, identity in enumerate(identities):
        step = identity['step']
        require(type(step) is int and 1 <= step <= len(support_document['training_windows']), 'unregistered training step')
        expected = support_document['training_windows'][step - 1]
        metadata = {k: v for k, v in identity.items() if k not in ('actual_batch_sha256', 'file_stat')}
        require(metadata == expected, 'actual window metadata differs from fixed registration')
        key = record_key(identity['record_path'], identity['protein'])
        plan = support_document['records'][key]
        graph = validate_record_plan(plan)
        require(identity.get('file_stat') == plan['hdf_stat'], 'actual batch HDF stat differs')
        n = len(plan['selected_geometry_row_ids'])
        require(n <= length and int(batch['lengths'][i]) == n
                and batch['protein_id'][i] == identity['protein']
                and batch['trajectory_id'][i] == identity['trajectory'], 'actual batch identity/prefix differs')
        static_atoms = torch.tensor(plan['legacy_mapping']['atom_mask'], dtype=torch.bool,
                                    device=batch['atom_mask'].device)
        require(not bool((batch['atom_mask'][i,:,:n] & ~static_atoms).any())
                and not bool(batch['atom_mask'][i,:,n:].any())
                and not bool(batch['residue_mask'][i,:,n:].any()), 'actual mask exceeds static atom/prefix support')
        for field, source in [('aatype','embedding_aatype'), ('residue_index','embedding_residue_index'),
                              ('chain_id','embedding_chain_id')]:
            value = batch[field]
            require(value.shape == (b, length) and value.dtype == torch.long and value.device == aa.device,
                    'legacy embedding shape/dtype/device differs')
            expected_value = torch.tensor(plan['legacy_mapping'][source], dtype=torch.long, device=aa.device)
            require(torch.equal(value[i, :n], expected_value), 'legacy ' + field + ' differs')
            padding = 20 if field == 'aatype' else -1
            require(bool((value[i, n:] == padding).all()), 'legacy padding differs')
        ids[i, :n] = torch.tensor(plan['selected_geometry_row_ids'], dtype=torch.long, device=aa.device)
        eligible[i, :n, :n] = graph.to(aa.device)
        audits.append(dict(record_plan_sha256=plan['record_plan_sha256'], pair_count=plan['pair_count'],
            record_key=key, physical_metadata_sha256=state_sha256(dict(
                geometry_row_ids=ids[i:i+1], physical_pair_eligibility=eligible[i:i+1]))))
    return {**batch, 'geometry_row_ids': ids, 'physical_pair_eligibility': eligible}, audits


def attach_physical_support(batch, identity, support):
    require(batch['aatype'].shape[0] == 1, 'fixed training attachment requires B=1')
    augmented, audits = attach_physical_pair_support(batch, support, identities=identity)
    return augmented, audits[0]
