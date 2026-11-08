"""Frozen TRAIN-cache panel support from complete PSF peptide metadata.

The cache contains cropped topology, so full-graph eligibility is supplied by
a separately registered metadata preflight. Coordinates never select pairs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from protein_jepa_frame.training.physical_distance_pair_support_v2 import (
    canonical_sha256, record_key, require, sha256_file, validate_record_plan,
)
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'frozen_h_distance_support_v2'
HORIZONS = (1, 4, 16, 64)


def canonical_metadata(value):
    """Normalize tuple/list storage only, preserving JSON fields and item order.

    Frozen topology hashes already use JSON encoding: a live tuple and its
    serialized list have identical hashes. Reject non-JSON objects/keys and
    nonfinite numbers rather than coercing them into a different identity.
    """
    def validate(item):
        if isinstance(item,dict):
            require(all(type(k) is str for k in item), 'metadata keys must be strings')
            for child in item.values():validate(child)
        elif isinstance(item,(tuple,list)):
            for child in item:validate(child)
        else:
            require(item is None or type(item) in (str,int,float,bool), 'unsupported metadata value')
    validate(value)
    return json.loads(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))


def full_file_stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def registered_windows(selection, cache_protocol):
    """Resolve all500 ordered identities from already frozen JSON metadata."""
    require(selection.get('schema_version') == 'backbone_readout_cache_v1'
            and selection.get('mode') == 'train' and selection.get('seed') == 42
            and selection.get('n_windows') == selection.get('selected_prefix') == 500,
            'requires complete TRAIN500 selection')
    rows = selection['full_selection']
    cp = cache_protocol['protocol']
    require(len(rows) == 500 and cp.get('mode') == 'train' and cp.get('n_windows') == 500
            and cp.get('dataset_role') == 'training_readout_calibration'
            and cp.get('max_residues') == 96 and cp.get('n_forecast_samples') == 0,
            'cache panel contract differs')
    metadata = [r['metadata'] for r in rows]
    require(len({r['protein'] for r in metadata}) == 500
            and [r['horizon'] for r in metadata] == list(HORIZONS)*125,
            'requires500 distinct proteins in fixed balanced horizon order')
    topologies = cache_protocol['provenance']['topology_metadata']
    require(len(topologies) == 500, 'incomplete historical topology metadata')
    windows = []
    for index, (item, topology) in enumerate(zip(rows, topologies)):
        record, m = item['record'], item['metadata']
        require(canonical_sha256(record) == item['record_sha256'], 'registered record hash differs')
        require(topology['metadata'] == m and record['protein_id'] == m['protein']
                and record['trajectory_id'] == m['trajectory']
                and record['temperature_k'] == m['temperature_k']
                and record['format'] == 'mdcath-hdf5' and record['frame_interval_ps'] == 1000.
                and m['target_frame'] == m['anchor_frame']+m['horizon']
                and m['horizon_ps'] == m['horizon']*1000.
                and m['context_frames'] == list(range(m['anchor_frame']-7, m['anchor_frame']+1))
                and m['context_frames'][0] >= 0 and m['target_frame'] < record['num_frames']
                and m['sampling_seed'] is None, 'registered window/record identity differs')
        path = str(Path(record['path']).resolve())
        require(path == record['path'] and topology['source_identity']['path'] == path
                and topology['source_identity']['protein_id'] == m['protein'], 'topology record path differs')
        windows.append(dict(cache_row_index=index, split='fit' if index < 400 else 'readout_heldout',
            metadata=copy.deepcopy(m), record=copy.deepcopy(record),
            record_key=record_key(path, m['protein']), historical_topology=copy.deepcopy(topology)))
    require(len({w['record_key'] for w in windows}) == 500, 'duplicate physical record plan')
    return windows


def cropped_topology(record_support, historical):
    """Rebuild the frozen cache view while retaining full graph separately."""
    full = record_support['full_topology']
    identity = canonical_metadata({k:v for k,v in full['source_identity'].items() if k != 'raw_identity_audit'})
    require(canonical_sha256(identity) == canonical_sha256(canonical_metadata(historical['source_identity']))
            and canonical_sha256(full['metadata_hashes']) == canonical_sha256(historical['metadata_hashes']),
            'full metadata differs from historical cache topology')
    ids = record_support['selected_geometry_row_ids']
    mapping = {old:new for new,old in enumerate(ids)}
    keep = [i for i,(a,b) in enumerate(full['peptide_edges']) if a in mapping and b in mapping]
    result = dict(schema_version=full['schema_version'], source_identity=copy.deepcopy(identity),
                  metadata_hashes=copy.deepcopy(full['metadata_hashes']))
    for k in ('residue_uids','residue_names','embedding_aatype','embedding_residue_index',
              'embedding_chain_id','chemical_aatype','atom_indices','atom_mask','geometry_row_ids'):
        result[k] = [copy.deepcopy(full[k][i]) for i in ids]
    result['peptide_edges'] = [[mapping[a],mapping[b]] for i in keep for a,b in [full['peptide_edges'][i]]]
    result['peptide_atom_indices'] = [full['peptide_atom_indices'][i] for i in keep]
    require(canonical_sha256(result) == historical['topology_sha256'],
            'rebuilt prefix topology differs from cached topology SHA')
    return canonical_metadata(result)


def make_plan(window, record_support):
    validate_record_plan(record_support)
    require(record_support['record_key'] == window['record_key']
            and record_support['protein'] == window['metadata']['protein']
            and record_support['record_path'] == window['record']['path']
            and record_support['max_residues'] == 96, 'wrong metadata record support')
    plan = canonical_metadata(dict(cache_row_index=window['cache_row_index'], split=window['split'],
        metadata=copy.deepcopy(window['metadata']), record_support=copy.deepcopy(record_support),
        cache_topology=cropped_topology(record_support, window['historical_topology']),
        historical_topology=copy.deepcopy(window['historical_topology'])))
    plan['plan_sha256'] = canonical_sha256(plan)
    return plan


def validate_plan(plan):
    require(plan['plan_sha256'] == canonical_sha256({k:v for k,v in plan.items() if k != 'plan_sha256'}),
            'cached-row plan hash differs')
    graph = validate_record_plan(plan['record_support'])
    require(plan['cache_topology'] == cropped_topology(plan['record_support'],plan['historical_topology']),
            'cached topology view differs')
    require(plan['metadata'] == plan['historical_topology']['metadata'], 'plan window metadata differs')
    return graph


def verify_provenance(protocol):
    root = Path.cwd().resolve()
    for path,digest in protocol['input_hashes'].items():
        require(sha256_file(path) == digest, 'input changed: '+path)
    for path,digest in protocol['source_hashes'].items():
        source = Path(path)
        require(not source.is_absolute() and source.resolve().is_relative_to(root), 'source escapes repository')
        snap = Path(protocol['source_snapshot'])/source
        require(snap.resolve().is_relative_to(Path(protocol['source_snapshot']).resolve()), 'snapshot escapes root')
        require(sha256_file(source) == sha256_file(snap) == digest, 'source/snapshot changed: '+path)
    for path,expected in protocol['selected_hdf5_stats'].items():
        require(full_file_stat(path) == expected, 'HDF metadata stat changed: '+path)


def load_support(path, expected_sha):
    """Fail closed unless every registered TRAIN500 metadata record completed."""
    path = Path(path)
    require(sha256_file(path) == expected_sha, 'support file hash differs')
    doc = json.loads(path.read_text())
    require(doc['schema_version'] == VERSION and doc['status'] == 'completed' and not doc['errors'],
            'support preflight incomplete or failed')
    require(all(doc.get(k) is False for k in ('trajectory_coordinates_loaded','cache_deserialized',
                'model_loaded','validation_loaded','test_opened')), 'metadata-only boundary differs')
    protocol_path = path.parent/'protocol.json'
    require(Path(doc['protocol_path']).resolve() == protocol_path.resolve()
            and sha256_file(protocol_path) == doc['protocol_sha256'], 'support protocol differs')
    protocol = json.loads(protocol_path.read_text())
    require(protocol['schema_version'] == VERSION, 'wrong support protocol schema')
    require(protocol.get('max_residues') == 96 and protocol.get('n_records') == 500
            and protocol.get('n_distinct_proteins') == 500
            and protocol.get('fit_indices') == list(range(400))
            and protocol.get('readout_heldout_indices') == list(range(400,500)), 'registered panel budget differs')
    execution = json.loads((path.parent/'execution.json').read_text())
    started = json.loads((path.parent/'started.json').read_text())
    require(execution.get('status') == 'completed' and execution.get('returncode') == 0
            and not execution.get('errors') and execution.get('support_sha256') == expected_sha
            and execution.get('pid',0) > 0
            and all(execution.get(k) == started.get(k) for k in ('pid','command','started_utc','protocol_sha256'))
            and execution.get('protocol_sha256') == doc['protocol_sha256']
            and isinstance(execution.get('command'),list) and bool(execution['command'])
            and bool(execution.get('started_utc')) and bool(execution.get('completed_utc'))
            and isinstance(execution.get('seconds'),(int,float)) and execution['seconds'] >= 0,
            'support lacks linked successful actual execution')
    verify_provenance(protocol)
    windows = protocol['windows']
    require(len(windows) == len(doc['plans']) == doc['completed_record_count'] == doc['registered_record_count'] == 500
            and canonical_sha256(windows) == protocol['windows_sha256'], 'incomplete500 support')
    require([w['cache_row_index'] for w in windows] == list(range(500))
            and len({w['metadata']['protein'] for w in windows}) == 500
            and [w['metadata']['horizon'] for w in windows] == list(HORIZONS)*125
            and [w['split'] for w in windows] == ['fit']*400+['readout_heldout']*100,
            'panel split/identity schedule differs')
    require(protocol['training_order'] == list(range(400))+list(range(100)), 'fixed500 update order differs')
    for window,plan in zip(windows,doc['plans']):
        expected = make_plan(window,plan['record_support'])
        require(plan == expected, 'support row differs from registration')
        rs = plan['record_support'];stat = protocol['selected_hdf5_stats'][rs['record_path']]
        require(rs['hdf_stat'] == {k:stat[k] for k in ('size','mtime_ns')}, 'record HDF stat differs')
        validate_plan(plan)
    return doc


def validate_cached_row(row, plan):
    """Check saved metadata/coordinates; return fixed observed-valid PSF pairs.

    No cached H is consumed, no tensor is mutated, and no distance/target variance
    or prediction determines eligibility. Nonfinite observed atoms fail visibly.
    """
    graph = validate_plan(plan)
    require(row['metadata'] == plan['metadata'], 'cached window metadata differs')
    topology = plan['cache_topology']
    actual_topology = canonical_metadata(row['topology'])
    require(canonical_sha256(actual_topology) == canonical_sha256(dict(sha256=canonical_sha256(topology),**topology)),
            'cached topology differs')
    n = len(topology['geometry_row_ids'])
    for name in ('embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype',
                 'geometry_row_ids','peptide_edges','atom_mask'):
        expected = torch.tensor(topology[name],dtype=torch.bool if name=='atom_mask' else torch.long)
        if name=='peptide_edges':expected=expected.reshape(-1,2)
        value = row[name]
        require(value.device.type == 'cpu' and value.dtype == expected.dtype
                and torch.equal(value,expected), 'cached '+name+' differs')
    mask = row['observed_mask']
    require(mask.shape == (n,) and mask.dtype == torch.bool and mask.device.type == 'cpu'
            and int(mask.sum()) >= 4 and bool(row['atom_mask'][mask].all()), 'invalid fixed observed support')
    target_mask = row['target_atom_mask']
    require(target_mask.shape == (n,4) and target_mask.dtype == torch.bool and target_mask.device.type == 'cpu'
            and not bool((target_mask & ~row['atom_mask']).any()) and bool(target_mask[mask].all()),
            'target mask lacks fixed observed backbone support')
    for name in ('observed_backbone','target_backbone'):
        value = row[name]
        require(value.shape == (1,n,4,3) and value.dtype == torch.float32 and value.device.type == 'cpu'
                and bool(torch.isfinite(value[:,mask]).all()), 'invalid fixed '+name)
    pairs = torch.nonzero(torch.triu(graph & mask[:,None] & mask[None,:],diagonal=1),as_tuple=False)
    require(len(pairs) > 0, 'empty fixed observed PSF pair support')
    audit = dict(cache_row_index=plan['cache_row_index'],split=plan['split'],plan_sha256=plan['plan_sha256'],
        record_plan_sha256=plan['record_support']['record_plan_sha256'],pair_count=len(pairs),
        observed_residue_count=int(mask.sum()),support_tensor_sha256=state_sha256(dict(
            pair_indices=pairs,observed_mask=mask,geometry_row_ids=row['geometry_row_ids'])))
    return dict(pair_indices=pairs,observed_mask=mask.clone(),audit=audit)


def attach_cached_physical_support(row,index,support):
    require(support.get('status') == 'completed' and not support.get('errors')
            and len(support['plans']) == 500 and type(index) is int and 0 <= index < 500,
            'complete support and valid cache index required')
    plan = support['plans'][index]
    require(plan['cache_row_index'] == index, 'support row order differs')
    return validate_cached_row(row,plan)
