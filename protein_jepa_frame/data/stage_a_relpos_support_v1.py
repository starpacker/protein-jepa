"""Pure TRAIN400 native-history/single-endpoint adapter; no readers or models.

The caller first verifies the complete frozen TRAIN500 support with its frozen
loader. This module binds those plans to the registered TrajectoryRecords;
plans alone omit the time origin. Only the first400 fit proteins are eligible.
No H, alignment, pair resampling, full future clip or h>=K rule is introduced.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch

from protein_jepa_frame.data import frozen_h_distance_support_v2 as support
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.manifest import TrajectoryRecord
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.prepare_backbone_readout_cache import WindowSpec, verify_clip

VERSION = 'stage_a_relpos_support_v1'
CLIP_KEYS = ('coords', 'atom_mask', 'residue_mask', 'aatype', 'residue_index',
             'chain_id', 'time_ps', 'temperature_k', 'dataset', 'protein_id', 'trajectory_id')
PAIR_KEYS = ('context', 'future', 'horizon_frames', 'horizon_time_ps', 'geometry_row_ids',
             'adaptation_metadata')
require = support.require


def fingerprint(value):
    """Hash JSON metadata and tensor bytes (including masked NaN bit patterns)."""
    leaves = {}
    def walk(item, path):
        if isinstance(item, dict):
            require(all(type(k) is str for k in item), 'fingerprint keys must be strings')
            leaves[path + '/keys'] = sorted(item)
            for key in sorted(item):
                walk(item[key], path + '/' + repr(key))
        elif isinstance(item, (list, tuple)):
            leaves[path + '/length'] = len(item)
            for i, child in enumerate(item):
                walk(child, path + '/' + str(i))
        else:
            leaves[path] = item
    walk(value, '')
    return state_sha256(leaves)


def _clip_view(clip):
    return {key: clip[key] for key in CLIP_KEYS}


def _window_spec(window):
    record = copy.deepcopy(window['record'])
    record['path'] = Path(record['path'])
    record = TrajectoryRecord(**record)
    m = window['metadata']
    require(record.format == 'mdcath-hdf5' and record.frame_interval_ps == 1000.
            and type(record.num_frames) is int and record.num_frames > 0
            and math.isfinite(record.first_frame_time_ps)
            and record.temperature_k is not None and math.isfinite(record.temperature_k)
            and record.temperature_k > 0, 'registered record/time contract differs')
    require(type(m['anchor_frame']) is int and m['anchor_frame'] >= 7
            and type(m['horizon']) is int and m['horizon'] in support.HORIZONS,
            'invalid anchor/horizon')
    spec = WindowSpec(record, m['anchor_frame'], m['horizon'])
    require(spec.target_frame < record.num_frames and spec.metadata('train') == m,
            'registered WindowSpec differs')
    require(str(record.path) == str(record.path.resolve())
            and window['record_key'] == support.record_key(record.path, record.protein_id),
            'registered record path/key differs')
    return spec


def select_adaptation_plans(verified_support, support_protocol):
    """Bind the entire verified metadata panel, then clone its exact first400.

    No files are read here. ``verified_support`` must already have passed
    frozen_h_distance_support_v2.load_support, including external provenance.
    The caller must hash-check the exact protocol file against the support's
    protocol_sha256 before parsing/passing it: a bare dict cannot authenticate
    that file SHA, and cache plans do not contain the record's time origin.
    The protocol is supplied explicitly, never reconstructed from coordinates.
    """
    before = fingerprint((verified_support, support_protocol))
    d, p = verified_support, support_protocol
    require(d.get('schema_version') == p.get('schema_version') == support.VERSION
            and d.get('status') == 'completed' and not d.get('errors')
            and d.get('registered_record_count') == d.get('completed_record_count') == 500,
            'complete verified TRAIN500 support required')
    plans, windows = d['plans'], p['windows']
    require(len(plans) == len(windows) == 500
            and support.canonical_sha256(windows) == p['windows_sha256']
            and p.get('n_records') == p.get('n_distinct_proteins') == 500
            and p.get('max_residues') == 96
            and p.get('fit_indices') == list(range(400))
            and p.get('readout_heldout_indices') == list(range(400, 500)),
            'registered whole500 panel/split differs')
    require(len({w['metadata']['protein'] for w in windows}) == 500
            and [w['metadata']['horizon'] for w in windows] == list(support.HORIZONS)*125,
            'whole500 protein/horizon order differs')
    selected = []
    for index, (plan, window) in enumerate(zip(plans, windows)):
        expected_split = 'fit' if index < 400 else 'readout_heldout'
        require(plan['cache_row_index'] == window['cache_row_index'] == index
                and plan['split'] == window['split'] == expected_split,
                'whole500 index/split order differs')
        _window_spec(window)
        require(support.make_plan(window, plan['record_support']) == plan,
                'plan differs from registered window')
        if index < 400:
            bound = dict(schema_version=VERSION, cache_plan=copy.deepcopy(plan),
                         window=copy.deepcopy(window),
                         support_protocol_sha256=d['protocol_sha256'],
                         registered_windows_sha256=p['windows_sha256'])
            bound['adaptation_plan_sha256'] = support.canonical_sha256(bound)
            selected.append(bound)
    require(fingerprint((verified_support, support_protocol)) == before, 'panel inputs mutated')
    return selected


def _validate_bound(bound):
    require(bound.get('schema_version') == VERSION and bound['adaptation_plan_sha256'] ==
            support.canonical_sha256({k:v for k,v in bound.items() if k != 'adaptation_plan_sha256'}),
            'adaptation plan fingerprint differs')
    plan, window = bound['cache_plan'], bound['window']
    index = plan['cache_row_index']
    require(type(index) is int and 0 <= index < 400 and window['cache_row_index'] == index
            and plan['split'] == window['split'] == 'fit', 'adaptation excludes last100/non-fit rows')
    require(plan == support.make_plan(window, plan['record_support']), 'bound support/window differs')
    return plan, _window_spec(window)


def _validate_clip(clip, topology, spec, endpoint):
    n, t = topology.n_residues, 1 if endpoint else 8
    types = dict(coords=torch.float32, atom_mask=torch.bool, residue_mask=torch.bool,
                 aatype=torch.long, residue_index=torch.long, chain_id=torch.long,
                 time_ps=torch.float32, temperature_k=torch.float32)
    for name, dtype in types.items():
        x = clip[name]
        require(isinstance(x, torch.Tensor) and x.device.type == 'cpu' and x.dtype == dtype
                and not x.requires_grad, 'clip must contain CPU dataset tensors: '+name)
    require(clip['residue_mask'].shape == (t,n)
            and clip['time_ps'].shape == (t,)
            and clip['temperature_k'].shape == ()
            and all(clip[k].shape == (n,) for k in ('aatype','residue_index','chain_id')),
            'clip identity/mask/time shapes differ')
    require(clip['protein_id'] == spec.protein_id and clip['trajectory_id'] == spec.trajectory_id
            and clip['dataset'] == (spec.record.dataset or 'unknown')
            and torch.equal(clip['temperature_k'], torch.tensor(spec.record.temperature_k,dtype=torch.float32)),
            'clip record identity/temperature differs')
    verify_clip(clip, topology, spec, endpoint=endpoint)


def prepare_adaptation_row(plan, context_clip, endpoint_clip, topology):
    """Validate eight native frames and one endpoint; return a detached copy.

    All eight frames retain the same complete observed backbone support.
    Partial/invisible residues are zeroed in every atom coordinate for the
    finite-input encoder; their original embedding IDs/UIDs are not changed.
    Missing atoms at a previously valid residue fail, never shrink the support.
    """
    inputs = dict(plan=plan, context=_clip_view(context_clip), endpoint=_clip_view(endpoint_clip),
                  topology=topology.as_metadata())
    before = fingerprint(inputs)
    cache, spec = _validate_bound(plan)
    expected = cache['cache_topology']
    require(support.canonical_metadata(topology.as_metadata()) == expected,
            'actual prefix topology differs from registered cache topology')
    require(4 <= topology.n_residues <= 96, 'registered prefix length must be4..96')
    for key in ('embedding_aatype','embedding_residue_index','embedding_chain_id',
                'geometry_row_ids','atom_mask'):
        value = getattr(topology,key)
        dtype = torch.bool if key == 'atom_mask' else torch.long
        require(value.device.type == 'cpu' and value.dtype == dtype
                and torch.equal(value,torch.tensor(expected[key],dtype=dtype)),
                'topology tensor differs: '+key)
    _validate_clip(context_clip,topology,spec,False)
    _validate_clip(endpoint_clip,topology,spec,True)
    times = context_clip['time_ps']
    require(bool(torch.isfinite(times).all())
            and torch.equal(torch.diff(times),torch.full((7,),1000.,dtype=torch.float32)),
            'actual context must have seven exact1000ps intervals')
    mask = context_clip['residue_mask'].all(0)
    require(int(mask.sum()) >= 4 and torch.equal(context_clip['residue_mask'],mask[None].expand(8,-1)),
            'fixed all8 observed support required')
    require(torch.equal(endpoint_clip['residue_mask'][0],mask)
            and bool(endpoint_clip['atom_mask'][0,mask].all()), 'endpoint lacks fixed observed support')
    gap = endpoint_clip['time_ps'][0] - context_clip['time_ps'][-1]
    require(bool(torch.isfinite(gap)) and float(gap) > 0 and float(gap) == spec.metadata('train')['horizon_ps'],
            'endpoint must have exact strict-future physical lag')
    def sanitized(clip):
        out = {k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in _clip_view(clip).items()}
        out['coords'] = torch.where(mask[None,:,None,None],out['coords'],0.)
        require(bool(torch.isfinite(out['coords']).all()), 'sanitized encoder coordinates nonfinite')
        return out
    row = dict(context=sanitized(context_clip),future=sanitized(endpoint_clip),
               horizon_frames=torch.tensor(spec.horizon,dtype=torch.long),horizon_time_ps=gap.clone(),
               geometry_row_ids=topology.geometry_row_ids.clone(),
               adaptation_metadata=dict(cache_row_index=cache['cache_row_index'],split='fit',
                   metadata=copy.deepcopy(cache['metadata']),residue_uids=copy.deepcopy(expected['residue_uids']),
                   topology_sha256=support.canonical_sha256(expected)))
    after = fingerprint(dict(plan=plan,context=_clip_view(context_clip),endpoint=_clip_view(endpoint_clip),
                             topology=topology.as_metadata()))
    require(before == after, 'adaptation inputs mutated')
    row['adaptation_audit'] = dict(cache_row_index=cache['cache_row_index'],split='fit',
        cache_plan_sha256=cache['plan_sha256'],adaptation_plan_sha256=plan['adaptation_plan_sha256'],
        input_sha256_before=before,input_sha256_after=after,
        observed_residue_count=int(mask.sum()),context_frames=8,future_frames=1,
        horizon_time_ps=float(gap),prepared_sha256=fingerprint(row))
    return row


def collate_adaptation_rows(rows):
    """Legacy paired collation with eight history frames and one future frame.

    Width is the batch maximum, at most96. Extra padding uses the unchanged
    legacy AA20/RI−1/chain−1 convention; original duplicate IDs remain intact.
    Call the existing temporal objective with ``lags=(0,)``.
    """
    require(bool(rows), 'cannot collate empty adaptation batch')
    before = fingerprint(rows)
    for row in rows:
        audit = row['adaptation_audit']
        require(audit['split'] == 'fit' and type(audit['cache_row_index']) is int
                and 0 <= audit['cache_row_index'] < 400
                and row['adaptation_metadata']['cache_row_index'] == audit['cache_row_index']
                and row['adaptation_metadata']['split'] == 'fit', 'non-fit prepared row')
        require(fingerprint({k:row[k] for k in PAIR_KEYS}) == audit['prepared_sha256'],
                'prepared row changed before collation')
        require(row['context']['coords'].shape[0] == 8 and row['future']['coords'].shape[0] == 1
                and row['context']['coords'].shape[1] == row['future']['coords'].shape[1] <= 96,
                'endpoint-only paired shape differs')
    batch = collate_paired_windows(rows)
    width = batch['context_coords'].shape[2]
    geometry_ids = torch.full((len(rows),width),-1,dtype=torch.long)
    for i,row in enumerate(rows):
        geometry_ids[i,:len(row['geometry_row_ids'])] = row['geometry_row_ids']
    batch['geometry_row_ids'] = geometry_ids
    batch['adaptation_metadata'] = copy.deepcopy([r['adaptation_metadata'] for r in rows])
    after = fingerprint(rows)
    require(before == after, 'collation mutated inputs')
    batch['adaptation_audit'] = dict(rows=copy.deepcopy([r['adaptation_audit'] for r in rows]),
        input_sha256_before=before,input_sha256_after=after,batch_sha256=fingerprint(batch))
    return batch
