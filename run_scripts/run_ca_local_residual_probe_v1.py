"""CPU-only, cached-H geometric CA decoder capacity comparison.

Independent register/smoke/run phases. No encoder, dynamics kernel, HDF reader,
trajectory sampler, validation selection, restart, or historical-file mutation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from evaluation.geometry import ca_geometry_summary, physical_distance_energy_score
from protein_jepa_frame.models.ca_geometry import project_ca_bonds
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
from protein_jepa_frame.models.geometric_local_residual_v1 import GeometricLocalResidualCADecoderV1
from protein_jepa_frame.training.train_lasp_ca_dynamics import energy_score, contact_probability_loss
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.eval_ca_endpoint_scores import canonical_pairs

VERSION = 'ca_local_residual_probe_v1'
ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('results/hs_joint_20261101/ca_local_residual_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/ca_local_residual_v1')
SCRIPT = 'run_scripts/run_ca_local_residual_probe_v1.py'
DOCUMENT = 'docs/CA_LOCAL_RESIDUAL_PROBE_V1_20261102.md'
TEST = 'tests/test_ca_local_residual_probe_v1.py'
INITIAL = 'artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt'
INITIAL_SHA = '396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda'
CACHE_BASE = Path('results/hs_joint_20261101/backbone_repair')
CACHES = {'train': str(CACHE_BASE/'readout_train500_v2.pt'), 'val': str(CACHE_BASE/'readout_val200_v2.pt')}
CACHE_SHA = {'train': 'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718',
             'val': 'd1a04f5861badc004f39d9496e3f245b6c32739a8d2deed5bb74840eaf605fdc'}
STATISTICS = RESULTS/'summary.protocol.json'
STATISTICS_SHA = '6019841688e0e39fb5bf52b27d9f04d05771e7db96d69fb4091e68f8746d87cf'
ARMS = ('base_tuned', 'local_residual')
ROUTES = ('baseline', *ARMS)
TASKS = ('oracle', 'forecast', 'anchor_identity', 'wrong_horizon_H')
STAGES = ('raw', 'projected')
HORIZONS = (1, 4, 16, 64)
WRONG_HORIZON = {1: 4, 4: 16, 16: 64, 64: 1}
CONFIG = dict(latent_dim=128, hidden=128, depth=3)
GEOMETRY = ('ca_bond_mean_angstrom', 'ca_bond_invalid_fraction',
            'ca_nonlocal_clash_fraction', 'ca_radius_gyration_angstrom')
SCALARS = ('coordinate_mse', 'ca_rmsd', 'rmsd_to_persistence_ratio',
           *(f'{kind}_{metric}' for kind in ('canonical', 'psf') for metric in GEOMETRY))
ENSEMBLE = ('coordinate_energy_score', 'physical_distance_energy_score', 'contact_brier')
KEY_FIELDS = ('protein', 'trajectory', 'temperature_k', 'anchor_frame', 'horizon')
HYPER = dict(steps=500, seed=42, torch_threads=2, batch_size=1, lr=1e-4,
    warmup_steps=50, cosine=True, final_lr=0., nonzero_lr_calls=499,
    lr_formula='1e-4*step/50 for step<=50; 1e-4*(1+cos(pi*(step-50)/450))/2 thereafter',
    optimizer='AdamW', weight_decay=.01,
    betas=[.9, .999], eps=1e-8, gradient_clip_norm=1., projection_steps=32,
    loss='sum observed-valid projected xyz squared error / (3 * Nvalid)',
    order='one pass through the immutable train500 cache in its existing order',
    selection='latest500 only; no validation during training, early stopping or tuning')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def local(path):
    p = (ROOT/Path(path)).resolve()
    require(p.is_relative_to(ROOT) and p.name != 'test.jsonl', 'outside repository or sealed test manifest')
    return p


def sha(path):
    digest = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_new(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(encoded)


def save_new(path, value):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(value, stream)


def read_json(path):
    return json.loads(local(path).read_text())


def tensor_inputs(row):
    return {key: value for key, value in row.items() if torch.is_tensor(value)}


def row_fingerprint(row):
    return state_sha256(tensor_inputs(row))


def key(row):
    metadata = row.get('metadata', row)
    return tuple(metadata[field] for field in KEY_FIELDS)


def inference_hash(checkpoint):
    component = checkpoint['component_checkpoint']
    state = {f'bundle.encoder.{k}': v for k, v in component['encoder'].items()}
    state.update({f'bundle.decoder.{k}': v for k, v in component['delta_ca_decoder'].items()})
    state.update({f'kernel.{k}': v for k, v in checkpoint['kernel'].items()})
    return state_sha256(state)


def source_files():
    names = {SCRIPT, DOCUMENT, TEST,
        'protein_jepa_frame/models/geometric_delta_decoder.py',
        'protein_jepa_frame/models/geometric_local_residual_v1.py',
        'tests/test_geometric_local_residual_v1.py',
        'protein_jepa_frame/models/ca_geometry.py', 'evaluation/geometry.py',
        'scripts/eval_ca_endpoint_scores.py', 'protein_jepa_frame/utils/contracts.py'}
    for module in tuple(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(ROOT):
                names.add(str(path.relative_to(ROOT)))
    return sorted(names)


def snapshot(directory, sources):
    directory = local(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        out = directory/name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('xb') as stream:
            stream.write(local(name).read_bytes())
        require(sha(out) == digest, 'snapshot differs')


def verify(protocol, digest=None):
    if digest is not None:
        require(sha(RESULTS/'protocol.json') == digest, 'protocol changed')
    for path, expected in {**protocol['input_hashes'], **protocol['source_hashes']}.items():
        require(sha(path) == expected, 'frozen source/input changed: '+path)
    for path, expected in protocol['source_hashes'].items():
        require(sha(Path(protocol['source_snapshot'])/path) == expected, 'snapshot changed: '+path)
    require(not torch.cuda.is_initialized(), 'CPU-only contract violated')
    return {**protocol['input_hashes'], **protocol['source_hashes']}


def register():
    require(not local(ARTIFACTS).exists(), 'artifacts already exist; no restart')
    require(local(RESULTS).is_dir() and {p.name for p in local(RESULTS).iterdir()} == {'summary.protocol.json'},
            'only the preregistered statistics protocol may exist')
    require(sha(STATISTICS) == STATISTICS_SHA, 'statistical registration changed')
    statistics = read_json(STATISTICS)
    inputs = {INITIAL: INITIAL_SHA, str(STATISTICS): STATISTICS_SHA, **{CACHES[m]: CACHE_SHA[m] for m in CACHES}}
    cached_sources = {}
    for mode, cache_path in CACHES.items():
        stem = Path(cache_path)
        for suffix in ('.protocol.json', '.selection.json', '.execution.json'):
            file = stem.with_suffix(suffix)
            inputs[str(file)] = sha(file)
        historical = read_json(stem.with_suffix('.protocol.json'))
        for path, expected in historical['input_hashes'].items():
            require(path not in inputs or inputs[path] == expected, 'cache input versions differ')
            inputs[path] = expected
        for path, expected in historical['source_hashes'].items():
            require(path not in cached_sources or cached_sources[path] == expected, 'cache source versions differ')
            cached_sources[path] = expected
            inputs[str(stem.with_suffix('.source_snapshot')/path)] = expected
    sources = {path: sha(path) for path in source_files()}
    for path, expected in cached_sources.items():
        require(sha(path) == expected, 'cache producer source changed: '+path)
        sources[path] = expected
    for path, expected in statistics['source_hashes'].items():
        require(sha(path) == expected, 'statistical source changed')
        sources[path] = expected
    for path, expected in inputs.items():
        require(sha(path) == expected, 'registered input differs: '+path)
    out = ARTIFACTS/'source_snapshot'
    snapshot(out, sources)
    protocol = dict(schema_version=VERSION, registered_utc=now(),
        arms=list(ARMS), routes=list(ROUTES), tasks=list(TASKS), stages=list(STAGES),
        model_config=CONFIG, hyperparameters=HYPER, statistics_protocol=str(STATISTICS),
        statistics_sha256=STATISTICS_SHA, source_hashes=sources, input_hashes=inputs,
        source_snapshot=str(out), caches=CACHES, cache_sha256=CACHE_SHA,
        initial_checkpoint=INITIAL, initial_checkpoint_sha256=INITIAL_SHA,
        observed_only_forward_arguments=['anchor_H', 'observed_R', 'observed_CA', 'embedding_aatype',
            'embedding_residue_index', 'embedding_chain_id', 'observed_mask'],
        future_forward_argument='cached oracle_H for supervised oracle routes, or immutable cached forecast_H for secondary routes',
        forbidden_forward_arguments=['target_backbone', 'target_atom_mask', 'chemical_aatype', 'peptide_edges'],
        gauge='target_backbone CA already fitted to observed anchor once; no new target or prediction alignment',
        cache_time='no time_ps tensor retained; first_frame_time_ps and 1000ps frame interval plus frame IDs retained in selection',
        topology='Both heads retain original legacy ri/ci P32. Actual undirected CA edges in the full cached PSF graph are an additional geometry diagnostic; nonlocal excludes graph distance one or two, includes distance >=3 or disconnected pairs. Cropped-away rows are not available.',
        expected_legacy_psf_mismatch_windows={'train': 7, 'val': 0},
        controls=dict(anchor_identity='replace oracle_H by same row anchor_H',
            wrong_horizon_H={str(k): v for k, v in WRONG_HORIZON.items()},
            strict_same_protein_same_horizon_time_permutation='unavailable: exactly one validation window per protein/horizon',
            inference='controls are descriptive readout-dependence diagnostics, not dynamical prediction evidence'),
        precision=dict(decoder='CPU float32', training_loss='float32 projected MSE',
            scoring='coordinates, new arithmetic and PSF fractions float64; frozen canonical invalid/clash fractions retain float32 Boolean means'),
        scalar_metrics=list(SCALARS), forecast_extra_metrics=list(ENSEMBLE),
        primary='val h16/64 oracle projected ca_rmsd, local_residual minus base_tuned, equal proteins',
        execution='CPU2 sequential stages in one actual process; smoke first4 has no optimizer; no automatic restart',
        limitations=['adaptive development reconstruction probe; oracle_H contains future information',
            'encoder and kernel tensors are read for provenance only, never instantiated or optimized',
            'fixed forecast_H is from original spatial_exact, not rerun with the updated decoder',
            'no whole future path in cache: seven historical metrics, ordered paths, full T1/T2A/T3 are not run',
            'one pass over 500 fixed training proteins is not an unconstrained capacity bound'],
        test_opened=False, hdf_opened=False, new_sampling=False, encoder_instantiated=False, kernel_instantiated=False)
    verify(protocol)
    write_new(RESULTS/'protocol.json', protocol)
    return protocol


def topology_support(row):
    mask = row['observed_mask'].bool()
    length = len(mask)
    ri, ci = row['embedding_residue_index'], row['embedding_chain_id']
    ids, edges = row['geometry_row_ids'], row['peptide_edges']
    require(ids.shape == (length,) and len(ids.unique()) == length, 'geometry row identity differs')
    require(edges.dtype == torch.long and edges.ndim == 2 and edges.shape[1] == 2, 'invalid explicit edges')
    pairs = [tuple(map(int, e)) for e in edges.tolist()]
    require(len(pairs) == len(set(pairs)) and all(0 <= a < length and 0 <= b < length and a != b for a,b in pairs),
            'duplicate/self/out-of-range peptide edge')
    all_edges = {tuple(sorted((a,b))) for a,b in pairs}
    actual = {(a,b) for a,b in all_edges if bool(mask[a] & mask[b])}
    legacy = {(i,i+1) for i in range(length-1) if bool(mask[i]&mask[i+1])
        and int(ci[i]) == int(ci[i+1]) and int(ri[i+1]-ri[i]) == 1}
    graph = torch.zeros(length, length, dtype=torch.bool)
    # Masking an intermediate atom never removes its known covalent topology.
    # Build graph distances on all cached PSF rows, then filter scored endpoints.
    for a,b in all_edges:
        graph[a,b] = graph[b,a] = True
    close = graph | ((graph.to(torch.int64) @ graph.to(torch.int64)) > 0)
    i,j = torch.triu_indices(length,length,offset=1)
    good = mask[i]&mask[j]&~close[i,j]
    psf_pairs = torch.stack((i[good],j[good]),1)
    bonds = torch.tensor(sorted(actual),dtype=torch.long).reshape(-1,2)
    canonical_pairs_mask = mask[i]&mask[j]&((ci[i]!=ci[j])|((ri[i]-ri[j]).abs()>=3))
    evidence = dict(n_residues=length, n_valid_residues=int(mask.sum()),
        n_legacy_edges=len(legacy), n_psf_edges=len(actual),
        n_common_edges=len(legacy&actual), legacy_only_edges=[list(e) for e in sorted(legacy-actual)],
        psf_only_edges=[list(e) for e in sorted(actual-legacy)],
        legacy_psf_equal=legacy==actual, topology_sha256=row['topology']['sha256'],
        n_canonical_nonlocal_pairs=int(canonical_pairs_mask.sum()), n_psf_nonlocal_pairs=len(psf_pairs))
    return bonds, psf_pairs, evidence


def validate_cache(cache, mode, initial, selection):
    rows, p = cache['rows'], cache['protocol']
    count = 500 if mode == 'train' else 200
    require(cache['schema_version'] == 'backbone_readout_cache_v1' and len(rows) == count,
            'cache schema/panel differs')
    expected = dict(mode=mode, n_windows=count, full_panel=True, complete_panel_windows=count,
        max_residues=96, context_frames=8, n_forecast_samples=0 if mode=='train' else 12,
        forecast_observed_only=True, before_target_read=True, optimizer_steps=0,
        test_opened=False, bioemu2_rerun=False, frame_interval_ps=1000.)
    require(all(p.get(k) == v for k,v in expected.items()), 'cache boundary/budget differs')
    require(p['checkpoint'] == INITIAL and cache['input_hashes'][INITIAL] == INITIAL_SHA,
            'cache checkpoint differs')
    model_hash = inference_hash(initial)
    require(p['model_tensor_sha256'] == model_hash == cache['completion']['model_sha256_before']
        == cache['completion']['model_sha256_after'] and cache['completion']['frozen_inputs_verified'],
        'cache representation/dynamics changed')
    require([r['metadata'] for r in rows] == [x['metadata'] for x in selection['full_selection']], 'cache order differs')
    require(len({key(r) for r in rows}) == count, 'duplicate cache keys')
    require(len({r['metadata']['protein'] for r in rows}) == (500 if mode=='train' else 50), 'protein count differs')
    require(Counter(r['metadata']['horizon'] for r in rows) == {h:count//4 for h in HORIZONS}, 'horizon support differs')
    if mode == 'val':
        require(Counter(r['metadata']['temperature_k'] for r in rows) == {t:40 for t in (320.,348.,379.,413.,450.)}, 'temperature support differs')
    mismatch, audits = 0, []
    for row, registered in zip(rows, selection['full_selection']):
        m = row['observed_mask']; length = len(m)
        require(m.dtype == torch.bool and m.ndim == 1 and 4 <= int(m.sum()) <= length <= 96, 'invalid observed support')
        require(row['atom_mask'].shape == row['target_atom_mask'].shape == (length,4)
            and bool(row['atom_mask'][m].all()) and bool(row['target_atom_mask'][m].all()), 'missing observed support')
        metadata = row['metadata']; record = registered['record']
        require(record['first_frame_time_ps'] == 0. and record['frame_interval_ps'] == 1000.
            and metadata['context_frames'] == list(range(metadata['anchor_frame']-7,metadata['anchor_frame']+1))
            and metadata['target_frame'] == metadata['anchor_frame']+metadata['horizon']
            and metadata['horizon_ps'] == metadata['horizon']*1000., 'cached time identity differs')
        require(row['boundary']['forecast_before_target_read'] is True
            and row['boundary']['future_reencoding_for_forecast'] == 0, 'future boundary differs')
        shapes = dict(observed_R=(1,length,3,3), observed_backbone=(1,length,4,3),
            target_backbone=(1,length,4,3), anchor_H=(1,length,128), oracle_H=(1,length,128), oracle_CA=(1,length,3))
        if mode=='val': shapes.update(forecast_H=(12,length,128),forecast_CA=(12,length,3))
        else: require('forecast_H' not in row and 'forecast_CA' not in row, 'train has forecasts')
        for name, shape in shapes.items():
            value = row[name]
            require(value.shape == shape and value.device.type == 'cpu' and value.dtype == torch.float32
                and not value.requires_grad and bool(torch.isfinite(value[:,m]).all()), 'bad cached tensor: '+name)
        for name in ('embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype',
                     'geometry_row_ids','peptide_edges','atom_mask'):
            require(row[name].tolist() == row['topology'][name], 'topology tensor differs: '+name)
        require(state_sha256({'backbone':row['target_backbone']}) == row['fingerprints']['aligned_label']
            and state_sha256({'H':row['oracle_H'],'CA':row['oracle_CA']}) == row['fingerprints']['oracle_features'],
            'cached label/oracle fingerprint differs')
        if mode=='val':
            require(state_sha256({k:row[k] for k in ('forecast_H','forecast_CA')}) == row['fingerprints']['forecast'], 'forecast fingerprint differs')
        _,_,top = topology_support(row)
        mismatch += int(not top['legacy_psf_equal'])
        audits.append(dict(metadata=metadata, input_tensor_sha256=row_fingerprint(row), topology=top))
    require(mismatch == (7 if mode=='train' else 0), 'registered topology mismatch count differs')
    return audits


def load_inputs(protocol):
    initial = torch.load(local(INITIAL), map_location='cpu', weights_only=False)
    component = initial['component_checkpoint']
    require(component['decoder_mode'] == 'geometric_ca' and component['calibration_args']['bond_projection_steps'] == 32,
            'D0 architecture/projection differs')
    caches, audits = {}, {}
    for mode,path in CACHES.items():
        caches[mode] = torch.load(local(path), map_location='cpu', weights_only=False)
        selection = read_json(caches[mode]['protocol']['selection_file'])
        audits[mode] = validate_cache(caches[mode], mode, initial, selection)
    require(not ({r['metadata']['protein'] for r in caches['train']['rows']}
                 & {r['metadata']['protein'] for r in caches['val']['rows']}), 'train/val protein overlap')
    state = {k:v.detach().clone() for k,v in component['delta_ca_decoder'].items()}
    identity = dict(initial_checkpoint_sha256=INITIAL_SHA, inference_tensor_sha256=inference_hash(initial),
        encoder_tensor_sha256=state_sha256(component['encoder']), kernel_tensor_sha256=state_sha256(initial['kernel']),
        decoder_tensor_sha256=state_sha256(state), complete_component_config_sha256=hashlib.sha256(canonical({
            k:component[k] for k in ('checkpoint_kind','decoder_mode','calibration_args')}).encode()).hexdigest(),
        projection_steps=32, encoder_instantiated=False, kernel_instantiated=False)
    return state, caches, audits, identity


def make_decoder(state, arm):
    require(arm in ROUTES, 'unknown decoder arm')
    if arm == 'local_residual':
        return GeometricLocalResidualCADecoderV1.from_base_state_dict(state, **CONFIG).float()
    with torch.random.fork_rng(devices=[]):
        result = GeometricDeltaCADecoder(**CONFIG).float()
    result.load_state_dict(state, strict=True)
    return result


def decode(decoder, row, future_h):
    """Forward reads cached H plus observed arguments; labels never enter here."""
    m = row['observed_mask'][None]
    future = future_h.detach().to(dtype=torch.float32,device='cpu')
    count, length, _ = future.shape
    require(length == m.shape[1], 'future H support differs')
    def expand(value): return value.detach().expand(count,*value.shape[1:])
    ah = expand(row['anchor_H'])
    anchor = expand(row['observed_backbone'][:,:,1,:])
    R = expand(row['observed_R'])
    masks = m.expand(count,-1)
    future = torch.where(masks[...,None],future,0.)
    ah = torch.where(masks[...,None],ah,0.)
    anchor = torch.where(masks[...,None],anchor,0.)
    R = torch.where(masks[...,None,None],R,0.)
    aa,ri,ci = [row[k][None].detach().expand(count,-1) for k in
                ('embedding_aatype','embedding_residue_index','embedding_chain_id')]
    aa = torch.where(masks,aa,0)
    raw = decoder(future,ah,aa,ri,ci,R,anchor,masks)
    projected = project_ca_bonds(raw,anchor,masks,ri,ci,iterations=32)
    require(bool(torch.isfinite(raw[masks]).all() and torch.isfinite(projected[masks]).all()), 'nonfinite decoder output')
    return {'raw':raw,'projected':projected}


def masked_mse(prediction, target, mask):
    require(prediction.shape == target.shape and mask.shape == prediction.shape[:-1] and bool(mask.any()), 'MSE shape/support differs')
    return (prediction[mask]-target.detach()[mask]).square().sum()/(3*mask.sum())


def loss_backward(decoder, row):
    decoder.train(); decoder.zero_grad(set_to_none=True)
    output = decode(decoder,row,row['oracle_H'])
    loss = masked_mse(output['projected'], row['target_backbone'][:,:,1,:], row['observed_mask'][None])
    require(bool(torch.isfinite(loss)), 'nonfinite loss')
    loss.backward()
    parameters = list(decoder.parameters())
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), 'missing/nonfinite decoder gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
    require(bool(torch.isfinite(norm)), 'nonfinite gradient norm')
    return loss, norm


def learning_rate(step):
    require(1 <= step <= 500, 'step outside fixed schedule')
    if step <= 50: return 1e-4*step/50
    return 1e-4*.5*(1+math.cos(math.pi*(step-50)/450))


def wrong_horizon_donors(rows):
    by_protein = defaultdict(dict)
    for index,row in enumerate(rows):
        protein,h = row['metadata']['protein'], row['metadata']['horizon']
        require(h not in by_protein[protein], 'same-protein/horizon is not unique')
        by_protein[protein][h] = index
    result = {}
    for group in by_protein.values():
        require(set(group) == set(HORIZONS), 'wrong-H needs complete horizon group')
        for h,index in group.items():
            donor = group[WRONG_HORIZON[h]]
            a,b = rows[index],rows[donor]
            for field in ('trajectory','temperature_k','anchor_frame','context_frames'):
                require(a['metadata'][field] == b['metadata'][field], 'wrong-H observed metadata differs')
            for field in ('observed_mask','observed_R','observed_backbone','anchor_H','embedding_aatype',
                          'embedding_residue_index','embedding_chain_id','chemical_aatype','geometry_row_ids','peptide_edges','atom_mask'):
                require(torch.equal(a[field],b[field]), 'wrong-H observed/topology tensors differ: '+field)
            require(a['topology'] == b['topology'], 'wrong-H topology provenance differs')
            result[index] = donor
    return result


def psf_geometry(ca, mask, bonds, pairs):
    x = ca.double()
    lengths = (x[bonds[:,0]]-x[bonds[:,1]]).norm(dim=-1)
    distances = (x[pairs[:,0]]-x[pairs[:,1]]).norm(dim=-1)
    return dict(ca_bond_mean_angstrom=float(lengths.mean()) if len(lengths) else None,
        ca_bond_invalid_fraction=float(((lengths<3.3)|(lengths>4.3)).double().mean()) if len(lengths) else None,
        ca_nonlocal_clash_fraction=float((distances<2.5).double().mean()) if len(distances) else None,
        ca_radius_gyration_angstrom=float((x[mask]-x[mask].mean(0)).square().sum(-1).mean().sqrt()))


def geometry(ca,row,support):
    bonds,pairs,evidence = support
    canonical_values = ca_geometry_summary(ca.double(),row['observed_mask'],
        chain_id=row['embedding_chain_id'],residue_index=row['embedding_residue_index'])
    psf_values = psf_geometry(ca,row['observed_mask'],bonds,pairs)
    return {f'{kind}_{name}':value for kind,values in [('canonical',canonical_values),('psf',psf_values)]
            for name,value in values.items()}


def mean_complete(values):
    return None if any(v is None for v in values) else float(np.mean(values))


@torch.no_grad()
def endpoint_ensemble_scores(samples, target, row):
    """Frozen U/V/Brier formulae, cached aligned target directly; NO Kabsch."""
    x,y = samples.double(),target.double()
    mask,ri,ci = [row[k] for k in ('observed_mask','embedding_residue_index','embedding_chain_id')]
    require(len(samples) == 12, 'fixed forecast ensemble requires twelve members')
    i,j = canonical_pairs(mask,ri,ci)
    predicted = ((x[:,i]-x[:,j]).norm(dim=-1)<8.).double().mean(0)
    truth = ((y[i]-y[j]).norm(dim=-1)<8.).double()
    return dict(coordinate_energy_score=float(energy_score(x[:,mask].flatten(1),y[None,mask].flatten(1))),
        physical_distance_energy_score=physical_distance_energy_score(x,y,mask,scale_angstrom=1.,max_pairs=512),
        contact_brier=float(contact_probability_loss(predicted,truth,'brier')))


@torch.no_grad()
def score_leaf(coords,row,support,*,forecast=False):
    mask = row['observed_mask']
    target = row['target_backbone'][0,:,1,:].double()
    anchor = row['observed_backbone'][0,:,1,:].double()
    denominator = float((anchor[mask]-target[mask]).square().sum(-1).mean().sqrt())
    members = []
    for sample in coords:
        error = (sample.double()[mask]-target[mask]).square()
        mse,rmsd = float(error.mean()),float(error.sum(-1).mean().sqrt())
        members.append(dict(coordinate_mse=mse,ca_rmsd=rmsd,
            rmsd_to_persistence_ratio=rmsd/denominator if denominator>0 else None,
            **geometry(sample,row,support)))
    scalars = {metric:mean_complete([r[metric] for r in members]) for metric in SCALARS}
    if forecast: scalars.update(endpoint_ensemble_scores(coords,target,row))
    require(all(value is None or math.isfinite(value) for value in scalars.values()), 'nonfinite evaluated scalar')
    return dict(scalars=scalars,per_member=members,n_members=len(coords),n_samples=len(coords),
        output_tensor_sha256=state_sha256({'CA':coords.detach().cpu()}),
        geometry_counts={k:support[2][k] for k in ('n_legacy_edges','n_psf_edges','n_canonical_nonlocal_pairs','n_psf_nonlocal_pairs')})


def reseed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU-only run required')


def zero_update_check(state,rows):
    models = {name:make_decoder(state,name).eval() for name in ROUTES}
    fingerprints = {name:state_sha256(model) for name,model in models.items()}
    values = []
    with torch.no_grad():
        for row in rows:
            predictions = {name:decode(model,row,row['oracle_H']) for name,model in models.items()}
            for stage in STAGES:
                require(all(torch.equal(predictions['baseline'][stage],predictions[a][stage])
                    and state_sha256({'CA':predictions['baseline'][stage]}) == state_sha256({'CA':predictions[a][stage]})
                    for a in ARMS),
                        'zero-update CPU outputs differ')
            values.append(dict(key=list(key(row)),input_tensor_sha256=row_fingerprint(row),
                output_sha256={stage:state_sha256({'CA':predictions['baseline'][stage]}) for stage in STAGES},
                route_output_sha256={name:{stage:state_sha256({'CA':predictions[name][stage]})
                    for stage in STAGES} for name in ROUTES}))
    require(bool((models['local_residual'].local_residual.weight==0).all()), 'initial W not zero')
    return models,dict(model_sha256=fingerprints,all_three_raw_and_projected_bitwise_equal=True,
        zero_additional_parameters=384,backend='CPU float32; old GPU cached outputs are not a bitwise comparator',rows=values)


def smoke(protocol):
    state,caches,audits,identity = load_inputs(protocol)
    rows = caches['train']['rows'][:4]
    models,initial = zero_update_check(state,rows)
    values=[]
    for arm in ARMS:
        for index,row in enumerate(rows):
            start=time.monotonic();before=row_fingerprint(row)
            loss,norm=loss_backward(models[arm],row)
            require(row_fingerprint(row)==before,'smoke input changed')
            require(state_sha256(models[arm])==initial['model_sha256'][arm],'smoke model changed')
            values.append(dict(arm=arm,row_index=index,key=list(key(row)),loss=float(loss.detach()),
                gradient_norm=float(norm),seconds=time.monotonic()-start,
                residual_gradient_nonzero=bool(models[arm].local_residual.weight.grad.abs().sum()>0) if arm=='local_residual' else None))
    report=dict(schema_version=VERSION,phase='smoke',optimizer_steps=0,optimizer_instantiated=False,
        initial_identity=identity,zero_update=initial,first4_rows=True,checks=values,
        model_state_unchanged=True,input_caches_unchanged=True,cuda_initialized=False)
    write_new(RESULTS/'smoke.json',report)
    return report


def train_arm(state,rows,arm,protocol,protocol_sha):
    out=local(ARTIFACTS/arm);out.mkdir(parents=True,exist_ok=False)
    reseed();model=make_decoder(state,arm)
    initial_hash=state_sha256(model)
    names=[name for name,_ in model.named_parameters()]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    start=time.monotonic()
    with (out/'metrics.jsonl').open('x') as metrics,(out/'batches.jsonl').open('x') as batches:
        for step,row in enumerate(rows,1):
            before=row_fingerprint(row)
            lr=learning_rate(step)
            for group in optimizer.param_groups:group['lr']=lr
            loss,norm=loss_backward(model,row)
            optimizer.step()
            require(row_fingerprint(row)==before,'training cache tensor changed')
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'nonfinite updated parameter')
            value=dict(step=step,metadata=row['metadata'],loss=float(loss.detach()),gradient_norm=float(norm),lr=lr)
            metrics.write(json.dumps(value,allow_nan=False)+'\n');metrics.flush()
            batches.write(json.dumps(dict(step=step,key=list(key(row)),input_tensor_sha256=before),sort_keys=True)+'\n');batches.flush()
            if step%25==0:print(f'{arm} {step}/500 loss={value["loss"]:.6g}',flush=True)
    require(len(rows)==500,'must train exactly500 rows')
    steps=[float(x['step']) for x in optimizer.state_dict()['state'].values()]
    require(len(steps)==len(names) and set(steps)=={500.},'all optimizer parameters must have500 updates')
    checkpoint=dict(schema_version=VERSION,arm=arm,step=500,model_config=CONFIG,head=model.state_dict(),
        parameter_names=names,optimizer=optimizer.state_dict(),hyperparameters=HYPER,
        initial_decoder_sha256=state_sha256(state),initial_model_sha256=initial_hash,
        final_model_sha256=state_sha256(model),protocol_sha256=protocol_sha,
        input_hashes=protocol['input_hashes'],source_hashes=protocol['source_hashes'],
        rng_state=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),
        encoder_instantiated=False,kernel_instantiated=False,validation=None,selection='latest500')
    require(checkpoint['final_model_sha256']!=initial_hash,'no decoder parameter changed')
    save_new(out/'checkpoint_latest.pt',checkpoint)
    return model.eval(),dict(arm=arm,steps=500,pid=os.getpid(),process_kind='in_process_sequential_stage',
        seconds=time.monotonic()-start,checkpoint=str(ARTIFACTS/arm/'checkpoint_latest.pt'),
        checkpoint_sha256=sha(out/'checkpoint_latest.pt'),metrics_sha256=sha(out/'metrics.jsonl'),
        batches_sha256=sha(out/'batches.jsonl'),all_optimizer_steps=500,active_optimizer_parameters=len(names))


@torch.no_grad()
def evaluate(models,caches,audits,identity,protocol,protocol_sha):
    donors=wrong_horizon_donors(caches['val']['rows'])
    raw=dict(schema_version=VERSION,protocol_sha256=protocol_sha,initial_identity=identity,
        cache_sha256=CACHE_SHA,source_hashes=protocol['source_hashes'],input_hashes=protocol['input_hashes'],
        conventions=dict(primary='fixed-anchor CA RMSD; predictions and cached aligned targets are never refitted',
            coordinate_mse='mean squared xyz error, per member, then mean members',
            ca_rmsd='sqrt(mean residue squared Euclidean error) per member, then mean members',
            coordinate_energy_score='frozen off-diagonal U estimator divided by sqrt(3*Nvalid), cached aligned target direct',
            physical_distance_energy_score='frozen physical-distance V estimator, deterministic512 pair cap',
            contact_brier='Brier of empirical hard8A contact frequencies on all canonical nonlocal pairs',
            precision=protocol['precision'],topology=protocol['topology']),
        controls=protocol['controls'],seven_metrics_run=False,encoder_instantiated=False,kernel_instantiated=False,
        hdf_opened=False,test_opened=False,new_sampling=False,per_window={'train':[],'val':[]})
    coordinates=dict(schema_version=VERSION,protocol_sha256=protocol_sha,per_window={'train':[],'val':[]})
    for mode in ('train','val'):
        for index,row in enumerate(caches[mode]['rows']):
            before=row_fingerprint(row);support=topology_support(row);mask=row['observed_mask']
            anchor=row['observed_backbone'][0,:,1,:].double();target=row['target_backbone'][0,:,1,:].double()
            denominator=float((anchor[mask]-target[mask]).square().sum(-1).mean().sqrt())
            record=dict(**row['metadata'],row_index=index,valid_residues=int(mask.sum()),
                topology_audit=support[2],input_fingerprints=dict(row_tensor_sha256=before,**row['fingerprints']),
                references=dict(persistence_ca_rmsd=denominator,persistence_geometry=geometry(anchor,row,support),
                    target_geometry=geometry(target,row,support)),
                wrong_horizon_donor=None,routes={})
            stored=dict(key=list(key(row)),row_index=index,routes={},reference_tensors={
                'target_ca':row['target_backbone'][0,:,1,:].detach().clone(),
                'anchor_ca':row['observed_backbone'][0,:,1,:].detach().clone(),
                **{name:row[name].detach().clone() for name in ('observed_mask',
                    'embedding_residue_index','embedding_chain_id','peptide_edges')}})
            if mode=='val':
                donor=caches[mode]['rows'][donors[index]]
                record['wrong_horizon_donor']=dict(row_index=donors[index],key=list(key(donor)),
                    oracle_H_sha256=state_sha256({'H':donor['oracle_H']}))
            for route,model in models.items():
                record['routes'][route]={task:None for task in TASKS}
                stored['routes'][route]={task:None for task in TASKS}
                tasks={'oracle':row['oracle_H']}
                if mode=='val':tasks.update(forecast=row['forecast_H'],anchor_identity=row['anchor_H'],wrong_horizon_H=donor['oracle_H'])
                for task,H in tasks.items():
                    result=decode(model,row,H)
                    record['routes'][route][task]={stage:score_leaf(value,row,support,forecast=task=='forecast') for stage,value in result.items()}
                    stored['routes'][route][task]={stage:value.detach().cpu().clone() for stage,value in result.items()}
            require(row_fingerprint(row)==before,'evaluation input changed')
            raw['per_window'][mode].append(record);coordinates['per_window'][mode].append(stored)
            if (index+1)%25==0:print(f'eval {mode} {index+1}/{len(caches[mode]["rows"])}',flush=True)
    save_new(RESULTS/'score_coordinates.pt',coordinates)
    raw['coordinate_sidecar']=dict(path=str(RESULTS/'score_coordinates.pt'),sha256=sha(RESULTS/'score_coordinates.pt'))
    raw['completed_utc']=now()
    write_new(RESULTS/'raw.json',raw)
    return dict(raw=str(RESULTS/'raw.json'),raw_sha256=sha(RESULTS/'raw.json'),**raw['coordinate_sidecar'])


def run(protocol,digest):
    smoke_record=read_json(RESULTS/'smoke.execution.json')
    require(smoke_record['status']=='completed' and smoke_record['exit_code']==0 and smoke_record['pid']>0,
            'actual completed smoke required')
    require(sha(RESULTS/'smoke.json')==smoke_record['output_sha256'],'smoke output changed')
    smoke_result=read_json(RESULTS/'smoke.json')
    require(smoke_result['optimizer_steps']==0 and smoke_result['zero_update']['all_three_raw_and_projected_bitwise_equal'], 'zero-step proof missing')
    state,caches,audits,identity=load_inputs(protocol)
    initial_models,zero_update=zero_update_check(state,caches['train']['rows']+caches['val']['rows'])
    write_new(RESULTS/'zero_update_full.json',dict(schema_version=VERSION,n_train=500,n_val=200,
        optimizer_steps=0,initial_identity=identity,zero_update=zero_update))
    baseline=initial_models['baseline'];del initial_models
    models={'baseline':baseline};stages=[]
    for arm in ARMS:
        started=now();frozen_before=verify(protocol,digest)
        model,evidence=train_arm(state,caches['train']['rows'],arm,protocol,digest)
        frozen_after=verify(protocol,digest)
        evidence.update(started_utc=started,completed_utc=now(),exit_code=0,status='completed',
            frozen_before=frozen_before,frozen_after=frozen_after)
        stages.append(evidence);write_new(RESULTS/f'{arm}.training.json',evidence);models[arm]=model
    require((local(ARTIFACTS/ARMS[0]/'batches.jsonl')).read_bytes()==(local(ARTIFACTS/ARMS[1]/'batches.jsonl')).read_bytes(),
            'two arms consumed different batches')
    before={a:state_sha256(m) for a,m in models.items()};started=now();frozen_before=verify(protocol,digest)
    evaluation=evaluate(models,caches,audits,identity,protocol,digest)
    require(before=={a:state_sha256(m) for a,m in models.items()},'evaluation changed models')
    stages.append(dict(stage='evaluate',pid=os.getpid(),process_kind='in_process_sequential_stage',
        started_utc=started,completed_utc=now(),exit_code=0,status='completed',
        frozen_before=frozen_before,frozen_after=verify(protocol,digest),**evaluation))
    return dict(stages=stages,output=str(RESULTS/'raw.json'),output_sha256=sha(RESULTS/'raw.json'),
        zero_update_full=dict(path=str(RESULTS/'zero_update_full.json'),sha256=sha(RESULTS/'zero_update_full.json')))


def execute(mode):
    path=RESULTS/'protocol.json';digest=sha(path);protocol=read_json(path)
    require(protocol['schema_version']==VERSION and protocol['hyperparameters']==HYPER,'protocol semantics changed')
    started=dict(phase=mode,pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=digest,
        source_sha256=sha(SCRIPT),cpu_threads=2,process_kind='single_actual_process_with_sequential_stages')
    write_new(RESULTS/f'{mode}.started.json',started)
    start=time.monotonic();out={}
    try:
        reseed();started['frozen_before']=verify(protocol,digest)
        if mode=='smoke':
            smoke(protocol);out=dict(output=str(RESULTS/'smoke.json'),output_sha256=sha(RESULTS/'smoke.json'))
        else:out=run(protocol,digest)
        frozen_after=verify(protocol,digest)
        write_new(RESULTS/f'{mode}.execution.json',dict(started,completed_utc=now(),seconds=time.monotonic()-start,
            status='completed',exit_code=0,frozen_inputs_sources_verified=True,frozen_after=frozen_after,**out))
    except BaseException as error:
        write_new(RESULTS/f'{mode}.execution.json',dict(started,completed_utc=now(),seconds=time.monotonic()-start,
            status='failed',exit_code=130 if isinstance(error,KeyboardInterrupt) else 1,error=f'{type(error).__name__}: {error}',**out))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('register','smoke','run'),required=True)
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'run from repository root')
    if args.mode=='register':
        register();print(json.dumps({'protocol':str(RESULTS/'protocol.json'),'sha256':sha(RESULTS/'protocol.json')}),flush=True)
    else:execute(args.mode)


if __name__=='__main__':
    main()
