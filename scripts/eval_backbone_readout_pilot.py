"""Frozen fixed-CA backbone-head comparison on a prepared development cache.

No encoder, dynamics sampler, trajectory loader, training, or test split is
opened here. Oracle readout and observed-only cached forecast are separate
routes. All scores use CPU float64 and all forecast members. Legacy ri/ci
geometry is explicitly audited against authoritative directed PSF edges.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import torch

from evaluation.backbone_metrics_v2 import backbone_error_summary, backbone_geometry_summary
from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutConfig, BackboneReadoutV2, HEAD_VERSION
from protein_jepa_frame.models.stage_c.backbone_objectives_v3 import backbone_geometry_losses_v3
from protein_jepa_frame.utils.contracts import state_sha256


REPO = Path(__file__).resolve().parents[1]
VERSION = 'fixed_ca_backbone_readout_pilot_eval_v1'
HEADS = ('init', 'coordinate', 'geometry')
SOURCES = (
    'scripts/eval_backbone_readout_pilot.py', 'evaluation/backbone_metrics_v2.py',
    'evaluation/geometry.py', 'protein_jepa_frame/models/backbone_readout_v2.py',
    'protein_jepa_frame/models/stage_c/backbone_objectives_v2.py',
    'protein_jepa_frame/models/stage_c/backbone_objectives_v3.py',
    'protein_jepa_frame/models/stage_c/structure_decoder.py',
    'protein_jepa_frame/models/stage_c/geometry.py',
    'protein_jepa_frame/models/stage_c/openfold_atom37.py',
    'protein_jepa_frame/utils/contracts.py',
    'protein_jepa_frame/vendored/openfold/np/residue_constants.py',
    'protein_jepa_frame/vendored/openfold/resources/stereo_chemical_props.txt',
    'protein_jepa_frame/vendored/openfold/utils/feats.py',
    'protein_jepa_frame/vendored/openfold/utils/rigid_utils.py',
    'protein_jepa_frame/vendored/openfold/utils/tensor_utils.py',
    'protein_jepa_frame/vendored/openfold/data/data_transforms.py',
    'protein_jepa_frame/vendored/PROVENANCE.json',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'input/output must be inside repository')
    require(path.name != 'test.jsonl' and 'bioemu' not in str(path).lower(), 'excluded input/output')
    return path


def sha(path):
    digest = hashlib.sha256()
    with local(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value):
    return state_sha256({'value': value})


def _plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def window_key(row):
    m = row['metadata']
    return (m['protein'], m['trajectory'], float(m['temperature_k']),
            int(m['anchor_frame']), int(m['horizon']))


def validate_row(row, *, n_forecast=12):
    mask = row['observed_mask']
    require(mask.dtype == torch.bool and mask.ndim == 1 and int(mask.sum()) >= 4,
            'observed_mask must be bool [L] with >=4 valid residues')
    length = len(mask)
    require(length <= 96, 'row exceeds registered 96-residue budget')
    for key in ('embedding_aatype', 'embedding_residue_index', 'embedding_chain_id',
                'chemical_aatype', 'geometry_row_ids'):
        require(row[key].dtype == torch.long and row[key].shape == (length,), f'{key} must be long [L]')
    for key in ('atom_mask', 'target_atom_mask'):
        require(row[key].dtype == torch.bool and row[key].shape == (length, 4), f'{key} must be bool [L,4]')
    require(bool((row['atom_mask'][mask, 1]).all()), 'observed valid residue lacks CA')
    shapes = {'observed_R': (1, length, 3, 3), 'observed_backbone': (1, length, 4, 3),
              'target_backbone': (1, length, 4, 3), 'anchor_H': (1, length, 128),
              'oracle_H': (1, length, 128), 'oracle_CA': (1, length, 3),
              'forecast_H': (n_forecast, length, 128), 'forecast_CA': (n_forecast, length, 3)}
    for key, shape in shapes.items():
        require(row[key].shape == shape and row[key].dtype == torch.float32 and row[key].device.type == 'cpu',
                f'{key} must be CPU float32 {shape}')
        valid = (row['atom_mask'] if key == 'observed_backbone' else row['target_atom_mask']) if key.endswith('backbone') else mask
        valid = valid & mask[:, None] if key.endswith('backbone') else valid
        require(bool(torch.isfinite(row[key][:, valid]).all()), f'nonfinite valid {key}')
    m = row['metadata']
    require(isinstance(m['protein'], str) and isinstance(m['trajectory'], str), 'window identity missing')
    for key in ('anchor_frame', 'horizon', 'target_frame', 'sampling_seed'):
        require(isinstance(m[key], int) and not isinstance(m[key], bool), f'integer metadata {key} required')
    require(m['anchor_frame'] >= 7 and m['horizon'] > 0 and m['target_frame'] == m['anchor_frame'] + m['horizon'],
            'target frame/horizon mismatch')
    require(math.isfinite(float(m['temperature_k'])), 'nonfinite temperature metadata')
    seed = int(hashlib.sha256(f"{m['protein']}|{m['trajectory']}|{m['anchor_frame']}|{m['horizon']}".encode()).hexdigest()[:12], 16) % (2**31-1)
    require(m['sampling_seed'] == seed, 'cached forecast sampling seed differs from fixed window protocol')
    # Authoritative topology validation, including empty edges, without any
    # reliance on legacy embedding residue numbers or author-chain labels.
    backbone_geometry_losses_v3(row['observed_backbone'].double(),
        chemical_aatype=row['chemical_aatype'], atom_mask=row['atom_mask'],
        residue_mask=mask, geometry_row_ids=row['geometry_row_ids'], peptide_edges=row['peptide_edges'])


def topology_audit(row):
    mask = row['observed_mask']
    ri, ci = row['embedding_residue_index'], row['embedding_chain_id']
    legacy = {(i, i+1) for i in range(len(mask)-1)
              if mask[i] and mask[i+1] and ci[i] == ci[i+1] and ri[i+1]-ri[i] == 1}
    actual = {(int(i), int(j)) for i, j in row['peptide_edges'] if mask[i] and mask[j]}
    return dict(explicit_peptide_edges=sorted(actual), legacy_v2_inferred_edges=sorted(legacy),
        edge_sets_equal=actual == legacy, explicit_only=sorted(actual-legacy), legacy_only=sorted(legacy-actual),
        legacy_connectivity_is_physical=actual == legacy,
        interpretation='v2 connection/torsion metrics use legacy adjacent-row ri/ci and are historical diagnostics when edges differ; v3 uses explicit PSF edges')


def control_permutation(row):
    key = '|'.join(map(str, window_key(row))) + '|oracle_H_residue_permutation'
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**63-1)
    indices = torch.where(row['observed_mask'])[0]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    permutation = indices[torch.randperm(len(indices), generator=generator)]
    result = row['oracle_H'].clone()
    result[:, indices] = result[:, permutation]
    return result, dict(seed=seed, valid_rows=indices.tolist(), source_rows=permutation.tolist(),
                        identity=bool(torch.equal(indices, permutation)))


def score_backbone(backbone, row, *, role, fixed_ca=None):
    backbone = backbone.detach().cpu().double()
    mask = row['observed_mask']
    score_mask = row['atom_mask'] & row['target_atom_mask'] & mask[:, None]
    require(bool(score_mask.any()), 'no common observed/target scoring atoms')
    require(backbone.ndim == 4 and backbone.shape[1:] == (len(mask), 4, 3), 'backbone sample shape differs')
    result = dict(role=role, n_samples=len(backbone), backbone_sha256=tensor_sha(backbone),
        error=backbone_error_summary(backbone, row['target_backbone'].double(), mask, atom_mask=score_mask),
        legacy_geometry=backbone_geometry_summary(backbone, mask, atom_mask=score_mask,
            chain_id=row['embedding_chain_id'], residue_index=row['embedding_residue_index'],
            aatype=row['chemical_aatype']),
        explicit_geometry=_plain(backbone_geometry_losses_v3(backbone,
            chemical_aatype=row['chemical_aatype'], atom_mask=score_mask,
            residue_mask=mask, geometry_row_ids=row['geometry_row_ids'], peptide_edges=row['peptide_edges'])))
    if fixed_ca is not None:
        require(torch.equal(backbone[:, mask, 1], fixed_ca.detach().cpu().double()[:, mask]), 'fixed CA changed')
        result.update(fixed_ca_bitwise_equal=True, fixed_ca_sha256=tensor_sha(fixed_ca))
    return result


@torch.no_grad()
def score_row(heads, row):
    """Decode all heads without future targets/masks entering any forward call."""
    require(set(heads) == set(HEADS), 'init/coordinate/geometry heads required')
    require(all(not module.training for head in heads.values() for module in head.modules()), 'strict eval mode required')
    validate_row(row)
    routes = {}
    permutation_H, permutation = control_permutation(row)
    inputs = {'oracle': (row['oracle_H'], row['oracle_CA']),
              'forecast': (row['forecast_H'], row['forecast_CA'])}
    for name, head in heads.items():
        device = next(head.parameters()).device
        variants = dict(inputs)
        if name in ('init', 'geometry'):
            variants.update(oracle_H_anchor=(row['anchor_H'], row['oracle_CA']),
                            oracle_H_zero=(torch.zeros_like(row['oracle_H']), row['oracle_CA']),
                            oracle_H_permuted=(permutation_H, row['oracle_CA']))
        for variant, (H, CA) in variants.items():
            n = len(H)
            output = head(H.to(device), row['observed_R'].expand(n, -1, -1, -1).to(device), CA.to(device),
                row['embedding_aatype'].expand(n, -1).to(device),
                row['chemical_aatype'].expand(n, -1).to(device), row['observed_mask'].expand(n, -1).to(device))
            require(output['atom_mask'].dtype == torch.bool and output['atom_mask'].shape == (n, len(row['observed_mask']), 4),
                    'head atom mask shape differs')
            needed = row['atom_mask'] & row['observed_mask'][:, None]
            require(bool(output['atom_mask'].cpu()[:, needed].all()), 'head omits observed topology atoms')
            require(torch.equal(output['backbone'].cpu()[:, row['observed_mask'], 1], CA[:, row['observed_mask']]),
                    'head changed fixed CA before scoring')
            role = ('forecast: observed-only cached H/CA' if variant == 'forecast' else
                    'oracle readout: fixed future-derived CA; H interventions diagnose readout reliance, not a forecast or a separately fitted CA-only model')
            routes[f'{name}/{variant}'] = score_backbone(output['backbone'], row, role=role, fixed_ca=CA)
    routes['persistence_MD'] = score_backbone(row['observed_backbone'], row, role='forecast control: observed real backbone')
    routes['target_MD'] = score_backbone(row['target_backbone'], row, role='reference: true aligned endpoint backbone, self error zero')
    mask = row['observed_mask'][:, None]
    return dict(**row['metadata'], topology=topology_audit(row), H_control_permutation=permutation,
        cache_fingerprints=row.get('fingerprints'), cache_boundary=row.get('boundary'),
        topology_source={key:row.get('topology', {}).get(key) for key in
                         ('sha256','schema_version','metadata_hashes','source_identity')},
        scoring=dict(valid_observed_residues=int(row['observed_mask'].sum()),
            observed_atom_counts=(row['atom_mask'] & mask).sum(0).tolist(),
            shared_scored_atom_counts=(row['atom_mask'] & row['target_atom_mask'] & mask).sum(0).tolist(),
            target_mask_used_only_for_scoring=True, arithmetic='CPU float64'), routes=routes)


def summarize(rows):
    """Equal members per window, equal windows per protein, equal proteins."""
    result = {}
    for horizon in sorted({r['horizon'] for r in rows}):
        selected = [r for r in rows if r['horizon'] == horizon]
        grouped, all_metrics = defaultdict(list), set()
        for row in selected:
            for route, values in row['routes'].items():
                groups = {'error': values['error']['mean'], 'legacy_geometry': values['legacy_geometry']['mean'],
                          'explicit_geometry': values['explicit_geometry']['losses']}
                for group, metrics in groups.items():
                    for metric, value in metrics.items():
                        all_metrics.add((route, group, metric))
                        if group == 'explicit_geometry':
                            count_name = 'n_nonbonded_atom_pairs' if metric == 'backbone_interresidue_clash' else 'n_'+metric
                            if not bool(torch.as_tensor(values['explicit_geometry']['counts'][count_name]).gt(0).any()):
                                value = None  # Training's differentiable empty zero is not measured good geometry.
                        if value is not None:
                            grouped[route, group, metric, row['protein']].append(value)
        aggregated = defaultdict(list)
        for (route, group, metric, protein), values in grouped.items():
            aggregated[route, group, metric].append(sum(values)/len(values))
        means = {}
        for route, group, metric in sorted(all_metrics):
            values = aggregated[route, group, metric]
            n_windows = sum(len(v) for (a,g,m,p),v in grouped.items() if (a,g,m)==(route,group,metric))
            means.setdefault(route, {}).setdefault(group, {})[metric] = dict(
                mean=sum(values)/len(values) if values else None, n_proteins=len(values), n_valid_windows=n_windows)
        result[str(horizon)] = dict(n_windows=len(selected), n_proteins=len({r['protein'] for r in selected}), means=means)
    return result


def validate_panel(rows, expected_windows=200, expected_proteins=50):
    require(len(rows) == expected_windows and len({window_key(r) for r in rows}) == len(rows), 'wrong panel budget or duplicate window')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['metadata']['protein']].append(row)
    require(len(grouped) == expected_proteins, 'wrong protein budget')
    for values in grouped.values():
        require(sorted(r['metadata']['horizon'] for r in values) == [1, 4, 16, 64], 'each protein needs four fixed horizons')
        require(len({(r['metadata']['trajectory'], r['metadata']['temperature_k'], r['metadata']['anchor_frame']) for r in values}) == 1,
                'horizon windows must share observed anchor and trajectory')
    if expected_windows == 200:
        require(all(sum(r['metadata']['temperature_k'] == t for r in rows) == 40 for t in (320.,348.,379.,413.,450.)),
                'formal panel requires registered temperature allocation')


def validate_cache(cache, allow_smoke=False):
    require(cache['schema_version'] == 'backbone_readout_cache_v1', 'cache schema differs')
    protocol = cache['protocol']
    expected = dict(mode='val', dataset_role='development_validation', forecast_observed_only=True,
        before_target_read=True, n_forecast_samples=12, horizons=[1,4,16,64], max_residues=96,
        context_frames=8, seed=0)
    require(all(protocol.get(key) == value for key,value in expected.items()), 'cache validation/observed-only contract differs')
    n = protocol.get('n_windows')
    require(n == 200 or (allow_smoke and n in (4,32)), 'formal cache requires 200 windows; smoke needs explicit opt-in')
    completion = cache.get('completion', {})
    require(completion.get('completed_utc') and completion.get('frozen_inputs_verified') is True
            and isinstance(completion.get('model_sha256_before'), str)
            and completion['model_sha256_before'] == completion.get('model_sha256_after'), 'cache completion/model freeze audit missing')
    validate_panel(cache['rows'], expected_windows=n, expected_proteins=n//4)
    for row in cache['rows']:
        boundary = row.get('boundary', {})
        anchor, target = row['metadata']['anchor_frame'], row['metadata']['target_frame']
        require(boundary.get('forecast_before_target_read') is True
                and boundary.get('future_reencoding_for_forecast') == 0
                and boundary.get('observed_frames') == list(range(anchor-7, anchor+1))
                and boundary.get('label_frames') == [target], 'cache row observed/label boundary differs')


def validate_checkpoints(saved, paths, *, formal_panel):
    """Audit fixed latest selection and the shared initial head/scale provenance."""
    weights = dict(c_n=1.,ca_c_n=4.,c_n_ca=4.,backbone_interresidue_clash=100.)
    initial = saved['init']
    initial_hash = state_sha256(initial['head'])
    require(initial['step']==0 and initial['protocol']['mode']=='init'
            and initial['protocol']['optimizer_steps']==0, 'init is not the zero-update control')
    for name,ck in saved.items():
        require(ck['frozen_inputs_verified'] is True and ck['head_tensor_sha256']==state_sha256(ck['head']),
                'checkpoint freeze/tensor hash audit differs')
        require(ck['protocol']['schema_version']=='fixed_ca_backbone_readout_training_v1', 'training protocol version differs')
        if name=='init': continue
        p=ck['protocol']
        require(ck['arm']==p['arm']==name and p['mode']=='train' and ck['step']==p['steps']==p['optimizer_steps'],
                'checkpoint arm/step registration differs')
        require(ck['step']==500 if formal_panel else ck['step'] in (4,32,500), 'wrong fixed latest step')
        require(not formal_panel or p['formal'] is True, 'smoke training cannot support formal evaluation')
        require(p['checkpoint_selection']=='fixed latest; no validation selection'
                and p['validation_used_for_training'] is False and p['encoder_CA_dynamics_frozen'] is True
                and p['seed']==42 and p['geometry_weights']==weights, 'training selection/objective contract differs')
        require(ck['initial_head_tensor_sha256']==initial_hash and local(p['initial'])==local(paths['init']),
                'trained head used another initialization')
        require(ck['head_tensor_sha256']!=initial_hash and ck['parameter_delta_l2']>0, 'trained head did not update')
        coefficient=p['geometry_lambda']
        require(coefficient==0 if name=='coordinate' else math.isfinite(coefficient) and .1<=coefficient<=10,
                'geometry coefficient differs from arm contract')
    left,right=saved['coordinate'],saved['geometry']
    require(left['step']==right['step'] and left['input_hashes']==right['input_hashes']
            and left['source_hashes']==right['source_hashes'], 'trained arms used different steps or inputs/sources')
    for field in ('train_cache','initial','scale_file','scale_sha256'):
        require(left['protocol'][field]==right['protocol'][field], f'trained arms differ in {field}')
    return initial_hash


def validate_scale(scale, saved, initial_hash):
    p=saved['geometry']['protocol']
    require(scale['frozen_inputs_verified'] is True and scale['head_tensor_sha256']==initial_hash
            and scale['protocol']['mode']=='calibrate'
            and scale['protocol']['train_cache']==p['train_cache']
            and scale['protocol']['initial']==p['initial']
            and scale['protocol']['geometry_weights']==p['geometry_weights'], 'scale calibration provenance differs')
    rows=scale['per_row'];n=min(32,saved['geometry']['step'])
    require(len(rows)==n==scale['protocol']['n_calibration_rows'], 'scale training-row budget differs')
    norms=[[float(r[key]) for r in rows] for key in ('coordinate_grad_norm','weighted_geometry_grad_norm')]
    require(all(math.isfinite(v) and v>=0 for group in norms for v in group), 'invalid calibration gradients')
    c,g=[sum(group)/n for group in norms]
    require(c>0 and g>0, 'degenerate calibration gradients')
    raw=.5*c/g;coefficient=min(10.,max(.1,raw))
    expected=dict(mean_coordinate_grad_norm=c,mean_weighted_geometry_grad_norm=g,unclipped_scale=raw,scale=coefficient)
    require(all(math.isclose(float(scale[k]),v,rel_tol=1e-12,abs_tol=1e-12) for k,v in expected.items())
            and math.isclose(p['geometry_lambda'],coefficient,rel_tol=1e-12,abs_tol=1e-12), 'geometry scale does not follow training-only registered formula')


def evaluate(cache_path, checkpoint_paths, output, *, device='cpu', allow_smoke=False):
    output = local(output)
    protocol_file, snapshot = output.with_suffix('.protocol.json'), output.with_suffix('.source_snapshot')
    require(not any(p.exists() for p in (output, protocol_file, snapshot)), 'refusing to overwrite result/protocol/source snapshot')
    inputs, declared_sources = {}, {}
    def track(path, expected=None):
        path = local(path); digest = sha(path)
        require(expected is None or digest == expected, f'input/source hash mismatch: {path}')
        require(path not in inputs or inputs[path] == digest, f'changed input/source: {path}')
        inputs[path] = digest
        return path
    cache_path = track(cache_path)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    validate_cache(cache, allow_smoke=allow_smoke)
    for category in ('source_hashes', 'input_hashes'):
        require(isinstance(cache[category], dict) and bool(cache[category]), f'cache {category} required')
        for path, digest in cache[category].items():
            tracked = track(path, digest)
            if category == 'source_hashes': declared_sources[str(tracked.relative_to(REPO))] = digest
    heads, saved = {}, {}
    for name in HEADS:
        path = track(checkpoint_paths[name]); saved[name] = torch.load(path, map_location='cpu', weights_only=False)
        ck = saved[name]
        require(ck['head_version'] == HEAD_VERSION and isinstance(ck['protocol'], dict), 'head version/protocol differs')
        for category in ('input_hashes', 'source_hashes'):
            require(isinstance(ck[category], dict) and bool(ck[category]), f'checkpoint {category} required')
            for path, digest in ck[category].items():
                tracked = track(path, digest)
                if category == 'source_hashes': declared_sources[str(tracked.relative_to(REPO))] = digest
        with torch.random.fork_rng(devices=[]):
            head = BackboneReadoutV2(BackboneReadoutConfig(**ck['head_config']))
        head.load_state_dict(ck['head'], strict=True)
        require(all(torch.isfinite(v).all() for v in head.state_dict().values()), 'nonfinite head state')
        heads[name] = head.to(device).eval().requires_grad_(False)
    require(saved['init']['step'] == 0 and saved['coordinate']['step'] == saved['geometry']['step'] > 0,
            'require zero-update init and matched fixed-step trained heads')
    require(all(saved[n]['head_config'] == saved['init']['head_config'] and
                saved[n]['initialization'] == saved['init']['initialization'] for n in HEADS), 'head architecture/initialization differs')
    initial_hash=validate_checkpoints(saved,checkpoint_paths,formal_panel=len(cache['rows'])==200)
    trained_protocol=saved['geometry']['protocol']
    scale_path=track(trained_protocol['scale_file'],trained_protocol['scale_sha256'])
    scale=json.loads(scale_path.read_text())
    validate_scale(scale,saved,initial_hash)
    source_hashes = {**declared_sources, **{path: sha(track(REPO/path)) for path in SOURCES}}
    state_before = {name: state_sha256(head) for name, head in heads.items()}
    protocol = dict(schema_version=VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
        dataset_role='development_validation', formal_panel=len(cache['rows']) == 200,
        cache=str(cache_path.relative_to(REPO)),
        cache_protocol=cache['protocol'], cache_schema_version=cache['schema_version'],
        cache_completion=cache['completion'], cache_provenance=cache.get('provenance'),
        scale_calibration=dict(path=str(scale_path.relative_to(REPO)),sha256=sha(scale_path),
            scale=scale['scale'],n_calibration_rows=scale['protocol']['n_calibration_rows']),
        checkpoints={name:dict(path=str(local(checkpoint_paths[name]).relative_to(REPO)), step=saved[name]['step'],
            protocol=saved[name]['protocol'], initialization=saved[name]['initialization']) for name in HEADS},
        input_hashes={str(p.relative_to(REPO)):d for p,d in inputs.items()}, source_hashes=source_hashes,
        optimizer_steps=0, test_opened=False, bioemu2_rerun=False,
        target_backbone_or_target_mask_given_to_head=False, oracle_inputs_are_future_derived=True,
        runtime=dict(device=str(device), torch_version=torch.__version__, torch_threads=torch.get_num_threads()),
        metric_arithmetic='CPU float64; head inference float32; exact cached CA checked before and after conversion',
        aggregation='equal samples within window, equal windows within protein, equal proteins per horizon; no CI or pass claim',
        limitations=['oracle is future-derived readout diagnostic', 'H interventions retain oracle CA and observed frames/sequence; not a separately fitted CA-only model',
            'legacy v2 connectivity is not physical where it disagrees with explicit PSF edges',
            'v3 flat-bottom losses are training diagnostics, not complete T1 or all-atom quality criteria; empty terms retain raw zero/count0 but aggregate as undefined'])
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(exist_ok=False)
    for path, digest in source_hashes.items():
        destination = snapshot/path; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream: stream.write((REPO/path).read_bytes())
        require(sha(destination) == digest, 'source snapshot differs')
    with protocol_file.open('x') as stream: json.dump(_plain(protocol), stream, indent=2, allow_nan=False)
    rows = []
    for index, row in enumerate(cache['rows']):
        rows.append(score_row(heads, row))
        if (index+1) % 10 == 0: print(f'backbone-readout windows={index+1}/{len(cache["rows"])}', flush=True)
    require(all(state_sha256(head) == state_before[name] for name,head in heads.items()), 'evaluation mutated head state')
    require(all(p.grad is None for head in heads.values() for p in head.parameters()), 'evaluation created parameter gradients')
    for path, digest in inputs.items(): require(sha(path) == digest, f'input/source changed during evaluation: {path}')
    result = dict(protocol, completed_utc=datetime.now(timezone.utc).isoformat(), frozen_inputs_verified=True,
        head_tensor_sha256=state_before, n_windows=len(rows), n_proteins=len({r['protein'] for r in rows}),
        topology_mismatch_windows=sum(not r['topology']['edge_sets_equal'] for r in rows),
        per_window=rows, by_horizon=summarize(rows))
    with output.open('x') as stream: json.dump(_plain(result), stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    for name in HEADS: parser.add_argument(f'--{name}-checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--allow-smoke', action='store_true', help='explicitly permit registered 4/32-window smoke caches; never labels them formal')
    parser.add_argument('--torch-threads', type=int, default=2)
    args = parser.parse_args(); torch.set_num_threads(args.torch_threads)
    evaluate(args.cache, {name:getattr(args, f'{name}_checkpoint') for name in HEADS}, args.out,
             device=args.device, allow_smoke=args.allow_smoke)


if __name__ == '__main__':
    main()
