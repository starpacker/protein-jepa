"""Preregister and audit the fixed-step development backbone readout pilot.

Read-only CPU aggregation. No fitting, sampling, checkpoint selection, test
access, metric rerun or pass threshold. Registration and result are exclusive.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from protein_jepa_frame.utils.contracts import state_sha256


REPO = Path(__file__).resolve().parents[1]
BASE = Path('results/hs_joint_20261101/backbone_repair_v2')
ARTIFACTS = Path('artifacts/hs_joint_20261101/backbone_repair_v2')
TRAIN_CACHE = Path('results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt')
VAL_CACHE = Path('results/hs_joint_20261101/backbone_repair/readout_val200_v2.pt')
DOCUMENT = Path('docs/BACKBONE_READOUT_PILOT_V2_20261101.md')
DOCUMENT_SHA = '6411030d03649786a11682d2501876da1a61ad30eaedbc2da34b6adb2f1a5496'
SCRIPT = Path('scripts/summarize_backbone_readout_pilot.py')
VERSION = 'backbone_readout_pilot_paired_summary_v1'
EVALUATOR_VERSION = 'fixed_ca_backbone_readout_pilot_eval_v1'
HORIZONS = (1, 4, 16, 64)
HEADS = ('init', 'coordinate', 'geometry')
GROUPS = ('overall', 'h1', 'h4', 'h16', 'h64')
CONTRASTS = (('geometry', 'coordinate'), ('geometry', 'init'), ('coordinate', 'init'))
CONTROLS = ('anchor', 'zero', 'permuted')
GEOMETRY = ('c_n', 'ca_c_n', 'c_n_ca', 'backbone_interresidue_clash')
METRICS = ('coordinate_rmsd_NCO_angstrom', *GEOMETRY)
WEIGHTS = dict(zip(GEOMETRY, (1., 4., 4., 100.)))
ROUTES = tuple(f'{h}/{r}' for h in HEADS for r in ('oracle', 'forecast')) + tuple(
    f'{h}/oracle_H_{c}' for h in ('init', 'geometry') for c in CONTROLS) + ('persistence_MD', 'target_MD')
BOOTSTRAPS = 10000
SEED = 20261101


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(path):
    path = Path(path)
    path = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(REPO.resolve()), 'path outside repository')
    require(path.name != 'test.jsonl' and 'bioemu' not in str(path).lower(), 'excluded path')
    return path


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), 'nonfinite/non-numeric statistic')
    return float(value)


def close(a, b):
    return math.isclose(number(a), number(b), rel_tol=1e-10, abs_tol=1e-12)


def training_close(a, b):
    # Training combines float32 tensors before serializing individual terms;
    # reconstructing their sum in Python float64 can differ by float32 rounding.
    return math.isclose(number(a), number(b), rel_tol=2e-6, abs_tol=1e-6)


def register(path):
    path = local(path)
    require(not path.exists(), 'refusing registration overwrite')
    require(not local(BASE/'evaluation200.json').exists(), 'registration requires unread/uncreated formal evaluation')
    require(sha(DOCUMENT) == DOCUMENT_SHA, 'frozen preregistration document differs')
    protocol = dict(schema_version=VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
        document=str(DOCUMENT), document_sha256=DOCUMENT_SHA,
        source_sha256=sha(SCRIPT), evaluation=str(BASE/'evaluation200.json'),
        output=str(BASE/'summary.json'), evaluator_schema=EVALUATOR_VERSION,
        contrasts=[list(x) for x in CONTRASTS], routes=list(ROUTES), groups=list(GROUPS),
        inferential_metrics=list(METRICS), h_control_contrast='intervention minus corresponding oracle',
        bootstrap=dict(n=BOOTSTRAPS, seed=SEED, unit='protein', interval='percentile_95',
            rng='numpy.default_rng restarted per comparison; sorted protein IDs',
            multiplicity='marginal exploratory; no adjustment or equivalence claim'),
        checkpoint_rule='fixed latest 500 steps, no validation selection',
        training_seed=42, training_windows=500, training_unique_proteins=500,
        validation_windows=200, validation_unique_proteins=50, forecast_members=12,
        geometry_weights=WEIGHTS, gradient_scale_rows=32,
        scale_formula='clip(.5*mean(coord_grad_norm)/mean(weighted_geometry_grad_norm),.1,10)',
        descriptive_only=['full backbone/atom coordinate error', 'independently fitted shape error', 'legacy geometry'],
        test_opened=False, bioemu2_rerun=False, no_pass_or_noninferiority_threshold=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    return protocol


def window_key(row):
    return (row['protein'], row['trajectory'], float(row['temperature_k']), row['anchor_frame'], row['horizon'])


def count_key(metric):
    return 'n_nonbonded_atom_pairs' if metric == 'backbone_interresidue_clash' else 'n_'+metric


def primary(route):
    """Derive NCO RMSD per member before averaging; never RMS average atoms."""
    n = route['n_samples']
    require(isinstance(n, int) and not isinstance(n, bool) and n > 0, 'invalid member count')
    errors = route['error']['per_sample']
    require(len(errors) == n, 'error member count differs')
    nco = []
    for sample in errors:
        numerator = 0.; denominator = 0
        for atom in ('N', 'C', 'O'):
            count = sample['n_atoms_'+atom]
            require(isinstance(count, int) and not isinstance(count, bool) and count >= 0, 'invalid NCO atom count')
            rms = sample['coordinate_rmsd_'+atom+'_angstrom']
            if count:
                require(rms is not None and number(rms) >= 0, 'missing NCO error with valid atoms')
                numerator += count * number(rms)**2
                denominator += count
            else:
                require(rms is None, 'empty atom error must be null')
        if denominator:
            nco.append(math.sqrt(numerator/denominator))
    result = {METRICS[0]: sum(nco)/len(nco) if nco else None}
    geometry = route['explicit_geometry']
    for metric in GEOMETRY:
        values, counts = geometry['per_sample'][metric], geometry['counts'][count_key(metric)]
        require(len(values) == len(counts) == n, 'geometry member count differs')
        valid = []
        for value, count in zip(values, counts):
            require(isinstance(count, int) and not isinstance(count, bool) and count >= 0, 'invalid geometry count')
            require(number(value) >= 0, 'negative geometry loss')
            if count:
                valid.append(number(value))
            else:
                require(value == 0, 'empty training geometry must retain raw zero')
        result[metric] = sum(valid)/len(valid) if valid else None
        require(close(geometry['losses'][metric], result[metric] or 0), 'geometry aggregate differs from members')
    return result


def descriptive(route):
    result = {f'error/{k}': v for k, v in route['error']['mean'].items()}
    result.update({f'legacy_geometry/{k}': v for k,v in route['legacy_geometry']['mean'].items()})
    values = primary(route)
    result['error/'+METRICS[0]] = values[METRICS[0]]
    result.update({'explicit_geometry/'+k: values[k] for k in GEOMETRY})
    return result


def selected(rows, group):
    return rows if group == 'overall' else [r for r in rows if r['horizon'] == int(group[1:])]


def paired(rows, candidate, reference, metric, *, n_bootstrap=BOOTSTRAPS, seed=SEED):
    by_protein = defaultdict(list)
    for row in rows:
        a, b = primary(row['routes'][candidate])[metric], primary(row['routes'][reference])[metric]
        if a is not None and b is not None:
            by_protein[row['protein']].append((a, b))
    proteins = []
    for protein, values in sorted(by_protein.items()):
        means = np.asarray(values, dtype=np.float64).mean(0)
        proteins.append(dict(protein=protein, candidate=float(means[0]), reference=float(means[1]),
                             delta=float(means[0]-means[1]), n_paired_windows=len(values)))
    base = dict(n_proteins=len(proteins), n_paired_windows=sum(p['n_paired_windows'] for p in proteins),
                per_protein=proteins)
    if not proteins:
        return dict(base, candidate_mean=None, reference_mean=None, delta=None, ci95=None)
    delta = np.asarray([p['delta'] for p in proteins], dtype=np.float64)
    ids = np.random.default_rng(seed).integers(0, len(delta), size=(n_bootstrap, len(delta)))
    interval = np.quantile(delta[ids].mean(1), [.025, .975], method='linear').tolist()
    return dict(base, candidate_mean=float(np.mean([p['candidate'] for p in proteins])),
                reference_mean=float(np.mean([p['reference'] for p in proteins])),
                delta=float(delta.mean()), ci95=interval)


def analysis(rows):
    main, controls, descriptions = {}, {}, {}
    for group in GROUPS:
        subset = selected(rows, group)
        main[group], controls[group], descriptions[group] = {}, {}, {}
        for candidate, reference in CONTRASTS:
            for route in ('oracle', 'forecast'):
                label = f'{candidate}-{reference}/{route}'
                main[group][label] = {metric: paired(subset, f'{candidate}/{route}', f'{reference}/{route}', metric)
                                      for metric in METRICS}
        for head in ('init', 'geometry'):
            for intervention in CONTROLS:
                label = f'{head}/H_{intervention}-oracle'
                controls[group][label] = {metric: paired(subset, f'{head}/oracle_H_{intervention}', f'{head}/oracle', metric)
                                          for metric in METRICS}
        for route in ROUTES:
            collected, names = defaultdict(list), set()
            for row in subset:
                for metric, value in descriptive(row['routes'][route]).items():
                    names.add(metric)
                    if value is not None:
                        collected[metric, row['protein']].append(number(value))
            descriptions[group][route] = {}
            for metric in sorted(names):
                values = [v for (m,_),v in collected.items() if m == metric]
                descriptions[group][route][metric] = dict(
                    mean=float(np.mean([np.mean(v) for v in values])) if values else None,
                    n_proteins=len(values), n_valid_windows=sum(map(len, values)))
    return dict(main_contrasts=main, H_controls=controls, descriptive=descriptions)


class Audit:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = local(path); key = str(path.relative_to(REPO)); digest = sha(path)
        require(expected is None or digest == expected, f'hash mismatch: {key}')
        require(key not in self.hashes or self.hashes[key] == digest, f'input changed: {key}')
        self.hashes[key] = digest
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text())

    def checkpoint(self, path):
        return torch.load(self.track(path), map_location='cpu', weights_only=False)

    def declarations(self, value, snapshot=None):
        for category in ('input_hashes', 'source_hashes'):
            require(isinstance(value.get(category), dict) and value[category], 'missing declared provenance')
            for path, digest in value[category].items():
                resolved = self.track(path, digest)
                if category == 'source_hashes' and snapshot is not None:
                    self.track(local(snapshot)/resolved.relative_to(REPO), digest)

    def verify(self):
        for path, digest in self.hashes.items():
            require(sha(path) == digest, f'input changed before summary write: {path}')


def check_training_protocol(protocol, mode):
    expected = dict(schema_version='fixed_ca_backbone_readout_training_v1', mode=mode,
        formal=True, seed=42, optimizer='AdamW', learning_rate=1e-4, final_learning_rate=1e-5,
        warmup_steps=25, weight_decay=1e-4, gradient_clip=1., geometry_weights=WEIGHTS,
        checkpoint_selection='fixed latest; no validation selection', encoder_CA_dynamics_frozen=True,
        validation_used_for_training=False, test_opened=False, bioemu2_rerun=False)
    require(all(protocol.get(k) == v for k,v in expected.items()), 'training protocol differs from preregistration')


def check_scale(scale, train_rows, initial_hash):
    check_training_protocol(scale['protocol'], 'calibrate')
    require(scale['protocol']['n_calibration_rows'] == 32 and scale['protocol']['optimizer_steps'] == 0,
            'gradient scale budget differs')
    require(scale['frozen_inputs_verified'] is True and scale['head_tensor_sha256'] == initial_hash, 'gradient scale initialization differs')
    rows = scale['per_row']
    require(len(rows) == 32 and [r['metadata'] for r in rows] == [r['metadata'] for r in train_rows[:32]],
            'gradient calibration did not use first 32 training rows')
    coord = sum(number(r['coordinate_grad_norm']) for r in rows)/32
    geom = sum(number(r['weighted_geometry_grad_norm']) for r in rows)/32
    require(all(number(r[k]) >= 0 for r in rows for k in ('coordinate_grad_norm','weighted_geometry_grad_norm')), 'negative calibration gradient norm')
    require(coord > 0 and geom > 0, 'nonpositive gradient scale means')
    require(close(coord, scale['mean_coordinate_grad_norm']) and close(geom, scale['mean_weighted_geometry_grad_norm']),
            'gradient norm means differ')
    raw = .5*coord/geom
    require(close(raw, scale['unclipped_scale']) and close(min(10.,max(.1,raw)), scale['scale']), 'incorrect scale formula')
    return dict(n_rows=32, mean_coordinate_grad_norm=coord, mean_weighted_geometry_grad_norm=geom,
                unclipped_scale=raw, scale=scale['scale'])


def validate_report(report):
    require(report['schema_version'] == EVALUATOR_VERSION and report['formal_panel'] is True, 'formal evaluator schema/panel required')
    require(report['dataset_role'] == 'development_validation' and report['n_windows'] == 200 and report['n_proteins'] == 50,
            'wrong development panel budget')
    require(report['frozen_inputs_verified'] is True and report['optimizer_steps'] == 0 and report['test_opened'] is False,
            'evaluator freeze/test contract differs')
    require(report['target_backbone_or_target_mask_given_to_head'] is False and report['oracle_inputs_are_future_derived'] is True,
            'oracle/forecast label semantics differ')
    p = report['cache_protocol']
    require(all(p.get(k) == v for k,v in dict(mode='val', dataset_role='development_validation', n_windows=200,
        n_forecast_samples=12, seed=0, horizons=list(HORIZONS), context_frames=8, max_residues=96,
        forecast_observed_only=True, before_target_read=True).items()), 'validation cache contract differs')
    rows = report['per_window']
    require(len(rows) == 200 and len({window_key(r) for r in rows}) == 200, 'missing or duplicate evaluation windows')
    proteins = defaultdict(list)
    for row in rows:
        proteins[row['protein']].append(row)
        require(set(row['routes']) == set(ROUTES), '14-route contract differs')
        require(row['scoring']['target_mask_used_only_for_scoring'] is True and row['scoring']['arithmetic'] == 'CPU float64',
                'scoring precision/mask boundary differs')
        require(row['cache_boundary']['forecast_before_target_read'] is True and
                row['cache_boundary']['future_reencoding_for_forecast'] == 0, 'forecast cache leaked future')
        for route, values in row['routes'].items():
            n = 12 if route.endswith('/forecast') else 1
            require(values['n_samples'] == values['error']['n_samples'] == values['legacy_geometry']['n_samples'] == n,
                    'incomplete member scoring')
            require(values['error']['schema_version'] == values['legacy_geometry']['schema_version'] == 'backbone_metrics_v2', 'metric version differs')
            require(values['explicit_geometry']['schema_version'] == 'backbone_objectives_v3.0_explicit_peptide_edges', 'PSF metric version differs')
            require(values['explicit_geometry']['conventions']['connection_tolerance_factor'] == 12.
                    and values['explicit_geometry']['conventions']['clash_overlap_angstrom'] == 1.5
                    and values['explicit_geometry']['conventions']['arithmetic'] == 'torch.float64', 'geometry metric convention differs')
            primary(values)
            if '/' in route:
                require(values['fixed_ca_bitwise_equal'] is True, 'CA preservation failed')
        for variant in ('oracle', 'forecast'):
            keys = [f'{h}/{variant}' for h in HEADS]
            if variant == 'oracle':
                keys += [f'{h}/oracle_H_{c}' for h in ('init','geometry') for c in CONTROLS]
            require(len({row['routes'][r]['fixed_ca_sha256'] for r in keys}) == 1, 'heads/controls used different CA')
    require(len(proteins) == 50, 'protein budget differs')
    for group in proteins.values():
        require(sorted(r['horizon'] for r in group) == list(HORIZONS), 'missing protein horizons')
        require(len({(r['trajectory'],r['temperature_k'],r['anchor_frame']) for r in group}) == 1, 'different horizon anchors')
    require(all(sum(r['temperature_k'] == t for r in rows) == 40 for t in (320.,348.,379.,413.,450.)), 'temperature allocation differs')
    return rows


def audit_artifacts(audit, report):
    """Validate fixed-step provenance without rerunning model or metrics."""
    train, val = audit.checkpoint(TRAIN_CACHE), audit.checkpoint(VAL_CACHE)
    for cache, path in ((train,TRAIN_CACHE),(val,VAL_CACHE)):
        require(cache['schema_version'] == 'backbone_readout_cache_v1', 'cache schema differs')
        audit.declarations(cache, local(path).with_suffix('.source_snapshot'))
        c = cache['completion']
        require(c['frozen_inputs_verified'] is True and c['model_sha256_before'] == c['model_sha256_after'], 'cache model changed')
    require(train['completion']['model_sha256_before'] == val['completion']['model_sha256_before'], 'train/validation frozen models differ')
    tp = train['protocol']
    require(all(tp.get(k) == v for k,v in dict(mode='train', dataset_role='training_readout_calibration', n_windows=500,
        seed=42, horizons=list(HORIZONS), context_frames=8, max_residues=96, n_forecast_samples=0).items()), 'training cache budget differs')
    train_rows = train['rows']
    train_proteins = {r['metadata']['protein'] for r in train_rows}
    val_keys = {window_key(r['metadata']) for r in val['rows']}
    require(len(train_rows) == len(train_proteins) == 500, 'training proteins are not unique')
    require([r['metadata']['horizon'] for r in train_rows] == [HORIZONS[i%4] for i in range(500)], 'training horizon order differs')
    require(not train_proteins & {r['metadata']['protein'] for r in val['rows']}, 'training/validation protein overlap')
    require(val_keys == {window_key(r) for r in report['per_window']} and val['protocol'] == report['cache_protocol'], 'evaluation panel differs from frozen cache')
    require(len(val['rows']) == len(val_keys) == 200, 'validation cache has duplicate/missing rows')
    require(report['cache'] == str(VAL_CACHE), 'evaluation used different validation cache')
    report_rows = {window_key(r): r for r in report['per_window']}
    for row in val['rows']:
        evaluated = report_rows[window_key(row['metadata'])]
        require(all(evaluated[k] == v for k,v in row['metadata'].items()), 'evaluation metadata differs from cache')
        require(evaluated['cache_fingerprints'] == row['fingerprints'] and evaluated['cache_boundary'] == row['boundary']
                and evaluated['topology_source']['sha256'] == row['topology']['sha256'], 'consumed input/topology fingerprint differs')
        for variant, ca in (('oracle',row['oracle_CA']),('forecast',row['forecast_CA'])):
            expected = state_sha256({'value':ca})
            require(all(evaluated['routes'][f'{head}/{variant}']['fixed_ca_sha256'] == expected for head in HEADS),
                    'evaluated CA differs from frozen cache tensor')
    initial_path = ARTIFACTS/'initial.pt'
    initial = audit.checkpoint(initial_path)
    audit.declarations(initial, local(initial_path).with_suffix('.source_snapshot'))
    require(initial['source_hashes'].get(str(DOCUMENT)) == DOCUMENT_SHA, 'initialization did not snapshot preregistration')
    check_training_protocol(initial['protocol'], 'init')
    require(initial['step'] == 0 and initial['frozen_inputs_verified'] is True, 'zero-update control differs')
    require(initial['initialization']['legacy_step'] == 6000 and initial['initialization']['legacy_checkpoint_sha256'] ==
            '5c02ad80b155edd853d788798d8512e757764f09f2fb9e8e97db06ccc2d88b0a', 'legacy initialization differs')
    initial_hash = state_sha256(initial['head'])
    require(initial_hash == initial['head_tensor_sha256'], 'initial tensor checksum differs')
    scale_path = BASE/'gradient_scale.json'
    scale = audit.json(scale_path)
    audit.declarations(scale, local(scale_path).with_suffix('.source_snapshot'))
    scale_info = check_scale(scale, train_rows, initial_hash)
    require(report['scale_calibration']['path'] == str(scale_path)
            and report['scale_calibration']['sha256'] == audit.hashes[str(scale_path)], 'evaluated scale calibration differs')
    for path in (TRAIN_CACHE, initial_path):
        require(scale['input_hashes'].get(str(path)) == audit.hashes[str(path)], 'scale uses different training cache/init')
    require(str(VAL_CACHE) not in scale['input_hashes'], 'calibration included validation cache')
    arms = {}
    for name in HEADS:
        path = initial_path if name == 'init' else ARTIFACTS/name/'checkpoint_latest.pt'
        ck = initial if name == 'init' else audit.checkpoint(path)
        require(report['checkpoints'][name]['path'] == str(path), 'evaluation did not use registered latest/init')
        require(ck['head_version'] == 'fixed_ca_backbone_readout_v2' and ck['head_config'] == initial['head_config']
                and ck['initialization'] == initial['initialization'], 'head initialization/config differs')
        require(state_sha256(ck['head']) == ck['head_tensor_sha256'] == report['head_tensor_sha256'][name], 'evaluated head tensors differ')
        require(report['checkpoints'][name]['protocol'] == ck['protocol'], 'evaluated checkpoint protocol differs')
        if name == 'init':
            continue
        audit.declarations(ck, ARTIFACTS/name/'source_snapshot')
        protocol = ck['protocol']; check_training_protocol(protocol, 'train')
        lam = scale_info['scale'] if name == 'geometry' else 0.
        require(ck['step'] == protocol['steps'] == protocol['optimizer_steps'] == 500 and ck['arm'] == protocol['arm'] == name,
                'fixed-step/arm contract differs')
        require(close(protocol['geometry_lambda'], lam) and ck['initial_head_tensor_sha256'] == initial_hash,
                'geometry lambda or initial tensors differ')
        require(ck['frozen_inputs_verified'] is True and ck['parameter_delta_l2'] > 0, 'head did not complete/update')
        for key in (str(TRAIN_CACHE),str(initial_path),str(scale_path)):
            require(ck['input_hashes'].get(key) == audit.hashes[key], 'arm trained with different inputs')
        require(str(VAL_CACHE) not in ck['input_hashes'], 'training included validation cache')
        frozen = [k for k in ck['head'] if k.startswith('ipa_blocks.') and k.endswith(('linear_k.bias','linear_b.bias'))]
        require(len(frozen) == 2*initial['head_config']['n_ipa_blocks'] and all(torch.equal(ck['head'][k],initial['head'][k]) for k in frozen), 'frozen head parameters changed')
        completion = audit.json(ARTIFACTS/name/'completion.json')
        require(completion['step'] == 500 and completion['arm'] == name and completion['frozen_inputs_verified'] is True,
                'training completion missing/differs')
        require(completion['checkpoint_sha256'] == audit.hashes[str(path)] and completion['head_tensor_sha256'] == ck['head_tensor_sha256'], 'completion checkpoint differs')
        require(completion['input_hashes'] == ck['input_hashes'] and completion['source_hashes'] == ck['source_hashes'], 'completion provenance differs')
        metrics_path = audit.track(ARTIFACTS/name/'metrics.jsonl')
        records = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        require(len(records) == 500 and [r['step'] for r in records] == list(range(1,501)), 'training log incomplete')
        require([r['metadata'] for r in records] == [r['metadata'] for r in train_rows], 'training row order differs')
        for row in records:
            require(training_close(row['objective'], number(row['coordinate_loss'])+lam*number(row['weighted_geometry_loss'])), 'logged objective differs')
            require(training_close(row['weighted_geometry_loss'],sum(WEIGHTS[k]*number(row['geometry_losses'][k]) for k in GEOMETRY)), 'logged geometry weights differ')
            step = row['step']
            lr = 1e-4*step/25 if step <= 25 else 1e-5+.5*9e-5*(1+math.cos(math.pi*(step-25)/475))
            require(close(row['lr'], lr), 'learning rate schedule differs')
        arms[name] = dict(step=500, geometry_lambda=lam, head_tensor_sha256=ck['head_tensor_sha256'],
            parameter_delta_l2=ck['parameter_delta_l2'], completed_utc=completion['completed_utc'])
    return dict(gradient_calibration=scale_info, arms=arms, training_unique_proteins=500,
                validation_unique_proteins=50, protein_overlap=0, frozen_model_sha256=train['completion']['model_sha256_before'])


def audit_executions(audit):
    jobs = [
        (TRAIN_CACHE.with_suffix('.execution.json'), 'scripts.prepare_backbone_readout_cache', {'--mode':'train','--out':str(TRAIN_CACHE)}),
        (VAL_CACHE.with_suffix('.execution.json'), 'scripts.prepare_backbone_readout_cache', {'--mode':'val','--out':str(VAL_CACHE)}),
        (BASE/'initial.execution.json', 'run_scripts.run_backbone_readout_pilot', {'--mode':'init','--out':str(ARTIFACTS/'initial.pt')}),
        (BASE/'gradient_scale_run.execution.json', 'run_scripts.run_backbone_readout_pilot', {'--mode':'calibrate','--out':str(BASE/'gradient_scale.json')}),
        *[(BASE/f'train_{arm}.execution.json', 'run_scripts.run_backbone_readout_pilot',
           {'--mode':'train','--arm':arm,'--steps':'500','--out':str(ARTIFACTS/arm)}) for arm in ('coordinate','geometry')],
        (BASE/'evaluation200_run.execution.json', 'scripts.eval_backbone_readout_pilot', {'--cache':str(VAL_CACHE),'--out':str(BASE/'evaluation200.json')}),
    ]
    completed = {}
    for path, module, flags in jobs:
        record = audit.json(path)
        require(record['exit_code'] == 0 and record.get('completed_utc'), f'incomplete/failed execution: {path}')
        require(datetime.fromisoformat(record['started_utc']) <= datetime.fromisoformat(record['completed_utc']), 'execution chronology differs')
        argv = record.get('command', record.get('argv', []))
        require('-m' in argv and argv[argv.index('-m')+1] == module and '--allow-smoke' not in argv, 'execution module/smoke differs')
        for flag, value in flags.items():
            require(argv.count(flag) == 1 and argv[argv.index(flag)+1] == value, f'execution flag differs: {flag}')
        if 'log' in record:
            audit.track(record['log'])
        completed[path.name] = dict(exit_code=0, started_utc=record['started_utc'], completed_utc=record['completed_utc'])
    return completed


def run(registration_path=BASE/'summary.protocol.json', output_path=BASE/'summary.json'):
    output_path = local(output_path)
    require(not output_path.exists(), 'refusing summary overwrite')
    audit = Audit(); registration = audit.json(registration_path)
    require(registration['schema_version'] == VERSION and registration['document_sha256'] == DOCUMENT_SHA,
            'summary registration differs')
    audit.track(DOCUMENT, DOCUMENT_SHA); audit.track(SCRIPT, registration['source_sha256'])
    require(registration['contrasts'] == [list(x) for x in CONTRASTS] and registration['inferential_metrics'] == list(METRICS)
            and registration['routes'] == list(ROUTES) and registration['groups'] == list(GROUPS)
            and registration['bootstrap']['n'] == BOOTSTRAPS and registration['bootstrap']['seed'] == SEED, 'statistics registration differs')
    require(output_path == local(registration['output']), 'output differs from registration')
    evaluation_path = local(registration['evaluation'])
    report = audit.json(evaluation_path)
    protocol = audit.json(evaluation_path.with_suffix('.protocol.json'))
    require(all(report.get(k) == v for k,v in protocol.items()), 'evaluation protocol/result mismatch')
    require(datetime.fromisoformat(registration['registered_utc']) < datetime.fromisoformat(report['registered_utc']), 'summary registered after evaluation start')
    audit.declarations(report, evaluation_path.with_suffix('.source_snapshot'))
    rows = validate_report(report)
    provenance = audit_artifacts(audit, report)
    provenance['executions'] = audit_executions(audit)
    results = analysis(rows)
    mismatch = [dict(key=list(window_key(r)), explicit_only=r['topology']['explicit_only'], legacy_only=r['topology']['legacy_only'])
                for r in rows if not r['topology']['edge_sets_equal']]
    require(len(mismatch) == report['topology_mismatch_windows'], 'topology audit count differs')
    summary = dict(schema_version=VERSION, completed_utc=datetime.now(timezone.utc).isoformat(),
        registration=registration, registration_sha256=audit.hashes[str(local(registration_path).relative_to(REPO))],
        evaluation_sha256=audit.hashes[str(evaluation_path.relative_to(REPO))], provenance=provenance,
        input_hashes=dict(audit.hashes), n_windows=200, n_proteins=50, inference_samples=12,
        runtime=dict(numpy_version=np.__version__, torch_version=torch.__version__, arithmetic='CPU float64'),
        all_fixed_CA_checks_passed=True, topology_mismatch_windows=mismatch,
        topology_interpretation='legacy connectivity metrics are historical where PSF edges differ; no canonical threshold replacement',
        limitations=['development pilot; one seed; reused development validation', 'marginal exploratory unadjusted intervals; zero-crossing is not equivalence',
            'oracle future-derived; H controls retain oracle CA, not CA-only retraining or an information bound',
            'permissive training geometry losses do not certify T1, all-atom quality or the full objective',
            'no model/step/weight selection, no new pass or noninferiority threshold'], **results)
    audit.verify()
    with output_path.open('x') as stream:
        json.dump(summary, stream, separators=(',', ':'), allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--registration', default=str(BASE/'summary.protocol.json'))
    parser.add_argument('--out', default=str(BASE/'summary.json'))
    args = parser.parse_args()
    result = register(args.registration) if args.register else run(args.registration,args.out)
    print(json.dumps({'schema_version':result['schema_version'], 'path':args.registration if args.register else args.out}))


if __name__ == '__main__':
    main()
