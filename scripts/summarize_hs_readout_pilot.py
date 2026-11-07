"""Audit the registered four-arm retention pilot and summarize fixed-step results.

CPU only; no model construction, trajectory reading, training, or test access.
Incomplete execution, altered sources/inputs, or partial pairing fail closed.
The saved BioEmu2 JSON is the only BioEmu2 artifact opened. All statistics are
exploratory; a confidence interval spanning zero does not establish equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / 'results/hs_joint_20261101/retention_v1'
ARTIFACTS = REPO / 'artifacts/hs_joint_20261101/retention_v1'
BIOEMU = REPO / 'results/lasp_v19/bioemu2_complete.json'
TRAIN = 'artifacts/mdcath_manifest_full/train.jsonl'
VAL = 'artifacts/mdcath_manifest_full/val.jsonl'
ARMS = ('frozen', 'joint_teacher', 'joint_observed', 'joint_strong_teacher')
SETTINGS = ((False, 1., .1, 0.), (True, 1., .1, 0.), (True, 1., .1, .1), (True, 10., 1., 0.))
ARM_FIELDS = ('train_encoder', 'teacher_weight', 'teacher_delta_weight', 'observed_readout_weight')
CONTRASTS = (('joint_observed', 'joint_teacher'), ('joint_observed', 'joint_strong_teacher'),
             ('joint_teacher', 'frozen'), ('joint_observed', 'frozen'), ('joint_strong_teacher', 'frozen'))
HORIZONS = [1, 4, 16, 64]
TEMPERATURES = [320., 348., 379., 413., 450.]
METRICS = ('rmsf_mae', 'rmsf_corr', 'root_mean_w2', 'variance_w2', 'md_pca_w2',
           'weak_contact_j', 'transient_contacts_j')
HIGHER_BETTER = {'rmsf_corr', 'weak_contact_j', 'transient_contacts_j'}
SCORES = ('coordinate_energy_score', 'physical_distance_energy_score', 'contact_brier')
GEOMETRY = ('ca_bond_mean_angstrom', 'ca_bond_invalid_fraction',
            'ca_nonlocal_clash_fraction', 'ca_radius_gyration_angstrom')
LEGACY_GEOMETRY = ('ca_bond', 'ca_clash_fraction', 'ca_angle_cos_mae')
CONTACT_COUNTS = ('weak_n_true', 'weak_n_pred', 'transient_n_true', 'transient_n_pred')
CONDITIONS = ('true_future_H', 'anchor_H', 'wrong_time_H', 'persistence')
GATE_METRICS = ('endpoint_rmsd', 'endpoint_shape_rmsd', 'path_rmsd', 'displacement_rmse',
    'anchor_rmsd', *METRICS, 'weak_n_true', 'weak_n_pred', 'transient_n_true', 'transient_n_pred',
    'nonlocal_distance_rmse', 'nonlocal_distance_change_rmse', 'nonlocal_contact_error',
    'nonlocal_contact_probability_mae', 'ca_bond', 'ca_clash_fraction', 'true_ca_clash_fraction',
    'ca_angle_cos_mae')
BOOTSTRAPS, SEED = 10000, 20261101
GATE_THRESHOLDS = dict(bootstrap_seed=SEED, bootstrap_reps=BOOTSTRAPS,
                      min_proteins=4, max_oracle_ratio=1/3, bond_limit=.1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    result = Path(value)
    result = (REPO / result).resolve() if not result.is_absolute() else result.resolve()
    require(result.name != 'test.jsonl', 'test manifest is sealed')
    return result


def label(value):
    return str(path(value).relative_to(REPO))


def sha(value):
    digest = hashlib.sha256()
    with path(value).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Inputs:
    """Record every file consumed and check all of them again before writing."""
    def __init__(self):
        self.hashes = {}

    def track(self, value, expected=None):
        file = path(value)
        actual = sha(file)
        require(expected is None or actual == expected, f'SHA256 mismatch: {file}')
        require(file not in self.hashes or self.hashes[file] == actual, f'input changed: {file}')
        self.hashes[file] = actual
        return file

    def read(self, value):
        return json.loads(self.track(value).read_text())

    def verify(self):
        for file, digest in self.hashes.items():
            require(sha(file) == digest, f'input changed during summary: {file}')


def integer(value):
    require(not isinstance(value, bool) and np.isfinite(value) and int(value) == value,
            f'noninteger window identity: {value}')
    return int(value)


def key(row):
    trajectory = row.get('trajectory', row.get('traj'))
    require(isinstance(trajectory, str), 'missing trajectory identity')
    return (row['protein'], trajectory, float(row['temperature_k']),
            integer(row['anchor_frame']), integer(row['horizon']))


def finite(mapping, names, where):
    require(all(name in mapping and np.isfinite(mapping[name]) for name in names),
            f'missing/nonfinite metrics: {where}')


def rows_for(data, where):
    rows = {}
    for row in data['per_window']:
        k = key(row)
        require(k not in rows, f'duplicate window: {where}/{k}')
        rows[k] = row
    require(len(rows) == data['n_windows'] == 200, f'incomplete panel: {where}')
    require(len({k[0] for k in rows}) == data['n_proteins'] == 50, f'protein count: {where}')
    require(Counter(k[4] for k in rows) == {h: 50 for h in HORIZONS}, f'horizon balance: {where}')
    require(Counter(k[2] for k in rows) == {t: 40 for t in TEMPERATURES}, f'temperature balance: {where}')
    for protein in {k[0] for k in rows}:
        selected = [k for k in rows if k[0] == protein]
        require(Counter(k[4] for k in selected) == {h: 1 for h in HORIZONS}
                and len({(k[1], k[2], k[3]) for k in selected}) == 1,
                f'expected one shared anchor/condition with four horizons: {where}/{protein}')
    return rows


def audit_protocol(protocol, inputs):
    require(tuple(protocol['arms']) == ARMS and protocol['optimizer_steps_per_arm'] == 500
            and protocol['test_opened'] is False and protocol['bioemu2_rerun'] is False,
            'unexpected registered experiment')
    require(protocol['contrasts'] == [f'{a} vs {b}' for a, b in CONTRASTS], 'contrasts differ')
    for arm, expected in zip(ARMS, SETTINGS):
        require(tuple(protocol['arms'][arm][k] for k in ARM_FIELDS) == expected, f'arm differs: {arm}')
    for name, expected in dict(n_windows=200, n_proteins=50, n_samples=12, seed=0, horizons=HORIZONS).items():
        require(protocol['evaluation'][name] == expected, f'registered evaluation differs: {name}')
    require(protocol['evaluation']['checkpoint'] == 'latest at exactly500 steps in every arm; no best/checkpoint selection',
            'checkpoint selection differs')
    for file, digest in protocol['input_hashes'].items():
        inputs.track(file, digest)
    require(label(BIOEMU) in protocol['input_hashes'] and TRAIN in protocol['input_hashes']
            and VAL in protocol['input_hashes'], 'missing registered baseline/manifests')
    for file, digest in protocol['source_hashes'].items():
        inputs.track(file, digest)
        inputs.track(ARTIFACTS / 'source_snapshot' / file, digest)
    return protocol


def audit_execution(root, arm, protocol, inputs):
    execution = inputs.read(root / f'{arm}.execution.json')
    require(execution['arm'] == arm and execution.get('completed_utc')
            and execution.get('frozen_inputs_verified') is True, f'{arm}: execution incomplete')
    require([job['kind'] for job in execution['jobs']] == ['train', 'ensemble', 'endpoint', 'readout'],
            f'{arm}: four ordered jobs required')
    latest = ARTIFACTS / arm / 'checkpoint_latest.pt'
    inputs.track(latest, execution['latest_sha256'])
    files = {kind: root / f'{arm}.{suffix}.json' for kind, suffix in
             (('ensemble', 'ensemble'), ('endpoint', 'endpoint'), ('readout', 'h_gate'))}
    training = [*protocol['common_training_arguments'], '--output', label(ARTIFACTS / arm)]
    settings = protocol['arms'][arm]
    for name in ARM_FIELDS[1:]:
        training += ['--' + name.replace('_', '-'), str(settings[name])]
    if settings['train_encoder']:
        training.append('--train-encoder')
    commands = {'train': ['-u', '-m', 'protein_jepa_frame.training.train_lasp_ca_dynamics', *training]}
    for kind, module in [('ensemble', 'eval_lasp_ca_dynamics'), ('endpoint', 'eval_ca_endpoint_scores'),
                         ('readout', 'gate_ca_representation')]:
        commands[kind] = ['-u', '-m', 'scripts.' + module,
            '--component' if kind == 'readout' else '--checkpoint', label(latest),
            '--out', label(files[kind]), '--max-windows', '200']
    for job in execution['jobs']:
        kind = job['kind']
        require(job.get('returncode') == 0 and job.get('completed_utc'), f'{arm}/{kind}: job incomplete or failed')
        require(job['command'][1:] == commands[kind], f'{arm}/{kind}: unregistered command')
        require(path(job['log']) == root / f'{arm}.{kind}.log', f'{arm}/{kind}: log location differs')
        inputs.track(job['log'])
        if kind != 'train':
            inputs.track(files[kind], job['output_sha256'])
    return latest, files, execution['latest_sha256']


def check_cli_arguments(recorded, arguments):
    index = 0
    while index < len(arguments):
        option = arguments[index]
        require(option.startswith('--'), f'invalid registered argument: {option}')
        name = option[2:].replace('-', '_')
        index += 1
        values = []
        while index < len(arguments) and not arguments[index].startswith('--'):
            values.append(arguments[index])
            index += 1
        require(name in recorded, f'checkpoint missing argument: {name}')
        actual = recorded[name]
        if not values:
            require(actual is True, f'registered flag disabled: {name}')
        elif isinstance(actual, list):
            require([str(x) for x in actual] == values, f'registered list differs: {name}')
        else:
            require(len(values) == 1 and type(actual)(values[0]) == actual,
                    f'registered value differs: {name}')


def common_validation(validation, args):
    require(set(validation['by_horizon']) == {str(h) for h in HORIZONS}, 'validation horizons differ')
    by_horizon = {}
    weights = {'latent_es': args['latent_weight'], 'coord_es': 1., 'contact': args['contact_weight'],
               'rmsf': args['rmsf_weight'], 'distance_es': args['distance_weight'], 'clash': args['clash_weight'],
               'slow_mode_scale': args['mode_weight'], 'slow_mode_distribution': args['mode_distribution_weight']}
    for horizon, terms in validation['by_horizon'].items():
        finite(terms, ('loss', *weights), f'validation/{horizon}')
        penalties = {}
        for term, weight in [('teacher', args['teacher_weight']), ('teacher_delta', args['teacher_delta_weight']),
                             ('observed_readout_mse', args['observed_readout_weight'])]:
            active = args['train_encoder'] and weight > 0
            require((term in terms) == active, f'validation retention term presence differs: {term}')
            if active:
                finite(terms, (term,), f'validation/{horizon}')
            penalties[term] = weight * terms.get(term, 0.)
        common = terms['loss'] - sum(penalties.values())
        reconstructed = sum(weight * terms[term] for term, weight in weights.items())
        require(np.isclose(common, reconstructed, atol=2e-6, rtol=2e-6), 'forecast loss decomposition differs')
        by_horizon[horizon] = dict(raw_total=terms['loss'], subtracted_penalties=penalties,
                                  common_forecast_loss=common, raw_terms=terms)
    require(np.isclose(validation['loss'], np.mean([x['raw_total'] for x in by_horizon.values()]),
                       atol=1e-12, rtol=1e-12), 'validation overall is not horizon mean')
    return dict(raw_total=validation['loss'],
                common_forecast_loss=float(np.mean([x['common_forecast_loss'] for x in by_horizon.values()])),
                by_horizon=by_horizon)


def same_tree(left, right):
    import torch
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and np.array_equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same_tree(x, y) for x, y in zip(left, right))
    return left == right


def audit_checkpoint(arm, latest, protocol, initial, inputs):
    import torch
    from protein_jepa_frame.utils.contracts import state_sha256
    checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
    require(checkpoint['checkpoint_kind'] == 'lasp_ca_dynamics' and checkpoint['step'] == 500, f'{arm}: not fixed step500')
    args, component = checkpoint['args'], checkpoint['component_checkpoint']
    require(args['resume'] is False and args['horizons'] == HORIZONS, f'{arm}: resumed or different horizons')
    require(tuple(args[k] for k in ARM_FIELDS) == tuple(protocol['arms'][arm][k] for k in ARM_FIELDS), f'{arm}: retention weights differ')
    check_cli_arguments(args, protocol['common_training_arguments'])
    require(path(args['output']) == ARTIFACTS / arm and path(args['manifest']) == path(TRAIN)
            and path(args['val_manifest']) == path(VAL), f'{arm}: run paths differ')
    require(checkpoint['kernel_config'] == initial['kernel_config'], f'{arm}: physical kernel config changed')
    initial_component = initial['component_checkpoint']
    require(same_tree({k: v for k, v in component.items() if k != 'encoder'},
                      {k: v for k, v in initial_component.items() if k != 'encoder'}),
            f'{arm}: decoder/component configuration changed')
    tensors = dict(kernel=state_sha256(checkpoint['kernel']), encoder=state_sha256(component['encoder']),
                   decoder=state_sha256(component['delta_ca_decoder']))
    require(tensors['decoder'] == protocol['source_tensor_hashes']['decoder'], f'{arm}: decoder updated')
    teacher = checkpoint.get('target_encoder')
    if args['train_encoder']:
        require(teacher is not None and state_sha256(teacher) == protocol['source_tensor_hashes']['encoder'],
                f'{arm}: authoritative teacher changed')
    else:
        require(teacher is None and tensors['encoder'] == protocol['source_tensor_hashes']['encoder'],
                f'{arm}: frozen encoder changed')
    combined = {f'bundle.encoder.{k}': v for k, v in component['encoder'].items()}
    combined.update({f'bundle.decoder.{k}': v for k, v in component['delta_ca_decoder'].items()})
    combined.update({f'kernel.{k}': v for k, v in checkpoint['kernel'].items()})
    require(all(torch.isfinite(v).all() for v in combined.values()), f'{arm}: nonfinite model tensors')
    drift = sum((v.double() - initial_component['encoder'][k].double()).square().sum().item()
                for k, v in component['encoder'].items())**.5
    norm = sum(v.double().square().sum().item() for v in initial_component['encoder'].values())**.5
    run = inputs.read(ARTIFACTS / arm / 'run.json')
    require(run['args'] == args, f'{arm}: run/checkpoint arguments differ')
    split = run['split_audit']
    require(all(split[name] == 0 for name in ('protein_overlap', 'domain_overlap', 'source_file_overlap')),
            f'{arm}: train/validation overlap')
    for role, manifest in (('train', TRAIN), ('val', VAL)):
        require(split[role + '_manifest_sha256'] == protocol['input_hashes'][manifest], f'{arm}: split hash differs')
    lines = [json.loads(line) for line in inputs.track(ARTIFACTS / arm / 'metrics.jsonl').read_text().splitlines() if line]
    require(not any('resumed_from' in row for row in lines), f'{arm}: unexpected resume log')
    final = [row for row in lines if row.get('step') == 500 and 'val' in row]
    beginning = [row for row in lines if row.get('step') == 0 and 'val' in row]
    require(len(final) == len(beginning) == 1 and final[0]['val'] == checkpoint['validation'],
            f'{arm}: final validation/checkpoint mismatch')
    require(any(row.get('step') == 500 and 'train' in row for row in lines), f'{arm}: missing final training record')
    result = dict(checkpoint=label(latest), checkpoint_sha256=inputs.hashes[path(latest)], step=500,
        tensor_hashes=tensors, inference_model_tensor_sha256=state_sha256(combined),
        teacher_tensor_sha256=state_sha256(teacher) if teacher is not None else None,
        encoder_relative_l2_drift=drift / norm, decoder_and_component_configuration_frozen=True,
        args=args, step_ps=checkpoint['kernel_config']['step_ps'],
        initial_validation=common_validation(beginning[0]['val'], args),
        final_validation=common_validation(checkpoint['validation'], args))
    return result


def audit_sources(data, protocol, inputs, where):
    from scripts.eval_ca_endpoint_scores import SOURCES
    from scripts.gate_ca_representation import SOURCE_PATHS
    expected = dict(endpoint=set(SOURCES), readout=set(SOURCE_PATHS), ensemble={
        'scripts/eval_lasp_ca_dynamics.py', 'protein_jepa_frame/training/train_lasp_ca_dynamics.py',
        'protein_jepa_frame/models/stage_b1b/model.py', 'protein_jepa_frame/models/correlated_bath_posterior.py'})
    recorded = {label(file): digest for file, digest in data['source_hashes'].items()}
    require(set(recorded) == expected[where.rsplit('/', 1)[1]], f'{where}: source hash coverage differs')
    for file, digest in recorded.items():
        require(protocol['source_hashes'].get(file) == digest, f'{where}: unregistered source: {file}')
        inputs.track(file, digest)
    return recorded


def audit_provenance(data, protocol, inputs, where):
    provenance = data['provenance']
    require(set(provenance['upstream_checkpoints']) == {'stage_a_ckpt', 'b1a_ckpt'}, f'{where}: incomplete upstream provenance')
    for split in provenance['training_splits']:
        require(all(split[name] == 0 for name in ('protein_overlap', 'domain_overlap', 'source_file_overlap')),
                f'{where}: split overlap')
        require(split['val_manifest_sha256'] == protocol['input_hashes'][VAL], f'{where}: validation hash differs')
        inputs.track(split['manifest'], split['train_manifest_sha256'])
    require(path(TRAIN) in {path(s['manifest']) for s in provenance['training_splits']}, f'{where}: missing train provenance')
    for parent in provenance['upstream_checkpoints'].values():
        inputs.track(parent['path'], parent['current_sha256'])
    return provenance


def audit_eval_protocol(recorded, where):
    expected = dict(context_frames=8, context_stride=1, frame_stride=1, max_residues=96,
        max_anchors_per_trajectory=4, temperatures=TEMPERATURES, seed=0, round_robin_proteins=True,
        round_robin_conditions=True, horizons_frames=HORIZONS, anchor_stride=1, skip_head_frames=0,
        temporal_lags=[], emit_multi_lag=True, max_proteins=None, max_trajectories_per_protein=None,
        max_windows_per_protein=None)
    require(path(recorded['manifest']) == path(VAL), f'{where}: wrong manifest')
    require(all(recorded[k] == v for k, v in expected.items()), f'{where}: standard200 protocol differs')


def audit_ensemble(data, arm):
    require(data['metric_version'] == 'empirical_w2_v2_pca_projection' and data['lasp_sampling'] == 'endpoint'
            and data['reference_kind'] == 'future_path_mixture', f'{arm}: ensemble task/version differs')
    args = data['args']
    expected = dict(n_samples=12, seed=0, max_windows=200, max_residues=96, horizons=HORIZONS,
        temperatures=TEMPERATURES, anchors_per_trajectory=4, round_robin_proteins=True,
        round_robin_conditions=True, prediction_kind='endpoint', s0_mode=None,
        coupling_scale=None, gamma_scale=None, gamma_floor=None)
    require(data['n_samples'] == 12 and path(args['manifest']) == path(VAL)
            and all(args[k] == v for k, v in expected.items()), f'{arm}: ensemble budget/override differs')
    rows = rows_for(data, arm + '/ensemble')
    for k, row in rows.items():
        require(row['prediction_kind'] == 'endpoint' and row['actual_n_samples'] == 12
                and row['observation_steps'] == [k[4]] * 12, f'{arm}: ensemble sample semantics differ')
        for field in ('lasp', 'persist'):
            finite(row[field], METRICS, (arm, k, field))
        finite(row['geometry_vs_anchor'], LEGACY_GEOMETRY, (arm, k))
        finite(row['contact_counts'], CONTACT_COUNTS, (arm, k))
    return rows


def audit_endpoint(data, arm, checkpoint):
    require(data['task_version'] == 'ca_conditional_endpoint_scores_v1'
            and data['dataset_role'] == 'development_validation' and data['prediction_kind'] == 'endpoint'
            and data['reference_kind'] == 'single_matched_future_endpoint'
            and data['future_information_given_to_sampler'] is False and data['checkpoint_step'] == 500
            and data['checkpoint_training_target'] == 'path_mixture', f'{arm}: endpoint task differs')
    require(data['model_tensor_sha256'] == checkpoint['inference_model_tensor_sha256'], f'{arm}: loaded inference tensors differ')
    audit_eval_protocol(data['protocol'], arm + '/endpoint')
    sampling = data['sampling']
    require((sampling['seed'], sampling['n_samples'], sampling['model_history_frames']) == (0, 12, 4),
            f'{arm}: endpoint budget differs')
    for name in ('s0_mode', 'noise_mix', 'noise_kernel', 'collective_mobility'):
        require(sampling[name] == checkpoint['args'][name], f'{arm}: sampling configuration differs: {name}')
    rows = rows_for(data, arm + '/endpoint')
    for k, row in rows.items():
        seed = int(hashlib.sha256(f'{k[0]}|{k[1]}|{k[3]}|{k[4]}'.encode()).hexdigest()[:12], 16) % (2**31 - 1)
        require(row['sampling_seed'] == seed and row['actual_n_samples'] == 12
                and row['observation_steps'] == [k[4]] * 12, f'{arm}: endpoint sampling differs at {k}')
        require(row['target_frame'] == k[3] + k[4] and row['context_frames'] == list(range(k[3]-7, k[3]+1))
                and row['horizon_ps'] == k[4] * checkpoint['step_ps'], f'{arm}: endpoint time differs at {k}')
        require(4 <= row['valid_residues'] <= 96 and row['nonlocal_pairs'] > 0, f'{arm}: endpoint residue/pair budget')
        for field in ('model', 'persistence'):
            finite(row['scores'][field], SCORES, (arm, k, field))
        for field in ('mean_sample_geometry', 'target_geometry', 'persistence_geometry'):
            finite(row[field], GEOMETRY, (arm, k, field))
        require(len(row['sample_geometry']) == 12, f'{arm}: incomplete sample geometry')
        for sample in row['sample_geometry']:
            finite(sample, GEOMETRY, (arm, k, 'sample geometry'))
        for metric in GEOMETRY:
            require(np.isclose(row['mean_sample_geometry'][metric], np.mean([s[metric] for s in row['sample_geometry']]),
                               rtol=1e-12, atol=1e-12), f'{arm}: sample geometry average differs')
    return rows


def audit_gate(data, arm, checkpoint, protocol):
    from scripts.gate_ca_representation import assess_rows
    require(data['purpose'] == 'true_H_readout_gate_only_not_forecast_evidence'
            and data['dataset_role'] == 'development_validation' and data['schema_version'] == 1
            and data['checkpoint_step'] == 500 and data['checkpoint_kind'] == 'lasp_ca_dynamics', f'{arm}: gate role differs')
    require(data['encoder_tensor_sha256'] == checkpoint['tensor_hashes']['encoder']
            and data['decoder_tensor_sha256'] == checkpoint['tensor_hashes']['decoder'], f'{arm}: gate tensors differ')
    for name, expected in GATE_THRESHOLDS.items():
        require(data['args'][name] == expected, f'{arm}: gate threshold differs: {name}')
    for role, manifest in (('train', TRAIN), ('val', VAL)):
        require(data[role + '_manifest_sha256'] == protocol['input_hashes'][manifest], f'{arm}: gate manifest differs')
    audit_eval_protocol(data['protocol'], arm + '/gate')
    rows = rows_for(data, arm + '/gate')
    for k, row in rows.items():
        require(row['target_frame'] == k[3] + k[4] and row['path_frames'] == k[4] + 1
                and row['context_frames'] == list(range(k[3]-7, k[3]+1))
                and row['horizon_ps'] == k[4] * checkpoint['step_ps'], f'{arm}: gate time differs at {k}')
        require(set(row['conditions']) == set(CONDITIONS), f'{arm}: gate conditions differ')
        finite(row, ('latent_delta_rms', 'valid_residues'), (arm, k))
        for condition in CONDITIONS:
            require(set(row['conditions'][condition]) == set(GATE_METRICS), f'{arm}: gate metric schema differs')
            finite(row['conditions'][condition], GATE_METRICS, (arm, k, condition))
    assessment = assess_rows(data['per_window'], seed=SEED, bootstrap_reps=BOOTSTRAPS,
                             min_proteins=4, max_oracle_ratio=1/3, bond_limit=.1)
    require(all(data[k] == assessment[k] for k in ('passed', 'by_horizon', 'per_protein')),
            f'{arm}: recorded gate assessment does not reproduce')
    return rows, assessment


def protein_values(rows, keys, getter):
    grouped = defaultdict(list)
    for k in keys:
        grouped[k[0]].append(getter(rows[k]))
    result = np.array([np.mean(grouped[protein]) for protein in sorted(grouped)], dtype=np.float64)
    require(len(result) > 0 and np.isfinite(result).all(), 'empty/nonfinite protein mean')
    return result


@lru_cache(maxsize=8)
def bootstrap_indices(count):
    return np.random.default_rng(SEED).integers(count, size=(BOOTSTRAPS, count))


def paired(candidate, reference, keys, candidate_getter, reference_getter, direction=None):
    first = protein_values(candidate, keys, candidate_getter)
    second = protein_values(reference, keys, reference_getter)
    difference = first - second
    ci = np.quantile(difference[bootstrap_indices(len(first))].mean(1), [.025, .975])
    result = dict(candidate=float(first.mean()), reference=float(second.mean()),
        candidate_minus_reference=float(difference.mean()), ci95=ci.tolist(),
        n_windows=len(keys), n_proteins=len(first))
    if direction is not None:
        result.update(improvement=float(direction * difference.mean()),
                      improvement_ci95=sorted((direction * ci).tolist()))
    return result


def comparison(candidate, reference, groups, metrics, getters, directions=None):
    return {group: {metric: paired(candidate, reference, keys,
        lambda row, m=metric: getters[0](row, m), lambda row, m=metric: getters[1](row, m),
        None if directions is None else directions[metric]) for metric in metrics}
        for group, keys in groups.items()}


def means(rows, groups, metrics, getter):
    return {group: {metric: float(protein_values(rows, keys, lambda row, m=metric: getter(row, m)).mean())
                   for metric in metrics} for group, keys in groups.items()}


def descriptive_ensemble_blocks(ensemble, groups):
    """Retain legacy geometry/count estimands separately from canonical geometry."""
    definitions = {
        'legacy_geometry_vs_anchor': dict(field='geometry_vs_anchor', metrics=LEGACY_GEOMETRY,
            reference_kind='observed_anchor',
            metric_semantics=dict(
                ca_bond='Mean absolute adjacent CA distance change from observed anchor, angstrom; '
                    'same chain and residue-index separation 1.',
                ca_clash_fraction='Fraction of nonlocal CA pairs with distance<3.0 angstrom; '
                    'nonlocal means different chains or residue-index separation>=3.',
                ca_angle_cos_mae='Mean absolute change from observed anchor in consecutive-edge cosine; '
                    'valid consecutive same-chain residue triples, dimensionless.'),
            interpretation='Descriptive legacy forecast geometry. Anchor angle/bond preservation is not '
                'future-angle accuracy. The 3.0-angstrom clash definition is separate from canonical '
                'CA geometry at 2.5 angstrom; no canonical threshold or pass rule is applied.'),
        'ensemble_contact_counts': dict(field='contact_counts', metrics=CONTACT_COUNTS,
            reference_kind='future_path_mixture_including_anchor',
            metric_semantics=dict(
                weak='Number of upper-triangle residue pairs with contact probability in [0.05,0.5).',
                transient='Number of upper-triangle residue pairs with contact probability in (0.01,0.99).',
                contact='CA distance<8.0 angstrom; all upper-triangle pairs, without a nonlocal exclusion.',
                n_true='Set size from the true future-path mixture including anchor.',
                n_pred='Set size from the 12 forecast endpoint draws.'),
            interpretation='Descriptive set sizes explaining the contact Jaccards. Equal-protein means '
                'of per-window counts, not pooled counts or fractions. Larger or smaller is not inherently '
                'better; reference counts are identical across arms.')}
    blocks = {}
    for name, definition in definitions.items():
        field, metrics = definition['field'], definition['metrics']
        getter = lambda row, metric, f=field: row[f][metric]
        blocks[name] = dict(**definition,
            arm_means={arm: means(ensemble[arm], groups, metrics, getter) for arm in ARMS},
            comparisons={f'{a}_vs_{b}': comparison(ensemble[a], ensemble[b], groups, metrics,
                (getter, getter)) for a, b in CONTRASTS})
    return blocks


def build_summary(root, output):
    import torch
    from protein_jepa_frame.utils.contracts import state_sha256
    torch.set_num_threads(2)
    root, output = path(root), path(output)
    require(not output.exists(), f'refusing to overwrite {output}')
    require(root == DEFAULT_ROOT and output.parent == root, 'summary must stay in the registered experiment directory')
    inputs = Inputs()
    inputs.track(__file__)
    # Check all completion records before opening any still-changing checkpoints.
    protocol = inputs.read(root / 'protocol.json')
    executions = {arm: audit_execution(root, arm, protocol, inputs) for arm in ARMS}
    audit_protocol(protocol, inputs)
    initial = torch.load(path(protocol['initial_checkpoint']), map_location='cpu', weights_only=False)
    require(label(protocol['initial_checkpoint']) in protocol['input_hashes']
            and initial['checkpoint_kind'] == 'lasp_ca_dynamics' and initial['step'] == 0
            and initial['probe_provenance']['source_training_step'] == protocol['source_training_step'],
            'initial checkpoint lineage differs')
    initial_tensors = dict(kernel=state_sha256(initial['kernel']),
        encoder=state_sha256(initial['component_checkpoint']['encoder']),
        decoder=state_sha256(initial['component_checkpoint']['delta_ca_decoder']))
    require(initial_tensors == protocol['source_tensor_hashes'], 'initial tensors differ from registration')
    checkpoints = {arm: audit_checkpoint(arm, executions[arm][0], protocol, initial, inputs) for arm in ARMS}
    normalized = [{k: v for k, v in checkpoints[a]['args'].items() if k not in (*ARM_FIELDS, 'output')} for a in ARMS]
    require(all(x == normalized[0] for x in normalized), 'training configurations differ beyond registered interventions')
    for arm in ARMS:
        for horizon in map(str, HORIZONS):
            require(np.isclose(checkpoints[arm]['initial_validation']['by_horizon'][horizon]['common_forecast_loss'],
                               checkpoints['frozen']['initial_validation']['by_horizon'][horizon]['common_forecast_loss'],
                               atol=2e-6, rtol=2e-6), f'{arm}: initial common forecast loss differs')
    documents, ensemble, endpoint, gates, assessments, source_lists = {}, {}, {}, {}, {}, defaultdict(list)
    for arm in ARMS:
        documents[arm] = {kind: inputs.read(file) for kind, file in executions[arm][1].items()}
        for kind, data in documents[arm].items():
            require(path(data['checkpoint']) == executions[arm][0] and data['checkpoint_sha256'] == executions[arm][2]
                    and data['optimizer_steps'] == 0 and data['test_opened'] is False, f'{arm}/{kind}: evaluation identity/role differs')
            source_lists[kind].append(audit_sources(data, protocol, inputs, arm + '/' + kind))
            if kind != 'ensemble':
                audit_provenance(data, protocol, inputs, arm + '/' + kind)
        ensemble[arm] = audit_ensemble(documents[arm]['ensemble'], arm)
        endpoint[arm] = audit_endpoint(documents[arm]['endpoint'], arm, checkpoints[arm])
        gates[arm], assessments[arm] = audit_gate(documents[arm]['readout'], arm, checkpoints[arm], protocol)
    for kind, maps in source_lists.items():
        require(all(m == maps[0] for m in maps), f'{kind}: source lists differ across arms')
    bio = inputs.read(BIOEMU)
    require(bio['baseline'] == 'bioemu2' and bio['metric_version'] == 'empirical_w2_v2_pca_projection'
            and bio['n_samples'] == 12 and path(bio['manifest']) == path(VAL), 'saved BioEmu2 task differs')
    for name, expected in dict(seed=0, n_samples=12, max_residues=96, history_frames=8,
        max_anchors_per_trajectory=4, cutoff=8., pca_k=3, weak_lo=.05, weak_hi=.5, trans_lo=.01,
        trans_hi=.99, horizons_ns=[1., 4., 16., 64.], manifest_order=False).items():
        require(bio['evaluation_args'][name] == expected, f'BioEmu2 reference adapter differs: {name}')
    baseline = rows_for(bio, 'saved BioEmu2')
    keys = set(baseline)
    for arm in ARMS:
        require(ensemble[arm].keys() == endpoint[arm].keys() == gates[arm].keys() == keys,
                f'{arm}: complete twelve-file/BioEmu2 pairing required; no intersection fallback')
        require(documents[arm]['endpoint']['metric_semantics'] == documents['frozen']['endpoint']['metric_semantics']
                and documents[arm]['readout']['metric_semantics'] == documents['frozen']['readout']['metric_semantics'],
                f'{arm}: metric semantics differ')
    persistence_max_abs = dict.fromkeys(METRICS, 0.)
    for k in sorted(keys):
        finite(baseline[k]['bioemu2'], METRICS, ('BioEmu2', k))
        finite(baseline[k]['persist'], METRICS, ('BioEmu2 persistence', k))
        base_e, base_p, base_g = ensemble['frozen'][k], endpoint['frozen'][k], gates['frozen'][k]
        for arm in ARMS:
            e, p, g = ensemble[arm][k], endpoint[arm][k], gates[arm][k]
            require(e['persist'] == base_e['persist'] and p['scores']['persistence'] == base_p['scores']['persistence']
                    and g['conditions']['persistence'] == base_g['conditions']['persistence'], f'{arm}: persistence differs at {k}')
            for field in ('target_frame', 'context_frames', 'horizon_ps', 'valid_residues'):
                require(p[field] == base_p[field] == g[field], f'{arm}: endpoint/gate reference metadata differs: {field}')
            for field in ('nonlocal_pairs', 'sampling_seed', 'observation_steps', 'target_geometry', 'persistence_geometry'):
                require(p[field] == base_p[field], f'{arm}: endpoint reference metadata differs: {field}')
            for metric in ('weak_n_true', 'transient_n_true'):
                require(e['contact_counts'][metric] == base_e['contact_counts'][metric], f'{arm}: reference contacts differ')
        for metric in METRICS:
            old, current = baseline[k]['persist'][metric], base_e['persist'][metric]
            require(np.isclose(old, current, atol=2e-6, rtol=1e-6), f'BioEmu2 reference persistence differs: {k}/{metric}')
            persistence_max_abs[metric] = max(persistence_max_abs[metric], abs(old-current))
    groups = dict(overall=sorted(keys))
    groups.update({f'h{h}': sorted(k for k in keys if k[4] == h) for h in HORIZONS})
    groups.update({f'T{int(t)}': sorted(k for k in keys if k[2] == t) for t in TEMPERATURES})
    direction = {m: 1 if m in HIGHER_BETTER else -1 for m in METRICS}
    def comparisons(rows, metrics, getter, directions=None):
        return {f'{a}_vs_{b}': comparison(rows[a], rows[b], groups, metrics, (getter, getter), directions) for a, b in CONTRASTS}
    seven = dict(metric_version=bio['metric_version'], reference_kind='future_path_mixture_including_anchor',
        comparisons=comparisons(ensemble, METRICS, lambda r, m: r['lasp'][m], direction),
        arm_means={arm: means(ensemble[arm], groups, METRICS, lambda r, m: r['lasp'][m]) for arm in ARMS},
        bioemu2_comparisons={arm: comparison(ensemble[arm], baseline, groups, METRICS,
            (lambda r, m: r['lasp'][m], lambda r, m: r['bioemu2'][m]), direction) for arm in ARMS},
        bioemu2_means=means(baseline, groups, METRICS, lambda r, m: r['bioemu2'][m]),
        persistence_means=means(ensemble['frozen'], groups, METRICS, lambda r, m: r['persist'][m]),
        empty_reference_windows={group: {metric: sum(ensemble['frozen'][k]['contact_counts'][metric] == 0 for k in selected)
            for metric in ('weak_n_true', 'transient_n_true')} for group, selected in groups.items()})
    endpoint_summary = dict(task_version=documents['frozen']['endpoint']['task_version'],
        metric_semantics=documents['frozen']['endpoint']['metric_semantics'],
        comparisons=comparisons(endpoint, SCORES, lambda r, m: r['scores']['model'][m], {m: -1 for m in SCORES}),
        arm_means={arm: means(endpoint[arm], groups, SCORES, lambda r, m: r['scores']['model'][m]) for arm in ARMS},
        persistence_means=means(endpoint['frozen'], groups, SCORES, lambda r, m: r['scores']['persistence'][m]),
        persistence_comparisons={arm: comparison(endpoint[arm], endpoint[arm], groups, SCORES,
            (lambda r, m: r['scores']['model'][m], lambda r, m: r['scores']['persistence'][m]), {m: -1 for m in SCORES}) for arm in ARMS})
    geometry = dict(semantics='Per-conformation canonical CA geometry: bond interval [3.3,4.3] angstrom; '
        'nonlocal clash <2.5 angstrom. Fractions are equal-protein means, not pooled counts. Bond mean/Rg have no fixed better direction.',
        comparisons=comparisons(endpoint, GEOMETRY, lambda r, m: r['mean_sample_geometry'][m]),
        arm_means={arm: means(endpoint[arm], groups, GEOMETRY, lambda r, m: r['mean_sample_geometry'][m]) for arm in ARMS},
        target_means=means(endpoint['frozen'], groups, GEOMETRY, lambda r, m: r['target_geometry'][m]),
        persistence_means=means(endpoint['frozen'], groups, GEOMETRY, lambda r, m: r['persistence_geometry'][m]))
    readout = dict(purpose='true future H oracle; no forecast or memory claim', thresholds=GATE_THRESHOLDS,
        metric_semantics=documents['frozen']['readout']['metric_semantics'],
        original_gate_assessments=assessments,
        gate_bootstrap_note='Original gate reproduced unchanged: SeedSequence([20261101,horizon]); all conditions, paired controls, ratio and checks retained.',
        true_future_H_comparisons=comparisons(gates, GATE_METRICS, lambda r, m: r['conditions']['true_future_H'][m]),
        condition_means={arm: {condition: means(gates[arm], groups, GATE_METRICS,
            lambda r, m, c=condition: r['conditions'][c][m]) for condition in CONDITIONS} for arm in ARMS})
    readout['oracle_persistence_ratios'] = {arm: {group:
        readout['condition_means'][arm]['true_future_H'][group]['endpoint_rmsd'] /
        readout['condition_means'][arm]['persistence'][group]['endpoint_rmsd']
        for group in groups} for arm in ARMS}
    readout['ratio_semantics'] = ('Ratio of equal-protein mean endpoint RMSDs, not mean per-window ratios. '
        'Original pass checks apply only per horizon; overall/temperature ratios are descriptive.')
    validation_summary = dict(definition='raw_total - teacher_weight*teacher - teacher_delta_weight*teacher_delta '
        '- observed_readout_weight*observed_readout_mse; frozen arm has no retention terms',
        panel='Original16 validation clips per horizon from training; not the standard200 panel. '
              'Logs do not identify protein clusters, so no bootstrap CI is assigned here.',
        arms={arm: checkpoints[arm]['final_validation'] for arm in ARMS},
        common_forecast_differences={f'{a}_vs_{b}': dict(
            overall=checkpoints[a]['final_validation']['common_forecast_loss'] -
                    checkpoints[b]['final_validation']['common_forecast_loss'],
            by_horizon={str(h): checkpoints[a]['final_validation']['by_horizon'][str(h)]['common_forecast_loss'] -
                               checkpoints[b]['final_validation']['by_horizon'][str(h)]['common_forecast_loss']
                        for h in HORIZONS}) for a, b in CONTRASTS})
    report = dict(schema_version=1, task='registered_hs_readout_retention_fixed500_summary',
        role=protocol['role'], test_opened=False, bioemu2_rerun=False, checkpoint_selection='latest at exactly500 steps',
        n_windows=200, n_proteins=50, n_samples=12, bootstrap_reps=BOOTSTRAPS, bootstrap_seed=SEED,
        registered_contrasts=protocol['contrasts'], group_sizes={name: dict(n_windows=len(selected),
            n_proteins=len({k[0] for k in selected})) for name, selected in groups.items()},
        estimand='equal windows within protein, then equal proteins within each group; all paired differences use the same keys/protein ordering',
        comparison_direction='candidate minus reference; direction-adjusted improvement positive is better when provided',
        confidence='Exploratory marginal 95% paired protein-bootstrap intervals; no multiplicity, training-seed or adaptive-selection correction. CI spanning0 is not equivalence.',
        scientific_limits=protocol['scientific_limits'] + [
            'No new pass/noninferiority rule; original H gate thresholds alone are reproduced.',
            'Temperature strata each contain10 different proteins; contrasts across temperatures are not within-protein temperature responses.',
            'Proper endpoint scores with one true endpoint target an expectation across windows, not a recovered conditional ensemble at each window.',
            'M=12 is shared, not a correction for model-dependent finite-M distance-V/Brier effects.',
            'Common-forecast validation loss uses the original16-window training-validation schedule, not the standard200 scoring panel; retained for descriptive comparison only.',
            'Only the saved BioEmu2 JSON is opened; its original evaluation_args/sources are retained as metadata, not rerun or silently re-adapted.'],
        protocol=label(root / 'protocol.json'), checkpoint_audit=checkpoints,
        pairing_audit=dict(complete_twelve_file_and_bioemu2_pairing=True, unique_keys=200,
            frozen_source_snapshots_verified=True, all_execution_jobs_exit0=True,
            checkpoint_tensor_and_decoder_teacher_constraints_verified=True,
            bioemu2_persistence_max_abs_difference=persistence_max_abs,
            ensemble_seed_evidence='reconstructed from recorded seed and hash-verified evaluator; per-window RNG/draws not serialized'),
        bioemu2_adapter=dict(path=label(BIOEMU), sha256=inputs.hashes[BIOEMU],
            evaluation_args=bio['evaluation_args'], original_sources_metadata_only=bio['sources']),
        validation_loss=validation_summary, seven_metrics=seven, endpoint_scores=endpoint_summary,
        canonical_ca_geometry=geometry, readout_gate=readout,
        **descriptive_ensemble_blocks(ensemble, groups))
    inputs.verify()
    report['input_sha256'] = {label(file): digest for file, digest in sorted(inputs.hashes.items())}
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    with output.open('x') as stream:
        stream.write(serialized)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--out', default=str(DEFAULT_ROOT / 'summary.json'))
    args = parser.parse_args()
    report = build_summary(args.root, args.out)
    print(json.dumps(dict(output=args.out, n_windows=report['n_windows'],
                          n_proteins=report['n_proteins'], contrasts=report['registered_contrasts']), indent=2))


if __name__ == '__main__':
    main()
