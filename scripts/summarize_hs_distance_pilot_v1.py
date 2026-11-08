"""Provenance-first CPU summary of the matched physical-distance target training pilot.

Register this wrapper separately before opening either new arm's score files.
The statistical estimands were registered independently before training. No
model construction, trajectory coordinate reads, sampling or optimization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

from scripts import hs_distance_pilot_statistics_v1 as numeric
from scripts import register_hs_distance_pilot_statistics_v1 as registration
from scripts import summarize_hs_ordered_pilot_v1 as legacy
from scripts import summarize_hs_target_pilot as old
from scripts import summarize_ca_ordered_paths as paths
from run_scripts import run_hs_distance_pilot_v1 as runner

BASE, ARTIFACTS = runner.RESULTS, runner.ARTIFACTS
IMPLEMENTATION = Path('results/hs_joint_20261101/distance_path_summary_v1')
SCRIPT = 'scripts/summarize_hs_distance_pilot_v1.py'
TEST = 'tests/test_hs_distance_summary_v1.py'
VERSION = 'hs_distance_path_pilot_summary_v1'
ARMS, JOBS = numeric.ARMS, runner.JOBS
SLOTS = dict(zip(ARMS, old.ARMS))
require = old.require
SOURCES = tuple(sorted(set((*legacy.SOURCES, *registration.SOURCES, SCRIPT, TEST,
    'run_scripts/run_hs_distance_pilot_v1.py',
    'protein_jepa_frame/training/train_lasp_ca_distance_v1.py',
    'protein_jepa_frame/training/physical_distance_augmented_objective_v1.py',
    'protein_jepa_frame/training/physical_distance_path_objective_v1.py', str(runner.DOCUMENT)))))


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path = old.local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with path.open('x') as stream:
        stream.write(encoded)


def slot(values):
    require(set(values) == set(ARMS), 'unexpected arm mapping')
    return {SLOTS[a]: values[a] for a in ARMS}


def contract():
    return dict(schema_version=VERSION, arms=list(ARMS),
        n_successful_logical_jobs=12, jobs_per_arm=list(JOBS),
        n_paired_intervals=402, standard_descriptive_means=670, suite_descriptive_means=1700,
        statistics_protocol=str(BASE/'summary.protocol.json'), output=str(BASE/'summary.json'),
        ordering='Both complete exit0 executions are checked before loading checkpoints or parsing score JSON.',
        data='All 500 actual training-window metadata and batch fingerprints must agree byte for byte across the two new arms.',
        audit='Reload checkpoint tensors on CPU; check initial identity, complete frozen component, optimizer support and 500 steps, final global/sampling RNG.',
        initial_h1='All scalar terms, selected totals and gradient norms agree numerically at fixed tolerance 2e-6+5e-4*abs(reference); retain signed errors. No bitwise-gradient assertion.',
        roles='increment minus absolute; both use marginal_path coordinate objective plus fixed weight-one auxiliary and plugin contact. Legacy positional slots are internal only.',
        suite='All 7200/full and 800/matched raw rows audited; overlapping full/matched rows must match exactly.',
        limits='Single-seed adaptive development pilot. Historical BioEmu2 T2A reference fingerprints are missing; no new paired T2A superiority claim. Frozen long-horizon H gate remains unchanged.',
        registration_timing='Scientific estimands must precede the experiment registration and training. This implementation may be registered after training begins, before it parses new score JSON.',
        test_opened=False, bioemu2_rerun=False, model_sampling=False)


def register():
    """Freeze implementation using protocols/sources only; never parse scores."""
    output = old.local(IMPLEMENTATION/'protocol.json')
    require(not output.exists() and not old.local(BASE/'summary.json').exists(), 'refusing registration overwrite')
    inputs = old.Inputs()
    experiment = inputs.read(BASE/'protocol.json')
    statistics = inputs.read(BASE/'summary.protocol.json')
    require(all(statistics.get(k) == v for k,v in registration.contract().items()), 'statistics contract changed')
    statistics_sha = inputs.hashes[old.local(BASE/'summary.protocol.json')]
    require(experiment['statistics_sha256'] == statistics_sha, 'experiment statistics digest differs')
    require(datetime.fromisoformat(statistics['registered_utc']) <= datetime.fromisoformat(experiment['registered_utc']),
            'statistics were registered after experiment protocol')
    for source,digest in statistics['source_hashes'].items():
        inputs.track(source,digest)
    for source in SOURCES:
        inputs.track(source)
    # Existence metadata is allowed even when training/evaluation already runs.
    # The registration makes no claim that outputs did not previously exist.
    existence = {str(BASE/f'{arm}.{kind}.json'):old.local(BASE/f'{arm}.{kind}.json').exists()
                 for arm in ARMS for kind in JOBS[1:]}
    snapshots = {}
    for source in SOURCES:
        target = old.local(IMPLEMENTATION/'source_snapshot'/source)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(old.local(source).read_bytes())
        digest = inputs.hashes[old.local(source)]
        inputs.track(target,digest)
        snapshots[old.label(target)] = digest
    inputs.verify()
    result = dict(contract(), registered_utc=now(),
        registration_stage='This wrapper has not parsed new-arm performance scores. The scientific statistics were separately frozen before training; this implementation registration may occur during or after the experiment.',
        score_file_existence=existence, score_json_parsed_during_registration=False,
        experiment_protocol_sha256=inputs.hashes[old.local(BASE/'protocol.json')], statistics_sha256=statistics_sha,
        source_hashes={source:inputs.hashes[old.local(source)] for source in SOURCES},
        snapshot_hashes=snapshots,
        input_hashes={old.label(p):digest for p,digest in inputs.hashes.items()
            if old.label(p) not in SOURCES and old.label(p) not in snapshots})
    write_new(output,result)
    return result


def execution_headers(executions, protocol):
    """No score/file inspection: reject any unfinished arm before further work."""
    require(set(executions) == set(ARMS), 'missing arm execution')
    frozen = {**protocol['input_hashes'], **protocol['source_hashes']}
    for arm in ARMS:
        record = executions[arm]
        require(record.get('arm') == arm and record.get('status') == 'completed'
            and record.get('completed_utc') and record.get('successful_logical_jobs') == len(JOBS),
            'both arms must complete all six jobs')
        require(type(record.get('runner_pid')) is int and record['runner_pid'] > 0, 'missing actual runner PID')
        require([j['kind'] for j in record['jobs']] == list(JOBS), 'missing/reordered logical jobs')
        for job in record['jobs']:
            require(type(job.get('returncode')) is int and job['returncode'] == 0
                and type(job.get('pid')) is int and job['pid'] > 0
                and job.get('started_utc') and job.get('completed_utc'), 'missing actual exit0 process')
            require(job['command'] == protocol['per_arm_commands'][arm][job['kind']], 'job command changed')
            require(job['frozen_before'] == job['frozen_after'] == frozen, 'job frozen hashes incomplete')


def audit_executions(executions, protocol, inputs):
    execution_headers(executions, protocol)
    for arm in ARMS:
        record = executions[arm]
        inputs.track(ARTIFACTS/arm/'checkpoint_latest.pt', record['latest_sha256'])
        for job in record['jobs']:
            kind = job['kind']
            require(old.local(job['log']) == old.local(BASE/f'{arm}.{kind}.log'), 'log path differs')
            inputs.track(job['log'], job['log_sha256'])
            if kind != 'train':
                output = BASE/f'{arm}.{kind}.json'
                inputs.track(output, job['output_sha256'])
                sidecar = output.with_suffix('.component.pt')
                require(old.local(sidecar).exists() == ('component_sha256' in job), 'sidecar identity missing')
                if 'component_sha256' in job:
                    inputs.track(sidecar, job['component_sha256'])
                if kind == 'ordered':
                    inputs.track(output.with_suffix('.protocol.json'), job['ordered_protocol_sha256'])


def audit_protocol(protocol, implementation, statistics, inputs):
    require(all(implementation.get(k) == v for k, v in contract().items()), 'wrapper registration changed')
    require(all(statistics.get(k) == v for k, v in registration.contract().items()), 'statistics registration changed')
    digest = inputs.hashes[old.local(BASE/'summary.protocol.json')]
    require(implementation['statistics_sha256'] == protocol['statistics_sha256'] == digest
        and inputs.hashes[old.local(BASE/'protocol.json')] == implementation['experiment_protocol_sha256'],
            'statistical registration hash differs')
    require(datetime.fromisoformat(statistics['registered_utc']) <= datetime.fromisoformat(protocol['registered_utc']),
            'statistics were registered after training protocol')
    for mapping in (statistics['source_hashes'], implementation['source_hashes']):
        for source, expected in mapping.items():
            inputs.track(source, expected)
    require(set(SOURCES) == set(implementation['source_hashes']), 'wrapper source coverage incomplete')
    expected_snapshots = {str(IMPLEMENTATION/'source_snapshot'/source):digest
                          for source,digest in implementation['source_hashes'].items()}
    require(implementation['snapshot_hashes'] == expected_snapshots, 'wrapper snapshot coverage differs')
    require(implementation['score_json_parsed_during_registration'] is False, 'wrapper read scores before registration')
    for path,digest in implementation['snapshot_hashes'].items():
        inputs.track(path,digest)
    for path,digest in implementation['input_hashes'].items():
        inputs.track(path,digest)
    require(protocol['schema_version'] == 'hs_distance_path_pilot_v1' and tuple(protocol['arms']) == ARMS
        and tuple(protocol['jobs']) == JOBS and protocol['test_opened'] is False
        and protocol['bioemu2_rerun'] is False, 'experiment scope differs')
    require(protocol['per_arm_commands'] == {a: runner.commands(a) for a in ARMS}, 'registered commands differ')
    require(protocol['checkpoint_selection'] == 'both latest500 only; no validation/best/early stopping', 'selection changed')
    require(old.local(protocol['statistics_protocol']) == old.local(BASE/'summary.protocol.json'), 'statistics path differs')
    require(old.local(protocol['initial_checkpoint']) == old.local(runner.INITIAL)
        and protocol['initial_checkpoint_sha256'] == runner.INITIAL_SHA, 'warm start changed')
    require(protocol['intervention']['distance_path_weight'] == 1.
        and protocol['intervention']['contact_estimator'] == 'plugin', 'registered auxiliary weight/contact changed')
    split = protocol['split_audit']
    require(all(split[k] == 0 for k in ('protein_overlap','domain_overlap','source_file_overlap')), 'split overlap')
    for role,path in (('train',old.shared.TRAIN),('val',old.shared.VAL)):
        require(split[role+'_manifest_sha256'] == contained_digest(protocol['input_hashes'],path),
                'registered split manifest identity differs')
    require(protocol['evaluation'] == dict(ensemble_windows=200, endpoint_windows=200, ordered_windows=200,
        proteins=50, samples=12, seed=0, horizons=[1, 4, 16, 64],
        suite=dict(windows=7200, proteins=12, trajectories=300, modes=['correct', 'wrong', 'none']),
        matched8=dict(windows=800, proteins=8, trajectories=200, modes=['correct'])), 'evaluation budget differs')
    expected_rows = protocol['training_windows']
    require(len(expected_rows) == 500 and [r['step'] for r in expected_rows] == list(range(1, 501))
        and [r['horizon'] for r in expected_rows] == [old.HORIZONS[i % 4] for i in range(500)], 'training schedule differs')
    for file, digest in protocol['input_hashes'].items():
        inputs.track(file, digest)
    for file, digest in protocol['source_hashes'].items():
        inputs.track(file, digest)
        inputs.track(Path(protocol['source_snapshot'])/file, digest)
    runner.verify(protocol, BASE/'protocol.json', inputs.hashes[old.local(BASE/'protocol.json')])


# Explicit frozen helpers; their module globals are never changed.
inference_hash = legacy.inference_hash
optimizer_pair = legacy.optimizer_pair


def audit_selected_loss(terms, arm):
    """Verify both additive totals and the exact serialized selected alias."""
    require(arm in ARMS, 'invalid distance objective')
    for key in ('loss','loss_base','loss_absolute','loss_increment',
                'distance_absolute_es','distance_increment_es'):
        old.number(terms[key])
    errors = {}
    for objective in ARMS:
        base, auxiliary = terms['loss_base'], terms['distance_'+objective+'_es']
        expected = base + auxiliary
        # Serialized float32 additions need not equal a float64 reconstruction.
        tolerance = 8*2**-23*max(2**-126,abs(expected),abs(base),abs(auxiliary))
        error = terms['loss_'+objective] - expected
        require(abs(error) <= tolerance, 'distance total is not base plus weight-one auxiliary')
        errors[objective] = dict(signed_error=error, tolerance=tolerance)
    require(terms['loss'] == terms['loss_'+arm], 'selected loss does not equal its registered alias')
    return errors


def h1_difference(candidate, reference):
    """The prior kernel diagnostic's fixed tolerance; no fitted tolerance."""
    old.number(candidate)
    old.number(reference)
    error = candidate-reference
    tolerance = 2e-6 + 5e-4*abs(reference)
    require(abs(error) <= tolerance, 'first h1 numerical identity differs')
    return dict(candidate=candidate, reference=reference, signed_error=error, tolerance=tolerance)


def audit_initial_h1(first_steps):
    require(set(first_steps) == set(ARMS), 'first-step arms differ')
    candidate,reference = (first_steps[arm] for arm in ARMS)
    require(all(row['step'] == row['horizon'] == 1 for row in first_steps.values()), 'initial h1 identity differs')
    require(candidate['lr'] == reference['lr'], 'first h1 learning rate differs')
    require(set(candidate['train']) == set(reference['train']), 'first h1 scalar terms differ')
    differences = {key:h1_difference(candidate['train'][key],reference['train'][key])
                   for key in sorted(reference['train'])}
    within_arm = {}
    for arm,row in first_steps.items():
        audit_selected_loss(row['train'],arm)
        within_arm[arm] = {kind:h1_difference(row['train'][inc],row['train'][absolute])
            for kind,inc,absolute in [('auxiliary','distance_increment_es','distance_absolute_es'),
                                      ('total','loss_increment','loss_absolute')]}
    gradient = h1_difference(candidate['grad_norm'],reference['grad_norm'])
    return dict(first_h1_numerical_identity=True,
        first_h1_identity_tolerance='abs(candidate-reference) <= 2e-6 + 5e-4*abs(reference)',
        first_h1_scalar_differences=differences, first_h1_within_arm_identity=within_arm,
        first_h1_gradient_norm_difference=gradient,
        first_h1_gradient_norms={arm:first_steps[arm]['grad_norm'] for arm in ARMS},
        first_h1_selected_losses={arm:first_steps[arm]['train']['loss'] for arm in ARMS})


def audit_training(checkpoints, initial, executions, protocol, inputs):
    from protein_jepa_frame.utils.contracts import state_sha256
    require(state_sha256(initial['kernel']) == protocol['initial_kernel_sha256']
        and initial['kernel_config'] == protocol['initial_kernel_config']
        and old.component_identity(initial['component_checkpoint']) == protocol['initial_component_identity'], 'initial checkpoint identity differs')
    checked, identities, raw_windows, first_steps = {}, {}, {}, {}
    for arm in ARMS:
        ck = checkpoints[arm]
        require(ck['args']['distance_path_objective'] == arm and ck['args']['coord_objective'] == 'marginal_path'
                and ck['args']['distance_path_weight'] == 1. and ck['args']['contact_estimator'] == 'plugin',
                'distance/contact/coordinate training estimand differs')
        actual = runner.checkpoint_audit(ck, arm, protocol)
        require(actual == executions[arm]['checkpoint_audit'], 'saved checkpoint audit disagrees')
        run = inputs.read(ARTIFACTS/arm/'run.json')
        started = inputs.read(ARTIFACTS/arm/'started.json')
        completed = inputs.read(ARTIFACTS/arm/'execution.json')
        require(run['args'] == ck['args'] and run['train_only'] is True and run['test_opened'] is False
            and run['no_validation_selection'] is True and run['validation_loaded'] is False,
            'training process/run contract differs')
        require(type(started['pid']) is int and started['pid'] == executions[arm]['jobs'][0]['pid']
            and started['run_kind'] == 'train' and started['distance_path_objective'] == arm,
            'actual trainer PID/estimator differs')
        source = 'protein_jepa_frame/training/train_lasp_ca_distance_v1.py'
        require(old.local(started['command'][0]) == old.local(source)
            and started['command'][1:] == protocol['per_arm_commands'][arm]['train'][4:]
            and started['source_sha256'] == protocol['source_hashes'][source], 'trainer command/source differs')
        require(all(completed.get(k) == value for k,value in started.items())
            and completed['status'] == 'completed' and type(completed['exit_code']) is int
            and completed['exit_code'] == 0 and completed.get('completed_utc')
            and old.local(completed['output']) == old.local(ARTIFACTS/arm/'checkpoint_latest.pt')
            and completed['output_sha256'] == executions[arm]['latest_sha256'], 'trainer completion differs')
        completion = completed['result']
        require(completion['optimizer_steps'] == 500 and completion['frozen_component_unchanged'] is True
            and completion['kernel_parameter_tensors'] == 73
            and completion['active_optimizer_states'] == len(ck['optimizer']['state']), 'trainer optimizer support differs')
        identities[arm] = inputs.read(ARTIFACTS/arm/'initial_identity.json')
        expected_names = list(initial['kernel'])
        require(len(expected_names) == len(set(expected_names)) == 73, 'initial kernel inventory differs')
        elements = sum(value.numel() for value in initial['kernel'].values())
        require(elements == 669429, 'initial kernel element count differs')
        expected_identity = dict(model_sha256=inference_hash(initial), component_sha256=inference_hash(initial, component_only=True),
            kernel_sha256=state_sha256(initial['kernel']), kernel_parameter_names=expected_names,
            kernel_parameter_elements=elements)
        require(identities[arm] == expected_identity, 'actual initialized model/inventory differs from warm start')
        parameter_ids = [identifier for group in ck['optimizer']['param_groups'] for identifier in group['params']]
        require(len(parameter_ids) == len(set(parameter_ids)) == 73
            and set(ck['optimizer']['state']) <= set(parameter_ids), 'optimizer full parameter inventory differs')
        path = inputs.track(ARTIFACTS/arm/'training_windows.jsonl')
        raw_windows[arm] = path.read_bytes()
        audited = runner.audit_training_windows(path, protocol['training_windows'])
        require(audited == executions[arm]['jobs'][0]['training_windows_audit'], 'training-window audit changed')
        for row in [json.loads(line) for line in raw_windows[arm].splitlines()]:
            require(re.fullmatch('[0-9a-f]{64}', row['actual_batch_sha256'])
                and row['file_stat'] == protocol['selected_hdf5_stats'][row['record_path']], 'batch hash/HDF5 stat differs')
        metrics = [json.loads(line) for line in inputs.track(ARTIFACTS/arm/'metrics.jsonl').read_text().splitlines()]
        require([r['step'] for r in metrics] == list(range(1, 501))
            and [r['horizon'] for r in metrics] == [old.HORIZONS[i % 4] for i in range(500)], 'training metrics budget differs')
        require(all('val' not in r and 'resumed_from' not in r for r in metrics), 'validation/resume present')
        for row in metrics:
            require(row['kernel_parameters'] == 73 and type(row['gradient_parameters']) is int
                and 0 < row['gradient_parameters'] <= len(ck['optimizer']['state']), 'metric gradient support differs')
            audit_selected_loss(row['train'],arm)
            for value in [row['grad_norm'], row['lr'], *row['train'].values()]:
                old.number(value)
        first_steps[arm] = {k: metrics[0][k] for k in ('step', 'horizon', 'grad_norm', 'lr', 'train')}
        checked[arm] = dict(checkpoint=str(ARTIFACTS/arm/'checkpoint_latest.pt'), checkpoint_sha256=executions[arm]['latest_sha256'],
            args=ck['args'], step=500, step_ps=ck['kernel_config']['step_ps'], inference_model_tensor_sha256=inference_hash(ck), audit=actual)
    require(raw_windows[ARMS[0]] == raw_windows[ARMS[1]], 'actual 500 training batches differ across arms')
    initial_h1 = audit_initial_h1(first_steps)
    for key in ('sampling_rng', 'rng_state'):
        runner.exact_tree(checkpoints[ARMS[0]][key], checkpoints[ARMS[1]][key], key)
    normalized = [{k: v for k, v in checkpoints[a]['args'].items() if k not in ('distance_path_objective', 'output')} for a in ARMS]
    require(normalized[0] == normalized[1], 'unexpected training argument difference')
    optimizer = optimizer_pair(*(checkpoints[a]['optimizer'] for a in ARMS))
    expected_pair = dict(initial_model_equal=True, actual_training_window_bytes_equal=True,
        final_sampling_and_global_rng_equal=True, only_argument_differences=['output', 'distance_path_objective'],
        compared_with_historical_endpoint_rng=False)
    require(inputs.read(BASE/'paired_training_audit.json') == expected_pair, 'runner paired audit differs')
    return checked, dict(**expected_pair, n_actual_training_windows=500, optimizer=optimizer,
        **initial_h1, kernel_parameter_tensors=73, final_model_and_optimizer_moments_may_differ=True,
        input_identity='Both new arms serialized identical actual loaded-batch fingerprints; this does not retroactively identify old endpoint training coordinates.')


def contained_digest(mapping, path):
    """Resolve absolute/relative aliases inside the repo; reject conflicts."""
    target = old.local(path)
    normalized = {}
    for original,digest in mapping.items():
        actual = old.local(original)
        require(actual not in normalized or normalized[actual] == digest, 'conflicting contained-path hashes')
        normalized[actual] = digest
    require(target in normalized, 'missing contained-path hash')
    return normalized[target]


def audit_ordered(data, arm, checked, protocol, inputs):
    from scripts.eval_ca_ordered_paths import SOURCES as required_sources, panel_spec, summarize as raw_summary
    output = BASE/f'{arm}.ordered.json'
    registered = inputs.read(output.with_suffix('.protocol.json'))
    require(all(data.get(k) == v for k, v in registered.items()), 'ordered output changed registration')
    require(data['task_version'] == 'ca_ordered_paths_development_v1' and data['dataset_role'] == 'development_validation'
        and data['optimizer_steps'] == 0 and data['model_selection'] is False and data['test_opened'] is False
        and data['bioemu2_rerun'] is False and data['future_information_given_to_sampler'] is False
        and data['frozen_inputs_verified'] is True and data.get('completed_utc'), 'ordered task incomplete/different')
    require(data['panel'] == json.loads(json.dumps(panel_spec([1, 4, 16, 64], 200),allow_nan=False)) and data['n_windows'] == 200
        and data['n_proteins'] == 50, 'ordered panel incomplete')
    require(old.local(data['args']['checkpoint']) == old.local(checked['checkpoint'])
        and contained_digest(data['input_hashes'],data['args']['checkpoint']) == checked['checkpoint_sha256']
        and data['model_tensor_sha256'] == checked['inference_model_tensor_sha256'], 'ordered loaded model differs')
    require(data['sampling'] == dict(seed=0, n_paths=12, model_history_frames=4, prediction_kind='endpoint',
        s0_mode='exact_spatial', one_continuous_rollout_per_member=True, no_future_reencoding=True, no_bath_reset=True),
        'ordered sampling contract differs')
    old.shared.audit_eval_protocol(data['protocol'], arm+'/ordered')
    old.shared.audit_provenance(data, protocol, inputs, arm+'/ordered')
    for path, digest in data['input_hashes'].items():
        inputs.track(path, digest)
    require(set(required_sources) <= set(data['source_hashes']), 'ordered required source coverage missing')
    for path, digest in data['source_hashes'].items():
        require(protocol['source_hashes'].get(path) == digest, 'unregistered ordered source')
        inputs.track(path, digest)
        inputs.track(output.with_suffix('.source_snapshot')/path, digest)
    rows = paths.audit_rows(data['per_window'], data['panel'])
    require(raw_summary(rows) == data['by_horizon'], 'ordered raw aggregate differs')
    return old.shared.rows_for(data, arm+'/ordered')


def cross_task_pairing(ensemble, endpoints, ordered, baseline):
    keys = set(baseline)
    require(len(keys) == 200, 'baseline panel incomplete')
    for arm in ARMS:
        require(set(ensemble[arm]) == set(endpoints[arm]) == set(ordered[arm]) == keys,
                'ordered/endpoint/ensemble/BioEmu keys differ')
        for key in keys:
            a, b = ordered[arm][key], endpoints[arm][key]
            for field in ('target_frame', 'context_frames', 'horizon_ps', 'valid_residues', 'sampling_seed'):
                require(a[field] == b[field], 'ordered/endpoint observation contract differs')
    numeric.observed_reference_pairing(ordered)


def suite_pairing(rows):
    for arm in ARMS:
        for key, row in rows[arm]['matched8'].items():
            require(key in rows[arm]['suite'] and row == rows[arm]['suite'][key], 'matched8/full raw row differs')
    for kind in ('suite', 'matched8'):
        require(set(rows[ARMS[0]][kind]) == set(rows[ARMS[1]][kind]), 'suite arm keys differ')
        for key in rows[ARMS[0]][kind]:
            a, b = (rows[arm][kind][key] for arm in ARMS)
            require(a['persist'] == b['persist'] and all(a['position'][m] == b['position'][m]
                for m in ('persist_kabsch_ca_rmsd', 'persist_aligned_ca_rmsd')), 'suite reference differs')


def count_intervals(value):
    if isinstance(value, dict):
        return int('ci95' in value) + sum(count_intervals(v) for v in value.values())
    if isinstance(value, list):
        return sum(map(count_intervals, value))
    return 0


def count_standard_means(value, in_means=False):
    scopes = ('arm_means','bioemu2_means','persistence_means','target_means',
              'geometry_vs_anchor','contact_counts')
    if isinstance(value,dict):
        return sum(count_standard_means(v,in_means or k in scopes) for k,v in value.items())
    if isinstance(value,list):
        return sum(count_standard_means(v,in_means) for v in value)
    return int(in_means and type(value) is float)


def count_suite_means(reports):
    count = 0
    for arm in ARMS:
        for kind in ('suite','matched8'):
            report = reports[arm][kind]
            cells = [*report['overall'].values(), *report['t2a'].values()]
            for group in ('by_temperature','by_horizon'):
                cells.extend(cell for modes in report[group].values() for cell in modes.values())
            count += sum(len(cell['means']) for cell in cells)
    return count


def build_summary():
    import torch
    output = old.local(BASE/'summary.json')
    require(not output.exists(), 'refusing summary overwrite')
    torch.set_num_threads(2)
    inputs = old.Inputs()
    implementation = inputs.read(IMPLEMENTATION/'protocol.json')
    statistics = inputs.read(BASE/'summary.protocol.json')
    protocol = inputs.read(BASE/'protocol.json')
    executions = {a: inputs.read(BASE/f'{a}.execution.json') for a in ARMS}
    # Both headers first: do not parse or load one completed arm while the other runs.
    execution_headers(executions, protocol)
    audit_protocol(protocol, implementation, statistics, inputs)
    audit_executions(executions, protocol, inputs)
    for arm in ARMS:
        require(datetime.fromisoformat(statistics['registered_utc']) <=
                datetime.fromisoformat(protocol['registered_utc']) <=
                datetime.fromisoformat(executions[arm]['jobs'][0]['started_utc']),
                'statistical/experiment registration did not precede training')
    initial = torch.load(inputs.track(runner.INITIAL, runner.INITIAL_SHA), map_location='cpu', weights_only=False)
    checkpoints = {a: torch.load(inputs.track(ARTIFACTS/a/'checkpoint_latest.pt'), map_location='cpu', weights_only=False) for a in ARMS}
    checked, paired_training = audit_training(checkpoints, initial, executions, protocol, inputs)
    gate = old.audit_reused_gate(protocol, initial, inputs)
    documents, ensemble, endpoints, ordered, suite_rows, suite_reports = {}, {}, {}, {}, {}, {}
    records = [json.loads(line) for line in inputs.track(old.shared.VAL).read_text().splitlines()]
    expected_suite = {kind: old.expected_suite_keys(records, kind) for kind in ('suite', 'matched8')}
    for arm in ARMS:
        documents[arm] = {kind: inputs.read(BASE/f'{arm}.{kind}.json') for kind in JOBS[1:]}
        for kind in ('ensemble', 'endpoint', 'suite', 'matched8'):
            data = documents[arm][kind]
            require(old.local(data['checkpoint']) == old.local(checked[arm]['checkpoint']), 'score checkpoint path differs')
            if kind in ('ensemble', 'endpoint'):
                require(data['checkpoint_sha256'] == checked[arm]['checkpoint_sha256']
                    and data['optimizer_steps'] == 0 and data['test_opened'] is False, 'score checkpoint/scope differs')
                old.shared.audit_sources(data, protocol, inputs, arm+'/'+kind)
                if kind == 'endpoint':
                    old.shared.audit_provenance(data, protocol, inputs, arm+'/'+kind)
        ensemble[arm] = old.shared.audit_ensemble(documents[arm]['ensemble'], arm)
        # The saved compatibility field remains target='endpoint' for both new arms.
        endpoints[arm] = old.audit_endpoint(documents[arm]['endpoint'], 'endpoint', checked[arm])
        ordered[arm] = audit_ordered(documents[arm]['ordered'], arm, checked[arm], protocol, inputs)
        suite_rows[arm], suite_reports[arm] = {}, {}
        for kind in ('suite', 'matched8'):
            suite_rows[arm][kind], suite_reports[arm][kind] = old.audit_suite(documents[arm][kind], arm, kind)
            require(set(suite_rows[arm][kind]) == expected_suite[kind], 'suite manifest selection differs')
    suite_pairing(suite_rows)
    bio, baseline = old.audit_bioemu(inputs)
    maximum = old.audit_pairing(slot(ensemble), slot(endpoints), baseline, slot(documents))
    cross_task_pairing(ensemble, endpoints, ordered, baseline)
    t2a = old.bioemu_t2a_evidence(slot({a: suite_rows[a]['matched8'] for a in ARMS}),
        slot({a: documents[a]['matched8'] for a in ARMS}), inputs)
    values = {**numeric.standard_analysis(ensemble, endpoints, baseline), **numeric.ordered_analysis(ordered)}
    require(count_intervals(values) == 402, 'paired interval coverage differs')
    require(count_standard_means(values) == 670, 'standard descriptive coverage differs')
    require(count_suite_means(suite_reports) == 1700, 'suite descriptive coverage differs')
    report = dict(schema_version=VERSION, created_utc=now(), arms=list(ARMS), candidate=ARMS[0], reference=ARMS[1],
        role='single_seed_adaptive_development_pilot', n_windows=200, n_proteins=50, n_samples=12,
        statistical_registration=str(BASE/'summary.protocol.json'), wrapper_registration=str(IMPLEMENTATION/'protocol.json'),
        implementation_sources=implementation['source_hashes'], statistical_contract=statistics,
        numerical_coverage=dict(paired_intervals=402,standard_descriptive_means=670,suite_descriptive_means=1700),
        checkpoint_audit=checked, paired_training_audit=paired_training, reused_H_gate=gate,
        execution_audit=dict(successful_logical_jobs=12, all_actual_exit0=True,
            source_inputs_snapshots_verified=True, ordered_reference_fingerprints_match_across_arms=True,
            all_standard_task_window_keys_match=True, matched8_full_raw_rows_exact=True),
        bootstrap=statistics['bootstrap'], endpoint_metric_semantics=documents[ARMS[0]]['endpoint']['metric_semantics'],
        suite_reports=suite_reports, bioemu2_matched8_reference_audit=t2a,
        bioemu2_adapter=dict(path=str(old.BIOEMU), sha256=inputs.hashes[old.local(old.BIOEMU)],
            evaluation_args=bio['evaluation_args'], persistence_max_abs_difference=maximum),
        scientific_limits=statistics['scientific_limits'], test_opened=False, bioemu2_rerun=False, new_model_sampling=False,
        **values)
    runner.verify(protocol, BASE/'protocol.json', inputs.hashes[old.local(BASE/'protocol.json')])
    inputs.verify()
    report['input_sha256'] = {old.label(p): d for p, d in sorted(inputs.hashes.items())}
    write_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('register', 'run'), required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        register()
        print(json.dumps(dict(path=str(IMPLEMENTATION/'protocol.json'), sha256=old.sha(IMPLEMENTATION/'protocol.json'))))
        return
    execution = IMPLEMENTATION/'execution.json'
    started = dict(pid=os.getpid(), started_utc=now(), mode='run', source_sha256=old.sha(SCRIPT))
    write_new(IMPLEMENTATION/'started.json', started)
    start = time.monotonic()
    try:
        build_summary()
        require(old.sha(SCRIPT) == started['source_sha256'], 'summary source changed during run')
        write_new(execution, dict(started, completed_utc=now(), seconds=time.monotonic()-start,
            exit_code=0, status='completed', output=str(BASE/'summary.json'), output_sha256=old.sha(BASE/'summary.json')))
    except BaseException as error:
        write_new(execution, dict(started, completed_utc=now(), seconds=time.monotonic()-start,
            exit_code=1, status='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
