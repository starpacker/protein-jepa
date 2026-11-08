"""Provenance-first CPU summary of the matched ordered-path training pilot.

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

from scripts import hs_ordered_pilot_statistics_v1 as numeric
from scripts import register_hs_ordered_pilot_statistics_v1 as registration
from scripts import summarize_hs_target_pilot as old
from scripts import summarize_ca_ordered_paths as paths
from run_scripts import run_hs_ordered_path_pilot_v1 as runner

BASE, ARTIFACTS = runner.RESULTS, runner.ARTIFACTS
IMPLEMENTATION = Path('results/hs_joint_20261101/ordered_path_summary_v1')
SCRIPT = 'scripts/summarize_hs_ordered_pilot_v1.py'
TEST = 'tests/test_hs_ordered_summary_v1.py'
VERSION = 'hs_ordered_path_pilot_summary_v1'
PROTOCOL_SHA = 'a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb'
STATISTICS_SHA = '4facecde654e82b780a587bdb0a81307da6902c43f6ec50fc305d92e74f78ff4'
REGISTRATION_EVIDENCE = {
    'results/hs_joint_20261101/ordered_path_register_v1.execution.json': '82b99aa3e8e1e8395fafac42587c058d2a966513ef6f7de041293b3e0ec77e9a',
    'results/hs_joint_20261101/ordered_path_register_v1.log': '2a4383505c9b3d85fe86787cc569893762577dcf2bd41939d2848cf9fabfd2be',
    'results/hs_joint_20261101/ordered_path_register_v2.execution.json': '4e83dc854c3314e5f960dcb8faca8cc69d4099ae3574ecaaceab41819ac7f863',
    'results/hs_joint_20261101/ordered_path_register_v2.log': '9fc57678b9544705a2347c83263d643a2c719c88d6e2412380d6cb0ad8bf8b57',
}
ARMS, JOBS = numeric.ARMS, runner.JOBS
SLOTS = dict(zip(ARMS, old.ARMS))
require = old.require
SOURCES = (*registration.SOURCES, SCRIPT, TEST,
    'run_scripts/run_hs_ordered_path_pilot_v1.py',
    'protein_jepa_frame/training/train_lasp_ca_ordered_v1.py',
    'protein_jepa_frame/training/ordered_path_objective_v1.py',
    'scripts/eval_ca_ordered_paths.py', 'evaluation/trajectory_metrics_v2.py',
    'evaluation/ca_trajectory_adapter.py', 'run_scripts/run_hs_target_pilot.py',
    'run_scripts/run_hs_markov_pilot_v1.py', str(runner.DOCUMENT))


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
        n_paired_intervals=402, standard_descriptive_means=670,
        statistics_protocol=str(BASE/'summary.protocol.json'),
        experiment_protocol_sha256=PROTOCOL_SHA, statistics_sha256=STATISTICS_SHA,
        output=str(BASE/'summary.json'),
        ordering='Both complete exit0 executions are checked before loading checkpoints or parsing score JSON.',
        data='All 500 actual training-window metadata and batch fingerprints must agree byte for byte across the two new arms.',
        audit='Reload checkpoint tensors on CPU; check initial identity, complete frozen component, optimizer support and 500 steps, final global/sampling RNG.',
        roles='joint_path minus marginal_path; old endpoint/path_mixture positional slots are internal only.',
        suite='All 7200/full and 800/matched raw rows audited; overlapping full/matched rows must match exactly.',
        limits='Single-seed adaptive development pilot. Historical BioEmu2 T2A reference fingerprints are missing; no new paired T2A superiority claim. Frozen long-horizon H gate remains unchanged.',
        test_opened=False, bioemu2_rerun=False, model_sampling=False)


def register():
    output = old.local(IMPLEMENTATION/'protocol.json')
    require(not output.exists() and not old.local(BASE/'summary.json').exists(), 'refusing registration overwrite')
    for arm in ARMS:
        for kind in JOBS[1:]:
            require(not old.local(BASE/f'{arm}.{kind}.json').exists(), 'new score exists before wrapper registration')
    inputs = old.Inputs()
    inputs.track(BASE/'protocol.json', PROTOCOL_SHA)
    inputs.track(BASE/'summary.protocol.json', STATISTICS_SHA)
    statistics = inputs.read(BASE/'summary.protocol.json')
    require(all(statistics.get(k) == v for k, v in registration.contract().items()), 'statistics contract changed')
    for source, digest in statistics['source_hashes'].items():
        inputs.track(source, digest)
    for source in sorted(set(SOURCES)):
        inputs.track(source)
    for source, expected in REGISTRATION_EVIDENCE.items():
        inputs.track(source, expected)
    inputs.verify()
    result = dict(contract(), registered_utc=now(),
        registration_stage='Before this author/wrapper reads any new ordered-pilot model score; statistical estimands were separately registered before training. Existing target/Markov development results are known.',
        registration_attempt_hashes=REGISTRATION_EVIDENCE,
        source_hashes={old.label(p): d for p, d in inputs.hashes.items() if p.suffix in ('.py', '.md')})
    write_new(output, result)
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
    require(implementation['statistics_sha256'] == protocol['statistics_sha256'] == digest == STATISTICS_SHA
        and inputs.hashes[old.local(BASE/'protocol.json')] == PROTOCOL_SHA,
            'statistical registration hash differs')
    require(datetime.fromisoformat(statistics['registered_utc']) <= datetime.fromisoformat(protocol['registered_utc']),
            'statistics were registered after training protocol')
    for mapping in (statistics['source_hashes'], implementation['source_hashes']):
        for source, expected in mapping.items():
            inputs.track(source, expected)
    require(set(SOURCES) <= set(implementation['source_hashes']), 'wrapper source coverage incomplete')
    require(protocol['schema_version'] == 'hs_ordered_path_pilot_v1' and tuple(protocol['arms']) == ARMS
        and tuple(protocol['jobs']) == JOBS and protocol['test_opened'] is False
        and protocol['bioemu2_rerun'] is False, 'experiment scope differs')
    require(protocol['per_arm_commands'] == {a: runner.commands(a) for a in ARMS}, 'registered commands differ')
    require(protocol['checkpoint_selection'] == 'both latest500 only; no validation/best/early stopping', 'selection changed')
    require(old.local(protocol['statistics_protocol']) == old.local(BASE/'summary.protocol.json'), 'statistics path differs')
    require(old.local(protocol['initial_checkpoint']) == old.local(runner.INITIAL)
        and protocol['initial_checkpoint_sha256'] == runner.INITIAL_SHA, 'warm start changed')
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


def inference_hash(checkpoint, *, component_only=False):
    from protein_jepa_frame.utils.contracts import state_sha256
    prefix = '' if component_only else 'bundle.'
    state = {f'{prefix}encoder.{k}': v for k, v in checkpoint['component_checkpoint']['encoder'].items()}
    state.update({f'{prefix}decoder.{k}': v for k, v in checkpoint['component_checkpoint']['delta_ca_decoder'].items()})
    if not component_only:
        state.update({f'kernel.{k}': v for k, v in checkpoint['kernel'].items()})
    return state_sha256(state)


def optimizer_pair(first, second):
    import torch
    require(old.shared.same_tree(first['param_groups'], second['param_groups']), 'optimizer parameter groups differ')
    require(set(first['state']) == set(second['state']) and bool(first['state']), 'optimizer support differs')
    for key in first['state']:
        a, b = first['state'][key], second['state'][key]
        require(set(a) == set(b) and float(torch.as_tensor(a['step'])) == float(torch.as_tensor(b['step'])) == 500,
                'optimizer state step/support differs')
        for name in a:
            x, y = a[name], b[name]
            if isinstance(x, torch.Tensor):
                require(isinstance(y, torch.Tensor) and x.shape == y.shape and x.dtype == y.dtype
                    and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'optimizer tensor metadata/finite differs')
            else:
                require(type(x) is type(y), 'optimizer scalar type differs')
    return dict(active_states=len(first['state']), steps=500, moments_may_differ=True)


def audit_training(checkpoints, initial, executions, protocol, inputs):
    from protein_jepa_frame.utils.contracts import state_sha256
    require(state_sha256(initial['kernel']) == protocol['initial_kernel_sha256']
        and initial['kernel_config'] == protocol['initial_kernel_config']
        and old.component_identity(initial['component_checkpoint']) == protocol['initial_component_identity'], 'initial checkpoint identity differs')
    checked, identities, raw_windows, first_steps = {}, {}, {}, {}
    for arm in ARMS:
        ck = checkpoints[arm]
        actual = runner.checkpoint_audit(ck, arm, protocol)
        require(actual == executions[arm]['checkpoint_audit'], 'saved checkpoint audit disagrees')
        run = inputs.read(ARTIFACTS/arm/'run.json')
        require(run['args'] == ck['args'] and run['train_only'] is True and run['test_opened'] is False
            and run['no_validation_selection'] is True
            and run['actual_pid'] == executions[arm]['jobs'][0]['pid'], 'training process/run contract differs')
        split = run['split_audit']
        require(all(split[k] == 0 for k in ('protein_overlap', 'domain_overlap', 'source_file_overlap')), 'split overlap')
        for role, path in (('train', old.shared.TRAIN), ('val', old.shared.VAL)):
            require(split[role+'_manifest_sha256'] == protocol['input_hashes'][path], 'split manifest identity differs')
        identities[arm] = inputs.read(ARTIFACTS/arm/'initial_identity.json')
        expected_identity = dict(model_sha256=inference_hash(initial), component_sha256=inference_hash(initial, component_only=True),
            kernel_sha256=state_sha256(initial['kernel']), kernel_config=initial['kernel_config'])
        require(identities[arm] == expected_identity, 'actual initialized model differs from warm start')
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
            for value in [row['grad_norm'], row['lr'], *row['train'].values()]:
                old.number(value)
        first_steps[arm] = {k: metrics[0][k] for k in ('step', 'horizon', 'grad_norm', 'lr', 'train')}
        checked[arm] = dict(checkpoint=str(ARTIFACTS/arm/'checkpoint_latest.pt'), checkpoint_sha256=executions[arm]['latest_sha256'],
            args=ck['args'], step=500, step_ps=ck['kernel_config']['step_ps'], inference_model_tensor_sha256=inference_hash(ck), audit=actual)
    require(raw_windows[ARMS[0]] == raw_windows[ARMS[1]], 'actual 500 training batches differ across arms')
    require(first_steps[ARMS[0]] == first_steps[ARMS[1]], 'first h1 equivalent loss/gradient differs')
    for key in ('sampling_rng', 'rng_state'):
        runner.exact_tree(checkpoints[ARMS[0]][key], checkpoints[ARMS[1]][key], key)
    normalized = [{k: v for k, v in checkpoints[a]['args'].items() if k not in ('coord_objective', 'output')} for a in ARMS]
    require(normalized[0] == normalized[1], 'unexpected training argument difference')
    optimizer = optimizer_pair(*(checkpoints[a]['optimizer'] for a in ARMS))
    expected_pair = dict(initial_model_equal=True, actual_training_window_bytes_equal=True,
        final_sampling_and_global_rng_equal=True, only_argument_differences=['output', 'coord_objective'],
        compared_with_historical_endpoint_rng=False)
    require(inputs.read(BASE/'paired_training_audit.json') == expected_pair, 'runner paired audit differs')
    return checked, dict(**expected_pair, n_actual_training_windows=500, optimizer=optimizer,
        first_h1_loss_and_gradient_equal=True, final_model_and_optimizer_moments_may_differ=True,
        input_identity='Both new arms serialized identical actual loaded-batch fingerprints; this does not retroactively identify old endpoint training coordinates.')


def audit_ordered(data, arm, checked, protocol, inputs):
    from scripts.eval_ca_ordered_paths import SOURCES as required_sources, panel_spec, summarize as raw_summary
    output = BASE/f'{arm}.ordered.json'
    registered = inputs.read(output.with_suffix('.protocol.json'))
    require(all(data.get(k) == v for k, v in registered.items()), 'ordered output changed registration')
    require(data['task_version'] == 'ca_ordered_paths_development_v1' and data['dataset_role'] == 'development_validation'
        and data['optimizer_steps'] == 0 and data['model_selection'] is False and data['test_opened'] is False
        and data['bioemu2_rerun'] is False and data['future_information_given_to_sampler'] is False
        and data['frozen_inputs_verified'] is True and data.get('completed_utc'), 'ordered task incomplete/different')
    require(data['panel'] == panel_spec([1, 4, 16, 64], 200) and data['n_windows'] == 200
        and data['n_proteins'] == 50, 'ordered panel incomplete')
    require(old.local(data['args']['checkpoint']) == old.local(checked['checkpoint'])
        and data['input_hashes'][data['args']['checkpoint']] == checked['checkpoint_sha256']
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


def audit_registration_attempts(inputs):
    """Registration environment failure is not a training/evaluation restart."""
    for path, digest in REGISTRATION_EVIDENCE.items():
        inputs.track(path, digest)
    rows = [inputs.read(f'results/hs_joint_20261101/ordered_path_register_v{i}.execution.json') for i in (1, 2)]
    require([r['exit_code'] for r in rows] == [1, 0] and rows[0]['command'] == rows[1]['command']
        and rows[0]['command'][-2:] == ['--mode', 'register'], 'registration attempts differ')
    for row in rows:
        require(row.get('completed_utc') and row.get('started_utc')
            and row['log'] in REGISTRATION_EVIDENCE, 'registration execution evidence incomplete')
    failure = inputs.track(rows[0]['log']).read_text()
    require('two GPUs required; no automatic device reassignment' in failure, 'registration failure cause differs')
    return dict(attempts=rows, source_and_log_hashes=REGISTRATION_EVIDENCE,
        interpretation='First registration failed at the two-GPU environment guard before formal training. The same frozen source was registered successfully on the second attempt. These are two registration attempts, not part of the twelve training/evaluation jobs or a training restart.')


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
    registration_attempts = audit_registration_attempts(inputs)
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
    report = dict(schema_version=VERSION, created_utc=now(), arms=list(ARMS), candidate=ARMS[0], reference=ARMS[1],
        role='single_seed_adaptive_development_pilot', n_windows=200, n_proteins=50, n_samples=12,
        statistical_registration=str(BASE/'summary.protocol.json'), wrapper_registration=str(IMPLEMENTATION/'protocol.json'),
        implementation_sources=implementation['source_hashes'], statistical_contract=statistics,
        checkpoint_audit=checked, paired_training_audit=paired_training, reused_H_gate=gate,
        registration_attempts=registration_attempts,
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
