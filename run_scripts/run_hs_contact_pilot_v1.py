"""Explicit register/run/smoke phases for the contact-estimator matched pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from run_scripts import run_hs_target_pilot as audit
from run_scripts.continue_hs_target_pilot_v2 import parse_cli, assert_endpoint_fixed96
from run_scripts.run_hs_markov_pilot_v1 import exact_tree
from protein_jepa_frame.training import train_lasp_ca_contact_v1 as training
from protein_jepa_frame.training import train_lasp_ca_ordered_v1 as ordered_training
from protein_jepa_frame.utils.contracts import state_sha256
from scripts import eval_ca_ordered_paths  # noqa: F401; include actual evaluation sources


RESULTS = Path('results/hs_joint_20261101/contact_estimator_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/contact_estimator_v1')
INITIAL = audit.INITIAL
INITIAL_SHA = audit.INITIAL_SHA
BASE_GATE = audit.BASE_GATE
BASE_GATE_SHA = audit.BASE_GATE_SHA
STATISTICS = RESULTS/'summary.protocol.json'
DOCUMENT = Path('docs/HS_CONTACT_ESTIMATOR_PILOT_V1_20261102.md')
TESTS = Path('tests/test_hs_contact_pilot_v1.py')
HISTORICAL = Path('results/hs_joint_20261101/ordered_path_v1/protocol.json')
HISTORICAL_SHA = 'a31b87128702cd796b25fbbef60c3bd1245304cfd369c668970e54c72c7235cb'
GRADIENT_DIAGNOSTIC = Path('results/hs_joint_20261101/contact_kernel_gradients_v2/result.json')
GRADIENT_DIAGNOSTIC_SHA = 'bd03074123092497c83d85ee7118934f79d10ee2c677a68c5f02c163948bf4fc'
ARMS = {'population_u': {'gpu': 0}, 'plugin': {'gpu': 1}}
JOBS = ('train', 'ensemble', 'endpoint', 'ordered', 'suite', 'matched8')
now, canonical = audit.now, audit.canonical
sha256_file, write_new, update_status = audit.sha256_file, audit.write_new, audit.update_status


def commands(arm):
    if arm not in ARMS:
        raise ValueError('unknown registered arm')
    latest = str(ARTIFACTS/arm/'checkpoint_latest.pt')
    result = {'train': [sys.executable, '-u', '-m',
        'protein_jepa_frame.training.train_lasp_ca_contact_v1',
        '--output', str(ARTIFACTS/arm), '--contact-estimator', arm]}
    original = audit.commands('endpoint')
    for kind in JOBS[1:]:
        if kind == 'ordered':
            command = [sys.executable, '-u', '-m', 'scripts.eval_ca_ordered_paths',
                '--checkpoint', latest, '--out', str(RESULTS/f'{arm}.{kind}.json'),
                '--max-windows', '200', '--horizons', '1', '4', '16', '64',
                '--device', 'cuda:0', '--torch-threads', '4']
        else:
            command = list(original[kind])
            command[command.index('--checkpoint')+1] = latest
            command[command.index('--out')+1] = str(RESULTS/f'{arm}.{kind}.json')
            if kind == 'endpoint':
                index = command.index('--max-residues')
                if command[index+1] != '96':
                    raise ValueError('unexpected frozen endpoint command')
                del command[index:index+2]
        result[kind] = command
    return result


def preflight_commands():
    """Parse each actual producer main without touching data/model execution."""
    assert_endpoint_fixed96()
    audit.assert_endpoint_default()
    parsed = {arm: {kind: parse_cli(command) for kind, command in commands(arm).items()}
              for arm in ARMS}
    for arm, jobs in parsed.items():
        if (jobs['train']['contact_estimator'] != arm or jobs['train']['run_kind'] != 'train'):
            raise ValueError('contact training parser contract differs')
        if jobs['endpoint']['max_windows'] != 200 or jobs['endpoint']['n_samples'] != 12:
            raise ValueError('endpoint parser budget differs')
        if jobs['ordered']['max_windows'] != 200 or jobs['ordered']['horizons'] != [1, 4, 16, 64]:
            raise ValueError('ordered panel differs')
        for kind, proteins, anchors, modes in [('suite', 12, 2, ['correct', 'wrong', 'none']),
                                               ('matched8', 8, 1, ['correct'])]:
            args = jobs[kind]
            if (args['proteins_per_temp'] != proteins or args['anchors_per_trajectory'] != anchors
                    or args['modes'] != modes or args['n_samples'] != 12 or not args['t2a']
                    or args['context_frames'] != 8 or args['max_residues'] != 96):
                raise ValueError('standard-suite parser budget differs')
    return parsed


def sources():
    root = Path.cwd().resolve()
    paths = set(audit.sources()) | set(eval_ca_ordered_paths.SOURCES) | {
        str(Path(__file__).resolve().relative_to(root)), str(DOCUMENT), str(TESTS),
        'protein_jepa_frame/training/train_lasp_ca_contact_v1.py',
        'protein_jepa_frame/training/train_lasp_ca_ordered_v1.py',
        'protein_jepa_frame/training/contact_corrected_path_objective_v1.py',
        'protein_jepa_frame/training/contact_ensemble_objective_v1.py',
        'tests/test_lasp_ca_contact_v1.py',
        'tests/test_contact_corrected_path_objective_v1.py',
        'tests/test_contact_ensemble_objective_v1.py',
        'protein_jepa_frame/training/ordered_path_objective_v1.py',
        'tests/test_ordered_path_objective_v1.py',
        'run_scripts/continue_hs_target_pilot_v2.py',
        'run_scripts/run_hs_markov_pilot_v1.py',
        'evaluation/ca_trajectory_adapter.py', 'evaluation/trajectory_metrics_v2.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(root):
                paths.add(str(path.relative_to(root)))
    return sorted(paths)


def source_snapshot(destination, expected):
    destination.mkdir(parents=True, exist_ok=False)
    for path, digest in expected.items():
        target = destination/path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(Path(path).read_bytes())
        if sha256_file(target) != digest:
            raise ValueError('source changed during exclusive snapshot')


def fixed_inputs():
    for path, expected in [(INITIAL, INITIAL_SHA), (BASE_GATE, BASE_GATE_SHA),
                           (HISTORICAL, HISTORICAL_SHA), (GRADIENT_DIAGNOSTIC, GRADIENT_DIAGNOSTIC_SHA)]:
        if sha256_file(path) != expected:
            raise ValueError('fixed upstream changed: '+str(path))
    initial = torch.load(INITIAL, map_location='cpu', weights_only=False)
    gate = json.loads(BASE_GATE.read_text())
    gate_checkpoint = Path(gate['checkpoint'])
    if sha256_file(gate_checkpoint) != gate['checkpoint_sha256']:
        raise ValueError('gate checkpoint changed')
    baseline = torch.load(gate_checkpoint, map_location='cpu', weights_only=False)
    exact_tree(initial['component_checkpoint'], baseline['component_checkpoint'], 'H-gate component')
    identity = audit.component_identity(initial['component_checkpoint'])
    if (gate['encoder_tensor_sha256'] != identity['encoder']
            or gate['decoder_tensor_sha256'] != identity['decoder']):
        raise ValueError('gate component hashes differ')
    ca = initial['component_checkpoint']['calibration_args']
    paths = [INITIAL, BASE_GATE, gate_checkpoint, audit.BIOEMU, HISTORICAL, GRADIENT_DIAGNOSTIC,
        Path(ordered_training.FIXED['manifest']), Path(ordered_training.FIXED['val_manifest']),
        Path(ca['stage_a_ckpt']), Path(ca['b1a_ckpt'])]
    return initial, identity, {str(path): sha256_file(path) for path in paths}


def training_stream():
    args = training.arguments('unused_metadata_only', 'plugin')
    loaders = {h: training.make_loader(args, args.manifest, h, True) for h in args.horizons}
    rows, stats = [], {}
    for step in range(1, args.steps+1):
        h = args.horizons[(step-1) % len(args.horizons)]
        row = ordered_training.training_identity(loaders[h], step, args.seed)
        rows.append(row)
        stats[row['record_path']] = ordered_training.file_stat(row['record_path'])
    if sha256_file(HISTORICAL) != HISTORICAL_SHA:
        raise ValueError('historical window registration changed')
    historical = json.loads(HISTORICAL.read_text())
    if rows != historical['training_windows'] or stats != historical['selected_hdf5_stats']:
        raise ValueError('new training stream differs from the fixed historical stream')
    return rows, stats


def split_audit():
    """Validate train/validation manifest metadata; never open trajectories/test."""
    from protein_jepa_frame.data.protocol import validate_manifest_split
    result = validate_manifest_split(ordered_training.FIXED['manifest'], ordered_training.FIXED['val_manifest'])
    if any(result[key] != 0 for key in ('protein_overlap','domain_overlap','source_file_overlap')):
        raise ValueError('training and validation split overlap')
    return result


def verify(protocol, protocol_path=None, protocol_sha=None):
    if protocol_path is not None and sha256_file(protocol_path) != protocol_sha:
        raise ValueError('registered protocol changed')
    immutable = {**protocol['input_hashes'], **protocol['source_hashes']}
    audit.verify_files(immutable)
    for path, digest in protocol['source_hashes'].items():
        if sha256_file(Path(protocol['source_snapshot'])/path) != digest:
            raise ValueError('registered source snapshot changed')
    for path, expected in protocol.get('selected_hdf5_stats', {}).items():
        if ordered_training.file_stat(path) != expected:
            raise ValueError('registered training trajectory stat changed')
    return immutable


def smoke(gpu):
    """Manually requested feasibility process, independently registered before data."""
    preflight_commands()
    _, identity, inputs = fixed_inputs()
    split = split_audit()
    stream, stats = training_stream()
    selected = stream[3]
    directory = RESULTS/'smoke_v1'
    directory.mkdir(parents=True, exist_ok=False)
    expected = {path: sha256_file(path) for path in sources()}
    snapshot = directory/'source_snapshot'
    source_snapshot(snapshot, expected)
    output = directory/'job'
    command = [sys.executable, '-u', '-m', 'protein_jepa_frame.training.train_lasp_ca_contact_v1',
        '--output', str(output), '--contact-estimator', 'population_u', '--run-kind', 'smoke']
    protocol = dict(schema_version='hs_contact_estimator_smoke_v1', registered_utc=now(),
        selected_training_window=selected, source_hashes=expected, input_hashes=inputs,
        selected_hdf5_stats={selected['record_path']: stats[selected['record_path']]},
        source_snapshot=str(snapshot), initial_component_identity=identity, split_audit=split,
        command=command, gpu=gpu, optimizer_steps=0, horizon=64, n_samples=12,
        purpose='memory/time/finite-gradient feasibility; no model or coefficient selection')
    path = directory/'protocol.json'; write_new(path, protocol)
    digest = sha256_file(path)
    status = dict(status='running', runner_pid=os.getpid(), command=command, started_utc=now())
    execution = directory/'execution.json'; write_new(execution, status)
    try:
        status['frozen_before'] = verify(protocol, path, digest)
        start = time.monotonic()
        log = directory/'run.log'
        with log.open('x') as handle:
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment(gpu))
            status['pid'] = child.pid; update_status(execution, status)
            code = child.wait()
        status.update(returncode=code, seconds=time.monotonic()-start,
                      completed_utc=now(), log=str(log), log_sha256=sha256_file(log))
        status['frozen_after'] = verify(protocol, path, digest)
        if code:
            raise RuntimeError(f'feasibility smoke exited {code}; failure preserved, no restart')
        report = output/'smoke.json'
        status.update(status='completed', output=str(report), output_sha256=sha256_file(report))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        update_status(execution, status)
    return status


def register():
    if ARTIFACTS.exists():
        raise FileExistsError('contact experiment already registered')
    if not RESULTS.exists() or {p.name for p in RESULTS.iterdir()} != {'smoke_v1', 'summary.protocol.json'}:
        raise ValueError('registration requires only completed smoke and independent statistics protocol')
    smoke_dir = RESULTS/'smoke_v1'
    sp = json.loads((smoke_dir/'protocol.json').read_text())
    verify(sp, smoke_dir/'protocol.json', sha256_file(smoke_dir/'protocol.json'))
    execution = json.loads((smoke_dir/'execution.json').read_text())
    if execution.get('status') != 'completed' or execution.get('returncode') != 0 or execution.get('pid', 0) <= 0:
        raise ValueError('requires an actual exit0 feasibility smoke')
    if sha256_file(execution['log']) != execution['log_sha256'] or sha256_file(execution['output']) != execution['output_sha256']:
        raise ValueError('smoke artifacts changed')
    result = json.loads(Path(execution['output']).read_text())
    if (result.get('optimizer_steps') != 0 or result.get('n_residues') != 96
            or set(result.get('arms', {})) != set(ARMS)
            or result.get('model_before_sha256') != result.get('model_after_sha256')):
        raise ValueError('smoke budget or zero-update model audit differs')
    parsed = preflight_commands()
    initial, identity, inputs = fixed_inputs()
    split = split_audit()
    current_sources = {path: sha256_file(path) for path in sources()}
    if any(current_sources.get(path) != digest for path, digest in sp['source_hashes'].items()):
        raise ValueError('sources changed since feasibility smoke; a reviewed new smoke version is required')
    for item in [STATISTICS, smoke_dir/'protocol.json', smoke_dir/'execution.json',
                 Path(execution['output']), Path(execution['log'])]:
        inputs[str(item)] = sha256_file(item)
    statistics = json.loads(STATISTICS.read_text())
    from scripts.register_hs_contact_pilot_statistics_v1 import contract as statistical_contract
    if any(statistics.get(key) != value for key,value in statistical_contract().items()):
        raise ValueError('contact statistical contract differs')
    for path, digest in statistics['source_hashes'].items():
        source = Path(path)
        if (source.is_absolute() or not source.resolve().is_relative_to(Path.cwd().resolve())
                or not source.is_file() or sha256_file(source) != digest):
            raise ValueError('statistical source missing, outside repository, or changed: '+path)
        if path in current_sources and current_sources[path] != digest:
            raise ValueError('statistical and experiment source digests differ')
        current_sources[path] = digest
    if torch.cuda.device_count() < 2:
        raise RuntimeError('two GPUs required; no automatic device reassignment')
    stream, stats = training_stream()
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    snapshot = ARTIFACTS/'source_snapshot'; source_snapshot(snapshot, current_sources)
    protocol = dict(schema_version='hs_contact_estimator_pilot_v1', registered_utc=now(),
        arms=ARMS, jobs=list(JOBS), per_arm_commands={a: commands(a) for a in ARMS},
        cli_parser_contracts=parsed, training_arguments={a: vars(training.arguments(ARTIFACTS/a,a)) for a in ARMS},
        initial_checkpoint=str(INITIAL), initial_checkpoint_sha256=INITIAL_SHA,
        initial_kernel_sha256=state_sha256(initial['kernel']), initial_component_identity=identity,
        initial_kernel_config=initial['kernel_config'], source_hashes=current_sources,
        input_hashes=inputs, source_snapshot=str(snapshot), training_windows=stream, split_audit=split,
        selected_hdf5_stats=stats, statistics_protocol=str(STATISTICS), statistics_sha256=sha256_file(STATISTICS),
        checkpoint_selection='both latest500 only; no validation/best/early stopping',
        primary='population_u minus plugin coordinate_path_energy_score; equal-protein h16/64 average',
        evaluation=dict(ensemble_windows=200, endpoint_windows=200, ordered_windows=200,
            proteins=50, samples=12, seed=0, horizons=[1,4,16,64],
            suite=dict(windows=7200, proteins=12, trajectories=300, modes=['correct','wrong','none']),
            matched8=dict(windows=800, proteins=8, trajectories=200, modes=['correct'])),
        h_gate_reuse=dict(path=str(BASE_GATE), sha256=BASE_GATE_SHA, long_horizon_failures_retained=True),
        historical_endpoint_role='supplementary only; old training used validation, new two-arm primary does not',
        train_stream_contract='same stateless seed/index formula; full historical global/sampling RNG identity is not asserted',
        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            gpu_names=[torch.cuda.get_device_name(i) for i in range(2)]),
        test_opened=False, bioemu2_rerun=False)
    verify(protocol)
    write_new(RESULTS/'protocol.json', protocol)
    return protocol


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='/tmp/lasp_stability_deps:.',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')


def checkpoint_audit(checkpoint, arm, protocol):
    if (checkpoint.get('checkpoint_kind') != 'lasp_ca_dynamics' or checkpoint.get('step') != 500
            or checkpoint.get('training_version') != training.VERSION):
        raise ValueError('requires contact latest exactly500')
    if canonical(checkpoint['args']) != canonical(protocol['training_arguments'][arm]):
        raise ValueError('saved training arguments differ from registration')
    initial = torch.load(INITIAL, map_location='cpu', weights_only=False)
    exact_tree(checkpoint['component_checkpoint'], initial['component_checkpoint'], 'frozen component')
    names = list(initial['kernel'])
    if len(names) != 73 or list(checkpoint['kernel']) != names:
        raise ValueError('full kernel inventory or order differs')
    groups = checkpoint['optimizer']['param_groups']
    if len(groups) != 1 or groups[0]['params'] != list(range(73)):
        raise ValueError('optimizer must retain all 73 parameter IDs in kernel order')
    unused = {i for i,name in enumerate(names)
              if name == 'energy.energy_head.2.bias' or name.startswith('history_posterior.')}
    if len(unused) != 9 or set(checkpoint['optimizer']['state']) != set(range(73))-unused:
        raise ValueError('optimizer active/unused support differs')
    for i in unused:
        exact_tree(checkpoint['kernel'][names[i]], initial['kernel'][names[i]], 'unused parameter')
    if (checkpoint.get('target_encoder') is not None or checkpoint.get('validation') is not None
            or checkpoint['kernel_config'] != protocol['initial_kernel_config']):
        raise ValueError('component/configuration or validation contract changed')
    if any(not bool(torch.isfinite(value).all()) for value in checkpoint['kernel'].values()):
        raise ValueError('nonfinite saved kernel')
    steps = [float(torch.as_tensor(state['step'])) for state in checkpoint['optimizer']['state'].values()]
    if not steps or any(step != 500 for step in steps):
        raise ValueError('active optimizer states must all record exactly500 updates')
    return dict(step=500, actual_args=checkpoint['args'], kernel_config=checkpoint['kernel_config'],
        kernel_tensor_sha256=state_sha256(checkpoint['kernel']),
        component_identity=audit.component_identity(checkpoint['component_checkpoint']),
        complete_frozen_component_verified=True, target_encoder_is_none=True,
        active_optimizer_states=len(steps), optimizer_steps=500,
        optimizer_parameter_tensors=73, unused_parameter_tensors=9,
        unused_parameters_unchanged=True,
        no_validation_selection=True, sampling_rng_sha256=state_sha256({'rng': checkpoint['sampling_rng']}))


def audit_training_windows(path, expected):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    metadata = [{k:v for k,v in row.items() if k not in ('actual_batch_sha256','file_stat')} for row in rows]
    if canonical(metadata) != canonical(expected):
        raise ValueError('actual training window metadata differs from registration')
    if any(len(row.get('actual_batch_sha256','')) != 64 or 'file_stat' not in row for row in rows):
        raise ValueError('actual training window fingerprints missing')
    return dict(windows=len(rows), metadata_matches_registration=True,
                actual_batch_fingerprints_retained=True, sha256=sha256_file(path))


def absent_outputs(arm, kind):
    path = RESULTS/f'{arm}.{kind}.json'
    candidates = [path, path.with_suffix('.component.pt')]
    if kind == 'ordered':
        candidates += [path.with_suffix('.protocol.json'), path.with_suffix('.source_snapshot')]
    for value in candidates:
        if value.exists():
            raise FileExistsError(value)
    return path


def run_arm(arm, protocol, protocol_sha):
    execution = RESULTS/f'{arm}.execution.json'
    status = dict(arm=arm, status='running', runner_pid=os.getpid(), started_utc=now(), jobs=[])
    write_new(execution, status)
    latest = ARTIFACTS/arm/'checkpoint_latest.pt'
    try:
        for kind in JOBS:
            before = verify(protocol, RESULTS/'protocol.json', protocol_sha)
            if kind == 'train':
                if (ARTIFACTS/arm).exists():
                    raise FileExistsError(ARTIFACTS/arm)
            else:
                output = absent_outputs(arm, kind)
                if sha256_file(latest) != status['latest_sha256']:
                    raise ValueError('checkpoint changed before evaluation')
            log = RESULTS/f'{arm}.{kind}.log'
            entry = dict(kind=kind, command=protocol['per_arm_commands'][arm][kind],
                frozen_before=before, started_utc=now(), log=str(log))
            status['jobs'].append(entry)
            start = time.monotonic()
            with log.open('x') as handle:
                child = subprocess.Popen(entry['command'], stdout=handle, stderr=subprocess.STDOUT,
                                         env=environment(protocol['arms'][arm]['gpu']))
                entry['pid'] = child.pid; update_status(execution, status)
                code = child.wait()
            entry.update(returncode=code, seconds=time.monotonic()-start, completed_utc=now(),
                         log_sha256=sha256_file(log))
            update_status(execution, status)
            entry['frozen_after'] = verify(protocol, RESULTS/'protocol.json', protocol_sha)
            if code:
                raise RuntimeError(f'{arm}/{kind} exited {code}; no restart')
            if kind == 'train':
                saved = torch.load(latest, map_location='cpu', weights_only=False)
                status['checkpoint_audit'] = checkpoint_audit(saved, arm, protocol)
                status['latest_sha256'] = sha256_file(latest)
                entry['training_windows_audit'] = audit_training_windows(
                    ARTIFACTS/arm/'training_windows.jsonl', protocol['training_windows'])
            else:
                if sha256_file(latest) != status['latest_sha256']:
                    raise ValueError('checkpoint changed during evaluation')
                entry['output_sha256'] = sha256_file(output)
                sidecar = output.with_suffix('.component.pt')
                if sidecar.exists():
                    entry['component_sha256'] = sha256_file(sidecar)
                if kind == 'ordered':
                    entry['ordered_protocol_sha256'] = sha256_file(output.with_suffix('.protocol.json'))
            update_status(execution, status)
            print(f'{arm}/{kind} completed exit0', flush=True)
        status.update(status='completed', completed_utc=now(), successful_logical_jobs=len(JOBS))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}', failed_utc=now())
        raise
    finally:
        update_status(execution, status)


def paired_training_audit():
    checkpoints = [torch.load(ARTIFACTS/arm/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
                   for arm in ARMS]
    initial = [json.loads((ARTIFACTS/arm/'initial_identity.json').read_text()) for arm in ARMS]
    if initial[0] != initial[1]:
        raise ValueError('two-arm initial model identity differs')
    windows = [(ARTIFACTS/arm/'training_windows.jsonl').read_bytes() for arm in ARMS]
    if windows[0] != windows[1]:
        raise ValueError('two-arm actual training windows/fingerprints differ')
    for key in ('sampling_rng', 'rng_state'):
        exact_tree(checkpoints[0][key], checkpoints[1][key], 'paired '+key)
    normalized = [{k:v for k,v in ck['args'].items() if k not in ('output','contact_estimator')}
                  for ck in checkpoints]
    if canonical(normalized[0]) != canonical(normalized[1]):
        raise ValueError('two arms differ beyond objective/output')
    return dict(initial_model_equal=True, actual_training_window_bytes_equal=True,
        final_sampling_and_global_rng_equal=True, only_argument_differences=['output','contact_estimator'],
        compared_with_historical_endpoint_rng=False)


def run():
    path = RESULTS/'protocol.json'; digest = sha256_file(path)
    protocol = json.loads(path.read_text()); verify(protocol, path, digest)
    for arm in ARMS:
        if (RESULTS/f'{arm}.execution.json').exists() or (ARTIFACTS/arm).exists():
            raise FileExistsError('arm already attempted; no restart')
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_arm, arm, protocol, digest): arm for arm in ARMS}
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as error:
                failures.append(futures[future]); print(str(error), flush=True)
    if failures:
        raise RuntimeError('failed arms: '+','.join(failures))
    verify(protocol, path, digest)
    write_new(RESULTS/'paired_training_audit.json', paired_training_audit())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight','smoke','register','run'), required=True)
    parser.add_argument('--gpu', type=int, choices=(0,1), default=0)
    args = parser.parse_args()
    if args.mode == 'preflight':
        print(json.dumps(preflight_commands(), indent=2))
    elif args.mode == 'smoke':
        smoke(args.gpu)
    elif args.mode == 'register':
        register()
    else:
        run()


if __name__ == '__main__':
    main()
