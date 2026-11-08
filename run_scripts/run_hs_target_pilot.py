"""Registered, matched endpoint-versus-path-target pilot; no automatic restart.

Run only after review. Two GPUs run one arm each. Frozen producers are invoked
unchanged; all outputs live in new target_v1 directories. Every subprocess PID,
exit status and immutable input/source digest is retained, including failures.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

from evaluation.loaders import sha256_file
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.eval_ca_endpoint_scores import SOURCES
# Import the actual suite dependencies before collecting loaded source paths.
from scripts import eval_lasp_ca_suite  # noqa: F401


ARTIFACTS = Path('artifacts/hs_joint_20261101/target_v1')
RESULTS = Path('results/hs_joint_20261101/target_v1')
INITIAL = Path('artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt')
INITIAL_SHA = '396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda'
BASE_GATE = Path('results/hs_joint_20261101/retention_v1/frozen.h_gate.json')
BASE_GATE_SHA = '25f0aa44148aa0539e7cb76352a9bdc891a0fdc85a08bac3230f1c41332c8f4b'
STATISTICS = RESULTS / 'summary.protocol.json'
STATISTICS_SHA = 'ca4a4c0b80dea75a0569ce5d3345ac3440e475acead93de50ee1f02379f9d5c4'
DOCUMENT = Path('docs/HS_TARGET_PILOT_V1_20261101.md')
BIOEMU = Path('results/lasp_v19/bioemu2_complete.json')
ARMS = {'endpoint': dict(gpu=0, target='endpoint'),
        'path_mixture': dict(gpu=1, target='path_mixture')}
JOBS = ('train', 'ensemble', 'endpoint', 'suite', 'matched8')
COMMON = ['--component', str(INITIAL), '--init-kernel', str(INITIAL),
    '--s0-mode', 'exact_spatial', '--noise-kernel', 'gaussian', '--noise-mix', '.8',
    '--noise-length', '8', '--noise-cutoff', '10', '--collective-mobility', '--history-conditioning',
    '--coupling-scale', '.1', '--gamma-scale', '.00002', '--gamma-floor', '.000002',
    '--steps', '500', '--stateless-data', '--train-windows', '0', '--cosine', '--warmup', '50',
    '--val-windows', '16', '--val-every', '100', '--n-samples', '12', '--grad-steps', '64',
    '--horizons', '1', '4', '16', '64', '--max-residues', '96', '--lr', '.0001',
    '--latent-weight', '.1', '--contact-objective', 'brier', '--contact-weight', '20',
    '--clash-weight', '40', '--distance-weight', '0', '--rmsf-weight', '0',
    '--mode-weight', '0', '--mode-distribution-weight', '0', '--observed-readout-weight', '0',
    '--seed', '42', '--manifest', 'artifacts/mdcath_manifest_full/train.jsonl',
    '--val-manifest', 'artifacts/mdcath_manifest_full/val.jsonl']
EXPECTED = dict(steps=500, stateless_data=True, train_windows=0, cosine=True, warmup=50,
    val_windows=16, val_every=100, n_samples=12, grad_steps=64, horizons=[1, 4, 16, 64],
    max_residues=96, lr=.0001, latent_weight=.1, contact_objective='brier', contact_weight=20.,
    clash_weight=40., distance_weight=0., rmsf_weight=0., mode_weight=0., mode_distribution_weight=0.,
    observed_readout_weight=0., seed=42, train_encoder=False, resume=False,
    s0_mode='exact_spatial', noise_kernel='gaussian', noise_mix=.8, noise_length=8., noise_cutoff=10.,
    collective_mobility=True, history_conditioning=True, coupling_scale=.1,
    gamma_scale=.00002, gamma_floor=.000002)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def update_status(path, value):
    # Only execution files created by this invocation are updated.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = stream.name
    os.replace(temporary, path)


def component_identity(component):
    configuration = {key: component[key] for key in
                     ('checkpoint_kind', 'decoder_mode', 'calibration_args')}
    return dict(encoder=state_sha256(component['encoder']),
                decoder=state_sha256(component['delta_ca_decoder']),
                configuration_sha256=hashlib.sha256(canonical(configuration).encode()).hexdigest(),
                configuration=configuration,
                projection_steps=component['calibration_args']['bond_projection_steps'])


def assert_endpoint_default(source=None):
    """Pin the frozen training sampler's implicit endpoint observation contract."""
    source = source or Path('protein_jepa_frame/training/train_lasp_ca_dynamics.py').read_text()
    tree = ast.parse(source)
    defaults = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'getattr'
                and len(node.args) == 3 and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 'prediction_kind']
    if (len(defaults) != 1 or not isinstance(defaults[0].args[2], ast.Constant)
            or defaults[0].args[2].value != 'endpoint'
            or any(isinstance(node, ast.Attribute) and node.attr == 'prediction_kind'
                   and isinstance(node.ctx, ast.Store) for node in ast.walk(tree))):
        raise ValueError('training prediction_kind endpoint default changed')


def commands(arm):
    if arm not in ARMS:
        raise ValueError('unregistered arm')
    latest = str(ARTIFACTS / arm / 'checkpoint_latest.pt')
    common_eval = ['--checkpoint', latest, '--n-samples', '12', '--max-residues', '96', '--seed', '0']
    suite = [sys.executable, '-u', '-m', 'scripts.eval_lasp_ca_suite', *common_eval,
             '--manifest', 'artifacts/mdcath_manifest_full/val.jsonl', '--t2a',
             '--context-frames', '8', '--horizons', '1', '4', '16', '64',
             '--temps', '320', '348', '379', '413', '450']
    return dict(
        train=[sys.executable, '-u', '-m', 'protein_jepa_frame.training.train_lasp_ca_dynamics',
               *COMMON, '--target', arm, '--output', str(ARTIFACTS / arm)],
        ensemble=[sys.executable, '-u', '-m', 'scripts.eval_lasp_ca_dynamics', *common_eval,
                  '--max-windows', '200', '--prediction-kind', 'endpoint',
                  '--horizons', '1', '4', '16', '64', '--out', str(RESULTS / f'{arm}.ensemble.json')],
        endpoint=[sys.executable, '-u', '-m', 'scripts.eval_ca_endpoint_scores', *common_eval,
                  '--max-windows', '200', '--out', str(RESULTS / f'{arm}.endpoint.json')],
        suite=[*suite, '--modes', 'correct', 'wrong', 'none', '--wrong-temp', '320',
               '--proteins-per-temp', '12', '--anchors-per-trajectory', '2',
               '--out', str(RESULTS / f'{arm}.suite.json')],
        matched8=[*suite, '--modes', 'correct', '--proteins-per-temp', '8',
                  '--anchors-per-trajectory', '1', '--out', str(RESULTS / f'{arm}.matched8.json')])


def sources():
    root = Path.cwd().resolve()
    paths = set(SOURCES) | {str(DOCUMENT), str(Path(__file__).resolve().relative_to(root)),
        'tests/test_hs_target_pilot.py', 'scripts/eval_lasp_ca_dynamics.py',
        'scripts/eval_lasp_ca_suite.py', 'evaluation/t2a_static.py',
        'protein_jepa_frame/training/lasp_run_state.py', 'protein_jepa_frame/data/collate.py',
        'protein_jepa_frame/data/manifest.py', 'protein_jepa_frame/training/train_utils.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == '.py' and path.is_file() and path.is_relative_to(root):
            paths.add(str(path.relative_to(root)))
    return sorted(paths)


def verify_files(expected):
    actual = {str(path): sha256_file(path) for path in expected}
    if actual != expected:
        raise ValueError('immutable input/source changed: ' + ', '.join(
            path for path in expected if actual[path] != expected[path]))
    return actual


def audit_checkpoint(checkpoint, arm, initial_identity):
    if checkpoint.get('checkpoint_kind') != 'lasp_ca_dynamics' or checkpoint.get('step') != 500:
        raise ValueError('requires the registered latest step500 checkpoint')
    args = checkpoint['args']
    for key, expected in dict(EXPECTED, target=arm, output=str(ARTIFACTS / arm),
                              component=str(INITIAL), init_kernel=str(INITIAL)).items():
        if args.get(key) != expected:
            raise ValueError(f'checkpoint registered argument differs: {key}')
    identity = component_identity(checkpoint['component_checkpoint'])
    if identity != initial_identity or checkpoint.get('target_encoder') is not None:
        raise ValueError('frozen encoder/decoder/calibration contract changed')
    return dict(step=500, actual_args=args, kernel_config=checkpoint['kernel_config'],
                kernel_tensor_sha256=state_sha256(checkpoint['kernel']),
                component_identity=identity, target_encoder_is_none=True,
                frozen_component_verified=True, h_gate_reused=str(BASE_GATE),
                h_gate_sha256=BASE_GATE_SHA)


def register():
    if ARTIFACTS.exists() or (RESULTS.exists() and
            {p.name for p in RESULTS.iterdir()} != {'summary.protocol.json'}):
        raise FileExistsError('Existing experiment/output; no automatic restart or overwrite')
    for path, expected in [(INITIAL, INITIAL_SHA), (BASE_GATE, BASE_GATE_SHA),
                           (STATISTICS, STATISTICS_SHA)]:
        if sha256_file(path) != expected:
            raise ValueError(f'registered input changed: {path}')
    assert_endpoint_default()
    initial = torch.load(INITIAL, map_location='cpu', weights_only=False)
    identity = component_identity(initial['component_checkpoint'])
    gate = json.loads(BASE_GATE.read_text())
    gate_checkpoint = Path(gate['checkpoint'])
    if sha256_file(gate_checkpoint) != gate['checkpoint_sha256']:
        raise ValueError('baseline H-gate checkpoint changed')
    gate_source = torch.load(gate_checkpoint, map_location='cpu', weights_only=False)
    if (component_identity(gate_source['component_checkpoint']) != identity
            or gate['encoder_tensor_sha256'] != identity['encoder']
            or gate['decoder_tensor_sha256'] != identity['decoder']):
        raise ValueError('H-gate reuse requires identical full component and configuration')
    if torch.cuda.device_count() < 2:
        raise RuntimeError('two visible GPUs required; no implicit device reassignment')
    source_hashes = {path: sha256_file(path) for path in sources()}
    ca = initial['component_checkpoint']['calibration_args']
    inputs = [INITIAL, BASE_GATE, gate_checkpoint, STATISTICS, BIOEMU,
        Path('artifacts/mdcath_manifest_full/train.jsonl'), Path('artifacts/mdcath_manifest_full/val.jsonl'),
        Path(ca['stage_a_ckpt']), Path(ca['b1a_ckpt'])]
    input_hashes = {str(path): sha256_file(path) for path in inputs}
    ARTIFACTS.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    snapshot = ARTIFACTS / 'source_snapshot'
    snapshot.mkdir()
    for path in source_hashes:
        destination = snapshot / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(Path(path).read_bytes())
        if sha256_file(destination) != source_hashes[path]:
            raise ValueError('source changed during snapshot: ' + path)
    protocol = dict(schema_version='hs_target_pilot_v1', registered_utc=now(),
        role='adaptive_development_matched_target_ablation', arms=ARMS,
        initial_checkpoint=str(INITIAL), source_training_step=initial['probe_provenance']['source_training_step'],
        initial_component_identity=identity, initial_kernel_sha256=state_sha256(initial['kernel']),
        optimizer_steps_per_arm=500, prediction_kind='endpoint', common_training_arguments=COMMON,
        per_arm_commands={arm: commands(arm) for arm in ARMS}, expected_checkpoint_arguments=EXPECTED,
        checkpoint_selection='latest exactly500, no early stopping or validation selection',
        statistics_protocol=str(STATISTICS), statistics_protocol_sha256=STATISTICS_SHA,
        primary='equal-protein mean of h16/h64 coordinate U-ES, endpoint minus path_mixture',
        h_gate_reuse=dict(path=str(BASE_GATE), sha256=BASE_GATE_SHA, full_component_identity=identity,
            reason='encoder/CA decoder/projection/calibration unchanged; long16/64 failures remain'),
        evaluation=dict(n_windows=200, n_proteins=50, n_samples=12, seed=0, horizons=[1, 4, 16, 64],
            suite=dict(expected_windows=7200, proteins_per_temp=12, replicas=5, anchors=2,
                       modes=['correct', 'wrong', 'none'], t2a=True),
            matched8=dict(expected_windows=800, proteins_per_temp=8, replicas=5, anchors=1,
                          modes=['correct'], t2a=True)),
        source_hashes=source_hashes, input_hashes=input_hashes, source_snapshot=str(snapshot),
        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                         gpu_names=[torch.cuda.get_device_name(i) for i in range(2)]),
        limits=['old endpoint experiments exist; novelty is this matched exact_spatial contrast',
                'single train seed and adaptively reused validation; no seed/multiplicity adjustment',
                'endpoint label versus empirical path mean changes target-estimator variance',
                'coordinate/latent ES plus soft-contact Brier and clash is not a pure proper-score objective',
                'H gate cannot improve with fixed components; no memory or full-goal claim',
                'historical seven metrics remain future-path-mixture scores; endpoint is separate',
                'suite T2A/T3 does not establish T2B/MSM or complete T1/192ns performance'],
        test_opened=False, bioemu2_rerun=False)
    verify_files(input_hashes)
    verify_files(source_hashes)
    write_new(RESULTS / 'protocol.json', protocol)
    return protocol


def run_arm(arm, protocol):
    path = RESULTS / f'{arm}.execution.json'
    status = dict(arm=arm, status='running', started_utc=now(), runner_pid=os.getpid(), jobs=[])
    write_new(path, status)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(ARMS[arm]['gpu']),
        PYTHONPATH='/tmp/lasp_stability_deps:.', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    immutable = dict(protocol['input_hashes'], **protocol['source_hashes'])
    latest = ARTIFACTS / arm / 'checkpoint_latest.pt'
    try:
        for kind in JOBS:
            command = protocol['per_arm_commands'][arm][kind]
            entry = dict(kind=kind, command=command, started_utc=now(),
                         log=str(RESULTS / f'{arm}.{kind}.log'))
            status['jobs'].append(entry)
            entry['frozen_before'] = verify_files(immutable)
            if kind != 'train':
                if sha256_file(latest) != status['latest_sha256']:
                    raise ValueError('checkpoint changed before evaluation')
                output = RESULTS / f'{arm}.{kind}.json'
                if output.exists() or output.with_suffix('.component.pt').exists():
                    raise FileExistsError(output)
            started = time.monotonic()
            with Path(entry['log']).open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                entry['pid'] = process.pid
                update_status(path, status)
                code = process.wait()
            entry.update(returncode=code, completed_utc=now(), seconds=time.monotonic() - started,
                         log_sha256=sha256_file(entry['log']))
            update_status(path, status)
            entry['frozen_after'] = verify_files(immutable)
            if code:
                raise RuntimeError(f'{arm}/{kind} exited {code}; no automatic restart')
            if kind == 'train':
                saved = torch.load(latest, map_location='cpu', weights_only=False)
                status['checkpoint_audit'] = audit_checkpoint(saved, arm, protocol['initial_component_identity'])
                status['latest_sha256'] = sha256_file(latest)
            else:
                if sha256_file(latest) != status['latest_sha256']:
                    raise ValueError('checkpoint changed during evaluation')
                entry['output_sha256'] = sha256_file(output)
            update_status(path, status)
            print(f'{arm}/{kind} completed exit0', flush=True)
        status.update(status='completed', completed_utc=now(), frozen_inputs_verified=True)
        return arm
    except BaseException as error:
        status.update(status='failed', failed_utc=now(), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        update_status(path, status)


def main():
    protocol = register()
    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(run_arm, arm, protocol): arm for arm in ARMS}
        for future in as_completed(tasks):
            try:
                print(f'{future.result()} all five jobs complete', flush=True)
            except Exception as error:
                failed.append(tasks[future])
                print(f'{tasks[future]} failed: {error}', flush=True)
    if failed:
        raise RuntimeError('failed arms (no restart): ' + ', '.join(failed))


if __name__ == '__main__':
    main()
