"""Registered four-arm H-readout retention experiment; no automatic restarts.

Run from the repository root. Every arm uses the same source weights, data
schedule and 500 steps. The fixed latest checkpoint is evaluated, regardless
of the arm's regularized validation minimum. Refuses existing run outputs.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from evaluation.loaders import sha256_file
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.eval_ca_endpoint_scores import SOURCES


ARTIFACTS = Path('artifacts/hs_joint_20261101/retention_v1')
RESULTS = Path('results/hs_joint_20261101/retention_v1')
INITIAL = Path('artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt')
INITIAL_SHA = '396e6e4df35019140324484bf8d560a292cf2688a11946f2e0ccde5cd6ea8dda'
ARMS = {
    'frozen': dict(gpu=0, train_encoder=False, teacher_weight=1., teacher_delta_weight=.1, observed_readout_weight=0.),
    'joint_teacher': dict(gpu=0, train_encoder=True, teacher_weight=1., teacher_delta_weight=.1, observed_readout_weight=0.),
    'joint_observed': dict(gpu=1, train_encoder=True, teacher_weight=1., teacher_delta_weight=.1, observed_readout_weight=.1),
    'joint_strong_teacher': dict(gpu=1, train_encoder=True, teacher_weight=10., teacher_delta_weight=1., observed_readout_weight=0.),
}
COMMON = ['--component', str(INITIAL), '--init-kernel', str(INITIAL),
    '--s0-mode', 'exact_spatial', '--noise-kernel', 'gaussian', '--noise-mix', '.8',
    '--noise-length', '8', '--noise-cutoff', '10', '--collective-mobility', '--history-conditioning',
    '--steps', '500', '--stateless-data', '--train-windows', '0', '--cosine', '--warmup', '50',
    '--val-windows', '16', '--val-every', '100', '--n-samples', '12', '--grad-steps', '64',
    '--target', 'path_mixture', '--max-residues', '96', '--lr', '.0001', '--encoder-lr-scale', '.1',
    '--latent-weight', '.1', '--contact-objective', 'brier', '--contact-weight', '20',
    '--rmsf-weight', '1', '--clash-weight', '40', '--mode-weight', '2', '--seed', '42']


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def register():
    if ARTIFACTS.exists() or (RESULTS / 'protocol.json').exists():
        raise FileExistsError('Existing experiment; inspect live processes/results before any resume.')
    assert sha256_file(INITIAL) == INITIAL_SHA
    ARTIFACTS.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(INITIAL, map_location='cpu', weights_only=False)
    sources = tuple(dict.fromkeys((*SOURCES, 'scripts/eval_lasp_ca_dynamics.py',
        'scripts/diagnose_ca_readout_retention.py', 'protein_jepa_frame/training/lasp_run_state.py',
        'protein_jepa_frame/training/train_utils.py', 'protein_jepa_frame/data/collate.py',
        str(Path(__file__).resolve().relative_to(Path.cwd())))))
    source_hashes = {p: sha256_file(p) for p in sources}
    for path in sources:
        destination = ARTIFACTS / 'source_snapshot' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    inputs = [str(INITIAL), 'artifacts/mdcath_manifest_full/train.jsonl',
              'artifacts/mdcath_manifest_full/val.jsonl', 'results/lasp_v19/bioemu2_complete.json',
              'results/hs_joint_20261101/readout_diagnosis/train64_frozen_vs_joint_best.json']
    protocol = dict(schema_version=1, registered_utc=now(), role='adaptive_development_ablation',
        purpose='test whether observed-frame structural supervision improves H readout and autonomous forecasts',
        initial_checkpoint=str(INITIAL), source_training_step=checkpoint['probe_provenance']['source_training_step'],
        source_tensor_hashes=dict(kernel=state_sha256(checkpoint['kernel']),
            encoder=state_sha256(checkpoint['component_checkpoint']['encoder']),
            decoder=state_sha256(checkpoint['component_checkpoint']['delta_ca_decoder'])),
        arms=ARMS, common_training_arguments=COMMON, optimizer_steps_per_arm=500,
        weight_choice='fixed observed weight0.1 based only on existing train32 gradient scale; '
            'strong teacher10/1 is an alternative control, not gradient matched (teacher gradient is zero initially)',
        contrasts=['joint_observed vs joint_teacher', 'joint_observed vs joint_strong_teacher',
                   'joint_teacher vs frozen', 'joint_observed vs frozen', 'joint_strong_teacher vs frozen'],
        training_task='endpoint draws fitted to future-path mixture excluding anchor; retained for this ablation',
        evaluation=dict(checkpoint='latest at exactly500 steps in every arm; no best/checkpoint selection',
            n_windows=200, n_proteins=50, n_samples=12, seed=0, horizons=[1,4,16,64],
            seven_metric_reference='future-path mixture including anchor, unchanged historical protocol',
            endpoint_reference='single matched future endpoint; three scores and canonical CA geometry',
            readout_gate='true future H oracle on same200 development windows; no forecast claim',
            validation_loss='compare common forecast loss after subtracting all teacher/readout penalties; raw totals differ'),
        uncertainty='all registered contrasts, horizons and temperatures; equal-protein paired10000-bootstrap '
            'marginal CI95, exploratory without multiplicity/training-seed/adaptive-selection correction',
        continuation_rule='No coefficient sweep after a failed screen. Retention is not a success unless long-horizon '
            'readout improves without reliably worsening other readout/forecast/geometry outcomes; all-horizon '
            'oracle ratio<=1/3 remains unmet until measured. Passing this screen never establishes full BioEmu2 superiority.',
        scientific_limits=['single training seed; four-arm comparison does not exclude every form of extra regularization',
            'observed readout MSE is structural supervision, not merely teacher distillation',
            'path-mixture training does not certify conditional endpoint dynamics; a later target ablation must be separate',
            'T1/T2A/T2B/order/T3/192ns and independent confirmation remain required for the full goal'],
        test_opened=False, bioemu2_rerun=False,
        input_hashes={p: sha256_file(p) for p in inputs}, source_hashes=source_hashes,
        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    write(RESULTS / 'protocol.json', protocol)
    return protocol


def run_arm(name, settings, protocol):
    output = ARTIFACTS / name
    status_path = RESULTS / f'{name}.execution.json'
    status = dict(arm=name, started_utc=now(), jobs=[])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(settings['gpu']),
                       PYTHONPATH='/tmp/lasp_stability_deps:.', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    arguments = [*COMMON, '--output', str(output)]
    for key in ('teacher_weight', 'teacher_delta_weight', 'observed_readout_weight'):
        arguments += ['--' + key.replace('_', '-'), str(settings[key])]
    if settings['train_encoder']:
        arguments.append('--train-encoder')
    latest = output / 'checkpoint_latest.pt'
    jobs = [('train', [sys.executable, '-u', '-m', 'protein_jepa_frame.training.train_lasp_ca_dynamics', *arguments]),
        ('ensemble', [sys.executable, '-u', '-m', 'scripts.eval_lasp_ca_dynamics', '--checkpoint', str(latest),
                      '--out', str(RESULTS / f'{name}.ensemble.json'), '--max-windows', '200']),
        ('endpoint', [sys.executable, '-u', '-m', 'scripts.eval_ca_endpoint_scores', '--checkpoint', str(latest),
                      '--out', str(RESULTS / f'{name}.endpoint.json'), '--max-windows', '200']),
        ('readout', [sys.executable, '-u', '-m', 'scripts.gate_ca_representation', '--component', str(latest),
                     '--out', str(RESULTS / f'{name}.h_gate.json'), '--max-windows', '200'])]
    for kind, command in jobs:
        for path, expected in {**protocol['input_hashes'], **protocol['source_hashes']}.items():
            assert sha256_file(path) == expected, f'Input changed: {path}'
        if kind != 'train':
            assert sha256_file(latest) == status['latest_sha256'], f'Checkpoint changed before {kind}'
        log = RESULTS / f'{name}.{kind}.log'
        entry = dict(kind=kind, command=command, log=str(log), started_utc=now())
        status['jobs'].append(entry)
        started = time.monotonic()
        with log.open('x') as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            write(status_path, status)
            result = process.wait()
        entry.update(returncode=result, seconds=time.monotonic()-started, completed_utc=now())
        write(status_path, status)
        if result:
            raise RuntimeError(f'{name}/{kind} failed with exit{result}; inspect {log}; no automatic restart')
        if kind == 'train':
            saved = torch.load(latest, map_location='cpu', weights_only=False)
            assert saved['step'] == 500
            status['latest_sha256'] = sha256_file(latest)
        elif kind != 'train':
            assert sha256_file(latest) == status['latest_sha256'], f'Checkpoint changed during {kind}'
            artifact = RESULTS / f'{name}.{dict(ensemble="ensemble", endpoint="endpoint", readout="h_gate")[kind]}.json'
            entry['output_sha256'] = sha256_file(artifact)
        write(status_path, status)
        print(f'{name} {kind} completed', flush=True)
    for path, expected in {**protocol['input_hashes'], **protocol['source_hashes']}.items():
        assert sha256_file(path) == expected, f'Input changed: {path}'
    status.update(completed_utc=now(), frozen_inputs_verified=True)
    write(status_path, status)
    return name


def main():
    protocol = register()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_arm, name, settings, protocol) for name, settings in ARMS.items()]
        for future in as_completed(futures):
            print(f'{future.result()} all tasks complete', flush=True)


if __name__ == '__main__':
    main()
