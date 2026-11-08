"""Explicit CLI-only continuation of the failed target pilot; never restart jobs."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from run_scripts import run_hs_target_pilot as old

RESULTS = old.RESULTS / 'continuation_v2'
DOCUMENT = Path('docs/HS_TARGET_PILOT_CLI_AMENDMENT_V2_20261101.md')
TESTS = Path('tests/test_hs_target_continuation.py')
JOBS = ('endpoint', 'suite', 'matched8')


def repaired_commands(original):
    """Exactly one invalid flag/value deletion; preserve every other argv token."""
    result = {}
    for arm in old.ARMS:
        result[arm] = {kind: list(original['per_arm_commands'][arm][kind]) for kind in JOBS}
        command = result[arm]['endpoint']
        if command.count('--max-residues') != 1:
            raise ValueError('expected exactly one original endpoint max-residues flag')
        index = command.index('--max-residues')
        if command[index + 1:index + 2] != ['96']:
            raise ValueError('original endpoint max-residues must equal96')
        del command[index:index + 2]
    return result


def parse_cli(command):
    """Run the real producer parser, stopping immediately after successful parsing."""
    if command[:3] != [sys.executable, '-u', '-m']:
        raise ValueError('unexpected Python CLI prefix')
    module = importlib.import_module(command[3])
    original_parse = argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        def __init__(self, namespace):
            self.namespace = namespace

    def capture(parser, args=None, namespace=None):
        parsed = original_parse(parser, args, namespace)
        raise Parsed(parsed)

    with patch.object(sys, 'argv', [command[3], *command[4:]]), \
            patch.object(argparse.ArgumentParser, 'parse_args', capture):
        try:
            module.main()
        except Parsed as parsed:
            return vars(parsed.namespace)
    raise RuntimeError('producer did not reach its real argument parser')


def assert_endpoint_fixed96():
    tree = ast.parse(Path('scripts/eval_ca_endpoint_scores.py').read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    calls = [n for n in ast.walk(run) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'EvalProtocol']
    if len(calls) != 1:
        raise ValueError('endpoint protocol constructor changed')
    values = [k.value for k in calls[0].keywords if k.arg == 'max_residues']
    if len(values) != 1 or not isinstance(values[0], ast.Constant) or values[0].value != 96:
        raise ValueError('endpoint no longer fixes max_residues96 internally')


def validate_failure_record(arm, execution, original):
    if execution.get('arm') != arm or execution.get('status') != 'failed':
        raise ValueError('requires the preserved failed arm execution')
    jobs = execution.get('jobs', [])
    if [(j.get('kind'), j.get('returncode')) for j in jobs] != [
            ('train', 0), ('ensemble', 0), ('endpoint', 2)]:
        raise ValueError('requires exactly train0/ensemble0/endpoint2; no further jobs')
    immutable = dict(original['input_hashes'], **original['source_hashes'])
    for job in jobs:
        if job.get('command') != original['per_arm_commands'][arm][job['kind']]:
            raise ValueError('original command differs from registration')
        if not isinstance(job.get('pid'), int) or job['pid'] <= 0:
            raise ValueError('missing actual original child PID')
        if job.get('frozen_before') != immutable or job.get('frozen_after') != immutable:
            raise ValueError('original before/after immutable audit missing or different')
        if old.sha256_file(job['log']) != job.get('log_sha256'):
            raise ValueError('original log changed')
    failure = Path(jobs[-1]['log']).read_text()
    if 'unrecognized arguments: --max-residues 96' not in failure:
        raise ValueError('original failure was not the specified parser error')
    if 'output_sha256' in jobs[-1]:
        raise ValueError('failed parser must not have produced an endpoint output')
    return jobs


def validate_absent_outputs(commands):
    absent = []
    for arm in old.ARMS:
        for kind in JOBS:
            command = commands[arm][kind]
            path = Path(command[command.index('--out') + 1])
            if path != old.RESULTS / f'{arm}.{kind}.json':
                raise ValueError('continuation output differs from original registered path')
            for candidate in (path, path.with_suffix('.component.pt')):
                if candidate.exists():
                    raise FileExistsError(candidate)
                absent.append(str(candidate))
    return absent


def verify_snapshot(protocol):
    old.verify_files(dict(protocol['input_hashes'], **protocol['source_hashes']))
    for path, digest in protocol['source_hashes'].items():
        if old.sha256_file(Path(protocol['source_snapshot']) / path) != digest:
            raise ValueError('continuation source snapshot changed: ' + path)


def register():
    if RESULTS.exists():
        raise FileExistsError('continuation exists; no restart or overwrite')
    original_path = old.RESULTS / 'protocol.json'
    original = old.json.loads(original_path.read_text())
    if original.get('schema_version') != 'hs_target_pilot_v1':
        raise ValueError('unexpected original protocol')
    if old.sha256_file(old.STATISTICS) != old.STATISTICS_SHA:
        raise ValueError('scientific statistical registration changed')
    old.verify_files(dict(original['input_hashes'], **original['source_hashes']))
    for path, digest in original['source_hashes'].items():
        if old.sha256_file(Path(original['source_snapshot']) / path) != digest:
            raise ValueError('original source snapshot changed: ' + path)
    old.assert_endpoint_default()
    assert_endpoint_fixed96()
    commands = repaired_commands(original)
    absent = validate_absent_outputs(commands)
    parser_contracts = {arm: {kind: parse_cli(commands[arm][kind]) for kind in JOBS}
                        for arm in old.ARMS}
    evidence = {}; inputs = dict(original['input_hashes'])
    inputs[str(original_path)] = old.sha256_file(original_path)
    audits = {}
    for arm in old.ARMS:
        path = old.RESULTS / f'{arm}.execution.json'
        execution = old.json.loads(path.read_text())
        jobs = validate_failure_record(arm, execution, original)
        checkpoint_path = old.ARTIFACTS / arm / 'checkpoint_latest.pt'
        checkpoint_digest = old.sha256_file(checkpoint_path)
        if checkpoint_digest != execution['latest_sha256']:
            raise ValueError('original final checkpoint changed')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        audit = old.audit_checkpoint(checkpoint, arm, original['initial_component_identity'])
        if old.canonical(audit) != old.canonical(execution['checkpoint_audit']):
            raise ValueError('original recorded checkpoint audit changed')
        audits[arm] = audit
        ensemble_path = old.RESULTS / f'{arm}.ensemble.json'
        if old.sha256_file(ensemble_path) != jobs[1]['output_sha256']:
            raise ValueError('completed original ensemble changed')
        ensemble = old.json.loads(ensemble_path.read_text())
        if ensemble.get('checkpoint') != str(checkpoint_path) or ensemble.get('n_windows') != 200:
            raise ValueError('original ensemble checkpoint or panel budget differs')
        if len(ensemble['per_window']) != 200 or len({r['protein'] for r in ensemble['per_window']}) != 50:
            raise ValueError('original ensemble must retain200 windows/50 proteins')
        evidence[arm] = dict(old_execution=str(path), old_execution_sha256=old.sha256_file(path),
            checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_digest,
            checkpoint_audit=audit, ensemble=str(ensemble_path), ensemble_sha256=old.sha256_file(ensemble_path),
            successful_old_jobs=['train', 'ensemble'], preserved_failed_job=dict(kind='endpoint', returncode=2),
            endpoint_result_absent_at_registration=True)
        for item in [path, checkpoint_path, ensemble_path, ensemble_path.with_suffix('.component.pt'),
                     *[Path(j['log']) for j in jobs]]:
            inputs[str(item)] = old.sha256_file(item)
    normalized = [{k: v for k, v in audits[arm]['actual_args'].items() if k not in ('target', 'output')}
                  for arm in old.ARMS]
    if old.canonical(normalized[0]) != old.canonical(normalized[1]):
        raise ValueError('training arms differ beyond target/output')
    if torch.cuda.device_count() < 2:
        raise RuntimeError('two visible GPUs required; no device reassignment')
    sources = dict(original['source_hashes'])
    for path in [str(Path(__file__).resolve().relative_to(Path.cwd().resolve())), str(DOCUMENT), str(TESTS)]:
        sources[path] = old.sha256_file(path)
    RESULTS.mkdir()
    snapshot = RESULTS / 'source_snapshot'; snapshot.mkdir()
    for path, digest in sources.items():
        destination = snapshot / path; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(Path(path).read_bytes())
        if old.sha256_file(destination) != digest:
            raise ValueError('source changed while snapshotting: ' + path)
    protocol = dict(schema_version='hs_target_pilot_cli_continuation_v2', registered_utc=old.now(),
        original_protocol=str(original_path), original_protocol_sha256=inputs[str(original_path)],
        statistics_protocol=str(old.STATISTICS), statistics_protocol_sha256=old.STATISTICS_SHA,
        reason='Original endpoint argparse rejected unsupported --max-residues96 before run(); exit2.',
        sole_amendment='Remove exactly --max-residues and96 from endpoint argv; frozen run() fixes96.',
        scientific_protocol_unchanged=True, checkpoints_unchanged=True, optimizer_steps=0,
        arms=old.ARMS, jobs=list(JOBS), per_arm_commands=commands, cli_parser_contracts=parser_contracts,
        original_evidence=evidence, outputs_verified_absent_before_registration=absent,
        logical_job_audit={arm: {kind: str((old.RESULTS if kind in ('train', 'ensemble') else RESULTS)
                                          / f'{arm}.execution.json') for kind in old.JOBS} for arm in old.ARMS},
        source_hashes=sources, input_hashes=inputs, source_snapshot=str(snapshot),
        test_opened=False, bioemu2_rerun=False, automatic_restart=False)
    verify_snapshot(protocol)
    old.write_new(RESULTS / 'protocol.json', protocol)
    return protocol


def run_arm(arm, protocol):
    path = RESULTS / f'{arm}.execution.json'
    status = dict(arm=arm, status='running', started_utc=old.now(), runner_pid=os.getpid(), jobs=[],
                  original_execution=protocol['original_evidence'][arm]['old_execution'],
                  original_execution_sha256=protocol['original_evidence'][arm]['old_execution_sha256'],
                  latest_sha256=protocol['original_evidence'][arm]['checkpoint_sha256'])
    old.write_new(path, status)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(old.ARMS[arm]['gpu']),
        PYTHONPATH='/tmp/lasp_stability_deps:.', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    immutable = dict(protocol['input_hashes'], **protocol['source_hashes'])
    try:
        for kind in JOBS:
            command = protocol['per_arm_commands'][arm][kind]
            output = Path(command[command.index('--out') + 1])
            if output.exists() or output.with_suffix('.component.pt').exists():
                raise FileExistsError(output)
            verify_snapshot(protocol)
            entry = dict(kind=kind, command=command, started_utc=old.now(),
                         log=str(RESULTS / f'{arm}.{kind}.log'), frozen_before=old.verify_files(immutable))
            status['jobs'].append(entry)
            started = time.monotonic()
            with Path(entry['log']).open('x') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
                entry['pid'] = process.pid
                old.update_status(path, status)
                code = process.wait()
            entry.update(returncode=code, completed_utc=old.now(), seconds=time.monotonic() - started,
                         log_sha256=old.sha256_file(entry['log']))
            old.update_status(path, status)
            entry['frozen_after'] = old.verify_files(immutable)
            verify_snapshot(protocol)
            if code:
                raise RuntimeError(f'{arm}/{kind} exited {code}; no automatic restart')
            entry['output_sha256'] = old.sha256_file(output)
            sidecar = output.with_suffix('.component.pt')
            if sidecar.exists():
                entry['component_sha256'] = old.sha256_file(sidecar)
            old.update_status(path, status)
            print(f'{arm}/{kind} completed exit0', flush=True)
        status.update(status='completed', completed_utc=old.now(), frozen_inputs_verified=True)
        return arm
    except BaseException as error:
        status.update(status='failed', failed_utc=old.now(), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        old.update_status(path, status)


def main():
    protocol = register()
    failed = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {pool.submit(run_arm, arm, protocol): arm for arm in old.ARMS}
        for future in as_completed(tasks):
            try:
                print(f'{future.result()} three continuation jobs complete', flush=True)
            except Exception as error:
                failed.append(tasks[future]); print(f'{tasks[future]} failed: {error}', flush=True)
    if failed:
        raise RuntimeError('failed continuation arms (no restart): ' + ', '.join(failed))


if __name__ == '__main__':
    main()
