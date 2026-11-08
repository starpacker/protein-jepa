from copy import deepcopy
from pathlib import Path

import pytest

from run_scripts import continue_hs_target_pilot_v2 as continuation


def original():
    return dict(per_arm_commands={arm: continuation.old.commands(arm) for arm in continuation.old.ARMS})


@pytest.mark.parametrize('arm', list(continuation.old.ARMS))
@pytest.mark.parametrize('kind', ['endpoint', 'suite', 'matched8'])
def test_real_producer_parser_for_every_continuation_command(arm, kind):
    command = continuation.repaired_commands(original())[arm][kind]
    args = continuation.parse_cli(command)
    assert args['seed'] == 0 and args['n_samples'] == 12
    assert args['checkpoint'].endswith(f'/{arm}/checkpoint_latest.pt')
    if kind == 'endpoint':
        assert args['max_windows'] == 200
        assert 'max_residues' not in args
        continuation.assert_endpoint_fixed96()
    else:
        assert args['max_residues'] == 96 and args['context_frames'] == 8
        assert args['horizons'] == [1, 4, 16, 64]
        assert args['temps'] == [320., 348., 379., 413., 450.]
        assert args['t2a'] is True
        assert (args['proteins_per_temp'], args['anchors_per_trajectory'], args['modes']) == (
            (12, 2, ['correct', 'wrong', 'none']) if kind == 'suite' else (8, 1, ['correct']))


@pytest.mark.parametrize('arm', list(continuation.old.ARMS))
def test_old_endpoint_command_really_fails_parser(arm):
    with pytest.raises(SystemExit) as error:
        continuation.parse_cli(original()['per_arm_commands'][arm]['endpoint'])
    assert error.value.code == 2


def test_sole_amendment_and_no_mutation():
    initial = original(); before = deepcopy(initial)
    repaired = continuation.repaired_commands(initial)
    assert initial == before
    for arm in continuation.old.ARMS:
        prior = initial['per_arm_commands'][arm]['endpoint']; index = prior.index('--max-residues')
        assert repaired[arm]['endpoint'] == prior[:index] + prior[index + 2:]
        for kind in ('suite', 'matched8'):
            assert repaired[arm][kind] == initial['per_arm_commands'][arm][kind]
    assert continuation.JOBS == ('endpoint', 'suite', 'matched8')


@pytest.mark.parametrize('mutation', ['missing', 'different', 'duplicate'])
def test_repair_rejects_unexpected_old_flag(mutation):
    initial = original(); command = initial['per_arm_commands']['endpoint']['endpoint']
    index = command.index('--max-residues')
    if mutation == 'missing':
        del command[index:index + 2]
    elif mutation == 'different':
        command[index + 1] = '64'
    else:
        command.extend(['--max-residues', '96'])
    with pytest.raises(ValueError):
        continuation.repaired_commands(initial)


@pytest.mark.parametrize('suffix', ['.json', '.component.pt'])
def test_refuses_result_or_sidecar_overwrite(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(continuation.old, 'RESULTS', tmp_path)
    commands = continuation.repaired_commands(original())
    assert len(continuation.validate_absent_outputs(commands)) == 12
    (tmp_path / ('endpoint.endpoint' + suffix)).write_text('preserved')
    with pytest.raises(FileExistsError):
        continuation.validate_absent_outputs(commands)
    assert (tmp_path / ('endpoint.endpoint' + suffix)).read_text() == 'preserved'


def failure_fixture(tmp_path):
    initial = original(); initial.update(input_hashes={}, source_hashes={})
    jobs = []
    for index, (kind, code) in enumerate([('train', 0), ('ensemble', 0), ('endpoint', 2)]):
        log = tmp_path / f'{kind}.log'
        log.write_text('unrecognized arguments: --max-residues 96' if code else 'completed')
        jobs.append(dict(kind=kind, returncode=code, pid=index + 100,
                         command=initial['per_arm_commands']['endpoint'][kind],
                         frozen_before={}, frozen_after={}, log=str(log),
                         log_sha256=continuation.old.sha256_file(log)))
    return initial, dict(arm='endpoint', status='failed', jobs=jobs)


def test_exact_original_failure_and_hash_required(tmp_path):
    initial, execution = failure_fixture(tmp_path)
    assert len(continuation.validate_failure_record('endpoint', execution, initial)) == 3
    Path(execution['jobs'][0]['log']).write_text('tampered')
    with pytest.raises(ValueError, match='log changed'):
        continuation.validate_failure_record('endpoint', execution, initial)


@pytest.mark.parametrize('mutation', ['bad_exit', 'extra_job', 'missing_pid', 'new_argv', 'produced_output', 'bad_audit'])
def test_rejects_unexpected_failure_evidence(tmp_path, mutation):
    initial, execution = failure_fixture(tmp_path)
    if mutation == 'bad_exit':
        execution['jobs'][2]['returncode'] = 1
    elif mutation == 'extra_job':
        execution['jobs'].append(dict(kind='suite', returncode=0))
    elif mutation == 'missing_pid':
        del execution['jobs'][1]['pid']
    elif mutation == 'new_argv':
        execution['jobs'][1]['command'] = ['different']
    elif mutation == 'produced_output':
        execution['jobs'][2]['output_sha256'] = 'unexpected'
    else:
        execution['jobs'][0]['frozen_after'] = {'changed': 'digest'}
    with pytest.raises(ValueError):
        continuation.validate_failure_record('endpoint', execution, initial)
