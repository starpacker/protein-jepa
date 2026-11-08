import copy
import json

import pytest

from scripts import summarize_hs_target_pilot_v2 as summary


def fixture_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(summary.v1,'REPO',tmp_path)
    base=tmp_path/summary.v1.BASE;base.mkdir(parents=True)
    artifact=tmp_path/summary.v1.ARTIFACTS/'endpoint';artifact.mkdir(parents=True)
    checkpoint=artifact/'checkpoint_latest.pt';checkpoint.write_bytes(b'immutable-checkpoint')
    input_file=tmp_path/'input.txt';input_file.write_text('input')
    source=tmp_path/'source.py';source.write_text('source')
    immutable={'input.txt':summary.v1.sha(input_file),'source.py':summary.v1.sha(source)}
    commands={kind:['python','-m',kind,'--checkpoint',str(summary.v1.ARTIFACTS/'endpoint'/'checkpoint_latest.pt'),
        '--max-residues','96','--out',str(summary.v1.BASE/f'endpoint.{kind}.json')]
        for kind in ('train','ensemble','endpoint','suite','matched8')}
    original=dict(input_hashes={'input.txt':immutable['input.txt']},source_hashes={'source.py':immutable['source.py']},
        per_arm_commands={'endpoint':commands})
    jobs=[]
    for kind,code in zip(('train','ensemble','endpoint'),(0,0,2)):
        log=base/f'endpoint.{kind}.log'
        log.write_text('done' if code==0 else
            'eval_ca_endpoint_scores.py: error: unrecognized arguments: --max-residues 96\n')
        job=dict(kind=kind,returncode=code,completed_utc='2026-11-01T00:00:00+00:00',pid=100+len(jobs),
            command=commands[kind],frozen_before=immutable.copy(),frozen_after=immutable.copy(),
            log=str(summary.v1.BASE/f'endpoint.{kind}.log'),log_sha256=summary.v1.sha(log))
        if kind=='ensemble':
            result=base/'endpoint.ensemble.json';result.write_text('{}')
            job['output_sha256']=summary.v1.sha(result)
        jobs.append(job)
    execution=dict(arm='endpoint',status='failed',failed_utc='2026-11-01T00:00:00+00:00',
        error='RuntimeError: endpoint/endpoint exited 2; no automatic restart',jobs=jobs,
        latest_sha256=summary.v1.sha(checkpoint),checkpoint_audit={'step':500})
    file=base/'endpoint.execution.json';file.write_text(json.dumps(execution))
    return original,execution,file


def test_command_repair_deletes_only_endpoint_unsupported_option(tmp_path,monkeypatch):
    original,_,_=fixture_attempt(tmp_path,monkeypatch)
    before=copy.deepcopy(original)
    repaired=summary.repaired_commands(original,'endpoint')
    assert original==before
    assert repaired['suite']==original['per_arm_commands']['endpoint']['suite']
    assert repaired['matched8']==original['per_arm_commands']['endpoint']['matched8']
    expected=original['per_arm_commands']['endpoint']['endpoint'].copy()
    i=expected.index('--max-residues');del expected[i:i+2]
    assert repaired['endpoint']==expected
    assert '--max-residues' not in repaired['endpoint']


@pytest.mark.parametrize('replacement',[[],['--max-residues','95'],['--max-residues','96','--max-residues','96']])
def test_repair_rejects_missing_different_or_duplicate_flag(tmp_path,monkeypatch,replacement):
    original,_,_=fixture_attempt(tmp_path,monkeypatch)
    original['per_arm_commands']['endpoint']['endpoint']=['python',*replacement]
    with pytest.raises(ValueError,match='flag|budget'):
        summary.repaired_commands(original,'endpoint')


def test_failed_attempt_is_preserved_not_normalized_to_success(tmp_path,monkeypatch):
    original,execution,file=fixture_attempt(tmp_path,monkeypatch)
    digest=summary.v1.sha(file)
    inputs=summary.v1.Inputs()
    result=summary.audit_failed_attempt('endpoint',original,inputs,digest)
    assert result==execution and result['status']=='failed'
    assert [job['returncode'] for job in result['jobs']]==[0,0,2]
    assert summary.v1.sha(file)==digest
    assert not (tmp_path/summary.v1.BASE/'endpoint.endpoint.json').exists()


@pytest.mark.parametrize('change',['completed','different_error','claims_output','bad_hash','bad_command'])
def test_failed_attempt_rejects_rewritten_or_unrelated_failure(tmp_path,monkeypatch,change):
    original,execution,file=fixture_attempt(tmp_path,monkeypatch)
    if change=='completed':execution['status']='completed'
    elif change=='different_error':execution['error']='out of memory'
    elif change=='claims_output':execution['jobs'][-1]['output_sha256']='fake'
    elif change=='bad_hash':execution['jobs'][0]['log_sha256']='fake'
    else:execution['jobs'][-1]['command']=['unexpected']
    file.write_text(json.dumps(execution))
    with pytest.raises(ValueError):summary.audit_failed_attempt('endpoint',original,summary.v1.Inputs())


def test_failure_requires_argparse_log_and_real_integer_exit_code(tmp_path,monkeypatch):
    original,execution,file=fixture_attempt(tmp_path,monkeypatch)
    log=summary.v1.local(execution['jobs'][-1]['log']);log.write_text('some other exit2')
    execution['jobs'][-1]['log_sha256']=summary.v1.sha(log);file.write_text(json.dumps(execution))
    with pytest.raises(ValueError,match='argparse'):summary.audit_failed_attempt('endpoint',original,summary.v1.Inputs())
    execution['jobs'][0]['returncode']=False;file.write_text(json.dumps(execution))
    with pytest.raises(ValueError,match='exit status'):summary.audit_failed_attempt('endpoint',original,summary.v1.Inputs())


def fixture_continuation(tmp_path,monkeypatch):
    original,old,file=fixture_attempt(tmp_path,monkeypatch)
    continuation=dict(original_evidence={'endpoint':{'old_execution_sha256':summary.v1.sha(file)}},
        input_hashes=original['input_hashes'],source_hashes=original['source_hashes'],
        per_arm_commands={'endpoint':summary.repaired_commands(original,'endpoint')})
    folder=tmp_path/summary.CONTINUATION;folder.mkdir()
    immutable=dict(continuation['input_hashes'],**continuation['source_hashes'])
    jobs=[]
    for kind in summary.REMAINING:
        log=folder/f'endpoint.{kind}.log';log.write_text('done')
        output=tmp_path/summary.v1.BASE/f'endpoint.{kind}.json';output.write_text('{}')
        job=dict(kind=kind,returncode=0,pid=200+len(jobs),completed_utc='2026-11-01T00:01:00+00:00',
            command=list(continuation['per_arm_commands']['endpoint'][kind]),frozen_before=immutable,frozen_after=immutable,
            log=str(summary.CONTINUATION/f'endpoint.{kind}.log'),log_sha256=summary.v1.sha(log),
            output_sha256=summary.v1.sha(output))
        if kind!='endpoint':
            sidecar=output.with_suffix('.component.pt');sidecar.write_bytes(b'component')
            job['component_sha256']=summary.v1.sha(sidecar)
        jobs.append(job)
    execution=dict(arm='endpoint',status='completed',completed_utc='2026-11-01T00:01:00+00:00',
        frozen_inputs_verified=True,original_execution=str(summary.v1.BASE/'endpoint.execution.json'),
        original_execution_sha256=summary.v1.sha(file),latest_sha256=old['latest_sha256'],jobs=jobs)
    path=folder/'endpoint.execution.json';path.write_text(json.dumps(execution))
    return continuation,old,execution,path


def test_continuation_requires_three_successes_and_keeps_old_failure_separate(tmp_path,monkeypatch):
    continuation,old,execution,_=fixture_continuation(tmp_path,monkeypatch)
    inputs=summary.v1.Inputs()
    result=summary.audit_continuation_execution('endpoint',continuation,old,inputs)
    assert result==execution and old['status']=='failed'
    assert [j['kind'] for j in result['jobs']]==list(summary.REMAINING)
    assert len([p for p in inputs.hashes if p.name.endswith('.component.pt')])==2


@pytest.mark.parametrize('state',['running','failed'])
def test_incomplete_suite_refuses_before_any_score_file_read(tmp_path,monkeypatch,state):
    continuation,old,execution,file=fixture_continuation(tmp_path,monkeypatch)
    execution['status']=state;execution['jobs']=execution['jobs'][:1]
    file.write_text(json.dumps(execution))
    inputs=summary.v1.Inputs()
    with pytest.raises(ValueError,match='incomplete'):
        summary.audit_continuation_execution('endpoint',continuation,old,inputs)
    assert list(inputs.hashes)==[file.resolve()]
    assert not (tmp_path/summary.OUTPUT).exists()


@pytest.mark.parametrize('mutation',['checkpoint','old_execution','command','missing_job','sidecar','exit_code'])
def test_continuation_rejects_changed_lineage_command_or_output(tmp_path,monkeypatch,mutation):
    continuation,old,execution,file=fixture_continuation(tmp_path,monkeypatch)
    if mutation=='checkpoint':execution['latest_sha256']='changed'
    elif mutation=='old_execution':execution['original_execution_sha256']='changed'
    elif mutation=='command':execution['jobs'][0]['command']+=['--other']
    elif mutation=='missing_job':execution['jobs'].pop()
    elif mutation=='exit_code':execution['jobs'][1]['returncode']=2
    else:execution['jobs'][1].pop('component_sha256')
    file.write_text(json.dumps(execution))
    with pytest.raises(ValueError):summary.audit_continuation_execution('endpoint',continuation,old,summary.v1.Inputs())


def test_v2_refuses_old_summary_or_existing_output(tmp_path,monkeypatch):
    monkeypatch.setattr(summary.v1,'REPO',tmp_path)
    with pytest.raises(ValueError,match='unregistered output'):
        summary.build_summary(tmp_path/summary.v1.BASE/'summary.json')
    file=tmp_path/summary.OUTPUT;file.parent.mkdir(parents=True);file.write_text('preserve')
    with pytest.raises(ValueError,match='overwrite'):summary.build_summary(file)
    assert file.read_text()=='preserve'


def test_build_stops_at_unfinished_suite_before_result_statistics(tmp_path,monkeypatch):
    continuation,old,execution,file=fixture_continuation(tmp_path,monkeypatch)
    monkeypatch.setattr(summary.v1,'ARMS',('endpoint',))
    for relative in (summary.SCRIPT,summary.v1.SCRIPT,summary.v1.SHARED):
        p=tmp_path/relative;p.parent.mkdir(exist_ok=True);p.write_text('synthetic source')
    monkeypatch.setattr(summary,'V1_SHA',summary.v1.sha(summary.v1.SCRIPT))
    original=dict(input_hashes=continuation['input_hashes'],source_hashes=continuation['source_hashes'],
        per_arm_commands={'endpoint':{j['kind']:j['command'] for j in old['jobs']}})
    (tmp_path/summary.v1.BASE/'protocol.json').write_text(json.dumps(original))
    (tmp_path/summary.v1.BASE/'summary.protocol.json').write_text('{}')
    (tmp_path/summary.CONTINUATION/'protocol.json').write_text(json.dumps(continuation))
    execution['status']='running';execution['jobs']=execution['jobs'][:2];file.write_text(json.dumps(execution))
    reached=[]
    def forbidden(*args,**kwargs):
        reached.append(True)
        raise AssertionError('must not read result statistics')
    monkeypatch.setattr(summary,'result_statistics',forbidden)
    with pytest.raises(ValueError,match='continuation incomplete'):
        summary.build_summary()
    assert reached==[] and not (tmp_path/summary.OUTPUT).exists()
