"""Audit the preserved CLI failure and continuation, with frozen v1 statistics.

This version changes execution provenance only. It never rewrites the original
failed executions, retrains a model, or changes the registered estimands.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import summarize_hs_target_pilot as v1


SCRIPT = Path('scripts/summarize_hs_target_pilot_v2.py')
CONTINUATION = v1.BASE / 'continuation_v2'
OUTPUT = v1.BASE / 'summary_v2.json'
VERSION = 'hs_matched_target_paired_summary_execution_v2'
V1_SHA = 'f0d3a23a69fa9ee99cafe8d1e60330176201c72212c212d9d6fa289b8b77ab08'
CONTINUATION_RUNNER = Path('run_scripts/continue_hs_target_pilot_v2.py')
CONTINUATION_RUNNER_SHA = '8f5da96eab25c8e131199c46a1274cb99b49a22c71be25471590bdaf07b7cb51'
REMAINING = ('endpoint', 'suite', 'matched8')
require = v1.require


def repaired_commands(original, arm):
    """The only command change is deleting the rejected endpoint option pair."""
    commands = {kind:list(original['per_arm_commands'][arm][kind]) for kind in REMAINING}
    endpoint = commands['endpoint']
    require(endpoint.count('--max-residues')==1, 'original endpoint flag is not unique')
    index = endpoint.index('--max-residues')
    require(index+1<len(endpoint) and endpoint[index+1]=='96', 'original endpoint residue budget differs')
    del endpoint[index:index+2]
    return commands


def audit_job(job, command, immutable, inputs, log, returncode):
    require(type(job.get('returncode')) is int and job['returncode']==returncode
        and job.get('completed_utc') and type(job.get('pid')) is int and job['pid']>0,
        'missing real process or unexpected exit status')
    require(job['command']==command, 'execution command differs')
    require(job['frozen_before']==immutable and job['frozen_after']==immutable,
        'execution frozen-hash evidence differs')
    require(v1.local(job['log'])==v1.local(log), 'execution log path differs')
    inputs.track(log,job['log_sha256'])


def audit_failed_attempt(arm, original, inputs, expected_sha=None):
    file = v1.BASE / f'{arm}.execution.json'
    inputs.track(file,expected_sha)
    execution = inputs.read(file)
    require(execution.get('arm')==arm and execution.get('status')=='failed'
        and execution.get('failed_utc') and not execution.get('completed_utc')
        and execution.get('error')==f'RuntimeError: {arm}/endpoint exited 2; no automatic restart',
        'original endpoint failure was not preserved')
    require([j['kind'] for j in execution['jobs']]==['train','ensemble','endpoint'],
        'unexpected original attempt history')
    immutable = dict(original['input_hashes'],**original['source_hashes'])
    latest = v1.ARTIFACTS / arm / 'checkpoint_latest.pt'
    inputs.track(latest,execution['latest_sha256'])
    for job,code in zip(execution['jobs'],(0,0,2)):
        kind = job['kind']
        audit_job(job,original['per_arm_commands'][arm][kind],immutable,inputs,
            v1.BASE/f'{arm}.{kind}.log',code)
        if kind=='ensemble':
            inputs.track(v1.BASE/f'{arm}.ensemble.json',job['output_sha256'])
    failed = execution['jobs'][2]
    require('output_sha256' not in failed, 'failed attempt unexpectedly claims an output')
    log = v1.local(failed['log']).read_text()
    require('eval_ca_endpoint_scores.py: error: unrecognized arguments: --max-residues 96' in log,
        'failure was not the registered argparse rejection')
    return execution


def audit_continuation_protocol(continuation, original, registration, old_executions, inputs):
    require(continuation['schema_version']=='hs_target_pilot_cli_continuation_v2'
        and continuation['scientific_protocol_unchanged'] is True
        and continuation['checkpoints_unchanged'] is True and continuation['optimizer_steps']==0
        and continuation['test_opened'] is False and continuation['bioemu2_rerun'] is False
        and continuation['automatic_restart'] is False, 'continuation scientific scope differs')
    require(continuation['arms']==original['arms'] and continuation['jobs']==list(REMAINING),
        'continuation jobs or arms differ')
    for role,expected in (('original_protocol',v1.BASE/'protocol.json'),
                          ('statistics_protocol',v1.BASE/'summary.protocol.json')):
        require(v1.local(continuation[role])==v1.local(expected)
            and continuation[role+'_sha256']==inputs.hashes[v1.local(expected)],
            f'continuation {role} identity differs')
    require(all(continuation['input_hashes'].get(p)==digest for p,digest in original['input_hashes'].items()),
        'original registered inputs not preserved')
    require(all(continuation['source_hashes'].get(p)==digest for p,digest in original['source_hashes'].items()),
        'original frozen source was changed')
    require(continuation['source_hashes'].get(str(CONTINUATION_RUNNER))==CONTINUATION_RUNNER_SHA,
        'continuation runner source differs from reviewed version')
    require(v1.local(continuation['source_snapshot'])==v1.local(CONTINUATION/'source_snapshot'),
        'continuation snapshot path differs')
    for file,digest in continuation['input_hashes'].items(): inputs.track(file,digest)
    for file,digest in continuation['source_hashes'].items():
        inputs.track(file,digest); inputs.track(CONTINUATION/'source_snapshot'/file,digest)
    absent = []
    for arm in v1.ARMS:
        require(continuation['per_arm_commands'][arm]==repaired_commands(original,arm),
            'continuation changed more than the rejected endpoint option')
        old = old_executions[arm]
        evidence = continuation['original_evidence'][arm]
        expected = dict(old_execution=str(v1.BASE/f'{arm}.execution.json'),
            old_execution_sha256=inputs.hashes[v1.local(v1.BASE/f'{arm}.execution.json')],
            checkpoint=str(v1.ARTIFACTS/arm/'checkpoint_latest.pt'),checkpoint_sha256=old['latest_sha256'],
            checkpoint_audit=old['checkpoint_audit'],ensemble=str(v1.BASE/f'{arm}.ensemble.json'),
            ensemble_sha256=old['jobs'][1]['output_sha256'],successful_old_jobs=['train','ensemble'],
            preserved_failed_job=dict(kind='endpoint',returncode=2),endpoint_result_absent_at_registration=True)
        require(evidence==expected, 'continuation original evidence disagrees with preserved failure')
        require(datetime.fromisoformat(old['failed_utc'])<=datetime.fromisoformat(continuation['registered_utc']),
            'continuation was registered before preserved failure')
        mapping = {kind:str((v1.BASE if kind in ('train','ensemble') else CONTINUATION)/f'{arm}.execution.json')
            for kind in v1.JOBS}
        require(continuation['logical_job_audit'][arm]==mapping, 'logical job provenance differs')
        for kind in REMAINING:
            output = v1.BASE/f'{arm}.{kind}.json'
            absent.extend([str(output),str(output.with_suffix('.component.pt'))])
            parser = continuation['cli_parser_contracts'][arm][kind]
            require(v1.local(parser['checkpoint'])==v1.local(v1.ARTIFACTS/arm/'checkpoint_latest.pt')
                and v1.local(parser['out'])==v1.local(output) and parser['n_samples']==12 and parser['seed']==0,
                'repaired real-parser checkpoint/output/sample contract differs')
            if kind=='endpoint':
                require('max_residues' not in parser and parser['max_windows']==200,
                    'endpoint parser contract changed')
            else:
                settings = registration['suite']['full' if kind=='suite' else 'matched8']
                require(parser['max_residues']==96 and parser['horizons']==list(v1.HORIZONS)
                    and parser['proteins_per_temp']==settings['proteins_per_temperature']
                    and parser['anchors_per_trajectory']==settings['anchors']
                    and parser['modes']==settings['modes'] and parser['t2a'] is True,
                    'suite real-parser budget differs')
    require(continuation['outputs_verified_absent_before_registration']==absent,
        'continuation did not register absence of all twelve pending output/sidecar paths')


def audit_continuation_execution(arm, continuation, old_execution, inputs):
    file = CONTINUATION/f'{arm}.execution.json'
    execution = inputs.read(file)
    require(execution.get('arm')==arm and execution.get('status')=='completed'
        and execution.get('completed_utc') and execution.get('frozen_inputs_verified') is True,
        f'{arm}: continuation incomplete or failed')
    require(execution['original_execution']==str(v1.BASE/f'{arm}.execution.json')
        and execution['original_execution_sha256']==continuation['original_evidence'][arm]['old_execution_sha256']
        and execution['latest_sha256']==old_execution['latest_sha256'], 'continuation lineage differs')
    require([j['kind'] for j in execution['jobs']]==list(REMAINING), 'all three continuation jobs required')
    immutable = dict(continuation['input_hashes'],**continuation['source_hashes'])
    for job in execution['jobs']:
        kind = job['kind']
        audit_job(job,continuation['per_arm_commands'][arm][kind],immutable,inputs,
            CONTINUATION/f'{arm}.{kind}.log',0)
        output = v1.BASE/f'{arm}.{kind}.json'
        inputs.track(output,job['output_sha256'])
        sidecar = output.with_suffix('.component.pt')
        require(v1.local(sidecar).exists()==('component_sha256' in job), 'continuation sidecar declaration differs')
        if 'component_sha256' in job: inputs.track(sidecar,job['component_sha256'])
    return execution


def result_statistics(original, registration, executions, inputs):
    """Run the unchanged v1 model/data audits and statistical functions."""
    import torch
    from protein_jepa_frame.utils.contracts import state_sha256
    initial = torch.load(inputs.track(v1.INITIAL),map_location='cpu',weights_only=False)
    require(initial['checkpoint_kind']=='lasp_ca_dynamics' and initial['step']==0
        and initial['probe_provenance']['source_training_step']==original['source_training_step'], 'initial lineage differs')
    require(v1.component_identity(initial['component_checkpoint'])==original['initial_component_identity']
        and state_sha256(initial['kernel'])==original['initial_kernel_sha256'], 'initial tensors differ')
    checkpoints = {a:torch.load(inputs.track(v1.ARTIFACTS/a/'checkpoint_latest.pt'),
        map_location='cpu',weights_only=False) for a in v1.ARMS}
    checked = {a:v1.audit_checkpoint(a,checkpoints[a],initial,executions[a],original,inputs) for a in v1.ARMS}
    normalized = [{k:v for k,v in checkpoints[a]['args'].items() if k not in ('target','output')} for a in v1.ARMS]
    require(normalized[0]==normalized[1], 'training differs beyond target/output')
    first,second = v1.ARMS
    require(checked[first]['first_training_step']==checked[second]['first_training_step'],
        'first h1 target-equivalent loss/gradient norm differs')
    require(torch.equal(checkpoints[first]['sampling_rng'],checkpoints[second]['sampling_rng']),
        'sampling RNG consumption differs')
    require(v1.shared.same_tree(checkpoints[first]['rng_state'],checkpoints[second]['rng_state']),
        'training global RNG state differs')
    gate = v1.audit_reused_gate(original,initial,inputs)
    documents,ensemble,endpoint,suite_rows,suite_reports = {},{},{},{},{}
    records = [json.loads(line) for line in inputs.track(v1.shared.VAL).read_text().splitlines() if line]
    suite_keys = {kind:v1.expected_suite_keys(records,kind) for kind in ('suite','matched8')}
    for arm in v1.ARMS:
        documents[arm] = {kind:inputs.read(v1.BASE/f'{arm}.{kind}.json') for kind in v1.JOBS[1:]}
        for kind,data in documents[arm].items():
            require(v1.local(data['checkpoint'])==v1.local(v1.ARTIFACTS/arm/'checkpoint_latest.pt'),
                'evaluation checkpoint path differs')
            if kind in ('ensemble','endpoint'):
                require(data['checkpoint_sha256']==executions[arm]['latest_sha256']
                    and data['optimizer_steps']==0 and data['test_opened'] is False, 'evaluation identity differs')
                v1.shared.audit_sources(data,original,inputs,arm+'/'+kind)
                if kind=='endpoint': v1.shared.audit_provenance(data,original,inputs,arm+'/'+kind)
        ensemble[arm] = v1.shared.audit_ensemble(documents[arm]['ensemble'],arm)
        endpoint[arm] = v1.audit_endpoint(documents[arm]['endpoint'],arm,checked[arm])
        suite_rows[arm],suite_reports[arm] = {},{}
        for kind in ('suite','matched8'):
            suite_rows[arm][kind],suite_reports[arm][kind] = v1.audit_suite(documents[arm][kind],arm,kind)
            require(set(suite_rows[arm][kind])==suite_keys[kind], 'suite raw keys differ from manifest protocol')
        for k,row in suite_rows[arm]['matched8'].items():
            require(row==suite_rows[arm]['suite'][k], 'same matched8/full-suite window is not reproduced exactly')
    for kind in ('suite','matched8'):
        require(set(suite_rows[first][kind])==set(suite_rows[second][kind]), 'suite arm keys differ')
        for k in suite_rows[first][kind]:
            a,b = [suite_rows[arm][kind][k] for arm in v1.ARMS]
            require(a['persist']==b['persist'] and all(a['position'][m]==b['position'][m]
                for m in ('persist_kabsch_ca_rmsd','persist_aligned_ca_rmsd')), 'suite references differ across arms')
    bio,baseline = v1.audit_bioemu(inputs)
    maximum = v1.audit_pairing(ensemble,endpoint,baseline,documents)
    statistics = v1.analysis(ensemble,endpoint,baseline)
    t2a = v1.bioemu_t2a_evidence({a:suite_rows[a]['matched8'] for a in v1.ARMS},
        {a:documents[a]['matched8'] for a in v1.ARMS},inputs)
    return dict(role=registration['role'],test_opened=False,bioemu2_rerun=False,
        n_windows=200,n_proteins=50,n_samples=12,
        statistical_registration=str(v1.BASE/'summary.protocol.json'),experiment_protocol=str(v1.BASE/'protocol.json'),
        bootstrap=registration['bootstrap'],scientific_limits=registration['scientific_limits'],
        checkpoint_selection=registration['checkpoint_rule'],checkpoint_audit=checked,
        paired_training_audit=dict(only_arguments_differ=['target','output'],same_final_sampling_rng=True,
            same_final_global_rng=True,same_stateless_training_data_configuration=True,
            first_h1_loss_terms_and_gradient_norm_bitwise_equal=True,
            input_identity_evidence='same train/val manifest hashes, stateless-data seed42, identical horizon '
                'cycle/500-step schedule and frozen loader source; no training-row identity list serialized'),
        reused_H_gate=gate,endpoint_metric_semantics=documents[first]['endpoint']['metric_semantics'],
        suite_reports=suite_reports,bioemu2_matched8_reference_audit=t2a,
        bioemu2_adapter=dict(path=str(v1.BIOEMU),sha256=inputs.hashes[v1.local(v1.BIOEMU)],
            evaluation_args=bio['evaluation_args'],original_sources_metadata_only=bio['sources'],
            persistence_max_abs_difference=maximum),validation_loss_note=registration['validation_loss'],**statistics)


def build_summary(output=OUTPUT):
    import torch
    output = v1.local(output)
    require(output==v1.local(OUTPUT) and not output.exists(), 'refusing overwrite or unregistered output location')
    torch.set_num_threads(2)
    inputs = v1.Inputs()
    inputs.track(SCRIPT); inputs.track(v1.SCRIPT,V1_SHA); inputs.track(v1.SHARED)
    registration = inputs.read(v1.BASE/'summary.protocol.json')
    original = inputs.read(v1.BASE/'protocol.json')
    continuation = inputs.read(CONTINUATION/'protocol.json')
    old_executions = {a:audit_failed_attempt(a,original,inputs,
        continuation['original_evidence'][a]['old_execution_sha256']) for a in v1.ARMS}
    # Never read score JSON/model tensors while a required suite is unfinished.
    new_executions = {a:audit_continuation_execution(a,continuation,old_executions[a],inputs) for a in v1.ARMS}
    v1.audit_protocol(original,registration,inputs)
    audit_continuation_protocol(continuation,original,registration,old_executions,inputs)
    report = result_statistics(original,registration,old_executions,inputs)
    report.update(schema_version=VERSION,statistical_schema_version=v1.VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),
        continuation_protocol=str(CONTINUATION/'protocol.json'),
        execution_audit=dict(total_process_attempts=12,successful_attempts=10,failed_attempts=2,
            completed_logical_jobs=10,original_execution_status='failed in both arms; preserved unchanged',
            original_successful_jobs=['train','ensemble'],continuation_successful_jobs=list(REMAINING),
            logical_job_sources=continuation['logical_job_audit'],
            frozen_sources_inputs_and_both_snapshots_verified=True,
            failure_records={a:dict(path=str(v1.BASE/f'{a}.execution.json'),
                sha256=inputs.hashes[v1.local(v1.BASE/f'{a}.execution.json')],status=old_executions[a]['status'],
                error=old_executions[a]['error'],failed_job=old_executions[a]['jobs'][2]) for a in v1.ARMS},
            continuation_records={a:dict(path=str(CONTINUATION/f'{a}.execution.json'),
                sha256=inputs.hashes[v1.local(CONTINUATION/f'{a}.execution.json')],
                status=new_executions[a]['status']) for a in v1.ARMS}),
        implementation_repair=dict(reason=continuation['reason'],sole_amendment=continuation['sole_amendment'],
            optimizer_steps=0,checkpoints_unchanged=True,scientific_protocol_unchanged=True,
            v1_summary_source_sha256=V1_SHA,new_summary_path=str(OUTPUT),
            note='The original endpoint attempts exited2 during argparse, before run(). The frozen endpoint '
                 'run fixes max_residues=96 internally. Only the invalid CLI option pair was removed. '
                 'No original execution is relabelled completed; all v1 numerical/statistical functions are reused.'))
    inputs.verify()
    report['input_sha256'] = {v1.label(file):digest for file,digest in sorted(inputs.hashes.items())}
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=str(OUTPUT))
    args = parser.parse_args()
    build_summary(args.out)
    print(json.dumps(dict(output=args.out,sha256=v1.sha(args.out))))


if __name__=='__main__':
    main()
