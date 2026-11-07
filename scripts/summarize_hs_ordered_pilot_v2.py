"""Serialization-only repair of the frozen ordered-pilot summary wrapper.

The v1 expected panel contains a tuple; persisted producer JSON contains a list.
This separately registered implementation canonicalizes the expected panel to
JSON. All scientific calculations and other audits remain the frozen helpers.
No global monkeypatching, new scoring, sampling, or statistical change occurs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from scripts import summarize_hs_ordered_pilot_v1 as v1

old, numeric, runner = v1.old, v1.numeric, v1.runner
BASE, ARTIFACTS, ARMS, JOBS = v1.BASE, v1.ARTIFACTS, v1.ARMS, v1.JOBS
IMPLEMENTATION = Path('results/hs_joint_20261101/ordered_path_summary_v2')
SCRIPT = 'scripts/summarize_hs_ordered_pilot_v2.py'
TEST = 'tests/test_hs_ordered_summary_v2.py'
VERSION = 'hs_ordered_path_pilot_summary_v2'
V1_REGISTRATION = v1.IMPLEMENTATION/'protocol.json'
V1_REGISTRATION_SHA = '04166f2856099c8b0d494e58924ed3ccb8f1072f47378cac879314fa6795975e'
V1_SOURCE_SHA = 'fc3ca858ab181160303f9f23d6d64b747c13665d445470e9e88a1fb67b7fa964'
SOURCES = tuple(sorted(set((*v1.SOURCES, SCRIPT, TEST))))
require, now, write_new = old.require, v1.now, v1.write_new


def repair_record():
    return dict(discovery='static review and synthetic JSON roundtrip; no real summary execution failure',
        defect='v1 compared persisted panel.horizons list with an in-memory expected tuple',
        repair='Only expected panel is normalized with json.loads(json.dumps(panel_spec(...))).',
        old_source_changed=False, statistical_definitions_changed=False,
        real_v1_summary_failure_claimed=False, old_registration_preserved=True,
        numerical_functions='Unchanged hs_ordered_pilot_statistics_v1.standard_analysis and ordered_analysis',
        total_paired_intervals=402)


def contract():
    result = dict(v1.contract())
    result.update(schema_version=VERSION, output=str(BASE/'summary_v2.json'),
        original_wrapper_registration=str(V1_REGISTRATION),
        original_wrapper_registration_sha256=V1_REGISTRATION_SHA,
        original_wrapper_source_sha256=V1_SOURCE_SHA,
        implementation_repair=repair_record(),
        source_snapshot=str(IMPLEMENTATION/'source_snapshot'))
    return result


def register():
    """Read/hash protocols and source only; record score-file existence, not values."""
    output = old.local(IMPLEMENTATION/'protocol.json')
    require(not output.exists() and not old.local(BASE/'summary_v2.json').exists(), 'refusing registration overwrite')
    prior_attempts = [v1.IMPLEMENTATION/'started.json', v1.IMPLEMENTATION/'execution.json', BASE/'summary.json']
    require(not any(old.local(p).exists() for p in prior_attempts),
            'v1 summary execution now exists; static-discovery registration context must be re-reviewed')
    inputs = old.Inputs()
    inputs.track(V1_REGISTRATION, V1_REGISTRATION_SHA)
    prior = inputs.read(V1_REGISTRATION)
    require(all(prior.get(k) == value for k,value in v1.contract().items()), 'v1 wrapper registration differs')
    inputs.track(v1.SCRIPT, V1_SOURCE_SHA)
    inputs.track(BASE/'protocol.json', v1.PROTOCOL_SHA)
    inputs.track(BASE/'summary.protocol.json', v1.STATISTICS_SHA)
    statistics = inputs.read(BASE/'summary.protocol.json')
    require(all(statistics.get(k) == value for k,value in v1.registration.contract().items()),
            'frozen statistical contract differs')
    for source,digest in {**statistics['source_hashes'], **prior['source_hashes']}.items():
        inputs.track(source,digest)
    for source in SOURCES:
        inputs.track(source)
    for path,digest in v1.REGISTRATION_EVIDENCE.items():
        inputs.track(path,digest)
    checked_utc = now()
    score_existence = {str(BASE/f'{arm}.{kind}.json'): old.local(BASE/f'{arm}.{kind}.json').exists()
                       for arm in ARMS for kind in JOBS[1:]}
    # Only the two new sources need new snapshots; the frozen v1/protocol chain
    # remains independently pinned and is fully re-audited during build_summary.
    snapshots = {}
    for source in (SCRIPT, TEST):
        target = old.local(IMPLEMENTATION/'source_snapshot'/source)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(old.local(source).read_bytes())
        digest = inputs.hashes[old.local(source)]
        inputs.track(target,digest)
        snapshots[str(target.relative_to(old.REPO))] = digest
    inputs.verify()
    sources = {source: inputs.hashes[old.local(source)] for source in SOURCES}
    record = dict(contract(), registered_utc=now(),
        registration_stage='Before this author/v2 wrapper reads any new ordered-pilot performance score. Outputs may already exist; scientific estimands and v1 implementation were registered earlier. This repairs a statically identified serialization mismatch.',
        score_file_existence_checked_utc=checked_utc, score_file_existence=score_existence,
        score_json_parsed_during_registration=False,
        prior_summary_execution_absent_at_registration=[str(p) for p in prior_attempts],
        source_hashes=sources, snapshot_hashes=snapshots,
        input_hashes={old.label(p):digest for p,digest in inputs.hashes.items()
                      if old.label(p) not in sources and old.label(p) not in snapshots})
    write_new(output,record)
    return record


def audit_implementation(implementation, inputs):
    require(all(implementation.get(k) == value for k,value in contract().items()), 'v2 implementation contract differs')
    require(implementation['score_json_parsed_during_registration'] is False
        and set(implementation['source_hashes']) == set(SOURCES), 'v2 source/registration scope differs')
    require(set(implementation['snapshot_hashes']) == {
        str(IMPLEMENTATION/'source_snapshot'/source) for source in (SCRIPT, TEST)}, 'v2 snapshot coverage differs')
    for source,digest in implementation['source_hashes'].items():
        inputs.track(source,digest)
    for source,digest in implementation['snapshot_hashes'].items():
        original = str(Path(source).relative_to(IMPLEMENTATION/'source_snapshot'))
        require(digest == implementation['source_hashes'][original], 'v2 snapshot digest differs')
        inputs.track(source,digest)
    for path,digest in implementation['input_hashes'].items():
        inputs.track(path,digest)
    inputs.track(V1_REGISTRATION,V1_REGISTRATION_SHA)
    inputs.track(v1.SCRIPT,V1_SOURCE_SHA)


def audit_ordered(data, arm, checked, protocol, inputs):
    """Identical v1 checks, except JSON-normalized expected panel representation."""
    from scripts.eval_ca_ordered_paths import SOURCES as required_sources, panel_spec, summarize as raw_summary
    output = BASE/f'{arm}.ordered.json'
    registered = inputs.read(output.with_suffix('.protocol.json'))
    require(all(data.get(k) == value for k,value in registered.items()), 'ordered output changed registration')
    require(data['task_version'] == 'ca_ordered_paths_development_v1' and data['dataset_role'] == 'development_validation'
        and data['optimizer_steps'] == 0 and data['model_selection'] is False and data['test_opened'] is False
        and data['bioemu2_rerun'] is False and data['future_information_given_to_sampler'] is False
        and data['frozen_inputs_verified'] is True and data.get('completed_utc'), 'ordered task incomplete/different')
    expected_panel = json.loads(json.dumps(panel_spec([1,4,16,64],200), allow_nan=False))
    require(data['panel'] == expected_panel and data['n_windows'] == 200
        and data['n_proteins'] == 50, 'ordered panel incomplete')
    require(old.local(data['args']['checkpoint']) == old.local(checked['checkpoint'])
        and data['input_hashes'][data['args']['checkpoint']] == checked['checkpoint_sha256']
        and data['model_tensor_sha256'] == checked['inference_model_tensor_sha256'], 'ordered loaded model differs')
    require(data['sampling'] == dict(seed=0, n_paths=12, model_history_frames=4, prediction_kind='endpoint',
        s0_mode='exact_spatial', one_continuous_rollout_per_member=True, no_future_reencoding=True, no_bath_reset=True),
        'ordered sampling contract differs')
    old.shared.audit_eval_protocol(data['protocol'], arm+'/ordered')
    old.shared.audit_provenance(data, protocol, inputs, arm+'/ordered')
    for path,digest in data['input_hashes'].items():
        inputs.track(path,digest)
    require(set(required_sources) <= set(data['source_hashes']), 'ordered required source coverage missing')
    for path,digest in data['source_hashes'].items():
        require(protocol['source_hashes'].get(path) == digest, 'unregistered ordered source')
        inputs.track(path,digest)
        inputs.track(output.with_suffix('.source_snapshot')/path,digest)
    rows = v1.paths.audit_rows(data['per_window'],data['panel'])
    require(raw_summary(rows) == data['by_horizon'], 'ordered raw aggregate differs')
    return old.shared.rows_for(data,arm+'/ordered')


def build_summary():
    """Explicit v1 orchestration with the repaired ordered audit; no global swaps."""
    import torch
    output = old.local(BASE/'summary_v2.json')
    require(not output.exists(), 'refusing summary overwrite')
    torch.set_num_threads(2)
    inputs = old.Inputs()
    implementation = inputs.read(IMPLEMENTATION/'protocol.json')
    statistics = inputs.read(BASE/'summary.protocol.json')
    protocol = inputs.read(BASE/'protocol.json')
    executions = {a:inputs.read(BASE/f'{a}.execution.json') for a in ARMS}
    v1.execution_headers(executions,protocol)
    audit_implementation(implementation,inputs)
    prior = inputs.read(V1_REGISTRATION)
    v1.audit_protocol(protocol,prior,statistics,inputs)
    v1.audit_executions(executions,protocol,inputs)
    registration_attempts = v1.audit_registration_attempts(inputs)
    initial = torch.load(inputs.track(runner.INITIAL,runner.INITIAL_SHA),map_location='cpu',weights_only=False)
    checkpoints = {a:torch.load(inputs.track(ARTIFACTS/a/'checkpoint_latest.pt'),map_location='cpu',weights_only=False) for a in ARMS}
    checked,paired_training = v1.audit_training(checkpoints,initial,executions,protocol,inputs)
    gate = old.audit_reused_gate(protocol,initial,inputs)
    documents,ensemble,endpoints,ordered,suite_rows,suite_reports = {},{},{},{},{},{}
    records = [json.loads(line) for line in inputs.track(old.shared.VAL).read_text().splitlines()]
    expected_suite = {kind:old.expected_suite_keys(records,kind) for kind in ('suite','matched8')}
    for arm in ARMS:
        documents[arm] = {kind:inputs.read(BASE/f'{arm}.{kind}.json') for kind in JOBS[1:]}
        for kind in ('ensemble','endpoint','suite','matched8'):
            data = documents[arm][kind]
            require(old.local(data['checkpoint']) == old.local(checked[arm]['checkpoint']), 'score checkpoint path differs')
            if kind in ('ensemble','endpoint'):
                require(data['checkpoint_sha256'] == checked[arm]['checkpoint_sha256']
                    and data['optimizer_steps'] == 0 and data['test_opened'] is False, 'score checkpoint/scope differs')
                old.shared.audit_sources(data,protocol,inputs,arm+'/'+kind)
                if kind == 'endpoint':
                    old.shared.audit_provenance(data,protocol,inputs,arm+'/'+kind)
        ensemble[arm] = old.shared.audit_ensemble(documents[arm]['ensemble'],arm)
        endpoints[arm] = old.audit_endpoint(documents[arm]['endpoint'],'endpoint',checked[arm])
        ordered[arm] = audit_ordered(documents[arm]['ordered'],arm,checked[arm],protocol,inputs)
        suite_rows[arm],suite_reports[arm] = {},{}
        for kind in ('suite','matched8'):
            suite_rows[arm][kind],suite_reports[arm][kind] = old.audit_suite(documents[arm][kind],arm,kind)
            require(set(suite_rows[arm][kind]) == expected_suite[kind], 'suite manifest selection differs')
    v1.suite_pairing(suite_rows)
    bio,baseline = old.audit_bioemu(inputs)
    maximum = old.audit_pairing(v1.slot(ensemble),v1.slot(endpoints),baseline,v1.slot(documents))
    v1.cross_task_pairing(ensemble,endpoints,ordered,baseline)
    t2a = old.bioemu_t2a_evidence(v1.slot({a:suite_rows[a]['matched8'] for a in ARMS}),
        v1.slot({a:documents[a]['matched8'] for a in ARMS}),inputs)
    values = {**numeric.standard_analysis(ensemble,endpoints,baseline), **numeric.ordered_analysis(ordered)}
    require(v1.count_intervals(values) == 402, 'paired interval coverage differs')
    report = dict(schema_version=VERSION,created_utc=now(),arms=list(ARMS),candidate=ARMS[0],reference=ARMS[1],
        role='single_seed_adaptive_development_pilot',n_windows=200,n_proteins=50,n_samples=12,
        statistical_registration=str(BASE/'summary.protocol.json'),wrapper_registration=str(IMPLEMENTATION/'protocol.json'),
        original_wrapper_registration=str(V1_REGISTRATION),implementation_repair=repair_record(),
        implementation_sources=implementation['source_hashes'],statistical_contract=statistics,
        checkpoint_audit=checked,paired_training_audit=paired_training,reused_H_gate=gate,
        registration_attempts=registration_attempts,
        execution_audit=dict(successful_logical_jobs=12,all_actual_exit0=True,
            source_inputs_snapshots_verified=True,ordered_reference_fingerprints_match_across_arms=True,
            all_standard_task_window_keys_match=True,matched8_full_raw_rows_exact=True),
        bootstrap=statistics['bootstrap'],endpoint_metric_semantics=documents[ARMS[0]]['endpoint']['metric_semantics'],
        suite_reports=suite_reports,bioemu2_matched8_reference_audit=t2a,
        bioemu2_adapter=dict(path=str(old.BIOEMU),sha256=inputs.hashes[old.local(old.BIOEMU)],
            evaluation_args=bio['evaluation_args'],persistence_max_abs_difference=maximum),
        scientific_limits=statistics['scientific_limits'],test_opened=False,bioemu2_rerun=False,new_model_sampling=False,
        **values)
    runner.verify(protocol,BASE/'protocol.json',inputs.hashes[old.local(BASE/'protocol.json')])
    inputs.verify()
    report['input_sha256'] = {old.label(p):digest for p,digest in sorted(inputs.hashes.items())}
    write_new(output,report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('register','run'),required=True)
    args = parser.parse_args()
    if args.mode == 'register':
        register()
        print(json.dumps(dict(path=str(IMPLEMENTATION/'protocol.json'),sha256=old.sha(IMPLEMENTATION/'protocol.json'))))
        return
    started = dict(pid=os.getpid(),started_utc=now(),mode='run',source_sha256=old.sha(SCRIPT))
    write_new(IMPLEMENTATION/'started.json',started)
    start = time.monotonic()
    try:
        build_summary()
        require(old.sha(SCRIPT) == started['source_sha256'], 'summary source changed during run')
        write_new(IMPLEMENTATION/'execution.json',dict(started,completed_utc=now(),seconds=time.monotonic()-start,
            exit_code=0,status='completed',output=str(BASE/'summary_v2.json'),output_sha256=old.sha(BASE/'summary_v2.json')))
    except BaseException as error:
        write_new(IMPLEMENTATION/'execution.json',dict(started,completed_utc=now(),seconds=time.monotonic()-start,
            exit_code=1,status='failed',error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
