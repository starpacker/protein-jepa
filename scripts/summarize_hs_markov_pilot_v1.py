"""Registered paired C0-versus-H+S summary, preserving all original executions.

Reuses frozen target-pilot statistics explicitly; no new scoring, training,
trajectory access, or BioEmu sampling. Registration never opens score JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import summarize_hs_target_pilot as stats
from scripts import summarize_hs_target_pilot_v2 as provenance
from run_scripts import run_hs_markov_pilot_v1 as runner


BASE = Path('results/hs_joint_20261101/markov_v1')
SCRIPT = Path('scripts/summarize_hs_markov_pilot_v1.py')
TEST = Path('tests/test_hs_markov_summary_v1.py')
VERSION = 'hs_markov_paired_summary_v1'
ARMS = ('markov_c0', 'trained_hs')
SLOTS = dict(zip(ARMS, stats.ARMS))
PINNED = {
    'scripts/summarize_hs_target_pilot.py': 'f0d3a23a69fa9ee99cafe8d1e60330176201c72212c212d9d6fa289b8b77ab08',
    'scripts/summarize_hs_target_pilot_v2.py': '98047422ad09fdebc3bca11db2d208d2ac4ba83cd4e4fb8e266ab3fd64b762a8',
    'scripts/summarize_hs_readout_pilot.py': '6b87d5207c9e56a5af6a77eba3a3c20ffd08b91f4c955384fd0cbcb9c7686599',
    'run_scripts/run_hs_markov_pilot_v1.py': '28b7dde33a5f836ebe6fab29b4a0f47545515fe9d40474f9c5740aa58cad4a98',
    'docs/HS_MARKOV_PILOT_V1_20261101.md': '1d3f07e92508246e3c068a78068385b2a400cb87380b93ad7c553fbe4abe094e',
    'tests/test_hs_markov_pilot_v1.py': '8c0b8402d69bfba8172afcbfe2ddb4b27f4d0ac761e43c5c80e003cdc6a259d9',
}
PROTOCOL_SHA = '5eca1e002b696e127cbb7c4954b82f39227363f41f3c0955c053a382a221bb1a'
require = stats.require


def contract():
    original = stats.statistical_contract()
    return dict(schema_version=VERSION, arms=list(ARMS), contrast='markov_c0 minus trained_hs',
        role='single_seed_adaptive_development_pilot', primary=original['primary'],
        groups=original['groups'], bootstrap=original['bootstrap'],
        historical_seven_metrics=original['historical_seven_metrics'],
        historical_metric_directions=original['historical_metric_directions'],
        historical_reference=original['historical_reference'],
        secondary_endpoint_scores=original['secondary_endpoint_scores'],
        endpoint_units=original['endpoint_units'],endpoint_estimator=original['endpoint_estimator'],
        canonical_ca_geometry=original['canonical_ca_geometry'],geometry_directions=original['geometry_directions'],
        baseline_comparisons='Each arm minus saved BioEmu2 on seven historical metrics; each arm minus persistence on three endpoint scores.',
        descriptive='All arm/baseline/persistence/target means; legacy anchor geometry and contact set counts descriptive only.',
        n_paired_comparisons=341,n_descriptive_means=670,n_windows=200,n_proteins=50,n_samples=12,
        new_jobs=['train','ensemble','endpoint'],checkpoint='latest exactly500; no best/checkpoint selection',
        reused_arm='target_v1 endpoint500 checkpoint and its unchanged ensemble200/endpoint200; no resampling',
        implementation='Frozen target analysis() uses positional arm slots. Map first slot to markov_c0 and second to trained_hs; rename all output arm/contrast keys. Numerical estimators unchanged.',
        inherited_slot_names_have_no_target_semantics=True,
        training='Identical frozen initialization and endpoint target; only coupling_scale .1->0 and output differ. Retain exact_spatial/history/bath8 and all RNG request shapes.',
        provenance='Three actual new exit0 jobs; original target argparse failures retained. Full component/cfg, final sampling/global RNG, optimizer state support/step budget, real C0 reload audited.',
        historical_data_limit='Old S500 did not serialize per-training-window identities or HDF5 content hashes. Current metadata replay/stat stability does not establish historical coordinate byte identity.',
        H_gate='Reuse frozen true-H readout gate solely because full encoder/decoder/component configuration is unchanged. No new forecast or memory evidence from that gate.',
        suite='No new full temperature suite or matched8; standard200 temperature strata only, not a full T2/T3 evaluation.',
        limits=['Warm start already trained with H+S; not from-scratch or matched effective capacity/compute.',
                'Single training seed, adaptive development validation; marginal exploratory intervals without multiplicity or selection adjustment.',
                'CI crossing zero is not equivalence; C0/S difference does not isolate physical memory or latent identifiability.',
                'Historical seven metrics use future-path mixture; three endpoint scores retain their separate finite-M definitions.'],
        test_opened=False,bioemu2_rerun=False)


def register():
    output = stats.local(BASE/'summary.protocol.json')
    require(not output.exists() and not stats.local(BASE/'summary.json').exists(), 'refusing existing registration/summary')
    inputs = stats.Inputs()
    protocol = inputs.read(BASE/'protocol.json')
    require(inputs.hashes[stats.local(BASE/'protocol.json')]==PROTOCOL_SHA, 'unexpected frozen experiment protocol')
    decision = inputs.read(BASE/'decision.json')
    require(inputs.hashes[stats.local(BASE/'decision.json')]==runner.DECISION_SHA, 'decision changed')
    primary = decision['primary']
    require(primary['metric']=='coordinate_energy_score' and primary['horizons']==[16,64]
        and primary['contrast']=='Markov_C0 minus trained_HS', 'primary differs from earlier decision')
    require(protocol['schema_version']==runner.VERSION and protocol['statistics']==primary, 'runner decision differs')
    for p,digest in PINNED.items(): inputs.track(p,digest)
    inputs.track(SCRIPT);inputs.track(TEST)
    sources = {stats.label(p):d for p,d in inputs.hashes.items() if p.suffix in ('.py','.md')}
    inputs.verify()
    record = dict(contract(),registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_stage='Before this summary implementation or author opened any S or C0 score JSON. Earlier decision fixed reuse and primary before target scores were read; training may already be running.',
        experiment_protocol=str(BASE/'protocol.json'),experiment_protocol_sha256=PROTOCOL_SHA,
        decision_sha256=runner.DECISION_SHA,source_hashes=sources)
    with output.open('x') as stream: json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=stats.sha(output))))


def analysis(ensemble,endpoint,baseline):
    require(set(ensemble)==set(endpoint)==set(ARMS),'expected exactly C0 and reused H+S')
    result = stats.analysis({SLOTS[a]:ensemble[a] for a in ARMS},{SLOTS[a]:endpoint[a] for a in ARMS},baseline)
    names = {**{v:k for k,v in SLOTS.items()},'endpoint_vs_path_mixture':'markov_c0_vs_trained_hs'}
    def rename(value):
        if isinstance(value,dict):return {names.get(k,k):rename(v) for k,v in value.items()}
        if isinstance(value,list):return [rename(v) for v in value]
        return value
    return rename(result)


def audit_new_execution(protocol,inputs):
    execution=inputs.read(BASE/'execution.json')
    require(execution.get('status')=='completed' and execution.get('completed_utc')
        and execution.get('frozen_files_verified') is True, 'Markov execution incomplete or failed')
    require(execution.get('protocol_sha256')==PROTOCOL_SHA and type(execution.get('runner_pid')) is int
        and execution['runner_pid']>0,'unregistered runner process')
    require([j['kind'] for j in execution['jobs']]==list(runner.JOBS),'all three new jobs required')
    for j in execution['jobs']:
        require(type(j.get('returncode')) is int and j['returncode']==0 and j.get('completed_utc')
            and type(j.get('pid')) is int and j['pid']>0,'missing actual successful Markov process')
        require(j['command']==protocol['commands'][j['kind']],'new execution command differs')
    # Only inspect score identities after complete successful execution is known.
    inputs.track(runner.ARTIFACTS/'checkpoint_latest.pt',execution['latest_sha256'])
    for j in execution['jobs']:
        require(stats.local(j['log'])==stats.local(BASE/f"markov.{j['kind']}.log"),'unexpected new log path')
        inputs.track(j['log'],j['log_sha256'])
        if j['kind']!='train':
            output=BASE/f"markov.{j['kind']}.json";inputs.track(output,j['output_sha256'])
            sidecar=output.with_suffix('.component.pt')
            require(stats.local(sidecar).exists()==('component_sha256' in j),'sidecar evidence differs')
            if 'component_sha256' in j:inputs.track(sidecar,j['component_sha256'])
    return execution


def audit_protocol(protocol,registration,inputs):
    require(all(registration[k]==v for k,v in contract().items()),'statistics registration changed')
    require(registration['experiment_protocol_sha256']==PROTOCOL_SHA
        and inputs.hashes[stats.local(BASE/'protocol.json')]==PROTOCOL_SHA,'experiment protocol changed')
    for p,d in registration['source_hashes'].items():inputs.track(p,d)
    for p,d in PINNED.items():require(registration['source_hashes'].get(p)==d,'frozen helper pin missing')
    require(protocol['schema_version']==runner.VERSION and protocol['commands']==runner.commands()
        and protocol['new_jobs']==list(runner.JOBS),'new run contract differs')
    require(protocol['automatic_restart'] is False and protocol['test_opened'] is False
        and protocol['bioemu2_rerun'] is False and protocol['complete_suite_in_new_budget'] is False,'new scope differs')
    for p,d in protocol['input_hashes'].items():inputs.track(p,d)
    for p,d in protocol['source_hashes'].items():
        inputs.track(p,d);inputs.track(Path(protocol['source_snapshot'])/p,d)
    runner.verify(protocol,PROTOCOL_SHA)
    require(len(protocol['selected_train500'])==500,'missing registered train stream')
    require([r['step'] for r in protocol['selected_train500']]==list(range(1,501))
        and [r['horizon'] for r in protocol['selected_train500']]==[stats.HORIZONS[i%4] for i in range(500)],'train stream schedule differs')
    require(protocol['historical_training_identity_limit']==registration['historical_data_limit'] or
        'not historical coordinate byte identity' in protocol['historical_training_identity_limit'],'historical input limitation missing')


def inference_hash(checkpoint):
    from protein_jepa_frame.utils.contracts import state_sha256
    values={f'bundle.encoder.{k}':v for k,v in checkpoint['component_checkpoint']['encoder'].items()}
    values.update({f'bundle.decoder.{k}':v for k,v in checkpoint['component_checkpoint']['delta_ca_decoder'].items()})
    values.update({f'kernel.{k}':v for k,v in checkpoint['kernel'].items()})
    return state_sha256(values)


def audit_markov_checkpoint(checkpoint,reused,execution,protocol,inputs):
    audited=runner.checkpoint_audit(checkpoint,reused,protocol['initial_component_identity'])
    require(audited==execution['checkpoint_audit'],'new checkpoint audit record differs')
    require(checkpoint['step']==500 and checkpoint['args']['train_encoder'] is False
        and checkpoint['args']['target']=='endpoint' and checkpoint['args']['resume'] is False,'new training scope differs')
    run=inputs.read(runner.ARTIFACTS/'run.json')
    require(run['args']==checkpoint['args'],'Markov run/checkpoint args differ')
    split=run['split_audit']
    require(all(split[k]==0 for k in ('protein_overlap','domain_overlap','source_file_overlap')),'train/val overlap')
    for role,p in (('train',stats.shared.TRAIN),('val',stats.shared.VAL)):
        require(split[role+'_manifest_sha256']==protocol['input_hashes'][p],'new split hash differs')
    records=[json.loads(line) for line in inputs.track(runner.ARTIFACTS/'metrics.jsonl').read_text().splitlines() if line]
    require(not any('resumed_from' in r for r in records),'unexpected new resume')
    require([r['step'] for r in records if 'train' in r]==[1,*range(10,501,10)],'new training log budget differs')
    starts=[r for r in records if r.get('step')==0 and 'val' in r]
    ends=[r for r in records if r.get('step')==500 and 'val' in r]
    require(len(starts)==len(ends)==1 and ends[0]['val']==checkpoint['validation'],'new checkpoint validation/log differs')
    require(all(step==500 for step in audited['optimizer_steps_by_state'].values())
        and bool(audited['optimizer_steps_by_state']),'optimizer update budget differs')
    return dict(checkpoint=str(runner.ARTIFACTS/'checkpoint_latest.pt'),checkpoint_sha256=execution['latest_sha256'],
        args=checkpoint['args'],step=500,step_ps=checkpoint['kernel_config']['step_ps'],
        inference_model_tensor_sha256=inference_hash(checkpoint),audit=audited,
        initial_target_specific_validation=stats.shared.common_validation(starts[0]['val'],checkpoint['args']),
        final_target_specific_validation=stats.shared.common_validation(ends[0]['val'],checkpoint['args']))


def read_scores(checked,original,new_protocol,inputs):
    documents,ensemble,endpoint={},{},{}
    for arm,folder,prefix in ((ARMS[0],BASE,'markov'),(ARMS[1],stats.BASE,'endpoint')):
        documents[arm]={kind:inputs.read(folder/f'{prefix}.{kind}.json') for kind in ('ensemble','endpoint')}
        for kind,data in documents[arm].items():
            require(stats.local(data['checkpoint'])==stats.local(checked[arm]['checkpoint'])
                and data['checkpoint_sha256']==checked[arm]['checkpoint_sha256'],'score checkpoint differs')
            require(data['optimizer_steps']==0 and data['test_opened'] is False,'score scope differs')
            stats.shared.audit_sources(data,new_protocol if arm==ARMS[0] else original,inputs,arm+'/'+kind)
            if kind=='endpoint':stats.shared.audit_provenance(data,original,inputs,arm+'/'+kind)
        ensemble[arm]=stats.shared.audit_ensemble(documents[arm]['ensemble'],arm)
        # Both checkpoints were trained on endpoint targets. The frozen audit's
        # arm argument encodes that training-target value, not this contrast role.
        endpoint[arm]=stats.audit_endpoint(documents[arm]['endpoint'],'endpoint',checked[arm])
    bio,baseline=stats.audit_bioemu(inputs)
    maximum=stats.audit_pairing({SLOTS[a]:ensemble[a] for a in ARMS},{SLOTS[a]:endpoint[a] for a in ARMS},baseline,
        {SLOTS[a]:documents[a] for a in ARMS})
    return documents,ensemble,endpoint,bio,baseline,maximum


def build_summary(output=BASE/'summary.json'):
    import torch
    output=stats.local(output)
    require(output==stats.local(BASE/'summary.json') and not output.exists(),'refusing overwrite or unregistered output')
    inputs=stats.Inputs();torch.set_num_threads(2)
    registration=inputs.read(BASE/'summary.protocol.json')
    protocol=inputs.read(BASE/'protocol.json')
    # Fail closed before model loading or score parsing while any new job runs.
    execution=audit_new_execution(protocol,inputs)
    audit_protocol(protocol,registration,inputs)
    original=inputs.read(stats.BASE/'protocol.json')
    original_registration=inputs.read(stats.BASE/'summary.protocol.json')
    continuation=inputs.read(provenance.CONTINUATION/'protocol.json')
    old_exec={a:provenance.audit_failed_attempt(a,original,inputs,
        continuation['original_evidence'][a]['old_execution_sha256']) for a in stats.ARMS}
    completed={a:provenance.audit_continuation_execution(a,continuation,old_exec[a],inputs) for a in stats.ARMS}
    stats.audit_protocol(original,original_registration,inputs)
    provenance.audit_continuation_protocol(continuation,original,original_registration,old_exec,inputs)
    initial=torch.load(inputs.track(stats.INITIAL),map_location='cpu',weights_only=False)
    reused=torch.load(inputs.track(runner.S_CHECKPOINT),map_location='cpu',weights_only=False)
    markov=torch.load(inputs.track(runner.ARTIFACTS/'checkpoint_latest.pt'),map_location='cpu',weights_only=False)
    checked={ARMS[1]:stats.audit_checkpoint('endpoint',reused,initial,old_exec['endpoint'],original,inputs)}
    checked[ARMS[0]]=audit_markov_checkpoint(markov,reused,execution,protocol,inputs)
    gate=stats.audit_reused_gate(original,initial,inputs)
    documents,ensemble,endpoint,bio,baseline,maximum=read_scores(checked,original,protocol,inputs)
    statistics=analysis(ensemble,endpoint,baseline)
    report=dict(schema_version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),role=registration['role'],
        n_windows=200,n_proteins=50,n_samples=12,arms=list(ARMS),candidate=ARMS[0],reference=ARMS[1],
        statistical_registration=str(BASE/'summary.protocol.json'),bootstrap=registration['bootstrap'],
        implementation_sources=registration['source_hashes'],implementation_note=registration['implementation'],
        checkpoint_audit=checked,reused_H_gate=gate,endpoint_metric_semantics=documents[ARMS[0]]['endpoint']['metric_semantics'],
        bioemu2_adapter=dict(path=str(stats.BIOEMU),sha256=inputs.hashes[stats.local(stats.BIOEMU)],
            evaluation_args=bio['evaluation_args'],persistence_max_abs_difference=maximum),
        execution_audit=dict(new_jobs_completed=3,new_failed_attempts=0,
            new_execution=dict(path=str(BASE/'execution.json'),sha256=inputs.hashes[stats.local(BASE/'execution.json')]),
            reused_S_jobs=['original train','original ensemble','continuation endpoint'],
            original_target_process_attempts=12,original_target_successful_logical_jobs=10,original_target_failed_attempts=2,
            failures_preserved={a:dict(path=str(stats.BASE/f'{a}.execution.json'),
                sha256=inputs.hashes[stats.local(stats.BASE/f'{a}.execution.json')],failed_job=old_exec[a]['jobs'][2]) for a in stats.ARMS},
            note='Original failures stay failed. New runner calls frozen-file checks before/after jobs and records final verified status; it does not serialize per-job frozen hash maps.'),
        paired_training=dict(only_arguments_differ=['coupling_scale','output'],same_initial_kernel_tensors=True,
            full_component_frozen=True,final_sampling_and_global_rng_identical=True,optimizer_state_support_and500steps_identical=True,
            final_kernel_and_optimizer_moment_values_allowed_to_differ=True,
            selected_training_metadata=protocol['selected_train500'],historical_input_limit=registration['historical_data_limit']),
        scientific_limits=registration['limits'],suite_scope=registration['suite'],
        test_opened=False,bioemu2_rerun=False,new_model_sampling=False,**statistics)
    runner.verify(protocol,PROTOCOL_SHA);inputs.verify()
    report['input_sha256']={stats.label(p):d for p,d in sorted(inputs.hashes.items())}
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--out',default=str(BASE/'summary.json'))
    args=parser.parse_args()
    if args.register:register()
    else:
        build_summary(args.out)
        print(json.dumps(dict(path=args.out,sha256=stats.sha(args.out))))


if __name__=='__main__':main()
