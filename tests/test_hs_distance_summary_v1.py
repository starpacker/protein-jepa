"""Synthetic provenance and estimand-boundary checks; no real pilot outputs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts import summarize_hs_distance_pilot_v1 as s
from scripts import eval_ca_ordered_paths as producer


def execution_fixture():
    protocol = dict(input_hashes={'input':'a'},source_hashes={'source':'b'},per_arm_commands={})
    executions = {}
    for arm in s.ARMS:
        commands = {kind:['python',kind,arm] for kind in s.JOBS}
        protocol['per_arm_commands'][arm] = commands
        executions[arm] = dict(arm=arm,status='completed',runner_pid=11,completed_utc='done',
            successful_logical_jobs=6,jobs=[dict(kind=kind,pid=12,returncode=0,
                started_utc='start',completed_utc='done',command=commands[kind],
                frozen_before={'input':'a','source':'b'},frozen_after={'input':'a','source':'b'}) for kind in s.JOBS])
    return executions,protocol


def test_both_complete_headers_before_loading_models_or_scores(monkeypatch):
    executions,protocol = execution_fixture()
    s.execution_headers(executions,protocol)
    executions['absolute']['status'] = 'running'
    reads = []
    class Inputs:
        def read(self,path):
            path=str(path);reads.append(path)
            for arm in s.ARMS:
                if path.endswith(f'/{arm}.execution.json'):return executions[arm]
            if path == str(s.BASE/'protocol.json'):return protocol
            return {}
    monkeypatch.setattr(s.old,'Inputs',Inputs)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('model loaded before both complete'))
    with pytest.raises(ValueError,match='both arms'):
        s.build_summary()
    assert not any(any(p.endswith(f'.{kind}.json') for kind in s.JOBS[1:]) for p in reads)


@pytest.mark.parametrize('mutation',['missing','exit','bool_exit','pid','command','hash'])
def test_incomplete_execution_evidence_fails(mutation):
    executions,protocol=execution_fixture();job=executions[s.ARMS[0]]['jobs'][2]
    if mutation=='missing':executions[s.ARMS[0]]['jobs'].pop()
    elif mutation=='exit':job['returncode']=1
    elif mutation=='bool_exit':job['returncode']=False
    elif mutation=='pid':job['pid']=0
    elif mutation=='command':job['command']=['changed']
    else:job['frozen_after']={'input':'wrong','source':'b'}
    with pytest.raises(ValueError):s.execution_headers(executions,protocol)


def terms(arm):
    return dict(loss=3.,loss_base=2.,loss_absolute=3.,loss_increment=3.,
        distance_absolute_es=1.,distance_increment_es=1.,contact=.05,
        coord_es=1.,coordinate_objective=1.,latent_es=.2,clash=.01)


def first_steps():
    return {a:dict(step=1,horizon=1,lr=2e-6,grad_norm=2.,train=terms(a)) for a in s.ARMS}


def test_first_h1_numerical_identity_preserves_rounding_errors():
    rows=first_steps()
    # Two totals in the same graph may round differently; selected alias stays exact.
    rows['increment']['train'].update(distance_increment_es=1.000001,
        loss_increment=3.000001,loss=3.000001)
    rows['increment']['grad_norm']=2.000001
    before=deepcopy(rows)
    result=s.audit_initial_h1(rows)
    assert result['first_h1_numerical_identity']
    assert result['first_h1_scalar_differences']['loss']['signed_error']>0
    assert result['first_h1_gradient_norm_difference']['signed_error']>0
    assert result['first_h1_selected_losses']=={'increment':3.000001,'absolute':3.}
    assert rows==before


@pytest.mark.parametrize('mutation',['shared','auxiliary','base','wrong_loss','unselected_total','lr','horizon','nan','gradient','missing'])
def test_initial_h1_mismatch_rejected(mutation):
    rows=first_steps();candidate=rows['increment']
    if mutation=='shared':candidate['train']['coord_es']+=.1
    elif mutation=='auxiliary':candidate['train']['distance_increment_es']+=.1
    elif mutation=='base':candidate['train']['loss_base']+=.1
    elif mutation=='wrong_loss':candidate['train']['loss']=2.
    elif mutation=='unselected_total':candidate['train']['loss_absolute']=2.
    elif mutation=='lr':candidate['lr']=0.
    elif mutation=='horizon':candidate['horizon']=4
    elif mutation=='gradient':candidate['grad_norm']+=.01
    elif mutation=='missing':candidate['train'].pop('contact')
    else:candidate['grad_norm']=float('nan')
    with pytest.raises(ValueError):s.audit_initial_h1(rows)


def ordered_fixture():
    data=dict(task_version='ca_ordered_paths_development_v1',dataset_role='development_validation',
        optimizer_steps=0,model_selection=False,test_opened=False,bioemu2_rerun=False,
        future_information_given_to_sampler=False,frozen_inputs_verified=True,completed_utc='done',
        panel=producer.panel_spec([1,4,16,64],200),n_windows=200,n_proteins=50,
        args={'checkpoint':str(s.old.REPO/'synthetic.pt')},input_hashes={'synthetic.pt':'sha'},
        model_tensor_sha256='tensor',sampling=dict(seed=0,n_paths=12,model_history_frames=4,
            prediction_kind='endpoint',s0_mode='exact_spatial',one_continuous_rollout_per_member=True,
            no_future_reencoding=True,no_bath_reset=True),protocol={},
        source_hashes={p:'source' for p in producer.SOURCES},per_window=['raw'],by_horizon={'h':1})
    data=json.loads(json.dumps(data))
    checked=dict(checkpoint='synthetic.pt',checkpoint_sha256='sha',inference_model_tensor_sha256='tensor')
    return data,checked,dict(source_hashes=deepcopy(data['source_hashes']))


class OrderedInputs:
    def __init__(self,data):self.registered={'panel':deepcopy(data['panel'])};self.tracked=[]
    def read(self,path):return deepcopy(self.registered)
    def track(self,*args):self.tracked.append(args)


def ordered_downstream(monkeypatch):
    calls=[]
    monkeypatch.setattr(s.old.shared,'audit_eval_protocol',lambda *a:calls.append('protocol'))
    monkeypatch.setattr(s.old.shared,'audit_provenance',lambda *a:calls.append('provenance'))
    def rows(raw,panel):calls.append('rows');assert isinstance(panel['horizons'],list);return raw
    monkeypatch.setattr(s.paths,'audit_rows',rows)
    monkeypatch.setattr(producer,'summarize',lambda rows:{'h':1})
    monkeypatch.setattr(s.old.shared,'rows_for',lambda *a:{'checked':1})
    return calls


def test_ordered_json_panel_and_absolute_relative_checkpoint_hash(monkeypatch):
    data,checked,protocol=ordered_fixture();before=deepcopy(data)
    calls=ordered_downstream(monkeypatch)
    assert s.audit_ordered(data,s.ARMS[0],checked,protocol,OrderedInputs(data))=={'checked':1}
    assert calls==['protocol','provenance','rows'] and data==before


@pytest.mark.parametrize('mutation',['panel','sha','tensor','source','aggregate','future'])
def test_ordered_boundary_repairs_preserve_fail_closed_audits(monkeypatch,mutation):
    data,checked,protocol=ordered_fixture();inputs=OrderedInputs(data)
    ordered_downstream(monkeypatch)
    if mutation=='panel':data['panel']['horizons']=[1,4,64,16];inputs=OrderedInputs(data)
    elif mutation=='sha':data['input_hashes']['synthetic.pt']='wrong'
    elif mutation=='tensor':data['model_tensor_sha256']='wrong'
    elif mutation=='source':data['source_hashes']={}
    elif mutation=='aggregate':data['by_horizon']={'h':2}
    else:data['future_information_given_to_sampler']=True
    with pytest.raises(ValueError):s.audit_ordered(data,s.ARMS[0],checked,protocol,inputs)


def test_contained_path_aliases_reject_outside_missing_and_conflicts():
    absolute=str(s.old.REPO/'synthetic.pt')
    assert s.contained_digest({'synthetic.pt':'sha',absolute:'sha'},absolute)=='sha'
    with pytest.raises(ValueError,match='conflicting'):
        s.contained_digest({'synthetic.pt':'sha',absolute:'other'},absolute)
    with pytest.raises(ValueError,match='outside'):
        s.contained_digest({'/outside/checkpoint.pt':'sha'},'/outside/checkpoint.pt')
    with pytest.raises(ValueError,match='missing'):
        s.contained_digest({'other.pt':'sha'},absolute)


def test_local_legacy_slots_and_contract_counts():
    values=dict(increment={'x':1},absolute={'x':2})
    assert s.slot(values)=={'endpoint':{'x':1},'path_mixture':{'x':2}}
    assert set(values)==set(s.ARMS)
    contract=s.contract()
    assert [contract[k] for k in ('n_paired_intervals','standard_descriptive_means','suite_descriptive_means')]==[402,670,1700]
    assert s.optimizer_pair is s.legacy.optimizer_pair and s.inference_hash is s.legacy.inference_hash


def test_registration_allows_existing_unparsed_scores_and_pins_snapshots(tmp_path,monkeypatch):
    monkeypatch.setattr(s.old,'REPO',tmp_path)
    monkeypatch.setattr(s,'BASE',Path('results/contact'))
    monkeypatch.setattr(s,'IMPLEMENTATION',Path('results/summary'))
    monkeypatch.setattr(s,'SOURCES',('new.py','tests.py'))
    monkeypatch.setattr(s.registration,'contract',lambda:{'science':'frozen'})
    for path in s.SOURCES:(tmp_path/path).write_text('# synthetic source\n')
    (tmp_path/s.BASE).mkdir(parents=True)
    science=dict(science='frozen',registered_utc='2026-11-02T00:00:00+00:00',source_hashes={})
    (tmp_path/s.BASE/'summary.protocol.json').write_text(json.dumps(science))
    protocol=dict(statistics_sha256=s.old.sha(s.BASE/'summary.protocol.json'),registered_utc='2026-11-02T01:00:00+00:00')
    (tmp_path/s.BASE/'protocol.json').write_text(json.dumps(protocol))
    score=tmp_path/s.BASE/'increment.ordered.json';score.write_text('INVALID JSON MUST NEVER BE PARSED')
    result=s.register()
    assert result['score_json_parsed_during_registration'] is False
    assert result['score_file_existence'][str(s.BASE/'increment.ordered.json')]
    for source in s.SOURCES:
        assert (tmp_path/s.IMPLEMENTATION/'source_snapshot'/source).read_bytes()==(tmp_path/source).read_bytes()
    with pytest.raises(ValueError,match='refusing registration overwrite'):s.register()


def test_registration_rejects_retrospective_scientific_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(s.old,'REPO',tmp_path)
    monkeypatch.setattr(s,'BASE',Path('results/contact'))
    monkeypatch.setattr(s,'IMPLEMENTATION',Path('results/summary'))
    monkeypatch.setattr(s.registration,'contract',lambda:{'science':'frozen'})
    (tmp_path/s.BASE).mkdir(parents=True)
    science=dict(science='frozen',registered_utc='2026-11-02T02:00:00+00:00',source_hashes={})
    (tmp_path/s.BASE/'summary.protocol.json').write_text(json.dumps(science))
    protocol=dict(statistics_sha256=s.old.sha(s.BASE/'summary.protocol.json'),registered_utc='2026-11-02T01:00:00+00:00')
    (tmp_path/s.BASE/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='after experiment'):s.register()
    assert not (tmp_path/s.IMPLEMENTATION).exists()


def synthetic_training(tmp_path,monkeypatch):
    from protein_jepa_frame.utils.contracts import state_sha256
    monkeypatch.setattr(s,'ARTIFACTS',tmp_path/'artifacts')
    monkeypatch.setattr(s,'BASE',tmp_path/'results')
    monkeypatch.setattr(s.old,'local',lambda p:Path(p).resolve())
    monkeypatch.setattr(s.old,'component_identity',lambda c:{'fixed':'component'})
    monkeypatch.setattr(s.runner,'checkpoint_audit',lambda ck,arm,p:{'checked':arm})
    class Inputs:
        def track(self,path,expected=None):return Path(path)
        def read(self,path):return json.loads(Path(path).read_text())
    def write(path,value,lines=False):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(json.dumps(r) for r in value)+'\n' if lines else json.dumps(value))
    initial=dict(component_checkpoint={'encoder':{'x':torch.tensor([1.])},'delta_ca_decoder':{'y':torch.tensor([2.])}},
        kernel={f'k{i}':torch.tensor([float(i)]) for i in range(73)},kernel_config={'step_ps':1000})
    initial['kernel']['k0']=torch.zeros(669429-72)
    windows=[dict(step=i,horizon=(1,4,16,64)[(i-1)%4],record_path='/synthetic.h5') for i in range(1,501)]
    source='protein_jepa_frame/training/train_lasp_ca_distance_v1.py'
    protocol=dict(initial_kernel_sha256=state_sha256(initial['kernel']),initial_kernel_config=initial['kernel_config'],
        initial_component_identity={'fixed':'component'},training_windows=windows,selected_hdf5_stats={'/synthetic.h5':{'size':10}},
        source_hashes={source:'trainer'},per_arm_commands={})
    identity=dict(model_sha256=s.inference_hash(initial),component_sha256=s.inference_hash(initial,component_only=True),
        kernel_sha256=state_sha256(initial['kernel']),kernel_parameter_names=list(initial['kernel']),kernel_parameter_elements=669429)
    checkpoints,executions={},{}
    for arm in s.ARMS:
        directory=s.ARTIFACTS/arm
        args=dict(output=str(directory),distance_path_objective=arm,distance_path_weight=1.,contact_estimator='plugin',coord_objective='marginal_path',seed=42)
        opt=dict(param_groups=[dict(params=list(range(73)),lr=.001)],state={0:dict(step=torch.tensor(500.),
            exp_avg=torch.tensor([1.,2.]),exp_avg_sq=torch.tensor([3.,4.]))})
        checkpoints[arm]=dict(initial,args=args,optimizer=opt,sampling_rng=torch.tensor([1,2],dtype=torch.uint8),rng_state={'global':[1,2]})
        command=['python','-u','-m','protein_jepa_frame.training.train_lasp_ca_distance_v1','--output',str(directory),'--distance-path-objective',arm]
        protocol['per_arm_commands'][arm]={'train':command}
        started=dict(pid=12,command=[source,*command[4:]],source_sha256='trainer',run_kind='train',distance_path_objective=arm)
        write(directory/'started.json',started)
        write(directory/'execution.json',dict(started,status='completed',exit_code=0,completed_utc='done',
            output=str(directory/'checkpoint_latest.pt'),output_sha256='checkpoint',
            result=dict(optimizer_steps=500,frozen_component_unchanged=True,kernel_parameter_tensors=73,active_optimizer_states=1)))
        write(directory/'run.json',dict(args=args,train_only=True,test_opened=False,no_validation_selection=True,validation_loaded=False))
        write(directory/'initial_identity.json',identity)
        actual=[dict(row,actual_batch_sha256='a'*64,file_stat={'size':10}) for row in windows]
        write(directory/'training_windows.jsonl',actual,True)
        metrics=[dict(step=row['step'],horizon=row['horizon'],grad_norm=2.000001 if arm=='increment' else 2.,
            lr=2e-6,kernel_parameters=73,gradient_parameters=1,train=terms(arm)) for row in windows]
        write(directory/'metrics.jsonl',metrics,True)
        executions[arm]=dict(checkpoint_audit={'checked':arm},latest_sha256='checkpoint',jobs=[dict(pid=12,
            training_windows_audit=s.runner.audit_training_windows(directory/'training_windows.jsonl',windows))])
    pair=dict(initial_model_equal=True,actual_training_window_bytes_equal=True,final_sampling_and_global_rng_equal=True,
        only_argument_differences=['output','distance_path_objective'],compared_with_historical_endpoint_rng=False)
    write(s.BASE/'paired_training_audit.json',pair)
    return checkpoints,initial,executions,protocol,Inputs(),write


def test_full_synthetic_500_step_audit_uses_new_metadata_and_numerical_h1_identity(tmp_path,monkeypatch):
    *args,write=synthetic_training(tmp_path,monkeypatch)
    checked,audit=s.audit_training(*args)
    assert set(checked)==set(s.ARMS) and audit['n_actual_training_windows']==500
    assert audit['kernel_parameter_tensors']==73 and audit['optimizer']['active_states']==1
    assert audit['first_h1_gradient_norms']=={'increment':2.000001,'absolute':2.}
    assert audit['first_h1_numerical_identity']
    path=s.ARTIFACTS/'absolute'/'training_windows.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['actual_batch_sha256']='b'*64;write(path,rows,True)
    args[2]['absolute']['jobs'][0]['training_windows_audit']=s.runner.audit_training_windows(path,args[3]['training_windows'])
    with pytest.raises(ValueError,match='actual 500 training batches differ'):s.audit_training(*args)


@pytest.mark.parametrize('mutation',['pid','validation','inventory','parameter_support','selected_loss','coord_objective','weight','contact','elements','rng'])
def test_new_training_audit_rejects_schema_and_estimator_drift(tmp_path,monkeypatch,mutation):
    *args,write=synthetic_training(tmp_path,monkeypatch)
    directory=s.ARTIFACTS/'increment'
    if mutation=='pid':
        path=directory/'started.json';data=json.loads(path.read_text());data['pid']=99;write(path,data)
    elif mutation=='validation':
        path=directory/'run.json';data=json.loads(path.read_text());data['validation_loaded']=True;write(path,data)
    elif mutation=='inventory':
        path=directory/'initial_identity.json';data=json.loads(path.read_text());data['kernel_parameter_names'].pop();write(path,data)
    elif mutation=='parameter_support':args[0]['increment']['optimizer']['param_groups'][0]['params'].pop()
    elif mutation=='selected_loss':
        path=directory/'metrics.jsonl';data=[json.loads(line) for line in path.read_text().splitlines()]
        data[100]['train']['loss']=2.;write(path,data,True)
    elif mutation=='coord_objective':args[0]['increment']['args']['coord_objective']='joint_path'
    elif mutation=='weight':args[0]['increment']['args']['distance_path_weight']=2.
    elif mutation=='contact':args[0]['increment']['args']['contact_estimator']='population_u'
    elif mutation=='elements':
        path=directory/'initial_identity.json';data=json.loads(path.read_text());data['kernel_parameter_elements']=73;write(path,data)
    else:args[0]['increment']['sampling_rng'][0]=9
    with pytest.raises((ValueError,AssertionError)):s.audit_training(*args)


def test_coverage_counts_exclude_original_suite_summaries_and_keep_standard_scopes():
    standard={'seven_metrics':{'arm_means':{a:{'overall':{'m':1.}} for a in s.ARMS}},
              'ordered_path_scores':{'descriptive':{'arbitrary':99.}},'group_sizes':{'n':200}}
    assert s.count_standard_means(standard)==2
    def cell(n):return {'means':{str(i):float(i) for i in range(n)}}
    reports={}
    for arm in s.ARMS:
        reports[arm]={}
        for kind,modes in (('suite',('correct','wrong','none')),('matched8',('correct',))):
            reports[arm][kind]=dict(
                overall={m:cell(20) for m in modes},
                by_temperature={str(t):{m:cell(20) for m in modes} for t in range(5)},
                by_horizon={str(h):{m:cell(20) for m in modes} for h in range(4)},
                t2a={str(t):cell(5) for t in range(5)},
                original_report_summaries={'means':{'exclude':42.}})
    assert s.count_suite_means(reports)==1700
    reports['absolute']['matched8']['overall']['correct']['means'].pop('0')
    assert s.count_suite_means(reports)==1699


def test_selected_alias_is_exact_and_both_totals_are_audited():
    values=terms('increment')
    values.update(distance_absolute_es=1.,distance_increment_es=2.,loss_absolute=3.,loss_increment=4.,loss=4.)
    result=s.audit_selected_loss(values,'increment')
    assert set(result)==set(s.ARMS)
    values['loss']=4.+1e-12
    with pytest.raises(ValueError,match='alias'):s.audit_selected_loss(values,'increment')
    values['loss']=4.;values['loss_absolute']=3.01
    with pytest.raises(ValueError,match='weight-one'):s.audit_selected_loss(values,'increment')


def test_selected_loss_rounding_tolerance_scales_with_tiny_auxiliaries():
    tiny=terms('increment')
    tiny.update(loss_base=1e-8,distance_absolute_es=2e-8,distance_increment_es=3e-8,
        loss_absolute=3e-8,loss_increment=4e-8,loss=4e-8)
    s.audit_selected_loss(tiny,'increment')
    tiny['loss_absolute']=0.
    with pytest.raises(ValueError,match='weight-one'):s.audit_selected_loss(tiny,'increment')


def test_h1_diagnostic_tolerance_has_fixed_absolute_and_relative_parts():
    assert s.h1_difference(1e-6,0.)['tolerance']==2e-6
    assert s.h1_difference(100.+.01,100.)['tolerance']==2e-6+5e-4*100.
    with pytest.raises(ValueError):s.h1_difference(2.1e-6,0.)
