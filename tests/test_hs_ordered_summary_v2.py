"""JSON serialization regression and v2 dispatch/provenance contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts import summarize_hs_ordered_pilot_v2 as s
from scripts import eval_ca_ordered_paths as producer
from test_hs_ordered_summary_v1 import execution_fixture


def ordered_fixture():
    data=dict(task_version='ca_ordered_paths_development_v1',dataset_role='development_validation',
        optimizer_steps=0,model_selection=False,test_opened=False,bioemu2_rerun=False,
        future_information_given_to_sampler=False,frozen_inputs_verified=True,completed_utc='done',
        panel=producer.panel_spec([1,4,16,64],200),n_windows=200,n_proteins=50,
        args={'checkpoint':'synthetic.pt'},input_hashes={'synthetic.pt':'sha'},
        model_tensor_sha256='tensor',sampling=dict(seed=0,n_paths=12,model_history_frames=4,
            prediction_kind='endpoint',s0_mode='exact_spatial',one_continuous_rollout_per_member=True,
            no_future_reencoding=True,no_bath_reset=True),protocol={},
        source_hashes={p:'source' for p in producer.SOURCES},per_window=['raw'],by_horizon={'h':1})
    # Exercise the actual producer -> JSON -> consumer boundary, rather than
    # passing the producer's tuple-containing Python object directly.
    data=json.loads(json.dumps(data))
    checked=dict(checkpoint='synthetic.pt',checkpoint_sha256='sha',inference_model_tensor_sha256='tensor')
    return data,checked,dict(source_hashes=deepcopy(data['source_hashes']))


class OrderedInputs:
    def __init__(self,data):
        self.registered={'panel':deepcopy(data['panel'])};self.tracked=[]
    def read(self,path):return deepcopy(self.registered)
    def track(self,*args):self.tracked.append(args)


def downstream(monkeypatch):
    calls=[]
    monkeypatch.setattr(s.old.shared,'audit_eval_protocol',lambda *a:calls.append('protocol'))
    monkeypatch.setattr(s.old.shared,'audit_provenance',lambda *a:calls.append('provenance'))
    def rows(raw,panel):
        calls.append('rows')
        assert isinstance(panel['horizons'],list)
        return raw
    monkeypatch.setattr(s.v1.paths,'audit_rows',rows)
    monkeypatch.setattr(producer,'summarize',lambda rows:{'h':1})
    monkeypatch.setattr(s.old.shared,'rows_for',lambda data,where:{'checked':1})
    return calls


def test_real_json_roundtrip_reproduces_v1_bug_and_v2_preserves_other_audits(monkeypatch):
    data,checked,protocol=ordered_fixture();inputs=OrderedInputs(data)
    calls=downstream(monkeypatch);before=deepcopy(data)
    original=s.v1.audit_ordered
    with pytest.raises(ValueError,match='panel incomplete'):
        original(data,s.ARMS[0],checked,protocol,inputs)
    assert not calls
    assert s.audit_ordered(data,s.ARMS[0],checked,protocol,inputs)=={'checked':1}
    assert calls==['protocol','provenance','rows']
    assert data==before and s.v1.audit_ordered is original
    assert len(inputs.tracked)==1+2*len(producer.SOURCES)


@pytest.mark.parametrize('field,value',[
    ('horizons',[1,4,64,16]),('horizons',[1,4,16]),('max_windows',196),
    ('complete_panel_windows',196),('full_panel',False),
    ('expected_full_panel_proteins',49),('name','long192_separate50'),('extra','unregistered'),
])
def test_bad_panel_still_rejected_after_json_roundtrip(monkeypatch,field,value):
    data,checked,protocol=ordered_fixture();data['panel'][field]=value
    calls=downstream(monkeypatch)
    with pytest.raises(ValueError,match='panel incomplete'):
        s.audit_ordered(data,s.ARMS[0],checked,protocol,OrderedInputs(data))
    assert not calls


@pytest.mark.parametrize('mutation,match',[
    ('sidecar','changed registration'),('source','source coverage'),('tensor','loaded model'),
    ('aggregate','raw aggregate'),('future','task incomplete'),('members','sampling contract'),
])
def test_fix_does_not_relax_nonpanel_audits(monkeypatch,mutation,match):
    data,checked,protocol=ordered_fixture();inputs=OrderedInputs(data)
    downstream(monkeypatch)
    if mutation=='sidecar':inputs.registered['panel']['max_windows']=196
    elif mutation=='source':data['source_hashes']={}
    elif mutation=='tensor':data['model_tensor_sha256']='changed'
    elif mutation=='aggregate':data['by_horizon']={'h':2}
    elif mutation=='future':data['future_information_given_to_sampler']=True
    else:data['sampling']['n_paths']=11
    with pytest.raises(ValueError,match=match):
        s.audit_ordered(data,s.ARMS[0],checked,protocol,inputs)


def test_incomplete_execution_stops_before_checkpoints_or_scores(monkeypatch):
    executions,protocol=execution_fixture();executions[s.ARMS[1]]['status']='running';reads=[]
    class Inputs:
        def read(self,path):
            path=str(path);reads.append(path)
            if path==str(s.BASE/'protocol.json'):return protocol
            for arm in s.ARMS:
                if path==str(s.BASE/f'{arm}.execution.json'):return executions[arm]
            return {}
    monkeypatch.setattr(s.old,'Inputs',Inputs)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint before complete jobs'))
    monkeypatch.setattr(s,'audit_ordered',lambda *a:pytest.fail('ordered scores before complete jobs'))
    with pytest.raises(ValueError,match='both arms'):s.build_summary()
    assert not any(any(path.endswith(f'.{kind}.json') for kind in s.JOBS[1:]) for path in reads)


def test_complete_build_uses_new_audit_and_frozen_statistics_without_global_patch(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'BASE',tmp_path/'results')
    monkeypatch.setattr(s,'IMPLEMENTATION',tmp_path/'wrapper')
    monkeypatch.setattr(s,'ARTIFACTS',tmp_path/'artifacts')
    monkeypatch.setattr(s.old,'local',lambda p:Path(p))
    monkeypatch.setattr(s.old,'label',lambda p:str(p))
    val=tmp_path/'val.jsonl';val.write_text('{}\n')
    monkeypatch.setattr(s.old.shared,'VAL',str(val))
    executions,protocol=execution_fixture()
    documents={}
    checked={a:dict(checkpoint=f'{a}.pt',checkpoint_sha256='ck') for a in s.ARMS}
    for arm in s.ARMS:
        for kind in s.JOBS[1:]:
            documents[str(s.BASE/f'{arm}.{kind}.json')]=dict(checkpoint=checked[arm]['checkpoint'],
                checkpoint_sha256='ck',optimizer_steps=0,test_opened=False,metric_semantics={})
    metadata={str(s.IMPLEMENTATION/'protocol.json'):{'source_hashes':{'new':'digest'}},
        str(s.BASE/'summary.protocol.json'):{'bootstrap':{'reps':10000},'scientific_limits':[]},
        str(s.BASE/'protocol.json'):protocol,str(s.V1_REGISTRATION):{}}
    metadata.update({str(s.BASE/f'{a}.execution.json'):executions[a] for a in s.ARMS})
    calls=[]
    class Inputs:
        def __init__(self):self.hashes={Path(s.old.BIOEMU):'bio',Path(s.BASE/'protocol.json'):'protocol'}
        def read(self,path):return deepcopy((metadata if str(path) in metadata else documents)[str(path)])
        def track(self,path,expected=None):self.hashes[Path(path)]='digest';return Path(path)
        def verify(self):calls.append('final_verify')
    monkeypatch.setattr(s.old,'Inputs',Inputs)
    monkeypatch.setattr(torch,'load',lambda *a,**k:{})
    monkeypatch.setattr(s,'audit_implementation',lambda *a:calls.append('implementation'))
    for name in ('audit_protocol','audit_executions','suite_pairing','cross_task_pairing'):
        monkeypatch.setattr(s.v1,name,lambda *a,_name=name:calls.append(_name))
    monkeypatch.setattr(s.v1,'audit_registration_attempts',lambda *a:{'preserved':True})
    monkeypatch.setattr(s.v1,'audit_training',lambda *a:(checked,{'unchanged':True}))
    monkeypatch.setattr(s.old,'audit_reused_gate',lambda *a:{'unchanged':True})
    monkeypatch.setattr(s.old,'expected_suite_keys',lambda *a:{'key'})
    for name in ('audit_sources','audit_provenance'):
        monkeypatch.setattr(s.old.shared,name,lambda *a:None)
    monkeypatch.setattr(s.old.shared,'audit_ensemble',lambda *a:{'key':{}})
    monkeypatch.setattr(s.old,'audit_endpoint',lambda *a:{'key':{}})
    def repaired(*args):calls.append(('repaired',args[1]));return {'key':{}}
    monkeypatch.setattr(s,'audit_ordered',repaired)
    def frozen_broken(*args):pytest.fail('v1 broken ordered audit was dispatched')
    monkeypatch.setattr(s.v1,'audit_ordered',frozen_broken)
    monkeypatch.setattr(s.old,'audit_suite',lambda *a:({'key':{}},{'unchanged':True}))
    monkeypatch.setattr(s.old,'audit_bioemu',lambda *a:({'evaluation_args':{}},{'key':{}}))
    monkeypatch.setattr(s.old,'audit_pairing',lambda *a:0.)
    monkeypatch.setattr(s.old,'bioemu_t2a_evidence',lambda *a:{'unchanged':True})
    def standard(*args):calls.append('standard_numeric');return {'standard':[{'ci95':[0.,1.]}]*341}
    def ordered(*args):calls.append('ordered_numeric');return {'ordered':[{'ci95':[0.,1.]}]*61}
    monkeypatch.setattr(s.numeric,'standard_analysis',standard)
    monkeypatch.setattr(s.numeric,'ordered_analysis',ordered)
    monkeypatch.setattr(s.runner,'verify',lambda *a:calls.append('runner_verify'))
    report=s.build_summary()
    assert s.v1.audit_ordered is frozen_broken
    assert [c for c in calls if isinstance(c,tuple)]==[('repaired',a) for a in s.ARMS]
    assert calls.count('standard_numeric')==calls.count('ordered_numeric')==1
    assert s.v1.count_intervals(report)==402
    assert report['schema_version']==s.VERSION and report['execution_audit']['successful_logical_jobs']==12
    assert report['implementation_repair']['real_v1_summary_failure_claimed'] is False
    assert (s.BASE/'summary_v2.json').is_file() and not (s.BASE/'summary.json').exists()
    with pytest.raises(ValueError,match='overwrite'):s.build_summary()


def test_metadata_register_allows_existing_unread_scores_and_preserves_old_protocol(monkeypatch,tmp_path):
    monkeypatch.setattr(s,'BASE',tmp_path/'results')
    monkeypatch.setattr(s,'IMPLEMENTATION',tmp_path/'wrapper')
    monkeypatch.setattr(s.old,'REPO',tmp_path)
    monkeypatch.setattr(s.old,'local',lambda p:Path(p) if Path(p).is_absolute() else tmp_path/p)
    monkeypatch.setattr(s.old,'label',lambda p:str(s.old.local(p).relative_to(tmp_path)))
    monkeypatch.setattr(s.v1,'IMPLEMENTATION',tmp_path/'v1_wrapper')
    monkeypatch.setattr(s,'V1_REGISTRATION',tmp_path/'v1_wrapper'/'protocol.json')
    s.BASE.mkdir();s.V1_REGISTRATION.parent.mkdir()
    s.V1_REGISTRATION.write_text('FROZEN OLD REGISTRATION')
    (s.BASE/'joint_path.ordered.json').write_text('INVALID JSON: MUST NEVER BE PARSED')
    for source in (s.SCRIPT,s.TEST):
        p=s.old.local(source);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source')
    reads=[]
    class Inputs:
        def __init__(self):self.hashes={}
        def track(self,path,expected=None):
            path=s.old.local(path);self.hashes[path]=expected or 'digest';return path
        def read(self,path):
            reads.append(str(path));self.track(path)
            if path==s.V1_REGISTRATION:return dict(s.v1.contract(),source_hashes={})
            if path==s.BASE/'summary.protocol.json':return dict(s.v1.registration.contract(),source_hashes={})
            pytest.fail('unexpected metadata registration read: '+str(path))
        def verify(self):pass
    monkeypatch.setattr(s.old,'Inputs',Inputs)
    record=s.register()
    assert record['score_file_existence'][str(s.BASE/'joint_path.ordered.json')] is True
    assert record['score_json_parsed_during_registration'] is False
    assert all(not p.endswith('.ordered.json') for p in reads)
    assert s.V1_REGISTRATION.read_text()=='FROZEN OLD REGISTRATION'
    assert record['n_paired_intervals']==402
    with pytest.raises(ValueError,match='overwrite'):s.register()


def test_v2_contract_preserves_scientific_contract_and_static_failure_history():
    original=s.v1.contract();revised=s.contract()
    for key in original:
        if key not in ('schema_version','output'):assert revised[key]==original[key],key
    assert revised['implementation_repair']['statistical_definitions_changed'] is False
    assert revised['implementation_repair']['real_v1_summary_failure_claimed'] is False
    assert s.numeric is s.v1.numeric
