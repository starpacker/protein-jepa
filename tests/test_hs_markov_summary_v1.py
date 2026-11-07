import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import summarize_hs_markov_pilot_v1 as summary
from test_hs_target_summary import synthetic_scores


def values():
    ensemble,endpoint,baseline=synthetic_scores()
    return ({a:ensemble[s] for a,s in summary.SLOTS.items()},
            {a:endpoint[s] for a,s in summary.SLOTS.items()},baseline)


def test_primary_is_c0_minus_hs_long_horizons_before_protein_bootstrap():
    result=summary.analysis(*values())
    primary=result['primary_long_horizon_coordinate_UES']
    expected=2*np.arange(1,51,dtype=float)
    draws=np.random.default_rng(20261101).choice(expected,size=(10000,50),replace=True)
    np.testing.assert_allclose(primary['ci95'],np.percentile(draws.mean(1),[2.5,97.5]),rtol=0,atol=1e-12)
    assert primary['candidate_minus_reference']==expected.mean()
    assert primary['improvement']==-expected.mean() and primary['better']=='lower'
    assert primary['n_windows']==100 and primary['n_proteins']==50
    assert [r['candidate_minus_reference'] for r in primary['per_protein']]==expected.tolist()


def test_all341_contrasts670_means_have_roles_and_no_old_target_aliases():
    result=summary.analysis(*values())
    pairs=[];mean_count=0
    def walk(value,in_mean=False):
        nonlocal mean_count
        if isinstance(value,dict):
            assert not {'endpoint_vs_path_mixture','path_mixture','endpoint'} & value.keys()
            if 'ci95' in value and 'candidate_minus_reference' in value:
                pairs.append(value);return
            for k,v in value.items():
                walk(v,in_mean or k in ('arm_means','bioemu2_means','persistence_means','target_means','geometry_vs_anchor','contact_counts'))
        elif isinstance(value,float) and in_mean:mean_count+=1
    walk(result)
    assert len(pairs)==341 and mean_count==670
    for field in ('seven_metrics','endpoint_scores','canonical_ca_geometry'):
        assert set(result[field]['markov_c0_vs_trained_hs'])==set(summary.stats.GROUPS)
    assert result['seven_metrics']['markov_c0_vs_trained_hs']['h16']['rmsf_corr']['better']=='higher'
    assert result['canonical_ca_geometry']['markov_c0_vs_trained_hs']['overall']['ca_bond_mean_angstrom']['better'] is None


def test_reused_hs_bioemu_and_persistence_statistics_keep_original_values():
    ens,ends,bio=synthetic_scores()
    original=summary.stats.analysis(ens,ends,bio)
    result=summary.analysis(*values())
    assert result['seven_metrics']['bioemu2_comparisons']['trained_hs']==original['seven_metrics']['bioemu2_comparisons']['path_mixture']
    assert result['endpoint_scores']['persistence_comparisons']['trained_hs']==original['endpoint_scores']['persistence_comparisons']['path_mixture']
    assert set(result['seven_metrics']['arm_means'])==set(summary.ARMS)


def test_no_missing_support_or_nonfinite_fallback():
    ens,ends,bio=values()
    key=next(k for k in bio if k[4]==64)
    del ends['markov_c0'][key]
    with pytest.raises(ValueError,match='complete standard200'):summary.analysis(ens,ends,bio)
    ens,ends,bio=values();ends['markov_c0'][key]['scores']['model']['coordinate_energy_score']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):summary.analysis(ens,ends,bio)


def execution_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(summary.stats,'REPO',tmp_path)
    base=tmp_path/summary.BASE;base.mkdir(parents=True)
    artifact=tmp_path/summary.runner.ARTIFACTS;artifact.mkdir(parents=True)
    checkpoint=artifact/'checkpoint_latest.pt';checkpoint.write_bytes(b'checkpoint')
    commands={k:['python',k] for k in summary.runner.JOBS}
    jobs=[]
    for i,kind in enumerate(summary.runner.JOBS):
        log=base/f'markov.{kind}.log';log.write_text('actual worker output')
        job=dict(kind=kind,command=commands[kind],pid=100+i,returncode=0,
            completed_utc='2026-11-02T01:00:00+00:00',log=str(summary.BASE/f'markov.{kind}.log'),
            log_sha256=summary.stats.sha(log))
        if kind!='train':
            output=base/f'markov.{kind}.json';output.write_text('{}')
            job['output_sha256']=summary.stats.sha(output)
            if kind=='ensemble':
                sidecar=output.with_suffix('.component.pt');sidecar.write_bytes(b'component')
                job['component_sha256']=summary.stats.sha(sidecar)
        jobs.append(job)
    execution=dict(status='completed',completed_utc='2026-11-02T01:00:00+00:00',frozen_files_verified=True,
        protocol_sha256=summary.PROTOCOL_SHA,runner_pid=99,jobs=jobs,latest_sha256=summary.stats.sha(checkpoint))
    path=base/'execution.json';path.write_text(json.dumps(execution))
    return dict(commands=commands),execution,path


def test_actual_three_jobs_hashes_and_sidecar_required(tmp_path,monkeypatch):
    protocol,execution,path=execution_fixture(tmp_path,monkeypatch)
    audited=summary.audit_new_execution(protocol,summary.stats.Inputs())
    assert audited==execution
    output=path.parent/'markov.endpoint.json';output.write_text('{"changed": true}')
    with pytest.raises(ValueError,match='SHA256'):summary.audit_new_execution(protocol,summary.stats.Inputs())


@pytest.mark.parametrize('change',['running','failed','missing','exit','bool_exit','no_pid','unregistered_command'])
def test_incomplete_or_failed_execution_rejected_before_score_identity_read(tmp_path,monkeypatch,change):
    protocol,execution,path=execution_fixture(tmp_path,monkeypatch)
    if change in ('running','failed'):execution['status']=change
    elif change=='missing':execution['jobs'].pop()
    elif change=='exit':execution['jobs'][-1]['returncode']=1
    elif change=='bool_exit':execution['jobs'][-1]['returncode']=False
    elif change=='no_pid':execution['jobs'][-1]['pid']=None
    else:execution['jobs'][-1]['command']=['different']
    path.write_text(json.dumps(execution))
    original=summary.stats.sha
    def no_scores(p):
        assert Path(p).name=='execution.json','premature checkpoint/score read'
        return original(p)
    monkeypatch.setattr(summary.stats,'sha',no_scores)
    with pytest.raises(ValueError):summary.audit_new_execution(protocol,summary.stats.Inputs())


def test_build_stops_before_models_and_scores_while_new_jobs_run(tmp_path,monkeypatch):
    import torch
    protocol,execution,path=execution_fixture(tmp_path,monkeypatch)
    execution['status']='running';path.write_text(json.dumps(execution))
    for name,value in (('protocol.json',protocol),('summary.protocol.json',{})):
        (path.parent/name).write_text(json.dumps(value))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('model loaded before complete jobs'))
    monkeypatch.setattr(summary,'read_scores',lambda *a,**k:pytest.fail('scores parsed before complete jobs'))
    with pytest.raises(ValueError,match='incomplete'):summary.build_summary()
    assert not (path.parent/'summary.json').exists()


def test_registration_only_metadata_pins_source_and_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(summary.stats,'REPO',tmp_path)
    base=tmp_path/summary.BASE;base.mkdir(parents=True)
    primary=dict(metric='coordinate_energy_score',horizons=[16,64],contrast='Markov_C0 minus trained_HS')
    for name,value in (('protocol.json',dict(schema_version=summary.runner.VERSION,statistics=primary)),('decision.json',dict(primary=primary))):
        (base/name).write_text(json.dumps(value))
    monkeypatch.setattr(summary,'PROTOCOL_SHA',summary.stats.sha(base/'protocol.json'))
    monkeypatch.setattr(summary.runner,'DECISION_SHA',summary.stats.sha(base/'decision.json'))
    for p in (summary.SCRIPT,summary.TEST,Path('frozen.py')):
        target=tmp_path/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('# fixture\n')
    monkeypatch.setattr(summary,'PINNED',{'frozen.py':summary.stats.sha('frozen.py')})
    original=summary.stats.Inputs.read
    def metadata_only(self,p):
        assert Path(p).name in ('protocol.json','decision.json')
        return original(self,p)
    monkeypatch.setattr(summary.stats.Inputs,'read',metadata_only)
    summary.register()
    registered=json.loads((base/'summary.protocol.json').read_text())
    assert registered['source_hashes'][str(summary.SCRIPT)]==summary.stats.sha(summary.SCRIPT)
    assert registered['n_paired_comparisons']==341 and registered['n_descriptive_means']==670
    with pytest.raises(ValueError,match='existing'):summary.register()


def test_contract_preserves_limits_and_no_full_suite_claim():
    c=summary.contract()
    assert c['contrast']=='markov_c0 minus trained_hs'
    assert c['bootstrap']['reps']==10000 and c['bootstrap']['seed']==20261101
    assert c['n_samples']==12 and c['primary']['horizons']==[16,64]
    assert 'No new full temperature suite' in c['suite']
    assert 'historical coordinate byte identity' in c['historical_data_limit']
    assert any('CI crossing zero is not equivalence' in s for s in c['limits'])
