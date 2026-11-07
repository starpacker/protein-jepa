"""Synthetic JSON and small rendered figures only; no trial outputs read."""
import csv
import json
import math

import pytest

from run_scripts import render_stage_a_relpos_adaptation_v1 as report


def scores():
    rows=[]
    for i in range(100):
        values={}
        for j,key in enumerate(report.SCORE_KEYS):
            value=.001*(i+1)*(1+j*.1)
            values[key]=dict(log_mse=value,mae_angstrom=value*10,mse_angstrom2=value*20,
                rmse_angstrom=math.sqrt(value*20),n_pairs=10+i,negative_count=i%2)
        rows.append(dict(cache_row_index=400+i,metadata=dict(protein='p'+str(i),trajectory='traj'+str(i),
            horizon=(1,4,16,64)[i%4],temperature_k=320.),n_pairs=10+i,scores=values))
    aggregates={}
    for group in report.GROUPS:
        selected=rows if group=='overall' else [r for r in rows if r['metadata']['horizon']==int(group[1:])]
        aggregates[group]={}
        for key in report.SCORE_KEYS:
            value={k:sum(r['scores'][key][k] for r in selected)/len(selected) for k in report.METRICS[:3]}
            value.update(rmse_angstrom=math.sqrt(value['mse_angstrom2']),n_proteins=len(selected),
                n_pairs=sum(r['n_pairs'] for r in selected),negative_count=sum(r['scores'][key]['negative_count'] for r in selected),
                mean_protein_negative_fraction=sum(r['scores'][key]['negative_count']/r['n_pairs'] for r in selected)/len(selected))
            aggregates[group][key]=value
    bootstrap=dict(primary='future_H/log_mse',draws=10000,contrasts={
        contrast:{role:{metric:dict(difference=.01,interval95=[-.02,.04]) for metric in report.METRICS}
                  for role in report.ROLES} for contrast in ('zero-pca','local-pca')})
    return dict(per_protein=rows,aggregates=aggregates,paired_bootstrap=bootstrap)


def learning():
    result={}
    for arm in report.ARMS:
        result['train_'+arm]=[dict(step=i,losses={'loss/total':1/i,'loss/base':.5/i},
            global_grad_norm_before_clip=2.,encoder_grad_norm_before_clip=1.,ranking_eligible_counts=list(range(7))) for i in range(1,501)]
        result['fit_'+arm]=[dict(step=i,cache_row_index=(i-1)%400,log_mse=1/i) for i in range(1,501)]
    return result


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'PRODUCER',tmp_path/'trial')
    monkeypatch.setattr(report,'VERIFIER',tmp_path/'verifier')
    monkeypatch.setattr(report,'OUTPUT',tmp_path/'rendered')
    for p in (report.PRODUCER,report.VERIFIER,tmp_path/'run_scripts',tmp_path/'tests'):p.mkdir()
    for p in (tmp_path/'run_scripts/render_stage_a_relpos_adaptation_v1.py',tmp_path/'tests/test_render_stage_a_relpos_adaptation_v1.py'):
        p.write_text('# synthetic source identity\n')
    report.write(report.PRODUCER/'protocol.json',{'synthetic':True});main=report.sha(report.PRODUCER/'protocol.json')
    checked={str(report.PRODUCER/'protocol.json'):main};logs=learning();result=scores()
    for phase in report.PHASES:
        folder=report.PRODUCER/phase;folder.mkdir()
        started=dict(pid=100,command=['synthetic'],started_utc='2026-01-01T00:00:00+00:00',protocol_sha256=main,phase=phase)
        report.write(folder/'started.json',started)
        value=dict(status='completed',phase=phase,protocol_sha256=main,
            resources=dict(host_peak_rss_bytes=1024,gpu_peak_allocated_bytes=2048),artifacts={})
        if phase=='score':value.update(result)
        if phase in logs:
            path=folder/'updates.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in logs[phase]))
            value['artifacts']['updates']=dict(path=str(path),sha256=report.sha(path))
            checked[str(path)]=report.sha(path)
        report.write(folder/'result.json',value)
        execution=dict(started,status='completed',returncode=0,completed_utc='2026-01-01T00:00:01+00:00',seconds=1.,
            result=dict(path=str(folder/'result.json'),sha256=report.sha(folder/'result.json')))
        report.write(folder/'execution.json',execution)
        for name in ('started.json','result.json','execution.json'):checked[str(folder/name)]=report.sha(folder/name)
    report.write(report.VERIFIER/'protocol.json',dict(main_protocol_sha256=main));vs=report.sha(report.VERIFIER/'protocol.json')
    started=dict(pid=200,command=['synthetic verifier'],started_utc='2026-01-02T00:00:00+00:00',protocol_sha256=vs)
    report.write(report.VERIFIER/'started.json',started)
    report.write(report.VERIFIER/'verification.json',dict(status='passed',rows=100,groups=5,arms=3,
        readout_roles=3,bootstrap_draws=10000,checked_hashes=checked))
    report.write(report.VERIFIER/'execution.json',dict(started,status='completed',returncode=0,
        completed_utc='2026-01-02T00:00:01+00:00',seconds=1.,verification_sha256=report.sha(report.VERIFIER/'verification.json')))
    return main,vs


def test_complete100_rows_50aggregates_and_mean_mse_root_convention(tmp_path,monkeypatch):
    assert report.VERIFIER.name=='stage_a_relpos_verification_v2'
    main,vs=fixture(tmp_path,monkeypatch);inputs=report.Inputs()
    _,results,executions,logs,_=report.verified_inputs(inputs,main,vs)
    proteins,aggregates,resources=report.tables(results['score'],results,executions)
    assert len(proteins)==100 and len(aggregates)==50 and len(resources)==9
    assert [p['cache_row_index'] for p in proteins]==list(range(400,500))
    value=next(r for r in aggregates if r['group']=='overall' and r['role']=='pca/future_H')
    assert value['rmse_angstrom']==math.sqrt(value['mse_angstrom2'])
    assert value['rmse_angstrom']!=sum(r['scores']['pca/future_H']['rmse_angstrom'] for r in results['score']['per_protein'])/100
    text=report.markdown(results['score'],aggregates,resources,main,vs)
    assert 'not a forecast' in text and 'PCA warm start' in text and 'Arm-local EMA teacher losses cannot rank' in text
    assert 'observed_H_substitution' in text and 'fixed secondary' in text and 'CA_persistence' in text
    assert '../stage_a_relpos_verification_v1/registration.execution.json' in text
    assert 'before SCORE but after training' in text and 'NumPy float32 JSON serialization' in text
    assert 'preserving the exact same bound and numerical formulas' in text
    inputs.finish()


@pytest.mark.parametrize('fault',['failed_verifier','changed_used_log','uncovered_score'])
def test_hash_linked_verification_gate_rejects_faults(tmp_path,monkeypatch,fault):
    main,vs=fixture(tmp_path,monkeypatch)
    if fault=='failed_verifier':
        p=report.VERIFIER/'execution.json';v=json.loads(p.read_text());v['returncode']=1;p.write_text(json.dumps(v))
    elif fault=='changed_used_log':
        p=report.PRODUCER/'train_pca/updates.jsonl';p.write_text(p.read_text()+'{}\n')
    else:
        p=report.VERIFIER/'verification.json';v=json.loads(p.read_text());v['checked_hashes'].pop(str(report.PRODUCER/'score/result.json'));p.write_text(json.dumps(v))
        e=report.VERIFIER/'execution.json';v=json.loads(e.read_text());v['verification_sha256']=report.sha(p);e.write_text(json.dumps(v))
    with pytest.raises(ValueError):report.verified_inputs(report.Inputs(),main,vs)


def test_one_exclusive_synthetic_render_all_points_outputs_and_provenance(tmp_path,monkeypatch):
    main,vs=fixture(tmp_path,monkeypatch)
    result=report.run(main,vs)
    assert result['status']=='completed' and result['returncode']==0
    with (report.OUTPUT/'per_protein.csv').open() as f:assert len(list(csv.DictReader(f)))==100
    for name in ('adaptation_updates','readout_updates'):
        with (report.OUTPUT/(name+'.csv')).open() as f:assert len(list(csv.DictReader(f)))==1500
    assert len(list(report.OUTPUT.glob('*.png')))==len(list(report.OUTPUT.glob('*.pdf')))==3
    manifest=json.loads((report.OUTPUT/'report.json').read_text())
    assert manifest['rows']==100 and all(report.sha(p)==digest for p,digest in manifest['outputs'].items())
    before=report.sha(report.OUTPUT/'execution.json')
    with pytest.raises(ValueError,match='no retry'):report.run(main,vs)
    assert report.sha(report.OUTPUT/'execution.json')==before


def test_failure_witness_preserved_without_retry(tmp_path,monkeypatch):
    main,vs=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(report,'verified_inputs',lambda *args:(_ for _ in ()).throw(ValueError('synthetic failure')))
    with pytest.raises(ValueError,match='synthetic failure'):report.run(main,vs)
    execution=json.loads((report.OUTPUT/'execution.json').read_text())
    assert execution['returncode']==1 and execution['failed_phase']=='verify'
    assert not (report.OUTPUT/'report.json').exists()
    with pytest.raises(ValueError,match='no retry'):report.run(main,vs)
