"""CPU synthetic contracts for the pre-registered ordered-path summary."""
import copy
import hashlib
import runpy

import pytest
import torch

from scripts import summarize_ca_ordered_paths as summary
from scripts.eval_ca_ordered_paths import DEFAULT_HORIZONS, panel_spec, score_window, temporal_controls


@pytest.fixture(scope='module')
def rows():
    fixture=runpy.run_path(str(summary.REPO/'tests/test_ca_ordered_paths.py'))['fixture']
    torch.set_num_threads(1)
    result=[]
    for h in DEFAULT_HORIZONS:
        model,window=fixture(h)
        result.append(score_window(model,window,n_samples=12))
    return result


def test_complete_horizon_prefix_and_continuous_single_reference_contract(rows):
    panel=panel_spec(DEFAULT_HORIZONS,4)
    assert summary.audit_rows(rows,panel)==rows
    full=[]
    for i in range(50):
        for original in rows:
            row=copy.deepcopy(original)
            row.update(protein=f'p{i}',trajectory=f'r{i}',temperature_k=(320.,348.,379.,413.,450.)[i%5])
            seed=int(hashlib.sha256(f"p{i}|r{i}|7|{row['horizon']}".encode()).hexdigest()[:12],16)%(2**31-1)
            row['sampling_seed']=seed
            row['temporal_controls']=temporal_controls(torch.zeros(1,row['horizon']+1,1,3),seed)[1]
            full.append(row)
    summary.audit_rows(full,panel_spec(DEFAULT_HORIZONS))
    with pytest.raises(RuntimeError,match='incomplete'):
        summary.audit_rows(full[:-1],panel_spec(DEFAULT_HORIZONS))


@pytest.mark.parametrize('bad',('duplicate','future_input','reset','reference_replicated','time','lag_join','reference_score'))
def test_contract_tampering_fails_closed(rows,bad):
    wrong=copy.deepcopy(rows)
    row=wrong[-1]
    if bad=='duplicate':wrong[-1]=wrong[0]
    elif bad=='future_input':row['continuous_path']['future_reencodings']=1
    elif bad=='reset':row['continuous_path']['bath_resets']=1
    elif bad=='reference_replicated':row['arms']['reference_MD']['n_paths']=12
    elif bad=='time':row['time_grid']['n_frames']+=1
    elif bad=='lag_join':row['arms']['forecast']['metrics']['lag_statistics']['prediction']['aggregate'][0]['n_pairs']+=1
    elif bad=='reference_score':row['arms']['reference_MD']['metrics']['scalars'][summary.SCORES[0]]=0.
    with pytest.raises((ValueError,RuntimeError)):
        summary.audit_rows(wrong,panel_spec(DEFAULT_HORIZONS,4))


def test_paired_scores_are_equal_protein_reproducible_and_allow_negative_scores():
    def row(protein,a,b):
        return dict(protein=protein,arms={name:dict(metrics=dict(scalars={summary.SCORES[0]:v}))
            for name,v in (('forecast',a),('persistence',b))})
    data=[row('a',-2.,-1.),row('a',0.,1.),row('b',9.,10.)]
    result=summary.paired(data,'forecast','persistence',summary.SCORES[0])
    assert result['candidate']==4. and result['reference']==5.
    assert result['candidate_minus_reference']==-1. and result['ci95']==[-1.,-1.]
    assert result['n_proteins']==2 and result['n_windows']==3
    assert result==summary.paired(data[::-1],'forecast','persistence',summary.SCORES[0])
    assert not {'passed','equivalent','noninferior'} & result.keys()
    with pytest.raises(ValueError,match='only joint'):
        summary.paired(data,'forecast','persistence','walk_rmsf_corr')


def test_constant_persistence_corr_is_excluded_with_raw_retained_and_ues_unchanged(rows):
    data=copy.deepcopy(rows)
    before=summary.summarize_panel(data)
    for row in data:row['arms']['persistence']['metrics']['scalars']['walk_rmsf_corr']=.21999
    after=summary.summarize_panel(data)
    assert set(after)=={'1','4','16','64'}
    for h in after:
        assert after[h]['comparisons']==before[h]['comparisons']
        assert after[h]['descriptive']['persistence']['walk_rmsf_corr']==dict(
            mean=None,n_valid_proteins=0,n_valid_windows=0)
        raw=after[h]['raw_persistence_walk_rmsf_corr']
        assert raw['aggregate']==dict(mean=.21999,n_valid_proteins=1,n_valid_windows=1)
        assert raw['per_window'][0]['value']==.21999
        assert len(after[h]['comparisons'])==3
        assert all('ci95' not in v for a in after[h]['descriptive'].values() for v in a.values())
    assert after['1']['descriptive']['forecast']['walk_rmsf_mae']['mean'] is None


def test_changed_inputs_sealed_paths_and_output_overwrite_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(summary,'REPO',tmp_path)
    monkeypatch.setattr(summary,'ROOT',tmp_path)
    file=tmp_path/'input.json';file.write_text('{}')
    inputs=summary.Inputs();inputs.read(file)
    file.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):
        inputs.verify()
    for name in ('test.jsonl','bioemu2.json'):
        with pytest.raises(ValueError,match='sealed'):
            summary.local(tmp_path/name)
    out=tmp_path/'summary.json';out.write_text('preserved')
    with pytest.raises(ValueError,match='overwrite'):
        summary.build_summary(out)
    assert out.read_text()=='preserved'
