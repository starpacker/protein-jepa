"""Synthetic-only label, estimand and pre-registration contracts."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import hs_contact_pilot_statistics_v1 as s
from scripts import register_hs_contact_pilot_statistics_v1 as registration
from test_hs_target_summary import synthetic_scores
from test_hs_ordered_pilot_statistics_v1 import fixture_row


def intervals(value):
    if isinstance(value,dict):
        return int('ci95' in value)+sum(intervals(v) for v in value.values())
    if isinstance(value,list):
        return sum(intervals(v) for v in value)
    return 0


def standard_mean_count(value, in_means=False):
    if isinstance(value,dict):
        if 'ci95' in value:
            return 0
        scopes = ('arm_means','bioemu2_means','persistence_means','target_means','geometry_vs_anchor','contact_counts')
        return sum(standard_mean_count(v,in_means or k in scopes) for k,v in value.items())
    return int(in_means and type(value) is float)


def ordered_rows():
    rows = {arm:{} for arm in s.ARMS}
    for i in range(50):
        for h in (1,4,16,64):
            key = (f'p{i:02}','x',(320.,348.,379.,413.,450.)[i//10],7,h)
            base = fixture_row()
            base.update(target_frame=7+h,horizon_ps=h*1000)
            for role in s.paths.ARMS:
                metrics = base['arms'][role]['metrics']
                metrics['scalars'] = dict.fromkeys((*s.SCORES,*s.paths.DESCRIPTIVE),0.)
                metrics['scalars']['distance_acf_corr_n_lags'] = 0
                if role == 'reference_MD':
                    for metric in s.SCORES: metrics['scalars'][metric] = None
                metrics['lag_statistics']['prediction'] = dict(aggregate=[dict(
                    acf=None,msd=0. if lag<=h else None,n_pairs=max(h+1-lag,0),n_acf_pairs=0)
                    for lag in s.paths.LAGS])
            for arm in s.ARMS:
                row = deepcopy(base)
                for metric in s.SCORES:
                    row['arms']['forecast']['metrics']['scalars'][metric] = (
                        (i+1)*({1:-9.,4:-9.,16:1.,64:3.}[h]) if arm=='population_u' else 0.)
                rows[arm][key] = row
    return rows


def test_standard_exact_frozen_equivalence_all341_intervals_and_labels():
    ensemble,endpoints,baseline = synthetic_scores()
    source_arms = s.ordered.endpoint.ARMS
    ensembles = {arm:ensemble[slot] for arm,slot in zip(s.ARMS,source_arms)}
    endpoints = {arm:endpoints[slot] for arm,slot in zip(s.ARMS,source_arms)}
    before = deepcopy((ensembles,endpoints,baseline))
    result = s.standard_analysis(ensembles,endpoints,baseline)
    expected = s.rename_ordered(s.ordered.standard_analysis(
        s.ordered_slots(ensembles),s.ordered_slots(endpoints),baseline))
    assert result == expected and intervals(result)==341
    assert standard_mean_count(result)==670
    assert (ensembles,endpoints,baseline)==before
    assert s.CONTRAST in result['seven_metrics']
    assert set(result['seven_metrics']['arm_means'])==set(s.ARMS)
    assert result['secondary_long_horizon_endpoint_UES']['candidate_minus_reference']==51.
    serialized = json.dumps(result)
    assert 'joint_path_vs_marginal_path' not in serialized and '"marginal_path"' not in serialized
    assert s.ordered.ARMS == ('joint_path','marginal_path')


def test_ordered_primary_sign_independent_bootstrap_and_null_support():
    rows = ordered_rows()
    before = deepcopy(rows)
    result = s.ordered_analysis(rows)
    expected = s.rename_ordered(s.ordered.ordered_analysis(s.ordered_slots(rows)))
    assert result==expected and rows==before and intervals(result)==61
    primary = result['primary_long_horizon_joint_path_UES']
    delta = 2*np.arange(1,51,dtype=float)
    generator = np.random.default_rng(20261101)
    interval = np.quantile(generator.choice(delta,size=(10000,50),replace=True).mean(1),[.025,.975])
    assert primary['candidate_minus_reference']==51. and primary['improvement']==-51.
    np.testing.assert_allclose(primary['ci95'],interval,rtol=0,atol=1e-12)
    assert primary['n_windows']==100 and primary['n_proteins']==50
    scores = result['ordered_path_scores']
    assert set(scores['comparisons'])=={s.CONTRAST,'population_u_vs_persistence','plugin_vs_persistence'}
    for arm in s.ARMS:
        h1 = scores['descriptive'][arm]['h1']
        assert h1['forecast']['lag_diagnostics']['32']['msd']['complete_panel_mean'] is None
        assert h1['persistence']['scalars']['walk_rmsf_corr']['complete_panel_mean'] is None
        assert h1['reference_MD']['scalars'][s.SCORES[0]]['n_valid_windows']==0


def test_label_mapping_is_explicit_nonmutating_and_rejects_collisions():
    source = {'joint_path':'joint_path','marginal_path_vs_persistence':[1.,None]}
    result = s.rename_ordered(source)
    assert result == {'population_u':'population_u','plugin_vs_persistence':[1.,None]}
    assert source['joint_path']=='joint_path'
    with pytest.raises(ValueError,match='collision'):
        s.rename_ordered({'joint_path':1,'population_u':2})
    with pytest.raises(ValueError,match='arms'):
        s.ordered_slots({'joint_path':{},'marginal_path':{}})


def test_missing_window_never_becomes_intersection():
    rows = ordered_rows()
    rows['plugin'].pop(next(iter(rows['plugin'])))
    with pytest.raises(ValueError,match='keys differ'):
        s.ordered_analysis(rows)


def test_contract_preserves_all_forecast_definitions_and_counts():
    old = registration.ordered_registration.contract()
    new = registration.contract()
    for key in ('primary','bootstrap','groups','ordered_path_scores','ordered_intervals','primary_intervals',
        'standard_intervals','total_paired_intervals','standard_endpoint_secondary','standard_endpoint_scores',
        'endpoint_units','endpoint_estimator','historical_seven_metrics','historical_metric_directions',
        'historical_reference','canonical_ca_geometry','geometry_directions','standard_descriptive_means',
        'suite','ordered_descriptive'):
        assert new[key]==old[key],key
    assert new['total_paired_intervals']==402 and new['standard_descriptive_means']==670
    assert new['suite_descriptive_means']==1700
    assert new['arms']==['population_u','plugin']
    assert new['training']['only_arm_arguments']==['contact_estimator','output']
    assert new['training']['kernel_parameter_tensors']==73
    assert 'Both arms' in new['training']['coordinate']
    assert any('not held-out' in item for item in new['scientific_limits'])
    assert registration.ordered_registration.contract()==old


def test_registration_pins_sources_without_score_reads_and_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(registration.old,'REPO',tmp_path)
    monkeypatch.setattr(registration,'SOURCES',('source.py',))
    (tmp_path/'source.py').write_text('# fixture\n')
    result = registration.register()
    output = tmp_path/registration.BASE/'summary.protocol.json'
    assert json.loads(output.read_text())==result
    assert result['source_hashes']=={'source.py':registration.old.sha('source.py')}
    with pytest.raises(ValueError,match='overwrite'):
        registration.register()


@pytest.mark.parametrize('suffix',['population_u.execution.json','plugin.ordered.json'])
def test_registration_rejects_started_or_scored_run(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(registration.old,'REPO',tmp_path)
    previous = tmp_path/registration.BASE/suffix
    previous.parent.mkdir(parents=True)
    previous.write_text('not parsed')
    with pytest.raises(ValueError):
        registration.register()
    assert not (tmp_path/registration.BASE/'summary.protocol.json').exists()
