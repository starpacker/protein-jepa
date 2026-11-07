from copy import deepcopy

import pytest

from scripts import hs_ordered_pilot_statistics_v1 as s


def test_descriptive_weights_proteins_equally_and_reports_missing_support():
    keys = [('a', 0), ('a', 1), ('b', 0)]
    rows = {keys[0]: 0., keys[1]: 2., keys[2]: 7.}
    out = s.descriptive(rows, keys, lambda x: x)
    assert out['complete_panel_mean'] == 4.
    rows[keys[0]] = None
    out = s.descriptive(rows, keys, lambda x: x)
    assert out['complete_panel_mean'] is None
    assert out['available_support_mean'] == 4.5
    assert out['n_valid_windows'] == 2
    assert out['missing_window_keys'] == [list(keys[0])]


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True])
def test_nonfinite_and_boolean_metrics_fail(value):
    with pytest.raises(ValueError, match='nonfinite'):
        s.descriptive({('a', 0): value}, [('a', 0)], lambda x: x)


def test_descriptive_empty_duplicate_or_missing_window_fails():
    for keys in ([], [('a', 0), ('a', 0)], [('b', 0)]):
        with pytest.raises(ValueError):
            s.descriptive({('a', 0): 0.}, keys, lambda x: x)


def test_deterministic_persistence_correlation_stays_undefined():
    row = dict(arms={'persistence': dict(metrics=dict(scalars={'walk_rmsf_corr': 1.}))})
    assert s.scalar(row, 'persistence', 'walk_rmsf_corr') is None
    assert row['arms']['persistence']['metrics']['scalars']['walk_rmsf_corr'] == 1.


def test_all_missing_is_null_not_zero():
    result = s.descriptive({('a', 0): None}, [('a', 0)], lambda x: x)
    assert result['complete_panel_mean'] is result['available_support_mean'] is None
    assert result['n_valid_proteins'] == result['n_valid_windows'] == 0


def test_endpoint_slot_names_cannot_escape_and_primary_is_secondary():
    original = dict(primary_long_horizon_coordinate_UES={'candidate': 2.},
        seven_metrics={'arm_means': {'endpoint': 1., 'path_mixture': 2.},
                       'endpoint_vs_path_mixture': {'overall': 3.}})
    renamed = s.rename_standard(original)
    assert 'primary_long_horizon_coordinate_UES' not in renamed
    assert renamed['secondary_long_horizon_endpoint_UES']['candidate'] == 2.
    assert renamed['seven_metrics']['arm_means'] == {'joint_path': 1., 'marginal_path': 2.}
    assert s.CONTRAST in renamed['seven_metrics']
    assert original['seven_metrics']['arm_means']['endpoint'] == 1.


def test_paired_estimand_rejects_missing_instead_of_intersection():
    candidate = {('a', 'x', 320., 7, 16): 1.}
    reference = {}
    with pytest.raises(ValueError, match='incomplete'):
        s.endpoint.paired(candidate, reference, list(candidate), lambda x: x, lambda x: x, 'lower')


def test_paired_sign_protein_weighting_and_order_are_fixed():
    keys = [('a', 'x', 320., 7, h) for h in (16, 64)] + [('b', 'x', 320., 7, 16)]
    candidate = dict(zip(keys, (0., 2., 7.)))
    reference = dict.fromkeys(keys, 2.)
    result = s.endpoint.paired(candidate, reference, keys, lambda x: x, lambda x: x, 'lower')
    permuted = s.endpoint.paired(candidate, reference, list(reversed(keys)), lambda x: x, lambda x: x, 'lower')
    assert result == permuted
    assert result['candidate'] == 4.
    assert result['candidate_minus_reference'] == 2.
    assert result['improvement'] == -2.


def fixture_row():
    row = dict(target_frame=23, context_frames=list(range(8)), horizon_ps=16000,
        valid_residues=8, time_grid={'step_ps': 1000}, n_forecast_paths=12,
        n_reference_paths=1, sampling_seed=3, temporal_controls={},
        input_fingerprints=dict(observed='o', reference='r', forecast_CA_path='f'), arms={})
    for role in s.paths.ARMS:
        row['arms'][role] = dict(metrics=dict(conventions={'pairs': [[0, 4]]},
            lag_statistics={'reference': {'n': 1}}, scalars={}))
    return row


@pytest.mark.parametrize('mutation', ['reference', 'seed', 'persistence', 'convention'])
def test_ordered_pairing_requires_same_reference_and_sampling_contract(mutation):
    key = ('a', 'x', 320., 7, 16)
    a = fixture_row()
    b = deepcopy(a)
    if mutation == 'reference': b['input_fingerprints']['reference'] = 'different'
    elif mutation == 'seed': b['sampling_seed'] = 4
    elif mutation == 'persistence': b['arms']['persistence']['extra'] = 1
    else: b['arms']['forecast']['metrics']['conventions']['pairs'] = [[1, 5]]
    with pytest.raises(ValueError):
        s.observed_reference_pairing({'joint_path': {key: a}, 'marginal_path': {key: b}})


def test_predictions_may_differ_but_observed_labels_must_match():
    key = ('a', 'x', 320., 7, 16)
    a = fixture_row()
    b = deepcopy(a)
    b['input_fingerprints']['forecast_CA_path'] = 'different'
    b['arms']['forecast']['metrics']['scalars']['new_prediction'] = 4.
    assert s.observed_reference_pairing({'joint_path': {key: a}, 'marginal_path': {key: b}}) == [key]


def test_complete_ordered_panel_primary_all_groups_and_undefined_lags():
    rows = {arm: {} for arm in s.ARMS}
    for i in range(50):
        for h in (1, 4, 16, 64):
            key = (f'p{i:02}', 'x', (320., 348., 379., 413., 450.)[i//10], 7, h)
            base = fixture_row()
            base.update(target_frame=7+h, horizon_ps=h*1000)
            for role in s.paths.ARMS:
                metrics = base['arms'][role]['metrics']
                metrics['scalars'] = dict.fromkeys((*s.SCORES, *s.paths.DESCRIPTIVE), 0.)
                metrics['scalars']['distance_acf_corr_n_lags'] = 0
                if role == 'reference_MD':
                    for metric in s.SCORES: metrics['scalars'][metric] = None
                metrics['lag_statistics']['prediction'] = dict(aggregate=[
                    dict(acf=None, msd=0. if lag <= h else None,
                         n_pairs=max(h+1-lag, 0), n_acf_pairs=0) for lag in s.paths.LAGS])
            for arm in s.ARMS:
                row = deepcopy(base)
                for metric in s.SCORES:
                    row['arms']['forecast']['metrics']['scalars'][metric] = (
                        {1: 0., 4: 0., 16: 1., 64: 3.}[h] if arm == 'joint_path' else 0.)
                rows[arm][key] = row
    result = s.ordered_analysis(rows)
    primary = result['primary_long_horizon_joint_path_UES']
    assert primary['candidate_minus_reference'] == 2.
    assert primary['ci95'] == [2., 2.]
    assert primary['n_proteins'] == 50 and primary['n_windows'] == 100
    comparisons = result['ordered_path_scores']['comparisons']
    assert sum(len(metrics) for groups in comparisons.values() for metrics in groups.values()) == 60
    report = result['ordered_path_scores']['descriptive']['joint_path']
    assert set(report) == set(s.endpoint.GROUPS)
    assert report['h1']['forecast']['lag_diagnostics']['32']['msd']['complete_panel_mean'] is None
    assert report['h64']['forecast']['lag_diagnostics']['32']['msd']['complete_panel_mean'] == 0.
    assert report['overall']['reference_MD']['scalars'][s.SCORES[0]]['n_valid_windows'] == 0
