from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import ca_local_residual_statistics_v1 as s


def test_equal_protein_weighting_and_opposite_sign_interval():
    keys = [('a', 0), ('a', 1), ('b', 0)]
    rows = dict(zip(keys, [(0., 2.), (2., 2.), (7., 2.)]))
    result = s.paired(rows, keys, lambda r: r[0], lambda r: r[1], 'lower')
    assert result['candidate'] == 4. and result['reference'] == 2.
    assert result['candidate_minus_reference'] == 2.
    assert result['improvement'] == -2.
    assert result['ci95'] == [-1., 5.]
    assert result['improvement_ci95'] == [-5., 1.]
    assert s.paired(rows, list(reversed(keys)), lambda r: r[0], lambda r: r[1], 'lower') == result


def test_missing_scalar_does_not_become_paired_subset_or_zero():
    rows = {('a', 0): (None, 1.), ('a', 1): (3., 1.), ('b', 0): (5., 2.)}
    result = s.paired(rows, list(rows), lambda r: r[0], lambda r: r[1], 'lower')
    assert result['candidate'] is result['ci95'] is result['candidate_minus_reference'] is None
    assert result['candidate_support']['available_support_mean'] == 4.
    assert result['candidate_support']['missing_window_keys'] == [['a', 0]]
    assert result['reference'] == 1.5
    assert result['status'] == 'undefined_on_complete_registered_panel'


def test_all_missing_and_no_direction():
    rows = {('a', 0): None}
    result = s.support(rows, list(rows), lambda r: r)
    assert result['complete_panel_mean'] is result['available_support_mean'] is None
    assert result['n_valid_proteins'] == result['n_valid_windows'] == 0
    result = s.paired({('a', 0): 3.}, [('a', 0)], lambda r: r, lambda r: r, None)
    assert result['ci95'] == [0., 0.] and result['improvement'] is None


@pytest.mark.parametrize('value', [True, np.bool_(False), float('inf'), float('nan'), '1'])
def test_bad_number_rejected(value):
    with pytest.raises(ValueError, match='finite number'):
        s.support({('a', 0): value}, [('a', 0)], lambda r: r)


@pytest.mark.parametrize('keys', [[], [('a', 0), ('a', 0)], [('b', 0)]])
def test_bad_support_rejected(keys):
    with pytest.raises(ValueError):
        s.support({('a', 0): 1.}, keys, lambda r: r)


def val_rows():
    return [dict(protein=f'p{i:02}', trajectory=f'p{i}_traj', temperature_k=s.TEMPERATURES[i//10],
                 anchor_frame=7, horizon=h, scores={'base_tuned': 2.,
                 'local_residual': 1. if h in (16, 64) else 100.})
            for i in range(50) for h in s.HORIZONS]


def test_real_json_roundtrip_complete_panel_and_primary_horizons():
    rows = s.indexed(json.loads(json.dumps(val_rows())), 'val')
    result = s.long_horizon_primary(rows, lambda r, a: r['scores'][a])
    assert result['candidate_minus_reference'] == -1. and result['ci95'] == [-1., -1.]
    assert result['n_windows'] == 100 and result['n_proteins'] == 50
    groups = s.groups_for(rows)
    assert len(groups) == 10 and len(groups['T320']) == 40 and len(groups['h64']) == 50


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'wrong_anchor', 'bool_horizon', 'wrong_temperature'])
def test_panel_failures(case):
    rows = deepcopy(val_rows())
    if case == 'duplicate': rows[-1] = deepcopy(rows[-2])
    elif case == 'missing': rows.pop()
    elif case == 'wrong_anchor': rows[0]['anchor_frame'] += 1
    elif case == 'bool_horizon': rows[0]['horizon'] = True
    elif case == 'wrong_temperature': rows[0]['temperature_k'] = 300.
    with pytest.raises(ValueError): s.indexed(rows, 'val')


def test_primary_missing_metric_and_missing_long_window_fail():
    rows = s.indexed(val_rows(), 'val')
    first = next(k for k in rows if k[4] == 16)
    rows[first]['scores']['local_residual'] = None
    with pytest.raises(ValueError, match='undefined'):
        s.long_horizon_primary(rows, lambda r, a: r['scores'][a])
    del rows[first]
    with pytest.raises(ValueError, match='both long horizons'):
        s.long_horizon_primary(rows, lambda r, a: r['scores'][a])


def test_bootstrap_indices_frozen_and_match_independent_seed():
    expected = np.random.default_rng(20261101).integers(3, size=(10000, 3))
    np.testing.assert_array_equal(s.bootstrap_indices(3), expected)
    with pytest.raises(ValueError): s.bootstrap_indices(3)[0, 0] = 0


def full_data():
    data = dict(schema_version='ca_local_residual_probe_v1', per_window={'train': [], 'val': val_rows()})
    for i in range(500):
        data['per_window']['train'].append(dict(protein=f'train{i}', trajectory=f'train{i}_traj',
            temperature_k=s.TEMPERATURES[i%5], anchor_frame=7, horizon=s.HORIZONS[i%4]))
    for split, rows in data['per_window'].items():
        for row in rows:
            row['routes'] = {}
            for arm, value in zip(s.ARMS, (3., 2., 1.)):
                row['routes'][arm] = {}
                for role in s.ROLES:
                    row['routes'][arm][role] = None if split == 'train' and role != 'oracle' else {
                        stage: dict(scalars={m: value for m in (s.FORECAST_SCALARS if role == 'forecast' else s.SCALARS)})
                        for stage in s.STAGES}
    return data


def test_complete_analysis_all_intervals_and_descriptive_controls_roundtrip():
    data = json.loads(json.dumps(full_data()))
    result = s.analysis(data)
    assert result['requested_interval_cells'] == result['complete_interval_cells'] == 1501
    assert result['undefined_interval_cells'] == 0
    assert result['primary']['candidate_minus_reference'] == -1.
    forecast = result['comparisons']['local_residual_minus_baseline']['forecast']['raw']['h64']
    assert forecast['coordinate_energy_score']['ci95'] == [-2., -2.]
    assert forecast['canonical_ca_bond_mean_angstrom']['improvement'] is None
    control = result['descriptive']['val']['local_residual']['wrong_horizon_H']['projected']['overall']
    assert control['ca_rmsd']['n_windows'] == 200
    assert set(result['descriptive']['train']['baseline']) == {'oracle'}
    assert s.contract()['requested_interval_cells'] == 1501


def test_undefined_secondary_retains_cell_count_and_fails_on_split_overlap():
    data = full_data()
    for arm in s.ARMS:
        data['per_window']['val'][0]['routes'][arm]['oracle']['raw']['scalars']['rmsd_to_persistence_ratio'] = None
    result = s.analysis(data)
    # One missing window belongs to overall, h1 and T320, in each of three contrasts.
    assert result['undefined_interval_cells'] == 9
    assert result['complete_interval_cells'] == 1492
    data['per_window']['train'][0]['protein'] = data['per_window']['val'][0]['protein']
    with pytest.raises(ValueError, match='split overlap'): s.analysis(data)
