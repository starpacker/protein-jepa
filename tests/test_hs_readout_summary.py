import copy
import json

import numpy as np
import pytest

from scripts import summarize_hs_readout_pilot as summary


def test_paired_bootstrap_uses_equal_proteins_and_keeps_uncertainty_descriptive():
    keys = [('a', 'a0', 320., 7, 1), ('a', 'a0', 320., 7, 4), ('b', 'b0', 348., 7, 1)]
    candidate = {k: {'value': v} for k, v in zip(keys, [1., 3., 10.])}
    reference = {k: {'value': v} for k, v in zip(keys, [0., 2., 9.])}
    get = lambda row: row['value']
    result = summary.paired(candidate, reference, keys, get, get, -1)
    assert result['candidate'] == 6.
    assert result['reference'] == 5.
    assert result['candidate_minus_reference'] == 1.
    assert result['ci95'] == [1., 1.]
    assert result['improvement_ci95'] == [-1., -1.]
    assert result['n_proteins'] == 2 and result['n_windows'] == 3
    assert not {'passed', 'equivalent', 'noninferior'} & result.keys()
    assert result == summary.paired(candidate, reference, keys[::-1], get, get, -1)


def synthetic_panel():
    rows = [dict(protein=f'p{i}', trajectory=f'p{i}_T{summary.TEMPERATURES[i//10]}',
                 temperature_k=summary.TEMPERATURES[i//10], anchor_frame=7, horizon=h)
            for i in range(50) for h in summary.HORIZONS]
    return dict(n_windows=200, n_proteins=50, per_window=rows)


def test_panel_requires_all_unique_balanced_window_keys():
    good = synthetic_panel()
    assert len(summary.rows_for(good, 'synthetic')) == 200
    duplicate = copy.deepcopy(good)
    duplicate['per_window'][-1] = duplicate['per_window'][0]
    with pytest.raises(ValueError, match='duplicate'):
        summary.rows_for(duplicate, 'duplicate')
    incomplete = copy.deepcopy(good)
    incomplete['per_window'].pop()
    with pytest.raises(ValueError, match='incomplete panel'):
        summary.rows_for(incomplete, 'missing')
    shifted = copy.deepcopy(good)
    shifted['per_window'][0]['anchor_frame'] += 1
    with pytest.raises(ValueError, match='shared anchor'):
        summary.rows_for(shifted, 'different anchor')


@pytest.mark.parametrize('setting', summary.SETTINGS)
def test_common_validation_removes_all_regularizers_with_the_actual_weights(setting):
    args = dict(zip(summary.ARM_FIELDS, setting), latent_weight=.1, contact_weight=20.,
                rmsf_weight=1., distance_weight=0., clash_weight=40., mode_weight=2.,
                mode_distribution_weight=0.)
    terms = dict(latent_es=2., coord_es=3., contact=.1, rmsf=.2, distance_es=1.,
                 clash=.01, slow_mode_scale=.2, slow_mode_distribution=1.)
    common = .1*2 + 3 + 20*.1 + .2 + 40*.01 + 2*.2
    terms['loss'] = common
    if args['train_encoder']:
        terms.update(teacher=.25, teacher_delta=.125)
        terms['loss'] += args['teacher_weight']*.25 + args['teacher_delta_weight']*.125
        if args['observed_readout_weight']:
            terms['observed_readout_mse'] = 5.
            terms['loss'] += args['observed_readout_weight']*5.
    validation = dict(loss=terms['loss'], by_horizon={str(h): dict(terms) for h in summary.HORIZONS})
    result = summary.common_validation(validation, args)
    assert result['common_forecast_loss'] == pytest.approx(common)
    assert result['raw_total'] == terms['loss']
    bad = copy.deepcopy(validation)
    bad['by_horizon']['1']['loss'] += 1.
    with pytest.raises(ValueError, match='decomposition'):
        summary.common_validation(bad, args)


def test_incomplete_execution_and_changed_inputs_fail_closed(tmp_path):
    file = tmp_path / 'frozen.execution.json'
    file.write_text(json.dumps(dict(arm='frozen', jobs=[dict(kind='train', returncode=0)])))
    inputs = summary.Inputs()
    with pytest.raises(ValueError, match='execution incomplete'):
        summary.audit_execution(tmp_path, 'frozen', {}, inputs)
    file.write_text('{}')
    with pytest.raises(ValueError, match='input changed'):
        inputs.verify()
    with pytest.raises(ValueError, match='sealed'):
        summary.path(tmp_path / 'test.jsonl')


def test_missing_metric_and_registered_argument_mismatch_fail_closed():
    with pytest.raises(ValueError, match='nonfinite'):
        summary.finite({'a': np.nan}, ['a'], 'synthetic')
    with pytest.raises(ValueError, match='registered value differs'):
        summary.check_cli_arguments({'observed_readout_weight': 0.}, ['--observed-readout-weight', '.1'])
    summary.check_cli_arguments({'train_encoder': True, 'steps': 500}, ['--train-encoder', '--steps', '500'])


def test_frozen_component_comparison_supports_nested_numpy_rng_state():
    left = dict(rng={'numpy': ('MT19937', np.array([1, 7, 13], dtype=np.uint32), 3, 0, .0)})
    right = copy.deepcopy(left)
    assert summary.same_tree(left, right)
    right['rng']['numpy'][1][1] = 8
    assert not summary.same_tree(left, right)
    assert not summary.same_tree(np.array([1], dtype=np.int32), np.array([1], dtype=np.int64))
    assert not summary.same_tree(np.array([1]), np.array([[1]]))


def test_legacy_geometry_and_counts_cover_all_registered_contrasts_and_groups():
    keys = sorted(summary.rows_for(synthetic_panel(), 'synthetic'))
    groups = dict(overall=keys)
    groups.update({f'h{h}': [k for k in keys if k[4] == h] for h in summary.HORIZONS})
    groups.update({f'T{int(t)}': [k for k in keys if k[2] == t] for t in summary.TEMPERATURES})
    rows = {arm: {k: dict(
        geometry_vs_anchor={metric: float(offset + k[4]) for metric in summary.LEGACY_GEOMETRY},
        contact_counts={metric: float(k[4] + (offset if metric.endswith('_pred') else 0))
                        for metric in summary.CONTACT_COUNTS}) for k in keys}
        for offset, arm in enumerate(summary.ARMS)}
    blocks = summary.descriptive_ensemble_blocks(rows, groups)
    assert set(blocks) == {'legacy_geometry_vs_anchor', 'ensemble_contact_counts'}
    assert '3.0-angstrom' in blocks['legacy_geometry_vs_anchor']['interpretation']
    assert 'not future-angle accuracy' in blocks['legacy_geometry_vs_anchor']['interpretation']
    assert 'not pooled counts' in blocks['ensemble_contact_counts']['interpretation']
    for name, metrics in [('legacy_geometry_vs_anchor', summary.LEGACY_GEOMETRY),
                          ('ensemble_contact_counts', summary.CONTACT_COUNTS)]:
        block = blocks[name]
        assert set(block['arm_means']) == set(summary.ARMS)
        assert set(block['comparisons']) == {f'{a}_vs_{b}' for a, b in summary.CONTRASTS}
        for (a, b) in summary.CONTRASTS:
            contrast = block['comparisons'][f'{a}_vs_{b}']
            assert set(contrast) == set(groups) and len(contrast) == 10
            for group in contrast.values():
                assert set(group) == set(metrics)
                for metric, result in group.items():
                    expected = 0. if metric.endswith('_true') else float(summary.ARMS.index(a)-summary.ARMS.index(b))
                    assert result['candidate_minus_reference'] == expected
                    assert result['ci95'] == [expected, expected]
                    assert not {'improvement', 'passed', 'equivalent', 'noninferior'} & result.keys()
