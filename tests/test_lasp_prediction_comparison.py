"""Fail-closed paired semantics comparison and signed gap regression tests."""
from copy import deepcopy

import pytest

from scripts.compare_lasp_prediction_semantics import paired_summary
from scripts.compare_lasp_bounded_paired import SHARED_METRICS


def artifact():
    rows = []
    for protein in ('a', 'b', 'c', 'd'):
        rows.append(dict(protein=protein, traj=protein+'_T348', anchor_frame=7,
            horizon=4., temperature_k=348., actual_n_samples=12, observation_steps=[4]*12,
            lasp={k:1. for k in SHARED_METRICS}, persist={k:2. for k in SHARED_METRICS}))
    return dict(per_window=rows, n_windows=4, n_samples=12, metric_version='v2',
                checkpoint_sha256='frozen-checkpoint')


def test_gap_direction_and_cluster_ci():
    baseline = artifact()
    candidate = deepcopy(baseline)
    for row in candidate['per_window']:
        row['lasp']['md_pca_w2'] += .4
        row['lasp']['rmsf_corr'] += .2
    summary = paired_summary(candidate, baseline, draws=100, diagnostic_only=True)
    pca = summary['metrics']['md_pca_w2']
    corr = summary['metrics']['rmsf_corr']
    assert pca['signed_error_gap_mixture_minus_endpoint'] == pytest.approx(.4)
    assert pca['ci95'] == pytest.approx([.4, .4])
    assert corr['signed_error_gap_mixture_minus_endpoint'] == pytest.approx(-.2)
    assert corr['one_sided_upper_983333'] == pytest.approx(-.2)


@pytest.mark.parametrize('field,value,message', [
    ('n_samples', None, 'budgets'),
    ('metric_version', None, 'versions'),
    ('checkpoint_sha256', 'different', 'checkpoint'),
])
def test_missing_or_incompatible_provenance_rejected(field, value, message):
    baseline = artifact()
    candidate = deepcopy(baseline)
    candidate[field] = value
    with pytest.raises(ValueError, match=message):
        paired_summary(candidate, baseline, diagnostic_only=True)


@pytest.mark.parametrize('mutation,message', [
    ('identity', 'identities'),
    ('duplicate', 'duplicate'),
    ('persistence', 'persistence'),
    ('actual_samples', 'sample count'),
    ('times', 'observation times'),
    ('nonfinite', 'nonfinite'),
])
def test_invalid_rows_fail_closed(mutation, message):
    baseline = artifact()
    candidate = deepcopy(baseline)
    row = candidate['per_window'][0]
    if mutation == 'identity': row['anchor_frame'] += 1
    if mutation == 'duplicate': candidate['per_window'].append(deepcopy(row))
    if mutation == 'persistence': row['persist']['md_pca_w2'] += 1
    if mutation == 'actual_samples': row['actual_n_samples'] = 11
    if mutation == 'times': row['observation_steps'][0] = 0
    if mutation == 'nonfinite': row['lasp']['md_pca_w2'] = float('nan')
    with pytest.raises(ValueError, match=message):
        paired_summary(candidate, baseline, diagnostic_only=True)


def test_small_panel_cannot_silently_be_called_standard200():
    baseline = artifact()
    with pytest.raises(ValueError, match='standard200'):
        paired_summary(baseline, baseline)
