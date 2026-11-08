"""Analytic CPU checks for joint-path scores and boundary-safe lag statistics."""
import json
import math

import pytest
import torch

from evaluation.trajectory_metrics_v2 import (
    joint_path_energy_scores, ordered_path_summary, path_lag_statistics,
)


def data(scales=(1., 1.1, 1.3, 1.7, 2.1)):
    anchor = torch.tensor([[-3., -1., 0.], [-1., 2., 1.], [1., -2., 2.],
                           [3., 1., -1.]], dtype=torch.float64)
    anchor -= anchor.mean(0)
    reference = torch.stack([scale * anchor for scale in scales])
    mask = torch.ones(4, dtype=torch.bool)
    times = torch.arange(len(reference), dtype=torch.float64) * 1000.
    keywords = dict(predicted_times_ps=times, reference_times_ps=times,
                    reference_mask=mask[None].expand(len(reference), -1),
                    residue_index=torch.arange(4), chain_id=torch.zeros(4, dtype=torch.long))
    return anchor, reference, mask, keywords


def lag_args(keywords):
    return {key: keywords[key] for key in ('residue_index', 'chain_id')}


def test_joint_score_distinguishes_same_time_marginals_with_different_path_dependence():
    anchor, _, mask, kw = data((1., 1., 1.))
    a, b = anchor, 2 * anchor
    reference = torch.stack((a, a, a))
    persistent = torch.stack((torch.stack((a, a, a)), torch.stack((a, b, b))))
    crossing = torch.stack((torch.stack((a, a, b)), torch.stack((a, b, a))))
    # Each time has the same {A,B} ensemble; only within-member dependence differs.
    for time in (1, 2):
        assert torch.equal(persistent[:, time].sort(dim=0).values, crossing[:, time].sort(dim=0).values)
    p = joint_path_energy_scores(persistent, reference, anchor, mask, **kw)
    q = joint_path_energy_scores(crossing, reference, anchor, mask, **kw)
    assert abs(p['coordinate_path_energy_score']) < 1e-12
    assert abs(p['physical_distance_path_energy_score']) < 1e-12
    expected_coordinate = float(anchor.norm()) * (1 - 1 / math.sqrt(2)) / math.sqrt(3 * 4 * 2)
    expected_distance = float((a[0] - a[3]).norm()) * (1 - 1 / math.sqrt(2)) / math.sqrt(2)
    assert q['coordinate_path_energy_score'] == pytest.approx(expected_coordinate, abs=1e-12)
    assert q['physical_distance_path_energy_score'] == pytest.approx(expected_distance, abs=1e-12)


def test_correct_shuffled_and_reversed_future_paths_have_time_matched_scores():
    anchor, reference, mask, kw = data()
    correct = reference[None].repeat(3, 1, 1, 1)
    shuffled = correct[:, [0, 1, 3, 2, 4]]
    reversed_future = torch.cat((correct[:, :1], correct[:, 1:].flip(1)), 1)
    baseline = ordered_path_summary(correct, reference, anchor, mask, **kw)
    assert baseline['scalars']['per_step_rmsd_mean'] < 1e-12
    for altered in (shuffled, reversed_future):
        summary = ordered_path_summary(altered, reference, anchor, mask, **kw)
        for metric in ('coordinate_path_energy_score', 'physical_distance_path_energy_score', 'per_step_rmsd_mean'):
            assert summary['scalars'][metric] > baseline['scalars'][metric] + .01
        # These controls preserve the conformer bag, so pooled spread/Rg cannot
        # establish order on their own.
        assert summary['scalars']['rg_mean'] == pytest.approx(baseline['scalars']['rg_mean'])
        assert summary['scalars']['walk_rmsf_mae'] == pytest.approx(baseline['scalars']['walk_rmsf_mae'], abs=1e-12)
    json.dumps(baseline, allow_nan=False)


def test_fixed_anchor_gauge_aligns_only_reference_not_prediction():
    anchor, reference, mask, kw = data()
    prediction = reference[None].repeat(2, 1, 1, 1)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    moved_truth = reference.clone()
    moved_truth[1:] = reference[1:] @ rotation + torch.tensor([13., -5., 2.])
    score = joint_path_energy_scores(prediction, moved_truth, anchor, mask, **kw)
    assert max(abs(value) for value in score.values()) < 1e-12
    shifted = prediction.clone()
    shifted[:, 1:, :, 0] += 2.
    score = joint_path_energy_scores(shifted, moved_truth, anchor, mask, **kw)
    assert score['coordinate_path_energy_score'] == pytest.approx(2 / math.sqrt(3))
    assert abs(score['physical_distance_path_energy_score']) < 1e-12


def test_lag_statistics_preserve_replica_boundaries_counts_and_zero_variance():
    anchor, _, mask, kw = data()
    ramp = torch.stack([scale * anchor for scale in (1., 2., 3., 4.)])
    constant = (10 * anchor)[None].repeat(3, 1, 1)
    result = path_lag_statistics([ramp, constant], [torch.arange(4), torch.arange(3) + 20],
                                mask, **lag_args(kw), lags=(0, 1, 2, 4))
    first, second = result['per_path']
    lag1 = first['by_lag'][1]
    distance_squared = float((anchor[0] - anchor[3]).square().sum())
    assert first['variance'] == pytest.approx(1.25 * distance_squared)
    assert lag1['n_pairs'] == 3 and lag1['lag_ps'] == 1
    assert lag1['acf'] == pytest.approx(1 / 3)
    assert lag1['msd'] == pytest.approx(distance_squared)
    assert second['variance'] == 0.
    assert all(row['acf'] is None for row in second['by_lag'])
    aggregate = result['aggregate'][1]
    assert aggregate['n_pairs'] == 5 and aggregate['n_acf_pairs'] == 3
    assert aggregate['acf'] == pytest.approx(1 / 3)
    assert aggregate['msd'] == pytest.approx(3 / 5 * distance_squared)
    assert result['aggregate'][-1] == dict(lag=4, lag_ps=4., n_pairs=0, n_acf_pairs=0, acf=None, msd=None)
    json.dumps(result, allow_nan=False)


def test_acf_msd_are_reversal_symmetric_but_detect_shuffled_local_increments():
    anchor, reference, mask, kw = data((1., 2., 3., 5., 8., 13.))
    times = torch.arange(len(reference))
    normal = path_lag_statistics(reference[None], times, mask, **lag_args(kw), lags=(1, 2))
    reverse = path_lag_statistics(reference.flip(0)[None], times, mask, **lag_args(kw), lags=(1, 2))
    shuffle = path_lag_statistics(reference[[0, 4, 2, 5, 1, 3]][None], times, mask, **lag_args(kw), lags=(1, 2))
    for left, right in zip(normal['aggregate'], reverse['aggregate']):
        assert left['acf'] == pytest.approx(right['acf'])
        assert left['msd'] == pytest.approx(right['msd'])
        assert left['n_pairs'] == right['n_pairs']
    assert normal['aggregate'][0]['msd'] != pytest.approx(shuffle['aggregate'][0]['msd'])
    # ACF/MSD are reversal-even signatures, not tests of kinetic irreversibility.


def test_padding_and_fixed_physical_pair_identities():
    anchor, reference, mask, kw = data()
    prediction = reference[None].repeat(2, 1, 1, 1)
    expected = ordered_path_summary(prediction, reference, anchor, mask, **kw)
    padded_anchor = torch.nn.functional.pad(anchor, (0, 0, 0, 2), value=float('nan'))
    padded_reference = torch.nn.functional.pad(reference, (0, 0, 0, 2), value=float('nan'))
    padded_prediction = torch.nn.functional.pad(prediction, (0, 0, 0, 2), value=float('nan'))
    padded_mask = torch.tensor([True] * 4 + [False] * 2)
    padded_kw = {**kw, 'reference_mask': padded_mask[None].expand(len(reference), -1),
                 'residue_index': torch.arange(6), 'chain_id': torch.zeros(6, dtype=torch.long)}
    actual = ordered_path_summary(padded_prediction, padded_reference, padded_anchor, padded_mask, **padded_kw)
    assert actual['scalars'] == expected['scalars']
    assert actual['conventions']['pair_indices'] == [[0, 3]]
    # Physical residue numbers and chains, not compacted array offsets, define pairs.
    changed_kw = {**kw, 'residue_index': torch.tensor([0, 10, 11, 0]),
                  'chain_id': torch.tensor([0, 0, 0, 1])}
    changed = ordered_path_summary(prediction, reference, anchor, mask, **changed_kw)
    assert changed['conventions']['pair_indices'] == [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]]
    subset = ordered_path_summary(prediction, reference, anchor, mask, **changed_kw, max_pairs=2)
    assert subset['conventions']['pair_indices'] == [[0, 1], [2, 3]]


def test_single_reference_is_not_replicated_and_single_member_has_no_u_score():
    anchor, reference, mask, kw = data((1., 1.2))
    single = ordered_path_summary(reference[None], reference, anchor, mask, **kw)
    assert single['scalars']['coordinate_path_energy_score'] is None
    assert single['scalars']['physical_distance_path_energy_score'] is None
    assert single['scalars']['walk_rmsf_mae'] is None
    assert single['scalars']['walk_rmsf_corr'] is None
    prediction = reference[None].repeat(12, 1, 1, 1)
    summary = ordered_path_summary(prediction, reference, anchor, mask, **kw)
    assert summary['lag_statistics']['prediction']['aggregate'][0]['n_pairs'] == 12
    assert summary['lag_statistics']['reference']['aggregate'][0]['n_pairs'] == 1
    assert len(summary['lag_statistics']['reference']['per_path']) == 1
    with pytest.raises(ValueError, match='M>=2'):
        joint_path_energy_scores(reference[None], reference, anchor, mask, **kw)


def test_contract_rejects_time_mask_anchor_and_nonfinite_mismatches():
    anchor, reference, mask, kw = data()
    prediction = reference[None].repeat(2, 1, 1, 1)
    def score(**changes):
        return joint_path_energy_scores(prediction, reference, anchor, mask, **{**kw, **changes})
    with pytest.raises(ValueError, match='timestamps must match'):
        score(predicted_times_ps=kw['predicted_times_ps'] + 1)
    with pytest.raises(ValueError, match='strictly increasing'):
        score(predicted_times_ps=torch.zeros(len(reference)))
    bad_mask = kw['reference_mask'].clone()
    bad_mask[-1, -1] = False
    with pytest.raises(ValueError, match='validity'):
        score(reference_mask=bad_mask)
    changed = prediction.clone()
    changed[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match='observed anchor'):
        joint_path_energy_scores(changed, reference, anchor, mask, **kw)
    changed[0, 1, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        joint_path_energy_scores(changed, reference, anchor, mask, **kw)
    with pytest.raises(ValueError, match='uniformly spaced'):
        path_lag_statistics(prediction, torch.tensor([0., 1., 2., 3., 5.]), mask, **lag_args(kw))
    with pytest.raises(ValueError, match='same physical time step'):
        path_lag_statistics(list(prediction), [torch.arange(5), torch.arange(5) * 2], mask, **lag_args(kw))


def test_persistence_zero_variance_and_geometry_are_explicit():
    anchor, reference, mask, kw = data()
    prediction = anchor[None, None].repeat(3, len(reference), 1, 1)
    result = ordered_path_summary(prediction, reference, anchor, mask, **kw)
    assert result['scalars']['path_length'] == 0.
    assert result['scalars']['end_to_end'] == 0.
    assert result['scalars']['tortuosity'] is None
    assert result['scalars']['distance_acf_lag1'] is None
    assert result['scalars']['distance_msd_lag1'] == 0.
    assert result['lag_statistics']['prediction']['aggregate'][0]['n_acf_pairs'] == 0
    assert result['scalars']['distance_acf_corr'] is None
    assert result['scalars']['distance_acf_corr_n_lags'] == 0
    json.dumps(result, allow_nan=False)


def test_float32_inputs_use_double_arithmetic_and_pooled_spread_is_disclosed():
    anchor, reference, mask, kw = data()
    anchor, reference = anchor.float(), reference.float()
    prediction = reference[None].repeat(3, 1, 1, 1)
    actual = ordered_path_summary(prediction, reference, anchor, mask, **kw)
    expected = ordered_path_summary(prediction.double(), reference.double(), anchor.double(), mask, **kw)
    assert actual == expected
    assert actual['conventions']['arithmetic_dtype'] == 'float64'
    assert 'between-member spread' in actual['conventions']['walk_rmsf']
    actual_lag = path_lag_statistics(prediction, kw['predicted_times_ps'], mask, **lag_args(kw))
    expected_lag = path_lag_statistics(prediction.double(), kw['predicted_times_ps'], mask, **lag_args(kw))
    assert actual_lag == expected_lag


def test_acf_correlation_requires_three_common_registered_defined_lags():
    anchor, reference, mask, kw = data((1., 2., 3., 5., 8., 13.))
    predicted = reference[None].repeat(2, 1, 1, 1)
    full = ordered_path_summary(predicted, reference, anchor, mask, **kw, lags=(1, 2, 4, 8))
    assert full['scalars']['distance_acf_corr'] == pytest.approx(1.)
    assert full['scalars']['distance_acf_corr_n_lags'] == 3
    assert full['conventions']['acf_correlation_valid_lags'] == [1, 2, 4]
    short = ordered_path_summary(predicted, reference, anchor, mask, **kw, lags=(1, 2, 8))
    assert short['scalars']['distance_acf_corr'] is None
    assert short['scalars']['distance_acf_corr_n_lags'] == 2
