import numpy as np

from protein_jepa_frame.eval.dynamics_benchmarks import (
    binary_forecast_metrics,
    chapman_kolmogorov_error,
    grouped_bootstrap,
    mean_first_passage_times,
    multiclass_forecast_metrics,
    path_ensemble_metrics,
    transition_counts,
    transition_operator_metrics,
)


def test_probabilistic_metrics_reward_calibrated_predictions():
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    result = multiclass_forecast_metrics(probabilities, np.asarray([0, 1]))

    assert result["nll"] < 0.25
    assert result["accuracy"] == 1.0
    assert result["ece"] < 0.2
    binary = binary_forecast_metrics(np.asarray([0.1, 0.9]), np.asarray([0, 1]))
    assert binary["brier"] < 0.02


def test_transition_operator_reports_mfpt_and_semigroup():
    reference = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    counts = transition_counts([[0, 0, 1, 1, 1, 0, 0]], num_states=2)
    assert counts.tolist() == [[2.0, 1.0], [1.0, 2.0]]
    mfpt = mean_first_passage_times(reference)
    assert np.isclose(mfpt[0, 1], 10.0)
    assert np.isclose(mfpt[1, 0], 5.0)
    metrics = transition_operator_metrics(reference, reference)
    assert metrics["row_total_variation"] == 0.0
    assert metrics["mfpt_relative_error"] == 0.0
    assert chapman_kolmogorov_error(reference, reference @ reference) < 1.0e-12


def test_grouped_bootstrap_does_not_treat_frames_as_independent():
    result = grouped_bootstrap([0.0, 1.0, 10.0, 11.0], ["a", "a", "b", "b"], samples=100, seed=2)
    assert result["groups"] == 2
    assert np.isclose(result["estimate"], 5.5)


def test_path_metrics_measure_transition_time_and_invalid_rollouts():
    result = path_ensemble_metrics(
        [["unfolded", "transition", "folded"], ["unfolded", "folded"]],
        [["unfolded", "transition", "transition", "folded"]],
    )
    assert result["predicted_valid_paths"] == 2
    assert result["predicted_failure_rate"] == 0.0
    assert result["transition_time_wasserstein"] > 0.0
