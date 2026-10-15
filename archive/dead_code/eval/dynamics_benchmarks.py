"""Metrics for evaluating probabilistic protein dynamics predictions.

The functions in this module are deliberately model agnostic. A model adapter
should produce state probabilities, event probabilities, or sampled paths; the
benchmark then scores those objects against held-out trajectories.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np


def _as_float_array(value: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def normalize_probabilities(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return finite row-normalized probabilities with a uniform fallback."""
    values = _as_float_array(probabilities)
    if values.ndim == 0:
        raise ValueError("probabilities must have at least one dimension")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = values.sum(axis=axis, keepdims=True)
    classes = values.shape[axis]
    uniform = np.full_like(values, 1.0 / max(classes, 1))
    return np.divide(values, total, out=uniform, where=total > 1.0e-12)


def calibration_bins(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    """Compute confidence, accuracy, and count for multiclass reliability bins."""
    probs = normalize_probabilities(probabilities)
    y = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or y.ndim != 1 or len(y) != len(probs):
        raise ValueError("probabilities must be [N, C] and targets must be [N]")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, max(1, bins) + 1)
    output = []
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        keep = (confidence >= left) & (
            confidence <= right if index == len(edges) - 2 else confidence < right
        )
        count = int(keep.sum())
        output.append(
            {
                "lower": float(left),
                "upper": float(right),
                "count": count,
                "confidence": float(confidence[keep].mean()) if count else 0.0,
                "accuracy": float(correct[keep].mean()) if count else 0.0,
            }
        )
    return output


def multiclass_forecast_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> dict:
    """Score probabilistic multiclass state forecasts.

    Targets are integer state ids. The returned ``nll`` and ``brier`` are the
    primary metrics; hard classification metrics are secondary diagnostics.
    """
    probs = normalize_probabilities(probabilities)
    y = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2 or y.ndim != 1 or len(y) != len(probs):
        raise ValueError("probabilities must be [N, C] and targets must be [N]")
    if np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("targets contain a state outside the probability columns")
    n = max(len(y), 1)
    selected = probs[np.arange(len(y)), y]
    nll = float(-np.log(np.clip(selected, 1.0e-12, 1.0)).mean())
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[y]
    brier = float(np.mean(np.square(probs - one_hot)))
    predicted = probs.argmax(axis=1)
    accuracy = float(np.mean(predicted == y))
    f1_values = []
    recalls = []
    for state in range(probs.shape[1]):
        tp = float(np.sum((predicted == state) & (y == state)))
        fp = float(np.sum((predicted == state) & (y != state)))
        fn = float(np.sum((predicted != state) & (y == state)))
        precision = tp / max(tp + fp, 1.0e-12)
        recall = tp / max(tp + fn, 1.0e-12)
        f1_values.append(2.0 * precision * recall / max(precision + recall, 1.0e-12))
        if tp + fn > 0:
            recalls.append(recall)
    reliability = calibration_bins(probs, y, bins=bins)
    ece = float(
        sum(item["count"] * abs(item["confidence"] - item["accuracy"]) for item in reliability)
        / n
    )
    return {
        "n": int(len(y)),
        "nll": nll,
        "brier": brier,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "ece": ece,
        "reliability": reliability,
    }


def binary_forecast_metrics(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> dict:
    """Score a calibrated binary event probability."""
    p = np.clip(_as_float_array(probability).reshape(-1), 0.0, 1.0)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(p) != len(y):
        raise ValueError("probability and target lengths differ")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("binary targets must be 0 or 1")
    n = max(len(y), 1)
    log_loss = float(-np.mean(y * np.log(np.clip(p, 1.0e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1.0e-12, 1.0))))
    brier = float(np.mean(np.square(p - y)))
    prediction = p >= 0.5
    accuracy = float(np.mean(prediction == y.astype(bool)))
    reliability = []
    edges = np.linspace(0.0, 1.0, max(1, bins) + 1)
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        keep = (p >= left) & (p <= right if index == len(edges) - 2 else p < right)
        count = int(keep.sum())
        reliability.append(
            {
                "lower": float(left),
                "upper": float(right),
                "count": count,
                "confidence": float(p[keep].mean()) if count else 0.0,
                "event_rate": float(y[keep].mean()) if count else 0.0,
            }
        )
    ece = float(
        sum(item["count"] * abs(item["confidence"] - item["event_rate"]) for item in reliability)
        / n
    )
    return {"n": int(len(y)), "log_loss": log_loss, "brier": brier, "accuracy": accuracy, "ece": ece, "reliability": reliability}


def transition_counts(
    sequences: Iterable[Sequence[int]],
    lag: int = 1,
    num_states: int | None = None,
) -> np.ndarray:
    """Count transitions without connecting the end of one trajectory to the next."""
    if lag <= 0:
        raise ValueError("lag must be positive")
    sequence_list = [np.asarray(sequence, dtype=np.int64).reshape(-1) for sequence in sequences]
    if num_states is None:
        maximum = max((int(sequence.max()) for sequence in sequence_list if len(sequence)), default=-1)
        num_states = maximum + 1
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    counts = np.zeros((num_states, num_states), dtype=np.float64)
    for sequence in sequence_list:
        if np.any(sequence < 0) or np.any(sequence >= num_states):
            raise ValueError("state id outside num_states")
        for source, target in zip(sequence[:-lag], sequence[lag:]):
            counts[source, target] += 1.0
    return counts


def row_normalize_matrix(matrix: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    values = _as_float_array(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("transition matrix must be square")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    values = np.clip(values, 0.0, None) + alpha
    totals = values.sum(axis=1, keepdims=True)
    identity = np.eye(values.shape[0], dtype=np.float64) / max(values.shape[0], 1)
    return np.divide(values, totals, out=identity, where=totals > 1.0e-12)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Compute the stationary row distribution of a finite Markov operator."""
    matrix = row_normalize_matrix(transition)
    distribution = np.full(matrix.shape[0], 1.0 / matrix.shape[0], dtype=np.float64)
    for _ in range(10000):
        updated = distribution @ matrix
        if np.max(np.abs(updated - distribution)) < 1.0e-12:
            return updated / updated.sum()
        distribution = updated
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    candidate = np.real(eigenvectors[:, index])
    if candidate.sum() < 0.0:
        candidate = -candidate
    candidate = np.clip(candidate, 0.0, None)
    return candidate / max(candidate.sum(), 1.0e-12)


def implied_timescales(transition: np.ndarray, lag_time: float = 1.0, count: int | None = None) -> np.ndarray:
    """Return non-trivial implied timescales sorted from slow to fast."""
    if lag_time <= 0.0:
        raise ValueError("lag_time must be positive")
    eigenvalues = np.linalg.eigvals(row_normalize_matrix(transition))
    eigenvalues = np.sort(np.abs(np.real(eigenvalues)))[::-1]
    eigenvalues = eigenvalues[1:]
    times = []
    for value in eigenvalues:
        if value <= 1.0e-12:
            times.append(0.0)
        elif value >= 1.0 - 1.0e-12:
            times.append(float("inf"))
        else:
            times.append(float(-lag_time / np.log(value)))
    result = np.asarray(times, dtype=np.float64)
    return result[:count] if count is not None else result


def mean_first_passage_times(transition: np.ndarray) -> np.ndarray:
    """Compute discrete-time MFPTs between all states.

    A singular transient block yields ``inf`` for the affected source-target
    pair. This is preferable to silently reporting a finite pseudoinverse
    value for an unreachable transition.
    """
    matrix = row_normalize_matrix(transition)
    n_states = matrix.shape[0]
    output = np.zeros((n_states, n_states), dtype=np.float64)
    for target in range(n_states):
        sources = np.asarray([state for state in range(n_states) if state != target], dtype=np.int64)
        if not len(sources):
            continue
        system = np.eye(len(sources)) - matrix[np.ix_(sources, sources)]
        try:
            values = np.linalg.solve(system, np.ones(len(sources)))
        except np.linalg.LinAlgError:
            values = np.full(len(sources), np.inf)
        output[sources, target] = values
    return output


def _row_js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probabilities(left)
    right = normalize_probabilities(right)
    midpoint = 0.5 * (left + right)
    left_term = np.where(left > 0.0, left * np.log(np.clip(left / midpoint, 1.0e-12, None)), 0.0)
    right_term = np.where(right > 0.0, right * np.log(np.clip(right / midpoint, 1.0e-12, None)), 0.0)
    return 0.5 * (left_term + right_term).sum(axis=1)


def transition_operator_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    lag_time: float = 1.0,
    timescale_count: int = 5,
) -> dict:
    """Compare two transition operators and their kinetic consequences."""
    predicted_matrix = row_normalize_matrix(predicted)
    reference_matrix = row_normalize_matrix(reference)
    if predicted_matrix.shape != reference_matrix.shape:
        raise ValueError("predicted and reference operators must have equal shape")
    row_tv = 0.5 * np.abs(predicted_matrix - reference_matrix).sum(axis=1)
    stationary_pred = stationary_distribution(predicted_matrix)
    stationary_ref = stationary_distribution(reference_matrix)
    pred_mfpt = mean_first_passage_times(predicted_matrix)
    ref_mfpt = mean_first_passage_times(reference_matrix)
    finite = np.isfinite(pred_mfpt) & np.isfinite(ref_mfpt)
    finite &= ~np.eye(pred_mfpt.shape[0], dtype=bool)
    mfpt_error = float(np.mean(np.abs(pred_mfpt[finite] - ref_mfpt[finite]))) if finite.any() else float("inf")
    mfpt_relative = (
        float(np.mean(np.abs(pred_mfpt[finite] - ref_mfpt[finite]) / np.maximum(ref_mfpt[finite], 1.0e-12)))
        if finite.any()
        else float("inf")
    )
    return {
        "row_total_variation": float(np.mean(row_tv)),
        "row_js_divergence": float(np.mean(_row_js_divergence(predicted_matrix, reference_matrix))),
        "stationary_l1": float(np.abs(stationary_pred - stationary_ref).sum()),
        "predicted_stationary": stationary_pred.tolist(),
        "reference_stationary": stationary_ref.tolist(),
        "predicted_implied_timescales": implied_timescales(predicted_matrix, lag_time, timescale_count).tolist(),
        "reference_implied_timescales": implied_timescales(reference_matrix, lag_time, timescale_count).tolist(),
        "mfpt_mae": mfpt_error,
        "mfpt_relative_error": mfpt_relative,
        "mfpt_pairs": int(finite.sum()),
    }


def chapman_kolmogorov_error(
    predicted_tau: np.ndarray,
    predicted_two_tau: np.ndarray,
) -> float:
    """Measure the normalized semigroup defect of two independently predicted operators."""
    left = row_normalize_matrix(predicted_tau)
    direct = row_normalize_matrix(predicted_two_tau)
    composed = left @ left
    return float(np.linalg.norm(direct - composed, ord="fro") / max(np.sqrt(direct.size), 1.0))


def integrated_brier_score(
    event_probabilities: np.ndarray,
    event_times: np.ndarray,
    event_observed: np.ndarray,
    horizons: Sequence[float],
) -> dict:
    """Score predicted event-by-horizon probabilities with right censoring.

    ``event_times`` is the first event time or censoring time. At horizon `h`,
    rows censored before `h` are omitted because their event status is unknown.
    """
    probabilities = np.clip(_as_float_array(event_probabilities), 0.0, 1.0)
    times = _as_float_array(event_times).reshape(-1)
    observed = np.asarray(event_observed, dtype=bool).reshape(-1)
    grid = _as_float_array(horizons).reshape(-1)
    if probabilities.shape != (len(times), len(grid)) or len(observed) != len(times):
        raise ValueError("event arrays have incompatible shapes")
    scores = []
    counts = []
    for index, horizon in enumerate(grid):
        known = observed | (times >= horizon)
        target = (observed & (times <= horizon)).astype(np.float64)
        count = int(known.sum())
        scores.append(float(np.mean(np.square(probabilities[known, index] - target[known]))) if count else float("nan"))
        counts.append(count)
    valid = np.isfinite(scores)
    return {
        "integrated_brier_score": float(np.mean(np.asarray(scores)[valid])) if valid.any() else float("nan"),
        "by_horizon": {str(float(h)): {"brier": score, "n": count} for h, score, count in zip(grid, scores, counts)},
    }


def empirical_survival(event_times: np.ndarray, event_observed: np.ndarray, horizons: Sequence[float]) -> np.ndarray:
    """Return the Kaplan-Meier survival curve on a fixed horizon grid."""
    times = _as_float_array(event_times).reshape(-1)
    observed = np.asarray(event_observed, dtype=bool).reshape(-1)
    curve = []
    for horizon in _as_float_array(horizons):
        survival = 1.0
        for time in np.sort(np.unique(times[(times <= horizon) & observed])):
            at_risk = np.sum(times >= time)
            events = np.sum((times == time) & observed)
            if at_risk:
                survival *= 1.0 - events / at_risk
        curve.append(survival)
    return np.asarray(curve, dtype=np.float64)


def survival_curve_error(
    predicted_event_probabilities: np.ndarray,
    event_times: np.ndarray,
    event_observed: np.ndarray,
    horizons: Sequence[float],
) -> dict:
    """Compare predicted event probabilities with the held-out survival curve."""
    grid = _as_float_array(horizons)
    observed_survival = empirical_survival(event_times, event_observed, grid)
    predicted_survival = 1.0 - np.mean(np.clip(_as_float_array(predicted_event_probabilities), 0.0, 1.0), axis=0)
    return {
        "survival_mae": float(np.mean(np.abs(predicted_survival - observed_survival))),
        "survival_rmse": float(np.sqrt(np.mean(np.square(predicted_survival - observed_survival)))),
        "predicted_survival": predicted_survival.tolist(),
        "observed_survival": observed_survival.tolist(),
    }


def path_feature_mmd(
    predicted_features: np.ndarray,
    reference_features: np.ndarray,
    bandwidth: float | None = None,
) -> float:
    """Compute a bounded RBF MMD between generated and reference path features."""
    left = _as_float_array(predicted_features)
    right = _as_float_array(reference_features)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("path features must be [N, D] with matching D")
    combined = np.concatenate([left, right], axis=0)
    distances = np.sqrt(np.maximum(((combined[:, None] - combined[None, :]) ** 2).sum(axis=-1), 0.0))
    if bandwidth is None:
        nonzero = distances[distances > 0.0]
        bandwidth = float(np.median(nonzero)) if nonzero.size else 1.0
    bandwidth = max(float(bandwidth), 1.0e-12)
    kernel = np.exp(-0.5 * np.square(distances / bandwidth))
    n_left, n_right = len(left), len(right)
    xx = kernel[:n_left, :n_left]
    yy = kernel[n_left:, n_left:]
    xy = kernel[:n_left, n_left:]
    return float(xx.mean() + yy.mean() - 2.0 * xy.mean())


def _quantile_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    """One-dimensional Wasserstein distance without requiring scipy."""
    left = np.sort(_as_float_array(left).reshape(-1))
    right = np.sort(_as_float_array(right).reshape(-1))
    if not len(left) or not len(right):
        return float("nan")
    grid = np.linspace(0.0, 1.0, max(len(left), len(right)))
    left_quantiles = np.quantile(left, grid)
    right_quantiles = np.quantile(right, grid)
    return float(np.mean(np.abs(left_quantiles - right_quantiles)))


def path_ensemble_metrics(
    predicted_paths: Sequence[Sequence[object]],
    reference_paths: Sequence[Sequence[object]],
    source_state: object = "unfolded",
    target_state: object = "folded",
) -> dict:
    """Compare sampled A-to-B paths using path-level statistics.

    A path is represented by a sequence of discrete states. Paths that do not
    start in ``source_state`` or do not reach ``target_state`` are retained in
    the failure count but excluded from transition-time statistics.
    """
    def valid_paths(paths: Sequence[Sequence[object]]) -> tuple[list[list[object]], int]:
        valid = []
        for path in paths:
            values = list(path)
            if values and values[0] == source_state and target_state in values[1:]:
                valid.append(values)
        return valid, len(paths) - len(valid)

    predicted, predicted_failures = valid_paths(predicted_paths)
    reference, reference_failures = valid_paths(reference_paths)
    if not predicted or not reference:
        return {
            "predicted_paths": len(predicted_paths),
            "reference_paths": len(reference_paths),
            "predicted_failure_rate": predicted_failures / max(len(predicted_paths), 1),
            "reference_failure_rate": reference_failures / max(len(reference_paths), 1),
            "transition_time_wasserstein": float("nan"),
            "path_feature_mmd": float("nan"),
            "path_coverage": 0.0,
        }

    def transition_time(path: list[object]) -> int:
        return next(index for index, state in enumerate(path[1:], start=1) if state == target_state)

    states = sorted({state for path in predicted + reference for state in path}, key=str)

    def features(paths: list[list[object]]) -> np.ndarray:
        rows = []
        for path in paths:
            counts = [path.count(state) / len(path) for state in states]
            rows.append([float(transition_time(path)), float(len(path)), *counts])
        return np.asarray(rows, dtype=np.float64)

    predicted_times = np.asarray([transition_time(path) for path in predicted], dtype=np.float64)
    reference_times = np.asarray([transition_time(path) for path in reference], dtype=np.float64)
    predicted_features = features(predicted)
    reference_features = features(reference)
    distances = np.sqrt(
        np.maximum(
            ((reference_features[:, None, :] - predicted_features[None, :, :]) ** 2).sum(axis=-1),
            0.0,
        )
    )
    threshold = float(np.percentile(distances.min(axis=1), 90.0))
    return {
        "predicted_paths": len(predicted_paths),
        "reference_paths": len(reference_paths),
        "predicted_valid_paths": len(predicted),
        "reference_valid_paths": len(reference),
        "predicted_failure_rate": predicted_failures / max(len(predicted_paths), 1),
        "reference_failure_rate": reference_failures / max(len(reference_paths), 1),
        "transition_time_wasserstein": _quantile_wasserstein(predicted_times, reference_times),
        "path_feature_mmd": path_feature_mmd(predicted_features, reference_features),
        "path_coverage": float(np.mean(distances.min(axis=1) <= threshold)),
        "predicted_transition_time_mean": float(predicted_times.mean()),
        "reference_transition_time_mean": float(reference_times.mean()),
    }


def grouped_bootstrap(
    values: Sequence[float],
    groups: Sequence[str],
    statistic: Callable[[np.ndarray], float] = np.mean,
    samples: int = 1000,
    seed: int = 0,
) -> dict:
    """Bootstrap a statistic over groups rather than individual frames."""
    values_array = _as_float_array(values)
    group_array = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(group_array.tolist())), dtype=object)
    if len(values_array) != len(group_array) or not len(unique):
        raise ValueError("values and groups must be non-empty and have equal length")
    group_values = [values_array[group_array == group] for group in unique]
    point = float(statistic(np.asarray([statistic(item) for item in group_values])))
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(max(1, samples)):
        selected = rng.randint(0, len(group_values), size=len(group_values))
        boot.append(float(statistic(np.asarray([statistic(group_values[index]) for index in selected]))))
    interval = np.percentile(boot, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(interval[0]), float(interval[1])], "groups": int(len(unique)), "samples": int(samples)}
