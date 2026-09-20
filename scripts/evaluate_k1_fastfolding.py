#!/usr/bin/env python3
"""K1 FastFolding comparison with train-fitted baselines.

The evaluator uses the same held-out event samples for every method:

* Persistence: keep the current discrete state.
* MSM: a three-state transition operator fitted on training scaffolds.
* tICA+MSM: native deeptime tICA plus a train-fitted clustered MSM.
* JEPA: probabilities produced by the frozen LSP-WM checkpoint and a
  train-scaffold-only readout.

The reference committor is the exact committor of the train-fitted reference
MSM.  Test labels are used only as held-out targets and to build the
held-out reference operator used for operator diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from protein_jepa_frame.eval.dynamics_benchmarks import (  # noqa: E402
    implied_timescales,
    multiclass_forecast_metrics,
    row_normalize_matrix,
    transition_counts,
    transition_operator_metrics,
)
from protein_jepa_frame.fastfolding.benchmark import (  # noqa: E402
    assign_group_splits,
    build_event_samples,
    load_jsonl,
    split_label_rows,
)


STATE_NAMES = ("unfolded", "transition", "folded")
STATE_TO_ID = {name: index for index, name in enumerate(STATE_NAMES)}


def _read_jsonl(path: Path) -> list[dict]:
    return load_jsonl(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_label_scheme(
    rows: list[dict],
    scheme: str,
    q_unfolded_max: float,
    q_folded_min: float,
) -> list[dict]:
    """Optionally re-derive ``state`` from the native contact fraction.

    The pre-annotated ``state`` column leaves only 8 ``unfolded`` frames in the
    whole 2600-frame label file, so folded-unfolded kinetics are not estimable
    under it.  ``q_native`` (native contact fraction, 0..1) is present for every
    frame, so a threshold scheme recovers a balanced three-state labelling
    without touching the split, the model predictions or the metric code:

    * ``unfolded``   ⇔ ``q_native <= q_unfolded_max``
    * ``folded``     ⇔ ``q_native >= q_folded_min``
    * ``transition`` ⇔ otherwise
    """
    if scheme == "native_state":
        return rows
    if scheme != "q_native":
        raise ValueError(f"unknown label scheme {scheme!r}")
    if not 0.0 <= q_unfolded_max < q_folded_min <= 1.0:
        raise ValueError(
            f"q_native thresholds must satisfy 0 <= {q_unfolded_max} < "
            f"{q_folded_min} <= 1"
        )
    relabelled = []
    for row in rows:
        if row.get("q_native") is None:
            raise ValueError(
                f"label row {row.get('protein_id')}/{row.get('frame')} has no q_native"
            )
        value = float(row["q_native"])
        if value <= q_unfolded_max:
            state = "unfolded"
        elif value >= q_folded_min:
            state = "folded"
        else:
            state = "transition"
        updated = dict(row)
        updated["state"] = state
        updated["state_source"] = "q_native_thresholds"
        relabelled.append(updated)
    return relabelled


def _group_sequences(rows: Iterable[dict]) -> list[list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["protein_id"], row.get("trajectory_id", row["protein_id"]))].append(row)
    sequences = []
    for values in groups.values():
        sequences.append(sorted(values, key=lambda row: int(row["frame"])))
    return sorted(sequences, key=lambda value: (value[0]["protein_id"], int(value[0]["frame"])))


def _state_sequences(rows: Iterable[dict]) -> list[np.ndarray]:
    output = []
    for sequence in _group_sequences(rows):
        values = [STATE_TO_ID.get(row.get("state")) for row in sequence]
        if all(value is not None for value in values) and len(values) >= 2:
            output.append(np.asarray(values, dtype=np.int64))
    return output


def _fit_reference_operator(rows: Iterable[dict], alpha: float = 1.0) -> np.ndarray:
    counts = transition_counts(_state_sequences(rows), lag=1, num_states=len(STATE_NAMES))
    return row_normalize_matrix(counts, alpha=alpha)


def _solve_committor(transition: np.ndarray, source: int = 0, target: int = 2) -> np.ndarray:
    n_states = transition.shape[0]
    boundary = np.asarray([source, target], dtype=np.int64)
    interior = np.asarray([state for state in range(n_states) if state not in boundary], dtype=np.int64)
    result = np.zeros(n_states, dtype=np.float64)
    result[target] = 1.0
    if len(interior):
        system = np.eye(len(interior)) - transition[np.ix_(interior, interior)]
        rhs = transition[np.ix_(interior, [target])].reshape(-1)
        result[interior] = np.linalg.solve(system, rhs)
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1.0e-12 or np.std(right) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic; NaN when one class is absent."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    ranks = _rankdata(scores) + 1.0
    positives = labels.astype(bool)
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _committor_metrics(probabilities: np.ndarray, source_states: np.ndarray, committor: np.ndarray) -> dict:
    """Committor comparison against the train-MSM committor of the source state.

    NOTE (metric degeneracy): ``probabilities @ committor`` reduces to
    ``committor[source_state]`` whenever the predictor is the identity operator
    (Persistence).  That makes the historical ``brier`` and ``pearson`` columns
    exactly 0 and 1 by construction, i.e. they were tautologies, not skill.  The
    identity control is reported explicitly below and ``_fold_forecast_metrics``
    provides the non-tautological alternative.
    """
    predicted = probabilities @ committor
    target = committor[source_states]
    identity = committor[source_states]
    return {
        "pearson": _correlation(predicted, target),
        "spearman": _correlation(_rankdata(predicted), _rankdata(target)),
        "mae": float(np.mean(np.abs(predicted - target))),
        "brier": float(np.mean(np.square(predicted - target))),
        "identity_operator_brier": float(np.mean(np.square(identity - target))),
        "constant_prior_brier": float(np.mean(np.square(float(np.mean(target)) - target))),
        "n_distinct_target_values": int(len(np.unique(np.round(target, 9)))),
        "auc": _roc_auc(predicted, (target > 0.5).astype(np.int64)),
        "n": int(len(target)),
    }


def _fold_forecast_metrics(probabilities: np.ndarray, target_states: np.ndarray) -> dict:
    """Non-tautological committor proxy.

    The target is the held-out *observed* fold indicator at the horizon frame
    (``1[target_state == folded]``) and the prediction is the model's folded
    probability column.  Persistence scores ``P(source == folded)`` on 98% of
    samples, so it cannot trivially reach zero error here.
    """
    folded = probabilities[:, STATE_TO_ID["folded"]]
    unfolded = probabilities[:, STATE_TO_ID["unfolded"]]
    label = (target_states == STATE_TO_ID["folded"]).astype(np.int64)
    base_rate = float(label.mean())
    return {
        "folded_brier": float(np.mean(np.square(folded - label))),
        "folded_log_loss": float(-np.mean(
            label * np.log(np.clip(folded, 1.0e-12, 1.0))
            + (1 - label) * np.log(np.clip(1.0 - folded, 1.0e-12, 1.0))
        )),
        "folded_auc": _roc_auc(folded, label),
        "unfolded_brier": float(np.mean(np.square(
            unfolded - (target_states == STATE_TO_ID["unfolded"]).astype(np.float64)))),
        "constant_base_rate_brier": float(np.mean(np.square(base_rate - label))),
        "base_rate": base_rate,
        "n": int(len(label)),
        "n_positive": int(label.sum()),
    }


def _bootstrap(values: dict[str, float], samples: int, seed: int) -> dict:
    finite = np.asarray([value for value in values.values() if np.isfinite(value)], dtype=np.float64)
    if len(finite) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [float(np.mean(finite[rng.integers(0, len(finite), size=len(finite))])) for _ in range(samples)],
        dtype=np.float64,
    )
    return {
        "mean": float(np.mean(finite)),
        "lower": float(np.percentile(draws, 2.5)),
        "upper": float(np.percentile(draws, 97.5)),
        "n": int(len(finite)),
    }


def _power_operator(transition: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return row_normalize_matrix(np.linalg.matrix_power(transition, horizon))


def _persistence_probabilities(source_states: np.ndarray, horizon: int) -> np.ndarray:
    del horizon
    return np.eye(len(STATE_NAMES), dtype=np.float64)[source_states]


def _state_msm_probabilities(
    source_states: np.ndarray, transition: np.ndarray, horizon: int
) -> np.ndarray:
    operator = _power_operator(transition, horizon)
    return operator[source_states]


def _feature_matrix(rows: Iterable[dict]) -> np.ndarray:
    features = []
    for row in rows:
        features.append(
            [
                float(row.get("q_native", 0.0)),
                float(row.get("rmsd_to_ref", 0.0)),
                float(row.get("radius_of_gyration", 0.0)),
            ]
        )
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("FastFolding labels do not contain finite physical features")
    return values


class TICAMSM:
    """A train-only tICA plus clustered MSM observation model."""

    def __init__(self, n_clusters: int, lag: int, seed: int):
        self.n_clusters = int(n_clusters)
        self.lag = int(lag)
        self.seed = int(seed)

    def fit(self, rows: list[dict]) -> "TICAMSM":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from deeptime.decomposition import TICA

        sequences = _group_sequences(rows)
        train_features = [_feature_matrix(sequence) for sequence in sequences]
        scaler = StandardScaler().fit(np.concatenate(train_features, axis=0))
        scaled = [scaler.transform(value) for value in train_features]
        tica = TICA(
            lagtime=self.lag,
            dim=min(2, scaled[0].shape[1]),
        ).fit(scaled).fetch_model()
        embedded = [np.asarray(tica.transform(value), dtype=np.float64) for value in scaled]
        self.scaler = scaler
        self.tica = tica
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(np.concatenate(embedded, axis=0))),
            n_init=20,
            random_state=self.seed,
        ).fit(np.concatenate(embedded, axis=0))
        self.n_states = int(self.kmeans.n_clusters)
        cluster_sequences = [self.kmeans.predict(value) for value in embedded]
        self.transition = row_normalize_matrix(
            transition_counts(cluster_sequences, lag=1, num_states=self.n_states),
            alpha=1.0,
        )
        state_counts = np.ones((self.n_states, len(STATE_NAMES)), dtype=np.float64)
        for sequence, clusters in zip(sequences, cluster_sequences):
            labels = np.asarray([STATE_TO_ID[row["state"]] for row in sequence], dtype=np.int64)
            for cluster, state in zip(clusters, labels):
                state_counts[cluster, state] += 1.0
        self.emission = state_counts / state_counts.sum(axis=1, keepdims=True)
        self.train_rows = len(rows)
        return self

    def probabilities(self, rows: list[dict], horizon: int) -> np.ndarray:
        features = self.scaler.transform(_feature_matrix(rows))
        embedded = np.asarray(self.tica.transform(features), dtype=np.float64)
        clusters = self.kmeans.predict(embedded)
        operator = _power_operator(self.transition, horizon)
        return operator[clusters] @ self.emission


def _load_model_predictions(path: Path) -> dict[str, np.ndarray]:
    rows = _read_jsonl(path)
    output = {}
    for row in rows:
        sample_id = row.get("sample_id")
        probabilities = row.get("state_probs")
        if sample_id is None or probabilities is None:
            continue
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (len(STATE_NAMES),):
            raise ValueError(f"invalid JEPA state_probs for {sample_id}")
        output[str(sample_id)] = values
    if not output:
        raise ValueError(f"no JEPA state predictions found in {path}")
    return output


def _prediction_rows(
    samples: list[dict],
    probabilities_by_id: dict[str, np.ndarray],
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    joined = []
    for sample in samples:
        probability = probabilities_by_id.get(sample["sample_id"])
        if probability is None:
            continue
        target = STATE_TO_ID.get(sample.get("target_state"))
        source = STATE_TO_ID.get(sample.get("source_state"))
        if target is None or source is None:
            continue
        joined.append(
            {
                "sample": sample,
                "probabilities": probability,
                "target": target,
                "source": source,
            }
        )
    if not joined:
        raise ValueError("no predictions matched held-out samples")
    return (
        joined,
        np.asarray([row["probabilities"] for row in joined], dtype=np.float64),
        np.asarray([row["target"] for row in joined], dtype=np.int64),
        np.asarray([row["source"] for row in joined], dtype=np.int64),
    )


def _operator_for_rows(joined: list[dict], probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_counts = np.zeros((len(STATE_NAMES), len(STATE_NAMES)), dtype=np.float64)
    predicted_counts = np.zeros_like(reference_counts)
    for row, probability in zip(joined, probabilities):
        reference_counts[row["source"], row["target"]] += 1.0
        predicted_counts[row["source"]] += probability
    return (
        row_normalize_matrix(predicted_counts, alpha=1.0),
        row_normalize_matrix(reference_counts, alpha=1.0),
    )


def _its_relative_error(pred_times: np.ndarray, ref_times: np.ndarray) -> float:
    """Symmetric log-timescale error scored on the *reference* timescale set.

    The timescale set to score on is defined only by the reference operator
    (the ground-truth train-MSM), which is identical for every method.  A
    predicted timescale that collapses to 0 or diverges to inf — the signature
    of a rank-deficient (e.g. constant-prior) operator — is *penalised*, not
    silently dropped.  The historical code masked on ``pred_times > 0`` too,
    which excluded a degenerate predictor's own failed timescales from its
    error and so rewarded exactly the degeneracy it should punish (a
    constant-prior JEPA operator scored on a single non-degenerate timescale
    beat MSM baselines scored on two).

    Degenerate predicted timescales are floored/capped to the finite dynamic
    range of the valid reference timescales before taking logs, so the error
    is a large finite number rather than ``inf``.
    """
    pred_times = np.asarray(pred_times, dtype=np.float64)
    ref_times = np.asarray(ref_times, dtype=np.float64)
    reference_valid = np.isfinite(ref_times) & (ref_times > 0.0)
    if not reference_valid.any():
        return float("nan")
    ref_valid = ref_times[reference_valid]
    floor = float(ref_valid.min()) * 1.0e-3
    cap = float(ref_valid.max()) * 1.0e3
    pred_scored = pred_times[reference_valid]
    # Map non-finite (inf timescale = eigenvalue 1) predictions onto the cap so
    # they contribute a bounded, large error instead of being dropped.
    pred_scored = np.where(np.isfinite(pred_scored), pred_scored, cap)
    pred_scored = np.clip(pred_scored, floor, cap)
    return float(np.mean(np.abs(np.log(pred_scored) - np.log(ref_valid))))


def _per_protein_operator_metrics(joined: list[dict], probabilities: np.ndarray) -> dict[str, dict]:
    by_protein: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(joined):
        by_protein[row["sample"]["protein_id"]].append(index)
    output = {}
    for protein, indices in sorted(by_protein.items()):
        subset = [joined[index] for index in indices]
        pred, ref = _operator_for_rows(subset, probabilities[indices])
        metrics = transition_operator_metrics(pred, ref, lag_time=1.0, timescale_count=2)
        metrics["its_relative_error"] = _its_relative_error(
            metrics["predicted_implied_timescales"],
            metrics["reference_implied_timescales"],
        )
        output[protein] = metrics
    return output


def _aggregate_operator(
    joined: list[dict],
    probabilities: np.ndarray,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, dict]:
    pooled_pred, pooled_ref = _operator_for_rows(joined, probabilities)
    pooled = transition_operator_metrics(pooled_pred, pooled_ref, lag_time=1.0, timescale_count=2)
    pooled["its_relative_error"] = _its_relative_error(
        pooled["predicted_implied_timescales"],
        pooled["reference_implied_timescales"],
    )
    per_protein = _per_protein_operator_metrics(joined, probabilities)
    aggregate = {}
    for metric in (
        "row_js_divergence",
        "row_total_variation",
        "stationary_l1",
        "its_relative_error",
        "mfpt_relative_error",
    ):
        aggregate[metric] = _bootstrap(
            {protein: values[metric] for protein, values in per_protein.items()},
            bootstrap_samples,
            seed,
        )
    aggregate["n_proteins"] = len(per_protein)
    return {"pooled": pooled, "bootstrap": aggregate}, per_protein


def evaluate(args: argparse.Namespace) -> dict:
    labels_path = Path(args.labels)
    prediction_path = Path(args.predictions)
    rows = _apply_label_scheme(
        _read_jsonl(labels_path),
        args.label_scheme,
        args.q_unfolded_max,
        args.q_folded_min,
    )
    assignments = assign_group_splits(
        rows,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    splits = split_label_rows(rows, assignments)
    train_samples = build_event_samples(splits["train"], args.context_frames, args.horizons)
    test_samples = build_event_samples(splits["test"], args.context_frames, args.horizons)
    model_predictions = _load_model_predictions(prediction_path)
    reference_operator = _fit_reference_operator(splits["train"])
    reference_committor = _solve_committor(reference_operator)
    train_state_counts = np.zeros(len(STATE_NAMES), dtype=np.float64)
    for row in splits["train"]:
        index = STATE_TO_ID.get(row.get("state"))
        if index is not None:
            train_state_counts[index] += 1.0
    train_marginal = train_state_counts / max(train_state_counts.sum(), 1.0)

    tica_model = TICAMSM(args.tica_clusters, args.tica_lag, args.seed).fit(splits["train"])
    methods = {"Persistence": {}, "MajorityClass": {}, "MSM": {}, "tICA+MSM": {}, "JEPA": {}}
    horizons = sorted({int(value) for value in args.horizons})
    output_by_horizon: dict[str, dict] = {}

    for horizon in horizons:
        horizon_samples = [sample for sample in test_samples if int(sample["horizon"]) == horizon]
        if not horizon_samples:
            continue
        source_states = np.asarray([STATE_TO_ID[sample["source_state"]] for sample in horizon_samples], dtype=np.int64)
        target_states = np.asarray([STATE_TO_ID[sample["target_state"]] for sample in horizon_samples], dtype=np.int64)
        target_rows = [sample["context_rows"][-1] for sample in horizon_samples]
        majority = np.repeat(train_marginal[None, :], len(horizon_samples), axis=0)
        baseline_probs = {
            "Persistence": _persistence_probabilities(source_states, horizon),
            "MajorityClass": majority,
            "MSM": _state_msm_probabilities(source_states, reference_operator, horizon),
            "tICA+MSM": tica_model.probabilities(target_rows, horizon),
        }
        jepa_joined, jepa_probs, _, _ = _prediction_rows(horizon_samples, model_predictions)
        all_method_metrics = {}
        for method_name, probabilities in baseline_probs.items():
            joined = [
                {
                    "sample": sample,
                    "probabilities": probability,
                    "target": target,
                    "source": source,
                }
                for sample, probability, target, source in zip(
                    horizon_samples, probabilities, target_states, source_states
                )
            ]
            forecast = multiclass_forecast_metrics(probabilities, target_states)
            forecast["committor_vs_train_msm"] = _committor_metrics(
                probabilities, source_states, reference_committor)
            forecast["fold_forecast"] = _fold_forecast_metrics(probabilities, target_states)
            operator, per_protein = _aggregate_operator(
                joined, probabilities, args.bootstrap_samples, args.seed
            )
            all_method_metrics[method_name] = {
                "forecast": forecast,
                "operator": operator,
                "per_protein_operator": per_protein,
                "n_samples": len(joined),
            }

        jepa_forecast = multiclass_forecast_metrics(jepa_probs, np.asarray([row["target"] for row in jepa_joined]))
        jepa_sources = np.asarray([row["source"] for row in jepa_joined], dtype=np.int64)
        jepa_targets = np.asarray([row["target"] for row in jepa_joined], dtype=np.int64)
        jepa_forecast["committor_vs_train_msm"] = _committor_metrics(
            jepa_probs, jepa_sources, reference_committor)
        jepa_forecast["fold_forecast"] = _fold_forecast_metrics(jepa_probs, jepa_targets)
        jepa_operator, jepa_per_protein = _aggregate_operator(
            jepa_joined, jepa_probs, args.bootstrap_samples, args.seed
        )
        all_method_metrics["JEPA"] = {
            "forecast": jepa_forecast,
            "operator": jepa_operator,
            "per_protein_operator": jepa_per_protein,
            "n_samples": len(jepa_joined),
        }
        all_method_metrics["class_imbalance"] = {
            "n_samples": len(horizon_samples),
            "source_state_counts": {
                name: int(np.sum(source_states == index)) for index, name in enumerate(STATE_NAMES)
            },
            "target_state_counts": {
                name: int(np.sum(target_states == index)) for index, name in enumerate(STATE_NAMES)
            },
            "majority_class": STATE_NAMES[int(np.argmax(np.bincount(target_states, minlength=len(STATE_NAMES))))],
            "majority_class_fraction": float(np.max(np.bincount(target_states, minlength=len(STATE_NAMES))) / len(target_states)),
            "jepa_samples": len(jepa_joined),
        }
        output_by_horizon[str(horizon)] = all_method_metrics

    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in ("nll", "brier", "accuracy", "balanced_accuracy", "macro_f1", "ece"):
            values = [
                horizon[method]["forecast"][metric]
                for horizon in output_by_horizon.values() if method in horizon
                and metric in horizon[method]["forecast"]
            ]
            summary[method][metric] = float(np.mean(values)) if values else float("nan")
        for metric in ("folded_brier", "folded_log_loss", "folded_auc", "constant_base_rate_brier"):
            values = [
                horizon[method]["forecast"]["fold_forecast"][metric]
                for horizon in output_by_horizon.values() if method in horizon
            ]
            finite = [value for value in values if np.isfinite(value)]
            summary[method][f"fold_forecast_{metric}"] = (
                float(np.mean(finite)) if finite else float("nan"))
        for metric in ("row_js_divergence", "row_total_variation", "stationary_l1",
                       "its_relative_error", "mfpt_relative_error"):
            values = [
                horizon[method]["operator"]["bootstrap"][metric]["mean"]
                for horizon in output_by_horizon.values() if method in horizon
            ]
            summary[method][metric] = float(np.mean(values)) if values else float("nan")

    # Degenerate-metric audit: for every headline metric, the best value reached
    # by a method with no dynamics at all (Persistence / MajorityClass).  A model
    # claim is only meaningful where it beats this floor.
    degenerate_metric_audit = {}
    for metric in ("nll", "brier", "accuracy", "balanced_accuracy", "macro_f1", "ece"):
        per_horizon = {}
        for horizon_key, payload in output_by_horizon.items():
            controls = [
                payload[name]["forecast"][metric]
                for name in ("Persistence", "MajorityClass")
                if name in payload and metric in payload[name]["forecast"]
            ]
            best = min(controls) if metric in {"nll", "brier", "ece"} else max(controls)
            jepa = payload.get("JEPA", {}).get("forecast", {}).get(metric, float("nan"))
            better = (
                (jepa < best) if metric in {"nll", "brier", "ece"} else (jepa > best)
            ) if np.isfinite(jepa) else False
            per_horizon[horizon_key] = {
                "best_degenerate_control": float(best),
                "jepa": float(jepa),
                "jepa_beats_control": bool(better),
                "margin": float(best - jepa) if metric in {"nll", "brier", "ece"} else float(jepa - best),
            }
        degenerate_metric_audit[metric] = per_horizon

    label_state_counts = {}
    for row in rows:
        name = row.get("state")
        if name is not None:
            label_state_counts[name] = label_state_counts.get(name, 0) + 1
    unfolded_frames = int(label_state_counts.get("unfolded", 0))
    unfolded_estimable = bool(unfolded_frames >= 50)

    return {
        "experiment_id": "K1",
        "protocol": {
            "name": "fastfolding_train_scaffold_holdout_v2",
            "dataset": "FastFolding",
            "labels_sha256": _sha256(labels_path),
            "predictions_sha256": _sha256(prediction_path),
            "context_frames": args.context_frames,
            "horizons": horizons,
            "state_names": list(STATE_NAMES),
            "split": "scaffold-disjoint",
            "train_fraction": args.train_fraction,

            "validation_fraction": args.validation_fraction,
            "reference_operator_fit": "training scaffold state sequences, alpha=1.0",
            "reference_committor": reference_committor.tolist(),
            "reference_committor_definition": "exact discrete committor of the train-fitted three-state MSM",
            "committor_metric_status": (
                "TAUTOLOGICAL for identity predictors: probabilities @ q_train == q_train[source] "
                "for Persistence, so brier/pearson are 0/1 by construction. Use "
                "by_horizon[*][method].forecast.fold_forecast for the non-tautological comparison."
            ),
            "label_state_counts": label_state_counts,
            "unfolded_frames_in_label_file": unfolded_frames,
            "label_scheme": args.label_scheme,
            "label_scheme_thresholds": (
                {"q_unfolded_max": args.q_unfolded_max, "q_folded_min": args.q_folded_min}
                if args.label_scheme == "q_native" else {}
            ),
            "bootstrap_unit": "protein",
            "bootstrap_samples": args.bootstrap_samples,
            "no_test_label_fitting": True,
        },
        "provenance": {
            "labels": str(labels_path),
            "predictions": str(prediction_path),
            "seed": args.seed,
            "python": sys.executable,
            "platform": platform.platform(),
            "split_assignments": assignments,
            "split_summary": {
                name: {
                    "rows": len(values),
                    "proteins": sorted({row["protein_id"] for row in values}),
                    "scaffolds": sorted({row["protein_id"].rsplit("-", 2)[0] for row in values}),
                }
                for name, values in splits.items()
            },
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "tica": {
                "lag": args.tica_lag,
                "clusters": args.tica_clusters,
                "train_rows": tica_model.train_rows,
                "native_deeptime": True,
            },
        },
        "summary": summary,
        "by_horizon": output_by_horizon,
        "degenerate_metric_audit": degenerate_metric_audit,
        "claim_boundary": (
            "JEPA is compared on the same held-out event samples as the train-fitted "
            "baselines. Committor is an exact train-MSM committor and is tautological for "
            "identity predictors, so only fold_forecast columns (and operator metrics) are "
            "used for claims. No superiority is claimed when the reported protein bootstrap "
            "interval overlaps zero or when the method does not beat the degenerate control. "
            "Labels come from the '" + args.label_scheme + "' scheme: results from different "
            "label schemes are not comparable."
        ),
        "gates": {
            "strict_run_complete": True,
            "native_tica_used": True,
            "true_committor_used": True,
            "majority_class_control_present": True,
            "unfolded_kinetics_estimable": unfolded_estimable,
            "jepa_beats_degenerate_control_on_nll": bool(
                all(entry["jepa_beats_control"]
                    for entry in degenerate_metric_audit["nll"].values())),
            "jepa_beats_degenerate_control_on_balanced_accuracy": bool(
                all(entry["jepa_beats_control"]
                    for entry in degenerate_metric_audit["balanced_accuracy"].values())),
            "all_requested_test_proteins_present": set(
                ["2F4K-0-protein", "CLN025-0-protein", "GTT-0-protein", "GTT-1-protein"]
            ).issubset(set(row["protein_id"] for row in splits["test"])),
        },
        "errors": [],
        "warnings": [
            "FastFolding source labels are sampled PDB frames; metrics are indexed by the sampled sequence order.",
            "The JEPA input file is a frozen checkpoint prediction artifact; it is not refit on test labels.",
            (
                "Under the '%s' label scheme the file contains %d 'unfolded' frames "
                "(counts: %s), so folded-unfolded kinetics %s."
                % (
                    args.label_scheme, unfolded_frames, label_state_counts,
                    "remain not estimable and accuracy stays dominated by the majority class"
                    if not unfolded_estimable
                    else "are estimable and the majority class no longer dominates",
                )
            ),
            "committor_vs_train_msm columns are tautological for Persistence (brier == 0 by construction); use fold_forecast.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--tica-lag", type=int, default=1)
    parser.add_argument("--tica-clusters", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument(
        "--label-scheme", choices=("native_state", "q_native"), default="native_state",
        help="native_state = the pre-annotated state column; q_native = re-derive "
             "unfolded/folded/transition from the native contact fraction",
    )
    parser.add_argument("--q-unfolded-max", type=float, default=0.4)
    parser.add_argument("--q-folded-min", type=float, default=0.6)
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
