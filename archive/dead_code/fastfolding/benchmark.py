"""Protein-disjoint FastFolding benchmark preparation and scoring."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from protein_jepa_frame.eval.dynamics_benchmarks import (
    binary_forecast_metrics,
    grouped_bootstrap,
    integrated_brier_score,
    multiclass_forecast_metrics,
    path_ensemble_metrics,
    row_normalize_matrix,
    survival_curve_error,
    transition_operator_metrics,
)


_SCAFFOLD_RE = re.compile(r"^(?P<scaffold>.+)-\d+-protein$")


def scaffold_id(protein_id: str) -> str:
    """Group numbered FastFolding replicas by their shared scaffold name."""
    match = _SCAFFOLD_RE.match(str(protein_id))
    return match.group("scaffold") if match else str(protein_id)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def assign_group_splits(
    rows: Sequence[dict],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, str]:
    """Assign all rows from a scaffold to one split.

    Fractions are applied to scaffold groups, not frames. For very small
    datasets, the function keeps each non-empty split populated when possible.
    """
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave test data")
    groups = sorted({scaffold_id(row["protein_id"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("at least three scaffold groups are required")
    rng = np.random.RandomState(seed)
    shuffled = list(np.asarray(groups)[rng.permutation(len(groups))])
    train_count = min(max(1, int(round(len(groups) * train_fraction)),), len(groups) - 2)
    validation_count = min(
        max(1, int(round(len(groups) * validation_fraction))),
        len(groups) - train_count - 1,
    )
    assignments = {}
    for group in shuffled[:train_count]:
        assignments[group] = "train"
    for group in shuffled[train_count : train_count + validation_count]:
        assignments[group] = "validation"
    for group in shuffled[train_count + validation_count :]:
        assignments[group] = "test"
    return assignments


def split_label_rows(rows: Sequence[dict], assignments: dict[str, str]) -> dict[str, list[dict]]:
    output = {"train": [], "validation": [], "test": []}
    for row in rows:
        group = scaffold_id(row["protein_id"])
        if group not in assignments:
            raise ValueError(f"missing split assignment for scaffold {group}")
        output[assignments[group]].append(row)
    for values in output.values():
        values.sort(key=lambda row: (row["protein_id"], row.get("trajectory_id", row["protein_id"]), int(row["frame"])))
    return output


def build_event_samples(
    rows: Sequence[dict],
    context_frames: int = 8,
    horizons: Sequence[int] = (1, 4, 8, 16),
) -> list[dict]:
    """Build causal context/target records without crossing trajectories."""
    if context_frames <= 0 or any(int(horizon) <= 0 for horizon in horizons):
        raise ValueError("context_frames and horizons must be positive")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["protein_id"], row.get("trajectory_id", row["protein_id"]))].append(row)
    samples = []
    for (protein_id, trajectory_id), sequence in sorted(grouped.items()):
        sequence.sort(key=lambda row: int(row["frame"]))
        for end in range(context_frames - 1, len(sequence)):
            for horizon in sorted(set(int(value) for value in horizons)):
                target_index = end + horizon
                if target_index >= len(sequence):
                    continue
                source = sequence[end]
                target = sequence[target_index]
                future_states = [item.get("state") for item in sequence[end + 1 : target_index + 1]]
                folded_hit = next((index for index, state in enumerate(future_states) if state == "folded"), None)
                unfolded_hit = next((index for index, state in enumerate(future_states) if state == "unfolded"), None)
                committor_target = None
                if folded_hit is not None and (unfolded_hit is None or folded_hit < unfolded_hit):
                    committor_target = 1
                elif unfolded_hit is not None and (folded_hit is None or unfolded_hit < folded_hit):
                    committor_target = 0
                sample_id = f"{protein_id}|{trajectory_id}|{int(source['frame'])}|{horizon}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "protein_id": protein_id,
                        "trajectory_id": trajectory_id,
                        "scaffold_id": scaffold_id(protein_id),
                        "context_frames": [int(item["frame"]) for item in sequence[end - context_frames + 1 : end + 1]],
                        "context_rows": sequence[end - context_frames + 1 : end + 1],
                        "future_rows": sequence[max(0, target_index - context_frames + 1) : target_index + 1],
                        "source_frame": int(source["frame"]),
                        "target_frame": int(target["frame"]),
                        "horizon": horizon,
                        "source_state": source.get("state"),
                        "target_state": target.get("state"),
                        "source_q_native": source.get("q_native"),
                        "target_q_native": target.get("q_native"),
                        "target_direction": target.get("direction"),
                        "source_committor_proxy": source.get("committor_proxy"),
                        "committor_target": committor_target,
                    }
                )
    return samples


def build_transition_paths(
    rows: Sequence[dict],
    source_state: str = "unfolded",
    target_state: str = "folded",
) -> list[list[str]]:
    """Extract observed contiguous source-to-target state paths."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["protein_id"], row.get("trajectory_id", row["protein_id"]))].append(row)
    paths = []
    for sequence in grouped.values():
        sequence.sort(key=lambda row: int(row["frame"]))
        states = [row.get("state") for row in sequence]
        start = None
        for index, state in enumerate(states):
            if state == source_state and start is None:
                start = index
            elif state == target_state and start is not None:
                paths.append(states[start : index + 1])
                start = None
    return paths


def build_hitting_time_samples(
    rows: Sequence[dict],
    context_frames: int = 8,
    source_state: str = "unfolded",
    target_state: str = "folded",
    competing_state: str | None = None,
) -> list[dict]:
    """Build censored first-passage records from each causal context."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["protein_id"], row.get("trajectory_id", row["protein_id"]))].append(row)
    samples = []
    for (protein_id, trajectory_id), sequence in sorted(grouped.items()):
        sequence.sort(key=lambda row: int(row["frame"]))
        for end in range(context_frames - 1, len(sequence)):
            source = sequence[end]
            if source.get("state") == target_state:
                continue
            future = [item.get("state") for item in sequence[end + 1 :]]
            target_hit = next((index + 1 for index, state in enumerate(future) if state == target_state), None)
            competing_hit = (
                next((index + 1 for index, state in enumerate(future) if state == competing_state), None)
                if competing_state is not None
                else None
            )
            if target_hit is not None and (competing_hit is None or target_hit < competing_hit):
                event_time = target_hit
                event_observed = True
            else:
                event_time = len(future) + 1
                event_observed = False
            samples.append(
                {
                    "sample_id": f"{protein_id}|{trajectory_id}|{int(source['frame'])}",
                    "protein_id": protein_id,
                    "trajectory_id": trajectory_id,
                    "scaffold_id": scaffold_id(protein_id),
                    "source_frame": int(source["frame"]),
                    "event_time": int(event_time),
                    "event_observed": bool(event_observed),
                }
            )
    return samples


def _state_ids(samples: Sequence[dict], state_names: Sequence[str]) -> dict[str, int]:
    mapping = {name: index for index, name in enumerate(state_names)}
    unknown = sorted({sample["target_state"] for sample in samples if sample.get("target_state") not in mapping})
    if unknown:
        raise ValueError(f"unknown states in labels: {unknown}")
    return mapping


def persistence_predictions(samples: Sequence[dict], state_names: Sequence[str]) -> list[dict]:
    """Create a mandatory persistence baseline in the prediction schema."""
    mapping = _state_ids(samples, state_names)
    output = []
    for sample in samples:
        source = sample.get("source_state")
        if source not in mapping:
            continue
        probabilities = [0.0] * len(state_names)
        probabilities[mapping[source]] = 1.0
        committor = {"unfolded": 0.0, "folded": 1.0}.get(source, 0.5)
        output.append(
            {
                "sample_id": sample["sample_id"],
                "state_names": list(state_names),
                "state_probs": probabilities,
                "committor_probability": committor,
            }
        )
    return output


def evaluate_predictions(
    samples: Sequence[dict],
    predictions: Sequence[dict],
    state_names: Sequence[str] = ("unfolded", "transition", "folded"),
    bootstrap_samples: int = 1000,
    reference_paths: Sequence[Sequence[object]] | None = None,
) -> dict:
    """Evaluate FastFolding state, committor, and transition-operator predictions.

    Prediction rows are keyed by ``sample_id`` and contain ``state_probs`` plus
    an optional ``committor_probability``. This makes model adapters small and
    keeps all baselines on exactly the same held-out samples.
    """
    sample_map = {sample["sample_id"]: sample for sample in samples}
    joined = []
    for prediction in predictions:
        sample = sample_map.get(prediction.get("sample_id"))
        if sample is None:
            continue
        row = {**sample, **prediction}
        joined.append(row)
    if not joined:
        raise ValueError("no prediction rows matched benchmark samples")
    mapping = _state_ids(joined, state_names)
    by_horizon = {}
    comm_values = []
    comm_targets = []
    comm_groups = []
    for horizon in sorted({int(row["horizon"]) for row in joined}):
        rows = [row for row in joined if int(row["horizon"]) == horizon and "state_probs" in row]
        if not rows:
            continue
        targets = np.asarray([mapping[row["target_state"]] for row in rows], dtype=np.int64)
        probabilities = np.asarray([row["state_probs"] for row in rows], dtype=np.float64)
        if probabilities.shape[1] != len(state_names):
            raise ValueError("state_probs does not match state_names")
        metrics = multiclass_forecast_metrics(probabilities, targets)
        metrics["samples"] = len(rows)
        by_horizon[str(horizon)] = {"state_forecast": metrics}
        comm_rows = [row for row in rows if row.get("committor_probability") is not None and row.get("committor_target") is not None]
        if comm_rows:
            probabilities_comm = np.asarray([row["committor_probability"] for row in comm_rows], dtype=np.float64)
            targets_comm = np.asarray([row["committor_target"] for row in comm_rows], dtype=np.float64)
            comm_metrics = binary_forecast_metrics(probabilities_comm, targets_comm)
            by_horizon[str(horizon)]["committor_proxy"] = comm_metrics
            comm_values.extend((probabilities_comm - targets_comm) ** 2)
            comm_targets.extend(targets_comm)
            comm_groups.extend(row["scaffold_id"] for row in comm_rows)

    operator_by_horizon = {}
    for horizon in sorted({int(row["horizon"]) for row in joined}):
        rows = [row for row in joined if int(row["horizon"]) == horizon and "state_probs" in row]
        reference_counts = np.zeros((len(state_names), len(state_names)), dtype=np.float64)
        predicted_counts = np.zeros_like(reference_counts)
        for row in rows:
            source = row.get("source_state")
            target = row.get("target_state")
            probabilities = row.get("state_probs")
            if source not in mapping or target not in mapping or probabilities is None:
                continue
            reference_counts[mapping[source], mapping[target]] += 1.0
            predicted_counts[mapping[source]] += np.asarray(probabilities, dtype=np.float64)
        reference = row_normalize_matrix(reference_counts, alpha=1.0)
        predicted = row_normalize_matrix(predicted_counts, alpha=1.0)
        operator_by_horizon[str(horizon)] = {
            "predicted": predicted.tolist(),
            "reference": reference.tolist(),
            "metrics": transition_operator_metrics(predicted, reference),
        }
    output = {
        "n_predictions": len(joined),
        "state_names": list(state_names),
        "by_horizon": by_horizon,
        "transition_operator": operator_by_horizon,
    }
    predicted_paths = [row["path_states"] for row in joined if row.get("path_states")]
    if predicted_paths and reference_paths is not None:
        output["transition_paths"] = path_ensemble_metrics(
            predicted_paths,
            reference_paths,
        )
    if comm_values:
        output["committor_proxy_bootstrap"] = grouped_bootstrap(
            comm_values,
            comm_groups,
            samples=bootstrap_samples,
        )
    return output


def evaluate_survival_predictions(
    samples: Sequence[dict],
    predictions: Sequence[dict],
    horizons: Sequence[int],
) -> dict:
    """Evaluate event-by-horizon probabilities and first-passage calibration."""
    sample_map = {sample["sample_id"]: sample for sample in samples}
    joined = [
        {**sample_map[prediction["sample_id"]], **prediction}
        for prediction in predictions
        if prediction.get("sample_id") in sample_map and prediction.get("event_probabilities") is not None
    ]
    if not joined:
        raise ValueError("no survival prediction rows matched hitting-time samples")
    probabilities = np.asarray([row["event_probabilities"] for row in joined], dtype=np.float64)
    if probabilities.shape != (len(joined), len(horizons)):
        raise ValueError("event_probabilities must have one value per requested horizon")
    event_times = np.asarray([row["event_time"] for row in joined], dtype=np.float64)
    event_observed = np.asarray([row["event_observed"] for row in joined], dtype=bool)
    return {
        "n_predictions": len(joined),
        "horizons": [int(horizon) for horizon in horizons],
        "integrated_brier": integrated_brier_score(probabilities, event_times, event_observed, horizons),
        "survival_curve": survival_curve_error(probabilities, event_times, event_observed, horizons),
    }
