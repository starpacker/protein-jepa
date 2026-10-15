#!/usr/bin/env python3
"""Generate FastFolding state/event predictions from a paired ProteinJEPA model.

The transition head is evaluated causally. A train-scaffold-only ridge readout
maps the predicted future latent state to FastFolding state probabilities. The
readout is an observation map for the benchmark, not an additional model
training run on the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.eval.paper_eval import _device, _load_checkpoint_spec, _model_from_spec
from protein_jepa_frame.fastfolding.benchmark import (
    assign_group_splits,
    build_event_samples,
    load_jsonl,
    split_label_rows,
)
from protein_jepa_frame.fastfolding.downstream import BackboneFrame, _move_batch, _window_to_clip


def _pair_clip(
    sample: dict,
    cache: dict[str, BackboneFrame],
    max_residues: int | None,
    temperature_k: float,
) -> dict:
    context = _window_to_clip(
        {"target": sample, "window": sample["context_rows"]},
        cache,
        max_residues=max_residues,
    )
    future = _window_to_clip(
        {"target": sample, "window": sample["future_rows"]},
        cache,
        max_residues=max_residues,
    )
    context["temperature_k"] = float(temperature_k)
    future["temperature_k"] = float(temperature_k)
    return {
        "context": context,
        "future": future,
        "horizon_frames": torch.tensor(int(sample["horizon"]), dtype=torch.long),
        "horizon_time_ps": torch.tensor(float(sample["target_frame"] - sample["source_frame"])),
    }


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    samples: list[dict],
    batch_size: int,
    max_residues: int | None,
    device: torch.device,
    temperature_k: float,
) -> np.ndarray:
    cache: dict[str, BackboneFrame] = {}
    features = []
    for start in range(0, len(samples), batch_size):
        pairs = [
            _pair_clip(sample, cache, max_residues, temperature_k)
            for sample in samples[start : start + batch_size]
        ]
        batch = _move_batch(collate_paired_windows(pairs), device)
        outputs = model.forward_paired(batch)
        feature = outputs.get("state_prediction")
        if feature is None:
            feature = outputs.get("slow_state_prediction")
        if feature is None:
            raise ValueError("checkpoint does not expose state_prediction or slow_state_prediction")
        features.append(feature.detach().cpu().numpy().astype(np.float64))
    if not features:
        raise ValueError("no FastFolding samples were available")
    return np.concatenate(features, axis=0)


def _fit_ridge_probabilities(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float,
    classes: int,
) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1.0e-8] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train_aug = np.concatenate([train, np.ones((len(train), 1))], axis=1)
    test_aug = np.concatenate([test, np.ones((len(test), 1))], axis=1)
    target = np.eye(classes, dtype=np.float64)[train_y]
    regularizer = ridge * np.eye(train_aug.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.pinv(train_aug.T @ train_aug + regularizer) @ train_aug.T @ target
    scores = test_aug @ weights
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True).clip(min=1.0e-12)


def _fit_binary_probability(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, ridge: float) -> np.ndarray:
    probabilities = _fit_ridge_probabilities(train_x, train_y.astype(np.int64), test_x, ridge, classes=2)
    return probabilities[:, 1]


def generate(args: argparse.Namespace) -> dict:
    rows = load_jsonl(args.labels)
    assignments = assign_group_splits(rows, seed=args.seed)
    splits = split_label_rows(rows, assignments)
    state_names = tuple(args.state_names)
    train_samples = build_event_samples(splits["train"], args.context_frames, args.horizons)
    test_samples = build_event_samples(splits["test"], args.context_frames, args.horizons)
    device = _device(args.device)
    spec = _load_checkpoint_spec(args.checkpoint)
    model, config = _model_from_spec(spec, device)
    if not config["model"].get("paired_forecast", False):
        raise ValueError("FastFolding transition benchmark requires paired_forecast=True")
    model.eval()
    temperature_k = float(args.temperature_k or config["model"].get("lspwm_reference_temperature_k", 348.0))
    train_features = _extract_features(
        model, train_samples, args.batch_size, args.max_residues, device, temperature_k
    )
    test_features = _extract_features(
        model, test_samples, args.batch_size, args.max_residues, device, temperature_k
    )
    mapping = {name: index for index, name in enumerate(state_names)}
    predictions = []
    for horizon in sorted(set(int(sample["horizon"]) for sample in test_samples)):
        train_indices = [index for index, sample in enumerate(train_samples) if int(sample["horizon"]) == horizon and sample.get("target_state") in mapping]
        test_indices = [index for index, sample in enumerate(test_samples) if int(sample["horizon"]) == horizon]
        if not train_indices or not test_indices:
            continue
        train_y = np.asarray([mapping[train_samples[index]["target_state"]] for index in train_indices], dtype=np.int64)
        state_probabilities = _fit_ridge_probabilities(
            train_features[train_indices],
            train_y,
            test_features[test_indices],
            args.ridge,
            len(state_names),
        )
        committor_indices = [index for index in train_indices if train_samples[index].get("committor_target") is not None]
        committor_test_indices = [index for index in test_indices if test_samples[index].get("committor_target") is not None]
        committor_probabilities = {}
        if committor_indices and committor_test_indices:
            committor_train_y = np.asarray([train_samples[index]["committor_target"] for index in committor_indices], dtype=np.int64)
            committor_values = _fit_binary_probability(
                train_features[committor_indices],
                committor_train_y,
                test_features[committor_test_indices],
                args.ridge,
            )
            committor_probabilities = dict(zip(committor_test_indices, committor_values))
        for position, index in enumerate(test_indices):
            row = {
                "sample_id": test_samples[index]["sample_id"],
                "state_names": list(state_names),
                "state_probs": state_probabilities[position].tolist(),
            }
            if index in committor_probabilities:
                row["committor_probability"] = float(committor_probabilities[index])
            predictions.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "checkpoint": args.checkpoint,
        "labels": str(args.labels),
        "output": str(args.output),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "predictions": len(predictions),
        "readout": "state_prediction_or_slow_state_prediction",
        "ridge": args.ridge,
        "temperature_k": temperature_k,
        "split": "scaffold-disjoint train-to-test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="name=/path/checkpoint.pt")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--state-names", nargs="+", default=["unfolded", "transition", "folded"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-residues", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature-k", type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(generate(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
