#!/usr/bin/env python3
"""Run the model-agnostic passive dynamics benchmarks.

FastFolding prediction JSONL schema:

    {"sample_id": "...", "state_probs": [0.1, 0.7, 0.2],
     "committor_probability": 0.4}

The state order is given by ``--state-names``. Model adapters should write one
prediction row for every context/horizon sample and must not include future
labels in the prediction file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_jepa_frame.fastfolding.benchmark import (
    assign_group_splits,
    build_event_samples,
    build_hitting_time_samples,
    build_transition_paths,
    evaluate_predictions,
    evaluate_survival_predictions,
    load_jsonl,
    persistence_predictions,
    split_label_rows,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_fastfolding(args: argparse.Namespace) -> dict:
    rows = load_jsonl(args.labels)
    assignments = assign_group_splits(
        rows,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    splits = split_label_rows(rows, assignments)
    state_names = tuple(args.state_names)
    split_samples = {
        name: build_event_samples(values, args.context_frames, args.horizons)
        for name, values in splits.items()
    }
    if args.predictions:
        prediction_rows = load_jsonl(args.predictions)
    else:
        prediction_rows = persistence_predictions(split_samples["test"], state_names)
    result = {
        "protocol": {
            "dataset": "FastFolding",
            "labels": str(args.labels),
            "predictions": str(args.predictions) if args.predictions else "persistence",
            "context_frames": args.context_frames,
            "horizons": list(args.horizons),
            "state_names": list(state_names),
            "split": "scaffold-disjoint",
            "seed": args.seed,
        },
        "split_summary": {
            name: {
                "rows": len(values),
                "samples": len(split_samples[name]),
                "scaffolds": sorted({row["scaffold_id"] for row in split_samples[name]}),
                "proteins": sorted({row["protein_id"] for row in split_samples[name]}),
            }
            for name, values in splits.items()
        },
        "test": evaluate_predictions(
            split_samples["test"],
            prediction_rows,
            state_names=state_names,
            bootstrap_samples=args.bootstrap_samples,
            reference_paths=build_transition_paths(splits["test"]),
        ),
    }
    if args.survival_predictions:
        survival_samples = build_hitting_time_samples(
            splits["test"],
            context_frames=args.context_frames,
        )
        result["test"]["survival"] = evaluate_survival_predictions(
            survival_samples,
            load_jsonl(args.survival_predictions),
            args.survival_horizons,
        )
    _write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fastfolding = subparsers.add_parser("fastfolding", help="Evaluate FastFolding state and event forecasts")
    fastfolding.add_argument("--labels", type=Path, required=True)
    fastfolding.add_argument("--predictions", type=Path, default=None)
    fastfolding.add_argument("--survival-predictions", type=Path, default=None)
    fastfolding.add_argument("--output", type=Path, required=True)
    fastfolding.add_argument("--context-frames", type=int, default=8)
    fastfolding.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    fastfolding.add_argument("--survival-horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    fastfolding.add_argument("--state-names", nargs="+", default=["unfolded", "transition", "folded"])
    fastfolding.add_argument("--train-fraction", type=float, default=0.6)
    fastfolding.add_argument("--validation-fraction", type=float, default=0.2)
    fastfolding.add_argument("--bootstrap-samples", type=int, default=1000)
    fastfolding.add_argument("--seed", type=int, default=6303)
    fastfolding.set_defaults(handler=run_fastfolding)
    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
