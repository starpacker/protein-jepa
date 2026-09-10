#!/usr/bin/env python3
"""Build protein-disjoint, replica-disjoint, and temperature-CV V3 splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TRAIN_TEMPERATURES = (320.0, 348.0, 379.0, 413.0)


def _read(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def build_v3_splits(
    rows: list[dict],
    *,
    train_proteins: int,
    validation_proteins: int,
    blind_proteins: int,
    seed: int,
    excluded_blind_proteins: set[str] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    required = {
        (temperature, str(replica))
        for temperature in TRAIN_TEMPERATURES + (450.0,)
        for replica in range(5)
    }
    by_protein: dict[str, list[dict]] = {}
    for row in rows:
        by_protein.setdefault(row["protein_id"], []).append(row)
    eligible = []
    for protein, values in by_protein.items():
        available = {
            (float(row["temperature_k"]), str(row.get("replica_id")))
            for row in values
        }
        if required <= available:
            eligible.append(protein)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    excluded = excluded_blind_proteins or set()
    blind_candidates = [protein for protein in eligible if protein not in excluded]
    if len(blind_candidates) < blind_proteins:
        raise ValueError("not enough proteins remain for a blind V3 split")
    blind = set(blind_candidates[-blind_proteins:])
    development = [protein for protein in eligible if protein not in blind]
    required_development = train_proteins + validation_proteins
    if len(development) < required_development:
        raise ValueError("not enough complete proteins for V3 development splits")
    train = set(development[:train_proteins])
    validation = set(
        development[train_proteins : train_proteins + validation_proteins]
    )

    def select(
        proteins: set[str], temperatures: set[float], replicas: set[str]
    ) -> list[dict]:
        return [
            row
            for row in rows
            if row["protein_id"] in proteins
            and float(row["temperature_k"]) in temperatures
            and str(row.get("replica_id")) in replicas
        ]

    splits = {
        "train": select(train, set(TRAIN_TEMPERATURES), {"0", "1", "2"}),
        "validation": select(validation, set(TRAIN_TEMPERATURES), {"3"}),
        "test": select(blind, set(TRAIN_TEMPERATURES), {"4"}),
        "temperature_test": select(blind, {450.0}, {"4"}),
    }
    for temperature in TRAIN_TEMPERATURES[1:]:
        splits[f"temperature_cv_{int(temperature)}_train"] = select(
            train,
            set(TRAIN_TEMPERATURES) - {temperature},
            {"0", "1", "2"},
        )
        splits[f"temperature_cv_{int(temperature)}_validation"] = select(
            validation, {temperature}, {"3"}
        )
    metadata = {
        "seed": seed,
        "train_proteins": sorted(train),
        "validation_proteins": sorted(validation),
        "blind_proteins": sorted(blind),
        "blind_excludes_previous_pilot": not bool(blind & excluded),
    }
    return splits, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="tmjepa_v3")
    parser.add_argument("--previous-pilot-manifest")
    parser.add_argument("--train-proteins", type=int, default=80)
    parser.add_argument("--validation-proteins", type=int, default=20)
    parser.add_argument("--blind-proteins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6303)
    args = parser.parse_args()
    previous = (
        {row["protein_id"] for row in _read(args.previous_pilot_manifest)}
        if args.previous_pilot_manifest
        else set()
    )
    splits, metadata = build_v3_splits(
        _read(args.manifest),
        train_proteins=args.train_proteins,
        validation_proteins=args.validation_proteins,
        blind_proteins=args.blind_proteins,
        seed=args.seed,
        excluded_blind_proteins=previous,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"metadata": metadata, "splits": {}}
    for name, values in splits.items():
        path = output_dir / f"{args.prefix}.{name}.jsonl"
        _write(path, values)
        summary["splits"][name] = {
            "path": str(path),
            "records": len(values),
            "proteins": len({row["protein_id"] for row in values}),
            "temperatures": sorted({float(row["temperature_k"]) for row in values}),
            "replicas": sorted({str(row.get("replica_id")) for row in values}),
        }
    summary_path = output_dir / f"{args.prefix}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
