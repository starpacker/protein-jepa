#!/usr/bin/env python3
"""Build replica-held-out and temperature-held-out TM-JEPA pilot manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def build_splits(
    rows: list[dict],
    *,
    num_proteins: int,
    heldout_temperature_k: float,
    seed: int,
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protein_id"])].append(row)
    eligible = []
    for protein, records in grouped.items():
        temperatures = {float(record["temperature_k"]) for record in records}
        replicas_by_temperature: dict[float, set[str]] = defaultdict(set)
        for record in records:
            replicas_by_temperature[float(record["temperature_k"])].add(str(record["replica_id"]))
        if heldout_temperature_k not in temperatures:
            continue
        if len(temperatures) < 2 or min(len(value) for value in replicas_by_temperature.values()) < 5:
            continue
        eligible.append(protein)
    if len(eligible) < num_proteins:
        raise ValueError(
            f"requested {num_proteins} proteins but only {len(eligible)} have complete temperature/replica support"
        )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = set(eligible[:num_proteins])
    splits: dict[str, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
        "temperature_test": [],
    }
    for row in rows:
        if str(row["protein_id"]) not in selected:
            continue
        temperature = float(row["temperature_k"])
        replica = str(row["replica_id"])
        if temperature == heldout_temperature_k:
            splits["temperature_test"].append(row)
        elif replica in {"0", "1", "2"}:
            splits["train"].append(row)
        elif replica == "3":
            splits["validation"].append(row)
        elif replica == "4":
            splits["test"].append(row)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="tmjepa_pilot20")
    parser.add_argument("--num-proteins", type=int, default=20)
    parser.add_argument("--heldout-temperature-k", type=float, default=450.0)
    parser.add_argument("--seed", type=int, default=6101)
    args = parser.parse_args()
    source = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = build_splits(
        _read_jsonl(source),
        num_proteins=args.num_proteins,
        heldout_temperature_k=args.heldout_temperature_k,
        seed=args.seed,
    )
    summary = {}
    for name, rows in splits.items():
        path = output_dir / f"{args.prefix}.{name}.jsonl"
        _write_jsonl(path, rows)
        summary[name] = {
            "path": str(path),
            "records": len(rows),
            "proteins": len({row["protein_id"] for row in rows}),
            "temperatures": sorted({float(row["temperature_k"]) for row in rows}),
            "replicas": sorted({str(row["replica_id"]) for row in rows}),
        }
    summary_path = output_dir / f"{args.prefix}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
