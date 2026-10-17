#!/usr/bin/env python3
"""Concatenate several JSONL manifests into one mixed manifest.

Each record keeps its own frame_interval_ps / temperature_k / dataset, so the
loader can reconstruct the correct absolute time for every trajectory.

Usage:
  python scripts/merge_manifests.py --out artifacts/mixed_manifest/train.jsonl \
      artifacts/mdcath_manifest_200dom/train.jsonl artifacts/atlas_manifest/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    per_dataset: dict[str, int] = {}
    with out.open("w") as fh:
        for inp in args.inputs:
            for line in Path(inp).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fh.write(json.dumps(rec) + "\n")
                n += 1
                per_dataset[rec.get("dataset", "?")] = per_dataset.get(rec.get("dataset", "?"), 0) + 1
    print(f"wrote {n} records to {out}")
    print(f"per dataset: {per_dataset}")


if __name__ == "__main__":
    main()
