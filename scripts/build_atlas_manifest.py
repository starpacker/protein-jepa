#!/usr/bin/env python3
"""Build a protein-level train/val/test manifest for the 1 ns/frame ATLAS NPZ.

The manifest records the SAME time contract as mdCATH:
  frame_interval_ps = 1000.0  (1 ns per stored frame)
  temperature_k     = 300.0

so the two datasets can be mixed under one absolute-time axis.  Splits are by
protein (all replicas of a protein stay in one split) to avoid leakage.

Usage:
  python scripts/build_atlas_manifest.py \
      --npz-dir /liuyutian/protein_jepa/datasets/ATLAS_NPZ_1ns \
      --output-dir artifacts/atlas_manifest
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/liuyutian/protein_jepa/datasets/ATLAS_NPZ_1ns")
    ap.add_argument("--output-dir", default="artifacts/atlas_manifest")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    records = []
    for npz in sorted(npz_dir.glob("*.npz")):
        # filename = <protein_id>_<replica>.npz ; protein ids may contain "_"
        stem = npz.stem
        protein_id, _, replica = stem.rpartition("_")
        if not protein_id:            # no replica suffix
            protein_id, replica = stem, "R0"
        records.append({
            "npz_path": str(npz.resolve()),
            "protein_id": protein_id,
            "trajectory_id": f"{protein_id}_{replica}",
            "replica_id": replica,
            "domain_id": protein_id,
            "format": "npz",
            "dataset": "atlas",
            "frame_interval_ps": 1000.0,
            "first_frame_time_ps": 0.0,
            "temperature_k": 300.0,
        })
    if not records:
        raise SystemExit(f"no npz files under {npz_dir}")

    proteins = sorted({r["protein_id"] for r in records})
    random.seed(args.seed)
    random.shuffle(proteins)
    n = len(proteins)
    n_train = int(args.train_ratio * n)
    n_val = int(args.val_ratio * n)
    split_of = {}
    for i, p in enumerate(proteins):
        split_of[p] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {s: (out_dir / f"{s}.jsonl").open("w") for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}
    for rec in records:
        s = split_of[rec["protein_id"]]
        writers[s].write(json.dumps(rec) + "\n")
        counts[s] += 1
    for w in writers.values():
        w.close()

    print(f"proteins: {n}  ({n_train} train / {n_val} val / {n - n_train - n_val} test)")
    print(f"records : {counts}")
    print(f"written : {out_dir}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    main()
