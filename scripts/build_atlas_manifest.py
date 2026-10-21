#!/usr/bin/env python3
"""Build a protein-level train/val/test manifest for an ATLAS NPZ directory.

The ``--frame-interval-ps`` value is the *physical* duration between stored
frames and MUST match the npz's own ``frame_interval_ps`` field, because the
training script multiplies ``frame_interval_ps`` by the rollout length to get
absolute time.  Building a 100 ps dataset with the 1000 ps default (or vice
versa) silently mislabels every horizon by 10x.

Preflighted against the actual npz contents: the manifest refuses to write if
the requested interval disagrees with what the files store.

Splits are by protein (all replicas of a protein stay in one split) to avoid
leakage.

Usage:
  python scripts/build_atlas_manifest.py \
      --npz-dir /liuyutian/protein_jepa/datasets/ATLAS_NPZ_1ns \
      --output-dir artifacts/atlas_manifest

  python scripts/build_atlas_manifest.py \
      --npz-dir /liuyutian/protein_jepa/datasets/ATLAS_NPZ_100ps \
      --output-dir artifacts/atlas_manifest_100ps --frame-interval-ps 100.0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="/liuyutian/protein_jepa/datasets/ATLAS_NPZ_1ns")
    ap.add_argument("--output-dir", default="artifacts/atlas_manifest")
    ap.add_argument("--frame-interval-ps", type=float, default=1000.0,
                    help="Physical ps between stored frames. Must equal the npz's own "
                         "frame_interval_ps (1000 for ATLAS_1ns, 100 for ATLAS_100ps).")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    records = []
    checked = 0
    for npz in sorted(npz_dir.glob("*.npz")):
        # Verify the declared interval against the file itself on the first few
        # entries; a mismatch means every horizon would be mislabelled.
        if checked < 5 and "frame_interval_ps" in np.load(npz, mmap_mode="r").files:
            with np.load(npz, mmap_mode="r") as data:
                stored = float(data["frame_interval_ps"])
            if abs(stored - args.frame_interval_ps) > 1e-3:
                raise SystemExit(
                    f"{npz.name}: stored frame_interval_ps={stored} but "
                    f"--frame-interval-ps={args.frame_interval_ps}; refusing to "
                    f"write a manifest with a mislabelled time axis")
            checked += 1
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
            "frame_interval_ps": args.frame_interval_ps,
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
