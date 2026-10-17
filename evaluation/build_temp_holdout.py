#!/usr/bin/env python3
"""Build held-out-TEMPERATURE manifests for mdCATH.

Writes two jsonl files next to the source manifest:
    val_temp<T>.jsonl      : val proteins, ONLY temperature T   (test split)
    train_notemp<T>.jsonl  : train proteins, all temperatures EXCEPT T  (train split)

This supports the "train at 4 temperatures -> test at the 5th" question.  The
evaluation runner only needs the val_temp<T> file (point --manifest at it, and
set temperatures=[T]); the train file documents the matching training protocol.

Usage:
  python -m evaluation.build_temp_holdout \
      --manifest artifacts/mdcath_manifest_200dom/val.jsonl \
      --train-manifest artifacts/mdcath_manifest_200dom/train.jsonl \
      --temperature 450
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def filter_manifest(src: str, dst: str, *, keep_T: float | None, drop_T: float | None):
    kept = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            T = rec.get("temperature_k")
            if keep_T is not None and (T is None or float(T) != keep_T):
                continue
            if drop_T is not None and T is not None and float(T) == drop_T:
                continue
            fout.write(json.dumps(rec) + "\n")
            kept += 1
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="val manifest")
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.manifest)
    os.makedirs(outdir, exist_ok=True)
    T = args.temperature
    tag = f"{T:g}"

    val_out = os.path.join(outdir, f"val_temp{tag}.jsonl")
    n = filter_manifest(args.manifest, val_out, keep_T=T, drop_T=None)
    print(f"{val_out}: {n} records at T={T}")

    if args.train_manifest:
        tr_out = os.path.join(outdir, f"train_notemp{tag}.jsonl")
        n2 = filter_manifest(args.train_manifest, tr_out, keep_T=None, drop_T=T)
        print(f"{tr_out}: {n2} records at T!={T}")


if __name__ == "__main__":
    main()
