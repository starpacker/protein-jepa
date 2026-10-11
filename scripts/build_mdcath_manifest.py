#!/usr/bin/env python3
"""Build manifest JSONL for mdCATH dataset — Phase 0 of fresh training plan.

Each line: JSON with domain_id, h5_path, temperature, replica, num_frames, etc.
Split: 80% train / 10% val / 10% test by CATH domain (no domain leakage).
Output: artifacts/mdcath_manifest/train.jsonl, val.jsonl, test.jsonl
"""
from __future__ import annotations

import json, random, sys
from pathlib import Path

import h5py
import numpy as np

TEMPERATURES = ["320", "348", "379", "413", "450"]
REPLICAS = ["0", "1", "2", "3", "4"]
FRAME_INTERVAL_PS = 1000.0


def discover_domains(data_dir: Path) -> list[tuple[str, Path]]:
    domains = []
    for h5_path in sorted(data_dir.glob("mdcath_dataset_*.h5")):
        domain_id = h5_path.stem.replace("mdcath_dataset_", "")
        domains.append((domain_id, h5_path))
    return domains


def read_frame_counts(h5_path: Path, domain_id: str) -> dict[tuple[str, str], int]:
    counts = {}
    with h5py.File(str(h5_path), "r") as f:
        dom = f[domain_id]
        for temp in TEMPERATURES:
            if temp not in dom:
                continue
            temp_group = dom[temp]
            for rep in REPLICAS:
                if rep not in temp_group:
                    continue
                n_frames = int(temp_group[rep]["coords"].shape[0])
                counts[(temp, rep)] = n_frames
    return counts


def build_manifest(data_dir: Path, output_dir: Path,
                   train_ratio=0.8, val_ratio=0.1, seed=42, max_domains=0):
    random.seed(seed)
    np.random.seed(seed)

    domains = discover_domains(data_dir)
    print(f"Found {len(domains)} domains")
    if max_domains > 0:
        domains = domains[:max_domains]
        print(f"  (limited to {max_domains})")

    random.shuffle(domains)
    n_total = len(domains)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {}
    for i, (domain_id, _) in enumerate(domains):
        if i < n_train:
            splits[domain_id] = "train"
        elif i < n_train + n_val:
            splits[domain_id] = "val"
        else:
            splits[domain_id] = "test"

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": open(output_dir / "train.jsonl", "w"),
        "val": open(output_dir / "val.jsonl", "w"),
        "test": open(output_dir / "test.jsonl", "w"),
    }

    total_traj = 0
    total_frames = 0
    skipped = 0

    for domain_id, h5_path in domains:
        split = splits[domain_id]
        try:
            counts = read_frame_counts(h5_path, domain_id)
        except Exception as e:
            print(f"  WARNING: skipping {domain_id}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if not counts:
            skipped += 1
            continue

        for (temp, rep), n_frames in counts.items():
            if n_frames < 10:
                continue
            record = {
                "h5_path": str(h5_path.resolve()),
                "protein_id": domain_id,
                "trajectory_id": f"{domain_id}_T{temp}_R{rep}",
                "domain_id": domain_id,
                "format": "mdcath-hdf5",
                "dataset": "mdCATH",
                "num_frames": n_frames,
                "frame_interval_ps": FRAME_INTERVAL_PS,
                "temperature_k": int(temp),
                "replica_id": rep,
                "metadata": {"hdf5_group": f"{domain_id}/{temp}/{rep}"},
            }
            writers[split].write(json.dumps(record) + "\n")
            total_traj += 1
            total_frames += n_frames

    for w in writers.values():
        w.close()

    print(f"\nManifest statistics:")
    print(f"  Domains: {len(domains)}, skipped: {skipped}")
    print(f"  Trajectories: {total_traj}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Split: {n_train} train / {n_val} val / {n_total - n_train - n_val} test")
    print(f"  Written to: {output_dir}/{{train,val,test}}.jsonl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/liuyutian/protein_jepa/datasets/mdCATH/repo/data")
    ap.add_argument("--output-dir", default="artifacts/mdcath_manifest")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-domains", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    build_manifest(data_dir, Path(args.output_dir), seed=args.seed,
                   max_domains=args.max_domains)
