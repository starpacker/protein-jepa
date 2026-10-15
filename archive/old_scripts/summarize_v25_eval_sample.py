#!/usr/bin/env python3
"""Summarize sample units for the existing V25 paired-window evaluator."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset


def run(args):
    cfg = load_config(args.config)
    ds = build_dataset(cfg.data)
    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
        generator=generator,
    )
    proteins = []
    trajectories = []
    horizons = []
    protein_to_traj = defaultdict(set)
    traj_to_pairs = Counter()
    protein_to_pairs = Counter()
    for i, batch in enumerate(loader):
        if args.max_batches and i >= args.max_batches:
            break
        pids = list(batch.get("protein_id", []))
        tids = list(batch.get("trajectory_id", []))
        hs = batch.get("horizon_frames")
        hs = hs.cpu().numpy().tolist() if hasattr(hs, "cpu") else []
        for pid, tid, h in zip(pids, tids, hs):
            proteins.append(pid)
            trajectories.append(tid)
            horizons.append(int(h))
            protein_to_traj[pid].add(tid)
            traj_to_pairs[tid] += 1
            protein_to_pairs[pid] += 1
    out = {
        "config": str(args.config),
        "manifest_path": cfg.data.manifest_path,
        "max_batches": args.max_batches,
        "n_pairs": len(proteins),
        "n_proteins": len(set(proteins)),
        "n_trajectories": len(set(trajectories)),
        "horizon_counts": dict(sorted(Counter(horizons).items())),
        "pairs_per_protein": {
            "min": min(protein_to_pairs.values()) if protein_to_pairs else 0,
            "mean": sum(protein_to_pairs.values()) / max(len(protein_to_pairs), 1),
            "max": max(protein_to_pairs.values()) if protein_to_pairs else 0,
        },
        "pairs_per_trajectory": {
            "min": min(traj_to_pairs.values()) if traj_to_pairs else 0,
            "mean": sum(traj_to_pairs.values()) / max(len(traj_to_pairs), 1),
            "max": max(traj_to_pairs.values()) if traj_to_pairs else 0,
        },
        "top_proteins_by_pairs": protein_to_pairs.most_common(20),
        "top_trajectories_by_pairs": traj_to_pairs.most_common(20),
    }
    return out


def write_summary(result, path: Path):
    lines = [
        "# V25 Paired-Window Evaluation Sample Summary",
        "",
        f"Manifest: `{result['manifest_path']}`",
        f"max batches: {result['max_batches']}",
        "",
        "| Unit | Count |",
        "|---|---:|",
        f"| pairs | {result['n_pairs']} |",
        f"| proteins | {result['n_proteins']} |",
        f"| trajectories | {result['n_trajectories']} |",
        "",
        "## Horizon Counts",
        "",
        "| horizon frames | count |",
        "|---:|---:|",
    ]
    for h, c in result["horizon_counts"].items():
        lines.append(f"| {h} | {c} |")
    lines.extend([
        "",
        "## Pair Distribution",
        "",
        f"Pairs/protein: min {result['pairs_per_protein']['min']}, "
        f"mean {result['pairs_per_protein']['mean']:.2f}, "
        f"max {result['pairs_per_protein']['max']}.",
        "",
        f"Pairs/trajectory: min {result['pairs_per_trajectory']['min']}, "
        f"mean {result['pairs_per_trajectory']['mean']:.2f}, "
        f"max {result['pairs_per_trajectory']['max']}.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=9101)
    args = p.parse_args()
    result = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        write_summary(result, Path(args.summary))
    print(json.dumps({
        "output": str(out),
        "summary": args.summary,
        "n_pairs": result["n_pairs"],
        "n_proteins": result["n_proteins"],
        "n_trajectories": result["n_trajectories"],
        "horizon_counts": result["horizon_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
