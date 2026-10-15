#!/usr/bin/env python3
"""Protein-grouped R1 compression probe for V25.

This is a reviewer-facing robustness check for the compression-memory story.
It reuses the V25 paired-window collection path, but ridge probes are trained
and validated on disjoint protein sets. This makes the uncertainty unit much
closer to the independent biological target than pair-level bootstrap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v25_r1_compression import _bootstrap, _collect, _ridge_eval


def _split_proteins(protein_ids: np.ndarray, seed: int, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    proteins = np.asarray(sorted(set(str(x) for x in protein_ids)))
    rng.shuffle(proteins)
    n_val = max(1, int(round(len(proteins) * val_fraction)))
    val_proteins = set(proteins[:n_val])
    val = np.asarray([str(pid) in val_proteins for pid in protein_ids], dtype=bool)
    train = ~val
    return train, val


def _evaluate_split(arrays: dict, seed: int, val_fraction: float) -> dict:
    Hc, Hf = arrays["H_current"], arrays["H_future"]
    Gc, Gf = arrays["G_current"], arrays["G_future"]
    hist, bath = arrays["history"], arrays["bath"]
    protein_ids = arrays["protein_id"]
    rng = np.random.default_rng(seed + 1309)
    perm = rng.permutation(len(Gc))
    hist_shuffled = hist[perm]
    bath_shuffled = bath[perm]
    bath_random = rng.normal(
        loc=bath.mean(axis=0, keepdims=True),
        scale=np.maximum(bath.std(axis=0, keepdims=True), 1.0e-6),
        size=bath.shape,
    )
    features = {
        "H256_current": (Hc, Hf),
        "H256_plus_CV_history": (np.concatenate([Hc, hist], axis=1), Hf),
        "G8_current": (Gc, Gf),
        "G8_plus_raw_history": (np.concatenate([Gc, hist], axis=1), Gf),
        "G8_plus_shuffled_history": (np.concatenate([Gc, hist_shuffled], axis=1), Gf),
        "G8_plus_bath": (np.concatenate([Gc, bath], axis=1), Gf),
        "G8_plus_shuffled_bath": (np.concatenate([Gc, bath_shuffled], axis=1), Gf),
        "G8_plus_random_bath": (np.concatenate([Gc, bath_random], axis=1), Gf),
        "G8_plus_history_and_bath": (np.concatenate([Gc, hist, bath], axis=1), Gf),
        "G8_plus_shuffled_history_and_bath": (np.concatenate([Gc, hist_shuffled, bath_shuffled], axis=1), Gf),
    }
    train_mask, val_mask = _split_proteins(protein_ids, seed, val_fraction)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    results = {}
    for name, (x, y) in features.items():
        results[name] = _ridge_eval(x[train_idx], y[train_idx], x[val_idx], y[val_idx])

    def gain(base: str, aug: str) -> float:
        return (results[base]["mse"] - results[aug]["mse"]) / max(results[base]["mse"], 1.0e-12)

    gains = {
        "H_history_gain_rel": gain("H256_current", "H256_plus_CV_history"),
        "G_history_gain_rel": gain("G8_current", "G8_plus_raw_history"),
        "G_history_alignment_gain_rel": gain("G8_plus_shuffled_history", "G8_plus_raw_history"),
        "G_bath_gain_rel": gain("G8_current", "G8_plus_bath"),
        "G_bath_alignment_gain_rel": gain("G8_plus_shuffled_bath", "G8_plus_bath"),
        "G_bath_vs_random_gain_rel": gain("G8_plus_random_bath", "G8_plus_bath"),
        "G_history_bath_gain_rel": gain("G8_current", "G8_plus_history_and_bath"),
        "G_history_bath_alignment_gain_rel": gain(
            "G8_plus_shuffled_history_and_bath", "G8_plus_history_and_bath"
        ),
    }
    return {
        "features": results,
        "gains": gains,
        "n_train_pairs": int(len(train_idx)),
        "n_val_pairs": int(len(val_idx)),
        "n_train_proteins": int(len(set(protein_ids[train_idx]))),
        "n_val_proteins": int(len(set(protein_ids[val_idx]))),
    }


def _aggregate(reps: list[dict]) -> dict:
    feature_names = reps[0]["features"].keys()
    gain_names = reps[0]["gains"].keys()
    out = {"features": {}, "gains": {}}
    for name in feature_names:
        out["features"][name] = {
            "mse": _bootstrap(np.asarray([r["features"][name]["mse"] for r in reps]), n_resamples=4000),
            "lambda_values": [r["features"][name]["lambda"] for r in reps],
        }
    for name in gain_names:
        out["gains"][name] = _bootstrap(np.asarray([r["gains"][name] for r in reps]), n_resamples=4000)
    out["split_sizes"] = {
        "train_pairs_mean": float(np.mean([r["n_train_pairs"] for r in reps])),
        "val_pairs_mean": float(np.mean([r["n_val_pairs"] for r in reps])),
        "train_proteins_mean": float(np.mean([r["n_train_proteins"] for r in reps])),
        "val_proteins_mean": float(np.mean([r["n_val_proteins"] for r in reps])),
    }
    return out


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    dataset = build_dataset(config.data)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
        generator=generator,
    )
    arrays = _collect(model, loader, device, args.max_batches)
    reps = [_evaluate_split(arrays, args.seed + i, args.val_fraction) for i in range(args.repeats)]
    return {
        "experiment_id": "V25_R1_protein_grouped_compression_probe",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "existing V25 paired-window collection; protein-disjoint ridge splits",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "n_pairs": int(len(arrays["H_current"])),
            "n_proteins": int(len(set(arrays["protein_id"]))),
            "repeated_protein_splits": args.repeats,
            "val_fraction": args.val_fraction,
            "uses_m1_strict": False,
        },
        "aggregate": _aggregate(reps),
        "replicates": reps,
        "claim_boundary": (
            "This evaluates R1-style compression closure with protein-disjoint ridge train/validation splits. "
            "It tests whether the G8 history/bath gains persist when validation windows come from held-out proteins."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# V25 R1 Protein-Grouped Compression Probe",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"n proteins: {result['protocol']['n_proteins']}",
        f"protein-disjoint splits: {result['protocol']['repeated_protein_splits']}",
        f"mean train/val proteins: {agg['split_sizes']['train_proteins_mean']:.1f} / {agg['split_sizes']['val_proteins_mean']:.1f}",
        f"mean train/val pairs: {agg['split_sizes']['train_pairs_mean']:.1f} / {agg['split_sizes']['val_pairs_mean']:.1f}",
        "",
        "## Protein-Grouped Ridge MSE",
        "",
        "| Feature set | MSE mean | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, row in agg["features"].items():
        stats = row["mse"]
        lines.append(f"| {name} | {stats['mean']:.6g} | [{stats['ci_lo']:.6g}, {stats['ci_hi']:.6g}] |")
    lines.extend([
        "",
        "## Protein-Grouped Relative Gains",
        "",
        "| Gain | mean | 95% CI |",
        "|---|---:|---:|",
    ])
    for name, stats in agg["gains"].items():
        lines.append(f"| {name} | {100*stats['mean']:.2f}% | [{100*stats['ci_lo']:.2f}%, {100*stats['ci_hi']:.2f}%] |")
    lines.extend(["", result["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate V25 R1 compression with protein-grouped splits")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--repeats", type=int, default=24)
    p.add_argument("--val-fraction", type=float, default=0.2)
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
        "n_pairs": result["protocol"]["n_pairs"],
        "n_proteins": result["protocol"]["n_proteins"],
        "gains": {k: v["mean"] for k, v in result["aggregate"]["gains"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
