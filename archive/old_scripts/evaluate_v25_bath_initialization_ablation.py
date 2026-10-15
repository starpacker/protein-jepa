#!/usr/bin/env python3
"""Bath-initialization ablation for V25 R1 compression.

Reviewer 3 explicitly requested comparing S0=0, current-only initialization,
and history initialization. This script evaluates those bath states as
features for predicting the future 8D thermodynamic CV under the accepted V25
paired-window protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v25_r1_compression import _bootstrap, _ridge_eval


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def _collect(model: ProteinJEPA, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    rows = {
        "G_current": [],
        "G_future": [],
        "bath_history": [],
        "bath_current_only": [],
        "horizon": [],
        "protein_id": [],
        "trajectory_id": [],
    }
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # Do not wrap the full forward in no_grad: the LSP-WM force path needs autograd.
        out = model.forward_paired(batch)
        hist = out["lspwm_history_features"].detach()
        g_current = out["lspwm_context_state"].detach()
        current_only_hist = g_current.repeat(1, 3)
        with torch.no_grad():
            bath_history = model.lspwm.history_to_bath(hist).detach()
            bath_current_only = model.lspwm.history_to_bath(current_only_hist).detach()
        rows["G_current"].append(g_current.cpu().numpy())
        rows["G_future"].append(out["lspwm_target_state"].detach().cpu().numpy())
        rows["bath_history"].append(bath_history.cpu().numpy())
        rows["bath_current_only"].append(bath_current_only.cpu().numpy())
        rows["horizon"].append(out["horizon_frames"].detach().cpu().numpy().reshape(-1))
        rows["protein_id"].extend(str(x) for x in batch.get("protein_id", []))
        rows["trajectory_id"].extend(str(x) for x in batch.get("trajectory_id", []))
    return {
        key: (np.concatenate(value, axis=0) if key not in {"protein_id", "trajectory_id"} else np.asarray(value))
        for key, value in rows.items()
    }


def _evaluate_split(arrays: dict, seed: int) -> dict:
    Gc = arrays["G_current"]
    Gf = arrays["G_future"]
    bath_history = arrays["bath_history"]
    bath_current = arrays["bath_current_only"]
    rng = np.random.default_rng(seed + 41)
    perm = rng.permutation(len(Gc))
    bath_shuffled = bath_history[perm]
    bath_random = rng.normal(
        loc=bath_history.mean(axis=0, keepdims=True),
        scale=np.maximum(bath_history.std(axis=0, keepdims=True), 1.0e-6),
        size=bath_history.shape,
    )
    bath_zero = np.zeros_like(bath_history)
    features = {
        "G8_current": Gc,
        "G8_plus_zero_bath": np.concatenate([Gc, bath_zero], axis=1),
        "G8_plus_current_only_bath": np.concatenate([Gc, bath_current], axis=1),
        "G8_plus_history_bath": np.concatenate([Gc, bath_history], axis=1),
        "G8_plus_shuffled_history_bath": np.concatenate([Gc, bath_shuffled], axis=1),
        "G8_plus_random_bath": np.concatenate([Gc, bath_random], axis=1),
    }
    train_idx, val_idx = _split_indices(len(Gc), seed)
    results = {
        name: _ridge_eval(x[train_idx], Gf[train_idx], x[val_idx], Gf[val_idx])
        for name, x in features.items()
    }

    def gain(base: str, aug: str) -> float:
        return (results[base]["mse"] - results[aug]["mse"]) / max(results[base]["mse"], 1.0e-12)

    gains = {
        "history_bath_vs_G8_current": gain("G8_current", "G8_plus_history_bath"),
        "history_bath_vs_zero_bath": gain("G8_plus_zero_bath", "G8_plus_history_bath"),
        "history_bath_vs_current_only_bath": gain("G8_plus_current_only_bath", "G8_plus_history_bath"),
        "history_bath_vs_shuffled_bath": gain("G8_plus_shuffled_history_bath", "G8_plus_history_bath"),
        "history_bath_vs_random_bath": gain("G8_plus_random_bath", "G8_plus_history_bath"),
        "current_only_bath_vs_zero_bath": gain("G8_plus_zero_bath", "G8_plus_current_only_bath"),
    }
    return {"features": results, "gains": gains, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}


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
    reps = [_evaluate_split(arrays, args.seed + i) for i in range(args.repeats)]
    feature_names = reps[0]["features"].keys()
    gain_names = reps[0]["gains"].keys()
    aggregate = {"features": {}, "gains": {}}
    for name in feature_names:
        aggregate["features"][name] = {
            "mse": _bootstrap(np.asarray([rep["features"][name]["mse"] for rep in reps]), n_resamples=4000),
            "lambda_values": [rep["features"][name]["lambda"] for rep in reps],
        }
    for name in gain_names:
        aggregate["gains"][name] = _bootstrap(np.asarray([rep["gains"][name] for rep in reps]), n_resamples=4000)
    return {
        "experiment_id": "V25_bath_initialization_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "V25 paired-window G8 future prediction with ridge probes",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "n_pairs": int(len(arrays["G_current"])),
            "n_proteins": int(len(set(arrays["protein_id"].tolist()))),
            "n_trajectories": int(len(set(arrays["trajectory_id"].tolist()))),
            "repeated_random_splits": args.repeats,
            "uses_m1_strict": False,
        },
        "aggregate": aggregate,
        "replicates": reps,
        "claim_boundary": (
            "This tests bath initialization as a predictive feature under paired-window probes. "
            "It is not a retrained C=0 no-memory model, but it directly compares S0=0, current-only, "
            "history, shuffled-history, and random bath initializations."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# V25 Bath Initialization Ablation",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"n proteins: {result['protocol']['n_proteins']}",
        f"n trajectories: {result['protocol']['n_trajectories']}",
        f"random splits: {result['protocol']['repeated_random_splits']}",
        "",
        "## Ridge MSE",
        "",
        "| Feature set | MSE mean | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, row in agg["features"].items():
        stats = row["mse"]
        lines.append(f"| {name} | {stats['mean']:.6g} | [{stats['ci_lo']:.6g}, {stats['ci_hi']:.6g}] |")
    lines.extend([
        "",
        "## Relative Gains",
        "",
        "| Gain | mean | 95% CI |",
        "|---|---:|---:|",
    ])
    for name, stats in agg["gains"].items():
        lines.append(f"| {name} | {100*stats['mean']:.2f}% | [{100*stats['ci_lo']:.2f}%, {100*stats['ci_hi']:.2f}%] |")
    lines.extend(["", result["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V25 bath initialization ablation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
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
        "gains": {name: row["mean"] for name, row in result["aggregate"]["gains"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
