#!/usr/bin/env python3
"""V25 R1 compression-closure supplement under paired-window protocol.

Question: why can the 256D JEPA state look nearly closed while the 8D
thermodynamic CV benefits from memory?

This evaluator uses the existing V25 paired-window data path and compares
ridge prediction of future states from:

* H256 current JEPA endpoint.
* H256 plus the 3-frame CV history used by V25's bath initializer.
* G8 current global CV.
* G8 plus raw 3-frame CV history.
* G8 plus inferred bath state S = history_to_bath(history).

The key positive pattern is expected to be:

* History gives little gain over H256.
* History/bath gives a larger gain over G8, showing that compression reopens
  memory and the bath carries useful missing information.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


RIDGE_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0)


def _standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, {"mean": mean, "std": std}


def _ridge_fit(train_x: np.ndarray, train_y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    gram = x.T @ x
    scale = float(np.trace(gram) / max(gram.shape[0], 1))
    reg = np.eye(gram.shape[0]) * lam * max(scale, 1.0e-12)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(gram + reg, x.T @ train_y)
    return w[:-1], w[-1]


def _mse(w: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x @ w + b - y) ** 2))


def _ridge_eval(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> dict:
    train_x, val_x, _ = _standardize(train_x, val_x)
    # Inner validation for lambda on the training prefix/random train set.
    cut = max(4, int(0.8 * len(train_x)))
    best = None
    scores = {}
    for lam in RIDGE_GRID:
        try:
            w, b = _ridge_fit(train_x[:cut], train_y[:cut], lam)
            score = _mse(w, b, train_x[cut:], train_y[cut:]) if len(train_x[cut:]) else 0.0
            scores[str(lam)] = score
            if best is None or score < best[0]:
                best = (score, lam)
        except np.linalg.LinAlgError:
            continue
    lam = float(best[1] if best else 1.0)
    w, b = _ridge_fit(train_x, train_y, lam)
    return {"mse": _mse(w, b, val_x, val_y), "lambda": lam}


def _bootstrap(values: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        mean = float(values.mean()) if len(values) else float("nan")
        return {"mean": mean, "ci_lo": float("nan"), "ci_hi": float("nan"), "n": int(len(values))}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    draws = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lo": float(np.quantile(draws, 0.025)),
        "ci_hi": float(np.quantile(draws, 0.975)),
        "n": int(len(values)),
    }


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def _collect(model: ProteinJEPA, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    rows = defaultdict(list)
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_paired(batch)
            hist = out["lspwm_history_features"]
            bath = model.lspwm.history_to_bath(hist).detach()
        rows["H_current"].append(out["context_endpoint"].detach().cpu().numpy())
        rows["H_future"].append(out["future_endpoint"].detach().cpu().numpy())
        rows["G_current"].append(out["lspwm_context_state"].detach().cpu().numpy())
        rows["G_future"].append(out["lspwm_target_state"].detach().cpu().numpy())
        rows["history"].append(hist.detach().cpu().numpy())
        rows["bath"].append(bath.cpu().numpy())
        rows["horizon"].append(out["horizon_frames"].detach().cpu().numpy().reshape(-1))
        rows["protein_id"].extend(batch.get("protein_id", ["unknown"] * len(out["horizon_frames"])))
    return {
        k: (np.concatenate(v, axis=0) if k != "protein_id" else np.asarray(v))
        for k, v in rows.items()
    }


def _evaluate_feature_sets(arrays: dict, seed: int) -> dict:
    Hc, Hf = arrays["H_current"], arrays["H_future"]
    Gc, Gf = arrays["G_current"], arrays["G_future"]
    hist, bath = arrays["history"], arrays["bath"]
    rng = np.random.default_rng(seed + 17)
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
        "H256_plus_shuffled_CV_history": (np.concatenate([Hc, hist_shuffled], axis=1), Hf),
        "G8_current": (Gc, Gf),
        "G8_plus_raw_history": (np.concatenate([Gc, hist], axis=1), Gf),
        "G8_plus_shuffled_history": (np.concatenate([Gc, hist_shuffled], axis=1), Gf),
        "G8_plus_bath": (np.concatenate([Gc, bath], axis=1), Gf),
        "G8_plus_shuffled_bath": (np.concatenate([Gc, bath_shuffled], axis=1), Gf),
        "G8_plus_random_bath": (np.concatenate([Gc, bath_random], axis=1), Gf),
        "G8_plus_history_and_bath": (np.concatenate([Gc, hist, bath], axis=1), Gf),
        "G8_plus_shuffled_history_and_bath": (np.concatenate([Gc, hist_shuffled, bath_shuffled], axis=1), Gf),
    }
    train_idx, val_idx = _split_indices(len(Hc), seed)
    results = {}
    for name, (x, y) in features.items():
        results[name] = _ridge_eval(x[train_idx], y[train_idx], x[val_idx], y[val_idx])

    # Relative gains: positive means the augmented state predicts better.
    def gain(base: str, aug: str) -> float:
        return (results[base]["mse"] - results[aug]["mse"]) / max(results[base]["mse"], 1.0e-12)

    gains = {
        "H_history_gain_rel": gain("H256_current", "H256_plus_CV_history"),
        "H_history_alignment_gain_rel": gain("H256_plus_shuffled_CV_history", "H256_plus_CV_history"),
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
    return {"features": results, "gains": gains, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}


def _replicate(arrays: dict, seeds: list[int]) -> dict:
    reps = []
    for seed in seeds:
        reps.append(_evaluate_feature_sets(arrays, seed))
    feature_names = reps[0]["features"].keys()
    gain_names = reps[0]["gains"].keys()
    aggregate = {"features": {}, "gains": {}}
    for name in feature_names:
        aggregate["features"][name] = {
            "mse": _bootstrap(np.asarray([r["features"][name]["mse"] for r in reps]), n_resamples=4000),
            "lambda_values": [r["features"][name]["lambda"] for r in reps],
        }
    for name in gain_names:
        aggregate["gains"][name] = _bootstrap(np.asarray([r["gains"][name] for r in reps]), n_resamples=4000)
    aggregate["replicates"] = reps
    return aggregate


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    dataset = build_dataset(config.data)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    arrays = _collect(model, loader, device, args.max_batches)
    seeds = [args.seed + i for i in range(args.repeats)]
    aggregate = _replicate(arrays, seeds)
    return {
        "experiment_id": "V25_R1_compression_paired_window",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "existing V25 paired-window protocol",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "n_pairs": int(len(arrays["H_current"])),
            "repeated_random_splits": args.repeats,
            "uses_m1_strict": False,
        },
        "aggregate": aggregate,
        "claim_boundary": (
            "This is a paired-window compression diagnostic, not a temporal-prefix proof. "
            "Positive G8 history/bath gain together with smaller H256 history gain supports "
            "the narrative that compression reintroduces memory."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# V25 R1 Compression Closure Supplement",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
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
    p = argparse.ArgumentParser(description="Evaluate V25 R1 compression closure supplement")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=6303)
    p.add_argument("--repeats", type=int, default=32)
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
        "gains": {k: v["mean"] for k, v in result["aggregate"]["gains"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
