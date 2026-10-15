#!/usr/bin/env python3
"""Nonlinear R1 compression probe for V25.

Reviewer 3 asked whether the history-closure story survives a nonlinear probe.
This script mirrors the R1 compression supplement, but replaces ridge with a
small fixed-budget MLP probe and keeps shuffled/random controls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v25_r1_compression import _bootstrap, _collect


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def _standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return (train - mean) / std, (val - mean) / std, mean, std


class ProbeMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp_eval(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    seed: int,
    hidden: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
) -> dict:
    torch.manual_seed(seed)
    train_x, val_x, _, _ = _standardize(train_x.astype(np.float32), val_x.astype(np.float32))
    train_y_s, val_y_s, y_mean, y_std = _standardize(train_y.astype(np.float32), val_y.astype(np.float32))
    rng = np.random.default_rng(seed + 1009)
    order = rng.permutation(len(train_x))
    inner_cut = max(8, int(0.85 * len(order)))
    fit_idx, early_idx = order[:inner_cut], order[inner_cut:]

    x_fit = torch.from_numpy(train_x[fit_idx]).to(device)
    y_fit = torch.from_numpy(train_y_s[fit_idx]).to(device)
    x_early = torch.from_numpy(train_x[early_idx]).to(device)
    y_early = torch.from_numpy(train_y_s[early_idx]).to(device)
    x_val = torch.from_numpy(val_x).to(device)

    model = ProbeMLP(train_x.shape[1], train_y.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(x_fit, y_fit)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 2027)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)

    best_state = None
    best_loss = float("inf")
    stale = 0
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            early_loss = torch.mean((model(x_early) - y_early) ** 2).item() if len(early_idx) else 0.0
        if early_loss < best_loss - 1.0e-5:
            best_loss = early_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    with torch.no_grad():
        pred_s = model(x_val).detach().cpu().numpy()
    pred = pred_s * y_std + y_mean
    mse = float(np.mean((pred - val_y) ** 2))
    return {"mse": mse, "epochs_run": int(_epoch + 1), "early_loss": float(best_loss)}


def _feature_sets(arrays: dict, seed: int) -> dict:
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
    return {
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


def _evaluate_split(arrays: dict, args: argparse.Namespace, device: torch.device, seed: int) -> dict:
    features = _feature_sets(arrays, seed)
    train_idx, val_idx = _split_indices(len(arrays["H_current"]), seed)
    results = {}
    for name, (x, y) in features.items():
        hidden = args.hidden_h if name.startswith("H256") else args.hidden_g
        results[name] = _mlp_eval(
            x[train_idx],
            y[train_idx],
            x[val_idx],
            y[val_idx],
            device=device,
            seed=seed + abs(hash(name)) % 100000,
            hidden=hidden,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
        )

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
    reps = [_evaluate_split(arrays, args, device, args.seed + i) for i in range(args.repeats)]
    feature_names = reps[0]["features"].keys()
    gain_names = reps[0]["gains"].keys()
    aggregate = {"features": {}, "gains": {}}
    for name in feature_names:
        aggregate["features"][name] = {
            "mse": _bootstrap(np.asarray([rep["features"][name]["mse"] for rep in reps]), n_resamples=4000),
            "epochs_run_mean": float(np.mean([rep["features"][name]["epochs_run"] for rep in reps])),
        }
    for name in gain_names:
        aggregate["gains"][name] = _bootstrap(np.asarray([rep["gains"][name] for rep in reps]), n_resamples=4000)
    return {
        "experiment_id": "V25_R1_nonlinear_compression_probe",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "existing V25 paired-window protocol",
            "probe": "fixed small MLP with early stopping",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "n_pairs": int(len(arrays["H_current"])),
            "repeated_random_splits": args.repeats,
            "hidden_h": args.hidden_h,
            "hidden_g": args.hidden_g,
            "uses_m1_strict": False,
        },
        "aggregate": aggregate,
        "replicates": reps,
        "claim_boundary": (
            "This is a nonlinear paired-window probe, not an autonomous rollout. "
            "Positive aligned-history/bath gains show that the compression-memory "
            "effect is not an artifact of a purely linear ridge probe."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    agg = result["aggregate"]
    lines = [
        "# V25 R1 Nonlinear Compression Probe",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"random splits: {result['protocol']['repeated_random_splits']}",
        f"probe: {result['protocol']['probe']}",
        "",
        "## MLP MSE",
        "",
        "| Feature set | MSE mean | 95% CI | mean epochs |",
        "|---|---:|---:|---:|",
    ]
    for name, row in agg["features"].items():
        stats = row["mse"]
        lines.append(
            f"| {name} | {stats['mean']:.6g} | [{stats['ci_lo']:.6g}, {stats['ci_hi']:.6g}] | "
            f"{row['epochs_run_mean']:.1f} |"
        )
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
    parser = argparse.ArgumentParser(description="Evaluate V25 R1 nonlinear compression probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-h", type=int, default=128)
    parser.add_argument("--hidden-g", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
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
