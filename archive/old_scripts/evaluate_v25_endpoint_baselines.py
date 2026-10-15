#!/usr/bin/env python3
"""Endpoint-skill baselines for the V25 paired-window K1 table.

Reviewer 3 noted that the deterministic endpoint skill table reports LaSP
without same-protocol baselines. This script evaluates simple baselines on the
same paired-window sample and horizons used by the V25 K1 evaluator.
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


def _one_hot(values: np.ndarray) -> np.ndarray:
    unique = sorted(set(int(x) for x in values.tolist()))
    index = {value: i for i, value in enumerate(unique)}
    out = np.zeros((len(values), len(unique)), dtype=np.float64)
    for row, value in enumerate(values):
        out[row, index[int(value)]] = 1.0
    return out


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    return idx[:cut], idx[cut:]


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _skill(pred: np.ndarray, target: np.ndarray) -> float:
    return 1.0 - _mse(pred, target) / max(_mse(np.zeros_like(target), target), 1.0e-12)


def _horizon_mean_predict(train_h: np.ndarray, train_y: np.ndarray, val_h: np.ndarray) -> np.ndarray:
    global_mean = train_y.mean(axis=0, keepdims=True)
    means = {}
    for h in sorted(set(int(x) for x in train_h.tolist())):
        mask = train_h == h
        if np.any(mask):
            means[h] = train_y[mask].mean(axis=0, keepdims=True)
    return np.concatenate([means.get(int(h), global_mean) for h in val_h], axis=0)


def _collect(model: ProteinJEPA, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    rows = {
        "H_current": [],
        "delta_target": [],
        "delta_lasp": [],
        "horizon": [],
        "protein_id": [],
        "trajectory_id": [],
    }
    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_paired(batch)
        if out.get("delta_target") is None or out.get("delta_prediction") is None:
            continue
        rows["H_current"].append(out["context_endpoint"].detach().cpu().numpy())
        rows["delta_target"].append(out["delta_target"].detach().cpu().numpy())
        rows["delta_lasp"].append(out["delta_prediction"].detach().cpu().numpy())
        rows["horizon"].append(out["horizon_frames"].detach().cpu().numpy().reshape(-1))
        rows["protein_id"].extend(str(x) for x in batch.get("protein_id", []))
        rows["trajectory_id"].extend(str(x) for x in batch.get("trajectory_id", []))
    if not rows["delta_target"]:
        raise RuntimeError("no endpoint predictions collected")
    return {
        key: (np.concatenate(value, axis=0) if key not in {"protein_id", "trajectory_id"} else np.asarray(value))
        for key, value in rows.items()
    }


def _evaluate_split(arrays: dict, seed: int) -> dict:
    target = arrays["delta_target"].astype(np.float64)
    lasp = arrays["delta_lasp"].astype(np.float64)
    current = arrays["H_current"].astype(np.float64)
    horizon = arrays["horizon"].astype(np.int64)
    horizon_oh = _one_hot(horizon)
    train_idx, val_idx = _split_indices(len(target), seed)

    rows = {}
    val_target = target[val_idx]
    rows["Persistence_zero_delta"] = {
        "mse": _mse(np.zeros_like(val_target), val_target),
        "skill": 0.0,
    }
    rows["LaSP_native_delta"] = {
        "mse": _mse(lasp[val_idx], val_target),
        "skill": _skill(lasp[val_idx], val_target),
    }
    horizon_pred = _horizon_mean_predict(horizon[train_idx], target[train_idx], horizon[val_idx])
    rows["Horizon_mean_delta"] = {
        "mse": _mse(horizon_pred, val_target),
        "skill": _skill(horizon_pred, val_target),
    }
    ridge_features = {
        "Ridge_H_current": current,
        "Ridge_H_current_plus_horizon": np.concatenate([current, horizon_oh], axis=1),
    }
    for name, x in ridge_features.items():
        result = _ridge_eval(x[train_idx], target[train_idx], x[val_idx], val_target)
        pred_mse = float(result["mse"])
        base_mse = rows["Persistence_zero_delta"]["mse"]
        rows[name] = {
            "mse": pred_mse,
            "skill": 1.0 - pred_mse / max(base_mse, 1.0e-12),
            "lambda": result["lambda"],
        }

    per_horizon = {}
    for h in sorted(set(int(x) for x in horizon.tolist())):
        mask = horizon[val_idx] == h
        if not np.any(mask):
            continue
        per_horizon[str(h)] = {
            name: {
                "mse": _mse(
                    (
                        np.zeros_like(val_target[mask])
                        if name == "Persistence_zero_delta"
                        else lasp[val_idx][mask]
                        if name == "LaSP_native_delta"
                        else horizon_pred[mask]
                        if name == "Horizon_mean_delta"
                        else np.zeros_like(val_target[mask])
                    ),
                    val_target[mask],
                )
            }
            for name in ["Persistence_zero_delta", "LaSP_native_delta", "Horizon_mean_delta"]
        }
        base = per_horizon[str(h)]["Persistence_zero_delta"]["mse"]
        for name in per_horizon[str(h)]:
            per_horizon[str(h)][name]["skill"] = 1.0 - per_horizon[str(h)][name]["mse"] / max(base, 1.0e-12)

    return {
        "metrics": rows,
        "per_horizon_basic": per_horizon,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
    }


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
    names = reps[0]["metrics"].keys()
    aggregate = {}
    for name in names:
        aggregate[name] = {
            "mse": _bootstrap(np.asarray([rep["metrics"][name]["mse"] for rep in reps]), n_resamples=4000),
            "skill": _bootstrap(np.asarray([rep["metrics"][name]["skill"] for rep in reps]), n_resamples=4000),
        }
    return {
        "experiment_id": "V25_endpoint_same_protocol_baselines",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "V25 paired-window endpoint delta prediction",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "seed": args.seed,
            "repeated_random_splits": args.repeats,
            "n_pairs": int(len(arrays["delta_target"])),
            "n_proteins": int(len(set(arrays["protein_id"].tolist()))),
            "n_trajectories": int(len(set(arrays["trajectory_id"].tolist()))),
            "horizon_counts": {str(int(h)): int(np.sum(arrays["horizon"] == h)) for h in sorted(set(arrays["horizon"].tolist()))},
        },
        "aggregate": aggregate,
        "replicates": reps,
        "claim_boundary": (
            "This is a same-sample endpoint-delta baseline supplement for Table 4. "
            "Ridge baselines are diagnostic supervised readouts fit on paired windows, not autonomous dynamics models."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# V25 Endpoint Same-Protocol Baselines",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"n proteins: {result['protocol']['n_proteins']}",
        f"n trajectories: {result['protocol']['n_trajectories']}",
        "",
        "| Method | MSE mean | MSE 95% CI | Skill mean | Skill 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["aggregate"].items():
        mse = row["mse"]
        skill = row["skill"]
        lines.append(
            f"| {name} | {mse['mean']:.6g} | [{mse['ci_lo']:.6g}, {mse['ci_hi']:.6g}] | "
            f"{skill['mean']:.4f} | [{skill['ci_lo']:.4f}, {skill['ci_hi']:.4f}] |"
        )
    lines.extend(["", result["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V25 endpoint baselines")
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
        "skills": {name: row["skill"]["mean"] for name, row in result["aggregate"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
