#!/usr/bin/env python3
"""Residual autocorrelation diagnostic for V25 paired-window dynamics.

This is a reviewer-facing replacement/companion for the opaque "DW" score:
collect paired-window global-CV displacements in trajectory order, calibrate
force and bath drifts with simple scalar least squares, then report residual
autocorrelation at window lags.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


def _scalar_fit(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.sum(x * x))
    if denom < 1.0e-30:
        return 0.0
    return float(np.sum(x * y) / denom)


def _two_scalar_fit(f: np.ndarray, b: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    gram = np.array(
        [[np.sum(f * f), np.sum(f * b)], [np.sum(f * b), np.sum(b * b)]],
        dtype=np.float64,
    )
    rhs = np.array([np.sum(f * y), np.sum(b * y)], dtype=np.float64)
    try:
        sol = np.linalg.solve(gram + np.eye(2) * 1.0e-12, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(gram + np.eye(2) * 1.0e-8, rhs, rcond=None)[0]
    return float(sol[0]), float(sol[1])


def _mse(x: np.ndarray) -> float:
    return float(np.mean(x * x))


def _acf_for_group(residual: np.ndarray, max_lag: int) -> dict[int, tuple[float, int]]:
    if len(residual) < 3:
        return {}
    centered = residual - residual.mean(axis=0, keepdims=True)
    var = float(np.mean(np.sum(centered * centered, axis=1)))
    if var < 1.0e-30:
        return {}
    out = {}
    for lag in range(1, max_lag + 1):
        if len(centered) <= lag:
            continue
        dots = np.sum(centered[:-lag] * centered[lag:], axis=1)
        out[lag] = (float(np.mean(dots) / var), int(len(dots)))
    return out


def _weighted_acf(rows: list[dict], residual_key: str, max_lag: int) -> dict[str, dict]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["trajectory_id"], row["horizon_frames"])].append(row)

    lag_values = defaultdict(list)
    lag_weights = defaultdict(list)
    used_groups = 0
    for group_rows in by_group.values():
        group_rows = sorted(group_rows, key=lambda r: r["start_frame"])
        residual = np.stack([r[residual_key] for r in group_rows], axis=0)
        group_acf = _acf_for_group(residual, max_lag)
        if group_acf:
            used_groups += 1
        for lag, (value, weight) in group_acf.items():
            lag_values[lag].append(value)
            lag_weights[lag].append(weight)

    out = {}
    for lag in range(1, max_lag + 1):
        vals = np.asarray(lag_values.get(lag, []), dtype=float)
        weights = np.asarray(lag_weights.get(lag, []), dtype=float)
        if len(vals) == 0:
            continue
        out[str(lag)] = {
            "mean": float(np.mean(vals)),
            "weighted_mean": float(np.average(vals, weights=weights)),
            "mean_abs": float(np.mean(np.abs(vals))),
            "weighted_mean_abs": float(np.average(np.abs(vals), weights=weights)),
            "n_groups": int(len(vals)),
            "n_pairs": int(np.sum(weights)),
        }
    out["_used_groups"] = {"n": int(used_groups)}
    return out


def _integrated_abs(acf: dict[str, dict], max_lag: int) -> float:
    vals = []
    for lag in range(1, max_lag + 1):
        row = acf.get(str(lag))
        if row is not None:
            vals.append(float(row["weighted_mean_abs"]))
    return float(np.sum(vals)) if vals else float("nan")


def _kernel_relative(model: ProteinJEPA, device: torch.device, times_ps: list[float]) -> dict[str, float]:
    taus, C = model.lspwm.multi_bath.parameters_at(dtype=torch.float64, device=device)
    taus_np = taus.detach().cpu().numpy()
    C_np = C.detach().cpu().numpy()
    k0 = C_np @ C_np.T
    k0_norm = max(float(np.linalg.norm(k0)), 1.0e-30)
    out = {}
    for t in times_ps:
        decay = np.exp(-float(t) / np.maximum(taus_np, 1.0e-30))
        kt = C_np @ np.diag(decay) @ C_np.T
        out[str(float(t))] = float(np.linalg.norm(kt) / k0_norm)
    return out


def _select_trajectory_blocks(dataset, max_blocks: int, starts_per_block: int) -> list[int] | None:
    if max_blocks <= 0:
        return None
    n_entries = len(dataset.entries)
    if n_entries == 0:
        return None
    entry_indices = np.linspace(0, n_entries - 1, num=min(max_blocks, n_entries), dtype=int)
    selected_entries = set(int(x) for x in entry_indices.tolist())
    selected: list[int] = []
    previous_cumulative = 0
    for entry_idx, entry in enumerate(dataset.entries):
        if entry_idx not in selected_entries:
            previous_cumulative = dataset.cumulative[entry_idx]
            continue
        count = int(entry.count)
        if count <= 0:
            previous_cumulative = dataset.cumulative[entry_idx]
            continue
        block = min(int(starts_per_block), count)
        local0 = max(0, (count - block) // 2)
        selected.extend(previous_cumulative + local0 + j for j in range(block))
        previous_cumulative = dataset.cumulative[entry_idx]
    return selected


def _collect_rows(model: ProteinJEPA, config, device: torch.device, args: argparse.Namespace) -> list[dict]:
    dataset = build_dataset(config.data)
    selected_indices = _select_trajectory_blocks(
        dataset, args.max_trajectory_blocks, args.starts_per_trajectory
    )
    loader_dataset = Subset(dataset, selected_indices) if selected_indices is not None else dataset
    loader = DataLoader(
        loader_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=False,
    )
    horizons = tuple(sorted({int(h) for h in config.data.paired_horizons_frames}))
    rows: list[dict] = []
    model.eval()
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        batch_start = batch_index * config.data.batch_size
        group_index = batch.get("context_group_index")
        if group_index is None:
            group_index = torch.arange(len(batch["horizon_frames"]), dtype=torch.long)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_paired(batch)
        Gc = out["lspwm_m1_global_current"].detach().cpu().numpy()
        Gf = out["lspwm_m1_global_future"].detach().cpu().numpy()
        force = out["lspwm_m1_dissipative_drift"].detach().cpu().numpy()
        bath = out["lspwm_m1_bath_drift"].detach().cpu().numpy()
        horizon_frames = out["horizon_frames"].detach().cpu().numpy().reshape(-1).astype(int)
        group_np = group_index.detach().cpu().numpy().reshape(-1).astype(int)
        traj_ids = batch["trajectory_id"]
        protein_ids = batch["protein_id"]
        for i in range(len(horizon_frames)):
            loader_index = int(batch_start + group_np[i])
            dataset_index = int(selected_indices[loader_index]) if selected_indices is not None else loader_index
            record, context_frames, _future_frames, _max_horizon = dataset._locate(dataset_index)
            h = int(horizon_frames[i])
            if h not in horizons:
                continue
            rows.append(
                {
                    "trajectory_id": traj_ids[i],
                    "protein_id": protein_ids[i],
                    "record_trajectory_id": record.trajectory_id,
                    "start_frame": int(context_frames[0]),
                    "horizon_frames": h,
                    "horizon_time_ps": float(h * (record.frame_interval_ps or 1.0)),
                    "delta": Gf[i] - Gc[i],
                    "force": force[i],
                    "bath": bath[i],
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    rows = _collect_rows(model, config, device, args)
    if not rows:
        raise RuntimeError("no paired rows collected")

    y = np.stack([r["delta"] for r in rows], axis=0)
    f = np.stack([r["force"] for r in rows], axis=0)
    b = np.stack([r["bath"] for r in rows], axis=0)
    alpha_f = _scalar_fit(f, y)
    alpha_b = _scalar_fit(b, y)
    alpha_full_f, alpha_full_b = _two_scalar_fit(f, b, y)

    for row in rows:
        yy = row["delta"]
        ff = row["force"]
        bb = row["bath"]
        row["resid_none"] = yy
        row["resid_force"] = yy - alpha_f * ff
        row["resid_memory"] = yy - alpha_b * bb
        row["resid_full"] = yy - alpha_full_f * ff - alpha_full_b * bb

    residual_mse = {
        "none": _mse(np.stack([r["resid_none"] for r in rows])),
        "force": _mse(np.stack([r["resid_force"] for r in rows])),
        "memory": _mse(np.stack([r["resid_memory"] for r in rows])),
        "full": _mse(np.stack([r["resid_full"] for r in rows])),
    }
    acf = {
        name: _weighted_acf(rows, f"resid_{name}", args.max_lag)
        for name in ("none", "force", "memory", "full")
    }
    integrated = {
        name: _integrated_abs(acf[name], args.max_lag)
        for name in ("none", "force", "memory", "full")
    }
    reduction = {
        "full_vs_none": (integrated["none"] - integrated["full"]) / max(integrated["none"], 1.0e-12),
        "full_vs_force": (integrated["force"] - integrated["full"]) / max(integrated["force"], 1.0e-12),
        "memory_vs_none": (integrated["none"] - integrated["memory"]) / max(integrated["none"], 1.0e-12),
    }
    window_stride_ps = float(config.data.window_stride) * float(
        np.median([r["horizon_time_ps"] / max(r["horizon_frames"], 1) for r in rows])
    )
    lag_times = [window_stride_ps * lag for lag in range(1, args.max_lag + 1)]
    return {
        "experiment_id": "V25_residual_acf_diagnostic",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "existing V25 paired-window protocol, sequential DataLoader order",
            "max_batches": args.max_batches,
            "max_trajectory_blocks": args.max_trajectory_blocks,
            "starts_per_trajectory": args.starts_per_trajectory,
            "n_pairs": int(len(rows)),
            "n_proteins": int(len({r["protein_id"] for r in rows})),
            "n_trajectories": int(len({r["trajectory_id"] for r in rows})),
            "max_lag_windows": args.max_lag,
            "window_stride_ps": window_stride_ps,
            "lag_times_ps": lag_times,
            "uses_m1_strict": False,
        },
        "calibration": {
            "force_scalar": alpha_f,
            "memory_scalar": alpha_b,
            "full_force_scalar": alpha_full_f,
            "full_memory_scalar": alpha_full_b,
            "note": "single global scalar coefficients fit on the collected paired-window sample",
        },
        "residual_mse": residual_mse,
        "acf_by_lag": acf,
        "integrated_abs_acf": integrated,
        "integrated_abs_acf_reduction": reduction,
        "kernel_relative_at_lag_times": _kernel_relative(model, device, lag_times),
        "claim_boundary": (
            "This is a paired-window residual autocorrelation diagnostic. It makes the "
            "temporal-residual claim auditable, but it is not an independently retrained "
            "no-memory baseline and should not be described as proof of unique physical "
            "identifiability."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# V25 Residual ACF Diagnostic",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"n proteins: {result['protocol']['n_proteins']}",
        f"n trajectories: {result['protocol']['n_trajectories']}",
        f"window stride: {result['protocol']['window_stride_ps']:.1f} ps",
        "",
        "## Scalar Calibration",
        "",
        "| coefficient | value |",
        "|---|---:|",
    ]
    for key, value in result["calibration"].items():
        if key == "note":
            continue
        lines.append(f"| {key} | {value:.6g} |")
    lines.extend([
        "",
        "## Residual MSE",
        "",
        "| Mode | MSE |",
        "|---|---:|",
    ])
    for name in ("none", "force", "memory", "full"):
        lines.append(f"| {name} | {result['residual_mse'][name]:.6g} |")
    lines.extend([
        "",
        "## Weighted Residual ACF",
        "",
        "| lag windows | lag ps | none | force | memory | full | kernel rel |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    lag_times = result["protocol"]["lag_times_ps"]
    for lag, lag_ps in enumerate(lag_times, start=1):
        vals = []
        for name in ("none", "force", "memory", "full"):
            vals.append(result["acf_by_lag"][name].get(str(lag), {}).get("weighted_mean", float("nan")))
        krel = result["kernel_relative_at_lag_times"].get(str(float(lag_ps)), float("nan"))
        lines.append(
            f"| {lag} | {lag_ps:.1f} | {vals[0]:.4f} | {vals[1]:.4f} | "
            f"{vals[2]:.4f} | {vals[3]:.4f} | {krel:.4f} |"
        )
    lines.extend([
        "",
        "## Integrated Absolute ACF",
        "",
        "| Mode | sum abs weighted ACF |",
        "|---|---:|",
    ])
    for name in ("none", "force", "memory", "full"):
        lines.append(f"| {name} | {result['integrated_abs_acf'][name]:.6g} |")
    red = result["integrated_abs_acf_reduction"]
    lines.extend([
        "",
        "## Reductions",
        "",
        "| Comparison | Reduction |",
        "|---|---:|",
        f"| full vs none | {100.0 * red['full_vs_none']:.2f}% |",
        f"| full vs force | {100.0 * red['full_vs_force']:.2f}% |",
        f"| memory vs none | {100.0 * red['memory_vs_none']:.2f}% |",
        "",
        result["claim_boundary"],
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate V25 residual autocorrelation diagnostics")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--max-lag", type=int, default=5)
    p.add_argument("--max-trajectory-blocks", type=int, default=0)
    p.add_argument("--starts-per-trajectory", type=int, default=8)
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
        "integrated_abs_acf": result["integrated_abs_acf"],
        "reductions": result["integrated_abs_acf_reduction"],
    }, indent=2))


if __name__ == "__main__":
    main()
