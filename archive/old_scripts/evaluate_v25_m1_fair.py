#!/usr/bin/env python3
"""V25 M1 reviewer supplement under the original paired-window protocol.

This script deliberately does NOT use ``evaluate_m1_strict.py``.  It follows
the existing V25 M1 data path: shuffled paired windows from the configured
manifest, the same checkpoint, and the same ``forward_paired`` outputs used by
``protein_jepa_frame/eval/m1k1r1_benchmark.py``.

It adds reviewer-requested controls:

1. Fair actual-drift metrics.  Force, memory, and full are each evaluated as a
   single 8D predicted displacement vector, not as asymmetric 1D vs 2D
   projection spaces.
2. The original subspace-coverage diagnostic is retained and compared against
   random 1D and random 2D subspace controls.
3. A global scalar calibration is reported for each drift source.  This tests
   whether the vector field points in the right direction when only one
   physically meaningless unit-conversion scalar is fitted.  Per-sample oracle
   projection is also reported as an upper bound on direction-only skill.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


EPS = 1.0e-12


def _paired_bootstrap(values: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=np.float64)
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


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, np.maximum(norm, EPS))


def _single_direction_projection(delta: np.ndarray, direction: np.ndarray) -> np.ndarray:
    u = _unit(direction)
    coeff = np.sum(delta * u, axis=-1, keepdims=True)
    return coeff * u


def _two_direction_projection(delta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    au = _unit(a)
    bu = _unit(b)
    ca = np.sum(delta * au, axis=-1, keepdims=True)
    cb = np.sum(delta * bu, axis=-1, keepdims=True)
    cab = np.sum(au * bu, axis=-1, keepdims=True)
    sin2 = np.maximum(1.0 - cab * cab, EPS)
    alpha = (ca - cab * cb) / sin2
    beta = (cb - cab * ca) / sin2
    return alpha * au + beta * bu


def _per_sample_skill(delta: np.ndarray, pred: np.ndarray) -> np.ndarray:
    d2 = np.sum(delta * delta, axis=-1)
    r2 = np.sum((delta - pred) ** 2, axis=-1)
    return 1.0 - r2 / np.maximum(d2, EPS)


def _cosine(delta: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sum(delta * pred, axis=-1) / (
        np.maximum(np.linalg.norm(delta, axis=-1), EPS)
        * np.maximum(np.linalg.norm(pred, axis=-1), EPS)
    )


def _global_scalar_fit(delta: np.ndarray, pred: np.ndarray) -> tuple[float, np.ndarray]:
    numerator = float(np.sum(delta * pred))
    denominator = float(np.sum(pred * pred))
    alpha = numerator / max(denominator, EPS)
    return alpha, alpha * pred


def _aggregate_mode(delta: np.ndarray, pred_rate: np.ndarray, dt: np.ndarray) -> dict:
    pred_raw = pred_rate * dt[:, None]
    alpha, pred_cal = _global_scalar_fit(delta, pred_raw)
    pred_oracle = _single_direction_projection(delta, pred_raw)

    baseline_mse = np.mean(np.sum(delta * delta, axis=-1))
    raw_mse = np.mean(np.sum((delta - pred_raw) ** 2, axis=-1))
    cal_mse = np.mean(np.sum((delta - pred_cal) ** 2, axis=-1))
    oracle_mse = np.mean(np.sum((delta - pred_oracle) ** 2, axis=-1))

    out = {
        "raw": {
            "mse": float(raw_mse / delta.shape[1]),
            "r2_global": float(1.0 - raw_mse / max(baseline_mse, EPS)),
            "skill": _paired_bootstrap(_per_sample_skill(delta, pred_raw)),
            "cosine": _paired_bootstrap(_cosine(delta, pred_raw)),
        },
        "scalar_calibrated": {
            "alpha": float(alpha),
            "mse": float(cal_mse / delta.shape[1]),
            "r2_global": float(1.0 - cal_mse / max(baseline_mse, EPS)),
            "skill": _paired_bootstrap(_per_sample_skill(delta, pred_cal)),
            "cosine": _paired_bootstrap(_cosine(delta, pred_cal)),
        },
        "oracle_magnitude": {
            "mse": float(oracle_mse / delta.shape[1]),
            "r2_global": float(1.0 - oracle_mse / max(baseline_mse, EPS)),
            "skill": _paired_bootstrap(_per_sample_skill(delta, pred_oracle)),
            "cosine": _paired_bootstrap(_cosine(delta, pred_oracle)),
        },
        "norms": {
            "pred_raw_norm_mean": float(np.linalg.norm(pred_raw, axis=-1).mean()),
            "pred_rate_norm_mean": float(np.linalg.norm(pred_rate, axis=-1).mean()),
            "delta_norm_mean": float(np.linalg.norm(delta, axis=-1).mean()),
        },
    }
    return out


def _subspace_diagnostics(delta: np.ndarray, force: np.ndarray, memory: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n, d = delta.shape
    pred_force = _single_direction_projection(delta, force)
    pred_memory = _single_direction_projection(delta, memory)
    pred_full_span = _two_direction_projection(delta, force, memory)

    random_1d_skills = []
    random_2d_skills = []
    fixed_random_1d_means = []
    fixed_random_2d_means = []
    for _ in range(256):
        r1 = rng.normal(size=(n, d))
        r2a = rng.normal(size=(n, d))
        r2b = rng.normal(size=(n, d))
        random_1d_skills.append(_per_sample_skill(delta, _single_direction_projection(delta, r1)))
        random_2d_skills.append(_per_sample_skill(delta, _two_direction_projection(delta, r2a, r2b)))

        g1 = rng.normal(size=(1, d))
        g2a = rng.normal(size=(1, d))
        g2b = rng.normal(size=(1, d))
        g1 = np.repeat(g1, n, axis=0)
        g2a = np.repeat(g2a, n, axis=0)
        g2b = np.repeat(g2b, n, axis=0)
        fixed_random_1d_means.append(
            _per_sample_skill(delta, _single_direction_projection(delta, g1)).mean()
        )
        fixed_random_2d_means.append(
            _per_sample_skill(delta, _two_direction_projection(delta, g2a, g2b)).mean()
        )
    random_1d = np.stack(random_1d_skills, axis=0).mean(axis=0)
    random_2d = np.stack(random_2d_skills, axis=0).mean(axis=0)
    fixed_random_1d_means = np.asarray(fixed_random_1d_means)
    fixed_random_2d_means = np.asarray(fixed_random_2d_means)

    full_minus_rand2 = _per_sample_skill(delta, pred_full_span) - random_2d
    force_minus_rand1 = _per_sample_skill(delta, pred_force) - random_1d
    memory_minus_rand1 = _per_sample_skill(delta, pred_memory) - random_1d
    full_span_skill = _per_sample_skill(delta, pred_full_span)
    force_skill = _per_sample_skill(delta, pred_force)
    memory_skill = _per_sample_skill(delta, pred_memory)

    return {
        "force_1d": _paired_bootstrap(force_skill),
        "memory_1d": _paired_bootstrap(memory_skill),
        "full_2d_span": _paired_bootstrap(full_span_skill),
        "random_per_sample_1d_expected": _paired_bootstrap(random_1d),
        "random_per_sample_2d_expected": _paired_bootstrap(random_2d),
        "random_fixed_global_1d_mean_over_draws": {
            "mean": float(fixed_random_1d_means.mean()),
            "std": float(fixed_random_1d_means.std()),
            "ci_lo": float(np.quantile(fixed_random_1d_means, 0.025)),
            "ci_hi": float(np.quantile(fixed_random_1d_means, 0.975)),
            "n": int(len(fixed_random_1d_means)),
        },
        "random_fixed_global_2d_mean_over_draws": {
            "mean": float(fixed_random_2d_means.mean()),
            "std": float(fixed_random_2d_means.std()),
            "ci_lo": float(np.quantile(fixed_random_2d_means, 0.025)),
            "ci_hi": float(np.quantile(fixed_random_2d_means, 0.975)),
            "n": int(len(fixed_random_2d_means)),
        },
        "force_minus_fixed_random_1d_mean": {
            "mean": float(force_skill.mean() - fixed_random_1d_means.mean()),
            "ci_lo": float(force_skill.mean() - np.quantile(fixed_random_1d_means, 0.975)),
            "ci_hi": float(force_skill.mean() - np.quantile(fixed_random_1d_means, 0.025)),
            "n": int(len(fixed_random_1d_means)),
        },
        "memory_minus_fixed_random_1d_mean": {
            "mean": float(memory_skill.mean() - fixed_random_1d_means.mean()),
            "ci_lo": float(memory_skill.mean() - np.quantile(fixed_random_1d_means, 0.975)),
            "ci_hi": float(memory_skill.mean() - np.quantile(fixed_random_1d_means, 0.025)),
            "n": int(len(fixed_random_1d_means)),
        },
        "full_minus_fixed_random_2d_mean": {
            "mean": float(full_span_skill.mean() - fixed_random_2d_means.mean()),
            "ci_lo": float(full_span_skill.mean() - np.quantile(fixed_random_2d_means, 0.975)),
            "ci_hi": float(full_span_skill.mean() - np.quantile(fixed_random_2d_means, 0.025)),
            "n": int(len(fixed_random_2d_means)),
        },
        "force_minus_random_1d": _paired_bootstrap(force_minus_rand1),
        "memory_minus_random_1d": _paired_bootstrap(memory_minus_rand1),
        "full_minus_random_2d": _paired_bootstrap(full_minus_rand2),
        "force_memory_cosine": _paired_bootstrap(_cosine(force, memory)),
    }


def _collect_arrays(model: ProteinJEPA, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    model.eval()
    rows = {"current": [], "future": [], "force": [], "memory": [], "total": [], "dt": []}
    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_paired(batch)
        required = [
            "lspwm_m1_global_current",
            "lspwm_m1_global_future",
            "lspwm_m1_dissipative_drift",
            "lspwm_m1_bath_drift",
            "lspwm_m1_total_drift",
            "lspwm_m1_dt",
        ]
        if any(out.get(k) is None for k in required):
            n_batches += 1
            continue
        rows["current"].append(out["lspwm_m1_global_current"].detach().cpu().numpy())
        rows["future"].append(out["lspwm_m1_global_future"].detach().cpu().numpy())
        rows["force"].append(out["lspwm_m1_dissipative_drift"].detach().cpu().numpy())
        rows["memory"].append(out["lspwm_m1_bath_drift"].detach().cpu().numpy())
        rows["total"].append(out["lspwm_m1_total_drift"].detach().cpu().numpy())
        dt = out["lspwm_m1_dt"].detach().cpu().numpy()
        rows["dt"].append(dt.reshape(-1))
        n_batches += 1

    if not rows["current"]:
        raise RuntimeError("no valid V25 M1 batches were collected")
    return {k: np.concatenate(v, axis=0) for k, v in rows.items()}


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if args.verbose:
        print(f"missing={len(missing)} unexpected={len(unexpected)}")

    dataset = build_dataset(config.data)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )

    arrays = _collect_arrays(model, loader, device, args.max_batches)
    delta = arrays["future"] - arrays["current"]
    dt = arrays["dt"].astype(np.float64)
    force = arrays["force"].astype(np.float64)
    memory = arrays["memory"].astype(np.float64)
    total = arrays["total"].astype(np.float64)

    modes = {
        "M_Force": _aggregate_mode(delta, force, dt),
        "M_Memory": _aggregate_mode(delta, memory, dt),
        "M_Full": _aggregate_mode(delta, total, dt),
    }
    signed_total_variants = {
        "F_plus_M": _aggregate_mode(delta, force + memory, dt),
        "negF_plus_M": _aggregate_mode(delta, -force + memory, dt),
        "F_minus_M": _aggregate_mode(delta, force - memory, dt),
        "negF_minus_M": _aggregate_mode(delta, -force - memory, dt),
        "force_only_sign_flipped": _aggregate_mode(delta, -force, dt),
        "memory_only_sign_flipped": _aggregate_mode(delta, -memory, dt),
    }
    signed_total_ranking = sorted(
        (
            {
                "variant": name,
                "calibrated_r2": payload["scalar_calibrated"]["r2_global"],
                "calibrated_skill": payload["scalar_calibrated"]["skill"]["mean"],
                "oracle_skill": payload["oracle_magnitude"]["skill"]["mean"],
                "cosine": payload["scalar_calibrated"]["cosine"]["mean"],
            }
            for name, payload in signed_total_variants.items()
        ),
        key=lambda row: (row["oracle_skill"], row["calibrated_r2"]),
        reverse=True,
    )
    none_mse = float(np.mean(np.sum(delta * delta, axis=-1)) / delta.shape[1])
    subspace = _subspace_diagnostics(delta, force, memory, seed=args.seed)

    output = {
        "experiment_id": "V25_M1_fair_paired_window",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "existing V25 M1 paired-window evaluator protocol",
            "manifest_path": config.data.manifest_path,
            "batch_size": config.data.batch_size,
            "max_batches": args.max_batches,
            "n_pairs": int(delta.shape[0]),
            "cv_dim": int(delta.shape[1]),
            "bootstrap_unit": "paired_window_same_as_existing_M1",
            "uses_m1_strict": False,
        },
        "baseline": {
            "M_None_mse": none_mse,
            "delta_norm_mean": float(np.linalg.norm(delta, axis=-1).mean()),
            "dt_unique": sorted(float(x) for x in np.unique(dt)[:20]),
        },
        "fair_actual_drift": modes,
        "signed_total_variants": signed_total_variants,
        "signed_total_ranking": signed_total_ranking,
        "subspace_diagnostic": subspace,
        "claim_boundary": (
            "Fair actual-drift compares one predicted 8D vector per mode. "
            "Scalar-calibrated rows fit a single global scalar per mode to correct unit scale; "
            "oracle-magnitude rows are direction-only upper bounds. "
            "Subspace diagnostics are retained only as geometric/decomposition diagnostics."
        ),
    }
    return output


def write_summary(result: dict, path: Path) -> None:
    fair = result["fair_actual_drift"]
    sub = result["subspace_diagnostic"]
    lines = [
        "# V25 M1 Fair Drift Supplement",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"Protocol: {result['protocol']['base']}",
        f"n pairs: {result['protocol']['n_pairs']}",
        "",
        "## Fair Actual-Drift Metrics",
        "",
        "| Mode | scalar alpha | calibrated R2 | calibrated mean skill | cosine | oracle direction skill |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["M_Force", "M_Memory", "M_Full"]:
        row = fair[name]
        lines.append(
            f"| {name} | {row['scalar_calibrated']['alpha']:.6g} | "
            f"{row['scalar_calibrated']['r2_global']:.4f} | "
            f"{row['scalar_calibrated']['skill']['mean']:.4f} "
            f"[{row['scalar_calibrated']['skill']['ci_lo']:.4f}, {row['scalar_calibrated']['skill']['ci_hi']:.4f}] | "
            f"{row['scalar_calibrated']['cosine']['mean']:.4f} | "
            f"{row['oracle_magnitude']['skill']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Signed Total-Drift Convention Sweep",
        "",
        "| Variant | calibrated R2 | calibrated mean skill | cosine | oracle direction skill |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in result.get("signed_total_ranking", []):
        lines.append(
            f"| {row['variant']} | {row['calibrated_r2']:.4f} | "
            f"{row['calibrated_skill']:.4f} | {row['cosine']:.4f} | "
            f"{row['oracle_skill']:.4f} |"
        )
    lines.extend([
        "",
        "## Subspace Diagnostic With Random Controls",
        "",
        "| Diagnostic | mean | 95% CI |",
        "|---|---:|---:|",
    ])
    for key in [
        "force_1d",
        "memory_1d",
        "full_2d_span",
        "random_per_sample_1d_expected",
        "random_per_sample_2d_expected",
        "random_fixed_global_1d_mean_over_draws",
        "random_fixed_global_2d_mean_over_draws",
        "force_minus_fixed_random_1d_mean",
        "memory_minus_fixed_random_1d_mean",
        "full_minus_fixed_random_2d_mean",
        "force_minus_random_1d",
        "memory_minus_random_1d",
        "full_minus_random_2d",
        "force_memory_cosine",
    ]:
        row = sub[key]
        lines.append(f"| {key} | {row['mean']:.4f} | [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] |")
    lines.append("")
    lines.append(result["claim_boundary"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate V25 M1 fair actual-drift supplement")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    result = run(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        write_summary(result, Path(args.summary))
    print(json.dumps({
        "output": str(out_path),
        "summary": args.summary,
        "n_pairs": result["protocol"]["n_pairs"],
        "M_Full_calibrated_R2": result["fair_actual_drift"]["M_Full"]["scalar_calibrated"]["r2_global"],
        "M_Full_oracle_skill": result["fair_actual_drift"]["M_Full"]["oracle_magnitude"]["skill"]["mean"],
        "full_minus_random_2d": result["subspace_diagnostic"]["full_minus_random_2d"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
