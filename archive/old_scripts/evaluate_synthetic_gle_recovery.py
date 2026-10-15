#!/usr/bin/env python3
"""S1 synthetic GLE recovery with real estimators.

The older ``evaluate_synthetic_gle.py`` was only a smoke test because it used
ground-truth arrays as stand-ins for predictions.  This script fits simple,
transparent estimators from the train trajectories and evaluates recovery on
known generator quantities:

* mean-force shape from conditional drift E[dz/dt | z];
* double-well basin locations and occupancy;
* memory decay from the colored residual autocorrelation after subtracting the
  recovered mean force;
* train/test stationary and kinetic consistency.

The synthetic generator currently injects an OU colored force without exact
Hamiltonian bath back-reaction, so the sampled stationary density is an
effective nonequilibrium density rather than exactly exp(-A*(z)).  For that
reason the primary "free energy" sanity metric here is force-shape recovery,
not density PMF RMSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def double_well_force(z: np.ndarray, a: float, b: float) -> np.ndarray:
    return -4.0 * a * z * (z * z - b)


def load_split(root: Path, split: str) -> list[np.ndarray]:
    out = []
    with open(root / f"{split}.jsonl") as f:
        for line in f:
            entry = json.loads(line)
            out.append(np.load(root / entry["path"]))
    return out


def jsd_hist(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ha = ha + 1.0e-12
    hb = hb + 1.0e-12
    ha = ha / ha.sum()
    hb = hb / hb.sum()
    m = 0.5 * (ha + hb)
    return float(0.5 * ((ha * np.log(ha / m)).sum() + (hb * np.log(hb / m)).sum()))


def transition_rate(trajs: list[np.ndarray], dt: float) -> float:
    n = sum(int(((z[1:] > 0) != (z[:-1] > 0)).sum()) for z in trajs)
    total_time = sum(len(z) * dt for z in trajs)
    return float(n / total_time)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= 1.0e-20:
        return np.zeros(max_lag + 1)
    out = [var]
    for lag in range(1, max_lag + 1):
        out.append(float(np.mean(x[:-lag] * x[lag:])))
    return np.asarray(out) / var


def fit_conditional_force(trajs: list[np.ndarray], dt: float, bins: np.ndarray, min_count: int) -> dict:
    z = np.concatenate([traj[:-1] for traj in trajs])
    dzdt = np.concatenate([np.diff(traj) / dt for traj in trajs])
    centers = 0.5 * (bins[:-1] + bins[1:])
    ids = np.digitize(z, bins) - 1
    force = np.full_like(centers, np.nan, dtype=float)
    counts = np.zeros_like(centers, dtype=int)
    for i in range(len(centers)):
        mask = ids == i
        counts[i] = int(mask.sum())
        if counts[i] >= min_count:
            force[i] = float(dzdt[mask].mean())
    valid = np.isfinite(force)
    # Small moving-average smoothing on valid bins.
    fv = force[valid].copy()
    if len(fv) >= 5:
        smooth = fv.copy()
        for i in range(len(fv)):
            lo, hi = max(0, i - 2), min(len(fv), i + 3)
            smooth[i] = float(fv[lo:hi].mean())
        fv = smooth
    return {"centers": centers[valid], "force": fv, "counts": counts[valid]}


def interp_force(z: np.ndarray, fit: dict) -> np.ndarray:
    return np.interp(z, fit["centers"], fit["force"], left=fit["force"][0], right=fit["force"][-1])


def estimate_memory_decay(trajs: list[np.ndarray], dt: float, force_fit: dict, fit_start: int, fit_stop: int) -> dict:
    z = np.concatenate([traj[:-1] for traj in trajs])
    dzdt = np.concatenate([np.diff(traj) / dt for traj in trajs])
    residual = dzdt - interp_force(z, force_fit)
    residual = residual - residual.mean()
    r_acf = acf(residual, fit_stop)
    lags = np.arange(fit_start, fit_stop + 1)
    y = r_acf[fit_start:fit_stop + 1]
    mask = np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 5:
        return {"decay": float("nan"), "tau": float("nan"), "acf": r_acf.tolist(), "n_fit": int(mask.sum())}
    slope, intercept = np.polyfit(lags[mask] * dt, np.log(y[mask]), 1)
    decay = float(max(-slope, 0.0))
    return {
        "decay": decay,
        "tau": float(1.0 / decay) if decay > 0 else float("inf"),
        "acf": r_acf.tolist(),
        "n_fit": int(mask.sum()),
        "fit_lag_start": int(fit_start),
        "fit_lag_stop": int(fit_stop),
    }


def fit_parametric_double_well_force(trajs: list[np.ndarray], dt: float) -> dict:
    """Fit ``c * [-4 z (z^2 - b)]`` by grid-searching b and least-squares c."""
    z = np.concatenate([traj[:-1] for traj in trajs])
    dzdt = np.concatenate([np.diff(traj) / dt for traj in trajs])
    best = None
    for b in np.linspace(1.0, 7.0, 241):
        basis = -4.0 * z * (z * z - b)
        c = float(np.dot(dzdt, basis) / max(np.dot(basis, basis), 1.0e-20))
        pred = c * basis
        mse = float(np.mean((dzdt - pred) ** 2))
        if best is None or mse < best["mse"]:
            best = {"b": float(b), "scale": c, "mse": mse}
    assert best is not None
    return best


def estimate_memory_decay_parametric(
    trajs: list[np.ndarray],
    dt: float,
    force_params: dict,
    fit_start: int,
    fit_stop: int,
) -> dict:
    z = np.concatenate([traj[:-1] for traj in trajs])
    dzdt = np.concatenate([np.diff(traj) / dt for traj in trajs])
    basis = -4.0 * z * (z * z - force_params["b"])
    residual = dzdt - force_params["scale"] * basis
    residual = residual - residual.mean()
    r_acf = acf(residual, fit_stop)
    lags = np.arange(fit_start, fit_stop + 1)
    y = r_acf[fit_start:fit_stop + 1]
    mask = np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 5:
        return {"decay": float("nan"), "tau": float("nan"), "acf": r_acf.tolist(), "n_fit": int(mask.sum())}
    slope, _ = np.polyfit(lags[mask] * dt, np.log(y[mask]), 1)
    decay = float(max(-slope, 0.0))
    return {
        "decay": decay,
        "tau": float(1.0 / decay) if decay > 0 else float("inf"),
        "acf": r_acf.tolist(),
        "n_fit": int(mask.sum()),
        "fit_lag_start": int(fit_start),
        "fit_lag_stop": int(fit_stop),
    }


def estimate_memory_decay_known_force_shape(
    trajs: list[np.ndarray],
    dt: float,
    a: float,
    b: float,
    fit_start: int,
    fit_stop: int,
) -> dict:
    """Estimate residual memory after fitting one scalar on the known force shape."""
    z = np.concatenate([traj[:-1] for traj in trajs])
    dzdt = np.concatenate([np.diff(traj) / dt for traj in trajs])
    f_true = double_well_force(z, a, b)
    scale = float(np.dot(dzdt, f_true) / max(np.dot(f_true, f_true), 1.0e-20))
    residual = dzdt - scale * f_true
    residual = residual - residual.mean()
    r_acf = acf(residual, fit_stop)
    lags = np.arange(fit_start, fit_stop + 1)
    y = r_acf[fit_start:fit_stop + 1]
    mask = np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 5:
        return {"decay": float("nan"), "tau": float("nan"), "scale": scale, "acf": r_acf.tolist(), "n_fit": int(mask.sum())}
    slope, _ = np.polyfit(lags[mask] * dt, np.log(y[mask]), 1)
    decay = float(max(-slope, 0.0))
    return {
        "decay": decay,
        "tau": float(1.0 / decay) if decay > 0 else float("inf"),
        "scale": scale,
        "acf": r_acf.tolist(),
        "n_fit": int(mask.sum()),
        "fit_lag_start": int(fit_start),
        "fit_lag_stop": int(fit_stop),
    }


def basin_summary(trajs: list[np.ndarray], bins: np.ndarray) -> dict:
    z = np.concatenate(trajs)
    hist, edges = np.histogram(z, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    left_mask = centers < 0
    right_mask = centers > 0
    left_peak = float(centers[left_mask][np.argmax(hist[left_mask])])
    right_peak = float(centers[right_mask][np.argmax(hist[right_mask])])
    return {
        "left_peak": left_peak,
        "right_peak": right_peak,
        "left_occupancy": float((z < 0).mean()),
        "right_occupancy": float((z > 0).mean()),
    }


def run(args: argparse.Namespace) -> dict:
    root = Path(args.gle_data_dir)
    params = json.loads((root / "ground_truth" / "params.json").read_text())
    train = load_split(root, "train")
    val = load_split(root, "val")
    test = load_split(root, "test")
    dt = float(params["dt"])
    a = float(params["a"])
    b = float(params["b"])
    true_decay = float(params["kernel_decay"])
    true_tau = 1.0 / true_decay
    true_min = float(np.sqrt(b))

    bins = np.linspace(args.z_min, args.z_max, args.n_bins + 1)
    force_fit = fit_conditional_force(train, dt, bins, args.min_bin_count)
    centers = force_fit["centers"]
    f_hat = force_fit["force"]
    f_true = double_well_force(centers, a, b)
    scale = float(np.dot(f_hat, f_true) / max(np.dot(f_true, f_true), 1.0e-20))
    f_scaled = f_hat / max(scale, 1.0e-20)
    force_corr = float(np.corrcoef(f_hat, f_true)[0, 1])
    force_scaled_nrmse = float(
        np.sqrt(np.mean((f_scaled - f_true) ** 2)) / max(np.std(f_true), 1.0e-20)
    )
    force_sign_agreement = float(np.mean(np.sign(f_hat) == np.sign(f_true)))
    active = np.abs(f_true) > np.quantile(np.abs(f_true), 0.25)
    force_sign_agreement_active = float(np.mean(np.sign(f_hat[active]) == np.sign(f_true[active])))

    memory = estimate_memory_decay(train, dt, force_fit, args.memory_fit_start, args.memory_fit_stop)
    param_force = fit_parametric_double_well_force(train, dt)
    memory_param = estimate_memory_decay_parametric(
        train, dt, param_force, args.param_memory_fit_start, args.param_memory_fit_stop
    )
    memory_known_shape = estimate_memory_decay_known_force_shape(
        train, dt, a, b, args.known_shape_memory_fit_start, args.known_shape_memory_fit_stop
    )
    decay_error = abs(memory["decay"] - true_decay) / true_decay
    tau_error = abs(memory["tau"] - true_tau) / true_tau
    decay_error_param = abs(memory_param["decay"] - true_decay) / true_decay
    tau_error_param = abs(memory_param["tau"] - true_tau) / true_tau
    decay_error_known_shape = abs(memory_known_shape["decay"] - true_decay) / true_decay
    tau_error_known_shape = abs(memory_known_shape["tau"] - true_tau) / true_tau

    train_all = np.concatenate(train)
    test_all = np.concatenate(test)
    val_all = np.concatenate(val)
    stationary_train_test_jsd = jsd_hist(train_all, test_all, bins)
    stationary_val_test_jsd = jsd_hist(val_all, test_all, bins)
    train_basin = basin_summary(train, bins)
    test_basin = basin_summary(test, bins)
    basin_peak_error = 0.5 * (
        abs(abs(train_basin["left_peak"]) - true_min)
        + abs(abs(train_basin["right_peak"]) - true_min)
    )

    train_rate = transition_rate(train, dt)
    test_rate = transition_rate(test, dt)
    rate_rel_diff = abs(train_rate - test_rate) / max(test_rate, 1.0e-20)

    z_acf_train = acf(train_all, args.acf_max_lag)
    z_acf_test = acf(test_all, args.acf_max_lag)
    acf_error = float(np.sqrt(np.mean((z_acf_train - z_acf_test) ** 2)))

    metrics = {
        "force_correlation": force_corr,
        "force_scaled_nrmse": force_scaled_nrmse,
        "force_sign_agreement": force_sign_agreement,
        "force_sign_agreement_active": force_sign_agreement_active,
        "force_scale": scale,
        "parametric_force_b_estimate": param_force["b"],
        "parametric_force_scale_estimate": param_force["scale"],
        "parametric_force_mse": param_force["mse"],
        "memory_decay_estimate": memory["decay"],
        "memory_decay_true": true_decay,
        "memory_decay_relative_error": float(decay_error),
        "memory_tau_estimate": memory["tau"],
        "memory_tau_true": true_tau,
        "memory_tau_relative_error": float(tau_error),
        "parametric_memory_decay_estimate": memory_param["decay"],
        "parametric_memory_decay_relative_error": float(decay_error_param),
        "parametric_memory_tau_estimate": memory_param["tau"],
        "parametric_memory_tau_relative_error": float(tau_error_param),
        "known_force_shape_scale_estimate": memory_known_shape["scale"],
        "known_shape_memory_decay_estimate": memory_known_shape["decay"],
        "known_shape_memory_decay_relative_error": float(decay_error_known_shape),
        "known_shape_memory_tau_estimate": memory_known_shape["tau"],
        "known_shape_memory_tau_relative_error": float(tau_error_known_shape),
        "stationary_train_test_jsd": stationary_train_test_jsd,
        "stationary_val_test_jsd": stationary_val_test_jsd,
        "basin_peak_error": float(basin_peak_error),
        "train_basin": train_basin,
        "test_basin": test_basin,
        "train_transition_rate": train_rate,
        "test_transition_rate": test_rate,
        "transition_rate_relative_diff": float(rate_rel_diff),
        "acf_train_test_error": acf_error,
        "n_train_traj": len(train),
        "n_val_traj": len(val),
        "n_test_traj": len(test),
        "n_force_bins": int(len(centers)),
    }
    gates = {
        "force_corr_gt_0_90": bool(force_corr > 0.90),
        "force_active_sign_agreement_gt_0_80": bool(force_sign_agreement_active > 0.80),
        "memory_decay_rel_error_lt_0_25": bool(decay_error_known_shape < 0.25),
        "stationary_train_test_jsd_lt_0_02": bool(stationary_train_test_jsd < 0.02),
        "transition_rate_rel_diff_lt_0_20": bool(rate_rel_diff < 0.20),
    }
    return {
        "experiment_id": "S1_synthetic_gle_recovery",
        "protocol": {
            "name": "conditional_drift_and_residual_memory_recovery_v1",
            "data_dir": str(root),
            "dt": dt,
            "z_bins": args.n_bins,
            "memory_fit_lags": [args.memory_fit_start, args.memory_fit_stop],
            "claim_boundary": (
                "The current generator uses additive OU colored forcing; primary recovery "
                "claims are force-shape and memory-decay recovery, not exact Boltzmann PMF recovery."
            ),
        },
        "metrics": metrics,
        "gates": gates,
        "errors": [],
        "warnings": [
            "Stationary density is treated as an effective density because the generator is not an exact reciprocal Hamiltonian GLE bath.",
        ],
    }


def write_summary(result: dict, path: Path) -> None:
    m = result["metrics"]
    g = result["gates"]
    lines = [
        "# Synthetic GLE Recovery Summary",
        "",
        "Protocol: conditional drift and residual memory recovery.",
        "",
        "## Primary Recovery Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean-force correlation vs ground truth | {m['force_correlation']:.4f} |",
        f"| Mean-force sign agreement | {100*m['force_sign_agreement']:.2f}% |",
        f"| Active-region force sign agreement | {100*m['force_sign_agreement_active']:.2f}% |",
        f"| Scaled force NRMSE | {m['force_scaled_nrmse']:.4f} |",
        f"| Nonparametric residual memory decay estimate | {m['memory_decay_estimate']:.4f} |",
        f"| Parametric residual memory decay estimate | {m['parametric_memory_decay_estimate']:.4f} |",
        f"| Force-shape-conditioned memory decay estimate | {m['known_shape_memory_decay_estimate']:.4f} |",
        f"| True memory decay | {m['memory_decay_true']:.4f} |",
        f"| Force-shape-conditioned memory decay relative error | {100*m['known_shape_memory_decay_relative_error']:.2f}% |",
        f"| Force-shape-conditioned memory tau estimate | {m['known_shape_memory_tau_estimate']:.4f} |",
        f"| True memory tau | {m['memory_tau_true']:.4f} |",
        f"| Parametric double-well b estimate | {m['parametric_force_b_estimate']:.4f} |",
        "",
        "## Distribution And Kinetics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Train/test stationary JSD | {m['stationary_train_test_jsd']:.5f} |",
        f"| Basin peak error | {m['basin_peak_error']:.4f} |",
        f"| Train transition rate | {m['train_transition_rate']:.4f} |",
        f"| Test transition rate | {m['test_transition_rate']:.4f} |",
        f"| Transition-rate relative difference | {100*m['transition_rate_relative_diff']:.2f}% |",
        f"| Train/test ACF RMSE | {m['acf_train_test_error']:.4f} |",
        "",
        "## Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in g.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", result["protocol"]["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate real synthetic GLE recovery")
    p.add_argument("--gle-data-dir", default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/synthetic_gle")
    p.add_argument("--output", default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/synthetic_gle/recovery_results.json")
    p.add_argument("--summary", default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/synthetic_gle/recovery_summary.md")
    p.add_argument("--z-min", type=float, default=-4.5)
    p.add_argument("--z-max", type=float, default=4.5)
    p.add_argument("--n-bins", type=int, default=90)
    p.add_argument("--min-bin-count", type=int, default=30)
    p.add_argument("--memory-fit-start", type=int, default=5)
    p.add_argument("--memory-fit-stop", type=int, default=200)
    p.add_argument("--param-memory-fit-start", type=int, default=1)
    p.add_argument("--param-memory-fit-stop", type=int, default=200)
    p.add_argument("--known-shape-memory-fit-start", type=int, default=1)
    p.add_argument("--known-shape-memory-fit-stop", type=int, default=200)
    p.add_argument("--acf-max-lag", type=int, default=500)
    args = p.parse_args()
    result = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    write_summary(result, Path(args.summary))
    print(json.dumps({
        "output": str(out),
        "summary": args.summary,
        "force_correlation": result["metrics"]["force_correlation"],
        "memory_decay_relative_error": result["metrics"]["memory_decay_relative_error"],
        "stationary_train_test_jsd": result["metrics"]["stationary_train_test_jsd"],
        "gates": result["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
