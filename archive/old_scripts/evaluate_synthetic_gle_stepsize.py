#!/usr/bin/env python3
"""Step-size sensitivity check for the synthetic GLE system.

Reviewer 3 asked for a small known system where the actual sampler is checked
at ``dt`` and ``dt/2``. We simulate the same synthetic GLE parameters over the
same physical time, downsample the half-step trajectories to the nominal
observation interval, and compare stationary/kinetic statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_synthetic_gle_recovery import acf, basin_summary, jsd_hist, transition_rate
from generate_synthetic_gle import double_well_force, generate_gle_trajectory


def generate_exact_ou_colored_trajectory(
    n_steps: int,
    dt: float,
    temperature: float,
    a: float,
    b: float,
    diffusion: float,
    kernel_decay: float,
    kernel_strength: float,
    seed: int,
    auxiliary_dim: int = 4,
) -> np.ndarray:
    """Generate the same colored-forcing model with exact OU auxiliary updates."""
    rng = np.random.default_rng(seed)
    z = np.zeros(n_steps, dtype=np.float64)
    eta = np.zeros((n_steps, auxiliary_dim), dtype=np.float64)
    z[0] = rng.normal(0.0, 1.0)
    eta[0] = rng.normal(0.0, np.sqrt(temperature), size=auxiliary_dim)
    rho = float(np.exp(-kernel_decay * dt))
    eta_sigma = float(np.sqrt(max(temperature * (1.0 - rho * rho), 0.0)))
    z_sigma = float(np.sqrt(2.0 * diffusion * temperature * dt))
    for t in range(1, n_steps):
        force = double_well_force(z[t - 1], a, b)
        aux_coupling = kernel_strength * eta[t - 1].sum()
        z[t] = z[t - 1] + dt * diffusion * (force + aux_coupling) / temperature
        z[t] += z_sigma * rng.normal()
        eta[t] = rho * eta[t - 1] + eta_sigma * rng.normal(size=auxiliary_dim)
    return z


def _simulate_set(
    params: dict,
    dt: float,
    n_steps: int,
    n_traj: int,
    seed: int,
    exact_ou: bool,
) -> list[np.ndarray]:
    out = []
    for i in range(n_traj):
        if exact_ou:
            z = generate_exact_ou_colored_trajectory(
                n_steps=n_steps,
                dt=dt,
                temperature=float(params["temperature"]),
                a=float(params["a"]),
                b=float(params["b"]),
                diffusion=float(params["diffusion"]),
                kernel_decay=float(params["kernel_decay"]),
                kernel_strength=float(params["kernel_strength"]),
                seed=seed + 7919 * i,
            )
        else:
            z, *_ = generate_gle_trajectory(
                n_steps=n_steps,
                dt=dt,
                temperature=float(params["temperature"]),
                a=float(params["a"]),
                b=float(params["b"]),
                diffusion=float(params["diffusion"]),
                kernel_decay=float(params["kernel_decay"]),
                kernel_strength=float(params["kernel_strength"]),
                seed=seed + 7919 * i,
            )
        noise = float(params.get("observation_noise", 0.0))
        if noise > 0:
            z = z + np.random.default_rng(seed + 7919 * i + 100000).normal(0.0, noise, size=z.shape)
        out.append(z)
    return out


def _concat(trajs: list[np.ndarray], burn_in: int) -> np.ndarray:
    return np.concatenate([traj[burn_in:] for traj in trajs])


def _acf_physical(trajs: list[np.ndarray], burn_in: int, max_lag: int) -> np.ndarray:
    values = _concat(trajs, burn_in)
    return acf(values, max_lag)


def _summarize(trajs: list[np.ndarray], dt: float, bins: np.ndarray, burn_in: int, acf_max_lag: int) -> dict:
    values = _concat(trajs, burn_in)
    return {
        "stationary_values_n": int(values.shape[0]),
        "transition_rate": transition_rate([traj[burn_in:] for traj in trajs], dt),
        "basin": basin_summary([traj[burn_in:] for traj in trajs], bins),
        "acf": _acf_physical(trajs, burn_in, acf_max_lag).tolist(),
    }


def run(args: argparse.Namespace) -> dict:
    root = Path(args.gle_data_dir)
    params = json.loads((root / "ground_truth" / "params.json").read_text())
    if args.temperature is not None:
        params["temperature"] = float(args.temperature)
    dt = float(params["dt"])
    if args.dt is not None:
        dt = float(args.dt)
    half_dt = dt / 2.0
    n_steps = int(args.traj_length or params["traj_length"])
    half_steps = 2 * n_steps - 1
    bins = np.linspace(args.z_min, args.z_max, args.n_bins + 1)

    nominal = _simulate_set(params, dt, n_steps, args.n_traj, args.seed, args.exact_ou)
    half_raw = _simulate_set(params, half_dt, half_steps, args.n_traj, args.seed, args.exact_ou)
    half_downsampled = [traj[::2] for traj in half_raw]

    burn_nominal = args.burn_in
    burn_half_raw = args.burn_in * 2
    nominal_summary = _summarize(nominal, dt, bins, burn_nominal, args.acf_max_lag)
    half_summary = _summarize(half_downsampled, dt, bins, burn_nominal, args.acf_max_lag)
    half_raw_summary = _summarize(half_raw, half_dt, bins, burn_half_raw, args.acf_max_lag * 2)

    nominal_values = _concat(nominal, burn_nominal)
    half_values = _concat(half_downsampled, burn_nominal)
    stationary_jsd = jsd_hist(nominal_values, half_values, bins)
    rate_rel_diff = abs(nominal_summary["transition_rate"] - half_summary["transition_rate"]) / max(
        abs(nominal_summary["transition_rate"]), 1.0e-20
    )
    acf_nominal = np.asarray(nominal_summary["acf"])
    acf_half = np.asarray(half_summary["acf"])
    acf_rmse = float(np.sqrt(np.mean((acf_nominal - acf_half) ** 2)))
    basin = {
        "left_occupancy_abs_diff": abs(
            nominal_summary["basin"]["left_occupancy"] - half_summary["basin"]["left_occupancy"]
        ),
        "right_occupancy_abs_diff": abs(
            nominal_summary["basin"]["right_occupancy"] - half_summary["basin"]["right_occupancy"]
        ),
        "left_peak_abs_diff": abs(nominal_summary["basin"]["left_peak"] - half_summary["basin"]["left_peak"]),
        "right_peak_abs_diff": abs(nominal_summary["basin"]["right_peak"] - half_summary["basin"]["right_peak"]),
    }
    gates = {
        "stationary_jsd_lt_0_02": bool(stationary_jsd < 0.02),
        "transition_rate_rel_diff_lt_0_15": bool(rate_rel_diff < 0.15),
        "acf_rmse_lt_0_10": bool(acf_rmse < 0.10),
        "basin_occupancy_abs_diff_lt_0_05": bool(
            max(basin["left_occupancy_abs_diff"], basin["right_occupancy_abs_diff"]) < 0.05
        ),
    }
    return {
        "experiment_id": "synthetic_gle_stepsize_sensitivity",
        "protocol": {
            "data_dir": str(root),
            "nominal_dt": dt,
            "half_dt": half_dt,
            "n_traj": args.n_traj,
            "nominal_steps": n_steps,
            "half_steps": half_steps,
            "physical_time": dt * (n_steps - 1),
            "temperature": float(params["temperature"]),
            "burn_in_nominal_steps": burn_nominal,
            "acf_max_lag_nominal_grid": args.acf_max_lag,
            "claim_boundary": (
                "This checks numerical step-size sensitivity of the controlled synthetic sampler. "
                "It is not a proof that real-protein V25 autonomous rollouts preserve the exact MD equilibrium."
            ),
            "sampler": "exact_ou_auxiliary" if args.exact_ou else "legacy_euler_auxiliary",
        },
        "metrics": {
            "stationary_jsd_nominal_vs_half_downsampled": stationary_jsd,
            "transition_rate_nominal": nominal_summary["transition_rate"],
            "transition_rate_half_downsampled": half_summary["transition_rate"],
            "transition_rate_half_raw": half_raw_summary["transition_rate"],
            "transition_rate_relative_diff": rate_rel_diff,
            "acf_rmse_nominal_vs_half_downsampled": acf_rmse,
            "basin_differences": basin,
            "nominal_basin": nominal_summary["basin"],
            "half_downsampled_basin": half_summary["basin"],
            "nominal_stationary_n": nominal_summary["stationary_values_n"],
            "half_stationary_n": half_summary["stationary_values_n"],
        },
        "gates": gates,
    }


def write_summary(result: dict, path: Path) -> None:
    m = result["metrics"]
    b = m["basin_differences"]
    lines = [
        "# Synthetic GLE Step-Size Sensitivity",
        "",
        f"Nominal dt: {result['protocol']['nominal_dt']}",
        f"Half dt: {result['protocol']['half_dt']}",
        f"n trajectories: {result['protocol']['n_traj']}",
        f"physical time per trajectory: {result['protocol']['physical_time']:.4f}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Stationary JSD, dt vs dt/2 downsampled | {m['stationary_jsd_nominal_vs_half_downsampled']:.5f} |",
        f"| Transition rate, dt | {m['transition_rate_nominal']:.5f} |",
        f"| Transition rate, dt/2 downsampled | {m['transition_rate_half_downsampled']:.5f} |",
        f"| Transition-rate relative difference | {100*m['transition_rate_relative_diff']:.2f}% |",
        f"| ACF RMSE, dt vs dt/2 downsampled | {m['acf_rmse_nominal_vs_half_downsampled']:.5f} |",
        f"| Left occupancy absolute difference | {b['left_occupancy_abs_diff']:.5f} |",
        f"| Right occupancy absolute difference | {b['right_occupancy_abs_diff']:.5f} |",
        f"| Left basin peak absolute difference | {b['left_peak_abs_diff']:.5f} |",
        f"| Right basin peak absolute difference | {b['right_peak_abs_diff']:.5f} |",
        "",
        "## Gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for key, value in result["gates"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", result["protocol"]["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic GLE dt vs dt/2 sensitivity")
    parser.add_argument("--gle-data-dir", default="artifacts/reviewer_experiments/synthetic_gle")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--n-traj", type=int, default=16)
    parser.add_argument("--traj-length", type=int, default=0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--z-min", type=float, default=-4.5)
    parser.add_argument("--z-max", type=float, default=4.5)
    parser.add_argument("--n-bins", type=int, default=90)
    parser.add_argument("--acf-max-lag", type=int, default=500)
    parser.add_argument("--legacy-aux-euler", dest="exact_ou", action="store_false")
    parser.set_defaults(exact_ou=True)
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
        "stationary_jsd": result["metrics"]["stationary_jsd_nominal_vs_half_downsampled"],
        "transition_rate_relative_diff": result["metrics"]["transition_rate_relative_diff"],
        "acf_rmse": result["metrics"]["acf_rmse_nominal_vs_half_downsampled"],
        "gates": result["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
