#!/usr/bin/env python3
"""Bootstrap the causal and closure value of V3 Mori memory."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _mean_lag_rms(result: dict) -> float:
    values = []
    for temperature in result["by_temperature"].values():
        values.extend(
            item["rms"]
            for item in temperature["closure"]["lagged_covariance"].values()
        )
    return sum(values) / max(len(values), 1)


def _bootstrap_interval(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def compare(
    overdamped: dict,
    mori: dict,
    *,
    bootstrap_samples: int = 10000,
    seed: int = 6303,
) -> dict:
    proteins = sorted(set(overdamped["by_protein"]) & set(mori["by_protein"]))
    if not proteins:
        raise ValueError("V3 ablations have no shared proteins")
    path_improvements = [
        overdamped["by_protein"][protein]["path_nll_per_dimension"]
        - mori["by_protein"][protein]["path_nll_per_dimension"]
        for protein in proteins
    ]
    energy_changes = [
        mori["by_protein"][protein]["energy_score"]
        - overdamped["by_protein"][protein]["energy_score"]
        for protein in proteins
    ]
    path_interval = _bootstrap_interval(
        path_improvements, samples=bootstrap_samples, seed=seed
    )
    energy_interval = _bootstrap_interval(
        energy_changes, samples=bootstrap_samples, seed=seed + 1
    )
    overdamped_lag = _mean_lag_rms(overdamped)
    mori_lag = _mean_lag_rms(mori)
    lag_reduction = (overdamped_lag - mori_lag) / max(overdamped_lag, 1.0e-12)
    mean_path_improvement = sum(path_improvements) / len(path_improvements)
    mean_energy_change = sum(energy_changes) / len(energy_changes)
    filter_advantage = mori["physics"]["mean_memory_filter_advantage"]
    gates = {
        "path_nll_improves_by_0.005": mean_path_improvement >= 0.005,
        "path_nll_bootstrap_lower_positive": path_interval[0] > 0.0,
        "multi_lag_residual_reduced_30_percent": lag_reduction >= 0.30,
        "endpoint_energy_not_degraded": mean_energy_change <= 0.01,
        "filtered_bath_beats_equilibrium_reset": filter_advantage >= 0.005,
        "mori_temperature_closure_passes": bool(mori.get("promotion_pass")),
    }
    return {
        "shared_proteins": len(proteins),
        "mean_path_nll_improvement": mean_path_improvement,
        "path_nll_improvement_95pct_interval": list(path_interval),
        "mean_endpoint_energy_score_change": mean_energy_change,
        "endpoint_energy_change_95pct_interval": list(energy_interval),
        "overdamped_mean_multi_lag_rms": overdamped_lag,
        "mori_mean_multi_lag_rms": mori_lag,
        "multi_lag_relative_reduction": lag_reduction,
        "mori_filter_advantage": filter_advantage,
        "gates": gates,
        "memory_promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overdamped", required=True)
    parser.add_argument("--mori", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=6303)
    args = parser.parse_args()
    result = compare(
        json.loads(Path(args.overdamped).read_text()),
        json.loads(Path(args.mori).read_text()),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
