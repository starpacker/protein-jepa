#!/usr/bin/env python3
"""Decide whether Mori memory improves over the overdamped TM-JEPA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(
    overdamped: dict,
    gle: dict,
    minimum_relative_improvement: float,
    minimum_path_nll_improvement: float = 0.001,
) -> dict:
    shared_horizons = sorted(set(overdamped["horizons"]) & set(gle["horizons"]), key=int)
    if not shared_horizons:
        raise ValueError("TM-JEPA ablations have no shared horizons")
    overdamped_skill = sum(overdamped["horizons"][key]["energy_skill"] for key in shared_horizons) / len(shared_horizons)
    gle_skill = sum(gle["horizons"][key]["energy_skill"] for key in shared_horizons) / len(shared_horizons)
    overdamped_ck = sum(
        overdamped["horizons"][key]["augmented_ck_energy_distance"] for key in shared_horizons
    ) / len(shared_horizons)
    gle_ck = sum(
        gle["horizons"][key]["augmented_ck_energy_distance"] for key in shared_horizons
    ) / len(shared_horizons)
    overdamped_ac = overdamped["closure"]["innovation_lag1_rms"]
    gle_ac = gle["closure"]["innovation_lag1_rms"]
    overdamped_path_nll = overdamped["closure"].get("path_nll_per_dimension")
    gle_path_nll = gle["closure"].get("path_nll_per_dimension")

    def relative_improvement(baseline: float, candidate: float) -> float:
        return (baseline - candidate) / max(abs(baseline), 1.0e-12)

    ac_improvement = relative_improvement(overdamped_ac, gle_ac)
    ck_improvement = relative_improvement(overdamped_ck, gle_ck)
    path_nll_improvement = (
        overdamped_path_nll - gle_path_nll
        if overdamped_path_nll is not None and gle_path_nll is not None
        else None
    )
    gates = {
        "energy_not_degraded": gle_skill >= overdamped_skill - 0.01,
        "innovation_autocorrelation_improved": ac_improvement >= minimum_relative_improvement,
        "path_nll_improved": path_nll_improvement is not None
        and path_nll_improvement >= minimum_path_nll_improvement,
        "memory_coupling_nonzero": gle["physics"]["mean_absolute_memory_coupling_per_ps"] > 0.0,
        "gle_diffusion_psd": gle["physics"]["minimum_diffusion_eigenvalue"] > 0.0,
    }
    return {
        "shared_horizons": [int(value) for value in shared_horizons],
        "overdamped_mean_energy_skill": overdamped_skill,
        "gle_mean_energy_skill": gle_skill,
        "overdamped_mean_ck_energy_distance": overdamped_ck,
        "gle_mean_ck_energy_distance": gle_ck,
        "overdamped_innovation_lag1_rms": overdamped_ac,
        "gle_innovation_lag1_rms": gle_ac,
        "overdamped_path_nll_per_dimension": overdamped_path_nll,
        "gle_path_nll_per_dimension": gle_path_nll,
        "path_nll_absolute_improvement": path_nll_improvement,
        "innovation_autocorrelation_relative_improvement": ac_improvement,
        # CK is a numerical semigroup diagnostic for both already-Markov
        # implementations.  It is reported, but is not evidence for memory.
        "ck_relative_improvement": ck_improvement,
        "gates": gates,
        "memory_promoted": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overdamped", required=True)
    parser.add_argument("--gle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.05)
    parser.add_argument("--minimum-path-nll-improvement", type=float, default=0.001)
    args = parser.parse_args()
    result = compare(
        json.loads(Path(args.overdamped).read_text()),
        json.loads(Path(args.gle).read_text()),
        args.minimum_relative_improvement,
        args.minimum_path_nll_improvement,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
