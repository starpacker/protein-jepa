#!/usr/bin/env python3
"""Compare strict-M1 closure results across checkpoints (v3 / v5 / v9)."""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

METRICS = [
    "prediction_mse",
    "innovation_whiteness",
    "fdt_residual",
    "stationary_jsd",
    "acf_error",
    "rollout_mean_norm",
    "semigroup_mean_defect",
]
BASELINES = ["Markov", "FreeEnergy", "Mori", "GenericMarkov", "GRU", "CausalTransformer"]
OUTCOMES = ["Full", "FullCalibrated"]
LOWER_IS_BETTER = set(METRICS)


def _mean(values: list[float]) -> float:
    return st.mean(values) if values else float("nan")


def summarise(path: str) -> dict:
    d = json.load(open(path))
    rows = {}
    for name, method in d["methods"].items():
        vals = {k: [] for k in METRICS}
        for _, mm in method["per_trajectory"].items():
            for k in METRICS:
                v = mm.get(k)
                if v is not None:
                    vals[k].append(v)
        rows[name] = {k: _mean(vals[k]) for k in METRICS}
    return {
        "path": path,
        "checkpoint": d["provenance"]["checkpoint"],
        "seed": d["provenance"]["seed"],
        "dt_ps": d["protocol"]["sampled_dt_ps"],
        "kernel_norm": d["protocol"]["native_kernel_norm_excluding_zero_lag"],
        "gates": d["gates"],
        "n_test_proteins": d["provenance"]["n_test_proteins"],
        "rows": rows,
        "comparisons": d["pairwise_full_minus_baseline"],
    }


def main() -> None:
    runs = [summarise(p) for p in sys.argv[1:]]

    print("=" * 130)
    print("STRICT M1 THERMODYNAMIC-CLOSURE COMPARISON")
    print("=" * 130)
    for r in runs:
        print(f"  {r['path']}")
        print(f"     checkpoint        : {r['checkpoint']}")
        print(f"     seed              : {r['seed']}   dt_ps={r['dt_ps']}   test proteins={r['n_test_proteins']}")
        print(f"     ||K[1:]||_F       : {r['kernel_norm']:.6g}")
        print(f"     gates             : {json.dumps(r['gates'])}")

    for metric in METRICS:
        print()
        print(f"### {metric}  (lower is better)")
        header = f"{'method':<20}" + "".join(f"{Path(r['path']).name[:26]:>28}" for r in runs)
        print(header)
        methods = ["Markov", "FreeEnergy", "Mori", *OUTCOMES, "GenericMarkov", "GRU", "CausalTransformer"]
        for m in methods:
            line = f"{m:<20}"
            for r in runs:
                v = r["rows"].get(m, {}).get(metric)
                line += f"{v:>28.6g}" if v is not None and v == v else f"{'n/a':>28}"
            print(line)

    print()
    print("=" * 130)
    print("PAIRED PROTEIN-BOOTSTRAP COMPARISON: Full - baseline  (negative = Full better)")
    print("=" * 130)
    for r in runs:
        print(f"\n--- {r['path']} ---")
        for outcome in OUTCOMES:
            for bl in BASELINES:
                key = f"{outcome}_vs_{bl}"
                comp = r["comparisons"].get(key)
                if not comp:
                    continue
                flags = []
                for met, v in comp.items():
                    if v["lower"] > 0 or v["upper"] < 0:
                        better = v["diff_mean"] < 0
                        flags.append(f"{met}{'(+)' if better else '(-)'}")
                print(f"  {key:<34} significant: {', '.join(flags) if flags else 'none'}")


if __name__ == "__main__":
    main()
