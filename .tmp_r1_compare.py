#!/usr/bin/env python3
"""Compare R1 history-gap results: LSP-WM checkpoints vs strict CA-coordinate baselines.

Usage:
  python3 .tmp_r1_compare.py <jepa_results.json> [<jepa_results.json> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GATE_THRESHOLD = 0.02  # evaluate_history_gap.py: pass when ci_lower > 0.02


def load_jepa(path: str) -> dict:
    d = json.load(open(path))
    metrics = d["metrics"]["by_horizon"]
    reps = [k for k in metrics if k != "horizons" and isinstance(metrics[k], dict)]
    rep = "lspwm" if "lspwm" in reps else reps[0]
    return {
        "path": path,
        "checkpoint": d["provenance"]["checkpoint_path"],
        "seed": d["provenance"]["random_seed"],
        "n_proteins": d["metrics"].get("n_proteins"),
        "latent_dim": metrics[rep].get("latent_dim"),
        "by_horizon": metrics[rep]["by_horizon"],
        "gate": d["gates"]["representation_gate"],
    }


def load_coordinates(path: str) -> dict:
    d = json.load(open(path))
    out = {}
    for rep, table in d["by_representation"].items():
        out[rep] = {}
        for key, v in table.items():
            out[rep][key] = {
                "current": v["mean_current_mse"],
                "delta": v["mean_delta_hist"],
                "ci": v["delta_hist_ci_95"],
                "n_proteins": v["n_proteins"],
            }
    return out


def main() -> None:
    jepa = [load_jepa(p) for p in sys.argv[1:] if p.endswith("results.json") and "coordinate" not in p]
    patterns = [p for p in sys.argv[1:] if "coordinate" in p]
    coords = load_coordinates(patterns[0]) if patterns else {}

    print("=" * 122)
    print("R1 HISTORY-GAP (representation closure)")
    print("=" * 122)
    for j in jepa:
        print(f"  {j['path']}")
        print(f"     checkpoint={j['checkpoint']}")
        print(f"     seed={j['seed']}  n_proteins={j['n_proteins']}  latent_dim={j['latent_dim']}  representation_gate={j['gate']}")
    print(f"  Gate rule: pass when protein-bootstrap CI lower bound > {GATE_THRESHOLD} (absolute MSE units).")

    for horizon in ["h=1", "h=2", "h=4", "h=8", "h=16", "h=32", "h=64"]:
        rows = []
        for j in jepa:
            table = j["by_horizon"].get(horizon)
            if not table:
                continue
            label = Path(j["path"]).parent.name.replace("representation_closure", "jepa").replace("_", "")
            for H, v in sorted(table["by_history_length"].items(), key=lambda kv: int(kv[0])):
                lo, hi = v["delta_hist_ci_95"]
                rows.append((f"JEPA[{label}]", H, v["mean_current_loss"], v["mean_delta_hist"], lo, hi))
        for rep in ("pca", "tica", "vamp"):
            if rep not in coords:
                continue
            key = f"{horizon}/H=2"
            v = coords[rep].get(key)
            if v:
                rows.append((rep.upper() + " (CA coords)", "2", v["current"], v["delta"], v["ci"][0], v["ci"][1]))
        if not rows:
            continue
        print()
        print(f"### horizon {horizon}")
        print(f"{'representation':<26}{'H':>3}{'current_loss':>16}{'delta_hist':>14}{'ci_low':>13}{'ci_high':>13}{'rel_delta':>11}{'gate':>8}{'winner':>8}")
        for name, H, cur, delta, lo, hi in rows:
            rel = delta / cur if cur else float("nan")
            gate = "PASS" if lo > GATE_THRESHOLD else "fail"
            better = "yes" if delta > 0 and lo > 0 else ("no" if hi < 0 else "tie")
            print(f"{name:<26}{H:>3}{cur:>16.5f}{delta:>14.5f}{lo:>13.5f}{hi:>13.5f}{rel:>10.2%}{gate:>8}{better:>8}")


if __name__ == "__main__":
    main()
