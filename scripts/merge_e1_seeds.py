"""Merge 3-seed E1 Representation Sufficiency Probe results into a mean table.

Reads ``results/todo_1107/e1_seed{42,43,44}.json`` and emits a Markdown table of
G_history (mean over seeds) plus per-probe R means, for each (target, tau).

R = mean squared error; G_history = R(P1) - R(P2). G > 0 means history carries
extra linearly-readable predictive information.
"""
from __future__ import annotations

import json
from pathlib import Path

SEEDS = [42, 43, 44]
TARGETS = ["A", "B", "C"]
TAUS = ["1", "4", "16"]
BASE = Path("results/todo_1107")


def load(seed: int) -> dict:
    return json.load(open(BASE / f"e1_seed{seed}.json"))


def main():
    seeds = {}
    for s in SEEDS:
        p = BASE / f"e1_seed{s}.json"
        if p.exists():
            seeds[s] = load(s)
    available = sorted(seeds)
    if not available:
        print("no seed results yet")
        return
    print(f"# E1 3-seed merge (available seeds: {available})\n")

    def mean(field: str, probe: str) -> str:
        vals = []
        for s in available:
            d = seeds[s]
            if field in d and tau in d[field] and tgt in d[field][tau]:
                r = d[field][tau][tgt].get(probe, {})
                if isinstance(r, dict) and "mse" in r:
                    vals.append(r["mse"])
        if not vals:
            return "—"
        return f"{sum(vals) / len(vals):.5f}"

    print("| 目标 | 时标 | R(P1) | R(P2) | R(P3) | G_history (mean±std) | 判读 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    names = {"A": "CA 距离变化", "B": "接触状态变化", "C": "集体运动投影"}
    for tau in TAUS:
        for tgt in TARGETS:
            g = []
            for s in available:
                d = seeds[s]
                if tau in d["targets"] and tgt in d["targets"][tau]:
                    g.append(d["targets"][tau][tgt]["G_history"])
            if not g:
                continue
            gm = sum(g) / len(g)
            if len(g) > 1:
                gsd = (sum((x - gm) ** 2 for x in g) / len(g)) ** 0.5
                gstr = f"{gm:+.4f}±{gsd:.4f}"
            else:
                gstr = f"{gm:+.4f}"
            r1 = mean("targets", "R_P1")
            r2 = mean("targets", "R_P2")
            r3 = mean("targets", "R_P3")
            verdict = "≈0" if abs(gm) < 0.1 else ("弱正" if gm > 0 else "≤0")
            print(f"| {names[tgt]} | {tau} ns | {r1} | {r2} | {r3} | {gstr} | {verdict} |")


if __name__ == "__main__":
    main()
