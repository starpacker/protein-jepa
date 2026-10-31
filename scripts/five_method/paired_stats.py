"""Paired bootstrap comparisons between methods on the shared window set.

Each method scored the *same* windows, so a paired test is the right one: it
removes between-protein difficulty from the comparison, which is what makes a
0.02 difference in RMSF-r meaningful or not.

Reported per (metric, horizon, method pair):
    mean_A, mean_B, delta = B - A, 95% bootstrap CI of the paired mean delta,
    win rate = fraction of windows where B is better than A.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

WIN_KEYS = ("protein_id", "trajectory_id", "anchor_frame")


def load(paths):
    recs = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                recs.append(r)
    return recs


def index(recs):
    """(method, window) -> record."""
    out = {}
    for r in recs:
        out[(r["method"],) + tuple(r[k] for k in WIN_KEYS)] = r
    return out


def paired(idx, A, B, horizon, *path, protein_set=None):
    """Paired values of one metric for methods A and B on common windows."""
    va, vb = [], []
    for key, ra in idx.items():
        if key[0] != A:
            continue
        if protein_set is not None and ra.get("protein_set") != protein_set:
            continue
        rb = idx.get((B,) + key[1:])
        if rb is None:
            continue
        try:
            a = ra["horizons"][str(horizon)]
            b = rb["horizons"][str(horizon)]
            for k in path:
                a = a[k]
                b = b[k]
        except (KeyError, TypeError):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and np.isfinite(a) and np.isfinite(b):
            va.append(float(a)); vb.append(float(b))
    return np.array(va), np.array(vb)


def boot_ci(d, n=4000, alpha=0.05, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    m = d[idx].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def compare(idx, A, B, horizons, metrics, higher_better, protein_set=None):
    rows = []
    for hs in horizons:
        for key, label in metrics:
            va, vb = paired(idx, A, B, hs, "ens", key, protein_set=protein_set)
            if va.size < 2:
                continue
            d = vb - va
            lo, hi = boot_ci(d)
            better = (vb > va) if higher_better[key] else (vb < va)
            rows.append(dict(horizon=hs, metric=key, label=label,
                             mean_A=float(va.mean()), mean_B=float(vb.mean()),
                             delta=float(d.mean()), ci_lo=lo, ci_hi=hi,
                             win_rate=float(better.mean()), n=int(d.size),
                             significant=bool(lo > 0 or hi < 0)))
    return rows


def markdown(rows, A, B, title):
    out = [f"\n#### {title}\n",
           "| horizon | metric | %s | %s | Δ (B−A) | 95%% CI | B better | verdict |"
           % (A, B),
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sig = "✅" if r["significant"] else "—"
        better = "↑" if "↑" in r["label"] else "↓"
        out.append(
            f"| {r['horizon']} ns | {r['label']} | {r['mean_A']:.3f} | "
            f"{r['mean_B']:.3f} | {r['delta']:+.3f} | "
            f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
            f"{100*r['win_rate']:.0f}% | {sig} |")
    return "\n".join(out)
