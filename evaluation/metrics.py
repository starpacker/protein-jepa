"""Aggregation + statistics for evaluation results.

Input: a flat list of PerProteinRow (one per protein × T × horizon × variant).
Output: nested aggregate tables with median/mean/SEM, delta vs persistence, and
a skill score, plus a paired bootstrap CI for (variant - persistence).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PerProteinRow:
    protein_id: str
    trajectory_id: str
    temperature_k: float
    horizon: int
    variant: str
    rmsd: float
    persist_rmsd: float
    oracle_rmsd: float | None = None
    msd: float | None = None
    n_residues: int = 0
    lag: int = 0


def _nanmedian(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def _nanmean_sem(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0


def skill(persist: float, model: float, oracle: float) -> float:
    """(persist - model) / (persist - oracle).  1=oracle, 0=persist, <0=worse."""
    denom = persist - oracle
    if denom is None or not np.isfinite(denom) or denom < 1e-6:
        return float("nan")
    return (persist - model) / denom


def paired_bootstrap_ci(a, b, n: int = 2000, seed: int = 0):
    """Mean(a-b) with 95% CI over paired samples."""
    a = np.asarray([x for x in a if x is not None and not math.isnan(x)], dtype=float)
    b = np.asarray([x for x in b if x is not None and not math.isnan(x)], dtype=float)
    m = min(len(a), len(b))
    if m == 0:
        return float("nan"), float("nan"), float("nan")
    d = a[:m] - b[:m]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, (n, m))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(rows: list[PerProteinRow]):
    """Nested dict: [temperature][horizon][variant] -> stats."""
    out: dict = {}
    for r in rows:
        out.setdefault(r.temperature_k, {}).setdefault(r.horizon, {}).setdefault(r.variant, []).append(r)

    result: dict = {}
    for T, by_h in out.items():
        result[T] = {}
        for h, by_v in by_h.items():
            block: dict = {}
            persist_vals = [r.persist_rmsd for r in by_v.get("persist", [])]
            for variant, rs in by_v.items():
                rmsds = [r.rmsd for r in rs]
                oracles = [r.oracle_rmsd for r in rs if r.oracle_rmsd is not None]
                orc = _nanmedian(oracles) if oracles else None
                pmed = _nanmedian(persist_vals) if persist_vals else float("nan")
                med = _nanmedian(rmsds)
                mean, sem = _nanmean_sem(rmsds)
                blk = {
                    "n": len(rs),
                    "rmsd_median": med,
                    "rmsd_mean": mean,
                    "rmsd_sem": sem,
                    "oracle_median": orc,
                    "persist_median": pmed,
                    "delta_vs_persist": (med - pmed) if np.isfinite(pmed) else float("nan"),
                    "msd_median": _nanmedian([r.msd for r in rs]),
                }
                if orc is not None:
                    blk["skill"] = skill(pmed, med, orc)
                # paired CI vs persistence
                pv = {r.protein_id + f"_{r.trajectory_id}": r.persist_rmsd for r in by_v.get("persist", [])}
                keys = [r.protein_id + f"_{r.trajectory_id}" for r in rs]
                b_paired = [pv[k] for k in keys if k in pv]
                a_paired = [r.rmsd for k, r in zip(keys, rs) if k in pv]
                if b_paired:
                    m, lo, hi = paired_bootstrap_ci(a_paired, b_paired)
                    blk["delta_ci"] = [m, lo, hi]
                    blk["delta_significant"] = bool(lo > 0 or hi < 0)
                block[variant] = blk
            result[T][h] = block
    return result
