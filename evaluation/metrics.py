"""Exact-window pairing, equal-protein aggregation and cluster bootstrap.

Raw rows describe windows, never independent proteins. All headline uncertainty
resamples proteins after averaging their anchors/replicas at each temperature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class WindowResult:
    protein_id: str
    trajectory_id: str
    temperature_k: float
    horizon: int
    variant: str
    rmsd: float
    persist_rmsd: float
    anchor_frame: int = -1
    target_frame: int = -1
    horizon_ps: float = 0.0
    oracle_rmsd: float | None = None
    anchor_decode_rmsd: float | None = None
    shape_rmsd: float | None = None
    msd: float | None = None
    sample_rmsds: list[float] = field(default_factory=list)
    sample_shape_rmsds: list[float] = field(default_factory=list)
    sample_msds: list[float] = field(default_factory=list)
    latent_msds: list[float] = field(default_factory=list)
    physical_energy_score: float | None = None
    persist_physical_energy_score: float | None = None
    n_residues: int = 0
    lag: int = 0
    diagnostic_only: bool = False
    sampling_seed: int = 0
    geometry: dict = field(default_factory=dict)
    sample_geometry: list[dict] = field(default_factory=list)
    target_geometry: dict = field(default_factory=dict)
    observables: dict = field(default_factory=dict)
    sample_observables: list[dict] = field(default_factory=list)
    dynamics_stats: dict = field(default_factory=dict)

    @property
    def window_key(self):
        return (self.protein_id, self.temperature_k, self.trajectory_id,
                self.anchor_frame, self.target_frame, self.horizon)


# Kept as an import alias; output files now distinguish windows and proteins.
PerProteinRow = WindowResult


def _finite(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def _nanmedian(xs):
    a = _finite(xs)
    return float(np.median(a)) if a else float("nan")


def _nanmean(xs):
    a = _finite(xs)
    return float(np.mean(a)) if a else float("nan")


def _nanmean_sem(xs):
    a = np.asarray(_finite(xs), dtype=float)
    if not a.size:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a))) if len(a) > 1 else float("nan")


def skill(persist: float, model: float, oracle: float) -> float:
    if oracle is None or not np.isfinite(persist-oracle) or persist-oracle <= 1e-6:
        return float("nan")
    return (persist-model)/(persist-oracle)


def paired_bootstrap_ci(a, b, n: int = 2000, seed: int = 0):
    """Paired mean difference; callers supply independent protein summaries.

    Invalid pairs are dropped jointly, never separately re-zipped.
    """
    if len(a) != len(b):
        raise ValueError("paired arrays must have equal length")
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    d = np.asarray([x-y for x, y in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def protein_summaries(rows: list[WindowResult]) -> list[dict]:
    """One equal-window mean per protein x temperature x horizon x variant."""
    groups: dict = {}
    seen = set()
    persistence = {}
    for row in rows:
        if row.anchor_frame < 0 or row.target_frame <= row.anchor_frame:
            raise ValueError("each result needs a valid anchor_frame and target_frame")
        key = row.window_key + (row.variant,)
        if key in seen:
            raise ValueError(f"duplicate window/variant result: {key}")
        seen.add(key)
        if row.variant == "persist":
            persistence[row.window_key] = row.rmsd
    for row in rows:
        if row.window_key in persistence and not np.isclose(
                row.persist_rmsd, persistence[row.window_key], rtol=1e-6, atol=1e-7):
            raise ValueError(f"persistence mismatch for exact window {row.window_key}")
        key = (row.protein_id, row.temperature_k, row.horizon, row.variant)
        groups.setdefault(key, []).append(row)
    result = []
    for (protein, temp, horizon, variant), rs in sorted(groups.items()):
        out = dict(protein_id=protein, temperature_k=temp, horizon=horizon,
                   variant=variant, n_windows=len(rs),
                   n_trajectories=len({r.trajectory_id for r in rs}),
                   diagnostic_only=any(r.diagnostic_only for r in rs))
        for name in ["rmsd", "persist_rmsd", "oracle_rmsd", "anchor_decode_rmsd",
                     "shape_rmsd", "msd", "physical_energy_score", "persist_physical_energy_score"]:
            out[name] = _nanmean([getattr(r, name) for r in rs])
        # Differences are paired BEFORE aggregation, with each original anchor.
        out["delta_vs_persist"] = _nanmean([r.rmsd-r.persist_rmsd for r in rs])
        out["sample_rmsd"] = _nanmean([_nanmean(r.sample_rmsds) for r in rs])
        out["sample_shape_rmsd"] = _nanmean([_nanmean(r.sample_shape_rmsds) for r in rs])
        out["sample_msd"] = _nanmean([_nanmean(r.sample_msds) for r in rs])
        out["physical_score_delta"] = _nanmean([
            r.physical_energy_score-r.persist_physical_energy_score for r in rs
            if r.physical_energy_score is not None and r.persist_physical_energy_score is not None])
        for name in ('geometry','target_geometry','observables','dynamics_stats'):
            keys = sorted({k for r in rs for k in getattr(r,name)})
            out[name] = {k:_nanmean([getattr(r,name).get(k) for r in rs]) for k in keys}
        for name in ('sample_geometry','sample_observables'):
            keys=sorted({k for r in rs for d in getattr(r,name) for k in d})
            out[name] = {k:_nanmean([_nanmean([d.get(k) for d in getattr(r,name)]) for r in rs]) for k in keys}
        result.append(out)
    return result


def aggregate(rows: list[WindowResult], *, seed: int = 0, bootstrap_reps: int = 2000):
    """Nested [temperature][horizon][variant], with proteins as uncertainty units."""
    grouped: dict = {}
    for p in protein_summaries(rows):
        grouped.setdefault(p["temperature_k"], {}).setdefault(p["horizon"], {}).setdefault(p["variant"], []).append(p)
    result = {}
    for temp, by_horizon in sorted(grouped.items()):
        result[temp] = {}
        for horizon, by_variant in sorted(by_horizon.items()):
            result[temp][horizon] = {}
            for variant, ps in sorted(by_variant.items()):
                rmsd = [p["rmsd"] for p in ps]
                persist = [p["persist_rmsd"] for p in ps]
                mean, sem = _nanmean_sem(rmsd)
                delta = paired_bootstrap_ci([p["delta_vs_persist"] for p in ps], [0.]*len(ps),
                                            n=bootstrap_reps, seed=seed)
                block = dict(n=len(ps), n_proteins=len(ps),
                             n_windows=sum(p["n_windows"] for p in ps),
                             rmsd_median=_nanmedian(rmsd), rmsd_mean=mean, rmsd_sem=sem,
                             persist_median=_nanmedian(persist),
                             oracle_median=_nanmedian([p["oracle_rmsd"] for p in ps]),
                             anchor_decode_median=_nanmedian([p["anchor_decode_rmsd"] for p in ps]),
                             delta_vs_persist=delta[0], delta_ci=list(delta),
                             delta_significant=len(ps) >= 2 and (delta[1] > 0 or delta[2] < 0),
                             uncertainty_unit="protein", insufficient_proteins=len(ps) < 2,
                             diagnostic_only=any(p["diagnostic_only"] for p in ps))
                block["skill"] = skill(block["persist_median"], block["rmsd_median"], block["oracle_median"])
                for name in ["shape_rmsd", "msd", "sample_rmsd", "sample_shape_rmsd", "sample_msd", "physical_energy_score"]:
                    block[name+"_mean"] = _nanmean([p[name] for p in ps])
                block["msd_median"] = _nanmedian([p["msd"] for p in ps])
                block["physical_score_delta_ci"] = list(paired_bootstrap_ci(
                    [p["physical_score_delta"] for p in ps], [0.]*len(ps), n=bootstrap_reps, seed=seed))
                for name in ('geometry','target_geometry','observables','dynamics_stats','sample_geometry','sample_observables'):
                    keys=sorted({k for p in ps for k in p[name]})
                    block[name]={k:_nanmean([p[name].get(k) for p in ps]) for k in keys}
                result[temp][horizon][variant] = block
    return result


def paired_variant_comparisons(rows, *, seed=0, bootstrap_reps=2000, pairs=None):
    """Exact-window paired model differences, averaged within then across protein.

    Missing variant windows fail; they are never silently intersected. Effects
    are descriptive when the oracle gate failed or only one protein is present.
    """
    protein_summaries(rows)  # validates duplicate keys and persistence agreement
    by_variant={}
    for r in rows:
        by_variant.setdefault(r.variant,{})[r.window_key]=r
    if pairs is None:
        pairs=[('h_plus_s','h_markov'),('h_plus_s','direct_jepa'),
               ('h_plus_s','persist'),('h_markov','persist'),('direct_jepa','persist')]
    result=[]
    for a,b in pairs:
        if a not in by_variant or b not in by_variant:
            continue
        if set(by_variant[a])!=set(by_variant[b]):
            raise ValueError(f'paired comparison has different window sets: {a} vs {b}')
        groups={}
        for key,ra in by_variant[a].items():
            rb=by_variant[b][key]
            metrics={'rmsd':(ra.rmsd,rb.rmsd),
                     'shape_rmsd':(ra.shape_rmsd,rb.shape_rmsd),
                     'physical_energy_score':(ra.physical_energy_score,rb.physical_energy_score),
                     'sample_rmsd':(_nanmean(ra.sample_rmsds),_nanmean(rb.sample_rmsds))}
            for metric in sorted(set(ra.observables)|set(rb.observables)):
                metrics['observable.'+metric]=(ra.observables.get(metric),rb.observables.get(metric))
            for metric,(va,vb) in metrics.items():
                groups.setdefault((ra.temperature_k,ra.horizon,metric),{}).setdefault(ra.protein_id,[]).append(
                    va-vb if va is not None and vb is not None else float('nan'))
        for (temp,horizon,metric),proteins in sorted(groups.items()):
            values=[_nanmean(deltas) for _,deltas in sorted(proteins.items())]
            n=int(sum(np.isfinite(x) for x in values))
            effect,lo,hi=paired_bootstrap_ci(values,[0.]*len(values),n=bootstrap_reps,seed=seed)
            result.append(dict(variant_a=a,variant_b=b,temperature_k=temp,horizon=horizon,
                metric=metric,delta_a_minus_b=effect,ci95=[lo,hi],n_proteins=n,
                uncertainty_unit='protein',lower_is_better=True,
                significant=n>=2 and (lo>0 or hi<0),
                diagnostic_only=any(r.diagnostic_only for r in rows)))
    return result
