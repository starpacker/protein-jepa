"""Pre-registered paired comparison of the S_0 arms.

Implements exactly the analysis fixed in LASP_CORRECT_FRAMEWORK_PLAN_20261031.md
BEFORE any arm was trained:

  * paired at window level (same protein/traj/temperature/anchor/horizon, so the
    sampler sees the same noise -- the eval derives its generator seed from the
    window key, not from the model seed),
  * protein-cluster bootstrap, 10000 draws, seed 20261024 (windows from one
    protein are not independent: resampling windows would understate the CI),
  * one-sided 98.33% lower confidence bound on the improvement; a "win" means
    that bound is > 0, a "loss" means the matching upper bound is < 0.

Improvement is oriented so positive always means the candidate is better:
delta = sign * (candidate - reference), sign = +1 for HIGHER_BETTER metrics.

Usage
    python -m scripts.compare_s0_arms \
        --run exact=exact_seed42=results/.../exact_seed42.ensemble.json \
        --compare exact:prior --compare gru:prior --compare exact:gru \
        --out results/.../paired.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.eval_lasp_final import METRICS, HIGHER_BETTER

N_BOOT = 10000
BOOT_SEED = 20261024
# one-sided 98.33% (alpha = 1/60); also keep the conventional two-sided 95%
LO_Q, HI_Q = 1.0 / 60.0, 1.0 - 1.0 / 60.0
CI_LO_Q, CI_HI_Q = 0.025, 0.975


def load_run(spec):
    """'name=path' -> (name, {(protein,traj,temp,anchor,horizon): row})"""
    name, path = spec.split("=", 1)
    data = json.loads(Path(path).read_text())
    rows = data["per_window"] if isinstance(data, dict) else data
    out = {}
    for r in rows:
        key = (r["protein"], r["traj"], r["temperature_k"],
               r["anchor_frame"], r["horizon"])
        if key in out:
            raise ValueError(f"duplicate window {key} in {path}")
        out[key] = r
    return name, out


def cluster_boot(values, clusters, n=N_BOOT, seed=BOOT_SEED):
    """Protein-resampled bootstrap distribution of the mean of ``values``."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.size == 0:
        return None
    uniq, inv = np.unique(c, return_inverse=True)
    if uniq.size < 2:
        return None                      # a single protein has no cluster CI
    sums = np.bincount(inv, weights=v, minlength=uniq.size)
    cnts = np.bincount(inv, minlength=uniq.size).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, uniq.size, size=(n, uniq.size))
    return sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)


def compare(cand, ref, field_c="lasp", field_r="lasp"):
    """Paired, per-horizon, per-metric improvement of cand over ref."""
    if cand.keys() != ref.keys():
        raise ValueError(
            "window sets differ -- refusing to compare a selected intersection "
            f"({len(cand)} vs {len(ref)} windows)")
    result = {}
    for h in sorted({k[-1] for k in cand}):
        keys = [k for k in cand if k[-1] == h]
        per_metric = {}
        for m in METRICS:
            x = np.array([cand[k][field_c][m] for k in keys], float)
            y = np.array([ref[k][field_r][m] for k in keys], float)
            valid = np.isfinite(x) & np.isfinite(y)
            if not valid.any():
                per_metric[m] = dict(n_valid=0)
                continue
            # Dropping non-finite windows silently would turn a broken rollout
            # into a "significant" result computed on a subset.  Say so loudly.
            if valid.sum() < 0.9 * len(keys):
                print(f"WARNING h={h} {m}: only {valid.sum()}/{len(keys)} "
                      f"finite paired windows")
            sign = 1.0 if m in HIGHER_BETTER else -1.0
            delta = sign * (x[valid] - y[valid])
            clusters = np.array([k[0] for k in keys])[valid]
            boot = cluster_boot(delta, clusters)
            entry = dict(
                candidate=float(np.mean(x[valid])),
                reference=float(np.mean(y[valid])),
                improvement=float(np.mean(delta)),
                n_valid=int(valid.sum()), n_total=len(keys),
            )
            if boot is None:
                entry.update(one_sided_lo=None, one_sided_hi=None,
                             ci95=[None, None], verdict="no_cluster_ci")
            else:
                lo = float(np.quantile(boot, LO_Q))
                hi = float(np.quantile(boot, HI_Q))
                entry.update(
                    one_sided_lo=lo, one_sided_hi=hi,
                    ci95=[float(np.quantile(boot, CI_LO_Q)),
                          float(np.quantile(boot, CI_HI_Q))],
                    verdict=("win" if lo > 0 else
                             "loss" if hi < 0 else "inconclusive"))
            per_metric[m] = entry
        result[str(h)] = per_metric
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   metavar="NAME=PATH", help="name=ensemble.json (repeatable)")
    p.add_argument("--compare", action="append", default=[], metavar="CAND:REF",
                   help="candidate:reference by run name (repeatable)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    runs = dict(load_run(s) for s in a.run)
    out = dict(protocol=dict(
        pairing="exact window key (protein,traj,temperature,anchor,horizon)",
        bootstrap=dict(clusters="protein", n=N_BOOT, seed=BOOT_SEED,
                       one_sided_confidence=1.0 - LO_Q),
        higher_better=sorted(HIGHER_BETTER),
        test_opened=False), runs={}, comparisons={})

    for name, rows in runs.items():
        out["runs"][name] = dict(
            n_windows=len(rows),
            n_proteins=len({k[0] for k in rows}),
            windows=[list(k) for k in sorted(rows)],
            mean={str(h): {m: float(np.mean([r["lasp"][m] for r in
                                             [rows[k] for k in rows if k[-1] == h]]))
                           for m in METRICS}
                  for h in sorted({k[-1] for k in rows})})

    for spec in a.compare:
        cname, rname = spec.split(":", 1)
        if cname not in runs or rname not in runs:
            raise SystemExit(f"unknown run in {spec!r}; have {sorted(runs)}")
        cmp_ = compare(runs[cname], runs[rname])
        out["comparisons"][f"{cname}_vs_{rname}"] = cmp_
        print(f"\n=== {cname} vs {rname} "
              f"(+ = {cname} better, one-sided 98.33% bound) ===")
        print(f"{'h':>4} {'metric':<22} {cname:>9} {rname:>9} "
              f"{'impr':>10} {'lo':>10} {'hi':>10}  verdict")
        for h, metrics in cmp_.items():
            for m, e in metrics.items():
                if not e.get("n_valid"):
                    print(f"{h:>4} {m:<22} {'--':>9} {'--':>9} {'--':>10}  no data")
                    continue
                print(f"{h:>4} {m:<22} {e['candidate']:>9.4f} "
                      f"{e['reference']:>9.4f} {e['improvement']:>+10.5f} "
                      f"{e['one_sided_lo'] if e['one_sided_lo'] is not None else float('nan'):>+10.5f} "
                      f"{e['one_sided_hi'] if e['one_sided_hi'] is not None else float('nan'):>+10.5f}  "
                      f"{e['verdict']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
