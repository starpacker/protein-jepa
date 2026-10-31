"""Aggregate the per-seed paired comparisons of the S_0 arms into a decision.

`compare_s0_arms.py` produces one paired JSON per (candidate, reference) pair
for ONE seed pair.  A single seed cannot decide anything here: the frozen SOTA
seeds disagree with each other significantly at h=64 (10 of 21 tests), so the
seed-to-seed spread is the yardstick the S_0 effect has to clear.

This script takes the per-seed paired files and reports, per horizon x metric:

  * the 3-seed mean improvement (candidate - reference, sign-oriented),
  * how many seeds gave a one-sided-98.33% significant win / loss,
  * the pre-registered verdict.

Pre-registered decision rule (fixed before training, see
LASP_CORRECT_FRAMEWORK_PLAN_20261031.md):
a comparison counts as a WIN only if the 3-seed mean improvement is positive
AND at least 2 of 3 seeds have their one-sided lower bound above zero.  It
counts as a LOSS if the mean is negative and 2 of 3 seeds have the upper
bound below zero.  Anything else is INCONCLUSIVE, and an inconclusive result
on the primary metrics is a substantive finding, not a failed run.

Usage
    python -m scripts.report_s0_arms --pair exact_vs_prior=s42=...s42.json \
        --pair exact_vs_prior=s43=...s43.json --pair exact_vs_prior=s44=...s44.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PRIMARY = ["rmsf_corr", "md_pca_w2", "variance_w2"]


def _strip_seed(name):
    """'exact_seed42_vs_prior_seed42' -> 'exact_vs_prior'."""
    return re.sub(r"_seed\d+", "", name)


def _is_horizon(key):
    try:
        float(key)
    except (TypeError, ValueError):
        return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True,
                   metavar="COMPARISON=SEED=PATH",
                   help="e.g. exact_vs_prior=s42=results/.../s42.paired.json")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # comparison -> seed -> {horizon: {metric: entry}}
    data = defaultdict(dict)
    for spec in a.pair:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise SystemExit(f"bad --pair {spec!r}, want COMPARISON=SEED=PATH")
        comp, seed, path = parts
        d = json.loads(Path(path).read_text())
        # A paired file may hold one or several comparisons, or be a bare
        # horizon-keyed table.  The comparison key carries the seed suffix
        # ("exact_seed42_vs_prior_seed42"); match it seed-insensitively so the
        # caller can name the comparison once for all three seeds.
        found = [k for k in d.get("comparisons", {}) if _strip_seed(k) == comp]
        if found:
            table = d["comparisons"][found[0]]
        elif comp in d.get("comparisons", {}):
            table = d["comparisons"][comp]
        elif all(_is_horizon(k) for k in d if k != "protocol"):
            table = d
        else:
            raise SystemExit(
                f"{path}: no comparison {comp!r}; available "
                f"{sorted(d.get('comparisons', {}))}")
        data[comp][seed] = table

    out = {"decision_rule": (
        "WIN if 3-seed mean improvement > 0 and >=2/3 seeds have one-sided "
        "98.33% lower bound > 0; LOSS if mean < 0 and >=2/3 upper bound < 0; "
        "otherwise INCONCLUSIVE"), "primary_metrics": PRIMARY,
        "comparisons": {}}

    for comp, seeds in sorted(data.items()):
        horizons = sorted({h for s in seeds.values() for h in s}, key=float)
        metrics = sorted({m for s in seeds.values() for h in s.values() for m in h})
        rows = {}
        for h in horizons:
            rows[h] = {}
            for m in metrics:
                vals, wins, losses, n = [], 0, 0, 0
                for seed, byh in seeds.items():
                    e = byh.get(h, {}).get(m)
                    if not e or not e.get("n_valid"):
                        continue
                    n += 1
                    vals.append(e["improvement"])
                    if e["verdict"] == "win":
                        wins += 1
                    elif e["verdict"] == "loss":
                        losses += 1
                if n == 0:
                    rows[h][m] = dict(n_seeds=0, verdict="no_data")
                    continue
                mean_impr = sum(vals) / len(vals)
                if mean_impr > 0 and wins >= 2:
                    verdict = "WIN"
                elif mean_impr < 0 and losses >= 2:
                    verdict = "LOSS"
                else:
                    verdict = "INCONCLUSIVE"
                rows[h][m] = dict(
                    n_seeds=n, mean_improvement=mean_impr,
                    improvements=vals, n_sig_win=wins, n_sig_loss=losses,
                    seed_agreement=("all" if (wins == n or losses == n) else
                                    "majority" if max(wins, losses) * 2 > n else
                                    "split"),
                    verdict=verdict, primary=m in PRIMARY)
        out["comparisons"][comp] = rows

        print(f"\n=== {comp} ===")
        print(f"{'h':>4} {'metric':<22} {'meanImpr':>10} {'wins':>5} "
              f"{'losses':>6} {'agree':<9} verdict")
        for h in horizons:
            for m in metrics:
                r = rows[h][m]
                if not r.get("n_seeds"):
                    continue
                star = " *" if r["primary"] else ""
                print(f"{h:>4} {m + star:<22} {r['mean_improvement']:>+10.5f} "
                      f"{r['n_sig_win']:>5} {r['n_sig_loss']:>6} "
                      f"{r['seed_agreement']:<9} {r['verdict']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
