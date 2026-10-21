"""Aggregate independent bioemu2 replicate runs into mean +/- std.

Reads results/bioemu2_reps/seed*.json (each a full sweep under its own sampling
seed) and reports, per horizon and metric, the across-replicate mean, standard
deviation, and min/max, alongside the persistence baseline.

Usage:
  python -m evaluation.baselines.summarize_reps --glob 'results/bioemu2_reps/seed*.json'
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
from collections import defaultdict

import numpy as np

METRICS = ["rmsf_corr", "rmsf_mae", "root_mean_w2", "variance_w2", "md_pca_w2",
           "weak_contact_j", "transient_contacts_j"]
BETTER_UP = {"rmsf_corr", "weak_contact_j", "transient_contacts_j"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="results/bioemu2_reps/seed*.json")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    files = sorted(_glob.glob(a.glob))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        raise SystemExit(f"no files matched {a.glob}")
    print(f"[summarize] {len(runs)} replicates: "
          f"{[f.split('/')[-1] for f in files]}\n")

    per = defaultdict(lambda: defaultdict(list))
    pers = {}
    for r in runs:
        for hz, blk in r["by_horizon"].items():
            for m in METRICS:
                per[hz][m].append(blk["metrics"][m]["model"])
                pers.setdefault((hz, m), blk["metrics"][m]["persistence"])

    out = {"n_replicates": len(runs), "files": files, "by_horizon": {}}
    rows = []
    for hz in sorted(per, key=int):
        hz_out = {"metrics": {}}
        for m in METRICS:
            v = np.asarray(per[hz][m], dtype=float)
            q = pers[(hz, m)]
            d = (v - q) if m in BETTER_UP else (q - v)
            hz_out["metrics"][m] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max()),
                "persistence": float(q),
                "improvement_mean": float(d.mean()),
                "improvement_std": float(d.std(ddof=1)) if d.size > 1 else 0.0,
            }
            rows.append((hz, m, v.mean(), v.std(ddof=1) if v.size > 1 else 0.0,
                         q, d.mean(), d.std(ddof=1) if d.size > 1 else 0.0))
        out["by_horizon"][hz] = hz_out

    print(f"{'hz':>4} {'metric':>22} {'mean':>9} {'std':>8} {'persist':>9} "
          f"{'impr':>9} {'impr_std':>9}")
    print("-" * 78)
    for hz, m, mn, sd, q, im, isd in rows:
        print(f"{hz:>4} {m:>22} {mn:>9.4f} {sd:>8.4f} {q:>9.4f} "
              f"{im:>+9.4f} {isd:>9.4f}")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
