"""S-ablation for Stage B (v4): GLE-with-bath vs matched Markov (no-S) baseline.

Spec §4 arm D ("Markov / S disabled") and §3 B3.4 ("S ablation demonstrates whether
extra history improves held-out multi-step and kinetic prediction, beyond a
parameter-count-matched Markov model").

Both arms are trained by ``train_stage_b_v4`` with identical data/splits/seed/steps,
differing only in ``--bath-dim`` (8 = full GLE with S bath; 0 = H-only Markov, no
auxiliary bath/memory state).  This script reads the two ``metrics.jsonl`` files and
reports, per arm, the final held-out:

  * ca_bond_violation_fraction  (the structural-validity gate, lower is better)
  * mean_displacement_angstrom  (autonomous rollout CA displacement error)
  * latent energy score         (rollout stability)

and the S-vs-Markov delta.  A negative delta on bond violation means the bath (S)
*helps* hold the decoded geometry on-manifold; a ~0 delta means the bath contributes
nothing and the dynamics can be simplified (disclose per spec §8 "no bath claim unless
disabling S measurably changes forecasts").

Usage:
    python scripts/v4_s_ablation.py \
        --with-s artifacts/v4_stage_b/run_bond_grad/metrics.jsonl \
        --no-s  artifacts/v4_stage_b/run_nobath/metrics.jsonl
"""
from __future__ import annotations

import argparse
import json


def _val_records(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "val" in row:
                recs.append((row["step"], row["val"]))
    return recs


def _final(recs: list[dict], key: str) -> tuple[int, float]:
    # Last validation record and its value for `key`.
    step, val = recs[-1]
    return step, float(val.get(key, float("nan")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-s", required=True)
    ap.add_argument("--no-s", required=True)
    a = ap.parse_args()

    s_recs = _val_records(a.with_s)
    ns_recs = _val_records(a.no_s)
    if not s_recs or not ns_recs:
        raise SystemExit("one or both metrics.jsonl files lack validation records")

    keys = ["validation/ca_bond_violation_fraction",
            "validation/mean_displacement_angstrom", "loss/latent_es"]
    labels = {
        "validation/ca_bond_violation_fraction": "ca_bond_violation_fraction (lower better)",
        "validation/mean_displacement_angstrom": "mean_displacement_angstrom (lower better)",
        "loss/latent_es": "latent energy score (lower better)",
    }

    print(f"{'metric':38}{'with-S':>10}{'no-S':>10}{'delta(S-Markov)':>16}")
    print("-" * 74)
    for k in keys:
        ss, sv = _final(s_recs, k)
        nss, nsv = _final(ns_recs, k)
        d = sv - nsv
        print(f"{labels[k]:38}{sv:>10.4f}{nsv:>10.4f}{d:>16.4f}")

    print()
    # The decisive S test: does the bath reduce bond violation beyond the Markov arm?
    _, s_bond = _final(s_recs, "validation/ca_bond_violation_fraction")
    _, ns_bond = _final(ns_recs, "validation/ca_bond_violation_fraction")
    if s_bond < ns_bond - 1e-3:
        verdict = "S HELPS: with-bath bond violation is lower than Markov no-S."
    elif abs(s_bond - ns_bond) <= 1e-3:
        verdict = "S NEUTRAL: bath does not measurably change bond validity (disclose, consider simpler dynamics)."
    else:
        verdict = "S HURTS: with-bath bond violation is higher (bath destabilizes decoded geometry)."
    print(f"Verdict (bond gate): {verdict}")
    print(f"  with-S final ca_bond_violation={s_bond:.4f}  no-S={ns_bond:.4f}")


if __name__ == "__main__":
    main()
