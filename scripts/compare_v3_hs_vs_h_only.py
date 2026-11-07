#!/usr/bin/env python3
"""H+S vs H-only verdict (Task 4, corrected): v3 SOTA vs v3 recipe with bath_dim=0.

H+S arm = v3 SOTA (artifacts/lasp_v3_velocity, seed 42), the accepted baseline.
H-only arm = the SAME v3 recipe with --bath-dim 0 (artifacts/lasp_v3_h_only_20261107).

Both arms are read through the v3 eval entries:
  * ensemble   -> scripts/eval_lasp_final.py  (by_horizon[h] -> {metric: {lasp, model1, persist, ...}})
  * trajectory -> scripts/eval_ca_ordered_paths_v3.py (by_horizon[h] -> {means: {forecast: {...}, reference_MD: {...}}})

Honest accounting: v3's S is inert (coupling_scale=1e-3, bath_var/R ~ 1e-8), so
a "no difference" result is EXPECTED and is reported plainly — it means v3's
*as-shipped* S adds nothing, not that a correctly-scaled S is useless.
"""
import argparse
import json
from pathlib import Path

ENSEMBLE_HIGHER_BETTER = {"rmsf_corr", "weak_contact_j", "transient_contacts_j"}
ENSEMBLE_LOWER_BETTER = {"rmsf_mae", "root_mean_w2", "variance_w2", "md_pca_w2"}


def load_ensemble(path):
    """eval_lasp_final output: by_horizon[str(hz_ns)] -> {metric: {lasp, ...}}."""
    d = json.loads(Path(path).read_text())
    out = {}
    for hz, rows in d.get("by_horizon", {}).items():
        out[float(hz)] = {m: rows[m]["lasp"] for m in rows if "lasp" in rows[m]}
    return out


def load_traj(path):
    """eval_ca_ordered_paths_v3 output: by_horizon[str(h)] -> means{forecast, reference_MD}."""
    d = json.loads(Path(path).read_text())
    out = {}
    for h, v in d.get("by_horizon", {}).items():
        means = v.get("means", {})
        f = means.get("forecast", {})
        r = means.get("reference_MD", {})
        out[int(h)] = dict(
            path_length=f.get("path_length", {}).get("mean"),
            acf_lag1=f.get("distance_acf_lag1", {}).get("mean"),
            ref_path_length=r.get("path_length", {}).get("mean"),
            ref_acf_lag1=r.get("reference_distance_acf_lag1", {}).get("mean"),
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hs-ensemble", default="results/todo_1107/e0_v3.ensemble.json")
    p.add_argument("--hs-ordered", default="results/todo_1107/v3_ordered_paths.json")
    p.add_argument("--ho-ensemble", required=True,
                   help="v3 H-only ensemble json (eval_lasp_final output)")
    p.add_argument("--ho-ordered", required=True,
                   help="v3 H-only ordered json (eval_ca_ordered_paths_v3 output)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    hs_e = load_ensemble(a.hs_ensemble)
    ho_e = load_ensemble(a.ho_ensemble)

    print("=== ENSEMBLE: H+S (v3 SOTA) vs H-only (v3 + bath_dim=0) ===")
    print(f"{'hz':>5} {'metric':22} {'H+S':>9} {'H-only':>9} {'Δ(H+S−H-only)':>14} {'S helps?':>9}")
    print("-" * 68)
    verdict = {}
    for hz in sorted(hs_e.keys()):
        for metric in list(ENSEMBLE_HIGHER_BETTER) + list(ENSEMBLE_LOWER_BETTER):
            hs_v = hs_e[hz].get(metric)
            ho_v = ho_e.get(hz, {}).get(metric)
            if hs_v is None or ho_v is None:
                continue
            higher = metric in ENSEMBLE_HIGHER_BETTER
            delta = hs_v - ho_v
            helps = (delta > 0) if higher else (delta < 0)
            print(f"{hz:>5} {metric:22} {hs_v:>9.4f} {ho_v:>9.4f} {delta:>+14.4f} {'YES' if helps else 'no':>9}")
            verdict[(hz, metric)] = helps
    n_yes = sum(verdict.values())
    n_all = len(verdict)
    print(f"\nS helps on {n_yes}/{n_all} ensemble cells "
          f"({'H+S > H' if n_yes == n_all and n_all else 'H+S == H or worse on some cells'})")

    hs_t = load_traj(a.hs_ordered)
    ho_t = load_traj(a.ho_ordered)
    print("\n=== TRAJECTORY: ACF Error / Path Error (lower better) ===")
    print(f"{'hz':>5} {'metric':20} {'H+S':>9} {'H-only':>9} {'S helps?':>9}")
    print("-" * 56)
    traj_verdict = {}
    for h in sorted(hs_t.keys()):
        # Path Error = |forecast path_length - reference path_length|
        for metric, f_key, r_key in (("path_error", "path_length", "ref_path_length"),
                                     ("acf_error", "acf_lag1", "ref_acf_lag1")):
            hs_f, hs_r = hs_t[h].get(f_key), hs_t[h].get(r_key)
            ho_f, ho_r = ho_t.get(h, {}).get(f_key), ho_t.get(h, {}).get(r_key)
            if None in (hs_f, hs_r, ho_f, ho_r):
                continue
            hs_err = abs(hs_f - hs_r)
            ho_err = abs(ho_f - ho_r)
            helps = hs_err < ho_err
            print(f"{h:>5} {metric:20} {hs_err:>9.4f} {ho_err:>9.4f} {'YES' if helps else 'no':>9}")
            traj_verdict[(h, metric)] = helps
    n_t = sum(traj_verdict.values())
    n_ta = len(traj_verdict)
    print(f"\nS helps on {n_t}/{n_ta} trajectory cells")

    if a.out:
        Path(a.out).write_text(json.dumps(dict(ensemble=verdict, trajectory=traj_verdict), indent=2, default=str))


if __name__ == "__main__":
    main()
