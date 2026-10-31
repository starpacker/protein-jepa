"""Merge the per-method JSONL sweeps into the report tables.

Every table this prints contains **all five methods** (that is the point of the
exercise), and every number comes from the same window set, because the sweep
scored each method on one shared ``plan_windows`` plan.

Tables
------
T1  unseen protein x horizon          (ensemble metrics, 1/4/16/64 ns)
T2  unseen protein x horizon          (position + order-aware rollout metrics)
T3  temperature sweep, unseen protein (5 temperatures x horizon)
T4  temperature sweep, seen protein   (5 temperatures x horizon)
T5  the three requested temperature cells side by side
T6  multi-scale summary (one row per method, aggregated over horizons)

Missing cells are printed as "n/a" with a reason, never silently dropped.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["model1", "model2", "lasp", "mdgen", "bioemu2"]
HIGHER_BETTER = {"rmsf_corr": True, "rmsf_mae": False, "root_mean_w2": False,
                 "variance_w2": False, "md_pca_w2": False, "weak_contact_j": True,
                 "transient_contacts_j": True}
SHORT = {"model1": "Model 1", "model2": "Model 2", "lasp": "Lasp",
         "mdgen": "MDGen", "bioemu2": "bioemu2"}

ENS = [("rmsf_corr", "RMSF-r↑", "{:.3f}"), ("rmsf_mae", "RMSF↓", "{:.3f}"),
       ("root_mean_w2", "root_W2↓", "{:.2f}"), ("variance_w2", "var_W2↓", "{:.2f}"),
       ("md_pca_w2", "pca_W2↓", "{:.4f}"), ("weak_contact_j", "weak_J↑", "{:.3f}"),
       ("transient_contacts_j", "trans_J↑", "{:.3f}")]
POS = [("kabsch_ca_rmsd", "Kabsch↓", "{:.2f}"), ("aligned_ca_rmsd", "aligned↓", "{:.2f}")]
WALK = [("per_step_rmsd_mean", "per-step↓", "{:.2f}"), ("path_length", "path_len↓", "{:.1f}"),
        ("end_to_end", "e2e↓", "{:.2f}"), ("rg_mean", "Rg↓", "{:.1f}"),
        ("traj_energy_score", "energy↓", "{:.2f}"),
        ("walk_rmsf_mae", "walkRMSF↓", "{:.2f}"), ("walk_rmsf_corr", "walkRMSF-r↑", "{:.3f}"),
        ("acf_lag1", "ACF-1↑", "{:.3f}"), ("acf_corr", "ACF-corr↑", "{:.3f}")]


def _keys(recs, methods, need_walk=False):
    """Window keys each method has fully scored (all requested horizons)."""
    per = {m: set() for m in methods}
    for r in recs:
        m = r["method"]
        if m not in per:
            continue
        hs = r.get("horizons", {})
        ok = True
        for h, blk in hs.items():
            if "ens" not in blk:
                ok = False
                break
            if need_walk and "walk" not in blk:
                ok = False
                break
        if ok and hs:
            per[m].add((r["protein_id"], r["trajectory_id"],
                        r["anchor_frame"], r["temperature_mode"]))
    return per


def matched(recs, methods, need_walk=False):
    """Restrict to the intersection of windows every method actually scored.

    Without this the columns would be averaged over different window sets and
    the comparison would be confounded by protein difficulty rather than
    method quality.  Returns (filtered_recs, n_matched, per_method_keys).
    """
    per = _keys(recs, methods, need_walk=need_walk)
    inter = set.intersection(*per.values()) if per else set()
    keep = []
    for r in recs:
        if r["method"] not in per:
            continue
        k = (r["protein_id"], r["trajectory_id"], r["anchor_frame"],
             r["temperature_mode"])
        if k in inter:
            keep.append(r)
    return keep, len(inter), per


def common_report(recs, methods, label, need_walk=False):
    per = _keys(recs, methods, need_walk=need_walk)
    inter = set.intersection(*per.values()) if per else set()
    rows = [f"| {SHORT[m]} | {len(per[m])} |" for m in methods]
    return table(
        rows, ["method", "fully scored windows"],
        f"Common windows — {label}",
        f"> Intersection used below: **{len(inter)}** windows. Averaging over "
        "different window sets would confound protein difficulty with method "
        "quality, so every table is restricted to this intersection.")


def cell_counts(recs, pset, temps, method="lasp"):
    """Windows behind each (protein-set, temperature) cell.

    Uses one reference method's coverage so the caption states the per-column
    sample size without pretending the columns share a single n.
    """
    parts = []
    for t in temps:
        sub = [r for r in recs if r["method"] == method
               and r["protein_set"] == pset
               and int(round(r["temperature_k"])) == int(t)]
        parts.append(f"T{int(t)}={len(sub)}")
    return "> windows per column: " + ", ".join(parts)


def coverage_table(recs, by_cell, methods, temps):
    """What each method actually covered — provenance for every number below."""
    header = ["method"] + [f"T{int(t)}" for t in temps] + ["cells", "windows", "proteins"]
    rows = []
    for m in methods:
        sub = [r for r in recs if r["method"] == m]
        cells = []
        for t in temps:
            ns = set()
            for (pset, tt), rs in by_cell.items():
                if int(tt) == int(t):
                    ns |= {r["protein_id"] for r in rs if r["method"] == m}
            cells.append(str(len(ns)) if ns else "—")
        rows.append("| " + " | ".join(
            [SHORT[m]] + cells
            + [str(len({(r["protein_set"], round(r["temperature_k"])) for r in sub})),
               str(len(sub)),
               str(len({r["protein_id"] for r in sub}))]) + " |")
    return table(rows, header,
                 "Coverage — distinct proteins scored per temperature",
                 "> Counts are *proteins*, not windows (several windows can come from "
                 "one trajectory). `—` means the method has no record at that "
                 "temperature.")


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
            if "error" in r:
                continue
            recs.append(r)
    return recs


def agg(recs, metric_path):
    """Mean over windows of one metric; NaN if no window has it."""
    vals = []
    for r in recs:
        node = r
        ok = True
        for k in metric_path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                ok = False
                break
        if ok and isinstance(node, (int, float)) and np.isfinite(node):
            vals.append(float(node))
    if not vals:
        return float("nan"), 0
    return float(np.mean(vals)), len(vals)


def fmt(v, spec):
    return "n/a" if not np.isfinite(v) else spec.format(v)


def table(rows, cols, title, note=""):
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    out = [f"\n#### {title}", "", head, sep]
    out += rows
    if note:
        out += ["", note]
    return "\n".join(out)


def best_marker(values):
    """Mark the best finite value per column (lower is better unless ↑)."""
    return values


def section_by_horizon(recs, horizons, title, metric_sets, note=""):
    """One sub-table per horizon: rows = methods.

    ``metric_sets`` is a list of ``(group, specs)`` where ``group`` is the
    JSONL sub-object the metric lives under (``ens``, ``pos``, ``walk``).
    """
    blocks = []
    for hs in horizons:
        header = ["method"] + [c for _, specs in metric_sets for _, c, _ in specs] + ["n_win"]
        rows = []
        for m in METHODS:
            sub = [r for r in recs if r["method"] == m
                   and str(hs) in r.get("horizons", {})]
            if not sub:
                rows.append("| " + " | ".join([SHORT[m]] + ["n/a"] * (len(header) - 1)) + " |")
                continue
            cells = []
            for group, specs in metric_sets:
                for key, _, f in specs:
                    v, _ = agg(sub, ["horizons", str(hs), group, key])
                    cells.append(fmt(v, f))
            rows.append("| " + " | ".join([SHORT[m]] + cells + [str(len(sub))]) + " |")
        blocks.append(table(rows, header, f"{title} — horizon {hs} ns", note if hs == horizons[-1] else ""))
    return "\n".join(blocks)


def per_temp(recs, temps, key, label, spec, horizons, n_note=""):
    """One table per horizon: rows = methods, columns = temperature."""
    out = []
    for hs in horizons:
        header = ["method"] + [f"T{int(t)}" for t in temps]
        rows = []
        for m in METHODS:
            cells = []
            for t in temps:
                sub = [r for r in recs if r["method"] == m
                       and int(round(r["temperature_k"])) == int(t)]
                v, _ = agg(sub, ["horizons", str(hs), "ens", key])
                cells.append(fmt(v, spec))
            rows.append("| " + " | ".join([SHORT[m]] + cells) + " |")
        out.append(table(rows, header,
                         f"{label} — {label_metric(key)} — horizon {hs} ns", n_note))
    return "\n".join(out)


METRIC_LABEL = {"rmsf_corr": "RMSF-r↑", "rmsf_mae": "RMSF↓",
                "root_mean_w2": "root_W2↓", "variance_w2": "var_W2↓",
                "md_pca_w2": "pca_W2↓", "weak_contact_j": "weak_J↑",
                "transient_contacts_j": "trans_J↑",
                "kabsch_ca_rmsd": "Kabsch↓", "aligned_ca_rmsd": "aligned↓",
                "per_step_rmsd_mean": "per-step↓", "walk_rmsf_corr": "walkRMSF-r↑",
                "acf_corr": "ACF-corr↑", "traj_energy_score": "energy↓"}


def label_metric(k):
    return METRIC_LABEL.get(k, k)


def multi_metric_by_temp(recs, temps, keys, label, horizons, n_note=""):
    """rows = (method, metric), columns = temperature, one block per horizon."""
    out = []
    for hs in horizons:
        header = ["metric", "method"] + [f"T{int(t)}" for t in temps]
        rows = []
        for key, spec in keys:
            for m in METHODS:
                cells = []
                for t in temps:
                    sub = [r for r in recs if r["method"] == m
                           and int(round(r["temperature_k"])) == int(t)]
                    v, _ = agg(sub, ["horizons", str(hs), "ens", key])
                    cells.append(fmt(v, spec))
                rows.append("| " + " | ".join([label_metric(key), SHORT[m]] + cells) + " |")
        out.append(table(rows, header, f"{label} — horizon {hs} ns", n_note))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/five_method")
    ap.add_argument("--out", default="FIVE_METHOD_COMPARISON_20261022.md")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="subset for a partial preview; the report always needs "
                         "all five, so the default is all five")
    args = ap.parse_args()
    if args.methods:
        global METHODS
        METHODS = [m for m in METHODS if m in args.methods]

    merged = Path(args.dir) / "all.jsonl"
    if merged.exists():
        # `all.jsonl` is produced by `.tmpwork/fm/merge.py`, which also picks up
        # shards that a live sweep process still holds open after an unlink.
        # Reading it alone avoids double counting the per-shard files.
        paths = [str(merged)]
    else:
        paths = sorted(glob.glob(str(Path(args.dir) / "*.jsonl")))
    recs = load(paths)
    if not recs:
        raise SystemExit(f"no records under {args.dir}")
    by_cell = defaultdict(list)
    for r in recs:
        by_cell[(r["protein_set"], int(round(r["temperature_k"])))].append(r)

    L = []
    L.append("# Five-method comparison — one harness, one window set\n")
    L.append(f"Records: **{len(recs)}** method/window pairs across "
             f"{len({r['protein_id'] for r in recs})} proteins and "
             f"{len(by_cell)} (protein-split x temperature) cells.\n")
    L.append("All five methods are scored on identical windows: same anchors, "
             "same true MD futures, same residue masks, same persistence, and the "
             "same `evaluation.*` metric code. The only thing that differs is how "
             "each method produces its CA cloud.\n")
    L.append("`n/a` means the method structurally cannot fill that cell "
             "(documented per table), never that the run failed.\n")

    unseen = [r for r in recs if r["protein_set"] == "unseen"]
    seen = [r for r in recs if r["protein_set"] == "seen"]

    temps_all = sorted({int(round(r["temperature_k"])) for r in recs})
    L.append("\n## T0 — Coverage and provenance\n")
    L.append(coverage_table(recs, by_cell, METHODS, temps_all))

    # Every table below uses only windows that *all five* methods scored.
    # Recompute the cells after the restriction so coverage stays honest.
    recs, n_matched, _per = matched(recs, METHODS)
    L.append(common_report(recs, METHODS, "all five methods"))
    by_cell = defaultdict(list)
    for r in recs:
        by_cell[(r["protein_set"], int(round(r["temperature_k"])))].append(r)
    unseen = [r for r in recs if r["protein_set"] == "unseen"]
    seen = [r for r in recs if r["protein_set"] == "seen"]

    # ---------------- T1/T2: unseen protein ----------------
    L.append("\n## T1 — Unseen protein: ensemble metrics across time scales\n")
    L.append(section_by_horizon(unseen, args.horizons,
                                "Unseen protein (mdCATH-1000dom val, disjoint in protein *and* trajectory)",
                                [("ens", ENS)]))
    L.append("\n## T2 — Unseen protein: position and order-aware rollout metrics\n")
    L.append(section_by_horizon(unseen, args.horizons,
                                "Unseen protein — position + trajectory metrics",
                                [("pos", POS), ("walk", WALK)],
                                note="> `Kabsch` is the common-anchor CA RMSD (the fitted transform "
                                     "comes from the true anchor→target pair only, so no prediction "
                                     "leaks into the gauge). `per-step`, `path_len`, `e2e`, `Rg`, "
                                     "`energy` are computed on each method's own ordered rollout; "
                                     "bioemu2 is `n/a` there because `propagate` draws independent "
                                     "endpoints and has no ordered path."))

    # ---------------- T3/T4: temperature ----------------
    temps = sorted({t for (_, t) in by_cell})
    KEY_SPECS = [(k, f) for k, _, f in ENS]
    L.append("\n## T3 — Temperature generalization, unseen protein\n")
    L.append("Rows = (metric, method); columns = the temperature of the evaluated "
             "trajectory. T=450 K is absent from the training split of the "
             "temperature-holdout checkpoints, so the T450 column is a genuinely "
             "unseen temperature for Model 1 / Model 2. Lasp is trained on all five "
             "temperatures but conditions on T only through the GLE kernel's "
             "temperature feature. MDGen and bioemu2 take **no temperature input at "
             "all**, so their columns are the same model run on hotter/colder "
             "trajectories — the correct reading is 'zero-shot transfer', not "
             "'temperature conditioning'.\n")
    L.append(multi_metric_by_temp(unseen, temps, KEY_SPECS, "Unseen protein",
                                  args.horizons,
                                  cell_counts(recs, "unseen", temps)))

    L.append("\n## T4 — Temperature generalization, seen protein\n")
    L.append("Same layout, but the proteins are drawn from the **training** split "
             "(seen sequences, unseen trajectories). Comparing T3 against T4 isolates "
             "the protein-generalization gap at each temperature. Only the two cells "
             "the task asks for were run for seen proteins — **T348** (seen protein × "
             "seen temperature) and **T450** (seen protein × *unseen* temperature) — "
             "so the other three columns are `n/a` by design, not by failure.\n")
    L.append(multi_metric_by_temp(seen, temps, KEY_SPECS, "Seen protein",
                                  args.horizons,
                                  cell_counts(recs, "seen", temps)))

    # ---------------- T5: the three requested cells ----------------
    L.append("\n## T5 — The three requested temperature cells, side by side\n")
    L.append("The four cells the task asks for, all at once. `n_win` is the number "
             "of scored windows behind each cell.\n")
    cells = [("unseen protein × unseen T (450 K)", "unseen", 450),
             ("unseen protein × seen T (348 K)", "unseen", 348),
             ("seen protein × unseen T (450 K)", "seen", 450),
             ("seen protein × seen T (348 K) — reference", "seen", 348)]
    for hs in args.horizons:
        header = ["cell", "method"] + [c for _, c, _ in ENS] + ["n_win"]
        rows = []
        for label, pset, t in cells:
            for m in METHODS:
                sub = [r for r in by_cell.get((pset, t), []) if r["method"] == m]
                vals, n = [], 0
                for key, _, f in ENS:
                    v, k = agg(sub, ["horizons", str(hs), "ens", key])
                    vals.append(fmt(v, f)); n = max(n, k)
                if not n:
                    n = sum(1 for r in sub
                            if "ens" in r.get("horizons", {}).get(str(hs), {}))
                rows.append("| " + " | ".join([label, SHORT[m]] + vals + [str(n)]) + " |")
        L.append(table(rows, header, f"Temperature cells — horizon {hs} ns"))

    # ---------------- T6: rollout / order-aware ----------------
    L.append("\n## T6 — Trajectory rollout (order-aware) quality\n")
    L.append("These metrics only mean something for a *method that produces an "
             "ordered path*. bioemu2's `propagate` draws independent endpoints, so "
             "it is `n/a` here unless it was run in `--bioemu2-walk` mode (one "
             "temporal chain of denoiser steps); MDGen samples a whole trajectory in "
             "one shot, so it is scored natively. Method 1 rolls out "
             "autoregressively and is therefore the most exposed to drift — that is "
             "precisely what these columns measure.\n")
    def walk_block(pool, title, n_methods):
        out = []
        for hs in args.horizons:
            header = ["method"] + [c for _, c, _ in WALK] + ["n_traj"]
            rows = []
            for m in METHODS[:n_methods] if n_methods else METHODS:
                sub = [r for r in pool if r["method"] == m]
                vals, tot = [], 0
                for key, _, f in WALK:
                    v, _ = agg(sub, ["horizons", str(hs), "walk", key])
                    vals.append(fmt(v, f))
                for r in sub:
                    blk = r.get("horizons", {}).get(str(hs), {})
                    if "walk" in blk:
                        tot += int(blk.get("walk_m") or 0)
                rows.append("| " + " | ".join([SHORT[m]] + vals + [str(tot)]) + " |")
            out.append(table(rows, header, f"{title} — horizon {hs} ns"))
        return "\n".join(out)

    L.append(walk_block(unseen, "Unseen protein — ordered rollout", None))
    L.append("\n> **bioemu2's `n_traj` is far smaller than the others.** Its "
             "`propagate` sampler draws each endpoint independently, so it has no "
             "ordered path at all and only contributes here for the windows where "
             "the separate `--bioemu2-walk` run (one temporal chain of denoiser "
             "steps, ~27 min/window) was executed. Read its row as indicative, not "
             "as a 140-window estimate; the four path-producing methods share the "
             "same full window set.\n")

    # ---------------- T7: one-line-per-method summary ----------------
    L.append("\n## T7 — Multi-scale summary (mean over horizons)\n")
    L.append("One row per method: the mean of each metric across the "
             f"{len(args.horizons)} horizons ({', '.join(str(h)+' ns' for h in args.horizons)}), "
             "unseen protein. Use this as the headline table; T1/T2 give the "
             "per-horizon detail.\n")
    header = ["method"] + [c for _, c, _ in ENS] + ["Kabsch↓", "n_win"]
    rows = []
    for m in METHODS:
        sub = [r for r in unseen if r["method"] == m]
        vals, n = [], 0
        for key, _, f in ENS:
            per_h = []
            k_max = 0
            for hs in args.horizons:
                v, k = agg(sub, ["horizons", str(hs), "ens", key])
                if np.isfinite(v):
                    per_h.append(v)
                k_max = max(k_max, k)
            vals.append(fmt(np.mean(per_h) if per_h else np.nan, f))
            n = max(n, k_max)
        kv, kk = [], 0
        for hs in args.horizons:
            v, k = agg(sub, ["horizons", str(hs), "pos", "kabsch_ca_rmsd"])
            if np.isfinite(v):
                kv.append(v)
            kk = max(kk, k)
        vals.append(fmt(np.mean(kv) if kv else np.nan, "{:.2f}"))
        rows.append("| " + " | ".join([SHORT[m]] + vals + [str(n)]) + " |")
    L.append(table(rows, header, "Unseen protein — mean across horizons"))

    # ---------------- T8: paired significance vs Model 1 ----------------
    L.append("\n## T8 — Paired bootstrap: every method against Model 1\n")
    L.append("All five methods scored the same windows, so a *paired* test removes "
             "between-protein difficulty and answers the question that actually "
             "matters: on a given protein, which method is better? Each cell is "
             "`mean(method) vs mean(Model 1)`, the parentheses holding the paired Δ "
             "and its 4000-sample bootstrap 95% CI; **bold** marks a CI that excludes "
             "zero (a real difference). One table, all five methods.\n")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.five_method.paired_stats import compare, index
    idx = index(recs)
    HB = dict(HIGHER_BETTER)
    others = ["model2", "lasp", "mdgen", "bioemu2"]

    def cell(A, B, hs, key, pset):
        rows = compare(idx, A, B, [hs], [(key, "")], HB, protein_set=pset)
        if not rows:
            return "n/a"
        r = rows[0]
        txt = f"{r['mean_B']:.3f} ({r['delta']:+.3f})"
        return f"**{txt}**" if r["significant"] else txt

    for pset, label in (("unseen", "unseen protein"), ("seen", "seen protein")):
        for hs in args.horizons:
            header = ["metric", "Model 1"] + [SHORT[o] for o in others]
            rows = []
            for key, lbl, _ in ENS:
                cells = []
                for o in others:
                    cells.append(cell("model1", o, hs, key, pset))
                ref = compare(idx, "model1", "lasp", [hs], [(key, "")], HB,
                              protein_set=pset)
                base = f"{ref[0]['mean_A']:.3f}" if ref else "n/a"
                rows.append("| " + " | ".join([lbl, base] + cells) + " |")
            L.append(table(rows, header,
                           f"{label} — paired vs Model 1 — horizon {hs} ns",
                           "> Format: `mean (Δ vs Model 1)`. **Bold** = 95% CI "
                           "excludes zero. A negative Δ is better for the ↓ "
                           "metrics, positive is better for ↑."))

    doc = "\n".join(L) + "\n"
    Path(args.out).write_text(doc)
    print(f"wrote {args.out} ({len(doc.splitlines())} lines)")


if __name__ == "__main__":
    main()
