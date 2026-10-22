"""TASK 3 - Temperature generalization for **Model 2** (from-none GLE).

Model 1's temperature matrix lives in ``eval_temperature_generalization.py`` (it
probes the ensemble head's ``[log horizon, (T-300)/150]`` conditioning).  Model 2
is structurally different: temperature enters the GLE kernel through
``temperature_proj`` / the history posterior / the bath, so we probe it through
``evaluation.run_eval_fromnone.evaluate`` with a ``temp_mode`` ablation:

    correct : feed the true trajectory temperature
    wrong   : feed a fixed wrong temperature (default 320 K)
    none    : drop temperature entirely (``temperature_k=None``)

The output mirrors the Model-1 matrix schema (cells keyed
``T{T}__{protein_set}`` with ``correct`` / ``wrong`` / ``none`` blocks) but carries
the from-none rollout metrics (RMSD / skill vs persistence / contact Jaccard),
reported per horizon.

Usage:
    python -m scripts.eval_temperature_generalization_model2 \
        --ckpt artifacts/fromnone/stage_cb1b_v4g/checkpoint_latest.pt \
        --manifest-seen-temp artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
        --manifest-unseen-temp artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl \
        --horizons 1 4 16 64 --max-proteins 20 --n-rollout-reps 4 \
        --out results/temperature_generalization_model2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.run_eval_fromnone import evaluate, parse_args as _rn_parse

MDCATH_TEMPS = (320.0, 348.0, 379.0, 413.0, 450.0)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest-seen-temp", required=True,
                   help="val proteins at T in {320,348,379,413} (val_not450)")
    p.add_argument("--manifest-unseen-temp", default=None,
                   help="held-out T=450 at held-out proteins (val_t450)")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--max-proteins", type=int, default=20)
    p.add_argument("--max-anchors-per-trajectory", type=int, default=4)
    p.add_argument("--n-rollout-reps", type=int, default=4)
    p.add_argument("--wrong-temp", type=float, default=320.0)
    p.add_argument("--modes", nargs="+", default=["correct", "wrong", "none"],
                   choices=["correct", "wrong", "none"],
                   help="Which conditioning ablations to run (skip 'wrong'/'none' "
                        "to halve/third the cost when the T input is known to be inert).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _cell(args, manifest, T):
    """Run correct/wrong/none cells at temperature T; return a metrics block."""
    block = {}
    for mode in args.modes:
        argv = [
            "--ckpt", args.ckpt, "--manifest", manifest,
            "--split", "temp", "--heldout-temperature", str(T),
            "--temp-mode", mode, "--temp-override", str(args.wrong_temp),
            "--horizons", *[str(h) for h in args.horizons],
            "--history-frames", str(args.history_frames),
            "--frame-stride", str(args.frame_stride),
            "--max-residues", str(args.max_residues),
            "--max-proteins", str(args.max_proteins),
            "--max-anchors-per-trajectory", str(args.max_anchors_per_trajectory),
            "--n-rollout-reps", str(args.n_rollout_reps),
            "--device", args.device, "--seed", str(args.seed),
        ]
        ev = evaluate(_rn_parse(argv))
        comp = {}
        for h, d in ev["by_horizon"].items():
            comp[h] = {
                "horizon_ns": d["horizon_ns"], "n_proteins": d["n_proteins"],
                "n_windows": d["n_windows"],
                "rmsd": d["rmsd"], "persist_rmsd": d["persist_rmsd"],
                "skill": d["skill"],
                "contact_jaccard": d["contact_jaccard"],
                "persist_contact_jaccard": d["persist_contact_jaccard"],
            }
        block[mode] = comp
    block["wrong_temp_used"] = args.wrong_temp
    return block


def main(argv=None):
    args = _parse_args(argv)
    results = {"model": "from-none (GLE)", "ckpt": args.ckpt,
               "n_rollout_reps": args.n_rollout_reps, "horizons": args.horizons,
               "modes": args.modes, "wrong_temp": args.wrong_temp, "cells": {}}

    for T in MDCATH_TEMPS:
        unseen = (T == 450.0)
        manifest = args.manifest_unseen_temp if unseen else args.manifest_seen_temp
        if manifest is None:
            continue
        key = "T%.0f__unseen_protein" % T
        print("[temp-gen-m2] %s ..." % key, flush=True)
        cell = _cell(args, manifest, T)
        cell.update({"T": T, "unseen_temperature": unseen,
                     "protein_set": "unseen_protein"})
        results["cells"][key] = cell

    # temperature-response curve: model RMSD vs T under the CORRECT condition.
    resp = {}
    for T in MDCATH_TEMPS:
        key = "T%.0f__unseen_protein" % T
        c = results["cells"].get(key)
        if not c:
            continue
        hh = str(max(int(h) for h in args.horizons))
        resp["T%.0f" % T] = c["correct"][hh]["rmsd"]
    results["temperature_response_rmsd_correct"] = resp
    results["temperature_monotone"] = (
        all(resp["T%.0f" % a] <= resp["T%.0f" % b] + 1e-6
            for a, b in zip(MDCATH_TEMPS[:-1], MDCATH_TEMPS[1:]))
        if len(resp) == len(MDCATH_TEMPS) else None)

    print(json.dumps(results, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print("[temp-gen-m2] wrote %s" % args.out)


if __name__ == "__main__":
    main()
