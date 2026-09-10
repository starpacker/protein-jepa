#!/usr/bin/env python3
"""Orchestrator: run all three baselines and produce a comparison table.

Usage:
    python -m baselines.run_all --subset 5  # Run on 5 proteins for testing
    python -m baselines.run_all             # Run on full validation set
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from baselines.tica_msm import run_tica_msm
from baselines.vampnets import run_vampnets
from baselines.spib import run_spib

# Paths
MANIFEST_DIR = Path("/liuyutian/protein_jepa/artifacts/manifests")
LABEL_DIR = Path("/liuyutian/protein_jepa/artifacts/msm_state_labels")
RESULTS_DIR = Path("/liuyutian/protein_jepa_frame/baselines/results")

VAL_MANIFEST = MANIFEST_DIR / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl"
VAL_LABELS = LABEL_DIR / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered_labels.json"

TRAIN_MANIFEST = MANIFEST_DIR / "mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered.jsonl"
TRAIN_LABELS = LABEL_DIR / "mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered_labels.json"

TEST_MANIFEST = MANIFEST_DIR / "mdcath.deeptime_lcc_pcca_v2.full_lag1.test.filtered.jsonl"
TEST_LABELS = LABEL_DIR / "mdcath.deeptime_lcc_pcca_v2.full_lag1.test.filtered_labels.json"


def print_comparison(results: dict[str, dict]) -> None:
    """Print a comparison table of all baseline methods."""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)

    methods = list(results.keys())
    if not methods:
        print("No results to compare.")
        return

    # VAMP-2 comparison
    print("\n--- VAMP-2 Scores ---")
    header = f"{'Method':<15}"
    for lag in [1, 2, 5, 10, 20]:
        header += f" {'lag=' + str(lag):>16}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<15}"
        vamp2 = results[method].get("vamp2_scores", {})
        for lag in [1, 2, 5, 10, 20]:
            key = f"lag_{lag}"
            if key in vamp2 and vamp2[key].get("mean") is not None:
                m = vamp2[key]["mean"]
                s = vamp2[key]["std"]
                row += f" {m:>8.4f}±{s:.4f}"
            else:
                row += f" {'N/A':>16}"
        print(row)

    # State accuracy comparison
    print("\n--- State Accuracy vs MSM Labels ---")
    header = f"{'Method':<15}  {'Top-1':>8}  {'Top-2':>8}  {'Balanced':>10}  {'Persistence':>12}  {'BrierSkill':>12}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<15}"
        acc = results[method].get("state_accuracy", {})
        if acc and acc.get("top1_accuracy") is not None:
            row += f"  {acc['top1_accuracy']:>8.4f}"
            row += f"  {acc['top2_accuracy']:>8.4f}"
            row += f"  {acc['balanced_accuracy']:>10.4f}"
            row += f"  {acc.get('persistence_accuracy', 0):>12.4f}"
        else:
            row += "  " + "N/A".rjust(8)
            row += "  " + "N/A".rjust(8)
            row += "  " + "N/A".rjust(10)
            row += "  " + "N/A".rjust(12)

        # Brier skill
        brier = results[method].get("brier_skill", {})
        if brier and brier.get("brier_skill") is not None:
            row += f"  {brier['brier_skill']:>12.4f}"
        else:
            row += "  " + "N/A".rjust(12)

        print(row)

    # MSM probe comparison
    print("\n--- MSM Probe Accuracy ---")
    header = f"{'Method':<15}  {'Mean Acc':>10}  {'Baseline':>10}  {'Above/Baseline':>16}  {'N':>5}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<15}"
        probe = results[method].get("msm_probe", {})
        if probe and probe.get("mean_accuracy") is not None:
            row += f"  {probe['mean_accuracy']:>10.4f}"
            row += f"  {probe['mean_baseline']:>10.4f}"
            row += f"  {probe['n_above_baseline']:>3d}/{probe['n_total']:<d}".rjust(16)
            row += f"  {probe['n_total']:>5d}"
        else:
            row += "  " + "N/A".rjust(10)
            row += "  " + "N/A".rjust(10)
            row += "  " + "N/A".rjust(16)
            row += "  " + "N/A".rjust(5)
        print(row)

    # Timing
    print("\n--- Timing ---")
    for method in methods:
        elapsed = results[method].get("time_seconds", 0)
        n_success = results[method].get("n_successful", 0)
        n_total = results[method].get("n_proteins", 0)
        print(f"  {method:<15}: {elapsed:>8.1f}s ({n_success}/{n_total} proteins)")


def main():
    parser = argparse.ArgumentParser(description="Run all baselines and compare")
    parser.add_argument("--subset", type=int, default=None,
                        help="Run on N proteins only (for testing)")
    parser.add_argument("--methods", nargs="+",
                        default=["tica_msm", "vampnet", "spib"],
                        choices=["tica_msm", "vampnet", "spib"],
                        help="Methods to run")
    parser.add_argument("--device", default="cpu",
                        help="Device for PyTorch (cpu, cuda, mps)")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="SPIB entropy regularization beta")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Max epochs for VAMPNet/SPIB training")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Run TICA+MSM
    if "tica_msm" in args.methods:
        all_results["tica_msm"] = run_tica_msm(
            manifest_path=VAL_MANIFEST,
            output_dir=RESULTS_DIR / "tica_msm",
            label_path=VAL_LABELS,
            max_proteins=args.subset,
        )

    # Run VAMPNet
    if "vampnet" in args.methods:
        all_results["vampnet"] = run_vampnets(
            manifest_path=VAL_MANIFEST,
            output_dir=RESULTS_DIR / "vampnet",
            label_path=VAL_LABELS,
            max_proteins=args.subset,
            max_epochs=args.max_epochs,
            device=args.device,
        )

    # Run SPIB
    if "spib" in args.methods:
        all_results["spib"] = run_spib(
            manifest_path=VAL_MANIFEST,
            output_dir=RESULTS_DIR / "spib",
            label_path=VAL_LABELS,
            max_proteins=args.subset,
            max_epochs=args.max_epochs,
            beta=args.beta,
            device=args.device,
        )

    # Print comparison
    print_comparison(all_results)

    # Save combined results
    combined = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "results": {k: v for k, v in all_results.items()},
    }
    out_path = RESULTS_DIR / "combined_results.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined results saved to {out_path}")


if __name__ == "__main__":
    main()