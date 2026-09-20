#!/usr/bin/env python3
"""Unified launcher for reviewer experiments.

Usage:
  python3 scripts/run_reviewer_experiment.py S1 --gle-data-dir artifacts/reviewer_experiments/synthetic_gle
  python3 scripts/run_reviewer_experiment.py R1 --checkpoint path/to/checkpoint.pt --config path/to/config.yaml
  python3 scripts/run_reviewer_experiment.py M1 --checkpoint path/to/checkpoint.pt --config path/to/config.json --manifest path/to/manifest.jsonl --output path/to/m1.json
  python3 scripts/run_reviewer_experiment.py K1 --labels path/to/fastfolding_labels.jsonl --predictions path/to/jepa_predictions.jsonl --output path/to/k1.json
  python3 scripts/run_reviewer_experiment.py ALL  # run all experiments
"""

from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "reviewer_experiments"

EXPERIMENTS = {
    "P0": {
        "script": "audit_lspwm_protocol.py",
        "output_dir": ARTIFACTS_DIR / "protocol",
        "description": "Protocol audit (existing script)",
    },
    "S1_gen": {
        "script": "generate_synthetic_gle.py",
        "output_dir": ARTIFACTS_DIR / "synthetic_gle",
        "description": "Generate synthetic GLE data",
    },
    "S1": {
        "script": "evaluate_synthetic_gle.py",
        "output_dir": ARTIFACTS_DIR / "synthetic_gle",
        "description": "Synthetic GLE recovery evaluation",
    },
    "R1": {
        "script": "evaluate_history_gap.py",
        "output_dir": ARTIFACTS_DIR / "representation_closure",
        "description": "History Gap representation closure",
    },
    "M1": {
        "script": "evaluate_m1_strict.py",
        "output_dir": ARTIFACTS_DIR / "thermodynamic_closure",
        "description": "Strict temporal-holdout thermodynamic closure ablation",
    },
    "K1": {
        "script": "evaluate_k1_fastfolding.py",
        "output_dir": ARTIFACTS_DIR / "protein_kinetics",
        "description": "FastFolding scaffold-disjoint protein kinetics evaluation",
    },
}


def run_experiment(exp_id: str, extra_args: list[str] | None = None) -> int:
    if exp_id not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_id}")
        print(f"Available: {list(EXPERIMENTS.keys())}")
        return 1

    info = EXPERIMENTS[exp_id]
    script_path = SCRIPTS_DIR / info["script"]
    if not script_path.exists():
        print(f"  [SKIP] Script not found: {script_path}")
        return 1

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"Running {exp_id}: {info['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0

    print(f"\n  Completed in {elapsed:.1f}s with exit code {result.returncode}")
    return result.returncode


def main():
    p = argparse.ArgumentParser(description="Unified reviewer experiment launcher")
    p.add_argument("experiment", nargs="?",
                   help="Experiment ID (S1, R1, M1, K1, S1_gen, P0) or ALL")
    p.add_argument("extra_args", nargs=argparse.REMAINDER,
                   help="Extra arguments passed to the experiment script")
    args = p.parse_args()

    if args.experiment is None:
        print("Available experiments:")
        for eid, info in sorted(EXPERIMENTS.items()):
            print(f"  {eid:6s}  {info['description']}")
        print("\nUsage: python3 run_reviewer_experiment.py <EXPERIMENT> [extra args]")
        return

    if args.experiment == "ALL":
        exit_codes = {}
        for eid in ["P0", "S1_gen", "S1", "R1", "M1", "K1"]:
            exit_codes[eid] = run_experiment(eid)
        failed = [eid for eid, code in exit_codes.items() if code != 0]
        if failed:
            print(f"\nFailed experiments: {failed}")
        else:
            print(f"\nAll experiments completed successfully.")
        return

    sys.exit(run_experiment(args.experiment, args.extra_args))


if __name__ == "__main__":
    main()
