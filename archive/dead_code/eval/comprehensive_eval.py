#!/usr/bin/env python3
"""Comprehensive evaluation for trained LSP-WM models.

Runs:
  1. LSP-WM internal metrics (endpoint skill, force matching, energy, Koopman, diagnostics)
  2. FastFolding benchmark (state predictions on fast-folding proteins)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


def run_lspwm_eval(checkpoint: Path, config: Path, output_dir: Path,
                    max_batches: int, bandwidth: float, device: str,
                    manifest: Path | None, batch_size: int, seed: int) -> dict:
    """Run the 5-module LSP-WM internal evaluation."""
    print("\n" + "=" * 60)
    print("LSP-WM INTERNAL EVALUATION")
    print("=" * 60, flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from protein_jepa_frame.config import load_config
    from protein_jepa_frame.data.dataset import build_dataset
    from protein_jepa_frame.data.collate import collate_paired_windows
    from protein_jepa_frame.data.sampler import HierarchicalClipSampler
    from torch.utils.data import DataLoader
    from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
    from protein_jepa_frame.eval.lspwm_eval import (
        eval_endpoint_skill, eval_force_matching,
        eval_energy_conservation, eval_koopman, eval_lspwm_components,
    )

    torch.manual_seed(seed)
    cfg = load_config(config)
    data_config = replace(
        cfg.data,
        manifest_path=str(manifest) if manifest is not None else cfg.data.manifest_path,
        batch_size=batch_size,
        num_workers=0,
        paired_all_horizons=True,
        sampling_mode="hierarchical",
        training_mode="paired_window",
    )
    model = ProteinJEPA(cfg.model).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", ckpt))
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"  Missing keys: {incompatible.missing_keys}")
    model.eval()

    dataset = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=max_batches * batch_size,
        seed=seed,
        dataset_weights=data_config.dataset_weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )

    results = {}
    for name, func, kwargs in [
        ("endpoint_skill", eval_endpoint_skill, {"max_batches": max_batches}),
        ("force_matching", eval_force_matching, {"max_batches": max_batches // 2, "bandwidth": bandwidth}),
        ("koopman", eval_koopman, {"max_batches": max_batches}),
        ("energy_conservation", eval_energy_conservation, {"max_batches": max(max_batches // 5, 2)}),
        ("components", eval_lspwm_components, {}),
    ]:
        print(f"\n[{name}]...", flush=True)
        try:
            if name == "components":
                r = func(model, device)
            else:
                r = func(model, loader, device, **kwargs)
            results[name] = r
            print(f"  OK", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            results[name] = {"error": str(e)}

    out_path = output_dir / "lspwm_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_fastfolding(checkpoint: Path, output_dir: Path,
                    labels: Path, device: str,
                    temperature_k: float | None = None) -> dict:
    """Run FastFolding benchmark via the generate+evaluate pipeline."""
    print("\n" + "=" * 60)
    print("FASTFOLDING BENCHMARK")
    print("=" * 60, flush=True)

    predictions_path = output_dir / "fastfolding_predictions.jsonl"
    results_path = output_dir / "fastfolding_results.json"
    script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "generate_fastfolding_predictions.py"

    if not script_path.exists():
        return {"error": f"Script not found: {script_path}"}

    # Step 1: Generate predictions
    print(f"\n[1/2] Generating FastFolding predictions...", flush=True)
    cmd = [
        sys.executable, str(script_path),
        "--checkpoint", str(checkpoint),
        "--labels", str(labels),
        "--output", str(predictions_path),
        "--context-frames", "8",
        "--horizons", "1", "4", "8", "16",
        "--state-names", "unfolded", "transition", "folded",
        "--batch-size", "4",
        "--device", device,
    ]
    if temperature_k is not None:
        cmd.extend(["--temperature-k", str(temperature_k)])

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    env = os.environ.copy()
    repo_in_path = repo_root in env.get("PYTHONPATH", "")
    if not repo_in_path:
        env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}"

    print(f"  Command: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        print(f"  ERROR generating predictions: {result.stderr[:500]}", flush=True)
        return {"error": result.stderr[:1000]}
    print(f"  Predictions saved to {predictions_path}", flush=True)

    # Step 2: Evaluate
    print(f"\n[2/2] Evaluating FastFolding predictions...", flush=True)
    eval_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluate_dynamics_benchmarks.py"
    cmd = [
        sys.executable, str(eval_script),
        "fastfolding",
        "--labels", str(labels),
        "--predictions", str(predictions_path),
        "--output", str(results_path),
        "--context-frames", "8",
        "--horizons", "1", "4", "8", "16",
        "--state-names", "unfolded", "transition", "folded",
    ]
    print(f"  Command: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        print(f"  ERROR evaluating: {result.stderr[:500]}", flush=True)
        return {"error": result.stderr[:1000]}

    try:
        with open(results_path) as f:
            ff_results = json.load(f)
        print(f"  FastFolding results saved to {results_path}", flush=True)
        return ff_results
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ERROR reading results: {e}", flush=True)
        return {"error": str(e)}
    print(f"\nLSP-WM results saved to {out_path}")
def print_summary(lspwm_results: dict, ff_results: dict):
    """Print a concise summary of all evaluation results."""
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    es = lspwm_results.get("endpoint_skill", {})
    if "deterministic_skill" in es:
        print(f"  Deterministic skill:     {es['deterministic_skill']:.4f}")

    fm = lspwm_results.get("force_matching", {})
    if "force_cosine" in fm:
        print(f"  Force cosine:            {fm['force_cosine']:.4f}")
    if "force_mse" in fm:
        print(f"  Force MSE:               {fm['force_mse']:.6f}")

    kp = lspwm_results.get("koopman", {})
    if "vamp2_score" in kp:
        print(f"  VAMP-2 score:            {kp['vamp2_score']:.2f}")

    ec = lspwm_results.get("energy_conservation", {})
    if "mean_circulation" in ec:
        print(f"  Mean circulation:        {ec['mean_circulation']:.4f}")

    comp = lspwm_results.get("components", {})
    if "energy_healthy" in comp:
        print(f"  Energy healthy:          {comp['energy_healthy']}")
        print(f"  Force healthy:           {comp['force_healthy']}")

    # FastFolding summary
    if "test" in ff_results and "by_horizon" in ff_results["test"]:
        print(f"\n  --- FastFolding ---")
        for h_name, h_data in sorted(ff_results["test"]["by_horizon"].items()):
            metrics = h_data.get("metrics", h_data)
            comm = metrics.get("committor_proxy", {})
            if comm:
                print(f"  FF h={h_name}: acc={comm.get('accuracy','?'):.3f} log_loss={comm.get('log_loss','?'):.3f}")
            tec = metrics.get("transition_operator", {})
            if tec:
                print(f"  FF h={h_name}: station_l1={tec.get('stationary_l1','?'):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LSP-WM evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Explicit evaluation manifest. Required for held-out protocol runs.",
    )
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fastfolding-labels", type=str,
                        default="/tmp/lspwm_fastfolding_labels.jsonl")
    parser.add_argument("--fastfolding-temperature-k", type=float, default=None)
    parser.add_argument("--skip-fastfolding", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    checkpoint = Path(args.checkpoint)
    config = Path(args.config)

    # Run LSP-WM internal evaluation
    lspwm_results = run_lspwm_eval(
        checkpoint, config, output_dir,
        max_batches=args.max_batches,
        bandwidth=args.bandwidth,
        device=device,
        manifest=Path(args.manifest) if args.manifest else None,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Run FastFolding benchmark
    ff_results = {}
    if not args.skip_fastfolding:
        labels_path = Path(args.fastfolding_labels)
        if labels_path.exists():
            ff_results = run_fastfolding(
                checkpoint, output_dir, labels_path,
                device=device,
                temperature_k=args.fastfolding_temperature_k,
            )
        else:
            print(f"\nFastFolding labels not found at {labels_path}, skipping.")

    print_summary(lspwm_results, ff_results)

    combined = {
        "checkpoint": str(checkpoint),
        "protocol": {
            "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
            "config": str(config.resolve()),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "bandwidth": args.bandwidth,
            "heldout_manifest_explicit": args.manifest is not None,
        },
        "lspwm_eval": lspwm_results,
        "fastfolding": ff_results,
    }
    combined_path = output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")
    print("Evaluation complete!", flush=True)


if __name__ == "__main__":
    main()
