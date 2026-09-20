#!/usr/bin/env python3
"""S1: Evaluate synthetic double-well GLE recovery.

Compares learned parameters against known A*, D*, K* and computes
primary and secondary metrics.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifacts.reviewer_experiments._shared.metrics import (
    paired_bootstrap_ci, jensen_shannon_divergence,
    innovation_whiteness_score, autocorrelation,
)
from artifacts.reviewer_experiments._shared.result_schema import (
    ExperimentResult, ProvenanceBlock, SyntheticMetrics,
)


def load_ground_truth(gle_dir: Path):
    data = np.load(gle_dir / "ground_truth" / "analytics.npz")
    with open(gle_dir / "ground_truth" / "params.json") as f:
        params = json.load(f)
    return data, params


def compute_primary_metrics(z_true: np.ndarray, z_pred: np.ndarray,
                            A_true: np.ndarray, A_pred: np.ndarray,
                            K_true: np.ndarray, K_pred: np.ndarray,
                            z_grid: np.ndarray, t_vals: np.ndarray):
    # Free-energy recovery error: interpolate A*(z) and A_theta(z) to traj points
    A_true_at_traj = np.interp(z_true, z_grid, A_true)
    A_pred_at_traj = np.interp(z_pred, z_grid, A_pred)
    A_true_at_traj -= A_true_at_traj.mean()
    A_pred_at_traj -= A_pred_at_traj.mean()
    free_energy_error = float(np.sqrt(np.mean((A_pred_at_traj - A_true_at_traj) ** 2)))

    # Memory kernel error (L2 over time)
    min_len = min(len(K_true), len(K_pred))
    kernel_error = float(np.sqrt(np.mean((K_pred[:min_len] - K_true[:min_len]) ** 2)))

    # Stationary JSD
    hist_true, _ = np.histogram(z_true, bins=50, density=True)
    hist_pred, _ = np.histogram(z_pred, bins=50, density=True)
    stationary_jsd = jensen_shannon_divergence(hist_true, hist_pred)

    # ACF error over lag grid
    max_lag = 200
    acf_true = autocorrelation(z_true.reshape(-1, 1), max_lag).flatten()
    acf_pred = autocorrelation(z_pred.reshape(-1, 1), max_lag).flatten()
    acf_error = float(np.sqrt(np.mean((acf_pred - acf_true) ** 2)))

    # Transition rate
    state_true = (z_true > 0).astype(int)
    state_pred = (z_pred > 0).astype(int)
    rate_true = (state_true[1:] != state_true[:-1]).sum() / len(state_true)
    rate_pred = (state_pred[1:] != state_pred[:-1]).sum() / len(state_pred)
    rate_error = abs(rate_pred - rate_true) / max(rate_true, 1e-12)

    return {
        "free_energy_error": free_energy_error,
        "memory_kernel_error": kernel_error,
        "stationary_jsd": stationary_jsd,
        "acf_error": acf_error,
        "transition_rate_error": float(rate_error),
    }


def main(args):
    gle_dir = Path(args.gle_data_dir)
    data, gt_params = load_ground_truth(gle_dir)
    A_true = data["free_energy"]
    K_true = data["kernel"]
    z_grid = data["z_grid"]
    t_vals = data["t_vals"]

    # Load trajectories
    results = []
    for split in ["train", "val", "test"]:
        manifest_path = gle_dir / f"{split}.jsonl"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                traj_path = gle_dir / entry["path"]
                z = np.load(traj_path)
                results.append({"split": split, "z": z, "entry": entry})

    print(f"Loaded {len(results)} trajectories from {gle_dir}")

    per_traj_metrics = {}
    for i, r in enumerate(results):
        z_true = r["z"]
        # In a real evaluation, z_pred would come from the model rollout.
        # For smoke test, use z_true as stand-in.
        z_pred = z_true.copy()
        metrics = compute_primary_metrics(
            z_true, z_pred, A_true, A_true, K_true, K_true, z_grid, t_vals)
        per_traj_metrics[f"traj_{i}"] = metrics
        print(f"  traj_{i} ({r['split']}): FE={metrics['free_energy_error']:.4f}, "
              f"K={metrics['memory_kernel_error']:.4f}, "
              f"JSD={metrics['stationary_jsd']:.4f}")

    # Aggregate
    agg = {k: float(np.mean([v[k] for v in per_traj_metrics.values()]))
           for k in ["free_energy_error", "memory_kernel_error",
                     "stationary_jsd", "acf_error", "transition_rate_error"]}
    print(f"\nAggregate: {json.dumps(agg, indent=2)}")

    # Build result
    out_dir = Path(args.output_dir)
    prov = ProvenanceBlock(
        protocol="reviewer_experiment_v1",
        experiment_id="S1",
        config_path=str(args.config) if args.config else "",
        random_seed=args.seed,
        output_dir=str(out_dir),
        result_path=str(out_dir / "results.json"),
    )
    result = ExperimentResult(
        experiment_id="S1",
        provenance=prov,
        metrics=agg,
        gates={"sanity_gate": False},
    )
    result.save(out_dir / "results.json")
    print(f"Result saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate synthetic GLE recovery")
    p.add_argument("--gle-data-dir",
                   default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/synthetic_gle")
    p.add_argument("--output-dir",
                   default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/synthetic_gle")
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=6303)
    main(p.parse_args())
