#!/usr/bin/env python3
"""
K1: Closed Latent Dynamics and Protein Kinetics Evaluation.

Compares the model's learned dynamics against a reference MSM on
per-protein latent trajectories:

Metrics:
  - Implied timescale error (RMS across timescales)
  - Committor correlation (Pearson between reference and model)
  - Committor MAE (mean absolute error)
  - MFPT relative error (per state)

Protocol:
  1. For each protein trajectory in latent space:
     a. Cluster (k-means) to get N discrete states
     b. Build reference MSM (count matrix → transition matrix)
     c. For each cluster center, run model rollout
     d. Build rollout MSM from model trajectories
     e. Compare implied timescales, committor, MFPT
  2. Aggregate with bootstrap CI over proteins.
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifacts.reviewer_experiments._shared.model_interface import (
    DynamicsModel, NullModel,
)
from artifacts.reviewer_experiments._shared.lspwm_adapter import LSPWMDynamicsModel
from artifacts.reviewer_experiments._shared.metrics import (
    paired_bootstrap_ci, autocorrelation,
)
from artifacts.reviewer_experiments._shared.result_schema import (
    ExperimentResult, ProvenanceBlock,
)

# ---------------------------------------------------------------------------
# MSM utilities
# ---------------------------------------------------------------------------

def cluster_latents(traj: np.ndarray, n_clusters: int, seed: int = 42
                    ) -> tuple[np.ndarray, np.ndarray]:
    """K-means clustering of latent trajectory.

    Returns (cluster_labels, cluster_centers).
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(traj.astype(np.float64))
    return labels, kmeans.cluster_centers_


def build_transition_matrix(labels: np.ndarray, lag: int = 1,
                            reversible: bool = True) -> np.ndarray:
    """Build row-stochastic transition matrix from discrete trajectory.

    Args:
        labels: Discrete state sequence (T,).
        lag: Lag time in frames.
        reversible: Symmetrize count matrix for detailed balance.

    Returns:
        T: (n_states, n_states) row-stochastic transition matrix.
    """
    n = int(labels.max()) + 1
    C = np.zeros((n, n), dtype=np.float64)
    for t in range(len(labels) - lag):
        i, j = labels[t], labels[t + lag]
        C[i, j] += 1.0

    if reversible:
        C = C + C.T

    # Row-normalize
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    T = C / row_sums
    return T


def compute_implied_timescales(T: np.ndarray, lag_time: float = 1.0
                                ) -> np.ndarray:
    """Implied timescales from transition matrix.

    t_i = -lag / ln(|lambda_i|) for eigenvalues sorted descending.
    Skips the first eigenvalue (≈ 1.0).
    """
    eigvals = np.linalg.eigvals(T)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eig_sorted = eigvals[idx]
    # Skip first (≈ 1.0), take the rest
    ts = -lag_time / np.log(np.maximum(np.abs(eig_sorted[1:]), 1e-12))
    return ts


def solve_committor(T: np.ndarray, source: list[int], target: list[int]
                    ) -> np.ndarray:
    """Solve discrete committor: q_A→B.

    q = 0 on source, 1 on target.
    (I - T_QQ) * q_int = T_QB * 1
    """
    n = T.shape[0]
    src_mask = np.zeros(n, dtype=bool); src_mask[source] = True
    tgt_mask = np.zeros(n, dtype=bool); tgt_mask[target] = True
    interior = ~(src_mask | tgt_mask)
    i_idx = np.where(interior)[0]
    T_QQ = T[np.ix_(i_idx, i_idx)]
    T_QB = T[np.ix_(i_idx, target)]
    A = np.eye(len(i_idx)) - T_QQ
    b = T_QB.sum(axis=1)
    q = np.zeros(n)
    q[source] = 0.0; q[target] = 1.0
    q[i_idx] = np.linalg.solve(A, b)
    return q


def compute_mfpt(T: np.ndarray, target: list[int]) -> np.ndarray:
    """MFPT to target set from all states.

    (I - T_QQ) * m = 1
    """
    n = T.shape[0]
    tgt_mask = np.zeros(n, dtype=bool); tgt_mask[target] = True
    interior = np.where(~tgt_mask)[0]
    T_QQ = T[np.ix_(interior, interior)]
    A = np.eye(len(interior)) - T_QQ
    m = np.full(n, np.nan)
    m[target] = 0.0
    m[interior] = np.linalg.solve(A, np.ones(len(interior)))
    return m


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def build_trajs_from_data(data: dict[str, dict], min_len: int = 50
                          ) -> dict[str, np.ndarray]:
    """Extract latent trajectories from per-protein data.

    Returns {pid: trajectory (total_len, D)}.
    """
    trajs = {}
    for pid, d in data.items():
        fs = d.get("frame_states")
        if fs is None: continue
        if fs.ndim == 3:
            B, T, D = fs.shape
            traj = fs.transpose(1, 0, 2).reshape(-1, D) if B > 1 else fs[0]
            if len(traj) >= min_len:
                trajs[pid] = traj
    return trajs


def evaluate_protein_kinetics(
    trajs: dict[str, np.ndarray],
    model: DynamicsModel,
    n_clusters: int = 50,
    lag_time: int = 5,
    n_rollout_per_cluster: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run kinetics evaluation for all protein trajectories.

    Returns per-protein metrics and aggregate.
    """
    rng = np.random.default_rng(seed)
    per_protein = {}

    for pid, traj in trajs.items():
        if verbose:
            print(f"  Processing {pid} (len={len(traj)})...")
        try:
            result = evaluate_single_protein(
                traj, model, n_clusters, lag_time,
                n_rollout_per_cluster, rng, verbose)
            per_protein[pid] = result
            if verbose:
                print(f"    ITS error={result['its_mae']:.4f} "
                      f"comm_corr={result['committor_corr']:.4f} "
                      f"MFPT err={result['mfpt_rel_error']:.4f}")
        except Exception as e:
            if verbose:
                print(f"    FAILED: {e}")
            per_protein[pid] = {"error": str(e)}

    return per_protein


def evaluate_single_protein(
    traj: np.ndarray, model: DynamicsModel,
    n_clusters: int, lag: int, n_rollout: int,
    rng: np.random.Generator, verbose: bool,
) -> dict:
    """Evaluate kinetics for one protein trajectory."""
    d = model.latent_dim

    # 1. Cluster reference trajectory
    labels, centers = cluster_latents(traj, n_clusters)
    T_ref = build_transition_matrix(labels, lag=1, reversible=True)

    # Define source/target sets for committor
    # Source = first cluster (most populated near start)
    # Target = last cluster (most populated near end)
    hist_counts = np.bincount(labels, minlength=n_clusters)
    # Find most populated at beginning and end
    half = len(labels) // 2
    begin_counts = np.bincount(labels[:half], minlength=n_clusters)
    end_counts = np.bincount(labels[half:], minlength=n_clusters)
    source = [int(begin_counts.argmax())]
    target = [int(end_counts.argmax())]
    if source == target:
        # Fallback: pick most and least populated
        source = [int(hist_counts.argmin())]
        target = [int(hist_counts.argmax())]

    # Reference kinetics
    its_ref = compute_implied_timescales(T_ref, lag_time=float(lag))
    comm_ref = solve_committor(T_ref, source, target)
    mfpt_ref = compute_mfpt(T_ref, target)

    # 2. Model rollout from each cluster center
    rollout_labels_dict = {}
    for i, center in enumerate(centers):
        # Rollout in latent space
        traj_roll = model.rollout(center, n_rollout, rng=rng)
        if np.any(np.isnan(traj_roll)):
            continue
        # Assign to nearest cluster
        from sklearn.metrics.pairwise import pairwise_distances_argmin_min
        roll_labels = pairwise_distances_argmin_min(traj_roll, centers)[0]
        rollout_labels_dict[i] = roll_labels

    if not rollout_labels_dict:
        return {"error": "all rollouts failed"}

    # Build aggregate rollout trajectory
    roll_agg = np.concatenate([rollout_labels_dict[i] for i in sorted(rollout_labels_dict)])
    T_roll = build_transition_matrix(roll_agg, lag=1, reversible=True)

    # 3. Compare kinetics

    # Implied timescales
    its_roll = compute_implied_timescales(T_roll, lag_time=float(lag))
    min_len = min(len(its_ref), len(its_roll))
    its_err = np.abs(np.log(its_ref[:min_len] + 1e-12) - np.log(its_roll[:min_len] + 1e-12))
    its_mae = float(np.mean(its_err))
    its_rel_err = float(np.mean(its_err / (its_ref[:min_len] + 1e-12)))

    # Committor correlation
    comm_roll = solve_committor(T_roll, source, target)
    if len(comm_ref) == len(comm_roll):
        comm_corr = float(np.corrcoef(comm_ref, comm_roll)[0, 1])
        comm_mae = float(np.mean(np.abs(comm_ref - comm_roll)))
    else:
        comm_corr = 0.0
        comm_mae = float("nan")

    # MFPT error
    mfpt_roll = compute_mfpt(T_roll, target)
    mfpt_ref_valid = mfpt_ref[~np.isnan(mfpt_ref)]
    mfpt_roll_valid = mfpt_roll[~np.isnan(mfpt_roll)]
    if len(mfpt_ref_valid) == len(mfpt_roll_valid) and len(mfpt_ref_valid) > 0:
        mfpt_rel_err = float(np.mean(
            np.abs(mfpt_ref_valid - mfpt_roll_valid) / (mfpt_ref_valid + 1e-12)))
    else:
        mfpt_rel_err = float("nan")

    return {
        "its_mae": its_mae,
        "its_rel_error": its_rel_err,
        "its_ref": its_ref[:min_len].tolist(),
        "its_roll": its_roll[:min_len].tolist(),
        "committor_corr": comm_corr,
        "committor_mae": comm_mae,
        "committor_ref": comm_ref.tolist(),
        "committor_roll": comm_roll.tolist(),
        "mfpt_rel_error": mfpt_rel_err,
        "n_states": n_clusters,
        "n_rollout_per_state": int(n_rollout),
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_kinetics(per_protein: dict) -> dict:
    """Compute aggregate metrics with bootstrap CI."""
    its_errs = {}
    comm_corrs = {}
    mfpt_errs = {}
    for pid, r in per_protein.items():
        if "error" in r: continue
        its_errs[pid] = r.get("its_mae", float("nan"))
        comm_corrs[pid] = r.get("committor_corr", float("nan"))
        mfpt_errs[pid] = r.get("mfpt_rel_error", float("nan"))

    # Filter NaN
    its_errs = {k: v for k, v in its_errs.items() if not np.isnan(v)}
    comm_corrs = {k: v for k, v in comm_corrs.items() if not np.isnan(v)}
    mfpt_errs = {k: v for k, v in mfpt_errs.items() if not np.isnan(v)}

    aggregate = {}
    if its_errs:
        aggregate["its_mae"] = paired_bootstrap_ci(its_errs, n_resamples=2000, seed=42)
    if comm_corrs:
        aggregate["committor_corr"] = paired_bootstrap_ci(comm_corrs, n_resamples=2000, seed=42)
    if mfpt_errs:
        aggregate["mfpt_rel_error"] = paired_bootstrap_ci(mfpt_errs, n_resamples=2000, seed=42)
    aggregate["n_valid_proteins"] = len(its_errs)
    return aggregate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="K1: Protein Kinetics Evaluation")
    p.add_argument("--model", choices=["lspwm", "null"], default="lspwm")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--config", default="")
    p.add_argument("--manifest",
                   default="/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl")
    p.add_argument("--output-dir",
                   default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/protein_kinetics")
    p.add_argument("--n-clusters", type=int, default=50,
                   help="Number of discrete states for MSM")
    p.add_argument("--lag-time", type=int, default=5,
                   help="Lag time for MSM in frames")
    p.add_argument("--n-rollout", type=int, default=200,
                   help="Rollout steps per cluster center")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=6303)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dummy", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print("K1: Protein Kinetics Evaluation")
    print("=" * 60)

    if args.dummy or not args.checkpoint:
        print("[Dummy mode]")
        model = NullModel()
        data = model.extract_latents()
        latent_dim = model.latent_dim
    elif args.model == "lspwm":
        print(f"[LSP-WM] {args.checkpoint}")
        model = LSPWMDynamicsModel()
        model.load(checkpoint_path=args.checkpoint, config_path=args.config,
                   manifest=args.manifest, device=args.device,
                   batch_size=args.batch_size, max_batches=args.max_batches,
                   seed=args.seed)
        data = model.extract_latents(max_batches=args.max_batches)
        latent_dim = model.latent_dim
    else:
        raise ValueError(f"Unknown model: {args.model}")

    trajs = build_trajs_from_data(data)
    print(f"  Proteins: {len(data)}, Trajectories: {len(trajs)}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  n_clusters: {args.n_clusters}, lag: {args.lag_time}")
    print()

    if not trajs:
        print("  No valid trajectories found!")
        return

    per_protein = evaluate_protein_kinetics(
        trajs, model, n_clusters=args.n_clusters, lag_time=args.lag_time,
        n_rollout_per_cluster=args.n_rollout, seed=args.seed, verbose=True)

    aggregate = aggregate_kinetics(per_protein)

    print(f"\n  Aggregate results:")
    for k, v in aggregate.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"    {k}: {v['mean']:.4f} [{v['lower']:.4f}, {v['upper']:.4f}]")
        else:
            print(f"    {k}: {v}")

    prov = ProvenanceBlock(
        protocol="reviewer_experiment_v1", experiment_id="K1",
        config_path=args.config, checkpoint_path=args.checkpoint,
        manifest_path=args.manifest, random_seed=args.seed,
        device=args.device, batch_size=args.batch_size,
        max_batches=args.max_batches, sampler_mode="hierarchical",
        output_dir=args.output_dir,
        result_path=str(Path(args.output_dir) / "results.json"),
        train_only_fitted_objects={
            "reference_msm_n_clusters": args.n_clusters,
            "reference_msm_lag_time": args.lag_time,
        },
    )
    result = ExperimentResult(
        experiment_id="K1", provenance=prov,
        metrics={
            "per_protein": per_protein,
            "aggregate": aggregate,
            "n_clusters": args.n_clusters,
            "lag_time": args.lag_time,
            "latent_dim": latent_dim,
        },
        gates={"kinetics_gate": bool(aggregate)},
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "results.json")
    print(f"\n  -> {out / 'results.json'}")
    model.close()


if __name__ == "__main__":
    main()
