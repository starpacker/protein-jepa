"""TICA + MSM Baseline: standard MSM pipeline on CA features.

Pipeline:
  1. Standardize features (per-feature z-score)
  2. TICA projection (lag=1, dim=10, kinetic_map scaling)
  3. KMeans clustering (48 microstates)
  4. Transition count estimation (lag=1, sliding)
  5. Maximum Likelihood MSM (reversible, largest connected component)
  6. PCCA+ coarse-graining to 6 metastable states
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from deeptime.clustering import KMeans as DtKMeans
from deeptime.decomposition import TICA
from deeptime.markov import TransitionCountEstimator, pcca
from deeptime.markov.msm import MaximumLikelihoodMSM

from baselines.data import extract_features_for_protein, load_all_proteins
from baselines.metrics import (
    compute_implied_timescales,
    compute_vamp2_scores,
    load_msm_labels,
)


def _tica_features(
    fit_features: list[np.ndarray],
    all_features: list[np.ndarray],
    lag: int = 1,
    dim: int = 10,
    seed: int = 42,
) -> list[np.ndarray]:
    """Apply TICA to features.

    Args:
        fit_features: Per-trajectory features for fitting (list of (F, D) arrays)
        all_features: Per-trajectory features to transform (list of (F, D) arrays)
        lag: TICA lag time
        dim: TICA output dimension
        seed: Random seed

    Returns:
        List of transformed feature arrays
    """
    scaler = StandardScaler()
    fit_concat = np.concatenate(fit_features, axis=0)
    scaler.fit(fit_concat)

    scaled_fit = [scaler.transform(f) for f in fit_features]
    scaled_all = [scaler.transform(f) for f in all_features]

    tica = TICA(lagtime=lag, dim=dim, epsilon=1e-6, scaling="kinetic_map")
    tica.fit(scaled_fit)

    return [np.asarray(tica.transform(f), dtype=np.float64) for f in scaled_all]


def _kmeans_microstates(
    reduced_fit: np.ndarray,
    n_microstates: int,
    seed: int = 42,
) -> tuple:
    """Cluster TICA-reduced features into microstates.

    Args:
        reduced_fit: (N, dim) TICA-reduced features for fitting
        n_microstates: Number of microstates
        seed: Random seed

    Returns:
        (kmeans_model, kmeans_fitted_model, assignments)
        where kmeans_model is the estimator and kmeans_fitted_model is the
        fitted model object that can be used for prediction.
    """
    effective_microstates = min(n_microstates, max(2, reduced_fit.shape[0] // 5))
    kmeans = DtKMeans(
        n_clusters=effective_microstates,
        max_iter=500,
        init_strategy="kmeans++",
        fixed_seed=int(seed),
        n_jobs=1,
    )
    kmeans.fit(reduced_fit)
    kmeans_fitted = kmeans.fetch_model()
    assignments = kmeans_fitted.transform(reduced_fit)
    return kmeans, kmeans_fitted, assignments.astype(np.int64)


def _fit_msm(
    micro_assignments: list[np.ndarray],
    lag: int = 1,
    n_states: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit an MSM and PCCA on microstate assignments.

    Args:
        micro_assignments: Per-trajectory microstate assignments
        lag: MSM lag time (frames)
        n_states: Target number of metastable states
        seed: Random seed

    Returns:
        dict with fit results including pcca_model, msm_model, state_assignments, etc.
    """
    # Count transitions
    count_model = (
        TransitionCountEstimator(lagtime=lag, count_mode="sliding")
        .fit(micro_assignments)
        .fetch_model()
    )
    counts = np.asarray(count_model.count_matrix, dtype=np.float64)

    if counts.sum() == 0:
        return {"status": "no_transitions"}

    # Fit MSM
    msm_estimator = MaximumLikelihoodMSM(
        reversible=True, allow_disconnected=False, sparse=False
    )
    msm_model = msm_estimator.fit(count_model).fetch_model()
    connected_microstates = np.asarray(msm_model.state_symbols(), dtype=np.int64)
    if connected_microstates.ndim != 1 or connected_microstates.size < 2:
        return {"status": "too_few_connected"}

    transition = np.asarray(msm_model.transition_matrix, dtype=np.float64)
    stationary = np.asarray(msm_model.stationary_distribution, dtype=np.float64)
    eigenvalues = np.asarray(msm_model.eigenvalues(), dtype=np.float64)

    # PCCA+
    n_micro = len(connected_microstates)
    requested_states = min(n_states, int(n_micro))
    pcca_model = None
    micro_to_state = None

    for candidate_states in range(requested_states, 1, -1):
        try:
            candidate_model = pcca(
                transition,
                candidate_states,
                stationary_distribution=stationary,
            )
            candidate_assignments = np.asarray(
                candidate_model.assignments, dtype=np.int64
            )
            if candidate_assignments.shape != (connected_microstates.size,):
                continue
            if candidate_assignments.min() < 0 or candidate_assignments.max() >= candidate_states:
                continue
            pcca_model = candidate_model
            micro_to_state = candidate_assignments
            effective_states = candidate_states
            break
        except Exception:
            continue

    if pcca_model is None or micro_to_state is None:
        return {"status": "pcca_failed"}

    # Map all microstates (including disconnected ones) to state -1
    all_micro = np.concatenate([a for a in micro_assignments if len(a) > 0])
    n_micro_total = int(all_micro.max()) + 1 if len(all_micro) > 0 else 0
    full_micro_to_state = np.full(n_micro_total, -1, dtype=np.int64)
    full_micro_to_state[connected_microstates] = micro_to_state

    # Assign states to all trajectories
    state_assignments = [full_micro_to_state[micro] for micro in micro_assignments]

    # Compute diagnostics
    flat = np.concatenate([s[s >= 0] for s in state_assignments])
    occ = np.bincount(flat, minlength=effective_states).astype(np.float64)
    occ_fraction = occ / max(float(occ.sum()), 1.0)
    occupied = int((occ > 0).sum())
    status = "success" if occupied >= 2 else "single_state"

    return {
        "status": status,
        "pcca_model": pcca_model,
        "msm_model": msm_model,
        "state_assignments": state_assignments,
        "full_micro_to_state": full_micro_to_state,
        "n_metastable_states": effective_states,
        "n_microstates": int(n_micro),
        "n_connected_microstates": int(connected_microstates.size),
        "occupied_states": occupied,
        "occupancy_fraction": occ_fraction.tolist(),
        "stationary_distribution": stationary.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "transition_matrix": transition.tolist(),
    }


def run_tica_msm(
    manifest_path: str | Path,
    output_dir: str | Path,
    label_path: str | Path | None = None,
    max_proteins: int | None = None,
    n_states: int = 6,
    tica_dim: int = 10,
    n_microstates: int = 48,
    msm_lag: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run TICA+MSM baseline on a manifest.

    Args:
        manifest_path: Path to JSONL manifest
        output_dir: Directory for output files
        label_path: Optional path to MSM labels for comparison
        max_proteins: Limit to first N proteins (for testing)
        n_states: Number of metastable states
        tica_dim: TICA output dimension
        n_microstates: Number of microstates for KMeans
        msm_lag: MSM lag time in frames
        seed: Random seed

    Returns:
        dict with results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("TICA + MSM BASELINE")
    print(f"{'='*60}")
    print(f"Manifest: {manifest_path}")
    t0 = time.time()

    # Load data
    grouped = load_all_proteins(manifest_path, max_proteins=max_proteins)
    print(f"Loaded {len(grouped)} proteins")

    # Load MSM labels if provided
    msm_labels = None
    if label_path:
        msm_labels = load_msm_labels(label_path)

    # Run per-protein TICA+MSM
    all_state_assignments: dict[str, np.ndarray] = {}
    all_features_dict: dict[str, np.ndarray] = {}
    protein_results = []
    success_count = 0

    for protein_idx, (pid, records) in enumerate(sorted(grouped.items())):
        print(f"  [{protein_idx + 1}/{len(grouped)}] {pid}...", end=" ")

        try:
            # Extract features
            features = extract_features_for_protein(records, frame_stride=1)
            all_features_dict.update(features)

            traj_ids = list(features.keys())
            traj_features = [features[tid] for tid in traj_ids]

            if len(traj_features) < 2:
                print("skip (<2 trajs)")
                continue

            # Split replicas: 60% fit, 40% score
            n_fit = max(1, int(len(traj_features) * 0.6))
            fit_features = traj_features[:n_fit]
            all_features = traj_features

            # TICA
            reduced = _tica_features(
                fit_features, all_features,
                lag=msm_lag, dim=tica_dim, seed=seed + protein_idx,
            )

            # KMeans microstates
            reduced_fit = np.concatenate(reduced[:n_fit], axis=0)
            kmeans_est, kmeans_fitted, micro_assignments = _kmeans_microstates(
                reduced_fit, n_microstates, seed=seed + protein_idx,
            )

            # Split micro assignments back
            split_points = np.cumsum([len(r) for r in reduced[:n_fit]])[:-1]
            micro_by_traj = list(np.split(micro_assignments, split_points))
            # For non-fit trajectories, predict microstates using fitted model
            for i in range(n_fit, len(reduced)):
                micro = kmeans_fitted.transform(reduced[i]).astype(np.int64)
                micro_by_traj.append(micro)

            # Fit MSM
            result = _fit_msm(micro_by_traj, lag=msm_lag, n_states=n_states, seed=seed + protein_idx)

            if result["status"] == "success":
                success_count += 1
                for tid, state_assign in zip(traj_ids, result["state_assignments"]):
                    all_state_assignments[tid] = state_assign

                protein_results.append({
                    "protein_id": pid,
                    "status": "success",
                    "n_trajectories": len(traj_ids),
                    "n_metastable_states": result["n_metastable_states"],
                    "occupied_states": result["occupied_states"],
                    "occupancy_fraction": result["occupancy_fraction"],
                })
                print(f"OK ({result['n_metastable_states']} states, {result['occupied_states']} occupied)")
            else:
                print(f"FAILED ({result['status']})")

        except Exception as e:
            print(f"ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nTICA+MSM done: {success_count}/{len(grouped)} proteins in {elapsed:.1f}s")

    # Compute metrics
    results = {
        "method": "tica_msm",
        "n_proteins": len(grouped),
        "n_successful": success_count,
        "time_seconds": elapsed,
        "params": {
            "n_states": n_states,
            "tica_dim": tica_dim,
            "n_microstates": n_microstates,
            "msm_lag": msm_lag,
            "seed": seed,
        },
        "protein_results": protein_results,
    }

    # VAMP-2 scores on state assignments (one-hot encoded)
    if all_state_assignments:
        # One-hot encode state assignments
        one_hot = {}
        for tid, labels in all_state_assignments.items():
            F = len(labels)
            oh = np.zeros((F, n_states), dtype=np.float32)
            for i, lbl in enumerate(labels):
                if 0 <= lbl < n_states:
                    oh[i, int(lbl)] = 1.0
            one_hot[tid] = oh

        results["vamp2_scores"] = compute_vamp2_scores(one_hot)

        # Implied timescales on features
        if all_features_dict:
            results["implied_timescales"] = compute_implied_timescales(all_features_dict)

        # State accuracy against MSM labels
        if msm_labels:
            from baselines.metrics import compute_state_accuracy, compute_msm_probe_accuracy
            results["state_accuracy"] = compute_state_accuracy(
                all_state_assignments, msm_labels, n_states=n_states
            )
            results["msm_probe"] = compute_msm_probe_accuracy(
                all_features_dict, msm_labels, n_states=n_states
            )

    # Save
    out_path = output_dir / "tica_msm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    return results