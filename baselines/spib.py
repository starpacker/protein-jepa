"""SPIB Baseline: State Predictive Information Bottleneck.

Simplified version using VAMP-2 score as surrogate for mutual information
between current and future state assignments, with entropy regularization
as the bottleneck.

Loss = -VAMP-2(membership_t, membership_{t+tau}) + beta * H(membership_t)

Reference: Wang et al. (2020) "State Predictive Information Bottleneck"
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baselines.data import extract_features_for_protein, load_all_proteins
from baselines.metrics import (
    compute_brier_skill,
    compute_implied_timescales,
    compute_msm_probe_accuracy,
    compute_vamp2_scores,
    load_msm_labels,
)
from baselines.vampnets import VAMPNet, compute_vamp2_loss, predict_memberships


def compute_entropy_regularization(memberships: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute entropy regularization term.

    H(z) = -mean(sum_k p_k * log(p_k + eps))

    Maximizing entropy encourages the model to use all states more evenly,
    acting as a bottleneck against collapsing to a single state.

    Args:
        memberships: (B, n_states) softmax probabilities
        eps: Small constant for numerical stability

    Returns:
        Scalar: negative entropy (to be minimized)
    """
    entropy = -torch.sum(memberships * torch.log(memberships + eps), dim=-1)  # (B,)
    return -entropy.mean()  # Negative so minimizing = maximizing entropy


def compute_spib_loss(
    memberships: list[torch.Tensor],
    lag: int = 1,
    beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute SPIB loss: -VAMP-2 + beta * entropy_regularization.

    Args:
        memberships: List of (F, n_states) membership tensors
        lag: Lag time for VAMP-2
        beta: Entropy regularization strength
        eps: Small constant for numerical stability

    Returns:
        Scalar loss
    """
    vamp2 = compute_vamp2_loss(memberships, lag=lag, eps=eps)  # Already negative
    all_mem = torch.cat(memberships, dim=0)
    entropy = compute_entropy_regularization(all_mem, eps=eps)
    return vamp2 + beta * entropy


def train_spib(
    features: list[np.ndarray],
    n_states: int = 6,
    lag: int = 1,
    beta: float = 0.1,
    lr: float = 1e-3,
    max_epochs: int = 200,
    batch_size: int = 256,
    early_stopping_patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[VAMPNet, dict[str, Any]]:
    """Train an SPIB model on one protein's features.

    Args:
        features: List of (F, D) feature arrays, one per trajectory
        n_states: Number of metastable states
        lag: Lag time for VAMP-2
        beta: Entropy regularization strength
        lr: Learning rate
        max_epochs: Maximum training epochs
        batch_size: Batch size
        early_stopping_patience: Patience for early stopping
        device: Device to train on
        seed: Random seed

    Returns:
        (trained_model, training_history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    D = features[0].shape[1]
    model = VAMPNet(input_dim=D, n_states=n_states).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Build time-lagged pairs
    X_t, X_tau = [], []
    for F in features:
        if len(F) > lag:
            X_t.append(F[:-lag])
            X_tau.append(F[lag:])

    if not X_t:
        return model, {"error": "no valid pairs"}

    X_t = np.concatenate(X_t, axis=0)
    X_tau = np.concatenate(X_tau, axis=0)
    N = len(X_t)

    # 80/20 train/val split
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(N * 0.8)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_t_train = torch.from_numpy(X_t[train_idx].astype(np.float32)).to(device)
    X_tau_train = torch.from_numpy(X_tau[train_idx].astype(np.float32)).to(device)
    X_t_val = torch.from_numpy(X_t[val_idx].astype(np.float32)).to(device)
    X_tau_val = torch.from_numpy(X_tau[val_idx].astype(np.float32)).to(device)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        X_t_shuffled = X_t_train[perm]
        X_tau_shuffled = X_tau_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_t = X_t_shuffled[start:end].to(device)
            batch_tau = X_tau_shuffled[start:end].to(device)

            optimizer.zero_grad()
            mem_t = model(batch_t)
            mem_tau = model(batch_tau)
            loss = compute_spib_loss([mem_t, mem_tau], lag=lag, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            mem_t_val = model(X_t_val.to(device))
            mem_tau_val = model(X_tau_val.to(device))
            val_loss = compute_spib_loss([mem_t_val, mem_tau_val], lag=lag, beta=beta).item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{max_epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def run_spib(
    manifest_path: str | Path,
    output_dir: str | Path,
    label_path: str | Path | None = None,
    max_proteins: int | None = None,
    n_states: int = 6,
    lag: int = 1,
    beta: float = 0.1,
    max_epochs: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Run SPIB baseline on a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SPIB BASELINE (beta={beta})")
    print(f"{'='*60}")
    t0 = time.time()

    grouped = load_all_proteins(manifest_path, max_proteins=max_proteins)
    print(f"Loaded {len(grouped)} proteins")

    msm_labels = None
    if label_path:
        msm_labels = load_msm_labels(label_path)

    all_memberships: dict[str, np.ndarray] = {}
    all_features_dict: dict[str, np.ndarray] = {}
    protein_results = []
    success_count = 0

    for protein_idx, (pid, records) in enumerate(sorted(grouped.items())):
        print(f"  [{protein_idx+1}/{len(grouped)}] {pid}...", end=" ")

        try:
            features = extract_features_for_protein(records, frame_stride=1)
            all_features_dict.update(features)

            traj_ids = list(features.keys())
            traj_features = [features[tid] for tid in traj_ids]

            if len(traj_features) < 2:
                print("skip (<2 trajs)")
                continue

            n_fit = max(1, int(len(traj_features) * 0.6))
            train_features = traj_features[:n_fit]

            model, history = train_spib(
                train_features, n_states=n_states, lag=lag, beta=beta,
                max_epochs=max_epochs, device=device, seed=seed + protein_idx,
            )

            for tid, feat in zip(traj_ids, traj_features):
                all_memberships[tid] = predict_memberships(model, feat, device=device)

            n_epochs = len(history.get("train_loss", []))
            val_loss = history.get("val_loss", [None])[-1] if history.get("val_loss") else None

            success_count += 1
            protein_results.append({
                "protein_id": pid, "status": "success",
                "n_trajectories": len(traj_ids),
                "n_epochs": n_epochs, "final_val_loss": val_loss,
            })
            print(f"OK ({n_epochs} epochs, val_loss={val_loss:.4f})")

        except Exception as e:
            print(f"ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nSPIB done: {success_count}/{len(grouped)} proteins in {elapsed:.1f}s")

    results: dict[str, Any] = {
        "method": "spib",
        "n_proteins": len(grouped),
        "n_successful": success_count,
        "time_seconds": elapsed,
        "params": {"n_states": n_states, "lag": lag, "beta": beta, "max_epochs": max_epochs, "seed": seed},
        "protein_results": protein_results,
    }

    if all_memberships:
        results["vamp2_scores"] = compute_vamp2_scores(all_memberships)
        if all_features_dict:
            results["implied_timescales"] = compute_implied_timescales(all_features_dict)
        if msm_labels:
            from baselines.metrics import compute_state_accuracy
            results["brier_skill"] = compute_brier_skill(all_memberships, msm_labels, n_states=n_states)
            hard_assignments = {tid: mem.argmax(axis=1).astype(np.int64) for tid, mem in all_memberships.items()}
            results["state_accuracy"] = compute_state_accuracy(hard_assignments, msm_labels, n_states=n_states)
            results["msm_probe"] = compute_msm_probe_accuracy(all_features_dict, msm_labels, n_states=n_states)

    out_path = output_dir / "spib_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")
    return results