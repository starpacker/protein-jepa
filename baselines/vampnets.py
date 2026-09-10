"""VAMPnets Baseline: MLP trained to maximize VAMP-2 score.

Trains a small MLP to map CA features to 6-state softmax membership
probabilities, using the VAMP-2 score as the training objective.

Reference: Mardt et al. (2018) "VAMPnets for deep learning of molecular kinetics"
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


class VAMPNet(nn.Module):
    """Small MLP mapping features to state membership probabilities.

    Architecture: D -> 256 -> 128 -> 64 -> n_states (softmax)
    """

    def __init__(
        self,
        input_dim: int,
        n_states: int = 6,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_states))
        layers.append(nn.Softmax(dim=-1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, D) input features

        Returns:
            (B, n_states) softmax membership probabilities
        """
        return self.net(x)


def compute_vamp2_loss(
    memberships: list[torch.Tensor],
    lag: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute differentiable VAMP-2 loss from membership sequences.

    Args:
        memberships: List of (F, n_states) membership tensors, one per trajectory
        lag: Lag time in frames

    Returns:
        Negative VAMP-2 score (scalar tensor, to be minimized)
    """
    n_states = memberships[0].shape[-1]
    device = memberships[0].device

    # Build time-lagged pairs
    t_pairs = []
    tau_pairs = []
    for mem in memberships:
        if len(mem) > lag:
            t_pairs.append(mem[:-lag])
            tau_pairs.append(mem[lag:])

    if not t_pairs:
        return torch.tensor(0.0, device=device, requires_grad=True)

    x_t = torch.cat(t_pairs, dim=0)  # (N, n_states)
    x_tau = torch.cat(tau_pairs, dim=0)
    N = x_t.shape[0]

    if N < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Center
    z_t = x_t - x_t.mean(dim=0, keepdim=True)
    z_tau = x_tau - x_tau.mean(dim=0, keepdim=True)

    # Covariance matrices
    C_00 = (z_t.T @ z_t) / (N - 1) + eps * torch.eye(n_states, device=device)
    C_tt = (z_tau.T @ z_tau) / (N - 1) + eps * torch.eye(n_states, device=device)
    C_0t = (z_t.T @ z_tau) / (N - 1)

    # Inverse square root via SVD
    try:
        U_00, S_00, V_00 = torch.linalg.svd(C_00)
        S_00_inv = torch.where(S_00 > eps, 1.0 / torch.sqrt(S_00), torch.zeros_like(S_00))
        C_00_inv_sqrt = U_00 @ torch.diag(S_00_inv) @ V_00

        U_tt, S_tt, V_tt = torch.linalg.svd(C_tt)
        S_tt_inv = torch.where(S_tt > eps, 1.0 / torch.sqrt(S_tt), torch.zeros_like(S_tt))
        C_tt_inv_sqrt = U_tt @ torch.diag(S_tt_inv) @ V_tt

        K = C_00_inv_sqrt @ C_0t @ C_tt_inv_sqrt
        score = torch.linalg.matrix_norm(K, ord="fro") ** 2
    except Exception:
        score = torch.tensor(0.0, device=device)

    return -score


def train_vampnet(
    features: list[np.ndarray],
    n_states: int = 6,
    lag: int = 1,
    lr: float = 1e-3,
    max_epochs: int = 200,
    batch_size: int = 256,
    early_stopping_patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
) -> tuple[VAMPNet, dict[str, Any]]:
    """Train a VAMPNet on one protein's features."""
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
            loss = compute_vamp2_loss([mem_t, mem_tau], lag=lag)
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
            val_loss = compute_vamp2_loss([mem_t_val, mem_tau_val], lag=lag).item()

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


def predict_memberships(model: VAMPNet, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Predict state memberships for a trajectory."""
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(features.astype(np.float32)).to(device)
        return model(X).cpu().numpy()


def run_vampnets(
    manifest_path: str | Path,
    output_dir: str | Path,
    label_path: str | Path | None = None,
    max_proteins: int | None = None,
    n_states: int = 6,
    lag: int = 1,
    max_epochs: int = 200,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Run VAMPnets baseline on a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("VAMPNet BASELINE")
    print(f"{'='*60}")
    print(f"Manifest: {manifest_path}")
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

            model, history = train_vampnet(
                train_features, n_states=n_states, lag=lag,
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
    print(f"\nVAMPNet done: {success_count}/{len(grouped)} proteins in {elapsed:.1f}s")

    results: dict[str, Any] = {
        "method": "vampnet",
        "n_proteins": len(grouped),
        "n_successful": success_count,
        "time_seconds": elapsed,
        "params": {"n_states": n_states, "lag": lag, "max_epochs": max_epochs, "seed": seed},
        "protein_results": protein_results,
    }

    if all_memberships:
        results["vamp2_scores"] = compute_vamp2_scores(all_memberships)
        if all_features_dict:
            results["implied_timescales"] = compute_implied_timescales(all_features_dict)
        if msm_labels:
            results["brier_skill"] = compute_brier_skill(all_memberships, msm_labels, n_states=n_states)
            from baselines.metrics import compute_state_accuracy
            hard_assignments = {tid: mem.argmax(axis=1).astype(np.int64) for tid, mem in all_memberships.items()}
            results["state_accuracy"] = compute_state_accuracy(hard_assignments, msm_labels, n_states=n_states)
            results["msm_probe"] = compute_msm_probe_accuracy(all_features_dict, msm_labels, n_states=n_states)

    out_path = output_dir / "vampnet_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved to {out_path}")
    return results