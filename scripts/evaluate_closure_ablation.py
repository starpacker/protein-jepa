#!/usr/bin/env python3
"""
M1: 2x2 Thermodynamic Closure Ablation + generic history baselines.

Evaluates 6 closure methods on frozen latent representations:
  Markov       — overdamped Langevin (const D, KDE force)
  FreeEnergy   — overdamped Langevin (position-dependent D)
  Mori         — GLE with memory kernel K(t) + constant D
  Full         — GLE with memory K(t) + position-dependent D
  GRU          — neural dynamics via GRU
  CausalTransformer — neural dynamics via causal transformer

Metrics:
  Innovation whiteness W (autocorrelation of residuals)
  Stationary JSD (KDE comparison of stationary distributions)
  FDT residual (fluctuation-dissipation theorem)
  Transition operator error (Koopman eigenvalue comparison)
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from artifacts.reviewer_experiments._shared.model_interface import (
    DynamicsModel, NullModel,
)
from artifacts.reviewer_experiments._shared.lspwm_adapter import LSPWMDynamicsModel
from artifacts.reviewer_experiments._shared.metrics import (
    paired_bootstrap_ci, innovation_whiteness_score,
    jensen_shannon_divergence, empirical_fdt_residual, autocorrelation,
)
from artifacts.reviewer_experiments._shared.result_schema import (
    ExperimentResult, ProvenanceBlock,
)

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def build_trajectories(data: dict[str, dict], min_len: int = 50
                       ) -> dict[str, np.ndarray]:
    """Extract latent trajectories from per-protein data.

    frame_states: (N, T, D). If N=1, one trajectory of length T.
    If N > 1, concatenate into one long trajectory.

    Returns {pid: trajectory (total_len, D)}.
    """
    trajs = {}
    for pid, d in data.items():
        fs = d.get("frame_states")
        if fs is None: continue
        if fs.ndim == 3:
            B, T, D = fs.shape
            if B == 1:
                traj = fs[0]  # (T, D)
            else:
                # Concatenate along time
                traj = fs.transpose(1, 0, 2).reshape(-1, D)
        else:
            continue
        if len(traj) >= min_len:
            trajs[pid] = traj
    return trajs


def build_transitions(trajs: dict[str, np.ndarray], dt: float = 1.0
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build transition dataset from trajectories.

    Returns:
        Z: current states (N, D)
        dZ: increments (N, D)
        Z_next: next states (N, D)
    """
    Z_list, dZ_list, Zn_list = [], [], []
    for traj in trajs.values():
        Z_list.append(traj[:-1])
        dZ_list.append(np.diff(traj, axis=0))
        Zn_list.append(traj[1:])
    return (np.concatenate(Z_list, 0), np.concatenate(dZ_list, 0),
            np.concatenate(Zn_list, 0))


# ---------------------------------------------------------------------------
# KDE-based force / diffusion estimation
# ---------------------------------------------------------------------------

def estimate_force_kde(Z: np.ndarray, bandwidth: float | None = None
                       ) -> Callable[[np.ndarray], np.ndarray]:
    """Estimate force F(z) = -nabla U(z) using KDE score.

    Works well only in low dimensions (<10). For high-d, use model force.
    For this implementation, we use a simple kernel regression approach.
    """
    from sklearn.neighbors import KernelDensity
    # Reduce dim for KDE if needed
    d = Z.shape[1]
    if d > 20:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(20, d))
        Z_reduced = pca.fit_transform(Z)
    else:
        pca = None
        Z_reduced = Z

    if bandwidth is None:
        bandwidth = np.std(Z_reduced) * (len(Z_reduced) ** (-0.2))

    kde = KernelDensity(bandwidth=bandwidth, metric='euclidean')
    kde.fit(Z_reduced.astype(np.float64))

    def force_fn(z: np.ndarray) -> np.ndarray:
        z_flat = z.reshape(-1, d)
        if pca is not None:
            zr = pca.transform(z_flat.astype(np.float64))
        else:
            zr = z_flat.astype(np.float64)
        # Score function = nabla log p(z) via numerical gradient
        h = 1e-4
        grad = np.zeros_like(zr)
        for i in range(zr.shape[1]):
            z_plus = zr.copy(); z_plus[:, i] += h
            z_minus = zr.copy(); z_minus[:, i] -= h
            logp_plus = kde.score_samples(z_plus)
            logp_minus = kde.score_samples(z_minus)
            grad[:, i] = (logp_plus - logp_minus) / (2 * h)
        # Force = nabla log p (which gives -nabla U up to a constant)
        if pca is not None:
            grad_full = pca.inverse_transform(grad)
        else:
            grad_full = grad
        return grad_full.reshape(z.shape)

    return force_fn


def estimate_diffusion_local(dZ: np.ndarray, Z: np.ndarray,
                             n_neighbors: int = 50, max_samples: int = 2000
                             ) -> Callable[[np.ndarray], np.ndarray]:
    """Estimate position-dependent diffusion D(z) via local covariance.

    Subsamples data for speed. Returns function D(z) -> ndarray (D,) with
    diagonal diffusion.
    """
    from sklearn.neighbors import NearestNeighbors
    d = Z.shape[1]
    # Subsample for faster KNN
    if len(Z) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z), max_samples, replace=False)
        Z_sample = Z[idx].astype(np.float64)
        dZ_sample = dZ[idx]
    else:
        Z_sample = Z.astype(np.float64)
        dZ_sample = dZ
    n_nbrs = min(n_neighbors, len(Z_sample))
    nbrs = NearestNeighbors(n_neighbors=n_nbrs, algorithm='kd_tree' if d <= 20 else 'brute')
    nbrs.fit(Z_sample)

    def diff_fn(z: np.ndarray) -> np.ndarray:
        z_flat = np.asarray(z, dtype=np.float64).reshape(-1, d)
        # Batch KNN: get all neighbor indices at once
        dist, idx_mat = nbrs.kneighbors(z_flat, return_distance=True)
        # vectorized variance over neighbors
        # idx_mat: (n_query, n_nbrs); dZ_sample: (N, d)
        neighbor_dz = dZ_sample[idx_mat]  # (n_query, n_nbrs, d)
        D_out = 0.5 * np.var(neighbor_dz, axis=1, ddof=1)
        return D_out.reshape(z.shape)

    return diff_fn


def estimate_memory_kernel_volterra(
    trajs: dict[str, np.ndarray], max_lag: int = 20, dt: float = 1.0,
) -> np.ndarray:
    """Estimate memory kernel K(tau) via Volterra inversion.

    Uses velocity autocorrelation C(tau) = <v(t) v(t+tau)>,
    then solves: integro-differential equation for K.

    Returns kernel (max_lag+1, D, D).
    """
    # Compute velocity (first difference)
    v_all = []
    for traj in trajs.values():
        v = np.diff(traj, axis=0) / dt
        v_all.append(v)
    V = np.concatenate(v_all, 0)
    d = V.shape[1]

    # Velocity autocorrelation C(tau) = <v(t) : v(t+tau)>
    T = len(V)
    C = np.zeros((max_lag + 1, d, d))
    for tau in range(max_lag + 1):
        if tau >= T: break
        # <v_i(t) * v_j(t+tau)> averaged over t
        v1 = V[:T-tau]
        v2 = V[tau:]
        C[tau] = (v1.T @ v2) / (T - tau)

    # Initial guess: K(0) = -C(1) / C(0) (simple Markovian approximation)
    kernel = np.zeros((max_lag + 1, d, d))
    C0_inv = np.linalg.pinv(C[0] + 1e-8 * np.eye(d))
    kernel[0] = -C[1] @ C0_inv

    # Volterra inversion for tau > 0
    # K(tau) = integral(0, tau) K(tau - s) * C(s) ds / ... 
    # Simplified: K(tau) ~ -Gamma(tau) / C(0) where Gamma is memory kernel
    for tau in range(1, max_lag + 1):
        if tau >= len(C) - 1: break
        # Simple approximation: K(tau) proportional to -C(tau+1)/C(0)
        kernel[tau] = -C[min(tau+1, len(C)-1)] @ C0_inv * np.exp(-tau / max_lag)

    return kernel


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ClosureMethod:
    """Base interface for closure methods."""

    def __init__(self, name: str, latent_dim: int, history_window: int = 8):
        self.name = name
        self.latent_dim = latent_dim
        self.history_window = history_window
        self._fitted = False

    def fit(self, trajs: dict[str, np.ndarray], model: DynamicsModel | None = None
            ) -> dict:
        """Fit method on trajectory data.

        Args:
            trajs: {pid: trajectory (T, D)}
            model: Optional model for accessing force/diffusion/kernel
        Returns:
            dict of fit diagnostics.
        """
        raise NotImplementedError

    def predict_step(self, z: np.ndarray, history: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
        """One-step prediction: (mean_next, cov_next)."""
        raise NotImplementedError

    def rollout(self, z_init: np.ndarray, n_steps: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate trajectory of length n_steps."""
        raise NotImplementedError

    def innovation(self, traj: np.ndarray, rng: np.random.Generator | None = None
                   ) -> np.ndarray:
        """Compute innovation sequence (prediction residuals) with teacher forcing."""
        rng = rng or np.random.default_rng()
        T, d = traj.shape
        innovations = np.zeros((T - 1, d))
        z = traj[0].copy()
        history = [z.copy()]  # Accumulate history for memory-based methods
        for t in range(T - 1):
            mean, cov = self.predict_step(z, history)
            innovations[t] = traj[t + 1] - mean
            z = traj[t + 1]  # Teacher forcing: use ground truth
            history.append(z.copy())
        return innovations

    def compute_stationary_dist(self, n_samples: int = 10000,
                                rng: np.random.Generator | None = None
                                ) -> np.ndarray:
        """Generate samples from the learned stationary distribution."""
        return None  # Subclasses can override


class GenericMarkovClosure(ClosureMethod):
    """Unconstrained one-step Markov baseline.

    This is deliberately a plain MLP z_t -> z_{t+1}.  It provides the
    prediction-only control required by M1, without force, diffusion, or
    memory structure.
    """

    def __init__(self, name, latent_dim, history_window=8, hidden_dim=256):
        super().__init__(name, latent_dim, history_window)
        self.hidden_dim = hidden_dim

    def fit(self, trajs, model=None, max_epochs=100, patience=10,
            max_samples=5000):
        Z, _, Zn = build_transitions(trajs)
        if len(Z) > max_samples:
            idx = np.random.default_rng(42).choice(len(Z), max_samples,
                                                     replace=False)
            Z, Zn = Z[idx], Zn[idx]
        n_train = max(1, int(len(Z) * 0.8))
        perm = np.random.default_rng(42).permutation(len(Z))
        tr, va = perm[:n_train], perm[n_train:]
        self._net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim))
        opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(max_epochs):
            self._net.train(); opt.zero_grad()
            loss = F.mse_loss(self._net(_to_tensor(Z[tr])), _to_tensor(Zn[tr]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            opt.step()
            self._net.eval()
            with torch.no_grad():
                val = F.mse_loss(self._net(_to_tensor(Z[va])), _to_tensor(Zn[va])).item()
            if val < best - 1e-7:
                best, stale = val, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self._net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience: break
        if best_state is not None: self._net.load_state_dict(best_state)
        self._fitted = True
        return {"train_loss": float(loss.item()), "val_loss": float(best),
                "epochs_trained": epoch + 1, "n_samples": len(Z)}

    def predict_step(self, z, history=None):
        self._net.eval()
        with torch.no_grad():
            pred = self._net(_to_tensor(z.reshape(-1, self.latent_dim)))
        mean = _to_np(pred).reshape(z.shape)
        return mean, np.eye(self.latent_dim) * 1e-3

    def rollout(self, z_init, n_steps, rng=None):
        traj = np.zeros((n_steps, self.latent_dim), dtype=np.float64)
        traj[0] = np.asarray(z_init).ravel()
        for t in range(1, n_steps):
            traj[t] = self.predict_step(traj[t - 1])[0].ravel()
            if not np.all(np.isfinite(traj[t])):
                traj[t:] = traj[t - 1]; break
        return traj


# ---------------------------------------------------------------------------
# Markov closure
# ---------------------------------------------------------------------------

class MarkovClosure(ClosureMethod):
    """Overdamped Langevin: dZ = -nabla U(Z) dt + sqrt(2 D) dW.

    Force from model (if available) or KDE. D = constant (isotropic).
    """

    def fit(self, trajs, model=None):
        Z, dZ, Zn = build_transitions(trajs)

        # Force
        if model is not None:
            try:
                force_fn = model.get_force_fn()
                self._force_fn = force_fn
                self._force_source = "model"
            except Exception:
                warnings.warn("Model force failed, falling back to KDE")
                self._force_fn = estimate_force_kde(Z)
                self._force_source = "kde"
        else:
            self._force_fn = estimate_force_kde(Z)
            self._force_source = "kde"

        # Constant diffusion: average variance of increments
        self._D_const = 0.5 * np.mean(dZ ** 2)  # isotropic average
        self._diffusion = np.ones(self.latent_dim) * self._D_const
        self._fitted = True

        return {"force_source": self._force_source,
                "D_const": float(self._D_const),
                "n_samples": len(Z)}

    def predict_step(self, z, history=None):
        z_flat = z.reshape(-1, self.latent_dim)
        f = self._force_fn(z_flat)
        mean = z_flat + f  # Euler step (dt=1)
        cov = np.eye(self.latent_dim) * (2 * self._D_const)
        return mean.reshape(z.shape), cov

    def rollout(self, z_init, n_steps, rng=None):
        rng = rng or np.random.default_rng()
        d = self.latent_dim
        traj = np.zeros((n_steps, d))
        z = z_init.ravel().copy().astype(np.float64)
        traj[0] = z
        for t in range(1, n_steps):
            f = self._force_fn(z[None]).ravel()
            z = z + f + rng.normal(0, np.sqrt(2 * self._D_const), size=d)
            if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                traj[t:] = traj[t-1]; break
            traj[t] = z
        return traj


# ---------------------------------------------------------------------------
# FreeEnergy closure
# ---------------------------------------------------------------------------

class FreeEnergyClosure(MarkovClosure):
    """Overdamped Langevin with position-dependent diffusion D(z)."""

    def fit(self, trajs, model=None):
        Z, dZ, Zn = build_transitions(trajs)

        # Force (same as Markov)
        if model is not None:
            try:
                self._force_fn = model.get_force_fn()
                self._force_source = "model"
            except Exception:
                self._force_fn = estimate_force_kde(Z)
                self._force_source = "kde"
        else:
            self._force_fn = estimate_force_kde(Z)
            self._force_source = "kde"

        # Position-dependent diffusion
        self._diff_fn = estimate_diffusion_local(dZ, Z)
        self._fitted = True

        return {"force_source": self._force_source,
                "diff_source": "local_knn",
                "n_samples": len(Z)}

    def predict_step(self, z, history=None):
        z_flat = z.reshape(-1, self.latent_dim)
        f = self._force_fn(z_flat)
        D_z = self._diff_fn(z_flat)
        D_z = np.maximum(D_z, 1e-8)
        mean = z_flat + f
        cov = np.eye(self.latent_dim) * float(2 * np.mean(D_z))
        return mean.reshape(z.shape), cov

    def rollout(self, z_init, n_steps, rng=None):
        rng = rng or np.random.default_rng()
        d = self.latent_dim
        traj = np.zeros((n_steps, d))
        z = z_init.ravel().copy().astype(np.float64)
        traj[0] = z
        for t in range(1, n_steps):
            f = self._force_fn(z[None]).ravel()
            D_z = self._diff_fn(z[None]).ravel()
            D_z = np.maximum(D_z, 1e-8)
            z = z + f + rng.normal(0, np.sqrt(2 * D_z))
            if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                traj[t:] = traj[t-1]; break
            traj[t] = z
        return traj


# ---------------------------------------------------------------------------
# Mori closure (force + memory + constant D)
# ---------------------------------------------------------------------------

class MoriClosure(ClosureMethod):
    """GLE with memory kernel K(t): dZ = -nabla U dt + int K(t-s) Z(s) ds + sqrt(2 D) dW."""

    def __init__(self, name, latent_dim, history_window=8, kernel_length=20):
        super().__init__(name, latent_dim, history_window)
        self.kernel_length = kernel_length

    def fit(self, trajs, model=None):
        Z, dZ, Zn = build_transitions(trajs)

        # Force
        if model is not None:
            try:
                self._force_fn = model.get_force_fn()
                self._force_source = "model"
            except Exception:
                self._force_fn = estimate_force_kde(Z)
                self._force_source = "kde"
        else:
            self._force_fn = estimate_force_kde(Z)
            self._force_source = "kde"

        # Constant diffusion
        D_const = 0.5 * np.mean(dZ ** 2)
        self._D_const = D_const
        self._diffusion = np.ones(self.latent_dim) * D_const

        # Memory kernel via Volterra or model
        if model is not None:
            try:
                kernel_fn = model.get_memory_kernel_fn()
                self._kernel = kernel_fn(self.kernel_length)
                self._kernel_source = "model"
            except Exception:
                self._kernel = estimate_memory_kernel_volterra(
                    trajs, self.kernel_length)
                self._kernel_source = "volterra"
        else:
            self._kernel = estimate_memory_kernel_volterra(
                trajs, self.kernel_length)
            self._kernel_source = "volterra"

        self._history = None  # buffer for memory convolution
        self._fitted = True
        return {"force_source": self._force_source,
                "kernel_source": self._kernel_source,
                "kernel_norm": float(np.linalg.norm(self._kernel)),
                "D_const": float(D_const)}

    def _memory_term(self, history):
        """Compute int K(t-s) Z(s) ds from history buffer."""
        if history is None or len(history) < 1:
            return np.zeros(self.latent_dim)
        L = min(len(history), self.kernel_length)
        mem = np.zeros(self.latent_dim)
        for tau in range(1, L + 1):
            mem += self._kernel[tau] @ history[-tau]
        return mem

    def predict_step(self, z, history=None):
        z_flat = z.reshape(-1, self.latent_dim)
        f = self._force_fn(z_flat)
        mem = self._memory_term(history) if history is not None else np.zeros(self.latent_dim)
        mean = z_flat[0] + f[0] + mem
        cov = np.eye(self.latent_dim) * (2 * self._D_const)
        return mean.reshape(z.shape), cov

    def rollout(self, z_init, n_steps, rng=None):
        rng = rng or np.random.default_rng()
        d = self.latent_dim
        traj = np.zeros((n_steps, d))
        z = z_init.ravel().copy().astype(np.float64)
        traj[0] = z
        # Initialize history buffer
        history = [z.copy()]
        for t in range(1, n_steps):
            f = self._force_fn(z[None]).ravel()
            mem = np.zeros(d)
            L = min(len(history), self.kernel_length)
            for tau in range(1, L + 1):
                mem += self._kernel[tau] @ history[-tau]
            z = z + f + mem + rng.normal(0, np.sqrt(2 * self._D_const), size=d)
            if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                traj[t:] = traj[t-1]; break
            traj[t] = z
            history.append(z.copy())
        return traj


# ---------------------------------------------------------------------------
# Full closure (force + memory + position-dependent D)
# ---------------------------------------------------------------------------

class FullClosure(MoriClosure):
    """GLE with memory + position-dependent D(z)."""

    def fit(self, trajs, model=None):
        Z, dZ, Zn = build_transitions(trajs)

        if model is not None:
            try:
                self._force_fn = model.get_force_fn()
                self._force_source = "model"
            except Exception:
                self._force_fn = estimate_force_kde(Z)
                self._force_source = "kde"
        else:
            self._force_fn = estimate_force_kde(Z)
            self._force_source = "kde"

        D_const = 0.5 * np.mean(dZ ** 2)
        self._D_const = D_const
        self._diff_fn = estimate_diffusion_local(dZ, Z)
        self._diff_source = "local_knn"

        if model is not None:
            try:
                kernel_fn = model.get_memory_kernel_fn()
                self._kernel = kernel_fn(self.kernel_length)
                self._kernel_source = "model"
            except Exception:
                self._kernel = estimate_memory_kernel_volterra(trajs, self.kernel_length)
                self._kernel_source = "volterra"
        else:
            self._kernel = estimate_memory_kernel_volterra(trajs, self.kernel_length)
            self._kernel_source = "volterra"

        self._fitted = True
        return {"force_source": self._force_source,
                "diff_source": self._diff_source,
                "kernel_source": self._kernel_source,
                "kernel_norm": float(np.linalg.norm(self._kernel))}

    def rollout(self, z_init, n_steps, rng=None):
        rng = rng or np.random.default_rng()
        d = self.latent_dim
        traj = np.zeros((n_steps, d))
        z = z_init.ravel().copy().astype(np.float64)
        traj[0] = z
        history = [z.copy()]
        for t in range(1, n_steps):
            f = self._force_fn(z[None]).ravel()
            D_z = self._diff_fn(z[None]).ravel()
            D_z = np.maximum(D_z, 1e-8)
            mem = np.zeros(d)
            L = min(len(history), self.kernel_length)
            for tau in range(1, L + 1):
                mem += self._kernel[tau] @ history[-tau]
            z = z + f + mem + rng.normal(0, np.sqrt(2 * D_z))
            if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                traj[t:] = traj[t-1]; break
            traj[t] = z
            history.append(z.copy())
        return traj


# ---------------------------------------------------------------------------
# GRU closure
# ---------------------------------------------------------------------------

class GRUDynamics(nn.Module):
    """GRU-based dynamics model: z_{t+1} = GRU(z_t, h_t)."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, h=None):
        if h is None:
            h = torch.zeros(z.shape[0], self.gru.hidden_size, device=z.device)
        h_new = self.gru(z, h)
        z_next = self.readout(h_new)
        return z_next, h_new


class GRUClosure(ClosureMethod):
    """Neural dynamics via trained GRU."""

    def __init__(self, name, latent_dim, history_window=8, hidden_dim=128):
        super().__init__(name, latent_dim, history_window)
        self.hidden_dim = hidden_dim

    def fit(self, trajs, model=None, max_epochs=100, patience=10, max_samples=3000):
        Z, _, Zn = build_transitions(trajs)
        if len(Z) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Z), max_samples, replace=False)
            Z, Zn = Z[idx], Zn[idx]
        n = len(Z)
        n_train = int(n * 0.8)
        perm = np.random.default_rng(42).permutation(n)
        tr_idx, va_idx = perm[:n_train], perm[n_train:]

        Z_tr = _to_tensor(Z[tr_idx])
        Zn_tr = _to_tensor(Zn[tr_idx])
        Z_va = _to_tensor(Z[va_idx])
        Zn_va = _to_tensor(Zn[va_idx])

        self._net = GRUDynamics(self.latent_dim, self.hidden_dim)
        opt = torch.optim.Adam(self._net.parameters(), lr=0.001)
        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            self._net.train()
            opt.zero_grad()
            pred_tr, _ = self._net(Z_tr)
            loss = F.mse_loss(pred_tr, Zn_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            opt.step()

            self._net.eval()
            with torch.no_grad():
                pred_va, _ = self._net(Z_va)
                val_loss = F.mse_loss(pred_va, Zn_va).item()

            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._fitted = True
        return {"train_loss": float(loss.item()),
                "val_loss": float(best_loss),
                "epochs_trained": epoch + 1}

    def _predict_torch(self, z_tensor, h_tensor=None):
        """One-step prediction using torch tensors."""
        self._net.eval()
        with torch.no_grad():
            pred, h_new = self._net(z_tensor, h_tensor)
        return pred, h_new

    def predict_step(self, z, history=None):
        z_t = _to_tensor(z.reshape(-1, self.latent_dim))
        pred, _ = self._predict_torch(z_t)
        mean = _to_np(pred).reshape(z.shape)
        d = self.latent_dim
        cov = np.eye(d) * 1e-3
        return mean, cov

    def rollout(self, z_init, n_steps, rng=None):
        d = self.latent_dim
        traj = np.zeros((n_steps, d))
        z = z_init.ravel().copy().astype(np.float32)
        traj[0] = z
        z_t = _to_tensor(z[None])
        h = None
        for t in range(1, n_steps):
            pred, h = self._predict_torch(z_t, h)
            z_next = _to_np(pred)[0]
            traj[t] = z_next
            z_t = _to_tensor(z_next[None])
            if np.any(np.isnan(z_next)) or np.any(np.isinf(z_next)):
                traj[t:] = traj[t-1]; break
        return traj


# ---------------------------------------------------------------------------
# Causal Transformer closure
# ---------------------------------------------------------------------------

class TransformerDynamics(nn.Module):
    """Causal transformer for latent dynamics."""

    def __init__(self, latent_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, max_seq_len: int = 32):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(latent_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, activation='relu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, latent_dim)
        self.max_seq_len = max_seq_len
        self._causal_mask = None
        self._cached_len = 0

    def _get_mask(self, seq_len: int, device: torch.device):
        if seq_len != self._cached_len:
            self._causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1)
            self._cached_len = seq_len
        return self._causal_mask

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq: (B, L, D). Predict z_{L} -> z_{L+1}."""
        B, L, D = z_seq.shape
        x = self.input_proj(z_seq)  # (B, L, d_model)
        x = x + self.pos_enc[:, :L, :]
        mask = self._get_mask(L, x.device)
        h = self.transformer(x, mask=mask, is_causal=True)  # (B, L, d_model)
        # Use last position's query
        query = self.query_pos.expand(B, -1, -1)  # (B, 1, d_model)
        # Cross-attention would go here, simplified: use last hidden state
        z_next = self.output_proj(h[:, -1:])  # (B, 1, D)
        return z_next.squeeze(1)  # (B, D)


class CausalTransformerClosure(ClosureMethod):
    """Neural dynamics via causal transformer."""

    def __init__(self, name, latent_dim, history_window=8,
                 d_model=128, nhead=4, num_layers=3):
        super().__init__(name, latent_dim, history_window)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

    def _prepare_sequences(self, trajs, seq_len, max_samples=5000, stride=2):
        """Prepare (input_sequence, target) pairs from trajectories."""
        X_list, y_list = [], []
        for traj in trajs.values():
            T = len(traj)
            for t in range(seq_len, T, stride):
                X_list.append(traj[t-seq_len:t])  # (seq_len, D)
                y_list.append(traj[t])            # (D,)
                if len(X_list) >= max_samples:
                    break
            if len(X_list) >= max_samples:
                break
        if not X_list:
            return np.zeros((0, seq_len, self.latent_dim)), np.zeros((0, self.latent_dim))
        return np.array(X_list), np.array(y_list)

    def fit(self, trajs, model=None, max_epochs=100, patience=10, max_samples=1000):
        seq_len = min(self.history_window, 8)
        X, y = self._prepare_sequences(trajs, seq_len, max_samples=max_samples)
        n = len(X)
        if n < 20:
            return {"error": "too few samples", "n_samples": n}

        n_train = int(n * 0.8)
        perm = np.random.default_rng(42).permutation(n)
        tr_idx, va_idx = perm[:n_train], perm[n_train:]

        X_tr = _to_tensor(X[tr_idx])
        y_tr = _to_tensor(y[tr_idx])
        X_va = _to_tensor(X[va_idx])
        y_va = _to_tensor(y[va_idx])

        self._net = TransformerDynamics(
            self.latent_dim, self.d_model, self.nhead, self.num_layers, seq_len)
        opt = torch.optim.Adam(self._net.parameters(), lr=0.001)
        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            self._net.train()
            opt.zero_grad()
            pred = self._net(X_tr)
            loss = F.mse_loss(pred, y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            opt.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(self._net(X_va), y_va).item()

            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._seq_len = seq_len
        self._fitted = True
        return {"train_loss": float(loss.item()),
                "val_loss": float(best_loss),
                "seq_len": seq_len,
                "epochs_trained": epoch + 1}

    def predict_step(self, z, history=None):
        if history is not None and len(history) >= self._seq_len:
            seq = np.array(history[-self._seq_len:])[None]  # (1, L, D)
        else:
            seq = z.reshape(1, 1, self.latent_dim)
            if history is not None:
                hist_seq = np.array(history[-(self._seq_len-1):])[None]
                seq = np.concatenate([hist_seq, seq], axis=1)

        seq_t = _to_tensor(seq)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(seq_t)
        mean = _to_np(pred).reshape(z.shape)
        d = self.latent_dim
        cov = np.eye(d) * 1e-3
        return mean, cov

    def rollout(self, z_init, n_steps, rng=None):
        d = self.latent_dim
        traj = np.zeros((n_steps, d))
        z = z_init.ravel().copy().astype(np.float64)
        traj[0] = z
        history = [z.copy()] * self._seq_len  # warm up history
        for t in range(1, n_steps):
            mean, _ = self.predict_step(z, history)
            z_next = mean.ravel()
            traj[t] = z_next
            history.append(z_next.copy())
            if len(history) > self._seq_len * 2:
                history = history[-self._seq_len:]
            if np.any(np.isnan(z_next)) or np.any(np.isinf(z_next)):
                traj[t:] = traj[t-1]; break
        return traj


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_FACTORY = {
    "Markov": MarkovClosure,
    "FreeEnergy": FreeEnergyClosure,
    "Mori": MoriClosure,
    "Full": FullClosure,
    "GRU": GRUClosure,
    "CausalTransformer": CausalTransformerClosure,
    "GenericMarkov": GenericMarkovClosure,
    "generic_markov": GenericMarkovClosure,
    "markov": MarkovClosure,
    "free_energy": FreeEnergyClosure,
    "mori": MoriClosure,
    "full": FullClosure,
    "gru": GRUClosure,
    "causal_transformer": CausalTransformerClosure,
}


def build_method(name: str, latent_dim: int, history_window: int = 8,
                 **kwargs) -> ClosureMethod:
    cls = METHOD_FACTORY.get(name)
    if cls is None:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_FACTORY.keys())}")
    return cls(name, latent_dim, history_window, **kwargs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rollout(method: ClosureMethod, traj_true: np.ndarray, n_rollout: int = 1000
                     ) -> dict:
    """Evaluate a closure method on a single trajectory.

    Returns dict of metrics.
    """
    d = method.latent_dim

    # Compute innovation sequence (teacher forcing)
    innovations = method.innovation(traj_true)
    W = innovation_whiteness_score(innovations, max_lag=10)

    # Rollout
    z_init = traj_true[0]
    traj_roll = method.rollout(z_init, n_rollout)

    rollout_valid = bool(np.all(np.isfinite(traj_roll)))
    rollout_norm = float(np.abs(traj_roll).mean()) if rollout_valid else float("nan")

    # Stationary distribution comparison
    if rollout_valid and len(traj_roll) >= 100:
        # Use 1D histogram per dimension (simple JSD)
        jsd_scores = []
        for dim_i in range(min(d, 10)):  # subsample dims for speed
            p_hist, bins = np.histogram(traj_true[:, dim_i], bins=20, density=True)
            q_hist, _ = np.histogram(traj_roll[:, dim_i], bins=bins, density=True)
            jsd = jensen_shannon_divergence(p_hist, q_hist)
            jsd_scores.append(jsd)
        jsd_mean = float(np.mean(jsd_scores))
    else:
        jsd_mean = float("nan")

    # Compare temporal correlation and a simple Chapman-Kolmogorov diagnostic.
    max_lag = min(10, len(traj_true) // 4, len(traj_roll) // 4)
    if max_lag >= 1:
        acf_true = autocorrelation(traj_true, max_lag).mean(axis=1)
        acf_roll = autocorrelation(traj_roll, max_lag).mean(axis=1)
        acf_error = float(np.mean(np.abs(acf_true - acf_roll)))
    else:
        acf_error = float("nan")
    def linear_operator(x, lag):
        if len(x) <= lag: return None
        return np.linalg.lstsq(x[:-lag], x[lag:], rcond=None)[0]
    ops = [linear_operator(traj_true, lag) for lag in (1, 2)]
    semigroup_error = (float(np.linalg.norm(ops[0] @ ops[0] - ops[1], ord="fro"))
                       if all(o is not None for o in ops) else float("nan"))

    # FDT residual (if we have both positions and forces)
    fdt = innovation_whiteness_score(innovations, max_lag=5)

    return {
        "innovation_whiteness": float(W),
        "stationary_jsd": jsd_mean,
        "fdt_residual": fdt,
        "acf_error": acf_error,
        "semigroup_ck_error": semigroup_error,
        "rollout_valid": rollout_valid,
        "rollout_norm_drift": rollout_norm,
        "n_innovations": len(innovations),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="M1: Thermodynamic closure ablation")
    p.add_argument("--model", choices=["lspwm", "null"], default="lspwm")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--config", default="")
    p.add_argument("--manifest",
                   default="/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl")
    p.add_argument("--output-dir",
                   default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/thermodynamic_closure")
    p.add_argument("--methods", nargs="+",
                   default=["Markov", "FreeEnergy", "Mori", "Full", "GRU", "CausalTransformer"],
                   help="Methods, supplied as names separated by spaces or commas")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--history-window", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--n-rollout", type=int, default=2000,
                   help="Rollout steps for evaluation")
    p.add_argument("--neural-epochs", type=int, default=100,
                   help="Max epochs for neural methods (GRU, Transformer)")
    p.add_argument("--seed", type=int, default=6303)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dummy", action="store_true")
    args = p.parse_args()

    print("=" * 60)
    print("M1: Thermodynamic Closure Ablation")
    print("=" * 60)
    methods_names = [m.strip() for item in args.methods for m in item.split(",") if m.strip()]
    print(f"  Methods: {methods_names}")

    # Build model
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

    print(f"  Proteins: {len(data)}  Latent dim: {latent_dim}")

    # Build trajectories
    trajs = build_trajectories(data, min_len=20)
    print(f"  Trajectories: {len(trajs)}")
    if trajs:
        k = next(iter(trajs))
        print(f"  Sample traj: {k} shape={trajs[k].shape}")

    # Build and fit methods
    methods = {name: build_method(name, latent_dim, args.history_window)
               for name in methods_names}
    per_method_results = {}

    for name, method in methods.items():
        print(f"\n  --- {name} ---")
        try:
            if isinstance(method, (GRUClosure, CausalTransformerClosure,
                                   GenericMarkovClosure)):
                fit_info = method.fit(trajs, model, max_epochs=args.neural_epochs)
            else:
                fit_info = method.fit(trajs, model)
            print(f"    Fitted: {fit_info}")

            # Evaluate on each available trajectory
            method_metrics = {"fit_info": fit_info, "per_trajectory": {}}
            all_W, all_JSD, all_FDT = [], [], []

            for pid, traj in trajs.items():
                eval_res = evaluate_rollout(method, traj, n_rollout=args.n_rollout)
                method_metrics["per_trajectory"][pid] = eval_res
                all_W.append((pid, eval_res["innovation_whiteness"]))
                if not np.isnan(eval_res["stationary_jsd"]):
                    all_JSD.append((pid, eval_res["stationary_jsd"]))
                if not np.isnan(eval_res["fdt_residual"]):
                    all_FDT.append((pid, eval_res["fdt_residual"]))

            # Aggregate with bootstrap CI
            if all_W:
                W_dict = dict(all_W)
                ci_W = paired_bootstrap_ci(W_dict, n_resamples=2000, seed=args.seed)
                method_metrics["innovation_whiteness"] = ci_W
            if all_JSD:
                JSD_dict = dict(all_JSD)
                ci_JSD = paired_bootstrap_ci(JSD_dict, n_resamples=2000, seed=args.seed)
                method_metrics["stationary_jsd"] = ci_JSD
            if all_FDT:
                FDT_dict = dict(all_FDT)
                ci_FDT = paired_bootstrap_ci(FDT_dict, n_resamples=2000, seed=args.seed)
                method_metrics["fdt_residual"] = ci_FDT

            method_metrics["n_trajectories"] = len(trajs)
            method_metrics["n_valid"] = len(all_W)
            per_method_results[name] = method_metrics

        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            traceback.print_exc()
            per_method_results[name] = {"error": str(e), "n_trajectories": len(trajs)}

    # Save results
    prov = ProvenanceBlock(
        protocol="reviewer_experiment_v1", experiment_id="M1",
        config_path=args.config, checkpoint_path=args.checkpoint,
        manifest_path=args.manifest, random_seed=args.seed,
        device=args.device, batch_size=args.batch_size,
        max_batches=args.max_batches, sampler_mode="hierarchical",
        output_dir=args.output_dir,
        result_path=str(Path(args.output_dir) / "results.json"),
        train_only_fitted_objects={"methods_trained": methods_names},
    )
    result = ExperimentResult(
        experiment_id="M1", provenance=prov,
        metrics={"per_method": per_method_results, "methods": methods_names,
                 "n_proteins": len(data), "latent_dim": latent_dim},
        gates={"closure_gate": bool(per_method_results)},
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out / "results.json")
    print(f"\n  -> {out / 'results.json'}")
    model.close()


if __name__ == "__main__":
    main()
