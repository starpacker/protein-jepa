#!/usr/bin/env python3
"""Strict M1 thermodynamic-closure evaluation.

This evaluator is intentionally separate from the historical M1 script.  It
uses deterministic trajectory windows, a temporal train/test split, method-
specific teacher-forced innovations, and protein-level bootstrap intervals.
There is no dependency or model fallback: a missing native model component is
an error and invalidates the run.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from artifacts.reviewer_experiments._shared.lspwm_adapter import (  # noqa: E402
    LSPWMDynamicsModel,
)
from artifacts.reviewer_experiments._shared.latent_trajectories import (  # noqa: E402
    count_available_clips,
    extract_ordered_latent_trajectories,
    split_temporal,
)
from artifacts.reviewer_experiments._shared.metrics import (  # noqa: E402
    autocorrelation,
    bootstrap_compare,
    innovation_whiteness_score,
    jensen_shannon_divergence,
    paired_bootstrap_ci,
)
from protein_jepa_frame.data.manifest import load_manifest  # noqa: E402

M1_METRICS = (
    "prediction_mse",
    "innovation_whiteness",
    "fdt_residual",
    "stationary_jsd",
    "stationary_jsd_noise_floor",
    "stationary_jsd_excess_over_floor",
    "acf_error",
    "semigroup_mean_defect",
    "rollout_mean_norm",
)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def extract_native_q_trajectories(
    adapter: LSPWMDynamicsModel,
    manifest_path: str,
    *,
    batch_size: int,
    clip_frames: int,
    window_stride: int,
    max_batches: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    """Extract one native LSP-WM q state per ordered trajectory window.

    Delegates to ``_shared.latent_trajectories`` so that R1 and M1 evaluate the
    identical state representation and time grid.
    """
    if adapter.model is None or adapter.model.lspwm is None:
        raise RuntimeError("strict M1 requires a loaded LSP-WM model")
    return extract_ordered_latent_trajectories(
        adapter,
        manifest_path,
        batch_size=batch_size,
        clip_frames=clip_frames,
        window_stride=window_stride,
        max_batches=max_batches,
    )



def transitions(trajectories: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    current = [sequence[:-1] for sequence in trajectories.values()]
    future = [sequence[1:] for sequence in trajectories.values()]
    if not current:
        raise RuntimeError("no transitions available")
    return np.concatenate(current), np.concatenate(future)


def _covariance(values: np.ndarray, dimension: int) -> np.ndarray:
    if len(values) < 2:
        return np.eye(dimension, dtype=np.float64) * 1.0e-6
    cov = np.cov(values, rowvar=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    if cov.shape != (dimension, dimension):
        raise ValueError(f"covariance has shape {cov.shape}, expected {(dimension, dimension)}")
    cov = 0.5 * (cov + cov.T)
    eig_min = float(np.linalg.eigvalsh(cov).min())
    if eig_min < 1.0e-8:
        cov = cov + np.eye(dimension) * (1.0e-8 - eig_min)
    return cov


def estimate_local_diffusion(
    z: np.ndarray,
    residual: np.ndarray,
    dt_ps: float,
    *,
    n_neighbors: int = 32,
    max_points: int = 2000,
) -> Callable[[np.ndarray], np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    if len(z) < 4:
        raise RuntimeError("not enough training transitions for local diffusion")
    rng = np.random.default_rng(6303)
    indices = np.arange(len(z))
    if len(indices) > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)
    z_fit = z[indices]
    residual_fit = residual[indices]
    neighbors = NearestNeighbors(n_neighbors=min(n_neighbors, len(z_fit)))
    neighbors.fit(z_fit)

    def diffusion(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float64).reshape(-1, z.shape[1])
        _, nearest = neighbors.kneighbors(query, return_distance=True)
        local = residual_fit[nearest]
        variance = np.var(local, axis=1, ddof=1) / (2.0 * dt_ps)
        return np.maximum(variance, 1.0e-10)

    return diffusion


def fit_memory_kernel(
    trajectories: dict[str, np.ndarray],
    drift_fn: Callable[[np.ndarray], np.ndarray] | None,
    dt_ps: float,
    kernel_length: int,
    state_center: np.ndarray | None = None,
    *,
    ridge_grid: tuple[float, ...] = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0),
) -> tuple[np.ndarray, dict]:
    """Fit ``K_j`` in ``z_{t+1} - z_t = drift(z_t) + dt * sum_j K_j (z_{t-j} - c)``.

    Convention (shared by every memory method in this file):
    ``kernel[j]`` multiplies ``(z_{t-j} - center)`` for ``j = 0 .. L-1``.  The
    historical implementation used ``j = 1 .. L`` for the fitted kernel and
    ``j = 1 .. L`` of the *native* bath for ``Full`` while the native
    ``K(0)`` lag was never used, so ``Full`` and ``Mori`` were misaligned by one
    lag.  A single convention removes that mismatch.

    ``drift_fn`` supplies the already-dt-scaled drift increment (for example
    ``D F(z) dt``); it may be ``None`` for the pure-memory cell.

    The ridge strength is chosen by an **inner** split of the training prefixes
    (first 60% of each trajectory's rows for the fit, last 40% for validation),
    because an under-regularised Volterra solve overfits catastrophically (1024
    coefficients on collinear lagged latents) and would make the memory cells
    look worse than they are.  Returns ``(kernel, diagnostics)``.
    """
    if kernel_length < 1:
        raise ValueError("kernel_length must be positive")
    if state_center is None:
        state_center = np.mean(
            np.concatenate(list(trajectories.values()), axis=0), axis=0
        )
    state_center = np.asarray(state_center, dtype=np.float64).reshape(-1)
    features = []
    targets = []
    rows_per_trajectory = []
    for sequence in trajectories.values():
        if len(sequence) <= kernel_length:
            continue
        drift = np.zeros_like(sequence) if drift_fn is None else np.asarray(drift_fn(sequence))
        start = len(features)
        for index in range(kernel_length - 1, len(sequence) - 1):
            history = np.concatenate(
                [sequence[index - lag] - state_center for lag in range(kernel_length)]
            )
            observed_increment = sequence[index + 1] - sequence[index]
            target = (observed_increment - drift[index]) / dt_ps
            features.append(history)
            targets.append(target)
        if len(features) > start:
            rows_per_trajectory.append((start, len(features)))
    if len(features) < 4:
        raise RuntimeError("not enough training transitions for finite-memory fit")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    # Latent coordinates are strongly collinear.  A fixed tiny ridge makes
    # the high-dimensional Volterra solve ill-conditioned and can create an
    # unstable rollout even when the one-step fit looks finite.
    ridge_scale = max(float(np.trace(x.T @ x) / max(x.shape[1], 1)), 1.0e-12)

    # Inner split: never mix a trajectory's own future rows into the fit, so the
    # ridge is selected on genuinely held-out increments.
    inner_fit, inner_validation = [], []
    for start, stop in rows_per_trajectory:
        count = stop - start
        cut = start + max(1, int(0.6 * count))
        if cut >= stop:
            continue
        inner_fit.append(np.arange(start, cut))
        inner_validation.append(np.arange(cut, stop))
    diagnostics: dict = {"ridge_scale": ridge_scale, "n_rows": int(len(x))}
    if inner_fit and inner_validation:
        fit_rows = np.concatenate(inner_fit)
        validation_rows = np.concatenate(inner_validation)
        x_fit, y_fit = x[fit_rows], y[fit_rows]
        x_validation, y_validation = x[validation_rows], y[validation_rows]
        scores = {}
        for multiplier in ridge_grid:
            reg = multiplier * ridge_scale * np.eye(x.shape[1], dtype=np.float64)
            trial = np.linalg.solve(x_fit.T @ x_fit + reg, x_fit.T @ y_fit)
            scores[float(multiplier)] = float(
                np.mean(np.square(x_validation @ trial - y_validation)))
        best_multiplier = min(scores, key=scores.get)
        diagnostics.update({
            "ridge_multiplier": float(best_multiplier),
            "ridge_multiplier_source": "inner_validation_split",
            "inner_validation_mse": scores,
            "n_inner_fit_rows": int(len(fit_rows)),
            "n_inner_validation_rows": int(len(validation_rows)),
        })
    else:
        best_multiplier = 1.0e-2
        diagnostics.update({
            "ridge_multiplier": float(best_multiplier),
            "ridge_multiplier_source": "default_no_inner_validation",
            "inner_validation_mse": {},
            "n_inner_fit_rows": 0,
            "n_inner_validation_rows": 0,
        })

    reg = best_multiplier * ridge_scale * np.eye(x.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(x.T @ x + reg, x.T @ y)
    full_fit_mse = float(np.mean(np.square(x @ coefficients - y)))
    diagnostics["full_train_rate_mse"] = full_fit_mse
    dimension = y.shape[1]
    kernel = coefficients.reshape(kernel_length, dimension, dimension).transpose(0, 2, 1)
    return kernel, diagnostics


class StrictClosure:
    def __init__(self, name: str, dimension: int, dt_ps: float, history_window: int):
        self.name = name
        self.dimension = dimension
        self.dt_ps = float(dt_ps)
        self.history_window = int(history_window)
        self._fitted = False

    def fit(self, trajectories: dict[str, np.ndarray], adapter: LSPWMDynamicsModel) -> dict:
        raise NotImplementedError

    def predict_step(
        self, z: np.ndarray, history: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def innovation(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residuals = []
        expected_covariances = []
        history: list[np.ndarray] = []
        for index in range(len(sequence) - 1):
            mean, covariance = self.predict_step(sequence[index], history)
            residuals.append(sequence[index + 1] - mean)
            expected_covariances.append(covariance)
            history.append(sequence[index].copy())
            history = history[-self.history_window :]
        return np.asarray(residuals), np.asarray(expected_covariances)

    def rollout(
        self, initial: np.ndarray, steps: int, rng: np.random.Generator
    ) -> np.ndarray:
        output = np.zeros((steps, self.dimension), dtype=np.float64)
        output[0] = initial
        history = [initial.copy()]
        for index in range(1, steps):
            mean, covariance = self.predict_step(output[index - 1], history)
            covariance = 0.5 * (covariance + covariance.T)
            chol = np.linalg.cholesky(covariance + np.eye(self.dimension) * 1.0e-10)
            output[index] = mean + chol @ rng.normal(size=self.dimension)
            history.append(output[index - 1].copy())
            history = history[-self.history_window :]
            if not np.all(np.isfinite(output[index])):
                raise FloatingPointError(f"{self.name} rollout became non-finite")
        return output


class PhysicalClosure(StrictClosure):
    def __init__(self, name: str, dimension: int, dt_ps: float, history_window: int):
        super().__init__(name, dimension, dt_ps, history_window)
        self.force_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def _force(self, z: np.ndarray) -> np.ndarray:
        if self.force_fn is None:
            raise RuntimeError(f"{self.name} force requested before fit")
        return self.force_fn(np.asarray(z).reshape(-1, self.dimension))

class TwoByTwoClosure(PhysicalClosure):
    """One ablation cell of the ``{force} x {memory}`` design.

    The historical M1 method table claimed a free-energy/memory ablation but
    ``Markov`` and ``FreeEnergy`` shared ``native_lspwm_potential`` and no cell
    with "memory but no free energy" existed at all.  This class makes the
    design literal: the four cells differ *only* by ``use_force`` and
    ``use_memory``.

    Shared update rule (identical for every cell, and identical to
    ``fit_memory_kernel``'s feature convention)::

        z_{t+1} = z_t + s_f * D F(z_t) * dt
                       + dt * sum_{j=0}^{L-1} K_j (z_{t-j} - center)
                       + eps,      eps ~ N(0, Sigma)

    ``kernel[0]`` multiplies the *current* state and ``kernel[j]`` multiplies
    ``z_{t-j}`` taken from ``history``.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        dt_ps: float,
        history_window: int,
        kernel_length: int,
        *,
        use_force: bool,
        use_memory: bool,
        kernel_source: str = "fitted_ridge",
    ):
        super().__init__(name, dimension, dt_ps, history_window)
        if kernel_source not in {"fitted_ridge", "native_bath"}:
            raise ValueError(f"unknown kernel_source {kernel_source}")
        self.kernel_length = int(kernel_length)
        self.use_force = bool(use_force)
        self.use_memory = bool(use_memory)
        self.kernel_source = kernel_source
        self.force_scale = 0.0
        self.kernel: np.ndarray | None = None
        self.memory_center = np.zeros(dimension, dtype=np.float64)
        self.diffusion = 1.0
        self.force_norm_train = 0.0
        self.degenerate_zero_force = False
        # Placeholder so that ``innovation`` can be used during ``fit`` to
        # estimate this cell's own residual covariance.
        self.noise_covariance = np.eye(dimension, dtype=np.float64) * 1.0e-6

    def _drift_increment(self, sequence: np.ndarray) -> np.ndarray:
        """``s_f * D F(z) * dt`` per state, used by the kernel fit."""
        out = np.zeros_like(np.asarray(sequence, dtype=np.float64))
        if self.use_force and self.force_fn is not None and self.force_scale != 0.0:
            out = self._force(sequence) * (self.force_scale * self.dt_ps)
        return out

    def fit(self, trajectories, adapter):
        values = np.concatenate(list(trajectories.values()), axis=0)
        self.memory_center = np.mean(values, axis=0)
        observed = np.concatenate(
            [sequence[1:] - sequence[:-1] for sequence in trajectories.values()], axis=0
        )
        # Noise scale of the force-free, memory-free cell.  Every cell below
        # re-estimates its own innovation covariance from its own residuals.
        self.diffusion = max(float(np.mean(np.square(observed)) / (2.0 * self.dt_ps)), 1.0e-10)

        if self.use_force:
            self.force_fn = adapter.get_force_fn()
            z, future = transitions(trajectories)
            native_increment = self._force(z) * self.dt_ps
            numerator = float(np.sum(native_increment * (future - z)))
            denominator = float(np.sum(native_increment * native_increment))
            if denominator <= 1.0e-12:
                # A dead force head is a *finding about the checkpoint*, not a
                # reason to abort: report the cell with force_scale = 0 (it then
                # coincides with the force-free cell, which the structural
                # fingerprint check flags explicitly).
                self.force_scale = 0.0
                self.force_norm_train = 0.0
            else:
                self.force_scale = float(np.clip(numerator / denominator, 0.0, 10.0))
                self.force_norm_train = float(np.linalg.norm(native_increment, axis=1).mean())
            self.degenerate_zero_force = bool(self.force_scale <= 1.0e-12)
        else:
            self.force_fn = None
            self.force_norm_train = 0.0
            self.degenerate_zero_force = False

        if self.use_memory:
            if self.kernel_source == "native_bath":
                adapter.kernel_dt_ps = self.dt_ps
                native_kernel = adapter.get_memory_kernel_fn()(self.kernel_length)
                self.kernel = np.asarray(native_kernel[: self.kernel_length], dtype=np.float64)
                kernel_diagnostics = {
                    "ridge_multiplier": None,
                    "ridge_multiplier_source": "native_bath_not_fitted",
                }
            else:
                self.kernel, kernel_diagnostics = fit_memory_kernel(
                    trajectories, self._drift_increment, self.dt_ps,
                    self.kernel_length, state_center=self.memory_center,
                )
            if not np.all(np.isfinite(self.kernel)):
                raise FloatingPointError(f"{self.name} kernel is non-finite")
        else:
            self.kernel = None
            kernel_diagnostics = {}

        residuals = np.concatenate(
            [self.innovation(sequence)[0] for sequence in trajectories.values()], axis=0
        )
        self.noise_covariance = _covariance(residuals, self.dimension)
        self._fitted = True
        return {
            "design_cell": {"use_force": self.use_force, "use_memory": self.use_memory},
            "force_source": "native_lspwm_potential" if self.use_force else "none",
            "effective_force_source": (
                "native_lspwm_potential"
                if (self.use_force and not self.degenerate_zero_force) else "none"
            ),
            "force_scale": self.force_scale,
            "force_increment_norm_train": self.force_norm_train,
            "effective_drift_increment_norm": self.force_scale * self.force_norm_train,
            "degenerate_zero_force": self.degenerate_zero_force,
            "diffusion_source": "train_increment_variance",
            "diffusion_mean": self.diffusion,
            "kernel_source": self.kernel_source if self.use_memory else "none",
            "kernel_norm": float(np.linalg.norm(self.kernel)) if self.kernel is not None else 0.0,
            "kernel_diagnostics": kernel_diagnostics,
            "kernel_lag_relative_norm": (
                [float(np.linalg.norm(self.kernel[j])
                       / max(np.linalg.norm(self.kernel[0]), 1.0e-30))
                 for j in range(self.kernel_length)] if self.kernel is not None else []
            ),
            "train_innovation_mse": float(np.mean(np.square(residuals))),
            "n_train_transitions": int(len(observed)),
        }

    def predict_step(self, z, history=None):
        state = np.asarray(z, dtype=np.float64).reshape(1, -1)[0]
        mean = state.copy()
        if self.use_force and self.force_scale != 0.0:
            mean = mean + self._force(state.reshape(1, -1))[0] * (self.force_scale * self.dt_ps)
        if self.kernel is not None:
            memory = self.kernel[0] @ (state - self.memory_center)
            for lag in range(1, self.kernel_length):
                if history is None or len(history) < lag:
                    break
                memory = memory + self.kernel[lag] @ (history[-lag] - self.memory_center)
            mean = mean + memory * self.dt_ps
        return mean, self.noise_covariance
class NativeFullClosure(TwoByTwoClosure):
    """Native LSP-WM closure: learned potential, state-dependent D, native bath.

    Unlike the historical ``FullClosure`` this class (a) uses the same memory
    lag convention as every other method, (b) exposes per-lag kernel norms so
    the effective memory depth is auditable, and (c) never silently zeroes its
    force.
    """

    def __init__(self, name, dimension, dt_ps, history_window, kernel_length):
        super().__init__(
            name, dimension, dt_ps, history_window, kernel_length,
            use_force=True, use_memory=True, kernel_source="native_bath",
        )
        self.diffusion_fn_native = None
        self.drift_calibration = 1.0

    def _attach(self, adapter):
        """Bind the native diffusion head before any prediction is requested."""
        self.diffusion_fn_native = adapter.get_diffusion_fn()

    def _native_drift(self, z: np.ndarray) -> np.ndarray:
        diffusion = np.asarray(self.diffusion_fn_native(z), dtype=np.float64)
        force = self._force(z)
        if diffusion.ndim == 2:
            return diffusion * force
        if diffusion.ndim == 3:
            return np.einsum("bij,bj->bi", diffusion, force)
        raise ValueError(f"native diffusion has unsupported shape {diffusion.shape}")

    def fit(self, trajectories, adapter):
        self._attach(adapter)
        info = super().fit(trajectories, adapter)
        # The native force and diffusion are used at their own scale.  The
        # train-prefix calibration only reports how well that scale matches the
        # observed increments; it is never allowed to zero the drift silently.
        self.force_scale = 1.0
        z, future = transitions(trajectories)
        native_increment = self._native_drift(z) * self.dt_ps
        numerator = float(np.sum(native_increment * (future - z)))
        denominator = float(np.sum(native_increment * native_increment))
        self.drift_calibration = (
            float(np.clip(numerator / denominator, 0.0, 10.0)) if denominator > 1.0e-12 else 0.0
        )
        residuals = []
        for index in range(len(z)):
            mean, _ = self.predict_step(z[index], [])
            residuals.append(future[index] - mean)
        self.noise_covariance = _covariance(np.asarray(residuals), self.dimension)
        info.update({
            "diffusion_source": "native_lspwm_state_dependent",
            "force_scale": 1.0,
            "effective_force_source": (
                "native_lspwm_diffusion_weighted" if denominator > 1.0e-12 else "none"
            ),
            "drift_calibration_scale": self.drift_calibration,
            "effective_drift_increment_norm": float(np.linalg.norm(native_increment, axis=1).mean()),
            "native_drift_increment_norm": float(np.linalg.norm(native_increment, axis=1).mean()),
            "native_diffusion_mean": float(np.mean(np.asarray(self.diffusion_fn_native(z)))),
        })
        return info

    def predict_step(self, z, history=None):
        z_array = np.asarray(z, dtype=np.float64).reshape(1, -1)
        state = z_array[0]
        mean = state + self._native_drift(z_array)[0] * self.dt_ps
        if self.kernel is not None:
            memory = self.kernel[0] @ (state - self.memory_center)
            for lag in range(1, self.kernel_length):
                if history is None or len(history) < lag:
                    break
                memory = memory + self.kernel[lag] @ (history[-lag] - self.memory_center)
            mean = mean + memory * self.dt_ps
        return mean, self.noise_covariance

class CalibratedFullClosure(TwoByTwoClosure):
    """Native physical closure with train-only mobility calibration.

    The checkpoint supplies the potential force, state-dependent diffusion, and
    finite-memory bath; their numerical scales are calibrated on the training
    prefix because the learned latent coordinate has arbitrary normalization.
    The fit records ``mobility_scale`` and flags the cell as degenerate when the
    fitted mobility collapses to zero, because a zero mobility makes the
    equation ``mean = z`` (a pure random walk) while still producing the best
    semigroup defect of the whole method table.
    """

    def __init__(self, name, dimension, dt_ps, history_window, kernel_length):
        super().__init__(
            name, dimension, dt_ps, history_window, kernel_length,
            use_force=True, use_memory=True, kernel_source="native_bath",
        )
        self.diffusion_fn_native = None
        self.mobility_scale = 0.0
        self.degenerate_native_drift = False

    def _attach(self, adapter):
        """Bind the native diffusion head before any prediction is requested."""
        self.diffusion_fn_native = adapter.get_diffusion_fn()

    def _native_drift(self, z: np.ndarray) -> np.ndarray:
        diffusion = np.asarray(self.diffusion_fn_native(z), dtype=np.float64)
        force = self._force(z)
        if diffusion.ndim == 2:
            return diffusion * force
        if diffusion.ndim == 3:
            return np.einsum("bij,bj->bi", diffusion, force)
        raise ValueError(f"native diffusion has unsupported shape {diffusion.shape}")

    def fit(self, trajectories, adapter):
        self._attach(adapter)
        info = super().fit(trajectories, adapter)
        self.force_scale = 1.0
        z, future = transitions(trajectories)
        native_increment = self._native_drift(z) * self.dt_ps
        denominator = float(np.sum(native_increment * native_increment))
        if denominator <= 1.0e-12:
            # The checkpoint's diffusion head is numerically zero, so the native
            # drift has no learned direction to calibrate.  Record the degeneracy
            # (the cell collapses onto the memory-only cell) instead of aborting.
            mobility = 0.0
        else:
            mobility = float(np.sum(native_increment * (future - z)) / denominator)
        self.mobility_scale = float(np.clip(mobility, 0.0, 10.0))
        self.degenerate_native_drift = bool(denominator <= 1.0e-12)
        residuals = []
        for index in range(len(z)):
            mean, _ = self.predict_step(z[index], [])
            residuals.append(future[index] - mean)
        self.noise_covariance = _covariance(np.asarray(residuals), self.dimension)
        info.update({
            "diffusion_source": "native_lspwm_state_dependent",
            "force_scale": 1.0,
            "mobility_scale": self.mobility_scale,
            "degenerate_zero_mobility": bool(self.mobility_scale <= 1.0e-12),
            "degenerate_native_drift": self.degenerate_native_drift,
            "effective_force_source": (
                "none" if (self.degenerate_native_drift or self.mobility_scale <= 1.0e-12)
                else "native_lspwm_diffusion_weighted"
            ),
            "effective_drift_increment_norm": float(
                np.linalg.norm(self.mobility_scale * native_increment, axis=1).mean()),
            "calibrated_drift_increment_norm": float(
                np.linalg.norm(self.mobility_scale * native_increment, axis=1).mean()),
            "native_drift_increment_norm": float(np.linalg.norm(native_increment, axis=1).mean()),
        })
        return info

    def predict_step(self, z, history=None):
        z_array = np.asarray(z, dtype=np.float64).reshape(1, -1)
        state = z_array[0]
        mean = state + self.mobility_scale * self._native_drift(z_array)[0] * self.dt_ps
        if self.kernel is not None:
            memory = self.kernel[0] @ (state - self.memory_center)
            for lag in range(1, self.kernel_length):
                if history is None or len(history) < lag:
                    break
                memory = memory + self.kernel[lag] @ (history[-lag] - self.memory_center)
            mean = mean + memory * self.dt_ps
        return mean, self.noise_covariance


class TrainMeanClosure(StrictClosure):
    """Null predictor: emit the training mean state with the training variance.

    Included so that every reported metric has a value for a model with no
    dynamics at all.  Any method whose metric is not better than this row is not
    evidence of dynamics.
    """

    def fit(self, trajectories, adapter):
        del adapter
        values = np.concatenate(list(trajectories.values()), axis=0)
        self.mean_state = values.mean(axis=0)
        self.noise_covariance = _covariance(values - self.mean_state, self.dimension)
        self._fitted = True
        return {"architecture": "constant_train_mean",
                "effective_force_source": "none",
                "effective_drift_increment_norm": 0.0,
                "n_train_transitions": int(len(values))}

    def predict_step(self, z, history=None):
        del z, history
        return self.mean_state.copy(), self.noise_covariance



class GenericMarkovClosure(StrictClosure):
    def fit(self, trajectories, adapter, *, epochs: int, seed: int):
        del adapter
        z, future = transitions(trajectories)
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(self.dimension, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, self.dimension)
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1.0e-3)
        x = torch.from_numpy(z).float()
        y = torch.from_numpy(future).float()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(self.net(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            residual = y - self.net(x)
        self.noise_covariance = _covariance(residual.numpy(), self.dimension)
        self._fitted = True
        return {"architecture": "MLP(256,256)", "train_loss": float(loss.item()),
                "n_train_transitions": int(len(z)), "epochs": epochs}

    def predict_step(self, z, history=None):
        del history
        with torch.no_grad():
            mean = self.net(torch.from_numpy(np.asarray(z, dtype=np.float32)[None]))[0].numpy()
        return mean, self.noise_covariance


class GRUClosure(StrictClosure):
    def fit(self, trajectories, adapter, *, epochs: int, seed: int):
        del adapter
        torch.manual_seed(seed)
        self.net = nn.GRU(self.dimension, 128, batch_first=True)
        self.readout = nn.Linear(128, self.dimension)
        optimizer = torch.optim.Adam(
            list(self.net.parameters()) + list(self.readout.parameters()), lr=1.0e-3
        )
        x_rows = []
        y_rows = []
        for sequence in trajectories.values():
            for index in range(1, len(sequence)):
                history = sequence[max(0, index - self.history_window):index]
                if len(history) < self.history_window:
                    history = np.concatenate(
                        [np.repeat(history[:1], self.history_window - len(history), axis=0), history]
                    )
                x_rows.append(history)
                y_rows.append(sequence[index])
        x = torch.from_numpy(np.asarray(x_rows)).float()
        y = torch.from_numpy(np.asarray(y_rows)).float()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            hidden, _ = self.net(x)
            loss = nn.functional.mse_loss(self.readout(hidden[:, -1]), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.net.parameters()) + list(self.readout.parameters()), 1.0)
            optimizer.step()
        with torch.no_grad():
            residual = y - self.readout(self.net(x)[0][:, -1])
        self.noise_covariance = _covariance(residual.numpy(), self.dimension)
        self._fitted = True
        return {"architecture": "GRU(128)", "train_loss": float(loss.item()),
                "n_train_transitions": int(len(x_rows)), "epochs": epochs}

    def predict_step(self, z, history=None):
        values = list(history or []) + [np.asarray(z)]
        values = values[-self.history_window:]
        if len(values) < self.history_window:
            values = [values[0]] * (self.history_window - len(values)) + values
        x = torch.from_numpy(np.asarray(values, dtype=np.float32)[None])
        with torch.no_grad():
            mean = self.readout(self.net(x)[0][:, -1])[0].numpy()
        return mean, self.noise_covariance


class TransformerClosure(StrictClosure):
    def fit(self, trajectories, adapter, *, epochs: int, seed: int):
        del adapter
        torch.manual_seed(seed)
        self.net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dim_feedforward=256,
                dropout=0.0, batch_first=True, activation="gelu",
            ), num_layers=2,
        )
        self.input_proj = nn.Linear(self.dimension, 128)
        self.output_proj = nn.Linear(128, self.dimension)
        optimizer = torch.optim.Adam(
            list(self.net.parameters()) + list(self.input_proj.parameters()) + list(self.output_proj.parameters()),
            lr=1.0e-3,
        )
        x_rows = []
        y_rows = []
        for sequence in trajectories.values():
            for index in range(1, len(sequence)):
                history = sequence[max(0, index - self.history_window):index]
                if len(history) < self.history_window:
                    history = np.concatenate(
                        [np.repeat(history[:1], self.history_window - len(history), axis=0), history]
                    )
                x_rows.append(history)
                y_rows.append(sequence[index])
        x = torch.from_numpy(np.asarray(x_rows)).float()
        y = torch.from_numpy(np.asarray(y_rows)).float()
        mask = torch.triu(torch.ones(self.history_window, self.history_window, dtype=torch.bool), diagonal=1)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            h = self.net(self.input_proj(x), mask=mask)
            loss = nn.functional.mse_loss(self.output_proj(h[:, -1]), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + list(self.input_proj.parameters()) + list(self.output_proj.parameters()), 1.0
            )
            optimizer.step()
        with torch.no_grad():
            residual = y - self.output_proj(self.net(self.input_proj(x), mask=mask)[:, -1])
        self.noise_covariance = _covariance(residual.numpy(), self.dimension)
        self._fitted = True
        return {"architecture": "causal_transformer(128,2 layers)", "train_loss": float(loss.item()),
                "n_train_transitions": int(len(x_rows)), "epochs": epochs}

    def predict_step(self, z, history=None):
        values = list(history or []) + [np.asarray(z)]
        values = values[-self.history_window:]
        if len(values) < self.history_window:
            values = [values[0]] * (self.history_window - len(values)) + values
        x = torch.from_numpy(np.asarray(values, dtype=np.float32)[None])
        mask = torch.triu(torch.ones(self.history_window, self.history_window, dtype=torch.bool), diagonal=1)
        with torch.no_grad():
            mean = self.output_proj(self.net(self.input_proj(x), mask=mask)[:, -1])[0].numpy()
        return mean, self.noise_covariance


def _stationary_jsd(reference: np.ndarray, generated: np.ndarray,
                    dimensions: int = 64, bins: int = 21) -> float:
    values = []
    for dimension in range(min(reference.shape[1], dimensions)):
        low = min(float(reference[:, dimension].min()), float(generated[:, dimension].min()))
        high = max(float(reference[:, dimension].max()), float(generated[:, dimension].max()))
        if not high > low:
            continue
        edges = np.linspace(low, high, bins)
        p, _ = np.histogram(reference[:, dimension], bins=edges)
        q, _ = np.histogram(generated[:, dimension], bins=edges)
        values.append(jensen_shannon_divergence(p, q))
    return float(np.mean(values)) if values else float("nan")


def _stationary_jsd_noise_floor(reference: np.ndarray, dimensions: int = 64,
                                bins: int = 21) -> float:
    """JSD between the two halves of the *reference* trajectory.

    The historical script compared a handful of reference frames against a
    256-step rollout, so any nominal JSD was dominated by finite-sample
    histogram noise.  Reporting the reference-vs-reference JSD makes that noise
    floor explicit: a method whose JSD is at or below this value has not been
    shown to reproduce the stationary distribution.
    """
    half = len(reference) // 2
    if half < 4:
        return float("nan")
    return _stationary_jsd(reference[:half], reference[half:], dimensions, bins)


def evaluate_one(method: StrictClosure, sequence: np.ndarray, rollout_steps: int, seed: int) -> dict:
    innovations, expected = method.innovation(sequence)
    predicted_mse = float(np.mean(np.square(innovations)))
    whiteness = float(innovation_whiteness_score(innovations, max_lag=min(3, len(innovations) - 2)))
    empirical_cov = _covariance(innovations, method.dimension)
    expected_cov = np.mean(expected, axis=0)
    fdt = float(
        np.linalg.norm(empirical_cov - expected_cov, ord="fro")
        / max(np.linalg.norm(expected_cov, ord="fro"), 1.0e-12)
    )

    rng = np.random.default_rng(seed)
    generated = method.rollout(sequence[0], rollout_steps, rng)
    if not np.all(np.isfinite(generated)):
        raise FloatingPointError(f"{method.name} rollout is non-finite")
    jsd = _stationary_jsd(sequence, generated)
    jsd_floor = _stationary_jsd_noise_floor(sequence)
    max_lag = min(5, len(sequence) // 3, len(generated) // 3)
    acf_error = float(
        np.mean(np.abs(autocorrelation(sequence, max_lag).mean(axis=1)
                       - autocorrelation(generated, max_lag).mean(axis=1)))
    )

    one_step = []
    two_step = []
    states = []
    for index in range(len(sequence) - 2):
        history = [sequence[j].copy() for j in range(max(0, index - method.history_window + 1), index)]
        first, _ = method.predict_step(sequence[index], history)
        second_history = history + [sequence[index].copy()]
        second, _ = method.predict_step(first, second_history)
        states.append(sequence[index])
        one_step.append(first)
        two_step.append(second)
    states = np.asarray(states)
    one_step = np.asarray(one_step)
    two_step = np.asarray(two_step)
    # Affine (homogeneous-coordinate) operators so that composition is
    # well defined.  The historical version fitted bias-free linear operators,
    # so a constant offset in the conditional mean was charged to the defect.
    ones = np.ones((len(states), 1))
    augmented_states = np.concatenate([states, ones], axis=1)
    augmented_one = np.concatenate([one_step, ones], axis=1)
    augmented_two = np.concatenate([two_step, ones], axis=1)
    t1 = np.linalg.lstsq(augmented_states, augmented_one, rcond=None)[0]
    t2 = np.linalg.lstsq(augmented_states, augmented_two, rcond=None)[0]
    ck_defect = float(
        np.linalg.norm(t1 @ t1 - t2, ord="fro")
        / max(np.linalg.norm(t2, ord="fro"), 1.0e-12)
    )
    return {
        "prediction_mse": predicted_mse,
        "innovation_whiteness": whiteness,
        "fdt_residual": fdt,
        "stationary_jsd": jsd,
        "stationary_jsd_noise_floor": jsd_floor,
        "stationary_jsd_excess_over_floor": (
            float(jsd - jsd_floor) if np.isfinite(jsd) and np.isfinite(jsd_floor) else float("nan")
        ),
        "acf_error": acf_error,
        "semigroup_mean_defect": ck_defect,
        "rollout_mean_norm": float(np.linalg.norm(generated, axis=1).mean()),
        "rollout_valid": True,
        "n_test_transitions": int(len(innovations)),
    }



def _aggregate_by_protein(per_trajectory: dict[str, dict], metadata: dict[str, dict]) -> dict[str, dict]:
    by_protein: dict[str, dict[str, list[float]]] = {}
    for trajectory_id, values in per_trajectory.items():
        protein_id = metadata[trajectory_id]["protein_id"]
        target = by_protein.setdefault(protein_id, {})
        for key, value in values.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                target.setdefault(key, []).append(float(value))
    return {
        protein_id: {key: float(np.mean(values)) for key, values in metrics.items()}
        for protein_id, metrics in by_protein.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict M1 model/baseline comparison")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--kernel-length", type=int, default=4)
    parser.add_argument("--neural-epochs", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=6303)
    args = parser.parse_args()

    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adapter = LSPWMDynamicsModel()
    adapter.load(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        manifest=args.manifest,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=1,
        seed=args.seed,
    )
    trajectories, metadata, loader_stats = extract_native_q_trajectories(
        adapter,
        args.manifest,
        batch_size=args.batch_size,
        clip_frames=args.clip_frames,
        window_stride=args.window_stride,
        max_batches=args.max_batches or None,
    )
    uncapped = count_available_clips(
        args.manifest, clip_frames=args.clip_frames, window_stride=args.window_stride,
    )
    train, test = split_temporal(trajectories)
    dimension = next(iter(train.values())).shape[1]
    dt_values = {round(float(value["dt_ps"]), 6) for value in metadata.values()}
    if len(dt_values) != 1:
        raise RuntimeError(f"strict M1 requires one common sampled dt, found {sorted(dt_values)}")
    dt_ps = float(next(iter(dt_values)))
    print(f"Extracted {len(trajectories)} trajectories, {len(train)} train, {len(test)} test, dt_ps={dt_ps}")
    if loader_stats["truncated_by_max_batches"]:
        print(
            f"[M1v2] WARNING: --max-batches {args.max_batches} truncated the evaluation set to "
            f"{loader_stats['clips_seen']} clips ({len(trajectories)} trajectories, "
            f"{loader_stats['n_proteins']} proteins) out of an estimated "
            f"{uncapped['est_clips']} clips ({uncapped['est_trajectories']} trajectories)"
        )

    methods: dict[str, StrictClosure] = {
        # Complete {force} x {memory} ablation.  ``M_None`` is the null cell the
        # historical table never contained, and ``M_Memory`` is the
        # "memory but no free energy" cell that was previously unimplemented.
        "TrainMean": TrainMeanClosure("TrainMean", dimension, dt_ps, args.history_window),
        "M_None": TwoByTwoClosure(
            "M_None", dimension, dt_ps, args.history_window, args.kernel_length,
            use_force=False, use_memory=False),
        "M_Force": TwoByTwoClosure(
            "M_Force", dimension, dt_ps, args.history_window, args.kernel_length,
            use_force=True, use_memory=False),
        "M_Memory": TwoByTwoClosure(
            "M_Memory", dimension, dt_ps, args.history_window, args.kernel_length,
            use_force=False, use_memory=True),
        "M_Full": TwoByTwoClosure(
            "M_Full", dimension, dt_ps, args.history_window, args.kernel_length,
            use_force=True, use_memory=True),
        # Native model closure variants.
        "Full": NativeFullClosure("Full", dimension, dt_ps, args.history_window, args.kernel_length),
        "FullCalibrated": CalibratedFullClosure(
            "FullCalibrated", dimension, dt_ps, args.history_window, args.kernel_length
        ),
        "GenericMarkov": GenericMarkovClosure("GenericMarkov", dimension, dt_ps, args.history_window),
        "GRU": GRUClosure("GRU", dimension, dt_ps, args.history_window),
        "CausalTransformer": TransformerClosure("CausalTransformer", dimension, dt_ps, args.history_window),
    }

    def _fit(method):
        if isinstance(method, (GenericMarkovClosure, GRUClosure, TransformerClosure)):
            return method.fit(train, adapter, epochs=args.neural_epochs, seed=args.seed)
        return method.fit(train, adapter)

    results: dict[str, dict] = {}
    failures: dict[str, int] = {}
    for method_name, method in methods.items():
        print(f"Fitting {method_name}")
        fit_info = _fit(method)
        per_trajectory = {}
        for trajectory_id, sequence in test.items():
            try:
                per_trajectory[trajectory_id] = evaluate_one(method, sequence, args.rollout_steps, args.seed)
            except Exception as e:
                failures[method_name] = failures.get(method_name, 0) + 1
                print(f"  WARNING: {method_name} failed on {trajectory_id}: {e}")
                continue
        per_protein = _aggregate_by_protein(per_trajectory, metadata)
        summary = {}
        for metric in M1_METRICS:
            values = {protein: row[metric] for protein, row in per_protein.items() if metric in row}
            summary[metric] = paired_bootstrap_ci(values, n_resamples=2000, seed=args.seed)
        results[method_name] = {
            "fit_info": fit_info,
            "per_trajectory": per_trajectory,
            "per_protein": per_protein,
            "summary": summary,
            "n_test_trajectories": len(per_trajectory),
            "n_failed_trajectories": failures.get(method_name, 0),
            "n_test_proteins": len(per_protein),
        }

    comparisons = {}
    candidates = ("M_Full", "M_Memory", "M_Force", "Full", "FullCalibrated")
    baselines = ("M_None", "TrainMean", "GenericMarkov", "GRU", "CausalTransformer")
    for candidate in candidates:
        for baseline in baselines:
            if candidate == baseline:
                continue
            entry = {}
            for metric in M1_METRICS:
                a = {p: row[metric] for p, row in results[baseline]["per_protein"].items() if metric in row}
                b = {p: row[metric] for p, row in results[candidate]["per_protein"].items() if metric in row}
                entry[metric] = bootstrap_compare(a, b, n_resamples=2000, seed=args.seed)
            comparisons[f"{candidate}_vs_{baseline}"] = entry

    native_kernel = np.asarray(adapter.get_memory_kernel_fn()(args.kernel_length), dtype=np.float64)
    native_kernel_norm = float(np.linalg.norm(native_kernel[1:]))
    native_kernel_lag_relative = [
        float(np.linalg.norm(native_kernel[j]) / max(np.linalg.norm(native_kernel[0]), 1.0e-30))
        for j in range(len(native_kernel))
    ]
    effective_memory_lags = int(sum(1 for value in native_kernel_lag_relative[1:] if value > 0.05))
    native_kernel_ratio = (
        float(native_kernel_norm / max(np.linalg.norm(native_kernel[0]), 1.0e-30))
        if len(native_kernel) else 0.0
    )

    def _paired(candidate: str, baseline: str, metric: str) -> dict:
        """Signed bootstrap of ``candidate - baseline`` for one metric."""
        if candidate == baseline:
            return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
        a = {p: row[metric] for p, row in results[baseline]["per_protein"].items() if metric in row}
        b = {p: row[metric] for p, row in results[candidate]["per_protein"].items() if metric in row}
        return bootstrap_compare(a, b, n_resamples=2000, seed=args.seed)

    def _significant(result: dict) -> bool:
        return bool(np.isfinite(result["lower"]) and np.isfinite(result["upper"])
                    and (result["upper"] < 0.0 or result["lower"] > 0.0))

    # Real closure gate: the memory term must add predictive information beyond
    # the force term *and* the force term must add information beyond memory,
    # each with a protein-bootstrap interval that excludes zero.
    closure_gate_metric = "prediction_mse"
    gate_mem_over_force = _paired("M_Full", "M_Force", closure_gate_metric)
    gate_force_over_mem = _paired("M_Full", "M_Memory", closure_gate_metric)
    gate_full_over_none = _paired("M_Full", "M_None", closure_gate_metric)
    closure_gate_details = {
        "metric": closure_gate_metric,
        "memory_adds_over_force": gate_mem_over_force,
        "memory_adds_over_force_significant": _significant(gate_mem_over_force),
        "force_adds_over_memory": gate_force_over_mem,
        "force_adds_over_memory_significant": _significant(gate_force_over_mem),
        "full_vs_null": gate_full_over_none,
        "full_beats_null_significant": _significant(gate_full_over_none),
    }
    closure_gate = bool(
        closure_gate_details["memory_adds_over_force_significant"]
        and closure_gate_details["force_adds_over_memory_significant"]
        and closure_gate_details["full_beats_null_significant"]
        and gate_mem_over_force["mean"] < 0.0
        and gate_force_over_mem["mean"] < 0.0
    )

    # ── Persistence-degeneracy audit for the one-step MSE gate ──────────────
    # ``prediction_mse`` is the mean squared one-step innovation.  When the
    # per-step latent displacement is small (E[||dz||^2] tiny), the identity map
    # (persistence, i.e. mean = z_t) is the Bayes-optimal one-step predictor, so
    # ``prediction_mse`` structurally rewards a model that adds *no* dynamics:
    # the force-free / memory-free cell ``M_None`` then attains the lowest MSE
    # while having the worst stationary distribution.  Any closure that injects
    # real drift or memory necessarily *raises* one-step MSE even as it improves
    # the stationary law and innovation whiteness.  We therefore report whether
    # the MSE gate is in this degenerate regime and add a distributional closure
    # gate that is not confounded by it.
    m_none_mse = results["M_None"]["summary"]["prediction_mse"]
    m_none_mse_mean = m_none_mse["mean"] if isinstance(m_none_mse, dict) else float(m_none_mse)
    persistence_increment_var = 2.0 * float(
        results["M_None"]["fit_info"].get("diffusion_mean", 0.0)
    ) * float(dt_ps)
    mse_gate_degenerate = bool(
        "diff_mean" in gate_full_over_none
        and gate_full_over_none["diff_mean"] > 0.0  # Full has HIGHER one-step MSE than doing nothing
        and m_none_mse_mean <= 1.05 * max(persistence_increment_var, 1.0e-30)
    )

    # Distributional closure gate: a physical world model is judged on whether it
    # reproduces the stationary distribution and yields white innovations, not on
    # one-step MSE.  ``M_Full`` must beat the strongest unstructured neural
    # baseline present (GRU / CausalTransformer / GenericMarkov) on both
    # ``stationary_jsd`` and ``innovation_whiteness`` with protein-bootstrap
    # intervals that exclude zero.
    neural_baselines = [
        name for name in ("GRU", "CausalTransformer", "GenericMarkov")
        if name in results
    ]
    distributional_gate_details: dict[str, dict] = {}
    for metric in ("stationary_jsd", "innovation_whiteness"):
        per_baseline = {}
        for base in neural_baselines:
            cmp = _paired("M_Full", base, metric)
            per_baseline[base] = {
                **cmp,
                "full_better_significant": bool(
                    "diff_mean" in cmp and _significant(cmp) and cmp["diff_mean"] < 0.0
                ),
            }
        distributional_gate_details[metric] = per_baseline
    distributional_closure_gate = bool(
        neural_baselines
        and all(
            distributional_gate_details["stationary_jsd"][base]["full_better_significant"]
            and distributional_gate_details["innovation_whiteness"][base]["full_better_significant"]
            for base in neural_baselines
        )
    )
    closure_gate_details["mse_gate_persistence_degenerate"] = mse_gate_degenerate
    closure_gate_details["persistence_increment_variance"] = persistence_increment_var
    closure_gate_details["distributional_gate"] = distributional_gate_details
    closure_gate_details["distributional_closure_gate"] = distributional_closure_gate

    # Structural fingerprint check: no two ablation cells may share a fitted
    # structure.  This is the invariant that would have caught the historical
    # ``Markov == FreeEnergy`` duplication automatically.  The drift entry is the
    # *effective* per-step drift increment norm (a fitted ``force_scale`` of zero
    # makes a ``use_force=True`` cell identical to a force-free one).
    fingerprints: dict[str, tuple] = {}
    for method_name, payload in results.items():
        info = payload["fit_info"]
        fingerprints[method_name] = (
            info.get("effective_force_source", info.get("force_source")),
            info.get("diffusion_source"),
            info.get("kernel_source"),
            round(float(info.get("effective_drift_increment_norm", 0.0)), 12),
            round(float(info.get("kernel_norm", 0.0)), 18),
            info.get("architecture"),
        )
    duplicates: dict[str, list[str]] = {}
    seen: dict[tuple, str] = {}
    for method_name, fingerprint in fingerprints.items():
        if fingerprint in seen:
            duplicates.setdefault(seen[fingerprint], []).append(method_name)
        else:
            seen[fingerprint] = method_name

    # Behavioural fingerprint: two cells that are structurally distinct but
    # produce bit-identical evaluation metrics *are the same model*.  This is how
    # the historical ``Markov == FreeEnergy`` duplication and the force_scale = 0
    # collapse of ``M_Force`` onto ``M_None`` become visible.
    behavioural_fingerprints: dict[str, tuple] = {}
    for method_name, payload in results.items():
        summary = payload["summary"]
        behavioural_fingerprints[method_name] = tuple(
            round(float(summary[metric]["mean"]), 14)
            for metric in M1_METRICS if metric in summary
        )
    behavioural_duplicates: dict[str, list[str]] = {}
    seen_behavioural: dict[tuple, str] = {}
    for method_name, fingerprint in behavioural_fingerprints.items():
        if fingerprint in seen_behavioural:
            behavioural_duplicates.setdefault(seen_behavioural[fingerprint], []).append(method_name)
        else:
            seen_behavioural[fingerprint] = method_name

    # Near-identical pairs: metric vectors that agree to within 0.1% are the same
    # model up to Monte-Carlo noise.  This is how ``Full`` and ``FullCalibrated``
    # collapse once the mobility is clipped to zero.
    near_identical: dict[str, dict] = {}
    behavioural_names = list(behavioural_fingerprints)
    for position, left in enumerate(behavioural_names):
        for right in behavioural_names[position + 1:]:
            a = np.asarray(behavioural_fingerprints[left], dtype=float)
            b = np.asarray(behavioural_fingerprints[right], dtype=float)
            denominator = np.maximum(np.abs(a), np.abs(b))
            denominator[denominator < 1.0e-30] = 1.0
            relative = float(np.max(np.abs(a - b) / denominator))
            if relative < 1.0e-3:
                near_identical[f"{left}_vs_{right}"] = {
                    "max_relative_difference": relative,
                }
    ablation_cells = ("M_None", "M_Force", "M_Memory", "M_Full")
    collapsed_ablation_pairs = {
        key: value for key, value in near_identical.items()
        if all(name in ablation_cells for name in key.split("_vs_"))
    }

    native_bath_informative = bool(effective_memory_lags >= 2)
    degenerate_cells = [
        name for name, payload in results.items()
        if payload["fit_info"].get("degenerate_zero_mobility")
    ]
    zero_force_cells = [
        name for name, payload in results.items()
        if payload["fit_info"].get("degenerate_zero_force")
    ]
    warnings = [
        "Full semigroup metric is a conditional-mean operator diagnostic, not a stochastic-kernel semigroup proof.",
        "FDT residual is a covariance calibration diagnostic for all methods; it is not an analytic thermodynamic identity for neural baselines.",
        "stationary_jsd is compared against stationary_jsd_noise_floor (reference-vs-reference JSD); values at or below the floor are not evidence of a matched stationary law.",
    ]
    if loader_stats["truncated_by_max_batches"]:
        warnings.append(
            f"--max-batches={args.max_batches} truncated the evaluation set: "
            f"{len(trajectories)} trajectories / {loader_stats['n_proteins']} proteins were evaluated "
            f"although the manifest contains {uncapped['est_trajectories']} usable trajectories."
        )
    if not native_bath_informative:
        warnings.append(
            f"The native bath kernel has only {effective_memory_lags} lag(s) with relative norm > 0.05 "
            f"at dt_ps={dt_ps}; the nominal {args.kernel_length}-lag memory is effectively shorter."
        )
    if duplicates:
        warnings.append(f"Duplicate structural fingerprints detected: {duplicates}")
    if behavioural_duplicates:
        warnings.append(
            "Duplicate behavioural fingerprints detected (bit-identical evaluation metrics, "
            f"i.e. the cells are the same model): {behavioural_duplicates}"
        )
    if near_identical:
        warnings.append(
            f"Behaviourally near-identical method pairs (max relative metric difference < 1e-3; "
            f"these are the same model up to Monte-Carlo noise): {sorted(near_identical)}"
        )
    if degenerate_cells:
        warnings.append(
            f"Degenerate zero-mobility cells (mean = z, i.e. a pure random walk): {degenerate_cells}"
        )
    if zero_force_cells:
        warnings.append(
            f"Degenerate zero-force cells (the checkpoint's force head is numerically zero, "
            f"so use_force=True collapses onto the force-free cell): {zero_force_cells}"
        )
    degenerate_native_drift_cells = [
        name for name, payload in results.items()
        if payload["fit_info"].get("degenerate_native_drift")
    ]
    if degenerate_native_drift_cells:
        warnings.append(
            f"Degenerate native drift (the checkpoint's diffusion head is numerically zero, so "
            f"the native drift direction cannot be calibrated): {degenerate_native_drift_cells}"
        )

    output = {
        "experiment_id": "M1",
        "protocol": {
            "name": "strict_temporal_holdout_v2",
            "fit_split": "first_80_percent_of_each_trajectory",
            "evaluation_split": "last_20_percent_of_each_trajectory",
            "aggregation": "trajectory_mean_within_protein_then_protein_bootstrap",
            "design": "complete {use_force} x {use_memory} ablation on one shared update rule",
            "ablation_cells": {
                "M_None": {"use_force": False, "use_memory": False, "kernel_source": "none"},
                "M_Force": {"use_force": True, "use_memory": False, "kernel_source": "none"},
                "M_Memory": {"use_force": False, "use_memory": True, "kernel_source": "fitted_ridge"},
                "M_Full": {"use_force": True, "use_memory": True, "kernel_source": "fitted_ridge"},
            },
            "memory_lag_convention": "kernel[j] multiplies (z_{t-j} - center) for j = 0..L-1",
            "sampled_dt_ps": dt_ps,
            "clip_frames": args.clip_frames,
            "window_stride": args.window_stride,
            "history_window": args.history_window,
            "kernel_length": args.kernel_length,
            "no_fallback": True,
            "native_kernel_lag_relative_norm": native_kernel_lag_relative,
            "native_kernel_norm_excluding_zero_lag": native_kernel_norm,
            "native_kernel_ratio_excluding_zero_lag": native_kernel_ratio,
            "native_effective_memory_lags": effective_memory_lags,
            "fdt_definition": "relative Frobenius residual between test innovation covariance and method-predicted covariance",
            "semigroup_definition": "defect of the affine conditional-mean operator in homogeneous coordinates, T1*T1-T2; not a stochastic-kernel proof",
        },
        "provenance": {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "manifest": str(args.manifest),
            "seed": args.seed,
            "device": str(args.device),
            "python": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "n_trajectories_extracted": len(trajectories),
            "n_train_trajectories": len(train),
            "n_test_trajectories": len(test),
            "n_train_proteins": len({metadata[key]["protein_id"] for key in train}),
            "n_test_proteins": len({metadata[key]["protein_id"] for key in test}),
            "loader_stats": loader_stats,
            "uncapped_manifest_capacity": uncapped,
            "train_only_objects": ["native q extractor", "local diffusion", "finite memory ridge", "neural predictors", "innovation covariance"],
        },
        "methods": results,
        "pairwise_full_minus_baseline": comparisons,
        "structural_fingerprints": {k: list(v) for k, v in fingerprints.items()},
        "structural_fingerprint_duplicates": duplicates,
        "behavioural_fingerprint_duplicates": behavioural_duplicates,
        "behaviourally_near_identical_pairs": near_identical,
        "closure_gate_details": closure_gate_details,
        "gates": {
            "strict_run_complete": True,
            "no_duplicate_structural_fingerprint": bool(not duplicates),
            "no_duplicate_behavioural_fingerprint": bool(not behavioural_duplicates),
            "ablation_cells_behaviourally_separated": bool(not collapsed_ablation_pairs),
            "native_memory_nonzero": bool(native_kernel_norm > 0.0),
            "native_memory_informative": native_bath_informative,
            "no_degenerate_zero_mobility": bool(not degenerate_cells),
            "no_degenerate_zero_force": bool(not zero_force_cells),
            "no_degenerate_native_drift": bool(not degenerate_native_drift_cells),
            "no_truncated_evaluation_set": bool(not loader_stats["truncated_by_max_batches"]),
            "n_test_trajectories_ge_30": bool(len(test) >= 30),
            "closure_gate": closure_gate,
            "mse_gate_not_persistence_degenerate": bool(not mse_gate_degenerate),
            "distributional_closure_gate": distributional_closure_gate,
        },
        "claim_boundary": "M1 values are a validation-set mechanistic comparison. A Full win is only claimed where the protein-bootstrap interval for Full minus baseline excludes zero; no claim is made for metrics whose interval crosses zero.",
        "errors": [],
        "warnings": warnings,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {out_path}")
    adapter.close()


if __name__ == "__main__":
    main()
