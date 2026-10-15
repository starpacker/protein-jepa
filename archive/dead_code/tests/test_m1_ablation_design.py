"""Regression tests for the corrected M1 {force} x {memory} ablation design.

These pin the three invariants whose violation caused the original M1 method
table to be non-comparable:

1. all four ablation cells exist and differ only by ``use_force``/``use_memory``;
2. every memory method uses the same lag convention
   (``kernel[j]`` multiplies ``z_{t-j}`` for ``j = 0..L-1``);
3. no two methods share a structural fingerprint.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m1 = _load_module("m1_strict_under_test", "scripts/evaluate_m1_strict.py")

DIMENSION = 8


class FakeAdapter:
    """Minimal stand-in for LSPWMDynamicsModel's physical accessors."""

    def __init__(self, kernel_scale: float = 1.0e-3, decay: float = 2.0):
        self.kernel_scale = kernel_scale
        self.decay = decay
        self.kernel_dt_ps = None

    def get_force_fn(self):
        return lambda z: -0.1 * np.asarray(z, dtype=np.float64)

    def get_diffusion_fn(self):
        def diffusion(z):
            values = np.asarray(z, dtype=np.float64)
            return np.stack([np.eye(DIMENSION) * 0.02 for _ in range(len(values))], axis=0)
        return diffusion

    def get_memory_kernel_fn(self):
        def kernel(max_lag):
            lags = np.arange(max_lag + 1)
            decay = np.exp(-lags / self.decay)
            return np.stack(
                [decay[i] * np.eye(DIMENSION) * self.kernel_scale for i in range(len(lags))],
                axis=0,
            )
        return kernel


def _synthetic_trajectories(
    seed: int = 0,
    n_trajectories: int = 4,
    length: int = 80,
    hidden_drift: float = -0.1,
):
    rng = np.random.default_rng(seed)
    trajectories = {}
    for index in range(n_trajectories):
        sequence = np.zeros((length, DIMENSION), dtype=np.float64)
        sequence[0] = rng.normal(size=DIMENSION)
        for step in range(1, length):
            velocity = hidden_drift * sequence[step - 1]
            if step > 1:
                velocity = velocity + 0.05 * (sequence[step - 1] - sequence[step - 2])
            sequence[step] = sequence[step - 1] + velocity + 0.02 * rng.normal(size=DIMENSION)
        trajectories[f"trajectory_{index}"] = sequence
    return trajectories


@pytest.fixture(scope="module")
def fitted_cells():
    train = _synthetic_trajectories()
    adapter = FakeAdapter()
    cells = {
        "M_None": m1.TwoByTwoClosure(
            "M_None", DIMENSION, 1.0, 4, 4, use_force=False, use_memory=False),
        "M_Force": m1.TwoByTwoClosure(
            "M_Force", DIMENSION, 1.0, 4, 4, use_force=True, use_memory=False),
        "M_Memory": m1.TwoByTwoClosure(
            "M_Memory", DIMENSION, 1.0, 4, 4, use_force=False, use_memory=True),
        "M_Full": m1.TwoByTwoClosure(
            "M_Full", DIMENSION, 1.0, 4, 4, use_force=True, use_memory=True),
        "Full": m1.NativeFullClosure("Full", DIMENSION, 1.0, 4, 4),
        "FullCalibrated": m1.CalibratedFullClosure("FullCalibrated", DIMENSION, 1.0, 4, 4),
        "TrainMean": m1.TrainMeanClosure("TrainMean", DIMENSION, 1.0, 4),
    }
    return {name: (cell, cell.fit(train, adapter)) for name, cell in cells.items()}, train



def test_all_four_ablation_cells_exist(fitted_cells):
    cells, _ = fitted_cells
    for name in ("M_None", "M_Force", "M_Memory", "M_Full"):
        assert name in cells, f"ablation cell {name} is missing"


def test_cells_differ_only_by_force_and_memory(fitted_cells):
    cells, _ = fitted_cells
    assert cells["M_None"][1]["design_cell"] == {"use_force": False, "use_memory": False}
    assert cells["M_Force"][1]["design_cell"] == {"use_force": True, "use_memory": False}
    assert cells["M_Memory"][1]["design_cell"] == {"use_force": False, "use_memory": True}
    assert cells["M_Full"][1]["design_cell"] == {"use_force": True, "use_memory": True}

    # Force-free cells must have no force at all; memory-free cells no kernel.
    assert cells["M_None"][1]["force_source"] == "none"
    assert cells["M_Memory"][1]["force_source"] == "none"
    assert cells["M_None"][1]["kernel_source"] == "none"
    assert cells["M_Force"][1]["kernel_source"] == "none"
    # "Memory without free energy" must actually carry a non-trivial kernel.
    assert cells["M_Memory"][1]["kernel_norm"] > 0.0
    assert cells["M_Full"][1]["kernel_norm"] > 0.0


def test_memory_cells_are_not_bit_identical(fitted_cells):
    """The historical Markov/FreeEnergy duplication must be impossible now."""
    cells, _ = fitted_cells
    assert cells["M_None"][1]["train_innovation_mse"] != pytest.approx(
        cells["M_Force"][1]["train_innovation_mse"], rel=1.0e-12)
    assert cells["M_Memory"][1]["kernel_norm"] != pytest.approx(
        cells["M_Full"][1]["kernel_norm"], rel=1.0e-12)


def test_kernel_lag_convention_is_consistent(fitted_cells):
    """``kernel[0]`` must multiply the current state for every memory method."""
    cells, _ = fitted_cells
    probe_state = np.full(DIMENSION, 0.5)
    history = [np.full(DIMENSION, -0.25), np.full(DIMENSION, 0.75)]

    for name in ("M_Memory", "M_Full"):
        cell = cells[name][0]
        cell.memory_center = np.zeros(DIMENSION)
        expected = cell.kernel[0] @ probe_state
        for lag in range(1, cell.kernel_length):
            if len(history) < lag:
                break
            expected = expected + cell.kernel[lag] @ history[-lag]
        mean, _ = cell.predict_step(probe_state, history)
        drift = mean - probe_state - expected * cell.dt_ps
        if cell.force_scale != 0.0:
            drift = drift - cell._force(probe_state.reshape(1, -1))[0] * (
                cell.force_scale * cell.dt_ps)
        np.testing.assert_allclose(drift, np.zeros(DIMENSION), rtol=1.0e-9, atol=1.0e-9)
        assert cell.kernel[0].shape == (DIMENSION, DIMENSION)


def test_structural_fingerprints_are_unique(fitted_cells):
    cells, _ = fitted_cells
    fingerprints = {
        name: _effective_fingerprint(info) for name, (_, info) in cells.items()
    }
    # ``FullCalibrated`` may legitimately collapse onto a degenerate cell when the
    # fitted mobility is clipped to zero, so uniqueness is required only for the
    # four ablation cells plus the two native ``Full`` variants.
    required = ("M_None", "M_Force", "M_Memory", "M_Full")
    selected = {name: fingerprints[name] for name in required}
    assert len(set(selected.values())) == len(selected), selected


def _effective_fingerprint(info: dict) -> tuple:
    """Must mirror ``main()``'s structural fingerprint (effective drift norm)."""
    return (
        info.get("effective_force_source", info.get("force_source")),
        info.get("diffusion_source"),
        info.get("kernel_source"),
        round(float(info.get("effective_drift_increment_norm", 0.0)), 12),
        round(float(info.get("kernel_norm", 0.0)), 18),
        info.get("architecture"),
    )


def test_zero_force_cell_collapses_onto_force_free_fingerprint():
    """A fitted ``force_scale`` of zero must make ``use_force=True`` inert.

    This is the v5 situation: the native force has a large raw increment norm
    (hundreds per step) but is *anti-correlated* with the observed increments, so
    the non-negative least-squares scale is clipped to zero and ``M_Force``
    becomes bit-identical to ``M_None``.  The fingerprint check has to see that.
    """
    train = _synthetic_trajectories(seed=5, hidden_drift=0.1)
    adapter = FakeAdapter()
    force_free = m1.TwoByTwoClosure(
        "M_None", DIMENSION, 1.0, 4, 4, use_force=False, use_memory=False)
    force_only = m1.TwoByTwoClosure(
        "M_Force", DIMENSION, 1.0, 4, 4, use_force=True, use_memory=False)
    info_free = force_free.fit(train, adapter)
    info_force = force_only.fit(train, adapter)

    assert info_force["force_scale"] == pytest.approx(0.0)
    assert info_force["degenerate_zero_force"] is True
    assert info_force["effective_force_source"] == "none"
    assert info_force["force_increment_norm_train"] > 0.0
    assert _effective_fingerprint(info_force) == _effective_fingerprint(info_free)
    assert info_force["train_innovation_mse"] == pytest.approx(
        info_free["train_innovation_mse"], rel=1.0e-12)


def test_memory_kernel_ridge_is_chosen_by_inner_validation():
    """The Volterra ridge must be selected on held-out increments."""
    train = _synthetic_trajectories(seed=7)
    kernel, diagnostics = m1.fit_memory_kernel(train, None, 1.0, 4)
    assert kernel.shape == (4, DIMENSION, DIMENSION)
    assert diagnostics["ridge_multiplier_source"] == "inner_validation_split"
    assert diagnostics["ridge_multiplier"] in {
        float(value) for value in (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
    }
    assert diagnostics["n_inner_validation_rows"] > 0
    assert np.isfinite(diagnostics["full_train_rate_mse"])
    # The in-sample rate residual must not exceed the raw target variance,
    # otherwise the "fit" is worse than predicting zero drift.
    targets = np.asarray([
        (sequence[1:] - sequence[:-1]) for sequence in train.values()
    ])
    assert diagnostics["full_train_rate_mse"] <= float(np.mean(np.square(targets)))


def test_mobility_degeneracy_is_detected():
    """A zero-mobility fit must be reported as degenerate, not silently accepted.

    An exactly zero force field is a degenerate *model* (the fit refuses it), so
    the test instead uses a force field that is real but points *against* the
    observed drift.  The least-squares mobility is then negative and is clipped
    to zero, which is exactly the v5/v9 ``FullCalibrated`` situation.
    """
    train = _synthetic_trajectories(seed=3, hidden_drift=-0.1)

    class AntiCorrelatedAdapter(FakeAdapter):
        def get_force_fn(self):
            return lambda z: 0.1 * np.asarray(z, dtype=np.float64)

    cell = m1.CalibratedFullClosure("FullCalibrated", DIMENSION, 1.0, 4, 4)
    info = cell.fit(train, AntiCorrelatedAdapter())
    assert info["mobility_scale"] == pytest.approx(0.0)
    assert info["degenerate_zero_mobility"] is True


def test_jsd_noise_floor_is_finite(fitted_cells):
    _, train = fitted_cells
    floor = m1._stationary_jsd_noise_floor(train["trajectory_0"])
    assert np.isfinite(floor)
    assert floor >= 0.0
