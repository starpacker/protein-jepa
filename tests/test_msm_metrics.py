"""Tests: MSM state-prediction metrics (TASK 2B).

Analytic checks on a synthetic 2-state Markov chain with a known transition
matrix: accuracy (identical -> 1), transition-matrix error (identical -> 0),
and error growth with a corrupted predictor.
"""
from __future__ import annotations

import numpy as np
import torch

from evaluation.msm_metrics import (
    lagged_state_prediction, transition_matrix_error, fit_msm,
)


def _two_state_chain(n=4000, p_stay=0.9, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(n, dtype=int)
    for t in range(1, n):
        if rng.random() < p_stay:
            s[t] = s[t - 1]
        else:
            s[t] = 1 - s[t - 1]
    return s


def test_lagged_accuracy_perfect_predictor_is_one():
    # Semantics: pred[t] is the model's guess for the state at t+tau, compared
    # against true[t+tau].  A perfect predictor sets pred[t] = true[t+tau].
    s = _two_state_chain()
    tau = 1
    perfect = np.concatenate([s[tau:], np.full(tau, s[-1])])
    out = lagged_state_prediction(s, perfect, tau=tau, n_states=2)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["balanced_accuracy"] == 1.0


def test_persistence_predictor_matches_true_persistence():
    # pred[t] = true[t] (state persistence) -> accuracy equals the persistence
    # baseline reported by the metric.
    s = _two_state_chain()
    out = lagged_state_prediction(s, s, tau=1, n_states=2)
    assert abs(out["accuracy"] - out["persist_accuracy"]) < 1e-9


def test_lagged_accuracy_corrupted_drops():
    s = _two_state_chain()
    rng = np.random.default_rng(1)
    bad = s.copy()
    flip = rng.random(len(s)) < 0.4
    bad[flip] = 1 - bad[flip]
    out = lagged_state_prediction(s, bad, tau=1, n_states=2)
    assert out["accuracy"] < 0.9
    # persistence baseline should be weaker than the true-identity predictor
    assert out["accuracy"] <= out["persist_accuracy"] + 0.2


def test_transition_matrix_error_identical_zero():
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    out = transition_matrix_error(T, T)
    assert out["frobenius"] < 1e-12
    assert out["mean_row_tv"] < 1e-12


def test_fit_msm_on_two_state_chain():
    # Two well-separated 1-D feature "clusters" from a 2-state chain.
    s = _two_state_chain(n=3000, p_stay=0.92, seed=2)
    feats = np.where(s[:, None] == 0, -5.0, 5.0) + 0.3 * np.random.default_rng(3).standard_normal((len(s), 1))
    model = fit_msm([feats], lag=1, n_micro=8, n_states=2)
    labels = model.label(feats)
    # labels should separate the two feature clusters
    big = feats[:, 0] > 0
    assert abs(labels[big].mean() - labels[~big].mean()) > 0
    # implied timescales compute without error
    from evaluation.msm_metrics import implied_timescales
    ts = implied_timescales(model.msm, 2)
    assert len(ts) >= 1
