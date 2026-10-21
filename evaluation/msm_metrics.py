"""MSM state-prediction metrics (TASK 2B, NEW 2026-10-21).

The part BioEmu's paper does NOT report (see ``BioEmu_指标梳理_蛋白质动力学基准.md``
§2.3): given a true history, does a model predict the correct *MSM state* after a
lag τ?  Implements:

    * lagged state accuracy / macro-F1 / balanced accuracy  (ŝ(t+τ) vs s(t+τ)),
    * state-transition-matrix error (per-row TV, JSD, Frobenius),
    * implied-timescale and MFPT errors,
    * a persistence-in-state baseline (ŝ(t+τ) = s(t)),

State labels come from a shared MSM: TICA on the *true* internal-distance
features, KMeans microstates, and a MaximumLikelihoodMSM (implicitly a coarse
state via the MSM).  Both true and predicted trajectories are projected through
the SAME TICA + centroids, so the label space is model-independent.

Design
------
* ``fit_msm(true_trajs, ...)`` -> an :class:`MSMStateModel` holding TICA,
  centroids and the MSM; ``label(feat)`` assigns discrete states.
* ``lagged_state_accuracy(pred_states_seq, true_states_seq, tau, k)``
* ``transition_matrix_error(T_pred, T_true)``
* ``implied_timescales(msm)``, ``mfpt(...)``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class MSMStateModel:
    """Shared TICA + KMeans + MSM used to label true and predicted trajectories."""
    tica: object
    kmeans: object
    msm: object
    lag: int
    n_states: int
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    macro_map: np.ndarray | None = None      # microstate -> macrostate (PCCA)
    n_macro: int = 0

    def transform(self, feats: np.ndarray) -> np.ndarray:
        """Project raw internal-distance features -> TICA space."""
        if self.feature_mean is not None:
            feats = (feats - self.feature_mean) / (self.feature_std + 1e-8)
        return self.tica.transform(feats)

    def label(self, feats: np.ndarray) -> np.ndarray:
        """Raw features -> discrete microstate labels."""
        y = self.transform(feats)
        return np.asarray(self.kmeans.transform(y)).reshape(-1)

    def label_macro(self, feats: np.ndarray) -> np.ndarray:
        """Raw features -> metastable macrostate labels (PCCA coarse-graining).

        Returns microstate labels unchanged when no PCCA map is available.
        """
        lab = self.label(feats).astype(int)
        if self.macro_map is None:
            return lab
        lab = np.clip(lab, 0, len(self.macro_map) - 1)
        return self.macro_map[lab]

    def _discretize_sequence(self, labels: np.ndarray) -> np.ndarray:
        """Collapse microstates to MSM coarse states (majority/nearest by pi)."""
        return labels


def _standardize(trajs: Sequence[np.ndarray]):
    allf = np.concatenate(trajs, axis=0)
    mu, sd = allf.mean(0), allf.std(0)
    return mu, sd


def fit_msm(true_feats: Sequence[np.ndarray], *, lag: int = 1, n_micro: int = 50,
            n_states: int = 5, seed: int = 0) -> MSMStateModel:
    """Fit TICA + KMeans + MSM on a list of per-trajectory feature arrays.

    Args:
        true_feats: list of [T_i, D] internal-distance feature arrays (true MD).
        lag: MSM lag in frames.
        n_micro: microstate count (KMeans). If n_states == n_micro this is the
            number of discrete MSM states used for the state-prediction accuracy.
        n_states: reserved (MSM coarse states via spectral clustering if needed).
        seed: unused placeholder for API stability (deeptime KMeans is deterministic).
    """
    from deeptime.decomposition import TICA
    from deeptime.clustering import KMeans

    mu, sd = _standardize(true_feats)
    z = [(f - mu) / (sd + 1e-8) for f in true_feats]
    tica = TICA(dim=2, lagtime=lag).fit(z).fetch_model()
    y = [tica.transform(f) for f in z]

    km = KMeans(n_clusters=n_micro, max_iter=100).fit(np.concatenate(y, axis=0)).fetch_model()
    disc = [km.transform(f).astype(int) for f in y]
    mask_disc = [d[d >= 0] for d in disc]

    from deeptime.markov.msm import MaximumLikelihoodMSM
    msm = MaximumLikelihoodMSM(reversible=True, lagtime=lag).fit_fetch(mask_disc)

    # PCCA coarse-graining: microstates -> `n_states` metastable macrostates.
    macro_map, n_macro = None, 0
    try:
        from deeptime.markov import pcca
        n_k = int(min(n_states, msm.n_states))
        if n_k >= 2:
            p = pcca(msm.transition_matrix, n_k)
            macro_map = np.asarray(p.assignments).astype(int)
            n_macro = int(macro_map.max()) + 1
    except Exception:
        macro_map, n_macro = None, 0

    return MSMStateModel(tica=tica, kmeans=km, msm=msm, lag=lag,
                         n_states=msm.n_states, feature_mean=mu, feature_std=sd,
                         macro_map=macro_map, n_macro=n_macro)


def lagged_state_prediction(true_states: np.ndarray, pred_states: np.ndarray,
                            tau: int, n_states: int) -> dict:
    """Compare predicted state at t+tau to the true state at t+tau.

    Both sequences must be aligned in time (same length, same origin).  Provides
    accuracy, macro-F1 and balanced accuracy (majority-inflation-resistant), and
    a state-persistence baseline.

    Returns per-lag scalars + confusion counts.
    """
    true_states = np.asarray(true_states).astype(int)
    pred_states = np.asarray(pred_states).astype(int)
    n = len(true_states)
    if n <= tau:
        return {"n_eval": 0}
    tgt = true_states[tau:]                       # s(t+tau)
    prd = pred_states[:-tau]                      # ŝ(t+tau) from history at t
    pers = true_states[:-tau]                     # persistence baseline
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    res = {
        "n_eval": int(len(tgt)),
        "accuracy": float(accuracy_score(tgt, prd)),
        "macro_f1": float(f1_score(tgt, prd, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(tgt, prd)),
        "persist_accuracy": float(accuracy_score(tgt, pers)),
        "persist_macro_f1": float(f1_score(tgt, pers, average="macro", zero_division=0)),
    }
    res["skill_vs_persistence"] = res["accuracy"] - res["persist_accuracy"]
    return res


def transition_matrix_error(T_pred: np.ndarray, T_true: np.ndarray) -> dict:
    """Per-row TV / JSD + Frobenius norm between two transition matrices.

    Rows are stationary-conditioned distributions over next states.
    """
    T_pred = np.asarray(T_pred, dtype=np.float64)
    T_true = np.asarray(T_true, dtype=np.float64)
    if T_pred.shape != T_true.shape:
        raise ValueError(f"shape mismatch {T_pred.shape} vs {T_true.shape}")
    eps = 1e-12
    P = T_pred / T_pred.sum(1, keepdims=True).clip(min=eps)
    Q = T_true / T_true.sum(1, keepdims=True).clip(min=eps)
    row_tv = 0.5 * np.abs(P - Q).sum(1)
    m = 0.5 * (P + Q)

    def kl(a, b):
        good = (a > 0)
        return np.where(good, a * np.log((a + eps) / (b + eps)), 0.0).sum(1)

    row_jsd = 0.5 * kl(P, m) + 0.5 * kl(Q, m)
    return {"frobenius": float(np.linalg.norm(P - Q)),
            "mean_row_tv": float(row_tv.mean()),
            "mean_row_jsd": float(row_jsd.mean()),
            "max_row_tv": float(row_tv.max())}


def implied_timescales(msm, n: int = 5) -> np.ndarray:
    """Top-n implied timescales (in lag units) of an MSM."""
    try:
        ts = msm.timescales()
        return np.asarray(ts[:n], dtype=np.float64)
    except Exception:
        return np.array([])


def implied_timescale_error(msm_pred, msm_true, n: int = 3) -> dict:
    """Relative error of the top-n implied timescales."""
    tp = implied_timescales(msm_pred, n)
    tt = implied_timescales(msm_true, n)
    m = min(len(tp), len(tt))
    if m == 0:
        return {"n": 0}
    rel = np.abs(tp[:m] - tt[:m]) / np.where(np.abs(tt[:m]) > 0, np.abs(tt[:m]), 1.0)
    return {"n": int(m), "mean_rel_error": float(rel.mean()),
            "pred_timescales": tp[:m].tolist(), "true_timescales": tt[:m].tolist()}


def mfpt_matrix(msm) -> np.ndarray:
    """Mean-first-passage-time matrix from an MSM (in lag units)."""
    return np.asarray(msm.mfpt())


__all__ = [
    "MSMStateModel", "fit_msm", "lagged_state_prediction",
    "transition_matrix_error", "implied_timescales", "implied_timescale_error",
    "mfpt_matrix",
]
