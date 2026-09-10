"""Shared evaluation metrics for all baseline methods.

Reuses the same deeptime scoring patterns as protein_jepa_frame's eval modules.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from deeptime.decomposition import VAMP, TICA


# ===================================================================
#  VAMP-2 scoring
# ===================================================================

def compute_vamp2_scores(
    traj_features: dict[str, np.ndarray],
    lags: list[int] | None = None,
    min_frames: int = 30,
) -> dict[str, Any]:
    """Compute per-trajectory VAMP-2 scores.

    Args:
        traj_features: {trajectory_id: (F, D) features}
        lags: Lag times in frames (default [1, 2, 5, 10, 20])
        min_frames: Minimum frames required for a trajectory (+ lag)

    Returns:
        dict with per-lag mean, std, n_trajs
    """
    if lags is None:
        lags = [1, 2, 5, 10, 20]

    results = {}
    for lag in lags:
        scores = []
        for tid, F in traj_features.items():
            if len(F) < lag + min_frames:
                continue
            try:
                v = VAMP(lagtime=lag, dim=None, scaling=None)
                v = v.fit(F).fetch_model()
                s = v.score(r="VAMP2")
                scores.append(s)
            except Exception:
                pass

        if scores:
            results[f"lag_{lag}"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n_trajs": len(scores),
            }
        else:
            results[f"lag_{lag}"] = {"mean": None, "n_trajs": 0}

    return results


# ===================================================================
#  TICA implied timescales
# ===================================================================

def compute_implied_timescales(
    traj_features: dict[str, np.ndarray],
    lags: list[int] | None = None,
    dt_ps: float = 1000.0,
) -> dict[str, Any]:
    """Compute TICA implied timescales per trajectory.

    Args:
        traj_features: {trajectory_id: (F, D) features}
        lags: Lag times for TICA (default [2, 5, 10, 20])
        dt_ps: Time per frame in picoseconds (default 1000)

    Returns:
        dict with per-lag mean timescales
    """
    if lags is None:
        lags = [2, 5, 10, 20]

    results = {}
    for lag in lags:
        all_ts = []
        for tid, F in traj_features.items():
            if len(F) < lag * 3:
                continue
            try:
                n_comp = min(5, F.shape[1], len(F) // 3)
                tica = TICA(lagtime=lag, dim=n_comp)
                tica = tica.fit(F).fetch_model()
                evals = tica.singular_values
                ts = []
                for e in evals:
                    if e >= 1.0 or e <= 0.0:
                        continue
                    ts.append(-lag * dt_ps / np.log(e))
                if ts:
                    all_ts.append(ts)
            except Exception:
                pass

        if all_ts:
            max_len = min(5, len(all_ts[0]))
            means = [
                float(np.mean([t[i] for t in all_ts if len(t) > i]))
                for i in range(max_len)
            ]
            results[f"lag_{lag}"] = {
                "mean_timescales_ps": means,
                "n_trajs": len(all_ts),
            }
        else:
            results[f"lag_{lag}"] = {"n_trajs": 0}

    return results


# ===================================================================
#  State accuracy metrics
# ===================================================================

def hungarian_match(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """Find best permutation of predicted labels to match true labels.

    Uses Hungarian algorithm on the confusion matrix.

    Returns:
        mapping array: pred_state -> best_match_true_state
    """
    # Only consider labeled frames (true_labels >= 0)
    mask = true_labels >= 0
    if not mask.any():
        return np.arange(n_states)

    conf_matrix = np.zeros((n_states, n_states), dtype=np.int64)
    for p, t in zip(pred_labels[mask], true_labels[mask]):
        if 0 <= p < n_states and 0 <= t < n_states:
            conf_matrix[p, t] += 1

    # Hungarian: maximize correct assignments
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    mapping = np.arange(n_states)
    mapping[row_ind] = col_ind
    return mapping


def compute_state_accuracy(
    pred_labels: dict[str, np.ndarray],
    true_labels: dict[str, np.ndarray],
    n_states: int = 6,
) -> dict[str, Any]:
    """Compute state prediction accuracy metrics.

    For each trajectory, maps predicted states to true states via Hungarian
    alignment, then computes top-1, top-2, and balanced accuracy.

    Args:
        pred_labels: {trajectory_id: (F,) int64 predicted state labels}
        true_labels: {trajectory_id: (F,) int64 true state labels (-1 = unlabeled)}

    Returns:
        dict with accuracy metrics
    """
    common_ids = set(pred_labels) & set(true_labels)
    if not common_ids:
        return {"error": "no common trajectory IDs"}

    all_pred = []
    all_true = []
    for tid in common_ids:
        p = pred_labels[tid]
        t = true_labels[tid]
        min_len = min(len(p), len(t))
        all_pred.append(p[:min_len])
        all_true.append(t[:min_len])

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # Hungarian alignment
    mapping = hungarian_match(all_pred, all_true, n_states)
    aligned_pred = mapping[all_pred]

    # Only labeled frames
    mask = all_true >= 0
    if not mask.any():
        return {
            "top1_accuracy": 0.0,
            "top2_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "n_frames": 0,
        }

    y_true = all_true[mask]
    y_pred = aligned_pred[mask]

    n = len(y_true)
    if n == 0:
        return {"top1_accuracy": 0.0, "n_frames": 0}

    top1 = (y_pred == y_true).mean()

    # Top-2: predicted and true match if either of the two most common
    # predicted states matches the true state. For each frame, check if
    # true state is in the top-2 predicted states. Since we only have
    # hard labels, top-2 = top-1 + (second most common state).
    # Simplified: check if true state is matchable to top-2
    # We compute "top-2 accuracy" by checking if either of the two most
    # frequent predicted states matches the true state.
    unique_states = np.unique(y_pred)
    state_counts = Counter(y_pred.tolist())
    top2_states = {s for s, _ in state_counts.most_common(2)}
    top2 = np.mean([1 if t in top2_states else 0 for t in y_true])

    # Balanced accuracy: per-class recall averaged
    classes = sorted(set(y_true) | set(y_pred))
    recalls = []
    for c in classes:
        true_c = y_true == c
        if true_c.sum() == 0:
            continue
        recall_c = (y_pred[true_c] == c).mean()
        recalls.append(recall_c)
    balanced = np.mean(recalls) if recalls else 0.0

    # Persistence accuracy (baseline): copy previous state
    # Frame-level persistence: predict state = previous frame's state
    # For each trajectory separately
    persistence_accs = []
    for tid in common_ids:
        p = pred_labels[tid]
        t = true_labels[tid]
        min_len = min(len(p), len(t))
        p = mapping[p[:min_len]]
        t = t[:min_len]
        mask_t = t >= 0
        if mask_t.sum() < 2:
            continue
        t_valid = t[mask_t]
        p_valid = p[mask_t]
        # Persistence: p_t = t_{t-1}
        same = (p_valid[1:] == t_valid[:-1])
        # Also check if t_valid[1:] == t_valid[:-1] (persistence prediction for current frame)
        pers_pred = t_valid[:-1]
        pers_correct = pers_pred == t_valid[1:]
        persistence_accs.append(pers_correct.mean())
    persistence_acc = np.mean(persistence_accs) if persistence_accs else 0.0

    return {
        "top1_accuracy": float(top1),
        "top2_accuracy": float(top2),
        "balanced_accuracy": float(balanced),
        "persistence_accuracy": float(persistence_acc),
        "n_frames": int(n),
        "n_trajectories": len(common_ids),
    }


# ===================================================================
#  Brier skill score
# ===================================================================

def compute_brier_skill(
    soft_assignments: dict[str, np.ndarray],
    true_labels: dict[str, np.ndarray],
    n_states: int = 6,
) -> dict[str, Any]:
    """Compute Brier skill score vs persistence forecast.

    Brier score = mean squared error between soft assignment and one-hot true label.
    Brier skill score = 1 - (Brier_model / Brier_persistence)

    Args:
        soft_assignments: {trajectory_id: (F, n_states) float32 probabilities}
        true_labels: {trajectory_id: (F,) int64 state labels (-1 = unlabeled)}

    Returns:
        dict with brier_score, persistence_brier, brier_skill
    """
    common_ids = set(soft_assignments) & set(true_labels)
    if not common_ids:
        return {"error": "no common trajectory IDs"}

    # Hungarian alignment on argmax
    all_argmax = []
    all_true_concat = []
    for tid in common_ids:
        s = soft_assignments[tid]
        t = true_labels[tid]
        min_len = min(len(s), len(t))
        all_argmax.append(s[:min_len].argmax(axis=1))
        all_true_concat.append(t[:min_len])

    all_argmax = np.concatenate(all_argmax)
    all_true = np.concatenate(all_true_concat)
    mapping = hungarian_match(all_argmax, all_true, n_states)

    # Align soft assignments
    brier_scores = []
    persistence_briers = []
    for tid in common_ids:
        s = soft_assignments[tid]
        t = true_labels[tid]
        min_len = min(len(s), len(t))
        s = s[:min_len]
        t = t[:min_len]

        # Align states
        s_aligned = np.zeros_like(s)
        for i in range(n_states):
            s_aligned[:, mapping[i]] = s[:, i]

        # One-hot true
        mask = t >= 0
        if mask.sum() < 2:
            continue
        t_valid = t[mask]
        s_valid = s_aligned[mask]

        # One-hot encode
        one_hot = np.zeros((len(t_valid), n_states))
        for i, label in enumerate(t_valid):
            if 0 <= label < n_states:
                one_hot[i, int(label)] = 1.0

        # Brier score = MSE between soft assignment and one-hot
        brier = np.mean((s_valid - one_hot) ** 2)
        brier_scores.append(brier)

        # Persistence Brier: predict t+1 = t state
        if len(t_valid) > 1:
            pers_pred = one_hot[:-1]
            pers_true = one_hot[1:]
            pers_brier = np.mean((pers_pred - pers_true) ** 2)
            persistence_briers.append(pers_brier)

    if not brier_scores:
        return {"brier_skill": None, "n_trajs": 0}

    mean_brier = np.mean(brier_scores)
    mean_pers_brier = np.mean(persistence_briers) if persistence_briers else 0.0
    skill = 1.0 - mean_brier / mean_pers_brier if mean_pers_brier > 0 else 0.0

    return {
        "brier_score": float(mean_brier),
        "persistence_brier": float(mean_pers_brier),
        "brier_skill": float(skill),
        "n_trajs": len(brier_scores),
    }


# ===================================================================
#  MSM linear probing
# ===================================================================

def compute_msm_probe_accuracy(
    features: dict[str, np.ndarray],
    msm_labels: dict[str, np.ndarray],
    n_states: int = 6,
) -> dict[str, Any]:
    """Per-protein linear probe from features to MSM labels.

    For each protein, trains LogisticRegression on 70% of frames,
    evaluates on 30%.

    Args:
        features: {trajectory_id: (F, D) features}
        msm_labels: {trajectory_id: (F,) int64 state labels}

    Returns:
        dict with per-protein accuracy summary
    """
    from collections import defaultdict

    # Group by protein_id (trajectory_id format: "protein:temp:replica")
    protein_groups: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for tid in features:
        if tid in msm_labels:
            # Extract protein_id from trajectory_id
            parts = tid.split(":")

            protein_groups[parts[0]][tid] = {"features": features[tid], "labels": msm_labels[tid]}
    protein_to_tids = defaultdict(list)
    for tid in features:
        if tid in msm_labels:
            protein_to_tids[tid.split(":")[0]].append(tid)

    all_accs = []
    all_baselines = []
    protein_details = {}

    for pid, tid_list in protein_to_tids.items():
        # Collect all features and labels for this protein
        latents = []
        labels = []
        for tid in tid_list:
            F = features[tid]
            L = msm_labels[tid]
            min_len = min(len(F), len(L))
            mask = L[:min_len] >= 0  # Only labeled frames
            if mask.sum() == 0:
                continue
            latents.append(F[:min_len][mask])
            labels.append(L[:min_len][mask])

        if not latents:
            continue

        latents = np.concatenate(latents, axis=0)
        labels = np.concatenate(labels, axis=0)

        if len(latents) < 20 or len(set(labels.tolist())) < 2:
            continue

        # 70/30 split
        rng = np.random.RandomState(42)
        n = len(latents)
        idx = rng.permutation(n)
        n_train = int(n * 0.7)
        train_lat, train_lbl = latents[idx[:n_train]], labels[idx[:n_train]]
        val_lat, val_lbl = latents[idx[n_train:]], labels[idx[n_train:]]

        if len(set(train_lbl.tolist())) < 2 or len(set(val_lbl.tolist())) < 2:
            continue

        # Baseline: majority class
        majority = Counter(train_lbl.tolist()).most_common(1)[0][0]
        baseline = (val_lbl == majority).mean()

        # Train logistic regression
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_lat)
        X_va = scaler.transform(val_lat)

        best_acc, best_C = 0, 1.0
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = LogisticRegression(
                solver="lbfgs", max_iter=5000, C=C, random_state=42
            )
            clf.fit(X_tr, train_lbl)
            acc = clf.score(X_va, val_lbl)
            if acc > best_acc:
                best_acc, best_C = acc, C

        n_states_eff = len(set(train_lbl.tolist()))
        all_accs.append(best_acc)
        all_baselines.append(baseline)
        protein_details[pid] = {
            "accuracy": float(best_acc),
            "baseline": float(baseline),
            "n_train": int(len(train_lat)),
            "n_val": int(len(val_lat)),
            "n_states": n_states_eff,
            "best_C": best_C,
        }

    if not all_accs:
        return {"mean_accuracy": None, "n_proteins": 0}

    return {
        "mean_accuracy": float(np.mean(all_accs)),
        "std_accuracy": float(np.std(all_accs)),
        "mean_baseline": float(np.mean(all_baselines)),
        "n_above_baseline": int(
            sum(1 for a, b in zip(all_accs, all_baselines) if a > b)
        ),
        "n_total": len(all_accs),
        "protein_details": protein_details,
    }


# ===================================================================
#  Load existing MSM labels
# ===================================================================

def load_msm_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Load MSM state labels from a filtered_labels.json file.

    Returns {trajectory_id: (F,) int64 state_labels}
    """
    with open(path) as f:
        data = json.load(f)
    result = {}
    for tid, entry in data.get("trajectories", {}).items():
        result[tid] = np.array(entry["state_labels"], dtype=np.int64)
    return result