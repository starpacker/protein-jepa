#!/usr/bin/env python3
"""R1 v6: lspwm_q only, with SVD-accelerated ridge regression.

SVD approach: factor the design matrix once per (trajectory, feature_set),
then evaluate all 8 ridge penalties with O(d*n) ops each instead of O(d³).

Numerically equivalent to np.linalg.solve(gram + λI, X^T y) up to floating
point — identical to evaluate_history_gap_v2.py.
"""
from __future__ import annotations
import json, sys, numpy as np, torch
from pathlib import Path

ROOT = Path('/liuyutian/protein_jepa_frame')
sys.path.insert(0, str(ROOT))

from artifacts.reviewer_experiments._shared.latent_trajectories import extract_ordered_latent_trajectories
from artifacts.reviewer_experiments._shared.lspwm_adapter import LSPWMDynamicsModel

RIDGE_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0)


def _ridge_svd(design_svd, train_y, penalty_multiplier, trace_over_D):
    """Ridge solve using pre-computed SVD of the design matrix.

    design_svd = (U, s, Vt) from np.linalg.svd(design, full_matrices=False)
      design is n × (D+1), includes intercept column.
    penalty = penalty_multiplier * trace_over_D
    Supports multi-output (n, D_out) targets.
    """
    U, s, Vt = design_svd
    train_y = np.asarray(train_y, dtype=np.float64)
    s2 = s * s
    Ut_y = U.T @ train_y  # (D+1, D_out)
    penalty = penalty_multiplier * trace_over_D
    scaling = s / (s2 + penalty)  # (D+1,)
    w_v = scaling[:, np.newaxis] * Ut_y  # (D+1, D_out) broadcast
    w_full = Vt.T @ w_v  # (D+1, D_out)
    return w_full[:-1], w_full[-1]


def _select_ridge_svd(design_svd, train_x_full, train_y_full,
                       inner_fraction, trace_over_D, grid=RIDGE_GRID):
    """Select ridge multiplier using inner validation on training prefix.

    design_svd: SVD of the FULL training design matrix (train + validation).
    We split the N rows into inner 75% / validation 25%.
    """
    N = len(train_y_full)
    cut = max(2, int(N * inner_fraction))
    if N - cut < 2:
        return 1.0e-3, {}

    # For evaluating on the validation fold we can't directly use the SVD
    # of the full matrix — need to re-fit on the inner fold for each multiplier.
    # But we use the SVD-based solve for each fit, which is fast.

    # Actually, SVD of inner fold would be more efficient if we had it.
    # Let's just compute SVD of inner fold once and use it for all multipliers.
    inner_x = train_x_full[:cut]
    inner_y = train_y_full[:cut]
    inner_design = np.concatenate([inner_x, np.ones((cut, 1), dtype=np.float64)], axis=1)
    U_i, s_i, Vt_i = np.linalg.svd(inner_design, full_matrices=False)

    val_x = train_x_full[cut:]
    val_y = train_y_full[cut:]

    scores = {}
    for multiplier in grid:
        w, b = _ridge_svd((U_i, s_i, Vt_i), inner_y, multiplier, trace_over_D)
        scores[float(multiplier)] = float(np.mean(np.square(val_x @ w + b - val_y)))

    return float(min(scores, key=scores.get)), scores


def _ridge_fast(train_x, train_y, penalty_multiplier, trace_over_D):
    """One-shot ridge using SVD (no inner selection)."""
    design = np.concatenate([train_x, np.ones((len(train_x), 1), dtype=np.float64)], axis=1)
    U, s, Vt = np.linalg.svd(design, full_matrices=False)
    return _ridge_svd((U, s, Vt), train_y, penalty_multiplier, trace_over_D)


def _mse(weights, bias, x, y):
    return float(np.mean(np.square(x @ weights + bias - y)))


def _history_features(sequence, history_len):
    block = np.stack(
        [sequence[history_len - 1 - lag: len(sequence) - lag] for lag in range(history_len)],
        axis=1)
    return block.reshape(block.shape[0], -1)


def _shuffled_history_features(sequence, history_len, shift):
    n_valid = len(sequence) - history_len + 1
    source = (np.arange(n_valid) + shift) % n_valid
    block = np.stack(
        [sequence[source + history_len - 1 - lag] for lag in range(history_len)], axis=1)
    return block.reshape(block.shape[0], -1)


def _probe_row(sequence, horizon, history_len, train_fraction=0.8, min_train=20, min_test=4):
    length = len(sequence)
    n_windows = length - history_len + 1
    n_samples = n_windows - horizon
    if n_samples < min_train + min_test:
        return None

    targets = sequence[history_len - 1 + horizon: length]
    current = sequence[history_len - 1: length - horizon]
    n_valid = len(current)

    features = {
        "current": current,
        "history": _history_features(sequence, history_len)[:n_valid],
        "current_tiled_H": np.concatenate([current] * history_len, axis=1),
        "history_shuffled": _shuffled_history_features(
            sequence, history_len, max(history_len, min(n_valid // 2, 37 + horizon))
        )[:n_valid],
    }

    cut = max(min_train, int(n_valid * train_fraction))
    if n_valid - cut < min_test:
        return None

    row = {"n_train": int(cut), "n_test": int(n_valid - cut),
           "target_std": float(np.mean(np.std(targets, axis=0)))}

    for name, matrix in features.items():
        # Pre-compute trace/D for the full training design
        train_design = np.concatenate([matrix[:cut], np.ones((cut, 1), dtype=np.float64)], axis=1)
        trace_over_D = float(np.trace(train_design.T @ train_design) / train_design.shape[1])

        mult, _ = _select_ridge_svd(
            None, matrix[:cut], targets[:cut],
            inner_fraction=0.75, trace_over_D=trace_over_D)
        w, b = _ridge_fast(matrix[:cut], targets[:cut], mult, trace_over_D)
        row[f"mse_{name}"] = _mse(w, b, matrix[cut:], targets[cut:])
        row[f"ridge_multiplier_{name}"] = float(mult)

    mean_pred = np.mean(targets[:cut], axis=0)
    row["mse_train_mean"] = float(np.mean(np.square(targets[cut:] - mean_pred)))
    return row


DELTA_DEFINITIONS = {
    "delta_hist_vs_current": ("mse_current", "mse_history"),
    "delta_hist_vs_tiled": ("mse_current_tiled_H", "mse_history"),
    "delta_hist_vs_shuffled": ("mse_history_shuffled", "mse_history"),
    "delta_mean_vs_current": ("mse_current", "mse_train_mean"),
}


def _bootstrap_mean(values, n_resamples=4000, seed=6303):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"mean": float(values.mean()) if len(values) else float("nan"),
                "lower": float("nan"), "upper": float("nan"), "n": int(len(values))}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()),
            "lower": float(np.quantile(draws, 0.025)),
            "upper": float(np.quantile(draws, 0.975)),
            "n": int(len(values))}


def aggregate(rows, metadata, seed=6303):
    from collections import defaultdict
    by_protein = defaultdict(list)
    for row in rows:
        by_protein[metadata[row["trajectory_id"]]["protein_id"]].append(row)
    protein_rows = {}
    for pid, vals in sorted(by_protein.items()):
        keys = vals[0].keys()
        protein_rows[pid] = {key: float(np.mean([float(v[key]) for v in vals]))
                             for key in keys if key != "trajectory_id"}
    output = {"n_trajectories": len(rows), "n_proteins": len(protein_rows)}
    for name, (base_key, ref_key) in DELTA_DEFINITIONS.items():
        delta = np.asarray([v[base_key] - v[ref_key] for v in protein_rows.values()], dtype=float)
        stats = _bootstrap_mean(delta, seed=seed)
        output[name] = stats
        output[f"{name}_ci_excludes_zero"] = bool(
            np.isfinite(stats["lower"]) and np.isfinite(stats["upper"])
            and (stats["lower"] > 0.0 or stats["upper"] < 0.0))
    output["mean_mse_current"] = float(np.mean([v["mse_current"] for v in protein_rows.values()]))
    output["mean_mse_history"] = float(np.mean([v["mse_history"] for v in protein_rows.values()]))
    output["mean_mse_current_tiled_H"] = float(np.mean([v["mse_current_tiled_H"] for v in protein_rows.values()]))
    output["mean_mse_history_shuffled"] = float(np.mean([v["mse_history_shuffled"] for v in protein_rows.values()]))
    output["mean_mse_train_mean"] = float(np.mean([v["mse_train_mean"] for v in protein_rows.values()]))
    output["per_protein_delta_hist_vs_current"] = {
        pid: vals["mse_current"] - vals["mse_history"]
        for pid, vals in sorted(protein_rows.items())}
    output["per_protein"] = protein_rows
    return output


def run_representation(name, sequences, metadata, horizons, history_lengths, seed):
    out = {}
    for horizon in horizons:
        for history_len in history_lengths:
            rows = []
            for tid, seq in sequences.items():
                if tid not in metadata:
                    continue
                row = _probe_row(seq, horizon, history_len)
                if row is None:
                    continue
                row["trajectory_id"] = tid
                rows.append(row)
            if len(rows) < 2:
                continue
            key = f"h={horizon}/H={history_len}"
            summary = aggregate(rows, metadata, seed=seed)
            summary["history_len"] = history_len
            summary["horizon_steps"] = horizon
            out[key] = summary
            print(f"  {name:10s} {key:12s} "
                  f"mse_cur={summary['mean_mse_current']:.4e} "
                  f"mse_hist={summary['mean_mse_history']:.4e} "
                  f"d_hist={summary['delta_hist_vs_current']['mean']:+.4e} "
                  f"[{summary['delta_hist_vs_current']['lower']:+.4e},"
                  f"{summary['delta_hist_vs_current']['upper']:+.4e}] "
                  f"d_vs_tile={summary['delta_hist_vs_tiled']['mean']:+.4e} "
                  f"n_prot={summary['n_proteins']}")
    return out


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def load_latents(checkpoint, config, manifest, device, batch_size, clip_frames, window_stride,
                 max_batches, min_states, cache_path):
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        print(f"[R1v6] loading cached latents from {cache}")
        with np.load(cache, allow_pickle=True) as handle:
            sequences = {key: handle[key] for key in handle.files if not key.startswith("meta_")}
            metadata = json.loads(str(handle["meta_metadata"].item()))
            loader_stats = json.loads(str(handle["meta_loader_stats"].item()))
        return sequences, metadata, loader_stats
    adapter = LSPWMDynamicsModel()
    adapter.load(checkpoint_path=checkpoint, config_path=config,
                 manifest=manifest, device=device, batch_size=batch_size,
                 max_batches=1, seed=6303)
    sequences, metadata, loader_stats = extract_ordered_latent_trajectories(
        adapter, manifest, batch_size=batch_size, clip_frames=clip_frames,
        window_stride=window_stride, max_batches=max_batches or None, min_states=min_states)
    adapter.close()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **sequences,
                            meta_metadata=np.array(json.dumps(metadata)),
                            meta_loader_stats=np.array(json.dumps(loader_stats)))
        print(f"[R1v6] cached latents -> {cache}")
    return sequences, metadata, loader_stats


def main():
    CKPT = '/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v6_kernel_fix/checkpoint_latest.pt'
    CFG = '/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v6_kernel_fix/eval_improved/eval_config.json'
    MANIFEST = '/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl'
    OUTPUT = '/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/representation_closure_v2/r1_history_gap_v2_v6.json'
    CACHE = '/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/representation_closure_v2/r1_latents_v6.npz'

    np.random.seed(6303)
    torch.manual_seed(6303)

    sequences, metadata, loader_stats = load_latents(
        CKPT, CFG, MANIFEST, 'cuda', 32, 24, 8, 0, 16, CACHE)

    dt_values = {round(float(v["dt_ps"]), 6) for v in metadata.values()}
    dt_ps = float(next(iter(dt_values)))

    output = {
        "experiment_id": "R1",
        "protocol": {
            "name": "history_gap_ordered_v2",
            "probe": "deterministic_ridge_scale_adaptive_l2_with_inner_validation_per_feature_set",
            "ridge_grid": list(RIDGE_GRID),
            "ridge_selection": "inner 75/25 split of the training prefix, per feature set",
            "ridge_solver": "svd-based (SVD once per design matrix, then O(d*n) per multiplier)",
            "capacity_controls": ["current_tiled_H", "history_shuffled"],
            "split": "per_trajectory_prefix(80%)/suffix(20%)",
            "aggregation": "trajectory_mean_within_protein_then_protein_bootstrap",
            "horizons": [1, 2, 4, 8, 16],
            "history_lengths": [1, 2, 4, 8],
            "representations": ["lspwm_q"],
            "clip_frames": 24, "window_stride": 8,
            "source_frames_per_step": 8, "no_fallback": True,
        },
        "provenance": {"checkpoint": CKPT, "config": CFG, "manifest": MANIFEST,
                       "seed": 6303, "device": "cuda", "python": sys.executable},
        # Coordinate baselines NOT re-run — PCA/tICA/VAMP on CA coordinates
        # are checkpoint-independent. v5 results are reused.
        "coordinate_baselines_note": "PCA/tICA/VAMP results from v5 are reusable (checkpoint-independent). See r1_history_gap_v2_v5.json.",
        "coordinate_baselines_source": "/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/representation_closure_v2/r1_history_gap_v2_v5.json",
        "representations": {}, "gates": {}, "errors": [], "warnings": [],
        "dataset": {
            "n_trajectories": len(sequences),
            "n_proteins": len({v["protein_id"] for v in metadata.values()}),
            "dt_ps": dt_ps, "loader_stats": loader_stats,
        },
        "physical_horizon_ps": {f"h={h}": h * dt_ps for h in [1, 2, 4, 8, 16]},
    }

    print(f"[R1v6] trajectories={len(sequences)} proteins={output['dataset']['n_proteins']} dt_ps={dt_ps}")

    probe_sequences = dict(sequences)
    probe_metadata = dict(metadata)
    dimension = int(next(iter(probe_sequences.values())).shape[1])
    print(f"[R1v6] === lspwm_q (dim={dimension}) n_traj={len(probe_sequences)} ===")
    result = run_representation("lspwm_q", probe_sequences, probe_metadata,
                                horizons=[1, 2, 4, 8, 16],
                                history_lengths=[1, 2, 4, 8], seed=6303)

    output["representations"]["lspwm_q"] = {
        "status": "ok", "dim": int(dimension),
        "n_trajectories": len(probe_sequences),
        "n_proteins": len({m["protein_id"] for m in probe_metadata.values()}),
        "by_condition": result,
    }
    passing = [
        key for key, value in result.items()
        if value["delta_hist_vs_current_ci_excludes_zero"]
        and value["delta_hist_vs_tiled_ci_excludes_zero"]
        and value["delta_hist_vs_tiled"]["mean"] > 0.0
    ]
    output["gates"]["lspwm_q_history_beats_current_and_tiled"] = bool(passing)
    output["gates"]["lspwm_q_passing_conditions"] = passing

    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n[R1v6] Written to {OUTPUT}")
    print(f"[R1v6] lspwm_q gate={bool(passing)}")
    for cond, data in sorted(result.items()):
        dh = data['delta_hist_vs_current']
        dt = data['delta_hist_vs_tiled']
        print(f"  {cond}: delta_hist={dh['mean']:.4f} [{dh['lower']:.4f},{dh['upper']:.4f}] excl0={data['delta_hist_vs_current_ci_excludes_zero']}, delta_vs_tiled={dt['mean']:.4f} excl0={data['delta_hist_vs_tiled_ci_excludes_zero']}")


if __name__ == "__main__":
    main()