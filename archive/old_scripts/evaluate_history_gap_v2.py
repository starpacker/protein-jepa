#!/usr/bin/env python3
"""R1v2: History Gap with *real* horizons and capacity-matched probes.

What was wrong with the original ``evaluate_history_gap.py``
------------------------------------------------------------
1. ``for hf in horizons`` only changed a dictionary key.  The endpoint pairs
   came from ``paired_horizons_frames=[24, 48, 96]`` sampling, so every reported
   "horizon" was the same experiment with a randomly mixed 24/48/96-frame lag.
2. The train/validation split was a random permutation of *frames*, so
   neighbouring frames of one trajectory landed in both splits (temporal
   leakage) and the reported unit was a frame, not a protein.
3. The "current" probe was a single ``nn.Linear(D, D)`` while the "history"
   probe was an MLP with 256-unit hidden layers, so the measured gap partly
   measured probe capacity rather than extra temporal information.
4. The PCA/tICA/VAMP "baselines" were fitted on the *JEPA latent*, not on CA
   coordinates, and were fitted on all frames including the evaluation ones.

What this script does instead
-----------------------------
* States are the ordered native LSP-WM ``q`` trajectory (one value per
  consecutive causal clip), extracted by
  ``_shared.latent_trajectories.extract_ordered_latent_trajectories``.  The
  physical step is ``frame_interval_ps * window_stride`` (8000 ps by default),
  so horizon ``h`` is a real, monotone time lag.
* Both probes are the *same* deterministic ridge learner with the *same*
  regularization rule; only the feature set changes.
* Two controls separate "extra parameters" from "extra history":
  ``current_tiled_H`` (identical feature count, no new information) and
  ``history_shuffled`` (same realizations, time alignment destroyed).
* Every trajectory is split into a training prefix and a held-out suffix.
* Inference is protein-level: trajectories are averaged within a protein and
  the bootstrap resamples proteins.
* Coordinate baselines (PCA/tICA/VAMP) are fitted on **train-prefix CA
  coordinates only**, with lag pairs taken strictly *within* trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from artifacts.reviewer_experiments._shared.latent_trajectories import (  # noqa: E402
    extract_ordered_latent_trajectories,
)
from artifacts.reviewer_experiments._shared.lspwm_adapter import (  # noqa: E402
    LSPWMDynamicsModel,
)



# ---------------------------------------------------------------------------
# Capacity-matched ridge probe
# ---------------------------------------------------------------------------

RIDGE_GRID = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0, 1000.0)


def _ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    penalty_multiplier: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an intercept ridge with a scale-adaptive regularizer.

    ``penalty = penalty_multiplier * trace(GᵀG)/D``.  The multiplier is selected
    per feature set by an inner split of the training prefix (see
    ``_select_ridge``); a *fixed* multiplier cannot be fair across feature sets,
    because a 2048-column history design needs far more shrinkage than a
    256-column current-state design.
    """
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    gram = design.T @ design
    penalty = penalty_multiplier * float(np.trace(gram) / max(design.shape[1], 1))
    reg = np.eye(design.shape[1], dtype=np.float64) * penalty
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(gram + reg, design.T @ train_y)
    return weights[:-1], weights[-1]


def _select_ridge(
    matrix: np.ndarray,
    targets: np.ndarray,
    *,
    inner_fraction: float = 0.75,
    grid: tuple[float, ...] = RIDGE_GRID,
) -> tuple[float, dict[str, float]]:
    """Pick the ridge multiplier on a held-out tail of the training prefix only.

    Returns ``(best_multiplier, validation_mse_by_multiplier)``.  Falls back to
    ``1e-3`` (and reports ``{}``) when the inner split is too small, which is
    recorded per trajectory.
    """
    cut = max(2, int(len(matrix) * inner_fraction))
    inner_x, inner_y = matrix[:cut], targets[:cut]
    validation_x, validation_y = matrix[cut:], targets[cut:]
    if len(validation_x) < 2:
        return 1.0e-3, {}
    scores = {}
    for multiplier in grid:
        weights, bias = _ridge(inner_x, inner_y, multiplier)
        scores[float(multiplier)] = _mse(weights, bias, validation_x, validation_y)
    return float(min(scores, key=scores.get)), scores


def _mse(weights: np.ndarray, bias: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(x, dtype=np.float64) @ weights + bias - y)))


def _history_features(sequence: np.ndarray, history_len: int) -> np.ndarray:
    """``[z_{t-H+1} .. z_t]`` flattened for every valid ``t``."""
    block = np.stack(
        [sequence[history_len - 1 - lag : len(sequence) - lag] for lag in range(history_len)],
        axis=1,
    )
    return block.reshape(block.shape[0], -1)


def _shuffled_history_features(
    sequence: np.ndarray, history_len: int, shift: int
) -> np.ndarray:
    """History windows drawn from a time-shifted copy of the same trajectory.

    The feature count and the marginal distribution of the entries match
    ``_history_features``; only the temporal alignment with the target is
    destroyed.
    """
    n_valid = len(sequence) - history_len + 1
    source = (np.arange(n_valid) + shift) % n_valid
    block = np.stack(
        [sequence[source + history_len - 1 - lag] for lag in range(history_len)],
        axis=1,
    )
    return block.reshape(block.shape[0], -1)


def _probe_row(
    sequence: np.ndarray,
    horizon: int,
    history_len: int,
    *,
    train_fraction: float = 0.8,
    min_train: int = 20,
    min_test: int = 4,
) -> dict | None:
    """Return per-trajectory held-out MSEs for every feature set."""
    length = len(sequence)
    n_windows = length - history_len + 1
    n_samples = n_windows - horizon
    if n_samples < min_train + min_test:
        return None

    targets = sequence[history_len - 1 + horizon : length]
    current = sequence[history_len - 1 : length - horizon]
    # ``_history_features`` returns one row per window start; the last
    # ``horizon`` rows have no observed target and are dropped here so that
    # every feature matrix is aligned with ``targets``.
    features = {
        "current": current,
        "history": _history_features(sequence, history_len)[:n_samples],
        "current_tiled_H": np.concatenate([current] * history_len, axis=1),
        "history_shuffled": _shuffled_history_features(
            sequence, history_len, max(history_len, min(n_samples // 2, 37 + horizon))
        )[:n_samples],
    }
    expected_rows = n_samples
    for name, matrix in features.items():
        if matrix.shape[0] != expected_rows:
            raise AssertionError(
                f"feature {name} has {matrix.shape[0]} rows, expected {expected_rows}"
            )

    cut = max(min_train, int(n_samples * train_fraction))
    if n_samples - cut < min_test:
        return None

    row = {
        "n_train": int(cut), "n_test": int(n_samples - cut),
        "target_std": float(np.mean(np.std(targets, axis=0))),
        "target_std_test": float(np.mean(np.std(targets[cut:], axis=0))),
    }
    for name, matrix in features.items():
        # Ridge strength is chosen per feature set on an inner split of the
        # training prefix, then refitted on the whole prefix.  Without this the
        # 2048-column history/tiled designs are systematically penalised
        # relative to the 256-column current-state design.
        multiplier, scores = _select_ridge(matrix[:cut], targets[:cut])
        weights, bias = _ridge(matrix[:cut], targets[:cut], multiplier)
        row[f"mse_{name}"] = _mse(weights, bias, matrix[cut:], targets[cut:])
        row[f"ridge_multiplier_{name}"] = float(multiplier)
        row[f"ridge_inner_validation_size_{name}"] = int(len(scores))
    # Floor control: predict the training mean of the target.
    mean_prediction = np.mean(targets[:cut], axis=0)
    row["mse_train_mean"] = float(np.mean(np.square(targets[cut:] - mean_prediction)))
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

DELTA_DEFINITIONS = {
    "delta_hist_vs_current": ("mse_current", "mse_history"),
    "delta_hist_vs_tiled": ("mse_current_tiled_H", "mse_history"),
    "delta_hist_vs_shuffled": ("mse_history_shuffled", "mse_history"),
    "delta_mean_vs_current": ("mse_current", "mse_train_mean"),
}


def _bootstrap_mean(values: np.ndarray, n_resamples: int = 4000, seed: int = 6303):
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


def aggregate(rows: list[dict], metadata: dict[str, dict], seed: int = 6303) -> dict:
    """Average trajectories within protein, then bootstrap over proteins."""
    by_protein: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_protein[metadata[row["trajectory_id"]]["protein_id"]].append(row)

    protein_rows = {}
    for protein_id, protein_values in sorted(by_protein.items()):
        keys = protein_values[0].keys()
        protein_rows[protein_id] = {
            key: float(np.mean([float(value[key]) for value in protein_values]))
            for key in keys if key != "trajectory_id"
        }

    output = {
        "n_trajectories": len(rows),
        "n_proteins": len(protein_rows),
        "mean_mse_current": float(np.mean([r["mse_current"] for r in protein_rows.values()])),
        "mean_mse_history": float(np.mean([r["mse_history"] for r in protein_rows.values()])),
        "mean_mse_current_tiled_H": float(
            np.mean([r["mse_current_tiled_H"] for r in protein_rows.values()])),
        "mean_mse_history_shuffled": float(
            np.mean([r["mse_history_shuffled"] for r in protein_rows.values()])),
        "mean_mse_train_mean": float(np.mean([r["mse_train_mean"] for r in protein_rows.values()])),
        "mean_target_std": float(np.mean([r["target_std"] for r in protein_rows.values()])),
        "per_protein_delta_hist_vs_current": {
            protein_id: values["mse_current"] - values["mse_history"]
            for protein_id, values in sorted(protein_rows.items())
        },
    }
    for name, (base_key, ref_key) in DELTA_DEFINITIONS.items():
        delta = np.asarray(
            [values[base_key] - values[ref_key] for values in protein_rows.values()],
            dtype=float,
        )
        stats = _bootstrap_mean(delta, seed=seed)
        output[name] = stats
        output[f"{name}_ci_excludes_zero"] = bool(
            np.isfinite(stats["lower"]) and np.isfinite(stats["upper"])
            and (stats["lower"] > 0.0 or stats["upper"] < 0.0)
        )
    current = output["mean_mse_current"]
    output["relative_gain_vs_current"] = float(
        (current - output["mean_mse_history"]) / current) if current > 0 else float("nan")
    output["per_protein"] = protein_rows
    return output




# ---------------------------------------------------------------------------
# Coordinate baselines (CA coordinates, train-prefix fit only)
# ---------------------------------------------------------------------------

def _coordinate_sequences(
    manifest_path: str,
    *,
    clip_frames: int,
    window_stride: int,
    min_states: int,
    max_proteins: int | None,
) -> dict[str, np.ndarray]:
    """Aligned CA features sampled on the clip-endpoint time grid.

    ``extract_features_for_protein`` returns every frame of a trajectory, so
    taking ``frame_stride=1`` and then slicing at the clip endpoints makes the
    coordinate and latent sequences share both their time origin and their
    physical step.  No interpolation or alignment fudge factor is needed.
    """
    from baselines.data import extract_features_for_protein, load_all_proteins

    grouped = load_all_proteins(manifest_path, max_proteins=max_proteins)
    endpoint_offset = clip_frames - 1
    out: dict[str, np.ndarray] = {}
    for protein_id, records in grouped.items():
        try:
            features = extract_features_for_protein(records, frame_stride=1)
        except Exception as exc:  # pragma: no cover - data dependent
            print(f"  [coord] skip {protein_id}: {type(exc).__name__}: {exc}")
            continue
        for trajectory_id, matrix in features.items():
            sampled = matrix[endpoint_offset::window_stride]
            if len(sampled) >= min_states:
                out[trajectory_id] = np.asarray(sampled, dtype=np.float64)
    return out


def _fit_transform_one_protein(
    name: str,
    sequences: dict[str, np.ndarray],
    train_ids: list[str],
    *,
    dim: int,
    lag: int,
    min_prefix: int = 8,
) -> dict[str, np.ndarray]:
    """Fit a representation on one protein's train prefixes, then transform.

    ``sequences`` must contain only trajectories of a **single** protein, so
    that every row lives in the same ``3 * n_residues`` feature space.  Only the
    per-trajectory 80% prefixes of ``train_ids`` are used for the fit; the
    evaluation suffixes never enter the estimator.
    """
    if not train_ids:
        raise RuntimeError(f"no trajectory is long enough to fit {name}")
    prefixes = {
        trajectory_id: sequences[trajectory_id][
            : max(min_prefix, int(len(sequences[trajectory_id]) * 0.8))
        ]
        for trajectory_id in train_ids
    }

    pooled = np.concatenate([prefixes[key] for key in train_ids], axis=0)
    mean = pooled.mean(axis=0, keepdims=True)
    scale = pooled.std(axis=0, keepdims=True)
    scale[scale < 1.0e-8] = 1.0

    def _standardize(matrix: np.ndarray) -> np.ndarray:
        return (matrix - mean) / scale

    if name == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=min(dim, pooled.shape[0] - 1, pooled.shape[1] - 1)).fit(_standardize(pooled))

        def transform(matrix: np.ndarray) -> np.ndarray:
            return model.transform(_standardize(matrix))
    elif name in {"tica", "vamp"}:
        from deeptime.decomposition import TICA, VAMP
        pairs_x0, pairs_x1 = [], []
        for trajectory_id in train_ids:
            prefix = _standardize(prefixes[trajectory_id])
            if len(prefix) <= lag:
                continue
            pairs_x0.append(prefix[:-lag])
            pairs_x1.append(prefix[lag:])
        if not pairs_x0:
            raise RuntimeError(f"no within-trajectory lag pairs available for {name}")
        # Lag pairs never cross a trajectory boundary.
        lagged = (np.concatenate(pairs_x0, axis=0), np.concatenate(pairs_x1, axis=0))
        estimator = TICA if name == "tica" else VAMP
        model = estimator(
            lagtime=lag, dim=min(dim, lagged[0].shape[1] - 1),
        ).fit(lagged).fetch_model()

        def transform(matrix: np.ndarray) -> np.ndarray:
            return np.asarray(model.transform(_standardize(matrix)))
    else:
        raise ValueError(f"unknown representation {name}")

    return {
        trajectory_id: np.asarray(transform(matrix), dtype=np.float64)
        for trajectory_id, matrix in sequences.items()
    }


def _fit_transform_representation(
    name: str,
    sequences: dict[str, np.ndarray],
    protein_of: dict[str, str],
    *,
    dim: int,
    lag: int,
    min_prefix: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    """Fit one representation **per protein** and transform every trajectory.

    Coordinate features have ``3 * n_residues`` dimensions, so two proteins live
    in different feature spaces and cannot be pooled into a single estimator
    (the v1 script effectively pretended they could).  Each protein therefore
    gets its own PCA / tICA / VAMP, fitted on that protein's train prefixes.

    Returns ``(sequences, dim_per_protein, skipped_reasons)``.
    """
    by_protein: dict[str, list[str]] = defaultdict(list)
    for trajectory_id, matrix in sequences.items():
        if len(matrix) >= 2 * min_prefix:
            by_protein[str(protein_of.get(trajectory_id, trajectory_id))].append(trajectory_id)

    transformed: dict[str, np.ndarray] = {}
    dims: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for protein_id in sorted(by_protein):
        train_ids = sorted(by_protein[protein_id])
        group = {
            trajectory_id: matrix
            for trajectory_id, matrix in sequences.items()
            if str(protein_of.get(trajectory_id, trajectory_id)) == protein_id
        }
        try:
            group_transformed = _fit_transform_one_protein(
                name, group, train_ids, dim=dim, lag=lag, min_prefix=min_prefix,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            skipped[protein_id] = f"{type(exc).__name__}: {exc}"
            continue
        transformed.update(group_transformed)
        dims[protein_id] = int(np.asarray(group_transformed[train_ids[0]]).shape[1])
    return transformed, dims, skipped


def _summarise_skip_reasons(skipped: dict[str, str]) -> str:
    """Collapse per-protein skip reasons into a short, deterministic summary."""
    if not skipped:
        return "no protein was skipped"
    counts: dict[str, int] = defaultdict(int)
    for reason in skipped.values():
        counts[reason] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{count}x {reason}" for reason, count in ordered)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_representation(
    name: str,
    sequences: dict[str, np.ndarray],
    metadata: dict[str, dict],
    *,
    horizons: list[int],
    history_lengths: list[int],
    seed: int,
) -> dict:
    output: dict[str, dict] = {}
    for horizon in horizons:
        for history_len in history_lengths:
            rows = []
            for trajectory_id, sequence in sequences.items():
                if trajectory_id not in metadata:
                    continue
                row = _probe_row(sequence, horizon, history_len)
                if row is None:
                    continue
                row["trajectory_id"] = trajectory_id
                rows.append(row)
            if len(rows) < 2:
                continue
            key = f"h={horizon}/H={history_len}"
            summary = aggregate(rows, metadata, seed=seed)
            summary["history_len"] = history_len
            summary["horizon_steps"] = horizon
            output[key] = summary
            print(
                f"  {name:10s} {key:12s} "
                f"mse_cur={summary['mean_mse_current']:.4e} "
                f"mse_hist={summary['mean_mse_history']:.4e} "
                f"mse_tile={summary['mean_mse_current_tiled_H']:.4e} "
                f"d_hist={summary['delta_hist_vs_current']['mean']:+.4e} "
                f"[{summary['delta_hist_vs_current']['lower']:+.4e},"
                f"{summary['delta_hist_vs_current']['upper']:+.4e}] "
                f"d_vs_tile={summary['delta_hist_vs_tiled']['mean']:+.4e} "
                f"n_prot={summary['n_proteins']}"
            )
    return output


def _load_or_extract_latents(args):
    """Return ``(sequences, metadata, loader_stats)``, using the npz cache."""
    cache = Path(args.latent_cache) if args.latent_cache else None
    if cache is not None and cache.exists():
        print(f"[R1v2] loading cached latents from {cache}")
        with np.load(cache, allow_pickle=True) as handle:
            sequences = {key: handle[key] for key in handle.files if not key.startswith("meta_")}
            metadata = json.loads(str(handle["meta_metadata"].item()))
            loader_stats = json.loads(str(handle["meta_loader_stats"].item()))
        if args.max_batches and loader_stats.get("truncated_by_max_batches"):
            print("[R1v2] cached latents were truncated by max_batches; re-extracting")
        else:
            return sequences, metadata, loader_stats

    adapter = LSPWMDynamicsModel()
    adapter.load(
        checkpoint_path=args.checkpoint, config_path=args.config,
        manifest=args.manifest, device=args.device,
        batch_size=args.batch_size, max_batches=1, seed=args.seed,
    )
    sequences, metadata, loader_stats = extract_ordered_latent_trajectories(
        adapter, args.manifest,
        batch_size=args.batch_size,
        clip_frames=args.clip_frames,
        window_stride=args.window_stride,
        max_batches=args.max_batches or None,
        min_states=args.min_states,
    )
    adapter.close()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, **sequences,
            meta_metadata=np.array(json.dumps(metadata)),
            meta_loader_stats=np.array(json.dumps(loader_stats)),
        )
        print(f"[R1v2] cached latents -> {cache}")
    return sequences, metadata, loader_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        default="artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt")
    parser.add_argument("--config",
                        default="artifacts/lspwm_v44_stage_b_v5_fixed_pmf/eval_improved/eval_config.json")
    parser.add_argument("--manifest", default="manifests/tmjepa_v3.validation.jsonl")
    parser.add_argument("--output",
                        default="artifacts/reviewer_experiments/representation_closure_v2/r1_history_gap_v2.json")
    parser.add_argument("--latent-cache",
                        default="artifacts/reviewer_experiments/representation_closure_v2/r1_latents.npz")
    parser.add_argument("--representations", nargs="+", default=["lspwm_q", "pca", "tica", "vamp"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--history-lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--clip-frames", type=int, default=24)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = no cap (full manifest)")
    parser.add_argument("--min-states", type=int, default=16)
    parser.add_argument("--max-proteins", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=6303)
    args = parser.parse_args()

    requested_deeptime = [
        name for name in args.representations if name in {"tica", "vamp"}
    ]
    if requested_deeptime:
        try:
            import deeptime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                f"R1v2 needs the 'deeptime' package for {requested_deeptime} ({exc}). "
                f"Use an interpreter that has it (the project environment is "
                f"/liuyutian/protein_jepa/.venv/bin/python) or drop those "
                f"representations from --representations."
            )

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output = {
        "experiment_id": "R1",
        "protocol": {
            "name": "history_gap_ordered_v2",
            "probe": "deterministic_ridge_scale_adaptive_l2_with_inner_validation_per_feature_set",
            "ridge_grid": list(RIDGE_GRID),
            "ridge_selection": "inner 75/25 split of the training prefix, per feature set",
            "capacity_controls": ["current_tiled_H", "history_shuffled"],
            "split": "per_trajectory_prefix(80%)/suffix(20%)",
            "aggregation": "trajectory_mean_within_protein_then_protein_bootstrap",
            "horizons": args.horizons,
            "history_lengths": args.history_lengths,
            "representations": args.representations,
            "clip_frames": args.clip_frames,
            "window_stride": args.window_stride,
            "source_frames_per_step": args.window_stride,
            "no_fallback": True,
        },
        "provenance": {
            "checkpoint": args.checkpoint, "config": args.config,
            "manifest": args.manifest, "seed": args.seed, "device": args.device,
            "python": sys.executable,
        },
        "representations": {},
        "gates": {},
        "errors": [],
        "warnings": [],
    }
    sequences, metadata, loader_stats = _load_or_extract_latents(args)


    dt_values = {round(float(value["dt_ps"]), 6) for value in metadata.values()}
    if len(dt_values) != 1:
        raise RuntimeError(f"R1v2 requires one common sampled dt, found {sorted(dt_values)}")
    dt_ps = float(next(iter(dt_values)))

    output["dataset"] = {
        "n_trajectories": len(sequences),
        "n_proteins": len({value["protein_id"] for value in metadata.values()}),
        "dt_ps": dt_ps,
        "loader_stats": loader_stats,
    }
    output["physical_horizon_ps"] = {f"h={h}": h * dt_ps for h in args.horizons}
    if loader_stats.get("truncated_by_max_batches"):
        output["warnings"].append(
            f"max_batches={args.max_batches} truncated the manifest: {loader_stats['clips_seen']} "
            f"clips were used out of an estimated {loader_stats['uncapped_est_batches']} batches"
        )
    print(f"[R1v2] trajectories={len(sequences)} proteins={output['dataset']['n_proteins']} "
          f"dt_ps={dt_ps} clips={loader_stats['clips_seen']}")

    coordinate_sequences = None
    prepared: dict[str, dict] = {}
    for name in args.representations:
        if name == "lspwm_q":
            prepared[name] = {
                "sequences": dict(sequences),
                "metadata": dict(metadata),
                "dim_per_protein": {},
                "dim": int(next(iter(sequences.values())).shape[1]),
                "skipped_proteins": {},
            }
            continue
        if coordinate_sequences is None:
            coordinate_sequences = _coordinate_sequences(
                args.manifest,
                clip_frames=args.clip_frames,
                window_stride=args.window_stride,
                min_states=args.min_states,
                max_proteins=args.max_proteins or None,
            )
            print(f"[R1v2] coordinate sequences: {len(coordinate_sequences)}")
        usable = {
            key: value for key, value in coordinate_sequences.items() if key in sequences
        }
        usable_protein_of = {key: metadata[key]["protein_id"] for key in usable}
        transformed, dims, skipped = _fit_transform_representation(
            name, usable, usable_protein_of, dim=args.dim, lag=args.lag,
        )
        prepared[name] = {
            "sequences": transformed,
            "metadata": {key: metadata[key] for key in transformed},
            "dim_per_protein": dims,
            "dim": sorted(set(dims.values())),
            "skipped_proteins": skipped,
        }
        print(f"[R1v2] {name}: {len(transformed)} trajectories, {len(dims)} proteins, "
              f"dims={sorted(set(dims.values()))}, skipped={len(skipped)}")

    failed_representations = {
        name: entry for name, entry in prepared.items() if not entry["sequences"]
    }
    usable_representations = [
        name for name in args.representations if name not in failed_representations
    ]
    protein_sets = {
        name: {meta["protein_id"] for meta in prepared[name]["metadata"].values()}
        for name in usable_representations
    }
    common_proteins = (
        sorted(set.intersection(*protein_sets.values())) if protein_sets else []
    )
    output["comparison_protein_ids"] = common_proteins
    output["proteins_per_representation"] = {
        name: sorted(values) for name, values in protein_sets.items()
    }
    if len(usable_representations) < len(args.representations):
        output["warnings"].append(
            f"representations with no usable protein: {sorted(failed_representations)}; "
            "they are excluded from the comparison protein set "
            "(a single empty representation must not void the whole comparison)"
        )
    if len(common_proteins) < 10:
        output["warnings"].append(
            f"only {len(common_proteins)} proteins are shared by every usable representation; "
            "the cross-representation comparison is restricted to that intersection"
        )
    print(f"[R1v2] common proteins across representations: {len(common_proteins)} "
          f"(usable representations: {usable_representations})")

    for name in args.representations:
        entry = prepared[name]
        if name in failed_representations:
            reason = _summarise_skip_reasons(entry["skipped_proteins"])
            output["representations"][name] = {
                "status": "failed",
                "dim": None,
                "skip_reason_summary": reason,
                "skipped_proteins": entry["skipped_proteins"],
                "n_trajectories": 0,
                "n_proteins": 0,
                "by_condition": {},
            }
            output["gates"][f"{name}_history_beats_current_and_tiled"] = False
            output["gates"][f"{name}_passing_conditions"] = []
            output["errors"].append(
                f"representation '{name}' produced no usable protein "
                f"({len(entry['skipped_proteins'])} proteins skipped): {reason}"
            )
            print(f"[R1v2] === {name} FAILED: {reason} ===")
            continue
        probe_sequences = {
            key: value for key, value in entry["sequences"].items()
            if entry["metadata"][key]["protein_id"] in common_proteins
        }
        probe_metadata = {key: entry["metadata"][key] for key in probe_sequences}
        dimension = entry["dim"]
        print(f"[R1v2] === {name} (dim={dimension}) n_traj={len(probe_sequences)} ===")
        result = run_representation(
            name, probe_sequences, probe_metadata,
            horizons=args.horizons, history_lengths=args.history_lengths, seed=args.seed,
        )
        output["representations"][name] = {
            "status": "ok",
            "dim": dimension,
            "dim_per_protein": entry["dim_per_protein"],
            "skipped_proteins": entry["skipped_proteins"],
            "n_trajectories": len(probe_sequences),
            "n_proteins": len({
                meta["protein_id"] for meta in probe_metadata.values()
            }),
            "n_train_windows_gate": all(
                value.get("n_train", 0) >= 2 for value in result.values()
            ),
            "by_condition": result,
        }
        passing = [
            key for key, value in result.items()
            if value["delta_hist_vs_current_ci_excludes_zero"]
            and value["delta_hist_vs_tiled_ci_excludes_zero"]
            and value["delta_hist_vs_tiled"]["mean"] > 0.0
        ]
        output["gates"][f"{name}_history_beats_current_and_tiled"] = bool(passing)
        output["gates"][f"{name}_passing_conditions"] = passing

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"[R1v2] wrote {out_path}")
    if output["errors"]:
        for message in output["errors"]:
            print(f"[R1v2] ERROR: {message}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
