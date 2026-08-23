#!/usr/bin/env python3
"""Paper-grade evaluation utilities for Protein JEPA Frame.

VAMP-2 scoring is delegated to deeptime's official VAMP implementation. The
surrounding code only handles feature extraction, blocked splits, baselines,
and result aggregation:

* blocked cross-validated VAMP-2 with train-selected preprocessing;
* learned representation vs deterministic geometry baselines;
* per-protein MSM probing with trajectory-heldout splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import TrajectoryRecord, load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA


DEFAULT_MANIFESTS = Path("/liuyutian/protein_jepa/artifacts/manifests")
DEFAULT_LABELS = Path("/liuyutian/protein_jepa/artifacts/msm_state_labels")
DEFAULT_VAL_MANIFEST = DEFAULT_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl"
DEFAULT_VAL_LABELS = DEFAULT_LABELS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered_labels.json"
MODEL_READOUT_CHOICES = [
    "mean",
    "state",
    "clip_state",
    "state_clip",
    "state_delta",
    "state_clip_delta",
    "state_mean",
    "state_clip_delta_mean",
    "kinetic",
]


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    checkpoint: Path | None
    artifact_dir: Path
    kind: str = "checkpoint"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _load_checkpoint_spec(spec: str) -> CheckpointSpec:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        name = Path(raw_path).parent.name
    if raw_path == "random":
        raise ValueError("use --include-random for a random-encoder baseline")
    checkpoint = Path(raw_path)
    artifact_dir = checkpoint.parent
    return CheckpointSpec(name=name, checkpoint=checkpoint, artifact_dir=artifact_dir)


def _model_from_spec(spec: CheckpointSpec, device: torch.device) -> tuple[ProteinJEPA, dict]:
    cfg = _load_json(spec.artifact_dir / "config.json")
    model_cfg = ModelConfig(**cfg["model"])
    model = ProteinJEPA(model_cfg).to(device)
    if spec.kind == "checkpoint":
        ckpt = torch.load(spec.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        cfg["_checkpoint_step"] = ckpt.get("step", "?")
        cfg["_checkpoint_epoch"] = ckpt.get("epoch", "?")
    else:
        cfg["_checkpoint_step"] = "random_init"
        cfg["_checkpoint_epoch"] = "random_init"
    model.eval()
    return model, cfg


def _data_cfg_from_artifact(artifact_dir: Path, batch_size: int, num_frames: int, window_stride: int) -> DataConfig:
    cfg = _load_json(artifact_dir / "config.json")
    return DataConfig(
        manifest_path=None,
        batch_size=batch_size,
        num_workers=0,
        num_frames=num_frames,
        frame_stride=1,
        window_stride=window_stride,
        max_residues=cfg["data"].get("max_residues"),
        coordinate_scale=cfg["data"].get("coordinate_scale", 1.0),
    )


def _window_overlap_ratio(num_frames: int, window_stride: int) -> float:
    if num_frames <= 0:
        return 0.0
    return float(max(0, num_frames - window_stride) / num_frames)


def _first_n_trajectories(records: list[TrajectoryRecord], n: int) -> list[TrajectoryRecord]:
    trajs: OrderedDict[str, TrajectoryRecord] = OrderedDict()
    for record in records:
        if record.trajectory_id not in trajs:
            trajs[record.trajectory_id] = record
        if len(trajs) >= n:
            break
    return list(trajs.values())


def _move_tensor_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _model_readout(model: ProteinJEPA, latents: torch.Tensor, residue_mask: torch.Tensor, readout: str) -> torch.Tensor:
    valid = residue_mask.unsqueeze(-1).float()
    frame_mean = (latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
    last_mean = frame_mean[:, -1]
    if readout == "mean":
        return last_mean
    states = model.encode_states(latents, residue_mask)
    frame_states = states["frame_states"]
    last_state = frame_states[:, -1]
    clip_state = states["clip_state"]
    if readout == "kinetic":
        kinetic = model.encode_kinetic_readout(latents, residue_mask)
        return kinetic[:, -1]
    if frame_states.shape[1] > 1:
        delta_state = frame_states[:, -1] - frame_states[:, -2]
    else:
        delta_state = torch.zeros_like(last_state)
    if readout == "state":
        return last_state
    if readout == "clip_state":
        return clip_state
    if readout == "state_clip":
        return torch.cat([last_state, clip_state], dim=-1)
    if readout == "state_delta":
        return torch.cat([last_state, delta_state], dim=-1)
    if readout == "state_clip_delta":
        return torch.cat([last_state, clip_state, delta_state], dim=-1)
    if readout == "state_mean":
        return torch.cat([last_state, last_mean], dim=-1)
    if readout == "state_clip_delta_mean":
        return torch.cat([last_state, clip_state, delta_state, last_mean], dim=-1)
    raise ValueError(f"unknown model readout {readout}")


@torch.no_grad()
def extract_model_frame_sequences(
    model: ProteinJEPA,
    records: list[TrajectoryRecord],
    data_cfg: DataConfig,
    device: torch.device,
    readout: str = "mean",
) -> dict[str, np.ndarray]:
    """Extract one pooled latent vector per saved frame for each trajectory."""
    dataset = TrajectoryClipDataset(records, data_cfg)
    index_map = {}
    for i in range(len(dataset)):
        record, frames = dataset._locate(i)
        index_map[i] = (record.trajectory_id, int(frames[-1]))

    loader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clips,
    )

    buffers: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    offset = 0
    for batch in loader:
        batch = _move_tensor_batch(batch, device)
        latents = model.context_encoder(
            coords=batch["coords"],
            residue_mask=batch["residue_mask"],
            aatype=batch["aatype"],
            residue_index=batch["residue_index"],
            chain_id=batch["chain_id"],
            token_mask=None,
        )
        pooled = _model_readout(model, latents, batch["residue_mask"], readout)
        B = pooled.shape[0]
        for b in range(B):
            tid, frame = index_map[offset + b]
            buffers[tid].append((frame, pooled[b].detach().cpu().numpy()))
        offset += B
    return _sorted_sequence_dict(buffers)


def extract_baseline_frame_sequences(
    records: list[TrajectoryRecord],
    data_cfg: DataConfig,
    baseline: str,
) -> dict[str, np.ndarray]:
    dataset = TrajectoryClipDataset(records, data_cfg)
    index_map = {}
    for i in range(len(dataset)):
        record, frames = dataset._locate(i)
        index_map[i] = (record.trajectory_id, int(frames[-1]))
    loader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clips,
    )
    buffers: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    offset = 0
    for batch in loader:
        feats = _baseline_features(batch, baseline, fixed_residues=data_cfg.max_residues)
        for b in range(feats.shape[0]):
            tid, frame = index_map[offset + b]
            buffers[tid].append((frame, feats[b]))
        offset += feats.shape[0]
    return _sorted_sequence_dict(buffers)


def _sorted_sequence_dict(buffers: dict[str, list[tuple[int, np.ndarray]]]) -> dict[str, np.ndarray]:
    out = {}
    for tid, items in buffers.items():
        items.sort(key=lambda item: item[0])
        out[tid] = np.stack([value for _, value in items], axis=0).astype(np.float64, copy=False)
    return out


def _baseline_features(batch: dict, baseline: str, fixed_residues: int | None = None) -> np.ndarray:
    coords = batch["coords"][:, -1].numpy()
    residue_mask = batch["residue_mask"][:, -1].numpy().astype(bool)
    if baseline == "geometry_stats":
        return np.stack([_geometry_stats(coords[i], residue_mask[i]) for i in range(coords.shape[0])], axis=0)
    if baseline == "centered_ca":
        return np.stack(
            [_centered_ca_flat(coords[i], residue_mask[i], fixed_residues=fixed_residues) for i in range(coords.shape[0])],
            axis=0,
        )
    raise ValueError(f"unknown baseline {baseline}")


def _geometry_stats(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = coords[mask]
    if valid.shape[0] == 0:
        return np.zeros(38, dtype=np.float64)
    ca = valid[:, 1, :]
    internal_pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3))
    parts = []
    for i, j in internal_pairs:
        d = np.linalg.norm(valid[:, i, :] - valid[:, j, :], axis=-1)
        parts.extend([d.mean(), d.std(), np.quantile(d, 0.1), np.quantile(d, 0.5), np.quantile(d, 0.9)])
    centered = ca - ca.mean(axis=0, keepdims=True)
    rg = math.sqrt(float((centered * centered).sum(axis=1).mean()))
    parts.append(rg)
    if ca.shape[0] >= 2:
        adj = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
        parts.extend([adj.mean(), adj.std(), adj.min(), adj.max()])
    else:
        parts.extend([0.0, 0.0, 0.0, 0.0])
    if ca.shape[0] >= 3:
        pd = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        iu = np.triu_indices(ca.shape[0], k=1)
        pair = pd[iu]
        parts.extend([pair.mean(), pair.std(), pair.min(), pair.max(), *np.quantile(pair, [0.1, 0.25, 0.5, 0.75, 0.9])])
    else:
        parts.extend([0.0] * 9)
    return np.asarray(parts, dtype=np.float64)


def _centered_ca_flat(coords: np.ndarray, mask: np.ndarray, fixed_residues: int | None = None) -> np.ndarray:
    fixed = fixed_residues or coords.shape[0]
    ca = np.zeros((fixed, 3), dtype=np.float64)
    n = min(coords.shape[0], fixed)
    local = coords[:n, 1, :].astype(np.float64, copy=True)
    local_mask = mask[:n]
    if local_mask.any():
        local[local_mask] -= local[local_mask].mean(axis=0, keepdims=True)
    local[~local_mask] = 0.0
    ca[:n] = local
    return ca.reshape(-1)


def _standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    keep = np.squeeze(std > 1.0e-8)
    if keep.ndim == 0:
        keep = np.asarray([bool(keep)])
    train_s = (train[:, keep] - mean[:, keep]) / std[:, keep]
    val_s = (val[:, keep] - mean[:, keep]) / std[:, keep]
    return train_s, val_s


def _pca_project(train: np.ndarray, val: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[1] <= max_dim:
        return train, val
    _, _, vt = np.linalg.svd(train, full_matrices=False)
    basis = vt[:max_dim].T
    return train @ basis, val @ basis


def _cov_inv_sqrt(cov: np.ndarray, eps: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    return (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T


def _vamp2_from_pairs(
    x0_train: np.ndarray,
    x1_train: np.ndarray,
    x0_val: np.ndarray,
    x1_val: np.ndarray,
    dims: list[int],
    eps: float,
    max_pre_dim: int,
) -> dict[str, float]:
    from deeptime.decomposition import VAMP

    train_pair = np.concatenate([x0_train, x1_train], axis=0)
    val_pair = np.concatenate([x0_val, x1_val], axis=0)
    train_pair, val_pair = _standardize(train_pair, val_pair)
    if train_pair.shape[1] == 0:
        return {}
    train_pair, val_pair = _pca_project(train_pair, val_pair, max_pre_dim)
    n0 = x0_train.shape[0]
    nv0 = x0_val.shape[0]
    x0_train = train_pair[:n0]
    x1_train = train_pair[n0:]
    x0_val = val_pair[:nv0]
    x1_val = val_pair[nv0:]

    feature_dim = x0_train.shape[1]
    out = {"feature_dim": float(feature_dim)}
    for dim in dims:
        score_dim = min(dim, feature_dim)
        train_model = VAMP(lagtime=1, dim=score_dim, scaling=None, epsilon=eps).fit((x0_train, x1_train)).fetch_model()
        test_model = VAMP(lagtime=1, dim=score_dim, scaling=None, epsilon=eps).fit((x0_val, x1_val)).fetch_model()
        train_score = train_model.score("VAMP2", epsilon=eps, dim=score_dim)
        cv_score = train_model.score("VAMP2", test_model=test_model, epsilon=eps, dim=score_dim)
        out[f"train_vamp2_dim_{dim}"] = float(train_score)
        out[f"cv_vamp2_dim_{dim}"] = float(cv_score)
        out[f"cv_vamp2_norm_dim_{dim}"] = float(cv_score / max(1, score_dim))
    return out


def blocked_cv_vamp2(
    sequences: dict[str, np.ndarray],
    lags: list[int],
    dims: list[int],
    n_folds: int,
    gap: int,
    eps: float,
    max_pre_dim: int,
    min_pairs: int,
) -> dict:
    results = {}
    for lag in lags:
        per_traj = {}
        fold_rows = []
        for tid, seq in sequences.items():
            n_pairs = len(seq) - lag
            if n_pairs < max(min_pairs, n_folds * 4):
                continue
            fold_edges = np.linspace(0, n_pairs, n_folds + 1, dtype=int)
            traj_rows = []
            for fold in range(n_folds):
                start, end = int(fold_edges[fold]), int(fold_edges[fold + 1])
                val_idx = np.arange(start, end)
                train_mask = np.ones(n_pairs, dtype=bool)
                train_start = max(0, start - gap)
                train_end = min(n_pairs, end + gap)
                train_mask[train_start:train_end] = False
                train_idx = np.nonzero(train_mask)[0]
                if len(val_idx) < 3 or len(train_idx) < 8:
                    continue
                x0_train, x1_train = seq[train_idx], seq[train_idx + lag]
                x0_val, x1_val = seq[val_idx], seq[val_idx + lag]
                try:
                    row = _vamp2_from_pairs(x0_train, x1_train, x0_val, x1_val, dims, eps, max_pre_dim)
                except np.linalg.LinAlgError:
                    continue
                if row:
                    row.update({"trajectory_id": tid, "fold": fold, "n_train_pairs": len(train_idx), "n_val_pairs": len(val_idx)})
                    traj_rows.append(row)
                    fold_rows.append(row)
            if traj_rows:
                per_traj[tid] = _aggregate_rows(traj_rows, dims)
        results[f"lag_{lag}"] = {
            "n_trajs": len(per_traj),
            "n_folds": len(fold_rows),
            "mean": _aggregate_rows(fold_rows, dims) if fold_rows else {},
            "per_trajectory": per_traj,
        }
    return results


def _aggregate_rows(rows: list[dict], dims: list[int]) -> dict:
    if not rows:
        return {}
    keys = ["feature_dim"]
    for dim in dims:
        keys.extend([f"train_vamp2_dim_{dim}", f"cv_vamp2_dim_{dim}", f"cv_vamp2_norm_dim_{dim}"])
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def run_vamp_cv(args: argparse.Namespace) -> dict:
    import deeptime

    device = _device(args.device)
    records = _first_n_trajectories(load_manifest(args.manifest), args.max_trajs)
    specs = [_load_checkpoint_spec(item) for item in args.checkpoint]
    baselines = args.baseline or ["geometry_stats", "centered_ca"]
    if args.include_random:
        base = specs[0]
        specs.append(CheckpointSpec(name="random_encoder", checkpoint=None, artifact_dir=base.artifact_dir, kind="random"))

    out = {
        "protocol": {
            "implementation": "deeptime.decomposition.VAMP + CovarianceKoopmanModel.score(test_model=...)",
            "deeptime_version": deeptime.__version__,
            "manifest": str(args.manifest),
            "max_trajs": args.max_trajs,
            "num_eval_trajs": len(records),
            "window_stride": args.window_stride,
            "window_overlap_ratio": _window_overlap_ratio(args.sequence_frames, args.window_stride),
            "sequence_frames": args.sequence_frames,
            "readout": args.readout,
            "lags": args.lags,
            "dims": args.dims,
            "n_folds": args.folds,
            "gap": args.gap,
            "max_pre_dim": args.max_pre_dim,
            "min_pairs": args.min_pairs,
        },
        "models": {},
        "baselines": {},
    }

    t0 = time.time()
    for spec in specs:
        print(f"[VAMP-CV] extracting {spec.name}", flush=True)
        model, cfg = _model_from_spec(spec, device)
        data_cfg = _data_cfg_from_artifact(
            spec.artifact_dir, args.batch_size, num_frames=args.sequence_frames, window_stride=args.window_stride
        )
        seqs = extract_model_frame_sequences(model, records, data_cfg, device, readout=args.readout)
        out["models"][spec.name] = {
            "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
            "checkpoint_step": cfg["_checkpoint_step"],
            "checkpoint_epoch": cfg["_checkpoint_epoch"],
            "training_objectives": {
                "msm_state_weight": cfg.get("optimization", {}).get("msm_state_weight", 0.0),
                "vamp2_weight": cfg.get("optimization", {}).get("vamp2_weight", 0.0),
            },
            "n_sequences": len(seqs),
            "vamp2_cv": blocked_cv_vamp2(
                seqs, args.lags, args.dims, args.folds, args.gap, args.eps, args.max_pre_dim, args.min_pairs
            ),
        }
        _write_json(args.output, {**out, "status": f"partial_after_model_{spec.name}"})

    baseline_cfg = _data_cfg_from_artifact(
        specs[0].artifact_dir, args.batch_size, num_frames=args.sequence_frames, window_stride=args.window_stride
    )
    for baseline in baselines:
        print(f"[VAMP-CV] extracting baseline {baseline}", flush=True)
        seqs = extract_baseline_frame_sequences(records, baseline_cfg, baseline)
        out["baselines"][baseline] = {
            "n_sequences": len(seqs),
            "vamp2_cv": blocked_cv_vamp2(
                seqs, args.lags, args.dims, args.folds, args.gap, args.eps, args.max_pre_dim, args.min_pairs
            ),
        }
        _write_json(args.output, {**out, "status": f"partial_after_baseline_{baseline}"})

    add_vamp_pairwise_comparisons(out)
    out["elapsed_s"] = round(time.time() - t0, 2)
    _write_json(args.output, out)
    _print_vamp_cv_summary(out, args.dims)
    return out


def _write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)
    print(f"saved {path}", flush=True)


def _metric_at(block: dict, lag: int, dim: int, key: str = "cv_vamp2_norm_dim") -> float | None:
    lag_block = block.get("vamp2_cv", {}).get(f"lag_{lag}", {})
    stat = lag_block.get("mean", {}).get(f"{key}_{dim}", {})
    return stat.get("mean")


def _print_vamp_cv_summary(out: dict, dims: list[int]) -> None:
    lag = out["protocol"]["lags"][0]
    dim = dims[-1]
    print(f"\nVAMP-CV summary, lag={lag}, dim={dim}, normalized score")
    for name, block in out["models"].items():
        value = _metric_at(block, lag, dim)
        print(f"  model {name:24s} {value if value is not None else 'NA'}")
    for name, block in out["baselines"].items():
        value = _metric_at(block, lag, dim)
        print(f"  base  {name:24s} {value if value is not None else 'NA'}")


def _trajectory_metric(block: dict, lag: int, dim: int, metric: str) -> dict[str, float]:
    lag_block = block.get("vamp2_cv", {}).get(f"lag_{lag}", {})
    per = lag_block.get("per_trajectory", {})
    key = f"{metric}_{dim}"
    out = {}
    for tid, stats in per.items():
        value = stats.get(key, {}).get("mean")
        if value is not None and np.isfinite(value):
            out[tid] = float(value)
    return out


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, seed: int = 0, n_boot: int = 2000) -> dict:
    delta = a - b
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, len(delta), size=len(delta))
        means[i] = delta[idx].mean()
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n_pairs": int(len(delta)),
        "fraction_positive": float(np.mean(delta > 0)),
    }


def add_vamp_pairwise_comparisons(out: dict) -> None:
    comparisons = {}
    for lag in out["protocol"]["lags"]:
        lag_key = f"lag_{lag}"
        comparisons[lag_key] = {}
        for dim in out["protocol"]["dims"]:
            dim_key = f"dim_{dim}"
            comparisons[lag_key][dim_key] = {}
            blocks = {f"model:{k}": v for k, v in out["models"].items()}
            blocks.update({f"baseline:{k}": v for k, v in out["baselines"].items()})
            metrics = {name: _trajectory_metric(block, lag, dim, "cv_vamp2_norm_dim") for name, block in blocks.items()}
            names = sorted(metrics)
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    shared = sorted(set(metrics[left]) & set(metrics[right]))
                    if len(shared) < 3:
                        continue
                    a = np.asarray([metrics[left][tid] for tid in shared], dtype=np.float64)
                    b = np.asarray([metrics[right][tid] for tid in shared], dtype=np.float64)
                    comparisons[lag_key][dim_key][f"{left} minus {right}"] = _bootstrap_delta(a, b)
    out["paired_bootstrap_comparisons"] = comparisons


def _ridge_future_score(
    x0_train: np.ndarray,
    x1_train: np.ndarray,
    x0_val: np.ndarray,
    x1_val: np.ndarray,
    lambdas: list[float],
    max_pre_dim: int,
    lambda_selection: str,
) -> dict:
    train_pair = np.concatenate([x0_train, x1_train], axis=0)
    val_pair = np.concatenate([x0_val, x1_val], axis=0)
    train_pair, val_pair = _standardize(train_pair, val_pair)
    if train_pair.shape[1] == 0:
        return {}
    train_pair, val_pair = _pca_project(train_pair, val_pair, max_pre_dim)
    n0 = x0_train.shape[0]
    nv0 = x0_val.shape[0]
    x0_train = train_pair[:n0]
    x1_train = train_pair[n0:]
    x0_val = val_pair[:nv0]
    x1_val = val_pair[nv0:]
    x_aug = np.concatenate([x0_train, np.ones((x0_train.shape[0], 1))], axis=1)
    v_aug = np.concatenate([x0_val, np.ones((x0_val.shape[0], 1))], axis=1)
    denom = float(np.mean((x1_val - x1_train.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 1.0e-12:
        return {}

    def fit_weights(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        x_aug_local = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        reg = lam * np.eye(x_aug_local.shape[1])
        reg[-1, -1] = 0.0
        try:
            return np.linalg.solve(x_aug_local.T @ x_aug_local + reg, x_aug_local.T @ y)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(x_aug_local.T @ x_aug_local + reg) @ x_aug_local.T @ y

    selected_lambda = None
    if lambda_selection == "train_holdout" and x0_train.shape[0] >= 12:
        split = min(x0_train.shape[0] - 3, max(8, int(round(0.8 * x0_train.shape[0]))))
        x_fit, y_fit = x0_train[:split], x1_train[:split]
        x_select, y_select = x0_train[split:], x1_train[split:]
        select_aug = np.concatenate([x_select, np.ones((x_select.shape[0], 1))], axis=1)
        best_select = {"mse": np.inf, "lambda": None}
        for lam in lambdas:
            weights = fit_weights(x_fit, y_fit, lam)
            pred = select_aug @ weights
            mse = float(np.mean((pred - y_select) ** 2))
            if mse < best_select["mse"]:
                best_select = {"mse": mse, "lambda": lam}
        selected_lambda = best_select["lambda"]
    elif lambda_selection != "validation":
        raise ValueError(f"unknown linear lambda selection {lambda_selection}")

    if selected_lambda is None:
        best_val = {"r2": -np.inf, "mse": None, "lambda": None}
        for lam in lambdas:
            weights = fit_weights(x0_train, x1_train, lam)
            pred = v_aug @ weights
            mse = float(np.mean((pred - x1_val) ** 2))
            r2 = 1.0 - mse / denom
            if r2 > best_val["r2"]:
                best_val = {"r2": float(r2), "mse": mse, "lambda": lam}
        return {
            **best_val,
            "feature_dim": float(x0_train.shape[1]),
            "mean_baseline_mse": denom,
            "lambda_selection": lambda_selection,
        }

    weights = fit_weights(x0_train, x1_train, selected_lambda)
    pred = v_aug @ weights
    mse = float(np.mean((pred - x1_val) ** 2))
    r2 = 1.0 - mse / denom
    return {
        "r2": float(r2),
        "mse": mse,
        "lambda": selected_lambda,
        "feature_dim": float(x0_train.shape[1]),
        "mean_baseline_mse": denom,
        "lambda_selection": lambda_selection,
    }


def blocked_cv_linear_dynamics(
    sequences: dict[str, np.ndarray],
    lags: list[int],
    n_folds: int,
    gap: int,
    lambdas: list[float],
    max_pre_dim: int,
    min_pairs: int,
    lambda_selection: str = "train_holdout",
) -> dict:
    results = {}
    for lag in lags:
        per_traj = {}
        rows = []
        for tid, seq in sequences.items():
            n_pairs = len(seq) - lag
            if n_pairs < max(min_pairs, n_folds * 4):
                continue
            fold_edges = np.linspace(0, n_pairs, n_folds + 1, dtype=int)
            traj_rows = []
            for fold in range(n_folds):
                start, end = int(fold_edges[fold]), int(fold_edges[fold + 1])
                val_idx = np.arange(start, end)
                train_mask = np.ones(n_pairs, dtype=bool)
                train_mask[max(0, start - gap) : min(n_pairs, end + gap)] = False
                train_idx = np.nonzero(train_mask)[0]
                if len(val_idx) < 3 or len(train_idx) < 8:
                    continue
                try:
                    score = _ridge_future_score(
                        seq[train_idx],
                        seq[train_idx + lag],
                        seq[val_idx],
                        seq[val_idx + lag],
                        lambdas,
                        max_pre_dim,
                        lambda_selection,
                    )
                except np.linalg.LinAlgError:
                    continue
                if score:
                    score.update({"trajectory_id": tid, "fold": fold, "n_train_pairs": len(train_idx), "n_val_pairs": len(val_idx)})
                    traj_rows.append(score)
                    rows.append(score)
            if traj_rows:
                per_traj[tid] = _aggregate_dynamics_rows(traj_rows)
        results[f"lag_{lag}"] = {
            "n_trajs": len(per_traj),
            "n_folds": len(rows),
            "mean": _aggregate_dynamics_rows(rows) if rows else {},
            "per_trajectory": per_traj,
        }
    return results


def _aggregate_dynamics_rows(rows: list[dict]) -> dict:
    out = {}
    for key in ["r2", "mse", "mean_baseline_mse", "feature_dim"]:
        vals = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def run_linear_dynamics(args: argparse.Namespace) -> dict:
    device = _device(args.device)
    records = _first_n_trajectories(load_manifest(args.manifest), args.max_trajs)
    specs = [_load_checkpoint_spec(item) for item in args.checkpoint]
    baselines = args.baseline or ["geometry_stats", "centered_ca"]
    out = {
        "protocol": {
            "manifest": str(args.manifest),
            "max_trajs": args.max_trajs,
            "window_stride": args.window_stride,
            "window_overlap_ratio": _window_overlap_ratio(args.sequence_frames, args.window_stride),
            "sequence_frames": args.sequence_frames,
            "readout": args.readout,
            "lags": args.lags,
            "n_folds": args.folds,
            "gap": args.gap,
            "max_pre_dim": args.max_pre_dim,
            "min_pairs": args.min_pairs,
            "lambdas": args.lambdas,
            "lambda_selection": args.lambda_selection,
        },
        "models": {},
        "baselines": {},
    }
    t0 = time.time()
    for spec in specs:
        print(f"[linear-dynamics] extracting {spec.name}", flush=True)
        model, cfg = _model_from_spec(spec, device)
        data_cfg = _data_cfg_from_artifact(
            spec.artifact_dir, args.batch_size, num_frames=args.sequence_frames, window_stride=args.window_stride
        )
        seqs = extract_model_frame_sequences(model, records, data_cfg, device, readout=args.readout)
        out["models"][spec.name] = {
            "checkpoint": str(spec.checkpoint),
            "checkpoint_step": cfg["_checkpoint_step"],
            "linear_dynamics_cv": blocked_cv_linear_dynamics(
                seqs,
                args.lags,
                args.folds,
                args.gap,
                args.lambdas,
                args.max_pre_dim,
                args.min_pairs,
                args.lambda_selection,
            ),
        }
    baseline_cfg = _data_cfg_from_artifact(
        specs[0].artifact_dir, args.batch_size, num_frames=args.sequence_frames, window_stride=args.window_stride
    )
    for baseline in baselines:
        print(f"[linear-dynamics] extracting baseline {baseline}", flush=True)
        seqs = extract_baseline_frame_sequences(records, baseline_cfg, baseline)
        out["baselines"][baseline] = {
            "linear_dynamics_cv": blocked_cv_linear_dynamics(
                seqs,
                args.lags,
                args.folds,
                args.gap,
                args.lambdas,
                args.max_pre_dim,
                args.min_pairs,
                args.lambda_selection,
            )
        }
    out["elapsed_s"] = round(time.time() - t0, 2)
    _write_json(args.output, out)
    return out


def load_msm_labels(path: Path) -> dict:
    data = _load_json(path)
    return data["trajectories"]


def assign_clip_label(record: TrajectoryRecord, frames: np.ndarray, msm_trajs: dict) -> int:
    labels = msm_trajs.get(record.trajectory_id, {}).get("state_labels", [])
    states = []
    for frame in frames:
        idx = int(frame)
        if idx < len(labels) and labels[idx] >= 0:
            states.append(int(labels[idx]))
    if not states:
        return -1
    return Counter(states).most_common(1)[0][0]


@torch.no_grad()
def extract_clip_latents_with_labels(
    model: ProteinJEPA,
    records: list[TrajectoryRecord],
    data_cfg: DataConfig,
    msm_labels: dict,
    device: torch.device,
    max_clips: int,
    readout: str = "mean",
) -> list[dict]:
    dataset = TrajectoryClipDataset(records, data_cfg)
    if len(dataset) > max_clips:
        rng = np.random.RandomState(42)
        indices = sorted(rng.choice(len(dataset), max_clips, replace=False).tolist())
        dataset = Subset(dataset, indices)
    inner = dataset.dataset if isinstance(dataset, Subset) else dataset
    index_map = {}
    for i in range(len(dataset)):
        orig = dataset.indices[i] if isinstance(dataset, Subset) else i
        index_map[i] = inner._locate(orig)
    loader = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_clips)
    rows = []
    offset = 0
    for batch in loader:
        batch = _move_tensor_batch(batch, device)
        latents = model.context_encoder(
            coords=batch["coords"],
            residue_mask=batch["residue_mask"],
            aatype=batch["aatype"],
            residue_index=batch["residue_index"],
            chain_id=batch["chain_id"],
            token_mask=None,
        )
        if readout == "mean":
            valid = batch["residue_mask"].unsqueeze(-1).float()
            pooled = (latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
            clip_valid = batch["residue_mask"].any(dim=2).unsqueeze(-1).float()
            clip_latents = (pooled * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)
        elif readout == "state":
            states = model.encode_states(latents, batch["residue_mask"])["frame_states"]
            clip_valid = batch["residue_mask"].any(dim=2).unsqueeze(-1).float()
            clip_latents = (states * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)
        elif readout == "clip_state":
            clip_latents = model.encode_states(latents, batch["residue_mask"])["clip_state"]
        elif readout == "kinetic":
            kinetic = model.encode_kinetic_readout(latents, batch["residue_mask"])
            clip_valid = batch["residue_mask"].any(dim=2).unsqueeze(-1).float()
            clip_latents = (kinetic * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)
        elif readout in {"state_clip", "state_delta", "state_clip_delta", "state_mean", "state_clip_delta_mean"}:
            clip_latents = _model_readout(model, latents, batch["residue_mask"], readout)
        else:
            raise ValueError(f"unknown model readout {readout}")
        for b in range(clip_latents.shape[0]):
            record, frames = index_map[offset + b]
            label = assign_clip_label(record, frames, msm_labels)
            if label >= 0:
                rows.append(
                    {
                        "protein_id": record.protein_id,
                        "trajectory_id": record.trajectory_id,
                        "start_frame": int(frames[0]),
                        "latent": clip_latents[b].detach().cpu().numpy().astype(np.float64),
                        "label": label,
                    }
                )
        offset += clip_latents.shape[0]
    return rows


def _ridge_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, lambdas: list[float]) -> dict:
    classes = np.array(sorted(set(train_y.tolist()) | set(val_y.tolist())), dtype=int)
    train_x, val_x = _standardize(train_x, val_x)
    x_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    v_aug = np.concatenate([val_x, np.ones((val_x.shape[0], 1))], axis=1)
    y_onehot = (train_y[:, None] == classes[None, :]).astype(np.float64)
    best = {"accuracy": -1.0, "lambda": None}
    for lam in lambdas:
        reg = lam * np.eye(x_aug.shape[1])
        reg[-1, -1] = 0.0
        try:
            weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_onehot)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y_onehot
        pred = classes[np.argmax(v_aug @ weights, axis=1)]
        acc = float(np.mean(pred == val_y))
        if acc > best["accuracy"]:
            best = {"accuracy": acc, "lambda": lam}
    majority = Counter(train_y.tolist()).most_common(1)[0][0]
    baseline = float(np.mean(val_y == majority))
    return {"accuracy": best["accuracy"], "baseline": baseline, "lambda": best["lambda"]}


def trajectory_holdout_probe(rows: list[dict], lambdas: list[float]) -> dict:
    by_protein: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_protein[row["protein_id"]].append(row)
    details = {}
    accs = []
    bases = []
    for pid, items in sorted(by_protein.items()):
        trajs = sorted({row["trajectory_id"] for row in items})
        if len(trajs) < 3 or len(items) < 30:
            continue
        seed = int(hashlib.sha256(pid.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        trajs = list(rng.permutation(trajs))
        n_train = max(1, int(round(0.7 * len(trajs))))
        train_trajs = set(trajs[:n_train])
        train = [row for row in items if row["trajectory_id"] in train_trajs]
        val = [row for row in items if row["trajectory_id"] not in train_trajs]
        if len(train) < 10 or len(val) < 5:
            continue
        train_y = np.asarray([row["label"] for row in train], dtype=int)
        val_y = np.asarray([row["label"] for row in val], dtype=int)
        if len(set(train_y.tolist())) < 2 or len(set(val_y.tolist())) < 2:
            continue
        train_x = np.stack([row["latent"] for row in train], axis=0)
        val_x = np.stack([row["latent"] for row in val], axis=0)
        score = _ridge_probe(train_x, train_y, val_x, val_y, lambdas)
        details[pid] = {
            **score,
            "n_train": len(train),
            "n_val": len(val),
            "n_train_trajs": len(train_trajs),
            "n_val_trajs": len(set(trajs) - train_trajs),
        }
        accs.append(score["accuracy"])
        bases.append(score["baseline"])
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "std_accuracy": float(np.std(accs)) if accs else None,
        "mean_baseline": float(np.mean(bases)) if bases else None,
        "n_above_baseline": int(sum(a > b for a, b in zip(accs, bases))),
        "n_total": len(accs),
        "protein_details": details,
    }


def run_msm_probe(args: argparse.Namespace) -> dict:
    device = _device(args.device)
    specs = [_load_checkpoint_spec(item) for item in args.checkpoint]
    records = load_manifest(args.manifest)
    labels = load_msm_labels(args.labels)
    results = {
        "protocol": {
            "manifest": str(args.manifest),
            "labels": str(args.labels),
            "split": "per-protein trajectory holdout",
            "max_clips": args.max_clips,
            "lambdas": args.lambdas,
            "readout": args.readout,
            "num_frames": getattr(args, "num_frames", None),
            "window_stride": getattr(args, "window_stride", None),
        },
        "models": {},
    }
    for spec in specs:
        print(f"[MSM-probe] extracting {spec.name}", flush=True)
        model, cfg = _model_from_spec(spec, device)
        data_cfg = _data_cfg_from_artifact(
            spec.artifact_dir,
            batch_size=args.batch_size,
            num_frames=getattr(args, "num_frames", None) or cfg["data"]["num_frames"],
            window_stride=getattr(args, "window_stride", None) or cfg["data"]["window_stride"],
        )
        rows = extract_clip_latents_with_labels(model, records, data_cfg, labels, device, args.max_clips, readout=args.readout)
        results["models"][spec.name] = {
            "checkpoint": str(spec.checkpoint),
            "checkpoint_step": cfg["_checkpoint_step"],
            "n_labeled_clips": len(rows),
            "trajectory_holdout_probe": trajectory_holdout_probe(rows, args.lambdas),
        }
    _write_json(args.output, results)
    return results


def _pipeline_preset(args: argparse.Namespace) -> dict:
    if args.preset == "strict":
        return {
            "max_trajs": args.max_trajs or 80,
            "max_clips": args.max_clips or 12000,
            "lags": args.lags or [1, 2, 5, 10, 20, 40],
            "dims": args.dims or [2, 5, 10, 20],
            "folds": args.folds or 5,
            "gap": args.gap if args.gap is not None else 40,
            "min_pairs": args.min_pairs or 40,
            "baselines": args.baseline or ["geometry_stats", "centered_ca"],
        }
    if args.preset == "full":
        return {
            "max_trajs": args.max_trajs or 50,
            "max_clips": args.max_clips or 8000,
            "lags": args.lags or [1, 2, 5, 10, 20],
            "dims": args.dims or [2, 5, 10, 20],
            "folds": args.folds or 5,
            "gap": args.gap if args.gap is not None else 20,
            "min_pairs": args.min_pairs or 40,
            "baselines": args.baseline or ["geometry_stats", "centered_ca"],
        }
    return {
        "max_trajs": args.max_trajs or 10,
        "max_clips": args.max_clips or 2000,
        "lags": args.lags or [1, 5, 20],
        "dims": args.dims or [2, 5, 10],
        "folds": args.folds or 3,
        "gap": args.gap if args.gap is not None else 10,
        "min_pairs": args.min_pairs or 20,
        "baselines": args.baseline or ["geometry_stats"],
    }


def _pipeline_summary(vamp: dict | None, msm: dict | None, linear: dict | None) -> dict:
    summary: dict = {"vamp_cv": {}, "msm_probe": {}, "linear_dynamics": {}}
    if vamp:
        lags = vamp["protocol"]["lags"]
        dims = vamp["protocol"]["dims"]
        selected_lag = lags[0]
        selected_dim = dims[min(1, len(dims) - 1)]
        key = f"cv_vamp2_dim_{selected_dim}"
        norm_key = f"cv_vamp2_norm_dim_{selected_dim}"
        summary["vamp_cv"]["selected_metric"] = {"lag": selected_lag, "dim": selected_dim}
        blocks = {f"model:{k}": v for k, v in vamp.get("models", {}).items()}
        blocks.update({f"baseline:{k}": v for k, v in vamp.get("baselines", {}).items()})
        for name, block in blocks.items():
            stats = block.get("vamp2_cv", {}).get(f"lag_{selected_lag}", {}).get("mean", {})
            summary["vamp_cv"][name] = {
                "cv_vamp2": stats.get(key, {}).get("mean"),
                "cv_vamp2_std": stats.get(key, {}).get("std"),
                "cv_vamp2_normalized": stats.get(norm_key, {}).get("mean"),
            }
    if msm:
        for name, block in msm.get("models", {}).items():
            score = block.get("trajectory_holdout_probe", {})
            summary["msm_probe"][f"model:{name}"] = {
                "accuracy": score.get("mean_accuracy"),
                "accuracy_std": score.get("std_accuracy"),
                "baseline": score.get("mean_baseline"),
                "above_baseline": f"{score.get('n_above_baseline')}/{score.get('n_total')}",
            }
    if linear:
        selected_lag = linear["protocol"]["lags"][0]
        blocks = {f"model:{k}": v for k, v in linear.get("models", {}).items()}
        blocks.update({f"baseline:{k}": v for k, v in linear.get("baselines", {}).items()})
        summary["linear_dynamics"]["selected_metric"] = {"lag": selected_lag}
        for name, block in blocks.items():
            stats = block.get("linear_dynamics_cv", {}).get(f"lag_{selected_lag}", {}).get("mean", {})
            summary["linear_dynamics"][name] = {
                "r2": stats.get("r2", {}).get("mean"),
                "r2_std": stats.get("r2", {}).get("std"),
            }
    return summary


def run_pipeline(args: argparse.Namespace) -> dict:
    preset = _pipeline_preset(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_window_stride = args.linear_window_stride or args.window_stride
    if args.require_contextual:
        if args.sequence_frames <= 1:
            raise ValueError("--require-contextual needs --sequence-frames > 1")
        if args.readout == "mean":
            raise ValueError("--require-contextual needs a temporal-state readout")
    common = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "device": args.device,
        "batch_size": args.batch_size,
        "readout": args.readout,
    }
    vamp_results = None
    msm_results = None
    linear_results = None

    if not args.skip_vamp:
        vamp_results = run_vamp_cv(
            SimpleNamespace(
                **common,
                output=output_dir / "vamp_cv.json",
                include_random=args.include_random,
                baseline=preset["baselines"],
                max_trajs=preset["max_trajs"],
                window_stride=args.window_stride,
                sequence_frames=args.sequence_frames,
                lags=preset["lags"],
                dims=preset["dims"],
                folds=preset["folds"],
                gap=preset["gap"],
                eps=args.eps,
                max_pre_dim=args.max_pre_dim,
                min_pairs=preset["min_pairs"],
            )
        )
    if not args.skip_msm:
        msm_results = run_msm_probe(
            SimpleNamespace(
                **common,
                labels=args.labels,
                output=output_dir / "msm_probe.json",
                max_clips=preset["max_clips"],
                lambdas=args.lambdas,
                num_frames=args.sequence_frames,
                window_stride=args.window_stride,
            )
        )
    if not args.skip_linear:
        linear_results = run_linear_dynamics(
            SimpleNamespace(
                **common,
                output=output_dir / "linear_dynamics.json",
                baseline=preset["baselines"],
                max_trajs=preset["max_trajs"],
                window_stride=linear_window_stride,
                sequence_frames=args.sequence_frames,
                lags=preset["lags"],
                folds=preset["folds"],
                gap=preset["gap"],
                max_pre_dim=args.max_pre_dim,
                min_pairs=preset["min_pairs"],
                lambdas=args.lambdas,
                lambda_selection=args.lambda_selection,
            )
        )

    summary = {
        "protocol": {
            "preset": args.preset,
            "checkpoints": args.checkpoint,
            "output_dir": str(output_dir),
            "readout": args.readout,
            "contextual_protocol": args.sequence_frames > 1 and args.readout != "mean",
            "require_contextual": args.require_contextual,
            "sequence_frames": args.sequence_frames,
            "window_stride": args.window_stride,
            "window_overlap_ratio": _window_overlap_ratio(args.sequence_frames, args.window_stride),
            "linear_window_stride": linear_window_stride,
            "linear_window_overlap_ratio": _window_overlap_ratio(args.sequence_frames, linear_window_stride),
            "linear_lambda_selection": args.lambda_selection,
            **preset,
        },
        "summary": _pipeline_summary(vamp_results, msm_results, linear_results),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["summary"], indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-grade Protein JEPA Frame evaluations")
    sub = parser.add_subparsers(dest="command", required=True)

    vamp = sub.add_parser("vamp-cv", help="Run blocked cross-validated VAMP-2")
    vamp.add_argument("--checkpoint", action="append", required=True, help="name=/path/checkpoint.pt; repeatable")
    vamp.add_argument("--include-random", action="store_true", help="Add random initialized same-architecture encoder")
    vamp.add_argument("--baseline", action="append", default=None, help="Baseline feature set")
    vamp.add_argument("--manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    vamp.add_argument("--output", type=Path, required=True)
    vamp.add_argument("--device", default="auto")
    vamp.add_argument("--max-trajs", type=int, default=50)
    vamp.add_argument("--batch-size", type=int, default=16)
    vamp.add_argument("--window-stride", type=int, default=5)
    vamp.add_argument("--sequence-frames", type=int, default=1)
    vamp.add_argument("--readout", choices=MODEL_READOUT_CHOICES, default="mean")
    vamp.add_argument("--lags", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    vamp.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20])
    vamp.add_argument("--folds", type=int, default=5)
    vamp.add_argument("--gap", type=int, default=20)
    vamp.add_argument("--eps", type=float, default=1.0e-5)
    vamp.add_argument("--max-pre-dim", type=int, default=64)
    vamp.add_argument("--min-pairs", type=int, default=40)
    vamp.set_defaults(func=run_vamp_cv)

    probe = sub.add_parser("msm-probe", help="Run trajectory-heldout MSM state probes")
    probe.add_argument("--checkpoint", action="append", required=True, help="name=/path/checkpoint.pt; repeatable")
    probe.add_argument("--manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    probe.add_argument("--labels", type=Path, default=DEFAULT_VAL_LABELS)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--device", default="auto")
    probe.add_argument("--batch-size", type=int, default=8)
    probe.add_argument("--max-clips", type=int, default=8000)
    probe.add_argument("--num-frames", type=int, default=None)
    probe.add_argument("--window-stride", type=int, default=None)
    probe.add_argument("--readout", choices=MODEL_READOUT_CHOICES, default="mean")
    probe.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    probe.set_defaults(func=run_msm_probe)

    dyn = sub.add_parser("linear-dynamics", help="Run blocked CV linear latent future prediction")
    dyn.add_argument("--checkpoint", action="append", required=True, help="name=/path/checkpoint.pt; repeatable")
    dyn.add_argument("--baseline", action="append", default=None, help="Baseline feature set")
    dyn.add_argument("--manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    dyn.add_argument("--output", type=Path, required=True)
    dyn.add_argument("--device", default="auto")
    dyn.add_argument("--max-trajs", type=int, default=50)
    dyn.add_argument("--batch-size", type=int, default=16)
    dyn.add_argument("--window-stride", type=int, default=5)
    dyn.add_argument("--sequence-frames", type=int, default=1)
    dyn.add_argument("--readout", choices=MODEL_READOUT_CHOICES, default="mean")
    dyn.add_argument("--lags", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    dyn.add_argument("--folds", type=int, default=5)
    dyn.add_argument("--gap", type=int, default=20)
    dyn.add_argument("--max-pre-dim", type=int, default=64)
    dyn.add_argument("--min-pairs", type=int, default=40)
    dyn.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    dyn.add_argument(
        "--lambda-selection",
        choices=["train_holdout", "validation"],
        default="train_holdout",
        help="Select ridge lambda on a train-only holdout split, or reproduce legacy validation-selected scores.",
    )
    dyn.set_defaults(func=run_linear_dynamics)

    pipe = sub.add_parser("pipeline", help="Run VAMP-CV, MSM probe, and linear dynamics as one checkpoint pipeline")
    pipe.add_argument("--checkpoint", action="append", required=True, help="name=/path/checkpoint.pt; repeatable")
    pipe.add_argument("--output-dir", type=Path, required=True)
    pipe.add_argument("--preset", choices=["quick", "full", "strict"], default="quick")
    pipe.add_argument("--manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    pipe.add_argument("--labels", type=Path, default=DEFAULT_VAL_LABELS)
    pipe.add_argument("--device", default="auto")
    pipe.add_argument("--batch-size", type=int, default=16)
    pipe.add_argument("--window-stride", type=int, default=5)
    pipe.add_argument("--linear-window-stride", type=int, default=None)
    pipe.add_argument("--sequence-frames", type=int, default=1)
    pipe.add_argument("--readout", choices=MODEL_READOUT_CHOICES, default="mean")
    pipe.add_argument(
        "--require-contextual",
        action="store_true",
        help="Fail if the pipeline is not using a multi-frame temporal-state readout.",
    )
    pipe.add_argument("--max-trajs", type=int, default=None)
    pipe.add_argument("--max-clips", type=int, default=None)
    pipe.add_argument("--lags", type=int, nargs="+", default=None)
    pipe.add_argument("--dims", type=int, nargs="+", default=None)
    pipe.add_argument("--folds", type=int, default=None)
    pipe.add_argument("--gap", type=int, default=None)
    pipe.add_argument("--min-pairs", type=int, default=None)
    pipe.add_argument("--baseline", action="append", default=None)
    pipe.add_argument("--include-random", action="store_true")
    pipe.add_argument("--eps", type=float, default=1.0e-5)
    pipe.add_argument("--max-pre-dim", type=int, default=64)
    pipe.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    pipe.add_argument(
        "--lambda-selection",
        choices=["train_holdout", "validation"],
        default="train_holdout",
        help="Select linear-dynamics ridge lambda on train-only holdout; use validation for legacy reproduction.",
    )
    pipe.add_argument("--skip-vamp", action="store_true")
    pipe.add_argument("--skip-msm", action="store_true")
    pipe.add_argument("--skip-linear", action="store_true")
    pipe.set_defaults(func=run_pipeline)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
