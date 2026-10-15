#!/usr/bin/env python3
"""Autonomous V25 latent-rollout quality evaluation.

This evaluator separates three paths that are easy to conflate in the paper:

1. the paired endpoint head (``delta_prediction``), which is not autonomous;
2. native stochastic endpoint sampling through the LSPWM endpoint API;
3. recursive world-model rollout, initialized from observed context only.

The recursive rollout never reads future frames during prediction. Future
states are used only as evaluation targets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


EPS = 1.0e-12


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _mse_rows(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().mean(dim=-1)


def _energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_term = torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(dim=1)
    pairwise = torch.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], dim=-1
    )
    return target_term - 0.5 * pairwise.mean(dim=(1, 2))


def _bootstrap(values: np.ndarray, *, n_resamples: int = 2000, seed: int = 17) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if values.size == 1:
        value = float(values[0])
        return {"mean": value, "ci_lo": value, "ci_hi": value, "n": 1}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, values.size, size=values.size)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(values.size),
    }


def _jsd_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], EPS, None))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _stationary_jsd(pred: np.ndarray, target: np.ndarray, *, bins: int = 32) -> float:
    if pred.shape[0] < 4 or target.shape[0] < 4:
        return float("nan")
    scores = []
    for dim in range(pred.shape[1]):
        lo = float(min(pred[:, dim].min(), target[:, dim].min()))
        hi = float(max(pred[:, dim].max(), target[:, dim].max()))
        if not np.isfinite(lo + hi) or hi <= lo:
            continue
        p_hist, edges = np.histogram(pred[:, dim], bins=bins, range=(lo, hi), density=False)
        q_hist, _ = np.histogram(target[:, dim], bins=edges, density=False)
        scores.append(_jsd_1d(p_hist, q_hist))
    return float(np.mean(scores)) if scores else float("nan")


def _acf_rmse_proxy(pred_delta: np.ndarray, target_delta: np.ndarray) -> float:
    """A paired-window lag-stat proxy for the endpoint displacement structure.

    True trajectory ACF needs consecutive generated paths. This proxy compares
    the dimension-wise displacement correlation matrices and is reported with
    a conservative label in the summary.
    """
    if pred_delta.shape[0] < pred_delta.shape[1] + 2:
        return float("nan")
    pred_c = pred_delta - pred_delta.mean(axis=0, keepdims=True)
    target_c = target_delta - target_delta.mean(axis=0, keepdims=True)
    pred_cov = np.corrcoef(pred_c, rowvar=False)
    target_cov = np.corrcoef(target_c, rowvar=False)
    return float(np.sqrt(np.nanmean((pred_cov - target_cov) ** 2)))


def _aux_from_projected_context(model: ProteinJEPA, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, str]:
    """Initialize the global auxiliary state from projected context CV frames."""
    lspwm = model.lspwm
    assert lspwm is not None
    sequence = out["lspwm_context_sequence"]
    mask = out["lspwm_context_mask"]
    ctx = int(getattr(lspwm, "auxiliary_initializer_context", 3))
    batch_idx = torch.arange(sequence.shape[0], device=sequence.device)
    counts = mask.long().sum(dim=1).clamp(min=1)
    frames = []
    for offset in range(ctx):
        idx = (counts - 1 - offset).clamp(min=0)
        frames.append(sequence[batch_idx, idx, :])
    aux_input = torch.cat(frames, dim=-1)
    expected = lspwm.auxiliary_initializer[0].normalized_shape[0]
    if aux_input.shape[-1] != expected:
        raise ValueError(
            f"projected context has {aux_input.shape[-1]} dims, initializer expects {expected}"
        )
    return lspwm.auxiliary_initializer(aux_input), "projected_context_auxiliary_initializer"


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    model.eval()
    if model.lspwm is None:
        raise ValueError("checkpoint does not contain an LSPWM model")

    dataset = build_dataset(config.data)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
        generator=generator,
    )

    rows: dict[str, list] = {
        "target": [],
        "current": [],
        "endpoint_head": [],
        "native_sample_mean": [],
        "recursive_zero_det": [],
        "recursive_history_det": [],
        "recursive_zero_sample_mean": [],
        "recursive_history_sample_mean": [],
        "persistence_mse": [],
        "horizon": [],
        "protein_id": [],
        "trajectory_id": [],
        "native_energy": [],
        "zero_energy": [],
        "history_energy": [],
        "persistence_energy": [],
        "native_msm_acc": [],
        "zero_msm_acc": [],
        "history_msm_acc": [],
        "native_msm_overlap": [],
        "zero_msm_overlap": [],
        "history_msm_overlap": [],
    }
    aux_status_counts: dict[str, int] = {}
    divergence_counts = {"zero": 0, "history": 0, "native": 0}

    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        out = model.forward_paired(values)
        current = out["lspwm_context_state"].detach()
        target = out["lspwm_target_state"].detach()
        condition = out["lspwm_condition"].detach()
        temperature = out["lspwm_temperature_k"].detach()
        horizon_time_ps = values["horizon_time_ps"].detach()
        history_features = out["lspwm_history_features"].detach()
        horizon = values["horizon_frames"].detach().cpu().numpy().reshape(-1)
        batch_size = current.shape[0]

        persistence_mse = _mse_rows(current, target)
        persistence_energy = torch.linalg.vector_norm(current - target, dim=-1)

        endpoint_head_status = "unavailable"
        if out.get("delta_prediction") is not None and out.get("delta_target") is not None:
            delta_prediction = out["delta_prediction"].detach()
            if delta_prediction.shape == current.shape:
                endpoint_head = current + delta_prediction
                endpoint_head_status = "latent_cv_delta"
            else:
                endpoint_head = torch.full_like(current, float("nan"))
                endpoint_head_status = (
                    f"excluded_shape_mismatch:{tuple(delta_prediction.shape)}_vs_{tuple(current.shape)}"
                )
        else:
            endpoint_head = torch.full_like(current, float("nan"))

        native_samples = model.lspwm.sample_endpoint(
            current,
            condition,
            history_features,
            horizon_time_ps,
            temperature,
            ensemble_size=args.ensemble_size,
        )
        native_mean = native_samples.mean(dim=1)
        native_energy = _energy_score(native_samples, target)

        rollout_steps = max(1, args.rollout_steps)
        dt = horizon_time_ps / float(rollout_steps)
        zero_aux = current.new_zeros(batch_size, model.lspwm.world_model.config.auxiliary_dim)

        zero_det_traj = model.lspwm.world_model.rollout(
            current,
            condition,
            dt,
            rollout_steps,
            temperature,
            stochastic=False,
            auxiliary=zero_aux,
            history_features=None,
        )
        zero_det = zero_det_traj["q"][:, -1].detach()

        zero_exp_current = current.repeat_interleave(args.ensemble_size, dim=0)
        zero_exp_condition = condition.repeat_interleave(args.ensemble_size, dim=0)
        zero_exp_temp = temperature.repeat_interleave(args.ensemble_size)
        zero_exp_dt = dt.repeat_interleave(args.ensemble_size)
        zero_exp_aux = zero_aux.repeat_interleave(args.ensemble_size, dim=0)
        zero_sample_traj = model.lspwm.world_model.rollout(
            zero_exp_current,
            zero_exp_condition,
            zero_exp_dt,
            rollout_steps,
            zero_exp_temp,
            stochastic=True,
            auxiliary=zero_exp_aux,
            history_features=None,
        )
        zero_samples = zero_sample_traj["q"][:, -1].reshape(batch_size, args.ensemble_size, -1)
        zero_sample_mean = zero_samples.mean(dim=1)
        zero_energy = _energy_score(zero_samples, target)

        try:
            history_aux, aux_status = _aux_from_projected_context(model, out)
        except Exception as exc:
            history_aux = zero_aux
            aux_status = f"zero_fallback:{type(exc).__name__}"
        aux_status_counts[aux_status] = aux_status_counts.get(aux_status, 0) + batch_size
        aux_status_counts[endpoint_head_status] = aux_status_counts.get(endpoint_head_status, 0) + 0

        hist_det_traj = model.lspwm.world_model.rollout(
            current,
            condition,
            dt,
            rollout_steps,
            temperature,
            stochastic=False,
            auxiliary=history_aux,
            history_features=None,
        )
        hist_det = hist_det_traj["q"][:, -1].detach()
        hist_exp_aux = history_aux.repeat_interleave(args.ensemble_size, dim=0)
        hist_sample_traj = model.lspwm.world_model.rollout(
            zero_exp_current,
            zero_exp_condition,
            zero_exp_dt,
            rollout_steps,
            zero_exp_temp,
            stochastic=True,
            auxiliary=hist_exp_aux,
            history_features=None,
        )
        hist_samples = hist_sample_traj["q"][:, -1].reshape(batch_size, args.ensemble_size, -1)
        hist_sample_mean = hist_samples.mean(dim=1)
        hist_energy = _energy_score(hist_samples, target)

        for name, tensor in [
            ("native", native_mean),
            ("zero", zero_sample_mean),
            ("history", hist_sample_mean),
        ]:
            bad = ~torch.isfinite(tensor).all(dim=-1)
            divergence_counts[name] += int(bad.sum().detach().cpu())

        target_roles = model.lspwm.role_readout(target)
        target_prob = target_roles["msm_logits"].softmax(dim=-1)
        target_state = target_roles["msm_logits"].argmax(dim=-1)

        role_metrics = {}
        for name, pred in [
            ("native", native_mean),
            ("zero", zero_sample_mean),
            ("history", hist_sample_mean),
        ]:
            roles = model.lspwm.role_readout(pred)
            prob = roles["msm_logits"].softmax(dim=-1)
            state = roles["msm_logits"].argmax(dim=-1)
            role_metrics[f"{name}_msm_acc"] = (state == target_state).float()
            role_metrics[f"{name}_msm_overlap"] = (prob * target_prob).sum(dim=-1)

        rows["target"].append(target.detach().cpu().numpy())
        rows["current"].append(current.detach().cpu().numpy())
        rows["endpoint_head"].append(endpoint_head.detach().cpu().numpy())
        rows["native_sample_mean"].append(native_mean.detach().cpu().numpy())
        rows["recursive_zero_det"].append(zero_det.detach().cpu().numpy())
        rows["recursive_history_det"].append(hist_det.detach().cpu().numpy())
        rows["recursive_zero_sample_mean"].append(zero_sample_mean.detach().cpu().numpy())
        rows["recursive_history_sample_mean"].append(hist_sample_mean.detach().cpu().numpy())
        rows["persistence_mse"].append(persistence_mse.detach().cpu().numpy())
        rows["horizon"].append(horizon)
        rows["protein_id"].extend(str(x) for x in batch.get("protein_id", []))
        rows["trajectory_id"].extend(str(x) for x in batch.get("trajectory_id", []))
        rows["native_energy"].append(native_energy.detach().cpu().numpy())
        rows["zero_energy"].append(zero_energy.detach().cpu().numpy())
        rows["history_energy"].append(hist_energy.detach().cpu().numpy())
        rows["persistence_energy"].append(persistence_energy.detach().cpu().numpy())
        for key, value in role_metrics.items():
            rows[key].append(value.detach().cpu().numpy())

    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in rows.items()
        if key not in {"protein_id", "trajectory_id"} and value
    }
    proteins = np.asarray(rows["protein_id"])
    trajectories = np.asarray(rows["trajectory_id"])
    target = arrays["target"]
    current = arrays["current"]
    persistence_mse = arrays["persistence_mse"]
    persistence_energy = arrays["persistence_energy"]

    method_map = {
        "endpoint_head_non_autonomous": "endpoint_head",
        "native_endpoint_sampler": "native_sample_mean",
        "recursive_zero_aux_det": "recursive_zero_det",
        "recursive_history_aux_det": "recursive_history_det",
        "recursive_zero_aux_ensemble": "recursive_zero_sample_mean",
        "recursive_history_aux_ensemble": "recursive_history_sample_mean",
    }

    aggregate = {}
    per_horizon = {}
    for method, key in method_map.items():
        pred = arrays[key]
        mse_rows = np.mean((pred - target) ** 2, axis=1)
        skill_rows = 1.0 - mse_rows / np.maximum(persistence_mse, EPS)
        delta_proxy = _acf_rmse_proxy(pred - current, target - current)
        aggregate[method] = {
            "mse": _bootstrap(mse_rows, n_resamples=args.bootstrap_resamples),
            "skill_vs_persistence": _bootstrap(skill_rows, n_resamples=args.bootstrap_resamples),
            "stationary_jsd": _stationary_jsd(pred, target, bins=args.bins),
            "displacement_correlation_rmse_proxy": delta_proxy,
        }
        if "ensemble" in method or method == "native_endpoint_sampler":
            energy_key = (
                "native_energy" if method == "native_endpoint_sampler"
                else "zero_energy" if "zero" in method
                else "history_energy"
            )
            energy_rows = arrays[energy_key]
            energy_skill = 1.0 - energy_rows / np.maximum(persistence_energy, EPS)
            aggregate[method]["energy_score"] = _bootstrap(
                energy_rows, n_resamples=args.bootstrap_resamples
            )
            aggregate[method]["energy_skill_vs_persistence"] = _bootstrap(
                energy_skill, n_resamples=args.bootstrap_resamples
            )
        if method in {"native_endpoint_sampler", "recursive_zero_aux_ensemble", "recursive_history_aux_ensemble"}:
            prefix = (
                "native" if method == "native_endpoint_sampler"
                else "zero" if "zero" in method
                else "history"
            )
            aggregate[method]["msm_accuracy"] = _bootstrap(
                arrays[f"{prefix}_msm_acc"], n_resamples=args.bootstrap_resamples
            )
            aggregate[method]["msm_soft_overlap"] = _bootstrap(
                arrays[f"{prefix}_msm_overlap"], n_resamples=args.bootstrap_resamples
            )

        per_horizon[method] = {}
        for h in sorted(set(int(x) for x in arrays["horizon"].tolist())):
            mask = arrays["horizon"] == h
            if not np.any(mask):
                continue
            h_mse = mse_rows[mask]
            h_skill = skill_rows[mask]
            per_horizon[method][str(h)] = {
                "rows": int(mask.sum()),
                "mse": _bootstrap(h_mse, n_resamples=args.bootstrap_resamples),
                "skill_vs_persistence": _bootstrap(
                    h_skill, n_resamples=args.bootstrap_resamples
                ),
                "stationary_jsd": _stationary_jsd(pred[mask], target[mask], bins=args.bins),
            }

    result = {
        "experiment_id": "V25_autonomous_rollout_quality",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "V25 paired-window evaluation sample",
            "max_batches": args.max_batches,
            "batch_size": config.data.batch_size,
            "n_pairs": int(target.shape[0]),
            "n_proteins": int(len(set(proteins.tolist()))),
            "n_trajectories": int(len(set(trajectories.tolist()))),
            "horizon_counts": {
                str(int(h)): int(np.sum(arrays["horizon"] == h))
                for h in sorted(set(arrays["horizon"].tolist()))
            },
            "frame_interval_ns": 1.0,
            "rollout_steps": args.rollout_steps,
            "ensemble_size": args.ensemble_size,
            "seed": args.seed,
        },
        "auxiliary_initialization_status": aux_status_counts,
        "divergence_counts": divergence_counts,
        "aggregate": aggregate,
        "per_horizon": per_horizon,
        "claim_boundary": (
            "endpoint_head_non_autonomous is an endpoint predictor, not an autonomous rollout. "
            "recursive_* rows are autonomous world-model rollouts initialized from observed context only; "
            "future frames are used only for scoring. The displacement-correlation metric is a paired-window "
            "proxy, not a true generated-trajectory ACF."
        ),
    }
    return result


def _fmt_ci(row: dict, digits: int = 4) -> str:
    return f"{row['mean']:.{digits}f} [{row['ci_lo']:.{digits}f}, {row['ci_hi']:.{digits}f}]"


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# V25 Autonomous Latent Rollout Quality",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"n proteins: {result['protocol']['n_proteins']}",
        f"n trajectories: {result['protocol']['n_trajectories']}",
        f"horizon counts: {result['protocol']['horizon_counts']}",
        f"rollout steps: {result['protocol']['rollout_steps']}",
        f"ensemble size: {result['protocol']['ensemble_size']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | MSE | Skill vs persistence | Stationary JSD | Disp-corr RMSE proxy | Energy skill | MSM overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in result["aggregate"].items():
        energy = row.get("energy_skill_vs_persistence")
        overlap = row.get("msm_soft_overlap")
        lines.append(
            f"| {method} | {_fmt_ci(row['mse'], 6)} | {_fmt_ci(row['skill_vs_persistence'], 4)} | "
            f"{row['stationary_jsd']:.5f} | {row['displacement_correlation_rmse_proxy']:.5f} | "
            f"{_fmt_ci(energy, 4) if energy else '-'} | {_fmt_ci(overlap, 4) if overlap else '-'} |"
        )
    lines.extend([
        "",
        "## Auxiliary Initialization",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ])
    for status, count in result["auxiliary_initialization_status"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "## Divergence Counts",
        "",
        "| Path | Count |",
        "|---|---:|",
    ])
    for name, count in result["divergence_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", result["claim_boundary"], ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lspwm_v44_stage_b_v25_comprehensive.yaml")
    parser.add_argument("--checkpoint", default="artifacts/lspwm_v44_stage_b_v25_comprehensive/checkpoint_step5400.pt")
    parser.add_argument("--output", default="artifacts/lspwm_v44_stage_b_v25_comprehensive/eval/autonomous_rollout_quality.json")
    parser.add_argument("--summary", default="artifacts/lspwm_v44_stage_b_v25_comprehensive/eval/autonomous_rollout_quality_summary.md")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9225)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        write_summary(result, Path(args.summary))
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
