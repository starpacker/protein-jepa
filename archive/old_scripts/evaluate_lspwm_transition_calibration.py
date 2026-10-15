#!/usr/bin/env python3
"""Diagnose one-step LSP-WM transition calibration.

The script keeps checkpoint selection and baseline fitting protein-disjoint:

* the training manifest is used only to fit ridge/VAR baselines and select
  their regularization on a train-only protein split;
* the validation manifest is used only for the final comparison;
* no encoder or world-model parameter is updated.

The report separates deterministic drift calibration from stochastic
covariance calibration.  This is intentionally a diagnostic, not a training
entry point.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
from protein_jepa_frame.models.lspwm import overdamped_transition_nll


DEFAULT_HORIZONS = (24, 48, 96)
DEFAULT_RIDGE_LAMBDAS = (
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    1.0e1,
    1.0e2,
)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _load_model(args: argparse.Namespace, device: torch.device) -> ProteinJEPA:
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"]
    model_config = model_config_from_dict(checkpoint["config"]["model"], state)
    model = ProteinJEPA(model_config).to(device)
    model.load_state_dict(upgrade_lspwm_state_dict(model, state), strict=True)
    model.eval()
    if model.lspwm is None:
        raise ValueError("checkpoint is not an LSP-WM checkpoint")
    return model


def _build_loader(
    config_path: str,
    manifest: str,
    *,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> DataLoader:
    source_config = load_config(config_path)
    data_config = replace(
        source_config.data,
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=0,
        paired_all_horizons=True,
        sampling_mode="hierarchical",
        training_mode="paired_window",
    )
    dataset = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=max_batches * batch_size,
        seed=seed,
        dataset_weights=data_config.dataset_weights,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )


@torch.no_grad()
def _collect_split(
    model: ProteinJEPA,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, np.ndarray]:
    """Collect paired states without running the LSP-WM training losses."""
    # forward_paired also owns the exact encoder/projection path used by the
    # endpoint evaluator.  Phase zero suppresses its optional diagnostic loss
    # and stochastic endpoint sampling while preserving the emitted states.
    previous_step = getattr(model, "_curriculum_step", None)
    model._curriculum_step = 0
    values: dict[str, list[Any]] = defaultdict(list)
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            outputs = model.forward_paired(_move(batch, device))
            for key in (
                "lspwm_context_state",
                "lspwm_target_state",
                "lspwm_condition",
                "lspwm_temperature_k",
                "lspwm_history_features",
                "horizon_frames",
                "horizon_time_ps",
            ):
                values[key].append(outputs[key].detach().cpu())
            values["protein_id"].extend(batch["protein_id"])
            values["trajectory_id"].extend(batch["trajectory_id"])
    finally:
        if previous_step is None:
            try:
                delattr(model, "_curriculum_step")
            except AttributeError:
                pass
        else:
            model._curriculum_step = previous_step

    if not values["horizon_frames"]:
        raise ValueError("split produced no batches")
    arrays = {
        key: torch.cat(items, dim=0).numpy()
        for key, items in values.items()
        if key not in {"protein_id", "trajectory_id"}
    }
    arrays["protein_id"] = np.asarray(values["protein_id"], dtype=str)
    arrays["trajectory_id"] = np.asarray(values["trajectory_id"], dtype=str)
    return arrays


def _fit_linear(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float,
    x_mean: np.ndarray | None = None,
    x_scale: np.ndarray | None = None,
    y_mean: np.ndarray | None = None,
    y_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("linear fit expects matching [rows, dimensions] arrays")
    if x_mean is None:
        x_mean = x.mean(axis=0)
    if x_scale is None:
        x_scale = x.std(axis=0)
    if y_mean is None:
        y_mean = y.mean(axis=0)
    if y_scale is None:
        y_scale = y.std(axis=0)
    x_scale = np.maximum(x_scale, 1.0e-6)
    y_scale = np.maximum(y_scale, 1.0e-6)
    xz = (x - x_mean) / x_scale
    yz = (y - y_mean) / y_scale
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1))], axis=1)
    regularizer = ridge_lambda * np.eye(x_aug.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    gram = x_aug.T @ x_aug + regularizer
    rhs = x_aug.T @ yz
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ rhs
    return {
        "weights": weights,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def _predict_linear(fit: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xz = (x - fit["x_mean"]) / fit["x_scale"]
    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1))], axis=1)
    return x_aug @ fit["weights"] * fit["y_scale"] + fit["y_mean"]


def _train_holdout_masks(
    protein_ids: np.ndarray,
    *,
    seed: int,
    holdout_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    proteins = np.unique(protein_ids)
    if proteins.size < 2:
        mask = np.ones(protein_ids.shape[0], dtype=bool)
        return mask, ~mask, proteins.tolist(), []
    rng = np.random.default_rng(seed)
    shuffled = proteins[rng.permutation(proteins.size)]
    holdout_count = min(
        proteins.size - 1,
        max(1, int(round(proteins.size * holdout_fraction))),
    )
    holdout = set(shuffled[-holdout_count:].tolist())
    select_mask = np.asarray([protein not in holdout for protein in protein_ids])
    holdout_mask = ~select_mask
    return (
        select_mask,
        holdout_mask,
        sorted(set(protein_ids[select_mask].tolist())),
        sorted(set(protein_ids[holdout_mask].tolist())),
    )


def _select_ridge_lambda(
    x: np.ndarray,
    y: np.ndarray,
    protein_ids: np.ndarray,
    lambdas: tuple[float, ...],
    *,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    fit_mask, select_mask, fit_proteins, select_proteins = _train_holdout_masks(
        protein_ids,
        seed=seed,
    )
    if not select_mask.any():
        selected = lambdas[len(lambdas) // 2]
        return selected, {
            "selection": "fixed_middle_no_holdout",
            "fit_rows": int(fit_mask.sum()),
            "select_rows": 0,
            "fit_proteins": fit_proteins,
            "select_proteins": select_proteins,
        }
    fit = {}
    scores = []
    for lam in lambdas:
        candidate = _fit_linear(x[fit_mask], y[fit_mask], ridge_lambda=lam)
        prediction = _predict_linear(candidate, x[select_mask])
        scores.append(
            {
                "lambda": float(lam),
                "mse": float(np.mean((prediction - y[select_mask]) ** 2)),
            }
        )
    best = min(scores, key=lambda item: (item["mse"], item["lambda"]))
    fit = {
        "selection": "train_protein_holdout",
        "fit_rows": int(fit_mask.sum()),
        "select_rows": int(select_mask.sum()),
        "fit_proteins": fit_proteins,
        "select_proteins": select_proteins,
        "scores": scores,
    }
    return float(best["lambda"]), fit


def _covariance_trace(residual: np.ndarray) -> tuple[float, float]:
    centered = residual - residual.mean(axis=0, keepdims=True)
    covariance_trace = float(np.mean(np.sum(centered * centered, axis=1)))
    second_moment_trace = float(np.mean(np.sum(residual * residual, axis=1)))
    return covariance_trace, second_moment_trace


def _gaussian_nll(
    residual: np.ndarray,
    covariance: np.ndarray,
    *,
    jitter: float = 1.0e-6,
) -> float:
    covariance = np.asarray(covariance, dtype=np.float64)
    dimension = residual.shape[1]
    covariance = covariance + np.eye(dimension) * jitter
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        return float("nan")
    try:
        solved = np.linalg.solve(covariance, residual.T).T
    except np.linalg.LinAlgError:
        return float("nan")
    quadratic = np.sum(residual * solved, axis=1)
    return float(0.5 * np.mean(quadratic + logdet + dimension * np.log(2.0 * np.pi)))


def _diagonal_covariance(residual: np.ndarray, floor: float = 1.0e-6) -> np.ndarray:
    variance = np.var(residual, axis=0)
    return np.diag(np.maximum(variance, floor))


def _empirical_covariance(residual: np.ndarray) -> np.ndarray:
    centered = residual - residual.mean(axis=0, keepdims=True)
    denominator = max(1, residual.shape[0] - 1)
    return centered.T @ centered / denominator


def _point_metrics(
    prediction: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
    persistence_mse: float,
) -> dict[str, float]:
    target_delta = target - current
    prediction_delta = prediction - current
    prediction_mse = float(np.mean((prediction - target) ** 2))
    target_norm = np.linalg.norm(target_delta, axis=1)
    prediction_norm = np.linalg.norm(prediction_delta, axis=1)
    alignment_denominator = (
        np.linalg.norm(target_delta, axis=1) * np.linalg.norm(prediction_delta, axis=1)
    )
    alignment = np.divide(
        np.sum(target_delta * prediction_delta, axis=1),
        alignment_denominator,
        out=np.zeros_like(target_norm),
        where=alignment_denominator > 1.0e-8,
    )
    return {
        "mse": prediction_mse,
        "skill_vs_persistence": 1.0 - prediction_mse / max(persistence_mse, 1.0e-12),
        "target_displacement_norm": float(target_norm.mean()),
        "prediction_displacement_norm": float(prediction_norm.mean()),
        "prediction_to_target_displacement_ratio": float(
            np.mean(prediction_norm) / max(float(np.mean(target_norm)), 1.0e-12)
        ),
        "mean_displacement_alignment": float(alignment.mean()),
    }


def _evaluate_lspwm(
    model: ProteinJEPA,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    world_model = model.lspwm.world_model
    current = torch.from_numpy(split["lspwm_context_state"]).to(device)
    target = torch.from_numpy(split["lspwm_target_state"]).to(device)
    condition = torch.from_numpy(split["lspwm_condition"]).to(device)
    temperature = torch.from_numpy(split["lspwm_temperature_k"]).to(device)
    history = torch.from_numpy(split["lspwm_history_features"]).to(device)
    horizon_frames = split["horizon_frames"]
    horizon_time_ps = torch.from_numpy(split["horizon_time_ps"]).to(device)
    auxiliary = current.new_zeros(current.shape[0], world_model.config.auxiliary_dim)

    with torch.no_grad():
        deterministic = world_model.step(
            current,
            auxiliary,
            condition,
            horizon_time_ps,
            temperature,
            stochastic=False,
        )["q"]
        correction = world_model.history_correction(
            current,
            history,
            horizon_time_ps,
            temperature,
        )
        deterministic = deterministic + correction
        coefficients = world_model.coefficients(current, temperature)
        covariance = 2.0 * coefficients["diffusion"] * horizon_time_ps[:, None, None]

    result: dict[int, dict[str, Any]] = {}
    tensors: dict[int, dict[str, np.ndarray]] = {}
    persistence_mse = {}
    for horizon in sorted(np.unique(horizon_frames).tolist()):
        mask = horizon_frames == horizon
        index = np.flatnonzero(mask)
        q = current[index]
        q_target = target[index]
        q_prediction = deterministic[index]
        q_covariance = covariance[index]
        q_condition = condition[index]
        q_temperature = temperature[index]
        q_history = history[index]
        q_dt = horizon_time_ps[index]
        residual = q_target - q_prediction
        persistence_residual = q_target - q
        persistence_mse[horizon] = float(persistence_residual.square().mean().item())
        nll = float(
            overdamped_transition_nll(
                world_model,
                q,
                q_target,
                q_condition,
                q_dt,
                q_temperature,
                history_features=q_history,
            ).item()
        )
        predicted_cov_trace = float(torch.diagonal(q_covariance, dim1=1, dim2=2).sum(dim=1).mean().item())
        tensors[horizon] = {
            "current": q.cpu().numpy(),
            "target": q_target.cpu().numpy(),
            "prediction": q_prediction.cpu().numpy(),
            "residual": residual.cpu().numpy(),
            "covariance": q_covariance.cpu().numpy(),
        }
        empirical_cov_trace, innovation_trace = _covariance_trace(
            tensors[horizon]["residual"]
        )
        result[horizon] = {
            "rows": int(mask.sum()),
            "nll": nll,
            "model_covariance_trace": predicted_cov_trace,
            "empirical_residual_covariance_trace": empirical_cov_trace,
            "empirical_innovation_second_moment_trace": innovation_trace,
            "covariance_trace_ratio_vs_residual_covariance": predicted_cov_trace
            / max(empirical_cov_trace, 1.0e-12),
            "covariance_trace_ratio_vs_innovation_second_moment": predicted_cov_trace
            / max(innovation_trace, 1.0e-12),
            "diffusion_diagonal_mean": float(
                torch.diagonal(coefficients["diffusion"][index], dim1=1, dim2=2)
                .mean()
                .item()
            ),
            "dt_ps_mean": float(q_dt.mean().item()),
        }
    return result, tensors


def _parse_lambdas(raw: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in raw.split(",") if value.strip())
    if not values or any(value < 0.0 for value in values):
        raise ValueError("--ridge-lambdas must contain non-negative numbers")
    return values


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model = _load_model(args, device)

    train_loader = _build_loader(
        args.config,
        args.train_manifest,
        batch_size=args.batch_size,
        max_batches=args.max_train_batches,
        seed=args.seed,
    )
    validation_loader = _build_loader(
        args.config,
        args.validation_manifest,
        batch_size=args.batch_size,
        max_batches=args.max_validation_batches,
        seed=args.seed,
    )
    train = _collect_split(model, train_loader, device, args.max_train_batches)
    validation = _collect_split(
        model,
        validation_loader,
        device,
        args.max_validation_batches,
    )
    lspwm_metrics, lspwm_tensors = _evaluate_lspwm(model, validation, device)
    lambdas = _parse_lambdas(args.ridge_lambdas)

    by_horizon: dict[str, Any] = {}
    ridge_selection: dict[str, Any] = {}
    for horizon in sorted(
        set(np.unique(train["horizon_frames"]).tolist())
        & set(np.unique(validation["horizon_frames"]).tolist())
    ):
        train_mask = train["horizon_frames"] == horizon
        validation_mask = validation["horizon_frames"] == horizon
        train_current = train["lspwm_context_state"][train_mask]
        train_target = train["lspwm_target_state"][train_mask]
        train_delta = train_target - train_current
        validation_current = validation["lspwm_context_state"][validation_mask]
        validation_target = validation["lspwm_target_state"][validation_mask]
        validation_proteins = validation["protein_id"][validation_mask]
        train_proteins = train["protein_id"][train_mask]

        persistence = validation_current
        persistence_mse = float(np.mean((persistence - validation_target) ** 2))
        target_lambda, target_selection = _select_ridge_lambda(
            train_current,
            train_target,
            train_proteins,
            lambdas,
            seed=args.seed + int(horizon),
        )
        delta_lambda, delta_selection = _select_ridge_lambda(
            train_current,
            train_delta,
            train_proteins,
            lambdas,
            seed=args.seed + 1000 + int(horizon),
        )
        target_fit = _fit_linear(
            train_current,
            train_target,
            ridge_lambda=target_lambda,
        )
        delta_fit = _fit_linear(
            train_current,
            train_delta,
            ridge_lambda=delta_lambda,
        )
        ridge_target_prediction = _predict_linear(target_fit, validation_current)
        ridge_delta_prediction = validation_current + _predict_linear(
            delta_fit,
            validation_current,
        )
        ridge_selection[str(horizon)] = {
            "target": {"lambda": target_lambda, **target_selection},
            "delta": {"lambda": delta_lambda, **delta_selection},
        }

        baseline_metrics = {
            "persistence": _point_metrics(
                persistence,
                validation_current,
                validation_target,
                persistence_mse,
            ),
            "ridge_target": _point_metrics(
                ridge_target_prediction,
                validation_current,
                validation_target,
                persistence_mse,
            ),
            "ridge_delta": _point_metrics(
                ridge_delta_prediction,
                validation_current,
                validation_target,
                persistence_mse,
            ),
        }
        lspwm_block = dict(lspwm_metrics[horizon])
        lspwm_block.update(
            _point_metrics(
                lspwm_tensors[horizon]["prediction"],
                lspwm_tensors[horizon]["current"],
                lspwm_tensors[horizon]["target"],
                persistence_mse,
            )
        )
        for name, prediction in (
            ("persistence", persistence),
            ("ridge_target", ridge_target_prediction),
            ("ridge_delta", ridge_delta_prediction),
        ):
            if name == "persistence":
                fit_train_prediction = train_current
                train_residual = train_target - fit_train_prediction
            elif name == "ridge_target":
                fit_train_prediction = _predict_linear(target_fit, train_current)
                train_residual = train_target - fit_train_prediction
            else:
                fit_train_prediction = train_current + _predict_linear(
                    delta_fit,
                    train_current,
                )
                train_residual = train_target - fit_train_prediction
            covariance = _empirical_covariance(train_residual)
            diagonal_covariance = _diagonal_covariance(train_residual)
            baseline_metrics[name]["train_residual_covariance_trace"] = float(
                np.trace(covariance)
            )
            baseline_metrics[name]["validation_gaussian_nll_diagonal"] = _gaussian_nll(
                validation_target - prediction,
                diagonal_covariance,
            )
        lspwm_block["validation_gaussian_nll_analytic"] = lspwm_block["nll"]
        by_horizon[str(horizon)] = {
            "rows": int(validation_mask.sum()),
            "train_rows_for_baseline": int(train_mask.sum()),
            "train_proteins": int(np.unique(train_proteins).size),
            "validation_proteins": int(np.unique(validation_proteins).size),
            "persistence_mse": persistence_mse,
            "baselines": baseline_metrics,
            "lspwm": lspwm_block,
        }

    lspwm_nll = [
        block["lspwm"]["nll"]
        for block in by_horizon.values()
        if np.isfinite(block["lspwm"]["nll"])
    ]
    return _jsonable(
        {
            "framework": "LSP-WM v4.4",
            "protocol": "protein-disjoint one-step transition calibration",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "config": str(Path(args.config).resolve()),
            "train_manifest": str(Path(args.train_manifest).resolve()),
            "validation_manifest": str(Path(args.validation_manifest).resolve()),
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_train_batches": args.max_train_batches,
            "max_validation_batches": args.max_validation_batches,
            "train_rows": int(train["horizon_frames"].shape[0]),
            "validation_rows": int(validation["horizon_frames"].shape[0]),
            "state_dim": int(train["lspwm_context_state"].shape[1]),
            "horizons": sorted(by_horizon, key=int),
            "ridge_lambdas": list(lambdas),
            "ridge_selection": ridge_selection,
            "mean_lspwm_nll": float(np.mean(lspwm_nll)) if lspwm_nll else float("nan"),
            "by_horizon": by_horizon,
            "claim_boundary": (
                "diagnostic comparison of frozen one-step state transitions; "
                "no training, checkpoint promotion, or test-set selection"
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--train-manifest",
        default="manifests/tmjepa_v3.train.jsonl",
    )
    parser.add_argument(
        "--validation-manifest",
        default="manifests/tmjepa_v3.validation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/phase2_transition_calibration",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=80)
    parser.add_argument("--max-validation-batches", type=int, default=80)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ridge-lambdas",
        default=",".join(str(value) for value in DEFAULT_RIDGE_LAMBDAS),
    )
    args = parser.parse_args()
    result = evaluate(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transition_calibration.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
