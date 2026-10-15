#!/usr/bin/env python3
"""Evaluate a frozen-latent, FDT-consistent LSP-WM calibration layer.

The intervention preserves the learned latent representation and changes only
two global physical scales:

    D_calibrated = mobility_scale * D_model
    A_calibrated = free_energy_scale * A_model

For the overdamped Langevin law

    dq = -D grad(A) dt + sqrt(2 D dt) dW,

the calibrated mean is therefore

    q_next = q - mobility_scale * free_energy_scale
                    * D grad(A) dt

and the calibrated covariance remains exactly

    Cov = 2 * mobility_scale * D * dt.

The mobility scale is fitted from train-only protein-holdout innovations after
a train-fitted ridge delta baseline. The free-energy scale is fitted from the
same train-only holdout, using the physical force direction and a
heteroscedastic Gaussian weight. Validation proteins are never used to fit or
select either scale.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_lspwm_transition_calibration import (
    DEFAULT_RIDGE_LAMBDAS,
    _build_loader,
    _collect_split,
    _fit_linear,
    _jsonable,
    _load_model,
    _point_metrics,
    _predict_linear,
    _select_ridge_lambda,
    _train_holdout_masks,
)


def _physics_arrays(
    model: torch.nn.Module,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    world_model = model.lspwm.world_model
    current = torch.from_numpy(split["lspwm_context_state"]).to(device)
    condition = torch.from_numpy(split["lspwm_condition"]).to(device)
    temperature = torch.from_numpy(split["lspwm_temperature_k"]).to(device)
    dt = torch.from_numpy(split["horizon_time_ps"]).to(device)
    auxiliary = current.new_zeros(current.shape[0], world_model.config.auxiliary_dim)

    if world_model.history_closure is not None:
        raise ValueError(
            "physical calibration requires the canonical Markov path; "
            "history closure must be disabled for this experiment"
        )
    with torch.no_grad():
        step = world_model.step(
            current,
            auxiliary,
            condition,
            dt,
            temperature,
            stochastic=False,
        )
        diffusion = world_model.coefficients(current, temperature)["diffusion"]
    covariance = 2.0 * diffusion * dt[:, None, None]
    return {
        "current": current.cpu().numpy(),
        "q_drift": step["q_drift"].cpu().numpy(),
        "covariance": covariance.cpu().numpy(),
        "condition": condition.cpu().numpy(),
        "horizon_frames": split["horizon_frames"],
        "horizon_time_ps": split["horizon_time_ps"],
        "protein_id": split["protein_id"],
    }


def _diagonal_variance(covariance: np.ndarray) -> np.ndarray:
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    return np.maximum(variance, 1.0e-12)


def _fit_mobility_scale(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """MLE scalar for Cov = alpha * covariance with fixed covariance shape."""
    variance = _diagonal_variance(covariance)
    quadratic = np.sum(residual * residual / variance, axis=1)
    alpha = float(np.sum(quadratic) / max(residual.shape[0] * residual.shape[1], 1))
    return max(alpha, 1.0e-12)


def _fit_free_energy_scale(
    delta: np.ndarray,
    force_displacement: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Fit a positive scalar potential scale along the model force direction."""
    variance = _diagonal_variance(covariance)
    weights = 1.0 / variance
    numerator = float(np.sum(weights * force_displacement * delta))
    denominator = float(np.sum(weights * force_displacement * force_displacement))
    if denominator <= 1.0e-12:
        return 0.0
    return max(0.0, numerator / denominator)


def _heteroscedastic_diagonal_nll(
    residual: np.ndarray,
    covariance: np.ndarray,
    scale: float = 1.0,
) -> float:
    variance = np.maximum(scale * _diagonal_variance(covariance), 1.0e-12)
    dimension = residual.shape[1]
    value = np.sum(
        residual * residual / variance + np.log(variance) + np.log(2.0 * np.pi),
        axis=1,
    )
    return float(0.5 * np.mean(value))


def _trace_metrics(
    residual: np.ndarray,
    covariance: np.ndarray,
    mobility_scale: float,
) -> dict[str, float]:
    centered = residual - residual.mean(axis=0, keepdims=True)
    empirical_trace = float(np.mean(np.sum(centered * centered, axis=1)))
    innovation_trace = float(np.mean(np.sum(residual * residual, axis=1)))
    model_trace = float(
        np.mean(np.trace(mobility_scale * covariance, axis1=1, axis2=2))
    )
    return {
        "model_covariance_trace": model_trace,
        "empirical_residual_covariance_trace": empirical_trace,
        "empirical_innovation_second_moment_trace": innovation_trace,
        "covariance_trace_ratio_vs_residual_covariance": model_trace
        / max(empirical_trace, 1.0e-12),
        "covariance_trace_ratio_vs_innovation_second_moment": model_trace
        / max(innovation_trace, 1.0e-12),
    }


def _fit_train_calibration(
    train: dict[str, np.ndarray],
    physics: dict[str, np.ndarray],
    ridge_lambdas: tuple[float, ...],
    *,
    seed: int,
) -> dict[str, Any]:
    holdout_records: list[dict[str, np.ndarray]] = []
    per_horizon: dict[str, Any] = {}
    horizons = sorted(np.unique(train["horizon_frames"]).tolist())
    for horizon in horizons:
        mask = train["horizon_frames"] == horizon
        current = train["lspwm_context_state"][mask]
        target = train["lspwm_target_state"][mask]
        delta = target - current
        proteins = train["protein_id"][mask]
        physics_mask = physics["horizon_frames"] == horizon
        if not np.array_equal(proteins, physics["protein_id"][physics_mask]):
            raise ValueError("physics and latent arrays are not aligned")

        ridge_lambda, selection = _select_ridge_lambda(
            current,
            delta,
            proteins,
            ridge_lambdas,
            seed=seed + int(horizon),
        )
        fit_mask, holdout_mask, fit_proteins, holdout_proteins = _train_holdout_masks(
            proteins,
            seed=seed + int(horizon),
        )
        fit = _fit_linear(
            current[fit_mask],
            delta[fit_mask],
            ridge_lambda=ridge_lambda,
        )
        holdout_delta = delta[holdout_mask]
        ridge_prediction = _predict_linear(fit, current[holdout_mask])
        ridge_residual = holdout_delta - ridge_prediction
        base_covariance = physics["covariance"][mask][holdout_mask]
        force_displacement = (
            physics["q_drift"][mask][holdout_mask]
            * train["horizon_time_ps"][mask][holdout_mask, None]
        )
        holdout_records.append(
            {
                "horizon": np.full(holdout_delta.shape[0], horizon, dtype=np.int64),
                "delta": holdout_delta,
                "ridge_residual": ridge_residual,
                "force_displacement": force_displacement,
                "covariance": base_covariance,
            }
        )
        per_horizon[str(horizon)] = {
            "ridge_lambda": ridge_lambda,
            "ridge_selection": selection,
            "fit_rows": int(fit_mask.sum()),
            "holdout_rows": int(holdout_mask.sum()),
            "fit_proteins": fit_proteins,
            "holdout_proteins": holdout_proteins,
        }

    combined = {
        key: np.concatenate([record[key] for record in holdout_records], axis=0)
        for key in (
            "horizon",
            "delta",
            "ridge_residual",
            "force_displacement",
            "covariance",
        )
    }
    mobility_scale = _fit_mobility_scale(
        combined["ridge_residual"],
        combined["covariance"],
    )
    free_energy_scale = _fit_free_energy_scale(
        combined["delta"],
        combined["force_displacement"] * mobility_scale,
        combined["covariance"],
    )
    return {
        "mobility_scale": mobility_scale,
        "free_energy_scale": free_energy_scale,
        "holdout_rows": int(combined["delta"].shape[0]),
        "holdout_proteins": int(
            np.unique(
                np.concatenate(
                    [
                        train["protein_id"][
                            train["horizon_frames"] == horizon
                        ][
                            _train_holdout_masks(
                                train["protein_id"][
                                    train["horizon_frames"] == horizon
                                ],
                                seed=seed + int(horizon),
                            )[1]
                        ]
                        for horizon in horizons
                    ]
                )
            ).size
        ),
        "train_holdout_ridge_residual_nll_before_calibration": _heteroscedastic_diagonal_nll(
            combined["ridge_residual"],
            combined["covariance"],
        ),
        "per_horizon": per_horizon,
    }


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
    train_physics = _physics_arrays(model, train, device)
    validation_physics = _physics_arrays(model, validation, device)
    ridge_lambdas = tuple(
        float(value) for value in args.ridge_lambdas.split(",") if value.strip()
    )
    calibration = _fit_train_calibration(
        train,
        train_physics,
        ridge_lambdas,
        seed=args.seed,
    )
    mobility_scale = calibration["mobility_scale"]
    free_energy_scale = calibration["free_energy_scale"]

    by_horizon: dict[str, Any] = {}
    for horizon in sorted(np.unique(validation["horizon_frames"]).tolist()):
        mask = validation["horizon_frames"] == horizon
        current = validation["lspwm_context_state"][mask]
        target = validation["lspwm_target_state"][mask]
        dt = validation["horizon_time_ps"][mask, None]
        raw_force_displacement = (
            validation_physics["q_drift"][mask] * dt
        )
        calibrated_prediction = current + (
            mobility_scale * free_energy_scale * raw_force_displacement
        )
        raw_prediction = current + raw_force_displacement
        covariance = validation_physics["covariance"][mask]
        calibrated_covariance = mobility_scale * covariance
        persistence_mse = float(np.mean((current - target) ** 2))
        calibrated_residual = target - calibrated_prediction
        raw_residual = target - raw_prediction
        block = {
            "rows": int(mask.sum()),
            "persistence": _point_metrics(
                current,
                current,
                target,
                persistence_mse,
            ),
            "raw_lspwm_force_step": _point_metrics(
                raw_prediction,
                current,
                target,
                persistence_mse,
            ),
            "calibrated_physical_step": _point_metrics(
                calibrated_prediction,
                current,
                target,
                persistence_mse,
            ),
            "raw_lspwm_nll": _heteroscedastic_diagonal_nll(
                raw_residual,
                covariance,
            ),
            "calibrated_physical_nll": _heteroscedastic_diagonal_nll(
                calibrated_residual,
                covariance,
                mobility_scale,
            ),
        }
        block["raw_lspwm_covariance"] = _trace_metrics(
            raw_residual,
            covariance,
            1.0,
        )
        block["calibrated_physical_covariance"] = _trace_metrics(
            calibrated_residual,
            covariance,
            mobility_scale,
        )
        block["mean_diffusion_diagonal"] = float(
            np.mean(
                np.diagonal(
                    covariance,
                    axis1=1,
                    axis2=2,
                )
                / (2.0 * validation["horizon_time_ps"][mask, None])
            )
        )
        block["dt_ps_mean"] = float(validation["horizon_time_ps"][mask].mean())
        by_horizon[str(horizon)] = block

    return _jsonable(
        {
            "framework": "LSP-WM v4.4",
            "protocol": "frozen-latent FDT-consistent physical scale calibration",
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
            "calibration": calibration,
            "by_horizon": by_horizon,
            "claim_boundary": (
                "small-scale frozen-representation calibration diagnostic; "
                "no encoder update, checkpoint promotion, or test-set selection"
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
        default="artifacts/phase2_physical_calibration",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=80)
    parser.add_argument("--max-validation-batches", type=int, default=80)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--ridge-lambdas",
        default=",".join(str(value) for value in DEFAULT_RIDGE_LAMBDAS),
    )
    args = parser.parse_args()
    result = evaluate(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "physical_calibration.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
