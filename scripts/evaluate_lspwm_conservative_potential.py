#!/usr/bin/env python3
"""Fit and validate a low-rank conservative potential correction.

This is the next small intervention after global mobility/free-energy
calibration.  The learned encoder, latent projection, and checkpoint remain
frozen.  Only a train-fitted scalar potential correction is added:

    Delta A(z, c) = b^T tanh(z) + tanh(z)^T W c.

The correction force is the exact negative gradient of this scalar potential.
The overdamped covariance is still tied to mobility through
``Cov = 2 * D * dt``.  PCA bases for the latent state and protein condition
are fit on training data only.  Ridge regularization is selected on a
train-only protein holdout; validation proteins are used only for reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_lspwm_physical_calibration import (
    _fit_train_calibration,
    _heteroscedastic_diagonal_nll,
    _physics_arrays,
)
from scripts.evaluate_lspwm_transition_calibration import (
    DEFAULT_RIDGE_LAMBDAS,
    _build_loader,
    _collect_split,
    _jsonable,
    _load_model,
    _point_metrics,
    _select_ridge_lambda,
    _train_holdout_masks,
)


def _fit_basis(
    values: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    dimension = min(int(dimension), right.shape[0])
    basis = right[:dimension].T
    scale = np.maximum((centered @ basis).std(axis=0), 1.0e-6)
    return values.mean(axis=0), basis / scale[None, :]


def _parameter_count(state_dim: int, condition_dim: int) -> int:
    return state_dim + state_dim * condition_dim


def _design_chunks(
    q: np.ndarray,
    condition: np.ndarray,
    covariance: np.ndarray,
    dt_ps: np.ndarray,
    *,
    mobility_scale: float,
    state_mean: np.ndarray,
    state_basis: np.ndarray,
    condition_mean: np.ndarray,
    condition_basis: np.ndarray,
    chunk_size: int = 64,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield displacement design matrices without materializing all rows."""
    n, state_dim = q.shape
    state_rank = state_basis.shape[1]
    condition_rank = condition_basis.shape[1]
    parameter_count = _parameter_count(state_rank, condition_rank)
    state_coordinates = (q - state_mean) @ state_basis
    condition_coordinates = (condition - condition_mean) @ condition_basis
    diffusion_diagonal = np.diagonal(covariance, axis1=1, axis2=2) / (
        2.0 * np.maximum(dt_ps[:, None], 1.0e-12)
    )
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        z = state_coordinates[start:stop]
        c = condition_coordinates[start:stop]
        factor = (
            -mobility_scale
            * dt_ps[start:stop, None]
            * diffusion_diagonal[start:stop]
        )
        design = np.empty((stop - start, state_dim, parameter_count), dtype=np.float64)
        offset = 0

        # b^T tanh(z).  The derivative is bounded by one.
        tanh_z = np.tanh(z)
        derivative = 1.0 - tanh_z * tanh_z
        design[:, :, offset : offset + state_rank] = (
            factor[:, :, None]
            * state_basis[None, :, :]
            * derivative[:, None, :]
        )
        offset += state_rank

        # tanh(z)^T W c, which is condition-dependent but remains scalar in q.
        for left in range(state_rank):
            for condition_index in range(condition_rank):
                design[:, :, offset] = (
                    factor
                    * state_basis[:, left][None, :]
                    * derivative[:, left, None]
                    * c[:, condition_index, None]
                )
                offset += 1
        if offset != parameter_count:
            raise AssertionError("potential design parameter count mismatch")
        yield (
            start,
            stop,
            design,
            np.maximum(
                mobility_scale
                * np.diagonal(
                    covariance[start:stop],
                    axis1=1,
                    axis2=2,
                ),
                1.0e-12,
            ),
        )


def _fit_design(
    q: np.ndarray,
    condition: np.ndarray,
    covariance: np.ndarray,
    dt_ps: np.ndarray,
    target: np.ndarray,
    *,
    mobility_scale: float,
    state_mean: np.ndarray,
    state_basis: np.ndarray,
    condition_mean: np.ndarray,
    condition_basis: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    parameter_count = _parameter_count(
        state_basis.shape[1],
        condition_basis.shape[1],
    )
    column_sum_squares = np.zeros(parameter_count, dtype=np.float64)
    for _, _, design, variance in _design_chunks(
        q,
        condition,
        covariance,
        dt_ps,
        mobility_scale=mobility_scale,
        state_mean=state_mean,
        state_basis=state_basis,
        condition_mean=condition_mean,
        condition_basis=condition_basis,
    ):
        weighted = design / np.sqrt(variance[:, :, None])
        column_sum_squares += (weighted * weighted).sum(axis=(0, 1))
    column_scale = np.sqrt(
        column_sum_squares / max(q.shape[0] * q.shape[1], 1)
    )
    column_scale = np.maximum(column_scale, 1.0e-8)
    gram = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    rhs = np.zeros(parameter_count, dtype=np.float64)
    for start, stop, design, variance in _design_chunks(
        q,
        condition,
        covariance,
        dt_ps,
        mobility_scale=mobility_scale,
        state_mean=state_mean,
        state_basis=state_basis,
        condition_mean=condition_mean,
        condition_basis=condition_basis,
    ):
        weighted_design = design / np.sqrt(variance[:, :, None])
        weighted_target = target[start:stop] / np.sqrt(variance)
        normalized_design = weighted_design / column_scale[None, None, :]
        gram += np.einsum(
            "ndp,ndq->pq",
            normalized_design,
            normalized_design,
        )
        rhs += np.einsum("ndp,nd->p", normalized_design, weighted_target)
    gram += ridge_lambda * np.eye(parameter_count)
    try:
        normalized_coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        normalized_coefficients = np.linalg.pinv(gram) @ rhs
    coefficients = normalized_coefficients / column_scale
    return coefficients, column_scale


def _predict_design(
    q: np.ndarray,
    condition: np.ndarray,
    covariance: np.ndarray,
    dt_ps: np.ndarray,
    coefficients: np.ndarray,
    *,
    mobility_scale: float,
    state_mean: np.ndarray,
    state_basis: np.ndarray,
    condition_mean: np.ndarray,
    condition_basis: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros_like(q, dtype=np.float64)
    for start, stop, design, _ in _design_chunks(
        q,
        condition,
        covariance,
        dt_ps,
        mobility_scale=mobility_scale,
        state_mean=state_mean,
        state_basis=state_basis,
        condition_mean=condition_mean,
        condition_basis=condition_basis,
    ):
        prediction[start:stop] = np.einsum(
            "ndp,p->nd",
            design,
            coefficients,
        )
    return prediction


def _fit_selection_mask(
    train: dict[str, np.ndarray],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit = np.zeros(train["horizon_frames"].shape[0], dtype=bool)
    holdout = np.zeros_like(fit)
    for horizon in sorted(np.unique(train["horizon_frames"]).tolist()):
        mask = train["horizon_frames"] == horizon
        local_fit, local_holdout, _, _ = _train_holdout_masks(
            train["protein_id"][mask],
            seed=seed + int(horizon),
        )
        indices = np.flatnonzero(mask)
        fit[indices[local_fit]] = True
        holdout[indices[local_holdout]] = True
    return fit, holdout


def _select_potential_lambda(
    train: dict[str, np.ndarray],
    physics: dict[str, np.ndarray],
    *,
    target: np.ndarray,
    mobility_scale: float,
    state_mean: np.ndarray,
    state_basis: np.ndarray,
    condition_mean: np.ndarray,
    condition_basis: np.ndarray,
    lambdas: tuple[float, ...],
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    fit_mask, holdout_mask = _fit_selection_mask(train, seed=seed)
    fit_coefficients = {}
    scores = []
    for ridge_lambda in lambdas:
        coefficients, _ = _fit_design(
            train["lspwm_context_state"][fit_mask],
            train["lspwm_condition"][fit_mask],
            physics["covariance"][fit_mask],
            train["horizon_time_ps"][fit_mask],
            target[fit_mask],
            mobility_scale=mobility_scale,
            state_mean=state_mean,
            state_basis=state_basis,
            condition_mean=condition_mean,
            condition_basis=condition_basis,
            ridge_lambda=ridge_lambda,
        )
        prediction = _predict_design(
            train["lspwm_context_state"][holdout_mask],
            train["lspwm_condition"][holdout_mask],
            physics["covariance"][holdout_mask],
            train["horizon_time_ps"][holdout_mask],
            coefficients,
            mobility_scale=mobility_scale,
            state_mean=state_mean,
            state_basis=state_basis,
            condition_mean=condition_mean,
            condition_basis=condition_basis,
        )
        scores.append(
            {
                "lambda": float(ridge_lambda),
                "mse": float(np.mean((prediction - target[holdout_mask]) ** 2)),
            }
        )
        fit_coefficients[ridge_lambda] = coefficients
    del fit_coefficients
    return min(scores, key=lambda item: (item["mse"], item["lambda"]))["lambda"], scores


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
    lambdas = tuple(
        float(value) for value in args.ridge_lambdas.split(",") if value.strip()
    )
    scalar = _fit_train_calibration(
        train,
        train_physics,
        lambdas,
        seed=args.seed,
    )
    mobility_scale = scalar["mobility_scale"]
    free_energy_scale = scalar["free_energy_scale"]

    state_mean, state_basis = _fit_basis(
        train["lspwm_context_state"],
        args.pca_dim,
    )
    condition_mean, condition_basis = _fit_basis(
        train["lspwm_condition"],
        args.condition_pca_dim,
    )
    train_force_displacement = (
        train_physics["q_drift"] * train["horizon_time_ps"][:, None]
    )
    train_scalar_displacement = (
        mobility_scale * free_energy_scale * train_force_displacement
    )
    correction_target = (
        train["lspwm_target_state"]
        - train["lspwm_context_state"]
        - train_scalar_displacement
    )
    selected_lambda, selection_scores = _select_potential_lambda(
        train,
        train_physics,
        target=correction_target,
        mobility_scale=mobility_scale,
        state_mean=state_mean,
        state_basis=state_basis,
        condition_mean=condition_mean,
        condition_basis=condition_basis,
        lambdas=lambdas,
        seed=args.seed,
    )
    coefficients, _ = _fit_design(
        train["lspwm_context_state"],
        train["lspwm_condition"],
        train_physics["covariance"],
        train["horizon_time_ps"],
        correction_target,
        mobility_scale=mobility_scale,
        state_mean=state_mean,
        state_basis=state_basis,
        condition_mean=condition_mean,
        condition_basis=condition_basis,
        ridge_lambda=selected_lambda,
    )

    by_horizon: dict[str, Any] = {}
    for horizon in sorted(np.unique(validation["horizon_frames"]).tolist()):
        mask = validation["horizon_frames"] == horizon
        current = validation["lspwm_context_state"][mask]
        target = validation["lspwm_target_state"][mask]
        dt = validation["horizon_time_ps"][mask]
        covariance = validation_physics["covariance"][mask]
        raw_force_displacement = validation_physics["q_drift"][mask] * dt[:, None]
        scalar_displacement = mobility_scale * free_energy_scale * raw_force_displacement
        correction_displacement = _predict_design(
            current,
            validation["lspwm_condition"][mask],
            covariance,
            dt,
            coefficients,
            mobility_scale=mobility_scale,
            state_mean=state_mean,
            state_basis=state_basis,
            condition_mean=condition_mean,
            condition_basis=condition_basis,
        )
        persistence = current
        scalar_prediction = current + scalar_displacement
        conservative_prediction = scalar_prediction + correction_displacement
        persistence_mse = float(np.mean((persistence - target) ** 2))
        scalar_residual = target - scalar_prediction
        conservative_residual = target - conservative_prediction
        by_horizon[str(horizon)] = {
            "rows": int(mask.sum()),
            "persistence": _point_metrics(
                persistence,
                current,
                target,
                persistence_mse,
            ),
            "scalar_calibration": _point_metrics(
                scalar_prediction,
                current,
                target,
                persistence_mse,
            ),
            "conservative_potential_calibration": _point_metrics(
                conservative_prediction,
                current,
                target,
                persistence_mse,
            ),
            "scalar_nll": _heteroscedastic_diagonal_nll(
                scalar_residual,
                covariance,
                mobility_scale,
            ),
            "conservative_potential_nll": _heteroscedastic_diagonal_nll(
                conservative_residual,
                covariance,
                mobility_scale,
            ),
            "covariance_trace": float(
                np.mean(
                    np.trace(
                        mobility_scale * covariance,
                        axis1=1,
                        axis2=2,
                    )
                )
            ),
            "empirical_conservative_residual_trace": float(
                np.mean(
                    np.sum(
                        (
                            conservative_residual
                            - conservative_residual.mean(axis=0, keepdims=True)
                        )
                        ** 2,
                        axis=1,
                    )
                )
            ),
            "covariance_to_conservative_residual_ratio": float(
                np.mean(
                    np.trace(
                        mobility_scale * covariance,
                        axis1=1,
                        axis2=2,
                    )
                )
                / max(
                    np.mean(
                        np.sum(
                            (
                                conservative_residual
                                - conservative_residual.mean(
                                    axis=0,
                                    keepdims=True,
                                )
                            )
                            ** 2,
                            axis=1,
                        )
                    ),
                    1.0e-12,
                )
            ),
            "correction_displacement_norm": float(
                np.linalg.norm(correction_displacement, axis=1).mean()
            ),
            "dt_ps_mean": float(dt.mean()),
        }

    return _jsonable(
        {
            "framework": "LSP-WM v4.4",
            "protocol": "frozen-latent conservative potential calibration",
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
            "state_pca_dim": int(state_basis.shape[1]),
            "condition_pca_dim": int(condition_basis.shape[1]),
            "scalar_calibration": scalar,
            "potential_correction": {
                "parameter_count": int(coefficients.size),
                "ridge_lambda": float(selected_lambda),
                "train_holdout_selection_scores": selection_scores,
                "scalar_potential_form": (
                    "Delta A = b^T tanh(z) + tanh(z)^T W c"
                ),
                "fit_data_boundary": (
                    "PCA, ridge selection, and coefficient fitting use train "
                    "proteins only"
                ),
            },
            "by_horizon": by_horizon,
            "claim_boundary": (
                "small-scale frozen-representation conservative-force "
                "diagnostic; no encoder update, checkpoint promotion, or "
                "test-set selection"
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
        default="artifacts/phase2_conservative_potential",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=80)
    parser.add_argument("--max-validation-batches", type=int, default=80)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--condition-pca-dim", type=int, default=8)
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
    output_path = output_dir / "conservative_potential.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
