#!/usr/bin/env python3
"""Temperature-resolved closure and causal-memory evaluation for TM-JEPA V3."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_tmjepa_v3_state_dict
from scripts.evaluate_tmjepa import _ck_error, _energy_score


def _diffusion_gates(
    minimum_eigenvalue: float,
    maximum_condition_number: float,
    minimum_increment_std: float,
    minimum_allowed_increment_std: float,
    maximum_allowed_condition_number: float,
) -> dict[str, bool]:
    """Separate mathematical PSD from practically usable diffusion."""
    return {
        "diffusion_psd": minimum_eigenvalue > 0.0,
        "diffusion_nondegenerate": (
            minimum_increment_std >= minimum_allowed_increment_std
        ),
        "diffusion_well_conditioned": (
            maximum_condition_number <= maximum_allowed_condition_number
        ),
    }


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _first_context_rows(batch: dict, row_count: int, device: torch.device) -> torch.Tensor:
    """Return one row per context after all-horizon collation duplicates it."""
    group = batch.get("context_group_index")
    if group is None:
        return torch.arange(row_count, device=device)
    group = group.to(device)
    return torch.stack(
        [torch.nonzero(group == value, as_tuple=False)[0, 0] for value in group.unique(sorted=True)]
    )


def _closure(
    records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    selected_temperature: int | None,
    lags: tuple[int, ...],
) -> dict:
    innovations = []
    paths = []
    sequences = []
    masks = []
    for innovation, mask, path, temperature in records:
        row_mask = (
            torch.ones_like(temperature, dtype=torch.bool)
            if selected_temperature is None
            else temperature.round().long() == selected_temperature
        )
        if not row_mask.any():
            continue
        value = innovation[row_mask]
        value_mask = mask[row_mask]
        innovations.append(value[value_mask])
        paths.append(path[row_mask][value_mask])
        sequences.append(value)
        masks.append(value_mask)
    if not innovations:
        return {}
    values = torch.cat(innovations)
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    identity = torch.eye(covariance.shape[0])
    lag_metrics = {}
    for lag in lags:
        left_parts = []
        right_parts = []
        for sequence, mask in zip(sequences, masks):
            if lag >= sequence.shape[1]:
                continue
            valid = mask[:, lag:] & mask[:, :-lag]
            if valid.any():
                left_parts.append(sequence[:, lag:][valid])
                right_parts.append(sequence[:, :-lag][valid])
        if not left_parts:
            continue
        left = torch.cat(left_parts)
        right = torch.cat(right_parts)
        left = left - left.mean(dim=0, keepdim=True)
        right = right - right.mean(dim=0, keepdim=True)
        lag_covariance = left.T @ right / max(left.shape[0] - 1, 1)
        lag_correlation = lag_covariance / (
            left.std(dim=0, unbiased=False).clamp(min=1.0e-8)[:, None]
            * right.std(dim=0, unbiased=False).clamp(min=1.0e-8)[None, :]
        )
        diagonal = torch.diagonal(lag_correlation)
        off_diagonal_mask = ~torch.eye(
            lag_correlation.shape[0], dtype=torch.bool
        )
        symmetric = 0.5 * (lag_correlation + lag_correlation.T)
        antisymmetric = 0.5 * (lag_correlation - lag_correlation.T)
        correlation_energy = lag_correlation.square().sum().clamp(min=1.0e-12)
        lag_metrics[str(lag)] = {
            "rms": float(lag_covariance.square().mean().sqrt()),
            "max_abs": float(lag_covariance.abs().max()),
            "correlation_rms": float(lag_correlation.square().mean().sqrt()),
            "correlation_max_abs": float(lag_correlation.abs().max()),
            "diagonal_correlation_rms": float(diagonal.square().mean().sqrt()),
            "diagonal_correlation_mean": float(diagonal.mean()),
            "diagonal_correlation_minimum": float(diagonal.min()),
            "diagonal_correlation_maximum": float(diagonal.max()),
            "off_diagonal_correlation_rms": float(
                lag_correlation[off_diagonal_mask].square().mean().sqrt()
            ),
            "symmetric_correlation_energy_fraction": float(
                symmetric.square().sum() / correlation_energy
            ),
            "antisymmetric_correlation_energy_fraction": float(
                antisymmetric.square().sum() / correlation_energy
            ),
        }
    return {
        "transitions": values.shape[0],
        "innovation_mean_rms": float(mean.square().mean().sqrt()),
        "innovation_covariance_rmse": float((covariance - identity).square().mean().sqrt()),
        "innovation_covariance_min_eigenvalue": float(torch.linalg.eigvalsh(covariance).min()),
        "innovation_covariance_max_eigenvalue": float(torch.linalg.eigvalsh(covariance).max()),
        "path_nll_per_dimension": float(torch.cat(paths).mean()),
        "lagged_covariance": lag_metrics,
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    data_config = replace(
        config.data,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=0,
        paired_all_horizons=True,
        sampling_mode="hierarchical",
    )
    dataset = build_dataset(data_config)
    expected_temperatures = sorted(
        {
            int(round(record.temperature_k))
            for record in dataset.records
            if record.temperature_k is not None
        }
    )
    expected_proteins = {record.protein_id for record in dataset.records}
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=args.max_batches * args.batch_size,
        seed=args.seed,
        dataset_weights=data_config.dataset_weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    model = ProteinJEPA(
        model_config_from_dict(checkpoint["config"]["model"], state)
    ).to(device)
    model.load_state_dict(upgrade_tmjepa_v3_state_dict(model, state), strict=True)
    model.eval()
    if args.base_step_ps is not None:
        model.tmjepa.dynamics.base_step_ps = float(args.base_step_ps)
    if args.projected_force_gate_steps is not None:
        model.tmjepa.projected_force_corrector.horizon_gate_steps = float(
            args.projected_force_gate_steps
        )
    if model.config.tmjepa_version != 3 or model.tmjepa is None:
        raise ValueError("evaluate_tmjepa_v3 requires a V3 checkpoint")

    horizon_totals: dict[int, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    temperature_horizons: dict[tuple[int, int], defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    closure_records = []
    diffusion_minima = []
    diffusion_condition_numbers = []
    diffusion_increment_minima = []
    observation_to_diffusion_ratios = []
    activation_energies = []
    memory_activation_energies = []
    divergence_values = []
    kernel_ratios = []
    posterior_information = []
    memory_drift = []
    rotation_strengths = []
    fdt_residuals = []
    observation_noise_traces = []
    temperatures_seen = set()
    proteins_seen = set()
    protein_totals: dict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        proteins_seen.update(values["protein_id"])
        outputs = model.forward_tmjepa(values)
        samples = outputs["tmjepa_samples"]
        target = outputs["tmjepa_target_cv"]
        current = outputs["tmjepa_current_cv"]
        observed_current = outputs.get("tmjepa_observed_current_cv", current)
        score = _energy_score(samples, target)
        persistence = torch.linalg.vector_norm(observed_current - target, dim=-1)
        # CK sampling is an auxiliary diagnostic. Isolate its RNG stream so
        # enabling the diagnostic cannot change endpoint metrics in later batches.
        if args.ck_samples > 1:
            fork_devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                ck_seed = args.seed + 1_000_003 + batch_index
                torch.manual_seed(ck_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(ck_seed)
                ck = _ck_error(model, outputs, args.ck_samples)
        else:
            ck = torch.zeros_like(score)
        reset_mean = torch.zeros_like(outputs["tmjepa_auxiliary_mean"])
        reset_logvar = torch.zeros_like(outputs["tmjepa_auxiliary_logvar"])
        reset_covariance = torch.eye(
            model.tmjepa.dynamics.auxiliary_dim,
            device=device,
            dtype=current.dtype,
        ).expand(current.shape[0], -1, -1)
        reset_samples, _ = model.tmjepa.dynamics.rollout(
            current,
            reset_mean,
            reset_logvar,
            outputs["tmjepa_condition"],
            outputs["tmjepa_temperature_k"],
            outputs["horizon_time_ps"],
            auxiliary_covariance=reset_covariance,
            ensemble_size=samples.shape[1],
        )
        reset_corrector = getattr(model.tmjepa, "projected_force_corrector", None)
        if reset_corrector is not None:
            reset_samples = reset_samples + reset_corrector(
                current,
                reset_mean,
                outputs["tmjepa_condition"],
                outputs["tmjepa_temperature_k"],
                outputs["horizon_time_ps"],
                outputs.get("tmjepa_projected_force_history_features"),
            )[:, None, :]
        reset_samples = model.tmjepa.observe_samples(
            reset_samples, outputs["tmjepa_observation_noise_covariance"]
        )
        reset_score = _energy_score(reset_samples, target)
        if not model.tmjepa.dynamics.memory_enabled:
            # A disabled bath has no intervention. Independent resampling here
            # would create a spurious nonzero memory advantage.
            reset_score = score
        fork_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            history_reset_samples, _ = model.tmjepa.dynamics.rollout(
                current,
                reset_mean,
                reset_logvar,
                outputs["tmjepa_condition"],
                outputs["tmjepa_temperature_k"],
                outputs["horizon_time_ps"],
                auxiliary_covariance=reset_covariance,
                ensemble_size=samples.shape[1],
            )
            if reset_corrector is not None:
                history_reset_samples = history_reset_samples + reset_corrector(
                    current,
                    reset_mean,
                    outputs["tmjepa_condition"],
                    outputs["tmjepa_temperature_k"],
                    outputs["horizon_time_ps"],
                    None,
                )[:, None, :]
            history_reset_samples = model.tmjepa.observe_samples(
                history_reset_samples,
                outputs["tmjepa_observation_noise_covariance"],
            )
        history_reset_score = _energy_score(history_reset_samples, target)
        for row, protein in enumerate(values["protein_id"]):
            item = protein_totals[protein]
            item["rows"] += 1.0
            item["score"] += float(score[row])
            item["persistence"] += float(persistence[row])
            valid_path = outputs["tmjepa_innovation_mask"][row]
            if valid_path.any():
                item["path_nll"] += float(
                    outputs["tmjepa_path_nll"][row][valid_path].mean()
                )
                item["path_rows"] += 1.0
        temperature = outputs["tmjepa_temperature_k"].round().long()
        temperatures_seen.update(int(value) for value in temperature.unique())
        for horizon in outputs["horizon_frames"].unique(sorted=True):
            selected = outputs["horizon_frames"] == horizon
            item = horizon_totals[int(horizon)]
            item["rows"] += float(selected.sum())
            item["score"] += float(score[selected].sum())
            item["persistence"] += float(persistence[selected].sum())
            item["reset_score"] += float(reset_score[selected].sum())
            item["history_reset_score"] += float(
                history_reset_score[selected].sum()
            )
            item["ck"] += float(ck[selected].sum())
            std = outputs["tmjepa_sample_variance"][selected].sqrt()
            covered = (target[selected] - outputs["tmjepa_prediction"][selected]).abs()
            item["covered90"] += float((covered <= 1.64485363 * std).sum())
            item["values"] += float(target[selected].numel())
            for temp in temperature[selected].unique():
                temp_selected = selected & (temperature == temp)
                grouped = temperature_horizons[(int(temp), int(horizon))]
                grouped["rows"] += float(temp_selected.sum())
                grouped["score"] += float(score[temp_selected].sum())
                grouped["persistence"] += float(persistence[temp_selected].sum())
                grouped_std = outputs["tmjepa_sample_variance"][temp_selected].sqrt()
                grouped_error = (
                    target[temp_selected] - outputs["tmjepa_prediction"][temp_selected]
                ).abs()
                grouped["covered90"] += float(
                    (grouped_error <= 1.64485363 * grouped_std).sum()
                )
                grouped["values"] += float(target[temp_selected].numel())

        context_rows = _first_context_rows(values, samples.shape[0], device)
        closure_records.append(
            (
                outputs["tmjepa_innovation"][context_rows].cpu(),
                outputs["tmjepa_innovation_mask"][context_rows].cpu(),
                outputs["tmjepa_path_nll"][context_rows].cpu(),
                temperature[context_rows].cpu(),
            )
        )
        diffusion = outputs["tmjepa_diffusion"][context_rows]
        coupling = model.tmjepa.dynamics.kinetic_parameters(
            outputs["tmjepa_condition"][context_rows],
            outputs["tmjepa_temperature_k"][context_rows],
            current[context_rows],
        )["coupling"]
        relaxation = outputs["tmjepa_relaxation_rates"][context_rows]
        integrated_kernel = torch.einsum(
            "bqr,br,bsr->bqs",
            coupling,
            relaxation.clamp(min=1.0e-12).reciprocal(),
            coupling,
        )
        kernel_ratios.append(
            (
                torch.diagonal(integrated_kernel, dim1=-2, dim2=-1).sum(-1)
                / torch.diagonal(diffusion, dim1=-2, dim2=-1).sum(-1).clamp(min=1.0e-12)
            ).cpu()
        )
        memory_drift.append(
            torch.einsum(
                "bqr,br->bq",
                coupling,
                outputs["tmjepa_auxiliary_mean"][context_rows],
            ).cpu()
        )
        posterior_information.append(
            (
                1.0
                - torch.diagonal(
                    outputs["tmjepa_auxiliary_covariance"][context_rows],
                    dim1=-2,
                    dim2=-1,
                ).mean(dim=-1)
            ).cpu()
        )
        diffusion_eigenvalues = torch.linalg.eigvalsh(diffusion)
        diffusion_minima.append(diffusion_eigenvalues.min(dim=-1).values.cpu())
        diffusion_condition_numbers.append(
            (
                diffusion_eigenvalues[:, -1]
                / diffusion_eigenvalues[:, 0].clamp(min=1.0e-12)
            ).cpu()
        )
        observation_covariance = outputs[
            "tmjepa_observation_noise_covariance"
        ][context_rows]
        observation_dt = (
            values["context_time_ps"][context_rows, 1]
            - values["context_time_ps"][context_rows, 0]
        ).abs().clamp(min=1.0e-6)
        diffusion_increment_minima.append(
            torch.sqrt(
                2.0
                * diffusion_eigenvalues[:, 0].clamp(min=0.0)
                * observation_dt
            ).cpu()
        )
        observation_to_diffusion_ratios.append(
            (
                torch.diagonal(observation_covariance, dim1=-2, dim2=-1).sum(-1)
                / (
                    observation_dt
                    * torch.diagonal(diffusion, dim1=-2, dim2=-1).sum(-1)
                ).clamp(min=1.0e-12)
            ).cpu()
        )
        activation_energies.append(
            outputs["tmjepa_activation_energy_kcal_per_mol"][context_rows].cpu()
        )
        memory_activation_energies.append(
            outputs["tmjepa_memory_activation_energy_kcal_per_mol"][context_rows].cpu()
        )
        divergence_values.append(
            outputs["tmjepa_diffusion_divergence"][context_rows].cpu()
        )
        generator = outputs["tmjepa_auxiliary_generator"][context_rows]
        relaxation_diagonal = torch.diag_embed(relaxation)
        rotation_strengths.append(
            outputs["tmjepa_auxiliary_rotation"][context_rows].abs().mean().cpu()
        )
        fdt_residuals.append(
            (
                generator
                + generator.transpose(-1, -2)
                - 2.0 * relaxation_diagonal
            ).abs().max().cpu()
        )
        observation_noise_traces.append(
            torch.diagonal(
                outputs["tmjepa_observation_noise_covariance"][context_rows],
                dim1=-2,
                dim2=-1,
            ).mean().cpu()
        )

    horizons = {}
    for horizon, item in sorted(horizon_totals.items()):
        rows = max(item["rows"], 1.0)
        score = item["score"] / rows
        persistence = item["persistence"] / rows
        horizons[str(horizon)] = {
            "rows": int(item["rows"]),
            "energy_score": score,
            "persistence_energy_score": persistence,
            "energy_skill": 1.0 - score / max(persistence, 1.0e-12),
            "memory_reset_energy_score": item["reset_score"] / rows,
            "memory_filter_advantage": item["reset_score"] / rows - score,
            "history_reset_energy_score": item["history_reset_score"] / rows,
            "history_filter_advantage": (
                item["history_reset_score"] / rows - score
            ),
            "coverage_90": item["covered90"] / max(item["values"], 1.0),
            "numerical_ck_energy_distance": item["ck"] / rows,
        }
    by_temperature = {}
    for temperature in expected_temperatures:
        grouped_horizons = {}
        for horizon in sorted(horizon_totals):
            item = temperature_horizons[(temperature, horizon)]
            rows = max(item["rows"], 1.0)
            score = item["score"] / rows
            persistence = item["persistence"] / rows
            grouped_horizons[str(horizon)] = {
                "rows": int(item["rows"]),
                "energy_skill": 1.0 - score / max(persistence, 1.0e-12),
                "coverage_90": item["covered90"] / max(item["values"], 1.0),
            }
        by_temperature[str(temperature)] = {
            "closure": _closure(closure_records, temperature, args.lags),
            "horizons": grouped_horizons,
        }
    closure = _closure(closure_records, None, args.lags)
    complete_grid = set(expected_temperatures) == temperatures_seen and all(
        horizon["rows"] > 0
        for item in by_temperature.values()
        for horizon in item["horizons"].values()
    )
    covariance_gate = complete_grid and all(
        item["closure"].get("innovation_covariance_rmse", float("inf")) < 0.15
        and item["closure"].get(
            "innovation_covariance_min_eigenvalue", float("-inf")
        )
        >= args.min_innovation_covariance_eigenvalue
        and item["closure"].get(
            "innovation_covariance_max_eigenvalue", float("inf")
        )
        <= args.max_innovation_covariance_eigenvalue
        for item in by_temperature.values()
    )
    whiteness_gate = complete_grid and all(
        all(
            metric["correlation_rms"] < 0.05
            for metric in item["closure"].get("lagged_covariance", {}).values()
        )
        for item in by_temperature.values()
    )
    coverage_gate = complete_grid and all(
        0.85 <= horizon["coverage_90"] <= 0.95
        for item in by_temperature.values()
        for horizon in item["horizons"].values()
        if horizon["rows"] > 0
    )
    diffusion_minima_tensor = torch.cat(diffusion_minima)
    diffusion_condition_numbers_tensor = torch.cat(diffusion_condition_numbers)
    minimum_diffusion = float(diffusion_minima_tensor.min())
    maximum_diffusion_condition_number = float(
        diffusion_condition_numbers_tensor.max()
    )
    minimum_diffusion_increment_std = float(
        torch.cat(diffusion_increment_minima).min()
    )
    mean_filter_advantage = sum(
        item["memory_filter_advantage"] for item in horizons.values()
    ) / max(len(horizons), 1)
    memory_enabled = bool(model.tmjepa.dynamics.memory_enabled)
    gates = {
        "complete_temperature_horizon_grid": complete_grid,
        "complete_protein_coverage": proteins_seen == expected_proteins,
        "positive_skill_every_temperature_and_horizon": complete_grid and all(
            horizon["energy_skill"] > 0.0
            for item in by_temperature.values()
            for horizon in item["horizons"].values()
            if horizon["rows"] > 0
        ),
        "coverage_calibrated_every_temperature": coverage_gate,
        "covariance_closed_every_temperature": covariance_gate,
        "multi_lag_white_every_temperature": whiteness_gate,
        "mori_filter_predictive": (not memory_enabled) or mean_filter_advantage > 0.005,
    }
    gates.update(
        _diffusion_gates(
            minimum_diffusion,
            maximum_diffusion_condition_number,
            minimum_diffusion_increment_std,
            args.min_diffusion_increment_std,
            args.max_diffusion_condition_number,
        )
    )
    result = {
        "method": "Thermodynamic Mori-JEPA V3",
        "metric_definitions": {
            "memory_filter_advantage": (
                "bath-reset energy score minus filtered energy score; causal "
                "history and endpoint observation models are held fixed"
            ),
            "history_filter_advantage": (
                "bath-and-history-reset energy score minus filtered energy score; "
                "the endpoint observation model is held fixed"
            ),
            "persistence_energy_score": (
                "energy score of the last observed CV as a point forecast"
            ),
            "coverage_90": "marginal per-CV empirical coverage",
            "numerical_ck_energy_distance": (
                "unbiased two-sample energy U-statistic on the augmented latent "
                "transition; finite-sample estimates may be slightly negative"
            ),
        },
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "evaluation_protocol": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "sampled_contexts": args.batch_size * args.max_batches,
            "ck_samples": args.ck_samples,
            "lags": list(args.lags),
            "base_step_ps_override": args.base_step_ps,
            "minimum_innovation_covariance_eigenvalue": (
                args.min_innovation_covariance_eigenvalue
            ),
            "maximum_innovation_covariance_eigenvalue": (
                args.max_innovation_covariance_eigenvalue
            ),
        },
        "sampling_coverage": {
            "expected_temperatures_k": expected_temperatures,
            "observed_temperatures_k": sorted(temperatures_seen),
            "expected_proteins": len(expected_proteins),
            "observed_proteins": len(proteins_seen),
            "missing_proteins": sorted(expected_proteins - proteins_seen),
        },
        "horizons": horizons,
        "closure": closure,
        "by_temperature": by_temperature,
        "by_protein": {
            protein: {
                "rows": int(item["rows"]),
                "energy_score": item["score"] / max(item["rows"], 1.0),
                "persistence_energy_score": item["persistence"]
                / max(item["rows"], 1.0),
                "path_nll_per_dimension": item["path_nll"]
                / max(item["path_rows"], 1.0),
            }
            for protein, item in sorted(protein_totals.items())
        },
        "physics": {
            "minimum_diffusion_eigenvalue": minimum_diffusion,
            "minimum_diffusion_increment_std": minimum_diffusion_increment_std,
            "minimum_allowed_diffusion_increment_std": (
                args.min_diffusion_increment_std
            ),
            "mean_diffusion_condition_number": float(
                diffusion_condition_numbers_tensor.mean()
            ),
            "maximum_diffusion_condition_number": maximum_diffusion_condition_number,
            "maximum_allowed_diffusion_condition_number": (
                args.max_diffusion_condition_number
            ),
            "mean_observation_to_diffusion_increment_variance_ratio": float(
                torch.cat(observation_to_diffusion_ratios).mean()
            ),
            "mean_activation_energy_kcal_per_mol": float(
                torch.cat(activation_energies).mean()
            ),
            "mean_memory_activation_energy_kcal_per_mol": float(
                torch.cat(memory_activation_energies).mean()
            ),
            "mean_absolute_ito_divergence_per_ps": float(
                torch.cat(divergence_values).abs().mean()
            ),
            "mean_integrated_memory_to_diffusion_ratio": float(
                torch.cat(kernel_ratios).mean()
            ),
            "mean_memory_drift_rms_per_ps": float(
                torch.cat(memory_drift).square().mean().sqrt()
            ),
            "mean_bath_information_fraction": float(
                torch.cat(posterior_information).mean()
            ),
            "mean_memory_filter_advantage": mean_filter_advantage,
            "mean_auxiliary_rotation_rate_per_ps": float(
                torch.stack(rotation_strengths).mean()
            ),
            "maximum_fdt_generator_residual": float(
                torch.stack(fdt_residuals).max()
            ),
            "mean_observation_noise_variance": float(
                torch.stack(observation_noise_traces).mean()
            ),
            "state_dependent_mobility": True,
            "exact_ito_divergence": True,
            "eyring_einstein_temperature_clock": True,
            "heat_capacity_free_energy": True,
            "bayesian_mori_filter": True,
            "integrator": (
                "local_exact_affine_van_loan_fdt"
                if model.tmjepa.dynamics.kinetics.coupled_bath_enabled
                else "local_exact_ou_fdt"
            ),
        },
        "promotion_gates": gates,
        "promotion_pass": all(gates.values()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--ck-samples", type=int, default=8)
    parser.add_argument("--lags", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--base-step-ps", type=float)
    parser.add_argument("--projected-force-gate-steps", type=float)
    parser.add_argument("--min-diffusion-increment-std", type=float, default=0.01)
    parser.add_argument("--max-diffusion-condition-number", type=float, default=100.0)
    parser.add_argument(
        "--min-innovation-covariance-eigenvalue", type=float, default=0.1
    )
    parser.add_argument(
        "--max-innovation-covariance-eigenvalue", type=float, default=2.5
    )
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
