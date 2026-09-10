#!/usr/bin/env python3
"""Evaluate TM-JEPA distributions, FDT construction, and augmented-state CK."""

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
from protein_jepa_frame.models.jepa import ProteinJEPA


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    first = torch.linalg.vector_norm(samples - target[:, None], dim=-1).mean(dim=1)
    pair = torch.cdist(samples, samples)
    count = samples.shape[1]
    if count > 1:
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=samples.device)
        second = pair[:, off_diagonal].mean(dim=1)
    else:
        second = torch.zeros_like(first)
    return first - 0.5 * second


def _two_sample_energy_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Unbiased two-sample energy statistic for equally sized ensembles.

    The U-statistic may be slightly negative at finite sample size. Keeping the
    signed estimate avoids the large horizon-dependent positive bias caused by
    self-distances or by clamping every row independently.
    """
    cross = torch.cdist(left, right).mean(dim=(1, 2))
    count_left = left.shape[1]
    count_right = right.shape[1]
    if count_left < 2 or count_right < 2:
        raise ValueError("unbiased energy distance requires at least two samples")
    within_left = torch.cdist(left, left).sum(dim=(1, 2)) / (
        count_left * (count_left - 1)
    )
    within_right = torch.cdist(right, right).sum(dim=(1, 2)) / (
        count_right * (count_right - 1)
    )
    return 2.0 * cross - within_left - within_right


@torch.no_grad()
def _ck_error(model: ProteinJEPA, outputs: dict[str, torch.Tensor], samples: int) -> torch.Tensor:
    tm = model.tmjepa
    if tm is None:
        raise ValueError("TM-JEPA model is not enabled")
    current = outputs["tmjepa_current_cv"]
    auxiliary_mean = outputs["tmjepa_auxiliary_mean"]
    auxiliary_logvar = outputs["tmjepa_auxiliary_logvar"]
    condition = outputs["tmjepa_condition"]
    temperature = outputs["tmjepa_temperature_k"]
    horizon = outputs["horizon_time_ps"]
    half = horizon * 0.5
    covariance_kwargs = (
        {"auxiliary_covariance": outputs["tmjepa_auxiliary_covariance"]}
        if "tmjepa_auxiliary_covariance" in outputs
        else {}
    )
    direct, _ = tm.dynamics.rollout(
        current,
        auxiliary_mean,
        auxiliary_logvar,
        condition,
        temperature,
        horizon,
        ensemble_size=samples,
        **covariance_kwargs,
    )
    first_q, first_a = tm.dynamics.rollout(
        current,
        auxiliary_mean,
        auxiliary_logvar,
        condition,
        temperature,
        half,
        ensemble_size=samples,
        **covariance_kwargs,
    )
    B, K, Q = first_q.shape
    condition_k = condition[:, None].expand(-1, K, -1).reshape(B * K, -1)
    temperature_k = temperature[:, None].expand(-1, K).reshape(-1)
    half_k = half[:, None].expand(-1, K).reshape(-1)
    composed, _ = tm.dynamics.rollout(
        first_q.reshape(B * K, Q),
        first_a.reshape(B * K, -1),
        torch.full_like(first_a.reshape(B * K, -1), -30.0),
        condition_k,
        temperature_k,
        half_k,
        ensemble_size=1,
    )
    composed = composed[:, 0].reshape(B, K, Q)
    return _two_sample_energy_distance(direct, composed)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    data_config = replace(
        config.data,
        manifest_path=str(args.manifest),
        batch_size=args.batch_size,
        num_workers=0,
        training_mode="paired_window",
        paired_all_horizons=True,
        sampling_mode="hierarchical",
    )
    dataset = build_dataset(data_config)
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
    checkpoint_config = checkpoint.get("config", {})
    model_values = checkpoint_config.get("model", config.model.__dict__)
    model = ProteinJEPA(model_config_from_dict(model_values, state)).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.tmjepa is None:
        raise ValueError("checkpoint does not contain TM-JEPA")

    totals: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    innovation_values: list[torch.Tensor] = []
    innovation_lagged: list[torch.Tensor] = []
    innovation_previous: list[torch.Tensor] = []
    path_nll_values: list[torch.Tensor] = []
    auxiliary_transition_kl_values: list[torch.Tensor] = []
    diffusion_minima: list[torch.Tensor] = []
    diagonal_diffusion_rates: list[torch.Tensor] = []
    correlated_diffusion_rates: list[torch.Tensor] = []
    coupling_rates: list[torch.Tensor] = []
    relaxation_times: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch = _move_batch(batch, device)
        outputs = model.forward_tmjepa(batch)
        samples = outputs["tmjepa_samples"]
        target = outputs["tmjepa_target_cv"]
        current = outputs["tmjepa_current_cv"]
        prediction = outputs["tmjepa_prediction"]
        std = outputs["tmjepa_sample_variance"].sqrt()
        score = _energy_score(samples, target)
        persistence = torch.linalg.vector_norm(current - target, dim=-1)
        ck = _ck_error(model, outputs, args.ck_samples) if args.ck_samples > 1 else torch.zeros_like(score)
        for horizon in outputs["horizon_frames"].unique(sorted=True):
            mask = outputs["horizon_frames"] == horizon
            item = totals[int(horizon)]
            item["rows"] += float(mask.sum())
            item["energy_score"] += float(score[mask].sum())
            item["persistence_score"] += float(persistence[mask].sum())
            item["squared_error"] += float((prediction[mask] - target[mask]).square().sum())
            item["values"] += float(target[mask].numel())
            item["coverage_50"] += float(
                ((target[mask] - prediction[mask]).abs() <= 0.67448975 * std[mask]).float().sum()
            )
            item["coverage_90"] += float(
                ((target[mask] - prediction[mask]).abs() <= 1.64485363 * std[mask]).float().sum()
            )
            item["ck_energy_distance"] += float(ck[mask].sum())
        innovation = outputs["tmjepa_innovation"]
        innovation_mask = outputs["tmjepa_innovation_mask"]
        innovation_values.append(innovation[innovation_mask].cpu())
        path_nll_values.append(outputs["tmjepa_path_nll"][innovation_mask].cpu())
        auxiliary_transition_kl_values.append(
            outputs["tmjepa_auxiliary_transition_kl"][innovation_mask].cpu()
        )
        lag_mask = innovation_mask[:, 1:] & innovation_mask[:, :-1]
        if lag_mask.any():
            innovation_lagged.append(innovation[:, 1:][lag_mask].cpu())
            innovation_previous.append(innovation[:, :-1][lag_mask].cpu())
        diffusion_minima.append(torch.linalg.eigvalsh(outputs["tmjepa_diffusion"]).min(dim=-1).values.cpu())
        diagonal_diffusion_rates.append(outputs["tmjepa_diagonal_diffusion_rates"].cpu())
        correlated_diffusion_rates.append(outputs["tmjepa_diffusion_rates"].cpu())
        coupling_rates.append(outputs["tmjepa_coupling_rates"].abs().cpu())
        relaxation_times.append(
            outputs["tmjepa_relaxation_rates"].clamp(min=1.0e-12).reciprocal().cpu()
        )

    horizons = {}
    all_positive = True
    for horizon, item in sorted(totals.items()):
        rows = max(item["rows"], 1.0)
        values = max(item["values"], 1.0)
        score = item["energy_score"] / rows
        persistence = item["persistence_score"] / rows
        skill = 1.0 - score / max(persistence, 1.0e-12)
        all_positive = all_positive and skill > 0.0
        horizons[str(horizon)] = {
            "rows": int(item["rows"]),
            "energy_score": score,
            "persistence_energy_score": persistence,
            "energy_skill": skill,
            "mean_cv_rmse": (item["squared_error"] / values) ** 0.5,
            "coverage_50": item["coverage_50"] / values,
            "coverage_90": item["coverage_90"] / values,
            "augmented_ck_energy_distance": item["ck_energy_distance"] / rows,
        }

    innovations = torch.cat(innovation_values) if innovation_values else torch.empty(0)
    if innovations.shape[0] > 1:
        mean = innovations.mean(dim=0)
        centered = innovations - mean
        covariance = centered.T @ centered / (centered.shape[0] - 1)
        identity = torch.eye(covariance.shape[0])
        innovation_mean_rms = float(mean.square().mean().sqrt())
        innovation_covariance_rmse = float((covariance - identity).square().mean().sqrt())
    else:
        innovation_mean_rms = None
        innovation_covariance_rmse = None
    if innovation_lagged:
        lagged = torch.cat(innovation_lagged)
        previous = torch.cat(innovation_previous)
        lagged = lagged - lagged.mean(dim=0, keepdim=True)
        previous = previous - previous.mean(dim=0, keepdim=True)
        lagged_covariance = lagged.T @ previous / max(lagged.shape[0] - 1, 1)
        innovation_lag1_rms = float(lagged_covariance.square().mean().sqrt())
        innovation_lag1_max = float(lagged_covariance.abs().max())
    else:
        innovation_lag1_rms = None
        innovation_lag1_max = None
    path_nll_per_dimension = (
        float(torch.cat(path_nll_values).mean()) if path_nll_values else None
    )
    auxiliary_transition_kl = (
        float(torch.cat(auxiliary_transition_kl_values).mean())
        if auxiliary_transition_kl_values
        else None
    )
    minimum_diffusion = float(torch.cat(diffusion_minima).min()) if diffusion_minima else None
    mean_diagonal_diffusion = (
        float(torch.cat(diagonal_diffusion_rates).mean()) if diagonal_diffusion_rates else None
    )
    mean_correlated_diffusion = (
        float(torch.cat(correlated_diffusion_rates).mean()) if correlated_diffusion_rates else None
    )
    mean_absolute_coupling = float(torch.cat(coupling_rates).mean()) if coupling_rates else None
    mean_relaxation = float(torch.cat(relaxation_times).mean()) if relaxation_times else None
    result = {
        "method": "Thermodynamic Mori-JEPA",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "horizons": horizons,
        "closure": {
            "innovation_mean_rms": innovation_mean_rms,
            "innovation_covariance_rmse": innovation_covariance_rmse,
            "innovation_lag1_rms": innovation_lag1_rms,
            "innovation_lag1_max_abs": innovation_lag1_max,
            "path_nll_per_dimension": path_nll_per_dimension,
            "auxiliary_transition_kl": auxiliary_transition_kl,
        },
        "physics": {
            "minimum_diffusion_eigenvalue": minimum_diffusion,
            "mean_diagonal_diffusion_rate_per_ps": mean_diagonal_diffusion,
            "mean_correlated_diffusion_rate_per_ps": mean_correlated_diffusion,
            "mean_absolute_memory_coupling_per_ps": mean_absolute_coupling,
            "mean_memory_relaxation_time_ps": mean_relaxation,
            "memory_enabled": bool(model.tmjepa.dynamics.memory_enabled),
            "fdt_parameter_tying": True,
            "temperature_explicit": True,
            "horizon_enters_drift": False,
            "strict_prefix_causal_filter": True,
            "integrator": "stochastic_heun",
            "maximum_integration_step_ps": model.tmjepa.dynamics.base_step_ps,
        },
        "promotion_gates": {
            "energy_skill_positive_every_horizon": all_positive,
            "diffusion_psd": minimum_diffusion is not None and minimum_diffusion > 0.0,
            "innovation_mean_closed": innovation_mean_rms is not None and innovation_mean_rms < 0.2,
            "innovation_covariance_closed": innovation_covariance_rmse is not None and innovation_covariance_rmse < 0.2,
            "innovation_temporally_white": innovation_lag1_rms is not None and innovation_lag1_rms < 0.1,
        },
    }
    result["promotion_pass"] = all(result["promotion_gates"].values())
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
    parser.add_argument("--seed", type=int, default=6101)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
