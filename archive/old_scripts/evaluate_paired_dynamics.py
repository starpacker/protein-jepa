from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA, v4_physics_terms, v4_slow_kinetic_terms


def _closed_local_metrics(item: dict[str, float]) -> dict[str, float]:
    values = max(item["values"], 1.0)
    pred_mse = item["pred_sse"] / values
    persistence_mse = item["persistence_sse"] / values
    metrics = {
        "mse": pred_mse,
        "persistence_mse": persistence_mse,
        "persistence_relative_skill": 1.0 - pred_mse / max(persistence_mse, 1.0e-12),
        "normalized_mse": item["normalized_pred_sse"] / values,
        "normalized_persistence_mse": item["normalized_persistence_sse"] / values,
    }
    metrics["normalized_persistence_relative_skill"] = 1.0 - metrics["normalized_mse"] / max(
        metrics["normalized_persistence_mse"], 1.0e-12
    )
    for control in ("shuffled_history", "shuffled_horizon"):
        raw_key = f"{control}_sse"
        normalized_key = f"normalized_{control}_sse"
        if item.get(raw_key, 0.0) > 0:
            metrics[f"{control}_mse"] = item[raw_key] / values
            metrics[f"normalized_{control}_mse"] = item[normalized_key] / values
    fixed = {}
    for key, value in item.items():
        if key.startswith("normalized_fixed_") and key.endswith("_sse"):
            horizon = key.removeprefix("normalized_fixed_").removesuffix("_sse")
            fixed[horizon] = value / values
    if fixed:
        metrics["normalized_fixed_horizon_mse"] = fixed
        best_horizon, best_mse = min(fixed.items(), key=lambda pair: pair[1])
        metrics["best_fixed_horizon"] = int(best_horizon)
        metrics["correct_vs_best_fixed_skill"] = 1.0 - metrics["normalized_mse"] / max(
            best_mse, 1.0e-12
        )
    uncertainty_values = item.get("uncertainty_values", 0.0)
    if uncertainty_values > 0:
        metrics.update(
            {
                "student_t_nll": item["student_t_nll_sum"] / uncertainty_values,
                "persistence_student_t_nll": item["persistence_student_t_nll_sum"] / uncertainty_values,
                "mean_normalized_scale": item["normalized_scale_sum"] / uncertainty_values,
                "coverage_1scale": item["coverage_1scale"] / uncertainty_values,
                "coverage_2scale": item["coverage_2scale"] / uncertainty_values,
            }
        )
    return metrics


def _closed_local_pair_metrics(item: dict[str, float]) -> dict[str, float]:
    values = max(item["values"], 1.0)
    mse = item["error_sse"] / values
    target_mse = item["target_sse"] / values
    normalized_mse = item["normalized_error_sse"] / values
    normalized_target_mse = item["normalized_target_sse"] / values
    return {
        "mse": mse,
        "zero_difference_mse": target_mse,
        "difference_relative_skill": 1.0 - mse / max(target_mse, 1.0e-12),
        "normalized_mse": normalized_mse,
        "normalized_zero_difference_mse": normalized_target_mse,
        "normalized_difference_relative_skill": 1.0 - normalized_mse / max(normalized_target_mse, 1.0e-12),
    }


def _v4_slow_metrics(item: dict[str, float]) -> dict[str, float]:
    values = max(item["values"], 1.0)
    prediction_mse = item["prediction_sse"] / values
    persistence_mse = item["persistence_sse"] / values
    metrics = {
        "brier_mse": prediction_mse,
        "persistence_brier_mse": persistence_mse,
        "persistence_relative_skill": 1.0 - prediction_mse / max(persistence_mse, 1.0e-12),
        "mean_lambda2": item["lambda2_sum"] / max(item["rows"], 1.0),
    }
    if item.get("shuffled_history_sse", 0.0) > 0:
        metrics["shuffled_history_brier_mse"] = item["shuffled_history_sse"] / values
    if item.get("shuffled_horizon_sse", 0.0) > 0:
        metrics["shuffled_horizon_brier_mse"] = item["shuffled_horizon_sse"] / values
    fixed = {}
    for key, value in item.items():
        if key.startswith("fixed_") and key.endswith("_sse"):
            horizon = key.removeprefix("fixed_").removesuffix("_sse")
            fixed[horizon] = value / values
    if fixed:
        metrics["fixed_horizon_brier_mse"] = fixed
        best_horizon, best_mse = min(fixed.items(), key=lambda pair: pair[1])
        metrics["best_fixed_horizon"] = int(best_horizon)
        metrics["correct_vs_best_fixed_skill"] = 1.0 - prediction_mse / max(best_mse, 1.0e-12)
    return metrics


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    max_batches: int,
    batch_size: int,
    seed: int,
    device_name: str,
) -> dict:
    config = load_config(config_path)
    data_config = replace(
        config.data,
        manifest_path=str(manifest_path),
        batch_size=batch_size,
        num_workers=0,
        training_mode="paired_window",
        sampling_mode="hierarchical",
        load_forces=False,
    )
    dataset = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=max_batches * batch_size,
        seed=seed,
        dataset_weights=data_config.dataset_weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    artifact_config = json.loads((checkpoint_path.parent / "config.json").read_text())
    model = ProteinJEPA(model_config_from_dict(artifact_config["model"], state)).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    slow_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    offline_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    local_observable_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    closed_local_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    closed_local_pair_sums: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    v4_slow_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    v4_slow_global: dict[str, float] = defaultdict(float)
    v4c_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    v4c_global: dict[str, float] = defaultdict(float)
    observable_control_values: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            outputs = model.forward_paired(batch)
            if "v4_slow_prediction" in outputs:
                v4_terms = v4_slow_kinetic_terms(outputs, lags=config.model.v4_slow_vamp_lags)
                for key in (
                    "vamp_loss", "balance_loss", "assignment_entropy", "reconstruction_loss",
                    "base_transition_loss", "base_transition_skill",
                    "rollout_transition_loss", "rollout_transition_skill", "mean_lambda2",
                ):
                    v4_slow_global[key] += float(v4_terms[key].detach().cpu())
                v4_slow_global["batches"] += 1.0
            if "v4c_prediction" in outputs:
                physics_terms = v4_physics_terms(
                    outputs,
                    lags=config.model.v4_physics_vamp_lags,
                    force_scale=config.model.v4_physics_force_scale,
                    per_trajectory_vamp=config.model.v4_physics_per_trajectory_vamp,
                    horizon_ranking_margin=config.optimization.v4_physics_horizon_ranking_margin,
                )
                for key in (
                    "vamp_loss", "balance_loss", "assignment_entropy", "reconstruction_loss",
                    "cross_covariance_loss", "increment_covariance_loss", "drift_loss",
                    "timescale_separation_loss",
                    "trajectory_balance_loss", "activity_loss", "latent_variance_loss",
                    "assignment_switching", "physical_activity",
                    "within_assignment_variance", "between_assignment_variance",
                    "identity_leakage_ratio",
                    "transition_nll", "horizon_ranking_loss", "correct_vs_best_fixed_skill",
                    "horizon_difference_loss", "target_horizon_difference",
                    "stationary_loss", "base_transition_loss", "base_transition_skill",
                    "rollout_transition_loss", "rollout_transition_skill",
                    "detailed_balance_residual", "mean_lambda2", "latent_autocorrelation",
                    "residual_autocorrelation",
                ):
                    v4c_global[key] += float(physics_terms[key].detach().cpu())
                v4c_global["batches"] += 1.0
            shuffled_history_outputs = None
            shuffled_horizon_outputs = None
            batch_size_actual = int(outputs["horizon_frames"].shape[0])
            group_index = batch.get("context_group_index")
            if batch_size_actual > 1:
                row_index = torch.arange(batch_size_actual, device=device)
                history_permutation = torch.roll(row_index, shifts=1)
                horizon_permutation = history_permutation
                if group_index is not None and group_index.unique().numel() > 1:
                    groups = group_index.unique(sorted=True)
                    source_group = {
                        int(group): int(groups[(position - 1) % len(groups)])
                        for position, group in enumerate(groups)
                    }
                    history_rows = []
                    for row in row_index:
                        group = int(group_index[row])
                        horizon = outputs["horizon_frames"][row]
                        candidate = torch.nonzero(
                            (group_index == source_group[group]) & (outputs["horizon_frames"] == horizon),
                            as_tuple=False,
                        ).flatten()
                        history_rows.append(candidate[0])
                    history_permutation = torch.stack(history_rows)
                    horizon_rows = []
                    for group in groups:
                        members = torch.nonzero(group_index == group, as_tuple=False).flatten()
                        rolled = torch.roll(members, shifts=1)
                        horizon_rows.extend(zip(members.detach().cpu().tolist(), rolled))
                    horizon_permutation = row_index.clone()
                    for destination, source in horizon_rows:
                        horizon_permutation[destination] = source
                shuffled_history_batch = dict(batch)
                for key, value in batch.items():
                    shuffle_context = key.startswith("context_") or key in {
                        "temperature_k", "dataset_index", "dataset", "protein_id", "trajectory_id"
                    }
                    if not shuffle_context:
                        continue
                    if torch.is_tensor(value) and value.shape[0] == batch_size_actual:
                        shuffled_history_batch[key] = value[history_permutation]
                    elif isinstance(value, list) and len(value) == batch_size_actual:
                        indices = history_permutation.detach().cpu().tolist()
                        shuffled_history_batch[key] = [value[index] for index in indices]
                shuffled_history_outputs = model.forward_paired(shuffled_history_batch)

                shuffled_horizon_batch = dict(batch)
                shuffled_horizon_batch["horizon_frames"] = batch["horizon_frames"][horizon_permutation]
                shuffled_horizon_batch["horizon_time_ps"] = batch["horizon_time_ps"][horizon_permutation]
                shuffled_horizon_outputs = model.forward_paired(shuffled_horizon_batch)
            if "closed_local_prediction" in outputs and group_index is not None:
                prediction = outputs["closed_local_prediction"]
                target = outputs["future_closed_local_state"]
                valid = outputs["closed_local_mask"]
                scale = outputs["closed_local_state_scale"].to(dtype=prediction.dtype)
                for group in group_index.unique(sorted=True):
                    members = torch.nonzero(group_index == group, as_tuple=False).flatten()
                    for left_position in range(members.numel()):
                        left = members[left_position]
                        for right_position in range(left_position + 1, members.numel()):
                            right = members[right_position]
                            pair_valid = valid[left] & valid[right]
                            if not pair_valid.any():
                                continue
                            left_horizon = int(outputs["horizon_frames"][left])
                            right_horizon = int(outputs["horizon_frames"][right])
                            key = tuple(sorted((left_horizon, right_horizon)))
                            prediction_difference = prediction[right][pair_valid] - prediction[left][pair_valid]
                            target_difference = target[right][pair_valid] - target[left][pair_valid]
                            error = prediction_difference - target_difference
                            item = closed_local_pair_sums[key]
                            item["values"] += float(error.numel())
                            item["error_sse"] += float(error.square().sum().cpu())
                            item["target_sse"] += float(target_difference.square().sum().cpu())
                            item["normalized_error_sse"] += float((error / scale).square().sum().cpu())
                            item["normalized_target_sse"] += float(
                                (target_difference / scale).square().sum().cpu()
                            )
            horizon_values = outputs["horizon_frames"].detach().cpu().tolist()
            delta_prediction = outputs["delta_prediction"]
            delta_target = outputs["delta_target"]
            persistence = torch.zeros_like(delta_target)
            state_prediction = outputs["state_prediction"]
            future_endpoint = outputs["future_endpoint"]
            context_endpoint = outputs.get("persistence_endpoint", outputs["context_endpoint"])
            for horizon in sorted(set(int(value) for value in horizon_values)):
                mask = outputs["horizon_frames"] == horizon
                if not mask.any():
                    continue
                pred = delta_prediction[mask]
                target = delta_target[mask]
                base = persistence[mask]
                pred_state = state_prediction[mask]
                future_state = future_endpoint[mask]
                context_state = context_endpoint[mask]
                item = sums[horizon]
                item["n"] += float(mask.sum().item())
                item["pred_delta_sse"] += float((pred - target).square().sum().cpu())
                item["persistence_delta_sse"] += float((base - target).square().sum().cpu())
                item["pred_state_sse"] += float((pred_state - future_state).square().sum().cpu())
                item["persistence_state_sse"] += float((context_state - future_state).square().sum().cpu())
                item["dim"] = float(pred.shape[-1])
                if shuffled_history_outputs is not None:
                    shuffled_history_prediction = shuffled_history_outputs["delta_prediction"][mask]
                    item["shuffled_history_delta_sse"] += float(
                        (shuffled_history_prediction - target).square().sum().cpu()
                    )
                if shuffled_horizon_outputs is not None:
                    shuffled_horizon_prediction = shuffled_horizon_outputs["delta_prediction"][mask]
                    item["shuffled_horizon_delta_sse"] += float(
                        (shuffled_horizon_prediction - target).square().sum().cpu()
                    )
                if "gaussian_log_sigma" in outputs:
                    log_sigma = outputs["gaussian_log_sigma"][mask]
                    sigma = log_sigma.exp()
                    error = target - pred
                    persistence_error = target
                    pred_nll = 0.5 * (
                        error.square() * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma + math.log(2.0 * math.pi)
                    )
                    persistence_nll = 0.5 * (
                        persistence_error.square() * torch.exp(-2.0 * log_sigma)
                        + 2.0 * log_sigma
                        + math.log(2.0 * math.pi)
                    )

                    def gaussian_crps(residual: torch.Tensor) -> torch.Tensor:
                        standardized = residual / sigma.clamp(min=1.0e-8)
                        phi = torch.exp(-0.5 * standardized.square()) / math.sqrt(2.0 * math.pi)
                        cdf = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
                        return sigma * (
                            standardized * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi)
                        )

                    item["gaussian_nll_sum"] += float(pred_nll.sum().cpu())
                    item["persistence_gaussian_nll_sum"] += float(persistence_nll.sum().cpu())
                    item["gaussian_crps_sum"] += float(gaussian_crps(error).sum().cpu())
                    item["persistence_gaussian_crps_sum"] += float(gaussian_crps(persistence_error).sum().cpu())
                    item["coverage_1sigma"] += float((error.abs() <= sigma).sum().cpu())
                    item["coverage_2sigma"] += float((error.abs() <= 2.0 * sigma).sum().cpu())
                    item["gaussian_values"] += float(error.numel())
                if "observable_prediction" in outputs:
                    observable_prediction = outputs["observable_prediction"][mask]
                    observable_target = outputs["observable_target"][mask]
                    observable_error = observable_prediction - observable_target
                    observable_persistence = torch.zeros_like(observable_target)
                    observable_persistence[:, 0] = 1.0
                    item["observable_abs_error"] += float(observable_error.abs().sum().cpu())
                    item["observable_squared_error"] += float(observable_error.square().sum().cpu())
                    item["observable_persistence_squared_error"] += float(
                        (observable_persistence - observable_target).square().sum().cpu()
                    )
                    item["observable_values"] += float(observable_error.numel())
                    controls = observable_control_values[horizon]
                    controls["target"].append(observable_target.detach().cpu())
                    controls["prediction"].append(observable_prediction.detach().cpu())
                    controls["persistence"].append(observable_persistence.detach().cpu())
                    if shuffled_history_outputs is not None:
                        shuffled_prediction = shuffled_history_outputs["observable_prediction"][mask]
                        shuffled_error = shuffled_prediction - observable_target
                        item["observable_shuffled_history_squared_error"] += float(
                            shuffled_error.square().sum().cpu()
                        )
                        controls["shuffled_history"].append(shuffled_prediction.detach().cpu())
                    if shuffled_horizon_outputs is not None:
                        shuffled_prediction = shuffled_horizon_outputs["observable_prediction"][mask]
                        shuffled_error = shuffled_prediction - observable_target
                        item["observable_shuffled_horizon_squared_error"] += float(
                            shuffled_error.square().sum().cpu()
                        )
                        controls["shuffled_horizon"].append(shuffled_prediction.detach().cpu())
                if "offline_delta_prediction" in outputs:
                    offline_item = offline_sums[horizon]
                    offline_pred = outputs["offline_delta_prediction"][mask]
                    offline_target = outputs["offline_delta_target"][mask]
                    offline_item["n"] += float(mask.sum().item())
                    offline_item["pred_sse"] += float((offline_pred - offline_target).square().sum().cpu())
                    offline_item["persistence_sse"] += float(offline_target.square().sum().cpu())
                    offline_item["dim"] = float(offline_pred.shape[-1])
                if "local_observable_prediction" in outputs:
                    local_item = local_observable_sums[horizon]
                    local_mask = outputs["local_observable_mask"][mask]
                    if local_mask.any():
                        local_prediction = outputs["local_observable_prediction"][mask][local_mask]
                        local_target = outputs["local_observable_target"][mask][local_mask]
                        local_persistence = torch.zeros_like(local_target)
                        local_persistence[:, 0] = 1.0
                        local_item["values"] += float(local_target.numel())
                        local_item["pred_sse"] += float((local_prediction - local_target).square().sum().cpu())
                        local_item["persistence_sse"] += float(
                            (local_persistence - local_target).square().sum().cpu()
                        )
                        if shuffled_history_outputs is not None:
                            shuffled_prediction = shuffled_history_outputs["local_observable_prediction"][mask][local_mask]
                            local_item["shuffled_history_sse"] += float(
                                (shuffled_prediction - local_target).square().sum().cpu()
                            )
                        if shuffled_horizon_outputs is not None:
                            shuffled_prediction = shuffled_horizon_outputs["local_observable_prediction"][mask][local_mask]
                            local_item["shuffled_horizon_sse"] += float(
                                (shuffled_prediction - local_target).square().sum().cpu()
                            )
                if "closed_local_prediction" in outputs:
                    closed_item = closed_local_sums[horizon]
                    closed_mask = outputs["closed_local_mask"][mask]
                    if closed_mask.any():
                        closed_prediction = outputs["closed_local_prediction"][mask][closed_mask]
                        closed_target = outputs["future_closed_local_state"][mask][closed_mask]
                        closed_current = outputs["context_closed_local_state"][mask][closed_mask]
                        closed_scale = outputs.get(
                            "closed_local_state_scale", torch.ones(closed_target.shape[-1], device=device)
                        ).to(dtype=closed_target.dtype)
                        closed_item["values"] += float(closed_target.numel())
                        closed_item["pred_sse"] += float((closed_prediction - closed_target).square().sum().cpu())
                        closed_item["persistence_sse"] += float((closed_current - closed_target).square().sum().cpu())
                        closed_item["normalized_pred_sse"] += float(
                            ((closed_prediction - closed_target) / closed_scale).square().sum().cpu()
                        )
                        closed_item["normalized_persistence_sse"] += float(
                            ((closed_current - closed_target) / closed_scale).square().sum().cpu()
                        )
                        if "closed_local_log_sigma" in outputs:
                            log_sigma = outputs["closed_local_log_sigma"][mask][closed_mask]
                            sigma = log_sigma.exp()
                            normalized_error = (closed_prediction - closed_target) / closed_scale
                            normalized_persistence_error = (closed_current - closed_target) / closed_scale
                            student_t_df = config.optimization.closed_local_student_t_df
                            normalizer = (
                                math.lgamma(0.5 * student_t_df)
                                - math.lgamma(0.5 * (student_t_df + 1.0))
                                + 0.5 * math.log(student_t_df * math.pi)
                            )

                            def student_t_nll(error: torch.Tensor) -> torch.Tensor:
                                standardized = error / sigma.clamp(min=1.0e-6)
                                return (
                                    log_sigma + normalizer
                                    + 0.5 * (student_t_df + 1.0)
                                    * torch.log1p(standardized.square() / student_t_df)
                                )

                            closed_item["student_t_nll_sum"] += float(
                                student_t_nll(normalized_error).sum().cpu()
                            )
                            closed_item["persistence_student_t_nll_sum"] += float(
                                student_t_nll(normalized_persistence_error).sum().cpu()
                            )
                            closed_item["normalized_scale_sum"] += float(sigma.sum().cpu())
                            closed_item["coverage_1scale"] += float(
                                (normalized_error.abs() <= sigma).sum().cpu()
                            )
                            closed_item["coverage_2scale"] += float(
                                (normalized_error.abs() <= 2.0 * sigma).sum().cpu()
                            )
                            closed_item["uncertainty_values"] += float(normalized_error.numel())
                        if shuffled_history_outputs is not None:
                            shuffled_prediction = shuffled_history_outputs["closed_local_prediction"][mask][closed_mask]
                            closed_item["shuffled_history_sse"] += float(
                                (shuffled_prediction - closed_target).square().sum().cpu()
                            )
                            closed_item["normalized_shuffled_history_sse"] += float(
                                ((shuffled_prediction - closed_target) / closed_scale).square().sum().cpu()
                            )
                        if shuffled_horizon_outputs is not None:
                            shuffled_prediction = shuffled_horizon_outputs["closed_local_prediction"][mask][closed_mask]
                            closed_item["shuffled_horizon_sse"] += float(
                                (shuffled_prediction - closed_target).square().sum().cpu()
                            )
                            closed_item["normalized_shuffled_horizon_sse"] += float(
                                ((shuffled_prediction - closed_target) / closed_scale).square().sum().cpu()
                            )
                        if "closed_local_fixed_predictions" in outputs:
                            fixed_horizons = outputs["closed_local_fixed_horizons"].detach().cpu().tolist()
                            for fixed_index, fixed_horizon in enumerate(fixed_horizons):
                                fixed_prediction = outputs["closed_local_fixed_predictions"][mask, fixed_index][
                                    closed_mask
                                ]
                                closed_item[f"normalized_fixed_{int(fixed_horizon)}_sse"] += float(
                                    ((fixed_prediction - closed_target) / closed_scale).square().sum().cpu()
                                )
                if "slow_delta_prediction" in outputs:
                    slow_item = slow_sums[horizon]
                    slow_pred = outputs["slow_delta_prediction"][mask]
                    slow_target = outputs["slow_delta_target"][mask]
                    slow_pred_state = outputs["slow_state_prediction"][mask]
                    slow_future = outputs["future_slow_state"][mask]
                    slow_context = outputs.get("slow_persistence_state", outputs["context_slow_state"])[mask]
                    slow_item["n"] += float(mask.sum().item())
                    slow_item["pred_delta_sse"] += float((slow_pred - slow_target).square().sum().cpu())
                    slow_item["persistence_delta_sse"] += float((torch.zeros_like(slow_target) - slow_target).square().sum().cpu())
                    slow_item["pred_state_sse"] += float((slow_pred_state - slow_future).square().sum().cpu())
                    slow_item["persistence_state_sse"] += float((slow_context - slow_future).square().sum().cpu())
                    slow_item["dim"] = float(slow_pred.shape[-1])
                if "v4_slow_prediction" in outputs:
                    v4_item = v4_slow_sums[horizon]
                    v4_prediction = outputs["v4_slow_prediction"][mask]
                    v4_target = outputs["v4_slow_target_assignment"][mask]
                    v4_current = outputs["v4_slow_current_assignment"][mask]
                    v4_item["values"] += float(v4_target.numel())
                    v4_item["rows"] += float(mask.sum())
                    v4_item["prediction_sse"] += float(
                        (v4_prediction - v4_target).square().sum().cpu()
                    )
                    v4_item["persistence_sse"] += float(
                        (v4_current - v4_target).square().sum().cpu()
                    )
                    transition = outputs["v4_slow_transition_matrix"][mask]
                    eigenvalues = torch.linalg.eigvals(transition).abs().sort(
                        dim=-1, descending=True
                    ).values
                    v4_item["lambda2_sum"] += float(eigenvalues[:, 1].sum().cpu())
                    if shuffled_history_outputs is not None:
                        shuffled = shuffled_history_outputs["v4_slow_prediction"][mask]
                        v4_item["shuffled_history_sse"] += float(
                            (shuffled - v4_target).square().sum().cpu()
                        )
                    if shuffled_horizon_outputs is not None:
                        shuffled = shuffled_horizon_outputs["v4_slow_prediction"][mask]
                        v4_item["shuffled_horizon_sse"] += float(
                            (shuffled - v4_target).square().sum().cpu()
                        )
                    if "v4_slow_fixed_predictions" in outputs:
                        fixed_horizons = outputs["v4_slow_fixed_horizons"].detach().cpu().tolist()
                        for fixed_index, fixed_horizon in enumerate(fixed_horizons):
                            fixed = outputs["v4_slow_fixed_predictions"][mask, fixed_index]
                            v4_item[f"fixed_{int(fixed_horizon)}_sse"] += float(
                                (fixed - v4_target).square().sum().cpu()
                            )
                if "v4c_prediction" in outputs:
                    v4c_item = v4c_sums[horizon]
                    v4c_prediction = outputs["v4c_prediction"][mask]
                    v4c_target = outputs["v4c_target_assignment"][mask]
                    v4c_current = outputs["v4c_current_assignment"][mask]
                    v4c_item["values"] += float(v4c_target.numel())
                    v4c_item["rows"] += float(mask.sum())
                    v4c_item["prediction_sse"] += float(
                        (v4c_prediction - v4c_target).square().sum().cpu()
                    )
                    v4c_item["persistence_sse"] += float(
                        (v4c_current - v4c_target).square().sum().cpu()
                    )
                    base_transition = outputs["v4c_base_transition_matrix"][mask]
                    eigenvalues = torch.linalg.eigvals(base_transition).abs().sort(
                        dim=-1, descending=True
                    ).values
                    v4c_item["lambda2_sum"] += float(eigenvalues[:, 1].sum().cpu())
                    if shuffled_history_outputs is not None:
                        shuffled = shuffled_history_outputs["v4c_prediction"][mask]
                        v4c_item["shuffled_history_sse"] += float(
                            (shuffled - v4c_target).square().sum().cpu()
                        )
                    if shuffled_horizon_outputs is not None:
                        shuffled = shuffled_horizon_outputs["v4c_prediction"][mask]
                        v4c_item["shuffled_horizon_sse"] += float(
                            (shuffled - v4c_target).square().sum().cpu()
                        )
                    if "v4c_fixed_predictions" in outputs:
                        fixed_horizons = outputs["v4c_fixed_horizons"].detach().cpu().tolist()
                        for fixed_index, fixed_horizon in enumerate(fixed_horizons):
                            fixed = outputs["v4c_fixed_predictions"][mask, fixed_index]
                            v4c_item[f"fixed_{int(fixed_horizon)}_sse"] += float(
                                (fixed - v4c_target).square().sum().cpu()
                            )

    result = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "checkpoint_step": checkpoint.get("step", "?"),
        "max_batches": max_batches,
        "batch_size": batch_size,
        "seed": seed,
        "horizons": {},
    }
    for horizon, item in sorted(sums.items()):
        n_values = item["n"] * item["dim"]
        pred_mse = item["pred_delta_sse"] / max(n_values, 1.0)
        persistence_mse = item["persistence_delta_sse"] / max(n_values, 1.0)
        state_denominator = item["persistence_state_sse"]
        metrics = {
            "n_pairs": int(item["n"]),
            "delta_mse": pred_mse,
            "persistence_delta_mse": persistence_mse,
            "persistence_relative_skill": 1.0 - pred_mse / max(persistence_mse, 1.0e-12),
            "endpoint_mse": item["pred_state_sse"] / max(n_values, 1.0),
            "endpoint_r2_vs_persistence": 1.0 - item["pred_state_sse"] / max(state_denominator, 1.0e-12),
        }
        if item.get("shuffled_history_delta_sse", 0.0) > 0:
            metrics["shuffled_history_delta_mse"] = item["shuffled_history_delta_sse"] / max(n_values, 1.0)
        if item.get("shuffled_horizon_delta_sse", 0.0) > 0:
            metrics["shuffled_horizon_delta_mse"] = item["shuffled_horizon_delta_sse"] / max(n_values, 1.0)
        if item.get("gaussian_values", 0.0) > 0:
            gaussian_values = item["gaussian_values"]
            metrics.update(
                {
                    "gaussian_nll": item["gaussian_nll_sum"] / gaussian_values,
                    "persistence_gaussian_nll": item["persistence_gaussian_nll_sum"] / gaussian_values,
                    "gaussian_crps": item["gaussian_crps_sum"] / gaussian_values,
                    "persistence_gaussian_crps": item["persistence_gaussian_crps_sum"] / gaussian_values,
                    "coverage_1sigma": item["coverage_1sigma"] / gaussian_values,
                    "coverage_2sigma": item["coverage_2sigma"] / gaussian_values,
                }
            )
        if item.get("observable_values", 0.0) > 0:
            observable_values = item["observable_values"]
            observable_mse = item["observable_squared_error"] / observable_values
            observable_persistence_mse = item["observable_persistence_squared_error"] / observable_values
            metrics.update(
                {
                    "observable_mae": item["observable_abs_error"] / observable_values,
                    "observable_mse": observable_mse,
                    "observable_persistence_mse": observable_persistence_mse,
                    "observable_persistence_relative_skill": 1.0
                    - observable_mse / max(observable_persistence_mse, 1.0e-12),
                }
            )
            if item.get("observable_shuffled_history_squared_error", 0.0) > 0:
                metrics["observable_shuffled_history_mse"] = (
                    item["observable_shuffled_history_squared_error"] / observable_values
                )
            if item.get("observable_shuffled_horizon_squared_error", 0.0) > 0:
                metrics["observable_shuffled_horizon_mse"] = (
                    item["observable_shuffled_horizon_squared_error"] / observable_values
                )
        result["horizons"][str(horizon)] = metrics
    if observable_control_values:
        result["observable_split_controls"] = {}
        for horizon, values in sorted(observable_control_values.items()):
            target = torch.cat(values["target"], dim=0)
            split = max(1, target.shape[0] // 2)
            if split >= target.shape[0]:
                continue
            train_target = target[:split]
            test_target = target[split:]
            mean_prediction = train_target.mean(dim=0, keepdim=True).expand_as(test_target)

            def split_mse(name: str) -> float:
                prediction = torch.cat(values[name], dim=0)[split:]
                return float((prediction - test_target).square().mean())

            control = {
                "fit_pairs": int(split),
                "test_pairs": int(test_target.shape[0]),
                "model_mse": split_mse("prediction"),
                "train_mean_mse": float((mean_prediction - test_target).square().mean()),
                "persistence_mse": split_mse("persistence"),
            }
            if values.get("shuffled_history"):
                control["shuffled_history_mse"] = split_mse("shuffled_history")
            if values.get("shuffled_horizon"):
                control["shuffled_horizon_mse"] = split_mse("shuffled_horizon")
            result["observable_split_controls"][str(horizon)] = control
    if slow_sums:
        result["slow_horizons"] = {}
        for horizon, item in sorted(slow_sums.items()):
            n_values = item["n"] * item["dim"]
            pred_mse = item["pred_delta_sse"] / max(n_values, 1.0)
            persistence_mse = item["persistence_delta_sse"] / max(n_values, 1.0)
            result["slow_horizons"][str(horizon)] = {
                "n_pairs": int(item["n"]),
                "delta_mse": pred_mse,
                "persistence_delta_mse": persistence_mse,
                "persistence_relative_skill": 1.0 - pred_mse / max(persistence_mse, 1.0e-12),
                "endpoint_mse": item["pred_state_sse"] / max(n_values, 1.0),
                "endpoint_r2_vs_persistence": 1.0 - item["pred_state_sse"] / max(item["persistence_state_sse"], 1.0e-12),
            }
    if offline_sums:
        result["offline_horizons"] = {}
        for horizon, item in sorted(offline_sums.items()):
            n_values = item["n"] * item["dim"]
            pred_mse = item["pred_sse"] / max(n_values, 1.0)
            persistence_mse = item["persistence_sse"] / max(n_values, 1.0)
            result["offline_horizons"][str(horizon)] = {
                "n_pairs": int(item["n"]),
                "delta_mse": pred_mse,
                "persistence_delta_mse": persistence_mse,
                "persistence_relative_skill": 1.0 - pred_mse / max(persistence_mse, 1.0e-12),
            }
    if local_observable_sums:
        result["local_observable_horizons"] = {}
        for horizon, item in sorted(local_observable_sums.items()):
            values = max(item["values"], 1.0)
            pred_mse = item["pred_sse"] / values
            persistence_mse = item["persistence_sse"] / values
            metrics = {
                "mse": pred_mse,
                "persistence_mse": persistence_mse,
                "persistence_relative_skill": 1.0 - pred_mse / max(persistence_mse, 1.0e-12),
            }
            if item.get("shuffled_history_sse", 0.0) > 0:
                metrics["shuffled_history_mse"] = item["shuffled_history_sse"] / values
            if item.get("shuffled_horizon_sse", 0.0) > 0:
                metrics["shuffled_horizon_mse"] = item["shuffled_horizon_sse"] / values
            result["local_observable_horizons"][str(horizon)] = metrics
    if closed_local_sums:
        result["closed_local_horizons"] = {}
        for horizon, item in sorted(closed_local_sums.items()):
            result["closed_local_horizons"][str(horizon)] = _closed_local_metrics(item)
    if closed_local_pair_sums:
        result["closed_local_horizon_pairs"] = {
            f"{left}-{right}": _closed_local_pair_metrics(item)
            for (left, right), item in sorted(closed_local_pair_sums.items())
        }
    if v4_slow_sums:
        result["v4_slow_horizons"] = {
            str(horizon): _v4_slow_metrics(item)
            for horizon, item in sorted(v4_slow_sums.items())
        }
        batches = max(v4_slow_global["batches"], 1.0)
        result["v4_slow_global"] = {
            key: value / batches
            for key, value in v4_slow_global.items()
            if key != "batches"
        }
    if v4c_sums:
        result["v4c_horizons"] = {
            str(horizon): _v4_slow_metrics(item)
            for horizon, item in sorted(v4c_sums.items())
        }
        batches = max(v4c_global["batches"], 1.0)
        result["v4c_global"] = {
            key: value / batches
            for key, value in v4c_global.items()
            if key != "batches"
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V2.1 paired-window dynamics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                config_path=args.config,
                checkpoint_path=args.checkpoint,
                manifest_path=args.manifest,
                output_path=args.output,
                max_batches=args.max_batches,
                batch_size=args.batch_size,
                seed=args.seed,
                device_name=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
