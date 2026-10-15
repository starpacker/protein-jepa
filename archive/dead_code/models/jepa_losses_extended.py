"""Extended loss functions: v4_physics, v5_contextual_dynamics, tmjepa, v51, etc.

Split from jepa.py to keep the core model class under 2000 lines.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from protein_jepa_frame.models.jepa_losses import (
    _latent_loss,
    metastable_vamp_terms,
    _per_trajectory_vamp_terms,
    _lag_autocorrelation,
    _whitened_cross_covariance_loss,
)
from protein_jepa_frame.models.lspwm import vamp2_loss as lspwm_vamp2_loss

def v4_physics_terms(
    outputs: dict,
    lags: tuple[int, ...] = (4, 8, 16),
    force_scale: float = 50.0,
    per_trajectory_vamp: bool = False,
    horizon_ranking_margin: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Representation, conservative force, and reversible kinetic terms for V4-C."""
    if "v4c_prediction" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {
            key: zero
            for key in (
                "reconstruction_loss", "vamp_loss", "balance_loss", "assignment_entropy",
                "cross_covariance_loss", "increment_covariance_loss", "drift_loss",
                "timescale_separation_loss",
                "trajectory_balance_loss", "activity_loss", "latent_variance_loss",
                "assignment_switching", "physical_activity",
                "within_assignment_variance", "between_assignment_variance",
                "identity_leakage_ratio",
                "transition_nll", "horizon_ranking_loss", "correct_vs_best_fixed_skill",
                "horizon_difference_loss", "target_horizon_difference",
                "force_loss", "force_cosine", "stationary_loss", "base_transition_loss",
                "base_transition_skill", "rollout_transition_loss", "rollout_transition_skill",
                "detailed_balance_residual", "mean_lambda2", "latent_autocorrelation",
                "residual_autocorrelation",
            )
        }
    vamp = metastable_vamp_terms(
        {
            "delta_target": outputs["delta_target"],
            "context_metastable_states": outputs["v4c_context_assignment"],
            "future_metastable_states": outputs["v4c_future_assignment"],
            "context_metastable_mask": outputs["v4c_context_mask"],
            "future_metastable_mask": outputs["v4c_future_mask"],
        },
        lags=lags,
    )
    trajectory_vamp = _per_trajectory_vamp_terms(
        outputs["v4c_context_assignment"],
        outputs["v4c_future_assignment"],
        outputs["v4c_context_mask"],
        outputs["v4c_future_mask"],
        lags,
    )
    if per_trajectory_vamp:
        vamp_loss = trajectory_vamp["vamp_loss"]
    else:
        vamp_loss = vamp["vamp_loss"]
    context_mask = outputs["v4c_context_mask"]
    future_mask = outputs["v4c_future_mask"]
    reconstruction_loss = 0.5 * (
        F.smooth_l1_loss(
            outputs["v4c_context_reconstruction"][context_mask],
            outputs["v4c_context_lowpass_target"][context_mask],
        )
        + F.smooth_l1_loss(
            outputs["v4c_future_reconstruction"][future_mask],
            outputs["v4c_future_lowpass_target"][future_mask],
        )
    )
    context_latent = outputs["v4c_context_latent"]
    context_residual = outputs["v4c_context_residual"]
    cross_covariance_loss = _whitened_cross_covariance_loss(
        context_latent[context_mask], context_residual[context_mask]
    )
    pair_mask = context_mask[:, 1:] & context_mask[:, :-1]
    latent_increment = context_latent[:, 1:] - context_latent[:, :-1]
    residual_increment = context_residual[:, 1:] - context_residual[:, :-1]
    increment_covariance_loss = _whitened_cross_covariance_loss(
        latent_increment[pair_mask], residual_increment[pair_mask]
    )
    base_lag = max(lags) if lags else 1
    latent_autocorrelation = _lag_autocorrelation(context_latent, context_mask, base_lag)
    residual_autocorrelation = _lag_autocorrelation(context_residual, context_mask, base_lag)
    timescale_separation_loss = F.relu(
        residual_autocorrelation.detach() + 0.05 - latent_autocorrelation
    )
    activity_lag = max(lags) if lags else 1
    assignment_changes = []
    physical_changes = []
    latent_variance_losses = []
    for assignment, physical, latent, mask in (
        (
            outputs["v4c_context_assignment"], outputs["v4c_context_dynamic_input"],
            outputs["v4c_context_latent"], context_mask,
        ),
        (
            outputs["v4c_future_assignment"], outputs["v4c_future_dynamic_input"],
            outputs["v4c_future_latent"], future_mask,
        ),
    ):
        valid_count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=latent.dtype)
        latent_mean = (latent * mask.unsqueeze(-1)).sum(dim=1) / valid_count
        latent_variance = (
            (latent - latent_mean.unsqueeze(1)).square() * mask.unsqueeze(-1)
        ).sum(dim=1) / valid_count
        latent_variance_losses.append(F.relu(0.25 - latent_variance).mean())
        if activity_lag < assignment.shape[1]:
            pair_valid = mask[:, :-activity_lag] & mask[:, activity_lag:]
            if pair_valid.any():
                assignment_changes.append(
                    (assignment[:, activity_lag:] - assignment[:, :-activity_lag])
                    .square().sum(dim=-1)[pair_valid]
                )
                physical_changes.append(
                    (physical[:, activity_lag:] - physical[:, :-activity_lag])
                    .square().mean(dim=-1)[pair_valid]
                )
    if assignment_changes:
        assignment_change = torch.cat(assignment_changes)
        physical_change = torch.cat(physical_changes)
        centered_assignment = assignment_change - assignment_change.mean()
        centered_physical = physical_change - physical_change.mean()
        activity_correlation = (centered_assignment * centered_physical).mean() / torch.sqrt(
            centered_assignment.square().mean() * centered_physical.square().mean()
        ).clamp(min=1.0e-8)
        activity_loss = 1.0 - activity_correlation
        assignment_switching = assignment_change.mean()
        physical_activity = physical_change.mean()
    else:
        activity_loss = context_latent.new_tensor(0.0)
        assignment_switching = context_latent.new_tensor(0.0)
        physical_activity = context_latent.new_tensor(0.0)
    latent_variance_loss = torch.stack(latent_variance_losses).mean()
    assignment_means = []
    within_assignment_variances = []
    for assignment, mask in (
        (outputs["v4c_context_assignment"], context_mask),
        (outputs["v4c_future_assignment"], future_mask),
    ):
        count = mask.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=assignment.dtype)
        mean = (assignment * mask.unsqueeze(-1)).sum(dim=1) / count
        variance = (
            (assignment - mean.unsqueeze(1)).square() * mask.unsqueeze(-1)
        ).sum(dim=(1, 2)) / (count.squeeze(1) * assignment.shape[-1])
        assignment_means.append(mean)
        within_assignment_variances.append(variance)
    assignment_means_tensor = torch.cat(assignment_means, dim=0)
    within_assignment_variance = torch.cat(within_assignment_variances).mean()
    between_assignment_variance = assignment_means_tensor.var(dim=0, unbiased=False).mean()
    identity_leakage_ratio = between_assignment_variance / within_assignment_variance.clamp(
        min=1.0e-8
    )
    drift_target = outputs["v4c_latent_delta"].detach()
    drift_prediction = outputs["v4c_energy_drift"]
    drift_loss = F.smooth_l1_loss(drift_prediction, drift_target)

    assignments = torch.cat(
        [outputs["v4c_context_assignment"], outputs["v4c_future_assignment"]], dim=1
    )
    assignment_mask = torch.cat([context_mask, future_mask], dim=1)
    occupancy = (assignments * assignment_mask.unsqueeze(-1)).sum(dim=1)
    occupancy = occupancy / assignment_mask.sum(dim=1, keepdim=True).clamp(min=1)
    stationary_loss = F.mse_loss(
        outputs["v4c_stationary_distribution"], occupancy.detach()
    )

    base_target = outputs["v4c_base_target"].detach()
    base_prediction = outputs["v4c_base_prediction"]
    base_current = outputs["v4c_base_current"]
    base_transition_loss = F.mse_loss(base_prediction, base_target)
    base_persistence = F.mse_loss(base_current, base_target)
    rollout_target = outputs["v4c_target_assignment"].detach()
    rollout_prediction = outputs["v4c_prediction"]
    rollout_current = outputs["v4c_current_assignment"]
    rollout_transition_loss = F.mse_loss(rollout_prediction, rollout_target)
    rollout_persistence = F.mse_loss(rollout_current, rollout_target)
    transition_nll = -(
        rollout_target * rollout_prediction.clamp(min=1.0e-8).log()
    ).sum(dim=-1).mean()
    horizon_ranking_loss = context_latent.new_tensor(0.0)
    correct_vs_best_fixed_skill = context_latent.new_tensor(0.0)
    if "v4c_fixed_predictions" in outputs:
        fixed_predictions = outputs["v4c_fixed_predictions"]
        fixed_horizons = outputs["v4c_fixed_horizons"]
        fixed_errors = (fixed_predictions - rollout_target[:, None]).square().mean(dim=-1)
        correct_error = (rollout_prediction - rollout_target).square().mean(dim=-1)
        wrong_mask = fixed_horizons.unsqueeze(0) != outputs["horizon_frames"].unsqueeze(1)
        ranking_values = F.relu(
            correct_error.unsqueeze(1) - fixed_errors + horizon_ranking_margin
        )[wrong_mask]
        if ranking_values.numel():
            horizon_ranking_loss = ranking_values.mean()
        best_wrong = fixed_errors.masked_fill(~wrong_mask, float("inf")).min(dim=1).values
        correct_vs_best_fixed_skill = (
            1.0 - correct_error / best_wrong.clamp(min=1.0e-8)
        ).mean()
    horizon_difference_loss = context_latent.new_tensor(0.0)
    target_horizon_difference = context_latent.new_tensor(0.0)
    group_index = outputs.get("context_group_index")
    if group_index is not None:
        difference_losses = []
        target_differences = []
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            for left_index in range(members.numel()):
                for right_index in range(left_index + 1, members.numel()):
                    left = members[left_index]
                    right = members[right_index]
                    predicted_difference = rollout_prediction[right] - rollout_prediction[left]
                    target_difference = rollout_target[right] - rollout_target[left]
                    difference_losses.append(
                        (predicted_difference - target_difference).square().mean()
                    )
                    target_differences.append(target_difference.square().mean())
        if difference_losses:
            horizon_difference_loss = torch.stack(difference_losses).mean()
            target_horizon_difference = torch.stack(target_differences).mean()

    force_loss = context_latent.new_tensor(0.0)
    force_cosine = context_latent.new_tensor(0.0)
    if "v4c_force_energy" in outputs:
        force_prediction = -torch.autograd.grad(
            outputs["v4c_force_energy"].sum(),
            outputs["v4c_force_coordinates"],
            create_graph=True,
            retain_graph=True,
        )[0]
        force_target = outputs["v4c_force_target"]
        force_mask = outputs["v4c_force_atom_mask"].unsqueeze(-1)
        valid = force_mask.to(dtype=force_prediction.dtype)
        atom_count = valid.sum(dim=(-3, -2), keepdim=True).clamp(min=1.0)
        prediction_mean = (force_prediction * valid).sum(dim=(-3, -2), keepdim=True) / atom_count
        target_mean = (force_target * valid).sum(dim=(-3, -2), keepdim=True) / atom_count
        force_prediction = (force_prediction - prediction_mean) / force_scale
        force_target = (force_target - target_mean) / force_scale
        force_loss = F.smooth_l1_loss(
            force_prediction[force_mask.expand_as(force_prediction)],
            force_target[force_mask.expand_as(force_target)],
            beta=0.25,
        )
        prediction_flat = (force_prediction * valid).flatten(start_dim=1)
        target_flat = (force_target * valid).flatten(start_dim=1)
        force_cosine = F.cosine_similarity(prediction_flat, target_flat, dim=-1).mean()

    with torch.no_grad():
        eigenvalues = torch.linalg.eigvals(outputs["v4c_base_transition_matrix"]).abs()
        mean_lambda2 = eigenvalues.sort(dim=-1, descending=True).values[:, 1].mean().real
    return {
        "reconstruction_loss": reconstruction_loss,
        "vamp_loss": vamp_loss,
        "balance_loss": vamp["balance_loss"],
        "assignment_entropy": vamp["assignment_entropy"],
        "trajectory_balance_loss": trajectory_vamp["trajectory_balance_loss"],
        "activity_loss": activity_loss,
        "latent_variance_loss": latent_variance_loss,
        "assignment_switching": assignment_switching,
        "physical_activity": physical_activity,
        "within_assignment_variance": within_assignment_variance,
        "between_assignment_variance": between_assignment_variance,
        "identity_leakage_ratio": identity_leakage_ratio,
        "cross_covariance_loss": cross_covariance_loss,
        "increment_covariance_loss": increment_covariance_loss,
        "timescale_separation_loss": timescale_separation_loss,
        "drift_loss": drift_loss,
        "force_loss": force_loss,
        "force_cosine": force_cosine,
        "stationary_loss": stationary_loss,
        "base_transition_loss": base_transition_loss,
        "base_transition_skill": 1.0 - base_transition_loss / base_persistence.clamp(min=1.0e-8),
        "rollout_transition_loss": rollout_transition_loss,
        "rollout_transition_skill": 1.0
        - rollout_transition_loss / rollout_persistence.clamp(min=1.0e-8),
        "transition_nll": transition_nll,
        "horizon_ranking_loss": horizon_ranking_loss,
        "correct_vs_best_fixed_skill": correct_vs_best_fixed_skill,
        "horizon_difference_loss": horizon_difference_loss,
        "target_horizon_difference": target_horizon_difference,
        "detailed_balance_residual": outputs["v4c_detailed_balance_residual"].abs().mean(),
        "mean_lambda2": mean_lambda2,
        "latent_autocorrelation": latent_autocorrelation,
        "residual_autocorrelation": residual_autocorrelation,
    }


def v5_contextual_dynamics_terms(
    outputs: dict,
    horizon_ranking_margin: float = 0.002,
) -> dict[str, torch.Tensor]:
    """Joint representation and forecast terms for contextual V5 dynamics."""
    if "v5_prediction" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {
            key: zero
            for key in (
                "transition_loss", "persistence_loss", "transition_skill",
                "state_reconstruction_loss", "feature_forecast_loss", "transition_nll",
                "observable_loss", "observable_zero_motion_loss", "observable_zero_motion_skill",
                "horizon_ranking_loss", "correct_vs_best_fixed_skill",
                "horizon_difference_loss", "target_horizon_difference",
                "observable_horizon_difference_loss", "observable_target_horizon_difference",
                "slot_diversity_loss", "gate_complexity_loss", "mean_active_slots",
                "effective_slots", "detailed_balance_residual",
            )
        }

    prediction = outputs["v5_prediction"]
    target = outputs["v5_target_assignment"].detach()
    current = outputs["v5_current_assignment"]
    slots = outputs["v5_context_slots"]
    transition_loss = F.mse_loss(prediction, target)
    persistence_loss = F.mse_loss(current, target)
    transition_nll = -(target * prediction.clamp(min=1.0e-8).log()).sum(dim=-1).mean()

    current_decoded = torch.bmm(current.unsqueeze(1), slots).squeeze(1)
    target_decoded = torch.bmm(target.unsqueeze(1), slots).squeeze(1)
    state_reconstruction_loss = 0.5 * (
        F.mse_loss(current_decoded, outputs["v5_current_feature"].detach())
        + F.mse_loss(target_decoded, outputs["v5_target_feature"].detach())
    )
    feature_forecast_loss = F.mse_loss(
        outputs["v5_predicted_feature"], outputs["v5_target_feature"].detach()
    )
    observable_target = outputs["v5_observable_target"].detach()
    observable_loss = F.smooth_l1_loss(
        outputs["v5_observable_prediction"], observable_target
    )
    observable_zero_motion = torch.zeros_like(observable_target)
    observable_zero_motion[:, 0] = 1.0
    observable_zero_motion_loss = F.smooth_l1_loss(
        observable_zero_motion, observable_target
    )

    gram = torch.bmm(slots, slots.transpose(1, 2))
    identity = torch.eye(slots.shape[1], device=slots.device, dtype=slots.dtype).unsqueeze(0)
    slot_diversity_loss = ((gram - identity) * (1.0 - identity)).square().mean()
    gates = outputs["v5_slot_gates"]
    active_count = gates.sum(dim=-1)
    gate_complexity_loss = gates.mean() + F.relu(2.0 - active_count).square().mean()
    normalized_gates = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
    effective_slots = torch.exp(
        -(normalized_gates * normalized_gates.clamp(min=1.0e-8).log()).sum(dim=-1)
    ).mean()

    horizon_ranking_loss = prediction.new_tensor(0.0)
    correct_vs_best_fixed_skill = prediction.new_tensor(0.0)
    if "v5_fixed_predictions" in outputs:
        fixed_predictions = outputs["v5_fixed_predictions"]
        fixed_horizons = outputs["v5_fixed_horizons"]
        fixed_errors = (fixed_predictions - target[:, None]).square().mean(dim=-1)
        correct_error = (prediction - target).square().mean(dim=-1)
        wrong_mask = fixed_horizons.unsqueeze(0) != outputs["horizon_frames"].unsqueeze(1)
        ranking = F.relu(
            correct_error.unsqueeze(1) - fixed_errors + horizon_ranking_margin
        )[wrong_mask]
        if ranking.numel():
            horizon_ranking_loss = ranking.mean()
        best_wrong = fixed_errors.masked_fill(~wrong_mask, float("inf")).min(dim=1).values
        finite = torch.isfinite(best_wrong)
        if finite.any():
            correct_vs_best_fixed_skill = (
                1.0 - correct_error[finite] / best_wrong[finite].clamp(min=1.0e-8)
            ).mean()

    horizon_difference_loss = prediction.new_tensor(0.0)
    target_horizon_difference = prediction.new_tensor(0.0)
    observable_horizon_difference_loss = prediction.new_tensor(0.0)
    observable_target_horizon_difference = prediction.new_tensor(0.0)
    group_index = outputs.get("context_group_index")
    if group_index is not None:
        difference_losses = []
        target_differences = []
        observable_difference_losses = []
        observable_target_differences = []
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            for left_index in range(members.numel()):
                for right_index in range(left_index + 1, members.numel()):
                    left = members[left_index]
                    right = members[right_index]
                    predicted_difference = prediction[right] - prediction[left]
                    target_difference = target[right] - target[left]
                    difference_losses.append(
                        (predicted_difference - target_difference).square().mean()
                    )
                    target_differences.append(target_difference.square().mean())
                    predicted_observable_difference = (
                        outputs["v5_observable_prediction"][right]
                        - outputs["v5_observable_prediction"][left]
                    )
                    target_observable_difference = (
                        observable_target[right] - observable_target[left]
                    )
                    observable_difference_losses.append(
                        (predicted_observable_difference - target_observable_difference)
                        .square().mean()
                    )
                    observable_target_differences.append(
                        target_observable_difference.square().mean()
                    )
        if difference_losses:
            horizon_difference_loss = torch.stack(difference_losses).mean()
            target_horizon_difference = torch.stack(target_differences).mean()
            observable_horizon_difference_loss = torch.stack(
                observable_difference_losses
            ).mean()
            observable_target_horizon_difference = torch.stack(
                observable_target_differences
            ).mean()

    return {
        "transition_loss": transition_loss,
        "persistence_loss": persistence_loss,
        "transition_skill": 1.0 - transition_loss / persistence_loss.clamp(min=1.0e-8),
        "state_reconstruction_loss": state_reconstruction_loss,
        "feature_forecast_loss": feature_forecast_loss,
        "observable_loss": observable_loss,
        "observable_zero_motion_loss": observable_zero_motion_loss,
        "observable_zero_motion_skill": 1.0
        - observable_loss / observable_zero_motion_loss.clamp(min=1.0e-8),
        "transition_nll": transition_nll,
        "horizon_ranking_loss": horizon_ranking_loss,
        "correct_vs_best_fixed_skill": correct_vs_best_fixed_skill,
        "horizon_difference_loss": horizon_difference_loss,
        "target_horizon_difference": target_horizon_difference,
        "observable_horizon_difference_loss": observable_horizon_difference_loss,
        "observable_target_horizon_difference": observable_target_horizon_difference,
        "slot_diversity_loss": slot_diversity_loss,
        "gate_complexity_loss": gate_complexity_loss,
        "mean_active_slots": active_count.mean(),
        "effective_slots": effective_slots,
        "detailed_balance_residual": outputs["v5_detailed_balance_residual"].abs().mean(),
    }


def _trajectory_centered_learned_cv_terms(
    learned_cv: torch.Tensor,
    mask: torch.Tensor,
    lags: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regularize dynamics after removing per-trajectory identity offsets."""
    weights = mask.to(learned_cv.dtype).unsqueeze(-1)
    count = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    trajectory_mean = (learned_cv * weights).sum(dim=1, keepdim=True) / count
    centered = (learned_cv - trajectory_mean) * weights
    values = centered[mask]
    zero = learned_cv.new_tensor(0.0)
    if values.shape[0] <= 1:
        return zero, zero, zero, zero

    std = torch.sqrt(values.var(dim=0, unbiased=False) + 1.0e-4)
    variance_loss = F.relu(0.25 - std).mean()
    normalized = values / std.detach().clamp(min=1.0e-3)
    covariance = normalized.T @ normalized / max(1, normalized.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().mean()

    def inverse_sqrt(matrix: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return (eigenvectors * eigenvalues.clamp(min=1.0e-6).rsqrt()) @ eigenvectors.T

    scores = []
    dim = learned_cv.shape[-1]
    for lag in sorted(set(int(value) for value in lags)):
        if lag <= 0 or lag >= learned_cv.shape[1]:
            continue
        pair_mask = mask[:, lag:] & mask[:, :-lag]
        x = centered[:, :-lag][pair_mask]
        y = centered[:, lag:][pair_mask]
        if x.shape[0] <= dim + 1:
            continue
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        denom = max(1, x.shape[0] - 1)
        identity = torch.eye(dim, device=x.device, dtype=x.dtype) * 1.0e-4
        c00 = x.T @ x / denom + identity
        c11 = y.T @ y / denom + identity
        c01 = x.T @ y / denom
        koopman = inverse_sqrt(c00) @ c01 @ inverse_sqrt(c11)
        scores.append(koopman.square().sum())
    vamp2_score = torch.stack(scores).mean() if scores else zero
    vamp_loss = -vamp2_score / float(dim)
    return variance_loss, covariance_loss, vamp_loss, vamp2_score


def tmjepa_terms(
    outputs: dict[str, torch.Tensor],
    memory_intervention_margin: float = 0.0,
    memory_relative_intervention_margin: float = 0.0,
    memory_ratio_min: float = 0.0,
    memory_ratio_max: float = 1.0e12,
    memory_ratio_target: float = 0.08,
    long_horizon_min_frames: int = 8,
    long_horizon_power: float = 1.0,
    vamp_lags: tuple[int, ...] = (1,),
    innovation_lags: tuple[int, ...] = (1, 2, 4, 8),
    diffusion_condition_target: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Proper distributional and Markov-closure objectives for TM-JEPA."""
    if "tmjepa_samples" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        terms = {
            key: zero
            for key in (
                "energy_score",
                "forecast_relative_mse_loss",
                "persistence_energy_score",
                "energy_skill",
                "transition_nll",
                "path_nll",
                "equilibrium_score_loss",
                "equilibrium_score_baseline",
                "equilibrium_score_skill",
                "innovation_mean_loss",
                "innovation_covariance_loss",
                "innovation_autocorrelation_loss",
                "innovation_covariance_spectral_loss",
                "innovation_covariance_worst_loss",
                "innovation_autocorrelation_worst_loss",
                "memory_intervention_loss",
                "memory_intervention_advantage",
                "memory_intervention_mse_advantage",
                "memory_relative_intervention_loss",
                "memory_relative_intervention_advantage",
                "memory_ratio_band_loss",
                "memory_log_ratio_loss",
                "memory_long_horizon_intervention_loss",
                "memory_long_horizon_intervention_advantage",
                "memory_long_horizon_relative_intervention_loss",
                "memory_long_horizon_relative_intervention_advantage",
                "memory_residual_loss",
                "memory_long_horizon_residual_loss",
                "long_horizon_energy_score",
                "long_horizon_forecast_relative_mse_loss",
                "long_horizon_transition_nll",
                "long_horizon_path_nll",
                "projected_force_regularization_loss",
                "auxiliary_kl_loss",
                "auxiliary_transition_kl_loss",
                "learned_cv_variance_loss",
                "learned_cv_covariance_loss",
                "vamp_loss",
                "vamp2_score",
                "mean_coverage_90",
                "soft_coverage_90",
                "coverage_calibration_loss",
                "minimum_diffusion_eigenvalue",
                "diffusion_condition_number",
                "diffusion_condition_loss",
                "mean_observation_noise_variance",
                "observation_to_diffusion_increment_variance_ratio",
                "correlated_observation_variance_fraction",
                "mean_endpoint_calibration_log_scale",
                "mean_endpoint_calibration_variance_multiplier",
                "integrated_memory_to_diffusion_ratio",
                "bath_information_fraction",
                "mean_relaxation_time_ps",
            )
        }
        learned_cv = outputs.get("tmjepa_learned_cv")
        if learned_cv is not None:
            mask = outputs["tmjepa_context_mask"]
            group_index = outputs.get("context_group_index")
            if group_index is not None:
                rows = torch.stack(
                    [
                        torch.nonzero(group_index == group, as_tuple=False).flatten()[0]
                        for group in group_index.unique(sorted=True)
                    ]
                )
                learned_cv = learned_cv[rows]
                mask = mask[rows]
            (
                terms["learned_cv_variance_loss"],
                terms["learned_cv_covariance_loss"],
                terms["vamp_loss"],
                terms["vamp2_score"],
            ) = _trajectory_centered_learned_cv_terms(
                learned_cv, mask, vamp_lags
            )
        return terms

    samples = outputs["tmjepa_samples"]
    target = outputs["tmjepa_target_cv"].detach()
    current = outputs["tmjepa_current_cv"].detach()
    observed_current = outputs.get("tmjepa_observed_current_cv", current).detach()
    projected_force_shift = outputs.get("tmjepa_projected_force_shift")
    projected_force_regularization_loss = (
        projected_force_shift.square().mean()
        if projected_force_shift is not None
        else samples.new_tensor(0.0)
    )
    horizon_frames = outputs.get("horizon_frames")
    if horizon_frames is not None:
        horizon = horizon_frames.detach().to(device=samples.device, dtype=samples.dtype)
        long_mask = horizon >= float(long_horizon_min_frames)
        if long_mask.any():
            normalized_horizon = horizon / horizon[long_mask].mean().clamp(min=1.0)
            long_horizon_weights = torch.where(
                long_mask,
                normalized_horizon.clamp(min=1.0e-6).pow(float(long_horizon_power)),
                torch.zeros_like(horizon),
            )
            long_horizon_weights = long_horizon_weights / long_horizon_weights[
                long_mask
            ].mean().clamp(min=1.0e-8)
        else:
            long_horizon_weights = torch.zeros(samples.shape[0], device=samples.device, dtype=samples.dtype)
    else:
        long_horizon_weights = torch.ones(samples.shape[0], device=samples.device, dtype=samples.dtype)

    def _row_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(values)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        numerator = (values * weights).sum()
        denominator = weights.expand_as(values).sum().clamp(min=1.0e-8)
        return numerator / denominator

    target_distance = torch.linalg.vector_norm(samples - target[:, None], dim=-1)
    pair_distance = torch.cdist(samples, samples)
    ensemble = samples.shape[1]
    if ensemble > 1:
        off_diagonal = ~torch.eye(ensemble, dtype=torch.bool, device=samples.device)
        diversity_by_row = pair_distance[:, off_diagonal].mean(dim=-1)
        diversity = diversity_by_row.mean()
    else:
        diversity_by_row = pair_distance.new_zeros(samples.shape[0])
        diversity = pair_distance.new_tensor(0.0)
    energy_score_by_row = target_distance.mean(dim=-1) - 0.5 * diversity_by_row
    energy_score = energy_score_by_row.mean()
    persistence_energy_score = torch.linalg.vector_norm(
        observed_current - target, dim=-1
    ).mean()
    energy_skill = 1.0 - energy_score / persistence_energy_score.clamp(min=1.0e-8)
    long_horizon_energy_score = _row_weighted_mean(
        energy_score_by_row, long_horizon_weights
    )

    prediction = outputs["tmjepa_prediction"]
    forecast_mse_by_row = (prediction - target).square().mean(dim=-1)
    persistence_mse_by_row = (observed_current - target).square().mean(dim=-1)
    forecast_relative_mse_by_row = forecast_mse_by_row / persistence_mse_by_row.clamp(
        min=1.0e-8
    )
    forecast_relative_mse_loss = forecast_relative_mse_by_row.mean()
    long_horizon_forecast_relative_mse_loss = _row_weighted_mean(
        forecast_relative_mse_by_row, long_horizon_weights
    )
    variance = outputs["tmjepa_sample_variance"].clamp(min=1.0e-8)
    transition_nll_by_value = 0.5 * (
        (target - prediction).square() / variance + variance.log() + math.log(2.0 * math.pi)
    )
    transition_nll = transition_nll_by_value.mean()
    long_horizon_transition_nll = _row_weighted_mean(
        transition_nll_by_value, long_horizon_weights
    )
    path_mask = outputs["tmjepa_innovation_mask"]
    path_values = outputs["tmjepa_path_nll"]
    path_nll = path_values[path_mask].mean() if path_mask.any() else samples.new_tensor(0.0)
    if path_mask.any():
        path_weights = long_horizon_weights[:, None].expand_as(path_values)
        long_horizon_path_nll = (
            path_values[path_mask] * path_weights[path_mask]
        ).sum() / path_weights[path_mask].sum().clamp(min=1.0e-8)
    else:
        long_horizon_path_nll = samples.new_tensor(0.0)
    equilibrium_score_loss = F.mse_loss(
        outputs["tmjepa_score_prediction"], outputs["tmjepa_score_target"].detach()
    )
    equilibrium_score_baseline = outputs["tmjepa_score_target"].detach().square().mean()
    equilibrium_score_skill = 1.0 - equilibrium_score_loss / equilibrium_score_baseline.clamp(
        min=1.0e-8
    )

    innovation = outputs["tmjepa_innovation"]
    innovation_mask = outputs["tmjepa_innovation_mask"]
    temperature = outputs.get("tmjepa_temperature_k")
    closure_groups = (
        [temperature.round().long() == value for value in temperature.round().long().unique()]
        if temperature is not None
        else [torch.ones(innovation.shape[0], dtype=torch.bool, device=innovation.device)]
    )
    mean_losses = []
    covariance_losses = []
    covariance_spectral_losses = []
    autocorrelation_losses = []
    for group in closure_groups:
        group_innovation = innovation[group]
        group_mask = innovation_mask[group]
        valid_innovation = group_innovation[group_mask]
        if valid_innovation.shape[0] > 1:
            innovation_mean = valid_innovation.mean(dim=0)
            mean_losses.append(innovation_mean.square().mean())
            centered = valid_innovation - innovation_mean
            covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
            identity = torch.eye(
                covariance.shape[0], device=covariance.device, dtype=covariance.dtype
            )
            covariance_losses.append((covariance - identity).square().mean())
            if valid_innovation.shape[0] > covariance.shape[0] + 1:
                covariance_eigenvalues = torch.linalg.eigvalsh(
                    0.5 * (covariance + covariance.T) + 1.0e-5 * identity
                ).clamp(min=1.0e-5)
                covariance_spectral_losses.append(
                    0.5
                    * (
                        covariance_eigenvalues
                        - covariance_eigenvalues.log()
                        - 1.0
                    ).mean()
                )
        for lag in innovation_lags:
            if lag <= 0 or lag >= group_innovation.shape[1]:
                continue
            lag_mask = group_mask[:, lag:] & group_mask[:, :-lag]
            if not lag_mask.any():
                continue
            lagged = group_innovation[:, lag:][lag_mask]
            previous = group_innovation[:, :-lag][lag_mask]
            lagged = lagged - lagged.mean(dim=0, keepdim=True)
            previous = previous - previous.mean(dim=0, keepdim=True)
            lagged_covariance = lagged.T @ previous / max(1, lagged.shape[0] - 1)
            lagged_scale = lagged.std(dim=0, unbiased=False).clamp(min=1.0e-4)
            previous_scale = previous.std(dim=0, unbiased=False).clamp(min=1.0e-4)
            lagged_correlation = lagged_covariance / (
                lagged_scale[:, None] * previous_scale[None, :]
            )
            diagonal_correlation = torch.diagonal(lagged_correlation)
            off_diagonal_correlation = lagged_correlation - torch.diag_embed(
                diagonal_correlation
            )
            off_diagonal_count = max(
                lagged_correlation.shape[0]
                * (lagged_correlation.shape[0] - 1),
                1,
            )
            autocorrelation_losses.append(
                diagonal_correlation.square().mean()
                + off_diagonal_correlation.square().sum() / off_diagonal_count
            )
    innovation_mean_loss = (
        torch.stack(mean_losses).mean() if mean_losses else samples.new_tensor(0.0)
    )
    innovation_covariance_loss = (
        torch.stack(covariance_losses).mean()
        if covariance_losses
        else samples.new_tensor(0.0)
    )
    innovation_autocorrelation_loss = (
        torch.stack(autocorrelation_losses).mean()
        if autocorrelation_losses
        else samples.new_tensor(0.0)
    )
    innovation_covariance_spectral_loss = (
        torch.stack(covariance_spectral_losses).mean()
        if covariance_spectral_losses
        else samples.new_tensor(0.0)
    )
    innovation_covariance_worst_loss = (
        torch.stack(covariance_losses).max()
        if covariance_losses
        else samples.new_tensor(0.0)
    )
    innovation_autocorrelation_worst_loss = (
        torch.stack(autocorrelation_losses).max()
        if autocorrelation_losses
        else samples.new_tensor(0.0)
    )

    if bool(outputs.get("tmjepa_memory_enabled", samples.new_zeros(1)).any()):
        filtered_error = (
            outputs["tmjepa_deterministic_filtered_prediction"] - target
        ).square().mean(dim=-1)
        reset_error = (
            outputs["tmjepa_deterministic_reset_prediction"] - target
        ).square().mean(dim=-1)
        memory_intervention_advantage = (reset_error - filtered_error).mean()
        memory_intervention_loss = F.relu(
            filtered_error - reset_error + memory_intervention_margin
        ).mean()
        memory_long_horizon_intervention_advantage = _row_weighted_mean(
            reset_error - filtered_error, long_horizon_weights
        )
        memory_long_horizon_intervention_loss = _row_weighted_mean(
            F.relu(filtered_error - reset_error + memory_intervention_margin),
            long_horizon_weights,
        )
        relative_advantage = (reset_error - filtered_error) / reset_error.clamp(
            min=1.0e-8
        )
        memory_relative_intervention_advantage = relative_advantage.mean()
        memory_relative_intervention_loss = F.relu(
            float(memory_relative_intervention_margin) - relative_advantage
        ).mean()
        memory_long_horizon_relative_intervention_advantage = _row_weighted_mean(
            relative_advantage, long_horizon_weights
        )
        memory_long_horizon_relative_intervention_loss = _row_weighted_mean(
            F.relu(float(memory_relative_intervention_margin) - relative_advantage),
            long_horizon_weights,
        )
        # Identify the history-dependent component directly.  The reset rollout
        # is a stop-gradient Markov baseline; only the filtered-minus-reset
        # correction receives gradients from this target.
        reset_detached = outputs[
            "tmjepa_deterministic_reset_prediction"
        ].detach()
        target_correction = (target - reset_detached).detach()
        correction_scale = (
            (target - current).square().mean(dim=-1).sqrt().detach().clamp(min=0.05)
        )
        correction = (
            outputs["tmjepa_deterministic_filtered_prediction"] - reset_detached
        )
        memory_residual_loss_by_row = F.smooth_l1_loss(
            correction / correction_scale[:, None],
            target_correction / correction_scale[:, None],
            reduction="none",
        ).mean(dim=-1)
        memory_residual_loss = memory_residual_loss_by_row.mean()
        memory_long_horizon_residual_loss = _row_weighted_mean(
            memory_residual_loss_by_row, long_horizon_weights
        )
    else:
        memory_intervention_advantage = samples.new_tensor(0.0)
        memory_intervention_loss = samples.new_tensor(0.0)
        memory_relative_intervention_advantage = samples.new_tensor(0.0)
        memory_relative_intervention_loss = samples.new_tensor(0.0)
        memory_long_horizon_intervention_advantage = samples.new_tensor(0.0)
        memory_long_horizon_intervention_loss = samples.new_tensor(0.0)
        memory_long_horizon_relative_intervention_advantage = samples.new_tensor(0.0)
        memory_long_horizon_relative_intervention_loss = samples.new_tensor(0.0)
        memory_residual_loss = samples.new_tensor(0.0)
        memory_long_horizon_residual_loss = samples.new_tensor(0.0)

    auxiliary_mean = outputs["tmjepa_auxiliary_mean"]
    auxiliary_logvar = outputs["tmjepa_auxiliary_logvar"]
    auxiliary_kl_loss = -0.5 * (
        1.0 + auxiliary_logvar - auxiliary_mean.square() - auxiliary_logvar.exp()
    ).mean()
    auxiliary_transition_kl = outputs["tmjepa_auxiliary_transition_kl"]
    auxiliary_transition_kl_loss = (
        auxiliary_transition_kl[innovation_mask].mean()
        if innovation_mask.any()
        else samples.new_tensor(0.0)
    )

    learned_cv_variance_loss = samples.new_tensor(0.0)
    learned_cv_covariance_loss = samples.new_tensor(0.0)
    vamp_loss = samples.new_tensor(0.0)
    vamp2_score = samples.new_tensor(0.0)
    learned_cv = outputs.get("tmjepa_learned_cv")
    if learned_cv is not None:
        mask = outputs["tmjepa_context_mask"]
        group_index = outputs.get("context_group_index")
        if group_index is not None:
            rows = torch.stack(
                [
                    torch.nonzero(group_index == group, as_tuple=False).flatten()[0]
                    for group in group_index.unique(sorted=True)
                ]
            )
            learned_cv = learned_cv[rows]
            mask = mask[rows]
        (
            learned_cv_variance_loss,
            learned_cv_covariance_loss,
            vamp_loss,
            vamp2_score,
        ) = _trajectory_centered_learned_cv_terms(learned_cv, mask, vamp_lags)

    std = variance.sqrt()
    covered = (target - prediction).abs() <= 1.6448536269514722 * std
    standardized_abs_error = (target - prediction).abs() / std.clamp(min=1.0e-6)
    per_value_soft_coverage = torch.sigmoid(
        (1.6448536269514722 - standardized_abs_error) / 0.1
    )
    soft_coverage_90 = per_value_soft_coverage.mean()
    coverage_losses = []
    temperature_k = outputs.get("tmjepa_temperature_k")
    if horizon_frames is not None and temperature_k is not None:
        rounded_temperature = temperature_k.round().long()
        for horizon in horizon_frames.detach().unique(sorted=True):
            horizon_mask = horizon_frames == horizon
            for temperature in rounded_temperature.detach().unique(sorted=True):
                group_mask = horizon_mask & (rounded_temperature == temperature)
                if group_mask.any():
                    group_coverage = per_value_soft_coverage[group_mask].mean()
                    coverage_losses.append((group_coverage - 0.9).square())
    if not coverage_losses and horizon_frames is not None:
        for horizon in horizon_frames.detach().unique(sorted=True):
            group_mask = horizon_frames == horizon
            if group_mask.any():
                group_coverage = per_value_soft_coverage[group_mask].mean()
                coverage_losses.append((group_coverage - 0.9).square())
    coverage_calibration_loss = (
        torch.stack(coverage_losses).mean()
        if coverage_losses
        else (soft_coverage_90 - 0.9).square()
    )
    diffusion_eigenvalues = torch.linalg.eigvalsh(outputs["tmjepa_diffusion"])
    minimum_diffusion_eigenvalue = diffusion_eigenvalues.min()
    diffusion_condition_numbers = (
        diffusion_eigenvalues[:, -1]
        / diffusion_eigenvalues[:, 0].clamp(min=1.0e-12)
    )
    diffusion_condition_number = diffusion_condition_numbers.mean()
    condition_target = max(float(diffusion_condition_target), 1.0)
    diffusion_condition_loss = F.relu(
        torch.log(diffusion_condition_numbers / condition_target)
    ).square().mean()
    observation_covariance = outputs.get("tmjepa_observation_noise_covariance")
    observation_noise_variance = (
        torch.diagonal(observation_covariance, dim1=-2, dim2=-1).mean()
        if observation_covariance is not None
        else samples.new_tensor(0.0)
    )
    observation_trace = (
        torch.diagonal(observation_covariance, dim1=-2, dim2=-1).sum(dim=-1)
        if observation_covariance is not None
        else samples.new_zeros(samples.shape[0])
    )
    diffusion_trace = torch.diagonal(
        outputs["tmjepa_diffusion"], dim1=-2, dim2=-1
    ).sum(dim=-1)
    base_step_ps = outputs.get(
        "tmjepa_base_step_ps", diffusion_trace.new_ones(diffusion_trace.shape)
    )
    observation_to_diffusion_increment_variance_ratio = (
        observation_trace
        / (base_step_ps * diffusion_trace).clamp(min=1.0e-12)
    ).mean()
    observation_mode_variance = outputs.get(
        "tmjepa_observation_noise_mode_variance"
    )
    correlated_observation_variance_fraction = (
        observation_mode_variance.sum(dim=-1)
        .div(observation_trace.clamp(min=1.0e-12))
        .mean()
        if observation_mode_variance is not None
        else samples.new_tensor(0.0)
    )
    calibration_log_scale = outputs.get("tmjepa_endpoint_calibration_log_scale")
    mean_endpoint_calibration_log_scale = (
        calibration_log_scale.mean()
        if calibration_log_scale is not None
        else samples.new_tensor(0.0)
    )
    mean_endpoint_calibration_variance_multiplier = (
        torch.exp(2.0 * calibration_log_scale).mean()
        if calibration_log_scale is not None
        else samples.new_tensor(1.0)
    )
    coupling = outputs["tmjepa_coupling_basis"] * outputs[
        "tmjepa_coupling_rates"
    ][:, None, :]
    integrated_memory_trace = (
        coupling.square().sum(dim=1)
        / outputs["tmjepa_relaxation_rates"].clamp(min=1.0e-12)
    ).sum(dim=-1)
    diffusion_trace = torch.diagonal(
        outputs["tmjepa_diffusion"], dim1=-2, dim2=-1
    ).sum(dim=-1)
    memory_ratio = integrated_memory_trace / diffusion_trace.clamp(min=1.0e-12)
    integrated_memory_to_diffusion_ratio = memory_ratio.mean()
    memory_ratio_lower = min(float(memory_ratio_min), float(memory_ratio_max))
    memory_ratio_upper = max(float(memory_ratio_min), float(memory_ratio_max))
    memory_ratio_band_loss = (
        F.relu(memory_ratio_lower - memory_ratio).square()
        + F.relu(memory_ratio - memory_ratio_upper).square()
    ).mean()
    memory_ratio_target = max(float(memory_ratio_target), 1.0e-8)
    memory_log_ratio_loss = (
        memory_ratio.clamp(min=1.0e-12).log()
        - math.log(memory_ratio_target)
    ).square().mean()
    auxiliary_covariance = outputs.get("tmjepa_auxiliary_covariance")
    if auxiliary_covariance is None:
        auxiliary_covariance = torch.diag_embed(auxiliary_logvar.exp())
    bath_information_fraction = (
        1.0
        - torch.diagonal(
            auxiliary_covariance, dim1=-2, dim2=-1
        ).mean(dim=-1)
    ).mean()
    mean_relaxation_time_ps = outputs["tmjepa_relaxation_rates"].clamp(min=1.0e-12).reciprocal().mean()
    return {
        "energy_score": energy_score,
        "forecast_relative_mse_loss": forecast_relative_mse_loss,
        "persistence_energy_score": persistence_energy_score,
        "energy_skill": energy_skill,
        "transition_nll": transition_nll,
        "path_nll": path_nll,
        "equilibrium_score_loss": equilibrium_score_loss,
        "equilibrium_score_baseline": equilibrium_score_baseline,
        "equilibrium_score_skill": equilibrium_score_skill,
        "innovation_mean_loss": innovation_mean_loss,
        "innovation_covariance_loss": innovation_covariance_loss,
        "innovation_autocorrelation_loss": innovation_autocorrelation_loss,
        "innovation_covariance_spectral_loss": (
            innovation_covariance_spectral_loss
        ),
        "innovation_covariance_worst_loss": innovation_covariance_worst_loss,
        "innovation_autocorrelation_worst_loss": (
            innovation_autocorrelation_worst_loss
        ),
        "memory_intervention_loss": memory_intervention_loss,
        "memory_intervention_advantage": memory_intervention_advantage,
        "memory_intervention_mse_advantage": memory_intervention_advantage,
        "memory_relative_intervention_loss": memory_relative_intervention_loss,
        "memory_relative_intervention_advantage": (
            memory_relative_intervention_advantage
        ),
        "memory_ratio_band_loss": memory_ratio_band_loss,
        "memory_log_ratio_loss": memory_log_ratio_loss,
        "memory_long_horizon_intervention_loss": (
            memory_long_horizon_intervention_loss
        ),
        "memory_long_horizon_intervention_advantage": (
            memory_long_horizon_intervention_advantage
        ),
        "memory_long_horizon_relative_intervention_loss": (
            memory_long_horizon_relative_intervention_loss
        ),
        "memory_long_horizon_relative_intervention_advantage": (
            memory_long_horizon_relative_intervention_advantage
        ),
        "memory_residual_loss": memory_residual_loss,
        "memory_long_horizon_residual_loss": memory_long_horizon_residual_loss,
        "long_horizon_energy_score": long_horizon_energy_score,
        "long_horizon_forecast_relative_mse_loss": (
            long_horizon_forecast_relative_mse_loss
        ),
        "long_horizon_transition_nll": long_horizon_transition_nll,
        "long_horizon_path_nll": long_horizon_path_nll,
        "projected_force_regularization_loss": projected_force_regularization_loss,
        "auxiliary_kl_loss": auxiliary_kl_loss,
        "auxiliary_transition_kl_loss": auxiliary_transition_kl_loss,
        "learned_cv_variance_loss": learned_cv_variance_loss,
        "learned_cv_covariance_loss": learned_cv_covariance_loss,
        "vamp_loss": vamp_loss,
        "vamp2_score": vamp2_score,
        "mean_coverage_90": covered.to(samples.dtype).mean(),
        "soft_coverage_90": soft_coverage_90,
        "coverage_calibration_loss": coverage_calibration_loss,
        "minimum_diffusion_eigenvalue": minimum_diffusion_eigenvalue,
        "diffusion_condition_number": diffusion_condition_number,
        "diffusion_condition_loss": diffusion_condition_loss,
        "mean_observation_noise_variance": observation_noise_variance,
        "observation_to_diffusion_increment_variance_ratio": (
            observation_to_diffusion_increment_variance_ratio
        ),
        "correlated_observation_variance_fraction": (
            correlated_observation_variance_fraction
        ),
        "mean_endpoint_calibration_log_scale": mean_endpoint_calibration_log_scale,
        "mean_endpoint_calibration_variance_multiplier": mean_endpoint_calibration_variance_multiplier,
        "integrated_memory_to_diffusion_ratio": integrated_memory_to_diffusion_ratio,
        "bath_information_fraction": bath_information_fraction,
        "mean_relaxation_time_ps": mean_relaxation_time_ps,
    }


def v51_sparse_motion_terms(
    outputs: dict,
    ranking_margin: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Predictive, anti-collapse, and causal-control terms for V5.1."""
    if "v51_predicted_delta" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {
            key: zero
            for key in (
                "delta_loss", "persistence_loss", "delta_skill", "nll_loss",
                "normalized_delta_loss", "online_delta_loss",
                "online_normalized_delta_loss", "displacement_magnitude_loss",
                "current_alignment_loss", "observable_loss", "oracle_observable_loss",
                "online_oracle_observable_loss", "observable_zero_motion_loss",
                "observable_skill", "oracle_observable_skill",
                "online_oracle_observable_skill", "horizon_ranking_loss",
                "correct_vs_best_wrong_skill", "history_ranking_loss", "history_advantage",
                "variance_loss", "covariance_loss", "load_balance_loss",
                "active_load_balance_loss", "mean_active_modes", "target_displacement",
                "predicted_displacement", "displacement_ratio", "causal_next_loss",
                "causal_next_skill", "horizon_consistency_loss", "horizon_consistency_skill",
                "history_truncation_loss", "history_truncation_advantage",
                "physical_horizon_consistency_loss", "physical_horizon_consistency_skill",
                "semigroup_loss", "semigroup_skill",
                "horizon_calibration_loss", "mode_diversity_loss",
                "mode_orthogonality_loss",
            )
        }

    prediction = outputs["v51_predicted_delta"]
    target = outputs["v51_target_delta"].detach()
    squared_error = (prediction - target).square()
    delta_loss = F.smooth_l1_loss(prediction, target)
    prediction_mse = squared_error.mean()
    persistence_loss = target.square().mean()
    per_sample_target_mse = target.square().mean(dim=-1).detach().clamp(min=1.0e-6)
    normalized_delta_loss = (
        squared_error.mean(dim=-1) / per_sample_target_mse
    ).mean()
    # Grouped training repeats one context at every horizon. Normalize within
    # each horizon before averaging so long-horizon displacement does not
    # erase the short-horizon calibration signal.
    horizon_calibration_terms = []
    horizon_frames = outputs.get("horizon_frames")
    if horizon_frames is not None:
        for horizon in horizon_frames.unique(sorted=True):
            mask = horizon_frames == horizon
            if mask.any():
                horizon_calibration_terms.append(
                    (squared_error[mask].mean(dim=-1) / per_sample_target_mse[mask]).mean()
                )
    horizon_calibration_loss = (
        torch.stack(horizon_calibration_terms).mean()
        if horizon_calibration_terms else prediction.new_tensor(0.0)
    )
    online_target = outputs["v51_online_target_delta"].detach()
    online_squared_error = (prediction - online_target).square()
    online_delta_loss = F.smooth_l1_loss(prediction, online_target)
    online_target_mse = online_target.square().mean(dim=-1).detach().clamp(min=1.0e-6)
    online_normalized_delta_loss = (
        online_squared_error.mean(dim=-1) / online_target_mse
    ).mean()
    target_norm = target.square().mean(dim=-1).sqrt().detach().clamp(min=1.0e-6)
    predicted_norm = prediction.square().mean(dim=-1).sqrt().clamp(min=1.0e-6)
    displacement_magnitude_loss = (
        predicted_norm.log() - target_norm.log()
    ).square().mean()
    log_sigma = outputs["v51_prediction_log_sigma"]
    nll_loss = 0.5 * (
        squared_error * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
    ).mean()
    current_alignment_loss = F.mse_loss(
        outputs["v51_online_current_feature"],
        outputs["v51_teacher_current_feature"].detach(),
    )

    observable_target = outputs["v51_observable_target"].detach()
    observable_prediction = outputs["v51_observable_prediction"]
    observable_loss = F.smooth_l1_loss(observable_prediction, observable_target)
    oracle_observable_loss = F.smooth_l1_loss(
        outputs["v51_oracle_observable"], observable_target
    )
    online_oracle_observable_loss = F.smooth_l1_loss(
        outputs["v51_online_oracle_observable"], observable_target
    )
    observable_zero_motion_loss = F.smooth_l1_loss(
        outputs["v51_no_rollout_observable"], observable_target
    )

    horizon_ranking_loss = prediction.new_tensor(0.0)
    correct_vs_best_wrong_skill = prediction.new_tensor(0.0)
    if "v51_fixed_predictions" in outputs:
        fixed = outputs["v51_fixed_predictions"]
        fixed_observable = outputs["v51_fixed_observable_predictions"]
        fixed_horizons = outputs["v51_fixed_horizons"]
        feature_errors = (
            fixed - outputs["v51_target_feature"].detach().unsqueeze(1)
        ).square().mean(dim=-1)
        correct_feature_error = (
            outputs["v51_predicted_feature"] - outputs["v51_target_feature"].detach()
        ).square().mean(dim=-1)
        observable_errors = (
            fixed_observable - observable_target.unsqueeze(1)
        ).square().mean(dim=-1)
        correct_observable_error = (
            observable_prediction - observable_target
        ).square().mean(dim=-1)
        wrong = fixed_horizons.unsqueeze(0) != outputs["horizon_frames"].unsqueeze(1)
        scale = target.square().mean(dim=-1).detach().clamp(min=1.0e-5)
        observable_scale = (
            outputs["v51_no_rollout_observable"] - observable_target
        ).square().mean(dim=-1).detach().clamp(min=1.0e-5)
        ranking = F.relu(
            correct_feature_error.unsqueeze(1) - feature_errors
            + ranking_margin * scale.unsqueeze(1)
        )
        observable_ranking = F.relu(
            correct_observable_error.unsqueeze(1) - observable_errors
            + ranking_margin * observable_scale.unsqueeze(1)
        )
        if wrong.any():
            horizon_ranking_loss = ranking[wrong].mean() + observable_ranking[wrong].mean()
            best_wrong = feature_errors.masked_fill(~wrong, float("inf")).min(dim=1).values
            finite = torch.isfinite(best_wrong)
            if finite.any():
                correct_vs_best_wrong_skill = (
                    1.0
                    - correct_feature_error[finite]
                    / best_wrong[finite].clamp(min=1.0e-8)
                ).mean()

    history_ranking_loss = prediction.new_tensor(0.0)
    history_advantage = prediction.new_tensor(0.0)
    if "v51_wrong_history_prediction" in outputs:
        wrong_feature_error = (
            outputs["v51_wrong_history_prediction"]
            - outputs["v51_target_feature"].detach()
        ).square().mean(dim=-1)
        correct_feature_error = (
            outputs["v51_predicted_feature"] - outputs["v51_target_feature"].detach()
        ).square().mean(dim=-1)
        wrong_observable_error = (
            outputs["v51_wrong_history_observable_prediction"] - observable_target
        ).square().mean(dim=-1)
        correct_observable_error = (
            observable_prediction - observable_target
        ).square().mean(dim=-1)
        feature_scale = target.square().mean(dim=-1).detach().clamp(min=1.0e-5)
        observable_scale = (
            outputs["v51_no_rollout_observable"] - observable_target
        ).square().mean(dim=-1).detach().clamp(min=1.0e-5)
        history_ranking_loss = (
            F.relu(
                correct_feature_error - wrong_feature_error
                + ranking_margin * feature_scale
            ).mean()
            + F.relu(
                correct_observable_error - wrong_observable_error
                + ranking_margin * observable_scale
            ).mean()
        )
        history_advantage = (
            1.0
            - correct_feature_error
            / wrong_feature_error.clamp(min=1.0e-8)
        ).mean()

    history_truncation_loss = prediction.new_tensor(0.0)
    history_truncation_advantage = prediction.new_tensor(0.0)
    if "v51_short_history_prediction" in outputs:
        short_feature_error = (
            outputs["v51_short_history_prediction"]
            - outputs["v51_target_feature"].detach()
        ).square().mean(dim=-1)
        correct_feature_error = (
            outputs["v51_predicted_feature"] - outputs["v51_target_feature"].detach()
        ).square().mean(dim=-1)
        short_observable_error = (
            outputs["v51_short_history_observable_prediction"] - observable_target
        ).square().mean(dim=-1)
        correct_observable_error = (
            observable_prediction - observable_target
        ).square().mean(dim=-1)
        feature_scale = target.square().mean(dim=-1).detach().clamp(min=1.0e-5)
        observable_scale = (
            outputs["v51_no_rollout_observable"] - observable_target
        ).square().mean(dim=-1).detach().clamp(min=1.0e-5)
        history_truncation_loss = (
            F.relu(
                correct_feature_error - short_feature_error
                + ranking_margin * feature_scale
            ).mean()
            + F.relu(
                correct_observable_error - short_observable_error
                + ranking_margin * observable_scale
            ).mean()
        )
        history_truncation_advantage = (
            1.0
            - correct_feature_error / short_feature_error.clamp(min=1.0e-8)
        ).mean()

    causal_next_loss = prediction.new_tensor(0.0)
    causal_next_skill = prediction.new_tensor(0.0)
    if "v51_causal_next_prediction" in outputs:
        causal_prediction = outputs["v51_causal_next_prediction"]
        causal_target = outputs["v51_causal_next_target"]
        if causal_prediction.numel():
            causal_error = (causal_prediction - causal_target).square().mean()
            causal_persistence = causal_target.square().mean().clamp(min=1.0e-8)
            causal_next_loss = F.smooth_l1_loss(causal_prediction, causal_target)
            causal_next_skill = 1.0 - causal_error / causal_persistence

    # Grouped paired samples contain the same observed context at several
    # horizons. Match differences between horizons so the rollout represents a
    # trajectory, rather than independently fitting an average endpoint.
    horizon_consistency_loss = prediction.new_tensor(0.0)
    horizon_consistency_skill = prediction.new_tensor(0.0)
    physical_horizon_consistency_loss = prediction.new_tensor(0.0)
    physical_horizon_consistency_skill = prediction.new_tensor(0.0)
    group_index = outputs.get("context_group_index")
    if group_index is not None:
        pair_predictions = []
        pair_targets = []
        physical_pair_predictions = []
        physical_pair_targets = []
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            if members.numel() < 2:
                continue
            pair_mask = torch.triu(
                torch.ones((members.numel(), members.numel()), dtype=torch.bool, device=prediction.device),
                diagonal=1,
            )
            pred_diff = prediction[members][:, None, :] - prediction[members][None, :, :]
            target_diff = target[members][:, None, :] - target[members][None, :, :]
            pair_predictions.append(pred_diff[pair_mask])
            pair_targets.append(target_diff[pair_mask])
            physical_prediction = outputs["v51_observable_prediction"][members]
            physical_target = observable_target[members]
            physical_diff = physical_prediction[:, None, :] - physical_prediction[None, :, :]
            physical_target_diff = physical_target[:, None, :] - physical_target[None, :, :]
            physical_pair_predictions.append(physical_diff[pair_mask])
            physical_pair_targets.append(physical_target_diff[pair_mask])
        if pair_predictions:
            pair_prediction = torch.cat(pair_predictions, dim=0)
            pair_target = torch.cat(pair_targets, dim=0)
            horizon_consistency_loss = F.smooth_l1_loss(pair_prediction, pair_target)
            horizon_consistency_skill = 1.0 - (
                (pair_prediction - pair_target).square().mean()
                / pair_target.square().mean().clamp(min=1.0e-8)
            )
            physical_pair_prediction = torch.cat(physical_pair_predictions, dim=0)
            physical_pair_target = torch.cat(physical_pair_targets, dim=0)
            # Physical channels have very different units and some horizon
            # pairs are nearly stationary. Robust channel normalization keeps
            # the observable constraint informative without overpowering the
            # latent trajectory loss.
            physical_scale = physical_pair_target.detach().abs().median(dim=0).values
            # Observable differences can be close to zero for stationary
            # channels. A physical floor prevents those channels from
            # dominating the shared latent trajectory objective.
            physical_scale = physical_scale.clamp(min=5.0e-2)
            normalized_physical_prediction = physical_pair_prediction / physical_scale
            normalized_physical_target = physical_pair_target / physical_scale
            pair_weight = physical_pair_target.detach().square().mean(dim=-1).sqrt()
            pair_weight = pair_weight / pair_weight.detach().median().clamp(min=1.0e-3)
            pair_weight = pair_weight.clamp(min=0.25, max=4.0)
            physical_pair_loss = F.smooth_l1_loss(
                normalized_physical_prediction,
                normalized_physical_target,
                reduction="none",
            ).mean(dim=-1)
            physical_horizon_consistency_loss = (
                physical_pair_loss * pair_weight
            ).mean()
            physical_horizon_consistency_skill = 1.0 - (
                (physical_pair_prediction - physical_pair_target).square().mean()
                / physical_pair_target.square().mean().clamp(min=1.0e-8)
            )

    semigroup_loss = prediction.new_tensor(0.0)
    semigroup_skill = prediction.new_tensor(0.0)
    if "v51_semigroup_prediction" in outputs:
        semigroup_prediction = outputs["v51_semigroup_prediction"]
        semigroup_target = outputs["v51_semigroup_target"]
        semigroup_loss = F.smooth_l1_loss(semigroup_prediction, semigroup_target)
        semigroup_skill = 1.0 - (
            (semigroup_prediction - semigroup_target).square().mean()
            / semigroup_target.square().mean().clamp(min=1.0e-8)
        )

    tokens = outputs["v51_dynamic_tokens"][outputs["v51_dynamic_token_mask"]]
    if tokens.shape[0] > 1:
        std = torch.sqrt(tokens.var(dim=0, unbiased=False) + 1.0e-4)
        variance_loss = F.relu(1.0 - std).mean()
        normalized = (tokens - tokens.mean(dim=0)) / std.clamp(min=1.0e-4)
        covariance = normalized.T @ normalized / max(1, normalized.shape[0] - 1)
        covariance = covariance - torch.diag_embed(torch.diagonal(covariance))
        covariance_loss = covariance.square().sum() / tokens.shape[1]
    else:
        variance_loss = prediction.new_tensor(0.0)
        covariance_loss = prediction.new_tensor(0.0)

    routing = outputs["v51_router_probabilities"]
    group_index = outputs.get("context_group_index")
    if group_index is not None:
        representatives = []
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            if members.numel():
                representatives.append(members[0])
        representative_index = torch.stack(representatives) if representatives else None
    else:
        representative_index = None
    context_routing = (
        routing.index_select(0, representative_index)
        if representative_index is not None else routing
    )
    mean_routing = context_routing.mean(dim=0)
    load_balance_loss = (
        mean_routing - 1.0 / routing.shape[-1]
    ).square().mean() * routing.shape[-1]
    active_mask = outputs["v51_active_mode_mask"].to(routing.dtype)
    soft_active_mask = outputs.get("v51_soft_active_mode_mask", active_mask)
    active_modes = active_mask.index_select(0, representative_index).sum(dim=-1).float().mean() if representative_index is not None else active_mask.sum(dim=-1).float().mean()
    target_active_frequency = active_modes.detach() / float(routing.shape[-1])
    context_entropy = -(context_routing.clamp(min=1.0e-8) * context_routing.clamp(min=1.0e-8).log()).sum(dim=-1)
    target_entropy = math.log(max(float(active_modes.detach()), 1.0))
    mode_diversity_loss = (
        (mean_routing - 1.0 / routing.shape[-1]).square().mean() * routing.shape[-1]
        + F.relu(context_entropy - target_entropy).mean() / math.log(float(routing.shape[-1]))
    )
    router_logits = outputs.get("v51_router_logits")
    if router_logits is not None:
        if representative_index is not None:
            router_logits = router_logits.index_select(0, representative_index)
        top = router_logits.topk(int(active_modes.detach()), dim=-1).values.mean(dim=-1)
        rest = router_logits.topk(
            router_logits.shape[-1] - int(active_modes.detach()), dim=-1
        ).values.mean(dim=-1)
        # Entropy minimization is flat at a uniform router. A top-vs-rest
        # margin supplies a non-zero gradient there while global balance keeps
        # different contexts from selecting one mode forever.
        mode_diversity_loss = mode_diversity_loss + F.relu(0.05 - (top - rest)).mean()
    active_load_balance_loss = (
        soft_active_mask.mean(dim=0) - target_active_frequency
    ).square().mean() * routing.shape[-1]
    mode_seed = outputs.get("v51_mode_seed")
    if mode_seed is not None and mode_seed.shape[0] > 1:
        normalized_seed = F.normalize(mode_seed, dim=-1)
        gram = normalized_seed @ normalized_seed.T
        gram = gram - torch.diag_embed(torch.diagonal(gram))
        mode_orthogonality_loss = gram.square().mean()
    else:
        mode_orthogonality_loss = prediction.new_tensor(0.0)

    return {
        "delta_loss": delta_loss,
        "persistence_loss": persistence_loss,
        "delta_skill": 1.0 - prediction_mse / persistence_loss.clamp(min=1.0e-8),
        "normalized_delta_loss": normalized_delta_loss,
        "horizon_calibration_loss": horizon_calibration_loss,
        "online_delta_loss": online_delta_loss,
        "online_normalized_delta_loss": online_normalized_delta_loss,
        "displacement_magnitude_loss": displacement_magnitude_loss,
        "nll_loss": nll_loss,
        "current_alignment_loss": current_alignment_loss,
        "observable_loss": observable_loss,
        "oracle_observable_loss": oracle_observable_loss,
        "online_oracle_observable_loss": online_oracle_observable_loss,
        "observable_zero_motion_loss": observable_zero_motion_loss,
        "observable_skill": 1.0
        - observable_loss / observable_zero_motion_loss.clamp(min=1.0e-8),
        "oracle_observable_skill": 1.0
        - oracle_observable_loss / observable_zero_motion_loss.clamp(min=1.0e-8),
        "online_oracle_observable_skill": 1.0
        - online_oracle_observable_loss
        / observable_zero_motion_loss.clamp(min=1.0e-8),
        "horizon_ranking_loss": horizon_ranking_loss,
        "correct_vs_best_wrong_skill": correct_vs_best_wrong_skill,
        "history_ranking_loss": history_ranking_loss,
        "history_advantage": history_advantage,
        "history_truncation_loss": history_truncation_loss,
        "history_truncation_advantage": history_truncation_advantage,
        "causal_next_loss": causal_next_loss,
        "causal_next_skill": causal_next_skill,
        "horizon_consistency_loss": horizon_consistency_loss,
        "horizon_consistency_skill": horizon_consistency_skill,
        "physical_horizon_consistency_loss": physical_horizon_consistency_loss,
        "physical_horizon_consistency_skill": physical_horizon_consistency_skill,
        "semigroup_loss": semigroup_loss,
        "semigroup_skill": semigroup_skill,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "load_balance_loss": load_balance_loss,
        "active_load_balance_loss": active_load_balance_loss,
        "mode_diversity_loss": mode_diversity_loss,
        "mode_orthogonality_loss": mode_orthogonality_loss,
        "mean_active_modes": active_modes,
        "target_displacement": target.square().mean().sqrt(),
        "predicted_displacement": prediction.square().mean().sqrt(),
        "displacement_ratio": predicted_norm.mean() / target_norm.mean().clamp(min=1.0e-8),
    }


def local_observable_prediction_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "local_observable_prediction" not in outputs:
        return outputs["delta_target"].new_tensor(0.0)
    mask = outputs["local_observable_mask"]
    if not mask.any():
        return outputs["local_observable_prediction"].new_tensor(0.0)
    return _latent_loss(
        outputs["local_observable_prediction"][mask],
        outputs["local_observable_target"][mask],
        loss_type,
    )


def closed_local_state_terms(
    outputs: dict, margin: float = 0.002, difference_min_gap: int = 0,
    robust_beta: float = 0.5, student_t_df: float = 3.0,
) -> dict[str, torch.Tensor]:
    if "closed_local_prediction" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {
            "state_loss": zero, "persistence_loss": zero, "skill": zero,
            "ranking_loss": zero, "wrong_horizon_loss": zero,
            "difference_loss": zero, "target_difference_mse": zero,
            "prediction_difference_mse": zero, "robust_state_loss": zero,
            "robust_difference_loss": zero, "student_t_nll": zero,
            "base_state_loss": zero, "base_robust_loss": zero, "base_skill": zero,
        }
    prediction = outputs["closed_local_prediction"]
    target = outputs["future_closed_local_state"]
    current = outputs["context_closed_local_state"]
    mask = outputs["closed_local_mask"]
    scale = outputs.get("closed_local_state_scale")
    if scale is None:
        scale = torch.ones(prediction.shape[-1], device=prediction.device, dtype=prediction.dtype)
    scale = scale.to(device=prediction.device, dtype=prediction.dtype)
    if scale.shape != (prediction.shape[-1],):
        raise ValueError("closed local state scale must contain one value per channel")
    if not mask.any():
        zero = prediction.new_tensor(0.0)
        return {
            "state_loss": zero, "persistence_loss": zero, "skill": zero,
            "ranking_loss": zero, "wrong_horizon_loss": zero,
            "difference_loss": zero, "target_difference_mse": zero,
            "prediction_difference_mse": zero, "robust_state_loss": zero,
            "robust_difference_loss": zero, "student_t_nll": zero,
            "base_state_loss": zero, "base_robust_loss": zero, "base_skill": zero,
        }
    normalized_error = (prediction[mask] - target[mask]) / scale
    state_loss = normalized_error.square().mean()
    robust_state_loss = F.smooth_l1_loss(
        normalized_error, torch.zeros_like(normalized_error), beta=robust_beta
    )
    persistence_loss = ((current[mask] - target[mask]) / scale).square().mean()
    student_t_nll = prediction.new_tensor(0.0)
    if "closed_local_log_sigma" in outputs:
        if student_t_df <= 2.0:
            raise ValueError("closed local Student-t degrees of freedom must exceed 2")
        log_sigma = outputs["closed_local_log_sigma"][mask]
        standardized = normalized_error / log_sigma.exp().clamp(min=1.0e-6)
        normalizer = (
            math.lgamma(0.5 * student_t_df)
            - math.lgamma(0.5 * (student_t_df + 1.0))
            + 0.5 * math.log(student_t_df * math.pi)
        )
        student_t_nll = (
            log_sigma + normalizer
            + 0.5 * (student_t_df + 1.0)
            * torch.log1p(standardized.square() / student_t_df)
        ).mean()
    ranking_terms = []
    wrong_terms = []
    difference_terms = []
    target_difference_terms = []
    prediction_difference_terms = []
    robust_difference_terms = []
    group_index = outputs.get("context_group_index")
    horizon_frames = outputs.get("horizon_frames")
    if group_index is not None:
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            if members.numel() < 2:
                continue
            if horizon_frames is not None:
                pair_positions = torch.triu_indices(
                    members.numel(), members.numel(), offset=1, device=members.device
                )
                left_indices = members[pair_positions[0]]
                right_indices = members[pair_positions[1]]
                if difference_min_gap > 0:
                    keep = (
                        horizon_frames[right_indices] - horizon_frames[left_indices]
                    ).abs() >= difference_min_gap
                    left_indices = left_indices[keep]
                    right_indices = right_indices[keep]
                pair_valid = mask[left_indices] & mask[right_indices]
                if pair_valid.any():
                    prediction_difference = (
                        prediction[right_indices] - prediction[left_indices]
                    ) / scale
                    target_difference = (target[right_indices] - target[left_indices]) / scale
                    prediction_difference = prediction_difference[pair_valid]
                    target_difference = target_difference[pair_valid]
                    difference_terms.append(F.mse_loss(prediction_difference, target_difference))
                    target_energy = target_difference.square().mean().sqrt().detach().clamp(min=0.5)
                    robust_difference_terms.append(
                        F.smooth_l1_loss(
                            (prediction_difference - target_difference) / target_energy,
                            torch.zeros_like(target_difference), beta=robust_beta,
                        )
                    )
                    target_difference_terms.append(target_difference.square().mean())
                    prediction_difference_terms.append(prediction_difference.square().mean())
            for target_index in members:
                valid = mask[target_index]
                if not valid.any():
                    continue
                for prediction_index in members:
                    if prediction_index == target_index:
                        continue
                    pair_valid = valid & mask[prediction_index]
                    if not pair_valid.any():
                        continue
                    pair_target = target[target_index][pair_valid]
                    correct_error = (
                        (prediction[target_index][pair_valid] - pair_target) / scale
                    ).square().mean()
                    wrong_error = (
                        (prediction[prediction_index][pair_valid] - pair_target) / scale
                    ).square().mean()
                    wrong_terms.append(wrong_error)
                    ranking_terms.append(torch.relu(correct_error - wrong_error + margin))
    ranking_loss = torch.stack(ranking_terms).mean() if ranking_terms else prediction.new_tensor(0.0)
    wrong_horizon_loss = torch.stack(wrong_terms).mean() if wrong_terms else prediction.new_tensor(0.0)
    difference_loss = torch.stack(difference_terms).mean() if difference_terms else prediction.new_tensor(0.0)
    target_difference_mse = (
        torch.stack(target_difference_terms).mean() if target_difference_terms else prediction.new_tensor(0.0)
    )
    prediction_difference_mse = (
        torch.stack(prediction_difference_terms).mean()
        if prediction_difference_terms else prediction.new_tensor(0.0)
    )
    robust_difference_loss = (
        torch.stack(robust_difference_terms).mean()
        if robust_difference_terms else prediction.new_tensor(0.0)
    )
    base_state_loss = prediction.new_tensor(0.0)
    base_robust_loss = prediction.new_tensor(0.0)
    base_skill = prediction.new_tensor(0.0)
    if "closed_local_base_prediction" in outputs:
        base_mask = outputs["closed_local_base_mask"]
        base_error = (
            outputs["closed_local_base_prediction"][base_mask]
            - outputs["closed_local_base_target"][base_mask]
        ) / scale
        base_persistence = (
            outputs["closed_local_base_current"][base_mask]
            - outputs["closed_local_base_target"][base_mask]
        ) / scale
        base_state_loss = base_error.square().mean()
        base_robust_loss = F.smooth_l1_loss(
            base_error, torch.zeros_like(base_error), beta=robust_beta
        )
        base_skill = 1.0 - base_state_loss / base_persistence.square().mean().clamp(min=1.0e-8)
    return {
        "state_loss": state_loss,
        "persistence_loss": persistence_loss,
        "skill": 1.0 - state_loss / persistence_loss.clamp(min=1.0e-8),
        "ranking_loss": ranking_loss,
        "wrong_horizon_loss": wrong_horizon_loss,
        "difference_loss": difference_loss,
        "target_difference_mse": target_difference_mse,
        "prediction_difference_mse": prediction_difference_mse,
        "robust_state_loss": robust_state_loss,
        "robust_difference_loss": robust_difference_loss,
        "student_t_nll": student_t_nll,
        "base_state_loss": base_state_loss,
        "base_robust_loss": base_robust_loss,
        "base_skill": base_skill,
    }


def gaussian_transition_nll(outputs: dict, loss_type: str = "mse") -> torch.Tensor:
    """Diagonal Gaussian NLL for the residual transition."""
    if "gaussian_delta_prediction" not in outputs:
        return outputs["predictions"].new_tensor(0.0) if "predictions" in outputs else outputs["delta_target"].new_tensor(0.0)
    residual = outputs["delta_target"] - outputs["gaussian_delta_prediction"]
    log_sigma = outputs["gaussian_log_sigma"]
    return 0.5 * (residual.square() * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma).mean()


def observable_prediction_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "observable_prediction" not in outputs:
        return outputs["delta_target"].new_tensor(0.0)
    return _latent_loss(outputs["observable_prediction"], outputs["observable_target"], loss_type)


def offline_forecast_terms(outputs: dict, loss_type: str = "smooth_l1") -> dict[str, torch.Tensor]:
    if "offline_delta_prediction" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {"delta_loss": zero, "state_loss": zero, "skill": zero}
    prediction = outputs["offline_delta_prediction"]
    target = outputs["offline_delta_target"]
    delta_loss = _latent_loss(prediction, target, loss_type)
    state_loss = _latent_loss(
        outputs["offline_state_prediction"], outputs["offline_future_endpoint"], loss_type
    )
    persistence = _latent_loss(torch.zeros_like(target), target, loss_type)
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
        "skill": 1.0 - delta_loss / persistence.clamp(min=1.0e-8),
    }
    MetastableStateHead,
