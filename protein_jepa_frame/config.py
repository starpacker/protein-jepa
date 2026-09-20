from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest_path: str | None = None
    batch_size: int = 4
    num_workers: int = 0
    num_frames: int = 8
    frame_stride: int = 1
    window_stride: int = 1
    max_residues: int | None = None
    coordinate_scale: float = 1.0
    synthetic_size: int = 128
    synthetic_residues: int = 64
    msm_labels_path: str | None = None
    sampling_mode: str = "clip_uniform"
    dataset_weights: dict[str, float] = field(default_factory=dict)
    training_mode: str = "clip"
    paired_horizons_frames: tuple[int, ...] = (8, 16, 24, 48)
    paired_nonoverlap_only: bool = False
    paired_all_horizons: bool = False
    paired_history_mode: str = "uniform"
    paired_recent_frames: int = 0
    paired_slow_stride: int = 0
    load_forces: bool = False


@dataclass
class MaskConfig:
    mode: str = "block"
    num_blocks: int = 2
    temporal_scale: tuple[float, float] = (0.25, 0.5)
    residue_scale: tuple[float, float] = (0.15, 0.4)
    min_context_ratio: float = 0.2
    future_context_ratio: float = 0.5
    future_gap_frames: int = 0


@dataclass
class ModelConfig:
    latent_dim: int = 128
    encoder_depth: int = 4
    temporal_depth: int = 0
    predictor_depth: int = 4
    state_predictor_depth: int = 2
    state_prediction_lags: tuple[int, ...] = ()
    frame_pooling: str = "mean"
    frame_pool_queries: int = 4
    lag_predictor_depth: int = 2
    lag_predictor_dropout: float | None = None
    lag_conditioning: str = "none"
    lag_prediction_mode: str = "direct"
    time_conditioning: str = "none"
    thermodynamic_conditioning: bool = False
    orientation_features: bool = False
    temporal_causal: bool = False
    temporal_causal_window: int = 0
    temporal_position_encoding: str = "learned"
    temporal_delta_features: bool = False
    motion_depth: int = 0
    motion_stopgrad_input: bool = False
    transition_source: str = "state"
    transition_thermodynamic_conditioning: bool = False
    linear_dynamics_lags: tuple[int, ...] = ()
    physchem_prediction_lags: tuple[int, ...] = ()
    physchem_predictor_depth: int = 2
    physchem_contact_cutoff: float = 8.0
    kinetic_readout_dim: int = 0
    kinetic_readout_depth: int = 2
    kinetic_readout_stopgrad: bool = False
    num_heads: int = 4
    num_rbf: int = 16
    dropout: float = 0.1
    max_frames: int = 256
    max_residues: int = 4096
    feature_norm: bool = True
    paired_forecast: bool = False
    paired_predictor_depth: int = 3
    learned_state_training: bool = False
    slow_state_enabled: bool = False
    slow_state_dim: int = 32
    slow_predictor_depth: int = 2
    slow_vamp_lags: tuple[int, ...] = (1, 2, 4, 8)
    slow_vamp_burn_in_frames: int = 0
    ls2_reversible_enabled: bool = False
    ls2_num_states: int = 12
    ls2_base_step_ps: float = 1000.0
    ls2_assignment_temperature: float = 0.5
    ls2_stationary_floor: float = 1.0e-4
    ls2_rate_scale: float = 0.1
    dual_temporal_towers: bool = False
    offline_temporal_depth: int = 0
    causal_temporal_depth: int = 0
    probabilistic_transition: bool = False
    transition_log_sigma_min: float = -5.0
    transition_log_sigma_max: float = 1.5
    observable_head: bool = False
    observable_dim: int = 4
    causal_encoder_gradient_scale: float = 1.0
    paired_teacher_space_delta: bool = False
    paired_target_tower: str = "causal"
    freeze_context_encoder: bool = False
    freeze_temporal_encoder: bool = False
    # Keep the EMA teacher fixed when a downstream stage fits only a
    # world-model head on top of a pretrained representation.
    freeze_target_encoders: bool = False
    freeze_offline_temporal_encoder: bool = False
    history_conditioned_transition: bool = False
    history_transition_strides: tuple[int, ...] = (1, 2, 4, 8)
    metastable_state_dim: int = 0
    observable_target_version: str = "v2"
    local_dynamics_enabled: bool = False
    local_observable_targets: bool = True
    local_dynamics_tokens: int = 4
    local_dynamics_dim: int = 32
    closed_local_state_enabled: bool = False
    closed_local_state_dim: int = 12
    closed_local_horizons: tuple[int, ...] = (48, 96, 192)
    closed_local_state_scale: tuple[float, ...] = (1.0,) * 12
    closed_local_state_mean: tuple[float, ...] = (0.0,) * 12
    closed_local_operator_mode: str = "experts"
    closed_local_relaxation_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    closed_local_velocity_strides: tuple[int, ...] = (1, 2, 4, 8)
    closed_local_log_sigma_min: float = -4.0
    closed_local_log_sigma_max: float = 2.0
    closed_local_base_step_frames: int = 16
    closed_local_history_frame_stride: int = 1
    closed_local_cross_segment_heads: int = 4
    freeze_closed_local_operator: bool = False
    v4_slow_state_enabled: bool = False
    v4_slow_latent_dim: int = 16
    v4_slow_num_states: int = 8
    v4_slow_vamp_lags: tuple[int, ...] = (4, 8, 16)
    v4_physics_state_enabled: bool = False
    v4_physics_latent_dim: int = 16
    v4_physics_num_states: int = 8
    v4_physics_vamp_lags: tuple[int, ...] = (4, 8, 16)
    v4_physics_lowpass_width: int = 5
    v4_physics_force_scale: float = 50.0
    v4_physics_dynamic_only: bool = False
    v4_physics_condition_on_equilibrium: bool = False
    v4_physics_per_trajectory_vamp: bool = False
    freeze_v4_physics_representation: bool = False
    v4_physics_training_fixed_controls: bool = False
    v5_contextual_dynamics_enabled: bool = False
    v5_slot_dim: int = 32
    v5_max_slots: int = 16
    v5_base_step_ps: float = 16000.0
    v5_max_rollout_steps: int = 16
    v5_assignment_temperature: float = 0.25
    v5_observable_dim: int = 12
    v5_control_horizons_frames: tuple[int, ...] = (48, 64, 96, 128, 192)
    v5_training_fixed_controls: bool = False
    v51_contextual_dynamics_enabled: bool = False
    v51_state_dim: int = 32
    v51_num_modes: int = 16
    v51_active_modes: int = 4
    v51_base_step_ps: float = 16000.0
    v51_max_rollout_steps: int = 16
    v51_router_temperature: float = 0.5
    v51_routing_noise_std: float = 0.0
    v51_active_temperature: float = 0.25
    v51_log_sigma_min: float = -5.0
    v51_log_sigma_max: float = 1.0
    v51_observable_dim: int = 12
    v51_observable_state_weight: float = 0.0
    v51_control_horizons_frames: tuple[int, ...] = (48, 64, 96, 128, 192)
    v51_training_controls: bool = True
    v51_causal_prediction_enabled: bool = False
    v51_history_truncation_enabled: bool = False
    v51_history_keep_fraction: float = 0.5
    v53_multi_scale_history_enabled: bool = False
    v53_slow_history_stride: int = 2
    v53_autonomous_rollout_enabled: bool = False
    v53_horizon_calibration_enabled: bool = False
    v53_horizon_calibration_scale: float = 0.5
    v53_frame_local_target_enabled: bool = False
    v53_physical_state_anchor_enabled: bool = False
    v53_physical_state_tokens: int = 4
    v53_learned_state_residual_scale: float = 0.1
    tmjepa_enabled: bool = False
    tmjepa_version: int = 2
    tmjepa_representation_only: bool = False
    tmjepa_num_segments: int = 3
    tmjepa_learned_cv_dim: int = 0
    tmjepa_learned_cv_burn_in_frames: int = 0
    tmjepa_condition_dim: int = 64
    tmjepa_auxiliary_dim: int = 4
    tmjepa_diffusion_rank: int = 4
    tmjepa_memory_enabled: bool = True
    tmjepa_coupled_bath_enabled: bool = True
    tmjepa_observation_noise_enabled: bool = True
    tmjepa_observation_noise_scale: float = 0.1
    tmjepa_observation_noise_rank: int = 0
    tmjepa_observation_noise_condition_scale: float = 0.0
    tmjepa_observation_noise_temperature_exponent: float = 1.0
    tmjepa_memory_intervention_enabled: bool = False
    tmjepa_observation_time_scale_ps: float = 1000.0
    tmjepa_base_step_ps: float = 1000.0
    tmjepa_max_rollout_steps: int = 256
    tmjepa_ensemble_size: int = 8
    tmjepa_diffusion_floor: float = 1.0e-8
    tmjepa_learned_cv_diffusion_floor_scale: float = 1.0
    tmjepa_diffusion_scale: float = 1.0e-5
    tmjepa_relaxation_rate_scale: float = 1.0e-3
    tmjepa_coupling_scale: float = 1.0e-3
    tmjepa_equilibrium_noise_std: float = 0.05
    tmjepa_equilibrium_noise_samples: int = 1
    tmjepa_contact_width_angstrom: float = 0.75
    tmjepa_reference_temperature_k: float = 348.0
    tmjepa_activation_energy_kcal_per_mol: float = 3.0
    tmjepa_activation_deviation_kcal_per_mol: float = 1.0
    tmjepa_state_mobility_scale: float = 0.5
    tmjepa_memory_times_ps: tuple[float, ...] = (
        1000.0,
        2000.0,
        4000.0,
        8000.0,
    )
    tmjepa_heat_capacity_scale_over_kb: float = 5.0
    tmjepa_endpoint_calibration_enabled: bool = False
    tmjepa_endpoint_calibration_scale: float = 0.0
    tmjepa_endpoint_calibration_temperature_scale: float = 0.0
    tmjepa_projected_force_enabled: bool = False
    tmjepa_projected_force_hidden_dim: int = 128
    tmjepa_projected_force_max_shift: float = 0.25
    tmjepa_projected_force_history_features: bool = False
    tmjepa_projected_force_latent_history_features: bool = False
    tmjepa_projected_force_horizon_gate_steps: float = 0.0
    tmjepa_memory_coupling_warmup_steps: int = 0
    tmjepa_kinetics_lr_scale: float = 1.0
    tmjepa_freeze_potential: bool = False
    tmjepa_freeze_kinetics: bool = False
    tmjepa_freeze_diffusion: bool = False
    tmjepa_freeze_memory: bool = False
    tmjepa_freeze_observation_noise: bool = False
    tmjepa_freeze_endpoint_calibration: bool = False
    tmjepa_freeze_learned_projection: bool = False
    tmjepa_cv_mean: tuple[float, ...] = ()
    tmjepa_cv_scale: tuple[float, ...] = ()

    # LSP-WM v4.1 independent coarse-grained dynamics
    lspwm_enabled: bool = False
    lspwm_num_segments: int = 3
    # This is the complete unified JEPA coarse state, not a small CV appendage.
    lspwm_learned_cv_dim: int = 256
    lspwm_condition_dim: int = 64
    lspwm_auxiliary_dim: int = 4
    lspwm_koopman_role_dim: int = 64
    lspwm_msm_role_dim: int = 32
    lspwm_geometric_role_dim: int = 96
    lspwm_auxiliary_role_dim: int = 64
    lspwm_role_hidden_dim: int = 256
    lspwm_reference_temperature_k: float = 348.0
    lspwm_diffusion_scale: float = 1.0e-6
    lspwm_coupling_scale: float = 1.0e-3
    lspwm_memory_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0)
    lspwm_mean_force_bandwidth: float = 1.5
    # Runtime KDE-buffer settings used by the DetSkill v3 experiment.  The
    # canonical world-model module does not consume these fields, but keeping
    # them in the config schema preserves checkpoint/config reproducibility.
    lspwm_kde_buffer_enabled: bool = False
    lspwm_kde_buffer_size: int = 0
    lspwm_kde_buffer_warmup: int = 0
    lspwm_freeze_learned_projection: bool = False
    lspwm_cv_mean: tuple[float, ...] = ()
    lspwm_cv_scale: tuple[float, ...] = ()
    # Arbitrary history residuals are not FDT-compliant; keep them disabled
    # until a causal filter and conditional residual covariance are validated.
    lspwm_history_closure_enabled: bool = False

    # LSP-WM v4.1 observation model (Ch.6)
    lspwm_observation_model: bool = False
    lspwm_observation_coord_dim: int = 0

    # LSP-WM v4.1 rollout self-consistency
    lspwm_rollout_enabled: bool = False
    lspwm_rollout_steps: int = 4
    lspwm_rollout_warmup_steps: int = 0
    lspwm_rollout_ensemble_size: int = 4
    lspwm_rollout_intermediate_supervision: bool = True
    lspwm_rollout_gaussian_nll_enabled: bool = True
    lspwm_dynamic_nll_enabled: bool = True
    # V4.4 thermodynamic improvements
    lspwm_force_loss_enabled: bool = True       # decoupled from dynamic_nll_enabled
    lspwm_state_dependent_diffusion: bool = False  # heteroscedastic diffusion head
    lspwm_potential_hidden_dim: int = 128      # PMF potential hidden dimension
    lspwm_potential_hidden_depth: int = 4      # PMF potential number of hidden layers
    # Frozen-latent physical transition calibration.  The calibration module
    # changes only the overdamped transition law; it never changes z.
    lspwm_physical_calibration_enabled: bool = False
    lspwm_physical_mobility_init: float = 1.0
    lspwm_physical_free_energy_init: float = 1.0
    lspwm_physical_log_scale_min: float = -12.0
    lspwm_physical_log_scale_max: float = 12.0
    lspwm_conservative_correction_enabled: bool = False
    lspwm_conservative_correction_rank: int = 16
    lspwm_conservative_correction_scale: float = 1.0
    lspwm_conservative_correction_rescale_with_free_energy: bool = False
    lspwm_reset_conservative_correction_on_init: bool = False
    lspwm_freeze_base_dynamics: bool = False
    lspwm_freeze_calibration_scales: bool = False

    # THM-JEPA: reversible (divergence-free) dynamics (v21)
    lspwm_reversible_dynamics_enabled: bool = False
    lspwm_reversible_operator_rank: int = 16
    lspwm_reversible_hidden_dim: int = 128
    lspwm_reversible_hidden_depth: int = 4

    # LSP-WM v4.4 first-principles architectural extensions
    # (used by LSPWMConfig via getattr in jepa.py construction)
    lspwm_midpoint_integration: bool = False
    lspwm_rollout_trajectory_steps: int = 0
    lspwm_semigroup_consistency_weight: float = 0.0

    # LSP-WM v4.1 curriculum phase switch steps (representation, thermodynamic_identification, dynamics, rollout, observation)

    lspwm_curriculum_steps: tuple[int, ...] = (0, 500, 1000, 1500, 2500)

    # EVA-6 autoregressive rollout
    e6_autoregressive_enabled: bool = False
    e6_rollout_steps: int = 8
    e6_state_dim: int = 16
    e6_hidden_dim: int = 256
    e6_noise_std: float = 0.05
    e6_temperature: float = 1.0
    e6_learned_state_noise_std: float = 0.02
    e6_freeze_fixed_cv_dynamics: bool = False
    e6_freeze_learned_state_projector: bool = False
    e6_freeze_encoder: bool = False


@dataclass
class OptimizationConfig:
    epochs: int = 10
    steps_per_epoch: int | None = None
    lr: float = 3e-4
    encoder_lr_scale: float = 0.2
    weight_decay: float = 0.05
    warmup_steps: int = 500
    ema_start: float = 0.996
    ema_end: float = 1.0
    grad_clip: float | None = 1.0
    precision: str = "float32"
    auxiliary_warmup_steps: int = 0
    residue_jepa_weight: float = 1.0
    state_jepa_weight: float = 0.0
    lagged_state_jepa_weight: float = 0.0
    semigroup_weight: float = 0.0
    state_transition_weight: float = 0.0
    physchem_weight: float = 0.0
    linear_dynamics_weight: float = 0.0
    variance_reg: float = 0.0
    state_variance_reg: float = 0.0
    state_feature_variance_reg: float = 0.0
    state_covariance_reg: float = 0.0
    msm_state_weight: float = 0.0
    vamp2_weight: float = 0.0
    vamp2_lag: int = 1
    kinetic_vamp_weight: float = 0.0
    kinetic_vamp_lag: int = 1
    paired_delta_weight: float = 1.0
    paired_state_weight: float = 0.25
    paired_relative_delta_weight: float = 0.0
    paired_endpoint_variance_weight: float = 0.02
    lspwm_dynamic_weight: float = 0.0
    lspwm_calibrated_dynamic_weight: float = 0.0
    lspwm_calibrated_endpoint_weight: float = 0.0
    lspwm_calibrated_force_weight: float = 0.0
    lspwm_calibration_regularization_weight: float = 0.0
    lspwm_force_weight: float = 0.0
    lspwm_koopman_weight: float = 0.0
    lspwm_msm_weight: float = 0.0
    lspwm_sequence_vamp_weight: float = 0.0
    lspwm_sequence_vamp_lag: int = 1
    lspwm_geometric_weight: float = 0.0
    lspwm_auxiliary_weight: float = 0.0
    lspwm_observation_weight: float = 0.0

    # LSP-WM v4.1 rollout self-consistency weights
    lspwm_rollout_energy_weight: float = 0.0
    lspwm_rollout_endpoint_weight: float = 0.0
    lspwm_rollout_gaussian_nll_weight: float = 0.0
    lspwm_rollout_distribution_weight: float = 0.0
    lspwm_rollout_koopman_weight: float = 0.0
    lspwm_rollout_force_weight: float = 0.0
    lspwm_rollout_trajectory_nll_weight: float = 0.0

    slow_delta_weight: float = 0.5
    slow_state_weight: float = 0.25
    slow_relative_delta_weight: float = 0.0
    slow_alignment_weight: float = 0.0
    slow_vamp_weight: float = 0.0
    slow_contrastive_weight: float = 0.1
    slow_variance_weight: float = 0.05
    slow_covariance_weight: float = 0.01
    ls2_state_weight: float = 0.0
    ls2_assignment_weight: float = 0.0
    ls2_nll_weight: float = 0.0
    ls2_balance_weight: float = 0.0
    ls2_entropy_weight: float = 0.0
    ls2_vamp_weight: float = 0.0
    gaussian_nll_weight: float = 0.0
    observable_weight: float = 0.0
    offline_forecast_weight: float = 0.0
    offline_lr_scale: float = 1.0
    causal_alignment_weight: float = 0.0
    metastable_vamp_weight: float = 0.0
    metastable_balance_weight: float = 0.0
    metastable_entropy_weight: float = 0.0
    local_observable_weight: float = 0.0
    closed_local_state_weight: float = 0.0
    horizon_ranking_weight: float = 0.0
    horizon_ranking_margin: float = 0.002
    horizon_difference_weight: float = 0.0
    horizon_difference_min_gap: int = 0
    closed_local_robust_state_weight: float = 0.0
    horizon_robust_difference_weight: float = 0.0
    closed_local_student_t_weight: float = 0.0
    closed_local_robust_beta: float = 0.5
    closed_local_student_t_df: float = 3.0
    closed_local_base_step_weight: float = 0.0
    closed_local_robust_base_step_weight: float = 0.0
    v4_slow_reconstruction_weight: float = 0.0
    v4_slow_vamp_weight: float = 0.0
    v4_slow_balance_weight: float = 0.0
    v4_slow_entropy_weight: float = 0.0
    v4_slow_base_transition_weight: float = 0.0
    v4_slow_rollout_transition_weight: float = 0.0
    v4_physics_reconstruction_weight: float = 0.0
    v4_physics_vamp_weight: float = 0.0
    v4_physics_balance_weight: float = 0.0
    v4_physics_entropy_weight: float = 0.0
    v4_physics_cross_covariance_weight: float = 0.0
    v4_physics_increment_covariance_weight: float = 0.0
    v4_physics_timescale_weight: float = 0.0
    v4_physics_trajectory_balance_weight: float = 0.0
    v4_physics_activity_weight: float = 0.0
    v4_physics_latent_variance_weight: float = 0.0
    v4_physics_drift_weight: float = 0.0
    v4_physics_force_weight: float = 0.0
    v4_physics_stationary_weight: float = 0.0
    v4_physics_base_transition_weight: float = 0.0
    v4_physics_rollout_transition_weight: float = 0.0
    v4_physics_transition_nll_weight: float = 0.0
    v4_physics_horizon_ranking_weight: float = 0.0
    v4_physics_horizon_ranking_margin: float = 0.002
    v4_physics_horizon_difference_weight: float = 0.0
    v5_transition_weight: float = 0.0
    v5_state_reconstruction_weight: float = 0.0
    v5_feature_forecast_weight: float = 0.0
    v5_observable_weight: float = 0.0
    v5_transition_nll_weight: float = 0.0
    v5_horizon_ranking_weight: float = 0.0
    v5_horizon_ranking_margin: float = 0.002
    v5_horizon_difference_weight: float = 0.0
    v5_observable_horizon_difference_weight: float = 0.0
    v5_slot_diversity_weight: float = 0.0
    v5_gate_complexity_weight: float = 0.0
    v51_delta_weight: float = 0.0
    v51_normalized_delta_weight: float = 0.0
    v51_horizon_calibration_weight: float = 0.0
    v51_online_delta_weight: float = 0.0
    v51_online_normalized_delta_weight: float = 0.0
    v51_displacement_magnitude_weight: float = 0.0
    v51_nll_weight: float = 0.0
    v51_current_alignment_weight: float = 0.0
    v51_observable_weight: float = 0.0
    v51_oracle_observable_weight: float = 0.0
    v51_online_oracle_observable_weight: float = 0.0
    v51_horizon_ranking_weight: float = 0.0
    v51_history_ranking_weight: float = 0.0
    v51_causal_prediction_weight: float = 0.0
    v51_horizon_consistency_weight: float = 0.0
    v51_history_truncation_weight: float = 0.0
    v51_physical_horizon_consistency_weight: float = 0.0
    v51_semigroup_weight: float = 0.0
    v53_history_warmup_steps: int = 0
    v53_evolution_warmup_steps: int = 0
    v51_ranking_margin: float = 0.01
    v51_variance_weight: float = 0.0
    v51_covariance_weight: float = 0.0
    v51_load_balance_weight: float = 0.0
    v51_active_load_balance_weight: float = 0.0
    v51_mode_diversity_weight: float = 0.0
    v51_mode_orthogonality_weight: float = 0.0
    v51_target_hard_sync_steps: int = 0
    tmjepa_energy_score_weight: float = 0.0
    tmjepa_forecast_relative_mse_weight: float = 0.0
    tmjepa_transition_nll_weight: float = 0.0
    tmjepa_path_nll_weight: float = 0.0
    tmjepa_equilibrium_score_weight: float = 0.0
    tmjepa_innovation_mean_weight: float = 0.0
    tmjepa_innovation_covariance_weight: float = 0.0
    tmjepa_innovation_autocorrelation_weight: float = 0.0
    tmjepa_innovation_covariance_spectral_weight: float = 0.0
    tmjepa_innovation_worst_group_weight: float = 0.0
    tmjepa_innovation_lags: tuple[int, ...] = (1, 2, 4, 8)
    tmjepa_diffusion_condition_weight: float = 0.0
    tmjepa_diffusion_condition_target: float = 100.0
    tmjepa_diffusion_lr_scale: float = 1.0
    tmjepa_observation_noise_lr_scale: float = 1.0
    tmjepa_projected_force_lr_scale: float = 1.0
    tmjepa_memory_intervention_weight: float = 0.0
    tmjepa_memory_intervention_margin: float = 0.0
    tmjepa_memory_relative_intervention_weight: float = 0.0
    tmjepa_memory_relative_intervention_margin: float = 0.0
    tmjepa_memory_ratio_band_weight: float = 0.0
    tmjepa_memory_ratio_min: float = 0.0
    tmjepa_memory_ratio_max: float = 1.0e12
    tmjepa_memory_log_ratio_weight: float = 0.0
    tmjepa_memory_ratio_target: float = 0.08
    tmjepa_memory_long_horizon_intervention_weight: float = 0.0
    tmjepa_memory_long_horizon_relative_intervention_weight: float = 0.0
    tmjepa_memory_residual_weight: float = 0.0
    tmjepa_memory_long_horizon_residual_weight: float = 0.0
    tmjepa_long_horizon_energy_score_weight: float = 0.0
    tmjepa_long_horizon_forecast_relative_mse_weight: float = 0.0
    tmjepa_long_horizon_transition_nll_weight: float = 0.0
    tmjepa_long_horizon_path_nll_weight: float = 0.0
    tmjepa_long_horizon_min_frames: int = 8
    tmjepa_long_horizon_power: float = 1.0
    tmjepa_auxiliary_kl_weight: float = 0.0
    tmjepa_auxiliary_transition_kl_weight: float = 0.0
    tmjepa_learned_cv_variance_weight: float = 0.0
    tmjepa_learned_cv_covariance_weight: float = 0.0
    tmjepa_vamp_weight: float = 0.0
    tmjepa_vamp_lags: tuple[int, ...] = (1,)
    tmjepa_coverage_calibration_weight: float = 0.0
    tmjepa_projected_force_regularization_weight: float = 0.0

    # EVA-6 autoregressive loss weights
    e6_rollout_state_weight: float = 0.0
    e6_rollout_variance_weight: float = 0.0
    e6_rollout_energy_score_weight: float = 0.0
    e6_rollout_equilibrium_weight: float = 0.0
    e6_rollout_consistency_weight: float = 0.0
    e6_rollout_diversity_weight: float = 0.0
    e6_paired_energy_score_weight: float = 0.0
    e6_representation_vamp2_weight: float = 0.0
    e6_rollout_observation_nll_weight: float = 0.0
    e6_rollout_vamp2_lags: tuple[int, ...] = (1, 2, 4, 8)
    e6_rollout_warmup_steps: int = 0
    e6_rollout_horizon_decay: float = 0.9
    e6_equilibrium_mmd_bandwidth: float = 1.0
    e6_equilibrium_samples: int = 64


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    seed: int = 0
    deterministic: bool = False
    device: str = "auto"
    output_dir: str = "artifacts/run"
    init_checkpoint: str | None = None
    init_exclude_prefixes: tuple[str, ...] = ()
    sync_target_after_init: bool = False
    resume_checkpoint: str | None = None
    strict_init: bool = True
    init_copy_temporal_to_offline: bool = False
    log_every: int = 20
    save_every: int = 1
    checkpoint_every_steps: int = 10000


def _coerce_dataclass(cls: type, values: dict[str, Any]) -> Any:
    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(field_map))
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    for name, spec in field_map.items():
        if name not in values:
            continue
        value = values[name]
        default_value = getattr(cls(), name)
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[name] = _coerce_dataclass(type(default_value), value)
        elif isinstance(default_value, tuple) and isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must decode to a mapping")
    return _coerce_dataclass(TrainConfig, raw)


def model_config_from_dict(values: dict[str, Any], state_dict: dict[str, Any] | None = None) -> ModelConfig:
    """Build a model config while recovering flags absent from old artifacts."""
    resolved = dict(values)
    keys = tuple(state_dict or ())
    if any(key.startswith("paired_delta_predictor.") for key in keys):
        resolved.setdefault("paired_forecast", True)
    if any(key.startswith("slow_state_projector.") for key in keys):
        resolved["slow_state_enabled"] = True
    if any(key.startswith("ls2_reversible_dynamics.") for key in keys):
        resolved["ls2_reversible_enabled"] = True
    if any(key.startswith("offline_temporal_encoder.") for key in keys):
        resolved["dual_temporal_towers"] = True
    if any(key.startswith("gaussian_transition.") for key in keys):
        resolved["probabilistic_transition"] = True
    if any(key.startswith("observable_head.") for key in keys):
        resolved["observable_head"] = True
    if any(key.startswith("paired_delta_predictor.summary.") for key in keys):
        resolved["history_conditioned_transition"] = True
    metastable_weights = []
    for key, value in (state_dict or {}).items():
        parts = key.split(".")
        if len(parts) == 4 and parts[:2] == ["metastable_state_head", "net"] and parts[3] == "weight":
            metastable_weights.append((int(parts[2]), value))
    if metastable_weights:
        resolved["metastable_state_dim"] = int(max(metastable_weights, key=lambda item: item[0])[1].shape[0])
    if any(key.startswith("local_dynamics_head.") for key in keys):
        resolved["local_dynamics_enabled"] = True
    local_token_weight = (state_dict or {}).get("local_dynamics_head.token_proj.1.weight")
    if local_token_weight is not None:
        resolved["local_dynamics_dim"] = int(local_token_weight.shape[0])
    local_segment_weight = (state_dict or {}).get("local_dynamics_head.segment_embed.weight")
    if local_segment_weight is not None:
        resolved["local_dynamics_tokens"] = int(local_segment_weight.shape[0])
    if any(key.startswith("closed_local_operator.") for key in keys):
        resolved["closed_local_state_enabled"] = True
    if any(key.startswith("v4_slow_kinetic_model.") for key in keys):
        resolved["v4_slow_state_enabled"] = True
    if any(key.startswith("v4_physics_slow_model.") for key in keys):
        resolved["v4_physics_state_enabled"] = True
    if any(key.startswith("v5_contextual_dynamics.") for key in keys):
        resolved["v5_contextual_dynamics_enabled"] = True
    if any(key.startswith("v51_contextual_dynamics.") for key in keys):
        resolved["v51_contextual_dynamics_enabled"] = True
    if any(key.startswith("tmjepa.") for key in keys):
        resolved["tmjepa_enabled"] = True
    if any(key.startswith("tmjepa.dynamics.potential.") for key in keys):
        resolved["tmjepa_version"] = 3
    known = {item.name for item in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in resolved.items() if key in known})


def config_to_dict(config: Any) -> Any:
    if is_dataclass(config):
        return {f.name: config_to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, tuple):
        return list(config)
    return config
