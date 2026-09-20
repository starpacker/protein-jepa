#!/usr/bin/env python3
import json

# Known ModelConfig fields (from config.py dataclass)
known_model_fields = {
    'latent_dim', 'encoder_depth', 'temporal_depth', 'predictor_depth',
    'state_predictor_depth', 'state_prediction_lags', 'frame_pooling',
    'frame_pool_queries', 'lag_predictor_depth', 'lag_predictor_dropout',
    'lag_conditioning', 'lag_prediction_mode', 'time_conditioning',
    'thermodynamic_conditioning', 'orientation_features', 'temporal_causal',
    'temporal_causal_window', 'temporal_position_encoding', 'temporal_delta_features',
    'motion_depth', 'motion_stopgrad_input', 'transition_source',
    'transition_thermodynamic_conditioning', 'linear_dynamics_lags',
    'physchem_prediction_lags', 'physchem_predictor_depth', 'physchem_contact_cutoff',
    'kinetic_readout_dim', 'kinetic_readout_depth', 'kinetic_readout_stopgrad',
    'num_heads', 'num_rbf', 'dropout', 'max_frames', 'max_residues',
    'feature_norm', 'paired_forecast', 'paired_predictor_depth',
    'learned_state_training', 'slow_state_enabled', 'slow_state_dim',
    'slow_predictor_depth', 'slow_vamp_lags', 'slow_vamp_burn_in_frames',
    'ls2_reversible_enabled', 'ls2_num_states', 'ls2_base_step_ps',
    'ls2_assignment_temperature', 'ls2_stationary_floor', 'ls2_rate_scale',
    'dual_temporal_towers', 'offline_temporal_depth', 'causal_temporal_depth',
    'probabilistic_transition', 'transition_log_sigma_min', 'transition_log_sigma_max',
    'observable_head', 'observable_dim', 'causal_encoder_gradient_scale',
    'paired_teacher_space_delta', 'paired_target_tower',
    'freeze_context_encoder', 'freeze_temporal_encoder', 'freeze_target_encoders',
    'freeze_offline_temporal_encoder', 'history_conditioned_transition',
    'history_transition_strides', 'metastable_state_dim',
    'observable_target_version', 'local_dynamics_enabled',
    'local_observable_targets', 'local_dynamics_tokens', 'local_dynamics_dim',
    'closed_local_state_enabled', 'closed_local_state_dim',
    'closed_local_horizons', 'closed_local_state_scale', 'closed_local_state_mean',
    'closed_local_operator_mode', 'closed_local_relaxation_scales',
    'closed_local_velocity_strides', 'closed_local_log_sigma_min',
    'closed_local_log_sigma_max', 'closed_local_base_step_frames',
    'closed_local_history_frame_stride', 'closed_local_cross_segment_heads',
    'freeze_closed_local_operator',
    'v4_slow_state_enabled', 'v4_slow_latent_dim', 'v4_slow_num_states',
    'v4_slow_vamp_lags', 'v4_physics_state_enabled', 'v4_physics_latent_dim',
    'v4_physics_num_states', 'v4_physics_vamp_lags', 'v4_physics_lowpass_width',
    'v4_physics_force_scale', 'v4_physics_dynamic_only',
    'v4_physics_condition_on_equilibrium', 'v4_physics_per_trajectory_vamp',
    'freeze_v4_physics_representation', 'v4_physics_training_fixed_controls',
    'v5_contextual_dynamics_enabled', 'v5_slot_dim', 'v5_max_slots',
    'v5_base_step_ps', 'v5_max_rollout_steps', 'v5_assignment_temperature',
    'v5_observable_dim', 'v5_control_horizons_frames', 'v5_training_fixed_controls',
    'v51_contextual_dynamics_enabled', 'v51_state_dim', 'v51_num_modes',
    'v51_active_modes', 'v51_base_step_ps', 'v51_max_rollout_steps',
    'tmjepa_enabled', 'lspwm_enabled', 'lspwm_encoder_dim', 'lspwm_state_dim',
    'lspwm_auxiliary_dim', 'lspwm_hidden_dim',
    'lspwm_num_segments', 'lspwm_learned_cv_dim', 'lspwm_condition_dim',
    'lspwm_koopman_role_dim', 'lspwm_msm_role_dim', 'lspwm_geometric_role_dim',
    'lspwm_auxiliary_role_dim', 'lspwm_role_hidden_dim',
    'lspwm_reference_temperature_k', 'lspwm_diffusion_scale', 'lspwm_coupling_scale',
    'lspwm_memory_times_ps', 'lspwm_mean_force_bandwidth',
    'lspwm_observation_model', 'lspwm_history_closure_enabled',
    'lspwm_midpoint_integration', 'lspwm_rollout_trajectory_steps',
    'lspwm_semigroup_consistency_weight', 'lspwm_rollout_enabled',
    'lspwm_dynamic_nll_enabled', 'lspwm_force_loss_enabled',
    'lspwm_state_dependent_diffusion', 'lspwm_freeze_learned_projection',
    'lspwm_potential_hidden_dim', 'lspwm_potential_hidden_depth',
    'lspwm_curriculum_steps', 'lspwm_cv_mean', 'lspwm_cv_scale',
    'lspwm_physical_calibration_enabled',
}

known_opt_fields = {
    'epochs', 'steps_per_epoch', 'lr', 'encoder_lr_scale', 'weight_decay',
    'warmup_steps', 'ema_start', 'ema_end', 'grad_clip', 'precision',
    'paired_delta_weight', 'paired_state_weight', 'paired_relative_delta_weight',
    'paired_endpoint_variance_weight',
    'lspwm_dynamic_weight', 'lspwm_force_weight', 'lspwm_koopman_weight',
    'lspwm_msm_weight', 'lspwm_sequence_vamp_weight', 'lspwm_sequence_vamp_lag',
    'lspwm_geometric_weight', 'lspwm_auxiliary_weight', 'lspwm_observation_weight',
    'lspwm_trajectory_weight', 'lspwm_semigroup_weight',
    'lspwm_rollout_energy_weight', 'lspwm_rollout_endpoint_weight',
    'lspwm_rollout_gaussian_nll_weight', 'lspwm_rollout_distribution_weight',
    'lspwm_rollout_koopman_weight', 'lspwm_rollout_force_weight',
    'lspwm_rollout_trajectory_nll_weight',
}

train_cfg_path = '/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v20_force_alignment/config.json'
with open(train_cfg_path) as f:
    train_cfg = json.load(f)

eval_cfg = {
    'seed': 6303,
    'device': 'cuda',
    'output_dir': '/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v20_force_alignment',
    'init_checkpoint': None,
    'init_exclude_prefixes': [],
    'strict_init': False,
    'sync_target_after_init': False,
    'log_every': 10,
    'save_every': 1,
    'checkpoint_every_steps': 100,
    'data': {
        'manifest_path': '/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.train.jsonl',
        'batch_size': 8,
        'num_workers': 0,
        'num_frames': 24,
        'frame_stride': 1,
        'window_stride': 8,
        'max_residues': 384,
        'sampling_mode': 'hierarchical',
        'training_mode': 'paired_window',
        'paired_horizons_frames': [24, 48, 96],
        'paired_nonoverlap_only': True,
        'paired_all_horizons': True,
    },
    'model': {k: v for k, v in train_cfg.get('model', {}).items()
              if k in known_model_fields},
    'optimization': {'epochs': 1},
}

n_model_filt = len(train_cfg.get('model', {}))
n_model_kept = len(eval_cfg['model'])
excluded = sorted(k for k in train_cfg.get('model', {}) if k not in known_model_fields)

output_path = '/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v20_force_alignment/eval_config.json'
with open(output_path, 'w') as f:
    json.dump(eval_cfg, f, indent=2)

print(f'Wrote {output_path}')
print(f'Model fields: {n_model_filt} -> {n_model_kept}')
print(f'Excluded: {excluded}')
