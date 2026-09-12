from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.predictor import (
    E6AutoregressiveDynamics,
    HistoryConditionedTransition,
    ContextualAdaptiveDynamics,
    ContextualSparseMotionDynamics,
    KineticReadoutHead,
    LaggedLinearDynamicsPredictor,
    LaggedPhyschemPredictor,
    LaggedStatePredictor,
    GaussianResidualTransition,
    ClosedLocalStateOperator,
    V4SlowKineticModel,
    V4PhysicsSlowModel,
    LocalDynamicsHead,
    MetastableStateHead,
    ObservableHead,
    PairedDeltaPredictor,
    ReversibleKineticOUDynamics,
    SlowStateProjector,
    SpatiotemporalPredictor,
    StatePredictor,
)
from protein_jepa_frame.models.temporal import TemporalStateEncoder
from protein_jepa_frame.models.thermodynamic import ThermodynamicMoriJEPA
from protein_jepa_frame.models.thermodynamic_v3 import ThermodynamicMoriJEPAV3
from protein_jepa_frame.models.lspwm import LSPWMProteinWorldModel


def upgrade_tmjepa_v3_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Upgrade historical V3 checkpoints without weakening strict loading."""
    upgraded = dict(state_dict)
    prefix = "tmjepa.dynamics.kinetics."
    memory_global = prefix + "memory_activation_energy_raw"
    diffusion_global = prefix + "activation_energy_raw"
    memory_deviation = prefix + "memory_activation_deviation"
    memory_rotation = prefix + "memory_rotation_raw"
    target_state = model.state_dict()
    memory_coupling_ramp = prefix + "_memory_coupling_ramp"
    if memory_coupling_ramp in target_state and memory_coupling_ramp not in upgraded:
        # Checkpoints predating the warmup buffer used the full memory coupling.
        upgraded[memory_coupling_ramp] = torch.ones_like(
            target_state[memory_coupling_ramp]
        )
    diffusion_floor_scale = prefix + "diffusion_floor_scale"
    if diffusion_floor_scale in target_state and diffusion_floor_scale not in upgraded:
        upgraded[diffusion_floor_scale] = target_state[diffusion_floor_scale].clone()
    if memory_global in target_state and memory_global not in upgraded:
        upgraded[memory_global] = upgraded[diffusion_global].clone()
    if memory_deviation in target_state and memory_deviation not in upgraded:
        upgraded[memory_deviation] = torch.zeros_like(target_state[memory_deviation])
    if memory_rotation in target_state and memory_rotation not in upgraded:
        upgraded[memory_rotation] = torch.zeros_like(target_state[memory_rotation])
    observation_prefix = prefix + "observation_noise_"
    for key in target_state:
        if key.startswith(observation_prefix) and key not in upgraded:
            upgraded[key] = target_state[key].clone()
    calibration_prefix = "tmjepa.endpoint_calibration."
    for key in target_state:
        if key.startswith(calibration_prefix) and key not in upgraded:
            upgraded[key] = target_state[key].clone()
    projected_force_prefix = "tmjepa.projected_force_corrector."
    for key in target_state:
        if (
            key.startswith(projected_force_prefix)
            and (
                key not in upgraded
                or tuple(upgraded[key].shape) != tuple(target_state[key].shape)
            )
        ):
            upgraded[key] = target_state[key].clone()
    return upgraded


def upgrade_lspwm_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Migrate pre-temperature LSP-WM candidates without silent loading.

    The temperature feature was appended to the PMF input and the unvalidated
    history closure was removed from the canonical path. Old checkpoints keep
    their learned columns; the new temperature column starts at zero and old
    history-only parameters are dropped explicitly.

    Also handles the transition from ConstantDiffusion to StateDependentDiffusion
    by dropping incompatible diffusion parameters (they are re-initialised).
    """
    upgraded = dict(state_dict)
    target_state = model.state_dict()
    first_layer = "lspwm.world_model.potential.network.0.weight"
    old_weight = upgraded.get(first_layer)
    new_weight = target_state.get(first_layer)
    if (
        old_weight is not None
        and new_weight is not None
        and old_weight.ndim == 2
        and new_weight.shape[0] == old_weight.shape[0]
        and new_weight.shape[1] == old_weight.shape[1] + 1
    ):
        migrated = new_weight.clone()
        migrated[:, :-1] = old_weight
        upgraded[first_layer] = migrated
    for key in tuple(upgraded):
        if key.startswith("lspwm.world_model.history_closure.") and key not in target_state:
            upgraded.pop(key)

    # Handle ConstantDiffusion ↔ StateDependentDiffusion transition
    # Drop incompatible diffusion keys (they will be re-initialised)
    has_old_diffusion = any(
        key.startswith("lspwm.world_model.diffusion.raw_factor") for key in upgraded
    )
    has_new_diffusion = any(
        key.startswith("lspwm.world_model.diffusion.net.") for key in upgraded
    )
    target_has_constant = any(
        key.startswith("lspwm.world_model.diffusion.raw_factor") for key in target_state
    )
    if has_old_diffusion and not target_has_constant:
        # Old ConstantDiffusion → new StateDependentDiffusion
        for key in tuple(upgraded):
            if key.startswith("lspwm.world_model.diffusion."):
                upgraded.pop(key)
    elif has_new_diffusion and target_has_constant:
        # New StateDependentDiffusion → old ConstantDiffusion (unlikely, but handle)
        for key in tuple(upgraded):
            if key.startswith("lspwm.world_model.diffusion."):
                upgraded.pop(key)

    # The auxiliary initializer was added after the original Stage B
    # checkpoints.  It is not used by the single-step endpoint evaluator, so
    # initialize only this new optional module from the target model.
    auxiliary_prefix = "lspwm.auxiliary_initializer."
    for key in target_state:
        if key.startswith(auxiliary_prefix) and key not in upgraded:
            upgraded[key] = target_state[key].clone()

    # The physical calibration head is intentionally absent from historical
    # checkpoints.  Initialize its missing keys from the new model so strict
    # loading remains possible without reinitializing the frozen latent path.
    calibration_prefix = "lspwm.world_model.physical_calibration."
    for key in target_state:
        if key.startswith(calibration_prefix) and key not in upgraded:
            upgraded[key] = target_state[key].clone()

    # KDE buffers are runtime state from the buffered DetSkill experiment and
    # are not part of the canonical LSP-WM model architecture.
    for key in tuple(upgraded):
        if key.startswith("lspwm._kde_buffer") and key not in target_state:
            upgraded.pop(key)

    # Handle potential network architecture changes (depth, hidden_dim)
    old_potential_keys = [k for k in upgraded if k.startswith("lspwm.world_model.potential.network.")]
    new_potential_keys = [k for k in target_state if k.startswith("lspwm.world_model.potential.network.")]
    if old_potential_keys and new_potential_keys:
        # Check if architectures match — if not, drop old potential weights
        # and let the new architecture use its default initialization
        has_mismatch = False
        if len(old_potential_keys) != len(new_potential_keys):
            has_mismatch = True
        else:
            for key in old_potential_keys:
                old_shape = upgraded[key].shape
                new_shape = target_state[key].shape
                if old_shape != new_shape:
                    has_mismatch = True
                    break
        if has_mismatch:
            for key in old_potential_keys:
                upgraded.pop(key)

    return upgraded


class ProteinJEPA(nn.Module):
    """V-JEPA mechanics adapted to protein trajectory clips."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.paired_target_tower not in {"causal", "offline"}:
            raise ValueError("paired_target_tower must be causal or offline")
        if config.paired_target_tower == "offline" and not config.dual_temporal_towers:
            raise ValueError("paired_target_tower=offline requires dual_temporal_towers")
        if config.history_conditioned_transition and not config.dual_temporal_towers:
            raise ValueError("history-conditioned transition requires dual temporal towers")
        if config.observable_target_version not in {"v2", "v3"}:
            raise ValueError("observable_target_version must be v2 or v3")
        if config.observable_head and config.observable_target_version == "v3" and config.observable_dim != 12:
            raise ValueError("V3 observable targets require observable_dim=12")
        if config.closed_local_state_enabled and not config.local_dynamics_enabled:
            raise ValueError("closed local state requires local dynamics tokens")
        if config.v4_physics_state_enabled and not config.closed_local_state_enabled:
            raise ValueError("V4-C physics state requires closed local states")
        if config.v4_physics_state_enabled and not config.dual_temporal_towers:
            raise ValueError("V4-C physics state requires frozen offline temporal features")
        if config.v4_physics_state_enabled and config.v4_slow_state_enabled:
            raise ValueError("V4-B and V4-C slow models cannot be enabled together")
        if config.v5_contextual_dynamics_enabled and not config.paired_forecast:
            raise ValueError("V5 contextual dynamics requires paired forecasting")
        if config.v51_contextual_dynamics_enabled and not config.paired_forecast:
            raise ValueError("V5.1 contextual dynamics requires paired forecasting")
        if config.v5_contextual_dynamics_enabled and config.v51_contextual_dynamics_enabled:
            raise ValueError("V5 and V5.1 contextual dynamics cannot be enabled together")
        if config.tmjepa_enabled and not config.paired_forecast:
            raise ValueError("TM-JEPA requires paired forecasting")
        if config.lspwm_enabled and not config.paired_forecast:
            raise ValueError("LSP-WM requires paired forecasting")
        if config.lspwm_enabled and config.tmjepa_enabled:
            raise ValueError("LSP-WM and TM-JEPA branches are mutually exclusive")
        if config.lspwm_enabled and (config.temporal_depth <= 0 or not config.temporal_causal):
            raise ValueError("LSP-WM requires a causal temporal encoder")
        if config.lspwm_enabled and config.lspwm_learned_cv_dim != config.latent_dim:
            raise ValueError(
                "LSP-WM unified state dimension must equal the JEPA latent_dim"
            )
        if config.lspwm_enabled and len(config.lspwm_memory_times_ps) != config.lspwm_auxiliary_dim:
            raise ValueError("LSP-WM memory times must match auxiliary_dim")
        if config.tmjepa_enabled and (
            config.v5_contextual_dynamics_enabled or config.v51_contextual_dynamics_enabled
        ):
            raise ValueError("TM-JEPA and legacy V5 dynamics cannot be enabled together")
        if config.tmjepa_enabled and config.tmjepa_version == 3 and config.tmjepa_learned_cv_dim:
            if config.temporal_depth <= 0 or not config.temporal_causal:
                raise ValueError("TM-JEPA V3 learned CVs require a causal temporal encoder")
            if config.temporal_causal_window <= 0:
                raise ValueError("TM-JEPA V3 learned CVs require a fixed causal window")
            causal_depth = (
                config.causal_temporal_depth
                if config.dual_temporal_towers and config.causal_temporal_depth > 0
                else config.temporal_depth
            )
            minimum_burn_in = causal_depth * (config.temporal_causal_window - 1)
            if config.tmjepa_learned_cv_burn_in_frames < minimum_burn_in:
                raise ValueError(
                    "TM-JEPA V3 learned-CV burn-in must fill the effective causal receptive field"
                )
        if config.tmjepa_projected_force_latent_history_features and (
            not config.tmjepa_enabled
            or config.tmjepa_version != 3
            or config.temporal_depth <= 0
            or config.tmjepa_learned_cv_dim <= 0
        ):
            raise ValueError(
                "latent projected-force history requires TM-JEPA V3 learned CVs "
                "and a temporal encoder"
            )
        if config.tmjepa_representation_only and (
            not config.tmjepa_enabled or config.tmjepa_learned_cv_dim <= 0
        ):
            raise ValueError("TM-JEPA representation-only mode requires learned CVs")
        if config.learned_state_training:
            if not config.paired_forecast or not config.slow_state_enabled:
                raise ValueError(
                    "learned-state training requires paired forecast and a slow-state projector"
                )
            if config.tmjepa_enabled:
                raise ValueError(
                    "learned-state training cannot use the fixed-CV TM-JEPA branch"
                )
            if config.freeze_context_encoder or config.freeze_temporal_encoder:
                raise ValueError("learned-state training requires trainable online encoders")
            if config.causal_encoder_gradient_scale <= 0.0:
                raise ValueError("learned-state training requires nonzero encoder gradients")
            if config.slow_vamp_burn_in_frames < 0:
                raise ValueError("slow-state VAMP burn-in must be non-negative")
            if any(lag <= 0 for lag in config.slow_vamp_lags):
                raise ValueError("slow-state VAMP lags must be positive")
        if config.ls2_reversible_enabled:
            if not config.learned_state_training:
                raise ValueError("LS2 reversible dynamics requires learned-state training")
            if config.ls2_num_states < 2:
                raise ValueError("LS2 reversible dynamics requires at least two states")
        self.context_encoder = ProteinGeometryEncoder(
            latent_dim=config.latent_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            num_rbf=config.num_rbf,
            dropout=config.dropout,
            max_residues=config.max_residues,
            orientation_features=config.orientation_features,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = SpatiotemporalPredictor(
            latent_dim=config.latent_dim,
            depth=config.predictor_depth,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_frames=config.max_frames,
            max_residues=config.max_residues,
            time_conditioning=config.time_conditioning,
        )
        causal_depth = config.causal_temporal_depth if config.dual_temporal_towers and config.causal_temporal_depth > 0 else config.temporal_depth
        self.temporal_encoder = (
            TemporalStateEncoder(
                latent_dim=config.latent_dim,
                depth=causal_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
                frame_pooling=config.frame_pooling,
                frame_pool_queries=config.frame_pool_queries,
                time_conditioning=config.time_conditioning,
                thermodynamic_conditioning=config.thermodynamic_conditioning,
                causal=True if config.dual_temporal_towers else config.temporal_causal,
                causal_window=config.temporal_causal_window,
                position_encoding=config.temporal_position_encoding,
                delta_features=config.temporal_delta_features,
                motion_depth=config.motion_depth,
                motion_stopgrad_input=config.motion_stopgrad_input,
            )
            if causal_depth > 0
            else None
        )
        self.target_temporal_encoder = copy.deepcopy(self.temporal_encoder) if self.temporal_encoder is not None else None
        if self.target_temporal_encoder is not None:
            for param in self.target_temporal_encoder.parameters():
                param.requires_grad = False
        self.offline_temporal_encoder = None
        self.target_offline_temporal_encoder = None
        if config.dual_temporal_towers:
            offline_depth = config.offline_temporal_depth if config.offline_temporal_depth > 0 else config.temporal_depth
            if offline_depth <= 0:
                raise ValueError("dual_temporal_towers requires a positive offline temporal depth")
            self.offline_temporal_encoder = TemporalStateEncoder(
                latent_dim=config.latent_dim,
                depth=offline_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
                frame_pooling=config.frame_pooling,
                frame_pool_queries=config.frame_pool_queries,
                time_conditioning=config.time_conditioning,
                thermodynamic_conditioning=config.thermodynamic_conditioning,
                causal=False,
                causal_window=0,
                position_encoding=config.temporal_position_encoding,
                delta_features=config.temporal_delta_features,
                motion_depth=0,
                motion_stopgrad_input=False,
            )
            self.target_offline_temporal_encoder = copy.deepcopy(self.offline_temporal_encoder)
            for param in self.target_offline_temporal_encoder.parameters():
                param.requires_grad = False
        self.state_predictor = (
            StatePredictor(
                latent_dim=config.latent_dim,
                depth=config.state_predictor_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
                time_conditioning=config.time_conditioning,
            )
            if self.temporal_encoder is not None
            else None
        )
        self.lagged_state_predictor = (
            LaggedStatePredictor(
                latent_dim=config.latent_dim,
                lags=config.state_prediction_lags,
                depth=config.lag_predictor_depth,
                dropout=config.dropout if config.lag_predictor_dropout is None else config.lag_predictor_dropout,
                conditioning=config.lag_conditioning,
                prediction_mode=config.lag_prediction_mode,
                time_conditioning=config.time_conditioning,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
            )
            if self.temporal_encoder is not None and len(config.state_prediction_lags) > 0
            else None
        )
        self.linear_dynamics_predictor = (
            LaggedLinearDynamicsPredictor(latent_dim=config.latent_dim, lags=config.linear_dynamics_lags)
            if self.temporal_encoder is not None and len(config.linear_dynamics_lags) > 0
            else None
        )
        self.physchem_predictor = (
            LaggedPhyschemPredictor(
                latent_dim=config.latent_dim,
                lags=config.physchem_prediction_lags,
                depth=config.physchem_predictor_depth,
                dropout=config.dropout,
            )
            if self.temporal_encoder is not None and len(config.physchem_prediction_lags) > 0
            else None
        )
        self.kinetic_readout = (
            KineticReadoutHead(
                latent_dim=config.latent_dim,
                output_dim=config.kinetic_readout_dim,
                depth=config.kinetic_readout_depth,
                dropout=config.dropout,
            )
            if self.temporal_encoder is not None and config.kinetic_readout_dim > 0
            else None
        )
        if config.paired_forecast and config.history_conditioned_transition:
            self.paired_delta_predictor = HistoryConditionedTransition(
                latent_dim=config.latent_dim,
                depth=config.paired_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
                strides=config.history_transition_strides,
            )
        else:
            self.paired_delta_predictor = (
                PairedDeltaPredictor(
                    latent_dim=config.latent_dim,
                    depth=config.paired_predictor_depth,
                    dropout=config.dropout,
                    thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
                )
                if config.paired_forecast
                else None
            )
        self.offline_paired_delta_predictor = (
            PairedDeltaPredictor(
                latent_dim=config.latent_dim,
                depth=config.paired_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
                zero_init=False,
            )
            if config.paired_forecast and config.dual_temporal_towers
            else None
        )
        self.gaussian_transition = (
            GaussianResidualTransition(
                latent_dim=config.latent_dim,
                depth=config.paired_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
                log_sigma_min=config.transition_log_sigma_min,
                log_sigma_max=config.transition_log_sigma_max,
            )
            if config.paired_forecast and config.probabilistic_transition
            else None
        )
        self.observable_head = (
            ObservableHead(
                latent_dim=config.latent_dim,
                output_dim=config.observable_dim,
                depth=config.paired_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
            )
            if config.paired_forecast and config.observable_head
            else None
        )
        self.slow_state_projector = (
            SlowStateProjector(
                latent_dim=config.latent_dim,
                output_dim=config.slow_state_dim,
                depth=config.slow_predictor_depth,
                dropout=config.dropout,
            )
            if config.paired_forecast and config.slow_state_enabled
            else None
        )
        self.slow_delta_predictor = (
            PairedDeltaPredictor(
                latent_dim=config.slow_state_dim,
                depth=config.slow_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
            )
            if config.paired_forecast and config.slow_state_enabled and not config.ls2_reversible_enabled
            else None
        )
        self.target_slow_state_projector = copy.deepcopy(self.slow_state_projector)
        if self.target_slow_state_projector is not None:
            for param in self.target_slow_state_projector.parameters():
                param.requires_grad = False
        self.ls2_reversible_dynamics = (
            ReversibleKineticOUDynamics(
                state_dim=config.slow_state_dim,
                num_states=config.ls2_num_states,
                base_step_ps=config.ls2_base_step_ps,
                assignment_temperature=config.ls2_assignment_temperature,
                stationary_floor=config.ls2_stationary_floor,
                rate_scale=config.ls2_rate_scale,
                dropout=config.dropout,
            )
            if config.ls2_reversible_enabled
            else None
        )
        self.metastable_state_head = (
            MetastableStateHead(
                latent_dim=config.latent_dim,
                output_dim=config.metastable_state_dim,
                depth=config.slow_predictor_depth,
                dropout=config.dropout,
            )
            if config.metastable_state_dim > 0
            else None
        )
        self.local_dynamics_head = (
            LocalDynamicsHead(
                latent_dim=config.latent_dim,
                token_dim=config.local_dynamics_dim,
                num_tokens=config.local_dynamics_tokens,
                output_dim=12,
                depth=config.paired_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
            )
            if config.local_dynamics_enabled
            else None
        )
        self.closed_local_operator = (
            ClosedLocalStateOperator(
                latent_dim=config.latent_dim,
                token_dim=config.local_dynamics_dim,
                state_dim=config.closed_local_state_dim,
                num_tokens=config.local_dynamics_tokens,
                horizons=config.closed_local_horizons,
                state_scale=config.closed_local_state_scale,
                state_mean=config.closed_local_state_mean,
                operator_mode=config.closed_local_operator_mode,
                relaxation_scales=config.closed_local_relaxation_scales,
                velocity_strides=config.closed_local_velocity_strides,
                log_sigma_min=config.closed_local_log_sigma_min,
                log_sigma_max=config.closed_local_log_sigma_max,
                base_step_frames=config.closed_local_base_step_frames,
                cross_segment_heads=config.closed_local_cross_segment_heads,
                depth=config.paired_predictor_depth,
                dropout=config.dropout,
            )
            if config.closed_local_state_enabled
            else None
        )
        self.v4_slow_kinetic_model = (
            V4SlowKineticModel(
                num_tokens=config.local_dynamics_tokens,
                state_dim=config.closed_local_state_dim,
                latent_dim=config.v4_slow_latent_dim,
                num_states=config.v4_slow_num_states,
                base_step_frames=config.closed_local_base_step_frames,
                history_frame_stride=config.closed_local_history_frame_stride,
                state_scale=config.closed_local_state_scale,
                dropout=config.dropout,
            )
            if config.v4_slow_state_enabled
            else None
        )
        self.v4_physics_slow_model = (
            V4PhysicsSlowModel(
                offline_dim=config.latent_dim,
                num_tokens=config.local_dynamics_tokens,
                state_dim=config.closed_local_state_dim,
                latent_dim=config.v4_physics_latent_dim,
                num_states=config.v4_physics_num_states,
                base_step_frames=config.closed_local_base_step_frames,
                history_frame_stride=config.closed_local_history_frame_stride,
                state_scale=config.closed_local_state_scale,
                lowpass_width=config.v4_physics_lowpass_width,
                dynamic_only=config.v4_physics_dynamic_only,
                condition_on_equilibrium=config.v4_physics_condition_on_equilibrium,
                training_fixed_controls=config.v4_physics_training_fixed_controls,
                dropout=config.dropout,
            )
            if config.v4_physics_state_enabled
            else None
        )
        self.v5_contextual_dynamics = (
            ContextualAdaptiveDynamics(
                latent_dim=config.latent_dim,
                slot_dim=config.v5_slot_dim,
                max_slots=config.v5_max_slots,
                base_step_ps=config.v5_base_step_ps,
                max_rollout_steps=config.v5_max_rollout_steps,
                assignment_temperature=config.v5_assignment_temperature,
                observable_dim=config.v5_observable_dim,
                control_horizons_frames=config.v5_control_horizons_frames,
                training_fixed_controls=config.v5_training_fixed_controls,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
                dropout=config.dropout,
            )
            if config.v5_contextual_dynamics_enabled
            else None
        )
        self.v51_contextual_dynamics = (
            ContextualSparseMotionDynamics(
                latent_dim=config.latent_dim,
                state_dim=config.v51_state_dim,
                num_modes=config.v51_num_modes,
                active_modes=config.v51_active_modes,
                base_step_ps=config.v51_base_step_ps,
                max_rollout_steps=config.v51_max_rollout_steps,
                router_temperature=config.v51_router_temperature,
                routing_noise_std=config.v51_routing_noise_std,
                active_temperature=config.v51_active_temperature,
                log_sigma_min=config.v51_log_sigma_min,
                log_sigma_max=config.v51_log_sigma_max,
                observable_dim=config.v51_observable_dim,
                observable_state_weight=config.v51_observable_state_weight,
                control_horizons_frames=config.v51_control_horizons_frames,
                training_controls=config.v51_training_controls,
                causal_prediction_enabled=config.v51_causal_prediction_enabled,
                history_truncation_enabled=config.v51_history_truncation_enabled,
                history_keep_fraction=config.v51_history_keep_fraction,
                multi_scale_history_enabled=config.v53_multi_scale_history_enabled,
                slow_history_stride=config.v53_slow_history_stride,
                autonomous_rollout_enabled=config.v53_autonomous_rollout_enabled,
                horizon_calibration_enabled=config.v53_horizon_calibration_enabled,
                horizon_calibration_scale=config.v53_horizon_calibration_scale,
                frame_local_target_enabled=config.v53_frame_local_target_enabled,
                physical_state_anchor_enabled=config.v53_physical_state_anchor_enabled,
                learned_state_residual_scale=config.v53_learned_state_residual_scale,
                dropout=config.dropout,
            )
            if config.v51_contextual_dynamics_enabled
            else None
        )
        self.e6_autoregressive_dynamics = (
            E6AutoregressiveDynamics(
                latent_dim=config.latent_dim,
                cv_dim=config.e6_state_dim,
                condition_dim=config.tmjepa_condition_dim,
                hidden_dim=config.e6_hidden_dim,
                rollout_steps=config.e6_rollout_steps,
                noise_std=config.e6_noise_std,
                learned_state_noise_std=config.e6_learned_state_noise_std,
                temperature=config.e6_temperature,
                dropout=config.dropout,
            )
            if config.e6_autoregressive_enabled
            else None
        )
        if config.tmjepa_enabled and config.tmjepa_version == 2:
            self.tmjepa = ThermodynamicMoriJEPA(
                latent_dim=config.latent_dim,
                num_segments=config.tmjepa_num_segments,
                learned_cv_dim=config.tmjepa_learned_cv_dim,
                condition_dim=config.tmjepa_condition_dim,
                auxiliary_dim=config.tmjepa_auxiliary_dim,
                memory_enabled=config.tmjepa_memory_enabled,
                observation_time_scale_ps=config.tmjepa_observation_time_scale_ps,
                base_step_ps=config.tmjepa_base_step_ps,
                max_rollout_steps=config.tmjepa_max_rollout_steps,
                ensemble_size=config.tmjepa_ensemble_size,
                diffusion_floor=config.tmjepa_diffusion_floor,
                diffusion_scale=config.tmjepa_diffusion_scale,
                relaxation_rate_scale=config.tmjepa_relaxation_rate_scale,
                coupling_scale=config.tmjepa_coupling_scale,
                equilibrium_noise_std=config.tmjepa_equilibrium_noise_std,
                contact_cutoff_angstrom=config.physchem_contact_cutoff,
                contact_width_angstrom=config.tmjepa_contact_width_angstrom,
            )
        elif config.tmjepa_enabled and config.tmjepa_version == 3:
            self.tmjepa = ThermodynamicMoriJEPAV3(
                latent_dim=config.latent_dim,
                num_segments=config.tmjepa_num_segments,
                learned_cv_dim=config.tmjepa_learned_cv_dim,
                condition_dim=config.tmjepa_condition_dim,
                auxiliary_dim=config.tmjepa_auxiliary_dim,
                diffusion_rank=config.tmjepa_diffusion_rank,
                memory_enabled=config.tmjepa_memory_enabled,
                coupled_bath_enabled=config.tmjepa_coupled_bath_enabled,
                observation_noise_enabled=(
                    config.tmjepa_observation_noise_enabled
                ),
                observation_noise_scale=config.tmjepa_observation_noise_scale,
                observation_noise_rank=config.tmjepa_observation_noise_rank,
                observation_noise_condition_scale=(
                    config.tmjepa_observation_noise_condition_scale
                ),
                observation_noise_temperature_exponent=(
                    config.tmjepa_observation_noise_temperature_exponent
                ),
                memory_intervention_enabled=(
                    config.tmjepa_memory_intervention_enabled
                ),
                reference_temperature_k=config.tmjepa_reference_temperature_k,
                activation_energy_kcal_per_mol=(
                    config.tmjepa_activation_energy_kcal_per_mol
                ),
                activation_deviation_kcal_per_mol=(
                    config.tmjepa_activation_deviation_kcal_per_mol
                ),
                state_mobility_scale=config.tmjepa_state_mobility_scale,
                memory_times_ps=config.tmjepa_memory_times_ps,
                base_step_ps=config.tmjepa_base_step_ps,
                max_rollout_steps=config.tmjepa_max_rollout_steps,
                ensemble_size=config.tmjepa_ensemble_size,
                diffusion_floor=config.tmjepa_diffusion_floor,
                learned_cv_diffusion_floor_scale=(
                    config.tmjepa_learned_cv_diffusion_floor_scale
                ),
                diffusion_scale=config.tmjepa_diffusion_scale,
                coupling_scale=config.tmjepa_coupling_scale,
                equilibrium_noise_std=config.tmjepa_equilibrium_noise_std,
                equilibrium_noise_samples=config.tmjepa_equilibrium_noise_samples,
                heat_capacity_scale_over_kb=(
                    config.tmjepa_heat_capacity_scale_over_kb
                ),
                endpoint_calibration_enabled=(
                    config.tmjepa_endpoint_calibration_enabled
                ),
                endpoint_calibration_scale=(
                    config.tmjepa_endpoint_calibration_scale
                ),
                endpoint_calibration_temperature_scale=(
                    config.tmjepa_endpoint_calibration_temperature_scale
                ),
                projected_force_enabled=config.tmjepa_projected_force_enabled,
                projected_force_hidden_dim=config.tmjepa_projected_force_hidden_dim,
                projected_force_max_shift=config.tmjepa_projected_force_max_shift,
                projected_force_history_features=(
                    config.tmjepa_projected_force_history_features
                ),
                projected_force_latent_history_features=(
                    config.tmjepa_projected_force_latent_history_features
                ),
                projected_force_horizon_gate_steps=(
                    config.tmjepa_projected_force_horizon_gate_steps
                ),
                learned_cv_burn_in_frames=(
                    config.tmjepa_learned_cv_burn_in_frames
                ),
                contact_cutoff_angstrom=config.physchem_contact_cutoff,
                contact_width_angstrom=config.tmjepa_contact_width_angstrom,
                cv_mean=config.tmjepa_cv_mean,
                cv_scale=config.tmjepa_cv_scale,
            )
        elif config.tmjepa_enabled:
            raise ValueError("tmjepa_version must be 2 or 3")
        else:
            self.tmjepa = None
        self.lspwm = (
            LSPWMProteinWorldModel(
                latent_dim=config.latent_dim,
                learned_cv_dim=config.lspwm_learned_cv_dim,
                num_segments=config.lspwm_num_segments,
                condition_dim=config.lspwm_condition_dim,
                auxiliary_dim=config.lspwm_auxiliary_dim,
                koopman_role_dim=config.lspwm_koopman_role_dim,
                msm_role_dim=config.lspwm_msm_role_dim,
                geometric_role_dim=config.lspwm_geometric_role_dim,
                auxiliary_role_dim=config.lspwm_auxiliary_role_dim,
                role_hidden_dim=config.lspwm_role_hidden_dim,
                reference_temperature_k=config.lspwm_reference_temperature_k,
                diffusion_scale=config.lspwm_diffusion_scale,
                relaxation_times_ps=config.lspwm_memory_times_ps,
                coupling_scale=config.lspwm_coupling_scale,
                contact_cutoff_angstrom=config.physchem_contact_cutoff,
                contact_width_angstrom=config.tmjepa_contact_width_angstrom,
                cv_mean=config.lspwm_cv_mean,
                cv_scale=config.lspwm_cv_scale,
                mean_force_bandwidth=config.lspwm_mean_force_bandwidth,
                history_closure_enabled=config.lspwm_history_closure_enabled,
                rollout_enabled=config.lspwm_rollout_enabled,
                rollout_steps=config.lspwm_rollout_steps,
                rollout_ensemble_size=config.lspwm_rollout_ensemble_size,
                rollout_warmup_steps=config.lspwm_rollout_warmup_steps,
                rollout_intermediate_supervision=(
                    config.lspwm_rollout_intermediate_supervision
                ),
                rollout_gaussian_nll_enabled=(
                    config.lspwm_rollout_gaussian_nll_enabled
                ),
                dynamic_nll_enabled=config.lspwm_dynamic_nll_enabled,
                force_loss_enabled=getattr(config, 'lspwm_force_loss_enabled', True),
                state_dependent_diffusion_enabled=getattr(config, 'lspwm_state_dependent_diffusion', False),
                potential_hidden_dim=getattr(config, 'lspwm_potential_hidden_dim', 128),
                potential_hidden_depth=getattr(config, 'lspwm_potential_hidden_depth', 4),
                physical_calibration_enabled=getattr(
                    config, "lspwm_physical_calibration_enabled", False
                ),
                physical_mobility_init=getattr(
                    config, "lspwm_physical_mobility_init", 1.0
                ),
                physical_free_energy_init=getattr(
                    config, "lspwm_physical_free_energy_init", 1.0
                ),
                physical_log_scale_min=getattr(
                    config, "lspwm_physical_log_scale_min", -12.0
                ),
                physical_log_scale_max=getattr(
                    config, "lspwm_physical_log_scale_max", 12.0
                ),
                conservative_correction_enabled=getattr(
                    config, "lspwm_conservative_correction_enabled", False
                ),
                conservative_correction_rank=getattr(
                    config, "lspwm_conservative_correction_rank", 16
                ),
                conservative_correction_scale=getattr(
                    config, "lspwm_conservative_correction_scale", 1.0
                ),
                conservative_correction_rescale_with_free_energy=getattr(
                    config,
                    "lspwm_conservative_correction_rescale_with_free_energy",
                    False,
                ),
                freeze_base_dynamics=getattr(
                    config, "lspwm_freeze_base_dynamics", False
                ),
                freeze_calibration_scales=getattr(
                    config, "lspwm_freeze_calibration_scales", False
                ),
            )
            if config.lspwm_enabled
            else None
        )
        if self.lspwm is not None and config.lspwm_freeze_learned_projection:
            for parameter in self.lspwm.learned_projection.parameters():
                parameter.requires_grad = False
            self.lspwm.learned_projection.eval()
        if self.tmjepa is not None:
            if config.tmjepa_freeze_learned_projection:
                projection = getattr(self.tmjepa, "learned_projection", None)
                if projection is None:
                    raise ValueError(
                        "tmjepa_freeze_learned_projection requires learned CVs"
                    )
                for parameter in projection.parameters():
                    parameter.requires_grad = False
            if config.tmjepa_freeze_potential:
                potential = getattr(self.tmjepa.dynamics, "potential", None)
                if potential is None:
                    raise ValueError("TM-JEPA V2 does not expose a separate potential")
                for parameter in potential.parameters():
                    parameter.requires_grad = False
                for parameter in self.tmjepa.condition_encoder.parameters():
                    parameter.requires_grad = False
            if config.tmjepa_freeze_kinetics:
                kinetics = getattr(self.tmjepa.dynamics, "kinetics", None)
                if kinetics is None:
                    raise ValueError("TM-JEPA V2 does not expose separate kinetics")
                for parameter in kinetics.parameters():
                    parameter.requires_grad = False
            kinetics = getattr(self.tmjepa.dynamics, "kinetics", None)
            if config.tmjepa_freeze_diffusion:
                if kinetics is None:
                    raise ValueError("TM-JEPA V2 does not expose separate kinetics")
                for module in (kinetics.mobility_head,):
                    for parameter in module.parameters():
                        parameter.requires_grad = False
                for parameter in (
                    kinetics.diffusion_basis,
                    kinetics.activation_energy_raw,
                    kinetics.diagonal_activation_deviation,
                    kinetics.mode_activation_deviation,
                ):
                    parameter.requires_grad = False
            if config.tmjepa_freeze_memory:
                if kinetics is None:
                    raise ValueError("TM-JEPA V2 does not expose separate kinetics")
                for parameter in kinetics.memory_head.parameters():
                    parameter.requires_grad = False
                kinetics.coupling_vectors.requires_grad = False
                for name in (
                    "memory_activation_energy_raw",
                    "memory_activation_deviation",
                    "memory_rotation_raw",
                ):
                    parameter = getattr(kinetics, name, None)
                    if parameter is not None:
                        parameter.requires_grad = False
            if config.tmjepa_freeze_observation_noise:
                if kinetics is None:
                    raise ValueError("TM-JEPA V2 does not expose separate kinetics")
                for name, parameter in kinetics.named_parameters():
                    if name.startswith("observation_noise_"):
                        parameter.requires_grad = False
            if config.tmjepa_freeze_endpoint_calibration:
                calibration = getattr(self.tmjepa, "endpoint_calibration", None)
                if calibration is not None:
                    for parameter in calibration.parameters():
                        parameter.requires_grad = False
        if config.freeze_v4_physics_representation and self.v4_physics_slow_model is not None:
            generator_prefixes = (
                "condition_trunk.",
                "equilibrium_condition_trunk.",
                "condition_embed.",
                "state_free_energy_head.",
                "symmetric_flux_head.",
            )
            for name, parameter in self.v4_physics_slow_model.named_parameters():
                parameter.requires_grad = name.startswith(generator_prefixes)
        if config.freeze_closed_local_operator and self.closed_local_operator is not None:
            for parameter in self.closed_local_operator.parameters():
                parameter.requires_grad = False
            self.closed_local_operator.eval()
        self.state_classifiers: nn.ModuleDict = nn.ModuleDict()
        if config.freeze_context_encoder:
            for param in self.context_encoder.parameters():
                param.requires_grad = False
            self.context_encoder.eval()
        if config.freeze_temporal_encoder:
            if self.temporal_encoder is None:
                raise ValueError("freeze_temporal_encoder requires a temporal encoder")
            for param in self.temporal_encoder.parameters():
                param.requires_grad = False
            self.temporal_encoder.eval()
        if config.freeze_offline_temporal_encoder and self.offline_temporal_encoder is not None:
            for param in self.offline_temporal_encoder.parameters():
                param.requires_grad = False
            self.offline_temporal_encoder.eval()
        self.target_encoder.eval()
        if self.target_temporal_encoder is not None:
            self.target_temporal_encoder.eval()
        if self.target_offline_temporal_encoder is not None:
            self.target_offline_temporal_encoder.eval()
        if self.target_slow_state_projector is not None:
            self.target_slow_state_projector.eval()
        if self.ls2_reversible_dynamics is not None:
            self.ls2_reversible_dynamics.target_assignment_head.eval()
        if self.e6_autoregressive_dynamics is not None:
            if config.e6_freeze_encoder:
                for param in self.context_encoder.parameters():
                    param.requires_grad = False
                self.context_encoder.eval()
                for param in self.temporal_encoder.parameters():
                    param.requires_grad = False
                self.temporal_encoder.eval()
            if config.e6_freeze_learned_state_projector:
                for param in self.e6_autoregressive_dynamics.state_projector.parameters():
                    param.requires_grad = False
            if config.e6_freeze_fixed_cv_dynamics and self.tmjepa is not None:
                for param in self.tmjepa.parameters():
                    param.requires_grad = False
        if config.lspwm_freeze_base_dynamics:
            if self.lspwm is None or not self.lspwm.physical_calibration_enabled:
                raise ValueError(
                    "lspwm_freeze_base_dynamics requires an enabled calibration head"
                )
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.lspwm.world_model.physical_calibration.parameters():
                parameter.requires_grad = True
            if config.lspwm_freeze_calibration_scales:
                calibration = self.lspwm.world_model.physical_calibration
                calibration.log_mobility_scale.requires_grad = False
                calibration.log_free_energy_scale.requires_grad = False

    def train(self, mode: bool = True) -> ProteinJEPA:
        super().train(mode)
        # EMA teachers are deterministic targets, not independently dropped-out views.
        self.target_encoder.eval()
        if self.config.freeze_context_encoder:
            self.context_encoder.eval()
        if self.config.freeze_temporal_encoder and self.temporal_encoder is not None:
            self.temporal_encoder.eval()
        if self.config.freeze_offline_temporal_encoder and self.offline_temporal_encoder is not None:
            self.offline_temporal_encoder.eval()
        if self.config.freeze_closed_local_operator and self.closed_local_operator is not None:
            self.closed_local_operator.eval()
        if self.target_temporal_encoder is not None:
            self.target_temporal_encoder.eval()
        if self.target_offline_temporal_encoder is not None:
            self.target_offline_temporal_encoder.eval()
        if self.target_slow_state_projector is not None:
            self.target_slow_state_projector.eval()
        if self.ls2_reversible_dynamics is not None:
            self.ls2_reversible_dynamics.target_assignment_head.eval()
        if self.config.freeze_v4_physics_representation and self.v4_physics_slow_model is not None:
            for module in (
                self.v4_physics_slow_model.frame_encoder,
                self.v4_physics_slow_model.slow_decoder,
                self.v4_physics_slow_model.assignment_head,
                self.v4_physics_slow_model.free_energy_head,
                self.v4_physics_slow_model.mobility_head,
            ):
                module.eval()
        return self

    def _transition_state_space(self, states: torch.Tensor) -> torch.Tensor:
        """Put state-transition inputs in the same space as EMA targets."""
        if not self.config.feature_norm:
            return states
        return F.layer_norm(states, (states.shape[-1],))

    @torch.no_grad()
    def sync_target_encoders(self) -> None:
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        if self.target_temporal_encoder is not None and self.temporal_encoder is not None:
            self.target_temporal_encoder.load_state_dict(self.temporal_encoder.state_dict())
        if self.target_offline_temporal_encoder is not None and self.offline_temporal_encoder is not None:
            self.target_offline_temporal_encoder.load_state_dict(self.offline_temporal_encoder.state_dict())
        if self.target_slow_state_projector is not None and self.slow_state_projector is not None:
            self.target_slow_state_projector.load_state_dict(self.slow_state_projector.state_dict())
        if self.ls2_reversible_dynamics is not None:
            self.ls2_reversible_dynamics.sync_target()
        if self.v51_contextual_dynamics is not None:
            self.v51_contextual_dynamics.sync_target_encoder()
        if self.tmjepa is not None:
            self.tmjepa.sync_target_projection()
        if self.lspwm is not None:
            self.lspwm.update_target_projection(0.0)
        self._curriculum_step = 0

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        # A frozen-latent downstream stage must preserve the exact teacher
        # representation used to construct its paired targets. Updating the
        # EMA target here would silently change the prediction problem while
        # the online encoder remains frozen.
        if self.config.freeze_target_encoders:
            return
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_temporal_encoder is not None and self.temporal_encoder is not None:
            for target, context in zip(self.target_temporal_encoder.parameters(), self.temporal_encoder.parameters()):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_offline_temporal_encoder is not None and self.offline_temporal_encoder is not None:
            for target, context in zip(self.target_offline_temporal_encoder.parameters(), self.offline_temporal_encoder.parameters()):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_slow_state_projector is not None and self.slow_state_projector is not None:
            for target, context in zip(self.target_slow_state_projector.parameters(), self.slow_state_projector.parameters()):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.ls2_reversible_dynamics is not None:
            self.ls2_reversible_dynamics.update_target(momentum)
        if self.v51_contextual_dynamics is not None:
            self.v51_contextual_dynamics.update_target_encoder(momentum)
        if self.tmjepa is not None:
            self.tmjepa.update_target_projection(momentum)
        if self.lspwm is not None:
            self.lspwm.update_target_projection(momentum)

    def forward(self, batch: dict, context_mask: torch.Tensor, target_mask: torch.Tensor) -> dict:
        coords = batch["coords"]
        residue_mask = batch["residue_mask"]
        aatype = batch["aatype"]
        residue_index = batch["residue_index"]
        chain_id = batch["chain_id"]
        time_ps = batch.get("time_ps")
        temperature_k = batch.get("temperature_k")
        dataset_index = batch.get("dataset_index")
        context_latents = self.context_encoder(
            coords=coords,
            residue_mask=residue_mask,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
            token_mask=context_mask,
        )
        with torch.no_grad():
            target_latents = self.target_encoder(
                coords=coords,
                residue_mask=residue_mask,
                aatype=aatype,
                residue_index=residue_index,
                chain_id=chain_id,
                token_mask=residue_mask,
            )
            if self.config.feature_norm:
                target_latents = F.layer_norm(target_latents, (target_latents.shape[-1],))
        predictions = self.predictor(
            context_latents=context_latents,
            context_mask=context_mask,
            target_mask=target_mask,
            residue_index=residue_index,
            chain_id=chain_id,
            time_ps=time_ps,
        )
        outputs = {
            "predictions": predictions,
            "targets": target_latents,
            "target_mask": target_mask,
            "context_latents": context_latents,
        }
        if self.temporal_encoder is not None:
            context_states = self.temporal_encoder(
                context_latents,
                context_mask,
                time_ps=time_ps,
                temperature_k=temperature_k,
                dataset_index=dataset_index,
            )
            outputs.update(context_states)
            with torch.no_grad():
                target_states = self.target_temporal_encoder(
                    target_latents,
                    residue_mask,
                    time_ps=time_ps,
                    temperature_k=temperature_k,
                    dataset_index=dataset_index,
                )
                target_frame_states = target_states["frame_states"]
                if self.config.feature_norm:
                    target_frame_states = F.layer_norm(target_frame_states, (target_frame_states.shape[-1],))
            if self.offline_temporal_encoder is not None:
                offline_states = self.offline_temporal_encoder(
                    context_latents,
                    context_mask,
                    time_ps=time_ps,
                    temperature_k=temperature_k,
                    dataset_index=dataset_index,
                )
                with torch.no_grad():
                    target_offline_states = self.target_offline_temporal_encoder(
                        target_latents,
                        residue_mask,
                        time_ps=time_ps,
                        temperature_k=temperature_k,
                        dataset_index=dataset_index,
                    )
                    offline_target_frame_states = target_offline_states["frame_states"]
                    if self.config.feature_norm:
                        offline_target_frame_states = F.layer_norm(
                            offline_target_frame_states, (offline_target_frame_states.shape[-1],)
                        )
                outputs["causal_frame_states"] = context_states["frame_states"]
                outputs["causal_target_frame_states"] = target_frame_states
                outputs["offline_frame_states"] = offline_states["frame_states"]
                outputs["offline_target_frame_states"] = offline_target_frame_states
                outputs["offline_clip_state"] = offline_states["clip_state"]
                context_states = offline_states
                target_states = target_offline_states
                target_frame_states = offline_target_frame_states
                outputs.update({"frame_states": context_states["frame_states"], "clip_state": context_states["clip_state"]})
            context_frame_mask = context_mask.any(dim=2)
            target_frame_mask = target_mask.any(dim=2)
            state_predictions = self.state_predictor(
                context_states=context_states["frame_states"],
                context_frame_mask=context_frame_mask,
                target_frame_mask=target_frame_mask,
                time_ps=time_ps,
            )
            outputs.update(
                {
                    "state_predictions": state_predictions,
                    "state_targets": target_frame_states,
                    "state_target_mask": target_frame_mask,
                    "context_frame_mask": context_frame_mask,
                    "target_frame_states": target_frame_states,
                }
            )
            if self.lagged_state_predictor is not None:
                transition_source = context_states["frame_states"]
                if self.config.transition_source == "motion":
                    if "motion_states" not in context_states:
                        raise ValueError("transition_source=motion requires model.motion_depth > 0")
                    transition_source = context_states["motion_states"]
                elif self.config.transition_source == "state_motion":
                    if "state_motion_states" not in context_states:
                        raise ValueError("transition_source=state_motion requires model.motion_depth > 0")
                    transition_source = context_states["state_motion_states"]
                elif self.config.transition_source == "state":
                    # feature_norm is applied to EMA targets below. Normalize the
                    # online source as well so the lag operator is trained in one
                    # closed coordinate system.
                    transition_source = self._transition_state_space(transition_source)
                else:
                    raise ValueError(f"unknown transition source {self.config.transition_source}")
                outputs["lagged_state_predictions"] = self.lagged_state_predictor(
                    transition_source,
                    time_ps=time_ps,
                    temperature_k=temperature_k,
                    dataset_index=dataset_index,
                )
                outputs["frame_valid_mask"] = residue_mask.any(dim=2)
                outputs["transition_states"] = transition_source
                outputs["transition_source_is_state_space"] = self.config.transition_source != "motion"
                lagged_state_targets = target_frame_states
                if self.config.transition_source == "state_motion":
                    if "state_motion_states" not in target_states:
                        raise ValueError("state_motion transition requires target motion states")
                    lagged_state_targets = target_states["state_motion_states"]
                    outputs["target_state_motion_states"] = lagged_state_targets
                outputs["lagged_state_targets"] = lagged_state_targets
                outputs["time_ps"] = time_ps
                outputs["temperature_k"] = temperature_k
                outputs["dataset_index"] = dataset_index
            if self.linear_dynamics_predictor is not None:
                outputs["linear_dynamics_predictions"] = self.linear_dynamics_predictor(context_states["frame_states"])
                outputs["frame_valid_mask"] = residue_mask.any(dim=2)
            if self.physchem_predictor is not None:
                outputs["physchem_predictions"] = self.physchem_predictor(context_states["frame_states"])
                targets, target_masks = compute_physchem_targets(
                    coords=coords,
                    atom_mask=batch["atom_mask"],
                    residue_mask=residue_mask,
                    residue_index=residue_index,
                    chain_id=chain_id,
                    lags=self.config.physchem_prediction_lags,
                    contact_cutoff=self.config.physchem_contact_cutoff,
                )
                outputs["physchem_targets"] = targets
                outputs["physchem_target_masks"] = target_masks
                outputs["frame_valid_mask"] = residue_mask.any(dim=2)
            if self.kinetic_readout is not None:
                kinetic_states = (
                    context_states["frame_states"].detach()
                    if self.config.kinetic_readout_stopgrad
                    else context_states["frame_states"]
                )
                outputs["kinetic_readout"] = self.kinetic_readout(kinetic_states)
                outputs["kinetic_readout_mask"] = context_frame_mask
        return outputs

    def encode_states(
        self,
        residue_latents: torch.Tensor,
        residue_mask: torch.Tensor,
        time_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return frame-level and clip-level state latents for dynamics readouts."""
        encoder = self.offline_temporal_encoder if self.offline_temporal_encoder is not None else self.temporal_encoder
        if encoder is not None:
            return encoder(
                residue_latents,
                residue_mask,
                time_ps=time_ps,
                temperature_k=temperature_k,
                dataset_index=dataset_index,
            )
        valid = residue_mask.unsqueeze(-1).float()
        frame_states = (residue_latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        clip_valid = residue_mask.any(dim=2).unsqueeze(-1).float()
        clip_state = (frame_states * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)
        return {"frame_states": frame_states, "clip_state": clip_state}

    def encode_causal_states(
        self,
        residue_latents: torch.Tensor,
        residue_mask: torch.Tensor,
        time_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.temporal_encoder is None:
            return self.encode_states(
                residue_latents,
                residue_mask,
                time_ps=time_ps,
                temperature_k=temperature_k,
                dataset_index=dataset_index,
            )
        return self.temporal_encoder(
            residue_latents,
            residue_mask,
            time_ps=time_ps,
            temperature_k=temperature_k,
            dataset_index=dataset_index,
        )

    @staticmethod
    def encode_endpoint_state(states: dict[str, torch.Tensor], frame_mask: torch.Tensor) -> torch.Tensor:
        """Select the last valid causal frame from a temporal state sequence."""
        frame_index = torch.arange(frame_mask.shape[1], device=frame_mask.device).view(1, -1)
        last = frame_mask.long().mul(frame_index).max(dim=1).values
        batch_index = torch.arange(frame_mask.shape[0], device=frame_mask.device)
        return states["frame_states"][batch_index, last]

    def forward_tmjepa(self, batch: dict) -> dict[str, torch.Tensor]:
        """Fast fixed-CV path that does not evaluate unused JEPA encoders."""
        if self.tmjepa is None:
            raise ValueError("TM-JEPA is not enabled")
        if self.tmjepa.learned_cv_dim:
            return self.forward_paired(batch)
        outputs = self.tmjepa(
            context_coords=batch["context_coords"],
            future_coords=batch["future_coords"],
            context_atom_mask=batch["context_atom_mask"],
            future_atom_mask=batch["future_atom_mask"],
            context_residue_mask=batch["context_residue_mask"],
            future_residue_mask=batch["future_residue_mask"],
            aatype=batch["context_aatype"],
            residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
            context_time_ps=batch["context_time_ps"],
            horizon_time_ps=batch["horizon_time_ps"],
            temperature_k=batch.get("temperature_k"),
            dataset_index=batch.get("dataset_index"),
        )
        current = outputs["tmjepa_current_cv"]
        target = outputs["tmjepa_target_cv"]
        prediction = outputs["tmjepa_prediction"]
        outputs.update(
            {
                "context_endpoint": current,
                "context_teacher_endpoint": current,
                "persistence_endpoint": current,
                "future_endpoint": target,
                "delta_target": target - current,
                "delta_prediction": prediction - current,
                "state_prediction": prediction,
                "horizon_frames": batch["horizon_frames"],
                "horizon_time_ps": batch["horizon_time_ps"],
                "transition_head_source": current,
            }
        )
        if "context_group_index" in batch:
            outputs["context_group_index"] = batch["context_group_index"]
        return outputs

    def forward_paired(self, batch: dict) -> dict[str, torch.Tensor]:
        """Encode paired windows for endpoint forecasting and slow dynamics."""
        if self.temporal_encoder is None or self.paired_delta_predictor is None:
            raise ValueError("paired forecast requires temporal_encoder and paired_delta_predictor")

        context_latents = self.context_encoder(
            coords=batch["context_coords"],
            residue_mask=batch["context_residue_mask"],
            aatype=batch["context_aatype"],
            residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
            token_mask=batch["context_residue_mask"],
        )
        causal_scale = float(self.config.causal_encoder_gradient_scale)
        if not 0.0 <= causal_scale <= 1.0:
            raise ValueError("causal_encoder_gradient_scale must be in [0, 1]")
        causal_context_latents = context_latents.detach() + causal_scale * (
            context_latents - context_latents.detach()
        )
        context_states = self.temporal_encoder(
            causal_context_latents,
            batch["context_residue_mask"],
            time_ps=batch.get("context_time_ps"),
            temperature_k=batch.get("temperature_k"),
            dataset_index=batch.get("dataset_index"),
        )
        offline_context_states = None
        if self.offline_temporal_encoder is not None:
            offline_context_states = self.offline_temporal_encoder(
                context_latents,
                batch["context_residue_mask"],
                time_ps=batch.get("context_time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
        with torch.no_grad():
            teacher_context_latents = self.target_encoder(
                coords=batch["context_coords"],
                residue_mask=batch["context_residue_mask"],
                aatype=batch["context_aatype"],
                residue_index=batch["context_residue_index"],
                chain_id=batch["context_chain_id"],
                token_mask=batch["context_residue_mask"],
            )
            future_latents = self.target_encoder(
                coords=batch["future_coords"],
                residue_mask=batch["future_residue_mask"],
                aatype=batch["future_aatype"],
                residue_index=batch["future_residue_index"],
                chain_id=batch["future_chain_id"],
                token_mask=batch["future_residue_mask"],
            )
            if self.config.feature_norm:
                teacher_context_latents = F.layer_norm(
                    teacher_context_latents, (teacher_context_latents.shape[-1],)
                )
                future_latents = F.layer_norm(future_latents, (future_latents.shape[-1],))
            teacher_context_states = self.target_temporal_encoder(
                teacher_context_latents,
                batch["context_residue_mask"],
                time_ps=batch.get("context_time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            future_states = self.target_temporal_encoder(
                future_latents,
                batch["future_residue_mask"],
                time_ps=batch.get("future_time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            causal_teacher_context_endpoint = self.encode_endpoint_state(
                teacher_context_states, batch["context_residue_mask"].any(dim=2)
            )
            causal_future_endpoint = self.encode_endpoint_state(
                future_states, batch["future_residue_mask"].any(dim=2)
            )
            if self.config.feature_norm:
                causal_teacher_context_endpoint = F.layer_norm(
                    causal_teacher_context_endpoint, (causal_teacher_context_endpoint.shape[-1],)
                )
                causal_future_endpoint = F.layer_norm(
                    causal_future_endpoint, (causal_future_endpoint.shape[-1],)
                )
            teacher_offline_context_endpoint = None
            future_offline_endpoint = None
            teacher_offline_context_states = None
            future_offline_states = None
            if self.target_offline_temporal_encoder is not None:
                teacher_offline_context_states = self.target_offline_temporal_encoder(
                    teacher_context_latents,
                    batch["context_residue_mask"],
                    time_ps=batch.get("context_time_ps"),
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )
                future_offline_states = self.target_offline_temporal_encoder(
                    future_latents,
                    batch["future_residue_mask"],
                    time_ps=batch.get("future_time_ps"),
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )
                teacher_offline_context_endpoint = self.encode_endpoint_state(
                    teacher_offline_context_states, batch["context_residue_mask"].any(dim=2)
                )
                future_offline_endpoint = self.encode_endpoint_state(
                    future_offline_states, batch["future_residue_mask"].any(dim=2)
                )
                if self.config.feature_norm:
                    teacher_offline_context_endpoint = F.layer_norm(
                        teacher_offline_context_endpoint, (teacher_offline_context_endpoint.shape[-1],)
                    )
                    future_offline_endpoint = F.layer_norm(
                        future_offline_endpoint, (future_offline_endpoint.shape[-1],)
                    )

            if self.config.paired_target_tower == "offline":
                context_teacher_endpoint = teacher_offline_context_endpoint
                future_endpoint = future_offline_endpoint
            else:
                context_teacher_endpoint = causal_teacher_context_endpoint
                future_endpoint = causal_future_endpoint

        context_endpoint = self.encode_endpoint_state(
            context_states, batch["context_residue_mask"].any(dim=2)
        )
        transition_source = self._transition_state_space(context_endpoint)
        if self.config.paired_teacher_space_delta:
            persistence_endpoint = context_teacher_endpoint
            delta_anchor = context_teacher_endpoint
            prediction_anchor = context_teacher_endpoint
        else:
            persistence_endpoint = transition_source
            delta_anchor = transition_source.detach()
            prediction_anchor = transition_source
        delta_target = future_endpoint - delta_anchor
        transition_head_source = transition_source
        if self.config.history_conditioned_transition:
            if offline_context_states is None:
                raise ValueError("history-conditioned transition requires offline context states")
            delta_prediction, transition_head_source = self.paired_delta_predictor(
                offline_context_states["frame_states"],
                batch["context_residue_mask"].any(dim=2),
                batch["horizon_time_ps"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
        else:
            delta_prediction = self.paired_delta_predictor(
                transition_source,
                batch["horizon_time_ps"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
        outputs = {
            "context_endpoint": transition_source,
            "context_teacher_endpoint": context_teacher_endpoint,
            "persistence_endpoint": persistence_endpoint,
            "future_endpoint": future_endpoint,
            "delta_target": delta_target,
            "delta_prediction": delta_prediction,
            "state_prediction": prediction_anchor + delta_prediction,
            "horizon_frames": batch["horizon_frames"],
            "horizon_time_ps": batch["horizon_time_ps"],
            "transition_head_source": transition_head_source,
        }
        if "context_group_index" in batch:
            outputs["context_group_index"] = batch["context_group_index"]
        if self.tmjepa is not None:
            if self.config.tmjepa_representation_only:
                outputs.update(
                    self.tmjepa.encode_learned_context(
                        context_states["frame_states"],
                        batch["context_residue_mask"].any(dim=2),
                    )
                )
            else:
                outputs.update(
                    self.tmjepa(
                        context_coords=batch["context_coords"],
                        future_coords=batch["future_coords"],
                        context_atom_mask=batch["context_atom_mask"],
                        future_atom_mask=batch["future_atom_mask"],
                        context_residue_mask=batch["context_residue_mask"],
                        future_residue_mask=batch["future_residue_mask"],
                        aatype=batch["context_aatype"],
                        residue_index=batch["context_residue_index"],
                        chain_id=batch["context_chain_id"],
                        context_time_ps=batch["context_time_ps"],
                        horizon_time_ps=batch["horizon_time_ps"],
                        temperature_k=batch.get("temperature_k"),
                        dataset_index=batch.get("dataset_index"),
                        context_frame_states=context_states["frame_states"],
                        future_frame_states=future_states["frame_states"],
                    )
                )
        if self.lspwm is not None:
            outputs.update(
                self.lspwm(
                    context_coords=batch["context_coords"],
                    future_coords=batch["future_coords"],
                    context_atom_mask=batch["context_atom_mask"],
                    future_atom_mask=batch["future_atom_mask"],
                    context_residue_mask=batch["context_residue_mask"],
                    future_residue_mask=batch["future_residue_mask"],
                    context_chain_id=batch["context_chain_id"],
                    future_chain_id=batch["future_chain_id"],
                    context_residue_index=batch["context_residue_index"],
                    future_residue_index=batch["future_residue_index"],
                    context_frame_states=context_states["frame_states"],
                    future_frame_states=future_states["frame_states"],
                    context_time_ps=batch["context_time_ps"],
                    horizon_time_ps=batch["horizon_time_ps"],
                    aatype=batch["context_aatype"],
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                    curriculum_step=self._curriculum_step if hasattr(self, '_curriculum_step') else None,
                    curriculum_steps=self.config.lspwm_curriculum_steps if hasattr(self.config, 'lspwm_curriculum_steps') else None,
                )
            )
        if self.v5_contextual_dynamics is not None:
            outputs.update(
                self.v5_contextual_dynamics(
                    context_states["frame_states"],
                    future_states["frame_states"],
                    batch["context_residue_mask"].any(dim=2),
                    batch["future_residue_mask"].any(dim=2),
                    batch["horizon_time_ps"],
                    batch["horizon_frames"],
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )
            )
            outputs["v5_observable_target"] = compute_v3_paired_observable_targets(
                batch["context_coords"], batch["future_coords"],
                batch["context_atom_mask"], batch["future_atom_mask"],
                batch["context_residue_mask"], batch["future_residue_mask"],
                batch["context_chain_id"], batch["context_residue_index"],
                contact_cutoff=self.config.physchem_contact_cutoff,
            )
        if self.v51_contextual_dynamics is not None:
            context_local_mask = batch["context_residue_mask"].unsqueeze(-1).to(
                dtype=causal_context_latents.dtype
            )
            context_local_states = (
                causal_context_latents * context_local_mask
            ).sum(dim=2) / context_local_mask.sum(dim=2).clamp(min=1.0)
            teacher_local_mask = batch["context_residue_mask"].unsqueeze(-1).to(
                dtype=teacher_context_latents.dtype
            )
            teacher_context_local_states = (
                teacher_context_latents * teacher_local_mask
            ).sum(dim=2) / teacher_local_mask.sum(dim=2).clamp(min=1.0)
            future_local_mask = batch["future_residue_mask"].unsqueeze(-1).to(
                dtype=future_latents.dtype
            )
            future_local_states = (
                future_latents * future_local_mask
            ).sum(dim=2) / future_local_mask.sum(dim=2).clamp(min=1.0)
            context_physical_states = None
            future_physical_states = None
            if self.config.v53_physical_state_anchor_enabled:
                context_physical, context_physical_mask = compute_v3_closed_local_state_sequence(
                    batch["context_coords"],
                    batch["context_atom_mask"],
                    batch["context_residue_mask"],
                    batch["context_chain_id"],
                    batch["context_residue_index"],
                    num_tokens=self.config.v53_physical_state_tokens,
                    contact_cutoff=self.config.physchem_contact_cutoff,
                )
                future_physical, future_physical_mask = compute_v3_closed_local_state_sequence(
                    batch["future_coords"],
                    batch["future_atom_mask"],
                    batch["future_residue_mask"],
                    batch["future_chain_id"],
                    batch["future_residue_index"],
                    num_tokens=self.config.v53_physical_state_tokens,
                    contact_cutoff=self.config.physchem_contact_cutoff,
                )
                context_physical_states = (
                    context_physical * context_physical_mask.unsqueeze(-1)
                ).flatten(start_dim=2)
                future_physical_states = (
                    future_physical * future_physical_mask.unsqueeze(-1)
                ).flatten(start_dim=2)
            v51_observable_target = compute_v3_paired_observable_targets(
                batch["context_coords"], batch["future_coords"],
                batch["context_atom_mask"], batch["future_atom_mask"],
                batch["context_residue_mask"], batch["future_residue_mask"],
                batch["context_chain_id"], batch["context_residue_index"],
                contact_cutoff=self.config.physchem_contact_cutoff,
            )
            wrong_context_frame_states = None
            wrong_context_frame_mask = None
            short_context_frame_states = None
            if self.config.v51_training_controls:
                time_length = causal_context_latents.shape[1]
                permutation = torch.arange(time_length, device=causal_context_latents.device)
                permutation[:-1] = permutation[:-1].flip(0)
                if self.config.v53_frame_local_target_enabled:
                    # Let dynamics reverse its fused learned/physical history.
                    wrong_context_frame_states = None
                else:
                    wrong_context_states = self.temporal_encoder(
                        causal_context_latents[:, permutation],
                        batch["context_residue_mask"][:, permutation],
                        time_ps=batch.get("context_time_ps"),
                        temperature_k=batch.get("temperature_k"),
                        dataset_index=batch.get("dataset_index"),
                    )
                    wrong_context_frame_states = wrong_context_states["frame_states"]
                if self.config.v51_history_truncation_enabled:
                    keep = max(1, int(round(time_length * self.config.v51_history_keep_fraction)))
                    wrong_context_frame_mask = batch["context_residue_mask"].any(dim=2).clone()
                    wrong_context_frame_mask[:, :-keep] = False
                    if self.config.v53_frame_local_target_enabled:
                        short_context_frame_states = None
                    else:
                        short_context_states = self.temporal_encoder(
                            causal_context_latents,
                            wrong_context_frame_mask.unsqueeze(-1).expand_as(batch["context_residue_mask"]),
                            time_ps=batch.get("context_time_ps"),
                            temperature_k=batch.get("temperature_k"),
                            dataset_index=batch.get("dataset_index"),
                        )
                        short_context_frame_states = short_context_states["frame_states"]
            outputs.update(
                self.v51_contextual_dynamics(
                    context_states["frame_states"],
                    teacher_context_states["frame_states"],
                    future_states["frame_states"],
                    batch["context_residue_mask"].any(dim=2),
                    batch["future_residue_mask"].any(dim=2),
                    batch.get("context_time_ps"),
                    batch.get("future_time_ps"),
                    batch["horizon_time_ps"],
                    batch["horizon_frames"],
                    context_local_states=context_local_states,
                    teacher_context_local_states=teacher_context_local_states,
                    future_local_states=future_local_states,
                    context_physical_states=context_physical_states,
                    future_physical_states=future_physical_states,
                    observable_target=v51_observable_target,
                    wrong_context_states=wrong_context_frame_states,
                    wrong_context_mask=wrong_context_frame_mask,
                    short_context_states=short_context_frame_states,
                    context_group_index=batch.get("context_group_index"),
                )
            )
            outputs["v51_observable_target"] = v51_observable_target
        if self.gaussian_transition is not None:
            gaussian_mean, gaussian_log_sigma = self.gaussian_transition(
                transition_source,
                batch["horizon_time_ps"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            outputs["gaussian_delta_prediction"] = gaussian_mean
            outputs["gaussian_log_sigma"] = gaussian_log_sigma
            outputs["delta_prediction"] = gaussian_mean
            outputs["state_prediction"] = prediction_anchor + gaussian_mean
        if offline_context_states is not None:
            offline_context_endpoint = self._transition_state_space(
                self.encode_endpoint_state(offline_context_states, batch["context_residue_mask"].any(dim=2))
            )
            outputs["offline_context_endpoint"] = offline_context_endpoint
            outputs["offline_context_teacher_endpoint"] = teacher_offline_context_endpoint
            outputs["offline_future_endpoint"] = future_offline_endpoint
            if self.offline_paired_delta_predictor is not None and future_offline_endpoint is not None:
                offline_delta_prediction = self.offline_paired_delta_predictor(
                    offline_context_endpoint,
                    batch["horizon_time_ps"],
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )
                if self.config.paired_teacher_space_delta:
                    offline_persistence_endpoint = teacher_offline_context_endpoint
                    offline_delta_anchor = teacher_offline_context_endpoint
                    offline_prediction_anchor = teacher_offline_context_endpoint
                else:
                    offline_persistence_endpoint = offline_context_endpoint
                    offline_delta_anchor = offline_context_endpoint.detach()
                    offline_prediction_anchor = offline_context_endpoint
                outputs["offline_delta_prediction"] = offline_delta_prediction
                outputs["offline_persistence_endpoint"] = offline_persistence_endpoint
                outputs["offline_delta_target"] = future_offline_endpoint - offline_delta_anchor
                outputs["offline_state_prediction"] = offline_prediction_anchor + offline_delta_prediction
        if self.observable_head is not None:
            outputs["observable_prediction"] = self.observable_head(
                transition_head_source,
                batch["horizon_time_ps"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            observable_target_fn = (
                compute_v3_paired_observable_targets
                if self.config.observable_target_version == "v3"
                else compute_paired_observable_targets
            )
            outputs["observable_target"] = observable_target_fn(
                batch["context_coords"], batch["future_coords"],
                batch["context_atom_mask"], batch["future_atom_mask"],
                batch["context_residue_mask"], batch["future_residue_mask"],
                batch["context_chain_id"], batch["context_residue_index"],
                contact_cutoff=self.config.physchem_contact_cutoff,
            )
        if self.metastable_state_head is not None:
            context_metastable_source = (
                offline_context_states["frame_states"]
                if offline_context_states is not None
                else context_states["frame_states"]
            )
            future_metastable_source = (
                future_offline_states["frame_states"]
                if future_offline_states is not None
                else future_states["frame_states"]
            )
            outputs["context_metastable_states"] = self.metastable_state_head(context_metastable_source)
            outputs["future_metastable_states"] = self.metastable_state_head(future_metastable_source)
            outputs["context_metastable_mask"] = batch["context_residue_mask"].any(dim=2)
            outputs["future_metastable_mask"] = batch["future_residue_mask"].any(dim=2)
        if self.local_dynamics_head is not None:
            if self.config.local_observable_targets:
                local_tokens, local_prediction, local_mask = self.local_dynamics_head(
                    context_latents[:, -1],
                    batch["context_residue_mask"][:, -1],
                    transition_head_source.detach(),
                    batch["horizon_time_ps"],
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )
                local_target, local_target_mask = compute_v3_local_observable_targets(
                    batch["context_coords"], batch["future_coords"],
                    batch["context_atom_mask"], batch["future_atom_mask"],
                    batch["context_residue_mask"], batch["future_residue_mask"],
                    batch["context_chain_id"], batch["context_residue_index"],
                    num_tokens=self.config.local_dynamics_tokens,
                    contact_cutoff=self.config.physchem_contact_cutoff,
                )
                outputs["local_observable_prediction"] = local_prediction
                outputs["local_observable_target"] = local_target
                outputs["local_observable_mask"] = local_mask & local_target_mask
            else:
                if self.config.closed_local_operator_mode == "compositional_rollout":
                    B, T, L, D = context_latents.shape
                    local_sequence, local_sequence_mask = self.local_dynamics_head.encode_tokens(
                        context_latents.reshape(B * T, L, D),
                        batch["context_residue_mask"].reshape(B * T, L),
                    )
                    local_sequence = local_sequence.reshape(
                        B, T, self.config.local_dynamics_tokens, self.config.local_dynamics_dim
                    )
                    local_sequence_mask = local_sequence_mask.reshape(
                        B, T, self.config.local_dynamics_tokens
                    )
                    local_tokens = local_sequence[:, -1]
                    local_mask = local_sequence_mask[:, -1]
                else:
                    local_tokens, local_mask = self.local_dynamics_head.encode_tokens(
                        context_latents[:, -1], batch["context_residue_mask"][:, -1]
                    )
            outputs["local_dynamics_tokens"] = local_tokens
            if self.closed_local_operator is not None:
                context_closed_sequence = None
                context_closed_sequence_mask = None
                if self.config.closed_local_operator_mode == "compositional_rollout":
                    context_closed_sequence, context_closed_sequence_mask = (
                        compute_v3_closed_local_state_sequence(
                            batch["context_coords"], batch["context_atom_mask"],
                            batch["context_residue_mask"], batch["context_chain_id"],
                            batch["context_residue_index"],
                            num_tokens=self.config.local_dynamics_tokens,
                        )
                    )
                    context_closed_state = context_closed_sequence[:, -1]
                    context_closed_mask = context_closed_sequence_mask[:, -1]
                else:
                    context_closed_state, context_closed_mask = compute_v3_closed_local_state(
                        batch["context_coords"], batch["context_atom_mask"], batch["context_residue_mask"],
                        batch["context_chain_id"], batch["context_residue_index"],
                        num_tokens=self.config.local_dynamics_tokens,
                    )
                future_closed_sequence = None
                future_closed_sequence_mask = None
                if self.v4_slow_kinetic_model is not None or self.v4_physics_slow_model is not None:
                    future_closed_sequence, future_closed_sequence_mask = (
                        compute_v3_closed_local_state_sequence(
                            batch["future_coords"], batch["future_atom_mask"],
                            batch["future_residue_mask"], batch["future_chain_id"],
                            batch["future_residue_index"],
                            num_tokens=self.config.local_dynamics_tokens,
                        )
                    )
                    future_closed_state = future_closed_sequence[:, -1]
                    future_closed_mask = future_closed_sequence_mask[:, -1]
                else:
                    future_closed_state, future_closed_mask = compute_v3_closed_local_state(
                        batch["future_coords"], batch["future_atom_mask"], batch["future_residue_mask"],
                        batch["future_chain_id"], batch["future_residue_index"],
                        num_tokens=self.config.local_dynamics_tokens,
                    )
                closed_velocity = None
                closed_velocity_mask = context_closed_mask
                if self.config.closed_local_operator_mode == "velocity_probabilistic":
                    closed_velocity, closed_velocity_mask = compute_v3_closed_local_velocity(
                        batch["context_coords"], batch["context_atom_mask"],
                        batch["context_residue_mask"], batch["context_chain_id"],
                        batch["context_residue_index"], context_closed_state,
                        self.closed_local_operator.state_scale,
                        strides=self.config.closed_local_velocity_strides,
                        num_tokens=self.config.local_dynamics_tokens,
                    )
                operator_outputs = self.closed_local_operator(
                    local_tokens.detach(), context_closed_state, transition_head_source.detach(),
                    batch["horizon_frames"], batch["horizon_time_ps"],
                    current_velocity=closed_velocity,
                    context_state_sequence=context_closed_sequence,
                    context_sequence_mask=context_closed_sequence_mask,
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )
                closed_prediction, closed_representation = operator_outputs[:2]
                outputs["context_closed_local_state"] = context_closed_state
                outputs["future_closed_local_state"] = future_closed_state
                outputs["closed_local_prediction"] = closed_prediction
                outputs["closed_local_representation"] = closed_representation
                outputs["closed_local_mask"] = (
                    local_mask & context_closed_mask & future_closed_mask & closed_velocity_mask
                )
                outputs["closed_local_state_scale"] = self.closed_local_operator.state_scale
                outputs["closed_local_state_mean"] = self.closed_local_operator.state_mean
                if context_closed_sequence is not None:
                    outputs["context_closed_local_sequence"] = context_closed_sequence
                    base_step = self.config.closed_local_base_step_frames
                    frame_stride = self.config.closed_local_history_frame_stride
                    if base_step % frame_stride != 0:
                        raise ValueError("closed local base step must be divisible by history frame stride")
                    lag = base_step // frame_stride
                    if lag >= context_closed_sequence.shape[1]:
                        raise ValueError("context window is too short for the closed local base step")
                    base_index = context_closed_sequence.shape[1] - 1 - lag
                    base_time = batch["context_time_ps"][:, -1] - batch["context_time_ps"][:, base_index]
                    base_horizon = torch.full_like(batch["horizon_frames"], base_step)
                    base_history_source = (
                        offline_context_states["frame_states"][:, base_index]
                        if offline_context_states is not None
                        else context_states["frame_states"][:, base_index]
                    )
                    base_outputs = self.closed_local_operator(
                        local_sequence[:, base_index].detach(),
                        context_closed_sequence[:, base_index], base_history_source.detach(),
                        base_horizon, base_time,
                        context_state_sequence=context_closed_sequence[:, :base_index + 1],
                        context_sequence_mask=context_closed_sequence_mask[:, :base_index + 1],
                        temperature_k=batch.get("temperature_k"),
                        dataset_index=batch.get("dataset_index"),
                    )
                    outputs["closed_local_base_prediction"] = base_outputs[0]
                    outputs["closed_local_base_current"] = context_closed_sequence[:, base_index]
                    outputs["closed_local_base_target"] = context_closed_sequence[:, -1]
                    outputs["closed_local_base_mask"] = (
                        local_sequence_mask[:, base_index]
                        & context_closed_sequence_mask[:, base_index]
                        & context_closed_sequence_mask[:, -1]
                    )
                    if not self.training:
                        fixed_predictions = []
                        fixed_step_time = batch["horizon_time_ps"] / (
                            batch["horizon_frames"].to(dtype=batch["horizon_time_ps"].dtype)
                            / float(base_step)
                        )
                        for fixed_horizon in self.config.closed_local_horizons:
                            fixed_frames = torch.full_like(batch["horizon_frames"], fixed_horizon)
                            fixed_time = fixed_step_time * (fixed_horizon / float(base_step))
                            fixed_predictions.append(
                                self.closed_local_operator(
                                    local_tokens.detach(), context_closed_state,
                                    transition_head_source.detach(), fixed_frames, fixed_time,
                                    context_state_sequence=context_closed_sequence,
                                    context_sequence_mask=context_closed_sequence_mask,
                                    temperature_k=batch.get("temperature_k"),
                                    dataset_index=batch.get("dataset_index"),
                                )[0]
                            )
                        outputs["closed_local_fixed_predictions"] = torch.stack(
                            fixed_predictions, dim=1
                        )
                        outputs["closed_local_fixed_horizons"] = torch.tensor(
                            self.config.closed_local_horizons,
                            device=context_closed_state.device,
                        )
                if closed_velocity is not None:
                    outputs["context_closed_local_velocity"] = closed_velocity
                if len(operator_outputs) == 3:
                    outputs["closed_local_log_sigma"] = operator_outputs[2]
                if "context_group_index" in batch:
                    outputs["context_group_index"] = batch["context_group_index"]
                if self.v4_slow_kinetic_model is not None:
                    if context_closed_sequence is None or future_closed_sequence is None:
                        raise ValueError("V4 slow state requires closed local state sequences")
                    outputs.update(
                        self.v4_slow_kinetic_model(
                            context_closed_sequence,
                            future_closed_sequence,
                            context_closed_sequence_mask,
                            future_closed_sequence_mask,
                            batch["horizon_frames"],
                            temperature_k=batch.get("temperature_k"),
                            dataset_index=batch.get("dataset_index"),
                        )
                    )
                if self.v4_physics_slow_model is not None:
                    if context_closed_sequence is None or future_closed_sequence is None:
                        raise ValueError("V4-C requires closed local state sequences")
                    if offline_context_states is None or future_offline_states is None:
                        raise ValueError("V4-C requires offline temporal states")
                    v4c_outputs = self.v4_physics_slow_model(
                        context_closed_sequence,
                        future_closed_sequence,
                        context_closed_sequence_mask,
                        future_closed_sequence_mask,
                        offline_context_states["frame_states"],
                        future_offline_states["frame_states"],
                        batch["horizon_frames"],
                        temperature_k=batch.get("temperature_k"),
                        dataset_index=batch.get("dataset_index"),
                    )
                    outputs.update(v4c_outputs)
                    if "context_forces" in batch and torch.is_grad_enabled():
                        force_rows = torch.arange(
                            batch["context_coords"].shape[0], device=batch["context_coords"].device
                        )
                        if "context_group_index" in batch:
                            force_rows = torch.stack(
                                [
                                    torch.nonzero(
                                        batch["context_group_index"] == group, as_tuple=False
                                    ).flatten()[0]
                                    for group in batch["context_group_index"].unique(sorted=True)
                                ]
                            )
                        force_coords = (
                            batch["context_coords"][force_rows, -1:].detach().requires_grad_(True)
                        )
                        force_state, force_state_mask = compute_v4c_force_state(
                            force_coords,
                            batch["context_atom_mask"][force_rows, -1:],
                            batch["context_residue_mask"][force_rows, -1:],
                            batch["context_chain_id"][force_rows],
                            batch["context_residue_index"][force_rows],
                            num_tokens=self.config.local_dynamics_tokens,
                        )
                        force_energy = self.v4_physics_slow_model.force_energy(
                            force_state[:, None],
                            v4c_outputs["v4c_equilibrium"][force_rows].detach(),
                            offline_context_states["frame_states"][force_rows, -1].detach(),
                        )
                        outputs["v4c_force_coordinates"] = force_coords
                        outputs["v4c_force_energy"] = force_energy
                        outputs["v4c_force_target"] = batch["context_forces"][force_rows, -1:]
                        outputs["v4c_force_atom_mask"] = batch["context_atom_mask"][force_rows, -1:]
                        outputs["v4c_force_state_mask"] = force_state_mask
        if self.slow_state_projector is not None and self.ls2_reversible_dynamics is not None:
            context_frame_mask = batch["context_residue_mask"].any(dim=2)
            future_frame_mask = batch["future_residue_mask"].any(dim=2)
            context_residue_weights = batch["context_residue_mask"].unsqueeze(-1).to(
                dtype=context_latents.dtype
            )
            future_residue_weights = batch["future_residue_mask"].unsqueeze(-1).to(
                dtype=future_latents.dtype
            )
            teacher_residue_weights = context_residue_weights.to(dtype=teacher_context_latents.dtype)
            context_instant_sequence = (
                context_latents * context_residue_weights
            ).sum(dim=2) / context_residue_weights.sum(dim=2).clamp(min=1.0)
            with torch.no_grad():
                teacher_context_instant_sequence = (
                    teacher_context_latents * teacher_residue_weights
                ).sum(dim=2) / teacher_residue_weights.sum(dim=2).clamp(min=1.0)
                future_instant_sequence = (
                    future_latents * future_residue_weights
                ).sum(dim=2) / future_residue_weights.sum(dim=2).clamp(min=1.0)
            context_slow_sequence = self.slow_state_projector(context_instant_sequence)
            context_slow_state = self.encode_endpoint_state(
                {"frame_states": context_slow_sequence}, context_frame_mask
            )
            with torch.no_grad():
                teacher_context_slow_sequence = self.target_slow_state_projector(
                    teacher_context_instant_sequence
                )
                future_slow_sequence = self.target_slow_state_projector(
                    future_instant_sequence
                )
                teacher_context_slow_state = self.encode_endpoint_state(
                    {"frame_states": teacher_context_slow_sequence}, context_frame_mask
                )
                future_slow_state = self.encode_endpoint_state(
                    {"frame_states": future_slow_sequence}, future_frame_mask
                )
            outputs.update(
                {
                    "context_slow_state": context_slow_state,
                    "teacher_context_slow_state": teacher_context_slow_state,
                    "future_slow_state": future_slow_state,
                    "context_slow_sequence": context_slow_sequence,
                    "future_slow_sequence": future_slow_sequence,
                    "context_slow_mask": context_frame_mask,
                    "future_slow_mask": future_frame_mask,
                }
            )
            ls2_outputs = self.ls2_reversible_dynamics(
                context_slow_state,
                future_slow_state,
                batch["horizon_time_ps"],
                batch["context_aatype"],
                batch["context_residue_mask"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
                context_sequence=context_slow_sequence,
            )
            with torch.no_grad():
                ls2_outputs["ls2_persistence_assignment"] = (
                    self.ls2_reversible_dynamics.assignments(
                        teacher_context_slow_state, target=True
                    )
                )
            ls2_outputs["ls2_persistence_state"] = teacher_context_slow_state
            outputs.update(ls2_outputs)
        elif self.slow_state_projector is not None:
            context_slow_state = self.slow_state_projector(transition_source)
            with torch.no_grad():
                future_slow_state = self.target_slow_state_projector(future_endpoint)
                teacher_context_slow_state = self.target_slow_state_projector(context_teacher_endpoint)
                future_slow_sequence = self.target_slow_state_projector(
                    future_states["frame_states"]
                )
            context_slow_sequence = self.slow_state_projector(
                context_states["frame_states"]
            )
            if self.config.paired_teacher_space_delta:
                slow_persistence_state = teacher_context_slow_state
                slow_delta_anchor = teacher_context_slow_state
                slow_prediction_anchor = teacher_context_slow_state
            else:
                slow_persistence_state = context_slow_state
                slow_delta_anchor = context_slow_state.detach()
                slow_prediction_anchor = context_slow_state
            slow_delta_target = future_slow_state - slow_delta_anchor
            slow_delta_prediction = self.slow_delta_predictor(
                context_slow_state,
                batch["horizon_time_ps"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            outputs.update(
                {
                    "context_slow_state": context_slow_state,
                    "teacher_context_slow_state": teacher_context_slow_state,
                    "slow_persistence_state": slow_persistence_state,
                    "future_slow_state": future_slow_state,
                    "slow_delta_target": slow_delta_target,
                    "slow_delta_prediction": slow_delta_prediction,
                    "slow_state_prediction": slow_prediction_anchor + slow_delta_prediction,
                    "context_slow_sequence": context_slow_sequence,
                    "future_slow_sequence": future_slow_sequence,
                    "context_slow_mask": batch["context_residue_mask"].any(dim=2),
                    "future_slow_mask": batch["future_residue_mask"].any(dim=2),
                }
            )
        return outputs

    def encode_kinetic_readout(
        self,
        residue_latents: torch.Tensor,
        residue_mask: torch.Tensor,
        time_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.kinetic_readout is None:
            raise ValueError("kinetic_readout is not configured for this model")
        states = self.encode_states(
            residue_latents,
            residue_mask,
            time_ps=time_ps,
            temperature_k=temperature_k,
            dataset_index=dataset_index,
        )["frame_states"]
        return self.kinetic_readout(states)

    def compute_msm_state_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Per-protein MSM state classification loss on context latents."""
        if "msm_state_labels" not in batch or batch["msm_state_labels"] is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        pooled = outputs.get("frame_states")
        if pooled is None:
            pooled = self.encode_states(
                outputs["context_latents"],
                batch["residue_mask"],
                time_ps=batch.get("time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )["frame_states"]
        labels = batch["msm_state_labels"]  # [B, T], -1 = unlabeled
        protein_ids = batch["protein_id"]  # list of str
        D = pooled.shape[-1]
        total_loss = torch.tensor(0.0, device=pooled.device)
        count = 0
        for b in range(pooled.shape[0]):
            pid = protein_ids[b]
            if pid not in self.state_classifiers:
                self.state_classifiers[pid] = nn.Linear(D, 6).to(pooled.device)
            classifier = self.state_classifiers[pid]
            logits = classifier(pooled[b])  # [T, 6]
            mask = labels[b] >= 0
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], labels[b][mask])
            total_loss = total_loss + loss
            count += 1
        return total_loss / max(count, 1)

    def compute_vamp2_loss(self, outputs: dict, batch: dict, lag: int = 1) -> torch.Tensor:
        """VAMP-2 maximization loss per clip (maximise -> minimise negative)."""
        _ = outputs  # unused, we need full-structure latents
        coords = batch["coords"]
        residue_mask = batch["residue_mask"]
        aatype = batch["aatype"]
        residue_index = batch["residue_index"]
        chain_id = batch["chain_id"]
        full_latents = self.context_encoder(
            coords=coords, residue_mask=residue_mask,
            aatype=aatype, residue_index=residue_index, chain_id=chain_id,
            token_mask=residue_mask,
        )
        pooled = self.encode_states(
            full_latents,
            residue_mask,
            time_ps=batch.get("time_ps"),
            temperature_k=batch.get("temperature_k"),
            dataset_index=batch.get("dataset_index"),
        )["frame_states"]  # [B, T, D]
        D = pooled.shape[-1]
        eps = 1e-3
        total_vamp2 = torch.tensor(0.0, device=pooled.device)
        n_clips = 0
        for b in range(pooled.shape[0]):
            seq = pooled[b]  # [T, D]
            if len(seq) < lag + 2:
                continue
            # Use a fixed random projection to a low-dimensional space
            # (T=8 is too few frames for stable 128D covariance estimation)
            if not hasattr(self, '_vamp2_projector') or self._vamp2_projector.shape[0] != D:
                proj = torch.randn(D, min(D, 8), device=seq.device)
                proj = proj / proj.norm(dim=0, keepdim=True)
                self.register_buffer('_vamp2_projector', proj)
            seq = seq @ self._vamp2_projector  # [T, d] with d = min(D, 8)
            d = seq.shape[-1]
            X0 = seq[:-lag]
            X1 = seq[lag:]
            X0 = X0 - X0.mean(dim=0, keepdim=True)
            X1 = X1 - X1.mean(dim=0, keepdim=True)
            N = X0.shape[0]
            try:
                # Compute whitening matrices without backprop (eigh gradients are unstable)
                with torch.no_grad():
                    C00 = X0.T @ X0 / (N - 1) + eps * torch.eye(d, device=X0.device)
                    C01 = X0.T @ X1 / (N - 1)
                    C11 = X1.T @ X1 / (N - 1) + eps * torch.eye(d, device=X1.device)
                    s0, U0 = torch.linalg.eigh(C00)
                    s1, U1 = torch.linalg.eigh(C11)
                    cutoff = 1e-6
                    s0_inv_sqrt = torch.where(s0 > cutoff, 1.0 / s0.sqrt(), torch.zeros_like(s0))
                    s1_inv_sqrt = torch.where(s1 > cutoff, 1.0 / s1.sqrt(), torch.zeros_like(s1))
                    C00_inv_half = U0 @ torch.diag(s0_inv_sqrt) @ U0.T
                    C11_inv_half = U1 @ torch.diag(s1_inv_sqrt) @ U1.T
                # Gradients flow through X0 and X1, NOT through the whitening matrices
                K = (X0 @ C00_inv_half).T @ (X1 @ C11_inv_half) / (N - 1)
                vamp2 = (K ** 2).sum()
                total_vamp2 = total_vamp2 + vamp2
                n_clips += 1
            except torch._C._LinAlgError:
                continue
        vamp2_mean = total_vamp2 / max(n_clips, 1)
        # Normalise to [0, 1]: 1 = trivial, proj_dim = perfect
        proj_dim = self._vamp2_projector.shape[-1]
        loss = 1.0 - (vamp2_mean - 1.0) / (proj_dim - 1.0)
        return loss


def _latent_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if pred.numel() == 0:
        raise ValueError("empty target mask")
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target)


def residue_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    pred = outputs["predictions"][outputs["target_mask"]]
    target = outputs["targets"][outputs["target_mask"]]
    return _latent_loss(pred, target, loss_type)


def state_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    pred = outputs["state_predictions"][outputs["state_target_mask"]]
    target = outputs["state_targets"][outputs["state_target_mask"]]
    return _latent_loss(pred, target, loss_type)


def state_transition_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    pred_delta = outputs["state_predictions"][:, 1:] - outputs["state_predictions"][:, :-1]
    target_delta = outputs["state_targets"][:, 1:] - outputs["state_targets"][:, :-1]
    frame_mask = outputs["state_target_mask"]
    valid_mask = outputs.get("frame_valid_mask", frame_mask)
    pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1] & valid_mask[:, 1:] & valid_mask[:, :-1]
    if not pair_mask.any():
        return outputs["predictions"].new_tensor(0.0)
    return _latent_loss(pred_delta[pair_mask], target_delta[pair_mask], loss_type)


def lagged_state_semigroup_loss(
    outputs: dict,
    lagged_state_predictor: LaggedStatePredictor | None,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    terms = lagged_state_semigroup_terms(outputs, lagged_state_predictor, loss_type=loss_type)
    if not terms:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack([term["loss"] for term in terms.values()]).mean()


def lagged_state_semigroup_terms(
    outputs: dict,
    lagged_state_predictor: LaggedStatePredictor | None,
    loss_type: str = "smooth_l1",
) -> dict[str, dict[str, torch.Tensor | int]]:
    """Compare direct and composed transitions from the final causal state.

    Composition is only valid when predictor inputs and outputs inhabit the
    same state space. A motion-only source predicts an absolute state and is
    therefore intentionally excluded.
    """
    if (
        "lagged_state_predictions" not in outputs
        or lagged_state_predictor is None
        or not outputs.get("transition_source_is_state_space", True)
    ):
        return {}
    predictions = outputs["lagged_state_predictions"]
    lags = sorted(predictions)
    available = set(lags)
    context_mask = outputs["context_frame_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | outputs["state_target_mask"])
    time_ps = outputs.get("time_ps")
    B, T = context_mask.shape
    frame_index = torch.arange(T, device=context_mask.device).view(1, T).expand(B, -1)
    source_index = torch.where(context_mask, frame_index, -1).max(dim=1).values
    terms: dict[str, dict[str, torch.Tensor | int]] = {}
    for first_lag in lags:
        first_pred = predictions[first_lag]
        if time_ps is not None and first_lag < time_ps.shape[1]:
            shifted_time = time_ps[:, first_lag:]
            if shifted_time.shape[1] < first_pred.shape[1]:
                pad = shifted_time[:, -1:].expand(-1, first_pred.shape[1] - shifted_time.shape[1])
                shifted_time = torch.cat([shifted_time, pad], dim=1)
            shifted_time = shifted_time[:, : first_pred.shape[1]]
        else:
            shifted_time = None
        composed_predictions = lagged_state_predictor(
            first_pred,
            time_ps=shifted_time,
            temperature_k=outputs.get("temperature_k"),
            dataset_index=outputs.get("dataset_index"),
        )
        for second_lag in lags:
            total_lag = first_lag + second_lag
            if total_lag not in available or total_lag >= T:
                continue
            intermediate_index = source_index + first_lag
            target_index = source_index + total_lag
            pair_mask = (source_index >= 0) & (target_index < T)
            if pair_mask.any():
                eligible = torch.nonzero(pair_mask, as_tuple=False).flatten()
                eligible = eligible[
                    valid_mask[eligible, intermediate_index[eligible]]
                    & valid_mask[eligible, target_index[eligible]]
                ]
                pair_mask = torch.zeros_like(pair_mask)
                pair_mask[eligible] = True
            if not pair_mask.any():
                continue
            batch_index = torch.arange(B, device=context_mask.device)[pair_mask]
            source = source_index[pair_mask]
            direct = predictions[total_lag][batch_index, source].detach()
            composed = composed_predictions[second_lag][batch_index, source]
            key = f"{first_lag}+{second_lag}={total_lag}"
            terms[key] = {
                "loss": _latent_loss(composed, direct, loss_type),
                "pairs": int(pair_mask.sum().item()),
            }
    return terms


def lagged_state_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    terms = lagged_state_jepa_terms(outputs, loss_type=loss_type)
    if not terms:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack([term["loss"] for term in terms.values()]).mean()


def paired_forecast_terms(outputs: dict, loss_type: str = "smooth_l1") -> dict[str, torch.Tensor]:
    """Return endpoint forecast losses and persistence-relative diagnostics."""
    prediction = outputs["delta_prediction"]
    target = outputs["delta_target"]
    zero = torch.zeros_like(target)
    delta_loss = _latent_loss(prediction, target, loss_type)
    persistence_loss = _latent_loss(zero, target, loss_type)
    state_loss = _latent_loss(outputs["state_prediction"], outputs["future_endpoint"], loss_type)
    relative_delta_loss = delta_loss / persistence_loss.detach().clamp(min=1.0e-6)
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
        "relative_delta_loss": relative_delta_loss,
        "persistence_delta_loss": persistence_loss,
        "skill": 1.0 - delta_loss / persistence_loss.clamp(min=1.0e-8),
        "pairs": torch.tensor(target.shape[0], device=target.device),
    }


def _slow_cross_entropy(prediction: torch.Tensor, target: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """In-batch temporal contrastive loss that prevents the slow head collapsing."""
    if prediction.shape[0] < 2:
        return prediction.new_tensor(0.0)
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(prediction.shape[0], device=prediction.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _continuous_slow_vamp2(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    lags: tuple[int, ...],
    burn_in_frames: int = 0,
    eps: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAMP-2 on within-trajectory fluctuations of a continuous learned state."""
    if burn_in_frames:
        sequence = sequence[:, burn_in_frames:]
        mask = mask[:, burn_in_frames:]
    zero = sequence.new_tensor(0.0)
    dim = sequence.shape[-1]
    scores = []
    eye = torch.eye(dim, device=sequence.device, dtype=sequence.dtype)
    for lag in sorted(set(int(value) for value in lags)):
        if lag <= 0 or lag >= sequence.shape[1]:
            continue
        x_parts = []
        y_parts = []
        for row in range(sequence.shape[0]):
            pair_mask = mask[row, :-lag] & mask[row, lag:]
            if not pair_mask.any():
                continue
            left = sequence[row, :-lag][pair_mask]
            right = sequence[row, lag:][pair_mask]
            # Remove static protein identity while retaining temporal changes.
            center = torch.cat([left, right], dim=0).mean(dim=0, keepdim=True)
            x_parts.append(left - center)
            y_parts.append(right - center)
        if not x_parts:
            continue
        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        if x.shape[0] <= dim + 1:
            continue
        denominator = max(x.shape[0] - 1, 1)
        c00 = x.T @ x / denominator + eps * eye
        c11 = y.T @ y / denominator + eps * eye
        c01 = x.T @ y / denominator
        with torch.no_grad():
            def inverse_sqrt(covariance: torch.Tensor) -> torch.Tensor:
                values, vectors = torch.linalg.eigh(covariance)
                return (vectors * values.clamp(min=eps).rsqrt().unsqueeze(0)) @ vectors.T

            w0 = inverse_sqrt(c00)
            w1 = inverse_sqrt(c11)
        koopman = w0 @ c01 @ w1
        scores.append(koopman.square().sum())
    score = torch.stack(scores).mean() if scores else zero
    return -score / float(dim), score


def slow_forecast_terms(
    outputs: dict,
    loss_type: str = "smooth_l1",
    vamp_lags: tuple[int, ...] = (1, 2, 4, 8),
    vamp_burn_in_frames: int = 0,
) -> dict[str, torch.Tensor]:
    """Losses for a low-dimensional slow/metastable endpoint representation."""
    prediction = outputs["slow_delta_prediction"]
    target = outputs["slow_delta_target"]
    delta_loss = _latent_loss(prediction, target, loss_type)
    state_loss = _latent_loss(outputs["slow_state_prediction"], outputs["future_slow_state"], loss_type)
    persistence_loss = _latent_loss(torch.zeros_like(target), target, loss_type)
    relative_delta_loss = (
        (prediction - target).square().mean(dim=-1)
        / target.detach().square().mean(dim=-1).clamp(min=1.0e-4)
    ).mean()
    alignment_loss = _latent_loss(
        outputs["context_slow_state"], outputs["teacher_context_slow_state"].detach(), loss_type
    )
    context = outputs["context_slow_state"]
    std = torch.sqrt(context.var(dim=0, unbiased=False) + 1.0e-4)
    variance_loss = torch.relu(1.0 - std).mean() if context.shape[0] > 1 else context.new_tensor(0.0)
    if context.shape[0] > 2:
        normalized = (context - context.mean(dim=0, keepdim=True)) / std.detach().clamp(min=1.0e-3)
        correlation = normalized.T @ normalized / max(context.shape[0] - 1, 1)
        off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
        covariance_loss = off_diagonal.square().mean()
    else:
        covariance_loss = context.new_tensor(0.0)
    vamp_loss, vamp2_score = _continuous_slow_vamp2(
        outputs["context_slow_sequence"],
        outputs["context_slow_mask"],
        vamp_lags,
        burn_in_frames=vamp_burn_in_frames,
    )
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
        "relative_delta_loss": relative_delta_loss,
        "alignment_loss": alignment_loss,
        "persistence_loss": persistence_loss,
        "skill": 1.0 - delta_loss / persistence_loss.clamp(min=1.0e-8),
        "contrastive_loss": _slow_cross_entropy(outputs["slow_state_prediction"], outputs["future_slow_state"]),
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "vamp_loss": vamp_loss,
        "vamp2_score": vamp2_score,
    }


def reversible_kinetic_terms(
    outputs: dict,
    vamp_lags: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, torch.Tensor]:
    """Losses and diagnostics for the LS2 reversible kinetic-OU predictor."""
    if "ls2_state_prediction" not in outputs:
        reference = outputs.get("delta_target", outputs.get("context_endpoint"))
        zero = reference.new_tensor(0.0)
        return {
            "state_loss": zero,
            "persistence_loss": zero,
            "state_skill": zero,
            "assignment_loss": zero,
            "assignment_persistence_loss": zero,
            "assignment_skill": zero,
            "nll": zero,
            "balance_loss": zero,
            "entropy_loss": zero,
            "vamp_loss": zero,
            "vamp2_score": zero,
            "detailed_balance_max": zero,
            "stationarity_error": zero,
        }
    prediction = outputs["ls2_state_prediction"]
    target = outputs["ls2_future_state"].detach()
    persistence = outputs["ls2_persistence_state"].detach()
    state_loss = F.smooth_l1_loss(prediction, target)
    persistence_loss = F.smooth_l1_loss(persistence, target)

    assignment_prediction = outputs["ls2_assignment_prediction"].clamp(min=1.0e-8)
    assignment_target = outputs["ls2_future_assignment"].detach()
    assignment_persistence = outputs["ls2_persistence_assignment"].detach()
    assignment_loss = (
        assignment_target
        * (
            assignment_target.clamp(min=1.0e-8).log()
            - assignment_prediction.log()
        )
    ).sum(dim=-1).mean()
    assignment_brier = (assignment_prediction - assignment_target).square().sum(dim=-1).mean()
    persistence_brier = (assignment_persistence - assignment_target).square().sum(dim=-1).mean()

    residual = target - prediction
    log_sigma = outputs["ls2_log_sigma"]
    nll = 0.5 * (
        residual.square() * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
    ).mean()
    usage = outputs["ls2_context_assignment"].mean(dim=0).clamp(min=1.0e-8)
    balance_loss = (usage * (usage * usage.numel()).log()).sum()
    entropy_loss = -(
        outputs["ls2_context_assignment"].clamp(min=1.0e-8)
        * outputs["ls2_context_assignment"].clamp(min=1.0e-8).log()
    ).sum(dim=-1).mean()
    vamp_loss, vamp2_score = _continuous_slow_vamp2(
        outputs["ls2_context_assignment_sequence"],
        outputs["context_slow_mask"],
        vamp_lags,
    )
    stationary = outputs["ls2_stationary_distribution"]
    transition = outputs["ls2_transition_matrix"]
    stationary_after = torch.bmm(stationary.unsqueeze(1), transition).squeeze(1)
    stationarity_error = (stationary_after - stationary).abs().max()
    return {
        "state_loss": state_loss,
        "persistence_loss": persistence_loss,
        "state_skill": 1.0 - state_loss / persistence_loss.clamp(min=1.0e-8),
        "assignment_loss": assignment_loss,
        "assignment_persistence_loss": persistence_brier,
        "assignment_skill": 1.0 - assignment_brier / persistence_brier.clamp(min=1.0e-8),
        "nll": nll,
        "balance_loss": balance_loss,
        "entropy_loss": entropy_loss,
        "vamp_loss": vamp_loss,
        "vamp2_score": vamp2_score,
        "detailed_balance_max": outputs["ls2_detailed_balance_residual"].abs().max(),
        "stationarity_error": stationarity_error,
    }


def paired_endpoint_variance_regularizer(outputs: dict) -> torch.Tensor:
    endpoint = outputs["context_endpoint"]
    if endpoint.shape[0] < 2:
        return endpoint.new_tensor(0.0)
    std = torch.sqrt(endpoint.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def paired_causal_alignment_loss(outputs: dict) -> torch.Tensor:
    if "context_teacher_endpoint" not in outputs:
        return outputs["context_endpoint"].new_tensor(0.0)
    return F.smooth_l1_loss(outputs["context_endpoint"], outputs["context_teacher_endpoint"].detach())


def lagged_state_jepa_terms(outputs: dict, loss_type: str = "smooth_l1") -> dict[int, dict[str, torch.Tensor | int]]:
    """Return per-lag losses from the final visible context state.

    Future masking can leave a gap that makes short lags miss the explicit
    target mask entirely. The EMA target encoder nevertheless provides a
    target for every valid frame. Anchoring each transition at the final
    visible context frame gives every configured horizon a causal source and
    prevents short-lag pairs from silently disappearing.
    """
    if "lagged_state_predictions" not in outputs:
        return {}
    context_mask = outputs["context_frame_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | outputs["state_target_mask"])
    target_states = outputs.get("lagged_state_targets", outputs["state_targets"])
    B, T = context_mask.shape
    frame_index = torch.arange(T, device=context_mask.device).view(1, T).expand(B, -1)
    source_index = torch.where(context_mask, frame_index, -1).max(dim=1).values
    terms: dict[int, dict[str, torch.Tensor | int]] = {}
    for lag, pred_all in outputs["lagged_state_predictions"].items():
        target_index = source_index + lag
        pair_mask = (source_index >= 0) & (target_index < T)
        if pair_mask.any():
            eligible = torch.nonzero(pair_mask, as_tuple=False).flatten()
            eligible = eligible[valid_mask[eligible, target_index[eligible]]]
            pair_mask = torch.zeros_like(pair_mask)
            pair_mask[eligible] = True
        if not pair_mask.any():
            continue
        batch_index = torch.arange(B, device=context_mask.device)[pair_mask]
        source = source_index[pair_mask]
        target = target_index[pair_mask]
        loss = _latent_loss(pred_all[batch_index, source], target_states[batch_index, target], loss_type)
        intrinsic_persistence = _latent_loss(
            target_states[batch_index, source], target_states[batch_index, target], loss_type
        )
        transition_states = outputs.get("transition_states")
        persistence = (
            _latent_loss(transition_states[batch_index, source], target_states[batch_index, target], loss_type)
            if transition_states is not None
            else intrinsic_persistence
        )
        source_alignment = (
            _latent_loss(transition_states[batch_index, source], target_states[batch_index, source], loss_type)
            if transition_states is not None
            else loss.new_tensor(0.0)
        )
        terms[int(lag)] = {
            "loss": loss,
            "persistence_loss": persistence,
            "intrinsic_persistence_loss": intrinsic_persistence,
            "source_alignment_loss": source_alignment,
            "pairs": int(pair_mask.sum().item()),
        }
    return terms


def physchem_prediction_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "physchem_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    context_mask = outputs["context_frame_mask"]
    target_mask = outputs["state_target_mask"]
    losses = []
    for lag, pred_all in outputs["physchem_predictions"].items():
        if lag >= pred_all.shape[1]:
            continue
        target_all = outputs["physchem_targets"][lag]
        physchem_mask = outputs["physchem_target_masks"][lag]
        pair_mask = context_mask[:, :-lag] & target_mask[:, lag:] & physchem_mask[:, :-lag]
        if pair_mask.any():
            losses.append(_latent_loss(pred_all[:, :-lag][pair_mask], target_all[:, :-lag][pair_mask], loss_type))
    if not losses:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack(losses).mean()


def linear_dynamics_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "linear_dynamics_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    context_mask = outputs["context_frame_mask"]
    target_mask = outputs["state_target_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | target_mask)
    target_states = outputs["state_targets"]
    losses = []
    for lag, pred_all in outputs["linear_dynamics_predictions"].items():
        if lag >= pred_all.shape[1]:
            continue
        pred = pred_all[:, :-lag]
        target = target_states[:, lag:]
        pair_mask = context_mask[:, :-lag] & target_mask[:, lag:] & valid_mask[:, lag:]
        if pair_mask.any():
            losses.append(_latent_loss(pred[pair_mask], target[pair_mask], loss_type))
    if not losses:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack(losses).mean()


def kinetic_vampnet_loss(outputs: dict, lag: int = 1, eps: float = 1.0e-3) -> torch.Tensor:
    if "kinetic_readout" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    if lag <= 0:
        raise ValueError("lag must be positive")
    z = outputs["kinetic_readout"]
    mask = outputs.get("kinetic_readout_mask")
    if mask is None:
        mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
    if lag >= z.shape[1]:
        return z.new_tensor(0.0)
    pair_mask = mask[:, :-lag] & mask[:, lag:]
    if not pair_mask.any():
        return z.new_tensor(0.0)
    x0 = z[:, :-lag][pair_mask]
    x1 = z[:, lag:][pair_mask]
    n, dim = x0.shape
    if n < dim + 2 or dim < 2:
        return z.new_tensor(0.0)

    x0 = x0 - x0.mean(dim=0, keepdim=True)
    x1 = x1 - x1.mean(dim=0, keepdim=True)
    eye = torch.eye(dim, device=z.device, dtype=z.dtype)
    with torch.no_grad():
        c00 = x0.T @ x0 / (n - 1) + eps * eye
        c11 = x1.T @ x1 / (n - 1) + eps * eye
        s0, u0 = torch.linalg.eigh(c00)
        s1, u1 = torch.linalg.eigh(c11)
        inv0 = torch.where(s0 > eps, s0.rsqrt(), torch.zeros_like(s0))
        inv1 = torch.where(s1 > eps, s1.rsqrt(), torch.zeros_like(s1))
        c00_inv_half = (u0 * inv0[None, :]) @ u0.T
        c11_inv_half = (u1 * inv1[None, :]) @ u1.T
    koopman = (x0 @ c00_inv_half).T @ (x1 @ c11_inv_half) / (n - 1)
    vamp2 = koopman.pow(2).sum()
    return 1.0 - vamp2 / float(dim)


def jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    return residue_jepa_loss(outputs, loss_type=loss_type)


def variance_regularizer(outputs: dict) -> torch.Tensor:
    pred = outputs["predictions"][outputs["target_mask"]]
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    std = torch.sqrt(pred.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def state_variance_regularizer(outputs: dict) -> torch.Tensor:
    if "state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    pred = outputs["state_predictions"][outputs["state_target_mask"]]
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    std = torch.sqrt(pred.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def state_feature_variance_regularizer(outputs: dict) -> torch.Tensor:
    if "frame_states" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    mask = outputs.get("context_frame_mask")
    states = outputs["frame_states"] if mask is None else outputs["frame_states"][mask]
    if states.shape[0] < 2:
        return outputs["predictions"].new_tensor(0.0)
    std = torch.sqrt(states.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def state_covariance_regularizer(outputs: dict) -> torch.Tensor:
    if "frame_states" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    mask = outputs.get("context_frame_mask")
    states = outputs["frame_states"] if mask is None else outputs["frame_states"][mask]
    if states.shape[0] < 2:
        return outputs["predictions"].new_tensor(0.0)
    states = states - states.mean(dim=0, keepdim=True)
    states = states / torch.sqrt(states.var(dim=0, unbiased=False, keepdim=True) + 1.0e-4)
    covariance = states.T @ states / max(1, states.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off_diagonal.square().sum() / states.shape[1]


def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = F.normalize(b1, dim=-1, eps=1.0e-8)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


@torch.no_grad()
def _backbone_torsions(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, L = residue_mask.shape
    angles = coords.new_zeros(B, T, L, 3)
    masks = torch.zeros(B, T, L, 3, dtype=torch.bool, device=coords.device)
    if L < 2:
        return angles, masks

    n = coords[:, :, :, 0]
    ca = coords[:, :, :, 1]
    c = coords[:, :, :, 2]
    valid_n = atom_mask[:, :, :, 0] & residue_mask
    valid_ca = atom_mask[:, :, :, 1] & residue_mask
    valid_c = atom_mask[:, :, :, 2] & residue_mask
    same_chain_prev = chain_id[:, 1:] == chain_id[:, :-1]
    same_chain_next = same_chain_prev

    phi_mask = valid_c[:, :, :-1] & valid_n[:, :, 1:] & valid_ca[:, :, 1:] & valid_c[:, :, 1:]
    phi_mask = phi_mask & same_chain_prev[:, None, :]
    angles[:, :, 1:, 0] = _dihedral(c[:, :, :-1], n[:, :, 1:], ca[:, :, 1:], c[:, :, 1:])
    masks[:, :, 1:, 0] = phi_mask

    psi_mask = valid_n[:, :, :-1] & valid_ca[:, :, :-1] & valid_c[:, :, :-1] & valid_n[:, :, 1:]
    psi_mask = psi_mask & same_chain_next[:, None, :]
    angles[:, :, :-1, 1] = _dihedral(n[:, :, :-1], ca[:, :, :-1], c[:, :, :-1], n[:, :, 1:])
    masks[:, :, :-1, 1] = psi_mask

    omega_mask = valid_ca[:, :, :-1] & valid_c[:, :, :-1] & valid_n[:, :, 1:] & valid_ca[:, :, 1:]
    omega_mask = omega_mask & same_chain_next[:, None, :]
    angles[:, :, :-1, 2] = _dihedral(ca[:, :, :-1], c[:, :, :-1], n[:, :, 1:], ca[:, :, 1:])
    masks[:, :, :-1, 2] = omega_mask
    return angles, masks


@torch.no_grad()
def compute_physchem_targets(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    lags: tuple[int, ...],
    contact_cutoff: float = 8.0,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    B, T, L = residue_mask.shape
    device = coords.device
    ca = coords[:, :, :, 1]
    ca_valid = atom_mask[:, :, :, 1] & residue_mask
    pair_valid = ca_valid[:, :, :, None] & ca_valid[:, :, None, :]
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    seq_sep = (residue_index[:, :, None] - residue_index[:, None, :]).abs() > 2
    pair_valid = pair_valid & same_chain[:, None] & seq_sep[:, None]
    flat_ca = ca.reshape(B * T, L, 3)
    distances = torch.cdist(flat_ca, flat_ca).reshape(B, T, L, L)
    contacts = (distances < contact_cutoff) & pair_valid

    torsions, torsion_mask = _backbone_torsions(coords, atom_mask, residue_mask, chain_id)
    targets: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    for lag in lags:
        target = coords.new_zeros(B, T, 4)
        valid = torch.zeros(B, T, dtype=torch.bool, device=device)
        if lag >= T:
            targets[int(lag)] = target
            masks[int(lag)] = valid
            continue

        current_contacts = contacts[:, :-lag]
        future_contacts = contacts[:, lag:]
        contact_count = current_contacts.sum(dim=(-1, -2))
        persisted = (current_contacts & future_contacts).sum(dim=(-1, -2))
        contact_persistence = persisted.float() / contact_count.float().clamp(min=1.0)
        contact_valid = contact_count > 0

        torsion_current = torsions[:, :-lag]
        torsion_future = torsions[:, lag:]
        torsion_pair_mask = torsion_mask[:, :-lag] & torsion_mask[:, lag:]
        torsion_delta = (
            (torch.sin(torsion_future) - torch.sin(torsion_current)).pow(2)
            + (torch.cos(torsion_future) - torch.cos(torsion_current)).pow(2)
        )
        torsion_sum = (torsion_delta * torsion_pair_mask.float()).sum(dim=2)
        torsion_count = torsion_pair_mask.sum(dim=2)
        torsion_mean = torsion_sum / torsion_count.float().clamp(min=1.0)
        torsion_valid = torsion_count > 0

        target[:, :-lag, 0] = contact_persistence
        target[:, :-lag, 1:] = torsion_mean
        valid[:, :-lag] = contact_valid & torsion_valid.all(dim=-1)
        targets[int(lag)] = target
        masks[int(lag)] = valid
    return targets, masks


@torch.no_grad()
def compute_paired_observable_targets(
    context_coords: torch.Tensor,
    future_coords: torch.Tensor,
    context_atom_mask: torch.Tensor,
    future_atom_mask: torch.Tensor,
    context_residue_mask: torch.Tensor,
    future_residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    contact_cutoff: float = 8.0,
) -> torch.Tensor:
    """Compute stable motion targets directly from the two endpoint frames.

    The four channels are contact persistence, mean backbone torsion change,
    radius-of-gyration change, and local CA-distance change. They are
    deliberately low-dimensional so they anchor the latent transition
    without turning MSM labels into a training target.
    """
    c = context_coords[:, -1]
    f = future_coords[:, -1]
    c_atoms = context_atom_mask[:, -1]
    f_atoms = future_atom_mask[:, -1]
    c_valid = context_residue_mask[:, -1] & c_atoms[:, :, 1]
    f_valid = future_residue_mask[:, -1] & f_atoms[:, :, 1]
    pair_valid = c_valid[:, :, None] & c_valid[:, None, :] & f_valid[:, :, None] & f_valid[:, None, :]
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    seq_sep = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
    nonlocal_pair = pair_valid & same_chain & (seq_sep > 2)
    c_dist = torch.cdist(c[:, :, 1], c[:, :, 1])
    f_dist = torch.cdist(f[:, :, 1], f[:, :, 1])
    c_contact = (c_dist < contact_cutoff) & nonlocal_pair
    f_contact = (f_dist < contact_cutoff) & nonlocal_pair
    contact_count = c_contact.sum(dim=(-1, -2)).float()
    contact_persistence = (c_contact & f_contact).sum(dim=(-1, -2)).float() / contact_count.clamp(min=1.0)

    c_torsion, c_torsion_mask = _backbone_torsions(
        context_coords[:, -1:], context_atom_mask[:, -1:], context_residue_mask[:, -1:], chain_id
    )
    f_torsion, f_torsion_mask = _backbone_torsions(
        future_coords[:, -1:], future_atom_mask[:, -1:], future_residue_mask[:, -1:], chain_id
    )
    torsion_mask = c_torsion_mask[:, 0] & f_torsion_mask[:, 0]
    torsion_delta = (
        (torch.sin(f_torsion[:, 0]) - torch.sin(c_torsion[:, 0])).square()
        + (torch.cos(f_torsion[:, 0]) - torch.cos(c_torsion[:, 0])).square()
    ).sqrt()
    torsion_change = (torsion_delta * torsion_mask.float()).sum(dim=(-1, -2)) / torsion_mask.sum(dim=(-1, -2)).float().clamp(min=1.0)

    def radius_of_gyration(ca: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.float().unsqueeze(-1)
        center = (ca * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        count = weights.sum(dim=(1, 2)).clamp(min=1.0)
        return torch.sqrt((((ca - center[:, None]) ** 2) * weights).sum(dim=(1, 2)) / count).clamp(min=1.0e-6)

    c_rg = radius_of_gyration(c[:, :, 1], c_valid)
    f_rg = radius_of_gyration(f[:, :, 1], f_valid)
    rg_change = (f_rg - c_rg) / 10.0

    local_pair = pair_valid & same_chain & (seq_sep > 0) & (seq_sep <= 8)
    local_count = local_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
    local_change = ((f_dist - c_dist).abs() * local_pair.float()).sum(dim=(-1, -2)) / local_count / 10.0
    return torch.stack([contact_persistence, torsion_change, rg_change, local_change], dim=-1).clamp(-5.0, 5.0)


@torch.no_grad()
def compute_v3_paired_observable_targets(
    context_coords: torch.Tensor,
    future_coords: torch.Tensor,
    context_atom_mask: torch.Tensor,
    future_atom_mask: torch.Tensor,
    context_residue_mask: torch.Tensor,
    future_residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    contact_cutoff: float = 8.0,
) -> torch.Tensor:
    """Twelve signed and event-resolved physical transition targets."""
    c = context_coords[:, -1]
    f = future_coords[:, -1]
    c_valid = context_residue_mask[:, -1] & context_atom_mask[:, -1, :, 1]
    f_valid = future_residue_mask[:, -1] & future_atom_mask[:, -1, :, 1]
    valid = c_valid & f_valid
    pair_valid = valid[:, :, None] & valid[:, None, :]
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    sequence_separation = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
    nonlocal_pair = pair_valid & same_chain & (sequence_separation > 2)
    c_dist = torch.cdist(c[:, :, 1], c[:, :, 1])
    f_dist = torch.cdist(f[:, :, 1], f[:, :, 1])
    c_contact = (c_dist < contact_cutoff) & nonlocal_pair
    f_contact = (f_dist < contact_cutoff) & nonlocal_pair
    contact_union = (c_contact | f_contact).sum(dim=(-1, -2)).float().clamp(min=1.0)
    retained = (c_contact & f_contact).sum(dim=(-1, -2)).float() / contact_union
    gained = ((~c_contact) & f_contact & nonlocal_pair).sum(dim=(-1, -2)).float() / contact_union
    lost = (c_contact & (~f_contact) & nonlocal_pair).sum(dim=(-1, -2)).float() / contact_union

    c_torsion, c_torsion_mask = _backbone_torsions(
        context_coords[:, -1:], context_atom_mask[:, -1:], context_residue_mask[:, -1:], chain_id
    )
    f_torsion, f_torsion_mask = _backbone_torsions(
        future_coords[:, -1:], future_atom_mask[:, -1:], future_residue_mask[:, -1:], chain_id
    )
    torsion_mask = c_torsion_mask[:, 0] & f_torsion_mask[:, 0]
    angle_delta = torch.atan2(
        torch.sin(f_torsion[:, 0] - c_torsion[:, 0]),
        torch.cos(f_torsion[:, 0] - c_torsion[:, 0]),
    )
    torsion_count = torsion_mask.sum(dim=1).float().clamp(min=1.0)
    signed_torsion = (angle_delta * torsion_mask.float()).sum(dim=1) / torsion_count
    absolute_torsion = (angle_delta.abs() * torsion_mask.float()).sum(dim=1) / torsion_count

    def radius_of_gyration(ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float().unsqueeze(-1)
        center = (ca * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        return torch.sqrt(
            (((ca - center[:, None]) ** 2) * weight).sum(dim=(1, 2))
            / weight.sum(dim=(1, 2)).clamp(min=1.0)
        ).clamp(min=1.0e-6)

    c_rg = radius_of_gyration(c[:, :, 1], valid)
    f_rg = radius_of_gyration(f[:, :, 1], valid)
    rg_change = (f_rg - c_rg) / 10.0

    aligned_rmsd = c_rg.new_zeros(c_rg.shape)
    for batch_index in range(c.shape[0]):
        mask = valid[batch_index]
        if int(mask.sum()) < 3:
            continue
        x = c[batch_index, mask, 1]
        y = f[batch_index, mask, 1]
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        u, _, vh = torch.linalg.svd(x.T @ y)
        correction = torch.ones(3, device=x.device, dtype=x.dtype)
        correction[-1] = torch.sign(torch.det(u @ vh)).clamp(min=-1.0, max=1.0)
        rotation = u @ torch.diag(correction) @ vh
        aligned_rmsd[batch_index] = torch.sqrt(((x @ rotation - y) ** 2).sum(dim=-1).mean()) / 10.0

    local_pair = pair_valid & same_chain & (sequence_separation > 0) & (sequence_separation <= 8)
    local_count = local_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
    signed_local_distance = ((f_dist - c_dist) * local_pair.float()).sum(dim=(-1, -2)) / local_count / 10.0
    return torch.cat(
        [
            torch.stack([retained, gained, lost], dim=-1),
            signed_torsion,
            absolute_torsion,
            rg_change[:, None],
            aligned_rmsd[:, None],
            signed_local_distance[:, None],
        ],
        dim=-1,
    ).clamp(-5.0, 5.0)


@torch.no_grad()
def compute_v3_local_observable_targets(
    context_coords: torch.Tensor,
    future_coords: torch.Tensor,
    context_atom_mask: torch.Tensor,
    future_atom_mask: torch.Tensor,
    context_residue_mask: torch.Tensor,
    future_residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the V3 physical transition vector within sequence segments."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    B, _, L = context_residue_mask.shape
    common = context_residue_mask[:, -1] & future_residue_mask[:, -1]
    position = torch.arange(L, device=common.device).view(1, L).expand(B, -1)
    length = common.sum(dim=1, keepdim=True).clamp(min=1)
    segment_id = torch.div(position * num_tokens, length, rounding_mode="floor").clamp(max=num_tokens - 1)
    targets = []
    masks = []
    for token_index in range(num_tokens):
        segment = (segment_id == token_index) & common
        context_segment_mask = context_residue_mask & segment[:, None]
        future_segment_mask = future_residue_mask & segment[:, None]
        targets.append(
            compute_v3_paired_observable_targets(
                context_coords, future_coords,
                context_atom_mask, future_atom_mask,
                context_segment_mask, future_segment_mask,
                chain_id, residue_index,
                contact_cutoff=contact_cutoff,
            )
        )
        masks.append(segment.sum(dim=1) >= 3)
    return torch.stack(targets, dim=1), torch.stack(masks, dim=1)


def _compute_v3_closed_local_state(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
    differentiable_distances: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one endpoint into composable, SE(3)-invariant local states."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    endpoint = coords[:, -1, :, 1]
    valid = residue_mask[:, -1] & atom_mask[:, -1, :, 1]
    B, L = valid.shape
    position = torch.arange(L, device=coords.device).view(1, L).expand(B, -1)
    length = valid.sum(dim=1, keepdim=True).clamp(min=1)
    segment_id = torch.div(position * num_tokens, length, rounding_mode="floor").clamp(max=num_tokens - 1)
    if differentiable_distances:
        delta = endpoint[:, :, None] - endpoint[:, None, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
    else:
        distances = torch.cdist(endpoint, endpoint)
    sequence_separation = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    torsions, torsion_mask = _backbone_torsions(coords[:, -1:], atom_mask[:, -1:], residue_mask[:, -1:], chain_id)
    torsions = torsions[:, 0]
    torsion_mask = torsion_mask[:, 0]
    all_weight = valid.float().unsqueeze(-1)
    protein_center = (endpoint * all_weight).sum(dim=1) / all_weight.sum(dim=1).clamp(min=1.0)
    states = []
    masks = []
    for token_index in range(num_tokens):
        segment = (segment_id == token_index) & valid
        segment_pair = segment[:, :, None] & segment[:, None, :] & same_chain
        nonlocal_pair = segment_pair & (sequence_separation > 2)
        possible = nonlocal_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
        contacts = (distances < contact_cutoff) & nonlocal_pair
        contact_density = contacts.sum(dim=(-1, -2)).float() / possible

        segment_torsion_mask = torsion_mask & segment.unsqueeze(-1)
        torsion_count = segment_torsion_mask.sum(dim=1).float().clamp(min=1.0)
        torsion_sin = (torch.sin(torsions) * segment_torsion_mask.float()).sum(dim=1) / torsion_count
        torsion_cos = (torch.cos(torsions) * segment_torsion_mask.float()).sum(dim=1) / torsion_count

        weight = segment.float().unsqueeze(-1)
        count = weight.sum(dim=1).clamp(min=1.0)
        center = (endpoint * weight).sum(dim=1) / count
        rg = torch.sqrt((((endpoint - center[:, None]) ** 2) * weight).sum(dim=(1, 2)) / count.squeeze(-1)).clamp(min=1.0e-6) / 10.0

        local_pair = segment_pair & (sequence_separation > 0) & (sequence_separation <= 8)
        local_count = local_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
        local_mean = (distances * local_pair.float()).sum(dim=(-1, -2)) / local_count / 10.0
        local_variance = (
            ((distances - local_mean[:, None, None] * 10.0).square() * local_pair.float()).sum(dim=(-1, -2))
            / local_count
        )
        local_std = torch.sqrt(local_variance.clamp(min=1.0e-6)) / 10.0
        center_distance = torch.linalg.vector_norm(center - protein_center, dim=-1) / 10.0
        contact_count = contacts.sum(dim=(-1, -2)).float().clamp(min=1.0)
        contact_order = (
            (sequence_separation.float() * contacts.float()).sum(dim=(-1, -2))
            / contact_count
            / length.squeeze(1).float().clamp(min=1.0)
        )
        states.append(
            torch.cat(
                [
                    contact_density[:, None], torsion_sin, torsion_cos,
                    rg[:, None], local_mean[:, None], local_std[:, None],
                    center_distance[:, None], contact_order[:, None],
                ],
                dim=-1,
            )
        )
        masks.append(segment.sum(dim=1) >= 3)
    return torch.stack(states, dim=1).clamp(-5.0, 5.0), torch.stack(masks, dim=1)


@torch.no_grad()
def compute_v3_closed_local_state(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_v3_closed_local_state(
        coords, atom_mask, residue_mask, chain_id, residue_index,
        num_tokens=num_tokens, contact_cutoff=contact_cutoff,
        differentiable_distances=False,
    )


def compute_v4c_force_state(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable endpoint state used only by conservative force matching."""
    return _compute_v3_closed_local_state(
        coords, atom_mask, residue_mask, chain_id, residue_index,
        num_tokens=num_tokens, contact_cutoff=contact_cutoff,
        differentiable_distances=True,
    )


@torch.no_grad()
def compute_v3_closed_local_state_sequence(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute closed local physical states at every observed frame."""
    B, T, L, A, C = coords.shape
    states, masks = compute_v3_closed_local_state(
        coords.reshape(B * T, 1, L, A, C),
        atom_mask.reshape(B * T, 1, L, A),
        residue_mask.reshape(B * T, 1, L),
        chain_id[:, None].expand(-1, T, -1).reshape(B * T, L),
        residue_index[:, None].expand(-1, T, -1).reshape(B * T, L),
        num_tokens=num_tokens,
        contact_cutoff=contact_cutoff,
    )
    return states.reshape(B, T, num_tokens, -1), masks.reshape(B, T, num_tokens)


@torch.no_grad()
def compute_v3_closed_local_velocity(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    current_state: torch.Tensor,
    state_scale: torch.Tensor,
    strides: tuple[int, ...] = (1, 2, 4, 8),
    num_tokens: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardized finite-difference rates from observed context frames only."""
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("closed local velocity strides must be positive")
    scale = state_scale.to(device=current_state.device, dtype=current_state.dtype)
    velocities = []
    masks = []
    for stride in strides:
        if stride >= coords.shape[1]:
            velocities.append(torch.zeros_like(current_state))
            masks.append(torch.zeros_like(current_state[..., 0], dtype=torch.bool))
            continue
        previous_state, previous_mask = compute_v3_closed_local_state(
            coords[:, :-stride], atom_mask[:, :-stride], residue_mask[:, :-stride],
            chain_id, residue_index, num_tokens=num_tokens,
        )
        velocities.append((current_state - previous_state) / scale / float(stride))
        masks.append(previous_mask)
    velocity = torch.stack(velocities, dim=2)
    velocity_mask = torch.stack(masks, dim=-1).all(dim=-1)
    return velocity, velocity_mask


def metastable_vamp_terms(
    outputs: dict,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    eps: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    """VAMP-2 and occupancy controls for the V3 metastable readout."""
    if "context_metastable_states" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {"vamp_loss": zero, "balance_loss": zero, "assignment_entropy": zero}
    sequences = [outputs["context_metastable_states"], outputs["future_metastable_states"]]
    masks = [outputs["context_metastable_mask"], outputs["future_metastable_mask"]]
    scores = []
    for lag in lags:
        x_parts = []
        y_parts = []
        for sequence, mask in zip(sequences, masks):
            if lag >= sequence.shape[1]:
                continue
            pair_mask = mask[:, :-lag] & mask[:, lag:]
            if pair_mask.any():
                x_parts.append(sequence[:, :-lag][pair_mask])
                y_parts.append(sequence[:, lag:][pair_mask])
        if not x_parts:
            continue
        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        dim = x.shape[-1]
        if x.shape[0] < dim + 2:
            continue
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            c00 = x.T @ x / max(x.shape[0] - 1, 1) + eps * eye
            c11 = y.T @ y / max(y.shape[0] - 1, 1) + eps * eye
            s0, u0 = torch.linalg.eigh(c00)
            s1, u1 = torch.linalg.eigh(c11)
            inv0 = torch.where(s0 > eps, s0.rsqrt(), torch.zeros_like(s0))
            inv1 = torch.where(s1 > eps, s1.rsqrt(), torch.zeros_like(s1))
            w0 = (u0 * inv0[None, :]) @ u0.T
            w1 = (u1 * inv1[None, :]) @ u1.T
        koopman = (x @ w0).T @ (y @ w1) / max(x.shape[0] - 1, 1)
        scores.append(koopman.square().sum())
    all_states = torch.cat(
        [sequence[mask] for sequence, mask in zip(sequences, masks)], dim=0
    )
    occupancy = all_states.mean(dim=0).clamp(min=1.0e-8)
    uniform = 1.0 / float(occupancy.numel())
    balance_loss = (occupancy * torch.log(occupancy / uniform)).sum()
    assignment_entropy = -(all_states.clamp(min=1.0e-8) * torch.log(all_states.clamp(min=1.0e-8))).sum(dim=-1).mean()
    if not scores:
        vamp_loss = all_states.new_tensor(0.0)
    else:
        score = torch.stack(scores).mean()
        vamp_loss = 1.0 - (score - 1.0) / max(all_states.shape[-1] - 1, 1)
    return {
        "vamp_loss": vamp_loss,
        "balance_loss": balance_loss,
        "assignment_entropy": assignment_entropy,
    }


def v4_slow_kinetic_terms(
    outputs: dict,
    lags: tuple[int, ...] = (4, 8, 16),
) -> dict[str, torch.Tensor]:
    """VAMP, reconstruction, and compositional transition terms for V4-B."""
    if "v4_slow_prediction" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {
            "vamp_loss": zero,
            "balance_loss": zero,
            "assignment_entropy": zero,
            "reconstruction_loss": zero,
            "base_transition_loss": zero,
            "base_transition_skill": zero,
            "rollout_transition_loss": zero,
            "rollout_transition_skill": zero,
            "mean_lambda2": zero,
        }
    vamp = metastable_vamp_terms(
        {
            "delta_target": outputs["delta_target"],
            "context_metastable_states": outputs["v4_slow_context_assignment"],
            "future_metastable_states": outputs["v4_slow_future_assignment"],
            "context_metastable_mask": outputs["v4_slow_context_mask"],
            "future_metastable_mask": outputs["v4_slow_future_mask"],
        },
        lags=lags,
    )
    context_mask = outputs["v4_slow_context_mask"]
    future_mask = outputs["v4_slow_future_mask"]
    reconstruction_loss = 0.5 * (
        F.smooth_l1_loss(
            outputs["v4_slow_context_reconstruction"][context_mask],
            outputs["v4_slow_context_normalized"][context_mask],
        )
        + F.smooth_l1_loss(
            outputs["v4_slow_future_reconstruction"][future_mask],
            outputs["v4_slow_future_normalized"][future_mask],
        )
    )
    base_target = outputs["v4_slow_base_target"].detach()
    base_prediction = outputs["v4_slow_base_prediction"]
    base_current = outputs["v4_slow_base_current"]
    base_transition_loss = F.mse_loss(base_prediction, base_target)
    base_persistence = F.mse_loss(base_current, base_target)
    rollout_target = outputs["v4_slow_target_assignment"].detach()
    rollout_prediction = outputs["v4_slow_prediction"]
    rollout_current = outputs["v4_slow_current_assignment"]
    rollout_transition_loss = F.mse_loss(rollout_prediction, rollout_target)
    rollout_persistence = F.mse_loss(rollout_current, rollout_target)
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvals(outputs["v4_slow_transition_matrix"]).abs()
        mean_lambda2 = eigenvalues.sort(dim=-1, descending=True).values[:, 1].mean().real
    return {
        "vamp_loss": vamp["vamp_loss"],
        "balance_loss": vamp["balance_loss"],
        "assignment_entropy": vamp["assignment_entropy"],
        "reconstruction_loss": reconstruction_loss,
        "base_transition_loss": base_transition_loss,
        "base_transition_skill": 1.0 - base_transition_loss / base_persistence.clamp(min=1.0e-8),
        "rollout_transition_loss": rollout_transition_loss,
        "rollout_transition_skill": 1.0
        - rollout_transition_loss / rollout_persistence.clamp(min=1.0e-8),
        "mean_lambda2": mean_lambda2,
    }


def _per_trajectory_vamp_terms(
    context_states: torch.Tensor,
    future_states: torch.Tensor,
    context_mask: torch.Tensor,
    future_mask: torch.Tensor,
    lags: tuple[int, ...],
    eps: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    """VAMP and occupancy computed after centering every trajectory independently."""
    scores = []
    occupancies = []
    entropies = []
    dim = context_states.shape[-1]
    eye = torch.eye(dim, device=context_states.device, dtype=context_states.dtype)
    for batch_index in range(context_states.shape[0]):
        sample_states = []
        for sequence, mask in (
            (context_states, context_mask), (future_states, future_mask)
        ):
            valid_states = sequence[batch_index][mask[batch_index]]
            if valid_states.numel():
                sample_states.append(valid_states)
                entropies.append(
                    -(valid_states.clamp(min=1.0e-8) * valid_states.clamp(min=1.0e-8).log())
                    .sum(dim=-1)
                    .mean()
                )
        if sample_states:
            occupancy = torch.cat(sample_states, dim=0).mean(dim=0).clamp(min=1.0e-8)
            occupancies.append(occupancy)
        for lag in lags:
            x_parts = []
            y_parts = []
            for sequence, mask in (
                (context_states, context_mask), (future_states, future_mask)
            ):
                if lag >= sequence.shape[1]:
                    continue
                pair_mask = mask[batch_index, :-lag] & mask[batch_index, lag:]
                if pair_mask.any():
                    x_parts.append(sequence[batch_index, :-lag][pair_mask])
                    y_parts.append(sequence[batch_index, lag:][pair_mask])
            if not x_parts:
                continue
            x = torch.cat(x_parts, dim=0)
            y = torch.cat(y_parts, dim=0)
            if x.shape[0] < dim + 2:
                continue
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)
            denominator = max(x.shape[0] - 1, 1)
            with torch.no_grad():
                c00 = x.T @ x / denominator + eps * eye
                c11 = y.T @ y / denominator + eps * eye
                s0, u0 = torch.linalg.eigh(c00)
                s1, u1 = torch.linalg.eigh(c11)
                w0 = (u0 * s0.clamp(min=eps).rsqrt().unsqueeze(0)) @ u0.T
                w1 = (u1 * s1.clamp(min=eps).rsqrt().unsqueeze(0)) @ u1.T
            koopman = (x @ w0).T @ (y @ w1) / denominator
            scores.append(koopman.square().sum())
    zero = context_states.new_tensor(0.0)
    if scores:
        score = torch.stack(scores).mean()
        vamp_loss = 1.0 - (score - 1.0) / max(dim - 1, 1)
    else:
        vamp_loss = zero
    if occupancies:
        occupancy = torch.stack(occupancies)
        uniform = 1.0 / float(dim)
        trajectory_balance = (
            occupancy * (occupancy / uniform).clamp(min=1.0e-8).log()
        ).sum(dim=-1).mean()
    else:
        trajectory_balance = zero
    assignment_entropy = torch.stack(entropies).mean() if entropies else zero
    return {
        "vamp_loss": vamp_loss,
        "trajectory_balance_loss": trajectory_balance,
        "assignment_entropy": assignment_entropy,
    }


def _whitened_cross_covariance_loss(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-3
) -> torch.Tensor:
    if x.shape[0] < 4:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    denominator = max(x.shape[0] - 1, 1)
    with torch.no_grad():
        def inverse_sqrt_covariance(values: torch.Tensor) -> torch.Tensor:
            covariance = values.T @ values / denominator
            covariance = covariance + eps * torch.eye(
                covariance.shape[0], device=values.device, dtype=values.dtype
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            inverse = eigenvalues.clamp(min=eps).rsqrt()
            return (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T

        x_whitener = inverse_sqrt_covariance(x)
        y_whitener = inverse_sqrt_covariance(y)
    cross = (x @ x_whitener).T @ (y @ y_whitener) / denominator
    return cross.square().mean()


def _lag_autocorrelation(values: torch.Tensor, mask: torch.Tensor, lag: int) -> torch.Tensor:
    if lag <= 0 or lag >= values.shape[1]:
        return values.new_tensor(0.0)
    pair_mask = mask[:, :-lag] & mask[:, lag:]
    if not pair_mask.any():
        return values.new_tensor(0.0)
    left = values[:, :-lag][pair_mask]
    right = values[:, lag:][pair_mask]
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    numerator = (left * right).mean()
    denominator = torch.sqrt(left.square().mean() * right.square().mean()).clamp(min=1.0e-8)
    return numerator / denominator


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
