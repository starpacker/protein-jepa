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
from protein_jepa_frame.models.opf import OPFPredictor, opf_terms
from protein_jepa_frame.models.temporal import TemporalStateEncoder
from protein_jepa_frame.models.thermodynamic import ThermodynamicMoriJEPA
from protein_jepa_frame.models.thermodynamic_v3 import ThermodynamicMoriJEPAV3
from protein_jepa_frame.models.lspwm import LSPWMProteinWorldModel
from protein_jepa_frame.models.jepa_losses import (
    _latent_loss,
    residue_jepa_loss,
    state_jepa_loss,
    state_transition_jepa_loss,
    lagged_state_semigroup_loss,
    lagged_state_semigroup_terms,
    lagged_state_jepa_loss,
    paired_forecast_terms,
    _slow_cross_entropy,
    _continuous_slow_vamp2,
    slow_forecast_terms,
    reversible_kinetic_terms,
    paired_endpoint_variance_regularizer,
    paired_causal_alignment_loss,
    lagged_state_jepa_terms,
    physchem_prediction_loss,
    linear_dynamics_jepa_loss,
    kinetic_vampnet_loss,
    jepa_loss,
    variance_regularizer,
    state_variance_regularizer,
    state_feature_variance_regularizer,
    state_covariance_regularizer,
    _dihedral,
    _backbone_torsions,
    compute_physchem_targets,
    compute_paired_observable_targets,
    compute_v3_paired_observable_targets,
    compute_v3_local_observable_targets,
    _compute_v3_closed_local_state,
    compute_v3_closed_local_state,
    compute_v4c_force_state,
    compute_v3_closed_local_state_sequence,
    compute_v3_closed_local_velocity,
    metastable_vamp_terms,
    v4_slow_kinetic_terms,
    _per_trajectory_vamp_terms,
    _whitened_cross_covariance_loss,
    _lag_autocorrelation,
)
from protein_jepa_frame.models.jepa_losses_extended import (
    v4_physics_terms,
    v5_contextual_dynamics_terms,
    _trajectory_centered_learned_cv_terms,
    tmjepa_terms,
    v51_sparse_motion_terms,
    local_observable_prediction_loss,
    closed_local_state_terms,
    gaussian_transition_nll,
    observable_prediction_loss,
    offline_forecast_terms,
)



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
        if config.opf_enabled and not config.paired_forecast:
            raise ValueError("OPF requires paired forecasting")
        if config.opf_enabled and config.opf_factor_count <= 0:
            raise ValueError("OPF factor count must be positive")
        if config.opf_enabled and config.latent_dim % config.opf_factor_count != 0:
            raise ValueError("OPF factor count must divide latent_dim")
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
        if config.lspwm_enabled and config.lspwm_learned_cv_dim > config.latent_dim:
            raise ValueError(
                "LSP-WM learned CV dimension cannot exceed JEPA latent_dim"
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
        # OPF: orthogonal predictive factorization of the paired delta target.
        # When factor_count == 1 this is the matched single-head baseline; OPF turns
        # the single full-dim prediction into K orthogonal r-dim factors + synthesis.
        self.opf_predictor = (
            OPFPredictor(
                latent_dim=config.latent_dim,
                num_factors=config.opf_factor_count if config.opf_enabled else 1,
                shared_trunk=config.opf_shared_trunk,
                depth=config.opf_predictor_depth,
                dropout=config.dropout,
                thermodynamic_conditioning=config.transition_thermodynamic_conditioning,
                activity_gamma=config.opf_activity_gamma,
                encoder_activity_gamma=config.opf_encoder_activity_gamma,
            )
            if config.paired_forecast and config.opf_enabled
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
                rollout_trajectory_steps=getattr(
                    config, "lspwm_rollout_trajectory_steps", 0
                ),
                semigroup_consistency_weight=getattr(
                    config, "lspwm_semigroup_consistency_weight", 0.0
                ),
                midpoint_integration=getattr(
                    config, "lspwm_midpoint_integration", False
                ),
                reversible_dynamics_enabled=getattr(
                    config, "lspwm_reversible_dynamics_enabled", False
                ),
                reversible_operator_rank=getattr(
                    config, "lspwm_reversible_operator_rank", 16
                ),
                reversible_hidden_dim=getattr(
                    config, "lspwm_reversible_hidden_dim", 128
                ),
                reversible_hidden_depth=getattr(
                    config, "lspwm_reversible_hidden_depth", 4
                ),
                per_residue_enabled=getattr(
                    config, "lspwm_per_residue_enabled", False
                ),
                per_residue_warmup_steps=getattr(
                    config, "lspwm_per_residue_warmup_steps", 2000
                ),
                residue_cv_dim=getattr(
                    config, "lspwm_residue_cv_dim", 8
                ),
                residue_gnn_depth=getattr(
                    config, "lspwm_residue_gnn_depth", 2
                ),
                residue_gnn_hidden=getattr(
                    config, "lspwm_residue_gnn_hidden", 64
                ),
                residue_force_direction_weight=getattr(
                    config, "lspwm_force_direction_weight", 0.3
                ),
                sliced_score_enabled=getattr(
                    config, "lspwm_sliced_score_enabled", False
                ),
                sliced_score_noise=getattr(
                    config, "lspwm_sliced_score_noise", 0.05
                ),
                sliced_score_projections=getattr(
                    config, "lspwm_sliced_score_projections", 4
                ),
                energy_graph_enabled=getattr(config, "lspwm_energy_graph_enabled", False),
                gauge_enabled=getattr(config, "lspwm_gauge_enabled", False),
                gauge_dim=getattr(config, "lspwm_gauge_dim", 6),
                bath_timescale_groups=getattr(config, "lspwm_bath_timescale_groups", (4, 8, 4)),
                bath_init_taus_ps=getattr(config, "lspwm_bath_init_taus_ps", (1000.0, 8000.0, 64000.0)),
                residue_conditional_enabled=getattr(config, "lspwm_residue_conditional_enabled", False),
                energy_graph_num_rbf=getattr(config, "lspwm_energy_graph_num_rbf", 8),
                energy_graph_hidden_dim=getattr(config, "lspwm_energy_graph_hidden_dim", 64),
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
        # ── OPF: orthogonal predictive factorization of the delta target ──
        # Replace the single full-dim prediction with the orthogonally factorized +
        # synthesized prediction. Geometric audit tensors are exported for M1.
        opf_prediction = None
        opf_factor_preds = None
        opf_factor_targets = None
        opf_factor_cond = None
        if self.opf_predictor is not None:
            opf_input = (
                offline_context_states["frame_states"][:, -1]
                if self.config.history_conditioned_transition and offline_context_states is not None
                else transition_source
            )
            if opf_input.dim() != 2:
                opf_input = opf_input.reshape(opf_input.shape[0], -1)
            opf_out = self.opf_predictor(
                opf_input,
                batch["horizon_time_ps"],
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            opf_prediction = opf_out["state_prediction"]
            opf_factor_preds = opf_out["factor_preds"]
            opf_factor_cond = opf_out["factor_cond"]
            # Project the stop-grad target into each factor's subspace (Eq 3).
            # The target latent is stopped (sg(z_t)); the projection P_k is NOT
            # detached so L_pred gradient shapes the projectors toward factor
            # coordinates that are predictable (paper Sec-2.2 "predictability
            # drives factorization"). opf_terms() must NOT re-detach these targets.
            opf_factor_targets = [
                (p.T @ delta_target.detach().T).T for p in self.opf_predictor.projectors
            ]
            # The synthesized delta is the OPF prediction (authoritative).
            delta_prediction = opf_prediction
        online_context_state = transition_source.detach()
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
            "online_context_state": online_context_state,
        }
        if self.opf_predictor is not None:
            outputs["opf_factor_preds"] = opf_factor_preds
            outputs["opf_factor_targets"] = opf_factor_targets
            outputs["opf_factor_cond"] = opf_factor_cond
            outputs["opf_projectors"] = self.opf_predictor
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
            lspwm_kwargs = dict(
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
            # v28: pass raw per-residue latents when per-residue dynamics are enabled
            if getattr(self.lspwm, 'per_residue_enabled', False):
                # context_latents from encoder: [B*T, L, 256] (flattened)
                # Derive B, T, L from the coordinate tensor
                B, T = batch["context_coords"].shape[:2]
                L = batch["context_coords"].shape[2]
                lspwm_kwargs["context_latents"] = context_latents.reshape(B, T, L, -1)
                # Reuse future_latents computed above (also flattened [B*T, L, D])
                B_future, T_future = batch["future_coords"].shape[:2]
                lspwm_kwargs["future_latents"] = future_latents.reshape(B_future, T_future, L, -1)
                # Cα distance graph from coordinates: [B, T, L, 3]
                lspwm_kwargs["context_ca_coords"] = batch["context_coords"][..., 1, :]
                lspwm_kwargs["future_ca_coords"] = batch["future_coords"][..., 1, :]
            outputs.update(self.lspwm(**lspwm_kwargs))
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


