from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.predictor import (
    HistoryConditionedTransition,
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
    SlowStateProjector,
    SpatiotemporalPredictor,
    StatePredictor,
)
from protein_jepa_frame.models.temporal import TemporalStateEncoder


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
            if config.paired_forecast and config.slow_state_enabled
            else None
        )
        self.target_slow_state_projector = copy.deepcopy(self.slow_state_projector)
        if self.target_slow_state_projector is not None:
            for param in self.target_slow_state_projector.parameters():
                param.requires_grad = False
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

    def train(self, mode: bool = True) -> ProteinJEPA:
        super().train(mode)
        # EMA teachers are deterministic targets, not independently dropped-out views.
        self.target_encoder.eval()
        if self.config.freeze_context_encoder:
            self.context_encoder.eval()
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

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
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
        if self.slow_state_projector is not None:
            context_slow_state = self.slow_state_projector(transition_source)
            with torch.no_grad():
                future_slow_state = self.target_slow_state_projector(future_endpoint)
                teacher_context_slow_state = self.target_slow_state_projector(context_teacher_endpoint)
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
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
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


def slow_forecast_terms(outputs: dict, loss_type: str = "smooth_l1") -> dict[str, torch.Tensor]:
    """Losses for a low-dimensional slow/metastable endpoint representation."""
    prediction = outputs["slow_delta_prediction"]
    target = outputs["slow_delta_target"]
    delta_loss = _latent_loss(prediction, target, loss_type)
    state_loss = _latent_loss(outputs["slow_state_prediction"], outputs["future_slow_state"], loss_type)
    persistence_loss = _latent_loss(torch.zeros_like(target), target, loss_type)
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
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
        "persistence_loss": persistence_loss,
        "skill": 1.0 - delta_loss / persistence_loss.clamp(min=1.0e-8),
        "contrastive_loss": _slow_cross_entropy(outputs["slow_state_prediction"], outputs["future_slow_state"]),
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
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
