"""Stage A model: EMA-target JEPA with joint masked + multi-horizon endpoint prediction.

Architecture invariant (from audit plan Gate 1):
- online branch: receives masked context, produces predictions
- EMA branch: always no_grad, eval mode, unmasked (full context) for targets
- multi-horizon targets use horizon_time_ps (seconds), not frame indices
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.dynamic_encoder import DynamicStateEncoder
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.predictor import (
    PairedDeltaPredictor,
    SpatiotemporalPredictor,
    StatePredictor,
)
from protein_jepa_frame.models.temporal import TemporalStateEncoder


@dataclass(frozen=True)
class StageAConfig:
    """Minimal config for Stage A training.
    
    This is a subset of ModelConfig that Stage A actually uses.
    Full ModelConfig is still accepted for backward compatibility.
    """
    latent_dim: int = 256
    encoder_depth: int = 4
    predictor_depth: int = 4
    temporal_depth: int = 2
    state_predictor_depth: int = 2
    num_heads: int = 4
    num_rbf: int = 16
    dropout: float = 0.1
    max_frames: int = 256
    max_residues: int = 4096
    feature_norm: bool = True
    time_conditioning: str = "none"
    thermodynamic_conditioning: bool = False
    orientation_features: bool = False
    frame_pooling: str = "mean"
    frame_pool_queries: int = 4
    temporal_causal: bool = False
    temporal_causal_window: int = 0
    temporal_position_encoding: str = "learned"
    temporal_delta_features: bool = False
    transition_thermodynamic_conditioning: bool = False
    causal_encoder_gradient_scale: float = 1.0
    paired_teacher_space_delta: bool = False
    paired_target_tower: str = "causal"
    dual_temporal_towers: bool = False
    offline_temporal_depth: int = 0
    causal_temporal_depth: int = 0
    opf_enabled: bool = False
    opf_factor_count: int = 4
    opf_shared_trunk: bool = True
    opf_predictor_depth: int = 2
    opf_activity_gamma: float = 0.01
    opf_encoder_activity_gamma: float = 0.01
    # Gate 1: coordinate-derived dynamic state encoder
    use_dynamic_encoder: bool = False
    dynamic_encoder_depth: int = 2
    dynamic_k_nearest: int = 8
    dynamic_use_displacement: bool = True
    dynamic_use_pair_distance: bool = True
    dynamic_use_torsion: bool = True
    dynamic_use_dt_conditioning: bool = True
    dynamic_fusion: str = "add"  # "add" | "gate" | "replace"

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "StageAConfig":
        """Extract the Stage A subset from a full ModelConfig."""
        return cls(**{
            k: getattr(config, k) for k in cls.__dataclass_fields__
            if hasattr(config, k)
        })


class StageAJEPA(nn.Module):
    """Canonical Stage A model: joint masked prediction + multi-horizon endpoints.
    
    Freeze contract (Gate 1):
    - context_encoder + temporal_encoder: online, receives mask
    - target_encoder + target_temporal_encoder: EMA, always eval + no_grad
    - predictor + state_predictor + paired_delta_predictor: online, receives context
    - When freeze_context_encoder=True, online encoder also eval + no_grad
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.stage_a_config = StageAConfig.from_model_config(config)

        # Spatial encoders (EMA pair)
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

        # Mask prediction head
        self.predictor = SpatiotemporalPredictor(
            latent_dim=config.latent_dim,
            depth=config.predictor_depth,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_frames=config.max_frames,
            max_residues=config.max_residues,
            time_conditioning=config.time_conditioning,
        )

        # Temporal encoder (EMA pair)
        causal_depth = (
            config.causal_temporal_depth
            if config.dual_temporal_towers and config.causal_temporal_depth > 0
            else config.temporal_depth
        )
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
                causal=config.temporal_causal if not config.dual_temporal_towers else True,
                causal_window=config.temporal_causal_window,
                position_encoding=config.temporal_position_encoding,
                delta_features=config.temporal_delta_features,
                motion_depth=0,
                motion_stopgrad_input=False,
            )
            if causal_depth > 0
            else None
        )
        self.target_temporal_encoder = (
            copy.deepcopy(self.temporal_encoder) if self.temporal_encoder is not None else None
        )
        if self.target_temporal_encoder is not None:
            for param in self.target_temporal_encoder.parameters():
                param.requires_grad = False

        # Offline (non-causal) temporal tower
        self.offline_temporal_encoder = None
        self.target_offline_temporal_encoder = None
        if config.dual_temporal_towers:
            offline_depth = (
                config.offline_temporal_depth
                if config.offline_temporal_depth > 0
                else config.temporal_depth
            )
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

        # Gate 1: coordinate-derived dynamic encoder (EMA pair)
        self.dynamic_encoder = None
        self.target_dynamic_encoder = None
        if config.use_dynamic_encoder:
            self.dynamic_encoder = DynamicStateEncoder(
                latent_dim=config.latent_dim,
                depth=config.dynamic_encoder_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
                k_nearest=config.dynamic_k_nearest,
                use_displacement=config.dynamic_use_displacement,
                use_pair_distance_changes=config.dynamic_use_pair_distance,
                use_torsion_changes=config.dynamic_use_torsion,
                use_dt_conditioning=config.dynamic_use_dt_conditioning,
            )
            self.target_dynamic_encoder = copy.deepcopy(self.dynamic_encoder)
            for param in self.target_dynamic_encoder.parameters():
                param.requires_grad = False

            # Fusion mechanism for combining static temporal and dynamic features
            if config.dynamic_fusion == "gate":
                self.dynamic_gate = nn.Parameter(torch.zeros(config.latent_dim))
            else:
                self.dynamic_gate = None

        # Frame-level state predictor
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

        # Endpoint delta predictor for paired horizons
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

        self._curriculum_step = 0

    def train(self, mode: bool = True) -> "StageAJEPA":
        super().train(mode)
        self.target_encoder.eval()
        if self.target_temporal_encoder is not None:
            self.target_temporal_encoder.eval()
        if self.target_offline_temporal_encoder is not None:
            self.target_offline_temporal_encoder.eval()
        if self.target_dynamic_encoder is not None:
            self.target_dynamic_encoder.eval()
        if self.config.freeze_context_encoder:
            self.context_encoder.eval()
        if self.config.freeze_temporal_encoder and self.temporal_encoder is not None:
            self.temporal_encoder.eval()
        return self

    @torch.no_grad()
    def sync_target_encoders(self) -> None:
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        if self.target_temporal_encoder is not None and self.temporal_encoder is not None:
            self.target_temporal_encoder.load_state_dict(self.temporal_encoder.state_dict())
        if self.target_offline_temporal_encoder is not None and self.offline_temporal_encoder is not None:
            self.target_offline_temporal_encoder.load_state_dict(
                self.offline_temporal_encoder.state_dict()
            )
        if self.target_dynamic_encoder is not None and self.dynamic_encoder is not None:
            self.target_dynamic_encoder.load_state_dict(self.dynamic_encoder.state_dict())
        self._curriculum_step = 0

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if self.config.freeze_target_encoders:
            return
        for target, context in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_temporal_encoder is not None and self.temporal_encoder is not None:
            for target, context in zip(
                self.target_temporal_encoder.parameters(),
                self.temporal_encoder.parameters(),
            ):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_offline_temporal_encoder is not None and self.offline_temporal_encoder is not None:
            for target, context in zip(
                self.target_offline_temporal_encoder.parameters(),
                self.offline_temporal_encoder.parameters(),
            ):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_dynamic_encoder is not None and self.dynamic_encoder is not None:
            for target, context in zip(
                self.target_dynamic_encoder.parameters(),
                self.dynamic_encoder.parameters(),
            ):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)

    def forward_mask_prediction(
        self, batch: dict, context_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> dict:
        """Clip-level masked JEPA prediction (residue + state level)."""
        coords = batch["coords"]
        residue_mask = batch["residue_mask"]
        aatype = batch["aatype"]
        residue_index = batch["residue_index"]
        chain_id = batch["chain_id"]
        time_ps = batch.get("time_ps")
        temperature_k = batch.get("temperature_k")
        dataset_index = batch.get("dataset_index")

        # Online branch: masked context
        context_latents = self.context_encoder(
            coords=coords, residue_mask=residue_mask,
            aatype=aatype, residue_index=residue_index,
            chain_id=chain_id, token_mask=context_mask,
        )

        # EMA branch: full visibility, no gradient
        with torch.no_grad():
            target_latents = self.target_encoder(
                coords=coords, residue_mask=residue_mask,
                aatype=aatype, residue_index=residue_index,
                chain_id=chain_id, token_mask=residue_mask,
            )
            if self.config.feature_norm:
                target_latents = F.layer_norm(target_latents, (target_latents.shape[-1],))

        predictions = self.predictor(
            context_latents=context_latents, context_mask=context_mask,
            target_mask=target_mask, residue_index=residue_index,
            chain_id=chain_id, time_ps=time_ps,
        )

        outputs = {
            "predictions": predictions,
            "targets": target_latents,
            "target_mask": target_mask,
            "context_latents": context_latents,
        }

        # Temporal / state prediction
        if self.temporal_encoder is not None:
            context_states = self.temporal_encoder(
                context_latents, context_mask,
                time_ps=time_ps, temperature_k=temperature_k,
                dataset_index=dataset_index,
            )

            # Gate 1: fuse coordinate-derived dynamic state into temporal states.
            # The dynamic encoder is driven purely by coordinate changes (aligned
            # displacement, pair-distance changes, torsion changes) so it injects
            # motion information that does NOT leak static identity or frame-index clocks.
            if self.dynamic_encoder is not None:
                dt = batch.get("dt_ps")
                dynamic_out = self.dynamic_encoder(
                    coords=coords,
                    residue_mask=residue_mask,  # Gate 1 fix: physical mask, not JEPA context_mask
                    residue_index=residue_index, chain_id=chain_id, dt=dt,
                )
                context_states = self._fuse_dynamic(context_states, dynamic_out)

            outputs.update(context_states)

            with torch.no_grad():
                target_states = self.target_temporal_encoder(
                    target_latents, residue_mask,
                    time_ps=time_ps, temperature_k=temperature_k,
                    dataset_index=dataset_index,
                )
                if self.target_dynamic_encoder is not None:
                    dt = batch.get("dt_ps")
                    target_dynamic_out = self.target_dynamic_encoder(
                        coords=coords, residue_mask=residue_mask,
                        residue_index=residue_index, chain_id=chain_id, dt=dt,
                    )
                    target_states = self._fuse_dynamic(target_states, target_dynamic_out)
                target_frame_states = target_states["frame_states"]
                if self.config.feature_norm:
                    target_frame_states = F.layer_norm(
                        target_frame_states, (target_frame_states.shape[-1],)
                    )

            context_frame_mask = context_mask.any(dim=2)
            target_frame_mask = target_mask.any(dim=2)
            state_predictions = self.state_predictor(
                context_states=context_states["frame_states"],
                context_frame_mask=context_frame_mask,
                target_frame_mask=target_frame_mask,
                time_ps=time_ps,
            )
            outputs.update({
                "state_predictions": state_predictions,
                "state_targets": target_frame_states,
                "state_target_mask": target_frame_mask,
                "context_frame_mask": context_frame_mask,
                "target_frame_states": target_frame_states,
            })

        return outputs

    def forward_paired(
        self, batch: dict
    ) -> dict[str, torch.Tensor]:
        """Paired-window endpoint prediction at physical horizons."""
        if self.temporal_encoder is None or self.paired_delta_predictor is None:
            raise ValueError("paired forecast requires temporal_encoder and paired_delta_predictor")

        # Online context encoding
        context_latents = self.context_encoder(
            coords=batch["context_coords"],
            residue_mask=batch["context_residue_mask"],
            aatype=batch["context_aatype"],
            residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
            token_mask=batch["context_residue_mask"],
        )
        
        causal_scale = float(getattr(self.config, 'causal_encoder_gradient_scale', 1.0))
        if not 0.0 <= causal_scale <= 1.0:
            raise ValueError("causal_encoder_gradient_scale must be in [0, 1]")
        gated_context_latents = context_latents.detach() + causal_scale * (
            context_latents - context_latents.detach()
        )

        context_states = self.temporal_encoder(
            gated_context_latents,
            batch["context_residue_mask"],
            time_ps=batch.get("context_time_ps"),
            temperature_k=batch.get("temperature_k"),
            dataset_index=batch.get("dataset_index"),
        )
        if self.dynamic_encoder is not None:
            dt = batch.get("context_dt_ps")
            dynamic_out = self.dynamic_encoder(
                coords=batch["context_coords"],
                residue_mask=batch["context_residue_mask"],
                residue_index=batch["context_residue_index"],
                chain_id=batch["context_chain_id"],
                dt=dt,
            )
            context_states = self._fuse_dynamic(context_states, dynamic_out)

        # EMA targets
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
                future_latents = F.layer_norm(
                    future_latents, (future_latents.shape[-1],)
                )
            teacher_context_states = self.target_temporal_encoder(
                teacher_context_latents,
                batch["context_residue_mask"],
                time_ps=batch.get("context_time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            if self.target_dynamic_encoder is not None:
                dt = batch.get("context_dt_ps")
                teacher_dynamic_out = self.target_dynamic_encoder(
                    coords=batch["context_coords"],
                    residue_mask=batch["context_residue_mask"],
                    residue_index=batch["context_residue_index"],
                    chain_id=batch["context_chain_id"],
                    dt=dt,
                )
                teacher_context_states = self._fuse_dynamic(teacher_context_states, teacher_dynamic_out)
            future_states = self.target_temporal_encoder(
                future_latents,
                batch["future_residue_mask"],
                time_ps=batch.get("future_time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            if self.target_dynamic_encoder is not None:
                dt = batch.get("future_dt_ps")
                future_dynamic_out = self.target_dynamic_encoder(
                    coords=batch["future_coords"],
                    residue_mask=batch["future_residue_mask"],
                    residue_index=batch["future_residue_index"],
                    chain_id=batch["future_chain_id"],
                    dt=dt,
                )
                future_states = self._fuse_dynamic(future_states, future_dynamic_out)

            # Endpoint encoding
            context_endpoint_ema = self._mean_over_valid(
                teacher_context_states["frame_states"],
                batch["context_residue_mask"].any(dim=2),
            )
            future_endpoint_ema = self._mean_over_valid(
                future_states["frame_states"],
                batch["future_residue_mask"].any(dim=2),
            )
            if self.config.feature_norm:
                context_endpoint_ema = F.layer_norm(
                    context_endpoint_ema, (context_endpoint_ema.shape[-1],)
                )
                future_endpoint_ema = F.layer_norm(
                    future_endpoint_ema, (future_endpoint_ema.shape[-1],)
                )

        # Online endpoint
        context_endpoint = self._mean_over_valid(
            context_states["frame_states"],
            batch["context_residue_mask"].any(dim=2),
        )
        if self.config.feature_norm:
            context_endpoint = F.layer_norm(
                context_endpoint, (context_endpoint.shape[-1],)
            )

        # Delta prediction
        if self.config.paired_teacher_space_delta:
            delta_anchor = context_endpoint_ema
            prediction_anchor = context_endpoint_ema
        else:
            delta_anchor = context_endpoint.detach()
            prediction_anchor = context_endpoint

        delta_target = future_endpoint_ema - delta_anchor
        delta_prediction = self.paired_delta_predictor(
            context_endpoint,
            batch["horizon_time_ps"],
            temperature_k=batch.get("temperature_k"),
            dataset_index=batch.get("dataset_index"),
        )

        outputs = {
            "context_endpoint": context_endpoint,
            "future_endpoint": future_endpoint_ema,
            "delta_target": delta_target,
            "delta_prediction": delta_prediction,
            "state_prediction": prediction_anchor + delta_prediction,
            "horizon_time_ps": batch["horizon_time_ps"],
            "horizon_frames": batch["horizon_frames"],
            "online_context_state": context_endpoint.detach(),
        }

        return outputs

    def forward(
        self,
        batch: dict,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        task: str = "mask",
    ) -> dict:
        """Unified forward: dispatch based on task mode.
        
        Args:
            batch: input batch dict
            context_mask: mask for context tokens (mask mode only)
            target_mask: mask for target tokens (mask mode only)
            task: "mask" for clip masked prediction, "paired" for endpoint forecast
            
        Returns:
            dict with task-specific outputs
        """
        if task == "mask":
            if context_mask is None or target_mask is None:
                raise ValueError("mask task requires context_mask and target_mask")
            return self.forward_mask_prediction(batch, context_mask, target_mask)
        elif task == "paired":
            return self.forward_paired(batch)
        else:
            raise ValueError(f"Unknown task: {task}")


    def _fuse_dynamic(
        self, temporal_states: dict[str, torch.Tensor], dynamic_states: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fuse coordinate-derived dynamic states into temporal states.

        Gate 1 requirement: dynamic state is driven by coordinate changes, not timestamps.
        The fusion mode determines how static temporal features and dynamic motion features
        are combined.

        Args:
            temporal_states: dict with "frame_states" [B, T, D], "clip_state" [B, D]
            dynamic_states: dict with "dynamic_frame_states" [B, T, D], "dynamic_clip_state" [B, D]

        Returns:
            fused dict with same keys as temporal_states
        """
        fusion = self.config.dynamic_fusion

        if fusion == "replace":
            # Replace temporal states entirely with dynamic states
            return {
                "frame_states": dynamic_states["dynamic_frame_states"],
                "clip_state": dynamic_states["dynamic_clip_state"],
            }
        elif fusion == "add":
            # Simple addition: static + dynamic
            return {
                "frame_states": temporal_states["frame_states"] + dynamic_states["dynamic_frame_states"],
                "clip_state": temporal_states["clip_state"] + dynamic_states["dynamic_clip_state"],
            }
        elif fusion == "gate":
            # Gated fusion: static + gate * dynamic
            gate = torch.sigmoid(self.dynamic_gate)
            return {
                "frame_states": temporal_states["frame_states"] + gate * dynamic_states["dynamic_frame_states"],
                "clip_state": temporal_states["clip_state"] + gate * dynamic_states["dynamic_clip_state"],
            }
        else:
            raise ValueError(f"Unknown dynamic_fusion mode: {fusion}")

    @staticmethod
    def _mean_over_valid(
        states: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean over time dimension using valid frames only."""
        # states: [B, T, D], valid_mask: [B, T]
        mask = valid_mask.unsqueeze(-1).to(states.dtype)
        return (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
