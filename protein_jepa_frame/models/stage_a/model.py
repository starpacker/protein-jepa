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
from protein_jepa_frame.models.time_features import LogTimeEmbedding


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


class DynamicJEPAHead(nn.Module):
    """Predict a future-view per-residue dynamic H from context dynamic H.

    The target is produced by the EMA dynamic encoder and detached by the
    caller.  This is a JEPA transition in latent space, not a supervised
    coordinate/displacement head.
    """

    def __init__(self, latent_dim: int, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.time_embed = LogTimeEmbedding(latent_dim)
        hidden = latent_dim * 2
        layers: list[nn.Module] = [
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(
        self,
        context_residue_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
    ) -> torch.Tensor:
        if context_residue_state.dim() != 3:
            raise ValueError("context_residue_state must be [B,L,D]")
        if horizon_time_ps.dim() != 1 or horizon_time_ps.shape[0] != context_residue_state.shape[0]:
            raise ValueError("horizon_time_ps must be [B]")
        time = self.time_embed(horizon_time_ps).to(dtype=context_residue_state.dtype)
        x = context_residue_state + time[:, None, :]
        return self.out(self.trunk(x))


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

            self.dynamic_jepa_predictor = DynamicJEPAHead(
                latent_dim=config.latent_dim,
                depth=max(1, config.paired_predictor_depth),
                dropout=config.dropout,
            )
        else:
            self.dynamic_jepa_predictor = None

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

        # iter_010: per-residue displacement prediction head.
        # Consumes a per-residue motion feature = [last-frame H_res,
        # H_res(t_last) - H_res(t_last-k) temporal delta] so the GRADIENT reaches
        # the encoder's temporal axis and cannot be satisfied by a static code.
        # Last layer zero-init: predicts exactly zero (= persistence) at load.
        disp_hidden = int(getattr(config, "disp_head_hidden", 128))
        self.disp_use_temporal_delta = bool(getattr(config, "disp_use_temporal_delta", True))
        self.disp_delta_k = int(getattr(config, "disp_delta_k", 4))
        disp_in = config.latent_dim * (2 if self.disp_use_temporal_delta else 1)
        self.displacement_head = nn.Sequential(
            nn.Linear(disp_in, disp_hidden),
            nn.GELU(),
            nn.LayerNorm(disp_hidden),
            nn.Linear(disp_hidden, 3),
        )
        nn.init.zeros_(self.displacement_head[-1].weight)
        nn.init.zeros_(self.displacement_head[-1].bias)


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
        def _ema_module(target_module: nn.Module, online_module: nn.Module) -> None:
            for target, online in zip(target_module.parameters(), online_module.parameters()):
                target.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)
            for target, online in zip(target_module.buffers(), online_module.buffers()):
                if target.dtype.is_floating_point:
                    target.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)
                else:
                    target.data.copy_(online.data)

        _ema_module(self.target_encoder, self.context_encoder)
        if self.target_temporal_encoder is not None and self.temporal_encoder is not None:
            _ema_module(self.target_temporal_encoder, self.temporal_encoder)
        if self.target_offline_temporal_encoder is not None and self.offline_temporal_encoder is not None:
            _ema_module(self.target_offline_temporal_encoder, self.offline_temporal_encoder)
        if self.target_dynamic_encoder is not None and self.dynamic_encoder is not None:
            _ema_module(self.target_dynamic_encoder, self.dynamic_encoder)

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
            "context_mask": context_mask,
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
                    observation_mask=context_mask,
                )
                context_states = self._fuse_dynamic(
                    context_states, dynamic_out, residue_latents=context_latents
                )

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
                        observation_mask=residue_mask,
                    )
                    target_states = self._fuse_dynamic(
                        target_states, target_dynamic_out, residue_latents=target_latents
                    )
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
            context_states = self._fuse_dynamic(
                context_states, dynamic_out, residue_latents=context_latents
            )

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
                teacher_context_states = self._fuse_dynamic(
                    teacher_context_states, teacher_dynamic_out,
                    residue_latents=teacher_context_latents,
                )
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
                future_states = self._fuse_dynamic(
                    future_states, future_dynamic_out, residue_latents=future_latents
                )

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
            # Explicit H surfaces for Stage-A diagnostics and the later Stage-B
            # handoff.  They are not used as a coordinate-supervised shortcut.
            "context_frame_states": context_states["frame_states"],
            "context_residue_states": context_states.get("residue_states"),
            "future_residue_states": future_states.get("residue_states"),
        }

        if self.dynamic_encoder is not None and self.dynamic_jepa_predictor is not None:
            context_dynamic_endpoint = context_states["dynamic_residue_states"][:, -1]
            future_dynamic_endpoint = future_states["dynamic_residue_states"][:, -1]
            dynamic_prediction = self.dynamic_jepa_predictor(
                context_dynamic_endpoint, batch["horizon_time_ps"]
            )
            dynamic_valid = (
                batch["context_residue_mask"][:, -1]
                & batch["future_residue_mask"][:, -1]
            )
            outputs.update({
                "context_dynamic_endpoint": context_dynamic_endpoint,
                "dynamic_prediction": dynamic_prediction,
                "dynamic_target": future_dynamic_endpoint.detach(),
                "dynamic_target_mask": dynamic_valid,
                "dynamic_context_mask": batch["context_residue_mask"][:, -1],
            })

        return outputs

    def forward_displacement(
        self, batch: dict
    ) -> dict[str, torch.Tensor]:
        """Per-residue displacement prediction from context H.

        Supervision: Kabsch-aligned ground-truth displacements computed from
        the last context frame anchor and the last future frame target, both in
        the per-residue anchor backbone frame.  This forces H to encode WHERE
        each residue is moving, not just where it is now.

        Returns dict with keys:
          disp_pred   [B, L, 3]  predicted displacement in anchor frame
          disp_target [B, L, 3]  Kabsch-aligned target displacement in anchor frame
          valid_mask  [B, L]     bool, residues valid in both context and future
        """
        # ── Encode context (last frame only for displacement) ──
        # context_encoder expects [B,T,L,4,3]; run full sequence, take last frame H
        context_latents = self.context_encoder(
            coords=batch["context_coords"],
            residue_mask=batch["context_residue_mask"],
            aatype=batch["context_aatype"],
            residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
            token_mask=batch["context_residue_mask"],
        )
        # context_latents: [B, T, L, D]; take last frame
        h_last = context_latents[:, -1]  # [B, L, D]

        # iter_010: temporal-delta motion feature. Gives the head (and thus the
        # encoder) a per-residue motion signal; a static code cannot drive it.
        if getattr(self, "disp_use_temporal_delta", False):
            n_t = context_latents.shape[1]
            k = max(1, min(int(getattr(self, "disp_delta_k", 4)), n_t - 1))
            h_prev = context_latents[:, -1 - k]  # [B, L, D]
            h_motion = torch.cat([h_last, h_last - h_prev], dim=-1)  # [B, L, 2D]
        else:
            h_motion = h_last

        # ── Compute displacement head prediction ──
        if not hasattr(self, "displacement_head"):
            raise AttributeError(
                "displacement_head not found; rebuild model with displacement head enabled"
            )
        disp_pred = self.displacement_head(h_motion)  # [B, L, 3]

        # Target stays the Kabsch-aligned local displacement (below); weighting
        # for the learnable regime is handled in the loss, not by band-limiting here.

        # ── Compute Kabsch-aligned ground-truth displacement target ──
        # The raw MD trajectory contains global rigid-body translation/rotation
        # of the whole protein inside the simulation box.  The eval protocol
        # Kabsch-superposes the prediction onto the target before scoring, which
        # removes exactly that rigid component.  We therefore must superpose the
        # future frame onto the anchor frame FIRST, and only then measure the
        # per-residue displacement.  Anything else supervises ~31 A of global
        # drift (measured) instead of the internal conformational change.
        anchor_bb = batch["context_coords"][:, -1]   # [B, L, 4, 3]
        future_bb = batch["future_coords"][:, -1]    # [B, L, 4, 3]

        ctx_mask = batch["context_residue_mask"][:, -1].bool()   # [B, L]
        fut_mask = batch["future_residue_mask"][:, -1].bool()    # [B, L]
        valid = ctx_mask & fut_mask                               # [B, L]

        anchor_ca = anchor_bb[..., 1, :]   # [B, L, 3]
        future_ca = future_bb[..., 1, :]   # [B, L, 3]

        # ── Per-sample Kabsch superposition of future CA onto anchor CA ──
        # (weighted by the shared valid mask; removes global rot + trans)
        B, L = valid.shape
        m = valid.to(anchor_ca.dtype).unsqueeze(-1)                # [B,L,1]
        n_valid = m.sum(dim=1, keepdim=True).clamp(min=1.0)        # [B,1,1]
        anchor_mean = (anchor_ca * m).sum(dim=1, keepdim=True) / n_valid  # [B,1,3]
        future_mean = (future_ca * m).sum(dim=1, keepdim=True) / n_valid  # [B,1,3]
        anchor_c = (anchor_ca - anchor_mean) * m                   # [B,L,3]
        future_c = (future_ca - future_mean) * m                   # [B,L,3]

        # H = sum_i future_c_i^T anchor_c_i  -> [B,3,3]
        H = future_c.transpose(-2, -1) @ anchor_c                  # [B,3,3]
        U, S, Vt = torch.linalg.svd(H)                             # batched SVD
        # Correct for reflection so R is a proper rotation
        det = torch.det(U @ Vt)                                    # [B]
        d_fix = torch.ones(B, 3, device=U.device, dtype=U.dtype)
        d_fix[:, -1] = torch.sign(det)
        # Kabsch rotation such that:  future_c @ R_fix^T  ==  anchor_c
        # (verified convention: R = V diag(d) U^T with H = future_c^T @ anchor_c)
        R_fix = Vt.transpose(-2, -1) @ torch.diag_embed(d_fix) @ U.transpose(-2, -1)  # [B,3,3]
        future_ca_aligned = future_c @ R_fix.transpose(-2, -1) + anchor_mean  # [B,L,3]

        # Internal (rigid-removed) global displacement
        disp_g = future_ca_aligned - anchor_ca                     # [B,L,3]

        # Per-residue anchor backbone frame (from the anchor structure)
        from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
        N_a  = anchor_bb[..., 0, :]
        C_a  = anchor_bb[..., 2, :]
        Rf, tf = RigidBodyFrame.from_backbone(N_a, anchor_ca, C_a)  # Rf [B,L,3,3]

        # Rotate into per-residue anchor frame (fully SE(3)-invariant target,
        # after the global Kabsch rigid-body removal above)
        disp_local = (Rf.transpose(-2, -1) @ disp_g.unsqueeze(-1)).squeeze(-1)  # [B,L,3]

        return {
            "disp_pred":   disp_pred,
            "disp_target": disp_local.detach(),
            "valid_mask":  valid,
        }

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
        self,
        temporal_states: dict[str, torch.Tensor],
        dynamic_states: dict[str, torch.Tensor],
        residue_latents: torch.Tensor | None = None,
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
            fused = {
                "frame_states": dynamic_states["dynamic_frame_states"],
                "clip_state": dynamic_states["dynamic_clip_state"],
            }
        elif fusion == "add":
            # Simple addition: static + dynamic
            fused = {
                "frame_states": temporal_states["frame_states"] + dynamic_states["dynamic_frame_states"],
                "clip_state": temporal_states["clip_state"] + dynamic_states["dynamic_clip_state"],
            }
        elif fusion == "gate":
            # Gated fusion: static + gate * dynamic
            gate = torch.sigmoid(self.dynamic_gate)
            fused = {
                "frame_states": temporal_states["frame_states"] + gate * dynamic_states["dynamic_frame_states"],
                "clip_state": temporal_states["clip_state"] + gate * dynamic_states["dynamic_clip_state"],
            }
        else:
            raise ValueError(f"Unknown dynamic_fusion mode: {fusion}")

        # Preserve the per-residue representation for Stage-B consumers and
        # diagnostics.  It is formed only after the dynamic branch has kept
        # residue identity, so localized motion is not irreversibly averaged.
        if "dynamic_residue_states" in dynamic_states:
            residue_dynamic = dynamic_states["dynamic_residue_states"]
            if fusion == "replace":
                fused["residue_states"] = residue_dynamic
            elif residue_latents is not None:
                fused["residue_states"] = residue_latents + residue_dynamic
            else:
                fused["residue_states"] = residue_dynamic
            fused["dynamic_residue_states"] = residue_dynamic
        return fused

    @staticmethod
    def _mean_over_valid(
        states: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean over time dimension using valid frames only."""
        # states: [B, T, D], valid_mask: [B, T]
        mask = valid_mask.unsqueeze(-1).to(states.dtype)
        return (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
