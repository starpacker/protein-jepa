from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.time_features import (
    LogTimeEmbedding,
    ThermodynamicConditionEmbedding,
    lag_delta_time,
    relative_clip_time,
)


class SpatiotemporalPredictor(nn.Module):
    """Transformer predictor over visible and target `[time, residue]` tokens."""

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
        max_residues: int = 4096,
        time_conditioning: str = "none",
    ):
        super().__init__()
        self.time_conditioning = time_conditioning
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, latent_dim))
        self.time_embed = nn.Embedding(max_frames, latent_dim)
        self.residue_embed = nn.Embedding(max_residues, latent_dim)
        self.chain_embed = nn.Embedding(128, latent_dim)
        if time_conditioning == "none":
            self.physical_time_embed = None
        elif time_conditioning == "physical":
            self.physical_time_embed = LogTimeEmbedding(latent_dim)
        else:
            raise ValueError(f"unknown time_conditioning {time_conditioning}")
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        time_ps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, L, D = context_latents.shape
        time_pos = torch.arange(T, device=context_latents.device).clamp(max=self.time_embed.num_embeddings - 1)
        residue_pos = residue_index.clamp(min=0, max=self.residue_embed.num_embeddings - 1)
        chain_pos = chain_id.clamp(min=0, max=self.chain_embed.num_embeddings - 1)
        pos = (
            self.time_embed(time_pos).view(1, T, 1, D)
            + self.residue_embed(residue_pos).view(B, 1, L, D)
            + self.chain_embed(chain_pos).view(B, 1, L, D)
        )
        relative_time = relative_clip_time(time_ps)
        if self.physical_time_embed is not None and relative_time is not None:
            pos = pos + self.physical_time_embed(relative_time).view(B, T, 1, D)
        x = context_latents * context_mask.unsqueeze(-1).float()
        x = x + self.mask_token * target_mask.unsqueeze(-1).float()
        x = x + pos
        token_valid = context_mask | target_mask
        flat = x.reshape(B, T * L, D)
        padding = ~token_valid.reshape(B, T * L)
        flat = self.blocks(flat, src_key_padding_mask=padding)
        flat = self.proj(self.norm(flat))
        return flat.reshape(B, T, L, D)


class StatePredictor(nn.Module):
    """Temporal JEPA predictor over frame-level state tokens."""

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
        time_conditioning: str = "none",
    ):
        super().__init__()
        self.time_conditioning = time_conditioning
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.time_embed = nn.Embedding(max_frames, latent_dim)
        if time_conditioning == "none":
            self.physical_time_embed = None
        elif time_conditioning == "physical":
            self.physical_time_embed = LogTimeEmbedding(latent_dim)
        else:
            raise ValueError(f"unknown time_conditioning {time_conditioning}")
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)
        self.proj = nn.Linear(latent_dim, latent_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        context_states: torch.Tensor,
        context_frame_mask: torch.Tensor,
        target_frame_mask: torch.Tensor,
        time_ps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_states.dim() != 3:
            raise ValueError("context_states must be [B,T,D]")
        B, T, D = context_states.shape
        time_pos = torch.arange(T, device=context_states.device).clamp(max=self.time_embed.num_embeddings - 1)
        x = context_states * context_frame_mask.unsqueeze(-1).float()
        x = x + self.mask_token * target_frame_mask.unsqueeze(-1).float()
        x = x + self.time_embed(time_pos).view(1, T, D)
        relative_time = relative_clip_time(time_ps)
        if self.physical_time_embed is not None and relative_time is not None:
            x = x + self.physical_time_embed(relative_time)
        token_valid = context_frame_mask | target_frame_mask
        x = self.blocks(x, src_key_padding_mask=~token_valid)
        return self.proj(self.norm(x))


class LaggedStatePredictor(nn.Module):
    """Predict EMA target states at several future temporal offsets."""

    def __init__(
        self,
        latent_dim: int = 128,
        lags: tuple[int, ...] = (1, 2, 4, 8),
        depth: int = 2,
        dropout: float = 0.1,
        conditioning: str = "none",
        prediction_mode: str = "direct",
        time_conditioning: str = "none",
        thermodynamic_conditioning: bool = False,
    ):
        super().__init__()
        if any(lag <= 0 for lag in lags):
            raise ValueError("state prediction lags must be positive")
        self.lags = tuple(int(lag) for lag in lags)
        self.conditioning = conditioning
        if prediction_mode not in {"direct", "residual"}:
            raise ValueError(f"unknown lag prediction mode {prediction_mode}")
        self.prediction_mode = prediction_mode
        self.time_conditioning = time_conditioning
        self.thermodynamic_conditioning = thermodynamic_conditioning
        if conditioning == "none":
            self.lag_embed = None
        elif conditioning == "embedding":
            self.lag_embed = nn.Embedding(max(self.lags) + 1, latent_dim)
        else:
            raise ValueError(f"unknown lag conditioning {conditioning}")
        if time_conditioning == "none":
            self.physical_lag_embed = None
        elif time_conditioning == "physical":
            self.physical_lag_embed = LogTimeEmbedding(latent_dim)
        else:
            raise ValueError(f"unknown time_conditioning {time_conditioning}")
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None
        hidden = latent_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({str(lag): nn.Linear(hidden, latent_dim) for lag in self.lags})
        if self.prediction_mode == "residual":
            for head in self.heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(
        self,
        frame_states: torch.Tensor,
        time_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        if frame_states.dim() != 3:
            raise ValueError("frame_states must be [B,T,D]")
        out = {}
        B, T, _ = frame_states.shape
        condition = None
        if self.condition_embed is not None:
            condition = self.condition_embed(
                temperature_k, dataset_index, B, frame_states.device, frame_states.dtype
            ).unsqueeze(1)
        for lag in self.lags:
            conditioned = frame_states
            if condition is not None:
                conditioned = conditioned + condition
            if self.lag_embed is not None:
                lag_id = torch.tensor(lag, device=frame_states.device, dtype=torch.long)
                conditioned = conditioned + self.lag_embed(lag_id).view(1, 1, -1)
            if self.physical_lag_embed is not None:
                delta_time = lag_delta_time(time_ps, lag, T, frame_states.device)
                if delta_time.shape[0] == 1 and B > 1:
                    delta_time = delta_time.expand(B, -1)
                conditioned = conditioned + self.physical_lag_embed(delta_time)
            update = self.heads[str(lag)](self.trunk(conditioned))
            out[lag] = frame_states + update if self.prediction_mode == "residual" else update
        return out


class PairedDeltaPredictor(nn.Module):
    """Shared physical-time-conditioned residual transition for paired windows."""

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        thermodynamic_conditioning: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.time_embed = LogTimeEmbedding(latent_dim)
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None
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
        self.delta = nn.Linear(hidden, latent_dim)
        if zero_init:
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        endpoint_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if endpoint_state.dim() != 2:
            raise ValueError("endpoint_state must be [B,D]")
        if horizon_time_ps.dim() != 1 or horizon_time_ps.shape[0] != endpoint_state.shape[0]:
            raise ValueError("horizon_time_ps must be [B]")
        B = endpoint_state.shape[0]
        condition = self.time_embed(horizon_time_ps).to(dtype=endpoint_state.dtype)
        if self.condition_embed is not None:
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, B, endpoint_state.device, endpoint_state.dtype
            )
        return self.delta(self.trunk(endpoint_state + condition))


class HistoryConditionedTransition(nn.Module):
    """Residual transition conditioned on multiscale state velocities.

    The history summary uses only the observed context window. Persistence is
    preserved at initialization by the zero-initialized output layer.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        thermodynamic_conditioning: bool = True,
        strides: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        if not strides or any(stride <= 0 for stride in strides):
            raise ValueError("history transition strides must be positive")
        self.strides = tuple(int(stride) for stride in strides)
        self.time_embed = LogTimeEmbedding(latent_dim)
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None
        summary_dim = latent_dim * (2 + 2 * len(self.strides))
        self.summary = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
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
        self.delta = nn.Linear(hidden, latent_dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def summarize(self, frame_states: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if frame_states.dim() != 3 or frame_mask.shape != frame_states.shape[:2]:
            raise ValueError("history states/mask must be [B,T,D] and [B,T]")
        B, T, D = frame_states.shape
        index = torch.arange(T, device=frame_states.device).view(1, T).expand(B, -1)
        last_index = torch.where(frame_mask, index, -1).max(dim=1).values.clamp(min=0)
        batch_index = torch.arange(B, device=frame_states.device)
        endpoint = frame_states[batch_index, last_index]
        valid = frame_mask.unsqueeze(-1).to(dtype=frame_states.dtype)
        mean = (frame_states * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        features = [endpoint, mean]
        for stride in self.strides:
            if stride >= T:
                features.extend([torch.zeros_like(endpoint), torch.zeros_like(endpoint)])
                continue
            pair_mask = frame_mask[:, stride:] & frame_mask[:, :-stride]
            delta = frame_states[:, stride:] - frame_states[:, :-stride]
            weights = pair_mask.unsqueeze(-1).to(dtype=delta.dtype)
            count = weights.sum(dim=1).clamp(min=1.0)
            delta_mean = (delta * weights).sum(dim=1) / count
            centered = (delta - delta_mean[:, None]) * weights
            delta_std = torch.sqrt(centered.square().sum(dim=1) / count + 1.0e-5)
            has_pairs = pair_mask.any(dim=1, keepdim=True).to(dtype=delta.dtype)
            features.extend([delta_mean * has_pairs, delta_std * has_pairs])
        return self.summary(torch.cat(features, dim=-1))

    def forward(
        self,
        frame_states: torch.Tensor,
        frame_mask: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.summarize(frame_states, frame_mask)
        B = history.shape[0]
        condition = self.time_embed(horizon_time_ps).to(dtype=history.dtype)
        if self.condition_embed is not None:
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, B, history.device, history.dtype
            )
        return self.delta(self.trunk(history + condition)), history


class MetastableStateHead(nn.Module):
    """VAMPNet-style soft state assignment over frozen frame states."""

    def __init__(self, latent_dim: int = 128, output_dim: int = 8, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        if output_dim < 2:
            raise ValueError("metastable_state_dim must be at least 2")
        hidden = latent_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, frame_states: torch.Tensor) -> torch.Tensor:
        if frame_states.dim() != 3:
            raise ValueError("metastable state head expects [B,T,D]")
        return torch.softmax(self.net(frame_states), dim=-1)


class LocalDynamicsHead(nn.Module):
    """Sequence-segment tokens supervised by local physical transitions."""

    def __init__(
        self,
        latent_dim: int = 128,
        token_dim: int = 32,
        num_tokens: int = 4,
        output_dim: int = 12,
        depth: int = 2,
        dropout: float = 0.1,
        thermodynamic_conditioning: bool = True,
    ):
        super().__init__()
        if num_tokens <= 0 or token_dim <= 0:
            raise ValueError("local dynamics token count and dimension must be positive")
        self.num_tokens = int(num_tokens)
        self.output_dim = int(output_dim)
        self.token_proj = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, token_dim), nn.GELU())
        self.history_proj = nn.Linear(latent_dim, token_dim)
        self.condition_proj = nn.Linear(latent_dim, token_dim)
        self.segment_embed = nn.Embedding(self.num_tokens, token_dim)
        self.time_embed = LogTimeEmbedding(latent_dim)
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None
        hidden = token_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, self.output_dim))
        self.predictor = nn.Sequential(*layers)
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        persistence = torch.zeros(self.output_dim)
        persistence[0] = 1.0
        self.register_buffer("physical_persistence", persistence)

    def _segment_assignment(self, residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residue_mask.dim() != 2:
            raise ValueError("local dynamics residue mask must be [B,L]")
        B, L = residue_mask.shape
        position = torch.arange(L, device=residue_mask.device).view(1, L).expand(B, -1)
        length = residue_mask.sum(dim=1, keepdim=True).clamp(min=1)
        segment = torch.div(position * self.num_tokens, length, rounding_mode="floor").clamp(max=self.num_tokens - 1)
        assignment = torch.nn.functional.one_hot(segment, num_classes=self.num_tokens).permute(0, 2, 1).bool()
        assignment = assignment & residue_mask[:, None]
        segment_mask = assignment.sum(dim=-1) >= 3
        return assignment, segment_mask

    def encode_tokens(self, residue_latents: torch.Tensor, residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residue_latents.dim() != 3 or residue_latents.shape[:2] != residue_mask.shape:
            raise ValueError("local dynamics latents/mask must be [B,L,D] and [B,L]")
        assignment, segment_mask = self._segment_assignment(residue_mask)
        weights = assignment.to(dtype=residue_latents.dtype)
        pooled = torch.einsum("bkl,bld->bkd", weights, residue_latents)
        pooled = pooled / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        tokens = self.token_proj(pooled)
        tokens = tokens * segment_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, segment_mask

    def forward(
        self,
        residue_latents: torch.Tensor,
        residue_mask: torch.Tensor,
        history_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, segment_mask = self.encode_tokens(residue_latents, residue_mask)
        B = tokens.shape[0]
        condition = self.time_embed(horizon_time_ps).to(dtype=tokens.dtype)
        if self.condition_embed is not None:
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, B, tokens.device, tokens.dtype
            )
        segment_ids = torch.arange(self.num_tokens, device=tokens.device)
        hidden = (
            tokens
            + self.history_proj(history_state).unsqueeze(1)
            + self.condition_proj(condition).unsqueeze(1)
            + self.segment_embed(segment_ids).unsqueeze(0)
        )
        prediction = self.predictor(hidden) + self.physical_persistence.view(1, 1, -1)
        return tokens, prediction, segment_mask


class ClosedLocalStateOperator(nn.Module):
    """Horizon-conditioned transition over a composable local state."""

    def __init__(
        self,
        latent_dim: int = 128,
        token_dim: int = 32,
        state_dim: int = 12,
        num_tokens: int = 4,
        horizons: tuple[int, ...] = (48, 96, 192),
        state_scale: tuple[float, ...] = (1.0,) * 12,
        state_mean: tuple[float, ...] | None = None,
        operator_mode: str = "experts",
        relaxation_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        velocity_strides: tuple[int, ...] = (1, 2, 4, 8),
        log_sigma_min: float = -4.0,
        log_sigma_max: float = 2.0,
        base_step_frames: int = 16,
        cross_segment_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("closed local horizons must be positive")
        self.horizons = tuple(sorted({int(horizon) for horizon in horizons}))
        self.state_dim = int(state_dim)
        if operator_mode not in {
            "experts", "continuous_relaxation", "velocity_probabilistic", "compositional_rollout"
        }:
            raise ValueError(f"unknown closed local operator mode {operator_mode}")
        self.operator_mode = operator_mode
        if len(state_scale) != self.state_dim or any(value <= 0 for value in state_scale):
            raise ValueError("closed local state scale must contain one positive value per channel")
        self.register_buffer("state_scale", torch.tensor(state_scale, dtype=torch.float32))
        if state_mean is None:
            state_mean = (0.0,) * self.state_dim
        if len(state_mean) != self.state_dim:
            raise ValueError("closed local state mean must contain one value per channel")
        self.register_buffer("state_mean", torch.tensor(state_mean, dtype=torch.float32))
        if base_step_frames <= 0:
            raise ValueError("closed local base step must be positive")
        self.base_step_frames = int(base_step_frames)
        if not velocity_strides or any(stride <= 0 for stride in velocity_strides):
            raise ValueError("closed local velocity strides must be positive")
        self.velocity_strides = tuple(int(stride) for stride in velocity_strides)
        if log_sigma_min >= log_sigma_max:
            raise ValueError("closed local log sigma bounds are invalid")
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
        self.state_proj = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, token_dim), nn.GELU())
        self.history_proj = nn.Linear(latent_dim, token_dim)
        self.segment_embed = nn.Embedding(num_tokens, token_dim)
        self.velocity_proj = None
        if self.operator_mode == "velocity_probabilistic":
            velocity_dim = state_dim * len(self.velocity_strides)
            self.velocity_proj = nn.Sequential(
                nn.LayerNorm(velocity_dim), nn.Linear(velocity_dim, token_dim), nn.GELU()
            )
        hidden = token_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*layers)
        self.experts = nn.ModuleDict()
        self.time_embed = None
        self.film = None
        self.relaxation_head = None
        self.uncertainty_head = None
        self.history_gru = None
        self.memory_update = None
        self.memory_proj = None
        self.condition_proj = None
        self.segment_mixer = None
        self.equilibrium_head = None
        self.rate_head = None
        self.drift_head = None
        if self.operator_mode == "experts":
            self.time_embed = LogTimeEmbedding(latent_dim)
            self.film = nn.Linear(latent_dim, token_dim * 2)
            self.experts = nn.ModuleDict({str(horizon): nn.Linear(hidden, state_dim) for horizon in self.horizons})
            for expert in self.experts.values():
                nn.init.zeros_(expert.weight)
                nn.init.zeros_(expert.bias)
        elif self.operator_mode in {"continuous_relaxation", "velocity_probabilistic"}:
            if not relaxation_scales or any(value <= 0 for value in relaxation_scales):
                raise ValueError("closed local relaxation scales must be positive")
            self.register_buffer(
                "relaxation_scales", torch.tensor(relaxation_scales, dtype=torch.float32)
            )
            num_modes = 1 + len(relaxation_scales)
            self.relaxation_head = nn.Linear(hidden, state_dim * num_modes)
            nn.init.zeros_(self.relaxation_head.weight)
            nn.init.zeros_(self.relaxation_head.bias)
            if self.operator_mode == "velocity_probabilistic":
                self.uncertainty_head = nn.Linear(hidden, state_dim * (num_modes + 1))
                nn.init.zeros_(self.uncertainty_head.weight)
                nn.init.zeros_(self.uncertainty_head.bias)
        else:
            if token_dim % cross_segment_heads != 0:
                raise ValueError("closed local token dimension must be divisible by cross-segment heads")
            memory_dim = token_dim * 2
            self.history_gru = nn.GRU(
                input_size=num_tokens * state_dim,
                hidden_size=memory_dim,
                batch_first=True,
            )
            self.memory_update = nn.GRUCell(num_tokens * state_dim, memory_dim)
            self.memory_proj = nn.Linear(memory_dim, token_dim)
            self.time_embed = LogTimeEmbedding(latent_dim)
            self.condition_embed = ThermodynamicConditionEmbedding(latent_dim)
            self.condition_proj = nn.Linear(latent_dim, token_dim)
            mixer_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=cross_segment_heads,
                dim_feedforward=token_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.segment_mixer = nn.TransformerEncoder(mixer_layer, num_layers=1)
            self.equilibrium_head = nn.Linear(token_dim, state_dim)
            self.rate_head = nn.Linear(token_dim, state_dim)
            self.drift_head = nn.Linear(token_dim, state_dim)
            nn.init.zeros_(self.equilibrium_head.weight)
            nn.init.zeros_(self.equilibrium_head.bias)
            nn.init.zeros_(self.rate_head.weight)
            nn.init.constant_(self.rate_head.bias, -1.5)
            nn.init.zeros_(self.drift_head.weight)
            nn.init.zeros_(self.drift_head.bias)

    def forward(
        self,
        local_tokens: torch.Tensor,
        current_state: torch.Tensor,
        history_state: torch.Tensor,
        horizon_frames: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        current_velocity: torch.Tensor | None = None,
        context_state_sequence: torch.Tensor | None = None,
        context_sequence_mask: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if local_tokens.dim() != 3 or current_state.shape[:2] != local_tokens.shape[:2]:
            raise ValueError("closed local tokens and state must be [B,K,*]")
        segment_ids = torch.arange(local_tokens.shape[1], device=local_tokens.device)
        if self.operator_mode == "compositional_rollout":
            if context_state_sequence is None:
                raise ValueError("compositional rollout requires a context state sequence")
            if context_state_sequence.dim() != 4 or context_state_sequence.shape[0] != current_state.shape[0]:
                raise ValueError("context state sequence must be [B,T,K,D]")
            if context_state_sequence.shape[2:] != current_state.shape[1:]:
                raise ValueError("context state sequence does not match current local state")
            steps = torch.div(horizon_frames, self.base_step_frames, rounding_mode="floor")
            if ((steps * self.base_step_frames) != horizon_frames).any() or (steps <= 0).any():
                raise ValueError("compositional horizons must be positive multiples of the base step")
            mean = self.state_mean.to(dtype=current_state.dtype).view(1, 1, 1, -1)
            scale = self.state_scale.to(dtype=current_state.dtype).view(1, 1, 1, -1)
            sequence = (context_state_sequence - mean) / scale
            if context_sequence_mask is not None:
                if context_sequence_mask.shape != context_state_sequence.shape[:3]:
                    raise ValueError("context sequence mask must be [B,T,K]")
                sequence = sequence * context_sequence_mask.unsqueeze(-1).to(dtype=sequence.dtype)
            _, memory_hidden = self.history_gru(sequence.flatten(start_dim=2))
            memory = memory_hidden[-1]
            anchor = (current_state - mean[:, 0]) / scale[:, 0]
            state = anchor
            segment_ids = torch.arange(local_tokens.shape[1], device=local_tokens.device)
            fixed = (
                local_tokens
                + self.history_proj(history_state).unsqueeze(1)
                + self.segment_embed(segment_ids).unsqueeze(0)
            )
            step_time = horizon_time_ps / steps.to(dtype=horizon_time_ps.dtype)
            condition = self.time_embed(step_time).to(dtype=state.dtype)
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, state.shape[0], state.device, state.dtype
            )
            fixed = fixed + self.condition_proj(condition).unsqueeze(1)
            prediction = state
            representation = fixed
            for step_index in range(1, int(steps.max().item()) + 1):
                hidden = fixed + self.state_proj(state) + self.memory_proj(memory).unsqueeze(1)
                representation = self.segment_mixer(hidden)
                equilibrium = anchor + self.equilibrium_head(representation)
                decay = torch.exp(-torch.nn.functional.softplus(self.rate_head(representation)))
                drift = self.drift_head(representation)
                state = equilibrium + decay * (state - equilibrium) + drift
                memory = self.memory_update(state.flatten(start_dim=1), memory)
                selected = (steps == step_index).view(-1, 1, 1)
                prediction = torch.where(selected, state, prediction)
            physical = prediction * scale[:, 0] + mean[:, 0]
            return physical, representation

        state_input = current_state
        if self.operator_mode == "velocity_probabilistic":
            if current_velocity is None:
                raise ValueError("velocity-probabilistic operator requires current_velocity")
            expected = (*current_state.shape[:2], len(self.velocity_strides), self.state_dim)
            if current_velocity.shape != expected:
                raise ValueError(f"closed local velocity must have shape {expected}")
            state_input = current_state / self.state_scale.to(dtype=current_state.dtype)
        hidden = (
            local_tokens
            + self.state_proj(state_input)
            + self.history_proj(history_state).unsqueeze(1)
            + self.segment_embed(segment_ids).unsqueeze(0)
        )
        if self.velocity_proj is not None:
            hidden = hidden + self.velocity_proj(current_velocity.flatten(start_dim=2))
        if self.operator_mode == "experts":
            condition = self.time_embed(horizon_time_ps).to(dtype=local_tokens.dtype)
            gamma, beta = self.film(condition).chunk(2, dim=-1)
            hidden = hidden * (1.0 + 0.5 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        representation = self.trunk(hidden)
        configured = torch.zeros(horizon_frames.shape[0], dtype=torch.bool, device=horizon_frames.device)
        for horizon in self.horizons:
            configured |= horizon_frames == horizon
        if self.operator_mode != "compositional_rollout" and not configured.all():
            unknown = sorted(set(int(value) for value in horizon_frames[~configured].detach().cpu().tolist()))
            raise ValueError(f"closed local operator received unconfigured horizons {unknown}")
        if self.operator_mode == "experts":
            delta = torch.zeros_like(current_state)
            for horizon in self.horizons:
                mask = horizon_frames == horizon
                if mask.any():
                    delta[mask] = self.experts[str(horizon)](representation[mask])
        else:
            tau = horizon_frames.to(dtype=representation.dtype) / float(max(self.horizons))
            scales = self.relaxation_scales.to(dtype=representation.dtype)
            relaxation_basis = 1.0 - torch.exp(-tau[:, None] / scales[None, :])
            basis = torch.cat([tau[:, None], relaxation_basis], dim=-1)
            coefficients = self.relaxation_head(representation).view(
                *representation.shape[:2], basis.shape[-1], self.state_dim
            )
            delta = torch.einsum("bkmd,bm->bkd", coefficients, basis)
            if self.operator_mode == "velocity_probabilistic":
                delta = delta * self.state_scale.to(dtype=delta.dtype)
                uncertainty = self.uncertainty_head(representation).view(
                    *representation.shape[:2], basis.shape[-1] + 1, self.state_dim
                )
                log_sigma = uncertainty[:, :, 0] + torch.einsum(
                    "bkmd,bm->bkd", uncertainty[:, :, 1:], basis
                )
                log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
                return current_state + delta, representation, log_sigma
        return current_state + delta, representation


class V4SlowKineticModel(nn.Module):
    """Trajectory-centered slow states with a composable stochastic matrix."""

    def __init__(
        self,
        num_tokens: int = 4,
        state_dim: int = 12,
        latent_dim: int = 16,
        num_states: int = 8,
        base_step_frames: int = 16,
        history_frame_stride: int = 2,
        state_scale: tuple[float, ...] = (1.0,) * 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_dim <= 0 or num_states < 2:
            raise ValueError("V4 slow latent and state dimensions must be positive")
        if base_step_frames <= 0 or history_frame_stride <= 0:
            raise ValueError("V4 slow temporal strides must be positive")
        if base_step_frames % history_frame_stride != 0:
            raise ValueError("V4 slow base step must be divisible by history frame stride")
        if len(state_scale) != state_dim or any(value <= 0 for value in state_scale):
            raise ValueError("V4 slow state scale must contain one positive value per channel")
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.num_states = int(num_states)
        self.base_step_frames = int(base_step_frames)
        self.history_frame_stride = int(history_frame_stride)
        self.register_buffer("state_scale", torch.tensor(state_scale, dtype=torch.float32))
        input_dim = self.num_tokens * self.state_dim
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, input_dim),
        )
        self.assignment_head = nn.Linear(latent_dim, num_states)
        self.memory_gru = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim)
        self.transition_head = nn.Linear(latent_dim, num_states * num_states)
        nn.init.zeros_(self.transition_head.weight)
        nn.init.zeros_(self.transition_head.bias)
        prior = torch.full((num_states, num_states), -2.0)
        prior.fill_diagonal_(2.0)
        self.register_buffer("transition_identity_prior", prior)

    def _normalize(
        self, sequence: torch.Tensor, equilibrium: torch.Tensor
    ) -> torch.Tensor:
        scale = self.state_scale.to(dtype=sequence.dtype).view(1, 1, 1, -1)
        return ((sequence - equilibrium) / scale).flatten(start_dim=2)

    def _encode(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.frame_encoder(normalized)
        assignment = torch.softmax(self.assignment_head(latent), dim=-1)
        reconstruction = self.decoder(latent)
        return latent, assignment, reconstruction

    @staticmethod
    def _rollout(
        assignment: torch.Tensor, transition: torch.Tensor, steps: torch.Tensor
    ) -> torch.Tensor:
        prediction = assignment
        state = assignment
        for step_index in range(1, int(steps.max().item()) + 1):
            state = torch.bmm(state.unsqueeze(1), transition).squeeze(1)
            prediction = torch.where((steps == step_index).unsqueeze(1), state, prediction)
        return prediction

    def forward(
        self,
        context_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        context_mask: torch.Tensor,
        future_mask: torch.Tensor,
        horizon_frames: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context_sequence.dim() != 4 or future_sequence.shape != context_sequence.shape:
            raise ValueError("V4 slow context and future sequences must be matching [B,T,K,D]")
        valid = context_mask.unsqueeze(-1).to(dtype=context_sequence.dtype)
        equilibrium = (context_sequence * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        context_normalized = self._normalize(context_sequence, equilibrium)
        future_normalized = self._normalize(future_sequence, equilibrium)
        context_latent, context_assignment, context_reconstruction = self._encode(context_normalized)
        future_latent, future_assignment, future_reconstruction = self._encode(future_normalized)
        _, memory_hidden = self.memory_gru(context_latent)
        condition = memory_hidden[-1] + self.condition_embed(
            temperature_k,
            dataset_index,
            context_sequence.shape[0],
            context_sequence.device,
            context_sequence.dtype,
        )
        transition_logits = self.transition_head(condition).view(
            -1, self.num_states, self.num_states
        ) + self.transition_identity_prior.to(dtype=condition.dtype)
        transition = torch.softmax(transition_logits, dim=-1)
        steps = torch.div(horizon_frames, self.base_step_frames, rounding_mode="floor")
        if ((steps * self.base_step_frames) != horizon_frames).any() or (steps <= 0).any():
            raise ValueError("V4 slow horizons must be positive multiples of the base step")
        endpoint = context_assignment[:, -1]
        rollout_prediction = self._rollout(endpoint, transition, steps)
        lag = self.base_step_frames // self.history_frame_stride
        if lag >= context_sequence.shape[1]:
            raise ValueError("V4 slow context is too short for its base step")
        base_prediction = torch.bmm(
            context_assignment[:, -1 - lag].unsqueeze(1), transition
        ).squeeze(1)
        result = {
            "v4_slow_context_latent": context_latent,
            "v4_slow_future_latent": future_latent,
            "v4_slow_context_assignment": context_assignment,
            "v4_slow_future_assignment": future_assignment,
            "v4_slow_context_reconstruction": context_reconstruction,
            "v4_slow_future_reconstruction": future_reconstruction,
            "v4_slow_context_normalized": context_normalized,
            "v4_slow_future_normalized": future_normalized,
            "v4_slow_context_mask": context_mask.all(dim=-1),
            "v4_slow_future_mask": future_mask.all(dim=-1),
            "v4_slow_transition_matrix": transition,
            "v4_slow_current_assignment": endpoint,
            "v4_slow_target_assignment": future_assignment[:, -1],
            "v4_slow_prediction": rollout_prediction,
            "v4_slow_base_current": context_assignment[:, -1 - lag],
            "v4_slow_base_target": context_assignment[:, -1],
            "v4_slow_base_prediction": base_prediction,
        }
        if not self.training:
            fixed = []
            for horizon in (48, 64, 96, 128, 192):
                fixed_steps = torch.full_like(steps, horizon // self.base_step_frames)
                fixed.append(self._rollout(endpoint, transition, fixed_steps))
            result["v4_slow_fixed_predictions"] = torch.stack(fixed, dim=1)
            result["v4_slow_fixed_horizons"] = torch.tensor(
                (48, 64, 96, 128, 192), device=steps.device
            )
        return result


class V4PhysicsSlowModel(nn.Module):
    """Residual slow representation with free energy and reversible kinetics."""

    def __init__(
        self,
        offline_dim: int,
        num_tokens: int = 4,
        state_dim: int = 12,
        latent_dim: int = 16,
        num_states: int = 8,
        base_step_frames: int = 16,
        history_frame_stride: int = 2,
        state_scale: tuple[float, ...] = (1.0,) * 12,
        lowpass_width: int = 5,
        dynamic_only: bool = False,
        condition_on_equilibrium: bool = False,
        training_fixed_controls: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_dim <= 0 or num_states < 2 or offline_dim <= 0:
            raise ValueError("V4-C dimensions must be positive")
        if lowpass_width <= 0 or lowpass_width % 2 == 0:
            raise ValueError("V4-C lowpass width must be a positive odd number")
        if base_step_frames <= 0 or history_frame_stride <= 0:
            raise ValueError("V4-C temporal strides must be positive")
        if base_step_frames % history_frame_stride != 0:
            raise ValueError("V4-C base step must be divisible by history stride")
        if len(state_scale) != state_dim or any(value <= 0 for value in state_scale):
            raise ValueError("V4-C state scale must contain one positive value per channel")
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.num_states = int(num_states)
        self.base_step_frames = int(base_step_frames)
        self.history_frame_stride = int(history_frame_stride)
        self.lowpass_width = int(lowpass_width)
        self.dynamic_only = bool(dynamic_only)
        self.condition_on_equilibrium = bool(condition_on_equilibrium)
        self.training_fixed_controls = bool(training_fixed_controls)
        self.register_buffer("state_scale", torch.tensor(state_scale, dtype=torch.float32))
        input_dim = self.num_tokens * self.state_dim
        encoder_dim = input_dim + offline_dim
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, latent_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.slow_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, input_dim),
        )
        self.assignment_head = nn.Linear(latent_dim, num_states)
        self.free_energy_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, 1),
        )
        self.mobility_head = nn.Linear(latent_dim, latent_dim)
        self.condition_trunk = nn.Sequential(
            nn.LayerNorm(offline_dim),
            nn.Linear(offline_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.equilibrium_condition_trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim)
        self.state_free_energy_head = nn.Linear(latent_dim, num_states)
        pair_count = num_states * (num_states - 1) // 2
        self.symmetric_flux_head = nn.Linear(latent_dim, pair_count)
        nn.init.zeros_(self.state_free_energy_head.weight)
        nn.init.zeros_(self.state_free_energy_head.bias)
        nn.init.zeros_(self.symmetric_flux_head.weight)
        nn.init.constant_(self.symmetric_flux_head.bias, -4.0)
        pair_i, pair_j = torch.triu_indices(num_states, num_states, offset=1)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)

    def _normalize(self, sequence: torch.Tensor, equilibrium: torch.Tensor) -> torch.Tensor:
        scale = self.state_scale.to(dtype=sequence.dtype).view(1, 1, 1, -1)
        return ((sequence - equilibrium) / scale).flatten(start_dim=2)

    def _lowpass(self, normalized: torch.Tensor) -> torch.Tensor:
        values = normalized.transpose(1, 2)
        values = torch.nn.functional.avg_pool1d(
            values,
            kernel_size=self.lowpass_width,
            stride=1,
            padding=self.lowpass_width // 2,
            count_include_pad=False,
        )
        return values.transpose(1, 2)

    def _encode(
        self, normalized: torch.Tensor, offline_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.frame_encoder(torch.cat([normalized, offline_states.detach()], dim=-1))
        assignment = torch.softmax(self.assignment_head(latent), dim=-1)
        reconstruction = self.slow_decoder(latent)
        energy = self.free_energy_head(latent).squeeze(-1)
        return latent, assignment, reconstruction, energy

    def _reversible_generator(
        self,
        offline_context: torch.Tensor,
        equilibrium: torch.Tensor,
        temperature_k: torch.Tensor | None,
        dataset_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = self.condition_trunk(offline_context.detach().mean(dim=1))
        if self.condition_on_equilibrium:
            condition = condition + self.equilibrium_condition_trunk(
                equilibrium.detach().flatten(start_dim=2).squeeze(1)
            )
        condition = condition + self.condition_embed(
            temperature_k,
            dataset_index,
            offline_context.shape[0],
            offline_context.device,
            offline_context.dtype,
        )
        state_energy = self.state_free_energy_head(condition)
        if temperature_k is None:
            inverse_temperature = torch.ones(
                offline_context.shape[0], 1, device=condition.device, dtype=condition.dtype
            )
        else:
            temperature = torch.nan_to_num(temperature_k, nan=300.0).clamp(250.0, 500.0)
            inverse_temperature = (300.0 / temperature).to(dtype=condition.dtype).unsqueeze(-1)
        stationary = torch.softmax(-state_energy * inverse_temperature, dim=-1)
        stationary = stationary.clamp(min=1.0e-4)
        stationary = stationary / stationary.sum(dim=-1, keepdim=True)

        pair_flux = torch.nn.functional.softplus(self.symmetric_flux_head(condition)) * 0.01
        symmetric_flux = condition.new_zeros(
            condition.shape[0], self.num_states, self.num_states
        )
        symmetric_flux[:, self.pair_i, self.pair_j] = pair_flux
        symmetric_flux[:, self.pair_j, self.pair_i] = pair_flux
        generator = symmetric_flux / stationary.unsqueeze(-1)
        generator = generator - torch.diag_embed(generator.sum(dim=-1))
        base_transition = torch.matrix_exp(generator)
        detailed_balance = (
            stationary.unsqueeze(-1) * base_transition
            - stationary.unsqueeze(-2) * base_transition.transpose(-1, -2)
        )
        return generator, base_transition, stationary, detailed_balance

    @staticmethod
    def _transition_at_steps(generator: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(generator * steps.to(dtype=generator.dtype).view(-1, 1, 1))

    def force_energy(
        self,
        physical_state: torch.Tensor,
        equilibrium: torch.Tensor,
        offline_endpoint: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self._normalize(physical_state, equilibrium)
        latent, _, _, energy = self._encode(normalized, offline_endpoint[:, None])
        return energy[:, 0]

    def forward(
        self,
        context_sequence: torch.Tensor,
        future_sequence: torch.Tensor,
        context_mask: torch.Tensor,
        future_mask: torch.Tensor,
        context_offline_states: torch.Tensor,
        future_offline_states: torch.Tensor,
        horizon_frames: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context_sequence.dim() != 4 or future_sequence.shape != context_sequence.shape:
            raise ValueError("V4-C physical sequences must be matching [B,T,K,D]")
        if context_offline_states.shape[:2] != context_sequence.shape[:2]:
            raise ValueError("V4-C context offline states must align with physical frames")
        if future_offline_states.shape[:2] != future_sequence.shape[:2]:
            raise ValueError("V4-C future offline states must align with physical frames")
        valid = context_mask.unsqueeze(-1).to(dtype=context_sequence.dtype)
        equilibrium = (context_sequence * valid).sum(dim=1, keepdim=True) / valid.sum(
            dim=1, keepdim=True
        ).clamp(min=1.0)
        context_normalized = self._normalize(context_sequence, equilibrium)
        future_normalized = self._normalize(future_sequence, equilibrium)
        context_lowpass = self._lowpass(context_normalized)
        future_lowpass = self._lowpass(future_normalized)
        if self.dynamic_only:
            context_encoder_input = context_lowpass
            future_encoder_input = future_lowpass
            context_offline_input = torch.zeros_like(context_offline_states)
            future_offline_input = torch.zeros_like(future_offline_states)
        else:
            context_encoder_input = context_normalized
            future_encoder_input = future_normalized
            context_offline_input = context_offline_states
            future_offline_input = future_offline_states
        context_latent, context_assignment, context_reconstruction, context_energy = self._encode(
            context_encoder_input, context_offline_input
        )
        future_latent, future_assignment, future_reconstruction, future_energy = self._encode(
            future_encoder_input, future_offline_input
        )
        context_residual = context_normalized - context_reconstruction
        future_residual = future_normalized - future_reconstruction

        generator, base_transition, stationary, detailed_balance = self._reversible_generator(
            context_offline_input, equilibrium, temperature_k, dataset_index
        )
        steps = torch.div(horizon_frames, self.base_step_frames, rounding_mode="floor")
        if ((steps * self.base_step_frames) != horizon_frames).any() or (steps <= 0).any():
            raise ValueError("V4-C horizons must be positive multiples of the base step")
        transition = self._transition_at_steps(generator, steps)
        endpoint = context_assignment[:, -1]
        rollout_prediction = torch.bmm(endpoint.unsqueeze(1), transition).squeeze(1)
        lag = self.base_step_frames // self.history_frame_stride
        if lag >= context_sequence.shape[1]:
            raise ValueError("V4-C context is too short for its base step")
        base_current = context_assignment[:, -1 - lag]
        base_prediction = torch.bmm(base_current.unsqueeze(1), base_transition).squeeze(1)

        energy_gradient = torch.zeros_like(context_latent)
        if torch.is_grad_enabled() and context_latent.requires_grad:
            energy_gradient = torch.autograd.grad(
                context_energy.sum(), context_latent, create_graph=self.training, retain_graph=True
            )[0]
        mobility = torch.nn.functional.softplus(self.mobility_head(context_latent)) + 1.0e-4
        drift = -mobility * energy_gradient
        latent_delta = (context_latent[:, lag:] - context_latent[:, :-lag]) / float(lag)

        result = {
            "v4c_context_latent": context_latent,
            "v4c_future_latent": future_latent,
            "v4c_context_assignment": context_assignment,
            "v4c_future_assignment": future_assignment,
            "v4c_context_reconstruction": context_reconstruction,
            "v4c_future_reconstruction": future_reconstruction,
            "v4c_context_lowpass_target": context_lowpass,
            "v4c_future_lowpass_target": future_lowpass,
            "v4c_context_dynamic_input": context_encoder_input,
            "v4c_future_dynamic_input": future_encoder_input,
            "v4c_context_residual": context_residual,
            "v4c_future_residual": future_residual,
            "v4c_context_energy": context_energy,
            "v4c_future_energy": future_energy,
            "v4c_energy_drift": drift[:, :-lag],
            "v4c_latent_delta": latent_delta,
            "v4c_context_mask": context_mask.all(dim=-1),
            "v4c_future_mask": future_mask.all(dim=-1),
            "v4c_generator": generator,
            "v4c_base_transition_matrix": base_transition,
            "v4c_transition_matrix": transition,
            "v4c_stationary_distribution": stationary,
            "v4c_detailed_balance_residual": detailed_balance,
            "v4c_current_assignment": endpoint,
            "v4c_target_assignment": future_assignment[:, -1],
            "v4c_prediction": rollout_prediction,
            "v4c_base_current": base_current,
            "v4c_base_target": context_assignment[:, -1],
            "v4c_base_prediction": base_prediction,
            "v4c_equilibrium": equilibrium,
        }
        if not self.training or self.training_fixed_controls:
            fixed = []
            for horizon in (48, 64, 96, 128, 192):
                fixed_steps = torch.full_like(steps, horizon // self.base_step_frames)
                fixed_transition = self._transition_at_steps(generator, fixed_steps)
                fixed.append(torch.bmm(endpoint.unsqueeze(1), fixed_transition).squeeze(1))
            result["v4c_fixed_predictions"] = torch.stack(fixed, dim=1)
            result["v4c_fixed_horizons"] = torch.tensor(
                (48, 64, 96, 128, 192), device=steps.device
            )
        return result


class GaussianResidualTransition(nn.Module):
    """Bounded diagonal Gaussian residual transition in latent state space."""

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        thermodynamic_conditioning: bool = True,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 1.5,
    ):
        super().__init__()
        if log_sigma_min >= log_sigma_max:
            raise ValueError("log_sigma_min must be smaller than log_sigma_max")
        self.time_embed = LogTimeEmbedding(latent_dim)
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
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
        self.mean = nn.Linear(hidden, latent_dim)
        self.log_sigma = nn.Linear(hidden, latent_dim)
        # Persistence is the correct neutral prior at initialization. The
        # variance starts finite and is learned from the data.
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.constant_(self.log_sigma.bias, -1.5)

    def forward(
        self,
        endpoint_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if endpoint_state.dim() != 2:
            raise ValueError("endpoint_state must be [B,D]")
        if horizon_time_ps.dim() != 1 or horizon_time_ps.shape[0] != endpoint_state.shape[0]:
            raise ValueError("horizon_time_ps must be [B]")
        B = endpoint_state.shape[0]
        condition = self.time_embed(horizon_time_ps).to(dtype=endpoint_state.dtype)
        if self.condition_embed is not None:
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, B, endpoint_state.device, endpoint_state.dtype
            )
        hidden = self.trunk(endpoint_state + condition)
        mean = self.mean(hidden)
        log_sigma = self.log_sigma(hidden).clamp(self.log_sigma_min, self.log_sigma_max)
        return mean, log_sigma


class ObservableHead(nn.Module):
    """Predict coordinate-derived, low-dimensional motion observables."""

    def __init__(
        self,
        latent_dim: int = 128,
        output_dim: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        thermodynamic_conditioning: bool = True,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("observable output dimension must be positive")
        hidden = latent_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)
        self.time_embed = LogTimeEmbedding(latent_dim)
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None

    def forward(
        self,
        endpoint_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if endpoint_state.dim() != 2:
            raise ValueError("endpoint_state must be [B,D]")
        B = endpoint_state.shape[0]
        condition = self.time_embed(horizon_time_ps).to(dtype=endpoint_state.dtype)
        if self.condition_embed is not None:
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, B, endpoint_state.device, endpoint_state.dtype
            )
        return self.net(endpoint_state + condition)


class SlowStateProjector(nn.Module):
    """Low-dimensional projection for slow/metastable dynamics readouts."""

    def __init__(self, latent_dim: int = 128, output_dim: int = 32, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("slow state dimension must be positive")
        hidden = latent_dim * 2
        layers: list[nn.Module] = [
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() != 2:
            raise ValueError("slow state projector expects [B,D]")
        return self.output_norm(self.net(state))


class LaggedLinearDynamicsPredictor(nn.Module):
    """Linear multi-lag state transition heads for Koopman-friendly latents."""

    def __init__(self, latent_dim: int = 128, lags: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        if any(lag <= 0 for lag in lags):
            raise ValueError("linear dynamics lags must be positive")
        self.lags = tuple(int(lag) for lag in lags)
        self.heads = nn.ModuleDict({str(lag): nn.Linear(latent_dim, latent_dim) for lag in self.lags})
        for head in self.heads.values():
            nn.init.eye_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, frame_states: torch.Tensor) -> dict[int, torch.Tensor]:
        if frame_states.dim() != 3:
            raise ValueError("frame_states must be [B,T,D]")
        return {lag: self.heads[str(lag)](frame_states) for lag in self.lags}


class LaggedPhyschemPredictor(nn.Module):
    """Predict low-dimensional physicochemical frame-pair events at multiple lags."""

    def __init__(
        self,
        latent_dim: int = 128,
        lags: tuple[int, ...] = (1, 2, 4, 8),
        output_dim: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if any(lag <= 0 for lag in lags):
            raise ValueError("physchem prediction lags must be positive")
        self.lags = tuple(int(lag) for lag in lags)
        self.output_dim = int(output_dim)
        hidden = latent_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({str(lag): nn.Linear(hidden, self.output_dim) for lag in self.lags})

    def forward(self, frame_states: torch.Tensor) -> dict[int, torch.Tensor]:
        if frame_states.dim() != 3:
            raise ValueError("frame_states must be [B,T,D]")
        h = self.trunk(frame_states)
        return {lag: self.heads[str(lag)](h) for lag in self.lags}


class KineticReadoutHead(nn.Module):
    """Low-dimensional VAMPNet-style readout from temporal frame states."""

    def __init__(self, latent_dim: int = 128, output_dim: int = 8, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        hidden = latent_dim * 2
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.extend([nn.Linear(hidden, output_dim), nn.LayerNorm(output_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, frame_states: torch.Tensor) -> torch.Tensor:
        if frame_states.dim() != 3:
            raise ValueError("frame_states must be [B,T,D]")
        return self.net(frame_states)
