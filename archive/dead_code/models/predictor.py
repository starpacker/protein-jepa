from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ContextualAdaptiveDynamics(nn.Module):
    """Continuous-time dynamics in a context-specific adaptive state basis.

    The slots are generated from the observed history, so slot indices do not
    carry global state semantics across proteins. The transition generator is
    recomputed after every integration step from the evolving state posterior
    and memory. This makes both the assignment and the dynamics depend on the
    observed motion, while retaining a valid rate matrix at every step.
    """

    def __init__(
        self,
        latent_dim: int,
        slot_dim: int = 32,
        max_slots: int = 16,
        base_step_ps: float = 16000.0,
        max_rollout_steps: int = 16,
        assignment_temperature: float = 0.25,
        observable_dim: int = 12,
        control_horizons_frames: tuple[int, ...] = (),
        training_fixed_controls: bool = False,
        thermodynamic_conditioning: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_dim <= 0 or slot_dim <= 0 or max_slots < 2:
            raise ValueError("V5 latent, slot, and state dimensions must be positive")
        if base_step_ps <= 0 or max_rollout_steps <= 0:
            raise ValueError("V5 integration scales must be positive")
        if assignment_temperature <= 0:
            raise ValueError("V5 assignment temperature must be positive")
        if observable_dim <= 0:
            raise ValueError("V5 observable dimension must be positive")
        self.latent_dim = int(latent_dim)
        self.slot_dim = int(slot_dim)
        self.max_slots = int(max_slots)
        self.base_step_ps = float(base_step_ps)
        self.max_rollout_steps = int(max_rollout_steps)
        self.assignment_temperature = float(assignment_temperature)
        self.observable_dim = int(observable_dim)
        self.control_horizons_frames = tuple(int(value) for value in control_horizons_frames)
        self.training_fixed_controls = bool(training_fixed_controls)

        self.history_encoder = nn.GRU(latent_dim, slot_dim, batch_first=True)
        self.history_norm = nn.LayerNorm(slot_dim)
        self.state_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, slot_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.history_state_projection = nn.Linear(slot_dim, slot_dim, bias=False)
        self.slot_seed = nn.Parameter(torch.empty(max_slots, slot_dim))
        self.slot_generator = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, max_slots * slot_dim),
        )
        self.slot_gate = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, max_slots),
        )
        self.condition_embed = (
            ThermodynamicConditionEmbedding(slot_dim) if thermodynamic_conditioning else None
        )
        self.posterior_to_memory = nn.Linear(slot_dim, slot_dim, bias=False)
        self.memory_cell = nn.GRUCell(slot_dim, slot_dim)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(slot_dim * 2),
            nn.Linear(slot_dim * 2, slot_dim),
            nn.SiLU(),
            nn.Linear(slot_dim, 1),
        )
        self.flux_head = nn.Sequential(
            nn.LayerNorm(slot_dim * 4),
            nn.Linear(slot_dim * 4, slot_dim * 2),
            nn.SiLU(),
            nn.Linear(slot_dim * 2, 1),
        )
        self.observable_head = nn.Sequential(
            nn.LayerNorm(slot_dim * 2),
            nn.Linear(slot_dim * 2, slot_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(slot_dim * 2, observable_dim),
        )
        self.log_rate_scale = nn.Parameter(torch.tensor(-6.0))
        pair_i, pair_j = torch.triu_indices(max_slots, max_slots, offset=1)
        self.register_buffer("pair_i", pair_i, persistent=False)
        self.register_buffer("pair_j", pair_j, persistent=False)
        nn.init.normal_(self.slot_seed, std=0.02)

    @staticmethod
    def _last_valid(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        index = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
        last = torch.where(mask, index, -1).max(dim=1).values.clamp(min=0)
        return sequence[torch.arange(sequence.shape[0], device=sequence.device), last]

    def _history(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if states.dim() != 3 or mask.shape != states.shape[:2]:
            raise ValueError("V5 history states and mask must be [B,T,D] and [B,T]")
        encoded, _ = self.history_encoder(states * mask.unsqueeze(-1).to(dtype=states.dtype))
        return self.history_norm(self._last_valid(encoded, mask))

    def _state_feature(
        self, states: torch.Tensor, mask: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        endpoint = self._last_valid(states, mask)
        feature = self.state_projection(endpoint) + self.history_state_projection(history)
        return torch.nn.functional.normalize(feature, dim=-1, eps=1.0e-6)

    def _context_slots(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.slot_generator(history).view(
            history.shape[0], self.max_slots, self.slot_dim
        )
        slots = torch.nn.functional.normalize(
            self.slot_seed.unsqueeze(0) + delta, dim=-1, eps=1.0e-6
        )
        gates = torch.sigmoid(self.slot_gate(history)).clamp(1.0e-4, 1.0 - 1.0e-4)
        return slots, gates

    def _assign(
        self, feature: torch.Tensor, slots: torch.Tensor, gates: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.einsum("bd,bkd->bk", feature, slots)
        logits = logits / self.assignment_temperature + gates.log()
        return torch.softmax(logits, dim=-1)

    def _generator(
        self,
        memory: torch.Tensor,
        posterior: torch.Tensor,
        slots: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.bmm(posterior.unsqueeze(1), slots).squeeze(1)
        condition = self.history_norm(memory + self.posterior_to_memory(state))
        expanded_condition = condition.unsqueeze(1).expand(-1, self.max_slots, -1)
        energies = self.energy_head(torch.cat([slots, expanded_condition], dim=-1)).squeeze(-1)
        stationary = torch.softmax(-energies + gates.log(), dim=-1)
        stationary = stationary.clamp(min=1.0e-4)
        stationary = stationary / stationary.sum(dim=-1, keepdim=True)

        left = slots[:, self.pair_i]
        right = slots[:, self.pair_j]
        pair_condition = condition.unsqueeze(1).expand(-1, self.pair_i.numel(), -1)
        pair_features = torch.cat([left, right, (left - right).abs(), pair_condition], dim=-1)
        pair_flux = torch.nn.functional.softplus(
            self.flux_head(pair_features).squeeze(-1) + self.log_rate_scale
        )
        pair_flux = pair_flux * gates[:, self.pair_i] * gates[:, self.pair_j]
        symmetric_flux = slots.new_zeros(
            slots.shape[0], self.max_slots, self.max_slots
        )
        symmetric_flux[:, self.pair_i, self.pair_j] = pair_flux
        symmetric_flux[:, self.pair_j, self.pair_i] = pair_flux
        generator = symmetric_flux / stationary.unsqueeze(-1)
        generator = generator - torch.diag_embed(generator.sum(dim=-1))
        detailed_balance = (
            stationary.unsqueeze(-1) * generator
            - stationary.unsqueeze(-2) * generator.transpose(-1, -2)
        )
        return generator, stationary, detailed_balance

    def _rollout(
        self,
        posterior: torch.Tensor,
        memory: torch.Tensor,
        slots: torch.Tensor,
        gates: torch.Tensor,
        horizon_time_ps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled_time = horizon_time_ps.to(dtype=posterior.dtype).clamp(min=0.0) / self.base_step_ps
        whole_steps = torch.floor(scaled_time).to(dtype=torch.long)
        if int(whole_steps.max().detach().cpu()) > self.max_rollout_steps:
            raise ValueError(
                f"V5 horizon requires {int(whole_steps.max())} steps, "
                f"above max_rollout_steps={self.max_rollout_steps}"
            )
        initial_generator, initial_stationary, detailed_balance = self._generator(
            memory, posterior, slots, gates
        )
        current = posterior
        current_memory = memory
        for step in range(int(whole_steps.max().detach().cpu())):
            generator, _, _ = self._generator(current_memory, current, slots, gates)
            transition = torch.matrix_exp(generator)
            candidate = torch.bmm(current.unsqueeze(1), transition).squeeze(1)
            candidate_state = torch.bmm(candidate.unsqueeze(1), slots).squeeze(1)
            candidate_memory = self.memory_cell(candidate_state, current_memory)
            active = (whole_steps > step).unsqueeze(-1)
            current = torch.where(active, candidate, current)
            current_memory = torch.where(active, candidate_memory, current_memory)
        fraction = (scaled_time - whole_steps.to(dtype=scaled_time.dtype)).view(-1, 1, 1)
        generator, _, _ = self._generator(current_memory, current, slots, gates)
        transition = torch.matrix_exp(generator * fraction)
        current = torch.bmm(current.unsqueeze(1), transition).squeeze(1)
        return current, initial_generator, initial_stationary, detailed_balance

    def forward(
        self,
        context_states: torch.Tensor,
        future_states: torch.Tensor,
        context_mask: torch.Tensor,
        future_mask: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        horizon_frames: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        history = self._history(context_states, context_mask)
        if self.condition_embed is not None:
            history = self.history_norm(
                history
                + self.condition_embed(
                    temperature_k,
                    dataset_index,
                    context_states.shape[0],
                    context_states.device,
                    context_states.dtype,
                )
            )
        slots, gates = self._context_slots(history)
        current_feature = self._state_feature(context_states, context_mask, history)
        current = self._assign(current_feature, slots, gates)
        prediction, generator, stationary, detailed_balance = self._rollout(
            current, history, slots, gates, horizon_time_ps
        )
        with torch.no_grad():
            future_history = self._history(future_states, future_mask)
            future_feature = self._state_feature(future_states, future_mask, future_history)
            target = self._assign(future_feature, slots.detach(), gates.detach())

        predicted_feature = torch.bmm(prediction.unsqueeze(1), slots).squeeze(1)
        result = {
            "v5_context_history": history,
            "v5_context_slots": slots,
            "v5_slot_gates": gates,
            "v5_current_assignment": current,
            "v5_target_assignment": target,
            "v5_prediction": prediction,
            "v5_current_feature": current_feature,
            "v5_target_feature": future_feature,
            "v5_predicted_feature": predicted_feature,
            "v5_observable_prediction": self.observable_head(
                torch.cat([predicted_feature, history], dim=-1)
            ),
            "v5_generator": generator,
            "v5_stationary_distribution": stationary,
            "v5_detailed_balance_residual": detailed_balance,
        }
        if self.training_fixed_controls or not self.training:
            ps_per_frame = horizon_time_ps / horizon_frames.to(dtype=horizon_time_ps.dtype).clamp(min=1)
            fixed = []
            for fixed_horizon in self.control_horizons_frames:
                fixed_time = ps_per_frame * float(fixed_horizon)
                fixed.append(self._rollout(current, history, slots, gates, fixed_time)[0])
            if fixed:
                result["v5_fixed_predictions"] = torch.stack(fixed, dim=1)
                result["v5_fixed_horizons"] = torch.tensor(
                    self.control_horizons_frames, device=horizon_frames.device
                )
        return result


class DynamicSequenceStateEncoder(nn.Module):
    """Encode conformation residuals and incoming motion into a continuous state."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        dropout: float = 0.0,
        multi_scale_history: bool = False,
        slow_history_stride: int = 2,
    ):
        super().__init__()
        if slow_history_stride <= 1:
            raise ValueError("V5.3 slow_history_stride must be greater than one")
        self.multi_scale_history = bool(multi_scale_history)
        self.slow_history_stride = int(slow_history_stride)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, state_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.history_encoder = nn.GRU(state_dim, state_dim, batch_first=True)
        self.slow_history_encoder = (
            nn.GRU(state_dim, state_dim, batch_first=True)
            if self.multi_scale_history else None
        )
        history_input_dim = state_dim * 4
        self.history_gate = (
            nn.Sequential(
                nn.LayerNorm(history_input_dim), nn.Linear(history_input_dim, 2)
            )
            if self.multi_scale_history else None
        )
        self.history_fusion = (
            nn.Sequential(
                nn.LayerNorm(history_input_dim),
                nn.Linear(history_input_dim, state_dim * 2),
                nn.GELU(),
                nn.Linear(state_dim * 2, state_dim),
            )
            if self.multi_scale_history else None
        )
        self.velocity_projection = (
            nn.Linear(latent_dim, state_dim) if self.multi_scale_history else None
        )
        self.acceleration_projection = (
            nn.Linear(latent_dim, state_dim) if self.multi_scale_history else None
        )
        self.endpoint_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, state_dim)
        )
        self.local_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.amplitude_projection = nn.Linear(latent_dim * 2, state_dim)
        self.static_projection = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, state_dim)
        )
        self.state_norm = nn.LayerNorm(state_dim)
        self.memory_norm = nn.LayerNorm(state_dim)
        nn.init.normal_(self.amplitude_projection.weight, std=1.0e-3)
        nn.init.zeros_(self.amplitude_projection.bias)
        nn.init.orthogonal_(self.local_projection[1].weight)
        nn.init.zeros_(self.local_projection[1].bias)
        if self.multi_scale_history:
            nn.init.zeros_(self.history_fusion[-1].weight)
            nn.init.zeros_(self.history_fusion[-1].bias)
            nn.init.zeros_(self.velocity_projection.weight)
            nn.init.zeros_(self.velocity_projection.bias)
            nn.init.zeros_(self.acceleration_projection.weight)
            nn.init.zeros_(self.acceleration_projection.bias)
            nn.init.constant_(self.history_gate[-1].bias[0], 2.0)
            nn.init.constant_(self.history_gate[-1].bias[1], -2.0)

    @staticmethod
    def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(dtype=states.dtype)
        return (states * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def last_valid(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        index = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
        last = torch.where(mask, index, -1).max(dim=1).values.clamp(min=0)
        rows = torch.arange(sequence.shape[0], device=sequence.device)
        return sequence[rows, last]

    @staticmethod
    def velocity(states: torch.Tensor, time_ps: torch.Tensor | None) -> torch.Tensor:
        velocity = torch.zeros_like(states)
        delta = states[:, 1:] - states[:, :-1]
        if time_ps is not None:
            dt = (time_ps[:, 1:] - time_ps[:, :-1]).abs().clamp(min=1.0e-6)
            scale = dt.mean(dim=1, keepdim=True).clamp(min=1.0e-6)
            delta = delta / (dt / scale).unsqueeze(-1)
        velocity[:, 1:] = delta
        return velocity

    def encode_local_conformation(
        self, states: torch.Tensor, mask: torch.Tensor, return_sequence: bool = False
    ) -> torch.Tensor:
        """Encode instantaneous conformations in one shared coordinate system."""
        if states.dim() != 3 or mask.shape != states.shape[:2]:
            raise ValueError("local conformation states and mask must be [B,T,D] and [B,T]")
        states = torch.nan_to_num(states)
        sequence = self.local_projection(states)
        sequence = torch.where(mask.unsqueeze(-1), sequence, torch.zeros_like(sequence))
        return sequence if return_sequence else self.last_valid(sequence, mask)

    def forward(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        time_ps: torch.Tensor | None = None,
        reference: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if states.dim() != 3 or mask.shape != states.shape[:2]:
            raise ValueError("dynamic states and mask must be [B,T,D] and [B,T]")
        # Transformer padding can produce non-finite values at fully masked
        # positions; multiplication by zero would leave NaNs in the GRU.
        states = torch.nan_to_num(states)
        states = torch.where(mask.unsqueeze(-1), states, torch.zeros_like(states))
        if reference is None:
            # Prefix statistics keep an auxiliary next-frame target genuinely causal.
            weight = mask.unsqueeze(-1).to(dtype=states.dtype)
            prefix_count = weight.cumsum(dim=1).clamp(min=1.0)
            reference_sequence = (states * weight).cumsum(dim=1) / prefix_count
            reference = reference_sequence[:, -1]
        else:
            reference_sequence = reference.unsqueeze(1).expand(-1, states.shape[1], -1)
        centered = states - reference_sequence
        motion = self.velocity(states, time_ps)
        tokens = self.token_projection(torch.cat([centered, motion], dim=-1))
        encoded, _ = self.history_encoder(tokens * mask.unsqueeze(-1).to(tokens.dtype))
        endpoint = self.last_valid(centered, mask)
        last_motion = self.last_valid(motion, mask)
        amplitude = self.amplitude_projection(torch.cat([endpoint, last_motion], dim=-1))
        recent_memory_sequence = self.memory_norm(encoded)
        memory_sequence = recent_memory_sequence
        if self.multi_scale_history:
            stride = self.slow_history_stride
            slow_tokens = tokens[:, ::stride]
            slow_mask = mask[:, ::stride]
            slow_encoded, _ = self.slow_history_encoder(
                slow_tokens * slow_mask.unsqueeze(-1).to(slow_tokens.dtype)
            )
            slow_memory = self.memory_norm(slow_encoded)
            slow_index = torch.arange(states.shape[1], device=states.device) // stride
            slow_sequence = slow_memory[:, slow_index.clamp(max=slow_memory.shape[1] - 1)]
            acceleration = torch.zeros_like(motion)
            acceleration[:, 1:] = motion[:, 1:] - motion[:, :-1]
            kinetic_input = torch.cat(
                [
                    recent_memory_sequence,
                    slow_sequence,
                    self.velocity_projection(motion),
                    self.acceleration_projection(acceleration),
                ],
                dim=-1,
            )
            gate = torch.softmax(self.history_gate(kinetic_input), dim=-1)
            memory_sequence = self.memory_norm(
                gate[..., :1] * recent_memory_sequence
                + gate[..., 1:] * slow_sequence
                + self.history_fusion(kinetic_input)
            )
            memory_sequence = memory_sequence * mask.unsqueeze(-1).to(memory_sequence.dtype)
        memory = self.last_valid(memory_sequence, mask)
        state = self.state_norm(self.endpoint_projection(endpoint) + memory) + amplitude
        state_sequence = (
            self.state_norm(self.endpoint_projection(centered) + memory_sequence)
            + self.amplitude_projection(torch.cat([centered, motion], dim=-1))
        )
        static_sequence = self.static_projection(reference_sequence)
        static = static_sequence[:, -1]
        if return_sequence:
            return (
                state,
                memory,
                static,
                tokens,
                state_sequence,
                memory_sequence,
                static_sequence,
            )
        return state, memory, static, tokens


class ContextualSparseMotionDynamics(nn.Module):
    """History-conditioned continuous dynamics with sparse local motion modes."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int = 32,
        num_modes: int = 16,
        active_modes: int = 4,
        base_step_ps: float = 16000.0,
        max_rollout_steps: int = 16,
        router_temperature: float = 0.5,
        routing_noise_std: float = 0.0,
        active_temperature: float = 0.25,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 1.0,
        observable_dim: int = 12,
        observable_state_weight: float = 0.0,
        control_horizons_frames: tuple[int, ...] = (),
        training_controls: bool = True,
        causal_prediction_enabled: bool = False,
        history_truncation_enabled: bool = False,
        history_keep_fraction: float = 0.5,
        multi_scale_history_enabled: bool = False,
        slow_history_stride: int = 2,
        autonomous_rollout_enabled: bool = False,
        horizon_calibration_enabled: bool = False,
        horizon_calibration_scale: float = 0.5,
        frame_local_target_enabled: bool = False,
        physical_state_anchor_enabled: bool = False,
        learned_state_residual_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_dim <= 0 or state_dim <= 0 or num_modes < 2:
            raise ValueError("V5.1 latent, state, and mode dimensions must be positive")
        if not 1 <= active_modes < num_modes:
            raise ValueError("V5.1 active_modes must be in [1, num_modes)")
        if (
            base_step_ps <= 0
            or max_rollout_steps <= 0
            or router_temperature <= 0
            or routing_noise_std < 0
            or active_temperature <= 0
            or observable_state_weight < 0
        ):
            raise ValueError("V5.1 rollout scales and router temperature must be positive")
        if log_sigma_min >= log_sigma_max:
            raise ValueError("V5.1 log-sigma bounds are invalid")
        self.state_dim = int(state_dim)
        self.num_modes = int(num_modes)
        self.active_modes = int(active_modes)
        self.base_step_ps = float(base_step_ps)
        self.max_rollout_steps = int(max_rollout_steps)
        self.router_temperature = float(router_temperature)
        self.routing_noise_std = float(routing_noise_std)
        self.active_temperature = float(active_temperature)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
        self.observable_dim = int(observable_dim)
        self.observable_state_weight = float(observable_state_weight)
        self.control_horizons_frames = tuple(int(value) for value in control_horizons_frames)
        self.training_controls = bool(training_controls)
        self.causal_prediction_enabled = bool(causal_prediction_enabled)
        if not 0.0 < history_keep_fraction <= 1.0:
            raise ValueError("V5.1 history_keep_fraction must be in (0, 1]")
        self.history_truncation_enabled = bool(history_truncation_enabled)
        self.history_keep_fraction = float(history_keep_fraction)
        self.multi_scale_history_enabled = bool(multi_scale_history_enabled)
        self.slow_history_stride = int(slow_history_stride)
        self.autonomous_rollout_enabled = bool(autonomous_rollout_enabled)
        self.horizon_calibration_enabled = bool(horizon_calibration_enabled)
        if horizon_calibration_scale < 0.0:
            raise ValueError("V5.3 horizon calibration scale must be non-negative")
        self.horizon_calibration_scale = float(horizon_calibration_scale)
        self.frame_local_target_enabled = bool(frame_local_target_enabled)
        self.physical_state_anchor_enabled = bool(physical_state_anchor_enabled)
        if learned_state_residual_scale < 0.0:
            raise ValueError("V5.3 learned state residual scale must be non-negative")
        self.learned_state_residual_scale = float(learned_state_residual_scale)

        self.state_encoder = DynamicSequenceStateEncoder(
            latent_dim,
            state_dim,
            dropout,
            multi_scale_history=self.multi_scale_history_enabled,
            slow_history_stride=self.slow_history_stride,
        )
        self.target_state_encoder = copy.deepcopy(self.state_encoder)
        for parameter in self.target_state_encoder.parameters():
            parameter.requires_grad = False
        self.target_state_encoder.eval()
        self.physical_history_projection = (
            nn.Linear(state_dim, latent_dim, bias=False)
            if self.physical_state_anchor_enabled else None
        )

        self.horizon_embed = LogTimeEmbedding(state_dim)
        self.router = nn.Sequential(
            nn.LayerNorm(state_dim * 4),
            nn.Linear(state_dim * 4, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, num_modes),
        )
        self.router_query = nn.Sequential(
            nn.LayerNorm(state_dim * 3),
            nn.Linear(state_dim * 3, state_dim),
            nn.SiLU(),
        )
        self.mode_seed = nn.Parameter(torch.empty(num_modes, state_dim))
        self.motion_expert = nn.Sequential(
            nn.LayerNorm(state_dim * 5),
            nn.Linear(state_dim * 5, state_dim * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 3, state_dim * 2),
        )
        self.terminal_delta = nn.Sequential(
            nn.LayerNorm(state_dim * 4),
            nn.Linear(state_dim * 4, state_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.horizon_calibration = nn.Sequential(
            nn.LayerNorm(state_dim * 4),
            nn.Linear(state_dim * 4, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.memory_input = nn.Sequential(
            nn.LayerNorm(state_dim * 2), nn.Linear(state_dim * 2, state_dim), nn.GELU()
        )
        self.memory_cell = nn.GRUCell(state_dim, state_dim)
        self.observable_decoder = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
            nn.Linear(state_dim * 2, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, observable_dim),
        )
        self.observable_state_projection = nn.Sequential(
            nn.LayerNorm(observable_dim),
            nn.Linear(observable_dim, state_dim),
        )
        zero_motion = torch.zeros(observable_dim)
        zero_motion[0] = 1.0
        self.register_buffer("zero_motion_observable", zero_motion, persistent=False)
        nn.init.normal_(self.mode_seed, std=0.02)
        nn.init.normal_(self.motion_expert[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.motion_expert[-1].bias)
        nn.init.normal_(self.terminal_delta[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.terminal_delta[-1].bias)
        nn.init.zeros_(self.horizon_calibration[-1].weight)
        nn.init.zeros_(self.horizon_calibration[-1].bias)
        nn.init.normal_(self.observable_state_projection[-1].weight, std=2.0e-2)
        nn.init.zeros_(self.observable_state_projection[-1].bias)
        if self.physical_history_projection is not None:
            nn.init.normal_(self.physical_history_projection.weight, std=2.0e-2)

    def train(self, mode: bool = True) -> ContextualSparseMotionDynamics:
        super().train(mode)
        self.target_state_encoder.eval()
        return self

    @torch.no_grad()
    def sync_target_encoder(self) -> None:
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for target, online in zip(
            self.target_state_encoder.parameters(), self.state_encoder.parameters()
        ):
            target.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)

    def _route(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        static: torch.Tensor,
        horizon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        effective_horizon = torch.zeros_like(horizon) if self.autonomous_rollout_enabled else horizon
        logits = self.router(torch.cat([state, memory, static, effective_horizon], dim=-1))
        # Tie routing to the learned local basis. The old router produced
        # nearly identical logits for long-history contexts, leaving the
        # top-k assignment effectively arbitrary. A context query against
        # normalized mode seeds makes assignment change with the observed
        # state while retaining context-specific, non-global semantics.
        query = F.normalize(self.router_query(torch.cat([state, memory, static], dim=-1)), dim=-1)
        keys = F.normalize(self.mode_seed, dim=-1)
        logits = logits + 0.5 * (query @ keys.T)
        route_logits = logits
        if self.training and self.routing_noise_std > 0.0:
            route_logits = logits + torch.randn_like(logits) * self.routing_noise_std
        probabilities = torch.softmax(route_logits / self.router_temperature, dim=-1)
        indices = route_logits.topk(self.active_modes, dim=-1).indices
        mask = torch.zeros_like(probabilities).scatter(1, indices, 1.0)
        selected = probabilities * mask
        hard = selected / selected.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
        routing = hard.detach() + probabilities - probabilities.detach()
        threshold = route_logits.topk(self.active_modes, dim=-1).values[:, -1:]
        soft_active = torch.sigmoid((logits - threshold.detach()) / self.active_temperature)
        soft_active = soft_active * (
            float(self.active_modes) / soft_active.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
        )
        return routing, probabilities, mask.bool(), soft_active.clamp(max=1.0), route_logits

    def _motion(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        static: torch.Tensor,
        horizon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        routing, probabilities, active_mask, soft_active, route_logits = self._route(
            state, memory, static, horizon
        )
        batch = state.shape[0]
        mode = self.mode_seed.unsqueeze(0).expand(batch, -1, -1)
        effective_horizon = torch.zeros_like(horizon) if self.autonomous_rollout_enabled else horizon
        condition = torch.cat([state, memory, static, effective_horizon], dim=-1)
        condition = condition.unsqueeze(1).expand(-1, self.num_modes, -1)
        parameters = self.motion_expert(torch.cat([condition, mode], dim=-1))
        drift, raw_log_sigma = parameters.chunk(2, dim=-1)
        log_sigma = raw_log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        weight = routing.unsqueeze(-1)
        mean_drift = (weight * drift).sum(dim=1)
        second_moment = (
            weight * (log_sigma.mul(2.0).exp() + drift.square())
        ).sum(dim=1)
        variance_rate = (second_moment - mean_drift.square()).clamp(min=1.0e-8)
        if self.autonomous_rollout_enabled:
            # In autonomous mode every contribution must be a local vector
            # field. Reusing terminal_delta here preserves the warm-started
            # V5.2 capacity without allowing a horizon-dependent endpoint
            # shortcut to dominate the semigroup.
            zero_horizon = torch.zeros_like(horizon)
            local_residual = self.terminal_delta(
                torch.cat([state, memory, static, zero_horizon], dim=-1)
            )
            mean_drift = mean_drift + local_residual
        return mean_drift, variance_rate, probabilities, active_mask, soft_active, route_logits

    def _rollout(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        static: torch.Tensor,
        horizon_time_ps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scaled_time = horizon_time_ps.to(dtype=state.dtype).clamp(min=0.0) / self.base_step_ps
        whole_steps = torch.floor(scaled_time).long()
        max_steps = int(whole_steps.max().detach().cpu())
        if max_steps > self.max_rollout_steps:
            raise ValueError(
                f"V5.1 horizon requires {max_steps} steps, above "
                f"max_rollout_steps={self.max_rollout_steps}"
            )
        current = state
        current_memory = memory
        variance = torch.zeros_like(state)
        horizon = self.horizon_embed(horizon_time_ps).to(dtype=state.dtype)
        _, _, initial_probabilities, initial_active, initial_soft_active, initial_logits = self._motion(
            current, current_memory, static, horizon
        )

        def integrate(dt: torch.Tensor) -> None:
            nonlocal current, current_memory, variance
            drift, variance_rate, _, _, _, _ = self._motion(
                current, current_memory, static, horizon
            )
            delta = drift * dt
            candidate = current + delta
            memory_input = self.memory_input(torch.cat([current, delta], dim=-1))
            candidate_memory = self.memory_cell(memory_input, current_memory)
            active = dt > 0
            current = torch.where(active, candidate, current)
            current_memory = torch.where(active, candidate_memory, current_memory)
            variance = variance + variance_rate * dt.abs()

        for step in range(max_steps):
            integrate((whole_steps > step).to(state.dtype).unsqueeze(-1))
        fraction = (scaled_time - whole_steps.to(scaled_time.dtype)).unsqueeze(-1)
        integrate(fraction)
        if not self.autonomous_rollout_enabled:
            correction = self.terminal_delta(torch.cat([state, memory, static, horizon], dim=-1))
            current = current + correction
        elif self.horizon_calibration_enabled and self.horizon_calibration_scale > 0.0:
            # This is a bounded endpoint residual for finite-data horizon
            # bias. It cannot alter the local vector field or zero-time state,
            # and its zero initialization preserves the autonomous warm start.
            calibration = self.horizon_calibration(
                torch.cat([state, memory, static, horizon], dim=-1)
            )
            current = current + self.horizon_calibration_scale * torch.tanh(calibration)
        log_sigma = 0.5 * variance.clamp(min=torch.exp(state.new_tensor(2 * self.log_sigma_min))).log()
        log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        return current, log_sigma, initial_probabilities, initial_active, initial_soft_active, initial_logits

    def _decode_observable(self, static: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(displacement)
        value = self.observable_decoder(torch.cat([static, displacement], dim=-1))
        anchor = self.observable_decoder(torch.cat([static, zero], dim=-1))
        return self.zero_motion_observable.to(value).unsqueeze(0) + value - anchor

    @staticmethod
    def _reverse_history_keep_endpoint(
        states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        reversed_states = states.clone()
        for row in range(states.shape[0]):
            valid = torch.nonzero(mask[row], as_tuple=False).flatten()
            if valid.numel() > 2:
                reversed_states[row, valid[:-1]] = states[row, valid[:-1].flip(0)]
        return reversed_states

    def forward(
        self,
        context_states: torch.Tensor,
        teacher_context_states: torch.Tensor,
        future_states: torch.Tensor,
        context_mask: torch.Tensor,
        future_mask: torch.Tensor,
        context_time_ps: torch.Tensor | None,
        future_time_ps: torch.Tensor | None,
        horizon_time_ps: torch.Tensor,
        horizon_frames: torch.Tensor,
        context_local_states: torch.Tensor | None = None,
        teacher_context_local_states: torch.Tensor | None = None,
        future_local_states: torch.Tensor | None = None,
        context_physical_states: torch.Tensor | None = None,
        future_physical_states: torch.Tensor | None = None,
        observable_target: torch.Tensor | None = None,
        wrong_context_states: torch.Tensor | None = None,
        wrong_context_mask: torch.Tensor | None = None,
        short_context_states: torch.Tensor | None = None,
        context_group_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        use_local_target = self.frame_local_target_enabled
        if use_local_target and (
            context_local_states is None
            or teacher_context_local_states is None
            or future_local_states is None
        ):
            raise ValueError("V5.3 frame-local targets require context, teacher, and future states")
        if self.physical_state_anchor_enabled:
            if context_physical_states is None or future_physical_states is None:
                raise ValueError("V5.3 physical state anchoring requires context and future states")
            if (
                context_physical_states.shape[-1] != self.state_dim
                or future_physical_states.shape[-1] != self.state_dim
            ):
                raise ValueError(
                    "V5.3 flattened physical state dimension must equal v51_state_dim"
                )
        # V5.3 history must be built from the same instantaneous conformation
        # sequence used by the transition state. Temporal frame states already
        # mix the full prefix and make reversal interventions almost invisible.
        history_input_states = context_local_states if use_local_target else context_states
        if self.physical_state_anchor_enabled:
            history_input_states = history_input_states + self.physical_history_projection(
                context_physical_states.to(dtype=history_input_states.dtype)
            )
        online_reference = self.state_encoder.masked_mean(history_input_states, context_mask)
        if self.causal_prediction_enabled:
            (
                history_current,
                memory,
                static,
                tokens,
                online_state_sequence,
                online_memory_sequence,
                online_static_sequence,
            ) = self.state_encoder(
                history_input_states, context_mask, context_time_ps, return_sequence=True
            )
        else:
            history_current, memory, static, tokens = self.state_encoder(
                history_input_states, context_mask, context_time_ps, reference=online_reference
            )
            online_state_sequence = online_memory_sequence = online_static_sequence = None
        if use_local_target:
            online_current = self.state_encoder.encode_local_conformation(context_local_states, context_mask)
            online_future = self.state_encoder.encode_local_conformation(future_local_states.detach(), future_mask)
            if self.causal_prediction_enabled:
                online_state_sequence = self.state_encoder.encode_local_conformation(
                    context_local_states, context_mask, return_sequence=True
                )
            if self.physical_state_anchor_enabled:
                context_physical = context_physical_states.to(dtype=online_current.dtype)
                future_physical = future_physical_states.to(dtype=online_future.dtype)
                online_current = self.state_encoder.last_valid(context_physical, context_mask) + (
                    self.learned_state_residual_scale * online_current
                )
                online_future = self.state_encoder.last_valid(future_physical, future_mask) + (
                    self.learned_state_residual_scale * online_future
                )
                if self.causal_prediction_enabled:
                    online_state_sequence = context_physical + (
                        self.learned_state_residual_scale * online_state_sequence
                    )
        else:
            online_current = history_current
            online_future, _, _, _ = self.state_encoder(
                future_states.detach(),
                future_mask,
                future_time_ps,
                reference=online_reference,
            )
        with torch.no_grad():
            if use_local_target:
                teacher_current = self.target_state_encoder.encode_local_conformation(
                    teacher_context_local_states, context_mask
                )
                teacher_future = self.target_state_encoder.encode_local_conformation(
                    future_local_states, future_mask
                )
                if self.physical_state_anchor_enabled:
                    teacher_current = self.target_state_encoder.last_valid(
                        context_physical_states.to(dtype=teacher_current.dtype), context_mask
                    ) + self.learned_state_residual_scale * teacher_current
                    teacher_future = self.target_state_encoder.last_valid(
                        future_physical_states.to(dtype=teacher_future.dtype), future_mask
                    ) + self.learned_state_residual_scale * teacher_future
            else:
                teacher_reference = self.target_state_encoder.masked_mean(
                    teacher_context_states, context_mask
                )
                teacher_current, _, _, _ = self.target_state_encoder(
                    teacher_context_states,
                    context_mask,
                    context_time_ps,
                    reference=teacher_reference,
                )
                teacher_future, _, _, _ = self.target_state_encoder(
                    future_states,
                    future_mask,
                    future_time_ps,
                    reference=teacher_reference,
                )

        rolled, log_sigma, probabilities, active_mask, soft_active, router_logits = self._rollout(
            online_current, memory, static, horizon_time_ps
        )
        predicted_delta = rolled - online_current
        online_target_delta = online_future - online_current
        observable_state_delta = torch.zeros_like(online_target_delta)
        if observable_target is not None and self.observable_state_weight > 0.0:
            observable_state_delta = (
                self.observable_state_projection(observable_target.to(dtype=online_current.dtype))
                * self.observable_state_weight
            )
        teacher_target_delta = teacher_future - teacher_current
        # A transition must be measured inside one coordinate system. In the
        # old frame-local path the source came from the online spatial tower
        # while the future came from the EMA tower; their representation offset
        # was much larger than the true horizon-dependent motion. EMA endpoints
        # provide a slowly moving, shared coordinate system for both ends.
        dynamics_target_delta = (
            teacher_target_delta if use_local_target else online_target_delta
        )
        feature_anchor = teacher_current if use_local_target else online_current
        target_delta = dynamics_target_delta + observable_state_delta
        predicted_feature = feature_anchor + predicted_delta
        result = {
            "v51_online_current_feature": online_current,
            "v51_online_future_feature": online_future,
            "v51_online_target_delta": online_target_delta,
            "v51_observable_state_delta": observable_state_delta,
            "v51_teacher_current_feature": teacher_current,
            "v51_teacher_target_feature": teacher_future,
            "v51_teacher_target_delta": teacher_target_delta,
            "v51_current_feature": feature_anchor,
            "v51_target_feature": feature_anchor + target_delta,
            "v51_target_delta": target_delta,
            "v51_predicted_delta": predicted_delta,
            "v51_predicted_feature": predicted_feature,
            "v51_prediction_log_sigma": log_sigma,
            "v51_context_memory": memory,
            "v51_history_state": history_current,
            "v51_static_condition": static,
            "v51_router_probabilities": probabilities,
            "v51_router_logits": router_logits,
            "v51_active_mode_mask": active_mask,
            "v51_soft_active_mode_mask": soft_active,
            "v51_mode_seed": self.mode_seed,
            "v51_dynamic_tokens": tokens,
            "v51_dynamic_token_mask": context_mask,
            "v51_observable_prediction": self._decode_observable(static, predicted_delta),
            "v51_no_rollout_observable": self._decode_observable(
                static, torch.zeros_like(predicted_delta)
            ),
            "v51_oracle_observable": self._decode_observable(static, target_delta),
            "v51_ema_oracle_observable": self._decode_observable(static, teacher_target_delta),
            "v51_online_oracle_observable": self._decode_observable(static, online_target_delta),
        }

        if context_group_index is not None:
            semigroup_predictions = []
            semigroup_targets = []
            for group in context_group_index.unique(sorted=True):
                members = torch.nonzero(context_group_index == group, as_tuple=False).flatten()
                if members.numel() < 2:
                    continue
                order = torch.argsort(horizon_time_ps[members])
                members = members[order]
                for left, right in zip(members[:-1], members[1:]):
                    elapsed = (horizon_time_ps[right] - horizon_time_ps[left]).clamp(min=0.0)
                    composed, _, _, _, _, _ = self._rollout(
                        rolled.index_select(0, left.view(1)),
                        memory.index_select(0, left.view(1)),
                        static.index_select(0, left.view(1)),
                        elapsed.view(1),
                    )
                    semigroup_predictions.append(composed.squeeze(0))
                    semigroup_targets.append(rolled[right].detach())
            if semigroup_predictions:
                result["v51_semigroup_prediction"] = torch.stack(semigroup_predictions)
                result["v51_semigroup_target"] = torch.stack(semigroup_targets)

        # Train the same transition operator on observed prefixes. The target is
        # the next causal state, so this term carries temporal direction into the
        # representation and cannot be solved from the endpoint alone.
        if self.causal_prediction_enabled and context_states.shape[1] > 1:
            prefix_valid = context_mask[:, :-1] & context_mask[:, 1:]
            prefix_state = online_state_sequence[:, :-1]
            prefix_memory = online_memory_sequence[:, :-1]
            prefix_static = online_static_sequence[:, :-1]
            if context_time_ps is None:
                prefix_dt = torch.ones(
                    prefix_state.shape[:2], device=prefix_state.device, dtype=prefix_state.dtype
                )
            else:
                prefix_dt = (
                    context_time_ps[:, 1:] - context_time_ps[:, :-1]
                ).abs().to(dtype=prefix_state.dtype)
            flat_valid = prefix_valid.reshape(-1)
            if flat_valid.any():
                flat_prediction, _, _, _, _, _ = self._rollout(
                    prefix_state.reshape(-1, self.state_dim)[flat_valid],
                    prefix_memory.reshape(-1, self.state_dim)[flat_valid],
                    prefix_static.reshape(-1, self.state_dim)[flat_valid],
                    prefix_dt.reshape(-1)[flat_valid],
                )
                flat_target = online_state_sequence[:, 1:].reshape(-1, self.state_dim)[flat_valid]
                flat_source = prefix_state.reshape(-1, self.state_dim)[flat_valid]
                # Only supervise the transition increment. Predicting the
                # absolute next state would reward a static representation and
                # dominate the long-horizon objective.
                result["v51_causal_next_prediction"] = flat_prediction - flat_source
                result["v51_causal_next_target"] = (flat_target - flat_source.detach()).detach()
                result["v51_causal_next_dt"] = prefix_dt.reshape(-1)[flat_valid]
            else:
                result["v51_causal_next_prediction"] = online_current.new_empty((0, self.state_dim))
                result["v51_causal_next_target"] = online_current.new_empty((0, self.state_dim))
                result["v51_causal_next_dt"] = online_current.new_empty((0,))

        if self.training_controls or not self.training:
            ps_per_frame = horizon_time_ps / horizon_frames.to(horizon_time_ps.dtype).clamp(min=1)
            fixed_features = []
            fixed_observables = []
            for fixed_horizon in self.control_horizons_frames:
                fixed_time = ps_per_frame * float(fixed_horizon)
                fixed_state, _, _, _, _, _ = self._rollout(
                    online_current, memory, static, fixed_time
                )
                fixed_delta = fixed_state - online_current
                fixed_features.append(feature_anchor + fixed_delta)
                fixed_observables.append(self._decode_observable(static, fixed_delta))
            if fixed_features:
                result["v51_fixed_predictions"] = torch.stack(fixed_features, dim=1)
                result["v51_fixed_observable_predictions"] = torch.stack(
                    fixed_observables, dim=1
                )
                result["v51_fixed_horizons"] = torch.tensor(
                    self.control_horizons_frames, device=horizon_frames.device
                )

            reversed_context = (
                self._reverse_history_keep_endpoint(history_input_states, context_mask)
                if wrong_context_states is None
                else wrong_context_states
            )
            wrong_history, wrong_memory, wrong_static, _ = self.state_encoder(
                reversed_context, context_mask, context_time_ps
            )
            wrong_current = online_current if use_local_target else wrong_history
            wrong_rolled, _, _, _, _, _ = self._rollout(
                wrong_current, wrong_memory, wrong_static, horizon_time_ps
            )
            wrong_delta = wrong_rolled - wrong_current
            result["v51_wrong_history_prediction"] = feature_anchor + wrong_delta
            result["v51_wrong_history_observable_prediction"] = self._decode_observable(
                wrong_static, wrong_delta
            )
            if wrong_context_mask is not None:
                if short_context_states is None:
                    short_context_states = history_input_states
                short_history, short_memory, short_static, _ = self.state_encoder(
                    short_context_states, wrong_context_mask, context_time_ps
                )
                short_current = online_current if use_local_target else short_history
                short_rolled, _, _, _, _, _ = self._rollout(
                    short_current, short_memory, short_static, horizon_time_ps
                )
                short_delta = short_rolled - short_current
                result["v51_short_history_prediction"] = feature_anchor + short_delta
                result["v51_short_history_observable_prediction"] = self._decode_observable(
                    short_static, short_delta
                )
        return result


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
        if state.dim() not in {2, 3}:
            raise ValueError("slow state projector expects [B,D] or [B,T,D]")
        return self.output_norm(self.net(state))


class ReversibleKineticOUDynamics(nn.Module):
    """Composable learned-state dynamics with an exact stationary measure.

    A reversible continuous-time Markov chain propagates soft metastable
    assignments. The within-state residual follows an exact diagonal OU
    transition. Both pieces therefore share one local generator across every
    requested horizon and converge to a sequence/temperature-conditioned
    equilibrium without using handcrafted collective variables.
    """

    def __init__(
        self,
        state_dim: int,
        num_states: int = 12,
        base_step_ps: float = 1000.0,
        assignment_temperature: float = 0.5,
        stationary_floor: float = 1.0e-4,
        rate_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if state_dim <= 0 or num_states < 2:
            raise ValueError("LS2 state dimension must be positive and num_states >= 2")
        if base_step_ps <= 0.0 or assignment_temperature <= 0.0 or rate_scale <= 0.0:
            raise ValueError("LS2 time, assignment temperature, and rate scale must be positive")
        if not 0.0 <= stationary_floor < 1.0 / num_states:
            raise ValueError("LS2 stationary floor must be in [0, 1 / num_states)")
        self.state_dim = int(state_dim)
        self.num_states = int(num_states)
        self.base_step_ps = float(base_step_ps)
        self.assignment_temperature = float(assignment_temperature)
        self.stationary_floor = float(stationary_floor)
        self.rate_scale = float(rate_scale)

        hidden = state_dim * 2
        self.assignment_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_states),
        )
        self.target_assignment_head = copy.deepcopy(self.assignment_head)
        for parameter in self.target_assignment_head.parameters():
            parameter.requires_grad = False

        self.sequence_embedding = nn.Embedding(32, state_dim)
        self.thermodynamic_condition = ThermodynamicConditionEmbedding(state_dim)
        self.condition_trunk = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.equilibrium_head = nn.Linear(state_dim, num_states)
        self.flux_head = nn.Linear(state_dim, num_states * num_states)
        nn.init.zeros_(self.equilibrium_head.weight)
        nn.init.zeros_(self.equilibrium_head.bias)
        nn.init.zeros_(self.flux_head.weight)
        nn.init.constant_(self.flux_head.bias, -8.0)

        self.prototypes = nn.Parameter(torch.empty(num_states, state_dim))
        nn.init.normal_(self.prototypes, std=0.2)
        self.log_relaxation_rate = nn.Parameter(torch.full((state_dim,), -8.0))
        self.log_equilibrium_variance = nn.Parameter(torch.zeros(state_dim))
        self.register_buffer(
            "off_diagonal_mask",
            1.0 - torch.eye(num_states),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_assignment_head.eval()
        return self

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target_assignment_head.load_state_dict(self.assignment_head.state_dict())

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, online in zip(
            self.target_assignment_head.parameters(), self.assignment_head.parameters()
        ):
            target.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)

    def assignments(self, state: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        head = self.target_assignment_head if target else self.assignment_head
        return torch.softmax(head(state) / self.assignment_temperature, dim=-1)

    def _static_condition(
        self,
        aatype: torch.Tensor,
        residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None,
        dataset_index: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if aatype.dim() != 2:
            raise ValueError("LS2 aatype must be [B,L]")
        if residue_mask.dim() == 3:
            residue_mask = residue_mask.any(dim=1)
        if residue_mask.shape != aatype.shape:
            raise ValueError("LS2 residue mask must match aatype")
        tokens = self.sequence_embedding(aatype.clamp(min=0, max=31))
        weights = residue_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        sequence = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        thermo = self.thermodynamic_condition(
            temperature_k,
            dataset_index,
            aatype.shape[0],
            aatype.device,
            sequence.dtype,
        )
        return self.condition_trunk(sequence + thermo).to(dtype=dtype)

    def generator(
        self,
        aatype: torch.Tensor,
        residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = self._static_condition(
            aatype, residue_mask, temperature_k, dataset_index, dtype
        )
        logits = self.equilibrium_head(condition)
        stationary = torch.softmax(logits, dim=-1)
        stationary = stationary * (1.0 - self.num_states * self.stationary_floor)
        stationary = stationary + self.stationary_floor

        raw_flux = self.flux_head(condition).view(-1, self.num_states, self.num_states)
        symmetric_flux = 0.5 * (raw_flux + raw_flux.transpose(-1, -2))
        symmetric_flux = F.softplus(symmetric_flux) * self.rate_scale / self.num_states
        symmetric_flux = symmetric_flux * self.off_diagonal_mask.to(dtype=dtype)
        generator = symmetric_flux / stationary.unsqueeze(-1)
        generator = generator - torch.diag_embed(generator.sum(dim=-1))
        detailed_balance = (
            stationary.unsqueeze(-1) * generator
            - stationary.unsqueeze(-2) * generator.transpose(-1, -2)
        )
        return generator, stationary, detailed_balance

    def propagate(
        self,
        assignment: torch.Tensor,
        residual: torch.Tensor,
        generator: torch.Tensor,
        horizon_time_ps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        steps = horizon_time_ps.to(dtype=generator.dtype) / self.base_step_ps
        transition = torch.matrix_exp(generator * steps.view(-1, 1, 1))
        next_assignment = torch.bmm(assignment.unsqueeze(1), transition).squeeze(1)
        relaxation = F.softplus(self.log_relaxation_rate) * self.rate_scale
        decay = torch.exp(-steps.unsqueeze(-1) * relaxation.unsqueeze(0))
        next_residual = decay * residual
        mean = next_assignment @ self.prototypes + next_residual
        equilibrium_variance = F.softplus(self.log_equilibrium_variance) + 1.0e-4
        variance = equilibrium_variance.unsqueeze(0) * (1.0 - decay.square()) + 1.0e-5
        log_sigma = 0.5 * torch.log(variance)
        return mean, log_sigma, next_assignment, transition

    def forward(
        self,
        context_state: torch.Tensor,
        future_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        aatype: torch.Tensor,
        residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
        context_sequence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_assignment = self.assignments(context_state)
        with torch.no_grad():
            future_assignment = self.assignments(future_state, target=True)
        generator, stationary, detailed_balance = self.generator(
            aatype,
            residue_mask,
            temperature_k=temperature_k,
            dataset_index=dataset_index,
            dtype=context_state.dtype,
        )
        context_residual = context_state - context_assignment @ self.prototypes
        mean, log_sigma, prediction, transition = self.propagate(
            context_assignment,
            context_residual,
            generator,
            horizon_time_ps,
        )
        outputs = {
            "ls2_context_state": context_state,
            "ls2_future_state": future_state,
            "ls2_state_prediction": mean,
            "ls2_log_sigma": log_sigma,
            "ls2_context_assignment": context_assignment,
            "ls2_future_assignment": future_assignment,
            "ls2_assignment_prediction": prediction,
            "ls2_context_residual": context_residual,
            "ls2_generator": generator,
            "ls2_transition_matrix": transition,
            "ls2_stationary_distribution": stationary,
            "ls2_detailed_balance_residual": detailed_balance,
        }
        if context_sequence is not None:
            outputs["ls2_context_assignment_sequence"] = self.assignments(context_sequence)
        return outputs


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


class E6AutoregressiveDynamics(nn.Module):
    """
    EVA-6: Autoregressive rollout trainer wrapping ThermodynamicMoriDynamicsV3.

    Adds a learned state residual projector (JEPA latent state -> CV residual),
    runs autoregressive rollouts, and computes training objectives for
    end-to-end learned-state dynamics with equilibrium correctness.
    """

    def __init__(
        self,
        latent_dim: int,
        cv_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        rollout_steps: int = 8,
        noise_std: float = 0.05,
        learned_state_noise_std: float = 0.02,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_dim <= 0 or cv_dim <= 0:
            raise ValueError("E6 latent and CV dimensions must be positive")
        if rollout_steps < 1:
            raise ValueError("E6 rollout steps must be at least 1")
        self.cv_dim = int(cv_dim)
        self.rollout_steps = int(rollout_steps)
        self.noise_std = float(noise_std)
        self.learned_state_noise_std = float(learned_state_noise_std)
        self.temperature = float(temperature)

        # Project learned state to CV residual
        self.state_projector = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cv_dim),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.ones(cv_dim) * 0.1)

        # Target projector (EMA)
        self.target_state_projector = copy.deepcopy(self.state_projector)
        for p in self.target_state_projector.parameters():
            p.requires_grad = False

        # Projected force corrector: state + condition -> correction
        self.force_corrector = nn.Sequential(
            nn.LayerNorm(cv_dim + condition_dim),
            nn.Linear(cv_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cv_dim),
            nn.Tanh(),
        )
        self.max_shift = nn.Parameter(torch.tensor(0.1))

        # Equilibrium distribution: learned state -> energy (for MMD-free matching)
        self.equilibrium_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> E6AutoregressiveDynamics:
        super().train(mode)
        self.target_state_projector.eval()
        return self

    def sync_target(self) -> None:
        self.target_state_projector.load_state_dict(self.state_projector.state_dict())

    def update_target(self, momentum: float) -> None:
        for target, online in zip(
            self.target_state_projector.parameters(),
            self.state_projector.parameters(),
        ):
            target.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)

    def project_state(
        self, learned_state: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        """Map learned state to CV residual."""
        projector = self.target_state_projector if target else self.state_projector
        residual = projector(learned_state)
        return residual * self.residual_scale

    @torch.no_grad()
    def equilibrium_score(
        self, learned_state: torch.Tensor
    ) -> torch.Tensor:
        """Energy landscape for equilibrium reweighting (lower = more likely)."""
        return self.equilibrium_head(learned_state).squeeze(-1)

    def forward(
        self,
        fixed_cv_sequence: torch.Tensor,
        learned_state_sequence: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
        temperature: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        v3_dynamics: nn.Module,
        target_fixed_cv_sequence: torch.Tensor | None = None,
        target_learned_state: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, D = fixed_cv_sequence.shape
        K = self.rollout_steps
        device = fixed_cv_sequence.device
        dtype = fixed_cv_sequence.dtype

        # Project learned state to residual
        context_residual = self.project_state(learned_state_sequence, target=False)
        combined_state = fixed_cv_sequence + context_residual

        # Last valid context state
        last_idx = mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(B, device=device)
        current_state = combined_state[batch_idx, last_idx.clamp(min=0)]

        # Run V3 filter on combined sequence to get auxiliary state
        filtered = v3_dynamics.filter(
            combined_state, mask, torch.zeros(B, T, device=device),
            condition, temperature,
        )
        auxiliary_mean = filtered["mean_sequence"][batch_idx, last_idx.clamp(min=0)]
        auxiliary_covariance = filtered["covariance_sequence"][batch_idx, last_idx.clamp(min=0)]
        auxiliary_logvar = torch.diagonal(auxiliary_covariance, dim1=-2, dim2=-1).clamp(min=1.0e-8).log()

        # Autoregressive rollout
        rollout_predictions = []
        rollout_ensembles = []
        rollout_targets = []
        rollout_aux_means = []
        state = current_state
        aux_mean = auxiliary_mean
        aux_logvar = auxiliary_logvar

        for step in range(K):
            dt = horizon_time_ps  # [B] or scalar

            # Projected force correction
            correction_input = torch.cat([state, condition], dim=-1)
            force_shift = self.force_corrector(correction_input) * self.max_shift
            force_shift = force_shift * (1.0 - torch.exp(-dt / max(1.0, dt.mean())))**2

            # V3 rollout for one step
            samples, aux_samples = v3_dynamics.rollout(
                state, aux_mean, aux_logvar, condition, temperature,
                dt.unsqueeze(-1) if dt.dim() == 0 else dt,
                auxiliary_covariance=auxiliary_covariance,
            )
            samples = samples + force_shift[:, None, :]
            prediction = samples.mean(dim=1)

            rollout_predictions.append(prediction)
            rollout_ensembles.append(samples)

            # Target for this step
            if target_fixed_cv_sequence is not None and step < target_fixed_cv_sequence.shape[1]:
                target = target_fixed_cv_sequence[:, step]
                if target_learned_state is not None:
                    target_residual = self.project_state(
                        target_learned_state[:, step], target=True
                    )
                    target = target + target_residual
                rollout_targets.append(target)
            else:
                rollout_targets.append(prediction.new_zeros(B, D))

            # Prepare for next step: use prediction + noise as next state
            if step < K - 1:
                state = prediction + torch.randn_like(prediction) * self.noise_std
                # Update auxiliary state
                aux_mean = aux_samples.mean(dim=1) if aux_samples is not None else aux_mean

        result = {
            "e6_rollout_predictions": torch.stack(rollout_predictions, dim=1),
            "e6_rollout_ensembles": torch.stack(rollout_ensembles, dim=2),
            "e6_rollout_targets": torch.stack(rollout_targets, dim=1),
            "e6_combined_state": combined_state,
            "e6_current_state": current_state,
            "e6_context_residual": context_residual,
            "e6_auxiliary_mean": auxiliary_mean,
            "e6_force_shift": force_shift,
        }
        return result


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
