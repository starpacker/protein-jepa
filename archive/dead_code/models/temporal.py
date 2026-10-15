from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.time_features import LogTimeEmbedding, ThermodynamicConditionEmbedding, relative_clip_time


class KineticQueryFramePooler(nn.Module):
    """Per-frame residue attention pooler for local kinetic signals."""

    def __init__(self, latent_dim: int, num_heads: int, num_queries: int, dropout: float):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        self.num_queries = int(num_queries)
        self.queries = nn.Parameter(torch.zeros(1, self.num_queries, latent_dim))
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(latent_dim * self.num_queries),
            nn.Linear(latent_dim * self.num_queries, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, residue_latents: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        if residue_latents.dim() != 4:
            raise ValueError("residue_latents must be [B,T,L,D]")
        B, T, L, D = residue_latents.shape
        flat_latents = residue_latents.reshape(B * T, L, D)
        flat_mask = residue_mask.reshape(B * T, L)
        if (~flat_mask).all(dim=1).any():
            flat_mask = flat_mask.clone()
            flat_mask[(~flat_mask).all(dim=1), 0] = True
        queries = self.queries.expand(B * T, -1, -1)
        pooled, _ = self.attn(queries, flat_latents, flat_latents, key_padding_mask=~flat_mask, need_weights=False)
        pooled = self.out(pooled.reshape(B * T, self.num_queries * D))
        frame_valid = residue_mask.any(dim=2).reshape(B * T, 1).float()
        return (pooled * frame_valid).reshape(B, T, D)


class TemporalStateEncoder(nn.Module):
    """Temporal transformer over per-frame pooled protein latents."""

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
        frame_pooling: str = "mean",
        frame_pool_queries: int = 4,
        time_conditioning: str = "none",
        thermodynamic_conditioning: bool = False,
        causal: bool = False,
        causal_window: int = 0,
        position_encoding: str = "learned",
        delta_features: bool = False,
        motion_depth: int = 0,
        motion_stopgrad_input: bool = False,
    ):
        super().__init__()
        self.time_conditioning = time_conditioning
        self.thermodynamic_conditioning = thermodynamic_conditioning
        self.causal = causal
        if causal_window < 0:
            raise ValueError("causal_window must be non-negative")
        if causal_window and not causal:
            raise ValueError("causal_window requires causal temporal attention")
        self.causal_window = int(causal_window)
        if position_encoding not in {"learned", "none"}:
            raise ValueError(f"unknown temporal position encoding {position_encoding}")
        self.position_encoding = position_encoding
        self.delta_features = delta_features
        self.motion_depth = int(motion_depth)
        self.motion_stopgrad_input = motion_stopgrad_input
        self.frame_pooling = frame_pooling
        if frame_pooling == "mean":
            self.frame_pooler = None
            self.frame_pool_gate = None
        elif frame_pooling == "kinetic_query":
            self.frame_pooler = KineticQueryFramePooler(latent_dim, num_heads, frame_pool_queries, dropout)
            self.frame_pool_gate = None
        elif frame_pooling == "mean_kinetic_query":
            self.frame_pooler = KineticQueryFramePooler(latent_dim, num_heads, frame_pool_queries, dropout)
            self.frame_pool_gate = nn.Parameter(torch.zeros(latent_dim))
        else:
            raise ValueError(f"unknown frame_pooling {frame_pooling}")
        if time_conditioning == "none":
            self.physical_time_embed = None
        elif time_conditioning == "physical":
            self.physical_time_embed = LogTimeEmbedding(latent_dim)
        else:
            raise ValueError(f"unknown time_conditioning {time_conditioning}")
        self.condition_embed = ThermodynamicConditionEmbedding(latent_dim) if thermodynamic_conditioning else None
        self.delta_proj = (
            nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, latent_dim), nn.GELU())
            if delta_features
            else None
        )
        if self.motion_depth > 0:
            self.motion_delta_proj = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
            )
            self.motion_time_embed = nn.Embedding(max_frames, latent_dim) if position_encoding == "learned" else None
            motion_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.motion_blocks = nn.TransformerEncoder(motion_layer, num_layers=self.motion_depth)
            self.motion_norm = nn.LayerNorm(latent_dim)
            self.motion_gate = nn.Parameter(torch.tensor(-5.0))
            self.state_motion_norm = nn.LayerNorm(latent_dim)
        else:
            self.motion_delta_proj = None
            self.motion_time_embed = None
            self.motion_blocks = None
            self.motion_norm = None
            self.motion_gate = None
            self.state_motion_norm = None
        self.time_embed = nn.Embedding(max_frames, latent_dim) if position_encoding == "learned" else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
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
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        residue_latents: torch.Tensor,
        residue_mask: torch.Tensor,
        time_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if residue_latents.dim() != 4:
            raise ValueError("residue_latents must be [B,T,L,D]")
        B, T, _, D = residue_latents.shape
        valid = residue_mask.unsqueeze(-1).float()
        mean_latents = (residue_latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        if self.frame_pooler is None:
            frame_latents = mean_latents
        elif self.frame_pool_gate is None:
            frame_latents = self.frame_pooler(residue_latents, residue_mask)
        else:
            update = self.frame_pooler(residue_latents, residue_mask)
            frame_latents = mean_latents + self.frame_pool_gate.view(1, 1, D) * update
        time_pos = torch.arange(T, device=residue_latents.device)
        frame_valid = residue_mask.any(dim=2)
        delta = torch.zeros_like(frame_latents)
        delta[:, 1:] = frame_latents[:, 1:] - frame_latents[:, :-1]
        delta_valid = torch.zeros_like(frame_valid)
        delta_valid[:, 1:] = frame_valid[:, 1:] & frame_valid[:, :-1]

        x = frame_latents
        if self.delta_proj is not None:
            x = x + self.delta_proj(delta) * delta_valid.unsqueeze(-1).to(dtype=x.dtype)
        if self.time_embed is not None:
            time_pos = time_pos.clamp(max=self.time_embed.num_embeddings - 1)
            x = x + self.time_embed(time_pos).view(1, T, D)
        relative_time = relative_clip_time(time_ps)
        if self.physical_time_embed is not None and relative_time is not None:
            x = x + self.physical_time_embed(relative_time)
        if self.condition_embed is not None:
            condition = self.condition_embed(temperature_k, dataset_index, B, x.device, x.dtype)
            x = x + condition.unsqueeze(1)
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)
        padding = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~frame_valid], dim=1)
        attention_mask = None
        if self.causal:
            attention_mask = torch.triu(
                torch.ones(T + 1, T + 1, dtype=torch.bool, device=x.device), diagonal=1
            )
            attention_mask[0] = False
            attention_mask[1:, 0] = True
            if self.causal_window:
                query = torch.arange(T, device=x.device)[:, None]
                key = torch.arange(T, device=x.device)[None, :]
                too_old = key < query - self.causal_window + 1
                attention_mask[1:, 1:] |= too_old
        x = self.blocks(x, mask=attention_mask, src_key_padding_mask=padding)
        x = self.norm(x)
        out = {"clip_state": x[:, 0], "frame_states": x[:, 1:]}
        if self.motion_blocks is not None:
            motion_delta = delta.detach() if self.motion_stopgrad_input else delta
            motion = self.motion_delta_proj(motion_delta)
            if self.motion_time_embed is not None:
                motion_pos = time_pos.clamp(max=self.motion_time_embed.num_embeddings - 1)
                motion = motion + self.motion_time_embed(motion_pos).view(1, T, D)
            motion = motion * delta_valid.unsqueeze(-1).to(dtype=motion.dtype)
            motion_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            motion_padding = ~delta_valid
            motion_padding = motion_padding.clone()
            motion_padding[:, 0] = False
            motion = self.motion_blocks(motion, mask=motion_mask, src_key_padding_mask=motion_padding)
            motion = self.motion_norm(motion) * delta_valid.unsqueeze(-1).to(dtype=motion.dtype)
            out["motion_states"] = motion
            state_motion = self.state_motion_norm(
                x[:, 1:].detach() + torch.sigmoid(self.motion_gate) * motion
            )
            out["state_motion_states"] = state_motion * frame_valid.unsqueeze(-1).to(dtype=state_motion.dtype)
        return out
