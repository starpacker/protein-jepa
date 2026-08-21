from __future__ import annotations

import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, latent_dim))
        self.time_embed = nn.Embedding(max_frames, latent_dim)
        self.residue_embed = nn.Embedding(max_residues, latent_dim)
        self.chain_embed = nn.Embedding(128, latent_dim)
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
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.time_embed = nn.Embedding(max_frames, latent_dim)
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
    ) -> torch.Tensor:
        if context_states.dim() != 3:
            raise ValueError("context_states must be [B,T,D]")
        B, T, D = context_states.shape
        time_pos = torch.arange(T, device=context_states.device).clamp(max=self.time_embed.num_embeddings - 1)
        x = context_states * context_frame_mask.unsqueeze(-1).float()
        x = x + self.mask_token * target_frame_mask.unsqueeze(-1).float()
        x = x + self.time_embed(time_pos).view(1, T, D)
        token_valid = context_frame_mask | target_frame_mask
        x = self.blocks(x, src_key_padding_mask=~token_valid)
        return self.proj(self.norm(x))
