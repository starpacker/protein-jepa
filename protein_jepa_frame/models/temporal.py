from __future__ import annotations

import torch
import torch.nn as nn


class TemporalStateEncoder(nn.Module):
    """Temporal transformer over per-frame pooled protein latents."""

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
    ):
        super().__init__()
        self.time_embed = nn.Embedding(max_frames, latent_dim)
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

    def forward(self, residue_latents: torch.Tensor, residue_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if residue_latents.dim() != 4:
            raise ValueError("residue_latents must be [B,T,L,D]")
        B, T, _, D = residue_latents.shape
        valid = residue_mask.unsqueeze(-1).float()
        frame_latents = (residue_latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        time_pos = torch.arange(T, device=residue_latents.device).clamp(max=self.time_embed.num_embeddings - 1)
        x = frame_latents + self.time_embed(time_pos).view(1, T, D)
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)
        frame_valid = residue_mask.any(dim=2)
        padding = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~frame_valid], dim=1)
        x = self.blocks(x, src_key_padding_mask=padding)
        x = self.norm(x)
        return {"clip_state": x[:, 0], "frame_states": x[:, 1:]}
