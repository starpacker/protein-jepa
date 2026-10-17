"""Legacy Stage-A V2 predictor used by the frozen evaluation checkpoints.

The working-tree Stage-A predictor later gained a spatial attention block and a
task embedding.  The evaluation checkpoints from ``stage_a_fix4`` contain the
earlier predictor whose state dict has ``predictor.transformer.*`` keys.  This
module keeps that checkpoint loading explicit; it does not change the semantic
contract:

    H_context [B,K,L,dH] -> pred_factors [B,L,J,r]

The factors are converted to an absolute future H by adding them to the
anchor factors in ``evaluation.loaders.ModelBundle.predict_future``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_a_v2.model import _PositionalEncoding, _TimeEmbedding


class LegacyCausalTransformerPredictor(nn.Module):
    """Checkpoint-compatible Stage-A predictor (pre spatial-attention version)."""

    def __init__(
        self,
        latent_dim: int,
        num_factors: int,
        factor_dim: int,
        depth: int = 4,
        num_heads: int = 4,
        max_frames: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_factors = num_factors
        self.factor_dim = factor_dim
        self.pos_enc = _PositionalEncoding(latent_dim, max_frames)
        self.time_embed = _TimeEmbedding(latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_head = nn.Linear(latent_dim, num_factors * factor_dim)

    @staticmethod
    def _causal_mask(k: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((k, k), float("-inf"), device=device)
        return mask.triu_(diagonal=1)

    def forward(
        self,
        h_ctx: torch.Tensor,
        lag_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        target_frame: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        task_id: int = 1,
    ) -> torch.Tensor:
        del valid_mask, task_id
        b, k, l, d = h_ctx.shape
        if target_frame is not None:
            x_hist = torch.cat([h_ctx[:, :-1], target_frame], dim=1)
        else:
            x_hist = h_ctx
        x = self.pos_enc(x_hist)
        cond = self.time_embed(lag_ps, temperature_k)
        x[:, -1] = x[:, -1] + cond[:, None, :]
        x = x.permute(0, 2, 1, 3).reshape(b * l, k, d)
        x = self.transformer(x, mask=self._causal_mask(k, x.device))
        summary = x[:, -1].reshape(b, l, d)
        out = self.out_head(summary)
        return out.reshape(b, l, self.num_factors, self.factor_dim)
