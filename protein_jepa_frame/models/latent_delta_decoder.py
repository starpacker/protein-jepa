"""CA-only experimental decoder for testing information in latent increments.

This component probe is not an atom37 decoder or a drop-in GLE checkpoint.
Anchor-local outputs guarantee rigid-motion equivariance and exact zero change
for zero latent increment. Dropout is disabled so paired subtraction is exact.
"""
import torch
from torch import nn


class LatentDeltaCADecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden=128, depth=3, max_update=12.0):
        super().__init__()
        self.max_update = max_update
        self.input = nn.Linear(2 * latent_dim + 6, hidden)
        self.aa = nn.Embedding(21, hidden)
        self.position = nn.Embedding(4096, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.,
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, future_h, anchor_h, aatype, residue_index, chain_id,
                rotation, anchor, mask):
        # Invariant local vectors to consecutive residues; never cross gaps/chains.
        left = torch.zeros_like(anchor)
        right = torch.zeros_like(anchor)
        valid = (mask[:, 1:] & mask[:, :-1]
                 & (chain_id[:, 1:] == chain_id[:, :-1])
                 & ((residue_index[:, 1:] - residue_index[:, :-1]) == 1))
        edge = (anchor[:, 1:] - anchor[:, :-1]) * valid[..., None]
        left[:, 1:] = -edge
        right[:, :-1] = edge
        local = torch.cat([
            (rotation.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
            for v in (left, right)
        ], dim=-1) / 3.8
        delta = future_h - anchor_h
        features = torch.cat((anchor_h, delta, local), dim=-1)
        baseline = torch.cat((anchor_h, torch.zeros_like(delta), local), dim=-1)
        # Both inputs share context and weights. No global orientation is exposed.
        context = self.aa(aatype) + self.position(residue_index.clamp(0, 4095))
        x = self.input(torch.cat((features, baseline), dim=0)) + context.repeat(2, 1, 1)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=~mask.repeat(2, 1))
        changed, fixed = self.head(x).chunk(2)
        displacement = self.max_update * (changed.tanh() - fixed.tanh())
        displacement = displacement * mask[..., None]
        return anchor + (rotation @ displacement.unsqueeze(-1)).squeeze(-1)
