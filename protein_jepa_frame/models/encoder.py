from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _internal_distances(coords: torch.Tensor) -> torch.Tensor:
    pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3))
    values = [torch.linalg.norm(coords[..., i, :] - coords[..., j, :], dim=-1) for i, j in pairs]
    return torch.stack(values, dim=-1)


class VisibleGraphAttentionLayer(nn.Module):
    """Distance-biased attention over visible residues inside each frame."""

    def __init__(self, dim: int, num_heads: int, num_rbf: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(num_rbf, num_heads, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, ca: torch.Tensor, valid: torch.Tensor, rbf: torch.Tensor) -> torch.Tensor:
        Bf, L, D = h.shape
        x = self.norm1(h)
        q = self.q(x).view(Bf, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(Bf, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(Bf, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores + self.edge_bias(rbf).permute(0, 3, 1, 2)
        key_mask = valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * key_mask.float()
        update = torch.matmul(attn, v).transpose(1, 2).reshape(Bf, L, D)
        h = h + self.dropout(self.out(update))
        h = h + self.dropout(self.mlp(self.norm2(h)))
        return h * valid.unsqueeze(-1).float()


class ProteinGeometryEncoder(nn.Module):
    """SE(3)-invariant residue-frame encoder for `[B,T,L,4,3]` backbone clips.

    The layer uses intra-residue distances and visible-only pairwise CA
    distances. It deliberately avoids absolute coordinates.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        num_rbf: int = 16,
        dropout: float = 0.1,
        max_residues: int = 4096,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_rbf = num_rbf
        self.aatype_embed = nn.Embedding(21, latent_dim)
        self.residue_embed = nn.Embedding(max_residues, latent_dim)
        self.chain_embed = nn.Embedding(128, latent_dim)
        self.geom_proj = nn.Linear(6, latent_dim)
        self.layers = nn.ModuleList(
            [VisibleGraphAttentionLayer(latent_dim, num_heads, num_rbf, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.register_buffer("rbf_centers", torch.linspace(2.0, 22.0, num_rbf), persistent=False)

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        width = (self.rbf_centers[1] - self.rbf_centers[0]).clamp(min=1.0)
        return torch.exp(-((distances.unsqueeze(-1) - self.rbf_centers) / width) ** 2)

    def forward(
        self,
        coords: torch.Tensor,
        residue_mask: torch.Tensor,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords.dim() != 5 or coords.shape[-2:] != (4, 3):
            raise ValueError("coords must be [B,T,L,4,3]")
        B, T, L = coords.shape[:3]
        valid = residue_mask if token_mask is None else (residue_mask & token_mask)
        valid_f = valid.reshape(B * T, L)
        coords_f = coords.reshape(B * T, L, 4, 3)
        ca = coords_f[:, :, 1, :]

        geom = _internal_distances(coords_f).clamp(max=50.0)
        aa = aatype.clamp(0, 20)
        res_pos = residue_index.clamp(min=0, max=self.residue_embed.num_embeddings - 1)
        chain_pos = chain_id.clamp(min=0, max=self.chain_embed.num_embeddings - 1)
        base = (
            self.geom_proj(geom)
            + self.aatype_embed(aa).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
            + self.residue_embed(res_pos).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
            + self.chain_embed(chain_pos).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
        )
        h = base * valid_f.unsqueeze(-1).float()

        distances = torch.cdist(ca, ca).clamp(max=80.0)
        pair_valid = valid_f[:, :, None] & valid_f[:, None, :]
        rbf = self._rbf(distances) * pair_valid.unsqueeze(-1).float()
        for layer in self.layers:
            h = layer(h, ca, valid_f, rbf)
        h = self.norm(h) * valid_f.unsqueeze(-1).float()
        return h.reshape(B, T, L, self.latent_dim)
