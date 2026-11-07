"""Versioned encoder candidate with explicit PCA/zero/local relpos semantics.

Parameter and buffer names match ProteinGeometryEncoder with relpos enabled.
Feature mode is mandatory configuration, deliberately not an extra state-dict
key: future loaders must bind it explicitly to a versioned checkpoint contract.
No frozen encoder, legacy loader, or model registry is modified.

All input coordinates must be finite, including hidden rows. This versioned
precondition does not silently repair other frozen geometry/attention channels.
Only local relpos feature construction sanitizes effective-hidden geometry.
"""
from __future__ import annotations

import torch

from protein_jepa_frame.models.encoder import (
    ProteinGeometryEncoder,
    _backbone_torsion_features,
    _internal_distances,
    _local_backbone_frames,
    _orientation_pair_bias,
    _within_frame_canonical_positions,
)

FEATURE_MODES = ('pca', 'zero', 'local')


def _mode(value: str) -> str:
    if type(value) is not str or value not in FEATURE_MODES:
        raise ValueError(f'feature_mode must be one of {FEATURE_MODES}')
    return value


def local_centroid_positions(coords: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Effective-visible centroid in each source residue-local row frame.

    coords [Bf,L,4,3], valid [Bf,L]. The only sanitization in this version is
    internal to this feature. Frames use the frozen eps=1e-8 normalization; finite
    collapsed frames remain on support. No normalization of output or scale fit.
    """
    safe = torch.where(valid[..., None, None], coords, torch.zeros_like(coords))
    ca = safe[:, :, 1, :]
    m = valid.to(ca.dtype).unsqueeze(-1)
    centroid = (ca*m).sum(dim=1, keepdim=True)/m.sum(dim=1, keepdim=True).clamp_min(1.)
    centered = (ca-centroid)*m
    frames = _local_backbone_frames(safe)
    return torch.einsum('blkc,blc->blk', frames, centered)


class RelposModeProteinGeometryEncoder(ProteinGeometryEncoder):
    """Same learned architecture, with a required versioned feature-mode label."""

    def __init__(
        self, latent_dim: int = 128, depth: int = 4, num_heads: int = 4,
        num_rbf: int = 16, dropout: float = 0.1, max_residues: int = 4096,
        orientation_features: bool = False, identity_embed_scale: float = 1.0,
        relative_position_features: bool = True, anchor_conditioned: bool = False,
        *, feature_mode: str,
    ):
        selected = _mode(feature_mode)
        if relative_position_features is not True:
            raise ValueError('relpos projection must remain present in every mode')
        if anchor_conditioned is not False:
            raise ValueError('anchor-conditioned branch is not part of this version')
        super().__init__(latent_dim=latent_dim, depth=depth, num_heads=num_heads,
            num_rbf=num_rbf, dropout=dropout, max_residues=max_residues,
            orientation_features=orientation_features, identity_embed_scale=identity_embed_scale,
            relative_position_features=True, anchor_conditioned=False)
        self.feature_mode = selected

    @property
    def feature_mode(self) -> str:
        return self._feature_mode

    @feature_mode.setter
    def feature_mode(self, value: str) -> None:
        self._feature_mode = _mode(value)

    def relpos_features(self, coords: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mode = _mode(self.feature_mode)
        if mode == 'pca':
            return _within_frame_canonical_positions(coords, valid)
        if mode == 'zero':
            return torch.zeros_like(coords[:, :, 1, :])
        return local_centroid_positions(coords, valid)

    def forward(
        self,
        coords: torch.Tensor,
        residue_mask: torch.Tensor,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        anchor_centroid: torch.Tensor | None = None,
        anchor_basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _mode(self.feature_mode)
        if self.anchor_conditioned or self.anchor_pos_proj is not None or anchor_centroid is not None or anchor_basis is not None:
            raise ValueError('anchor-conditioned branch and supplied anchor arguments are forbidden')
        if self.relpos_proj is None:
            raise ValueError('relpos projection must remain present in every mode')
        if not bool(torch.isfinite(coords).all()):
            raise ValueError('all coordinates, including hidden rows, must be finite')
        # Frozen forward body below, with only the relpos feature provider
        # substituted. The unsupported anchor branch is excluded above.
        if coords.dim() != 5 or coords.shape[-2:] != (4, 3):
            raise ValueError("coords must be [B,T,L,4,3]")
        B, T, L = coords.shape[:3]
        valid = residue_mask if token_mask is None else (residue_mask & token_mask)
        valid_f = valid.reshape(B * T, L)
        coords_f = coords.reshape(B * T, L, 4, 3)
        ca = coords_f[:, :, 1, :]

        geom = _internal_distances(coords_f).clamp(max=50.0)
        if self.orientation_features:
            torsions = _backbone_torsion_features(coords, valid, residue_index, chain_id)
            geom = torch.cat([geom, torsions.reshape(B * T, L, 6)], dim=-1)
        aa = aatype.clamp(0, 20)
        res_pos = residue_index.clamp(min=0, max=self.residue_embed.num_embeddings - 1)
        chain_pos = chain_id.clamp(min=0, max=self.chain_embed.num_embeddings - 1)
        base = (
            self.geom_proj(geom) * self.geom_gain
            + self.aatype_embed(aa).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
            + self.residue_embed(res_pos).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
            + self.chain_embed(chain_pos).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
        )
        if self.relpos_proj is not None:
            relpos = self.relpos_features(coords_f, valid_f)
            base = base + self.relpos_proj(relpos)
        h = base * valid_f.unsqueeze(-1).float()

        distances = torch.cdist(ca, ca).clamp(max=80.0)
        pair_valid = valid_f[:, :, None] & valid_f[:, None, :]
        rbf = self._rbf(distances) * pair_valid.unsqueeze(-1).float()
        orientation_bias = None
        if self.orientation_bias is not None:
            orientation_bias = _orientation_pair_bias(
                coords_f, self.orientation_bias.weight
            ) * pair_valid.unsqueeze(-1).float()
        for layer in self.layers:
            h = layer(h, ca, valid_f, rbf, orientation_bias=orientation_bias)
        h = self.norm(h) * valid_f.unsqueeze(-1).float()
        return h.reshape(B, T, L, self.latent_dim)
