"""Static/dynamic factorization for the v4 representation contract.

Implements the Stage A interfaces from PROTEIN_JEPA_V4_STATIC_DYNAMIC_GLE_GOAL.md:

  H_static = E_s(c, X_0)          -- time-invariant identity/fold context
  H_dyn,t  = E_d(X_t | H_static)  -- magnitude-carrying conformational state
  X_hat,t  = D(H_static, H_dyn,t)

The split is built *around* the existing ``ProteinGeometryEncoder`` rather than
rewriting it, so checkpoint weights transfer.  The key change that the Phase 0
audit demanded is that ``H_dyn`` must be able to carry **magnitude**: the existing
encoder's final ``LayerNorm(latent_dim, elementwise_affine=False)`` projects every
residue onto a unit sphere (norm 11.3137 exactly), which freezes displacement
magnitude out of the representation.

Design:

  * ``StaticEncoder`` derives H_static from the *anchor frame only* (sequence
    identity embeddings + the anchor's within-frame canonical position), plus a
    small learned context head.  It is constant along a trajectory by construction
    and never touches future frames (leakage-safe).
  * ``DynamicEncoder`` is the existing geometry encoder with the magnitude-locking
    norm removed and an optional per-residue magnitude head added, so ‖H_dyn‖ can
    grow with real conformational displacement.  It is conditioned on H_static via
    an additive/concatenated modulation.
  * ``ConditionalStructureDecoder`` decodes ``D(H_static, H_dyn,t)``.  For the
    Stage A gate we reuse the existing delta-CA decoders (GeometricDelta / LatentDelta
    / BoundedEdge) but feed them an augmented input that carries the static context,
    so ablations can show whether H_dyn (vs H_static alone) drives the displacement.

These are *interfaces*; the GLE (Stage B) is not wired here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.encoder import (
    ProteinGeometryEncoder,
    _within_frame_canonical_positions,
)


class StaticEncoder(nn.Module):
    """Time-invariant identity + anchor-fold context.

    Inputs are the observed anchor backbone ``X_0`` and sequence metadata only.
    The output ``H_static`` is constant across a rollout (it has no time axis).

    Optional OpenFold-informed static branch (spec §1 / A1.2): when
    ``use_openfold`` is set, the forward also accepts ``openfold_single``
    (per-residue single representation, ``[B,L,ds]``) and ``openfold_pair``
    (pair representation, ``[B,L,L,dz]``) and projects them into the static
    context via a small learned head.  These features are computed *only* from
    permitted context (sequence and, where the forecast task includes it, the
    observed anchor) and are time-constant by construction, so they carry no
    future-frame leakage.  With ``use_openfold=False`` (the ablation baseline)
    the encoder is exactly the identity + anchor-frame canonical-position model.
    """

    def __init__(self, latent_dim: int = 128, max_residues: int = 4096,
                 use_openfold: bool = False, openfold_single_dim: int = 384,
                 openfold_pair_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_openfold = use_openfold
        self.aatype_embed = nn.Embedding(21, latent_dim)
        self.residue_embed = nn.Embedding(max_residues, latent_dim)
        self.chain_embed = nn.Embedding(128, latent_dim)
        # Anchor-frame position channel: where each residue sits in the anchor's
        # canonical PCA frame.  SE(3)-invariant and time-constant (anchor only).
        self.anchor_pos_proj = nn.Linear(3, latent_dim, bias=False)
        nn.init.normal_(self.anchor_pos_proj.weight, std=0.1)
        # OpenFold-informed static projections (only registered when use_openfold,
        # so the no-OpenFold ablation baseline keeps its exact checkpoint layout
        # and remains strict-loadable).
        self.openfold_single_proj = None
        self.openfold_pair_proj = None
        if use_openfold:
            self.openfold_single_proj = nn.Linear(openfold_single_dim, latent_dim, bias=False)
            self.openfold_pair_proj = nn.Linear(openfold_pair_dim, latent_dim, bias=False)
            nn.init.normal_(self.openfold_single_proj.weight, std=0.02)
            nn.init.normal_(self.openfold_pair_proj.weight, std=0.02)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, anchor_coords, residue_mask, aatype, residue_index, chain_id,
                openfold_single=None, openfold_pair=None):
        """anchor_coords [B,L,4,3]; returns H_static [B,L,D].

        ``openfold_single``/``openfold_pair`` are optional OpenFold-informed
        static features; they are ignored (and may be omitted) when
        ``use_openfold`` is False.
        """
        B, L = anchor_coords.shape[:2]
        aa = aatype.clamp(0, 20)
        res = residue_index.clamp(0, self.residue_embed.num_embeddings - 1)
        chain = chain_id.clamp(0, self.chain_embed.num_embeddings - 1)
        x = (
            self.aatype_embed(aa)
            + self.residue_embed(res)
            + self.chain_embed(chain)
        )
        # Anchor's own within-frame canonical position: centroid-relative CA in
        # the anchor PCA basis.  SE(3)-invariant and time-constant (anchor only).
        # _within_frame_canonical_positions takes [Bf, L, 4, 3] and [Bf, L].
        pos = _within_frame_canonical_positions(anchor_coords, residue_mask)  # [B,L,3]
        x = x + self.anchor_pos_proj(pos)
        if self.use_openfold:
            # Per-residue single rep + pair rep reduced to per-residue context.
            if openfold_single is not None:
                x = x + self.openfold_single_proj(openfold_single)
            if openfold_pair is not None:
                pair_sum = openfold_pair.sum(dim=2)  # [B,L,dz] -> sum over partners
                x = x + self.openfold_pair_proj(pair_sum)
        return self.head(x) * residue_mask.unsqueeze(-1).float()


class DynamicEncoder(nn.Module):
    """Magnitude-carrying conformational encoder, conditioned on H_static.

    Wraps a ``ProteinGeometryEncoder`` with the magnitude-locking LayerNorm
    removed, and conditions the base features on ``H_static``.  The output
    ``H_dyn`` is free to change magnitude with real displacement.
    """

    def __init__(self, latent_dim: int = 128, **encoder_kwargs):
        super().__init__()
        # Build the base geometry encoder WITHOUT the magnitude-locking norm.
        self.encoder = ProteinGeometryEncoder(latent_dim=latent_dim, **encoder_kwargs)
        # Replace the final norm with a NON-NORMALIZING projection.  LayerNorm
        # always divides by the per-residue std, pinning norm to sqrt(D) regardless
        # of its affine flag; that is the magnitude lock the Phase 0 audit found.
        # A plain learned linear (with a small init) lets H_dyn magnitude grow
        # with real displacement.
        self.encoder.norm = nn.Identity()
        # Static conditioning: project H_static into a per-residue modulation.
        self.static_proj = nn.Linear(latent_dim, latent_dim)
        self.norm = nn.Linear(latent_dim, latent_dim)
        nn.init.normal_(self.norm.weight, std=0.02)
        nn.init.zeros_(self.norm.bias)

    def forward(self, coords, residue_mask, aatype, residue_index, chain_id,
                h_static=None, token_mask=None, anchor_centroid=None, anchor_basis=None):
        # The base encoder expects residue_mask as [B,T,L] (or [B,1,L] per-frame)
        # while aatype/residue_index/chain_id are [B,L].  Accept both conventions:
        # expand a [B,L] mask to the coords time dim.
        if residue_mask.dim() == 2:
            T = coords.shape[1]
            residue_mask = residue_mask.unsqueeze(1).expand(-1, T, -1)
        # Encode with the (now magnitude-free) base encoder.
        h = self.encoder(
            coords, residue_mask, aatype, residue_index, chain_id,
            token_mask=token_mask, anchor_centroid=anchor_centroid, anchor_basis=anchor_basis,
        )
        if h_static is not None:
            # Condition on static context: additive per-residue modulation.
            # h_static may be [B,L,D] broadcast against [B,T,L,D].
            if h_static.dim() == 3:
                h_static = h_static.unsqueeze(1)
            h = h + self.static_proj(h_static)
        h = self.norm(h)
        # Re-apply the mask: the final linear has a bias that would otherwise
        # leak into padded/chain-break residues.
        h = h * residue_mask.unsqueeze(-1).float()
        return h


class ConditionalStructureDecoder(nn.Module):
    """Decoder contract D(H_static, H_dyn,t) -> CA.

    Wraps a delta-CA decoder, but first projects H_static and H_dyn into a
    *shared* latent space so the wrapped decoder's ``future_h - anchor_h`` delta
    means "conformational offset from the static reference" rather than the
    subtraction of two incompatible spaces.  The static context conditions
    identity/fold; the dynamic branch carries the magnitude that drives
    displacement.  This is the v4 contract X_hat = D(H_static, H_dyn,t).
    """

    def __init__(self, decoder: nn.Module, static_dim: int = 128, dynamic_dim: int = 128):
        super().__init__()
        self.decoder = decoder
        self.static_dim = static_dim
        # Project both branches into a shared conditioning space before the
        # wrapped delta-decoder computes future_h - anchor_h.
        self.static_proj = nn.Linear(static_dim, dynamic_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, dynamic_dim)
        nn.init.normal_(self.static_proj.weight, std=0.02)
        nn.init.normal_(self.dynamic_proj.weight, std=0.02)
        nn.init.zeros_(self.static_proj.bias)
        nn.init.zeros_(self.dynamic_proj.bias)

    def forward(self, h_dyn, h_static, aatype, residue_index, chain_id,
                rotation, anchor, mask):
        # h_dyn [B,L,D], h_static [B,L,D] -> shared space; delta = dyn - static.
        future_h = self.dynamic_proj(h_dyn)
        anchor_h = self.static_proj(h_static)
        return self.decoder(future_h, anchor_h, aatype, residue_index, chain_id,
                            rotation, anchor, mask)
