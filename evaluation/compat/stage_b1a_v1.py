"""Stage B1a: representation calibration.

Two sub-phases (STAGE_ABC_REDESIGN_PLAN.md §2.0):

  1. Frozen encoder baseline
     Train oracle decoder D_χ(H_t, c) and physical-future probe, encoder frozen.

  2. Controlled unfreezing (optional)
     Unfreeze top N encoder blocks + adapter, encoder lr ≪ decoder lr.
     Anchor loss keeps H close to frozen-A reference.

This module owns:
  - StageBOracleConfig: all knobs for B1a
  - StageBOracleDecoder: oracle decoder + physical probe + anchor head,
    wrapping StructureDecoder with H-as-context and H-as-state
  - freeze_encoder_partial(): helper to selectively unfreeze top blocks
  - normalize_H(): per-channel normalization using train-set statistics

Contract:
  - Encoder parameters may only be touched here; never inside B1b.
  - After B1a completes, call freeze_for_b1b() to permanently lock the encoder
    and register the H normalization statistics as frozen buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.compat.structure_decoder_v1 import (
    DecoderConfig,
    StructureDecoder,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageBOracleConfig:
    # Latent dimensions
    latent_dim: int = 128           # dH; encoder output width
    aux_dim: int = 64               # width of physical-future probe hidden layer

    # Physical probe targets (per residue).  Side-chain torsions remain
    # disabled until atom14 labels are available.
    num_distance_pairs: int = 20    # top-k CA distance pairs to predict
    num_torsion_outputs: int = 3    # φ,ψ,ω as sin/cos pairs → 6 scalars
    predict_contacts: bool = True   # binary contact (<8Å) per pair

    # Controlled unfreezing
    unfreeze_top_n_blocks: int = 0  # 0 = frozen baseline; 1-2 for fine-tune phase
    encoder_lr_scale: float = 0.1   # encoder lr = base_lr * encoder_lr_scale

    # IPA decoder size (matched to StructureDecoder defaults)
    decoder_c_hidden: int = 128
    decoder_n_ipa_blocks: int = 4
    decoder_n_heads: int = 4
    decoder_torsion_bins: int = 36

    # Translation budget for the oracle decoder.  B1a is an *oracle*: it is
    # handed the TRUE future H, so it must be able to reach the true future
    # structure.  Measured per-residue internal (Kabsch-aligned) CA displacement
    # over 8-48 ns horizons is 3.4-4.8 Å mean with per-sample means up to ~18 Å.
    # The Stage-C defaults (0.5 Å local + 2.0 Å Q = 2.5 Å total) are far too
    # tight for an oracle and floor FAPE/RMSD regardless of H quality, so we
    # open the budget here.  This does NOT make the forecasting task easier
    # because future H is given; it only removes an artificial decoder cap.
    decoder_max_translation_update: float = 8.0
    decoder_max_q_translation_update: float = 12.0
    decoder_max_com_update: float = 20.0
    # Enable the direct H-conditioned position head so the oracle decoder can
    # actually place CA where H says it is, instead of being stuck at anchor.
    decoder_direct_h_position: bool = True
    decoder_max_direct_position_update: float = 25.0

    @property
    def torsion_output_dim(self) -> int:
        return self.num_torsion_outputs * 2  # sin + cos per angle


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel H normalizer (train-set statistics)
# ─────────────────────────────────────────────────────────────────────────────

class HNormalizer(nn.Module):
    """Per-channel standardization of H using train-set mean/std.

    Statistics are registered as non-parameter buffers and must be computed
    once from train data before B1b starts (call fit()).  The floor on std
    prevents division by zero on inactive channels.
    """

    MIN_STD: float = 1e-3

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("mean_", torch.zeros(latent_dim))
        self.register_buffer("std_", torch.ones(latent_dim))
        self.register_buffer("fitted_", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, h_samples: torch.Tensor) -> None:
        """Fit from a representative H tensor [..., dH] collected over the training set."""
        flat = h_samples.reshape(-1, self.latent_dim).float()
        self.mean_.copy_(flat.mean(dim=0))
        self.std_.copy_(flat.std(dim=0).clamp(min=self.MIN_STD))
        self.fitted_.fill_(True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Standardize H: [..., dH] → [..., dH].  Returns h unchanged if not fitted."""
        if not self.fitted_.item():
            return h
        return (h - self.mean_) / self.std_

    def inverse(self, h_norm: torch.Tensor) -> torch.Tensor:
        return h_norm * self.std_ + self.mean_


# ─────────────────────────────────────────────────────────────────────────────
# Physical future probe
# ─────────────────────────────────────────────────────────────────────────────

class PhysicalFutureProbe(nn.Module):
    """Lightweight MLP predicting observable physical quantities from H.

    Predicts per-residue torsion sin/cos and per-pair distances/contacts
    from H_{t+τ}.  This is only used to check H info-content; it is frozen
    after B1a and reused by B1b/B2 as the evaluation Φ.

    All inputs are normalized H; all outputs are directly supervised.
    """

    def __init__(self, cfg: StageBOracleConfig) -> None:
        super().__init__()
        dH = cfg.latent_dim
        hid = cfg.aux_dim
        n_pairs = cfg.num_distance_pairs

        # Torsion head: per-residue
        self.torsion_head = nn.Sequential(
            nn.Linear(dH, hid), nn.GELU(), nn.Linear(hid, cfg.torsion_output_dim)
        )
        # Distance head: per-pair, using concat of two residue H vectors
        self.dist_head = nn.Sequential(
            nn.Linear(dH * 2, hid), nn.GELU(), nn.Linear(hid, 1)
        )
        # Contact head (binary, same input)
        self.contact_head = nn.Sequential(
            nn.Linear(dH * 2, hid), nn.GELU(), nn.Linear(hid, 1)
        )
        self.n_pairs = n_pairs
        self._predict_contacts = cfg.predict_contacts

        nn.init.zeros_(self.torsion_head[-1].weight)
        nn.init.zeros_(self.torsion_head[-1].bias)
        nn.init.zeros_(self.dist_head[-1].weight)
        nn.init.zeros_(self.dist_head[-1].bias)

    def forward(
        self,
        H: torch.Tensor,              # [B, L, dH]  normalised future H
        pair_idx: torch.Tensor,       # [B, n_pairs, 2]  residue index pairs
        valid_mask: torch.Tensor,     # [B, L]
    ) -> dict[str, torch.Tensor]:
        B, L, D = H.shape

        torsion_pred = self.torsion_head(H)   # [B, L, 6] for phi/psi/omega sin/cos

        # Gather pair features
        i_idx = pair_idx[:, :, 0]    # [B, n_pairs]
        j_idx = pair_idx[:, :, 1]    # [B, n_pairs]
        H_i = H[torch.arange(B, device=H.device)[:, None], i_idx]  # [B, n_pairs, D]
        H_j = H[torch.arange(B, device=H.device)[:, None], j_idx]
        pair_feat = torch.cat([H_i, H_j], dim=-1)  # [B, n_pairs, 2D]

        dist_pred = self.dist_head(pair_feat).squeeze(-1)     # [B, n_pairs]
        contact_logit = self.contact_head(pair_feat).squeeze(-1)  # [B, n_pairs]

        return {
            "torsion_pred": torsion_pred,
            "dist_pred": dist_pred,
            "contact_logit": contact_logit,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main B1a model
# ─────────────────────────────────────────────────────────────────────────────

class StageBOracleDecoder(nn.Module):
    """Oracle decoder + physical probe for Stage B1a calibration.

    Wraps StructureDecoder so it accepts H as both context (h_frozen) and
    dynamics variable (q_dynamics).  The StructureDecoder expects:
        h_frozen:    [B, L, context_dim]
        q_dynamics:  [B, L, q_dim]

    We pass the same H for both when context_dim == q_dim == dH.

    Also contains:
      - PhysicalFutureProbe: to check H info-content on future frames
      - Anchor head: optional MSE toward a frozen Stage A H reference
      - HNormalizer: for per-channel standardization (fit before B1b)
    """

    def __init__(self, config: StageBOracleConfig) -> None:
        super().__init__()
        self.config = config
        dH = config.latent_dim

        # Structure decoder reused from Stage C, reconfigured for H-as-state
        decoder_cfg = DecoderConfig(
            context_dim=dH,
            q_dim=dH,
            c_hidden=config.decoder_c_hidden,
            n_ipa_blocks=config.decoder_n_ipa_blocks,
            n_heads=config.decoder_n_heads,
            torsion_bins=config.decoder_torsion_bins,
            max_translation_update=config.decoder_max_translation_update,
            max_q_translation_update=config.decoder_max_q_translation_update,
            max_com_update=config.decoder_max_com_update,
            direct_h_position=config.decoder_direct_h_position,
            max_direct_position_update=config.decoder_max_direct_position_update,
            freeze_ab=True,
        )
        self.decoder = StructureDecoder(decoder_cfg)

        # Physical future probe (frozen after B1a)
        self.probe = PhysicalFutureProbe(config)

        # Anchor: soft L2 distillation from frozen Stage A H output
        # (no trainable params — computed externally as a loss term)

        # H normalizer
        self.h_normalizer = HNormalizer(dH)

    def decode_structure(
        self,
        H: torch.Tensor,          # [B, L, dH]  H at target time (normalised or raw)
        aatype: torch.Tensor,     # [B, L]
        R_anchor: torch.Tensor,   # [B, L, 3, 3]  last observed frame rotations
        t_anchor: torch.Tensor,   # [B, L, 3]     last observed frame translations
        residue_mask: torch.Tensor,  # [B, L]
    ) -> dict[str, torch.Tensor]:
        """Decode structure from H.  H is used as both context and dynamics."""
        return self.decoder(
            aatype=aatype,
            h_frozen=H,        # context_dim pathway
            q_dynamics=H,      # q_dim pathway — same H as state
            R_anchor=R_anchor,
            t_anchor=t_anchor,
            residue_mask=residue_mask,
        )

    def probe_future(
        self,
        H_future: torch.Tensor,   # [B, L, dH]
        pair_idx: torch.Tensor,   # [B, n_pairs, 2]
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run physical future probe on H at target time."""
        H_norm = self.h_normalizer(H_future)
        return self.probe(H_norm, pair_idx, valid_mask)

    def freeze_for_b1b(self) -> None:
        """Permanently freeze the H normalizer and decoder after B1a.

        Call after H normalizer has been fitted and B1a training is done.
        This is a one-way operation — B1b encoder must be separately frozen
        by the caller (encoder is not owned here).
        """
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        for p in self.probe.parameters():
            p.requires_grad_(False)
        self.h_normalizer.requires_grad_(False)


# ─────────────────────────────────────────────────────────────────────────────
# Controlled-unfreezing helper
# ─────────────────────────────────────────────────────────────────────────────

def freeze_encoder_partial(
    encoder: nn.Module,
    unfreeze_top_n_blocks: int,
) -> list[nn.Parameter]:
    """Freeze all encoder params except the top N attention blocks.

    Returns the list of now-trainable parameters for the optimizer group.

    Args:
        encoder: ProteinGeometryEncoder instance.
        unfreeze_top_n_blocks: number of blocks from the end to keep trainable.
            0 = fully frozen (returns empty list).
    """
    # Freeze everything first
    for p in encoder.parameters():
        p.requires_grad_(False)

    if unfreeze_top_n_blocks <= 0:
        return []

    trainable: list[nn.Parameter] = []

    # Unfreeze top N blocks from encoder.layers
    if hasattr(encoder, "layers") and isinstance(encoder.layers, nn.ModuleList):
        n = len(encoder.layers)
        for i, block in enumerate(encoder.layers):
            if i >= n - unfreeze_top_n_blocks:
                for p in block.parameters():
                    p.requires_grad_(True)
                    trainable.append(p)

    # Always unfreeze the final LayerNorm
    if hasattr(encoder, "norm"):
        for p in encoder.norm.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    return trainable
