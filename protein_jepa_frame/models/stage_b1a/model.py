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

from protein_jepa_frame.models.stage_c.structure_decoder import (
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
    decoder_dropout: float = 0.1

    # Anchor-local displacement budget for the oracle decoder.  B1a is an
    # *oracle*: it is handed the TRUE future H, so it must be able to reach the
    # true future internal conformation.  Measured per-residue internal
    # (Kabsch-aligned) CA displacement over 8-48 ns horizons is 3.4-4.8 Å mean
    # with per-sample means up to ~18 Å; the budget must exceed that or FAPE/RMSD
    # floors regardless of H quality.  The decoder's ONLY positional path is
    # this anchor-local displacement (X_hat = R_anchor @ d_local + t_anchor);
    # see STAGE_AB_REDESIGN_PLAN.md §2.0.0.
    decoder_max_local_position_update: float = 12.0
    # Global-context pooling over the state path in the decoder's single
    # representation.  Held-out ablations of this channel were done on a
    # same-protein split and are NOT yet shown to pass a clean protein-level
    # gate (see H_DECODABILITY_REPORT.md §3); it is kept because it is sound in
    # principle (per-residue H alone carries no protein-global reference).
    decoder_global_h_context: bool = True

    # ── Anchor-free positional decoding ──────────────────────────────────────
    # When True the decoder no longer adds a bounded per-residue displacement to
    # a copy of the anchor structure.  Instead H predicts the whole internal CA
    # cloud in a single canonical frame and the anchor (R_anchor, t_anchor) is
    # applied only as a post-hoc rigid placement (see StructureDecoder).  This
    # removes the structural shortcut that made real_H ≈ zero_H: with the anchor
    # supplying the base geometry, H only had to emit ~0 and the positional
    # gradient through H vanished.
    decoder_anchor_free: bool = True
    decoder_coord_scale: float = 40.0

    @property
    def torsion_output_dim(self) -> int:
        return self.num_torsion_outputs * 2  # sin + cos per angle

    @classmethod
    def from_saved(cls, config_dict: dict | None) -> "StageBOracleConfig":
        """Build from a saved config dict, dropping unknown/legacy keys.

        Old B1a checkpoints recorded decoder knobs that have since been removed
        (decoder_max_q_translation_update, decoder_max_com_update,
        decoder_direct_h_position, decoder_h_relative_position*, ...).  Those
        are silently dropped so the HNormalizer/probe can still be loaded from
        a legacy checkpoint.
        """
        if not config_dict:
            return cls()
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in valid})


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
        if flat.shape[0] == 0 or not torch.isfinite(flat).all():
            raise ValueError("normalizer requires finite, nonempty training H")
        self.mean_.copy_(flat.mean(dim=0))
        self.std_.copy_(flat.std(dim=0, correction=0).clamp(min=self.MIN_STD))
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

    Wraps StructureDecoder, which has a SINGLE state pathway:

        q_dynamics: [B, L, dH]   invariant state Q := H_t

    plus the equivariant rigid-body basis ``anchor = (R_anchor, t_anchor)``
    that supplies absolute placement only.  Decoding is

        X_hat = anchor_frame  x  d_local(Q := H_t, c)

    (STAGE_AB_REDESIGN_PLAN.md §2.0.0).  There is no separate h_frozen context
    pathway — H_t IS the dynamics state on the mainline.

    Also contains:
      - PhysicalFutureProbe: to check H info-content on future frames
      - Anchor head: optional MSE toward a frozen Stage A H reference
      - HNormalizer: for per-channel standardization (fit before B1b)
    """

    def __init__(self, config: StageBOracleConfig) -> None:
        super().__init__()
        self.config = config
        dH = config.latent_dim

        # Structure decoder reused from Stage C, reconfigured for H-as-state.
        # One state pathway (q_dim = dH = context), no h_frozen context pathway.
        decoder_cfg = DecoderConfig(
            q_dim=dH,
            c_hidden=config.decoder_c_hidden,
            n_ipa_blocks=config.decoder_n_ipa_blocks,
            n_heads=config.decoder_n_heads,
            torsion_bins=config.decoder_torsion_bins,
            dropout=config.decoder_dropout,
            max_local_position_update=config.decoder_max_local_position_update,
            freeze_ab=True,
            global_h_context=getattr(config, "decoder_global_h_context", True),
            anchor_free=getattr(config, "decoder_anchor_free", True),
            coord_scale=getattr(config, "decoder_coord_scale", 40.0),
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
        H: torch.Tensor,          # [B, L, dH] raw encoder H only
        aatype: torch.Tensor,     # [B, L]
        R_anchor: torch.Tensor,   # [B, L, 3, 3]  last observed frame rotations
        t_anchor: torch.Tensor,   # [B, L, 3]     last observed frame translations
        residue_mask: torch.Tensor,  # [B, L]
    ) -> dict[str, torch.Tensor]:
        """Decode structure from H (the invariant state) + anchor basis.

        ``R_anchor/t_anchor`` are the last observed frame's rigid-body basis;
        they fix absolute placement only.  The internal conformation
        ``d_local`` is predicted from ``H`` alone.
        """
        if H.shape[-1] != self.config.latent_dim:
            raise ValueError("decoder expects raw H with the bundle latent dimension")
        return self.decoder(
            aatype=aatype,
            q_dynamics=H,      # single state pathway: Q := H_t
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

    @torch.no_grad()
    def refit_normalizer(self, raw_h_samples: torch.Tensor, preserve_probe: bool = True) -> None:
        """Refit train statistics, preserving the probe's raw-H mapping exactly.

        A changed encoder needs new dynamics coordinates. The decoder consumes
        raw H and is unaffected; adjust each probe input affine map to avoid
        invalidating a calibrated probe when its normalization changes.
        """
        old_mean = self.h_normalizer.mean_.clone()
        old_std = self.h_normalizer.std_.clone()
        fitted = bool(self.h_normalizer.fitted_)
        self.h_normalizer.fit(raw_h_samples)
        if preserve_probe and fitted:
            ratio = self.h_normalizer.std_ / old_std
            offset = (self.h_normalizer.mean_ - old_mean) / old_std
            for layer, repeats in ((self.probe.torsion_head[0], 1),
                                   (self.probe.dist_head[0], 2),
                                   (self.probe.contact_head[0], 2)):
                original = layer.weight.clone()
                layer.bias.add_(original @ offset.repeat(repeats))
                layer.weight.mul_(ratio.repeat(repeats)[None])

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
        self.eval()


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
