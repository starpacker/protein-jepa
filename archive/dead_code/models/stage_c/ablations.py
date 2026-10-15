"""P0.3: Ablation decoder variants for isolating Q/H contributions.

Three critical decoders:
  QZeroDecoder   — Q replaced by zeros; tests if H alone drives structure
  QShuffleDecoder — Q shuffled across batch; tests if Q semantics matter
  HZeroDecoder   — H replaced by zeros; tests if Q alone drives structure
  QNoiseDecoder  — Q with Gaussian noise; tests Q sensitivity
  QRandomDecoder — Q from random normal; tests if Q content matters

All share the same backbone decoder (StructureDecoder), only modifying inputs.

Decision table:
  Ablation  | If RMSD ≈ oracle | Implication
  ----------|-------------------|-------------
  Q=0       | Yes              | Q unused — decoder learned from H alone
  Q=0       | No (worse)       | Q genuinely contributes
  Q-shuffle | Yes ≈ oracle     | Q carries no residue-specific information
  Q-noise   | Yes ≈ oracle     | Q magnitude/direction unused
  Q-random  | Yes ≈ oracle     | Q completely unused by decoder
  H=0       | Yes              | H unused — decoder learned from Q alone
  H=0       | No (catastrophic)| H critical — geometry comes from frozen context
"""

from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_c.structure_decoder import StructureDecoder


class QZeroDecoder(nn.Module):
    """Decoder with Q replaced by zeros — tests if H alone drives structure."""

    def __init__(self, backbone_decoder: StructureDecoder):
        super().__init__()
        self.decoder = backbone_decoder

    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, residue_mask):
        q_zero = torch.zeros_like(q_original)
        return self.decoder(aatype, h_frozen, q_zero, R_anchor, t_anchor, residue_mask)


class QShuffleDecoder(nn.Module):
    """Decoder with Q shuffled across batch — tests if Q semantics matter.

    With batch_size=1, shuffle is identity; this is primarily useful when
    evaluating with batch_size >= 4.
    """

    def __init__(self, backbone_decoder: StructureDecoder):
        super().__init__()
        self.decoder = backbone_decoder

    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, residue_mask):
        B = q_original.shape[0]
        if B > 1:
            idx = torch.randperm(B, device=q_original.device)
            q_shuffled = q_original[idx]
        else:
            q_shuffled = q_original
        return self.decoder(aatype, h_frozen, q_shuffled, R_anchor, t_anchor, residue_mask)


class HZeroDecoder(nn.Module):
    """Decoder with H replaced by zeros — tests if Q alone drives structure."""

    def __init__(self, backbone_decoder: StructureDecoder):
        super().__init__()
        self.decoder = backbone_decoder

    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, residue_mask):
        h_zero = torch.zeros_like(h_frozen)
        return self.decoder(aatype, h_zero, q_original, R_anchor, t_anchor, residue_mask)


class QNoiseDecoder(nn.Module):
    """Decoder with Q + Gaussian noise (σ=1.0) — tests Q sensitivity."""

    def __init__(self, backbone_decoder: StructureDecoder, noise_std: float = 1.0):
        super().__init__()
        self.decoder = backbone_decoder
        self.noise_std = noise_std

    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, residue_mask):
        q_noisy = q_original + torch.randn_like(q_original) * self.noise_std
        return self.decoder(aatype, h_frozen, q_noisy, R_anchor, t_anchor, residue_mask)


class QRandomDecoder(nn.Module):
    """Decoder with Q replaced by random normal — tests if Q content matters at all."""

    def __init__(self, backbone_decoder: StructureDecoder):
        super().__init__()
        self.decoder = backbone_decoder

    def forward(self, aatype, h_frozen, q_original, R_anchor, t_anchor, residue_mask):
        q_random = torch.randn_like(q_original)
        return self.decoder(aatype, h_frozen, q_random, R_anchor, t_anchor, residue_mask)


__all__ = [
    "QZeroDecoder",
    "QShuffleDecoder",
    "HZeroDecoder",
    "QNoiseDecoder",
    "QRandomDecoder",
]
