"""T3 — Model-side ensemble producer (NEW, 2026-10-18).

The forecast task is under-determined at the single-frame level (see
`STAGE_A_REPAIR_RESULTS_20261018.md` §9–§10), so we model the *ensemble* of
plausible futures rather than one absolute frame.  This module wraps the
retained, frozen pieces (Stage-A geometry encoder + B1a anchored decoder) and
adds ONE small new, differentiable head:

    H_anchor = Encoder(context_frames)[-1]              (frozen)
    mu, log_sigma = GaussianHHead(H_anchor)             (new, trainable)
    H^{(n)} = mu + sigma * eps_n,  eps_n ~ N(0, I)      (N samples)
    X^{(n)}_CA = Decoder(H^{(n)}, anchor basis)         (frozen, anchored)

The N decoded CA clouds are the model ensemble; they feed the ensemble-level
losses in `protein_jepa_frame/training/ensemble_losses.py`.

NOTHING here modifies the precise-atom-coordinate decoding path
(`stage_c/structure_decoder.py`, `stage_b1a/*`).  The decoder is used exactly
as-is; only a new latent head is introduced.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GaussianHHead", "EnsembleProducer"]


class GaussianHHead(nn.Module):
    """Per-residue latent Gaussian head: H_anchor -> (mu, log_sigma) in H space.

    Small residual MLP; ``log_sigma`` is bounded so sampling stays stable.

    Optional per-sample conditioning (D1, 2026-10-19): if ``cond_dim > 0`` the
    head receives an extra broadcast vector (e.g. normalized log horizon time,
    temperature) concatenated to every residue's H_anchor.  This lets the same
    frozen stack express "how much flexibility to expect" as a function of the
    forecast timescale / temperature.  ``cond_dim=0`` reproduces the original
    head exactly.
    """

    def __init__(self, dim: int, hidden: int = 128, sigma_min: float = 1e-3,
                 sigma_max: float = 5.0, cond_dim: int = 0):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.cond_dim = int(cond_dim)
        in_dim = dim + self.cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * dim),
        )
        # Start deterministic-ish (small sigma) so early training is stable.
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[dim:] = -4.0      # log_sigma init (sigma ~ 0.02)

    def _merge_cond(self, h_anchor: torch.Tensor, cond: torch.Tensor | None):
        if self.cond_dim == 0:
            return h_anchor
        if cond is None:
            raise ValueError("GaussianHHead built with cond_dim>0; cond is required")
        if cond.shape[0] != h_anchor.shape[0]:
            raise ValueError(f"cond batch {cond.shape[0]} != H batch {h_anchor.shape[0]}")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"cond last dim {cond.shape[-1]} != cond_dim {self.cond_dim}")
        c = cond.unsqueeze(1).expand(-1, h_anchor.shape[1], -1)   # [B, L, c]
        return torch.cat([h_anchor, c], dim=-1)

    def forward(self, h_anchor: torch.Tensor,
                cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_sigma = self.net(self._merge_cond(h_anchor, cond)).chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(min=torch.log(torch.tensor(self.sigma_min)).item(),
                                    max=torch.log(torch.tensor(self.sigma_max)).item())
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def sample(self, h_anchor: torch.Tensor, n: int, generator=None,
               eps: torch.Tensor | None = None,
               cond: torch.Tensor | None = None) -> torch.Tensor:
        """[B, L, dH] -> [B, n, L, dH] samples around mu with per-residue sigma."""
        mu, sigma = self.forward(h_anchor, cond=cond)
        if eps is None:
            eps = torch.randn(mu.shape[0], n, mu.shape[1], mu.shape[2],
                              device=mu.device, dtype=mu.dtype, generator=generator)
        return mu.unsqueeze(1) + sigma.unsqueeze(1) * eps


class EnsembleProducer(nn.Module):
    """Frozen encoder + trainable Gaussian H head + frozen anchored decoder.

    Parameters
    ----------
    encoder : ProteinGeometryEncoder (frozen)
    decoder : StageBOracleDecoder (frozen) exposing decode_structure(H, aatype,
              R_anchor, t_anchor, residue_mask)
    head    : GaussianHHead
    """

    def __init__(self, encoder, decoder, head: GaussianHHead):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_frame(self, coords, residue_mask, aatype, residue_index, chain_id):
        """[B, L, 4, 3] -> H [B, L, dH] (no grad; encoder is frozen)."""
        c5 = coords.unsqueeze(1)
        m3 = residue_mask.unsqueeze(1)
        H = self.encoder(c5, m3, aatype, residue_index, chain_id, token_mask=m3)
        return H[:, 0]

    def sample_future_ca(
        self, context_coords, residue_mask, aatype, residue_index, chain_id,
        R_anchor, t_anchor, n_samples: int, generator=None,
        eps: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
    ) -> dict:
        """Produce an ensemble of future CA clouds from the anchor.

        Args:
            context_coords: [B, K, L, 4, 3] context frames (anchor = last one).
            cond:           optional [B, c] per-sample conditioning (D1).
        Returns:
            dict(ca_samples [B, n, L, 3], ca_mean [B, L, 3],
                 ca_mean_decoded [B, L, 3] (frozen decoder applied to mu),
                 backbone_samples [B, n, L, 4, 3] (N, CA, C, O; OpenFold),
                 atom37_samples [B, n, L, 37, 3] (OpenFold ordering),
                 torsion_samples [B, n, L, 7] (phi, psi, omega, chi1..chi4),
                 H_samples, mu, sigma)

        The ``*_samples`` backbone/atom37/torsion outputs are the *same* OpenFold
        reconstruction the frozen decoder already produced internally
        (``out["backbone_direct"]``/``out["atom37_coords"]``/``out["torsions"]``);
        they are surfaced here so evaluation/export can score a full mainchain
        instead of only Cα.  The CA column of ``backbone_samples`` equals
        ``ca_samples`` exactly (both are ``backbone_coords[:, :, 1, :]``).
        """
        B, K, L, _, _ = context_coords.shape
        # Encode the anchor frame as the H-anchor (frozen encoder).
        anchor_coords = context_coords[:, -1]                     # [B, L, 4, 3]
        H_anchor = self.encode_frame(anchor_coords, residue_mask, aatype,
                                     residue_index, chain_id)     # [B, L, dH]
        H_samples = self.head.sample(H_anchor, n_samples, generator=generator,
                                     eps=eps, cond=cond)
        mu, sigma = self.head.forward(H_anchor, cond=cond)

        flat_H = H_samples.reshape(B * n_samples, L, H_samples.shape[-1])
        flat_R = R_anchor[:, None].expand(B, n_samples, L, 3, 3).reshape(B * n_samples, L, 3, 3)
        flat_t = t_anchor[:, None].expand(B, n_samples, L, 3).reshape(B * n_samples, L, 3)
        flat_aa = aatype[:, None].expand(B, n_samples, L).reshape(B * n_samples, L)
        flat_mask = residue_mask[:, None].expand(B, n_samples, L).reshape(B * n_samples, L)
        out = self.decoder.decode_structure(flat_H, flat_aa, flat_R, flat_t, flat_mask)
        ca = out["backbone_coords"][:, :, 1, :].reshape(B, n_samples, L, 3)
        # Full mainchain (N, CA, C, O) + full heavy-atom atom37, both produced by
        # the frozen decoder's OFFICIAL OpenFold reconstruction path.  These extra
        # keys are surfaced only when the decoder actually exposes them, so a
        # minimal decoder (returning just ``backbone_coords``) keeps working and
        # existing consumers of ``ca_samples`` are never broken.
        backbone = out["backbone_coords"].reshape(B, n_samples, L, 4, 3)
        # Deterministic ("mode") decode: frozen decoder applied to the mean H.
        out_mu = self.decoder.decode_structure(mu, aatype, R_anchor, t_anchor, residue_mask)
        ca_mu = out_mu["backbone_coords"][:, :, 1, :]
        result = {
            "ca_samples": ca,
            "ca_mean": ca.mean(dim=1),
            "ca_mean_decoded": ca_mu,
            "backbone_samples": backbone,
            "H_samples": H_samples,
            "mu": mu,
            "sigma": sigma,
        }
        if "atom37_coords" in out:
            result["atom37_samples"] = out["atom37_coords"].reshape(B, n_samples, L, -1, 3)
        if "atom37_mask" in out:
            result["atom37_mask"] = out["atom37_mask"].reshape(B, n_samples, L, -1)
        if "torsions" in out:
            result["torsion_samples"] = out["torsions"].reshape(B, n_samples, L, -1)
        return result
