"""Stage B2 autonomous decode modes: D_chi(H_t, c) vs D_chi'(H_t, S_t, c).

Implements §3.7 of STAGE_AB_REDESIGN_PLAN.md.  This file is ADDITIVE: it does
not modify or replace the primary decoder path (§3.1–§3.6).  It adds a second,
independently parametrised decoder that additionally receives the bath state
S_t, and a shared autonomous rollout that can decode with either.

Design contract
---------------
The B1b kernel is the single source of the (H, S) transition.  This module calls
``kernel.sample_step`` directly, so training/deployment kernel consistency
(plan §8.3 test 7) is preserved by construction — no second copy of D/C/Gamma.

Decode modes
------------
  "h_only"      D_chi (H_t, c)          primary; also the B1a oracle decoder.
                 H_t is routed into BOTH the context and the q pathway
                 (matches StageBOracleDecoder.decode_structure).

  "h_plus_bath" D_chi'(H_t, S_t, c)     comparison only (§3.7).
                 H_t -> context pathway; S_t -> q pathway (q_dim = dS).
                 S_t comes strictly from the kernel recursion; no history,
                 anchor, or basin bypass is added.

Red lines respected (§11)
-------------------------
  - The primary decoder is unchanged; D_chi' is a separate module with its own
    parameters, selectable at evaluation time.
  - S_t is never fabricated for the direct mode; this rollout is autonomous only.
  - D_chi' adds exactly ONE extra input (S_t).  The geometry generation layer
    (OpenFold frames/torsions/atom37) is identical to §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_c.structure_decoder import (
    DecoderConfig,
    StructureDecoder,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BathAwareDecoderConfig:
    """Sizing for the two B2 decoders.

    ``latent_dim`` is dH, ``bath_dim`` is dS.  The bath decoder routes S_t
    through the decoder's q pathway, so ``q_dim == bath_dim``.
    """
    latent_dim: int = 128           # dH
    bath_dim: int = 8               # dS

    c_hidden: int = 128
    n_ipa_blocks: int = 4
    n_heads: int = 4
    torsion_bins: int = 36

    # Translation budgets.  These mirror the B1a oracle budget so that D_chi
    # and D_chi' are compared on equal footing (only the input differs).
    max_translation_update: float = 8.0
    max_q_translation_update: float = 12.0
    max_com_update: float = 20.0
    direct_h_position: bool = True
    max_direct_position_update: float = 25.0

    def h_only_decoder_config(self) -> DecoderConfig:
        return DecoderConfig(
            context_dim=self.latent_dim,
            q_dim=self.latent_dim,          # H into q pathway (B1a convention)
            c_hidden=self.c_hidden,
            n_ipa_blocks=self.n_ipa_blocks,
            n_heads=self.n_heads,
            torsion_bins=self.torsion_bins,
            max_translation_update=self.max_translation_update,
            max_q_translation_update=self.max_q_translation_update,
            max_com_update=self.max_com_update,
            direct_h_position=self.direct_h_position,
            max_direct_position_update=self.max_direct_position_update,
            freeze_ab=True,
        )

    def bath_decoder_config(self) -> DecoderConfig:
        return DecoderConfig(
            context_dim=self.latent_dim,
            q_dim=self.bath_dim,            # S into q pathway
            c_hidden=self.c_hidden,
            n_ipa_blocks=self.n_ipa_blocks,
            n_heads=self.n_heads,
            torsion_bins=self.torsion_bins,
            max_translation_update=self.max_translation_update,
            max_q_translation_update=self.max_q_translation_update,
            max_com_update=self.max_com_update,
            direct_h_position=self.direct_h_position,
            max_direct_position_update=self.max_direct_position_update,
            freeze_ab=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Decoder wrappers with a unified call signature
# ─────────────────────────────────────────────────────────────────────────────

class HOnlyDecoder(nn.Module):
    """D_chi(H_t, c): the primary decoder path (unchanged semantics).

    Owns its own StructureDecoder.  The optional ``S_t`` argument is accepted
    only so both decoders share one call signature; it is IGNORED here.
    """

    mode = "h_only"

    def __init__(self, cfg: BathAwareDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decoder = StructureDecoder(cfg.h_only_decoder_config())

    def decode(
        self,
        H: torch.Tensor,                 # [B, L, dH]
        S: torch.Tensor | None,          # ignored
        aatype: torch.Tensor,            # [B, L]
        R_anchor: torch.Tensor,          # [B, L, 3, 3]
        t_anchor: torch.Tensor,          # [B, L, 3]
        residue_mask: torch.Tensor,      # [B, L]
        return_dict: bool = True,
        **kwargs,
    ):
        return self.decoder(
            aatype=aatype,
            h_frozen=H,
            q_dynamics=H,                # H into both pathways (B1a convention)
            R_anchor=R_anchor,
            t_anchor=t_anchor,
            residue_mask=residue_mask,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)


class BathAwareDecoder(nn.Module):
    """D_chi'(H_t, S_t, c): comparison decoder that also receives S_t.

    H_t -> context pathway; S_t -> q pathway.  Independently parametrised from
    HOnlyDecoder — training one never touches the other.
    """

    mode = "h_plus_bath"

    def __init__(self, cfg: BathAwareDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decoder = StructureDecoder(cfg.bath_decoder_config())

    def decode(
        self,
        H: torch.Tensor,                 # [B, L, dH]
        S: torch.Tensor,                 # [B, L, dS]
        aatype: torch.Tensor,
        R_anchor: torch.Tensor,
        t_anchor: torch.Tensor,
        residue_mask: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        if S is None:
            raise ValueError("BathAwareDecoder requires S_t; got None")
        if S.shape[-1] != self.cfg.bath_dim:
            raise ValueError(
                f"S_t last dim must be dS={self.cfg.bath_dim}, got {S.shape[-1]}"
            )
        return self.decoder(
            aatype=aatype,
            h_frozen=H,
            q_dynamics=S,                # S_t enters here
            R_anchor=R_anchor,
            t_anchor=t_anchor,
            residue_mask=residue_mask,
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self.decode(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Autonomous rollout + decode
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutonomousRolloutOutput:
    """Result of an autonomous rollout.

    H:   [B, n_steps+1, L, dH]  latent trajectory including H0 at index 0
    S:   [B, n_steps+1, L, dS]  bath trajectory (None when dS == 0)
    coords: {mode_name: [B, n_steps+1, L, 4, 3]} decoded backbone per mode
    outputs:{mode_name: list[dict]} raw decoder output dicts
    """
    H: torch.Tensor
    S: torch.Tensor | None
    coords: dict[str, torch.Tensor]
    outputs: dict[str, list]


def _init_bath(kernel, H0, residue_mask, temperature_k, H_hist, dt_hist,
               S0, generator):
    """Initialise S_0 exactly as StageB1b.rollout does (sample, not mean)."""
    cfg = kernel.cfg
    if not cfg.use_bath:
        return None
    if S0 is not None:
        return S0
    dS = cfg.bath_dim
    B, L = H0.shape[:2]
    dev, dtype = H0.device, H0.dtype
    if H_hist is None:
        m0 = torch.zeros(B, L, dS, device=dev, dtype=dtype)
        chol0 = torch.eye(dS, device=dev, dtype=dtype).reshape(1, 1, dS, dS).expand(
            B, L, dS, dS
        ).contiguous()
    else:
        m0, chol0 = kernel.history_posterior(H_hist, dt_hist, temperature_k, residue_mask)
    eps = torch.randn(m0.shape, device=dev, dtype=dtype, generator=generator)
    return m0 + torch.einsum("blij,blj->bli", chol0, eps)


@torch.no_grad()
def autonomous_rollout_and_decode(
    kernel,                              # StageB1b (frozen for B2)
    decoders: dict[str, nn.Module],      # {"h_only": ..., "h_plus_bath": ...}
    H0: torch.Tensor,                    # [B, L, dH] last observed H
    aatype: torch.Tensor,                # [B, L]
    R_anchor: torch.Tensor,              # [B, L, 3, 3]
    t_anchor: torch.Tensor,              # [B, L, 3]
    residue_mask: torch.Tensor,          # [B, L]
    n_steps: int,
    temperature_k=None,
    H_hist: torch.Tensor | None = None,  # [B, K, L, dH]
    dt_hist: torch.Tensor | None = None, # [B, K]
    seq_feature: torch.Tensor | None = None,
    S0: torch.Tensor | None = None,
    generator=None,
    decode_modes: tuple[str, ...] = ("h_only", "h_plus_bath"),
) -> AutonomousRolloutOutput:
    """Roll out (H, S) with the B1b kernel, decoding each step in each mode.

    The transition is ``kernel.sample_step`` — the SAME kernel used by the NLL
    and by deployment, so mode consistency holds by construction.  S_t fed to
    the bath decoder is the kernel's own recursion output; nothing else.

    Args:
        kernel:   a StageB1b instance whose parameters are frozen for B2.
        decoders: mapping mode name -> decoder module.  A mode is only run if
                  it is both present here and listed in ``decode_modes``.
        H0:       starting latent state (last observed frame H).
        n_steps:  number of Euler steps to take (horizon = n_steps * h).
        seq_feature: static (T, c)-conditioning feature held fixed over the
                  rollout (§2.2 static conditions).  Defaults to H0.
    """
    cfg = kernel.cfg
    B, L = H0.shape[:2]
    if seq_feature is None:
        seq_feature = H0

    S = _init_bath(kernel, H0, residue_mask, temperature_k, H_hist, dt_hist,
                   S0, generator)

    H_traj = [H0]
    S_traj = [S] if S is not None else None
    outputs: dict[str, list] = {m: [] for m in decode_modes}
    coords: dict[str, list] = {m: [] for m in decode_modes}

    def _decode_step(H, S):
        for m in decode_modes:
            if m not in decoders:
                raise KeyError(f"decode mode {m!r} requested but no decoder given")
            out = decoders[m].decode(
                H=H, S=S, aatype=aatype, R_anchor=R_anchor,
                t_anchor=t_anchor, residue_mask=residue_mask,
            )
            outputs[m].append(out)
            coords[m].append(out["backbone_coords"])

    # Decode the initial state too, so trajectories are length n_steps+1.
    _decode_step(H0, S)

    H = H0
    for _ in range(n_steps):
        H, S = kernel.sample_step(
            H, S, residue_mask, temperature_k, seq_feature, generator,
        )
        H_traj.append(H)
        if S is not None and S_traj is not None:
            S_traj.append(S)
        _decode_step(H, S)

    H_stacked = torch.stack(H_traj, dim=1)
    S_stacked = torch.stack(S_traj, dim=1) if S_traj is not None else None
    coords_out = {m: torch.stack(coords[m], dim=1) for m in decode_modes}
    return AutonomousRolloutOutput(
        H=H_stacked, S=S_stacked, coords=coords_out, outputs=outputs,
    )


__all__ = [
    "BathAwareDecoderConfig",
    "HOnlyDecoder",
    "BathAwareDecoder",
    "AutonomousRolloutOutput",
    "autonomous_rollout_and_decode",
]
