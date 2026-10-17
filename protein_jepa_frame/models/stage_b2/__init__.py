"""Stage B2: shared structure decoding and autonomous rollout.

Two autonomous decode modes (§3.6–§3.7 of STAGE_AB_REDESIGN_PLAN.md):

  h_only      — D_χ(Q:=H_t, anchor, c)          the primary decoder (B1a oracle)
  h_plus_bath — D_χ'(Q:=H_t, S_t, anchor, c)   comparison decoder, extra S_t

Both share the invariant x equivariant decomposition (plan §2.0.0):
``X_hat = anchor_frame  x  d_local(Q:=H_t, c [, S_t])``; anchor supplies the
rigid-body basis only.  The two decoders differ ONLY by the extra S_t input.

The B1b dynamics kernel (StageB1b) is not owned here; it is passed in already
constructed and frozen so that B2 only trains its own decoder parameters.
"""

from protein_jepa_frame.models.stage_b2.bath_aware_decoder import (
    BathAwareDecoderConfig,
    HOnlyDecoder,
    BathAwareDecoder,
    AutonomousRolloutOutput,
    autonomous_rollout_and_decode,
)

__all__ = [
    "BathAwareDecoderConfig",
    "HOnlyDecoder",
    "BathAwareDecoder",
    "AutonomousRolloutOutput",
    "autonomous_rollout_and_decode",
]
