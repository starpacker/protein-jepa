"""Stage B2: shared structure decoding and autonomous rollout.

Two autonomous decode modes (§3.6–§3.7 of STAGE_AB_REDESIGN_PLAN.md):

  h_only      — D_χ(H_t, c)         the primary decoder, identical to B1a oracle
  h_plus_bath — D_χ'(H_t, S_t, c)  comparison decoder that also receives S_t

The B1b dynamics kernel (StageB1b) is not owned here; it is passed in already
constructed and frozen so that B2 only trains its own decoder parameters.
"""

from protein_jepa_frame.models.stage_b2.bath_aware_decoder import (
    BathAwareDecoderConfig,
    BathAwareDecoder,
    AutonomousRolloutOutput,
    autonomous_rollout_and_decode,
)

__all__ = [
    "BathAwareDecoderConfig",
    "BathAwareDecoder",
    "AutonomousRolloutOutput",
    "autonomous_rollout_and_decode",
]
