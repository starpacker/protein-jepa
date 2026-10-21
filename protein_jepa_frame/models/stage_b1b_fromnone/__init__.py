"""From-none Stage B: joint (encoder H, GLE dynamics, CA decoder).

The Stage A encoder is deliberately NOT used here.  A fresh, randomly
initialised ProteinGeometryEncoder is co-trained end-to-end with the H/S
dynamics kernel (StageB1b) and a CA-displacement decoder, so that the
inter-frame change in H is *forced* to decode into inter-frame CA motion.
This is the "from-none" escape from the diagnosis that a frozen Stage A H is
a same-frame invariant descriptor whose difference carries no geometric
information (ridge R^2 < 0).

Modules:
  FromNoneConfig      : sizing for the decoder head.
  SimpleCADecoder     : lightweight per-residue MLP displacement head.
  MessagePassingCADecoder : residue-graph self-attention head (H only).
  CRIDecoder          : cross-residue + H<->S interactive head (recommended).
"""
from protein_jepa_frame.models.stage_b1b_fromnone.decoder import (
    FromNoneConfig,
    SimpleCADecoder,
)
from protein_jepa_frame.models.stage_b1b_fromnone.mp_decoder import (
    CRIDecoder,
    MessagePassingCADecoder,
    MessagePassingConfig,
)

__all__ = [
    "FromNoneConfig",
    "SimpleCADecoder",
    "MessagePassingConfig",
    "MessagePassingCADecoder",
    "CRIDecoder",
]
