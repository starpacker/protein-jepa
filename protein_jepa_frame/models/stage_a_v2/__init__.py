"""Stage A V2: clean JEPA-Anything-aligned JEPA for protein dynamics.

Uses ProteinGeometryEncoder (online + EMA teacher) with OPF projectors and a
causal Transformer predictor.  Wired to jepa_anything_core losses directly.
"""

from protein_jepa_frame.models.stage_a_v2.model import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_a_v2.objectives import (
    StageAV2LossWeights,
    compute_stage_a_v2_loss,
)

__all__ = [
    "StageAV2",
    "StageAV2Config",
    "StageAV2LossWeights",
    "compute_stage_a_v2_loss",
]
