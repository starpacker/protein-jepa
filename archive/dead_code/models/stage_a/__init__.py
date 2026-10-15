"""Stage A: EMA-target JEPA with joint masked prediction and multi-horizon endpoints."""

from protein_jepa_frame.models.stage_a.model import StageAJEPA, StageAConfig
from protein_jepa_frame.models.stage_a.objectives import (
    stage_a_mask_loss,
    stage_a_horizon_loss,
    stage_a_total_loss,
)

__all__ = [
    "StageAJEPA",
    "StageAConfig",
    "stage_a_mask_loss",
    "stage_a_horizon_loss",
    "stage_a_total_loss",
]
