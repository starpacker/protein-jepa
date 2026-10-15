"""Stage C utilities needed by Stage B1a.

Only these four files are live; everything else has been archived.
"""

from protein_jepa_frame.models.stage_c.geometry import (
    RigidBodyFrame,
    InvariantPointAttention,
    rotation_matrix_from_quaternion,
    quaternion_from_rotation_matrix,
)

from protein_jepa_frame.models.stage_c.structure_decoder import (
    DecoderConfig,
    StructureDecoder,
    NUM_TORSIONS,
    ATOM37_DIM,
)

from protein_jepa_frame.models.stage_c.objectives import (
    fape_loss,
    frame_aligned_fape,
    torsion_loss,
    pairwise_distance_loss,
    stage_c_total_loss,
)

from protein_jepa_frame.models.stage_c.openfold_atom37 import (
    frames_to_atom37,
    frames_to_atom37_from_backbone_tensors,
    decoder_frames_to_atom37,
)

__all__ = [
    "RigidBodyFrame",
    "InvariantPointAttention",
    "rotation_matrix_from_quaternion",
    "quaternion_from_rotation_matrix",
    "DecoderConfig",
    "StructureDecoder",
    "NUM_TORSIONS",
    "ATOM37_DIM",
    "fape_loss",
    "frame_aligned_fape",
    "torsion_loss",
    "pairwise_distance_loss",
    "stage_c_total_loss",
    "frames_to_atom37",
    "frames_to_atom37_from_backbone_tensors",
    "decoder_frames_to_atom37",
]
