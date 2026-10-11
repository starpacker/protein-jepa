"""Stage C: Structure decoding from predicted latent states.

Subpackage organisation:
  geometry.py          — SE(3) equivariance primitives, rigid frames, IPA
  structure_decoder.py — Equivariant IPA decoder with torsion prediction
  objectives.py        — FAPE, torsion, distance, bond, clash losses
  direct_head.py       — Direct endpoint prediction head (Gate 6)
  rollout.py           — Autonomous recursive rollout (Gate 6)
"""

from protein_jepa_frame.models.stage_c.geometry import (
    RigidBodyFrame,
    InvariantPointAttention,
    rotation_matrix_from_quaternion,
    quaternion_from_rotation_matrix,
    test_se3_equivariance,
)

from protein_jepa_frame.models.stage_c.structure_decoder import (
    DecoderConfig,
    StructureDecoder,
    FrameUpdate,
    TorsionHead,
    ContactGraphBuilder,
    reconstruct_backbone_from_torsions,
    reconstruct_atom37,
    NUM_TORSIONS,
    ATOM37_DIM,
)

from protein_jepa_frame.models.stage_c.objectives import (
    fape_loss,
    frame_aligned_fape,
    torsion_loss,
    pairwise_distance_loss,
    bond_length_violation,
    clash_penalty,
    stage_c_total_loss,
)

from protein_jepa_frame.models.stage_c.direct_head import (
    DirectHeadConfig,
    DirectEndpointHead,
)

from protein_jepa_frame.models.stage_c.rollout import (
    RolloutConfig,
    AutonomousRollout,
    rollout_schedule,
)

from protein_jepa_frame.models.stage_c.readout import (
    ReadoutConfig,
    PhysicalObservableReadout,
    readout_distance_loss,
    readout_contact_loss,
    readout_rmsf_loss,
    readout_total_loss,
    compute_target_contacts,
    compute_target_rmsf,
)

from protein_jepa_frame.models.stage_c.model import (
    StageCModel,
)

__all__ = [
    "StageCModel",
    "RigidBodyFrame",
    "InvariantPointAttention",
    "rotation_matrix_from_quaternion",
    "quaternion_from_rotation_matrix",
    "test_se3_equivariance",
    "DecoderConfig",
    "StructureDecoder",
    "FrameUpdate",
    "TorsionHead",
    "ContactGraphBuilder",
    "reconstruct_backbone_from_torsions",
    "reconstruct_atom37",
    "NUM_TORSIONS",
    "ATOM37_DIM",
    "fape_loss",
    "frame_aligned_fape",
    "torsion_loss",
    "pairwise_distance_loss",
    "bond_length_violation",
    "clash_penalty",
    "stage_c_total_loss",
    "DirectHeadConfig",
    "DirectEndpointHead",
    "RolloutConfig",
    "AutonomousRollout",
    "rollout_schedule",
    "ReadoutConfig",
    "PhysicalObservableReadout",
    "readout_distance_loss",
    "readout_contact_loss",
    "readout_rmsf_loss",
    "readout_total_loss",
    "compute_target_contacts",
    "compute_target_rmsf",
]