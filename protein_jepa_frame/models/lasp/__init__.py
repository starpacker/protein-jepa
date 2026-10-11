"""Latent Statistical Physics (LaSP) — canonical Stage B/C primitives.

Subpackage organisation follows the audit plan (STAGE_C_CODE_AUDIT_AND_IMPLEMENTATION_PLAN.md):

  lasp/
    utils.py          — _as_batch_temperature
    energy.py         — scalar potentials (MeanForcePotential, ResidueMessagePassingPotential,
                         ReversiblePotential, UnifiedEnergyGraph, SkewSymmetricOperator)
    diffusion.py      — diffusion tensors (ConstantDiffusion, StateDependentDiffusion, DiagonalDiffusion)
    bath.py           — Mori bath and gauge structures
    integrator.py     — GLE integration (LSPWMWorldModel, step, step_midpoint, rollout)
    objectives.py     — losses (KDE, transition NLL, FDT, cycle, Koopman)
    projection.py     — latent projection / role readout
    protein_bridge.py — LSPWMProteinWorldModel (host bridge)
"""

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature

from protein_jepa_frame.models.lasp.energy import (
    MeanForcePotential,
    ResidueMessagePassingPotential,
    ReversiblePotential,
    SkewSymmetricOperator,
    UnifiedEnergyGraph,
    sliced_score_matching_loss,
    residue_force_direction_loss,
    force_score_G_loss,
    force_score_Q_loss,
    self_consistency_loss,
    stationary_distribution_loss,
)

from protein_jepa_frame.models.lasp.diffusion import (
    ConstantDiffusion,
    StateDependentDiffusion,
    DiagonalDiffusion,
)

from protein_jepa_frame.models.lasp.bath import (
    MultiTimescaleBath,
    GaugeProjector,
    ResidueConditionalDecoder,
    GaugeFixedBath,
    ProjectedForceClosure,
)

from protein_jepa_frame.models.lasp.projection import (
    UnifiedLatentRoleReadout,
    msm_consistency_loss,
)

from protein_jepa_frame.models.lasp.objectives import (
    _estimate_kde_bandwidth,
    kde_mean_force,
    mean_force_matching_loss,
    calibrated_mean_force_matching_loss,
    _gaussian_transition_nll_from_parameters,
    overdamped_transition_nll,
    mori_transition_nll,
    calibrated_overdamped_transition_nll,
    fdt_residual,
    empirical_fdt_residual,
    energy_loop_circulation,
    FlowMatchObservationModel,
    asymmetric_cycle_loss,
    local_density_weights,
    vamp2_loss,
    fit_koopman_operator,
    koopman_diagnostics,
)

from protein_jepa_frame.models.lasp.integrator import (
    BoundedConservativePotential,
    PhysicalTransitionCalibration,
    LSPWMConfig,
    LSPWMWorldModel,
    PoissonOperator,
    DissipationOperator,
)

from protein_jepa_frame.models.lasp.protein_bridge import (
    LSPWMProteinWorldModel,
)

__all__ = [
    "_as_batch_temperature",
    "MeanForcePotential",
    "ResidueMessagePassingPotential",
    "ReversiblePotential",
    "SkewSymmetricOperator",
    "UnifiedEnergyGraph",
    "sliced_score_matching_loss",
    "residue_force_direction_loss",
    "force_score_G_loss",
    "force_score_Q_loss",
    "self_consistency_loss",
    "stationary_distribution_loss",
    "ConstantDiffusion",
    "StateDependentDiffusion",
    "DiagonalDiffusion",
    "MultiTimescaleBath",
    "GaugeProjector",
    "ResidueConditionalDecoder",
    "GaugeFixedBath",
    "ProjectedForceClosure",
    "UnifiedLatentRoleReadout",
    "msm_consistency_loss",
    "_estimate_kde_bandwidth",
    "kde_mean_force",
    "mean_force_matching_loss",
    "calibrated_mean_force_matching_loss",
    "_gaussian_transition_nll_from_parameters",
    "overdamped_transition_nll",
    "mori_transition_nll",
    "calibrated_overdamped_transition_nll",
    "fdt_residual",
    "empirical_fdt_residual",
    "energy_loop_circulation",
    "FlowMatchObservationModel",
    "asymmetric_cycle_loss",
    "local_density_weights",
    "vamp2_loss",
    "fit_koopman_operator",
    "koopman_diagnostics",
    "BoundedConservativePotential",
    "PhysicalTransitionCalibration",
    "LSPWMConfig",
    "LSPWMWorldModel",
    "PoissonOperator",
    "DissipationOperator",
    "LSPWMProteinWorldModel",
]
