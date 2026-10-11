"""LSP-WM backward-compatibility shim.

All canonical implementations now live in the ``protein_jepa_frame.models.lasp``
subpackage. This module re-exports them so that existing checkpoint loading
paths and import statements continue to work without modification.

New code should import directly from ``protein_jepa_frame.models.lasp``.
"""

from protein_jepa_frame.models.lasp import (
    # utils
    _as_batch_temperature,
    # energy
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
    # diffusion
    ConstantDiffusion,
    StateDependentDiffusion,
    DiagonalDiffusion,
    # bath
    MultiTimescaleBath,
    GaugeProjector,
    ResidueConditionalDecoder,
    GaugeFixedBath,
    ProjectedForceClosure,
    # projection
    UnifiedLatentRoleReadout,
    msm_consistency_loss,
    # objectives
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
    _estimate_kde_bandwidth,
    fit_koopman_operator,
    koopman_diagnostics,
    PoissonOperator,
    DissipationOperator,
    # integrator
    BoundedConservativePotential,
    PhysicalTransitionCalibration,
    LSPWMConfig,
    LSPWMWorldModel,
    # protein_bridge
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
    "_estimate_kde_bandwidth",
    "fit_koopman_operator",
    "koopman_diagnostics",
    "PoissonOperator",
    "DissipationOperator",
    "BoundedConservativePotential",
    "PhysicalTransitionCalibration",
    "LSPWMConfig",
    "LSPWMWorldModel",
    "LSPWMProteinWorldModel",
]
