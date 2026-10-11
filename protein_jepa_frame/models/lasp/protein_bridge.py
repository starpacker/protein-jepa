"""Protein bridge module for LSP-WM — connects JEPA encoder to LaSP dynamics.

Contains:
  - LSPWMProteinWorldModel: full protein bridge module
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.thermodynamic import (
    FixedBackboneCollectiveVariables,
    ProteinConditionEncoder,
    _last_valid,
)

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature
from protein_jepa_frame.models.lasp.energy import (
    MeanForcePotential,
    ResidueMessagePassingPotential,
    UnifiedEnergyGraph,
    sliced_score_matching_loss,
    residue_force_direction_loss,
    force_score_G_loss,
    force_score_Q_loss,
    self_consistency_loss,
    stationary_distribution_loss,
)
from protein_jepa_frame.models.lasp.diffusion import DiagonalDiffusion
from protein_jepa_frame.models.lasp.bath import (
    MultiTimescaleBath,
    GaugeProjector,
    ResidueConditionalDecoder,
)
from protein_jepa_frame.models.lasp.projection import (
    UnifiedLatentRoleReadout,
    msm_consistency_loss,
)
from protein_jepa_frame.models.lasp.objectives import (
    kde_mean_force,
    mean_force_matching_loss,
    overdamped_transition_nll,
    calibrated_overdamped_transition_nll,
    mori_transition_nll,
    asymmetric_cycle_loss,
    local_density_weights,
    vamp2_loss,
)
from protein_jepa_frame.models.lasp.integrator import LSPWMConfig, LSPWMWorldModel


class LSPWMProteinWorldModel(nn.Module):
    """Full protein bridge for the independent LSP-WM framework.

    The host JEPA supplies causal frame states. This module owns a unified
    coarse state and all coarse dynamics, so it can be trained and ablated
    without inheriting EVA5C's state-dependent diffusion or free-energy head.

    ``learned_cv_dim`` is retained as a checkpoint/configuration name for
    compatibility, but it now denotes the complete LSP-WM state dimension. The
    fixed physical CVs are supervision and diagnostics, never concatenated into
    the learned state.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        learned_cv_dim: int = 4,
        num_segments: int = 3,
        condition_dim: int = 64,
        auxiliary_dim: int = 4,
        reference_temperature_k: float = 348.0,
        diffusion_scale: float = 1.0e-6,
        relaxation_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0),
        coupling_scale: float = 1.0e-3,
        contact_cutoff_angstrom: float = 8.0,
        contact_width_angstrom: float = 0.75,
        cv_mean: tuple[float, ...] = (),
        cv_scale: tuple[float, ...] = (),
        mean_force_bandwidth: float = 1.5,
        history_closure_enabled: bool = False,
        observation_model: bool = False,
        observation_coord_dim: int = 0,
        koopman_role_dim: int = 64,
        msm_role_dim: int = 32,
        geometric_role_dim: int = 96,
        auxiliary_role_dim: int = 64,
        role_hidden_dim: int = 256,
        rollout_enabled: bool = False,
        rollout_steps: int = 4,
        rollout_ensemble_size: int = 4,
        rollout_warmup_steps: int = 0,
        rollout_intermediate_supervision: bool = True,
        rollout_gaussian_nll_enabled: bool = True,
        dynamic_nll_enabled: bool = True,
        force_loss_enabled: bool = True,
        state_dependent_diffusion_enabled: bool = False,
        potential_hidden_dim: int = 128,
        potential_hidden_depth: int = 4,
        physical_calibration_enabled: bool = False,
        physical_mobility_init: float = 1.0,
        physical_free_energy_init: float = 1.0,
        physical_log_scale_min: float = -12.0,
        physical_log_scale_max: float = 12.0,
        conservative_correction_enabled: bool = False,
        conservative_correction_rank: int = 16,
        conservative_correction_scale: float = 1.0,
        conservative_correction_rescale_with_free_energy: bool = False,
        freeze_base_dynamics: bool = False,
        freeze_calibration_scales: bool = False,
        # Rollout / curriculum (v4.4)
        rollout_trajectory_steps: int = 0,
        semigroup_consistency_weight: float = 0.0,
        midpoint_integration: bool = False,
        # THM-JEPA: reversible (divergence-free) dynamics
        reversible_dynamics_enabled: bool = False,
        reversible_operator_rank: int = 16,
        reversible_hidden_dim: int = 128,
        reversible_hidden_depth: int = 4,
        # v28: per-residue dynamics
        per_residue_enabled: bool = False,
        per_residue_warmup_steps: int = 2000,  # V27: delay per-residue training
        residue_cv_dim: int = 8,
        residue_gnn_depth: int = 2,
        residue_gnn_hidden: int = 64,
        residue_force_direction_weight: float = 0.3,
        sliced_score_enabled: bool = False,
        sliced_score_noise: float = 0.05,
        sliced_score_projections: int = 4,
        energy_graph_enabled: bool = False,
        gauge_enabled: bool = False,
        gauge_dim: int = 6,
        bath_timescale_groups: tuple[int, int, int] = (4, 8, 4),
        bath_init_taus_ps: tuple[float, float, float] = (1000.0, 8000.0, 64000.0),
        residue_conditional_enabled: bool = False,
        energy_graph_num_rbf: int = 8,
        energy_graph_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or learned_cv_dim < 1:
            raise ValueError("LSP-WM needs a positive latent and learned-CV dimension")
        self.latent_dim = int(latent_dim)
        self.learned_cv_dim = int(learned_cv_dim)
        self.mean_force_bandwidth = float(mean_force_bandwidth)
        self.rollout_enabled = bool(rollout_enabled)
        self.rollout_steps = int(rollout_steps)
        self.rollout_ensemble_size = int(rollout_ensemble_size)
        self.rollout_warmup_steps = int(rollout_warmup_steps)
        self.rollout_intermediate_supervision = bool(
            rollout_intermediate_supervision
        )
        self.rollout_gaussian_nll_enabled = bool(rollout_gaussian_nll_enabled)
        self.dynamic_nll_enabled = bool(dynamic_nll_enabled)
        self.force_loss_enabled = bool(force_loss_enabled)
        self.state_dependent_diffusion_enabled = bool(state_dependent_diffusion_enabled)
        self.physical_calibration_enabled = bool(physical_calibration_enabled)
        self.freeze_base_dynamics = bool(freeze_base_dynamics)
        self.freeze_calibration_scales = bool(freeze_calibration_scales)
        self.collective_variables = FixedBackboneCollectiveVariables(
            num_segments=num_segments,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            contact_width_angstrom=contact_width_angstrom,
        )
        self.fixed_cv_dim = self.collective_variables.output_dim
        if cv_mean and len(cv_mean) != self.fixed_cv_dim:
            raise ValueError("LSP-WM cv_mean must match the fixed CV dimension")
        if cv_scale and len(cv_scale) != self.fixed_cv_dim:
            raise ValueError("LSP-WM cv_scale must match the fixed CV dimension")
        self.register_buffer(
            "cv_mean",
            torch.tensor(cv_mean or (0.0,) * self.fixed_cv_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "cv_scale",
            torch.tensor(cv_scale or (1.0,) * self.fixed_cv_dim, dtype=torch.float32),
        )
        self.learned_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, learned_cv_dim),
            nn.Tanh(),
        )
        self.target_learned_projection = copy.deepcopy(self.learned_projection)
        for parameter in self.target_learned_projection.parameters():
            parameter.requires_grad = False
        self.condition_encoder = ProteinConditionEncoder(condition_dim)
        self.role_readout = UnifiedLatentRoleReadout(
            learned_cv_dim,
            self.fixed_cv_dim,
            condition_dim,
            koopman_dim=koopman_role_dim,
            msm_dim=msm_role_dim,
            geometric_dim=geometric_role_dim,
            auxiliary_dim=auxiliary_role_dim,
            hidden_dim=role_hidden_dim,
        )
        self.world_model = LSPWMWorldModel(
            LSPWMConfig(
                state_dim=learned_cv_dim,
                condition_dim=condition_dim,
                auxiliary_dim=auxiliary_dim,
                reference_temperature_k=reference_temperature_k,
                diffusion_scale=diffusion_scale,
                relaxation_times_ps=relaxation_times_ps,
                coupling_scale=coupling_scale,
                history_feature_dim=(3 * learned_cv_dim if history_closure_enabled else 0),
                history_closure_hidden_dim=256,
                history_closure_max_shift=0.55,
                history_closure_horizon_gate_steps=2.0,
                observation_model=observation_model,
                observation_coord_dim=observation_coord_dim or learned_cv_dim,
                state_dependent_diffusion_enabled=state_dependent_diffusion_enabled,
                force_loss_enabled=force_loss_enabled,
                potential_hidden_dim=potential_hidden_dim,
                potential_hidden_depth=potential_hidden_depth,
                physical_calibration_enabled=physical_calibration_enabled,
                physical_mobility_init=physical_mobility_init,
                physical_free_energy_init=physical_free_energy_init,
                physical_log_scale_min=physical_log_scale_min,
                physical_log_scale_max=physical_log_scale_max,
                conservative_correction_enabled=conservative_correction_enabled,
                conservative_correction_rank=conservative_correction_rank,
                conservative_correction_scale=conservative_correction_scale,
                conservative_correction_rescale_with_free_energy=(
                    conservative_correction_rescale_with_free_energy
                ),
                midpoint_integration=midpoint_integration, # Rev-1: BAOAB path
                rollout_trajectory_steps=rollout_trajectory_steps, # Rev-2: multi-step NLL
                semigroup_consistency_weight=semigroup_consistency_weight, # Rev-4
                reversible_dynamics_enabled=reversible_dynamics_enabled,
                reversible_operator_rank=reversible_operator_rank,
                reversible_hidden_dim=reversible_hidden_dim,
                reversible_hidden_depth=reversible_hidden_depth,
            )
        )
        if self.freeze_base_dynamics:
            if not self.physical_calibration_enabled:
                raise ValueError(
                    "freeze_base_dynamics requires physical calibration"
                )
            self.freeze_base_dynamics_parameters()
        if self.freeze_calibration_scales:
            if not self.physical_calibration_enabled:
                raise ValueError(
                    "freeze_calibration_scales requires physical calibration"
                )
            self.world_model.physical_calibration.log_mobility_scale.requires_grad = False
            self.world_model.physical_calibration.log_free_energy_scale.requires_grad = False

        # ── Rev-3: Learned auxiliary state initializer ──────────────
        # Predicts the bath (memory) state from the last N context frames,
        # so multi-step rollouts start with a data-informed memory trace.
        self.auxiliary_initializer_context: int = 3
        self.auxiliary_initializer = nn.Sequential(
            nn.LayerNorm(learned_cv_dim * self.auxiliary_initializer_context),
            nn.Linear(learned_cv_dim * self.auxiliary_initializer_context, 128),
            nn.SiLU(),
            nn.Linear(128, auxiliary_dim),
        )

        # ── v28: Per-residue latent dynamics ──────────────────────────
        # Each residue gets its own CV state q_i ∈ R^{cv_dim}, bypassing
        # the global mean-pooling that destroys all spatial information.
        self.per_residue_enabled = bool(per_residue_enabled)
        self.per_residue_warmup_steps = int(per_residue_warmup_steps)
        self.residue_cv_dim = int(residue_cv_dim)
        self.residue_gnn_depth = int(residue_gnn_depth)
        self.residue_gnn_hidden = int(residue_gnn_hidden)
        self.residue_force_direction_weight = float(residue_force_direction_weight)
        self.sliced_score_enabled = bool(sliced_score_enabled)
        self.sliced_score_noise = float(sliced_score_noise)
        self.sliced_score_projections = int(sliced_score_projections)
        self.energy_graph_enabled = bool(energy_graph_enabled)
        self.gauge_enabled = bool(gauge_enabled)
        self.residue_conditional_enabled = bool(residue_conditional_enabled)

        if self.per_residue_enabled:
            # Project raw per-residue latents → per-residue CV state
            self.residue_projection = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, self.residue_cv_dim),
                nn.Tanh(),
            )
            # GNN-based per-residue potential energy
            self.residue_potential = ResidueMessagePassingPotential(
                cv_dim=self.residue_cv_dim,
                condition_dim=condition_dim,
                hidden_dim=self.residue_gnn_hidden,
                depth=self.residue_gnn_depth,
                reference_temperature_k=reference_temperature_k,
            )
            if self.energy_graph_enabled:
                self.unified_energy = UnifiedEnergyGraph(
                    cv_dim=self.residue_cv_dim, global_dim=self.residue_cv_dim,
                    condition_dim=condition_dim, num_rbf=energy_graph_num_rbf,
                    hidden_dim=energy_graph_hidden_dim,
                )
                self.multi_bath = MultiTimescaleBath(
                    state_dim=self.residue_cv_dim,
                    num_fast=bath_timescale_groups[0], num_mid=bath_timescale_groups[1],
                    num_slow=bath_timescale_groups[2], init_taus=bath_init_taus_ps,
                )
                self.gauge_projector = GaugeProjector(latent_dim, gauge_dim) if self.gauge_enabled else None
                self.residue_decoder = (ResidueConditionalDecoder(self.residue_cv_dim, self.residue_cv_dim)
                                        if self.residue_conditional_enabled else None)
                self.mori_diffusion = DiagonalDiffusion(
                    state_dim=self.residue_cv_dim,
                    condition_dim=condition_dim,
                    reference_temperature_k=reference_temperature_k,
                    diffusion_scale=5e-5,
                )
                # V25 FIX #2: Per-dimension learnable diffusion magnitude correction.
                # Raw scaled diffusion gives dissipative ~5e-5 vs dG ~0.007 (120x).
                # This parameter lets the optimizer learn the right per-dim scale.
                self.diffusion_scale_correction = nn.Parameter(
                    torch.zeros(self.residue_cv_dim)  # log-space, 0 → 1×
                )
                self._diffusion_scale_correction_scale = nn.Parameter(
                    torch.tensor(1.0)  # global scalar multiplier
                )
                # V25: History→bath aux projector for M1 ablation
                # auxiliary_initializer outputs 4D (config auxiliary_dim),
                # but MultiTimescaleBath expects sum(group_sizes) = 16D.
                # This MLP bridges the gap using history features.
                self.history_to_bath = nn.Sequential(
                    nn.LayerNorm(self.residue_cv_dim * 3),  # 3 history frames
                    nn.Linear(self.residue_cv_dim * 3, 64),
                    nn.SiLU(),
                    nn.Linear(64, sum(bath_timescale_groups)),
                    nn.Tanh(),
                )

    @property
    def state_dim(self) -> int:
        return self.learned_cv_dim

    # ── V25: Per-residue dynamics integration ───────────────────────────
    # Each residue has its own CV state q_i ∈ R^D. The per-residue step
    # applies the Mori-Zwanzig Langevin equation to each residue
    # independently, using:
    #   - Per-residue forces ∇_i A(Q,G) from UnifiedEnergyGraph or
    #     ResidueMessagePassingPotential (includes pair interactions)
    #   - Global auxiliary s (16D Mori bath) coupling uniformly to all
    #     residues via the same coupling matrix C
    #   - Per-residue noise with FDT-consistent covariance 2D·dt
    #
    # Update equation for residue i:
    #   q_i(t+dt) = q_i(t) + [-D · ∇_i A + C · s] · dt + √(2D·dt) · ξ_i
    #
    # where s is predicted from 3-frame CV history via history_to_bath MLP.
    def per_residue_step(
        self,
        q_per_res: torch.Tensor,        # [B, L, D]
        residue_force: torch.Tensor,     # [B, L, D]  pre-computed force
        residue_mask: torch.Tensor,      # [B, L]
        condition: torch.Tensor | None,  # [B, C]
        dt_ps: torch.Tensor | float,
        temperature_k: torch.Tensor | None,
        *,
        stochastic: bool = True,
    ) -> dict:
        """One step of per-residue overdamped Langevin + global Mori bath."""
        if q_per_res.ndim != 3 or q_per_res.shape[2] != self.residue_cv_dim:
            raise ValueError(
                f"per_residue_step: q_per_res must be [B, L, {self.residue_cv_dim}], "
                f"got {list(q_per_res.shape)}"
            )
        B, L, D = q_per_res.shape
        dt = torch.as_tensor(dt_ps, device=q_per_res.device, dtype=q_per_res.dtype)
        if dt.ndim == 0:
            dt = dt.expand(B)
        if dt.shape != (B,) or torch.any(dt <= 0):
            raise ValueError("dt_ps must be positive scalar or shape [batch]")

        temperature = _as_batch_temperature(
            temperature_k, B, q_per_res,
            self.world_model.config.reference_temperature_k,
        )

        # Pool global G from per-residue CVs for diffusion and aux prediction
        m = residue_mask.to(dtype=q_per_res.dtype)
        G_current = (q_per_res * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)  # [B, D]

        # Diffusion: per-dimension diagonal, function of global G
        diffusion = self.mori_diffusion(
            G_current, condition, temperature
        )  # [B, D, D] diagonal

        # Apply per-dim learnable scale correction
        if hasattr(self, 'diffusion_scale_correction'):
            corr = torch.exp(self.diffusion_scale_correction) * torch.relu(
                self._diffusion_scale_correction_scale
            )
            corr_diag = torch.diag_embed(corr.to(dtype=diffusion.dtype))  # [D, D]
            diffusion = diffusion @ corr_diag  # [B, D, D]

        # Bath coupling: same C for all residues
        taus, C_bath = self.multi_bath.parameters_at(
            dtype=q_per_res.dtype, device=q_per_res.device
        )  # C_bath: [D, A_global]
        rates = taus.reciprocal()  # [A_global]
        coupling = C_bath.unsqueeze(0).expand(B, -1, -1)  # [B, D, A_global]

        # Auxiliary: predict from 3-frame global CV history  →  [B, A_global]
        auxiliary = self._predict_per_residue_aux(q_per_res, residue_mask, G_current)
        A_global = auxiliary.shape[1]

        # Drift: -D ∇A + C s  (per-residue)
        # residue_force = -grad A (already the negative gradient, conservative force F)
        # So the correct drift is: +D · F  (not -D · F)
        # FDT: drift = -D ∇A = D · F where F = -∇A (the conservative force)
        dissipative = torch.einsum("bdd,bLd->bLd", diffusion, residue_force)  # [B, L, D]  P0-2 FIXED
        # C · s: [B, D, A] @ [B, A] → [B, D], broadcast → [B, L, D]
        bath_drift = torch.einsum("bda,ba->bd", coupling, auxiliary).unsqueeze(1)  # [B, 1, D]
        drift = dissipative + bath_drift  # [B, L, D]

        # Auxiliary evolution (OU process, same for all residues since aux is global)
        aux_drift = -auxiliary * rates.unsqueeze(0)  # [B, A_global]

        # Noise: FDT-consistent
        q_noise = q_per_res.new_zeros(B, L, D)
        aux_noise = auxiliary.new_zeros(B, A_global)
        if stochastic:
            q_cov = 2.0 * diffusion * dt[:, None, None]  # [B, D, D]
            # Cholesky: [B, D, D]; noise: [B, L, D]
            chol = torch.linalg.cholesky(q_cov)
            eps = torch.randn(B, L, D, device=q_per_res.device, dtype=q_per_res.dtype)
            q_noise = torch.einsum("bdd,bLd->bLd", chol, eps)

            aux_std = (2.0 * rates.unsqueeze(0) * dt[:, None]).sqrt()  # [B, A]
            aux_noise = torch.randn_like(auxiliary) * aux_std

        # Apply updates (masked — only valid residues evolve)
        dt_exp = dt[:, None, None]  # [B, 1, 1]
        next_q = q_per_res + drift * dt_exp + q_noise
        # Invalid residues stay frozen
        invalid = (~residue_mask.bool()).unsqueeze(-1)
        next_q = next_q.masked_scatter(invalid, q_per_res.masked_select(invalid))

        next_aux = auxiliary + aux_drift * dt[:, None] + aux_noise

        return {
            "q": next_q,               # [B, L, D]
            "auxiliary": next_aux,      # [B, A_global]
            "drift": drift,             # [B, L, D]
            "dissipative": dissipative,  # [B, L, D]
            "bath_drift": bath_drift,   # [B, 1, D]
            "diffusion": diffusion,     # [B, D, D]
            "coupling": coupling,       # [B, D, A_global]
            "rates": rates,             # [A_global]
            "temperature_k": temperature,
        }

    def _predict_per_residue_aux(
        self,
        q_per_res: torch.Tensor,   # [B, L, D]
        residue_mask: torch.Tensor, # [B, L]
        G_current: torch.Tensor,    # [B, D]
    ) -> torch.Tensor:
        """Predict global auxiliary (bath) state from per-residue CVs.

        Uses the 3-frame history (last, last-1, last-2) pooled to global G,
        fed through history_to_bath MLP. This breaks the force-bath degeneracy
        by using context frames rather than steady-state force alignment.
        """
        B, L, D = q_per_res.shape
        m = residue_mask.to(dtype=q_per_res.dtype)  # [B, L]
        # V27: Per-residue-aware 3-frame context for bath prediction.
        # Use the per-residue CVs themselves (not only the pooled G) to
        # estimate past and reference states from the spatial distribution.
        # Frame 0 (retrospective): mean of per-residue CVs with down-weight
        #   on high-variance residues (those in motion), biased toward origin.
        # Frame 1 (static reference): zero state (origin).
        # Frame 2 (present): pooled mean G_current.
        D_local = q_per_res.shape[2]  # residue_cv_dim, matches G_current
        m = residue_mask.to(dtype=q_per_res.dtype)  # [B, L]
        # Retrospective: per-residue squared norm → inverse-variance weight
        q_sq_norm = q_per_res.square().sum(dim=-1)  # [B, L]
        # Soft inverse: residues with larger displacement get lower weight
        inv_w = 1.0 / (1.0 + q_sq_norm.clamp(min=1e-8))  # [B, L]
        w = inv_w * m  # [B, L]
        G_retro = (q_per_res * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, D_local]
        G_static = q_per_res.new_zeros(B, D_local)
        # Ensure G_current is same dim as q_per_res features
        if G_current.shape[-1] != D_local:
            G_current = G_current[..., :D_local]
        frames = [G_retro, G_static, G_current]  # past → present trajectory
        aux_input = torch.cat(frames, dim=-1)  # [B, 3*D_local]
        aux_history = self.history_to_bath(aux_input) * 2.0  # [B, A_global]
        return aux_history

    # ── Per-residue transition NLL ────────────────────────────────────
    def per_residue_transition_nll(
        self,
        q_current: torch.Tensor,      # [B, L, D]
        q_next: torch.Tensor,          # [B, L, D] observed next state (detached)
        residue_force: torch.Tensor,   # [B, L, D]
        residue_mask: torch.Tensor,    # [B, L]
        condition: torch.Tensor | None,
        dt_ps: torch.Tensor | float,
        temperature_k: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gaussian NLL of observed per-residue displacement under model drift.

        Observed:  disp_i = q_next_i - q_current_i
        Predicted: pred_i = (-D·∇_i A + C·s) · dt
        Covariance: 2·D·dt  (FDT-consistent)
        """
        B, L, D = q_current.shape
        dt = torch.as_tensor(dt_ps, device=q_current.device, dtype=q_current.dtype)
        if dt.ndim == 0:
            dt = dt.expand(B)
        m = residue_mask.to(dtype=q_current.dtype)

        # Pool G for diffusion
        G = (q_current * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1.0)

        temperature = _as_batch_temperature(
            temperature_k, B, q_current,
            self.world_model.config.reference_temperature_k,
        )
        diffusion = self.mori_diffusion(G, condition, temperature)
        if hasattr(self, 'diffusion_scale_correction'):
            corr = torch.exp(self.diffusion_scale_correction) * torch.relu(
                self._diffusion_scale_correction_scale
            )
            diffusion = diffusion @ torch.diag_embed(corr.to(dtype=diffusion.dtype))

        # Coupling and auxiliary
        _, C_bath = self.multi_bath.parameters_at(
            dtype=q_current.dtype, device=q_current.device
        )
        coupling = C_bath.unsqueeze(0).expand(B, -1, -1)  # [B, D, A]
        auxiliary = self._predict_per_residue_aux(q_current, residue_mask, G)

        # Drift prediction: residue_force = -grad A = F (conservative force)
        # Correct drift: D · F (not -D · F)
        dissipative = torch.einsum("bdd,bLd->bLd", diffusion, residue_force)  # [B, L, D]  P0-2 FIXED
        bath = torch.einsum("bda,ba->bd", coupling, auxiliary).unsqueeze(1)     # [B, 1, D]
        drift = (dissipative + bath) * dt[:, None, None]  # [B, L, D]

        # Observed displacement
        disp = (q_next - q_current).detach()  # [B, L, D]

        # Covariance: 2 * D * dt  (per-dimension, shared across residues)
        # For diagonal D, each residue has the same D components.
        # diag(D) = [B, D]; cov_per_dim = 2 * diag(D) * dt  = [B, D]
        D_diag = diffusion.diagonal(dim1=-2, dim2=-1)  # [B, D]
        var = (2.0 * D_diag * dt[:, None]).clamp_min(1e-8)  # [B, D]
        log_var = var.log()  # [B, D]

        # NLL: 0.5 * Σ_d [(disp_d - drift_d)² / var_d + log(var_d) + log(2π)]
        residual_sq = (disp - drift).square()  # [B, L, D]
        nll_per_res = 0.5 * (residual_sq / var.unsqueeze(1) + log_var.unsqueeze(1) + math.log(2 * math.pi))
        # V27: No clamp — gradients must flow to energy graph.
        # Weight and warmup ramp in train.py control the contribution.
        # Average over valid residues and dimensions
        nll = (nll_per_res.sum(dim=-1) * m).sum() / m.sum().clamp_min(1.0)
        return nll

    def freeze_base_dynamics_parameters(self) -> None:
        """Freeze every LSP-WM parameter except the physical calibration head."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.world_model.physical_calibration.parameters():
            parameter.requires_grad = True

    def calibration_parameters(self) -> list[nn.Parameter]:
        if not self.physical_calibration_enabled:
            return []
        return [
            parameter
            for parameter in self.world_model.physical_calibration.parameters()
            if parameter.requires_grad
        ]

    @torch.no_grad()
    def update_target_projection(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("target projection momentum must be in [0, 1]")
        for target, online in zip(
            self.target_learned_projection.parameters(),
            self.learned_projection.parameters(),
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)

    def init_auxiliary_from_context(
        self, context_sequence: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict initial auxiliary (bath) state from recent context frames.

        Rev-3: Instead of always starting the Mori bath at zero, this method
        learns to infer the latent memory state from the last N context
        frame states (projected through ``learned_projection``).

        Args:
            context_sequence: [batch, seq_len, latent_dim] frame states.
            context_mask: [batch, seq_len] boolean mask (True = valid).

        Returns:
            auxiliary_init: [batch, auxiliary_dim] predicted bath state.
        """
        # Project latent frame states to learned CV space
        with torch.set_grad_enabled(self.training):
            q = self.learned_projection(context_sequence)
        ctx = self.auxiliary_initializer_context
        if context_mask is not None:
            # Use the last `ctx` valid frames per batch element
            count = context_mask.long().sum(dim=1)  # [batch]
            batch_idx = torch.arange(q.shape[0], device=q.device)
            frames = []
            for offset in range(ctx):
                idx = (count - 1 - offset).clamp(min=0).long()
                frames.append(q[batch_idx, idx, :])
            input_tensor = torch.cat(frames, dim=-1)  # [batch, ctx * d_cv]
        else:
            input_tensor = q[:, -ctx:, :].reshape(q.shape[0], -1)
        return self.auxiliary_initializer(input_tensor)

    def encode_fixed_state(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        chain_id: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> torch.Tensor:
        fixed = self.collective_variables(
            coords,
            atom_mask,
            residue_mask,
            chain_id,
            residue_index,
        )
        return (fixed - self.cv_mean.to(fixed)) / self.cv_scale.to(fixed).clamp(min=1.0e-6)

    def encode_state(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        chain_id: torch.Tensor,
        residue_index: torch.Tensor,
        frame_states: torch.Tensor,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        projector = self.target_learned_projection if target else self.learned_projection
        return projector(frame_states)

    @staticmethod
    def history_features(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.long().sum(dim=1)
        index = (count - 1).clamp(min=0)
        previous_index = (count - 2).clamp(min=0)
        rows = torch.arange(sequence.shape[0], device=sequence.device)
        last = sequence[rows, index]
        previous = sequence[rows, previous_index]
        rank = mask.long().cumsum(dim=1)
        recent = mask & (rank > (count[:, None] - 4).clamp(min=0))
        recent_mean = (sequence * recent.unsqueeze(-1)).sum(dim=1) / recent.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        global_mean = (sequence * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
        return torch.cat([last - recent_mean, recent_mean - global_mean, last - previous], dim=-1)

    def forward(
        self,
        *,
        context_coords: torch.Tensor,
        future_coords: torch.Tensor,
        context_atom_mask: torch.Tensor,
        future_atom_mask: torch.Tensor,
        context_residue_mask: torch.Tensor,
        future_residue_mask: torch.Tensor,
        context_chain_id: torch.Tensor,
        future_chain_id: torch.Tensor,
        context_residue_index: torch.Tensor,
        future_residue_index: torch.Tensor,
        context_frame_states: torch.Tensor,
        future_frame_states: torch.Tensor,
        context_time_ps: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        aatype: torch.Tensor,
        temperature_k: torch.Tensor | None,
        dataset_index: torch.Tensor | None,
        curriculum_step: int | None = None,
        curriculum_steps: tuple[int, ...] | None = None,
        rollout_steps: int | None = None,
        rollout_ensemble_size: int | None = None,
        # v28 per-residue optional inputs
        context_latents: torch.Tensor | None = None,
        future_latents: torch.Tensor | None = None,
        context_ca_coords: torch.Tensor | None = None,
        future_ca_coords: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_fixed = self.encode_fixed_state(
            context_coords,
            context_atom_mask,
            context_residue_mask,
            context_chain_id,
            context_residue_index,
        )
        future_fixed = self.encode_fixed_state(
            future_coords,
            future_atom_mask,
            future_residue_mask,
            future_chain_id,
            future_residue_index,
        )
        context_q = self.encode_state(
            context_coords,
            context_atom_mask,
            context_residue_mask,
            context_chain_id,
            context_residue_index,
            context_frame_states,
        )
        with torch.no_grad():
            future_q = self.encode_state(
                future_coords,
                future_atom_mask,
                future_residue_mask,
                future_chain_id,
                future_residue_index,
                future_frame_states,
                target=True,
            )
        context_mask = context_residue_mask.any(dim=2)
        future_mask = future_residue_mask.any(dim=2)
        condition = self.condition_encoder(
            aatype,
            context_residue_index,
            context_residue_mask.any(dim=1),
            dataset_index,
        )
        temperature = _as_batch_temperature(
            temperature_k,
            context_q.shape[0],
            context_q,
            self.world_model.config.reference_temperature_k,
        )
        current = _last_valid(context_q, context_mask)
        target = _last_valid(future_q, future_mask).detach()
        current_fixed = _last_valid(context_fixed, context_mask)
        target_fixed = _last_valid(future_fixed, future_mask).detach()
        history = self.history_features(context_q, context_mask)
        reference = context_q[context_mask].detach()

        outputs = {
            "lspwm_context_state": current,
            "lspwm_target_state": target,
            "lspwm_context_sequence": context_q,
            "lspwm_context_mask": context_mask,
            "lspwm_condition": condition,
            "lspwm_temperature_k": temperature,
            "lspwm_history_features": history,
            "lspwm_context_fixed_cv": current_fixed,
            "lspwm_target_fixed_cv": target_fixed,
        }

        # Compute phase-aware canonical losses (Ch.7.2)
        # Use the protein-world-model's method which looks at curriculum_step and config thresholds
        effective_step = curriculum_step if curriculum_step is not None else 5000
        phase = self.get_active_phase(effective_step, curriculum_steps)

        # I. Representation loss (L_repr) — handled by JEPA predictor, not here
        # II. Koopman loss (L_koop) — VAMP-2 on current/target (always on)
        # III. Thermodynamic identification (L_force + L_dyn)
        # IV. Rollout self-consistency (multi-step)
        # V. Observation model (L_obs)

        # Force loss (L_force) — PMF mean force matching
        # Active in: thermodynamic_identification, dynamics, rollout
        is_force_active = phase not in ("representation",) and (
            self.force_loss_enabled or self.dynamic_nll_enabled
        )
        if is_force_active:
            if reference.shape[0] >= 2:
                mean_force_target = kde_mean_force(
                    current.detach(),
                    reference,
                    self.mean_force_bandwidth,
                )
                outputs["lspwm_mean_force_target"] = mean_force_target
                force_loss = mean_force_matching_loss(
                    self.world_model.potential,
                    current,
                    mean_force_target,
                    condition,
                    temperature,
                )
                outputs["lspwm_force_loss"] = force_loss
                if self.physical_calibration_enabled:
                    calibrated_force = self.world_model.calibrated_mean_force(
                        current,
                        condition,
                        temperature,
                        create_graph=True,
                    )
                    outputs["lspwm_calibrated_mean_force"] = calibrated_force
                    outputs["lspwm_calibrated_force_loss"] = (
                        calibrated_force - mean_force_target.detach()
                    ).square().mean(dim=-1).mean()

        # ── v28: Per-residue latent dynamics ──────────────────────────
        # Each residue gets its own CV state.  The encoder's raw per-residue
        # latents bypass the temporal-encoder mean-pooling so we recover
        # all L residues with their own q_i ∈ R^{cv_dim}.
        if (self.per_residue_enabled and context_latents is not None
                and context_ca_coords is not None and context_latents.ndim == 4):
            batch_size, num_frames, num_residues, _ = context_latents.shape

            # Context per-residue CVs: [B, T, L, cv_dim]
            q_context = self.residue_projection(context_latents)

            # Frame-level mask: [B, T]  (mask is already [B, T, L] from paired windows)
            frame_mask = context_residue_mask.any(dim=-1).to(dtype=q_context.dtype)
            # Residue-level mask for each frame: [B, T, L]
            rmask_3d = context_residue_mask.to(dtype=q_context.dtype)
            last_idx = (frame_mask.sum(dim=1) - 1).clamp(min=0).long()  # [B]
            q_current_res = q_context[torch.arange(batch_size, device=q_context.device), last_idx]
            rmask_current = rmask_3d[torch.arange(batch_size, device=rmask_3d.device), last_idx]

            # Cα distance graph: [B, L, 3] → [B, L, L]
            ca_last = context_ca_coords[torch.arange(batch_size, device=context_ca_coords.device), last_idx]
            ca_distances_last = torch.cdist(ca_last, ca_last)
            if self.gauge_projector is not None:
                gauge, _ = self.gauge_projector(
                    context_latents[torch.arange(batch_size, device=context_latents.device), last_idx],
                    ca_last, rmask_current,
                )
                outputs["lspwm_gauge_norm"] = gauge.norm(dim=-1).mean().detach()

            # Hierarchical path: a single global G is pooled from Q_local, while
            # the unified energy keeps residue and cross-scale coupling terms.
            global_current = (q_current_res * rmask_current.unsqueeze(-1)).sum(1) / rmask_current.sum(1, keepdim=True).clamp_min(1.0)

            # ── cv_vamp2 on global G: provides gradient to residue_projection
            # in ALL phases (critical fix for Phase 1 degeneracy).
            # VAMP-2 compares the per-residue-pooled G_current with the
            # per-residue-pooled G_future from the target encoder latents.
            if (future_latents is not None and future_latents.ndim == 4
                    and batch_size >= 2):
                q_future_vamp = self.residue_projection(future_latents.detach())
                ff_mask_v = future_residue_mask.any(dim=-1).to(dtype=q_future_vamp.dtype)
                ff_last_idx_v = (ff_mask_v.sum(dim=1) - 1).clamp(min=0).long()
                q_future_last_v = q_future_vamp[torch.arange(batch_size, device=q_future_vamp.device), ff_last_idx_v]
                frmask_last_v = future_residue_mask.to(dtype=q_future_last_v.dtype)[
                    torch.arange(batch_size, device=q_future_last_v.device), ff_last_idx_v]
                global_future_v = (q_future_last_v * frmask_last_v.unsqueeze(-1)).sum(1) / frmask_last_v.sum(1, keepdim=True).clamp_min(1.0)
                outputs["lspwm_cv_vamp2_loss"] = vamp2_loss(
                    global_current, global_future_v.detach(),
                )

            if self.energy_graph_enabled:
                if self.training:
                    energy_terms = self.unified_energy(
                        q_current_res, global_current, ca_distances_last, rmask_current,
                        condition, temperature,
                    )
                    residue_energy = energy_terms["energy_per_residue"]
                    residue_force = -torch.autograd.grad(
                        energy_terms["energy_total"].sum(), q_current_res,
                        create_graph=True, retain_graph=True,
                    )[0]
                    global_force = -torch.autograd.grad(
                        energy_terms["energy_total"].sum(), global_current,
                        create_graph=True, retain_graph=True,
                    )[0]
                else:
                    # Eval: compute forces via local enable_grad context
                    with torch.enable_grad():
                        _q_q = q_current_res.detach().clone().requires_grad_(True)
                        _G_q = global_current.detach().clone().requires_grad_(True)
                        _cond = condition.detach() if condition is not None else None
                        _temp = temperature.detach() if temperature is not None else None
                        _energy = self.unified_energy(
                            _q_q, _G_q, ca_distances_last, rmask_current, _cond, _temp
                        )
                        residue_force = -torch.autograd.grad(
                            _energy['energy_total'].sum(), _q_q, create_graph=False, retain_graph=True
                        )[0]
                        global_force = -torch.autograd.grad(
                            _energy['energy_total'].sum(), _G_q, create_graph=False
                        )[0]
                    energy_terms = _energy
                    residue_energy = energy_terms["energy_per_residue"]
                outputs["lspwm_global_force_norm"] = global_force.norm(dim=-1).mean().detach()
                outputs["lspwm_energy_global"] = energy_terms["energy_global"].detach()
                outputs["lspwm_energy_coupling_var"] = energy_terms["energy_coupling"].var().detach()
            else:
                residue_force = self.residue_potential.force(
                    q_current_res, ca_distances_last, rmask_current, condition, create_graph=True,
                )
                residue_energy = self.residue_potential(q_current_res, ca_distances_last, rmask_current, condition)
            outputs["lspwm_residue_force_norm"] = residue_force.norm(dim=-1).mean().detach()
            outputs["lspwm_residue_energy"] = residue_energy.detach()

            # Sliced score matching (replaces KDE when state dim > 16)
            if self.sliced_score_enabled and phase not in ("representation",):
                sliced_score = sliced_score_matching_loss(
                    self.residue_potential,
                    q_current_res,
                    ca_distances_last,
                    rmask_current,
                    condition,
                    noise_std=self.sliced_score_noise,
                    num_projections=self.sliced_score_projections,
                )
                outputs["lspwm_sliced_score_loss"] = sliced_score

            # Force direction loss (hinge): aligns force with observed displacement.
            # Gate on per_residue_enabled + future_latents availability instead of
            # model-level weight (the actual contribution is determined by the
            # optimization-section lspwm_force_direction_weight in train.py).
            if (self.per_residue_enabled
                    and phase not in ("representation",)
                    and future_latents is not None
                    and future_latents.ndim == 4):
                # Future per-residue CVs: [B, T_future, L, cv_dim]
                q_future = self.residue_projection(future_latents.detach())
                future_frame_mask = future_residue_mask.any(dim=-1).to(dtype=q_future.dtype)
                future_last_idx = (future_frame_mask.sum(dim=1) - 1).clamp(min=0).long()
                q_next_res = q_future[torch.arange(batch_size, device=q_future.device), future_last_idx]

                if self.energy_graph_enabled:
                    disp_for_loss = (q_next_res - q_current_res).detach()
                    f_norm = residue_force.norm(dim=-1).clamp_min(1e-8)
                    d_norm = disp_for_loss.norm(dim=-1).clamp_min(1e-8)
                    cos = (residue_force * disp_for_loss).sum(-1) / (f_norm * d_norm)
                    direction_loss = (torch.relu(-cos) * rmask_current).sum() / rmask_current.sum().clamp_min(1.0)
                else:
                    direction_loss = residue_force_direction_loss(
                        self.residue_potential, q_current_res, q_next_res,
                        ca_distances_last, rmask_current, condition,
                    )
                outputs["lspwm_residue_direction_loss"] = direction_loss
                # Diagnostic: force-direction alignment
                with torch.no_grad():
                    disp = q_next_res - q_current_res
                    force_norm = residue_force.norm(dim=-1).clamp(min=1e-8)
                    disp_norm = disp.norm(dim=-1).clamp(min=1e-8)
                    cos_sim = (residue_force * disp).sum(dim=-1) / (force_norm * disp_norm)
                outputs["lspwm_residue_force_cosine"] = (
                        (cos_sim * rmask_current).sum() / rmask_current.sum().clamp(min=1)
                    )
                # Amplitude supervision: force ≈ disp/dt (rate-space comparison).
                # dt_ps is large (24k-96k), so we normalise both sides into
                # rate units (ps⁻¹) to avoid exploding loss.
                dt_ps_val = horizon_time_ps[:, None, None].to(residue_force.dtype).clamp_min(1.0)
                outputs["lspwm_residue_amplitude_loss"] = (
                    ((residue_force - disp / dt_ps_val) ** 2).sum(-1) * rmask_current
                ).sum() / rmask_current.sum().clamp_min(1.0) / self.residue_cv_dim
                if self.residue_decoder is not None:
                    g_next = (q_next_res * rmask_current.unsqueeze(-1)).sum(1) / rmask_current.sum(1, keepdim=True).clamp_min(1.0)
                    mu, sigma = self.residue_decoder(q_current_res, global_current, g_next)
                    nll = 0.5 * (((q_next_res.detach() - mu) / sigma) ** 2 + 2.0 * sigma.log())
                    outputs["lspwm_residue_conditional_loss"] = (nll.mean(-1) * rmask_current).sum() / rmask_current.sum().clamp_min(1.0)

            # ── v22: Hierarchical world model with explicit energy graph ──
            # KDE force score on global G, sliced score on Q_local,
            # self-consistency between A_g and phi, and stationary MMD.
            if (self.energy_graph_enabled
                    and phase not in ("representation",)):
                # Force score on G: targets grad_G A(Q,G) via KDE on global state
                # Use the global CV samples (from learned_projection) as reference
                # for KDE. As training progresses, pool(Q_local) and learned_projection
                # converge through cv_vamp2.
                # V25 FIX: wrap in torch.enable_grad() so autograd.grad works during eval.
                if reference.shape[0] >= 2:
                    with torch.enable_grad():
                        outputs["lspwm_force_score_G_loss"] = force_score_G_loss(
                            self.unified_energy,
                            q_current_res, global_current, ca_distances_last,
                            rmask_current, condition, temperature,
                            reference,
                            bandwidth=self.mean_force_bandwidth,
                        )
                # Force score on Q: sliced score matching via UnifiedEnergyGraph
                # V25 FIX: wrap in torch.enable_grad() to support eval-mode forward.
                with torch.enable_grad():
                    outputs["lspwm_force_score_Q_loss"] = force_score_Q_loss(
                        self.unified_energy,
                        q_current_res, global_current, ca_distances_last,
                        rmask_current, condition, temperature,
                        noise_std=self.sliced_score_noise,
                        num_projections=self.sliced_score_projections,
                    )
                # Self-consistency: A_g(G) + mean_i[phi(q_i,G)] ≈ KDE(G)
                if reference.shape[0] >= 2:
                    with torch.enable_grad():
                        outputs["lspwm_self_consistency_loss"] = self_consistency_loss(
                            self.unified_energy,
                            q_current_res, global_current, ca_distances_last,
                        rmask_current, condition, temperature,
                        reference,
                        bandwidth=self.mean_force_bandwidth,
                    )
            # Stationary MMD: penalize distribution shift between rollout and MD
            if (phase in ("dynamics", "rollout", "observation")
                    and self.energy_graph_enabled
                    and reference.shape[0] >= 4):
                half = reference.shape[0] // 2
                outputs["lspwm_stationary_loss"] = stationary_distribution_loss(
                    reference[:half],
                    reference[half:],
                )

        # Dynamic NLL (L_dyn) — overdamped transition likelihood
        # Active in: thermodynamic_identification, dynamics, rollout
        if phase not in ("representation",):
            dynamic_nll = overdamped_transition_nll(
                self.world_model,
                current,
                target,
                condition,
                horizon_time_ps,
                temperature,
                history_features=history,
            )
            outputs["lspwm_dynamic_nll"] = dynamic_nll
            if self.physical_calibration_enabled:
                calibrated_parameters = self.world_model.calibrated_transition(
                    current,
                    condition,
                    horizon_time_ps,
                    temperature,
                    history_features=history,
                )
                outputs["lspwm_calibrated_transition_mean"] = calibrated_parameters[
                    "mean"
                ]
                outputs["lspwm_calibrated_dynamic_nll"] = (
                    calibrated_overdamped_transition_nll(
                        self.world_model,
                        current,
                        target,
                        condition,
                        horizon_time_ps,
                        temperature,
                        history_features=history,
                    )
                )
                outputs["lspwm_calibration_regularization"] = (
                    self.world_model.physical_calibration.regularization_loss(
                        current,
                        condition,
                    )
                )
                outputs["lspwm_physical_mobility_scale"] = (
                    self.world_model.physical_calibration.mobility_scale.detach()
                )
                outputs["lspwm_physical_free_energy_scale"] = (
                    self.world_model.physical_calibration.free_energy_scale.detach()
                )

            # ── v23: Mori transition NLL (FDT bath on global G) ──
            # Enabled when energy_graph is active and bath is instantiated.
            # Provides FDT-consistent transition likelihood with memory.
            global_future = None  # always initialised for M1 ablation use
            if (self.energy_graph_enabled
                and hasattr(self, "multi_bath")
                and self.multi_bath is not None
                and phase in ("thermodynamic_identification", "dynamics", "rollout", "observation")):
                # Predict future global state from future per-residue latents
                if (future_latents is not None and future_latents.ndim == 4):
                    q_future_all = self.residue_projection(future_latents.detach())
                    ff_mask = future_residue_mask.any(dim=-1).to(dtype=q_future_all.dtype)
                    ff_last_idx = (ff_mask.sum(dim=1) - 1).clamp(min=0).long()
                    q_future_last = q_future_all[
                        torch.arange(batch_size, device=q_future_all.device), ff_last_idx
                    ]
                    ff_rmask = future_residue_mask.to(dtype=q_future_last.dtype)[
                        torch.arange(batch_size, device=q_future_last.device), ff_last_idx
                    ]
                    global_future = (q_future_last * ff_rmask.unsqueeze(-1)).sum(1) / \
                                    ff_rmask.sum(1, keepdim=True).clamp_min(1.0)
                else:
                    # Fallback: pool from future_q (geometric state)
                    global_future = q_future_all = None

                if global_future is not None and global_future.shape == global_current.shape:
                    # V24: initialise auxiliary from steady-state estimate
                    # V25 FIX: wrap entire mori NLL path in enable_grad for eval mode.
                    # mori_transition_nll calls force_G() which uses autograd.grad().
                    with torch.no_grad():
                        with torch.enable_grad():
                            force_G_init = self.unified_energy.force_G(
                                q_current_res, global_current, ca_distances_last,
                                rmask_current, condition, temperature,
                            ).detach()
                    taus, C_bath = self.multi_bath.parameters_at(
                        dtype=global_current.dtype, device=global_current.device
                    )
                    gamma = taus.reciprocal()
                    aux_ss = -torch.einsum(
                        "da,bd->ba", C_bath, force_G_init
                    ) / gamma.unsqueeze(0).clamp(min=1e-6)
                    aux_ss = torch.tanh(aux_ss * 0.1) * 10.0  # soft clamp
                    with torch.enable_grad():
                        mori_result = mori_transition_nll(
                            self.unified_energy,
                            self.multi_bath,
                            self.mori_diffusion,
                            q_current_res,
                            global_current,
                            q_current_res,
                            global_future,
                            ca_distances_last,
                            rmask_current,
                            condition,
                            temperature,
                            horizon_time_ps,
                            auxiliary=aux_ss,
                        )
                    outputs["lspwm_mori_nll"] = mori_result["nll"]
                    outputs["lspwm_global_current"] = global_current.detach()
                    if global_future is not None:
                        outputs["lspwm_global_future"] = global_future.detach()
                    with torch.no_grad():
                        outputs["lspwm_mori_force_G_norm"] = (
                            mori_result["force_G"].norm(dim=-1).mean()
                        )
                        outputs["lspwm_mori_bath_drift_norm"] = (
                            mori_result["bath_drift"].norm(dim=-1).mean()
                        )
                        outputs["lspwm_mori_dissipative_norm"] = (
                            mori_result["dissipative_drift"].norm(dim=-1).mean()
                        )
                        outputs["lspwm_mori_total_drift_norm"] = (
                            mori_result["total_drift"].norm(dim=-1).mean()
                        )
                else:
                    outputs["lspwm_mori_nll"] = current.new_zeros(())
                    outputs["lspwm_global_current"] = global_current.detach() if global_current is not None else current.new_zeros(0)
                    if global_future is not None:
                        outputs["lspwm_global_future"] = global_future.detach()

            # ── M1 ablation outputs: force + memory decomposition on global CVs ──
            # Always compute force_G and bath drift when energy graph is active
            # (even when mori_nll is disabled), so M1 eval can ablate them independently.
            # V25 FIX #1: Predict aux from history (3-frame context) instead of
            # steady-state force to break cos(force,bath)≈0.857 degeneracy.
            if (self.energy_graph_enabled
                    and hasattr(self, "multi_bath")
                    and self.multi_bath is not None
                    and hasattr(self, 'unified_energy')
                    and (global_current is not None) and (global_current.shape[0] > 0)):
                # Compute conservative force on G via energy graph
                with torch.enable_grad():
                    force_G = self.unified_energy.force_G(
                        q_current_res, global_current, ca_distances_last,
                        rmask_current, condition, temperature,
                    ).detach()  # [B, D]

                # V25 FIX #1: Predict aux from per-residue history context.
                # Use last 3 frames of per-residue CVs pooled to global,
                # then linearly project to aux dim. This breaks the
                # cos(force,bath)≈0.857 degeneracy where aux_ss ≈ -C·force/γ
                # makes bath drift just a linear transform of force.
                B_pooled, T_pooled, L_pooled, _ = q_context.shape
                frame_mask_per = context_residue_mask.any(dim=-1).to(dtype=q_context.dtype)
                n_ctx = min(3, T_pooled)
                frames_pooled = []
                for offset in range(n_ctx):
                    idx_pool = (frame_mask_per.sum(dim=1) - 1 - offset).clamp(min=0).long()
                    q_frame_pool = q_context[torch.arange(B_pooled, device=q_context.device), idx_pool]  # [B, L, D]
                    rmask_frame = context_residue_mask[torch.arange(B_pooled, device=context_residue_mask.device), idx_pool]
                    g_frame = (q_frame_pool * rmask_frame.unsqueeze(-1)).sum(1) / rmask_frame.sum(1, keepdim=True).clamp_min(1.0)
                    frames_pooled.append(g_frame)
                # Pad with first frame if fewer than 3
                while len(frames_pooled) < 3:
                    frames_pooled.insert(0, frames_pooled[0])
                aux_input = torch.cat(frames_pooled, dim=-1).detach()  # [B, 3*D]
                # V25: history→bath projector (handles 3*D→bath_aux_dim mapping)
                aux_history = self.history_to_bath(aux_input)  # [B, bath_aux_dim]
                # Scale for stability (tanh already gives [-1,1])
                aux_history = aux_history * 2.0  # [B, bath_aux_dim]

                # V25 FIX #2: Per-dimension learnable diffusion scale.
                # The raw 5e-5 scale gives dissipative_drift ~5e-8.
                # We add a per-dim learnable multiplier to reach ~O(dG).
                if hasattr(self, 'diffusion_scale_correction'):
                    corr = torch.exp(self.diffusion_scale_correction) * torch.relu(self._diffusion_scale_correction_scale)
                else:
                    corr = torch.ones(self.residue_cv_dim, device=global_current.device, dtype=global_current.dtype)

                diffusion_raw = self.mori_diffusion(global_current, condition, temperature)  # [B, D, D]
                # Apply per-dim correction (diagonal reweighting)
                corr_diag = torch.diag_embed(corr)  # [D, D]
                diffusion_adj = diffusion_raw @ corr_diag  # [B, D, D]
                dissipative_drift = -torch.einsum(
                    "bij,bj->bi", diffusion_adj, force_G
                )  # [B, D]

                # Bath (memory) drift from history-predicted aux
                taus, C_bath = self.multi_bath.parameters_at(
                    dtype=global_current.dtype, device=global_current.device
                )
                C_expanded = C_bath.unsqueeze(0).expand(global_current.shape[0], -1, -1)
                bath_drift = torch.einsum("bda,ba->bd", C_expanded, aux_history)

                outputs["lspwm_m1_force_G"] = force_G.detach()
                outputs["lspwm_m1_dissipative_drift"] = dissipative_drift.detach()
                outputs["lspwm_m1_bath_drift"] = bath_drift.detach()
                outputs["lspwm_m1_total_drift"] = (dissipative_drift + bath_drift).detach()
                outputs["lspwm_m1_global_current"] = global_current.detach()
                if global_future is not None:
                    outputs["lspwm_m1_global_future"] = global_future.detach()
                if isinstance(horizon_time_ps, torch.Tensor):
                    outputs["lspwm_m1_dt"] = horizon_time_ps.detach()

            # ── V25: Per-residue dynamics integration ───────────────────
            # Each residue i has q_i ∈ R^D with its own conservative force
            # ∇_i A(Q,G) (from UnifiedEnergyGraph or ResidueMPP).
            # We now INTEGRATE the per-residue Langevin equation:
            #
            #   q_i(t+dt) = q_i(t) + [-D·∇_i A + C·s]·dt + √(2D·dt)·ξ_i
            #
            # with a global Mori bath s predicted from history.
            # This closes the gap between the rich per-residue energy
            # function and the dynamics integration, which previously
            # only operated on the pooled global CV G.
            if (self.per_residue_enabled
                    and phase not in ("representation",)
                    and effective_step is not None
                    and effective_step >= self.per_residue_warmup_steps
                    and context_latents is not None
                    and context_latents.ndim == 4
                    and hasattr(self, 'unified_energy')
                    and hasattr(self, 'multi_bath')
                    and hasattr(self, 'residue_projection')):
                # ── Per-residue dynamics step ──
                # Use pre-computed residue_force from UnifiedEnergyGraph
                # (already available from the energy_graph_enabled block above)
                if 'residue_force' in locals() and residue_force is not None:
                    per_res_step = self.per_residue_step(
                        q_per_res=q_current_res,
                        residue_force=residue_force,
                        residue_mask=rmask_current,
                        condition=condition,
                        dt_ps=horizon_time_ps,
                        temperature_k=temperature,
                        stochastic=True,
                    )
                    q_next_pred = per_res_step["q"]  # [B, L, D]

                    # ── Per-residue prediction skill ──
                    # Compare predicted vs observed next-state displacement
                    # weighted by valid residues
                    with torch.no_grad():
                        # Get future per-residue CVs for target
                        if (future_latents is not None
                                and future_latents.ndim == 4):
                            ff_mask = future_residue_mask.any(dim=-1).to(
                                dtype=context_latents.dtype)
                            ff_last_idx = (ff_mask.sum(dim=1) - 1).clamp(min=0).long()
                            q_future_last = self.residue_projection(
                                future_latents
                            )[torch.arange(batch_size, device=
                                context_latents.device), ff_last_idx]
                            ff_rmask_curr = future_residue_mask.to(
                                dtype=context_latents.dtype)[
                                torch.arange(batch_size, device=
                                    context_latents.device), ff_last_idx]

                            # Per-residue prediction error
                            pred_disp = q_next_pred - q_current_res  # [B, L, D]
                            true_disp = (q_future_last - q_current_res).detach()
                            per_res_sq = ((pred_disp - true_disp).square()
                                          .sum(dim=-1))  # [B, L]
                            # Persistence baseline
                            pers_sq = true_disp.square().sum(dim=-1)  # [B, L]
                            # Common mask
                            common_mask = rmask_current * ff_rmask_curr
                            n_valid = common_mask.sum().clamp_min(1.0)
                            pred_mse = (per_res_sq * common_mask).sum() / n_valid
                            pers_mse = (pers_sq * common_mask).sum() / n_valid
                            per_res_skill = 1.0 - pred_mse / pers_mse.clamp(min=1e-8)
                            outputs["lspwm_per_residue_pred_mse"] = pred_mse.detach()
                            outputs["lspwm_per_residue_skill"] = per_res_skill.detach()

                    # ── Per-residue transition NLL (training loss) ──
                    if (future_latents is not None
                            and future_latents.ndim == 4
                            and hasattr(self, '_predict_per_residue_aux')):
                        ff_mask = future_residue_mask.any(dim=-1).to(
                            dtype=context_latents.dtype)
                        ff_last_idx = (ff_mask.sum(dim=1) - 1).clamp(min=0).long()
                        q_future_last = self.residue_projection(
                            future_latents.detach()
                        )[torch.arange(batch_size, device=
                            context_latents.device), ff_last_idx]
                        per_res_nll = self.per_residue_transition_nll(
                            q_current=q_current_res,
                            q_next=q_future_last,
                            residue_force=residue_force,
                            residue_mask=rmask_current,
                            condition=condition,
                            dt_ps=horizon_time_ps,
                            temperature_k=temperature,
                        )
                        outputs["lspwm_per_residue_transition_nll"] = per_res_nll

                    # ── Diagnostics ──
                    outputs["lspwm_per_residue_drift_norm"] = (
                        per_res_step["drift"].norm(dim=-1).mean().detach()
                    )
                    outputs["lspwm_per_residue_dissipative_norm"] = (
                        per_res_step["dissipative"].norm(dim=-1).mean().detach()
                    )
                    outputs["lspwm_per_residue_bath_norm"] = (
                        per_res_step["bath_drift"].squeeze(1).norm(dim=-1).mean().detach()
                    )
                    outputs["lspwm_per_residue_aux_norm"] = (
                        per_res_step["auxiliary"].norm(dim=-1).mean().detach()
                    )

        # Koopman VAMP-2 loss (L_koop) — Ch.7 (always active for representation shaping)
        role_losses, current_roles = self.role_readout.losses(
            current, target, current_fixed, condition
        )
        outputs.update(role_losses)
        outputs["lspwm_koopman_features"] = current_roles["koopman"]
        outputs["lspwm_msm_logits"] = current_roles["msm_logits"]
        outputs["lspwm_geometric_features"] = current_roles["geometric"]
        outputs["lspwm_auxiliary_features"] = current_roles["auxiliary"]

        # Rollout self-consistency losses (Phase IV) — multi-step GENERIC rollout
        if phase == "rollout" and self.rollout_enabled:
            rollout_warmup = max(0, self.rollout_warmup_steps)
            ramp = min(1.0, float(effective_step - curriculum_steps[3]) / max(rollout_warmup, 1)) if curriculum_steps else 1.0
            n_steps = max(1, int(self.rollout_steps * ramp))
            n_ensemble = max(1, int(self.rollout_ensemble_size * ramp))
            # Extract real future frames for rollout trajectory supervision
            future_sequence_for_rollout = future_q.detach()  # [batch, T, state_dim]
            rollout_losses = self.compute_rollout_losses(
                current,
                target,
                condition,
                history,
                temperature,
                horizon_time_ps,
                reference,
                current_fixed,
                rollout_steps=n_steps,
                ensemble_size=n_ensemble,
                bandwidth=self.mean_force_bandwidth,
                future_sequence=(
                    future_sequence_for_rollout
                    if self.rollout_intermediate_supervision
                    else None
                ),
            )
            outputs.update(rollout_losses)

        # Track endpoint energy skill (diagnostic, not a loss)
        if phase in ("dynamics", "rollout", "observation") and current.shape[0] >= 2:
            diag_ensemble = max(4, self.rollout_ensemble_size if self.rollout_enabled else 4)
            ensemble_samples = self.sample_endpoint(
                current, condition, history, horizon_time_ps, temperature,
                ensemble_size=diag_ensemble,
            )
            score = torch.linalg.vector_norm(
                ensemble_samples - target[:, None, :], dim=-1
            ).mean(dim=1)
            persistence = torch.linalg.vector_norm(current - target, dim=-1)
            skill = 1.0 - score / persistence.clamp(min=1.0e-8)
            outputs["lspwm_endpoint_energy_skill"] = skill.mean().detach()

            # Deterministic skill (mean-drift only, no noise)
            # This separates PMF quality from noise and provides a cleaner
            # diagnostic for whether the learned PMF improves predictions.
            if self.physical_calibration_enabled:
                deterministic_q = outputs["lspwm_calibrated_transition_mean"]
            else:
                deterministic = self.world_model.step(
                    current,
                    auxiliary=current.new_zeros(
                        current.shape[0], self.world_model.config.auxiliary_dim
                    ),
                    condition=condition,
                    dt_ps=horizon_time_ps,
                    temperature_k=temperature,
                    stochastic=False,
                )["q"]
                correction = self.world_model.history_correction(
                    current, history, horizon_time_ps, temperature,
                )
                deterministic_q = deterministic + correction
            deterministic_distance = torch.linalg.vector_norm(
                deterministic_q - target, dim=-1
            )
            deterministic_skill = 1.0 - deterministic_distance / persistence.clamp(min=1.0e-8)
            outputs["lspwm_deterministic_skill"] = deterministic_skill.mean().detach()

        # Observation loss (L_obs) — Ch.6, only when decoder exists
        # Active in: observation
        if phase == "observation" and self.world_model.observation_model is not None and current.shape[0] >= 2:
            density_weights = local_density_weights(current, bandwidth=self.mean_force_bandwidth)
            obs_loss = asymmetric_cycle_loss(
                self.world_model.observation_model,
                current,
                target,
                current,
                target,
                weights=density_weights,
            )
            outputs["lspwm_observation_loss"] = obs_loss

        # Record which phase is active
        outputs["lspwm_active_phase"] = phase

        # ── v21 THM-JEPA: reversible force diagnostics ──
        # When reversible dynamics are enabled, decompose the total drift into its
        # three components and report their magnitudes and cosine alignment with
        # the observed increment.  This lets us watch the reversible force
        # emerge from zero during training.
        if (phase not in ("representation",)
                and self.world_model.skew_operator is not None
                and current.shape[0] >= 2):
            extra_aux = current.new_zeros(
                current.shape[0], self.world_model.config.auxiliary_dim
            )
            dt_ps = horizon_time_ps
            with torch.no_grad():
                coeffs = self.world_model.coefficients(current, temperature)
                diffusion = coeffs["diffusion"]
                coupling = coeffs["coupling"]
                _grad = -self.world_model.potential.mean_force(
                    current, condition, temperature, create_graph=False,
                )
                dissipative_drift = -torch.einsum("bij,bj->bi", diffusion, _grad)
                bath_drift = torch.einsum(
                    "bqa,ba->bq",
                    coupling.expand(current.shape[0], -1, -1),
                    extra_aux,
                )
                rev_grad = self.world_model.reversible_potential.mean_force(
                    current, condition, temperature, create_graph=False,
                )
                reversible_drift = self.world_model.skew_operator.apply(rev_grad)
                total_drift = dissipative_drift + bath_drift + reversible_drift

                # Observed increment
                observed_incr = target - current  # [batch, d]

                outputs["lspwm_dissipative_force_norm"] = (
                    dissipative_drift.norm(dim=-1).mean().detach()
                )
                outputs["lspwm_reversible_force_norm"] = (
                    reversible_drift.norm(dim=-1).mean().detach()
                )
                outputs["lspwm_bath_drift_norm"] = (
                    bath_drift.norm(dim=-1).mean().detach()
                )
                outputs["lspwm_total_drift_norm"] = (
                    total_drift.norm(dim=-1).mean().detach()
                )
                # Cosine similarity between total force drift and observed increment
                num = (total_drift * observed_incr).sum(dim=-1)
                den = total_drift.norm(dim=-1) * observed_incr.norm(dim=-1).clamp(min=1e-8)
                outputs["lspwm_drift_incr_cosine"] = (
                    (num / den).mean().detach()
                )
                # Reversible fraction of total drift magnitude
                total_mag = total_drift.norm(dim=-1).clamp(min=1e-8)
                outputs["lspwm_reversible_drift_fraction"] = (
                    (reversible_drift.norm(dim=-1) / total_mag).mean().detach()
                )

        return outputs

    def get_active_phase(self, curriculum_step: int | None, config_curriculum_steps: tuple[int, ...] | None = None) -> str:
        """Return the active curriculum phase for the given step.

        Phases (Ch.7.2):
            I   (representation)         : L_repr + L_koop (shaping the representation)
            II  (thermodynamic_id)       : L_force + L_dyn (fit PMF and diffusion)
            III (dynamics)               : L_dyn (fine-tune transition)
            IV  (rollout)                : multi-step rollout self-consistency
            V   (observation)            : L_obs (decoder only)
        """
        if curriculum_step is None:
            return "dynamics"
        phases = ("representation", "thermodynamic_identification", "dynamics", "rollout", "observation")
        thresholds = config_curriculum_steps or (0, 500, 1000, 1500, 2500)
        for i in range(len(phases) - 1, -1, -1):
            if curriculum_step >= thresholds[i]:
                return phases[i]
        return phases[0]

    @torch.no_grad()
    def compute_rollout_diagnostics(
        self,
        current: torch.Tensor,
        condition: torch.Tensor,
        history_features: torch.Tensor,
        temperature_k: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        *,
        rollout_steps: int = 4,
        ensemble_size: int = 4,
    ) -> dict[str, torch.Tensor]:
        """Run multi-step rollout diagnostics without gradient computation.

        Returns trajectory states and diagnostics for the rollout phase.
        """
        batch = current.shape[0]
        device = current.device
        dt_per_step = horizon_time_ps / max(rollout_steps, 1)
        expanded_current = current.repeat_interleave(ensemble_size, dim=0)
        expanded_condition = condition.repeat_interleave(ensemble_size, dim=0)
        expanded_temperature = temperature_k.repeat_interleave(ensemble_size)
        expanded_dt = dt_per_step.repeat_interleave(ensemble_size)
        auxiliary = current.new_zeros(expanded_current.shape[0], self.world_model.config.auxiliary_dim)
        trajectory = self.world_model.rollout(
            expanded_current,
            expanded_condition,
            expanded_dt,
            rollout_steps,
            expanded_temperature,
            stochastic=True,
            auxiliary=auxiliary,
            history_features=history_features.repeat_interleave(ensemble_size, dim=0),
        )
        states = trajectory["q"]  # [batch*ensemble, steps+1, state_dim]
        states = states.reshape(batch, ensemble_size, rollout_steps + 1, -1)
        return {
            "lspwm_rollout_trajectory": states,  # [batch, ensemble, steps+1, state_dim]
            "lspwm_rollout_steps": rollout_steps,
        }

    def compute_rollout_losses(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
        history_features: torch.Tensor,
        temperature_k: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        reference: torch.Tensor,
        current_fixed: torch.Tensor,
        *,
        rollout_steps: int = 4,
        ensemble_size: int = 4,
        bandwidth: float = 1.5,
        vamp_ridge: float = 1.0e-6,
        future_sequence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-step rollout self-consistency losses (Phase IV).

        The rollout is a GENERIC Langevin trajectory from ``current`` to
        ``target`` in ``rollout_steps`` equal steps. Five losses are computed:

            1. Rollout endpoint energy score:
               Compares the final rollout state to the target in energy space.
               Score = ||q_final - target|| - 0.5 * pairwise ||q_i - q_j||
               This is the same metric used in evaluation, now a training loss.

            2. Deterministic observed endpoint loss:
               Directly supervises the mean rollout at the matched held-out
               horizon. This is the primary long-horizon drift objective.

            3. Ensemble Gaussian endpoint NLL:
               Matches ensemble mean and variance to the observed endpoint,
               preventing diffusion collapse or an over-dispersed rollout.

            4. Rollout distribution matching:
               KDE log-density of rollout intermediate states under the
               empirical data distribution from ``reference``. Penalises
               rollout states that drift into low-density regions.

            5. Rollout trajectory NLL:
               Per-step negative log-likelihood of rollout intermediate states
               against the real future frames (``future_sequence``). Provides
               supervised signal at each rollout step, not just the endpoint.

            6. Rollout Koopman consistency:
               VAMP-2 score between consecutive rollout states. Ensures the
               rollout trajectory has the same dynamical structure as real data.

            7. Rollout force consistency:
               PMF mean-force matching on rollout intermediate states using
               the same KDE target as the standard force loss.
        """
        losses = {}
        batch = current.shape[0]
        device = current.device
        if batch < 2 or rollout_steps < 1:
            losses["lspwm_rollout_energy_loss"] = current.new_zeros(())
            losses["lspwm_rollout_endpoint_loss"] = current.new_zeros(())
            losses["lspwm_rollout_gaussian_nll"] = current.new_zeros(())
            losses["lspwm_rollout_distribution_loss"] = current.new_zeros(())
            losses["lspwm_rollout_koopman_loss"] = current.new_zeros(())
            losses["lspwm_rollout_force_loss"] = current.new_zeros(())
            return losses

        dt_per_step = horizon_time_ps / max(rollout_steps, 1)
        expanded_current = current.repeat_interleave(ensemble_size, dim=0)
        expanded_condition = condition.repeat_interleave(ensemble_size, dim=0)
        expanded_temperature = temperature_k.repeat_interleave(ensemble_size)
        expanded_dt = dt_per_step.repeat_interleave(ensemble_size)
        auxiliary = current.new_zeros(
            expanded_current.shape[0], self.world_model.config.auxiliary_dim
        )
        trajectory = self.world_model.rollout(
            expanded_current,
            expanded_condition,
            expanded_dt,
            rollout_steps,
            expanded_temperature,
            stochastic=True,
            auxiliary=auxiliary,
            history_features=history_features.repeat_interleave(ensemble_size, dim=0),
        )
        rollout_states = trajectory["q"]  # [batch*ensemble, steps+1, state_dim]
        rollout_states = rollout_states.reshape(
            batch, ensemble_size, rollout_steps + 1, -1
        )  # [batch, ensemble, steps+1, state_dim]

        # A deterministic endpoint supplies a low-variance, directly observed
        # long-horizon drift target. Unlike bridge interpolation, the target is
        # the actual paired future endpoint at the declared physical horizon.
        deterministic = self.world_model.rollout(
            current,
            condition,
            dt_per_step,
            rollout_steps,
            temperature_k,
            stochastic=False,
            auxiliary=current.new_zeros(
                batch, self.world_model.config.auxiliary_dim
            ),
        )
        losses["lspwm_rollout_endpoint_loss"] = F.smooth_l1_loss(
            deterministic["q"][:, -1], target
        )

        # --- 1. Rollout endpoint energy score ---
        q_final = rollout_states[:, :, -1, :]  # [batch, ensemble, state_dim]
        target_energy = torch.linalg.vector_norm(
            q_final - target[:, None, :], dim=-1
        )  # [batch, ensemble]
        pairwise = torch.linalg.vector_norm(
            q_final[:, :, None, :] - q_final[:, None, :, :], dim=-1
        )  # [batch, ensemble, ensemble]
        energy_score = target_energy.mean(dim=1) - 0.5 * pairwise.mean(dim=(1, 2))
        losses["lspwm_rollout_energy_loss"] = energy_score.mean()

        if self.rollout_gaussian_nll_enabled and ensemble_size >= 2:
            ensemble_mean = q_final.mean(dim=1)
            ensemble_variance = q_final.var(dim=1, unbiased=False).clamp(min=1.0e-4)
            endpoint_residual = target - ensemble_mean
            losses["lspwm_rollout_gaussian_nll"] = 0.5 * (
                endpoint_residual.square() / ensemble_variance
                + ensemble_variance.log()
            ).mean()
        else:
            # One rollout contains no variance information. In particular, a
            # variance floor would turn ordinary stochastic endpoint error
            # into an arbitrarily large and misleading Gaussian NLL.
            losses["lspwm_rollout_gaussian_nll"] = current.new_zeros(())

        # --- 2. Rollout distribution matching ---
        # KDE log-density of rollout intermediate states under reference
        intermediate = rollout_states[:, :, :-1, :]  # [batch, ensemble, steps, state_dim]
        flat_intermediate = intermediate.reshape(-1, current.shape[-1])  # [batch*ensemble*steps, state_dim]
        if reference.shape[0] >= 2:
            bw = bandwidth * bandwidth
            # delta = flat_intermediate [N, d] - reference [M, d] -> [N, M, d]
            delta = flat_intermediate[:, None, :] - reference[None, :, :]
            log_kernel = -0.5 * delta.square().sum(dim=-1) / bw
            log_density = torch.logsumexp(log_kernel, dim=1) - math.log(max(reference.shape[0], 1))
            # Penalise low density: loss = -mean(log_density)
            density_loss = -log_density.mean()
            losses["lspwm_rollout_distribution_loss"] = density_loss

        # --- 3. Rollout trajectory NLL ---
        # Per-step NLL of rollout intermediate states against real future frames.
        # Uses the same overdamped transition NLL as the dynamics phase.
        if future_sequence is not None:
            # future_sequence: [batch, T, state_dim] — real future frames
            rollout_nll = 0.0
            n_steps_used = 0
            for step_idx in range(1, rollout_steps + 1):
                q_step = rollout_states[:, :, step_idx, :]  # [batch, ensemble, state_dim]
                # Find the corresponding real future frame by time
                step_frac = float(step_idx) / float(rollout_steps)
                frame_idx = int(step_frac * (future_sequence.shape[1] - 1))
                frame_idx = min(frame_idx, future_sequence.shape[1] - 1)
                real_target = future_sequence[:, frame_idx, :]  # [batch, state_dim]
                # Compute NLL per ensemble member, average
                for e in range(ensemble_size):
                    q_e = q_step[:, e, :]  # [batch, state_dim]
                    nll = overdamped_transition_nll(
                        self.world_model, q_e, real_target, condition,
                        dt_per_step, temperature_k,
                        history_features=history_features,
                    )
                    rollout_nll = rollout_nll + nll
                    n_steps_used += 1
            if n_steps_used > 0:
                # V27: Clamp rollout trajectory NLL to prevent explosion
                # The per-sample NLL is already clamped in _gaussian_transition_nll_from_parameters,
                # but the average over many steps can still be large if diffusion collapses
                avg_nll = rollout_nll / n_steps_used
                losses["lspwm_rollout_trajectory_nll"] = avg_nll.clamp(max=100.0)

        # --- 4. Rollout Koopman consistency ---
        # VAMP-2 between consecutive rollout states.
        # vamp2_loss returns -VAMP2 (to minimise), so we use it directly.
        source = rollout_states[:, :, :-1, :].reshape(-1, current.shape[-1])
        target_roll = rollout_states[:, :, 1:, :].reshape(-1, current.shape[-1])
        if source.shape[0] >= 2 and source.shape[0] == target_roll.shape[0]:
            rollout_koopman = vamp2_loss(source, target_roll, ridge=vamp_ridge)
            losses["lspwm_rollout_koopman_loss"] = rollout_koopman

        # --- 4. Rollout force consistency ---
        # PMF mean-force matching on rollout intermediate states
        if reference.shape[0] >= 2:
            rollout_flat = flat_intermediate.detach()
            mean_force_target = kde_mean_force(
                rollout_flat, reference, bandwidth
            )
            # Repeat condition for each ensemble member and each rollout step
            # flat_intermediate shape: [batch * ensemble * steps, state_dim]
            expanded_condition = condition.repeat_interleave(
                ensemble_size * rollout_steps, dim=0
            )
            expanded_temperature = temperature_k.repeat_interleave(
                ensemble_size * rollout_steps, dim=0
            )
            force_loss = mean_force_matching_loss(
                self.world_model.potential,
                rollout_flat,
                mean_force_target,
                expanded_condition,
                expanded_temperature,
            )
            losses["lspwm_rollout_force_loss"] = force_loss

        # Ensure all losses are present
        for key in (
                     "lspwm_rollout_energy_loss",
                     "lspwm_rollout_endpoint_loss",
                     "lspwm_rollout_gaussian_nll",
                     "lspwm_rollout_distribution_loss",
                     "lspwm_rollout_koopman_loss", "lspwm_rollout_force_loss",
                     "lspwm_rollout_trajectory_nll"):
            if key not in losses:
                losses[key] = current.new_zeros(())
        return losses

    def sample_endpoint(
        self,
        current: torch.Tensor,
        condition: torch.Tensor,
        history_features: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        *,
        ensemble_size: int = 8,
    ) -> torch.Tensor:
        """Sample ``[batch, ensemble, state_dim]`` endpoints for evaluation."""
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2")
        batch = current.shape[0]
        expanded_current = current.repeat_interleave(ensemble_size, dim=0)
        expanded_condition = condition.repeat_interleave(ensemble_size, dim=0)
        expanded_history = history_features.repeat_interleave(ensemble_size, dim=0)
        expanded_horizon = horizon_time_ps.repeat_interleave(ensemble_size)
        expanded_temperature = temperature_k.repeat_interleave(ensemble_size)
        auxiliary = current.new_zeros(batch * ensemble_size, self.world_model.config.auxiliary_dim)
        if self.world_model.physical_calibration.enabled:
            parameters = self.world_model.calibrated_transition(
                expanded_current,
                expanded_condition,
                expanded_horizon,
                expanded_temperature,
                history_features=expanded_history,
            )
            covariance = 2.0 * parameters["diffusion"] * expanded_horizon[:, None, None]
            cholesky = torch.linalg.cholesky(covariance)
            noise = torch.einsum(
                "bi,bji->bj",
                torch.randn_like(expanded_current),
                cholesky,
            )
            samples = parameters["mean"] + noise
        else:
            result = self.world_model.step(
                expanded_current,
                auxiliary,
                expanded_condition,
                expanded_horizon,
                expanded_temperature,
                stochastic=True,
            )
            correction = self.world_model.history_correction(
                current,
                history_features,
                horizon_time_ps,
                temperature_k,
            )
            samples = result["q"] + correction.repeat_interleave(
                ensemble_size, dim=0
            )
        return samples.reshape(batch, ensemble_size, -1)
