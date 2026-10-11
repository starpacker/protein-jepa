"""GLE integration and world model core.

Contains:
  - BoundedConservativePotential: low-rank bounded free-energy correction
  - PhysicalTransitionCalibration: FDT-consistent scalar calibration
  - LSPWMConfig: serializable config dataclass
  - LSPWMWorldModel: canonical passive Markovian Mori embedding with
    step, step_midpoint, rollout, semigroup consistency
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature
from protein_jepa_frame.models.lasp.energy import (
    MeanForcePotential,
    ReversiblePotential,
    SkewSymmetricOperator,
)
from protein_jepa_frame.models.lasp.diffusion import (
    ConstantDiffusion,
    StateDependentDiffusion,
)
from protein_jepa_frame.models.lasp.bath import (
    GaugeFixedBath,
    ProjectedForceClosure,
)
from protein_jepa_frame.models.lasp.objectives import (
    FlowMatchObservationModel,
    vamp2_loss,
    kde_mean_force,
    mean_force_matching_loss,
    overdamped_transition_nll,
    calibrated_overdamped_transition_nll,
    asymmetric_cycle_loss,
    local_density_weights,
)
from protein_jepa_frame.models.lasp.objectives import (
    PoissonOperator,
    DissipationOperator,
)


class BoundedConservativePotential(nn.Module):
    """Low-rank bounded free-energy correction in the existing latent space."""

    def __init__(
        self, state_dim: int, condition_dim: int,
        *, rank: int = 16, scale: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or condition_dim < 0 or rank < 1 or scale < 0.0:
            raise ValueError("invalid conservative correction dimensions or scale")
        self.state_dim = int(state_dim)
        self.condition_dim = int(condition_dim)
        self.rank = int(rank)
        self.scale = float(scale)
        self.raw_state_basis = nn.Parameter(torch.randn(state_dim, rank) * 0.02)
        self.bias = nn.Parameter(torch.zeros(rank))
        self.condition_basis = nn.Parameter(torch.zeros(condition_dim, rank))

    def _condition(self, q: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_dim == 0:
            return q.new_zeros(q.shape[0], 0)
        if condition is None:
            return q.new_zeros(q.shape[0], self.condition_dim)
        if condition.shape != (q.shape[0], self.condition_dim):
            raise ValueError("condition shape does not match conservative correction")
        return condition.to(device=q.device, dtype=q.dtype)

    def _state_basis(self) -> torch.Tensor:
        return torch.tanh(self.raw_state_basis)

    def forward(
        self, q: torch.Tensor, condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.ndim != 2 or q.shape[-1] != self.state_dim:
            raise ValueError("q has the wrong conservative correction shape")
        state_features = torch.tanh(q @ self._state_basis())
        condition_features = self._condition(q, condition)
        amplitudes = self.bias
        if self.condition_dim > 0:
            amplitudes = amplitudes + condition_features @ self.condition_basis
        amplitudes = torch.tanh(amplitudes) / math.sqrt(float(self.rank))
        return self.scale * (state_features * amplitudes).sum(dim=-1)

    def gradient(
        self, q: torch.Tensor, condition: torch.Tensor | None = None,
        *, create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``grad DeltaA``; the corresponding mean force is ``-gradient``."""
        with torch.enable_grad():
            q_leaf = q if q.requires_grad else q.detach().requires_grad_(True)
            energy = self(q_leaf, condition)
            return torch.autograd.grad(
                energy.sum(), q_leaf,
                create_graph=create_graph, retain_graph=create_graph,
            )[0]


class PhysicalTransitionCalibration(nn.Module):
    """FDT-consistent scalar and conservative transition calibration."""

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        *,
        enabled: bool = False,
        mobility_init: float = 1.0,
        free_energy_init: float = 1.0,
        log_scale_min: float = -12.0,
        log_scale_max: float = 12.0,
        conservative_enabled: bool = False,
        conservative_rank: int = 16,
        conservative_scale: float = 1.0,
        conservative_rescale_with_free_energy: bool = False,
    ) -> None:
        super().__init__()
        if mobility_init <= 0.0 or free_energy_init <= 0.0:
            raise ValueError("physical calibration scales must be positive")
        if log_scale_min >= log_scale_max:
            raise ValueError("physical calibration log-scale bounds are invalid")
        self.enabled = bool(enabled)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)
        self.conservative_enabled = bool(conservative_enabled)
        self.conservative_rescale_with_free_energy = bool(
            conservative_rescale_with_free_energy
        )
        self.log_mobility_scale = nn.Parameter(
            torch.tensor(math.log(float(mobility_init)))
        )
        self.log_free_energy_scale = nn.Parameter(
            torch.tensor(math.log(float(free_energy_init)))
        )
        self.conservative_correction = (
            BoundedConservativePotential(
                state_dim, condition_dim,
                rank=conservative_rank, scale=conservative_scale,
            )
            if self.conservative_enabled else None
        )

    def _scale(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(self.log_scale_min, self.log_scale_max).exp()

    @property
    def mobility_scale(self) -> torch.Tensor:
        return self._scale(self.log_mobility_scale)

    @property
    def free_energy_scale(self) -> torch.Tensor:
        return self._scale(self.log_free_energy_scale)

    def correction_gradient(
        self, q: torch.Tensor, condition: torch.Tensor | None,
        *, create_graph: bool = True,
    ) -> torch.Tensor:
        if self.conservative_correction is None:
            return q.new_zeros(q.shape)
        return self.conservative_correction.gradient(
            q, condition, create_graph=create_graph,
        )

    def regularization_loss(
        self, q: torch.Tensor, condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.conservative_correction is None:
            return q.new_zeros(())
        correction_gradient = self.correction_gradient(
            q, condition, create_graph=True,
        )
        correction_energy = self.conservative_correction(q, condition)
        return correction_gradient.square().mean() + 0.01 * correction_energy.square().mean()

@dataclass(frozen=True)
class LSPWMConfig:
    """Small serializable contract for the standalone LSP-WM core."""

    state_dim: int = 256
    condition_dim: int = 64
    auxiliary_dim: int = 4
    reference_temperature_k: float = 348.0
    diffusion_scale: float = 1.0e-3
    relaxation_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0)
    coupling_scale: float = 1.0e-3
    history_feature_dim: int = 0
    history_closure_hidden_dim: int = 128
    history_closure_max_shift: float = 0.55
    history_closure_horizon_gate_steps: float = 2.0

    # Potential architecture
    potential_hidden_dim: int = 128
    potential_hidden_depth: int = 4

    # Observation model
    observation_model: bool = False
    observation_coord_dim: int = 0
    observation_hidden_dim: int = 256
    observation_num_layers: int = 3
    observation_flow_steps: int = 16

    # Curriculum
    curriculum_phases: tuple[str, ...] = (
        "representation", "koopman", "thermodynamic_identification",
        "dynamics", "observation",
    )
    curriculum_phase_steps: tuple[int, ...] = (0, 1000, 2500, 5000, 10000)

    # State-dependent diffusion
    state_dependent_diffusion_enabled: bool = False

    # Separate force-loss flag
    force_loss_enabled: bool = True

    # FDT-consistent physical transition calibration
    physical_calibration_enabled: bool = False
    physical_mobility_init: float = 1.0
    physical_free_energy_init: float = 1.0
    physical_log_scale_min: float = -12.0
    physical_log_scale_max: float = 12.0
    conservative_correction_enabled: bool = False
    conservative_correction_rank: int = 16
    conservative_correction_scale: float = 1.0
    conservative_correction_rescale_with_free_energy: bool = False

    # v4.4 architectural extensions
    midpoint_integration: bool = False
    rollout_trajectory_steps: int = 0
    semigroup_consistency_weight: float = 0.0
    sliced_score_matching_enabled: bool = False
    sliced_score_matching_noise: float = 0.05

    # v21 THM-JEPA: reversible (divergence-free) dynamics
    reversible_dynamics_enabled: bool = False
    reversible_operator_rank: int = 16
    reversible_hidden_dim: int = 128
    reversible_hidden_depth: int = 4


class LSPWMWorldModel(nn.Module):
    """Canonical passive Markovian Mori embedding.

    The executed path is the dimensionless overdamped system:

        dq = [-D grad A(q) + C s] dt + sqrt(2D) dW_q
        ds = [-Gamma s - C^T grad A(q)] dt + sqrt(2Gamma) dW_s

    ``C`` and ``Gamma`` come from the same QR-gauge-fixed bath object used for
    the reported memory kernel.
    """

    def __init__(self, config: LSPWMConfig = LSPWMConfig()) -> None:
        super().__init__()
        if len(config.relaxation_times_ps) != config.auxiliary_dim:
            raise ValueError("bath times must match auxiliary_dim")
        self.config = config
        self.potential = MeanForcePotential(
            config.state_dim, config.condition_dim,
            hidden_dim=config.potential_hidden_dim,
            hidden_depth=config.potential_hidden_depth,
            reference_temperature_k=config.reference_temperature_k,
        )
        if config.state_dependent_diffusion_enabled:
            self.diffusion = StateDependentDiffusion(
                config.state_dim, config.condition_dim,
                hidden_dim=128,
                reference_temperature_k=config.reference_temperature_k,
                diffusion_scale=config.diffusion_scale,
            )
        else:
            self.diffusion = ConstantDiffusion(
                config.state_dim,
                reference_temperature_k=config.reference_temperature_k,
                diffusion_scale=config.diffusion_scale,
            )
        self.physical_calibration = PhysicalTransitionCalibration(
            config.state_dim, config.condition_dim,
            enabled=config.physical_calibration_enabled,
            mobility_init=config.physical_mobility_init,
            free_energy_init=config.physical_free_energy_init,
            log_scale_min=config.physical_log_scale_min,
            log_scale_max=config.physical_log_scale_max,
            conservative_enabled=config.conservative_correction_enabled,
            conservative_rank=config.conservative_correction_rank,
            conservative_scale=config.conservative_correction_scale,
            conservative_rescale_with_free_energy=(
                config.conservative_correction_rescale_with_free_energy
            ),
        )
        self.bath = GaugeFixedBath(
            config.state_dim, config.auxiliary_dim,
            relaxation_times_ps=config.relaxation_times_ps,
            coupling_scale=config.coupling_scale,
        )
        if config.reversible_dynamics_enabled:
            self.reversible_potential = ReversiblePotential(
                config.state_dim, config.condition_dim,
                hidden_dim=config.reversible_hidden_dim,
                hidden_depth=config.reversible_hidden_depth,
                reference_temperature_k=config.reference_temperature_k,
            )
            self.skew_operator = SkewSymmetricOperator(
                config.state_dim, rank=config.reversible_operator_rank,
            )
        else:
            self.reversible_potential = None
            self.skew_operator = None
        self.history_closure = (
            ProjectedForceClosure(
                config.state_dim, config.history_feature_dim,
                hidden_dim=config.history_closure_hidden_dim,
                max_shift=config.history_closure_max_shift,
                horizon_gate_steps=config.history_closure_horizon_gate_steps,
                base_step_ps=config.relaxation_times_ps[0],
                reference_temperature_k=config.reference_temperature_k,
            )
            if config.history_feature_dim > 0 else None
        )
        self.poisson = PoissonOperator(config.state_dim, config.auxiliary_dim)
        self.dissipation = DissipationOperator(
            config.state_dim, config.auxiliary_dim,
            reference_temperature_k=config.reference_temperature_k,
        )
        self.observation_model = (
            FlowMatchObservationModel(
                config.state_dim, config.observation_coord_dim,
                hidden_dim=config.observation_hidden_dim,
                num_layers=config.observation_num_layers,
                flow_matching_steps=config.observation_flow_steps,
            )
            if config.observation_model else None
        )

    def history_correction(
        self,
        q: torch.Tensor,
        history_features: torch.Tensor | None,
        dt_ps: torch.Tensor | float,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if self.history_closure is None:
            return q.new_zeros(q.shape)
        if history_features is None:
            raise ValueError("history closure requires history_features")
        return self.history_closure(q, history_features, dt_ps, temperature_k)

    def coefficients(
        self,
        q: torch.Tensor,
        temperature_k: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        diffusion = self.diffusion(q.shape[0], temperature_k, reference=q)
        bath = self.bath.parameters_at(dtype=q.dtype, device=q.device)
        return {"diffusion": diffusion, **bath}

    def calibrated_transition(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None,
        dt_ps: torch.Tensor | float,
        temperature_k: torch.Tensor | float | None = None,
        *,
        history_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        calibration = self.physical_calibration
        if not calibration.enabled:
            raise ValueError("physical transition calibration is disabled")
        if q.ndim != 2 or q.shape[-1] != self.config.state_dim:
            raise ValueError("q must have shape [batch, state_dim]")
        dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
        if dt.ndim == 0:
            dt = dt.expand(q.shape[0])
        if dt.shape != (q.shape[0],) or torch.any(dt <= 0.0):
            raise ValueError("dt_ps must be positive scalar or shape [batch]")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.config.reference_temperature_k,
        )
        raw_gradient = -self.potential.mean_force(
            q, condition, temperature, create_graph=True,
        )
        correction_gradient = calibration.correction_gradient(
            q, condition, create_graph=True,
        )
        diffusion = self.diffusion(
            q.shape[0], temperature, reference=q,
        ) * calibration.mobility_scale
        if calibration.conservative_rescale_with_free_energy:
            total_gradient = calibration.free_energy_scale * (
                raw_gradient + correction_gradient
            )
        else:
            total_gradient = (
                calibration.free_energy_scale * raw_gradient + correction_gradient
            )
        mean = q - torch.einsum("bij,bj->bi", diffusion, total_gradient) * dt[:, None]
        if history_features is not None and self.history_closure is not None:
            mean = mean + self.history_correction(
                q, history_features, dt, temperature,
            )
        return {
            "mean": mean, "diffusion": diffusion,
            "gradient": total_gradient, "raw_gradient": raw_gradient,
            "correction_gradient": correction_gradient,
            "temperature_k": temperature, "dt_ps": dt,
            "mobility_scale": calibration.mobility_scale,
            "free_energy_scale": calibration.free_energy_scale,
        }

    def calibrated_mean_force(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        calibration = self.physical_calibration
        if not calibration.enabled:
            raise ValueError("physical transition calibration is disabled")
        if q.ndim != 2 or q.shape[-1] != self.config.state_dim:
            raise ValueError("q must have shape [batch, state_dim]")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.config.reference_temperature_k,
        )
        base_force = self.potential.mean_force(
            q, condition, temperature, create_graph=create_graph,
        )
        correction_gradient = calibration.correction_gradient(
            q, condition, create_graph=create_graph,
        )
        if calibration.conservative_rescale_with_free_energy:
            return calibration.free_energy_scale * (base_force - correction_gradient)
        return calibration.free_energy_scale * base_force - correction_gradient

    def step(
        self,
        q: torch.Tensor,
        auxiliary: torch.Tensor,
        condition: torch.Tensor | None,
        dt_ps: float | torch.Tensor,
        temperature_k: torch.Tensor | float | None = None,
        *,
        stochastic: bool = True,
        use_generic: bool = True,
        history_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if q.ndim != 2 or q.shape[-1] != self.config.state_dim:
            raise ValueError("q must have shape [batch, state_dim]")
        if auxiliary.shape != (q.shape[0], self.config.auxiliary_dim):
            raise ValueError("auxiliary must have shape [batch, auxiliary_dim]")
        dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
        if dt.ndim == 0:
            dt = dt.expand(q.shape[0])
        if dt.shape != (q.shape[0],) or torch.any(dt <= 0.0):
            raise ValueError("dt_ps must be positive scalar or shape [batch]")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.config.reference_temperature_k,
        )
        gradient = -self.potential.mean_force(
            q, condition, temperature, create_graph=self.training,
        )
        coefficients = self.coefficients(q, temperature)
        diffusion = coefficients["diffusion"]
        coupling = coefficients["coupling"]
        rates = coefficients["rates"]
        del use_generic
        q_drift = -torch.einsum("bij,bj->bi", diffusion, gradient)
        q_drift = q_drift + torch.einsum(
            "bqa,ba->bq", coupling.expand(q.shape[0], -1, -1), auxiliary
        )
        # THM-JEPA: add reversible (divergence-free) drift
        if self.skew_operator is not None:
            reversible_gradient = self.reversible_potential.mean_force(
                q, condition, temperature, create_graph=self.training,
            )
            reversible_drift = self.skew_operator.apply(reversible_gradient)
            q_drift = q_drift + reversible_drift
        auxiliary_drift = -auxiliary * rates.unsqueeze(0)
        auxiliary_drift = auxiliary_drift - torch.einsum(
            "bqa,bq->ba", coupling.expand(q.shape[0], -1, -1), gradient,
        )
        q_noise = q.new_zeros(q.shape)
        auxiliary_noise = auxiliary.new_zeros(auxiliary.shape)
        if stochastic:
            q_covariance = 2.0 * diffusion * dt[:, None, None]
            cholesky = torch.linalg.cholesky(q_covariance)
            q_noise = torch.einsum("bi,bji->bj", torch.randn_like(q), cholesky)
            auxiliary_std = (2.0 * rates.unsqueeze(0) * dt[:, None]).sqrt()
            auxiliary_noise = torch.randn_like(auxiliary) * auxiliary_std
        next_q = q + q_drift * dt[:, None] + q_noise
        if history_features is not None and self.history_closure is not None:
            raise ValueError(
                "history closure is disabled in the canonical FDT path; "
                "use a validated memory state instead"
            )
        next_auxiliary = auxiliary + auxiliary_drift * dt[:, None] + auxiliary_noise
        reversible_info = {}
        if self.skew_operator is not None:
            reversible_info["reversible_gradient"] = reversible_gradient
            reversible_info["reversible_drift"] = reversible_drift
        return {
            "q": next_q, "auxiliary": next_auxiliary,
            "q_drift": q_drift, "auxiliary_drift": auxiliary_drift,
            "gradient": gradient, "diffusion": diffusion,
            "coupling": coupling, "rates": rates,
            "temperature_k": temperature,
            **reversible_info,
        }

    def step_midpoint(
        self,
        q: torch.Tensor,
        auxiliary: torch.Tensor,
        condition: torch.Tensor | None,
        dt_ps: float | torch.Tensor,
        temperature_k: torch.Tensor | float | None = None,
        *,
        stochastic: bool = True,
        history_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """BAOAB-style midpoint step (Rev-1)."""
        if q.ndim != 2 or q.shape[-1] != self.config.state_dim:
            raise ValueError("q must have shape [batch, state_dim]")
        if auxiliary.shape != (q.shape[0], self.config.auxiliary_dim):
            raise ValueError("auxiliary must have shape [batch, auxiliary_dim]")
        dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
        if dt.ndim == 0:
            dt = dt.expand(q.shape[0])
        if dt.shape != (q.shape[0],) or torch.any(dt <= 0.0):
            raise ValueError("dt_ps must be positive scalar or shape [batch]")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.config.reference_temperature_k,
        )
        coefficients = self.coefficients(q, temperature)
        diffusion = coefficients["diffusion"]
        coupling = coefficients["coupling"]
        rates = coefficients["rates"]
        dt_expand = dt[:, None, None]
        # B/2: half-step diffusion noise
        if stochastic:
            q_cov = 2.0 * diffusion * dt_expand
            chol = torch.linalg.cholesky(q_cov)
            noise_b1 = torch.einsum("bi,bji->bj", torch.randn_like(q), chol)
            q_half = q + 0.5 * noise_b1
        else:
            q_half = q
        # A: full-step drift evaluated at MIDPOINT
        gradient = -self.potential.mean_force(
            q_half, condition, temperature, create_graph=self.training,
        )
        q_drift = -torch.einsum("bij,bj->bi", diffusion, gradient)
        q_drift = q_drift + torch.einsum(
            "bqa,ba->bq", coupling.expand(q.shape[0], -1, -1), auxiliary
        )
        if self.skew_operator is not None:
            reversible_gradient = self.reversible_potential.mean_force(
                q_half, condition, temperature, create_graph=self.training,
            )
            reversible_drift = self.skew_operator.apply(reversible_gradient)
            q_drift = q_drift + reversible_drift
        auxiliary_drift = -auxiliary * rates.unsqueeze(0)
        auxiliary_drift = auxiliary_drift - torch.einsum(
            "bqa,bq->ba", coupling.expand(q.shape[0], -1, -1), gradient,
        )
        q_full = q_half + q_drift * dt[:, None]
        auxiliary_full = auxiliary + auxiliary_drift * dt[:, None]
        # B/2: second half-step diffusion
        if stochastic:
            noise_b2 = torch.einsum("bi,bji->bj", torch.randn_like(q), chol)
            next_q = q_full + 0.5 * noise_b2
            auxiliary_std = (2.0 * rates.unsqueeze(0) * dt[:, None]).sqrt()
            next_auxiliary = auxiliary_full + torch.randn_like(auxiliary) * auxiliary_std
        else:
            next_q = q_full
            next_auxiliary = auxiliary_full
        if history_features is not None and self.history_closure is not None:
            raise ValueError(
                "history closure is disabled in the canonical FDT path; "
                "use a validated memory state instead"
            )
        reversible_info = {}
        if self.skew_operator is not None:
            reversible_info["reversible_gradient"] = reversible_gradient
            reversible_info["reversible_drift"] = reversible_drift
        return {
            "q": next_q, "auxiliary": next_auxiliary,
            "q_drift": q_drift, "auxiliary_drift": auxiliary_drift,
            "gradient": gradient, "diffusion": diffusion,
            "coupling": coupling, "rates": rates,
            "temperature_k": temperature,
            **reversible_info,
        }

    def rollout(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None,
        dt_ps: float,
        steps: int,
        temperature_k: torch.Tensor | float | None = None,
        *,
        stochastic: bool = True,
        auxiliary: torch.Tensor | None = None,
        use_generic: bool = True,
        history_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if steps < 1:
            raise ValueError("steps must be positive")
        if auxiliary is None:
            auxiliary = q.new_zeros(q.shape[0], self.config.auxiliary_dim)
        states = [q]
        auxiliaries = [auxiliary]
        current_q, current_auxiliary = q, auxiliary
        step_fn = self.step_midpoint if self.config.midpoint_integration else self.step
        for _ in range(steps):
            if self.config.midpoint_integration:
                result = step_fn(
                    current_q, current_auxiliary, condition,
                    dt_ps, temperature_k, stochastic=stochastic,
                    history_features=history_features,
                )
            else:
                result = step_fn(
                    current_q, current_auxiliary, condition,
                    dt_ps, temperature_k, stochastic=stochastic,
                    use_generic=use_generic,
                    history_features=history_features,
                )
            current_q = result["q"]
            current_auxiliary = result["auxiliary"]
            states.append(current_q)
            auxiliaries.append(current_auxiliary)
        return {
            "q": torch.stack(states, dim=1),
            "auxiliary": torch.stack(auxiliaries, dim=1),
        }

    def rollout_trajectory_nll(
        self,
        q_seq: torch.Tensor,
        condition: torch.Tensor | None,
        dt_ps: float | torch.Tensor,
        temperature_k: torch.Tensor | float | None = None,
        *,
        auxiliary_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Joint NLL of an observed trajectory under K-step model rollout."""
        batch, seq_len, d = q_seq.shape
        if seq_len < 2:
            raise ValueError("q_seq must have at least 2 frames for trajectory NLL")
        device, dtype = q_seq.device, q_seq.dtype
        if auxiliary_init is not None:
            auxiliary = auxiliary_init
        else:
            auxiliary = q_seq.new_zeros(batch, self.config.auxiliary_dim)
        dt = torch.as_tensor(dt_ps, device=device, dtype=dtype)
        if dt.ndim == 0:
            dt = dt.expand(batch)
        temperature = _as_batch_temperature(
            temperature_k, batch, q_seq, self.config.reference_temperature_k,
        )
        step_fn = self.step_midpoint if self.config.midpoint_integration else self.step
        total_nll = q_seq.new_zeros(())
        for t in range(1, seq_len):
            result = step_fn(
                q_seq[:, t - 1, :].contiguous(), auxiliary, condition,
                dt, temperature, stochastic=False, history_features=None,
            )
            q_pred_mean = result["q"]
            diffusion = result["diffusion"]
            auxiliary = result["auxiliary"]
            d_dim = d
            diag_mask = torch.eye(d_dim, device=device, dtype=torch.bool).unsqueeze(0)
            diag_vals = diffusion[diag_mask.expand_as(diffusion)].view(batch, d_dim)
            cov_diag = 2.0 * diag_vals * dt[:, None].clamp(min=1e-6, max=1e4)
            residual = q_seq[:, t, :] - q_pred_mean
            res_scaled = residual / cov_diag.clamp(min=1e-8).sqrt()
            res_sq = res_scaled.square()
            soft_sq = (1.0 + res_sq).sqrt() - 1.0
            quadratic = soft_sq.sum(dim=-1)
            logdet = cov_diag.clamp(min=1e-12, max=1e4).log().sum(dim=-1)
            step_nll = 0.5 * (quadratic + logdet + d_dim * math.log(2.0 * math.pi))
            step_nll = step_nll.clamp(max=50.0)
            total_nll = total_nll + step_nll.mean()
        return total_nll / (seq_len - 1)

    def semigroup_consistency_loss(
        self,
        source: torch.Tensor,
        target_dt: torch.Tensor,
        target_2dt: torch.Tensor,
        ridge: float = 1.0e-6,
    ) -> torch.Tensor:
        """Enforce T(2Δt) ≈ T(Δt)² via the Koopman operator."""
        source_c = source - source.mean(dim=0, keepdim=True)
        target_dt_c = target_dt - target_dt.mean(dim=0, keepdim=True)
        target_2dt_c = target_2dt - target_2dt.mean(dim=0, keepdim=True)
        C_00 = (source_c.T @ source_c) / (source.shape[0] - 1) + ridge * torch.eye(
            source.shape[-1], device=source.device, dtype=source.dtype,
        )
        C_01 = (source_c.T @ target_dt_c) / (source.shape[0] - 1)
        C_02 = (source_c.T @ target_2dt_c) / (source.shape[0] - 1)
        C_00_inv = torch.linalg.solve(
            C_00, torch.eye(source.shape[-1], device=source.device, dtype=source.dtype),
        )
        K_dt = C_00_inv @ C_01
        K_2dt = C_00_inv @ C_02
        return (K_2dt - K_dt @ K_dt).square().sum()

    def get_active_phase(self, step: int) -> str:
        phases = self.config.curriculum_phases
        thresholds = self.config.curriculum_phase_steps
        for i in range(len(phases) - 1, -1, -1):
            if step >= thresholds[i]:
                return phases[i]
        return phases[0]

    def compute_phase_losses(
        self,
        q: torch.Tensor,
        target_q: torch.Tensor,
        condition: torch.Tensor | None,
        dt_ps: float | torch.Tensor,
        temperature_k: torch.Tensor | float | None = None,
        history_features: torch.Tensor | None = None,
        *,
        phase: str = "dynamics",
        koopman_source: torch.Tensor | None = None,
        koopman_target: torch.Tensor | None = None,
        vamp_ridge: float = 1.0e-6,
        q_seq: torch.Tensor | None = None,
        auxiliary_init: torch.Tensor | None = None,
        target_q_2dt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = q.device
        losses = {}
        phase_map = {
            "representation": set(),
            "koopman": {"koopman"},
            "thermodynamic_identification": {"force", "dynamics"},
            "dynamics": {"dynamics"},
            "observation": {"observation"},
        }
        active = phase_map.get(phase, set())

        if "koopman" in active and koopman_source is not None and koopman_target is not None:
            losses["lspwm_koopman_loss"] = vamp2_loss(
                koopman_source, koopman_target, ridge=vamp_ridge,
            )

        if "force" in active:
            reference = q.detach()
            if reference.shape[0] >= 2:
                mean_force_target = kde_mean_force(
                    q.detach(), reference, bandwidth=1.5,
                )
                force_loss = mean_force_matching_loss(
                    self.potential, q, mean_force_target, condition, temperature_k,
                )
                losses["lspwm_force_loss"] = force_loss

        if "dynamics" in active:
            dyn_loss = overdamped_transition_nll(
                self, q, target_q, condition, dt_ps, temperature_k,
                history_features=history_features,
            )
            losses["lspwm_dynamic_nll"] = dyn_loss

        if self.config.rollout_trajectory_steps > 0 and q_seq is not None:
            if q_seq.shape[1] >= 2:
                trajectory_nll = self.rollout_trajectory_nll(
                    q_seq, condition, dt_ps, temperature_k,
                    auxiliary_init=auxiliary_init,
                )
                losses["lspwm_trajectory_nll"] = trajectory_nll

        if self.config.semigroup_consistency_weight > 0.0 and target_q_2dt is not None:
            semigroup_loss = self.semigroup_consistency_loss(
                q, target_q, target_q_2dt, ridge=vamp_ridge,
            )
            losses["lspwm_semigroup_loss"] = semigroup_loss

        if "observation" in active and self.observation_model is not None:
            if q.shape[0] >= 2:
                cycle = asymmetric_cycle_loss(
                    self.observation_model, q, target_q, q, target_q,
                )
                losses["lspwm_observation_loss"] = cycle

        return losses
