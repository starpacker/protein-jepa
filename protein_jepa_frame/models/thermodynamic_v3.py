from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.thermodynamic import (
    K_B_KCAL_PER_MOL_K,
    FixedBackboneCollectiveVariables,
    ProteinConditionEncoder,
    _last_valid,
)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _nth_last_valid(values: torch.Tensor, mask: torch.Tensor, offset: int) -> torch.Tensor:
    """Return the offset-th valid item from the end, falling back to the first."""
    valid_count = mask.long().sum(dim=1)
    index = (valid_count - 1 - int(offset)).clamp(min=0)
    gather_index = index[:, None, None].expand(-1, 1, values.shape[-1])
    return values.gather(1, gather_index).squeeze(1)


def _recent_history_features(
    values: torch.Tensor,
    mask: torch.Tensor,
    time_ps: torch.Tensor,
    *,
    base_step_ps: float,
) -> torch.Tensor:
    """Causal velocity/acceleration summary for Mori projected-force closure."""
    last = _nth_last_valid(values, mask, 0)
    prev = _nth_last_valid(values, mask, 1)
    prev2 = _nth_last_valid(values, mask, 2)
    first = _nth_last_valid(values.flip(dims=(1,)), mask.flip(dims=(1,)), 0)

    t_last = _nth_last_valid(time_ps[..., None], mask, 0).squeeze(-1)
    t_prev = _nth_last_valid(time_ps[..., None], mask, 1).squeeze(-1)
    t_prev2 = _nth_last_valid(time_ps[..., None], mask, 2).squeeze(-1)
    t_first = _nth_last_valid(
        time_ps.flip(dims=(1,))[..., None],
        mask.flip(dims=(1,)),
        0,
    ).squeeze(-1)

    scale = values.new_tensor(float(base_step_ps))
    dt = (t_last - t_prev).abs().clamp(min=1.0)
    dt_prev = (t_prev - t_prev2).abs().clamp(min=1.0)
    dt_total = (t_last - t_first).abs().clamp(min=1.0)
    velocity = (last - prev) * (scale / dt[:, None])
    previous_velocity = (prev - prev2) * (scale / dt_prev[:, None])
    acceleration = velocity - previous_velocity
    drift = (last - first) * (scale / dt_total[:, None])

    enough_two = mask.long().sum(dim=1) >= 2
    enough_three = mask.long().sum(dim=1) >= 3
    velocity = torch.where(enough_two[:, None], velocity, torch.zeros_like(velocity))
    drift = torch.where(enough_two[:, None], drift, torch.zeros_like(drift))
    acceleration = torch.where(
        enough_three[:, None],
        acceleration,
        torch.zeros_like(acceleration),
    )
    return torch.cat([velocity, acceleration, drift], dim=-1)


def _latent_history_features(
    frame_states: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Summarize causal JEPA states at local and clip-wide time scales."""
    last = _nth_last_valid(frame_states, mask, 0)
    previous = _nth_last_valid(frame_states, mask, 1)
    weights = mask.to(frame_states.dtype).unsqueeze(-1)
    global_mean = (frame_states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    valid_rank = mask.long().cumsum(dim=1)
    valid_count = mask.long().sum(dim=1, keepdim=True)
    recent_mask = mask & (valid_rank > (valid_count - 4).clamp(min=0))
    recent_weights = recent_mask.to(frame_states.dtype).unsqueeze(-1)
    recent_mean = (frame_states * recent_weights).sum(dim=1) / recent_weights.sum(
        dim=1
    ).clamp(min=1.0)
    has_previous = mask.long().sum(dim=1) >= 2
    local_increment = torch.where(
        has_previous[:, None],
        last - previous,
        torch.zeros_like(last),
    )
    return torch.cat(
        [last, last - recent_mean, recent_mean - global_mean, local_increment],
        dim=-1,
    )


class HeatCapacityFreeEnergy(nn.Module):
    """Temperature-consistent PMF with a constant heat-capacity correction."""

    def __init__(
        self,
        cv_dim: int,
        condition_dim: int,
        reference_temperature_k: float = 348.0,
        heat_capacity_scale_over_kb: float = 5.0,
    ):
        super().__init__()
        if reference_temperature_k <= 0.0 or heat_capacity_scale_over_kb < 0.0:
            raise ValueError("TM-JEPA V3 thermodynamic scales are invalid")
        hidden_dim = max(64, condition_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        self.heat_capacity_scale_over_kb = float(heat_capacity_scale_over_kb)
        self.trunk = nn.Sequential(
            nn.Linear(cv_dim + condition_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.reference_enthalpy_over_kbt_head = nn.Linear(hidden_dim, 1)
        self.reference_entropy_over_kb_head = nn.Linear(hidden_dim, 1)
        self.heat_capacity_over_kb_head = nn.Linear(hidden_dim, 1)
        for head in (
            self.reference_enthalpy_over_kbt_head,
            self.reference_entropy_over_kb_head,
            self.heat_capacity_over_kb_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        q: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([q, condition], dim=-1))
        enthalpy = self.reference_enthalpy_over_kbt_head(hidden).squeeze(-1)
        entropy = self.reference_entropy_over_kb_head(hidden).squeeze(-1)
        heat_capacity = self.heat_capacity_scale_over_kb * torch.tanh(
            self.heat_capacity_over_kb_head(hidden).squeeze(-1)
        )
        ratio = temperature_k / self.reference_temperature_k
        heat_capacity_basis = 1.0 - ratio.reciprocal() - ratio.log()
        beta_free_energy = (
            ratio.reciprocal() * enthalpy
            - entropy
            + heat_capacity * heat_capacity_basis
        )
        free_energy = beta_free_energy * K_B_KCAL_PER_MOL_K * temperature_k
        entropy_at_temperature = entropy + heat_capacity * ratio.log()
        return {
            "free_energy": free_energy,
            "beta_free_energy": beta_free_energy,
            "reference_enthalpy_over_kbt": enthalpy,
            "entropy_over_kb": entropy_at_temperature,
            "heat_capacity_over_kb": heat_capacity,
        }


class EyringEinsteinKinetics(nn.Module):
    """PSD mobility and passive Mori kernel with structured temperature clocks."""

    def __init__(
        self,
        cv_dim: int,
        condition_dim: int,
        auxiliary_dim: int,
        diffusion_rank: int,
        memory_enabled: bool,
        diffusion_floor: float,
        diffusion_scale: float,
        coupling_scale: float,
        reference_temperature_k: float,
        activation_energy_kcal_per_mol: float,
        activation_deviation_kcal_per_mol: float,
        state_mobility_scale: float,
        memory_times_ps: tuple[float, ...],
        coupled_bath_enabled: bool = False,
        observation_noise_enabled: bool = False,
        observation_noise_scale: float = 0.1,
        observation_noise_rank: int = 0,
        observation_noise_condition_scale: float = 0.0,
        observation_noise_temperature_exponent: float = 1.0,
        diffusion_floor_scale: tuple[float, ...] = (),
    ):
        super().__init__()
        if len(memory_times_ps) != auxiliary_dim:
            raise ValueError("TM-JEPA V3 needs one identifiable time per Mori mode")
        if not 1 <= diffusion_rank <= cv_dim:
            raise ValueError("TM-JEPA V3 diffusion rank must be between 1 and cv_dim")
        if any(value <= 0.0 for value in memory_times_ps):
            raise ValueError("TM-JEPA V3 memory times must be positive")
        if not 0 <= observation_noise_rank <= cv_dim:
            raise ValueError(
                "TM-JEPA V3 observation-noise rank must be between 0 and cv_dim"
            )
        if observation_noise_condition_scale < 0.0:
            raise ValueError(
                "TM-JEPA V3 observation-noise condition scale must be non-negative"
            )
        if diffusion_floor_scale and (
            len(diffusion_floor_scale) != cv_dim
            or any(value <= 0.0 for value in diffusion_floor_scale)
        ):
            raise ValueError(
                "TM-JEPA V3 diffusion-floor scales must be positive and match cv_dim"
            )
        self.cv_dim = int(cv_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.diffusion_rank = int(diffusion_rank)
        self.memory_enabled = bool(memory_enabled)
        self.coupled_bath_enabled = bool(coupled_bath_enabled)
        self.observation_noise_enabled = bool(observation_noise_enabled)
        self.memory_coupling_warmup_steps = 0
        self.observation_noise_scale = float(observation_noise_scale)
        self.observation_noise_rank = int(observation_noise_rank)
        self.observation_noise_condition_scale = float(
            observation_noise_condition_scale
        )
        self.observation_noise_temperature_exponent = float(
            observation_noise_temperature_exponent
        )
        self.diffusion_floor = float(diffusion_floor)
        self.register_buffer(
            "diffusion_floor_scale",
            torch.tensor(
                diffusion_floor_scale or (1.0,) * cv_dim,
                dtype=torch.float32,
            ),
        )
        self.diffusion_scale = float(diffusion_scale)
        self.coupling_scale = float(coupling_scale)
        self.reference_temperature_k = float(reference_temperature_k)
        self.activation_deviation_kcal_per_mol = float(
            activation_deviation_kcal_per_mol
        )
        self.state_mobility_scale = float(state_mobility_scale)
        self.register_buffer(
            "reference_relaxation_rates",
            torch.tensor(memory_times_ps, dtype=torch.float32).reciprocal(),
        )

        self.mobility_head = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, 3 * cv_dim + diffusion_rank),
        )
        self.memory_head = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(
                condition_dim * 2,
                cv_dim * auxiliary_dim + 2 * auxiliary_dim,
            ),
        )
        self.observation_noise_head = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, cv_dim + observation_noise_rank),
        )
        for head in (
            self.mobility_head,
            self.memory_head,
            self.observation_noise_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.mobility_head[-1].bias[:cv_dim].fill_(_inverse_softplus(0.1))
            mode_start = 3 * cv_dim
            self.mobility_head[-1].bias[
                mode_start : mode_start + diffusion_rank
            ].fill_(_inverse_softplus(0.1))
            amplitude_start = cv_dim * auxiliary_dim
            self.memory_head[-1].bias[
                amplitude_start : amplitude_start + auxiliary_dim
            ].fill_(math.atanh(0.1))

        self.diffusion_basis = nn.Parameter(
            torch.randn(cv_dim, diffusion_rank) * 0.05
        )
        self.coupling_vectors = nn.Parameter(
            torch.randn(cv_dim, auxiliary_dim) * 0.2
        )
        self.register_buffer("_memory_coupling_ramp", torch.ones(1))
        activation_fraction = activation_energy_kcal_per_mol / 10.0
        if not 0.0 < activation_fraction < 1.0:
            raise ValueError("TM-JEPA V3 activation energy must be between 0 and 10 kcal/mol")
        self.activation_energy_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(activation_fraction))
        )
        self.memory_activation_energy_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(activation_fraction))
        )
        self.diagonal_activation_deviation = nn.Parameter(torch.zeros(cv_dim))
        self.mode_activation_deviation = nn.Parameter(torch.zeros(diffusion_rank))
        self.memory_activation_deviation = nn.Parameter(torch.zeros(auxiliary_dim))
        self.memory_rotation_raw = nn.Parameter(
            torch.randn(auxiliary_dim, auxiliary_dim) * 0.1
        )
        self.observation_noise_raw = nn.Parameter(
            torch.full((cv_dim,), _inverse_softplus(0.1))
        )
        self.observation_noise_mode_raw = nn.Parameter(
            torch.full((observation_noise_rank,), _inverse_softplus(0.01))
        )
        self.observation_noise_basis = nn.Parameter(
            torch.randn(cv_dim, observation_noise_rank) * 0.05
        )

    def _activation_energy(
        self,
        deviation: torch.Tensor,
        global_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_raw is None:
            global_raw = self.activation_energy_raw
        global_energy = 10.0 * torch.sigmoid(global_raw)
        return global_energy + self.activation_deviation_kcal_per_mol * torch.tanh(
            deviation
        )

    def _clock(
        self, temperature_k: torch.Tensor, activation_energy: torch.Tensor
    ) -> torch.Tensor:
        inverse_difference = (
            1.0 / self.reference_temperature_k - 1.0 / temperature_k[..., None]
        )
        exponent = activation_energy * inverse_difference / K_B_KCAL_PER_MOL_K
        return (
            temperature_k[..., None] / self.reference_temperature_k
        ) * torch.exp(exponent.clamp(min=-6.0, max=6.0))

    def forward(
        self,
        q: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mobility_raw = self.mobility_head(condition)
        memory_raw = self.memory_head(condition)
        Q, R, M = self.cv_dim, self.auxiliary_dim, self.diffusion_rank
        (
            raw_diagonal,
            raw_linear,
            raw_quadratic,
            raw_modes,
        ) = torch.split(mobility_raw, [Q, Q, Q, M], dim=-1)
        (
            raw_coupling_delta,
            raw_coupling_amplitude,
            raw_relaxation_adjustment,
        ) = torch.split(memory_raw, [Q * R, R, R], dim=-1)

        bounded_q = torch.tanh(q)
        linear = self.state_mobility_scale * torch.tanh(raw_linear)
        quadratic = self.state_mobility_scale * torch.tanh(raw_quadratic)
        diagonal_logits = (
            raw_diagonal
            + linear * bounded_q
            + quadratic * bounded_q.square()
        )
        diagonal_energy = self._activation_energy(
            self.diagonal_activation_deviation
        )
        mode_energy = self._activation_energy(self.mode_activation_deviation)
        diagonal_clock = self._clock(temperature_k, diagonal_energy)
        mode_clock = self._clock(temperature_k, mode_energy)
        diagonal_rates = (
            self.diffusion_floor * self.diffusion_floor_scale.to(q)
            + F.softplus(diagonal_logits) * self.diffusion_scale
        ) * diagonal_clock
        mode_rates = F.softplus(raw_modes) * self.diffusion_scale * mode_clock

        derivative_logits = (
            linear + 2.0 * quadratic * bounded_q
        ) * (1.0 - bounded_q.square())
        diagonal_divergence = (
            torch.sigmoid(diagonal_logits)
            * self.diffusion_scale
            * diagonal_clock
            * derivative_logits
        )
        diffusion_basis = torch.linalg.qr(self.diffusion_basis, mode="reduced")[0]
        diffusion = torch.diag_embed(diagonal_rates) + torch.einsum(
            "qr,br,sr->bqs", diffusion_basis, mode_rates, diffusion_basis
        )

        coupling_delta = raw_coupling_delta.reshape(-1, Q, R)
        coupling_direction = F.normalize(
            self.coupling_vectors.unsqueeze(0) + 0.1 * coupling_delta,
            dim=1,
            eps=1.0e-8,
        )
        memory_energy = self._activation_energy(
            self.memory_activation_deviation,
            self.memory_activation_energy_raw,
        )
        memory_clock = self._clock(temperature_k, memory_energy)
        coupling_amplitude = (
            torch.tanh(raw_coupling_amplitude)
            * self.coupling_scale
            * memory_clock
        )
        if not self.memory_enabled:
            coupling_amplitude = torch.zeros_like(coupling_amplitude)
        coupling_amplitude = coupling_amplitude * self._memory_coupling_ramp
        coupling = coupling_direction * coupling_amplitude[:, None, :]
        relaxation = self.reference_relaxation_rates.to(q).unsqueeze(0) * torch.exp(
            0.3 * torch.tanh(raw_relaxation_adjustment)
        )
        relaxation = relaxation * memory_clock
        rotation_raw = torch.tanh(self.memory_rotation_raw)
        rotation_raw = 0.5 * (rotation_raw - rotation_raw.T)
        rotation_scale = 2.0 * torch.sqrt(
            relaxation[:, :, None] * relaxation[:, None, :]
        )
        rotation = rotation_scale * rotation_raw.unsqueeze(0)
        if not self.coupled_bath_enabled or not self.memory_enabled:
            rotation = torch.zeros_like(rotation)
        dissipation = torch.diag_embed(relaxation)
        auxiliary_generator = dissipation + rotation
        observation_adjustment = self.observation_noise_condition_scale * torch.tanh(
            self.observation_noise_head(condition)
        )
        diagonal_adjustment = observation_adjustment[:, :Q]
        mode_adjustment = observation_adjustment[:, Q:]
        temperature_factor = (
            temperature_k / self.reference_temperature_k
        ).clamp(min=0.5, max=2.0).pow(
            self.observation_noise_temperature_exponent
        )
        observation_variance = (
            F.softplus(self.observation_noise_raw + diagonal_adjustment)
            * self.observation_noise_scale
            * temperature_factor[:, None]
        )
        if self.observation_noise_rank:
            observation_basis = torch.linalg.qr(
                self.observation_noise_basis, mode="reduced"
            )[0]
            observation_mode_variance = (
                F.softplus(self.observation_noise_mode_raw + mode_adjustment)
                * self.observation_noise_scale
                * temperature_factor[:, None]
            )
            observation_covariance = torch.diag_embed(
                observation_variance
            ) + torch.einsum(
                "qr,br,sr->bqs",
                observation_basis,
                observation_mode_variance,
                observation_basis,
            )
        else:
            observation_basis = q.new_zeros(Q, 0)
            observation_mode_variance = q.new_zeros(q.shape[0], 0)
            observation_covariance = torch.diag_embed(observation_variance)
        if not self.observation_noise_enabled:
            observation_variance = torch.zeros_like(observation_variance)
            observation_mode_variance = torch.zeros_like(observation_mode_variance)
            observation_covariance = torch.zeros_like(observation_covariance)
        return {
            "diffusion": diffusion,
            "diffusion_divergence": diagonal_divergence,
            "diagonal_diffusion_rates": diagonal_rates,
            "diffusion_rates": mode_rates,
            "diffusion_basis": diffusion_basis,
            "coupling": coupling,
            "coupling_basis": coupling_direction,
            "coupling_rates": coupling_amplitude,
            "relaxation_rates": relaxation,
            "auxiliary_generator": auxiliary_generator,
            "auxiliary_noise_rate": 2.0 * dissipation,
            "auxiliary_rotation": rotation,
            "observation_noise_covariance": observation_covariance,
            "observation_noise_diagonal_variance": observation_variance,
            "observation_noise_mode_variance": observation_mode_variance,
            "observation_noise_basis": observation_basis,
            "diagonal_activation_energy": diagonal_energy.expand(q.shape[0], -1),
            "mode_activation_energy": mode_energy.expand(q.shape[0], -1),
            "memory_activation_energy": memory_energy.expand(q.shape[0], -1),
            "memory_clock": memory_clock,
        }


class ThermodynamicMoriDynamicsV3(nn.Module):
    """Itô-consistent thermodynamic GLE with a physical Bayesian observer."""

    def __init__(
        self,
        cv_dim: int,
        condition_dim: int,
        auxiliary_dim: int = 5,
        diffusion_rank: int = 4,
        memory_enabled: bool = True,
        coupled_bath_enabled: bool = False,
        observation_noise_enabled: bool = False,
        observation_noise_scale: float = 0.1,
        observation_noise_rank: int = 0,
        observation_noise_condition_scale: float = 0.0,
        observation_noise_temperature_exponent: float = 1.0,
        reference_temperature_k: float = 348.0,
        activation_energy_kcal_per_mol: float = 3.0,
        activation_deviation_kcal_per_mol: float = 1.0,
        state_mobility_scale: float = 0.5,
        memory_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
        base_step_ps: float = 250.0,
        max_rollout_steps: int = 256,
        ensemble_size: int = 8,
        diffusion_floor: float = 1.0e-8,
        diffusion_scale: float = 1.0e-5,
        coupling_scale: float = 1.0e-3,
        equilibrium_noise_std: float = 0.05,
        equilibrium_noise_samples: int = 1,
        heat_capacity_scale_over_kb: float = 5.0,
        diffusion_floor_scale: tuple[float, ...] = (),
    ):
        super().__init__()
        if (
            base_step_ps <= 0.0
            or max_rollout_steps <= 0
            or ensemble_size < 2
            or equilibrium_noise_samples < 1
        ):
            raise ValueError("TM-JEPA V3 integration settings are invalid")
        self.cv_dim = int(cv_dim)
        self.condition_dim = int(condition_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.memory_enabled = bool(memory_enabled)
        self.base_step_ps = float(base_step_ps)
        self.max_rollout_steps = int(max_rollout_steps)
        self.ensemble_size = int(ensemble_size)
        self.equilibrium_noise_std = float(equilibrium_noise_std)
        self.equilibrium_noise_samples = int(equilibrium_noise_samples)
        self.potential = HeatCapacityFreeEnergy(
            cv_dim,
            condition_dim,
            reference_temperature_k=reference_temperature_k,
            heat_capacity_scale_over_kb=heat_capacity_scale_over_kb,
        )
        self.kinetics = EyringEinsteinKinetics(
            cv_dim,
            condition_dim,
            auxiliary_dim,
            diffusion_rank,
            memory_enabled,
            diffusion_floor,
            diffusion_scale,
            coupling_scale,
            reference_temperature_k,
            activation_energy_kcal_per_mol,
            activation_deviation_kcal_per_mol,
            state_mobility_scale,
            memory_times_ps,
            coupled_bath_enabled,
            observation_noise_enabled,
            observation_noise_scale,
            observation_noise_rank,
            observation_noise_condition_scale,
            observation_noise_temperature_exponent,
            diffusion_floor_scale,
        )

    @staticmethod
    def _temperature(
        temperature_k: torch.Tensor | None,
        batch: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if temperature_k is None:
            return reference.new_full((batch,), 300.0)
        return torch.nan_to_num(temperature_k.to(reference), nan=300.0).clamp(
            250.0, 600.0
        )

    def free_energy(
        self,
        q: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self._temperature(temperature_k, q.shape[0], q)
        values = self.potential(q, condition, temperature)
        return (
            values["free_energy"],
            values["beta_free_energy"],
            values["entropy_over_kb"],
            values["heat_capacity_over_kb"],
        )

    def _energy_gradient(
        self,
        q: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        with torch.enable_grad():
            if not q.requires_grad:
                q = q.detach().requires_grad_(True)
            values = self.potential(q, condition, temperature_k)
            gradient = torch.autograd.grad(
                values["beta_free_energy"].sum(),
                q,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
        return q, gradient, values

    def kinetic_parameters(
        self,
        condition: torch.Tensor,
        temperature_k: torch.Tensor | None,
        q: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if q is None:
            q = condition.new_zeros(condition.shape[0], self.cv_dim)
        temperature = self._temperature(temperature_k, condition.shape[0], condition)
        return self.kinetics(q, condition, temperature)

    @staticmethod
    def _bath_coefficients(
        relaxation: torch.Tensor, dt: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        dt_value = dt.unsqueeze(-1)
        rate = relaxation.clamp(min=1.0e-12)
        x = rate * dt_value
        phi = torch.exp(-x)
        one_minus = -torch.expm1(-x)
        history = one_minus / rate
        force_integral = (x - one_minus) / rate.square()
        exact_numerator = 2.0 * x - 3.0 + 4.0 * phi - phi.square()
        series_numerator = (
            (2.0 / 3.0) * x.pow(3)
            - 0.5 * x.pow(4)
            + (7.0 / 30.0) * x.pow(5)
        )
        integral_variance = torch.where(
            x.abs() < 0.1, series_numerator, exact_numerator
        ) / rate.square()
        cross_covariance = one_minus.square() / rate
        auxiliary_variance = 1.0 - phi.square()
        return {
            "phi": phi,
            "history": history,
            "force_integral": force_integral,
            "integral_variance": integral_variance.clamp(min=0.0),
            "cross_covariance": cross_covariance.clamp(min=0.0),
            "auxiliary_variance": auxiliary_variance.clamp(min=0.0),
        }

    def _coupled_bath_transition(
        self,
        auxiliary: torch.Tensor,
        diffusion: torch.Tensor,
        coupling: torch.Tensor,
        generator: torch.Tensor,
        auxiliary_noise_rate: torch.Tensor,
        base_drift: torch.Tensor,
        force: torch.Tensor,
        dt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Exact affine-Gaussian step for frozen local q coefficients."""
        batch = auxiliary.shape[0]
        q_dim = self.cv_dim
        a_dim = self.auxiliary_dim
        state_dim = q_dim + a_dim
        drift_matrix = auxiliary.new_zeros(batch, state_dim, state_dim)
        drift_matrix[:, :q_dim, q_dim:] = coupling
        drift_matrix[:, q_dim:, q_dim:] = -generator
        affine_drift = torch.cat([base_drift, force], dim=-1)
        noise_rate = auxiliary.new_zeros(batch, state_dim, state_dim)
        noise_rate[:, :q_dim, :q_dim] = 2.0 * diffusion
        noise_rate[:, q_dim:, q_dim:] = auxiliary_noise_rate

        # Van Loan block exponential returns F, integral exp(A s)Q ds, and
        # the affine mean integral in one operation.
        van_loan = auxiliary.new_zeros(batch, 2 * state_dim + 1, 2 * state_dim + 1)
        van_loan[:, :state_dim, :state_dim] = drift_matrix
        van_loan[:, :state_dim, state_dim : 2 * state_dim] = noise_rate
        van_loan[:, :state_dim, -1] = affine_drift
        van_loan[:, state_dim : 2 * state_dim, state_dim : 2 * state_dim] = (
            -drift_matrix.transpose(-1, -2)
        )
        exponential = torch.matrix_exp(van_loan * dt[:, None, None])
        transition = exponential[:, :state_dim, :state_dim]
        covariance_factor = exponential[:, :state_dim, state_dim : 2 * state_dim]
        process_covariance = covariance_factor @ transition.transpose(-1, -2)
        process_covariance = 0.5 * (
            process_covariance + process_covariance.transpose(-1, -2)
        )
        affine_offset = exponential[:, :state_dim, -1]

        history_matrix = transition[:, :q_dim, q_dim:]
        auxiliary_transition = transition[:, q_dim:, q_dim:]
        deterministic_q_increment = affine_offset[:, :q_dim]
        deterministic_auxiliary_offset = affine_offset[:, q_dim:]
        return {
            "history_matrix": history_matrix,
            "auxiliary_transition_matrix": auxiliary_transition,
            "deterministic_q_increment": deterministic_q_increment,
            "deterministic_auxiliary_offset": deterministic_auxiliary_offset,
            "q_mean_increment": deterministic_q_increment
            + torch.einsum("bqr,br->bq", history_matrix, auxiliary),
            "auxiliary_mean": deterministic_auxiliary_offset
            + torch.einsum("brs,bs->br", auxiliary_transition, auxiliary),
            "q_covariance": process_covariance[:, :q_dim, :q_dim],
            "q_auxiliary_covariance": process_covariance[:, :q_dim, q_dim:],
            "auxiliary_covariance": process_covariance[:, q_dim:, q_dim:],
        }

    def _local_transition(
        self,
        q: torch.Tensor,
        auxiliary: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
        dt: torch.Tensor,
        *,
        create_graph: bool,
    ) -> dict[str, torch.Tensor]:
        q_leaf, gradient, potential = self._energy_gradient(
            q, condition, temperature_k, create_graph=create_graph
        )
        kinetic = self.kinetics(q_leaf, condition, temperature_k)
        diffusion = kinetic["diffusion"]
        coupling = kinetic["coupling"]
        relaxation = kinetic["relaxation_rates"]
        force = -torch.einsum("bqr,bq->br", coupling, gradient)
        base_drift = -torch.einsum("bij,bj->bi", diffusion, gradient)
        base_drift = base_drift + kinetic["diffusion_divergence"]
        if self.kinetics.coupled_bath_enabled:
            coupled = self._coupled_bath_transition(
                auxiliary,
                diffusion,
                coupling,
                kinetic["auxiliary_generator"],
                kinetic["auxiliary_noise_rate"],
                base_drift,
                force,
                dt,
            )
            return {
                "q": q_leaf,
                "gradient": gradient,
                "potential": potential,
                "kinetic": kinetic,
                **coupled,
            }

        coefficients = self._bath_coefficients(relaxation, dt)
        history_matrix = coupling * coefficients["history"].unsqueeze(1)
        force_displacement = torch.einsum(
            "bqr,br,br->bq", coupling, coefficients["force_integral"], force
        )
        q_mean_increment = (
            base_drift * dt.unsqueeze(-1)
            + torch.einsum("bqr,br->bq", history_matrix, auxiliary)
            + force_displacement
        )
        auxiliary_mean = (
            coefficients["phi"] * auxiliary
            + coefficients["history"] * force
        )
        q_covariance = 2.0 * dt[:, None, None] * diffusion
        q_covariance = q_covariance + torch.einsum(
            "bqr,br,bsr->bqs",
            coupling,
            coefficients["integral_variance"],
            coupling,
        )
        q_auxiliary_covariance = coupling * coefficients[
            "cross_covariance"
        ].unsqueeze(1)
        auxiliary_covariance = torch.diag_embed(
            coefficients["auxiliary_variance"]
        )
        auxiliary_transition = torch.diag_embed(coefficients["phi"])
        deterministic_q_increment = (
            base_drift * dt.unsqueeze(-1) + force_displacement
        )
        deterministic_auxiliary_offset = coefficients["history"] * force
        return {
            "q": q_leaf,
            "gradient": gradient,
            "potential": potential,
            "kinetic": kinetic,
            "history_matrix": history_matrix,
            "auxiliary_transition_matrix": auxiliary_transition,
            "deterministic_q_increment": deterministic_q_increment,
            "deterministic_auxiliary_offset": deterministic_auxiliary_offset,
            "force_displacement": force_displacement,
            "base_displacement": base_drift * dt.unsqueeze(-1),
            "q_mean_increment": q_mean_increment,
            "auxiliary_mean": auxiliary_mean,
            "q_covariance": q_covariance,
            "q_auxiliary_covariance": q_auxiliary_covariance,
            "auxiliary_covariance": auxiliary_covariance,
            **coefficients,
        }

    def filter(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        time_ps: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.kinetics.observation_noise_enabled:
            return self._filter_with_observation_noise(
                sequence, mask, time_ps, condition, temperature_k
            )
        B, T, _ = sequence.shape
        mean = sequence.new_zeros(B, self.auxiliary_dim)
        covariance = torch.eye(
            self.auxiliary_dim, device=sequence.device, dtype=sequence.dtype
        ).expand(B, -1, -1).clone()
        means = [mean]
        covariances = [covariance]
        innovations = []
        path_nll = []
        pair_mask = mask[:, 1:] & mask[:, :-1]
        identity_q = torch.eye(
            self.cv_dim, device=sequence.device, dtype=sequence.dtype
        ).unsqueeze(0)
        identity_a = torch.eye(
            self.auxiliary_dim, device=sequence.device, dtype=sequence.dtype
        ).unsqueeze(0)
        for step in range(T - 1):
            dt = (time_ps[:, step + 1] - time_ps[:, step]).abs().clamp(
                min=1.0e-6
            )
            local = self._local_transition(
                sequence[:, step],
                mean,
                condition,
                temperature_k,
                dt,
                create_graph=self.training,
            )
            history_matrix = local["history_matrix"]
            auxiliary_transition = local["auxiliary_transition_matrix"]
            observed_increment = sequence[:, step + 1] - sequence[:, step]
            predicted_increment = (
                local["deterministic_q_increment"]
                + torch.einsum("bqr,br->bq", history_matrix, mean)
            )
            residual = observed_increment - predicted_increment
            innovation_covariance = local["q_covariance"] + torch.einsum(
                "bqr,brs,bks->bqk", history_matrix, covariance, history_matrix
            )
            innovation_covariance = innovation_covariance + 1.0e-8 * identity_q
            cholesky = torch.linalg.cholesky(innovation_covariance)
            standardized = torch.cholesky_solve(
                residual.unsqueeze(-1), cholesky
            ).squeeze(-1)
            whitened = torch.linalg.solve_triangular(
                cholesky, residual.unsqueeze(-1), upper=False
            ).squeeze(-1)
            log_determinant = 2.0 * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(dim=-1)
            nll = 0.5 * (
                whitened.square().sum(dim=-1)
                + log_determinant
                + self.cv_dim * math.log(2.0 * math.pi)
            ) / float(self.cv_dim)

            predicted_auxiliary_mean = (
                torch.einsum("brs,bs->br", auxiliary_transition, mean)
                + local["deterministic_auxiliary_offset"]
            )
            predicted_auxiliary_covariance = (
                auxiliary_transition
                @ covariance
                @ auxiliary_transition.transpose(-1, -2)
                + local["auxiliary_covariance"]
            )
            cross_auxiliary_observation = torch.einsum(
                "brs,bsk,bqk->brq",
                auxiliary_transition,
                covariance,
                history_matrix,
            ) + local["q_auxiliary_covariance"].transpose(-1, -2)
            kalman_gain = torch.cholesky_solve(
                cross_auxiliary_observation.transpose(-1, -2), cholesky
            ).transpose(-1, -2)
            updated_mean = predicted_auxiliary_mean + torch.einsum(
                "brq,bq->br", kalman_gain, residual
            )
            updated_covariance = predicted_auxiliary_covariance - torch.einsum(
                "brq,bqs,bks->brk",
                kalman_gain,
                innovation_covariance,
                kalman_gain,
            )
            updated_covariance = 0.5 * (
                updated_covariance + updated_covariance.transpose(-1, -2)
            ) + 1.0e-7 * identity_a
            valid = pair_mask[:, step]
            mean = torch.where(valid[:, None], updated_mean, mean)
            covariance = torch.where(
                valid[:, None, None], updated_covariance, covariance
            )
            means.append(mean)
            covariances.append(covariance)
            innovations.append(torch.where(valid[:, None], whitened, 0.0))
            path_nll.append(torch.where(valid, nll, 0.0))
        return {
            "mean_sequence": torch.stack(means, dim=1),
            "covariance_sequence": torch.stack(covariances, dim=1),
            "innovation": torch.stack(innovations, dim=1),
            "path_nll": torch.stack(path_nll, dim=1),
            "mask": pair_mask,
            "observation_error_mean_sequence": sequence.new_zeros(
                B, T, self.cv_dim
            ),
            "observation_error_covariance_sequence": sequence.new_zeros(
                B, T, self.cv_dim, self.cv_dim
            ),
        }

    def _filter_with_observation_noise(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
        time_ps: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Kalman closure for a physical bath and additive frame noise."""
        batch, steps, _ = sequence.shape
        a_dim, q_dim = self.auxiliary_dim, self.cv_dim
        state_dim = a_dim + q_dim
        auxiliary_mean = sequence.new_zeros(batch, a_dim)
        error_mean = sequence.new_zeros(batch, q_dim)
        initial_kinetic = self.kinetic_parameters(
            condition, temperature_k, sequence[:, 0]
        )
        observation_covariance = initial_kinetic["observation_noise_covariance"]
        covariance = sequence.new_zeros(batch, state_dim, state_dim)
        covariance[:, :a_dim, :a_dim] = torch.eye(
            a_dim, device=sequence.device, dtype=sequence.dtype
        )
        covariance[:, a_dim:, a_dim:] = observation_covariance
        auxiliary_means = [auxiliary_mean]
        auxiliary_covariances = [covariance[:, :a_dim, :a_dim]]
        error_means = [error_mean]
        error_covariances = [covariance[:, a_dim:, a_dim:]]
        innovations = []
        path_nll = []
        pair_mask = mask[:, 1:] & mask[:, :-1]
        identity_q = torch.eye(
            q_dim, device=sequence.device, dtype=sequence.dtype
        ).unsqueeze(0)
        identity_state = torch.eye(
            state_dim, device=sequence.device, dtype=sequence.dtype
        ).unsqueeze(0)

        for step in range(steps - 1):
            dt = (time_ps[:, step + 1] - time_ps[:, step]).abs().clamp(
                min=1.0e-6
            )
            latent_q = sequence[:, step] - error_mean
            local = self._local_transition(
                latent_q,
                auxiliary_mean,
                condition,
                temperature_k,
                dt,
                create_graph=self.training,
            )
            history = local["history_matrix"]
            auxiliary_transition = local["auxiliary_transition_matrix"]
            observation_covariance = local["kinetic"][
                "observation_noise_covariance"
            ]
            observation_matrix = torch.cat(
                [history, -identity_q.expand(batch, -1, -1)], dim=-1
            )
            observed_increment = sequence[:, step + 1] - sequence[:, step]
            predicted_increment = (
                local["deterministic_q_increment"]
                + torch.einsum("bqr,br->bq", history, auxiliary_mean)
                - error_mean
            )
            residual = observed_increment - predicted_increment
            observation_process_covariance = (
                local["q_covariance"] + observation_covariance
            )
            innovation_covariance = (
                observation_process_covariance
                + observation_matrix
                @ covariance
                @ observation_matrix.transpose(-1, -2)
                + 1.0e-8 * identity_q
            )
            cholesky = torch.linalg.cholesky(innovation_covariance)
            whitened = torch.linalg.solve_triangular(
                cholesky, residual.unsqueeze(-1), upper=False
            ).squeeze(-1)
            log_determinant = 2.0 * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(dim=-1)
            nll = 0.5 * (
                whitened.square().sum(dim=-1)
                + log_determinant
                + q_dim * math.log(2.0 * math.pi)
            ) / float(q_dim)

            state_transition = sequence.new_zeros(
                batch, state_dim, state_dim
            )
            state_transition[:, :a_dim, :a_dim] = auxiliary_transition
            predicted_state_mean = torch.cat(
                [
                    local["auxiliary_mean"],
                    sequence.new_zeros(batch, q_dim),
                ],
                dim=-1,
            )
            state_process_covariance = sequence.new_zeros(
                batch, state_dim, state_dim
            )
            state_process_covariance[:, :a_dim, :a_dim] = local[
                "auxiliary_covariance"
            ]
            state_process_covariance[:, a_dim:, a_dim:] = observation_covariance
            predicted_state_covariance = (
                state_transition
                @ covariance
                @ state_transition.transpose(-1, -2)
                + state_process_covariance
            )
            process_cross = torch.cat(
                [
                    local["q_auxiliary_covariance"].transpose(-1, -2),
                    observation_covariance,
                ],
                dim=1,
            )
            state_observation_cross = (
                state_transition
                @ covariance
                @ observation_matrix.transpose(-1, -2)
                + process_cross
            )
            kalman_gain = torch.cholesky_solve(
                state_observation_cross.transpose(-1, -2), cholesky
            ).transpose(-1, -2)
            updated_mean = predicted_state_mean + torch.einsum(
                "brq,bq->br", kalman_gain, residual
            )
            updated_covariance = predicted_state_covariance - (
                kalman_gain
                @ innovation_covariance
                @ kalman_gain.transpose(-1, -2)
            )
            updated_covariance = 0.5 * (
                updated_covariance + updated_covariance.transpose(-1, -2)
            ) + 1.0e-7 * identity_state
            valid = pair_mask[:, step]
            state_mean = torch.cat([auxiliary_mean, error_mean], dim=-1)
            state_mean = torch.where(valid[:, None], updated_mean, state_mean)
            covariance = torch.where(
                valid[:, None, None], updated_covariance, covariance
            )
            auxiliary_mean = state_mean[:, :a_dim]
            error_mean = state_mean[:, a_dim:]
            auxiliary_means.append(auxiliary_mean)
            auxiliary_covariances.append(covariance[:, :a_dim, :a_dim])
            error_means.append(error_mean)
            error_covariances.append(covariance[:, a_dim:, a_dim:])
            innovations.append(torch.where(valid[:, None], whitened, 0.0))
            path_nll.append(torch.where(valid, nll, 0.0))

        return {
            "mean_sequence": torch.stack(auxiliary_means, dim=1),
            "covariance_sequence": torch.stack(auxiliary_covariances, dim=1),
            "observation_error_mean_sequence": torch.stack(error_means, dim=1),
            "observation_error_covariance_sequence": torch.stack(
                error_covariances, dim=1
            ),
            "innovation": torch.stack(innovations, dim=1),
            "path_nll": torch.stack(path_nll, dim=1),
            "mask": pair_mask,
        }

    def rollout(
        self,
        q: torch.Tensor,
        auxiliary_mean: torch.Tensor,
        auxiliary_logvar: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor | None,
        horizon_time_ps: torch.Tensor,
        *,
        auxiliary_covariance: torch.Tensor | None = None,
        ensemble_size: int | None = None,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensemble = int(ensemble_size or self.ensemble_size)
        B = q.shape[0]
        temperature = self._temperature(temperature_k, B, q)
        if auxiliary_covariance is None:
            auxiliary_covariance = torch.diag_embed(auxiliary_logvar.exp())
        covariance_cholesky = torch.linalg.cholesky(
            auxiliary_covariance
            + 1.0e-8
            * torch.eye(self.auxiliary_dim, device=q.device, dtype=q.dtype)
        )
        initial_noise = torch.randn(
            B, ensemble, self.auxiliary_dim, device=q.device, dtype=q.dtype
        ) if stochastic else q.new_zeros(B, ensemble, self.auxiliary_dim)
        a_state = auxiliary_mean[:, None] + torch.einsum(
            "bij,bkj->bki", covariance_cholesky, initial_noise
        )
        q_state = q[:, None].expand(-1, ensemble, -1).clone()
        q_state = q_state.reshape(B * ensemble, self.cv_dim)
        a_state = a_state.reshape(B * ensemble, self.auxiliary_dim)
        condition_e = condition[:, None].expand(-1, ensemble, -1).reshape(
            B * ensemble, -1
        )
        temperature_e = temperature[:, None].expand(-1, ensemble).reshape(-1)
        total_time = horizon_time_ps.to(q).clamp(min=0.0)
        steps = torch.ceil(total_time / self.base_step_ps).long()
        max_steps = int(steps.max().detach().cpu())
        if max_steps > self.max_rollout_steps:
            raise ValueError(
                f"TM-JEPA V3 horizon needs {max_steps} steps, above "
                f"max_rollout_steps={self.max_rollout_steps}"
            )
        elapsed = torch.zeros_like(total_time)
        identity = torch.eye(
            self.cv_dim + self.auxiliary_dim,
            device=q.device,
            dtype=q.dtype,
        ).unsqueeze(0)
        for _ in range(max_steps):
            dt = torch.minimum(
                (total_time - elapsed).clamp(min=0.0),
                total_time.new_full(total_time.shape, self.base_step_ps),
            )
            dt_e = dt[:, None].expand(-1, ensemble).reshape(-1)
            local = self._local_transition(
                q_state,
                a_state,
                condition_e,
                temperature_e,
                dt_e,
                create_graph=self.training,
            )
            joint_covariance = torch.cat(
                [
                    torch.cat(
                        [local["q_covariance"], local["q_auxiliary_covariance"]],
                        dim=-1,
                    ),
                    torch.cat(
                        [
                            local["q_auxiliary_covariance"].transpose(-1, -2),
                            local["auxiliary_covariance"],
                        ],
                        dim=-1,
                    ),
                ],
                dim=-2,
            )
            cholesky = torch.linalg.cholesky(joint_covariance + 1.0e-8 * identity)
            noise = torch.randn(
                B * ensemble,
                self.cv_dim + self.auxiliary_dim,
                device=q.device,
                dtype=q.dtype,
            ) if stochastic else q.new_zeros(
                B * ensemble, self.cv_dim + self.auxiliary_dim
            )
            noise = torch.einsum("bij,bj->bi", cholesky, noise)
            active = dt_e > 0.0
            q_candidate = q_state + local["q_mean_increment"] + noise[:, : self.cv_dim]
            a_candidate = local["auxiliary_mean"] + noise[:, self.cv_dim :]
            q_state = torch.where(active[:, None], q_candidate, q_state)
            a_state = torch.where(active[:, None], a_candidate, a_state)
            elapsed = elapsed + dt
        return (
            q_state.reshape(B, ensemble, self.cv_dim),
            a_state.reshape(B, ensemble, self.auxiliary_dim),
        )


class EndpointVarianceCalibrator(nn.Module):
    """Bounded effective-noise rescaling for finite-horizon CV rollouts."""

    def __init__(
        self,
        cv_dim: int,
        *,
        enabled: bool = False,
        max_log_scale: float = 0.0,
        temperature_log_scale: float = 0.0,
        reference_temperature_k: float = 348.0,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.max_log_scale = float(max_log_scale)
        self.temperature_log_scale = float(temperature_log_scale)
        self.reference_temperature_k = float(reference_temperature_k)
        self.bias = nn.Parameter(torch.zeros(cv_dim))
        self.log_horizon_slope = nn.Parameter(torch.zeros(cv_dim))
        self.short_horizon_slope = nn.Parameter(torch.zeros(cv_dim))
        self.temperature_slope = nn.Parameter(torch.zeros(cv_dim))
        self.short_temperature_slope = nn.Parameter(torch.zeros(cv_dim))

    def forward(
        self,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        shape = (horizon_time_ps.shape[0], self.bias.shape[0])
        if not self.enabled or self.max_log_scale <= 0.0:
            return reference.new_zeros(shape)
        horizon_feature = torch.log1p(
            horizon_time_ps.to(reference).clamp(min=0.0) / 1000.0
        )[:, None]
        short_feature = torch.exp(
            -horizon_time_ps.to(reference).clamp(min=0.0) / 2000.0
        )[:, None]
        temperature_feature = (
            temperature_k.to(reference) / self.reference_temperature_k
        ).log()[:, None]
        raw = (
            self.bias
            + self.log_horizon_slope * horizon_feature
            + self.short_horizon_slope * short_feature
        )
        if self.temperature_log_scale > 0.0:
            raw = raw + (
                self.temperature_slope
                * temperature_feature
                * self.temperature_log_scale
            ) + (
                self.short_temperature_slope
                * temperature_feature
                * short_feature
                * self.temperature_log_scale
            )
        return self.max_log_scale * torch.tanh(raw)


class ProjectedForceCorrector(nn.Module):
    """Bounded history-conditioned mean correction for unresolved slow forces."""

    def __init__(
        self,
        cv_dim: int,
        condition_dim: int,
        auxiliary_dim: int,
        *,
        enabled: bool = False,
        hidden_dim: int = 128,
        max_shift: float = 0.25,
        history_feature_dim: int = 0,
        horizon_gate_steps: float = 0.0,
        reference_temperature_k: float = 348.0,
        base_step_ps: float = 1000.0,
    ):
        super().__init__()
        if (
            hidden_dim <= 0
            or max_shift < 0.0
            or reference_temperature_k <= 0.0
            or base_step_ps <= 0.0
            or horizon_gate_steps < 0.0
        ):
            raise ValueError("projected-force corrector scales are invalid")
        self.enabled = bool(enabled)
        self.max_shift = float(max_shift)
        self.history_feature_dim = int(history_feature_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        self.base_step_ps = float(base_step_ps)
        self.horizon_gate_steps = float(horizon_gate_steps)
        input_dim = cv_dim + auxiliary_dim + self.history_feature_dim + condition_dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cv_dim),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        q: torch.Tensor,
        auxiliary_mean: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        history_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled or self.max_shift == 0.0:
            return q.new_zeros(q.shape)
        if history_features is None:
            history_features = q.new_zeros(q.shape[0], self.history_feature_dim)
        elif history_features.shape[-1] != self.history_feature_dim:
            raise ValueError("projected-force history feature dimension mismatch")
        horizon = (horizon_time_ps.to(q) / self.base_step_ps).clamp(min=0.0)
        temperature = temperature_k.to(q).clamp(min=1.0)
        features = torch.cat(
            [
                q,
                auxiliary_mean,
                history_features.to(q),
                condition,
                torch.log1p(horizon).unsqueeze(-1),
                torch.sqrt(horizon + 1.0e-6).unsqueeze(-1),
                (temperature / self.reference_temperature_k - 1.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        correction = self.max_shift * torch.tanh(self.net(features))
        if self.horizon_gate_steps > 0.0:
            # A coordinate response to an unresolved force is O(t^2) at short time.
            gate = -torch.expm1(-horizon / self.horizon_gate_steps)
            correction = correction * gate.square().unsqueeze(-1)
        return correction


class ThermodynamicMoriJEPAV3(nn.Module):
    """Fixed-measure CVs, temperature law, Bayesian Mori state, and rollout."""

    def __init__(
        self,
        latent_dim: int,
        num_segments: int = 3,
        learned_cv_dim: int = 0,
        condition_dim: int = 64,
        auxiliary_dim: int = 5,
        diffusion_rank: int = 4,
        memory_enabled: bool = True,
        coupled_bath_enabled: bool = False,
        observation_noise_enabled: bool = False,
        observation_noise_scale: float = 0.1,
        observation_noise_rank: int = 0,
        observation_noise_condition_scale: float = 0.0,
        observation_noise_temperature_exponent: float = 1.0,
        memory_intervention_enabled: bool = False,
        reference_temperature_k: float = 348.0,
        activation_energy_kcal_per_mol: float = 3.0,
        activation_deviation_kcal_per_mol: float = 1.0,
        state_mobility_scale: float = 0.5,
        memory_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
        base_step_ps: float = 250.0,
        max_rollout_steps: int = 256,
        ensemble_size: int = 8,
        diffusion_floor: float = 1.0e-8,
        diffusion_scale: float = 1.0e-5,
        coupling_scale: float = 1.0e-3,
        equilibrium_noise_std: float = 0.05,
        equilibrium_noise_samples: int = 1,
        heat_capacity_scale_over_kb: float = 5.0,
        learned_cv_diffusion_floor_scale: float = 1.0,
        endpoint_calibration_enabled: bool = False,
        endpoint_calibration_scale: float = 0.0,
        endpoint_calibration_temperature_scale: float = 0.0,
        projected_force_enabled: bool = False,
        projected_force_hidden_dim: int = 128,
        projected_force_max_shift: float = 0.25,
        projected_force_history_features: bool = False,
        projected_force_latent_history_features: bool = False,
        projected_force_horizon_gate_steps: float = 0.0,
        learned_cv_burn_in_frames: int = 0,
        contact_cutoff_angstrom: float = 8.0,
        contact_width_angstrom: float = 0.75,
        cv_mean: tuple[float, ...] = (),
        cv_scale: tuple[float, ...] = (),
    ):
        super().__init__()
        if (
            learned_cv_dim < 0
            or learned_cv_burn_in_frames < 0
            or learned_cv_diffusion_floor_scale <= 0.0
        ):
            raise ValueError("TM-JEPA V3 learned-CV settings must be non-negative")
        self.learned_cv_dim = int(learned_cv_dim)
        self.learned_cv_burn_in_frames = int(learned_cv_burn_in_frames)
        self.memory_intervention_enabled = bool(memory_intervention_enabled)
        self.projected_force_history_features = bool(projected_force_history_features)
        self.projected_force_latent_history_features = bool(
            projected_force_latent_history_features
        )
        self.collective_variables = FixedBackboneCollectiveVariables(
            num_segments=num_segments,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            contact_width_angstrom=contact_width_angstrom,
        )
        self.fixed_cv_dim = self.collective_variables.output_dim
        self.cv_dim = self.fixed_cv_dim + self.learned_cv_dim
        self.condition_encoder = ProteinConditionEncoder(condition_dim)
        self.learned_projection = (
            nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, self.learned_cv_dim),
                nn.Tanh(),
            )
            if self.learned_cv_dim
            else None
        )
        self.target_learned_projection = copy.deepcopy(self.learned_projection)
        if self.target_learned_projection is not None:
            for parameter in self.target_learned_projection.parameters():
                parameter.requires_grad = False
        self.dynamics = ThermodynamicMoriDynamicsV3(
            self.cv_dim,
            condition_dim,
            auxiliary_dim=auxiliary_dim,
            diffusion_rank=diffusion_rank,
            memory_enabled=memory_enabled,
            coupled_bath_enabled=coupled_bath_enabled,
            observation_noise_enabled=observation_noise_enabled,
            observation_noise_scale=observation_noise_scale,
            observation_noise_rank=observation_noise_rank,
            observation_noise_condition_scale=(
                observation_noise_condition_scale
            ),
            observation_noise_temperature_exponent=(
                observation_noise_temperature_exponent
            ),
            reference_temperature_k=reference_temperature_k,
            activation_energy_kcal_per_mol=activation_energy_kcal_per_mol,
            activation_deviation_kcal_per_mol=activation_deviation_kcal_per_mol,
            state_mobility_scale=state_mobility_scale,
            memory_times_ps=memory_times_ps,
            base_step_ps=base_step_ps,
            max_rollout_steps=max_rollout_steps,
            ensemble_size=ensemble_size,
            diffusion_floor=diffusion_floor,
            diffusion_scale=diffusion_scale,
            coupling_scale=coupling_scale,
            equilibrium_noise_std=equilibrium_noise_std,
            equilibrium_noise_samples=equilibrium_noise_samples,
            heat_capacity_scale_over_kb=heat_capacity_scale_over_kb,
            diffusion_floor_scale=(
                (1.0,) * self.fixed_cv_dim
                + (float(learned_cv_diffusion_floor_scale),) * self.learned_cv_dim
            ),
        )
        self.endpoint_calibration = EndpointVarianceCalibrator(
            self.cv_dim,
            enabled=endpoint_calibration_enabled,
            max_log_scale=endpoint_calibration_scale,
            temperature_log_scale=endpoint_calibration_temperature_scale,
            reference_temperature_k=reference_temperature_k,
        )
        self.projected_force_corrector = ProjectedForceCorrector(
            self.cv_dim,
            condition_dim,
            auxiliary_dim,
            enabled=projected_force_enabled,
            hidden_dim=projected_force_hidden_dim,
            max_shift=projected_force_max_shift,
            history_feature_dim=(
                (3 * self.cv_dim if projected_force_history_features else 0)
                + (4 * latent_dim if projected_force_latent_history_features else 0)
            ),
            horizon_gate_steps=projected_force_horizon_gate_steps,
            reference_temperature_k=reference_temperature_k,
            base_step_ps=base_step_ps,
        )
        allowed_stat_dims = {self.fixed_cv_dim, self.cv_dim}
        if cv_mean and len(cv_mean) not in allowed_stat_dims:
            raise ValueError("TM-JEPA V3 cv_mean has the wrong dimension")
        if cv_scale and len(cv_scale) not in allowed_stat_dims:
            raise ValueError("TM-JEPA V3 cv_scale has the wrong dimension")
        mean = torch.tensor(cv_mean, dtype=torch.float32) if cv_mean else torch.zeros(self.cv_dim)
        scale = torch.tensor(cv_scale, dtype=torch.float32) if cv_scale else torch.ones(self.cv_dim)
        if mean.numel() == self.fixed_cv_dim and self.learned_cv_dim:
            mean = torch.cat([mean, torch.zeros(self.learned_cv_dim)])
        if scale.numel() == self.fixed_cv_dim and self.learned_cv_dim:
            scale = torch.cat([scale, torch.ones(self.learned_cv_dim)])
        if torch.any(scale <= 0.0):
            raise ValueError("TM-JEPA V3 CV scales must be positive")
        self.register_buffer("cv_mean", mean)
        self.register_buffer("cv_scale", scale)

    @torch.no_grad()
    def sync_target_projection(self) -> None:
        if self.learned_projection is not None and self.target_learned_projection is not None:
            self.target_learned_projection.load_state_dict(
                self.learned_projection.state_dict()
            )

    @torch.no_grad()
    def update_target_projection(self, momentum: float) -> None:
        if self.learned_projection is None or self.target_learned_projection is None:
            return
        for target, online in zip(
            self.target_learned_projection.parameters(),
            self.learned_projection.parameters(),
        ):
            target.data.mul_(momentum).add_(
                online.detach().data, alpha=1.0 - momentum
            )

    def train(self, mode: bool = True) -> ThermodynamicMoriJEPAV3:
        super().train(mode)
        if self.target_learned_projection is not None:
            self.target_learned_projection.eval()
        return self

    def standardize(self, q: torch.Tensor) -> torch.Tensor:
        return (q - self.cv_mean.to(q)) / self.cv_scale.to(q).clamp(min=1.0e-6)

    def unstandardize(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.cv_scale.to(q) + self.cv_mean.to(q)

    def observe_samples(
        self,
        latent_samples: torch.Tensor,
        observation_covariance: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the configured endpoint observation model to latent samples."""
        if not self.dynamics.kinetics.observation_noise_enabled:
            return latent_samples
        identity = torch.eye(
            self.cv_dim,
            device=latent_samples.device,
            dtype=latent_samples.dtype,
        )
        cholesky = torch.linalg.cholesky(
            observation_covariance + 1.0e-8 * identity
        )
        noise = torch.einsum(
            "bij,bkj->bki", cholesky, torch.randn_like(latent_samples)
        )
        return latent_samples + noise

    def encode_learned_context(
        self,
        context_frame_states: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Project a stationary-history JEPA sequence without running dynamics."""
        if self.learned_projection is None:
            raise ValueError("learned-context encoding requires learned CVs")
        learned = self.learned_projection(context_frame_states)
        if self.learned_cv_burn_in_frames:
            if self.learned_cv_burn_in_frames >= learned.shape[1]:
                raise ValueError(
                    "learned-CV burn-in must leave at least one context frame"
                )
            learned = learned[:, self.learned_cv_burn_in_frames :]
            context_mask = context_mask[:, self.learned_cv_burn_in_frames :]
        return {
            "tmjepa_learned_cv": learned,
            "tmjepa_context_mask": context_mask,
        }

    def forward(
        self,
        *,
        context_coords: torch.Tensor,
        future_coords: torch.Tensor,
        context_atom_mask: torch.Tensor,
        future_atom_mask: torch.Tensor,
        context_residue_mask: torch.Tensor,
        future_residue_mask: torch.Tensor,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        context_time_ps: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None,
        dataset_index: torch.Tensor | None,
        context_frame_states: torch.Tensor | None = None,
        future_frame_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_raw = self.collective_variables(
            context_coords,
            context_atom_mask,
            context_residue_mask,
            chain_id,
            residue_index,
        )
        future_raw = self.collective_variables(
            future_coords,
            future_atom_mask,
            future_residue_mask,
            chain_id,
            residue_index,
        )
        context_cv = (
            context_raw - self.cv_mean[: self.fixed_cv_dim].to(context_raw)
        ) / self.cv_scale[: self.fixed_cv_dim].to(context_raw).clamp(min=1.0e-6)
        future_cv = (
            future_raw - self.cv_mean[: self.fixed_cv_dim].to(future_raw)
        ) / self.cv_scale[: self.fixed_cv_dim].to(future_raw).clamp(min=1.0e-6)
        if self.learned_projection is not None:
            if context_frame_states is None or future_frame_states is None:
                raise ValueError(
                    "TM-JEPA V3 learned CVs require context and future frame states"
                )
            context_learned = self.learned_projection(context_frame_states)
            with torch.no_grad():
                future_learned = self.target_learned_projection(
                    future_frame_states
                )
            context_cv = torch.cat([context_cv, context_learned], dim=-1)
            future_cv = torch.cat([future_cv, future_learned], dim=-1)
        context_mask = context_residue_mask.any(dim=2)
        future_mask = future_residue_mask.any(dim=2)
        if self.learned_cv_dim and self.learned_cv_burn_in_frames:
            if self.learned_cv_burn_in_frames >= context_cv.shape[1]:
                raise ValueError(
                    "learned-CV burn-in must leave at least one context frame"
                )
            start = self.learned_cv_burn_in_frames
            context_cv = context_cv[:, start:]
            context_mask = context_mask[:, start:]
            context_time_ps = context_time_ps[:, start:]
        condition = self.condition_encoder(
            aatype,
            residue_index,
            context_residue_mask.any(dim=1),
            dataset_index,
        )
        temperature = self.dynamics._temperature(
            temperature_k, context_cv.shape[0], context_cv
        )
        filtered = self.dynamics.filter(
            context_cv, context_mask, context_time_ps, condition, temperature
        )
        observed_current = _last_valid(context_cv, context_mask)
        target = _last_valid(future_cv, future_mask).detach()
        auxiliary_mean = _last_valid(filtered["mean_sequence"], context_mask)
        auxiliary_covariance = _last_valid(
            filtered["covariance_sequence"], context_mask
        )
        auxiliary_logvar = torch.diagonal(
            auxiliary_covariance, dim1=-2, dim2=-1
        ).clamp(min=1.0e-8).log()
        observation_error_mean = _last_valid(
            filtered["observation_error_mean_sequence"], context_mask
        )
        observation_error_covariance = _last_valid(
            filtered["observation_error_covariance_sequence"], context_mask
        )
        current = observed_current - observation_error_mean
        history_parts = []
        if self.projected_force_history_features:
            history_parts.append(
                _recent_history_features(
                    context_cv,
                    context_mask,
                    context_time_ps,
                    base_step_ps=self.dynamics.base_step_ps,
                )
            )
        if self.projected_force_latent_history_features:
            if context_frame_states is None:
                raise ValueError(
                    "latent projected-force history requires causal frame states"
                )
            latent_states = context_frame_states
            latent_mask = context_residue_mask.any(dim=2)
            if self.learned_cv_dim and self.learned_cv_burn_in_frames:
                latent_states = latent_states[:, self.learned_cv_burn_in_frames :]
                latent_mask = latent_mask[:, self.learned_cv_burn_in_frames :]
            history_parts.append(_latent_history_features(latent_states, latent_mask))
        history_features = torch.cat(history_parts, dim=-1) if history_parts else None
        samples, auxiliary_samples = self.dynamics.rollout(
            current,
            auxiliary_mean,
            auxiliary_logvar,
            condition,
            temperature,
            horizon_time_ps,
            auxiliary_covariance=auxiliary_covariance,
        )
        projected_force_shift = self.projected_force_corrector(
            current,
            auxiliary_mean,
            condition,
            temperature,
            horizon_time_ps,
            history_features,
        )
        samples = samples + projected_force_shift[:, None, :]
        kinetic_at_current = self.dynamics.kinetic_parameters(
            condition, temperature, current
        )
        endpoint_observation_covariance = kinetic_at_current[
            "observation_noise_covariance"
        ]
        samples = self.observe_samples(samples, endpoint_observation_covariance)
        calibration_log_scale = self.endpoint_calibration(
            horizon_time_ps, temperature, samples
        )
        if self.endpoint_calibration.enabled:
            prediction_anchor = samples.mean(dim=1, keepdim=True)
            samples = prediction_anchor + torch.exp(
                calibration_log_scale[:, None, :]
            ) * (samples - prediction_anchor)
        if self.dynamics.memory_enabled and self.memory_intervention_enabled:
            deterministic_filtered, _ = self.dynamics.rollout(
                current,
                auxiliary_mean,
                auxiliary_logvar,
                condition,
                temperature,
                horizon_time_ps,
                auxiliary_covariance=auxiliary_covariance,
                ensemble_size=1,
                stochastic=False,
            )
            deterministic_filtered = deterministic_filtered + projected_force_shift[:, None, :]
            reset_mean = torch.zeros_like(auxiliary_mean)
            reset_covariance = torch.eye(
                self.dynamics.auxiliary_dim,
                device=current.device,
                dtype=current.dtype,
            ).expand(current.shape[0], -1, -1)
            deterministic_reset, _ = self.dynamics.rollout(
                current,
                reset_mean,
                torch.zeros_like(auxiliary_logvar),
                condition,
                temperature,
                horizon_time_ps,
                auxiliary_covariance=reset_covariance,
                ensemble_size=1,
                stochastic=False,
            )
            projected_force_reset_shift = self.projected_force_corrector(
                current,
                reset_mean,
                condition,
                temperature,
                horizon_time_ps,
                history_features,
            )
            deterministic_reset = deterministic_reset + projected_force_reset_shift[:, None, :]
            history_reset_shift = self.projected_force_corrector(
                current,
                reset_mean,
                condition,
                temperature,
                horizon_time_ps,
                None,
            )
            deterministic_history_reset, _ = self.dynamics.rollout(
                current,
                reset_mean,
                torch.zeros_like(auxiliary_logvar),
                condition,
                temperature,
                horizon_time_ps,
                auxiliary_covariance=reset_covariance,
                ensemble_size=1,
                stochastic=False,
            )
            deterministic_history_reset = (
                deterministic_history_reset + history_reset_shift[:, None, :]
            )
            deterministic_filtered = deterministic_filtered[:, 0]
            deterministic_reset = deterministic_reset[:, 0]
            deterministic_history_reset = deterministic_history_reset[:, 0]
        else:
            deterministic_filtered = current
            deterministic_reset = current
            deterministic_history_reset = current
            projected_force_reset_shift = torch.zeros_like(projected_force_shift)
            history_reset_shift = torch.zeros_like(projected_force_shift)
        prediction = samples.mean(dim=1)
        sample_variance = samples.var(dim=1, unbiased=False).clamp(min=1.0e-8)
        current_leaf, current_gradient, potential = self.dynamics._energy_gradient(
            current, condition, temperature, create_graph=self.training
        )
        equilibrium_q = observed_current.detach()
        score_samples = self.dynamics.equilibrium_noise_samples
        score_q = equilibrium_q[:, None].expand(-1, score_samples, -1)
        score_noise = (
            torch.randn_like(score_q) * self.dynamics.equilibrium_noise_std
        )
        noisy_q = (score_q + score_noise).reshape(-1, self.cv_dim)
        score_condition = condition[:, None].expand(
            -1, score_samples, -1
        ).reshape(-1, condition.shape[-1])
        score_temperature = temperature[:, None].expand(
            -1, score_samples
        ).reshape(-1)
        _, noisy_gradient, _ = self.dynamics._energy_gradient(
            noisy_q,
            score_condition,
            score_temperature,
            create_graph=self.training,
        )
        noisy_gradient = noisy_gradient.reshape_as(score_noise)
        kinetic = self.dynamics.kinetics(current_leaf, condition, temperature)
        zero_transition_kl = filtered["path_nll"].new_zeros(
            filtered["path_nll"].shape
        )
        current_raw = (
            current_leaf[..., : self.fixed_cv_dim]
            * self.cv_scale[: self.fixed_cv_dim].to(current_leaf)
            + self.cv_mean[: self.fixed_cv_dim].to(current_leaf)
        )
        physical_current = self.collective_variables.to_physical(current_raw)
        log_jacobian = self.collective_variables.log_abs_det_to_unconstrained(
            physical_current
        ) - self.cv_scale[: self.fixed_cv_dim].log().sum()
        result = {
            "tmjepa_current_cv": current_leaf,
            "tmjepa_observed_current_cv": observed_current,
            "tmjepa_target_cv": target,
            "tmjepa_samples": samples,
            "tmjepa_auxiliary_samples": auxiliary_samples,
            "tmjepa_prediction": prediction,
            "tmjepa_projected_force_shift": projected_force_shift,
            "tmjepa_projected_force_reset_shift": projected_force_reset_shift,
            "tmjepa_projected_force_history_reset_shift": history_reset_shift,
            "tmjepa_projected_force_history_features": (
                history_features
                if history_features is not None
                else current.new_zeros(current.shape[0], 0)
            ),
            "tmjepa_deterministic_filtered_prediction": deterministic_filtered,
            "tmjepa_deterministic_reset_prediction": deterministic_reset,
            "tmjepa_deterministic_history_reset_prediction": (
                deterministic_history_reset
            ),
            "tmjepa_memory_enabled": current.new_full(
                (current.shape[0],), float(self.dynamics.memory_enabled)
            ),
            "tmjepa_sample_variance": sample_variance,
            "tmjepa_free_energy": potential["free_energy"],
            "tmjepa_entropy_over_kb": potential["entropy_over_kb"],
            "tmjepa_heat_capacity_over_kb": potential["heat_capacity_over_kb"],
            "tmjepa_beta_force_gradient": current_gradient,
            "tmjepa_score_prediction": -noisy_gradient,
            "tmjepa_score_target": -score_noise / self.dynamics.equilibrium_noise_std**2,
            "tmjepa_score_base_cv": equilibrium_q,
            "tmjepa_auxiliary_mean": auxiliary_mean,
            "tmjepa_auxiliary_logvar": auxiliary_logvar,
            "tmjepa_auxiliary_covariance": auxiliary_covariance,
            "tmjepa_observation_error_mean": observation_error_mean,
            "tmjepa_observation_error_covariance": observation_error_covariance,
            "tmjepa_observation_noise_covariance": endpoint_observation_covariance,
            "tmjepa_observation_noise_diagonal_variance": kinetic[
                "observation_noise_diagonal_variance"
            ],
            "tmjepa_observation_noise_mode_variance": kinetic[
                "observation_noise_mode_variance"
            ],
            "tmjepa_endpoint_calibration_log_scale": calibration_log_scale,
            "tmjepa_auxiliary_mean_sequence": filtered["mean_sequence"],
            "tmjepa_innovation": filtered["innovation"],
            "tmjepa_innovation_mask": filtered["mask"],
            "tmjepa_path_nll": filtered["path_nll"],
            "tmjepa_auxiliary_transition_kl": zero_transition_kl,
            "tmjepa_diffusion": kinetic["diffusion"],
            "tmjepa_diffusion_divergence": kinetic["diffusion_divergence"],
            "tmjepa_diagonal_diffusion_rates": kinetic["diagonal_diffusion_rates"],
            "tmjepa_diffusion_rates": kinetic["diffusion_rates"],
            "tmjepa_coupling_rates": kinetic["coupling_rates"],
            "tmjepa_relaxation_rates": kinetic["relaxation_rates"],
            "tmjepa_auxiliary_generator": kinetic["auxiliary_generator"],
            "tmjepa_auxiliary_rotation": kinetic["auxiliary_rotation"],
            "tmjepa_mode_basis": kinetic["diffusion_basis"],
            "tmjepa_coupling_basis": kinetic["coupling_basis"],
            "tmjepa_activation_energy_kcal_per_mol": kinetic[
                "diagonal_activation_energy"
            ],
            "tmjepa_memory_activation_energy_kcal_per_mol": kinetic[
                "memory_activation_energy"
            ],
            "tmjepa_temperature_k": temperature,
            "tmjepa_base_step_ps": current.new_full(
                (current.shape[0],), self.dynamics.base_step_ps
            ),
            "tmjepa_current_physical_cv": physical_current,
            "tmjepa_target_physical_cv": self.collective_variables.to_physical(
                target[..., : self.fixed_cv_dim]
                * self.cv_scale[: self.fixed_cv_dim].to(target)
                + self.cv_mean[: self.fixed_cv_dim].to(target)
            ),
            "tmjepa_sample_physical_cv": self.collective_variables.to_physical(
                samples[..., : self.fixed_cv_dim]
                * self.cv_scale[: self.fixed_cv_dim].to(samples)
                + self.cv_mean[: self.fixed_cv_dim].to(samples)
            ),
            "tmjepa_physical_free_energy": potential["free_energy"]
            - K_B_KCAL_PER_MOL_K * temperature * log_jacobian,
            "tmjepa_context_cv_sequence": context_cv,
            "tmjepa_future_cv_sequence": future_cv,
            "tmjepa_context_mask": context_mask,
            "tmjepa_condition": condition,
            "tmjepa_v3": current.new_ones(current.shape[0]),
        }
        if self.learned_cv_dim:
            result["tmjepa_learned_cv"] = context_cv[..., -self.learned_cv_dim :]
        return result
