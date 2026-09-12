"""Canonical primitives for the Latent Statistical Physics World Model.

The module is deliberately independent from the legacy TM-JEPA V2/V3 paths.
It provides the contracts defined in the repository README: a scalar PMF, a
global constant diffusion tensor, a gauge-fixed Markovian bath, an overdamped
rollout, and empirical Koopman diagnostics.

Architecture follows the GENERIC (General Equation for Non-Equilibrium
Reversible-Irreversible Coupling) formulation from Chapter 9:

    y_dot = L(y) nabla E(y) + M(y) nabla S(y) + sqrt(2 k_B T M(y)) xi

where L is the Poisson operator (antisymmetric, reversible), M is the
dissipation operator (symmetric PSD, irreversible), E is the total energy
(PMF + bath energy), and S is the entropy.

The five canonical losses from Chapter 7 are:
    L_repr : JEPA predictive representation
    L_koop : VAMP-2 score maximisation (Koopman-compatible subspace)
    L_dyn  : GLE overdamped transition NLL
    L_force: PMF mean force matching
    L_obs  : Asymmetric cycle consistency (observation model)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.thermodynamic import (
    FixedBackboneCollectiveVariables,
    ProteinConditionEncoder,
    _last_valid,
)


def _as_batch_temperature(
    temperature_k: torch.Tensor | float | None,
    batch: int,
    reference: torch.Tensor,
    reference_temperature_k: float,
) -> torch.Tensor:
    if temperature_k is None:
        temperature = reference.new_full((batch,), reference_temperature_k)
    elif torch.is_tensor(temperature_k):
        temperature = temperature_k.to(device=reference.device, dtype=reference.dtype)
        if temperature.ndim == 0:
            temperature = temperature.expand(batch)
        if temperature.shape != (batch,):
            raise ValueError("temperature_k must be scalar or have shape [batch]")
    else:
        temperature = reference.new_full((batch,), float(temperature_k))
    if not bool(torch.isfinite(temperature).all()) or torch.any(temperature <= 0.0):
        raise ValueError("temperature_k must be finite and positive")
    return temperature.clamp(max=5000.0)


class MeanForcePotential(nn.Module):
    """A scalar dimensionless PMF whose force is always conservative.

    Temperature is an explicit conditioning variable. The feature is the
    dimensionless log-temperature ratio, so a multi-temperature dataset does
    not force all equilibrium PMFs to share one unconstrained surface.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 128,
        hidden_depth: int = 2,
        reference_temperature_k: float = 348.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or condition_dim < 0 or hidden_dim < 4 or reference_temperature_k <= 0.0:
            raise ValueError("invalid LSP-WM potential dimensions")
        if hidden_depth < 1:
            raise ValueError("hidden_depth must be at least 1")
        self.state_dim = int(state_dim)
        self.condition_dim = int(condition_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        input_dim = self.state_dim + self.condition_dim + 1
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _condition(self, q: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_dim == 0:
            return q.new_zeros(*q.shape[:-1], 0)
        if condition is None:
            return q.new_zeros(*q.shape[:-1], self.condition_dim)
        if condition.shape != (*q.shape[:-1], self.condition_dim):
            raise ValueError("condition shape does not match the potential")
        return condition.to(dtype=q.dtype, device=q.device)

    def forward(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if q.shape[-1] != self.state_dim:
            raise ValueError("q has the wrong state dimension")
        temperature = _as_batch_temperature(
            temperature_k,
            q.shape[0],
            q,
            self.reference_temperature_k,
        )
        temperature_feature = torch.log(
            temperature / self.reference_temperature_k
        ).reshape(*q.shape[:-1], 1)
        return self.network(
            torch.cat([q, self._condition(q, condition), temperature_feature], dim=-1)
        ).squeeze(-1)

    def mean_force(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``-grad A`` in the same dimensionless units as the PMF."""
        with torch.enable_grad():
            q_leaf = q if q.requires_grad else q.detach().requires_grad_(True)
            energy = self(q_leaf, condition, temperature_k)
            gradient = torch.autograd.grad(
                energy.sum(),
                q_leaf,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
        return -gradient


class ConstantDiffusion(nn.Module):
    """Global PSD diffusion tensor with the required temperature clock.

    The Cholesky factor is shared across state, protein, and condition. Only
    the scalar temperature factor changes between environments.
    """

    def __init__(
        self,
        state_dim: int,
        *,
        reference_temperature_k: float = 348.0,
        diffusion_scale: float = 1.0e-3,
        diagonal_floor: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if state_dim < 1 or reference_temperature_k <= 0.0:
            raise ValueError("invalid constant-diffusion dimensions")
        if diffusion_scale <= 0.0 or diagonal_floor <= 0.0:
            raise ValueError("diffusion scales must be positive")
        self.state_dim = int(state_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        self.diffusion_scale = float(diffusion_scale)
        self.diagonal_floor = float(diagonal_floor)
        raw = torch.eye(state_dim)
        raw.diagonal().copy_(torch.log(torch.expm1(raw.diagonal())))
        self.raw_factor = nn.Parameter(raw)

    def factor(self) -> torch.Tensor:
        lower = torch.tril(self.raw_factor, diagonal=-1)
        diagonal = F.softplus(torch.diagonal(self.raw_factor)) + self.diagonal_floor
        return lower + torch.diag(diagonal)

    def matrix(
        self,
        batch: int | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference is None:
            reference = self.raw_factor
        if batch is None:
            if torch.is_tensor(temperature_k) and temperature_k.ndim > 0:
                batch = int(temperature_k.shape[0])
            else:
                batch = 1
        temperature = _as_batch_temperature(
            temperature_k,
            batch,
            reference,
            self.reference_temperature_k,
        )
        factor = self.factor().to(dtype=reference.dtype, device=reference.device)
        base = self.diffusion_scale * (factor @ factor.T)
        return base.unsqueeze(0) * (
            temperature / self.reference_temperature_k
        ).view(-1, 1, 1)

    def forward(
        self,
        batch: int,
        temperature_k: torch.Tensor | float | None = None,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.matrix(batch, temperature_k, reference=reference)


class StateDependentDiffusion(nn.Module):
    """Heteroscedastic diffusion head: predicts per-sample diagonal from state + condition.

    Replaces the global ConstantDiffusion with a neural network that outputs
    the diffusion diagonal conditioned on the current state and protein condition.
    This allows the model to adapt its variance per sample (e.g., high variance
    for uncertain transition states, low variance for stable folded states).
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        hidden_dim: int = 128,
        *,
        reference_temperature_k: float = 348.0,
        diffusion_scale: float = 1.0e-3,
        diagonal_floor: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if state_dim < 1 or reference_temperature_k <= 0.0:
            raise ValueError("invalid state-dependent diffusion dimensions")
        if diffusion_scale <= 0.0 or diagonal_floor <= 0.0:
            raise ValueError("diffusion scales must be positive")
        self.state_dim = int(state_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        self.diffusion_scale = float(diffusion_scale)
        self.diagonal_floor = float(diagonal_floor)

        # Network predicts log-diagonal modulation factor from state only
        # (condition is protein-level info not available in the diffusion call path)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Learnable global log-amplitude (initialised so initial scale ≈ diffusion_scale)
        self.log_amplitude = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        batch: int | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, state_dim, state_dim] diagonal diffusion matrix.

        When ``reference`` is provided it is used as the state vector ``q``
        to predict per-sample diffusion. Fallback: identity × diffusion_scale.
        """
        if reference is None or reference.ndim == 1 or reference.shape[-1] != self.state_dim:
            # Fallback: return identity-scaled diagonal
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            diag = self.diffusion_scale * torch.exp(self.log_amplitude) * torch.ones(
                int(batch or 1), self.state_dim, device=device, dtype=dtype
            )
        else:
            # Predict per-sample diffusion diagonal from state
            # reference is [batch, state_dim] — only the first state_dim dims are used
            q = reference[..., :self.state_dim].contiguous()
            # Build condition placeholder: if condition_dim > 0, fall back to zeros
            # The condition is embedded in the world model, not passed here directly
            raw = self.net(q)  # [batch, state_dim]
            diag = self.diffusion_scale * torch.exp(self.log_amplitude) * (
                F.softplus(raw) + self.diagonal_floor
            )

        # Temperature scaling (same as ConstantDiffusion)
        temperature = _as_batch_temperature(
            temperature_k,
            int(batch or diag.shape[0]),
            diag,
            self.reference_temperature_k,
        )
        diag = diag * (temperature / self.reference_temperature_k).unsqueeze(-1)

        return torch.diag_embed(diag.clamp(min=1e-12))  # [batch, state_dim, state_dim]


class GaugeFixedBath(nn.Module):
    """Minimal, orthogonal and spectrally ordered bath realization."""

    def __init__(
        self,
        state_dim: int,
        auxiliary_dim: int,
        *,
        relaxation_times_ps: tuple[float, ...] = (1000.0, 2000.0, 4000.0, 8000.0),
        coupling_scale: float = 1.0e-3,
    ) -> None:
        super().__init__()
        if state_dim < 1 or auxiliary_dim < 1 or auxiliary_dim > state_dim:
            raise ValueError("bath dimension must be between 1 and state_dim")
        relaxation_times_ps = tuple(float(value) for value in relaxation_times_ps)
        if len(relaxation_times_ps) != auxiliary_dim or any(
            value <= 0.0 for value in relaxation_times_ps
        ):
            raise ValueError("relaxation_times_ps must be positive and complete")
        if tuple(sorted(relaxation_times_ps)) != relaxation_times_ps:
            raise ValueError("relaxation_times_ps must be in ascending order")
        if coupling_scale < 0.0:
            raise ValueError("coupling_scale must be non-negative")
        self.state_dim = int(state_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.coupling_scale = float(coupling_scale)
        self.raw_basis = nn.Parameter(torch.randn(state_dim, auxiliary_dim) * 0.02)
        self.raw_amplitude = nn.Parameter(torch.full((auxiliary_dim,), -2.0))
        self.register_buffer(
            "reference_relaxation_rates",
            torch.tensor(relaxation_times_ps, dtype=torch.float32).reciprocal(),
        )

    def basis(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.raw_basis, mode="reduced")
        sign = torch.sign(torch.diagonal(r)).masked_fill(
            torch.diagonal(r).abs() < 1.0e-8,
            1.0,
        )
        return q * sign.unsqueeze(0)

    def parameters_at(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        basis = self.basis().to(dtype=dtype, device=device)
        rates = self.reference_relaxation_rates.to(dtype=dtype, device=device)
        amplitudes = self.coupling_scale * F.softplus(self.raw_amplitude).to(
            dtype=dtype,
            device=device,
        )
        coupling = basis * amplitudes.unsqueeze(0)
        return {
            "basis": basis,
            "rates": rates,
            "amplitudes": amplitudes,
            "coupling": coupling,
        }

    def forward(
        self,
        time_ps: torch.Tensor,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        dtype = dtype or self.raw_basis.dtype
        device = device or self.raw_basis.device
        params = self.parameters_at(dtype=dtype, device=device)
        time = time_ps.to(dtype=dtype, device=device).reshape(-1)
        decay = torch.exp(-time[:, None] * params["rates"][None, :])
        return torch.einsum(
            "qa,ta,ra->tqr",
            params["coupling"],
            decay,
            params["coupling"],
        )


class ProjectedForceClosure(nn.Module):
    """Bounded unresolved-force correction with an O(dt^2) short-time gate."""

    def __init__(
        self,
        state_dim: int,
        history_feature_dim: int,
        *,
        hidden_dim: int = 128,
        max_shift: float = 0.55,
        horizon_gate_steps: float = 2.0,
        base_step_ps: float = 1000.0,
        reference_temperature_k: float = 348.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or history_feature_dim < 1 or hidden_dim < 4:
            raise ValueError("invalid projected-force closure dimensions")
        if max_shift < 0.0 or horizon_gate_steps <= 0.0 or base_step_ps <= 0.0:
            raise ValueError("projected-force closure scales are invalid")
        self.state_dim = int(state_dim)
        self.history_feature_dim = int(history_feature_dim)
        self.max_shift = float(max_shift)
        self.horizon_gate_steps = float(horizon_gate_steps)
        self.base_step_ps = float(base_step_ps)
        self.reference_temperature_k = float(reference_temperature_k)
        input_dim = state_dim + history_feature_dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        q: torch.Tensor,
        history_features: torch.Tensor,
        dt_ps: torch.Tensor | float,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if q.ndim != 2 or q.shape[-1] != self.state_dim:
            raise ValueError("q has the wrong projected-force state shape")
        if history_features.shape != (q.shape[0], self.history_feature_dim):
            raise ValueError("history_features has the wrong projected-force shape")
        dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
        if dt.ndim == 0:
            dt = dt.expand(q.shape[0])
        if dt.shape != (q.shape[0],):
            raise ValueError("dt_ps must be scalar or have shape [batch]")
        temperature = _as_batch_temperature(
            temperature_k,
            q.shape[0],
            q,
            self.reference_temperature_k,
        )
        horizon = (dt / self.base_step_ps).clamp(min=0.0)
        features = torch.cat(
            [
                q,
                history_features,
                torch.log1p(horizon).unsqueeze(-1),
                torch.sqrt(horizon + 1.0e-6).unsqueeze(-1),
                (temperature / self.reference_temperature_k - 1.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        shift = self.max_shift * torch.tanh(self.net(features))
        gate = -torch.expm1(-horizon / self.horizon_gate_steps)
        return shift * gate.square().unsqueeze(-1)


def _estimate_kde_bandwidth(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    nearest_fraction: float = 0.25,
) -> torch.Tensor:
    """Estimate a local KDE bandwidth for the mean-force target.

    Uses a fraction of the median nearest-neighbour distance. This is more
    robust than Silverman's rule in high dimensions (d >> n) where the full
    Silverman formula converges to a constant near 1.0 regardless of scale.

    Args:
        query: [batch, state_dim] query points.
        reference: [ref_batch, state_dim] reference (training) points.
        nearest_fraction: fraction of median min-distance to use (default 0.25).

    Returns:
        scalar bandwidth tensor on the same device, clamped to [1e-4, 2.0].
    """
    n, d = reference.shape
    with torch.no_grad():
        # Sub-sample to avoid O(B²) cost
        sample_size = min(n, 512)
        idx = torch.randperm(n, device=reference.device)[:sample_size]
        ref_sample = reference[idx]
        # Pairwise distances between query and reference sample
        dists = torch.cdist(query, ref_sample)  # [batch, sample_size]
        # Query points are often drawn from the same set as the KDE reference.
        # Excluding exact self/duplicate matches prevents the adaptive
        # bandwidth from collapsing to the clamp floor and producing a
        # meaningless zero force target.
        dists = dists.masked_fill(dists <= 1.0e-8, float("inf"))
        # Median of minimum distances per query point
        min_dists, _ = dists.min(dim=1)
        finite = torch.isfinite(min_dists)
        if finite.any():
            median_min_dist = min_dists[finite].median()
        else:
            # A completely collapsed reference has no identifiable local
            # scale; retain the lower clamp and let the caller mark the
            # resulting force target as degenerate.
            median_min_dist = query.new_tensor(0.0)
        # Use a fraction of the median nearest-neighbour distance.
        # This gives a local kernel that adapts to the data density.
        bw = nearest_fraction * median_min_dist.clamp(min=1e-8)
    return bw.clamp(min=1e-4, max=2.0).detach()


def kde_mean_force(
    query: torch.Tensor,
    reference: torch.Tensor,
    bandwidth: float | torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Estimate ``grad log p(query)`` with a Gaussian KDE.

    When ``bandwidth`` is 0 or negative, the bandwidth is estimated adaptively
    from the data using ``_estimate_kde_bandwidth``. This is the recommended
    setting for Stage B fitting where a fixed global bandwidth (e.g. 4.0) gives
    near-zero gradients for Tanh-normalised states in [-1, 1].

    The result is a dimensionless mean force target. It is intentionally a
    force estimate, rather than a density or histogram target, so it can train
    a scalar potential with a conservative gradient.
    """
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must have shape [N, state_dim]")
    if reference.shape[0] < 1:
        raise ValueError("reference must contain at least one sample")
    # Adaptive bandwidth estimation
    bw_tensor = query.new_tensor(bandwidth)
    if torch.any(bw_tensor <= 0.0):
        bw_tensor = _estimate_kde_bandwidth(query, reference)
    scale = bw_tensor.square()
    outputs = []
    step = query.shape[0] if chunk_size is None else max(1, int(chunk_size))
    for start in range(0, query.shape[0], step):
        block = query[start : start + step]
        delta = reference.unsqueeze(0) - block.unsqueeze(1)
        log_kernel = -0.5 * delta.square().sum(dim=-1) / scale
        weights = torch.softmax(log_kernel, dim=1)
        outputs.append((weights.unsqueeze(-1) * delta).sum(dim=1) / scale)
    return torch.cat(outputs, dim=0)


def mean_force_matching_loss(
    potential: MeanForcePotential,
    q: torch.Tensor,
    target_mean_force: torch.Tensor,
    condition: torch.Tensor | None = None,
    temperature_k: torch.Tensor | float | None = None,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match a scalar PMF gradient to an independently estimated mean force."""
    if target_mean_force.shape != q.shape:
        raise ValueError("target_mean_force must have the same shape as q")
    predicted = potential.mean_force(
        q, condition, temperature_k, create_graph=True
    )
    residual = predicted - target_mean_force.detach()
    per_sample = residual.square().mean(dim=-1)
    if weights is None:
        return per_sample.mean()
    if weights.shape != per_sample.shape:
        raise ValueError("weights must have shape [batch]")
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0e-8)


def calibrated_mean_force_matching_loss(
    world_model: "LSPWMWorldModel",
    q: torch.Tensor,
    target_mean_force: torch.Tensor,
    condition: torch.Tensor | None = None,
    temperature_k: torch.Tensor | float | None = None,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match the FDT-calibrated total mean force to a KDE score target.

    The target is a score estimate in the fixed latent coordinates.  The
    prediction includes both the calibrated base PMF force and the exact
    gradient of the bounded conservative correction, so the force objective
    is aligned with the transition mean used by the endpoint evaluator.
    """
    if target_mean_force.shape != q.shape:
        raise ValueError("target_mean_force must have the same shape as q")
    predicted = world_model.calibrated_mean_force(
        q, condition, temperature_k, create_graph=True
    )
    residual = predicted - target_mean_force.detach()
    per_sample = residual.square().mean(dim=-1)
    if weights is None:
        return per_sample.mean()
    if weights.shape != per_sample.shape:
        raise ValueError("weights must have shape [batch]")
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0e-8)


def _gaussian_transition_nll_from_parameters(
    mean: torch.Tensor,
    target_q: torch.Tensor,
    diffusion: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a Gaussian transition NLL from a mean and PSD diffusion."""
    if mean.shape != target_q.shape or mean.ndim != 2:
        raise ValueError("mean and target_q must have shape [batch, state_dim]")
    if diffusion.shape != (mean.shape[0], mean.shape[1], mean.shape[1]):
        raise ValueError("diffusion must have shape [batch, state_dim, state_dim]")
    if dt.shape != (mean.shape[0],) or torch.any(dt <= 0.0):
        raise ValueError("dt must be positive and have shape [batch]")
    residual = target_q - mean
    d = mean.shape[-1]

    # Check if the diffusion matrix is effectively diagonal (from StateDependentDiffusion)
    # Heuristic: mean |off_diag| / mean |diag| < 5e-4 → use cheap diagonal path
    d_dim = d
    diag_mask = torch.eye(d_dim, device=diffusion.device, dtype=torch.bool).unsqueeze(0)
    off_diag = diffusion[~diag_mask.expand_as(diffusion)]
    diag_vals = diffusion[diag_mask.expand_as(diffusion)].view(-1, d_dim)  # [batch, d]
    off_diag_norm = off_diag.abs().mean()
    diag_norm = diag_vals.abs().mean().clamp(min=1e-12)
    is_diagonal = bool((off_diag_norm / diag_norm) < 5e-4)

    if is_diagonal:
        # Diagonal case: cheap O(d) per sample
        cov_diag = 2.0 * diag_vals * dt[:, None]  # [batch, d]
        quadratic = (residual.square() / cov_diag.clamp(min=1e-12)).sum(dim=-1)
        logdet = cov_diag.clamp(min=1e-12).log().sum(dim=-1)
    else:
        # Full matrix case: O(d^3) per sample
        covariance = 2.0 * diffusion * dt[:, None, None]
        residual_3d = residual.unsqueeze(-1)
        solved = torch.linalg.solve(covariance, residual_3d).squeeze(-1)
        quadratic = (residual * solved).sum(dim=-1)
        sign, logdet = torch.linalg.slogdet(covariance)
        if torch.any(sign <= 0.0):
            raise ValueError("transition covariance is not positive definite")

    return 0.5 * (quadratic + logdet + d * math.log(2.0 * math.pi)).mean()


def overdamped_transition_nll(
    world_model: "LSPWMWorldModel",
    q: torch.Tensor,
    target_q: torch.Tensor,
    condition: torch.Tensor | None,
    dt_ps: float | torch.Tensor,
    temperature_k: torch.Tensor | float | None = None,
    history_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian one-step NLL for the uncalibrated overdamped state."""
    if q.shape != target_q.shape or q.ndim != 2:
        raise ValueError("q and target_q must have shape [batch, state_dim]")
    dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
    if dt.ndim == 0:
        dt = dt.expand(q.shape[0])
    if dt.shape != (q.shape[0],) or torch.any(dt <= 0.0):
        raise ValueError("dt_ps must be positive scalar or shape [batch]")
    temperature = _as_batch_temperature(
        temperature_k,
        q.shape[0],
        q,
        world_model.config.reference_temperature_k,
    )
    gradient = -world_model.potential.mean_force(
        q, condition, temperature, create_graph=True
    )
    diffusion = world_model.diffusion(q.shape[0], temperature, reference=q)
    mean = q - torch.einsum("bij,bj->bi", diffusion, gradient) * dt[:, None]
    mean = mean + world_model.history_correction(
        q,
        history_features,
        dt,
        temperature,
    )
    return _gaussian_transition_nll_from_parameters(mean, target_q, diffusion, dt)


def calibrated_overdamped_transition_nll(
    world_model: "LSPWMWorldModel",
    q: torch.Tensor,
    target_q: torch.Tensor,
    condition: torch.Tensor | None,
    dt_ps: float | torch.Tensor,
    temperature_k: torch.Tensor | float | None = None,
    history_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL for the trainable FDT-consistent calibrated transition."""
    if q.shape != target_q.shape or q.ndim != 2:
        raise ValueError("q and target_q must have shape [batch, state_dim]")
    parameters = world_model.calibrated_transition(
        q,
        condition,
        dt_ps,
        temperature_k,
        history_features=history_features,
    )
    return _gaussian_transition_nll_from_parameters(
        parameters["mean"],
        target_q,
        parameters["diffusion"],
        parameters["dt_ps"],
    )


def fdt_residual(
    diffusion: torch.Tensor,
    noise_covariance: torch.Tensor,
    dt_ps: float | torch.Tensor,
) -> torch.Tensor:
    """Return a relative residual for ``Cov[dW] = 2 D dt``."""
    if diffusion.shape != noise_covariance.shape or diffusion.ndim != 3:
        raise ValueError("diffusion and noise_covariance must have shape [batch, d, d]")
    dt = torch.as_tensor(dt_ps, device=diffusion.device, dtype=diffusion.dtype)
    if dt.ndim == 0:
        dt = dt.view(1, 1, 1)
    elif dt.shape == (diffusion.shape[0],):
        dt = dt.view(-1, 1, 1)
    elif dt.shape != (diffusion.shape[0], 1, 1):
        raise ValueError("dt_ps must be scalar or have shape [batch]")
    expected = 2.0 * diffusion * dt
    return (noise_covariance - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp(min=1.0e-12)


@torch.no_grad()
def empirical_fdt_residual(
    world_model: "LSPWMWorldModel",
    q: torch.Tensor,
    condition: torch.Tensor | None,
    dt_ps: torch.Tensor | float,
    temperature_k: torch.Tensor | float | None,
    *,
    samples: int = 64,
) -> dict[str, torch.Tensor]:
    """Estimate FDT residuals from independently sampled one-step noise.

    The deterministic drift is subtracted before estimating covariance. This
    keeps the gate tied to the executed integrator rather than to an analytic
    claim or a hard-coded zero.
    """
    if samples < 2:
        raise ValueError("at least two stochastic samples are required")
    if q.ndim != 2:
        raise ValueError("q must have shape [batch, state_dim]")
    dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
    if dt.ndim == 0:
        dt = dt.expand(q.shape[0])
    if dt.shape != (q.shape[0],):
        raise ValueError("dt_ps must be scalar or have shape [batch]")
    auxiliary = q.new_zeros(q.shape[0], world_model.config.auxiliary_dim)
    deterministic = world_model.step(
        q,
        auxiliary,
        condition,
        dt,
        temperature_k,
        stochastic=False,
    )
    q_noise = []
    auxiliary_noise = []
    for _ in range(samples):
        stochastic = world_model.step(
            q,
            auxiliary,
            condition,
            dt,
            temperature_k,
            stochastic=True,
        )
        q_noise.append(stochastic["q"] - deterministic["q"])
        auxiliary_noise.append(stochastic["auxiliary"] - deterministic["auxiliary"])
    q_noise = torch.stack(q_noise, dim=0)
    auxiliary_noise = torch.stack(auxiliary_noise, dim=0)
    q_covariance = torch.einsum("nbi,nbj->bij", q_noise, q_noise) / samples
    auxiliary_covariance = torch.einsum(
        "nbi,nbj->bij", auxiliary_noise, auxiliary_noise
    ) / samples
    coefficients = world_model.coefficients(q, temperature_k)
    q_expected = 2.0 * coefficients["diffusion"] * dt[:, None, None]
    auxiliary_expected = torch.diag_embed(
        2.0 * coefficients["rates"].unsqueeze(0) * dt[:, None]
    )
    q_residual = fdt_residual(coefficients["diffusion"], q_covariance, dt)
    auxiliary_residual = (
        (auxiliary_covariance - auxiliary_expected).square().mean().sqrt()
        / auxiliary_expected.square().mean().sqrt().clamp(min=1.0e-12)
    )
    # A full d x d covariance estimated from N independent draws has a
    # non-zero relative Frobenius error even when the FDT holds exactly:
    # E[||S - I||_F^2 / ||I||_F^2] = (d + 1) / N.  Reporting the raw
    # residual alone therefore turns ordinary Monte Carlo variance into an
    # apparent FDT violation when d is comparable to, or larger than, N.
    q_dimension = q.shape[-1]
    auxiliary_dimension = auxiliary.shape[-1]
    q_sampling_baseline = q.new_tensor((q_dimension + 1.0) / samples).sqrt()
    auxiliary_sampling_baseline = q.new_tensor(
        (auxiliary_dimension + 1.0) / samples
    ).sqrt()
    q_sampling_normalized = q_residual / q_sampling_baseline.clamp(min=1.0e-12)
    auxiliary_sampling_normalized = auxiliary_residual / auxiliary_sampling_baseline.clamp(
        min=1.0e-12
    )
    analytic_q_residual = fdt_residual(
        coefficients["diffusion"], q_expected, dt
    )
    analytic_auxiliary_residual = (
        (auxiliary_expected - auxiliary_expected).square().mean().sqrt()
        / auxiliary_expected.square().mean().sqrt().clamp(min=1.0e-12)
    )
    return {
        "q_covariance": q_covariance,
        "auxiliary_covariance": auxiliary_covariance,
        "q_expected_covariance": q_expected,
        "auxiliary_expected_covariance": auxiliary_expected,
        "q_fdt_residual": q_residual,
        "auxiliary_fdt_residual": auxiliary_residual,
        "fdt_residual": torch.maximum(q_residual, auxiliary_residual),
        "q_fdt_sampling_baseline": q_sampling_baseline,
        "auxiliary_fdt_sampling_baseline": auxiliary_sampling_baseline,
        "q_fdt_sampling_normalized": q_sampling_normalized,
        "auxiliary_fdt_sampling_normalized": auxiliary_sampling_normalized,
        "fdt_sampling_normalized": torch.maximum(
            q_sampling_normalized, auxiliary_sampling_normalized
        ),
        "analytic_q_fdt_residual": analytic_q_residual,
        "analytic_auxiliary_fdt_residual": analytic_auxiliary_residual,
        "analytic_fdt_residual": torch.maximum(
            analytic_q_residual, analytic_auxiliary_residual
        ),
        "samples": q.new_tensor(float(samples)),
    }


def energy_loop_circulation(
    potential: MeanForcePotential,
    loop_q: torch.Tensor,
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically integrate the conservative force around a closed loop."""
    if loop_q.ndim != 2 or loop_q.shape[0] < 3:
        raise ValueError("loop_q must contain at least three vertices")
    if not torch.allclose(loop_q[0], loop_q[-1]):
        loop_q = torch.cat([loop_q, loop_q[:1]], dim=0)
        if condition is not None:
            condition = torch.cat([condition, condition[:1]], dim=0)
    force = potential.mean_force(loop_q, condition, create_graph=False)
    delta = loop_q[1:] - loop_q[:-1]
    gradient = -force
    circulation = 0.5 * (gradient[1:] + gradient[:-1])
    return (circulation * delta).sum(dim=-1).abs().sum()


# ---------------------------------------------------------------------------
# Chapter 6 — Observation Model (decoder)
# ---------------------------------------------------------------------------


class FlowMatchObservationModel(nn.Module):
    """Observation likelihood p(x | z) via flow matching.

    This decoder maps the coarse-grained latent state z (state_dim) back to
    observable protein coordinates. It is trained with an asymmetric cycle
    consistency loss that *never* backpropagates into the dynamics.

    Architecture follows the principle (Ch.6.1–6.2):
        z_t -> z_{t+1}   (dynamics, no decoder)
        z_t -> x_t       (observation model, decoder only)
    """

    def __init__(
        self,
        latent_dim: int,
        coord_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 3,
        flow_matching_steps: int = 16,
    ) -> None:
        super().__init__()
        if latent_dim < 1 or coord_dim < 1 or hidden_dim < 4 or num_layers < 1:
            raise ValueError("invalid observation model dimensions")
        self.latent_dim = int(latent_dim)
        self.coord_dim = int(coord_dim)
        self.flow_matching_steps = int(flow_matching_steps)
        layers = []
        input_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.SiLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, coord_dim))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observable coordinates."""
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have shape [batch, {self.latent_dim}]")
        return self.network(z)

    def flow_matching_loss(
        self,
        z_0: torch.Tensor,
        x_1: torch.Tensor,
        *,
        noise_std: float = 0.05,
    ) -> torch.Tensor:
        """Conditional flow matching loss (Ch.6.2).

        Samples t ~ Uniform(0, 1) and computes the conditional vector field
        that transports a Gaussian prior to the data distribution.
        """
        if z_0.shape != x_1.shape or z_0.ndim != 2:
            raise ValueError("z_0 and x_1 must have the same shape [batch, coord_dim]")
        batch = z_0.shape[0]
        device = z_0.device
        t = torch.rand(batch, device=device)
        x_0 = torch.randn_like(z_0) * noise_std
        x_t = (1.0 - t[:, None]) * x_0 + t[:, None] * x_1
        target_velocity = x_1 - x_0
        predicted_velocity = self.forward(x_t) - x_t
        return (predicted_velocity - target_velocity).square().mean()


def asymmetric_cycle_loss(
    decoder: FlowMatchObservationModel,
    z_context: torch.Tensor,
    z_future: torch.Tensor,
    x_context: torch.Tensor,
    x_future: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    cycle_stopgrad: bool = True,
) -> torch.Tensor:
    """Asymmetric cycle consistency loss (Ch.6.3).

    The cycle loss only trains the decoder. When ``cycle_stopgrad=True``,
    gradients are blocked from entering the dynamics path.

    ``z_context -> decode -> x_context_recon -> encode -> z_context_cycle``
    is NOT enforced. Instead:

        z_future -> decode -> x_future_pred
        x_future -> encode -> z_future     (only if an inverse encoder exists)

    Here we use the simpler observation-side cycle:

        z -> decode -> x_recon
        x -> encode -> z_cycle  (mean squared error only on decoder params)
    """
    if z_context.shape != z_future.shape:
        raise ValueError("z_context and z_future must have the same shape")
    if x_context.shape != x_future.shape:
        raise ValueError("x_context and x_future must have the same shape")
    x_pred = decoder(z_future)
    recon_loss = (x_pred - x_future.detach()).square().mean(dim=-1)
    if weights is not None:
        if weights.shape != (z_context.shape[0],):
            raise ValueError("weights must have shape [batch]")
        recon_loss = recon_loss * weights
    return recon_loss.mean()


def local_density_weights(
    z: torch.Tensor,
    *,
    alpha: float = 0.5,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Local density weight w(z) = p_data(z)^alpha (Ch.6.4).

    High-density regions get higher weight, so the decoder focuses on
    well-sampled areas. Transition states are naturally down-weighted.
    """
    if z.ndim != 2:
        raise ValueError("z must have shape [batch, state_dim]")
    n = z.shape[0]
    delta = z.unsqueeze(0) - z.unsqueeze(1)
    log_kernel = -0.5 * delta.square().sum(dim=-1) / bandwidth
    log_prob = torch.logsumexp(log_kernel, dim=1) - math.log(max(n, 1))
    weight = torch.exp(alpha * log_prob)
    return weight / weight.sum().clamp(min=1.0e-8)


# ---------------------------------------------------------------------------
# Chapter 7 — VAMP-2 Koopman loss (L_koop)
# ---------------------------------------------------------------------------


def vamp2_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
) -> torch.Tensor:
    """Differentiable VAMP-2 score maximisation (Ch.7 L_koop).

    Maximising the VAMP-2 score encourages the latent representation to span
    a Koopman-invariant subspace (Ch.2 Proposition). The loss is negative
    VAMP-2, so minimising it maximises the kinetic variance explained.

    Reference: Wu & Noé, "Variational Approach for Learning Markov Processes
    from Time Series Data" (2019).
    """
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have shape [N, state_dim]")
    if source.shape[0] < 2:
        return source.new_tensor(0.0)
    source_centered = source - source.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    n = max(1, source.shape[0] - 1)
    c00 = source_centered.T @ source_centered / n
    c11 = target_centered.T @ target_centered / n
    c01 = source_centered.T @ target_centered / n
    eye = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    # The explicit eigendecomposition is useful for diagnostics, but its
    # backward pass is undefined when minibatch covariance eigenvalues repeat
    # or the covariance is rank deficient. Solves against a ridge-regularized
    # covariance give the same finite-sample VAMP-2 trace and stable gradients.
    c00_reg = c00 + ridge * eye
    c11_reg = c11 + ridge * eye
    left = torch.linalg.solve(c00_reg, c01)
    score = torch.linalg.solve(c11_reg, c01.T @ left).diagonal().sum()
    return -score  # negative VAMP-2 = loss to minimise


# ---------------------------------------------------------------------------
# Chapter 9 — GENERIC operators for the extended state
# ---------------------------------------------------------------------------


class PoissonOperator(nn.Module):
    """Anti-symmetric Poisson operator L(y) for reversible dynamics.

    In the overdamped limit (Ch.0.2), the Poisson operator is zero for the
    positional coordinates. However, the extended state (q, s) has a non-trivial
    symplectic structure from the Markovian embedding:

        L = [[0,  C],
             [-C^T, 0]]

    This ensures the reversible part of the GENERIC equation preserves the
    energy: L(y) nabla E(y) is orthogonal to nabla S(y) (degeneracy condition).
    """

    def __init__(self, state_dim: int, auxiliary_dim: int) -> None:
        super().__init__()
        if state_dim < 1 or auxiliary_dim < 1:
            raise ValueError("Poisson operator needs positive dimensions")
        self.state_dim = int(state_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.coupling = nn.Parameter(torch.randn(state_dim, auxiliary_dim) * 0.02)

    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        """Apply the Poisson operator to a gradient vector.

        Args:
            gradient: concatenated [nabla_q E, nabla_s E] of shape [batch, state_dim + aux_dim]

        Returns:
            L @ gradient of shape [batch, state_dim + aux_dim]
        """
        if gradient.ndim != 2 or gradient.shape[-1] != self.state_dim + self.auxiliary_dim:
            raise ValueError("gradient must have shape [batch, state_dim + auxiliary_dim]")
        q_grad = gradient[:, :self.state_dim]
        s_grad = gradient[:, self.state_dim:]
        coupling = self.coupling.to(dtype=gradient.dtype, device=gradient.device)
        q_part = torch.einsum("qa,ba->bq", coupling, s_grad)
        s_part = -torch.einsum("qa,bq->ba", coupling, q_grad)
        return torch.cat([q_part, s_part], dim=-1)


class DissipationOperator(nn.Module):
    """Symmetric PSD dissipation operator M(y) for irreversible dynamics.

    In the GENERIC formalism, M(y) governs the irreversible dynamics and
    satisfies the degeneracy condition M(y) nabla E(y) = 0 (energy conservation
    by the dissipative part).

    For the overdamped Langevin with Markovian embedding:

        M = [[D/(k_B T),  0],
             [0,          gamma/(k_B T)]]

    where D is the diffusion tensor and gamma is the bath friction.
    """

    def __init__(
        self,
        state_dim: int,
        auxiliary_dim: int,
        *,
        reference_temperature_k: float = 348.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or auxiliary_dim < 1:
            raise ValueError("dissipation operator needs positive dimensions")
        self.state_dim = int(state_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.reference_temperature_k = float(reference_temperature_k)

    def forward(
        self,
        gradient: torch.Tensor,
        diffusion: torch.Tensor,
        rates: torch.Tensor,
        temperature_k: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the dissipation operator to a gradient vector.

        M @ gradient = [D/(k_B T) * nabla_q E,  gamma/(k_B T) * nabla_s E]

        Args:
            gradient: concatenated [nabla_q E, nabla_s E] of shape [batch, state_dim + aux_dim]
            diffusion: diffusion tensor [batch, state_dim, state_dim]
            rates: bath relaxation rates [auxiliary_dim]
            temperature_k: temperature [batch]

        Returns:
            M @ gradient of shape [batch, state_dim + aux_dim]
        """
        if gradient.ndim != 2 or gradient.shape[-1] != self.state_dim + self.auxiliary_dim:
            raise ValueError("gradient must have shape [batch, state_dim + auxiliary_dim]")
        batch = gradient.shape[0]
        q_grad = gradient[:, :self.state_dim]
        s_grad = gradient[:, self.state_dim:]
        kbt = (temperature_k / self.reference_temperature_k).to(
            dtype=gradient.dtype, device=gradient.device
        ).view(-1, 1, 1)
        q_part = torch.einsum("bij,bj->bi", diffusion / kbt, q_grad)
        gamma = (rates * self.reference_temperature_k).to(
            dtype=gradient.dtype, device=gradient.device
        )
        s_part = s_grad * gamma.unsqueeze(0) / (temperature_k[:, None].to(
            dtype=gradient.dtype, device=gradient.device
        ) + 1.0e-8)
        return torch.cat([q_part, s_part], dim=-1)


def fit_koopman_operator(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    """Fit ``target ~= source @ K.T`` and return a centered linear readout."""
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must have the same shape [N, state_dim]")
    if source.shape[0] < 2 or ridge < 0.0:
        raise ValueError("at least two samples and non-negative ridge are required")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    x = source - source_mean
    y = target - target_mean
    gram = x.T @ x / source.shape[0]
    cross = x.T @ y / source.shape[0]
    identity = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    solution = torch.linalg.solve(gram + ridge * identity, cross)
    operator = solution.T
    prediction = x @ operator.T + target_mean
    residual = prediction - target
    return {
        "operator": operator,
        "source_mean": source_mean,
        "target_mean": target_mean,
        "prediction": prediction,
        "residual": residual,
        "residual_mse": residual.square().mean(),
    }


def _vamp2_score(source: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    source = source - source.mean(dim=0)
    target = target - target.mean(dim=0)
    n = max(1, source.shape[0] - 1)
    c00 = source.T @ source / n
    c11 = target.T @ target / n
    c01 = source.T @ target / n
    eye = torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    eval0, vec0 = torch.linalg.eigh(c00 + ridge * eye)
    eval1, vec1 = torch.linalg.eigh(c11 + ridge * eye)
    inv0 = vec0 @ torch.diag(eval0.clamp(min=ridge).rsqrt()) @ vec0.T
    inv1 = vec1 @ torch.diag(eval1.clamp(min=ridge).rsqrt()) @ vec1.T
    whitened = inv0 @ c01 @ inv1
    return whitened.square().sum()


def koopman_diagnostics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    lag_time: float = 1.0,
    ridge: float = 1.0e-5,
    predictor_jacobian_variance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return empirical compatibility diagnostics for one lag."""
    if lag_time <= 0.0:
        raise ValueError("lag_time must be positive")
    fit = fit_koopman_operator(source, target, ridge=ridge)
    eigenvalues = torch.linalg.eigvals(fit["operator"])
    magnitude = eigenvalues.abs().clamp(min=1.0e-8, max=1.0 - 1.0e-8)
    implied_timescales = -source.new_tensor(lag_time) / magnitude.log()
    jacobian_variance = (
        predictor_jacobian_variance
        if predictor_jacobian_variance is not None
        else source.new_zeros(())
    )
    return {
        **fit,
        "eigenvalues": eigenvalues,
        "implied_timescales": implied_timescales,
        "vamp2_score": _vamp2_score(source, target, ridge),
        "predictor_jacobian_variance": jacobian_variance,
    }


class BoundedConservativePotential(nn.Module):
    """Low-rank bounded free-energy correction in the existing latent space.

    The correction is a scalar potential, so its force is integrable by
    construction.  ``tanh`` bounds both the state features and the
    condition-dependent amplitudes.  Zero initialization of the amplitudes
    makes the calibrated model exactly reduce to scalar calibration at step 0.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int,
        *,
        rank: int = 16,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or condition_dim < 0 or rank < 1 or scale < 0.0:
            raise ValueError("invalid conservative correction dimensions or scale")
        self.state_dim = int(state_dim)
        self.condition_dim = int(condition_dim)
        self.rank = int(rank)
        self.scale = float(scale)
        self.raw_state_basis = nn.Parameter(
            torch.randn(state_dim, rank) * 0.02
        )
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
        # Keep the force field bounded even if optimization is aggressive.
        return torch.tanh(self.raw_state_basis)

    def forward(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
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
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``grad DeltaA``; the corresponding mean force is ``-gradient``."""
        with torch.enable_grad():
            q_leaf = q if q.requires_grad else q.detach().requires_grad_(True)
            energy = self(q_leaf, condition)
            return torch.autograd.grad(
                energy.sum(),
                q_leaf,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]


class PhysicalTransitionCalibration(nn.Module):
    """FDT-consistent scalar and conservative transition calibration.

    For the overdamped Langevin law

        dq = -D grad(A) dt + sqrt(2 D dt) dW,

    this module uses ``D_cal = m D`` and
        ``grad(A_cal) = s grad(A) + grad(DeltaA)``.  An optional mode treats
    ``DeltaA`` as being expressed in the same dimensionless energy units as
    ``A`` and applies ``s`` to both terms.  The same mobility scale therefore
    controls both drift mobility and covariance, preserving the
    Einstein/FDT relationship.
    """

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
                state_dim,
                condition_dim,
                rank=conservative_rank,
                scale=conservative_scale,
            )
            if self.conservative_enabled
            else None
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
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        if self.conservative_correction is None:
            return q.new_zeros(q.shape)
        return self.conservative_correction.gradient(
            q,
            condition,
            create_graph=create_graph,
        )

    def regularization_loss(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.conservative_correction is None:
            return q.new_zeros(())
        correction_gradient = self.correction_gradient(
            q,
            condition,
            create_graph=True,
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

    # Potential architecture (Ch.4 PMF)
    potential_hidden_dim: int = 128
    potential_hidden_depth: int = 4

    # Observation model (Ch.6)
    observation_model: bool = False
    observation_coord_dim: int = 0
    observation_hidden_dim: int = 256
    observation_num_layers: int = 3
    observation_flow_steps: int = 16

    # Curriculum (Ch.7.2)
    curriculum_phases: tuple[str, ...] = (
        "representation",
        "koopman",
        "thermodynamic_identification",
        "dynamics",
        "observation",
    )
    curriculum_phase_steps: tuple[int, ...] = (
        0, 1000, 2500, 5000, 10000
    )

    # State-dependent diffusion (heteroscedastic)
    state_dependent_diffusion_enabled: bool = False

    # Separate force-loss flag (decoupled from dynamic_nll_enabled)
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

    # ──────────────────────────────────────────────────────────────
    # v4.4 architectural extensions (first-principles revisions)
    # ──────────────────────────────────────────────────────────────

    # Midpoint/BAOAB integration (Rev-1): evaluates gradient at q+drift*dt/2
    # instead of only at q, giving O(dt^2) accuracy and better long-time
    # stability for multi-step rollouts (FastFolding H>1).
    midpoint_integration: bool = False

    # Multi-step rollout trajectory NLL (Rev-2): trains the path likelihood
    # P[q(t0:tK) | q(t0)] by rolling out K steps and computing the joint
    # Gaussian NLL of the observed trajectory.  rollout_trajectory_steps
    # controls K (0 disables).
    rollout_trajectory_steps: int = 0

    # Semigroup consistency weight (Rev-4): enforces T(2Δt) ≈ T(Δt)^2 via
    # the Koopman operator, implementing §6's cross-timescale constraint.
    # 0.0 disables the loss.
    semigroup_consistency_weight: float = 0.0

    # Sliced score matching (Rev-6 alternative): when enabled, replaces the
    # KDE-based mean_force_matching_loss with sliced score matching, which
    # avoids bandwidth selection entirely and scales better to >16 dims.
    sliced_score_matching_enabled: bool = False
    sliced_score_matching_noise: float = 0.05


class LSPWMWorldModel(nn.Module):
    """Canonical passive Markovian Mori embedding.

    The executed path is the dimensionless overdamped system

        dq = [-D grad A(q) + C s] dt + sqrt(2D) dW_q
        ds = [-Gamma s - C^T grad A(q)] dt + sqrt(2Gamma) dW_s

    ``C`` and ``Gamma`` come from the same QR-gauge-fixed bath object used for
    the reported memory kernel. ``use_generic`` remains as a compatibility
    argument for old callers, but both branches execute this canonical law.
    """

    def __init__(self, config: LSPWMConfig = LSPWMConfig()) -> None:
        super().__init__()
        if len(config.relaxation_times_ps) != config.auxiliary_dim:
            raise ValueError("bath times must match auxiliary_dim")
        self.config = config
        self.potential = MeanForcePotential(
            config.state_dim,
            config.condition_dim,
            hidden_dim=config.potential_hidden_dim,
            hidden_depth=config.potential_hidden_depth,
            reference_temperature_k=config.reference_temperature_k,
        )
        if config.state_dependent_diffusion_enabled:
            self.diffusion = StateDependentDiffusion(
                config.state_dim,
                config.condition_dim,
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
            config.state_dim,
            config.condition_dim,
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
            config.state_dim,
            config.auxiliary_dim,
            relaxation_times_ps=config.relaxation_times_ps,
            coupling_scale=config.coupling_scale,
        )
        self.history_closure = (
            ProjectedForceClosure(
                config.state_dim,
                config.history_feature_dim,
                hidden_dim=config.history_closure_hidden_dim,
                max_shift=config.history_closure_max_shift,
                horizon_gate_steps=config.history_closure_horizon_gate_steps,
                base_step_ps=config.relaxation_times_ps[0],
                reference_temperature_k=config.reference_temperature_k,
            )
            if config.history_feature_dim > 0
            else None
        )
        # GENERIC operators (Ch.9)
        self.poisson = PoissonOperator(config.state_dim, config.auxiliary_dim)
        self.dissipation = DissipationOperator(
            config.state_dim,
            config.auxiliary_dim,
            reference_temperature_k=config.reference_temperature_k,
        )
        # Observation model (Ch.6)
        self.observation_model = (
            FlowMatchObservationModel(
                config.state_dim,
                config.observation_coord_dim,
                hidden_dim=config.observation_hidden_dim,
                num_layers=config.observation_num_layers,
                flow_matching_steps=config.observation_flow_steps,
            )
            if config.observation_model
            else None
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
        diffusion = self.diffusion(
            q.shape[0], temperature_k, reference=q
        )
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
        """Return the FDT-consistent calibrated one-step Gaussian parameters."""
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
            temperature_k,
            q.shape[0],
            q,
            self.config.reference_temperature_k,
        )
        raw_gradient = -self.potential.mean_force(
            q,
            condition,
            temperature,
            create_graph=True,
        )
        correction_gradient = calibration.correction_gradient(
            q,
            condition,
            create_graph=True,
        )
        diffusion = self.diffusion(
            q.shape[0],
            temperature,
            reference=q,
        ) * calibration.mobility_scale
        if calibration.conservative_rescale_with_free_energy:
            total_gradient = calibration.free_energy_scale * (
                raw_gradient + correction_gradient
            )
        else:
            total_gradient = (
                calibration.free_energy_scale * raw_gradient
                + correction_gradient
            )
        mean = q - torch.einsum(
            "bij,bj->bi", diffusion, total_gradient
        ) * dt[:, None]
        if history_features is not None and self.history_closure is not None:
            mean = mean + self.history_correction(
                q,
                history_features,
                dt,
                temperature,
            )
        return {
            "mean": mean,
            "diffusion": diffusion,
            "gradient": total_gradient,
            "raw_gradient": raw_gradient,
            "correction_gradient": correction_gradient,
            "temperature_k": temperature,
            "dt_ps": dt,
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
        """Return the calibrated conservative force in latent coordinates."""
        calibration = self.physical_calibration
        if not calibration.enabled:
            raise ValueError("physical transition calibration is disabled")
        if q.ndim != 2 or q.shape[-1] != self.config.state_dim:
            raise ValueError("q must have shape [batch, state_dim]")
        temperature = _as_batch_temperature(
            temperature_k,
            q.shape[0],
            q,
            self.config.reference_temperature_k,
        )
        base_force = self.potential.mean_force(
            q,
            condition,
            temperature,
            create_graph=create_graph,
        )
        correction_gradient = calibration.correction_gradient(
            q,
            condition,
            create_graph=create_graph,
        )
        if calibration.conservative_rescale_with_free_energy:
            return calibration.free_energy_scale * (
                base_force - correction_gradient
            )
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
            temperature_k,
            q.shape[0],
            q,
            self.config.reference_temperature_k,
        )
        gradient = -self.potential.mean_force(
            q,
            condition,
            temperature,
            create_graph=self.training,
        )
        coefficients = self.coefficients(q, temperature)
        diffusion = coefficients["diffusion"]
        coupling = coefficients["coupling"]
        rates = coefficients["rates"]

        # Canonical passive Mori dynamics. The coupling is the same
        # QR-gauge-fixed bath coupling used to define the memory kernel. The
        # old GENERIC helper is retained as an API primitive, but it is not
        # used here because its previous entropy scaling omitted the PMF drift.
        del use_generic
        q_drift = -torch.einsum("bij,bj->bi", diffusion, gradient)
        q_drift = q_drift + torch.einsum(
            "bqa,ba->bq", coupling.expand(q.shape[0], -1, -1), auxiliary
        )
        auxiliary_drift = -auxiliary * rates.unsqueeze(0)
        auxiliary_drift = auxiliary_drift - torch.einsum(
            "bqa,bq->ba",
            coupling.expand(q.shape[0], -1, -1),
            gradient,
        )

        q_noise = q.new_zeros(q.shape)
        auxiliary_noise = auxiliary.new_zeros(auxiliary.shape)
        if stochastic:
            q_covariance = 2.0 * diffusion * dt[:, None, None]
            cholesky = torch.linalg.cholesky(q_covariance)
            q_noise = torch.einsum(
                "bi,bji->bj",
                torch.randn_like(q),
                cholesky,
            )
            auxiliary_std = (2.0 * rates.unsqueeze(0) * dt[:, None]).sqrt()
            auxiliary_noise = torch.randn_like(auxiliary) * auxiliary_std
        next_q = q + q_drift * dt[:, None] + q_noise
        # A learned history residual is intentionally excluded from the
        # canonical path unless its conditional covariance is modeled and
        # validated against FDT.
        if history_features is not None and self.history_closure is not None:
            raise ValueError(
                "history closure is disabled in the canonical FDT path; "
                "use a validated memory state instead"
            )
        next_auxiliary = auxiliary + auxiliary_drift * dt[:, None] + auxiliary_noise
        return {
            "q": next_q,
            "auxiliary": next_auxiliary,
            "q_drift": q_drift,
            "auxiliary_drift": auxiliary_drift,
            "gradient": gradient,
            "diffusion": diffusion,
            "coupling": coupling,
            "rates": rates,
            "temperature_k": temperature,
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
        """BAOAB-style midpoint step (Rev-1).

        Unlike the Euler-Maruyama ``step()`` which evaluates the gradient
        only at the start state ``q``, this method:

        1. Draws noise and applies half-step diffusion (B/2).
        2. Evaluates the gradient at the **midpoint** after the half-step
           diffusion but before the drift (A).
        3. Applies the full drift from the midpoint (A).
        4. Applies the second half-step diffusion (B/2).

        For the overdamped Langevin system with Mori bath:

            dq/dt = -D ∇A(q) + C s + √(2D) ξ_q

        the BAOAB splitting preserves the Boltzmann distribution for
        quadratic potentials and gives O(dt²) global accuracy.
        """
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

        d_dim = q.shape[-1]
        dt_expand = dt[:, None, None]  # [batch, 1, 1]

        # ── B/2: half-step diffusion noise ──
        if stochastic:
            q_cov = 2.0 * diffusion * dt_expand  # [batch, d, d]
            chol = torch.linalg.cholesky(q_cov)
            noise_b1 = torch.einsum(
                "bi,bji->bj", torch.randn_like(q), chol
            )
            q_half = q + 0.5 * noise_b1
        else:
            q_half = q

        # ── A: full-step drift evaluated at MIDPOINT q_half ──
        gradient = -self.potential.mean_force(
            q_half, condition, temperature, create_graph=self.training,
        )
        q_drift = -torch.einsum("bij,bj->bi", diffusion, gradient)
        q_drift = q_drift + torch.einsum(
            "bqa,ba->bq", coupling.expand(q.shape[0], -1, -1), auxiliary
        )

        auxiliary_drift = -auxiliary * rates.unsqueeze(0)
        auxiliary_drift = auxiliary_drift - torch.einsum(
            "bqa,bq->ba",
            coupling.expand(q.shape[0], -1, -1),
            gradient,
        )

        q_full = q_half + q_drift * dt[:, None]
        auxiliary_full = auxiliary + auxiliary_drift * dt[:, None]

        # ── B/2: second half-step diffusion ──
        if stochastic:
            noise_b2 = torch.einsum(
                "bi,bji->bj", torch.randn_like(q), chol
            )
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

        return {
            "q": next_q,
            "auxiliary": next_auxiliary,
            "q_drift": q_drift,
            "auxiliary_drift": auxiliary_drift,
            "gradient": gradient,
            "diffusion": diffusion,
            "coupling": coupling,
            "rates": rates,
            "temperature_k": temperature,
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
                    current_q,
                    current_auxiliary,
                    condition,
                    dt_ps,
                    temperature_k,
                    stochastic=stochastic,
                    history_features=history_features,
                )
            else:
                result = step_fn(
                    current_q,
                    current_auxiliary,
                    condition,
                    dt_ps,
                    temperature_k,
                    stochastic=stochastic,
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

    # ── Rev-2: Multi-step rollout trajectory NLL ────────────────────
    def rollout_trajectory_nll(
        self,
        q_seq: torch.Tensor,        # [batch, seq_len, state_dim]
        condition: torch.Tensor | None,
        dt_ps: float | torch.Tensor,
        temperature_k: torch.Tensor | float | None = None,
        *,
        auxiliary_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Joint NLL of an observed trajectory under K-step model rollout.

        Unlike ``overdamped_transition_nll()`` which evaluates only a single
        step, this method rolls out the world model for ``seq_len - 1`` steps
        and computes the Gaussian NLL at each step, **training the path
        likelihood** P[q(t0:tK) | q(t0)].

        Args:
            q_seq: Observed trajectory [batch, seq_len, state_dim].
            condition: Static condition per batch element.
            dt_ps: Time step per transition (scalar or [batch]).
            temperature_k: Temperature per batch element.
            auxiliary_init: Optional pre-initialized auxiliary state.
                When None, uses zeros (standard Mori embedding).

        Returns:
            Scalar trajectory NLL averaged over steps and batch.
        """
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
                q_seq[:, t - 1, :].contiguous(),
                auxiliary,
                condition,
                dt,
                temperature,
                stochastic=False,  # evaluate NLL, no extra noise
                history_features=None,
            )
            q_pred_mean = result["q"]  # deterministic next state
            diffusion = result["diffusion"]
            auxiliary = result["auxiliary"]  # propagate for next step

            # Gaussian NLL: -log N(q_obs | q_pred_mean, Σ)
            # Σ = 2 * D * dt  (from FDT)
            d_dim = d
            diag_mask = torch.eye(d_dim, device=device, dtype=torch.bool).unsqueeze(0)
            diag_vals = diffusion[diag_mask.expand_as(diffusion)].view(batch, d_dim)
            cov_diag = 2.0 * diag_vals * dt[:, None]  # [batch, d]
            residual = q_seq[:, t, :] - q_pred_mean
            quadratic = (residual.square() / cov_diag.clamp(min=1e-12)).sum(dim=-1)
            logdet = cov_diag.clamp(min=1e-12).log().sum(dim=-1)
            step_nll = 0.5 * (quadratic + logdet + d_dim * math.log(2.0 * math.pi))
            total_nll = total_nll + step_nll.mean()

        return total_nll / (seq_len - 1)

    # ── Rev-4: Semigroup consistency loss ──────────────────────────
    def semigroup_consistency_loss(
        self,
        source: torch.Tensor,    # [batch, state_dim]
        target_dt: torch.Tensor, # [batch, state_dim]
        target_2dt: torch.Tensor,# [batch, state_dim]
        ridge: float = 1.0e-6,
    ) -> torch.Tensor:
        """Enforce T(2Δt) ≈ T(Δt)² via the Koopman operator.

        The Koopman operator K maps observables linearly:
            K_Δt g(q_t) = E[g(q_{t+Δt}) | q_t]

        The semigroup property from §6 requires:
            K_{2Δt} = K_Δt ◦ K_Δt

        We compute this by building empirical Koopman matrices at both lags
        and measuring their mismatch as a Frobenius-norm penalty.

        This is the **cross-timescale consistency** constraint missing from
        the v4.4 single-lag VAMP-2 score.
        """
        # Center the data
        source_c = source - source.mean(dim=0, keepdim=True)
        target_dt_c = target_dt - target_dt.mean(dim=0, keepdim=True)
        target_2dt_c = target_2dt - target_2dt.mean(dim=0, keepdim=True)

        # Empirical covariance and cross-covariance (standard Koopman estimation)
        C_00 = (source_c.T @ source_c) / (source.shape[0] - 1) + ridge * torch.eye(
            source.shape[-1], device=source.device, dtype=source.dtype
        )
        C_01 = (source_c.T @ target_dt_c) / (source.shape[0] - 1)
        C_02 = (source_c.T @ target_2dt_c) / (source.shape[0] - 1)

        # K_Δt = C_00^{-1} @ C_01  (T(Δt))
        # K_{2Δt} = C_00^{-1} @ C_02  (T(2Δt))
        C_00_inv = torch.linalg.solve(C_00, torch.eye(
            source.shape[-1], device=source.device, dtype=source.dtype
        ))
        K_dt = C_00_inv @ C_01
        K_2dt = C_00_inv @ C_02

        # Semigroup consistency: ||K_{2Δt} - K_Δt @ K_Δt||_F^2
        consistency = (K_2dt - K_dt @ K_dt).square().sum()
        return consistency

    def get_active_phase(self, step: int) -> str:
        """Return the active curriculum phase (Ch.7.2) for the given step."""
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
        # Rev-2: optional multi-step trajectory for rollout NLL
        q_seq: torch.Tensor | None = None,
        auxiliary_init: torch.Tensor | None = None,
        # Rev-4: optional second-lag target for semigroup consistency
        target_q_2dt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the subset of canonical losses active for the given phase.

        Phases (Ch.7.2):
            I  (representation) : L_repr  (handled by JEPA, not here)
            II (koopman)        : L_koop  (VAMP-2 score)
            III (thermodynamic) : L_dyn + L_force
            IV (dynamics)       : L_dyn
            V  (observation)    : L_obs

        Extended phases (v4.4):
            * Rev-2 rollout trajectory NLL  (when q_seq is provided)
            * Rev-4 semigroup consistency   (when target_q_2dt is provided)
        """
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
                koopman_source, koopman_target, ridge=vamp_ridge
            )

        if "force" in active:
            reference = q.detach()
            if reference.shape[0] >= 2:
                mean_force_target = kde_mean_force(
                    q.detach(), reference, bandwidth=1.5
                )
                force_loss = mean_force_matching_loss(
                    self.potential,
                    q,
                    mean_force_target,
                    condition,
                    temperature_k,
                )
                losses["lspwm_force_loss"] = force_loss

        if "dynamics" in active:
            dyn_loss = overdamped_transition_nll(
                self, q, target_q, condition, dt_ps, temperature_k,
                history_features=history_features,
            )
            losses["lspwm_dynamic_nll"] = dyn_loss

        # ── Rev-2: Multi-step rollout trajectory NLL ──────────────
        if self.config.rollout_trajectory_steps > 0 and q_seq is not None:
            if q_seq.shape[1] >= 2:
                trajectory_nll = self.rollout_trajectory_nll(
                    q_seq, condition, dt_ps, temperature_k,
                    auxiliary_init=auxiliary_init,
                )
                losses["lspwm_trajectory_nll"] = trajectory_nll

        # ── Rev-4: Semigroup consistency loss ─────────────────────
        if self.config.semigroup_consistency_weight > 0.0 and target_q_2dt is not None:
            semigroup_loss = self.semigroup_consistency_loss(
                q, target_q, target_q_2dt, ridge=vamp_ridge
            )
            losses["lspwm_semigroup_loss"] = semigroup_loss

        if "observation" in active and self.observation_model is not None:
            if q.shape[0] >= 2:
                cycle = asymmetric_cycle_loss(
                    self.observation_model, q, target_q, q, target_q
                )
                losses["lspwm_observation_loss"] = cycle

        return losses


class UnifiedLatentRoleReadout(nn.Module):
    """Read several scientific views from one shared coarse state.

    The role dimensions describe probe capacity, not fixed coordinates in ``z``.
    Every probe sees the complete state, so the JEPA representation remains a
    single 256-dimensional object and can reuse information across objectives.
    """

    def __init__(
        self,
        state_dim: int,
        fixed_cv_dim: int,
        condition_dim: int,
        *,
        koopman_dim: int = 64,
        msm_dim: int = 32,
        geometric_dim: int = 96,
        auxiliary_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        dimensions = (state_dim, fixed_cv_dim, condition_dim, koopman_dim, msm_dim, geometric_dim, auxiliary_dim)
        if any(value < 1 for value in dimensions) or hidden_dim < 4:
            raise ValueError("unified latent role dimensions must be positive")
        self.state_dim = int(state_dim)
        self.fixed_cv_dim = int(fixed_cv_dim)
        self.condition_dim = int(condition_dim)
        self.koopman_dim = int(koopman_dim)
        self.msm_dim = int(msm_dim)
        self.geometric_dim = int(geometric_dim)
        self.auxiliary_dim = int(auxiliary_dim)

        def readout(output_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.koopman = readout(self.koopman_dim)
        self.msm = readout(self.msm_dim)
        self.geometric = readout(self.geometric_dim)
        self.auxiliary = readout(self.auxiliary_dim)
        self.geometric_decoder = nn.Sequential(
            nn.LayerNorm(self.geometric_dim),
            nn.Linear(self.geometric_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.fixed_cv_dim),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        if z.ndim != 2 or z.shape[-1] != self.state_dim:
            raise ValueError(f"z must have shape [batch, {self.state_dim}]")
        return {
            "koopman": self.koopman(z),
            "msm_logits": self.msm(z),
            "geometric": self.geometric(z),
            "auxiliary": self.auxiliary(z),
        }

    def losses(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        fixed_cv: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return role-guidance losses without partitioning the latent vector."""
        current_roles = self(current)
        with torch.no_grad():
            target_roles = self(target.detach())
        target_auxiliary = condition.detach()
        if target_auxiliary.shape[-1] < self.auxiliary_dim:
            target_auxiliary = F.pad(
                target_auxiliary, (0, self.auxiliary_dim - target_auxiliary.shape[-1])
            )
        elif target_auxiliary.shape[-1] > self.auxiliary_dim:
            target_auxiliary = target_auxiliary[..., : self.auxiliary_dim]
        geometric_prediction = self.geometric_decoder(current_roles["geometric"])
        return {
            "lspwm_koopman_loss": vamp2_loss(
                current_roles["koopman"], target_roles["koopman"], ridge=1.0e-3
            ),
            "lspwm_msm_loss": msm_consistency_loss(
                current_roles["msm_logits"], target_roles["msm_logits"]
            ),
            "lspwm_geometric_loss": F.smooth_l1_loss(
                geometric_prediction, fixed_cv.detach()
            ),
            "lspwm_auxiliary_loss": F.smooth_l1_loss(
                current_roles["auxiliary"], target_auxiliary
            ),
        }, current_roles


def msm_consistency_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    *,
    balance_weight: float = 0.1,
    confidence_weight: float = 0.02,
) -> torch.Tensor:
    """Self-supervised soft-MSM objective for paired coarse states.

    The first term preserves state identity across a lag, the second prevents a
    single state from absorbing the batch, and the last term makes assignments
    usable as soft metastable states. It has no fixed partition of ``z``.
    """
    if source_logits.shape != target_logits.shape or source_logits.ndim != 2:
        raise ValueError("MSM logits must have the same shape [batch, num_states]")
    source = source_logits.softmax(dim=-1)
    target = target_logits.softmax(dim=-1).detach()
    overlap = (source * target).sum(dim=-1).clamp(min=1.0e-8)
    marginal = source.mean(dim=0)
    num_states = source.shape[-1]
    balance = num_states * (marginal - 1.0 / num_states).square().sum()
    entropy = -(source * source.clamp(min=1.0e-8).log()).sum(dim=-1).mean()
    return -overlap.log().mean() + balance_weight * balance + confidence_weight * entropy


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
                midpoint_integration=False,      # Rev-1: configurable per experiment
                rollout_trajectory_steps=0,      # Rev-2: set >0 to enable
                semigroup_consistency_weight=0.0, # Rev-4: set >0 to enable
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

    @property
    def state_dim(self) -> int:
        return self.learned_cv_dim

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
                losses["lspwm_rollout_trajectory_nll"] = rollout_nll / n_steps_used

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
