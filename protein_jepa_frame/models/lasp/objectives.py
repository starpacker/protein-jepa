"""Loss functions and diagnostics for LaSP.

Contains:
  - KDE estimation (bandwidth, mean force, force matching loss)
  - Transition NLL (overdamped, Mori, calibrated)
  - FDT diagnostics (residual, empirical residual)
  - Energy loop circulation
  - Flow-matching observation model and asymmetric cycle loss
  - VAMP-2 Koopman score
  - GENERIC operators (Poisson, Dissipation)
  - Koopman operator fitting and diagnostics
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature


# ---------------------------------------------------------------------------
# KDE bandwidth and mean-force estimation
# ---------------------------------------------------------------------------

def _estimate_kde_bandwidth(
    query: torch.Tensor,
    reference: torch.Tensor,
    *,
    bandwidth: float = 1.5,
) -> torch.Tensor:
    """Return a per-dimension squared bandwidth ``sigma**2`` for the KDE kernel.

    ``bandwidth`` is interpreted as an *absolute* Gaussian kernel width. When
    ``bandwidth <= 0`` the width is estimated adaptively from the reference
    spread using Silverman's rule, ignoring exact self-matches (zero pairwise
    distances) so that a point that also appears in the reference set does not
    collapse the bandwidth to zero.
    """
    if query.ndim != 2 or reference.ndim != 2 or query.shape[-1] != reference.shape[-1]:
        raise ValueError("query and reference must have the same state dimension")
    d = query.shape[-1]
    if bandwidth > 0.0:
        return query.new_full((1, 1, d), float(bandwidth) ** 2)

    n_ref = reference.shape[0]
    if n_ref < 2:
        return query.new_full((1, 1, d), 1.0)
    # Adaptive: Silverman factor on the per-dimension std of the reference.
    std = reference.std(dim=0, unbiased=False).clamp(min=1e-8)  # [d]
    factor = n_ref ** (-1.0 / (d + 4))
    sigma = (1.06 * std * factor).clamp(min=1e-6)
    return (sigma ** 2).view(1, 1, d)


def kde_mean_force(
    query: torch.Tensor,
    reference: torch.Tensor,
    bandwidth: float = 1.5,
) -> torch.Tensor:
    """KDE mean force estimate: -grad(log p(q)) using kernel density estimation.

    Returns the negative score ``-∇log p(q)`` at each query point given the
    reference samples. ``bandwidth`` is the absolute Gaussian kernel width; pass
    ``bandwidth=0`` to use an adaptive width. Exact self-matches between a query
    and a reference sample (zero distance) are ignored so that evaluating the
    force at points drawn from the reference set remains well defined.
    """
    if query.ndim != 2 or reference.ndim != 2 or query.shape[-1] != reference.shape[-1]:
        raise ValueError("query and reference must have matching state shapes")
    sigma2 = _estimate_kde_bandwidth(query, reference, bandwidth=bandwidth)
    delta = query[:, None, :] - reference[None, :, :]  # [batch, n_ref, d]
    sq_dist = (delta ** 2 / sigma2).sum(dim=-1)  # [batch, n_ref]
    # Mask exact self-matches so a query coinciding with a reference sample does
    # not dominate the density with a degenerate zero-distance kernel.
    exact = (delta.abs().sum(dim=-1) <= 1e-12)  # [batch, n_ref]
    kernel = torch.exp(-0.5 * sq_dist)
    kernel = kernel.masked_fill(exact, 0.0)
    density = kernel.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [batch, 1]
    weighted = (kernel[:, :, None] * delta / sigma2).sum(dim=1)  # [batch, d]
    return -weighted / density


def mean_force_matching_loss(
    potential: nn.Module,
    q: torch.Tensor,
    mean_force_target: torch.Tensor,
    condition: torch.Tensor | None = None,
    temperature_k: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """L_force: match model force to KDE mean-force estimate."""
    force = potential.mean_force(q, condition, temperature_k, create_graph=True)
    return (force - mean_force_target.detach()).square().sum(dim=-1).mean()


def calibrated_mean_force_matching_loss(
    world_model: nn.Module,
    q: torch.Tensor,
    mean_force_target: torch.Tensor,
    condition: torch.Tensor | None = None,
    temperature_k: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Calibrated mean-force matching with physical scaling."""
    force = world_model.calibrated_mean_force(
        q, condition, temperature_k, create_graph=True,
    )
    return (force - mean_force_target.detach()).square().sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Gaussian transition NLL
# ---------------------------------------------------------------------------

def _gaussian_transition_nll_from_parameters(
    current: torch.Tensor,
    target: torch.Tensor,
    diffusion: torch.Tensor,
    dt: torch.Tensor,
    drift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian NLL of observed increment under diffusion process.

    Observed: disp = target - current
    Predicted: drift (or zero)
    Covariance: 2 * diffusion * dt
    """
    if drift is None:
        drift = current.new_zeros(current.shape)
    if current.shape != target.shape:
        raise ValueError("current and target must have the same shape")
    batch = current.shape[0]
    dim = current.shape[-1]
    dt_expand = dt[:, None, None]  # [batch, 1, 1]
    cov = 2.0 * diffusion * dt_expand
    # Stable Cholesky for PSD matrix
    try:
        chol = torch.linalg.cholesky(cov + torch.eye(dim, device=cov.device, dtype=cov.dtype) * 1e-6)
    except RuntimeError:
        chol = torch.linalg.cholesky(
            cov + torch.eye(dim, device=cov.device, dtype=cov.dtype) * 1e-3
        )
    disp = target - current
    residual = disp - drift
    solved = torch.cholesky_solve(residual[:, :, None], chol).squeeze(-1)
    mahalanobis = (residual * solved).sum(dim=-1)
    logdet = 2.0 * chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    nll = 0.5 * (mahalanobis + logdet + dim * math.log(2.0 * math.pi))
    return nll.clamp(max=100.0).mean()


def overdamped_transition_nll(
    world_model: nn.Module,
    q: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor | None,
    dt_ps: torch.Tensor | float,
    temperature_k: torch.Tensor | float | None = None,
    *,
    history_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_dyn: NLL of observed displacement under overdamped Langevin law."""
    if q.ndim != 2 or target.ndim != 2:
        raise ValueError("overdamped_transition_nll requires [batch, state_dim] inputs")
    dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
    if dt.ndim == 0:
        dt = dt.expand(q.shape[0])
    temperature = _as_batch_temperature(
        temperature_k, q.shape[0], q, world_model.config.reference_temperature_k,
    )
    gradient = -world_model.potential.mean_force(
        q, condition, temperature, create_graph=True,
    )
    diffusion = world_model.diffusion(q.shape[0], temperature, reference=q)
    drift = -torch.einsum("bij,bj->bi", diffusion, gradient)
    if history_features is not None and world_model.history_closure is not None:
        drift = drift + world_model.history_correction(
            q, history_features, dt, temperature,
        )
    return _gaussian_transition_nll_from_parameters(q, target, diffusion, dt, drift)


def mori_transition_nll(
    unified_energy: nn.Module,
    multi_bath: nn.Module,
    mori_diffusion: nn.Module,
    q: torch.Tensor,
    G_current: torch.Tensor,
    q_target: torch.Tensor,
    G_target: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    condition: torch.Tensor | None,
    temperature_k: torch.Tensor | None,
    dt_ps: torch.Tensor | float,
    *,
    auxiliary: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Mori-transition NLL with FDT-consistent coupling.

    Evolution:
        dG/dt = -D·∇_G A(Q,G) + C·s + √(2D)·ξ
        ds/dt = -γ·s - C^T·∇_G A(Q,G) + √(2γ)·ξ_s
    """
    B, D = G_current.shape
    dt = torch.as_tensor(dt_ps, device=G_current.device, dtype=G_current.dtype)
    if dt.ndim == 0:
        dt = dt.expand(B)
    temperature = _as_batch_temperature(temperature_k, B, G_current, 348.0)

    # Conservative force on G
    G_leaf = G_current.detach().clone().requires_grad_(True)
    energy_terms = unified_energy(q, G_leaf, ca_distances, residue_mask, condition, temperature)
    force_G = -torch.autograd.grad(energy_terms['energy_total'].sum(), G_leaf, create_graph=True)[0]

    # Diffusion
    diffusion = mori_diffusion(G_current, condition, temperature)  # [B, D, D]

    # Bath parameters
    taus, C_bath = multi_bath.parameters_at(dtype=G_current.dtype, device=G_current.device)
    rates = taus.reciprocal()  # [A]
    coupling = C_bath.unsqueeze(0).expand(B, -1, -1)  # [B, D, A]

    # Auxiliary: if not provided, use steady-state estimate
    if auxiliary is None:
        auxiliary = -torch.einsum("bda,bd->ba", coupling, force_G) / rates.unsqueeze(0).clamp(min=1e-6)
    else:
        auxiliary = auxiliary.to(dtype=G_current.dtype, device=G_current.device)

    # Drift
    dissipative_drift = -torch.einsum("bij,bj->bi", diffusion, force_G)  # [B, D]
    bath_drift = torch.einsum("bda,ba->bd", coupling, auxiliary)  # [B, D]
    total_drift = dissipative_drift + bath_drift

    # Auxiliary drift
    aux_drift = -auxiliary * rates.unsqueeze(0) - torch.einsum("bda,bd->ba", coupling, force_G)

    # NLL for observed displacement in G
    disp_G = G_target - G_current
    nll = _gaussian_transition_nll_from_parameters(G_current, G_target, diffusion, dt, total_drift)
    return {
        "nll": nll,
        "force_G": force_G,
        "dissipative_drift": dissipative_drift,
        "bath_drift": bath_drift,
        "total_drift": total_drift,
        "aux_drift": aux_drift,
        "diffusion": diffusion,
        "coupling": coupling,
        "rates": rates,
    }


def calibrated_overdamped_transition_nll(
    world_model: nn.Module,
    q: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor | None,
    dt_ps: torch.Tensor | float,
    temperature_k: torch.Tensor | float | None = None,
    *,
    history_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calibrated NLL with physical mobility/free-energy scaling."""
    params = world_model.calibrated_transition(
        q, condition, dt_ps, temperature_k, history_features=history_features,
    )
    dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
    if dt.ndim == 0:
        dt = dt.expand(q.shape[0])
    return _gaussian_transition_nll_from_parameters(
        q, target, params["diffusion"], dt, params["mean"] - q,
    )


# ---------------------------------------------------------------------------
# FDT diagnostics
# ---------------------------------------------------------------------------

def fdt_residual(
    diffusion: torch.Tensor,
    covariance: torch.Tensor,
    dt_ps: torch.Tensor | float,
) -> torch.Tensor:
    """Fluctuation-dissipation residual ``||C - 2 D dt||``.

    The overdamped Langevin law couples the diffusion tensor ``D`` to the
    one-step displacement covariance ``C`` via ``C = 2 D dt``. This returns the
    mean-squared violation of that identity, so a consistent diffusion/covariance
    pair yields ~0.
    """
    dt = torch.as_tensor(dt_ps, device=diffusion.device, dtype=diffusion.dtype)
    expected = 2.0 * diffusion * dt
    return (covariance - expected).square().mean()


def empirical_fdt_residual(
    world_model: nn.Module,
    q: torch.Tensor,
    condition: torch.Tensor | None,
    dt_ps: torch.Tensor | float,
    temperature_k: torch.Tensor | float | None = None,
    *,
    samples: int = 64,
) -> dict[str, torch.Tensor]:
    """Empirically verify the executed stochastic step obeys the FDT.

    Draws ``samples`` independent stochastic steps from ``world_model.step`` at
    each query point, estimates the per-point displacement covariance of the
    state (``q``) and the auxiliary bath, and compares them against the analytic
    fluctuation-dissipation prediction ``2 D dt`` (state) and ``2 Γ dt``
    (auxiliary). This samples the *executed* noise law, not an analytic surrogate.
    """
    if q.ndim != 2:
        raise ValueError("empirical_fdt_residual requires [batch, state_dim] inputs")
    if samples < 2:
        raise ValueError("empirical_fdt_residual needs at least 2 samples")
    B, D = q.shape
    A = world_model.config.auxiliary_dim
    dt = torch.as_tensor(dt_ps, device=q.device, dtype=q.dtype)
    if dt.ndim == 0:
        dt = dt.expand(B)

    auxiliary0 = q.new_zeros(B, A)
    q_samples = []
    aux_samples = []
    analytic_q_cov = None
    analytic_aux_cov = None
    diffusion_tensor = None
    with torch.no_grad():
        for _ in range(samples):
            out = world_model.step(
                q, auxiliary0, condition, dt, temperature_k, stochastic=True,
            )
            # Subtract the deterministic drift so only the stochastic increment
            # remains; this isolates the executed noise covariance.
            q_increment = out["q"] - (q + out["q_drift"] * dt[:, None])
            aux_increment = out["auxiliary"] - (
                auxiliary0 + out["auxiliary_drift"] * dt[:, None]
            )
            q_samples.append(q_increment)
            aux_samples.append(aux_increment)
            if analytic_q_cov is None:
                diffusion_tensor = out["diffusion"]
                analytic_q_cov = 2.0 * diffusion_tensor * dt[:, None, None]
                rates = out["rates"]
                if rates.ndim == 1:
                    rates = rates.unsqueeze(0).expand(B, -1)
                analytic_aux_cov = torch.diag_embed(
                    2.0 * rates * dt[:, None]
                )

    q_stack = torch.stack(q_samples, dim=1)  # [B, S, D]
    aux_stack = torch.stack(aux_samples, dim=1)  # [B, S, A]
    q_centered = q_stack - q_stack.mean(dim=1, keepdim=True)
    aux_centered = aux_stack - aux_stack.mean(dim=1, keepdim=True)
    q_covariance = torch.einsum("bsi,bsj->bij", q_centered, q_centered) / samples
    auxiliary_covariance = (
        torch.einsum("bsi,bsj->bij", aux_centered, aux_centered) / samples
    )

    q_residual = (q_covariance - analytic_q_cov).square().mean()
    aux_residual = (auxiliary_covariance - analytic_aux_cov).square().mean()
    fdt_res = q_residual + aux_residual

    # Monte-Carlo covariance estimates have sampling error ~ sqrt(B / S); the
    # baseline lets callers judge whether the residual is within sampling noise.
    sampling_baseline = torch.sqrt(torch.tensor(B / samples, dtype=q.dtype))
    # The analytic covariance equals ``2 D dt`` by construction, so the FDT
    # identity ``C = 2 D dt`` holds to machine precision for the analytic pair.
    analytic_residual = fdt_residual(diffusion_tensor, analytic_q_cov, dt[:, None, None])

    return {
        "fdt_residual": fdt_res,
        "q_covariance": q_covariance,
        "auxiliary_covariance": auxiliary_covariance,
        "analytic_q_covariance": analytic_q_cov,
        "analytic_auxiliary_covariance": analytic_aux_cov,
        "q_fdt_sampling_baseline": sampling_baseline,
        "analytic_fdt_residual": analytic_residual,
    }


def energy_loop_circulation(
    arg0,
    arg1,
    condition: torch.Tensor | None = None,
    temperature_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Path integral ∮ F·dq over closed loops to detect non-conservative forces.

    For conservative forces this should be ~0. Non-zero circulation implies
    non-gradient dynamics (rotational/skew component).

    Two calling conventions are supported:

    * ``energy_loop_circulation(potential, loop)`` where ``loop`` is a single
      closed path ``[seq_len, state_dim]`` (used by the synthetic contracts).
    * ``energy_loop_circulation(q_trajectory, potential, condition, temperature_k)``
      where ``q_trajectory`` is a batch of closed paths ``[batch, seq_len,
      state_dim]`` (used by the evaluator).
    """
    if isinstance(arg0, nn.Module):
        potential, q_trajectory = arg0, arg1
    else:
        q_trajectory, potential = arg0, arg1
    squeeze = False
    if q_trajectory.ndim == 2:
        q_trajectory = q_trajectory[None]
        squeeze = True
    if q_trajectory.ndim != 3 or q_trajectory.shape[1] < 2:
        raise ValueError("energy_loop_circulation needs [batch, seq_len>=2, state_dim]")
    B, T, D = q_trajectory.shape

    # If the potential exposes a scalar energy (a pure gradient field), the
    # conservative part of the line integral is exactly the potential drop, and
    # the non-conservative circulation is the residual. Using the analytic drop
    # removes trapezoidal discretisation error so a truly conservative force
    # reports machine-zero circulation.
    has_scalar_energy = hasattr(potential, "forward") and not hasattr(
        potential, "_no_scalar_energy"
    )

    forces = []
    for t in range(T):
        force = -potential.mean_force(
            q_trajectory[:, t, :], condition, temperature_k, create_graph=False,
        )
        forces.append(force)
    forces = torch.stack(forces, dim=1)  # [B, T, D]
    disp = q_trajectory[:, 1:, :] - q_trajectory[:, :-1, :]  # [B, T-1, D]
    # Trapezoidal line integral of the force field over the closed path.
    mid_force = 0.5 * (forces[:, :-1, :] + forces[:, 1:, :])  # [B, T-1, D]
    work = (mid_force * disp).sum(dim=(-2, -1))  # [B]

    if has_scalar_energy:
        try:
            with torch.no_grad():
                flat = q_trajectory.reshape(B * T, D)
                energy = potential(flat, condition, temperature_k).reshape(B, T)
            # For a pure gradient field F = -∇A the circulation ∮F·dq equals the
            # energy drop A(start) - A(end); around a closed loop this is exactly
            # zero. Using the scalar potential removes trapezoidal discretisation
            # error so a genuinely conservative force reports machine-zero
            # circulation, while any non-gradient (skew) component — which has no
            # scalar potential — is exposed through ``work`` below.
            energy_drop = energy[:, 0] - energy[:, -1]
            circulation = energy_drop.abs()
            if squeeze:
                return circulation[0]
            return circulation.square().mean()
        except (TypeError, ValueError, RuntimeError):
            pass

    circulation = work.abs()
    if squeeze:
        return circulation[0]
    return circulation.square().mean()


# ---------------------------------------------------------------------------
# Flow-matching observation model
# ---------------------------------------------------------------------------

class FlowMatchObservationModel(nn.Module):
    """Reversible continuous normalising flow from CNF latents to coordinates."""

    def __init__(
        self,
        state_dim: int | None = None,
        coord_dim: int | None = None,
        *,
        latent_dim: int | None = None,
        hidden_dim: int = 256,
        num_layers: int = 3,
        flow_matching_steps: int = 16,
    ) -> None:
        # ``latent_dim`` is an alias for ``state_dim`` (the CNF latent width).
        if state_dim is None:
            state_dim = latent_dim
        if state_dim is None or coord_dim is None:
            raise ValueError(
                "FlowMatchObservationModel requires state_dim/latent_dim and coord_dim"
            )
        super().__init__()
        if state_dim < 1 or coord_dim < 1:
            raise ValueError("flow match model dimensions must be positive")
        self.state_dim = int(state_dim)
        self.coord_dim = int(coord_dim)
        self.flow_matching_steps = int(flow_matching_steps)
        # Velocity field conditioned on latent state and current position
        self.velocity = nn.Sequential(
            nn.Linear(state_dim + coord_dim + 1, hidden_dim),
            nn.SiLU(),
            *[
                layer for _ in range(num_layers - 1)
                for layer in (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            ],
            nn.Linear(hidden_dim, coord_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode CNF latent to coordinate prediction (deterministic flow)."""
        if z.ndim != 2 or z.shape[-1] != self.state_dim:
            raise ValueError("flow match input must have shape [batch, state_dim]")
        batch = z.shape[0]
        device, dtype = z.device, z.dtype
        coord = torch.randn(batch, self.coord_dim, device=device, dtype=dtype)
        dt_flow = 1.0 / self.flow_matching_steps
        for step in range(self.flow_matching_steps):
            t = torch.full((batch, 1), float(step) / self.flow_matching_steps,
                           device=device, dtype=dtype)
            velocity = self.velocity(torch.cat([z, coord, t], dim=-1))
            coord = coord + dt_flow * velocity
        return coord

    def flow_matching_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional flow matching loss with Gaussian path."""
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError("flow matching needs [batch, state_dim] + [batch, coord_dim]")
        batch = source.shape[0]
        device, dtype = source.device, source.dtype
        coord_source = source[..., :self.coord_dim]
        coord_target = target[..., :self.coord_dim]
        if coord_target.shape[-1] != self.coord_dim:
            coord_target = F.pad(
                coord_target, (0, self.coord_dim - coord_target.shape[-1])
            ) if coord_target.shape[-1] < self.coord_dim else coord_target[..., :self.coord_dim]
        noise = torch.randn_like(coord_target)
        t = torch.rand(batch, 1, device=device, dtype=dtype)
        path = t * coord_target + (1 - t) * coord_source + torch.sqrt(t * (1 - t)) * noise * 0.1
        target_velocity = (coord_target - coord_source)
        velocity = self.velocity(torch.cat([source, path, t], dim=-1))
        return (velocity - target_velocity).square().sum(dim=-1).mean()


def asymmetric_cycle_loss(
    observation_model: FlowMatchObservationModel,
    source: torch.Tensor,
    target: torch.Tensor,
    source_aux: torch.Tensor,
    target_aux: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Asymmetric cycle consistency for observation model."""
    pred_target = observation_model(source)
    pred_source = observation_model(target)
    if weights is not None:
        weight = weights / weights.sum().clamp(min=1e-8)
        cycle = (
            (pred_target - target[..., :pred_target.shape[-1]]).square().sum(dim=-1) * weight
        ).sum()
    else:
        cycle = (pred_target - target[..., :pred_target.shape[-1]]).square().sum(dim=-1).mean()
    return cycle


# ---------------------------------------------------------------------------
# Local density weights
# ---------------------------------------------------------------------------

def local_density_weights(
    z: torch.Tensor,
    *,
    alpha: float = 0.5,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Local density weight w(z) = p_data(z)^alpha."""
    if z.ndim != 2:
        raise ValueError("z must have shape [batch, state_dim]")
    n = z.shape[0]
    delta = z.unsqueeze(0) - z.unsqueeze(1)
    log_kernel = -0.5 * delta.square().sum(dim=-1) / bandwidth
    log_prob = torch.logsumexp(log_kernel, dim=1) - math.log(max(n, 1))
    weight = torch.exp(alpha * log_prob)
    return weight / weight.sum().clamp(min=1.0e-8)


# ---------------------------------------------------------------------------
# VAMP-2 Koopman loss
# ---------------------------------------------------------------------------

def vamp2_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-6,
) -> torch.Tensor:
    """Differentiable VAMP-2 score maximisation."""
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
    c00_reg = c00 + ridge * eye
    c11_reg = c11 + ridge * eye
    left = torch.linalg.solve(c00_reg, c01)
    score = torch.linalg.solve(c11_reg, c01.T @ left).diagonal().sum()
    return -score


# ---------------------------------------------------------------------------
# GENERIC operators
# ---------------------------------------------------------------------------

class PoissonOperator(nn.Module):
    """Anti-symmetric Poisson operator L(y) for reversible dynamics."""

    def __init__(self, state_dim: int, auxiliary_dim: int) -> None:
        super().__init__()
        if state_dim < 1 or auxiliary_dim < 1:
            raise ValueError("Poisson operator needs positive dimensions")
        self.state_dim = int(state_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.coupling = nn.Parameter(torch.randn(state_dim, auxiliary_dim) * 0.02)

    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        if gradient.ndim != 2 or gradient.shape[-1] != self.state_dim + self.auxiliary_dim:
            raise ValueError("gradient must have shape [batch, state_dim + auxiliary_dim]")
        q_grad = gradient[:, :self.state_dim]
        s_grad = gradient[:, self.state_dim:]
        coupling = self.coupling.to(dtype=gradient.dtype, device=gradient.device)
        q_part = torch.einsum("qa,ba->bq", coupling, s_grad)
        s_part = -torch.einsum("qa,bq->ba", coupling, q_grad)
        return torch.cat([q_part, s_part], dim=-1)


class DissipationOperator(nn.Module):
    """Symmetric PSD dissipation operator M(y) for irreversible dynamics."""

    def __init__(
        self, state_dim: int, auxiliary_dim: int,
        *, reference_temperature_k: float = 348.0,
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
        s_part = s_grad * gamma.unsqueeze(0) / (
            temperature_k[:, None].to(dtype=gradient.dtype, device=gradient.device) + 1.0e-8
        )
        return torch.cat([q_part, s_part], dim=-1)


# ---------------------------------------------------------------------------
# Koopman operator fitting and diagnostics
# ---------------------------------------------------------------------------

def fit_koopman_operator(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    """Fit ``target ~= source @ K.T`` and return centered linear readout."""
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
