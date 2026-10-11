"""Stage B loss functions for per-residue LaSP world model.

Gate 2 objectives:
- Transition NLL: per-residue Gaussian likelihood using same D, F as integrator
- Force floor: penalizes |F| below a minimum target (prevents flat energy landscape)
- Force direction: penalizes force magnitude explosion (soft ceiling)
- Energy regularization: prevents energy divergence
- Rollout NLL (P1.3): N-step autonomous rollout endpoint MSE with FDT noise
- Semigroup consistency (P1.3): z_{t→t+2} ≈ transition(transition(z_t, 1), 1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _transition(model, q, s, *, ca_distances, residue_mask,
                temperature_k, dt_ps, stochastic=False,
                max_substep_dt_ps=20.0, noise_scale=1.0):
    """Use the canonical closed-loop transition when available."""
    if hasattr(model, "transition"):
        return model.transition(
            q, s, ca_distances=ca_distances, residue_mask=residue_mask,
            temperature_k=temperature_k, dt_ps=dt_ps, stochastic=stochastic,
            max_substep_dt_ps=max_substep_dt_ps, noise_scale=noise_scale,
        )
    q_next, s_next = model.step(
        q, s, ca_distances=ca_distances, residue_mask=residue_mask,
        temperature_k=temperature_k, dt_ps=dt_ps, stochastic=stochastic,
    )
    return q_next, s_next, None


def _force_floor_loss(
    forces: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    target_threshold: float = 0.1,
    weight_inside: float = 1.0,
) -> torch.Tensor:
    """Barrier loss that penalises |F| falling below a minimum target.

    The loss is 0 when |F| >= target and rises quadratically for smaller forces.
    This prevents the energy landscape from collapsing into a perfectly flat
    potential where the model predicts q_next ≈ q_current regardless of
    conformation — the root cause of 100 % divergence in autonomous rollouts.

    With forces ~3e-4 and FDT noise ~0.59/dim, the noise/drift ratio of
    10,000:1 guarantees an unbounded random walk.  A floor of 0.1 still leaves
    noise 3× drift, well within the Langevin regime, while providing enough
    restoring force to keep the trajectory near the folded basin.

    Args:
        forces:          [B, L, D] conservative forces (-dA/dQ)
        residue_mask:    [B, L] valid-residue mask
        target_threshold: minimum force norm we require (default 0.1)
        weight_inside:   relative weight for forces below target vs above
    """
    f_norm = forces.norm(dim=-1)  # [B, L]
    below = F.relu(target_threshold - f_norm)  # 0 when |F| >= target
    if residue_mask is not None:
        penalty = below * residue_mask.float()
        avg = (penalty.pow(2) * weight_inside).sum() / residue_mask.sum().clamp(min=1.0)
    else:
        avg = (below.pow(2) * weight_inside).mean()
    return avg


def stage_b_total_loss(
    outputs: dict,
    *,
    nll_weight: float = 1.0,
    force_floor_weight: float = 0.0,
    force_floor_target: float = 0.1,
    force_direction_weight: float = 0.1,
    energy_reg_weight: float = 0.01,
    force_cosine_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total Stage B loss using per-residue Q, F, D.

    Args:
        outputs: from StageBLASP.forward()
            - q: [B, L, D]
            - forces: [B, L, D]
            - diffusion: [B, L]
            - transition_nll: scalar (optional)
            - energy_total: scalar (optional)
            - residue_mask: [B, L]
        force_floor_weight: weight for minimum-force barrier (prevents flat landscape)
        force_floor_target:  force norm target below which penalty rises
        force_direction_weight: weight for soft ceiling on force magnitude
    """
    device = outputs["q"].device
    total = torch.tensor(0.0, device=device)
    log = {}
    residue_mask = outputs.get("residue_mask")

    # ── Transition NLL ──
    if "transition_nll" in outputs and nll_weight > 0:
        nll = outputs["transition_nll"]
        total = total + nll_weight * nll
        log["transition_nll"] = float(nll.detach().cpu())

    # ── Force floor (prevents energy landscape collapse) ──
    if "forces" in outputs and force_floor_weight > 0:
        forces = outputs["forces"]  # [B, L, D]
        floor_loss = _force_floor_loss(
            forces, residue_mask,
            target_threshold=force_floor_target,
        )
        total = total + force_floor_weight * floor_loss
        log["force_floor_loss"] = float(floor_loss.detach().cpu())

    # ── Force magnitude regularization (soft ceiling) ──
    if "forces" in outputs and force_direction_weight > 0:
        forces = outputs["forces"]  # [B, L, D]
        f_norm = forces.norm(dim=-1)  # [B, L]
        if residue_mask is not None:
            force_mag = (f_norm * residue_mask.float()).sum() / residue_mask.sum().clamp(min=1.0)
        else:
            force_mag = f_norm.mean()
        total = total + force_direction_weight * force_mag
        log["force_mag_loss"] = float(force_mag.detach().cpu())

    # ── Force cosine alignment (with q_next if available) ──
    if "forces" in outputs and force_cosine_weight > 0 and "q_next" in outputs:
        dq = outputs["q_next"] - outputs["q"]
        forces = outputs["forces"]
        cos_sim = torch.cosine_similarity(
            forces.reshape(-1, forces.shape[-1]),
            dq.reshape(-1, dq.shape[-1]),
            dim=-1,
        ).reshape(forces.shape[:-1])
        if residue_mask is not None:
            cos_mean = (cos_sim * residue_mask.float()).sum() / residue_mask.sum().clamp(min=1)
        else:
            cos_mean = cos_sim.mean()
        # The force should point along the observed displacement.  This keeps
        # the force-floor regularizer from learning an arbitrary constant-norm
        # vector field, which produces a non-flat but non-restoring landscape.
        total = total + force_cosine_weight * (1.0 - cos_mean)
        log["force_cosine"] = float(cos_mean.detach().cpu())

    # ── Energy regularization ──
    if "energy_total" in outputs and energy_reg_weight > 0:
        energy = outputs["energy_total"]
        energy_loss = energy.pow(2).mean() * 0.01
        total = total + energy_reg_weight * energy_loss
        log["energy_reg_loss"] = float(energy_loss.detach().cpu())

    log["total_loss"] = float(total.detach().cpu())
    return total, log


# ═══════════════════════════════════════════════════════════════════════
# P1.3: Rollout and semigroup consistency losses
# ═══════════════════════════════════════════════════════════════════════

def rollout_mse_loss(
    model,
    q_0: torch.Tensor,
    s_0: torch.Tensor,
    n_steps: int,
    dt_ps: float,
    q_target: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    temperature_k: torch.Tensor | None = None,
    *,
    stochastic: bool = True,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """P2.1: Autonomous rollout endpoint MSE with FDT noise.

    Roll forward n_steps autonomously with FDT-consistent noise, then compute
    MSE w.r.t. q_target.  The noise is necessary so the model learns to produce
    restoring forces that are competitive with thermal fluctuations — without
    it the energy landscape collapses to |F|≈0 and autonomous rollouts diverge.

    Uses MSE instead of NLL to avoid the expensive compute_energy_and_forces
    call at the final state (which uses create_graph=True internally).

    NaN-safety: when forces are small (early training), a long stochastic rollout
    accumulates unbounded noise.  If the result is NaN, we return a safe zero
    loss — the force_floor loss will push |F| up and the rollout will become
    stable within a few epochs.

    Args:
        model: StageBLASP instance
        q_0: [B, L, D] initial Q
        s_0: [B, L, M] initial bath state
        n_steps: number of integration steps
        dt_ps: step size in ps
        q_target: [B, L, D] target Q (from future coords)
        ca_distances, residue_mask, temperature_k: environment
        stochastic: whether to inject FDT noise (default True)
        noise_scale: multiplier on noise std (1.0 = full FDT; <1.0 reduces noise)

    Returns:
        scalar MSE loss (or 0.0 if rollout diverged)
    """
    q_t = q_0.clone()
    s_t = s_0.clone()

    for step_i in range(n_steps):
        q_t, s_t, _ = _transition(model,
            q_t, s_t,
            ca_distances=ca_distances,
            residue_mask=residue_mask,
            temperature_k=temperature_k,
            dt_ps=dt_ps,
            stochastic=stochastic,
            noise_scale=noise_scale,
            max_substep_dt_ps=max(float(dt_ps), 100.0),
        )
        # Truncate intermediate state history, but keep the final state attached
        # so this auxiliary objective has a real gradient.
        q_t = q_t.detach() if step_i < n_steps - 1 else q_t
        if s_t is not None:
            s_t = s_t.detach()

        # Early-break if rollout diverged (force floor hasn't pushed |F| up yet)
        if step_i < n_steps - 1 and (torch.isnan(q_t).any() or torch.isinf(q_t).any()):
            return torch.tensor(0.0, device=q_0.device, dtype=q_0.dtype)

    # Simple MSE between rollout endpoint and target
    if torch.isnan(q_t).any() or torch.isinf(q_t).any():
        return torch.tensor(0.0, device=q_0.device, dtype=q_0.dtype)
    mask_f = residue_mask.unsqueeze(-1).float()
    diff = q_target - q_t
    mse = (diff.pow(2) * mask_f).sum() / mask_f.sum().clamp(min=1)
    return mse


def rollout_nll_loss(
    model,
    q_0: torch.Tensor,
    s_0: torch.Tensor,
    n_steps: int,
    dt_ps: float,
    q_target: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    temperature_k: torch.Tensor | None = None,
    *,
    stochastic: bool = True,
) -> torch.Tensor:
    """P1.3: Autonomous rollout endpoint NLL with FDT noise.

    Roll forward n_steps autonomously, then compute NLL w.r.t. q_target.

    Args:
        model: StageBLASP instance
        q_0: [B, L, D] initial Q
        s_0: [B, L, M] initial bath state
        n_steps: number of integration steps
        dt_ps: step size in ps
        q_target: [B, L, D] target Q (from future coords)
        ca_distances, residue_mask, temperature_k: environment
        stochastic: whether to inject FDT noise (default True)

    Returns:
        scalar NLL loss
    """
    q_t = q_0.clone()
    s_t = s_0.clone()

    for step_i in range(n_steps):
        q_t, s_t, _ = _transition(model,
            q_t, s_t,
            ca_distances=ca_distances,
            residue_mask=residue_mask,
            temperature_k=temperature_k,
            dt_ps=dt_ps,
            stochastic=stochastic,
        )
        q_t = q_t.detach() if step_i < n_steps - 1 else q_t
        if s_t is not None:
            s_t = s_t.detach()

        if step_i < n_steps - 1 and (torch.isnan(q_t).any() or torch.isinf(q_t).any()):
            return torch.tensor(0.0, device=q_0.device, dtype=q_0.dtype)

    if torch.isnan(q_t).any() or torch.isinf(q_t).any():
        return torch.tensor(0.0, device=q_0.device, dtype=q_0.dtype)

    # Compute forces at final state for NLL denominator
    force_out = model.compute_energy_and_forces(
        q_t,
        ca_distances, residue_mask, temperature_k,
    )
    forces_final = force_out["forces"]
    g_final = model.compute_g(q_t.detach(), residue_mask)
    diff_final = model.compute_per_residue_diffusion(q_t.detach(), g_final, temperature_k)

    # NLL: p(q_target | q_final, D(q_final)*dt) — use one-step NLL formulation
    mask_f = residue_mask.unsqueeze(-1).float()
    D = diff_final.unsqueeze(-1).clamp(min=1e-8)
    dt_tensor = torch.full((q_t.shape[0],), dt_ps, device=q_t.device, dtype=q_t.dtype)
    dt_s = dt_tensor.clamp(min=1e-6).unsqueeze(1).unsqueeze(1)
    # Mean = q_final + dt * D * F (one more step from final position)
    mean_pred = q_t.detach() + dt_s * D * forces_final
    cov = (2.0 * dt_s * D).clamp(min=1e-8)
    diff = q_target - mean_pred
    log_var = torch.log(cov)
    nll = 0.5 * (log_var + diff.pow(2) / cov)
    return (nll * mask_f).sum() / mask_f.sum().clamp(min=1)


def semigroup_consistency_loss(
    model,
    q_0: torch.Tensor,
    s_0: torch.Tensor,
    dt_ps: float,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    temperature_k: torch.Tensor | None = None,
    *,
    stochastic: bool = False,
    multiplier: float = 5.0,
) -> torch.Tensor:
    """P1.3: Semigroup consistency — z_{t→t+2k} ≈ transition(transition(z_t), k).

    Uses large-step (multiplier × dt_ps) to amplify force-field curvature.
    When the energy is locally linear (small Hessian), the 1+1 and 2 paths
    produce nearly identical endpoints.  Multiplying dt by 5 makes the
    displacement ~0.075×|F| per dim, enough for the force difference between
    intermediate states to create a measurable MSE.

    Deterministic (stochastic=False) is required — stochastic noise would
    dominate integration error for the variance.

    Returns:
        scalar MSE consistency loss
    """
    # Use amplified step sizes so the non-linear energy curvature can be seen.
    # Without amplification, |F|≈0.06 × dt=100ps → ΔQ ≈ 0.015 per dim, the
    # energy is locally linear over that displacement, and the difference
    # between 1+1 and 2 step paths is numerically zero in float32.
    base_dt = dt_ps * float(multiplier)
    big_step = float(base_dt)  # one substep per transition to avoid OOM

    # Two steps of base_dt
    q_a, s_a, _ = _transition(model, q_0.clone(), s_0.clone(),
        ca_distances=ca_distances, residue_mask=residue_mask,
        temperature_k=temperature_k, dt_ps=base_dt, stochastic=stochastic,
        max_substep_dt_ps=big_step)
    q_a = q_a.detach()
    if s_a is not None:
        s_a = s_a.detach()
    q_a2, _, _ = _transition(model, q_a, s_a,
        ca_distances=ca_distances, residue_mask=residue_mask,
        temperature_k=temperature_k, dt_ps=base_dt, stochastic=stochastic,
        max_substep_dt_ps=big_step)

    # One step of 2*base_dt
    q_b, _, _ = _transition(model, q_0.clone(), s_0.clone(),
        ca_distances=ca_distances, residue_mask=residue_mask,
        temperature_k=temperature_k, dt_ps=2.0 * base_dt, stochastic=stochastic,
        max_substep_dt_ps=2.0 * big_step)

    if (torch.isnan(q_a2).any() or torch.isinf(q_a2).any() or
        torch.isnan(q_b).any() or torch.isinf(q_b).any()):
        return torch.tensor(0.0, device=q_0.device, dtype=q_0.dtype)

    mse = ((q_a2 - q_b) ** 2)
    mask_f = residue_mask.unsqueeze(-1).float()
    return (mse * mask_f).sum() / mask_f.sum().clamp(min=1)


__all__ = [
    "stage_b_total_loss",
    "rollout_mse_loss",
    "rollout_nll_loss",
    "semigroup_consistency_loss",
]
