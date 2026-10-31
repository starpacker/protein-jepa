"""Replay local joint-state derivatives along recorded LASP microstates."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.models.lasp_midpoint import CoupledMidpoint
from scripts.lasp_curvature import mobility_curvature


def _norm(h, s):
    return (h.double().square().sum() + s.double().square().sum()).sqrt()


def microstate_vjp(kernel, trace, anchor, temperature, mask, noise_factor,
                   brownian_h, brownian_s, endpoint_gradient, integrator,
                   curvature_iterations=20):
    """Replay actual transitions and propagate an endpoint covector locally.

    Anchor-conditioned coefficients stay detached. Curvature is sampled at the
    three largest local gain transitions. Gains are directional, not spectral
    norms, and a correlation with curvature alone does not prove causation.
    """
    started = time.perf_counter()
    microsteps = brownian_h.shape[2]
    dt = kernel.cfg.step_ps / microsteps
    with torch.no_grad():
        D, C, Gamma = kernel.bath(anchor, temperature)
        tail = F.softplus(kernel.energy.log_tail).detach()
    midpoint = None
    if integrator == "coupled_midpoint_v1":
        midpoint = CoupledMidpoint(D, C, Gamma, tail, mask, dt, noise_factor)
    vh = endpoint_gradient.detach()
    vs = torch.zeros_like(trace[-1]["S_after"], device=anchor.device)
    terminal = float(_norm(vh, vs))
    if terminal == 0:
        raise ValueError("microstate VJP requires a nonzero endpoint covector")
    rows = []
    max_difference = 0.
    for index in reversed(range(len(trace))):
        saved = trace[index]
        H = saved["H_before"].to(anchor.device).requires_grad_(True)
        S = saved["S_before"].to(anchor.device).requires_grad_(True)
        macro, micro = saved["macro_step"], saved["micro_step"]
        z_h, z_s = brownian_h[:, macro, micro], brownian_s[:, macro, micro]
        with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
            force = kernel.compute_forces(H, mask, temperature, create_graph=True)
            if midpoint is None:
                next_h, next_s = kernel._collective_velocity_step_fixed(
                    H, S, mask, temperature, D, C, Gamma, dt, create_graph=True,
                    noise_H=z_h, noise_S=z_s, noise_factor=noise_factor)
            else:
                next_h, next_s = midpoint.step(H, S, force, z_h, z_s)
            with torch.no_grad():
                difference = max(float((next_h - saved["H_after"].to(H.device)).abs().max()),
                                 float((next_s - saved["S_after"].to(H.device)).abs().max()))
                max_difference = max(max_difference, difference)
                if not torch.allclose(next_h, saved["H_after"].to(H.device), rtol=1e-5, atol=1e-5) or not torch.allclose(
                        next_s, saved["S_after"].to(H.device), rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f"microstate replay disagrees at {index}: {difference}")
                before = float(_norm(vh, vs))
                force_rms = float(force.double().square().mean().sqrt())
                displacement_rms = float((next_h - H).double().square().mean().sqrt())
            vh, vs = torch.autograd.grad((next_h, next_s), (H, S), (vh, vs))
        after = float(_norm(vh, vs))
        if not torch.isfinite(vh).all() or not torch.isfinite(vs).all():
            raise FloatingPointError(f"nonfinite local microstate VJP at {index}")
        rows.append({"index": index, "macro_step": macro, "micro_step": micro,
                     "physical_time_ps": (index + 1) * dt,
                     "local_directional_gain": after / before if before else None,
                     "gain_from_endpoint": after / terminal,
                     "force_rms": force_rms, "displacement_rms": displacement_rms,
                     "forward_max_abs_difference": difference,
                     "solver": saved["solver"]})
        vh, vs = vh.detach(), vs.detach()
    ranked = sorted(rows, key=lambda row: row["local_directional_gain"] or 0., reverse=True)
    if curvature_iterations:
        for row in ranked[:3]:
            state = trace[row["index"]]["H_before"].to(anchor.device)
            row["curvature_before"] = mobility_curvature(
                kernel, state, anchor, temperature, mask, noise_factor, curvature_iterations)
            row["curvature_before"]["micro_dt_times_ritz_min"] = (
                dt * row["curvature_before"]["ritz_min_per_ps"])
            row["curvature_before"]["micro_dt_times_ritz_max"] = (
                dt * row["curvature_before"]["ritz_max_per_ps"])
    return {"method": "joint H/S local VJP at recorded microstates with replay agreement",
            "steps": sorted(rows, key=lambda row: row["index"]),
            "joint_gain_from_endpoint": float(_norm(vh, vs)) / terminal,
            "forward_max_abs_difference": max_difference,
            "initial_H_covector": vh.detach(), "initial_S_covector": vs.detach(),
            "elapsed_seconds": time.perf_counter() - started}
