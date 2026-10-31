"""Joint-state VJP localization for the frozen-anchor collective Euler path."""
from __future__ import annotations

import math
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def joint_path_vjp(kernel, trajectory, bath, anchor, temperature, mask, noise_factor, endpoint_gradient):
    """Propagate the loss covector through local H/S Jacobians on a saved path.

    Additive noise has zero state derivative. D/C/Gamma and spatial covariance
    stay anchor-conditioned, as in the v1 sampler. This measures state recurrence,
    not parameter derivatives or an optimizer replay.
    """
    if kernel.cfg.bath_coupling != "velocity" or bath is None:
        raise ValueError("joint probe requires the collective velocity bath")
    with torch.no_grad():
        D, C, gamma = kernel.bath(anchor, temperature)
        root = D.sqrt()
    h = kernel.cfg.step_ps
    rm = mask[..., None].to(trajectory.dtype)
    v_h = endpoint_gradient.detach()
    v_s = torch.zeros_like(bath[:, -1])

    def norm(a, b):
        return float((a.double().square().sum() + b.double().square().sum()).sqrt())

    terminal_norm = norm(v_h, v_s)
    rows = []
    for step in reversed(range(trajectory.shape[1] - 1)):
        H = trajectory[:, step].detach().requires_grad_(True)
        S = bath[:, step].detach().requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
            force = kernel.compute_forces(H, mask, temperature, create_graph=True)
            drive = root * (force - torch.einsum("blij,blj->bli", C, S))
            if noise_factor is not None:
                drive = torch.einsum("bij,bjd->bid", noise_factor.transpose(-1, -2), drive)
                drive = torch.einsum("bij,bjd->bid", noise_factor, drive)
            displacement = h * root * drive
            next_h = (H + displacement) * rm
            next_s = (S + torch.einsum("blij,bli->blj", C, displacement) - h * gamma * S) * rm
            before = norm(v_h, v_s)
            v_h, v_s = torch.autograd.grad((next_h, next_s), (H, S), (v_h, v_s))
        after = norm(v_h, v_s)
        finite = bool(torch.isfinite(v_h).all() and torch.isfinite(v_s).all())
        rows.append(dict(
            step=step,
            finite=finite,
            input_covector_norm64=after if math.isfinite(after) else None,
            output_covector_norm64=before if math.isfinite(before) else None,
            directional_vjp_gain=after / before if before > 0 and math.isfinite(after) else None,
            gain_from_endpoint=after / terminal_norm if terminal_norm > 0 and math.isfinite(after) else None,
        ))
        v_h, v_s = v_h.detach(), v_s.detach()
        if not finite:
            break
    return dict(
        method="joint H/S local VJP along saved path, endpoint-loss covector",
        terminal_norm64=terminal_norm,
        steps=sorted(rows, key=lambda row: row["step"]),
    )
