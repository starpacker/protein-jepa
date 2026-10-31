"""Development-only Ritz estimates of frozen-mobility energy curvature."""
from __future__ import annotations

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def mobility_curvature(kernel, state, anchor, temperature, mask, noise_factor=None,
                       iterations=20, seed=1729):
    """Estimate extreme Ritz values of N^T Hess(U) N, M=N N^T.

    The calculation uses masked latent noise coordinates, with N=sqrt(D) F.
    D and F stay fixed at the observed anchor. Finite Lanczos Ritz values are
    directional evidence, not certified bounds on the full Hessian spectrum.
    All samples form one block-diagonal operator; sample extrema are pooled.
    """
    if kernel.cfg.force_clip is not None:
        raise ValueError("energy curvature requires an unclipped energy force")
    with torch.no_grad():
        D, _, _ = kernel.bath(anchor, temperature)
        root = D.sqrt().detach()
    rm = mask[..., None].to(state.dtype)

    def forward(v):
        v = v * rm
        if noise_factor is not None:
            v = torch.einsum("bij,bjd->bid", noise_factor, v)
        return root * v * rm

    def adjoint(v):
        v = root * v * rm
        if noise_factor is not None:
            v = torch.einsum("bij,bjd->bid", noise_factor.transpose(-1, -2), v)
        return v * rm

    def dot(a, b):
        return (a.double() * b.double()).sum()

    generator = torch.Generator(device=state.device).manual_seed(seed)
    q = torch.randn(state.shape, dtype=state.dtype, device=state.device,
                    generator=generator) * rm
    q = q / dot(q, q).sqrt().to(q.dtype)
    basis, alphas, betas = [], [], []
    with sdpa_kernel(SDPBackend.MATH), torch.enable_grad():
        H = state.detach().requires_grad_(True)
        force = kernel.compute_forces(H, mask, temperature, create_graph=True)
        for index in range(iterations):
            basis.append(q)
            hessian_v = -torch.autograd.grad(
                force, H, grad_outputs=forward(q), retain_graph=True)[0]
            z = adjoint(hessian_v).detach()
            if not torch.isfinite(z).all():
                return {"finite": False, "iterations": index + 1}
            alpha = dot(q, z)
            alphas.append(float(alpha))
            if index:
                z = z - betas[-1] * basis[-2]
            z = z - alpha.to(z.dtype) * q
            # Two reorthogonalization passes control float32 loss of orthogonality.
            for _ in range(2):
                for previous in basis:
                    z = z - dot(previous, z).to(z.dtype) * previous
            beta = float(dot(z, z).sqrt())
            if beta < 1e-12 or index + 1 == iterations:
                break
            betas.append(beta)
            q = z / beta
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        off = torch.tensor(betas[:len(alphas) - 1], dtype=torch.float64)
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    eigenvalues = torch.linalg.eigvalsh(T)
    low, high = float(eigenvalues[0]), float(eigenvalues[-1])
    return {"finite": True, "iterations": len(alphas),
            "operator": "N^T Hess(U) N, N=sqrt(D) F; masked pooled sample operator",
            "ritz_min_per_ps": low, "ritz_max_per_ps": high,
            "macro_dt_times_ritz_min": kernel.cfg.step_ps * low,
            "macro_dt_times_ritz_max": kernel.cfg.step_ps * high,
            "certified_spectral_bounds": False}
