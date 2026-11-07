"""Experimental exact bath filtering with cross-residue mobility.

This module supplies the StageB1b ``exact_spatial`` mode with a full-covariance
filter for the discrete transition

    dH = h M (f - C S) + eta_H,       Cov(eta_H) = 2 h M,
    S_next = (I - h Gamma) S + C.T dH + eta_S,
                                      Cov(eta_S) = 2 h Gamma,

where the two noises are independent, C is block diagonal over residues and
M = sqrt(D) (K kron I_latent) sqrt(D). The returned Cholesky is a FULL bath
factor: independent per-residue sampling would discard its correlations.

All residues supplied here must be valid. Integrating this with padded batches
requires packing valid residues and restoring their indices outside the filter.
K and the prior covariance must be strictly positive definite. This is exact
Gaussian conditioning for the stated Euler transition, not an exact continuous
time integrator or evidence of improved forecasts.
"""
from __future__ import annotations

import torch
from torch import Tensor


def apply_spatial_mobility(value: Tensor, diffusion: Tensor,
                           spatial_covariance: Tensor) -> Tensor:
    """Apply M to [B,L,dH] without constructing the [L*dH,L*dH] matrix."""
    root = diffusion.sqrt()
    return root * (spatial_covariance @ (root * value))


def bath_mobility_gram(diffusion: Tensor, coupling: Tensor,
                       spatial_covariance: Tensor) -> Tensor:
    """Return C.T M C as [B,L*dS,L*dS], using block-diagonal C."""
    batch, length, _, bath_dim = coupling.shape
    weighted = diffusion.sqrt()[..., None] * coupling
    rows = weighted.transpose(-1, -2).reshape(batch, length * bath_dim, -1)
    gram = (rows @ rows.transpose(-1, -2)).reshape(
        batch, length, bath_dim, length, bath_dim)
    gram = gram * spatial_covariance[:, :, None, :, None]
    return gram.reshape(batch, length * bath_dim, length * bath_dim)


def correlated_bath_filter_step(
    mean: Tensor,
    prior_cholesky: Tensor,
    displacement: Tensor,
    force: Tensor,
    diffusion: Tensor,
    coupling: Tensor,
    gamma: Tensor,
    spatial_covariance: Tensor,
    step: float | Tensor,
    *,
    validate: bool = True,
) -> tuple[Tensor, Tensor]:
    """Condition on one observed dH, then propagate S to the next H frame.

    mean/gamma: [B,L,dS]; displacement/force/diffusion: [B,L,dH];
    coupling: [B,L,dH,dS]; spatial_covariance K: [B,L,L];
    prior_cholesky: [B,L*dS,L*dS], lower triangular, residue-major order.
    step is scalar or [B]. Returns next mean and a full lower Cholesky.

    If P = L L.T and Q = (h/2) C.T M C, use
    P_plus = L (I + L.T Q L)^-1 L.T, avoiding explicit precision matrices.
    The minus sign in m_plus = m - .5 P_plus C.T (dH-hMf+hMCm)
    comes from the observation matrix -hMC. Once dH is observed, C.T dH is
    deterministic: adding C.T (2hM) C to the next covariance would count that
    noise twice.

    No hidden jitter changes the likelihood. Ill-conditioned inputs raise a
    Cholesky error; dtype/conditioning must be addressed by the caller and its
    matching sampler. validate=False skips numerical checks after callers have
    established the contract, while retaining shape checks.
    """
    batch, length, bath_dim = mean.shape
    latent_dim = displacement.shape[-1]
    bath_size = length * bath_dim
    shapes = {
        'prior_cholesky': (prior_cholesky, (batch, bath_size, bath_size)),
        'displacement': (displacement, (batch, length, latent_dim)),
        'force': (force, (batch, length, latent_dim)),
        'diffusion': (diffusion, (batch, length, latent_dim)),
        'coupling': (coupling, (batch, length, latent_dim, bath_dim)),
        'gamma': (gamma, (batch, length, bath_dim)),
        'spatial_covariance': (spatial_covariance, (batch, length, length)),
    }
    for name, (value, expected) in shapes.items():
        if tuple(value.shape) != expected:
            raise ValueError(f'{name} must have shape {expected}')
        if value.dtype != mean.dtype or value.device != mean.device:
            raise ValueError(f'{name} must share mean dtype and device')
    h = torch.as_tensor(step, device=mean.device, dtype=mean.dtype)
    if h.numel() == 1:
        h = h.reshape(1).expand(batch)
    elif h.shape != (batch,):
        raise ValueError('step must be scalar or [B]')
    if validate:
        if not torch.isfinite(mean).all() or not torch.isfinite(h).all():
            raise ValueError('mean and step must be finite')
        if any(not torch.isfinite(value).all() for value, _ in shapes.values()):
            raise ValueError('filter inputs must be finite')
        if not (h > 0).all() or not (diffusion > 0).all() or not (gamma >= 0).all():
            raise ValueError('step/diffusion must be positive and gamma nonnegative')
        if not torch.allclose(spatial_covariance, spatial_covariance.transpose(-1, -2)):
            raise ValueError('spatial_covariance must be symmetric positive definite')
        _, info = torch.linalg.cholesky_ex(spatial_covariance)
        if (info != 0).any():
            raise ValueError('spatial_covariance must be symmetric positive definite')
        if (prior_cholesky.diagonal(dim1=-2, dim2=-1) <= 0).any() or \
                torch.count_nonzero(prior_cholesky.triu(diagonal=1)):
            raise ValueError('prior_cholesky must be a positive-diagonal lower factor')

    h_matrix = h[:, None, None]
    information = .5 * h_matrix * bath_mobility_gram(
        diffusion, coupling, spatial_covariance)
    whitened = prior_cholesky.transpose(-1, -2) @ information @ prior_cholesky
    whitened = .5 * (whitened + whitened.transpose(-1, -2))
    whitened = whitened + torch.eye(bath_size, device=mean.device, dtype=mean.dtype)
    whitened_cholesky = torch.linalg.cholesky(whitened)
    solved = torch.linalg.solve_triangular(
        whitened_cholesky, prior_cholesky.transpose(-1, -2), upper=False)
    posterior_covariance = solved.transpose(-1, -2) @ solved

    mean_bath_force = torch.einsum('blhs,bls->blh', coupling, mean)
    residual = displacement - h_matrix * apply_spatial_mobility(
        force - mean_bath_force, diffusion, spatial_covariance)
    score = torch.einsum('blhs,blh->bls', coupling, residual).reshape(batch, bath_size)
    correction = .5 * (posterior_covariance @ score[..., None]).squeeze(-1)
    posterior_mean = mean - correction.reshape_as(mean)

    decay = 1. - h_matrix * gamma
    next_mean = decay * posterior_mean + torch.einsum(
        'blhs,blh->bls', coupling, displacement)
    flat_decay = decay.reshape(batch, bath_size)
    next_covariance = posterior_covariance * flat_decay[:, :, None] * flat_decay[:, None, :]
    next_covariance = next_covariance + torch.diag_embed(
        (2. * h_matrix * gamma).reshape(batch, bath_size))
    next_covariance = .5 * (next_covariance + next_covariance.transpose(-1, -2))
    return next_mean, torch.linalg.cholesky(next_covariance)
