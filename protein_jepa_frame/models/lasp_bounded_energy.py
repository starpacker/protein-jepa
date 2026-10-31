"""Opt-in bounded-curvature nonconvex energy; legacy LASP energy is unchanged.

For fixed temperature and mask, residual Hessian spectral norm is at most
``curvature_budget``. Local ridge functions give independent residue blocks;
collective ridge functions act on sum(H)/sqrt(n_valid), an operator of norm one.
The positive quadratic tail is separate. This is not a contraction guarantee.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class BoundedRidgeEnergy(nn.Module):
    def __init__(self, latent_dim, features=128, curvature_budget=0.5,
                 tail_init=0.01, reference_temperature_k=348.):
        super().__init__()
        if latent_dim < 1 or features < 1:
            raise ValueError("dimensions must be positive")
        if not math.isfinite(curvature_budget) or curvature_budget <= 0:
            raise ValueError("curvature budget must be finite and positive")
        if not math.isfinite(tail_init) or tail_init <= 0:
            raise ValueError("tail must be finite and positive")
        self.curvature_budget = float(curvature_budget)
        self.reference_temperature_k = float(reference_temperature_k)
        self.weight = nn.Parameter(torch.randn(2, features, latent_dim) / math.sqrt(latent_dim))
        self.phase = nn.Parameter(torch.zeros(2, features))
        self.temperature_phase = nn.Parameter(torch.zeros(2, features))
        self.amplitude = nn.Parameter(torch.randn(2, features) * 0.01)
        self.allocation = nn.Parameter(torch.zeros(2, features))
        self.log_tail = nn.Parameter(torch.full((latent_dim,), math.log(math.expm1(tail_init))))

    def coefficients(self):
        # Smooth row norm < 1; no power-iteration estimate in the bound.
        weight = self.weight / (1 + self.weight.square().sum(-1, keepdim=True)).sqrt()
        amplitude = (self.curvature_budget * self.allocation.flatten().softmax(0)
                     .reshape_as(self.amplitude) * self.amplitude.tanh())
        return weight, amplitude

    def forward(self, H, residue_mask, temperature_k=None):
        mask = residue_mask.to(H.dtype)
        count = mask.sum(-1).clamp_min(1)
        H = H * mask[..., None]
        weight, amplitude = self.coefficients()
        if temperature_k is None:
            temp = H.new_zeros(H.shape[0])
        else:
            temp = torch.as_tensor(temperature_k, device=H.device, dtype=H.dtype)
            temp = temp.expand(H.shape[0]).reshape(-1)
            if not bool(torch.isfinite(temp).all() and (temp > 0).all()):
                raise ValueError("temperature must be finite and positive")
            temp = (temp / self.reference_temperature_k).log()
        phase = self.phase[None] + temp[:, None, None] * self.temperature_phase[None]
        local = torch.einsum('bld,kd->blk', H, weight[0]) + phase[:, 0, None]
        pooled = H.sum(1) / count.sqrt()[:, None]
        collective = pooled @ weight[1].T + phase[:, 1]
        residual = ((local.cos() * amplitude[0]).sum(-1) * mask).sum()
        residual = residual + (collective.cos() * amplitude[1]).sum()
        tail = 0.5 * (F.softplus(self.log_tail) * H.square()).sum()
        return tail + residual

    def residual_hessian_bound(self):
        weight, amplitude = self.coefficients()
        return (amplitude.abs() * weight.square().sum(-1)).sum()
