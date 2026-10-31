"""Coupled linear midpoint step for frozen-anchor collective LASP dynamics.

The quadratic H potential and the complete H/S mobility are implicit. The
nonlinear residual force is explicit. This is an IMEX method, not an exact
transition for the nonlinear learned energy. Shared H/bath noise is retained.
"""
from __future__ import annotations

import math
from copy import copy

import torch
from torch.autograd.function import once_differentiable


class _ImplicitSolve(torch.autograd.Function):
    """Differentiate A^-1 b through an adjoint solve, including zero b."""

    @staticmethod
    def forward(ctx, rhs, root_D, C, root_Gamma, root_tail, factor, mask, solver):
        solution = solver._solve_cg(rhs)
        ctx.solver = solver
        ctx.save_for_backward(solution, root_D, C, root_Gamma, root_tail, factor, mask)
        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        solution, root_D, C, root_Gamma, root_tail, factor, mask = ctx.saved_tensors
        solver = copy(ctx.solver)
        original = (root_D, C, root_Gamma, root_tail, factor)
        names = ("root_D", "C", "root_Gamma", "root_tail", "factor")
        leaves = []
        for index, (name, value) in enumerate(zip(names, original), 1):
            leaf = None if value is None else value.detach().requires_grad_(ctx.needs_input_grad[index])
            setattr(solver, name, leaf)
            leaves.append(leaf)
        solver.mask = mask
        with torch.no_grad():
            adjoint = solver._solve_cg(grad_output)
        requested = [leaf for leaf in leaves if leaf is not None and leaf.requires_grad]
        parameter_grads = []
        if requested:
            with torch.enable_grad():
                applied = solver.system(solution.detach())
                parameter_grads = list(torch.autograd.grad(
                    applied, requested, grad_outputs=-adjoint, allow_unused=True))
        grads = iter(parameter_grads)
        result = [next(grads) if leaf is not None and leaf.requires_grad else None
                  for leaf in leaves]
        return (adjoint, *result, None, None)


def _inner(a, b):
    return (a * b).flatten(1).sum(1).reshape(-1, 1, 1)


class CoupledMidpoint:
    """Matrix-free SPD solve in energy coordinates y=(sqrt(tail)*H, S).

    With Q=diag(tail,I), B=sqrt(Q) G sqrt(Q) is symmetric PSD. The
    linear-only transition is (I+dt*B/2)^-1 (I-dt*B/2); applying the same
    inverse to sqrt(2*dt)*sqrt(Q)*L noise preserves unit covariance in y.
    This covariance identity does not assert exact finite-time OU covariance.
    """

    def __init__(self, D, C, Gamma, tail, mask, dt, noise_factor=None,
                 tolerance=1e-6, max_iterations=128):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("midpoint dt must be finite and positive")
        if not 0 < tolerance < 1 or max_iterations < 1:
            raise ValueError("invalid midpoint solver tolerance or iteration limit")
        if not torch.isfinite(tail).all() or not (tail > 0).all():
            raise ValueError("midpoint needs a finite positive quadratic tail")
        self.root_D = D.sqrt()
        self.C = C
        self.root_Gamma = Gamma.sqrt()
        self.root_tail = tail.sqrt()
        self.mask = mask.unsqueeze(-1).to(D.dtype)
        self.dt = dt
        self.factor = noise_factor
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.dH = D.shape[-1]
        self.last_solve = None

    def mobility(self, x):
        """Apply joint G, including spatial covariance and H/S cross terms."""
        h, s = x[..., :self.dH], x[..., self.dH:]
        p = (h + torch.einsum("blij,blj->bli", self.C, s)) * self.mask
        drive = self.root_D * p
        if self.factor is not None:
            drive = torch.einsum("bij,bjd->bid", self.factor.transpose(-1, -2), drive)
            drive = torch.einsum("bij,bjd->bid", self.factor, drive)
        mh = self.root_D * drive * self.mask
        ms = (torch.einsum("blij,bli->blj", self.C, mh)
              + self.root_Gamma.square() * s) * self.mask
        return torch.cat((mh, ms), dim=-1)

    def scale(self, x):
        return torch.cat((self.root_tail * x[..., :self.dH], x[..., self.dH:]), dim=-1)

    def block(self, y):
        return self.scale(self.mobility(self.scale(y)))

    def system(self, y):
        return y + (0.5 * self.dt) * self.block(y)

    def solve(self, rhs):
        return _ImplicitSolve.apply(rhs, self.root_D, self.C, self.root_Gamma,
                                    self.root_tail, self.factor, self.mask, self)

    def _solve_cg(self, rhs):
        # Each sample has independent CG coefficients and a checked true residual.
        # The custom adjoint differentiates the solve independently of early exit.
        x = torch.zeros_like(rhs)
        r = rhs
        p = r
        rr = _inner(r, r)
        rhs_norm = rr.detach().sqrt()
        threshold = self.tolerance * rhs_norm.clamp_min(torch.finfo(rhs.dtype).tiny)
        iterations = 0
        tiny = torch.finfo(rhs.dtype).tiny
        for iterations in range(1, self.max_iterations + 1):
            if bool((rr.detach().sqrt() <= threshold).all()):
                break
            active = rr.detach().sqrt() > threshold
            ap = self.system(p)
            denominator = _inner(p, ap)
            if bool((active & ((denominator.detach() <= 0) | ~torch.isfinite(denominator.detach()))).any()):
                raise FloatingPointError("midpoint CG lost positive definiteness")
            alpha = torch.where(active, rr / denominator.clamp_min(tiny), torch.zeros_like(rr))
            x = x + alpha * p
            r = r - alpha * ap
            rr_new = _inner(r, r)
            beta = torch.where(active, rr_new / rr.clamp_min(tiny), torch.zeros_like(rr))
            p = r + beta * p
            rr = rr_new
        # Recompute the true residual rather than trusting recursive CG residuals.
        residual = (self.system(x) - rhs).detach()
        relative = _inner(residual, residual).sqrt() / rhs_norm.clamp_min(tiny)
        self.last_solve = {"iterations": iterations,
                           "relative_residual_max": float(relative.max())}
        if not torch.isfinite(relative).all() or bool((relative > 5 * self.tolerance).any()):
            raise FloatingPointError(f"midpoint CG did not converge: {self.last_solve}")
        return x

    def step(self, H, S, force, noise_H, noise_S):
        y = self.scale(torch.cat((H, S), dim=-1))
        residual_force = force + self.root_tail.square() * H
        residual_drive = self.scale(self.mobility(torch.cat(
            (residual_force, torch.zeros_like(S)), dim=-1)))
        if self.factor is not None:
            noise_H = torch.einsum("bij,bjd->bid", self.factor, noise_H)
        eta_h = self.root_D * noise_H * self.mask
        eta_s = (torch.einsum("blij,bli->blj", self.C, eta_h)
                 + self.root_Gamma * noise_S) * self.mask
        eta = self.scale(torch.cat((eta_h, eta_s), dim=-1))
        rhs = y - 0.5 * self.dt * self.block(y) + self.dt * residual_drive
        rhs = rhs + math.sqrt(2 * self.dt) * eta
        result = self.solve(rhs)
        return (result[..., :self.dH] / self.root_tail * self.mask,
                result[..., self.dH:] * self.mask)
