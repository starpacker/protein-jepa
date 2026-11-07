"""Temporary frozen-encoder relpos intervention and pairwise H bookkeeping.

No encoder is constructed or called here. The caller owns baseline replay,
checkpoint/configuration provenance, RNG/state audits, and all-frame gating.
The shared-P control measures gauge sensitivity; equivariance is not expected.
Independent pairwise replacements do not define a consistent new trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.modules import module as module_hooks

CENTERED_H_EPS = 1.0e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _validate_mask(mask: torch.Tensor, length: int) -> None:
    _require(isinstance(mask, torch.Tensor) and mask.device.type == 'cpu'
             and mask.dtype == torch.bool and mask.shape == (length,), 'mask must be CPU bool [L]')
    _require(bool(mask.any()), 'empty fixed support')


def _permutation(permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(isinstance(permutation, torch.Tensor) and permutation.device.type == 'cpu'
             and permutation.dtype in (torch.float32, torch.float64)
             and permutation.shape == (3, 3), 'P must be CPU float32/64 [3,3]')
    P = permutation.detach().double().clone()
    _require(bool(((P == 0) | (P == 1) | (P == -1)).all())
             and torch.equal(P.abs().sum(0), torch.ones(3, dtype=torch.float64))
             and torch.equal(P.abs().sum(1), torch.ones(3, dtype=torch.float64)),
             'P must be an exact signed permutation')
    order = P.abs().argmax(0)
    signs = P[order, torch.arange(3)]
    return P, order, signs


def _no_other_hooks(encoder: nn.Module, own_module: nn.Module | None = None,
                    own_id: int | None = None) -> None:
    for name in ('_global_forward_pre_hooks', '_global_forward_hooks',
                 '_global_backward_pre_hooks', '_global_backward_hooks'):
        _require(not getattr(module_hooks, name, {}), 'global hooks are not allowed')
    for child in encoder.modules():
        for name in ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks'):
            keys = set(getattr(child, name, {}))
            if child is own_module and name == '_forward_pre_hooks':
                keys.discard(own_id)
            _require(not keys, 'other encoder hooks are not allowed')


@contextmanager
def relpos_input_override(
    encoder: nn.Module, expected_canonical: torch.Tensor,
    permutation: torch.Tensor, mask: torch.Tensor,
) -> Iterator[dict[str, Any]]:
    """Replace exactly one B1 relpos input, then remove this hook even on error.

    Call under no_grad with an eval/frozen CPU float32 encoder. The captured
    input must match expected_canonical *bitwise*, including signed zeros.
    P acts on columns: output[...,j] = sign[j]*input[...,order[j]]. An identity
    P returns the original input object. Other P values use gather/sign only,
    avoiding an extra floating-point matrix multiplication. No input is mutated.
    """
    _require(isinstance(encoder, nn.Module), 'encoder must be an nn.Module')
    _require(not torch.is_grad_enabled(), 'no_grad is required')
    _require(not torch.is_autocast_enabled('cpu'), 'CPU autocast is forbidden')
    _require(all(not child.training for child in encoder.modules()), 'encoder must be in eval mode')
    _require(all(not p.requires_grad and p.grad is None and p.device.type == 'cpu'
                 and p.dtype == torch.float32 for p in encoder.parameters()),
             'encoder parameters must be frozen gradient-free CPU float32')
    _require(all(b.device.type == 'cpu' for b in encoder.buffers()), 'encoder buffers must be on CPU')
    target = getattr(encoder, 'relpos_proj', None)
    _require(type(target) is nn.Linear and target.in_features == 3,
             'encoder.relpos_proj must be the intended nn.Linear with three inputs')
    _require(isinstance(expected_canonical, torch.Tensor) and expected_canonical.device.type == 'cpu'
             and expected_canonical.dtype == torch.float32 and expected_canonical.ndim == 3
             and expected_canonical.shape[0] == 1 and expected_canonical.shape[2] == 3,
             'canonical input must be CPU float32 [1,L,3]')
    _validate_mask(mask, expected_canonical.shape[1])
    _require(bool(torch.isfinite(expected_canonical).all())
             and bool((expected_canonical[:, ~mask] == 0).all()), 'invalid canonical fixed support/padding')
    expected = expected_canonical.detach().clone()
    P, order, signs = _permutation(permutation)
    identity = torch.equal(P, torch.eye(3, dtype=torch.float64))
    _no_other_hooks(encoder)
    witness: dict[str, Any] = dict(permutation=P, call_count=0, is_identity=identity,
                                   identity_input_object_preserved=False, hook_removed=False)

    def hook(module: nn.Module, args: tuple[Any, ...]):
        _require(not torch.is_grad_enabled() and not torch.is_autocast_enabled('cpu'),
                 'forward requires no_grad and disabled CPU autocast')
        witness['call_count'] += 1
        _require(witness['call_count'] == 1, 'relpos hook must be called exactly once')
        _require(module is target and len(args) == 1 and isinstance(args[0], torch.Tensor),
                 'unexpected relpos invocation')
        actual = args[0]
        _require(actual.device.type == 'cpu' and actual.dtype == torch.float32
                 and _bitwise_equal(actual, expected), 'captured relpos differs from source float32 canonical')
        _no_other_hooks(encoder, target, handle.id)
        if identity:
            changed = actual
        else:
            gathered = actual.index_select(-1, order)
            changed = torch.where(signs[None, None] < 0, -gathered, gathered)
        witness['input'] = actual.detach().clone()
        witness['output'] = changed.detach().clone()
        witness['identity_input_object_preserved'] = identity and changed is actual
        return (changed,)

    handle = target.register_forward_pre_hook(hook)
    try:
        yield witness
        _require(witness['call_count'] == 1, 'relpos hook must be called exactly once')
        _no_other_hooks(encoder, target, handle.id)
    finally:
        handle.remove()
        witness['hook_removed'] = handle.id not in target._forward_pre_hooks


def _center(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean(-1, keepdim=True)


def _squared(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(-1)


def _mean(x: torch.Tensor, valid: torch.Tensor | None = None) -> float | None:
    x = x if valid is None else x[valid]
    return float(x.mean()) if x.numel() else None


@torch.no_grad()
def pair_counterfactual_audit(
    H0: torch.Tensor, H1: torch.Tensor, H1_next_only: torch.Tensor,
    H0_shared: torch.Tensor, H1_shared: torch.Tensor, mask: torch.Tensor,
) -> dict[str, Any]:
    """All-support raw/centered increment effects; no normalization or selection.

    next_only effect is E1; shared_P effect is E1-E0. Shared next H must be the
    exact same output as next_only next H. Tangent directions for original and
    next_only use original H0; shared_P uses H0_shared. These are explicitly
    different reference directions, not a common-projector improvement metric.
    Undefined centered directions remain NaN with a mask and null summaries.
    """
    _require(isinstance(H0, torch.Tensor) and H0.ndim == 3 and H0.shape[0] == 1
             and H0.shape[-1] > 0, 'H must have shape [1,L,d]')
    _validate_mask(mask, H0.shape[1])
    tensors = (H0, H1, H1_next_only, H0_shared, H1_shared)
    for H in tensors:
        _require(isinstance(H, torch.Tensor) and H.shape == H0.shape
                 and H.dtype == torch.float32 and H.device.type == 'cpu',
                 'all H must have identical CPU float32 [1,L,d] shape')
        _require(bool(torch.isfinite(H[:, mask]).all()), 'nonfinite H on fixed support')
    _require(_bitwise_equal(H1_next_only, H1_shared), 'shared next H differs bitwise from reused next-only H')
    x0, x1, y1, y0, shared1 = [H[0, mask].double() for H in tensors]
    baseline = x1 - x0
    E0, E1 = y0 - x0, y1 - x1
    arrays = dict(valid_indices=torch.nonzero(mask, as_tuple=False).flatten(),
                  effect_current=E0, effect_next=E1)
    summary: dict[str, Any] = dict(residue_count=int(mask.sum()), latent_dim=H0.shape[-1])
    modes = (
        ('original', x0, x1, torch.zeros_like(E1), 'original_current_H'),
        ('next_only', x0, y1, E1, 'original_current_H'),
        ('shared_P', y0, shared1, E1 - E0, 'shared_P_current_H'),
    )
    for name, start, end, effect, reference in modes:
        increment = end - start
        centered_increment = _center(increment)
        centered_baseline, centered_effect = _center(baseline), _center(effect)
        raw_cross = 2 * (baseline * effect).sum(-1)
        centered_cross = 2 * (centered_baseline * centered_effect).sum(-1)
        raw_closure = _squared(increment) - _squared(baseline) - _squared(effect) - raw_cross
        centered_closure = (_squared(centered_increment) - _squared(centered_baseline)
                            - _squared(centered_effect) - centered_cross)
        z0, z1 = _center(start), _center(end)
        radius = z0.norm(dim=-1)
        valid = radius > CENTERED_H_EPS
        radial = torch.full_like(radius, float('nan'))
        tangent_squared = torch.full_like(radius, float('nan'))
        direction = z0[valid] / radius[valid, None]
        radial[valid] = (direction * centered_increment[valid]).sum(-1)
        tangent_squared[valid] = _squared(centered_increment[valid] - radial[valid, None] * direction)
        values = dict(increment=increment, centered_increment=centered_increment, effect=effect,
            centered_effect=centered_effect, raw_increment_squared=_squared(increment),
            centered_increment_squared=_squared(centered_increment), raw_effect_squared=_squared(effect),
            centered_effect_squared=_squared(centered_effect), raw_cross=raw_cross, centered_cross=centered_cross,
            raw_norm_squared_change=_squared(end)-_squared(start),
            centered_norm_squared_change=_squared(z1)-_squared(z0),
            channel_increment_squared=increment.mean(-1).square()*increment.shape[-1],
            raw_decomposition_closure=raw_closure, centered_decomposition_closure=centered_closure,
            increment_identity_residual=increment-baseline-effect,
            centered_increment_identity_residual=centered_increment-centered_baseline-centered_effect,
            tangent_valid=valid, tangent_initial_centered_norm=radius,
            radial_increment=radial, radial_increment_squared=radial.square(),
            tangent_increment_squared=tangent_squared,
            finite_chord_closure=(_squared(z1)-_squared(z0)
                - (2*radius*radial+radial.square()+tangent_squared)))
        arrays.update({name+'__'+key: value for key, value in values.items()})
        metrics = {key+'_mean': _mean(values[key]) for key in (
            'raw_increment_squared', 'centered_increment_squared', 'raw_effect_squared',
            'centered_effect_squared', 'raw_cross', 'centered_cross', 'raw_norm_squared_change',
            'centered_norm_squared_change', 'channel_increment_squared')}
        metrics.update(tangent_reference=reference, tangent_valid_count=int(valid.sum()),
            tangent_undefined_count=int((~valid).sum()),
            tangent_increment_squared_mean=_mean(tangent_squared, valid),
            radial_increment_mean=_mean(radial, valid), radial_increment_squared_mean=_mean(radial.square(), valid),
            raw_decomposition_max_abs_error=float(raw_closure.abs().max()),
            centered_decomposition_max_abs_error=float(centered_closure.abs().max()),
            increment_identity_max_abs_error=float(values['increment_identity_residual'].abs().max()),
            centered_increment_identity_max_abs_error=float(values['centered_increment_identity_residual'].abs().max()),
            finite_chord_max_abs_error=float(values['finite_chord_closure'][valid].abs().max()) if bool(valid.any()) else None)
        summary[name] = metrics
    return {'arrays': arrays, 'summary': summary}
