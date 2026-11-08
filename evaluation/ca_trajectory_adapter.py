"""Ordered CA trajectories from the existing endpoint sampler, for evaluation.

Initialization, full-covariance bath sampling and every transition remain owned
by ``CADynamics.sample``. One scoped wrapper retains its already computed H/S
path; it never re-encodes a prediction or restarts the bath. Ordinary no_grad is
required instead of inference_mode because the energy force uses autograd.

The model must be exclusively owned by the calling evaluation worker. Concurrent
or recursive adapter calls sharing a kernel are rejected, as are foreign-thread
calls intercepted by the scoped rollout wrapper. Direct access to the same
model outside this adapter must not overlap an adapter call.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import weakref

import torch

from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames


_ACTIVE = weakref.WeakKeyDictionary()
_ACTIVE_LOCK = threading.Lock()
_MISSING = object()


@dataclass(frozen=True)
class CATrajectory:
    """Paths with axes [batch, member, time including zero, residue, feature].

    CA at time zero is the observed anchor, not a decoder reconstruction.
    ``latent`` and ``bath`` retain H0 and the actual sampled S0. All subsequent
    CA frames use the same observed H0, backbone rotation and anchor; the final
    CA frame is the exact endpoint returned by the original sampler. Padding
    retains the sampler/decoder convention and must be masked when scoring.
    """

    ca: torch.Tensor
    latent: torch.Tensor
    bath: torch.Tensor | None
    step_ps: float


@torch.no_grad()
def sample_ca_trajectory(
    model, observed, mask, aa, ri, ci, temperature, steps, n_samples,
    generator=None, *, decode_chunk_size=64, context_dt_ps=None,
) -> CATrajectory:
    """Evaluate one continuous path per member from observed context only.

    ``observed`` is [B,K,L,4,3], ``mask`` is the observed [B,L] validity mask.
    Sampling uses exactly the existing sample API and RNG order. Additional
    decoding is bounded by ``decode_chunk_size`` and consumes no random draws
    for the deterministic, eval-mode component bundle. No parameter flags,
    training modes or parameter gradients are changed.

    The model treats adjacent history frames as ``kernel.cfg.step_ps`` apart.
    Supply ``context_dt_ps`` to assert that physical data spacing explicitly;
    this adapter does not infer spacing from coordinates or resample the path.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('use no_grad, not inference_mode: energy forces require autograd')
    if any(module.training for module in model.modules()):
        raise ValueError('trajectory evaluation requires model.eval() on all modules')
    if getattr(model, 'prediction_kind', 'endpoint') != 'endpoint':
        raise ValueError('ordered trajectories require endpoint prediction_kind')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (steps, n_samples, decode_chunk_size)):
        raise ValueError('steps, n_samples and decode_chunk_size must be positive integers')
    if observed.ndim != 5 or observed.shape[-2:] != (4, 3):
        raise ValueError('observed must have shape [B,K,L,4,3]')
    b, k, length = observed.shape[:3]
    if b < 1 or k < 1 or mask.shape != (b, length) or mask.dtype != torch.bool:
        raise ValueError('observed history and boolean [B,L] mask must be nonempty and compatible')
    if model.history_conditioning and k < 4:
        raise ValueError('history-conditioned trajectories require four observed frames')
    if any(value.shape != (b, length) for value in (aa, ri, ci)):
        raise ValueError('sequence metadata must have shape [B,L]')
    step_ps = float(model.kernel.cfg.step_ps)
    if not math.isfinite(step_ps) or step_ps <= 0:
        raise ValueError('kernel step_ps must be finite and positive')
    if context_dt_ps is not None and not math.isclose(
            float(context_dt_ps), step_ps, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('observed context spacing must equal kernel step_ps')

    kernel = model.kernel
    owner = threading.get_ident()
    with _ACTIVE_LOCK:
        if kernel in _ACTIVE:
            raise RuntimeError('recursive or concurrent trajectory calls sharing a kernel are forbidden')
        _ACTIVE[kernel] = owner

    local_rollout = kernel.__dict__.get('rollout', _MISSING)
    original_rollout = kernel.rollout
    captured = {}

    def retain_rollout(*args, **kwargs):
        if threading.get_ident() != owner or captured:
            raise RuntimeError('trajectory evaluation requires exactly one exclusive rollout')
        captured['entered'] = True
        h_path, s_path = original_rollout(*args, **{**kwargs, 'return_bath': True})
        captured.update(latent=h_path, bath=s_path)
        return h_path

    try:
        kernel.rollout = retain_rollout
        endpoint_ca, endpoint_h = model.sample(
            observed, mask, aa, ri, ci, temperature, steps, n_samples, generator)
        if 'latent' not in captured:
            raise RuntimeError('sample did not execute the expected rollout')
        h_path, s_path = captured['latent'], captured['bath']
        q = b * n_samples
        if h_path.shape[:3] != (q, steps + 1, length):
            raise RuntimeError('rollout returned an incompatible latent path')
        if s_path is not None and s_path.shape[:3] != (q, steps + 1, length):
            raise RuntimeError('rollout returned an incompatible bath path')
        if not torch.equal(h_path[:, -1].reshape_as(endpoint_h), endpoint_h):
            raise RuntimeError('sample endpoint does not match the captured path terminal state')

        repeat = lambda value: value[:, None].expand(
            b, n_samples, *value.shape[1:]).reshape(q, *value.shape[1:])
        rotation, anchor = _extract_backbone_frames(observed[:, -1])
        rotation, anchor = repeat(rotation), repeat(anchor)
        aa, ri, ci, rm = map(repeat, (aa, ri, ci, mask))
        ca_path = endpoint_ca.new_empty(q, steps + 1, length, 3)
        ca_path[:, 0] = anchor
        # Keep the original decoding result, including its exact numerical
        # batch shape. Chunking intermediate frames cannot perturb the endpoint.
        ca_path[:, -1] = endpoint_ca.reshape(q, length, 3)
        for offset in range(0, q * (steps - 1), decode_chunk_size):
            index = torch.arange(offset, min(offset + decode_chunk_size, q * (steps - 1)),
                                 device=h_path.device)
            member, time = index // (steps - 1), index % (steps - 1) + 1
            ca_path[member, time] = model.bundle.decode_ca(
                h_path[member, time], h_path[member, 0], aa[member], ri[member], ci[member],
                rotation[member], anchor[member], rm[member])
        return CATrajectory(
            ca=ca_path.reshape(b, n_samples, steps + 1, length, 3).detach(),
            latent=h_path.reshape(b, n_samples, steps + 1, length, -1).detach(),
            bath=None if s_path is None else s_path.reshape(b, n_samples, steps + 1, length, -1).detach(),
            step_ps=step_ps,
        )
    finally:
        if local_rollout is _MISSING:
            delattr(kernel, 'rollout')
        else:
            kernel.rollout = local_rollout
        with _ACTIVE_LOCK:
            del _ACTIVE[kernel]
