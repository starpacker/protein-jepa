"""Trajectory-level metrics for a *walk* of future frames (order-aware).

The ensemble metrics (``evaluation.ensemble_metrics*``) score a *bag* of
conformers and are order-free: they cannot tell a smooth diffusion apart from a
shuffled set of the same conformers.  A GLE rollout is a *sequence*, so we add
order-aware metrics that (a) constrain the path shape and (b) score it as a
distribution of paths:

    per_step_rmsd      [T]  common-anchor RMSD of each future frame (no leakage)
    path_shape              internal path length / end-to-end / tortuosity
    rg_curve           [T]  radius of gyration per frame
    acf_rmsd                autocorrelation of the internal-distance vector
    autocorr_time           integrated relaxation time of that ACF
    traj_energy_score       energy score of a whole ensemble of PATHS
    walk_rmsf               ensemble RMSF over the pooled frames of the walk

All metrics take CA coordinates:
    pred : [M, T, L, 3]   M sample trajectories
    true : [T, L, 3]      the true future frames
    anchor : [L, 3]       the real last context frame (for the common gauge)
    mask : [L] bool
"""
from __future__ import annotations

import torch

from evaluation.ensemble_metrics import align_ensemble, rmsf_profile
from evaluation.geometry import (
    anchor_align_rotation, apply_rigid, _kabsch_rotation,
)


def _ok(mask: torch.Tensor) -> bool:
    return int(mask.bool().sum()) >= 3


def _align_frames_to_anchor(x: torch.Tensor, anchor: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """Kabsch-align every frame of ``x`` [..., L, 3] onto the anchor cloud.

    Removes the global rigid drift so path-length quantities measure internal
    motion.  Uses ONLY the model's own frames (no target involved).
    """
    m = mask.to(x.dtype)
    mm = m[..., None]
    denom = m.sum().clamp_min(1.0)
    refc = (anchor - (anchor * mm).sum(-2, keepdim=True) / denom) * mm
    out = torch.empty_like(x)
    import itertools
    for idx in itertools.product(*[range(s) for s in x.shape[:-2]]):
        p = (x[idx] - (x[idx] * mm).sum(-2, keepdim=True) / denom) * mm
        R = _kabsch_rotation(p, refc)
        out[idx] = p @ R
    return out


def per_step_rmsd(pred: torch.Tensor, true: torch.Tensor, anchor: torch.Tensor,
                  mask: torch.Tensor) -> dict:
    """Common-anchor RMSD of each future frame.

    For frame t we fit the *real* rigid transform anchor->true[t] (never sees a
    prediction) and apply it to every sample, exactly like
    ``geometry.common_anchor_rmsd`` but per step.  Returns the mean-over-samples
    curve ``[T]`` plus its mean (area under the curve, lower is better).
    """
    T = true.shape[0]
    curve = torch.full((T,), float("nan"), dtype=pred.dtype, device=pred.device)
    for t in range(T):
        if not _ok(mask):
            continue
        R, tr = anchor_align_rotation(anchor, true[t], mask)
        aligned = apply_rigid(pred[:, t], R, tr)              # [M, L, 3]
        err = (aligned[:, mask] - true[t, mask]).square().sum(-1)   # [M, L]
        curve[t] = err.mean(-1).sqrt().mean()                 # mean over M
    return {"per_step_rmsd": curve,
            "per_step_rmsd_mean": float(torch.nanmean(curve))}


def path_shape(pred: torch.Tensor, anchor: torch.Tensor, mask: torch.Tensor) -> dict:
    """Internal-motion path statistics of a walk ([M, T, L, 3]).

    ``path_length``  : sum_t mean_res ||x_t - x_{t-1}||  (x_0 = anchor), after
                       aligning each frame to the anchor (internal motion only).
    ``end_to_end``   : mean_res ||x_T - x_0|| in that gauge.
    ``tortuosity``   : path_length / end_to_end  (>1 = curved), guarded.
    """
    x = torch.cat([anchor[None, None].expand(pred.shape[0], 1, *anchor.shape),
                   pred], dim=1)                              # [M, T+1, L, 3]
    xa = _align_frames_to_anchor(x, anchor, mask)             # [M, T+1, L, 3]
    step = (xa[:, 1:] - xa[:, :-1]).norm(dim=-1)[..., mask]   # [M, T, Lv]
    plen = float(step.sum(1).mean())                          # mean over M of path len
    e2e = float((xa[:, -1] - xa[:, 0]).norm(dim=-1)[..., mask].mean())
    tort = plen / e2e if e2e > 1e-6 else float("nan")
    return {"path_length": plen, "end_to_end": e2e, "tortuosity": tort}


def rg_curve(pred: torch.Tensor, mask: torch.Tensor) -> dict:
    """Radius of gyration per frame ([M, T, L, 3]); rotation invariant."""
    m = mask.to(pred.dtype).view(*([1] * (pred.dim() - 2)), mask.shape[0], 1)
    denom = m.sum(dim=-2, keepdim=True).clamp_min(1.0)          # [.,1,1]
    x = pred * m
    mean = x.sum(-2, keepdim=True) / denom
    rg = ((x - mean).square().sum(-1).sum(-1) / denom[..., 0]).clamp_min(0).sqrt()
    return {"rg_curve": rg.mean(0), "rg_mean": float(rg.mean())}


def _dist_vec(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Upper-triangular nonlocal CA distance vector [T, P] (rotation invariant)."""
    L = x.shape[-2]
    i, j = torch.triu_indices(L, L, offset=3, device=x.device)
    keep = mask.bool()[i] & mask.bool()[j]
    i, j = i[keep], j[keep]
    return (x[..., i, :] - x[..., j, :]).norm(dim=-1)         # [..., T, P]


def autocorr_time(x: torch.Tensor, mask: torch.Tensor, k: int = 1) -> dict:
    """Normalized autocorrelation of the internal-distance vector.

    ``acf`` is ``rho[k] = <v_t . v_{t+k}> / <v_t . v_t>`` with v_t the
    mean-centred internal-distance vector at frame t.  NOTE: the *integrated*
    sum ``sum_k rho[k]`` is a mathematical identity (it equals ``-rho[0]`` for a
    mean-centred series) and therefore useless; we instead report the
    **lag-1 autocorrelation** ``rho[k]`` (default lag 1), a compact and
    informative walk-smoothness signature: high = the walk drifts coherently in
    internal-distance space, low/negative = it jitters frame-to-frame.
    """
    v = _dist_vec(x, mask)                                   # [T, P]
    if v.shape[0] < 2:
        return {"acf": torch.zeros(1, device=v.device),
                "acf_lag1": float("nan")}
    v = v - v.mean(0, keepdim=True)
    denom = (v * v).sum().clamp_min(1e-12)
    T = v.shape[0]
    rho = torch.stack([(v[: T - lag] * v[lag:]).sum() / denom for lag in range(T)])
    return {"acf": rho, "acf_lag1": float(rho[min(k, T - 1)])}


def traj_energy_score(pred: torch.Tensor, true: torch.Tensor,
                      mask: torch.Tensor) -> float:
    """Energy score of the *path* distribution (proper scoring rule, lower better).

    Treats each sample trajectory as a point in (T*L*3) but scores it through
    the pairwise-CA-distance marginal at each frame (scale-free of the global
    gauge), averaged over frames.  M=1 reduces to a plain prediction error.
    """
    from evaluation.geometry import physical_distance_energy_score
    scores = []
    for t in range(true.shape[0]):
        s = physical_distance_energy_score(pred[:, t], true[t], mask,
                                           return_tensor=False)
        if s == s:                                            # not NaN
            scores.append(s)
    return float(sum(scores) / len(scores)) if scores else float("nan")


def walk_rmsf(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> dict:
    """Ensemble RMSF over the POOLED frames of the walk vs the true walk."""
    from evaluation.ensemble_metrics import rmsf_profile_error
    pr = align_ensemble(pred.reshape(-1, pred.shape[-2], 3), mask)
    tr = align_ensemble(true, mask)
    err = rmsf_profile_error(pr, tr, mask)
    return {"walk_rmsf_mae": float(err["rmsf_mae"]),
            "walk_rmsf_corr": float(err["rmsf_corr"])}


def walk_summary(pred: torch.Tensor, true: torch.Tensor, anchor: torch.Tensor,
                 mask: torch.Tensor, *, acf_k: int = 1) -> dict:
    """All order-aware walk metrics for one window (scalars only)."""
    out: dict[str, float] = {}
    ps = per_step_rmsd(pred, true, anchor, mask)
    out["per_step_rmsd_mean"] = ps["per_step_rmsd_mean"]
    out.update({k: v for k, v in path_shape(pred, anchor, mask).items()})
    rgc = rg_curve(pred, mask)
    out["rg_mean"] = rgc["rg_mean"]           # scalar only (drop the [T] curve)
    out["traj_energy_score"] = traj_energy_score(pred, true, mask)
    out.update(walk_rmsf(pred, true, mask))
    # true-walk ACF for reference + correlation of the pred ACF with the true ACF
    p_acf = autocorr_time(pred.reshape(-1, pred.shape[-2], 3), mask)
    t_acf = autocorr_time(true, mask)
    out["acf_lag1"] = p_acf["acf_lag1"]                       # predicted walk's lag-1 ACF
    out["true_acf_lag1"] = t_acf["acf_lag1"]
    na = min(p_acf["acf"].numel(), t_acf["acf"].numel())
    a, b = p_acf["acf"][:na], t_acf["acf"][:na]
    if na >= 3:
        a = a - a.mean(); b = b - b.mean()
        d = a.norm() * b.norm()
        out["acf_corr"] = float((a * b).sum() / d) if float(d) > 1e-9 else float("nan")
    else:
        out["acf_corr"] = float("nan")
    return out
