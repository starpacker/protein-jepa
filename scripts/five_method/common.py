"""Shared plumbing for the five-method comparison.

Everything here is deliberately *method-agnostic*:

* one ``EvalProtocol`` window iterator for all methods (same anchors, same
  true MD futures, same residue masks);
* one deterministic seeding scheme keyed by the *window identity* (never by
  loop position), with ``fork_rng`` so methods cannot perturb each other;
* one metric implementation (``evaluation.*``), never a per-method copy.

The five methods are only allowed to differ in `how they produce the CA cloud`.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows

METHODS = ("model1", "model2", "lasp", "mdgen", "bioemu2")
PRETTY = {
    "model1": "Model 1 (Stage-A ensemble head)",
    "model2": "Model 2 (from-none GLE v4g)",
    "lasp": "Lasp (Stage A + GLE, unfrozen decoder)",
    "mdgen": "MDGen (zero-shot ATLAS)",
    "bioemu2": "bioemu2 (zero-shot, bioemu2-preview)",
}

ENSEMBLE_METRICS = ["rmsf_corr", "rmsf_mae", "root_mean_w2", "variance_w2",
                    "md_pca_w2", "weak_contact_j", "transient_contacts_j"]
POSITION_METRICS = ["kabsch_ca_rmsd", "aligned_ca_rmsd"]
WALK_METRICS = ["per_step_rmsd_mean", "path_length", "end_to_end", "rg_mean",
                "traj_energy_score", "walk_rmsf_mae", "walk_rmsf_corr",
                "acf_lag1", "acf_corr", "tortuosity"]

# True == higher is better
HIGHER_BETTER = {
    "rmsf_corr": True, "rmsf_mae": False, "root_mean_w2": False,
    "variance_w2": False, "md_pca_w2": False, "weak_contact_j": True,
    "transient_contacts_j": True,
    "kabsch_ca_rmsd": False, "aligned_ca_rmsd": False,
    "per_step_rmsd_mean": False, "path_length": False, "end_to_end": False,
    "rg_mean": False, "traj_energy_score": False, "walk_rmsf_mae": False,
    "walk_rmsf_corr": True, "acf_lag1": True, "acf_corr": True,
    "tortuosity": False,
}


# --------------------------------------------------------------------------- #
# deterministic per-window seeding
# --------------------------------------------------------------------------- #
def window_seed(window, run: int = 0, salt: str = "") -> int:
    """Stable seed derived from the window's *identity*.

    Using identity (protein|trajectory|anchor|horizon) rather than loop position
    means re-ordering the manifest, changing ``--max-windows`` or running a
    different method cannot silently change what noise a method saw, and every
    method gets a reproducible independent stream.
    """
    if isinstance(window, dict):
        pid, tid = window["protein_id"], window["trajectory_id"]
        anchor, hor = window["anchor_frame"], window["horizon"]
        temp = window["temperature_k"]
    else:
        pid, tid = window.protein_id, window.trajectory_id
        anchor, hor = window.anchor_frame, window.horizon
        temp = window.temperature_k
    key = (f"{pid}|{tid}|{anchor}|{hor}|{int(round(float(temp)))}|{salt}")
    h = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    return int((h + 1_000_003 * int(run)) % (2 ** 63 - 1))


def set_linalg_backend(who: str) -> None:
    """Pin torch's preferred linalg backend for ``who``.

    The two zero-shot baselines have *incompatible* requirements on this machine:
    MDGen's ``rot_to_quat`` does a batched ``eigh`` that crashes on cuSOLVER
    (``CUSOLVER_STATUS_INTERNAL_ERROR`` for the 4x4 quaternion matrices) but
    works on MAGMA, while bioemu2's rigid alignment calls
    ``torch.linalg.svd(..., driver="gesvd")``, which MAGMA rejects outright.
    Our own metric code runs on either.  So the backend is a property of the
    method being executed, not a global setting.
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.preferred_linalg_library(
            "magma" if who == "mdgen" else "cusolver")
    except Exception:
        pass


def fork_seed(seed: int, device: torch.device | str | None = None):
    """Context manager: seed global torch RNG inside a forked RNG state."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    ctx = torch.random.fork_rng(devices=devices)
    return _SeededCtx(ctx, seed)


class _SeededCtx:
    def __init__(self, ctx, seed):
        self.ctx, self.seed = ctx, int(seed)

    def __enter__(self):
        self.ctx.__enter__()
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed % (2 ** 32 - 1))
        return self

    def __exit__(self, *exc):
        return self.ctx.__exit__(*exc)


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def make_protocol(args, *, horizons_frames: Iterable[int], manifest: str,
                  temperature: float | None = None,
                  max_windows: int | None = None) -> EvalProtocol:
    return EvalProtocol(
        manifest=manifest,
        horizons_frames=tuple(int(h) for h in horizons_frames),
        context_frames=int(args.history_frames),
        context_stride=1, frame_stride=1,
        max_residues=int(args.max_residues),
        max_proteins=getattr(args, "max_proteins", None),
        temperatures=(float(temperature),) if temperature is not None else None,
        max_anchors_per_trajectory=getattr(args, "max_anchors_per_trajectory", None),
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False,
        seed=int(getattr(args, "seed", 0)),
    )


def plan_windows(proto: EvalProtocol, max_windows: int | None = None) -> list[dict]:
    """Materialise the *window plan* once, so every method scores the same set.

    Returns lightweight records (index + identity + the tensors kept on CPU).
    Holding the plan means an MDGen/bioemu2 failure cannot shift the window set
    of the other methods.
    """
    out: list[dict] = []
    for i, w in enumerate(iter_windows(proto)):
        if max_windows is not None and len(out) >= max_windows:
            break
        rmask = w.context["residue_mask"].all(dim=0).bool()
        if int(rmask.sum()) < 4:
            continue
        fp = getattr(w, "future_path", None)
        if fp is None:
            continue
        coords = fp["coords"][:, :, 1, :][:, rmask]           # [T,L,3] true CA walk
        if coords.shape[0] < 2:
            continue
        ctx = {k: (v.detach().clone() if torch.is_tensor(v) else v)
               for k, v in w.context.items()}
        out.append({
            "index": len(out),
            "protein_id": w.protein_id,
            "trajectory_id": w.trajectory_id,
            "temperature_k": float(w.temperature_k),
            "anchor_frame": int(w.anchor_frame),
            "horizon": int(w.horizon),
            "horizon_ns": float(w.horizon) * (
                float(w.context["time_ps"][-1] - w.context["time_ps"][-2]) / 1000.0)
            if w.context.get("time_ps") is not None and w.context["time_ps"].numel() > 1
            else float(w.horizon),
            "context": ctx,
            "residue_mask": rmask,
            "true_ca_path": coords,                            # [T,L,3] (masked residues)
            "n_residues": int(rmask.sum()),
            "window": w,          # the original EvalWindow (io helpers need it)
        })
    return out


def ctx_batched(win: dict, device) -> dict:
    """Move one window's context to device with a leading batch dim of 1."""
    ctx = win["context"]
    moved = {}
    for k, v in ctx.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device).unsqueeze(0)
        else:
            moved[k] = v
    return moved


def true_walk(win: dict, device) -> torch.Tensor:
    return win["true_ca_path"].to(device)


# --------------------------------------------------------------------------- #
# metric application
# --------------------------------------------------------------------------- #
def score_ensemble(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor | None,
                   *, cutoff=8.0, pca_k=3) -> dict:
    from evaluation.ensemble_metrics_ext import extended_ensemble_summary
    return extended_ensemble_summary(pred.float(), true.float(), mask,
                                     cutoff=cutoff, pca_k=pca_k)


def score_position(samples: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> dict:
    """Common-anchor + aligned CA RMSD, averaged over the sample cloud."""
    from evaluation.geometry import common_anchor_rmsd, aligned_rmsd
    ca = [common_anchor_rmsd(samples[i], anchor, target, mask)
          for i in range(samples.shape[0])]
    al = [aligned_rmsd(samples[i], target, mask) for i in range(samples.shape[0])]
    return {"kabsch_ca_rmsd": float(np.nanmean(ca)),
            "aligned_ca_rmsd": float(np.nanmean(al))}


def score_walk(pred_walk: torch.Tensor, true: torch.Tensor, anchor: torch.Tensor,
               mask: torch.Tensor) -> dict:
    """Order-aware (trajectory) metrics for a [M,T,L,3] predicted walk."""
    from evaluation.trajectory_metrics import walk_summary
    out = walk_summary(pred_walk.float(), true.float(), anchor.float(), mask)
    return {k: float(out[k]) for k in WALK_METRICS if k in out}


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def dump(path, obj):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
