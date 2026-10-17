"""Evaluation window protocol.

Turns a manifest into a deterministic stream of evaluation windows:

    for each (trajectory, anchor position):
        context = K frames ending at the anchor
        for each horizon tau in horizons:
            target  = the frame tau raw-frames after the anchor

The context and every target share ONE clip read (so the reader/format logic is
identical to training), and the context/target windows are disjoint by
construction: the target index is `anchor + tau` with `tau >= 1`, and the context
occupies `[anchor-K+1 .. anchor]`.  No target frame is ever inside the context.

This module has NO model knowledge and NO metric knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.manifest import load_manifest, TrajectoryRecord
from protein_jepa_frame.data.dataset import TrajectoryClipDataset


@dataclass
class EvalProtocol:
    manifest: str
    horizons_frames: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)
    context_frames: int = 8                # K
    context_stride: int = 1                # uniform stride within context
    frame_stride: int = 1                  # sampling stride of raw trajectory
    max_proteins: int | None = None        # cap trajectories for smoke runs
    temperatures: tuple[float, ...] | None = None   # subset of T (None = all)
    anchor_stride: int = 1                 # sub-sample anchors along the trajectory
    skip_head_frames: int = 0              # skip equilibration (raw frames)
    seed: int = 0


@dataclass
class EvalWindow:
    protein_id: str
    trajectory_id: str
    temperature_k: float
    anchor_frame: int                       # raw frame index of the anchor
    horizon: int                            # in raw frames == ns (mdCATH)
    horizon_ps: float
    context: dict                           # collated-ready clip dict (K frames)
    target: dict                            # clip dict (1 frame)
    context_frames: np.ndarray = field(default_factory=lambda: np.array([]))
    target_frame: int = -1


def _select_trajectories(records, proto: EvalProtocol):
    recs = list(records)
    if proto.temperatures is not None:
        keep = {float(t) for t in proto.temperatures}
        recs = [r for r in recs if r.temperature_k is not None and float(r.temperature_k) in keep]
    if proto.max_proteins is not None:
        # deterministic: keep first N distinct trajectories in manifest order
        recs = recs[: proto.max_proteins]
    return recs


def _context_offsets(proto: EvalProtocol) -> np.ndarray:
    """Raw-frame offsets of the context relative to the anchor (last = 0)."""
    K = proto.context_frames
    s = proto.context_stride
    return np.arange(-(K - 1) * s, 1, s, dtype=np.int64)


def _num_raw_frames(record: TrajectoryRecord, stride: int) -> int:
    n = record.num_frames or 0
    return max(0, n // stride) if n else 0


def iter_windows(proto: EvalProtocol):
    """Yield EvalWindow objects deterministically.

    Uses TrajectoryClipDataset's readers under the hood by reading a single clip
    that spans the whole needed range, so mdCATH/atlas/npz formats are handled by
    the same code path as training.
    """
    records = _select_trajectories(load_manifest(proto.manifest), proto)
    max_h = max(proto.horizons_frames)
    ctx_off = _context_offsets(proto)
    span_start_off = int(ctx_off.min())                     # negative
    read_len = -span_start_off + max_h + 1                  # raw frames to read

    for rec in records:
        n_raw = rec.num_frames or 0
        if n_raw <= 0:
            continue
        # Build a clip dataset over this single record, but read variable windows
        # directly through the record reader to keep full control.
        ds = TrajectoryClipDataset([rec], DataConfig(manifest_path=None, num_frames=read_len))
        # anchor positions: raw frame index of the LAST context frame
        first_anchor = proto.skip_head_frames + (-span_start_off)
        last_anchor = n_raw - 1 - max_h
        for anchor in range(first_anchor, last_anchor + 1, proto.anchor_stride):
            start = anchor + span_start_off
            frames = start + np.arange(read_len, dtype=np.int64)
            try:
                clip = ds._read_clip(rec, frames)
            except Exception:
                continue
            clip_off = -span_start_off     # index of anchor inside the clip
            ctx_idx = clip_off + ctx_off   # [K] positions in clip
            ctx = _slice_clip(clip, ctx_idx)
            ctx_frames = frames[ctx_idx]
            for tau in proto.horizons_frames:
                tgt_idx = np.array([clip_off + tau], dtype=np.int64)
                tgt = _slice_clip(clip, tgt_idx)
                yield EvalWindow(
                    protein_id=rec.protein_id,
                    trajectory_id=rec.trajectory_id,
                    temperature_k=float(rec.temperature_k) if rec.temperature_k is not None else float("nan"),
                    anchor_frame=int(anchor),
                    horizon=int(tau),
                    horizon_ps=float(tau) * float(getattr(rec, "frame_interval_ps", 1000.0) or 1000.0),
                    context=ctx,
                    target=tgt,
                    context_frames=ctx_frames,
                    target_frame=int(frames[clip_off + tau]),
                )


_FRAME_KEYS = {"coords", "atom_mask", "residue_mask", "time_ps", "forces", "msm_state_labels"}


def _slice_clip(clip: dict, idx: np.ndarray) -> dict:
    out = {}
    for k, v in clip.items():
        if k in _FRAME_KEYS and torch.is_tensor(v):
            out[k] = v[torch.as_tensor(idx)]
        else:
            out[k] = v
    return out
