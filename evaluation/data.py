"""Deterministic, fail-closed history/target windows using training readers.

Horizons are measured in raw-frame offsets, matching data.protocol.WindowProtocol.
Context stride is in observation steps. Every emitted physical horizon is unique
per anchor; multi-lag aliases never duplicate the same scoring window.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.protocol import validate_endpoint_times


@dataclass
class EvalProtocol:
    manifest: str
    horizons_frames: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)
    context_frames: int = 8
    context_stride: int = 1
    frame_stride: int = 1
    max_residues: int | None = None
    max_proteins: int | None = None
    max_trajectories_per_protein: int | None = None
    max_anchors_per_trajectory: int | None = None
    temperatures: tuple[float, ...] | None = None
    anchor_stride: int = 1
    skip_head_frames: int = 0
    seed: int = 0
    temporal_lags: tuple[int, ...] = ()
    emit_multi_lag: bool = True

    def __post_init__(self):
        if not self.horizons_frames or min(self.horizons_frames) < 1:
            raise ValueError("all horizons must be positive")
        for name in ["context_frames", "context_stride", "frame_stride", "anchor_stride"]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ["max_residues", "max_proteins", "max_trajectories_per_protein", "max_anchors_per_trajectory"]:
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive or None")
        if self.skip_head_frames < 0 or any(l < 0 for l in self.temporal_lags):
            raise ValueError("negative skip/lag is invalid")


@dataclass
class EvalWindow:
    protein_id: str
    trajectory_id: str
    temperature_k: float
    anchor_frame: int
    horizon: int
    horizon_ps: float
    context: dict
    target: dict
    context_frames: np.ndarray = field(default_factory=lambda: np.array([]))
    target_frame: int = -1
    lag: int = 0
    future_path: dict | None = None  # observed consecutive future states, diagnostic only

    def validate(self):
        if not len(self.context_frames) or self.context_frames[-1] != self.anchor_frame:
            raise ValueError("anchor must be the last history frame")
        if np.any(np.diff(self.context_frames) <= 0):
            raise ValueError("history frames must increase strictly")
        if self.target_frame <= self.anchor_frame or self.target_frame in self.context_frames:
            raise ValueError("scored target must be strictly after the history")
        if self.horizon < 1 or self.horizon_ps <= 0:
            raise ValueError("physical horizon must be positive")
        ct = self.context.get("time_ps")
        tt = self.target.get("time_ps")
        if ct is not None and tt is not None:
            validate_endpoint_times(ct, tt)
            dt = float(tt[-1]-ct[-1])
            if not np.isclose(dt, self.horizon_ps, rtol=1e-5, atol=1e-3):
                raise ValueError(f"time metadata mismatch: {dt} vs {self.horizon_ps}")
        return self


def _select_trajectories(records, proto):
    recs = list(records)
    if proto.temperatures is not None:
        recs = [r for r in recs if r.temperature_k is not None and float(r.temperature_k) in proto.temperatures]
    selected, proteins, counts = [], set(), {}
    for rec in recs:
        if rec.protein_id not in proteins:
            if proto.max_proteins is not None and len(proteins) >= proto.max_proteins:
                continue
            proteins.add(rec.protein_id)
        count = counts.get(rec.protein_id, 0)
        if proto.max_trajectories_per_protein is not None and count >= proto.max_trajectories_per_protein:
            continue
        selected.append(rec)
        counts[rec.protein_id] = count+1
    return selected


def _context_offsets(proto):
    stride = proto.context_stride * proto.frame_stride
    return np.arange(-(proto.context_frames-1)*stride, 1, stride, dtype=np.int64)


def iter_windows(proto):
    records = _select_trajectories(load_manifest(proto.manifest), proto)
    lags = proto.temporal_lags if proto.emit_multi_lag and proto.temporal_lags else (0,)
    gaps = sorted({int(h)-int(l)*proto.frame_stride for h in proto.horizons_frames for l in lags
                   if h-l*proto.frame_stride >= 1})
    if not gaps:
        raise ValueError("no valid future horizons remain")
    ctx_off = _context_offsets(proto)
    if any(g % proto.frame_stride for g in gaps):
        raise ValueError("all evaluation gaps must be divisible by frame_stride")
    max_raw_h = max(gaps)
    read_len = -int(ctx_off.min())+max_raw_h+1
    for rec in records:
        n_raw = rec.num_frames or TrajectoryClipDataset._read_num_frames(rec)
        if n_raw < read_len+proto.skip_head_frames:
            continue
        ds = TrajectoryClipDataset([rec], DataConfig(manifest_path=None, num_frames=read_len,
                                                   max_residues=proto.max_residues))
        first = proto.skip_head_frames-int(ctx_off.min())
        last = n_raw-1-max_raw_h
        for count, anchor in enumerate(range(first, last+1, proto.anchor_stride*proto.frame_stride)):
            if proto.max_anchors_per_trajectory is not None and count >= proto.max_anchors_per_trajectory:
                break
            start = anchor+int(ctx_off.min())
            frames = start+np.arange(read_len, dtype=np.int64)
            # Reader errors must be visible, never silently alter the benchmark.
            clip = ds._read_clip(rec, frames)
            offset = -int(ctx_off.min())
            ctx_idx = offset+ctx_off
            ctx = _slice_clip(clip, ctx_idx)
            for gap in gaps:
                raw_gap = gap
                tgt = _slice_clip(clip, np.array([offset+raw_gap]))
                path_idx = offset+np.arange(0, raw_gap+1, proto.frame_stride)
                w = EvalWindow(rec.protein_id, rec.trajectory_id,
                               float(rec.temperature_k) if rec.temperature_k is not None else 348.,
                               anchor, gap, raw_gap*float(rec.frame_interval_ps or 1.), ctx, tgt,
                               frames[ctx_idx], anchor+raw_gap,
                               future_path=_slice_clip(clip, path_idx))
                yield w.validate()


_FRAME_KEYS = {"coords", "atom_mask", "residue_mask", "time_ps", "forces", "msm_state_labels"}


def _slice_clip(clip, idx):
    return {k: v[torch.as_tensor(idx)] if k in _FRAME_KEYS and torch.is_tensor(v) else v
            for k, v in clip.items()}
