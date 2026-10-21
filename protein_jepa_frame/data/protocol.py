"""Shared physical-time and causal-window contracts for all stages.

Endpoint labels are always strictly after context.  Autonomous dynamics use a
continuous clip, sharing exactly the last history frame with sequence[0].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import torch

from protein_jepa_frame.data.manifest import load_manifest


@dataclass(frozen=True)
class WindowProtocol:
    history_frames: int = 8
    frame_stride: int = 1
    horizons_frames: tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96)
    temporal_lags: tuple[int, ...] = (0, 4, 6, 7)
    history_mode: str = 'uniform'
    time_unit: str = 'ps'

    def __post_init__(self):
        if self.history_frames < 1 or self.frame_stride < 1:
            raise ValueError('history_frames and frame_stride must be positive')
        if self.history_mode != 'uniform' or self.time_unit != 'ps':
            raise ValueError('new protocol requires uniform history and physical time in ps')
        if not self.horizons_frames or any(h < 1 for h in self.horizons_frames):
            raise ValueError('horizons must be positive raw-frame offsets')
        if any(l < 0 or l >= self.history_frames for l in self.temporal_lags):
            raise ValueError('temporal_lags must index the future clip')
        if any(h % self.frame_stride for h in self.horizons_frames):
            raise ValueError('horizons must be divisible by frame_stride')
        if any(h - l * self.frame_stride <= 0
               for h in self.horizons_frames for l in self.temporal_lags):
            raise ValueError('each selected future endpoint must lie strictly after the anchor')

    def to_dict(self):
        return asdict(self)

    @property
    def gaps_frames(self):
        lags = self.temporal_lags or (0,)
        return tuple(sorted({h-l*self.frame_stride for h in self.horizons_frames for l in lags}))


def protocol_from_args(args):
    return WindowProtocol(
        history_frames=args.history_frames,
        frame_stride=getattr(args, 'frame_stride', 1),
        horizons_frames=tuple(getattr(args, 'paired_horizons_frames', (8,16,24,32,48,64,96))),
        temporal_lags=tuple(getattr(args, 'temporal_lags', (0,))),
        history_mode=getattr(args, 'paired_history_mode', 'uniform'),
    )


def apply_protocol(config, protocol):
    """Set the paired-window fields; callers retain dataset/batch options."""
    config.num_frames = protocol.history_frames
    config.frame_stride = protocol.frame_stride
    config.paired_history_mode = protocol.history_mode
    config.paired_horizons_frames = protocol.horizons_frames
    config.paired_nonoverlap_only = False
    return config


def validate_manifest_split(train_manifest, val_manifest, *, heldout_temperature=None):
    """Audit a protein-holdout split before fitting any module/statistics."""
    train = load_manifest(train_manifest)
    val = load_manifest(val_manifest)
    train_proteins, val_proteins = {r.protein_id for r in train}, {r.protein_id for r in val}
    overlap = train_proteins & val_proteins
    if overlap:
        raise ValueError(f'train/validation protein overlap: {sorted(overlap)[:10]}')
    train_domains = {r.domain_id for r in train if r.domain_id}
    val_domains = {r.domain_id for r in val if r.domain_id}
    domain_overlap = train_domains & val_domains
    if domain_overlap:
        raise ValueError(f'train/validation domain overlap: {sorted(domain_overlap)[:10]}')
    # Renaming a manifest record must not turn the same physical trajectory
    # into an unseen protein. HDF5 mdCATH files contain all domain replicas.
    # Materialized NPZ partitions retain their physical HDF5 origin. Renaming
    # or copying such a view must not create a fictitious protein holdout.
    def physical_source(r):
        return str(Path((r.metadata or {}).get('source_path', r.path)).resolve())
    train_sources = {physical_source(r) for r in train}
    val_sources = {physical_source(r) for r in val}
    if train_sources & val_sources:
        raise ValueError('train/validation source-file overlap under different protein IDs')
    if heldout_temperature is not None and any(r.temperature_k == heldout_temperature for r in train):
        raise ValueError('held-out temperature occurs in the training manifest')
    return {
        'train_proteins':len(train_proteins), 'val_proteins':len(val_proteins),
        'protein_overlap':0, 'domain_overlap':0, 'source_file_overlap':0,
        'train_manifest_sha256':hashlib.sha256(Path(train_manifest).read_bytes()).hexdigest(),
        'val_manifest_sha256':hashlib.sha256(Path(val_manifest).read_bytes()).hexdigest(),
        'train_temperatures':sorted({r.temperature_k for r in train if r.temperature_k is not None}),
        'val_temperatures':sorted({r.temperature_k for r in val if r.temperature_k is not None}),
    }


def validate_endpoint_times(context_time_ps, target_time_ps):
    context = torch.as_tensor(context_time_ps)
    target = torch.as_tensor(target_time_ps, device=context.device)
    if not torch.isfinite(context).all() or not torch.isfinite(target).all():
        raise ValueError('physical timestamps must be finite')
    if context.shape[-1] < 1:
        raise ValueError('empty context')
    if context.shape[-1] > 1 and not (context.diff(dim=-1) > 0).all():
        raise ValueError('history timestamps must be strictly increasing')
    if target.ndim == context.ndim - 1:
        target = target.unsqueeze(-1)
    if not (target > context[..., -1:]).all():
        raise ValueError('future targets must be strictly after the history anchor (no leakage)')


def split_contiguous_window(batch, history_frames, horizon_steps, step_ps=None):
    """Split collate_clips output into history and supervised [anchor,...,future].

    Returns flat context_*/future_* namespaces compatible with the paired
    collator.  future_* contains N+1 frames and starts at the real anchor;
    future_*[:,1:] is supervision only and must never enter the producer.
    """
    K, N = int(history_frames), int(horizon_steps)
    if K < 1 or N < 1:
        raise ValueError('history_frames and horizon_steps must be positive')
    coords = batch['coords']
    if coords.ndim != 5 or coords.shape[1] != K+N:
        raise ValueError(f'continuous clip must contain exactly K+N={K+N} frames')
    times = batch['time_ps']
    if times.shape != coords.shape[:2] or not torch.isfinite(times).all():
        raise ValueError('invalid physical time shape or nonfinite values')
    dt = times.diff(dim=1)
    if not (dt > 0).all():
        raise ValueError('continuous timestamps must be strictly increasing')
    if step_ps is not None:
        if not math.isfinite(step_ps) or step_ps <= 0:
            raise ValueError('step_ps must be finite and positive')
        if not torch.allclose(dt, torch.full_like(dt, step_ps), atol=1e-3, rtol=1e-5):
            raise ValueError('physical observation dt does not match the dynamics step_ps')
    else:
        if not torch.allclose(dt, dt[:, :1].expand_as(dt), atol=1e-3, rtol=1e-5):
            raise ValueError('autonomous training requires uniform physical time intervals')
    frame_keys = {'coords', 'residue_mask', 'atom_mask', 'time_ps', 'dt_ps', 'forces', 'msm_state_labels'}
    out = {}
    for key, value in batch.items():
        if key in frame_keys and torch.is_tensor(value):
            out['context_'+key] = value[:, :K]
            out['future_'+key] = value[:, K-1:]
        else:
            out['context_'+key] = value
            out['future_'+key] = value
            out[key] = value
    history_times = out['context_time_ps']
    future_times = out['future_time_ps']
    validate_endpoint_times(history_times, future_times[:, 1:])
    if not torch.equal(history_times[:, -1], future_times[:, 0]):
        raise ValueError('history[-1] must be the same anchor as supervised sequence[0]')
    # The first history token has no preceding observation; use the known
    # sampling interval rather than feeding an artificial log(0) time feature.
    hist_dt = history_times.diff(dim=1, prepend=history_times[:, :1])
    hist_dt[:, 0] = dt[:, 0]
    out['context_dt_ps'] = hist_dt
    out['future_dt_ps'] = torch.cat((dt[:, K-1:K], dt[:, K-1:]), dim=1)
    out['valid_mask'] = batch['residue_mask'].all(dim=1)
    return out
