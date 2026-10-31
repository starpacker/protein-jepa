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
    # Round-robin scheduling ACROSS PROTEINS.
    #
    # The default (False) walk is manifest order: every window of trajectory 1,
    # then trajectory 2, ...  Any caller that stops after N windows therefore
    # scores the FIRST FEW TRAJECTORIES ONLY.  On the 1000dom val manifest that
    # means a 200-window budget is 100% one 60-residue protein (1nhwC00) -- the
    # benchmark silently degenerates into a single-protein test and the
    # per-window bootstrap CI becomes meaningless.
    #
    # round_robin_proteins=True interleaves: window 1 of protein A, protein B,
    # protein C, ..., then window 2 of A, B, C, ...  So ANY prefix of the stream
    # is spread over as many proteins as available, which is what a benchmark
    # with a window budget needs.
    round_robin_proteins: bool = False
    # Optional cap on windows contributed by any one protein (round-robin only).
    max_windows_per_protein: int | None = None
    # Interleave (temperature, replica) alongside the protein axis.
    #
    # Per-protein round-robin alone still walks ONE trajectory at a time, so a
    # short prefix is all-T320 (the manifest lists T320_R0..R4 first) and only the
    # first few horizons.  That is a second silent single-condition benchmark.
    # With round_robin_conditions=True the bucket key becomes
    # (protein, condition index), where the condition index enumerates the
    # protein's (temperature, replica) trajectories -- so round r emits
    # protein A / T320, protein B / T320, ..., protein A / T348, ... and a
    # prefix covers proteins AND temperatures AND replicas.
    round_robin_conditions: bool = False

    def __post_init__(self):
        if self.round_robin_conditions and not self.round_robin_proteins:
            raise ValueError("round_robin_conditions requires round_robin_proteins=True")
        if not self.horizons_frames or min(self.horizons_frames) < 1:
            raise ValueError("all horizons must be positive")
        for name in ["context_frames", "context_stride", "frame_stride", "anchor_stride"]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ["max_residues", "max_proteins", "max_trajectories_per_protein",
                     "max_anchors_per_trajectory", "max_windows_per_protein"]:
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive or None")
        if not self.round_robin_proteins and self.max_windows_per_protein is not None:
            raise ValueError(
                "max_windows_per_protein requires round_robin_proteins=True "
                "(in manifest order its prefix semantics would be misleading)")
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


def _window_geometry(proto):
    """Resolve the (lags, gaps, ctx offsets, read length) shared by all records."""
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
    return gaps, ctx_off, max_raw_h, read_len


def _iter_record_windows(rec, proto, geometry):
    """Yield every scored window for one trajectory record (manifest order)."""
    for group in _iter_record_groups(rec, proto, geometry):
        yield from group


def _iter_record_groups(rec, proto, geometry):
    """Yield ONE LIST of windows per anchor: all horizons for that anchor.

    The anchor is the atomic unit of a benchmark sample -- every horizon is
    scored against the same anchor frame, and the expensively-read clip is shared
    between them.  Splitting a record's horizons across different anchors would
    both re-read the clip and make horizon coverage depend on the truncation
    point, so callers that interleave must do it at THIS granularity.
    """
    gaps, ctx_off, max_raw_h, read_len = geometry
    n_raw = rec.num_frames or TrajectoryClipDataset._read_num_frames(rec)
    if n_raw < read_len+proto.skip_head_frames:
        return
    ds = TrajectoryClipDataset([rec], DataConfig(manifest_path=None, num_frames=read_len,
                                               max_residues=proto.max_residues))
    first = proto.skip_head_frames-int(ctx_off.min())
    last = n_raw-1-max_raw_h
    offset = -int(ctx_off.min())
    ctx_idx = offset+ctx_off
    for count, anchor in enumerate(range(first, last+1, proto.anchor_stride*proto.frame_stride)):
        if proto.max_anchors_per_trajectory is not None and count >= proto.max_anchors_per_trajectory:
            break
        frames = anchor+int(ctx_off.min())+np.arange(read_len, dtype=np.int64)
        # Reader errors must be visible, never silently alter the benchmark.
        clip = ds._read_clip(rec, frames)
        ctx = _slice_clip(clip, ctx_idx)
        group = []
        for gap in gaps:
            raw_gap = gap
            tgt = _slice_clip(clip, np.array([offset+raw_gap]))
            path_idx = offset+np.arange(0, raw_gap+1, proto.frame_stride)
            w = EvalWindow(rec.protein_id, rec.trajectory_id,
                           float(rec.temperature_k) if rec.temperature_k is not None else 348.,
                           anchor, gap, raw_gap*float(rec.frame_interval_ps or 1.), ctx, tgt,
                           frames[ctx_idx], anchor+raw_gap,
                           future_path=_slice_clip(clip, path_idx))
            group.append(w.validate())
        yield group


def _round_robin_windows(records, proto, geometry):
    """Interleave windows so every prefix covers many proteins AND conditions.

    Manifest order spends the whole budget on the first trajectory family, which
    on the 1000dom val split is a single 60-residue domain.  Bucketing by protein
    alone is not enough either: each protein has 25 trajectories
    (5 temperatures x 5 replicas), so a 200-window budget still put 100 windows
    on protein #1 before touching protein #2.

    This walks the (protein, condition) grid on a LATIN-SQUARE diagonal: in round
    ``r``, protein ``i`` is sampled at condition ``(r + i) % n_cond[i]``.  So

      * consecutive windows already change protein, and
      * the condition (i.e. temperature/replica) varies *within* a round instead
        of being constant across it.

    With 4 horizons per anchor group, a 200-window budget therefore yields 50
    anchor groups spread over 50 proteins and all five temperatures, instead of
    one protein at one temperature.  Uneven manifests stay correct because the
    offset is taken modulo each protein's own condition count.  A bucket that
    still has anchor groups left is revisited on later rounds, so nothing is
    dropped when trajectories are long.

    Within one trajectory the order is preserved (anchors ascending, then
    horizons).  ``proto.seed`` shuffles the PROTEIN order for multi-seed
    repeats.  Only the current trajectory of each visited protein is buffered,
    so a trajectory that is never reached is never read from disk.
    """
    # (protein, condition_index) -> list of trajectory records
    buckets: dict = {}
    order: list = []                     # proteins, in manifest order
    seen_cond: dict[str, int] = {}
    for rec in records:
        pid = rec.protein_id
        if pid not in seen_cond:
            seen_cond[pid] = 0
            order.append(pid)
        c = seen_cond[pid]
        seen_cond[pid] = c + 1
        key = (pid, c) if proto.round_robin_conditions else (pid, 0)
        buckets.setdefault(key, []).append(rec)

    if proto.seed:
        rng = np.random.default_rng(proto.seed)
        rng.shuffle(order)

    max_cond: dict[str, int] = {}
    for (pid, c) in buckets:
        max_cond[pid] = max(max_cond.get(pid, 0), c + 1)

    per_protein: dict[str, int] = {}
    cursors = {k: 0 for k in buckets}    # which trajectory inside this bucket
    pending: dict = {}                   # buffered ANCHOR GROUPS (list of windows)

    def _live(key) -> bool:
        return cursors[key] < len(buckets[key])

    for rnd in range(10 ** 9):
        # Termination is "no bucket has a trajectory left", NOT "nothing was
        # emitted": a round fully suppressed by max_windows_per_protein must not
        # end the sweep while unread trajectories remain.
        if not any(_live(k) for k in buckets):
            return
        for i, pid in enumerate(order):
            if proto.round_robin_conditions:
                key = (pid, (rnd + i) % max_cond[pid])
            else:
                key = (pid, 0)
            if key not in buckets or not _live(key):
                continue
            # Pull the next non-empty anchor group, advancing through exhausted
            # trajectories of this bucket as needed.
            while key not in pending and _live(key):
                gen = _iter_record_groups(buckets[key][cursors[key]], proto, geometry)
                pending[key] = (gen, None)
                grp = next(gen, None)
                if grp is None:
                    cursors[key] += 1
                    pending.pop(key, None)
                    continue
                pending[key] = (gen, grp)
            if key not in pending:
                continue
            gen, grp = pending[key]
            emit = []
            for w in grp:
                if proto.max_windows_per_protein is not None:
                    if per_protein.get(w.protein_id, 0) >= proto.max_windows_per_protein:
                        continue
                per_protein[w.protein_id] = per_protein.get(w.protein_id, 0) + 1
                emit.append(w)
            # Advance to the next anchor group once this one is drained.
            nxt = next(gen, None)
            if nxt is None:
                cursors[key] += 1
                pending.pop(key, None)
            else:
                pending[key] = (gen, nxt)
            yield from emit


def iter_windows(proto):
    records = _select_trajectories(load_manifest(proto.manifest), proto)
    geometry = _window_geometry(proto)
    if proto.round_robin_proteins:
        yield from _round_robin_windows(records, proto, geometry)
        return
    for rec in records:
        yield from _iter_record_windows(rec, proto, geometry)


_FRAME_KEYS = {"coords", "atom_mask", "residue_mask", "time_ps", "forces", "msm_state_labels"}


def _slice_clip(clip, idx):
    return {k: v[torch.as_tensor(idx)] if k in _FRAME_KEYS and torch.is_tensor(v) else v
            for k, v in clip.items()}
