"""Coverage guarantees for the round-robin window scheduler.

The legacy manifest-order walk spent an entire 200-window benchmark budget on
ONE protein (1nhwC00, 60 residues), so every reported metric was a single-domain
anecdote.  These tests pin the properties that make a window prefix usable as a
dataset-level benchmark: many proteins, several temperatures, every horizon.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.data import EvalProtocol, _round_robin_windows, _window_geometry


def _record(pid, cond, frames=200):
    """Minimal stand-in for a TrajectoryRecord; scheduling never reads frames."""
    class _Rec:
        pass
    r = _Rec()
    r.protein_id, r.trajectory_id = pid, f"{pid}_T{cond}"
    r.temperature_k = (320., 348., 379., 413., 450.)[cond % 5]
    r.num_frames, r.frame_interval_ps = frames, 1000.
    r.format, r.metadata, r.path = "synthetic", {}, ""
    return r


def test_prefix_covers_many_proteins_and_all_temperatures(monkeypatch):
    import evaluation.data as data

    def fake_groups(rec, proto, geometry):
        # 3 anchors x 2 horizons per trajectory, no disk access
        for a in range(3):
            group = []
            for h in (1, 4):
                group.append(type("W", (), {
                    "protein_id": rec.protein_id,
                    "trajectory_id": rec.trajectory_id,
                    "temperature_k": rec.temperature_k,
                    "horizon": h,
                })())
            yield group

    monkeypatch.setattr(data, "_iter_record_groups", fake_groups)
    proto = EvalProtocol("x", horizons_frames=(1, 4), context_frames=8,
                         round_robin_proteins=True, round_robin_conditions=True)
    records = [_record(f"p{i:02d}", c) for i in range(10) for c in range(5)]
    geometry = _window_geometry(proto)

    windows = list(_round_robin_windows(records, proto, geometry))
    assert len(windows) == 10 * 5 * 3 * 2

    # first 20 windows (10 proteins x 2 horizons) already cover every protein
    head = windows[:20]
    assert len({w.protein_id for w in head}) == 10
    # the temperature axis is interleaved WITHIN the first round, not deferred
    head2 = windows[:40]
    assert len({w.temperature_k for w in head2}) >= 3


def test_every_window_is_emitted_exactly_once(monkeypatch):
    import evaluation.data as data

    def fake_groups(rec, proto, geometry):
        for a in range(2):
            yield [type("W", (), {"protein_id": rec.protein_id,
                                  "trajectory_id": rec.trajectory_id,
                                  "temperature_k": rec.temperature_k,
                                  "anchor": a, "horizon": 1})()]

    monkeypatch.setattr(data, "_iter_record_groups", fake_groups)
    proto = EvalProtocol("x", horizons_frames=(1,), context_frames=8,
                         round_robin_proteins=True, round_robin_conditions=True)
    records = [_record(f"p{i}", c) for i in range(4) for c in range(3)]
    windows = list(_round_robin_windows(records, proto, _window_geometry(proto)))
    keys = [(w.protein_id, w.trajectory_id, w.anchor) for w in windows]
    # each (trajectory, anchor) is visited exactly once: nothing dropped, nothing
    # double-scored, which is what makes a capped run a fair subset.
    assert len(keys) == len(set(keys)) == 4 * 3 * 2


def test_max_windows_per_protein_caps_without_ending_early(monkeypatch):
    import evaluation.data as data

    def fake_groups(rec, proto, geometry):
        for a in range(5):
            yield [type("W", (), {"protein_id": rec.protein_id,
                                  "trajectory_id": rec.trajectory_id,
                                  "temperature_k": rec.temperature_k,
                                  "horizon": 1})()]

    monkeypatch.setattr(data, "_iter_record_groups", fake_groups)
    proto = EvalProtocol("x", horizons_frames=(1,), context_frames=8,
                         round_robin_proteins=True, round_robin_conditions=True,
                         max_windows_per_protein=3)
    records = [_record(f"p{i}", c) for i in range(6) for c in range(2)]
    windows = list(_round_robin_windows(records, proto, _window_geometry(proto)))
    counts = {}
    for w in windows:
        counts[w.protein_id] = counts.get(w.protein_id, 0) + 1
    # every protein is present (no early termination) and none exceeds the cap
    assert len(counts) == 6
    assert max(counts.values()) <= 3


def test_round_robin_conditions_requires_round_robin_proteins():
    with pytest.raises(ValueError, match="round_robin_conditions"):
        EvalProtocol("x", round_robin_conditions=True)
    with pytest.raises(ValueError, match="max_windows_per_protein"):
        EvalProtocol("x", max_windows_per_protein=5)


def test_seed_shuffles_protein_order_deterministically():
    a = EvalProtocol("x", round_robin_proteins=True, seed=0)
    b = EvalProtocol("x", round_robin_proteins=True, seed=0)
    c = EvalProtocol("x", round_robin_proteins=True, seed=7)
    assert a.seed == b.seed == 0 and c.seed == 7
