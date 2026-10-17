"""Assert the evaluation window protocol never leaks target into context.

Acceptance criterion (EVALUATION_PLAN.md §7.3): context and every target frame
are disjoint in raw frames, and the anchor is the last context frame.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import EvalProtocol, iter_windows, _context_offsets  # noqa: E402

MANIFEST = "artifacts/mdcath_manifest_200dom/val.jsonl"


@pytest.mark.skipif(not Path(MANIFEST).exists(), reason="manifest not present")
def test_windows_are_disjoint_and_anchored():
    proto = EvalProtocol(manifest=MANIFEST, horizons_frames=(1, 2, 4, 8),
                         context_frames=8, max_proteins=2)
    n = 0
    for w in iter_windows(proto):
        ctx = set(w.context_frames.tolist())
        assert w.target_frame not in ctx, "target leaks into context"
        assert w.target_frame == w.anchor_frame + w.horizon
        assert w.context_frames[-1] == w.anchor_frame, "anchor must be last context frame"
        assert len(w.context_frames) == 8
        n += 1
        if n > 200:
            break
    assert n > 0


def test_context_offsets_end_at_zero():
    off = _context_offsets(EvalProtocol(manifest="x", context_frames=4, context_stride=2))
    assert off[-1] == 0
    assert list(off) == [-6, -4, -2, 0]
