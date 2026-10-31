"""Focused no-update diagnostic regressions; executable without pytest."""
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.diagnose_lasp_stability import _gradient_stats, _panel


def test_norm_overflow_is_not_element_nonfiniteness():
    value = torch.full((100,), 1e19)
    stats = _gradient_stats([("energy.weight", value)], [value])
    assert stats["finite"]
    assert stats["norm32"] is None
    assert abs(stats["norm64"] / 1e20 - 1) < 1e-6


def test_panel_prefers_distinct_proteins():
    windows = [SimpleNamespace(protein_id=p, horizon=h) for p in ["a", "a", "b"] for h in [1, 64]]
    with patch("scripts.diagnose_lasp_stability.iter_windows", return_value=iter(windows)):
        panel = _panel("unused", [320], [1, 64], 96, 2, 2)
    assert len(panel) == 4
    for horizon in [1, 64]:
        assert {w.protein_id for w in panel if w.horizon == horizon} == {"a", "b"}


if __name__ == "__main__":
    test_norm_overflow_is_not_element_nonfiniteness()
    test_panel_prefers_distinct_proteins()
    print("PASS: float32 overflow discrimination and distinct-protein panel")
