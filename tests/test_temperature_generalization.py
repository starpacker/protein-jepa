"""TASK 3 tests: temperature conditioning override + RMSF-response metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_temperature_generalization import _build_cond, _rmsf_corr, MDCATH_TEMPS


def _paired(temp_k: float, B: int = 2, horizon_ps: float = 1e3):
    return {
        "context_coords": torch.zeros(B, 4, 5, 4, 3),
        "temperature_k": torch.full((B,), float(temp_k)),
        "horizon_time_ps": torch.full((B,), float(horizon_ps)),
    }


def test_five_mdcath_temperatures():
    assert MDCATH_TEMPS == (320.0, 348.0, 379.0, 413.0, 450.0)


def test_cond_dim_zero_is_none():
    assert _build_cond(_paired(348.0), 0, torch.device("cpu")) is None


def test_cond_correct_temperature_uses_true_value():
    p = _paired(413.0)
    dev = torch.device("cpu")
    c = _build_cond(p, 2, dev)
    # second column is (T-300)/150
    assert torch.allclose(c[:, 1], torch.full((2,), (413.0 - 300.0) / 150.0))


def test_cond_override_sets_constant_temperature():
    p = _paired(348.0)                      # true T = 348
    dev = torch.device("cpu")
    c = _build_cond(p, 2, dev, temp_override=450.0)
    assert torch.allclose(c[:, 1], torch.full((2,), (450.0 - 300.0) / 150.0))


def test_cond_drop_zeroes_temperature_slot():
    p = _paired(450.0)                      # true T = 450 (max)
    dev = torch.device("cpu")
    c = _build_cond(p, 2, dev, temp_drop=True)
    assert torch.allclose(c[:, 1], torch.zeros(2))


def test_cond_horizon_column_unchanged_by_temp_override():
    p = _paired(348.0)
    dev = torch.device("cpu")
    a = _build_cond(p, 2, dev)
    b = _build_cond(p, 2, dev, temp_override=450.0)
    assert torch.allclose(a[:, 0], b[:, 0])          # horizon column identical


def test_rmsf_corr_perfect_when_model_matches_true():
    B, n, L = 2, 8, 6
    true_rmsf = torch.rand(B, L) * 2.0
    # construct a cloud whose per-residue RMSF equals true_rmsf exactly-ish
    ca = (torch.randn(B, n, L, 3) * true_rmsf[:, None, :, None]
          - 0.0)                                     # std over n == true_rmsf
    mask = torch.ones(B, L, dtype=torch.bool)
    r = _rmsf_corr(ca, true_rmsf, mask)
    assert r.shape == (B,)
    assert float(r.min()) > 0.8                    # high (sampling noise in std)


def test_rmsf_corr_zero_for_constant_true():
    B, n, L = 1, 6, 5
    true_rmsf = torch.ones(B, L)                     # no variance -> undefined r
    ca = torch.randn(B, n, L, 3)
    r = _rmsf_corr(ca, true_rmsf, torch.ones(B, L, dtype=torch.bool))
    assert torch.isfinite(r).all()                   # guarded, not NaN
