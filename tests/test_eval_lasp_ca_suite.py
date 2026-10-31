"""Tests: T3 conditioning ablations in the CA evaluation suite.

The mechanism test for "is temperature actually used?" compares a correct
temperature against a deliberately wrong one.  That comparison is only
meaningful if the two temperatures differ, so the suite must never emit a
"wrong" arm that silently equals the observed temperature.
"""
from __future__ import annotations

from scripts.eval_lasp_ca_suite import (MDCATH_TEMPS, temperature_tensor,
                                        window_seed)


def test_correct_mode_uses_observed_temperature():
    out = temperature_tensor('correct', 348.0, 320.0, 'cpu')
    assert float(out[0]) == 348.0


def test_none_mode_omits_temperature():
    assert temperature_tensor('none', 348.0, 320.0, 'cpu') is None


def test_wrong_mode_uses_the_requested_wrong_temperature():
    out = temperature_tensor('wrong', 450.0, 320.0, 'cpu')
    assert float(out[0]) == 320.0


def test_wrong_mode_never_collapses_onto_the_observed_temperature():
    # Observed T == default wrong T is exactly the degenerate cell: at T=320 K the
    # "wrong" arm would otherwise re-run the "correct" arm.
    out = temperature_tensor('wrong', 320.0, 320.0, 'cpu')
    assert float(out[0]) != 320.0
    assert float(out[0]) in MDCATH_TEMPS
    # furthest mdCATH temperature from 320 K is 450 K
    assert float(out[0]) == max(MDCATH_TEMPS)


def test_wrong_mode_fallback_is_the_furthest_temperature():
    assert float(temperature_tensor('wrong', 450.0, 450.0, 'cpu')[0]) == 320.0
    assert float(temperature_tensor('wrong', 379.0, 379.0, 'cpu')[0]) == 450.0


def test_window_seed_is_independent_of_the_t3_arm():
    """correct/wrong/none must share noise: only the temperature may differ."""
    args = ('P1', 'P1_T320_R0', 7, 16.0)
    # The signature takes no mode at all -- arms cannot accidentally differ.
    s = window_seed(0, *args)
    assert s == window_seed(0, *args)
    assert 0 <= s < 2 ** 31 - 1


def test_window_seed_tracks_window_identity_not_loop_position():
    a = window_seed(0, 'P1', 'P1_T320_R0', 7, 16.0)
    b = window_seed(0, 'P1', 'P1_T320_R0', 8, 16.0)
    c = window_seed(0, 'P1', 'P1_T320_R0', 7, 64.0)
    d = window_seed(0, 'P2', 'P2_T320_R0', 7, 16.0)
    assert len({a, b, c, d}) == 4
    # a different suite seed changes every window seed
    assert window_seed(1, 'P1', 'P1_T320_R0', 7, 16.0) != a


def test_none_mode_is_the_reference_temperature_not_absent():
    """Document the T3 'none' semantics that the suite actually implements.

    The temperature feature is log(T / T_ref) and `None` maps to zeros, so the
    "no temperature" arm is bit-identical to feeding T = T_ref (348 K). This is
    NOT "temperature withheld"; it is a fixed-reference-temperature control. The
    test pins the equivalence so a table cannot quietly describe it as absence.
    """
    import torch
    from protein_jepa_frame.models.stage_b1b import StageB1bConfig
    from protein_jepa_frame.models.stage_b1b.model import HSpaceEnergy

    cfg = StageB1bConfig()
    energy = HSpaceEnergy(cfg)
    B = 3
    absent = energy._temp_feature(None, B, 'cpu', torch.float32)
    at_ref = energy._temp_feature(
        torch.full((B,), float(cfg.reference_temperature_k)), B, 'cpu', torch.float32)
    assert torch.equal(absent, at_ref)
    assert torch.count_nonzero(absent) == 0
    # A non-reference temperature is a genuinely different input.
    elsewhere = energy._temp_feature(torch.full((B,), 320.0), B, 'cpu', torch.float32)
    assert not torch.equal(elsewhere, absent)
