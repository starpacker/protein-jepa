"""The trainer must never let one bad batch poison the weights.

Measured on 2026-10-22: `lasp_v3_velocity` went non-finite at step 8,630 and
kept running for 5,000+ more steps, ending with 64/73 kernel tensors NaN;
`lasp_v5_pos03` did the same from step 12,700.  `clip_grad_norm_` cannot catch
this -- the norm of a NaN gradient is NaN and `clamp(max=c)` of NaN is NaN -- so
the skip has to be explicit.
"""
import math

import pytest
import torch

from protein_jepa_frame.training.train_stage_b_hs import _safe_step


def _setup(lr=1e-3):
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 1)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr)
    for pg in opt.param_groups:
        pg["base_lr"] = pg["lr"]
    return lin, opt


def _snapshot(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def test_finite_step_updates_weights():
    lin, opt = _setup()
    before = _snapshot(lin)
    loss = lin(torch.randn(8, 4)).pow(2).mean()
    ok, gn = _safe_step(opt, list(lin.parameters()), loss, 1.0)
    assert ok and math.isfinite(gn)
    assert any(not torch.equal(v, before[k]) for k, v in lin.state_dict().items())


def test_nonfinite_loss_is_skipped_and_weights_untouched():
    lin, opt = _setup()
    before = _snapshot(lin)
    loss = lin(torch.randn(8, 4)).pow(2).mean() * float("nan")
    ok, gn = _safe_step(opt, list(lin.parameters()), loss, 1.0)
    assert not ok
    for k, v in lin.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed despite a non-finite loss"


def test_nonfinite_gradient_is_skipped_and_weights_untouched():
    """The actual failure mode: a NaN arrives through the gradient, not the loss."""
    lin, opt = _setup()
    before = _snapshot(lin)
    x = torch.randn(8, 4)
    loss = lin(x).pow(2).mean()
    # Poison the gradient after the loss looks perfectly finite.
    loss = loss + (x.sum() * float("nan"))
    ok, gn = _safe_step(opt, list(lin.parameters()), loss, 1.0)
    assert not ok
    for k, v in lin.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} changed despite a NaN gradient"


def test_optimizer_state_not_advanced_on_skip():
    """A skipped step must not consume AdamW moments, or recovery is biased."""
    lin, opt = _setup()
    x = torch.randn(8, 4)
    ok, _ = _safe_step(opt, list(lin.parameters()), lin(x).pow(2).mean(), 1.0)
    assert ok
    n_before = sum(v.numel() for s in opt.state.values() for v in s.values()
                   if torch.is_tensor(v))
    ok, _ = _safe_step(opt, list(lin.parameters()),
                       lin(x).pow(2).mean() * float("nan"), 1.0)
    assert not ok
    n_after = sum(v.numel() for s in opt.state.values() for v in s.values()
                  if torch.is_tensor(v))
    assert n_after == n_before, "a skipped step mutated optimizer state"


def test_recovery_after_skip():
    """After a skip the trainer must still learn from the next good batch."""
    lin, opt = _setup(lr=1e-2)
    x = torch.randn(8, 4)
    _safe_step(opt, list(lin.parameters()), lin(x).pow(2).mean() * float("nan"), 1.0)
    before = _snapshot(lin)
    ok, _ = _safe_step(opt, list(lin.parameters()), lin(x).pow(2).mean(), 1.0)
    assert ok
    assert any(not torch.equal(v, before[k]) for k, v in lin.state_dict().items())


def test_lr_groups_applied_only_on_success():
    lin, opt = _setup(lr=1e-3)
    x = torch.randn(8, 4)
    _safe_step(opt, list(lin.parameters()), lin(x).pow(2).mean() * float("nan"),
               1.0, lr_groups=0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3), "lr moved on a skipped step"
    _safe_step(opt, list(lin.parameters()), lin(x).pow(2).mean(), 1.0, lr_groups=0.5)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)
