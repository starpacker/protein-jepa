"""Tests for ensemble-level losses (T4)."""
from __future__ import annotations

import torch

from protein_jepa_frame.training.ensemble_losses import (
    EnsembleLossWeights,
    compute_ensemble_loss,
)
from protein_jepa_frame.training.ensemble_targets import build_ensemble_targets


def _true_and_pred(B=2, n=8, L=7, T=5, seed=0):
    """Build a true window and a model ensemble close to it."""
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(L, 3)
    base[:, 0] = torch.arange(L, dtype=torch.float32) * 4.0

    def window(nn_):
        noise = torch.randn(B, nn_, L, 1, generator=g) * 0.6
        ca = base[None, None] + torch.cat(
            [torch.zeros(B, nn_, L, 1), noise, torch.zeros(B, nn_, L, 1)], dim=-1)
        coords = torch.zeros(B, nn_, L, 4, 3)
        coords[:, :, :, 1, :] = ca
        return coords

    true_coords = window(T)
    mask = torch.ones(B, T, L, dtype=torch.bool)
    targets = build_ensemble_targets(true_coords, mask)
    pred_coords = window(n)                      # model ensemble window
    pred_ca = pred_coords[:, :, :, 1, :]         # [B,n,L,3]
    true_hist = targets.dist_hist_true()
    pred_hist = targets.dist_hist_from_ca(pred_ca)
    mask2 = targets.mask
    return targets, pred_ca, pred_hist, true_hist, mask2


def test_loss_shapes_and_finiteness():
    t, pc, ph, th, m = _true_and_pred()
    w = EnsembleLossWeights()
    c = compute_ensemble_loss(pc, t.rmsf, t.contact, ph, th, m,
                              (t.dist_hi - t.dist_lo) / (t.dist_bins - 1), w)
    assert torch.isfinite(c.total) and c.total >= 0
    d = c.as_loggable_dict()
    for k in ("ens/total", "ens/rmsf", "ens/contact", "ens/dist"):
        assert k in d


def test_identical_ensemble_gives_small_loss():
    # use the true window as the model ensemble -> losses ~ 0
    t, _, _, th, m = _true_and_pred()
    true_ca = t.ca_window
    ph = t.dist_hist_from_ca(true_ca)
    w = EnsembleLossWeights()
    c = compute_ensemble_loss(true_ca, t.rmsf, t.contact, ph, th, m,
                              (t.dist_hi - t.dist_lo) / (t.dist_bins - 1), w)
    assert float(c.rmsf) < 1e-4
    assert float(c.dist) < 1e-4


def test_loss_is_differentiable():
    t, pc, ph, th, m = _true_and_pred()
    x = pc.clone().requires_grad_(True)
    ph = t.dist_hist_from_ca(x)
    w = EnsembleLossWeights()
    c = compute_ensemble_loss(x, t.rmsf, t.contact, ph, th, m,
                              (t.dist_hi - t.dist_lo) / (t.dist_bins - 1), w)
    c.total.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_loss_decreases_when_matching():
    t, pc, ph, th, m = _true_and_pred(seed=3)
    w = EnsembleLossWeights()
    binw = (t.dist_hi - t.dist_lo) / (t.dist_bins - 1)
    # mismatch: flipped/averaged ensemble
    bad = torch.zeros_like(pc)
    bad_hist = t.dist_hist_from_ca(bad)
    c_bad = compute_ensemble_loss(bad, t.rmsf, t.contact, bad_hist, th, m, binw, w)
    c_good = compute_ensemble_loss(t.ca_window, t.rmsf, t.contact,
                                   t.dist_hist_from_ca(t.ca_window), th, m, binw, w)
    assert float(c_good.total) < float(c_bad.total)


def test_weight_validation_rejects_negative():
    w = EnsembleLossWeights(rmsf_weight=-1.0)
    try:
        w.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_contact_loss_uses_cutoff():
    t, pc, ph, th, m = _true_and_pred()
    w = EnsembleLossWeights(rmsf_weight=0.0, dist_weight=0.0)
    binw = (t.dist_hi - t.dist_lo) / (t.dist_bins - 1)
    c1 = compute_ensemble_loss(pc, t.rmsf, t.contact, ph, th, m, binw, w, cutoff=8.0)
    c2 = compute_ensemble_loss(pc, t.rmsf, t.contact, ph, th, m, binw, w, cutoff=20.0)
    # different cutoffs -> different contact loss
    assert abs(float(c1.contact) - float(c2.contact)) > 1e-6


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)


# ---------------------------------------------------------------------------
# D2 (2026-10-19): atom-position guide term
# ---------------------------------------------------------------------------

def _pos_pair(B=2, L=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    true_ca = torch.randn(B, L, 3, generator=g)
    true_ca[:, :, 0] = torch.arange(L).float()[None] * 3.0
    mask = torch.ones(B, L)
    return true_ca, mask


def test_position_loss_zero_when_exact():
    from protein_jepa_frame.training.ensemble_losses import _position_loss
    true_ca, mask = _pos_pair()
    # exact match (up to a global rigid motion) -> ~0
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pred = true_ca @ R.T + torch.tensor([5., -3., 2.])
    val = float(_position_loss(pred, true_ca, mask))
    assert val < 1e-6, val


def test_position_loss_increases_with_perturbation():
    from protein_jepa_frame.training.ensemble_losses import _position_loss
    true_ca, mask = _pos_pair()
    small = true_ca + 0.1 * torch.randn_like(true_ca)
    large = true_ca + 2.0 * torch.randn_like(true_ca)
    v_small = float(_position_loss(small, true_ca, mask))
    v_large = float(_position_loss(large, true_ca, mask))
    assert v_large > v_small


def test_position_loss_is_differentiable():
    from protein_jepa_frame.training.ensemble_losses import _position_loss
    true_ca, mask = _pos_pair()
    pred = (true_ca + 0.5 * torch.randn_like(true_ca)).requires_grad_(True)
    _position_loss(pred, true_ca, mask).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_compute_loss_requires_pos_inputs_when_weighted():
    t, pc, ph, th, m = _true_and_pred()
    w = EnsembleLossWeights(pos_weight=1.0)
    binw = (t.dist_hi - t.dist_lo) / (t.dist_bins - 1)
    try:
        compute_ensemble_loss(pc, t.rmsf, t.contact, ph, th, m, binw, w)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when pos_weight>0 without coords")


def test_compute_loss_pos_changes_total():
    t, pc, ph, th, m = _true_and_pred()
    true_ca = t.ca_window[:, 0]                       # [B,L,3] a true future frame
    pred_ca = pc[:, 0]                                # [B,L,3] a model frame
    binw = (t.dist_hi - t.dist_lo) / (t.dist_bins - 1)
    base = compute_ensemble_loss(pc, t.rmsf, t.contact, ph, th, m, binw,
                                 EnsembleLossWeights())
    withpos = compute_ensemble_loss(pc, t.rmsf, t.contact, ph, th, m, binw,
                                    EnsembleLossWeights(pos_weight=3.0),
                                    true_ca_pos=true_ca, pred_ca_pos=pred_ca)
    assert withpos.pos is not None
    assert float(withpos.total) > float(base.total)
    assert "ens/pos" in withpos.as_loggable_dict()
