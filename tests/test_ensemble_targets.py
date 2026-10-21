"""Tests for ensemble-target construction from future windows (T2)."""
from __future__ import annotations

import torch

from protein_jepa_frame.training.ensemble_targets import (
    extract_ca_window,
    build_ensemble_targets,
    persistence_ensemble,
    oracle_ensemble,
    pair_indices,
)


def _window(B=2, T=5, L=7, seed=0):
    """Synthetic future window [B, T, L, 4, 3] with CA fluctuated in y."""
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(L, 3)
    base[:, 0] = torch.arange(L, dtype=torch.float32) * 4.0
    noise = torch.randn(B, T, L, 1, generator=g) * 0.5
    ca = base[None, None] + torch.cat(
        [torch.zeros(B, T, L, 1), noise, torch.zeros(B, T, L, 1)], dim=-1)
    coords = torch.zeros(B, T, L, 4, 3)
    coords[:, :, :, 1, :] = ca          # CA at atom index 1
    mask = torch.ones(B, T, L, dtype=torch.bool)
    return coords, mask


def test_extract_ca_window_shapes():
    coords, mask = _window(B=3, T=5, L=7)
    ca, m = extract_ca_window(coords, mask)
    assert ca.shape == (3, 5, 7, 3)
    assert m.shape == (3, 7)
    assert torch.allclose(ca, coords[:, :, :, 1, :])


def test_build_targets_shapes_and_finiteness():
    coords, mask = _window(B=2, T=5, L=7)
    t = build_ensemble_targets(coords, mask)
    assert t.rmsf.shape == (2, 7)
    assert t.contact.shape == (2, 7, 7)
    assert torch.isfinite(t.rmsf).all()
    assert (t.rmsf >= 0).all()
    assert (t.contact >= 0).all() and (t.contact <= 1).all()
    h = t.dist_hist_true()
    assert h.shape[0] == 2 and h.shape[2] == t.dist_bins
    # histograms normalized per pair
    assert torch.allclose(h.sum(-1), torch.ones_like(h.sum(-1)), atol=1e-5)


def test_targets_are_rigid_invariant():
    coords, mask = _window(B=1, T=5, L=7)
    t1 = build_ensemble_targets(coords, mask)
    # apply a global rotation+translation to the whole window
    ang = 0.5
    R = torch.tensor([[torch.cos(torch.tensor(ang)), -torch.sin(torch.tensor(ang)), 0],
                      [torch.sin(torch.tensor(ang)), torch.cos(torch.tensor(ang)), 0],
                      [0, 0, 1.]])
    c2 = coords.clone()
    c2[..., 1, :] = coords[..., 1, :] @ R + 10.0
    t2 = build_ensemble_targets(c2, mask)
    assert torch.allclose(t1.rmsf, t2.rmsf, atol=1e-3)
    assert torch.allclose(t1.contact, t2.contact, atol=1e-3)


def test_persistence_ensemble_is_constant():
    coords, mask = _window(B=2, T=5, L=7)
    p = persistence_ensemble(coords, n=8, residue_mask=mask)
    assert p.shape == (2, 8, 7, 3)
    # all replicas identical
    assert torch.allclose(p[:, 0], p[:, -1])
    # RMSF of a constant ensemble is (near) zero
    from evaluation.ensemble_metrics import rmsf_profile, align_ensemble
    r = rmsf_profile(align_ensemble(p[0], mask[0, 0]))
    assert float(r.max()) < 1e-5


def test_oracle_ensemble_equals_window():
    coords, mask = _window(B=2, T=5, L=7)
    o = oracle_ensemble(coords, mask)
    assert torch.allclose(o, coords[:, :, :, 1, :])


def test_pair_indices_deterministic_and_small():
    mask = torch.ones(4, 200, dtype=torch.bool)
    i1, j1 = pair_indices(200, mask[0], max_pairs=128)
    i2, j2 = pair_indices(200, mask[1], max_pairs=128)
    assert torch.equal(i1, i2) and torch.equal(j1, j2)   # mask-independent
    assert len(i1) == 128
    assert (i1 < j1).all()


def test_single_frame_window_rejected():
    coords, mask = _window(B=1, T=1, L=5)
    try:
        build_ensemble_targets(coords, mask)
    except ValueError as e:
        assert "future frames" in str(e)
    else:
        raise AssertionError("expected ValueError for T<2")


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
