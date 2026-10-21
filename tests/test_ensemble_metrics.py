"""Unit tests for ensemble-level metrics (T1, 2026-10-18).

Each metric is checked on an analytically determined toy case so a failure
means the metric (not the caller) is wrong.
"""
from __future__ import annotations

import math

import torch

from evaluation.ensemble_metrics import (
    rmsf_profile,
    align_ensemble,
    contact_probability_map,
    internal_distance_matrix,
    covariance_slow_modes,
    rmsf_profile_error,
    contact_probability_error,
    distance_distribution_emd,
    ensemble_summary,
)


def _toy_ensemble(n=200, L=6, amp=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(L, 3)
    base[:, 0] = torch.arange(L, dtype=torch.float32) * 3.8
    # fluctuate only along y with per-residue amplitude
    amp_vec = torch.linspace(0.0, amp, L)[:, None]
    noise = torch.randn(n, L, 1, generator=g) * amp_vec            # [N,L,1]
    x = base[None] + torch.cat([torch.zeros(n, L, 1), noise,
                                torch.zeros(n, L, 1)], dim=-1)
    return x


def test_rmsf_matches_analytic():
    # two residues with known sigma; RMSF == sigma for zero-mean gaussian
    x = _toy_ensemble(n=20000, L=6, amp=1.0, seed=1)
    r = rmsf_profile(x)
    amp_vec = torch.linspace(0.0, 1.0, 6)
    assert torch.allclose(r, amp_vec, atol=3e-2), (r, amp_vec)


def test_rmsf_mask_zeros_invalid():
    x = _toy_ensemble(n=50, L=6)
    m = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
    r = rmsf_profile(x, m)
    assert torch.allclose(r[3:], torch.zeros(3))


def test_align_ensemble_removes_rigid_motion():
    x = _toy_ensemble(n=32, L=6)
    # Per-frame TRANSLATION.  Alignment is defined w.r.t. a fixed reference, so
    # the *shape fluctuation* (hence RMSF) must be identical whether or not a
    # rigid motion was applied first.
    g = torch.Generator().manual_seed(3)
    y = x + torch.randn(x.shape[0], 1, 3, generator=g) * 5
    r_y = rmsf_profile(align_ensemble(y))
    r_x = rmsf_profile(align_ensemble(x))
    assert torch.allclose(r_y, r_x, atol=1e-2), (r_x, r_y)


def test_align_ensemble_removes_global_rotation():
    x = _toy_ensemble(n=32, L=6)
    ang = 0.7
    R = torch.tensor([[math.cos(ang), -math.sin(ang), 0],
                      [math.sin(ang), math.cos(ang), 0],
                      [0, 0, 1.]])
    y = x @ R + 3.0
    r_y = rmsf_profile(align_ensemble(y))
    r_x = rmsf_profile(align_ensemble(x))
    assert torch.allclose(r_y, r_x, atol=1e-2), (r_x, r_y)


def test_contact_probability_bounds_and_diag():
    x = _toy_ensemble(n=100, L=6, amp=0.5)
    cp = contact_probability_map(x, cutoff=8.0)
    assert cp.shape == (6, 6)
    assert torch.allclose(torch.diagonal(cp), torch.zeros(6))
    assert (cp >= 0).all() and (cp <= 1).all()


def test_contact_probability_known_pairs():
    # residues 0,1 fixed at 3.8A -> always in contact; 0,5 far -> never
    x = torch.zeros(50, 6, 3)
    x[:, :, 0] = torch.tensor([0., 3.8, 7.6, 11.4, 15.2, 30.0])
    cp = contact_probability_map(x, cutoff=8.0)
    assert cp[0, 1] == 1.0
    assert cp[0, 5] == 0.0


def test_internal_distance_matrix_matches_direct():
    x = _toy_ensemble(n=20, L=6)
    dm = internal_distance_matrix(x)
    d_direct = torch.cdist(x, x).mean(dim=0)
    assert torch.allclose(dm, d_direct, atol=1e-5)
    assert torch.allclose(torch.diagonal(dm), torch.zeros(6), atol=1e-5)


def test_covariance_slow_modes_ordering_and_scale():
    # isotropic fluctuation of amplitude s -> total variance = 3L s^2
    n, L, s = 5000, 4, 0.7
    x = torch.randn(n, L, 3) * s
    ev = covariance_slow_modes(x, k=4)
    assert ev.shape == (4,)
    assert (ev[:-1] >= ev[1:] - 1e-6).all()            # descending
    # all 3L modes carry the total variance; top-4 are a subset of it
    ev_all = covariance_slow_modes(x, k=3 * L)
    total = float(ev_all.sum())
    assert abs(total - 3 * L * s * s) / (3 * L * s * s) < 0.15, total
    assert float(ev[0]) >= float(ev_all[-1]) - 1e-6


def test_identical_ensembles_have_zero_error():
    x = _toy_ensemble(n=64, L=6, amp=0.8)
    assert rmsf_profile_error(x, x)["rmsf_mae"] == 0.0
    assert abs(rmsf_profile_error(x, x)["rmsf_corr"] - 1.0) < 1e-5
    assert contact_probability_error(x, x)["contact_prob_mae"] == 0.0
    assert abs(contact_probability_error(x, x)["contact_prob_jaccard"] - 1.0) < 1e-5
    assert distance_distribution_emd(x, x)["dist_emd"] < 1e-5


def test_rmsf_error_increases_with_mismatch():
    x = _toy_ensemble(n=200, L=6, amp=1.0, seed=5)
    y = _toy_ensemble(n=200, L=6, amp=2.0, seed=5)
    e_same = rmsf_profile_error(x, x)["rmsf_mae"]
    e_diff = rmsf_profile_error(y, x)["rmsf_mae"]
    assert e_diff > e_same


def test_distance_emd_positive_and_scales():
    x = _toy_ensemble(n=200, L=8, amp=1.0, seed=7)
    y = _toy_ensemble(n=200, L=8, amp=1.0, seed=8)
    z = _toy_ensemble(n=200, L=8, amp=4.0, seed=7)
    e_xy = distance_distribution_emd(y, x)["dist_emd"]
    e_xz = distance_distribution_emd(z, x)["dist_emd"]
    assert e_xy > 0 and e_xz > 0
    # larger amplitude mismatch -> larger EMD
    assert e_xz > e_xy


def test_ensemble_summary_keys():
    x = _toy_ensemble(n=32, L=6)
    s = ensemble_summary(x, x)
    for k in ("rmsf_mae", "rmsf_corr", "contact_prob_mae",
              "contact_prob_jaccard", "dist_emd"):
        assert k in s, k


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)


# --------------------------------------------------------------------------- #
# extended metrics (2026-10-19): ROOT MEAN W2 / VARIANCE W2 / MD-PCA W2 /
# WEAK CONTACT J / TRANSIENT CONTACTS J
# --------------------------------------------------------------------------- #
from evaluation.ensemble_metrics import rmsf_profile_error as _rpe  # noqa: E402


def rmsf_corr(pred_x, true_x, mask=None):
    return torch.tensor(_rpe(pred_x, true_x, mask)["rmsf_corr"])


from evaluation.ensemble_metrics_ext import (  # noqa: E402
    w2_1d,
    root_mean_w2,
    variance_w2,
    md_pca_w2,
    weak_contact_jaccard,
    transient_contact_jaccard,
    extended_ensemble_summary,
)


def test_w2_1d_identity_and_shift():
    a = torch.randn(500, dtype=torch.float64)
    assert float(w2_1d(a, a)) == 0.0
    b = a + 1.5
    assert abs(float(w2_1d(a, b)) - 1.5) < 1e-6, "translation shift = exact W2"


def test_root_mean_w2_oracle_zero_and_monotone():
    x = _toy_ensemble(n=300, L=6)
    mask = torch.ones(6, dtype=torch.bool)
    assert float(root_mean_w2(x, x, mask)) < 1e-9
    small = x + torch.randn_like(x) * 0.1
    big = x + torch.randn_like(x) * 1.0
    assert float(root_mean_w2(small, x, mask)) < float(root_mean_w2(big, x, mask))


def test_variance_w2_oracle_zero_and_responds_to_spread():
    x = _toy_ensemble(n=300, L=6)
    mask = torch.ones(6, dtype=torch.bool)
    assert float(variance_w2(x, x, mask)) < 1e-9
    # an ensemble with compressed fluctuations should differ
    y = x.mean(0, keepdim=True) + (x - x.mean(0, keepdim=True)) * 0.3
    assert float(variance_w2(y, x, mask)) > 1e-6


def test_md_pca_w2_oracle_zero_and_finite_grad():
    x = _toy_ensemble(n=300, L=6)
    mask = torch.ones(6, dtype=torch.bool)
    assert float(md_pca_w2(x, x, mask, k=3)) < 1e-9
    z = (x + torch.randn_like(x) * 0.4).requires_grad_(True)
    val = md_pca_w2(z, x, mask, k=3)
    val.backward()
    assert torch.isfinite(z.grad).all()


def test_weak_contact_jaccard_range_and_perfect():
    x = _toy_ensemble(n=800, L=8, amp=2.0, seed=3)
    mask = torch.ones(8, dtype=torch.bool)
    assert abs(float(weak_contact_jaccard(x, x, mask)) - 1.0) < 1e-6
    y = x + torch.randn_like(x) * 0.6
    j = float(weak_contact_jaccard(y, x, mask))
    assert 0.0 <= j <= 1.0


def test_transient_contact_jaccard_perfect_and_range():
    x = _toy_ensemble(n=800, L=8, amp=2.0, seed=4)
    mask = torch.ones(8, dtype=torch.bool)
    assert abs(float(transient_contact_jaccard(x, x, mask)) - 1.0) < 1e-6
    y = x + torch.randn_like(x) * 0.5
    j = float(transient_contact_jaccard(y, x, mask))
    assert 0.0 <= j <= 1.0


def test_extended_summary_keys_and_oracle():
    x = _toy_ensemble(n=200, L=6)
    mask = torch.ones(6, dtype=torch.bool)
    s = extended_ensemble_summary(x, x, mask)
    core = {"rmsf_mae", "rmsf_corr", "root_mean_w2", "variance_w2", "md_pca_w2",
            "weak_contact_j", "transient_contacts_j"}
    assert core <= set(s)
    for k in ("rmsf_mae", "root_mean_w2", "variance_w2", "md_pca_w2"):
        assert s[k] < 1e-9, k
    assert abs(s["rmsf_corr"] - 1.0) < 1e-6, "oracle RMSF profile corr = 1"
    assert abs(s["weak_contact_j"] - 1.0) < 1e-6
    assert abs(s["transient_contacts_j"] - 1.0) < 1e-6


def test_rmsf_corr_direction_and_oracle():
    x = _toy_ensemble(n=300, L=8, amp=1.0, seed=7)
    mask = torch.ones(8, dtype=torch.bool)
    assert abs(float(rmsf_corr(x, x, mask)) - 1.0) < 1e-6
    # a scaled (amplitude-only) perturbation keeps the profile shape => corr ~ 1
    y = x.mean(0, keepdim=True) + (x - x.mean(0, keepdim=True)) * 0.5
    assert float(rmsf_corr(y, x, mask)) > 0.99


def test_jaccard_full_exposes_degenerate_sizes():
    from evaluation.ensemble_metrics_ext import weak_contact_jaccard_full
    x = _toy_ensemble(n=600, L=8, amp=2.0, seed=9)
    mask = torch.ones(8, dtype=torch.bool)
    # a single-frame (delta) prediction => binary {0,1} map => weak set empty
    delta = x[0][None].expand(8, -1, -1)
    out = weak_contact_jaccard_full(delta, x, mask)
    assert out["n_pred"] == 0, "delta ensemble cannot express weak contacts"
    # truth has weak contacts (amp=2 gives fractional probabilities) => J == 0
    if out["n_true"] > 0:
        assert out["weak_contact_j"] == 0.0
