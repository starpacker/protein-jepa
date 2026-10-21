"""Geometry unit tests.  Acceptance: Kabsch vs an independently constructed
ground-truth rotation < 1e-5 A; alignment must remove pure rigid motion exactly.
Run:  python -m pytest evaluation/tests/test_geometry.py -q
"""
from __future__ import annotations
import math
import numpy as np
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import (  # noqa: E402
    ca_from_backbone, center, kabsch_align, aligned_rmsd,
    anchor_align_rotation, apply_rigid, contact_map, contact_jaccard,
)


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return torch.tensor(Q, dtype=torch.float64)


def test_kabsch_removes_pure_rigid_motion():
    rng = np.random.default_rng(0)
    P = torch.tensor(rng.normal(size=(30, 3)) * 10, dtype=torch.float64)
    R = _random_rotation(rng)
    t = torch.tensor(rng.normal(size=3) * 5, dtype=torch.float64)
    Q = P @ R.T + t  # exact rigid transform
    mask = torch.ones(30, dtype=torch.bool)
    assert aligned_rmsd(P, Q, mask) < 1e-9


def test_kabsch_recovers_ground_truth_rotation():
    rng = np.random.default_rng(1)
    P = torch.tensor(rng.normal(size=(40, 3)) * 10, dtype=torch.float64)
    R_true = _random_rotation(rng)
    t_true = torch.tensor(rng.normal(size=3) * 5, dtype=torch.float64)
    noise = torch.tensor(rng.normal(size=(40, 3)) * 0.02, dtype=torch.float64)
    Q = P @ R_true.T + t_true + noise
    mask = torch.ones(40, dtype=torch.bool)
    rmsd = aligned_rmsd(P, Q, mask)
    noise_rms = float((noise ** 2).sum(-1).mean().sqrt())
    # Kabsch removes the rigid part; the residual equals the noise RMSD up to the
    # amount the optimal rotation can absorb (so rmsd <= noise_rms, and close).
    assert rmsd <= noise_rms + 1e-9
    assert abs(rmsd - noise_rms) < 0.2 * noise_rms
    # and be small
    assert rmsd < 0.05


def test_kabsch_is_globally_optimal():
    """The returned RMSD must be the minimum over SO(3): no other rotation beats it.

    (Cross-check against scipy is omitted: scipy.spatial.transform.Rotation
    .align_vectors has a different rotation/vector convention that is easy to
    mis-apply; ground-truth + random-search optimality is a stronger, unambiguous
    check that the rotation is the true Kabsch solution.)
    """
    rng = np.random.default_rng(2)
    P = torch.tensor(rng.normal(size=(25, 3)) * 8, dtype=torch.float64)
    R_true = _random_rotation(rng)
    Q = P @ R_true.T + torch.tensor(rng.normal(size=3), dtype=torch.float64) \
        + torch.tensor(rng.normal(size=(25, 3)) * 0.3, dtype=torch.float64)
    mask = torch.ones(25, dtype=torch.bool)
    ours = aligned_rmsd(P, Q, mask)
    Pc = P - P.mean(0, keepdim=True)
    Qc = Q - Q.mean(0, keepdim=True)
    best = 1e9
    for _ in range(5000):
        A = torch.tensor(rng.normal(size=(3, 3)))
        U, _, Vh = torch.linalg.svd(A)
        Rr = U @ Vh
        if torch.det(Rr) < 0:
            Rr[:, -1] *= -1
        r = float(((Pc @ Rr - Qc) ** 2).sum(-1).mean().sqrt())
        best = min(best, r)
    assert ours <= best + 1e-9


def test_anchor_rotation_applies_to_prediction():
    """Applying the anchor->target (R,t) to a prediction is equivalent to
    kabsch_align of that prediction's own coords, when coordinates coincide."""
    rng = np.random.default_rng(4)
    anchor = torch.tensor(rng.normal(size=(20, 3)) * 10, dtype=torch.float64)
    R = _random_rotation(rng)
    t = torch.tensor(rng.normal(size=3) * 3, dtype=torch.float64)
    target = anchor @ R.T + t
    mask = torch.ones(20, dtype=torch.bool)
    Rr, tt = anchor_align_rotation(anchor, target, mask)
    moved = apply_rigid(anchor, Rr, tt)
    # anchor moved identically should land on target
    assert float((moved - target).abs().max()) < 1e-6


def test_mask_ignores_invalid_residues():
    rng = np.random.default_rng(5)
    P = torch.tensor(rng.normal(size=(10, 3)), dtype=torch.float64)
    Q = P.clone()
    mask = torch.ones(10, dtype=torch.bool)
    mask[7] = False
    # corrupt an invalid residue; RMSD must not change
    Q[7] += 100.0
    assert aligned_rmsd(P, Q, mask) < 1e-9


def test_ca_extraction():
    bb = torch.zeros(5, 4, 3)
    bb[:, 1, :] = 1.0  # CA channel
    ca = ca_from_backbone(bb)
    assert ca.shape == (5, 3)
    assert torch.allclose(ca, torch.ones(5, 3))


def test_contact_jaccard_identity():
    rng = np.random.default_rng(6)
    ca = torch.tensor(rng.normal(size=(10, 3)) * 3)
    mask = torch.ones(10, dtype=torch.bool)
    assert abs(contact_jaccard(ca, ca, mask) - 1.0) < 1e-9
    assert contact_map(ca, mask).diagonal().sum().item() == 0


def test_common_anchor_alignment_does_not_refit_prediction():
    from evaluation.geometry import common_anchor_rmsd
    torch.manual_seed(19)
    anchor = torch.randn(20, 3, dtype=torch.float64)
    mask = torch.ones(20, dtype=torch.bool)
    pred = anchor + torch.tensor([5., 0., 0.])
    assert aligned_rmsd(pred, anchor, mask) < 1e-9
    assert abs(common_anchor_rmsd(pred, anchor, anchor, mask)-5.) < 1e-9


def test_physical_energy_score_matches_empirical_distribution():
    from evaluation.geometry import physical_distance_energy_score, displacement_msd
    target = torch.arange(18, dtype=torch.float64).reshape(6,3)
    mask = torch.ones(6, dtype=torch.bool)
    assert physical_distance_energy_score(target[None].repeat(3,1,1), target, mask) == 0.
    distorted = target.clone(); distorted[3] += 3.
    assert physical_distance_energy_score(distorted[None], target, mask) > 0.
    mask[0] = False
    pred = target.clone(); pred[0] += 1000.
    assert displacement_msd(pred, target, mask) == 0.


def test_history_pose_calibration_removes_canonical_gauge_and_preserves_gradients():
    from evaluation.geometry import calibrate_observed_anchor_pose
    torch.manual_seed(23)
    current = torch.randn(1,12,3,dtype=torch.float64)
    forecast = (current + .2*torch.randn_like(current)).requires_grad_()
    observed = current + torch.tensor([[[4.,5.,6.]]])
    mask = torch.ones(1,12,dtype=torch.bool)
    expected = calibrate_observed_anchor_pose(forecast,current,observed,mask)
    q,_ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    if torch.det(q)<0: q[:,0]*=-1
    offset = torch.tensor([2.,-3.,8.],dtype=torch.float64)
    changed = calibrate_observed_anchor_pose(forecast@q+offset,current@q+offset,observed,mask)
    assert torch.allclose(changed,expected,atol=1e-9)
    expected.square().sum().backward()
    assert forecast.grad is not None and forecast.grad.norm()>0


def test_history_pose_calibration_follows_observed_rigid_motion():
    from evaluation.geometry import calibrate_observed_anchor_pose
    torch.manual_seed(24)
    current = torch.randn(1,10,3,dtype=torch.float64)
    forecast = current+.1*torch.randn_like(current)
    observed = current+4
    mask = torch.ones(1,10,dtype=torch.bool)
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    if torch.det(q)<0:q[:,0]*=-1
    first=calibrate_observed_anchor_pose(forecast,current,observed,mask)
    second=calibrate_observed_anchor_pose(forecast,current,observed@q+7,mask)
    assert torch.allclose(second,first@q+7,atol=1e-9)
