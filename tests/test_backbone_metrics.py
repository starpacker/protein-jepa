"""Tests: backbone (N, CA, C, O) metrics + OpenFold Cα lifter (TASK 1, 2026-10-21).

Each metric is checked on an analytically determined toy case (failure => the
metric, not the caller, is wrong).  Covers:

* backbone RMSD: self -> 0, rigid transform -> 0, perturbation -> > 0;
* per-atom-type RMSD consistency;
* geometry validity of an IDEAL backbone (all bonds/angles valid, ω trans);
* the lifter's invariants: Cα exact, ideal bond lengths from OpenFold tables.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from evaluation.backbone_metrics import (
    backbone_geometry_summary,
    backbone_rmsd,
    backbone_per_atom_rmsd,
    IDEAL_BOND,
)
from protein_jepa_frame.models.stage_c.openfold_ca_lifter import lift_ca_to_backbone


def _ideal_helix(L=60):
    th = 2 * math.pi / 3.6
    r, t = 2.3, 1.5
    ca = np.array([[r * math.cos(i * th), r * math.sin(i * th), i * t] for i in range(L)],
                  dtype=np.float64)
    return torch.tensor(ca)


def _rot(q):
    w, x, y, z = q
    return torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def test_backbone_rmsd_self_is_zero():
    bb = torch.randn(2, 12, 4, 3)
    mask = torch.ones(2, 12, dtype=torch.bool)
    assert backbone_rmsd(bb, bb, mask) < 1e-6
    for v in backbone_per_atom_rmsd(bb, bb, mask).values():
        assert v < 1e-6


def test_backbone_rmsd_rigid_transform_is_zero():
    bb = torch.randn(1, 15, 4, 3)
    mask = torch.ones(1, 15, dtype=torch.bool)
    q = torch.tensor([0.3, 0.5, -0.2, 0.7]); q = q / q.norm()
    R = _rot(q)
    moved = bb @ R.T + torch.tensor([7.0, -2.0, 4.0])
    assert backbone_rmsd(moved, bb, mask) < 1e-5


def test_backbone_rmsd_perturbation_positive():
    bb = torch.randn(1, 15, 4, 3)
    mask = torch.ones(1, 15, dtype=torch.bool)
    noisy = bb + 0.2 * torch.randn_like(bb)
    assert backbone_rmsd(noisy, bb, mask) > 0.05


def test_lifter_ca_is_exact_and_bonds_ideal():
    ca = _ideal_helix(60)
    bb, _, _ = lift_ca_to_backbone(ca[None])
    # Cα preserved exactly
    assert (bb[0, :, 1] - ca).abs().max().item() < 1e-5
    # bond lengths match OpenFold's own ideal values
    n_ca = (bb[0, :, 0] - bb[0, :, 1]).norm(dim=-1).mean().item()
    ca_c = (bb[0, :, 2] - bb[0, :, 1]).norm(dim=-1).mean().item()
    c_o = (bb[0, :, 3] - bb[0, :, 2]).norm(dim=-1).mean().item()
    assert abs(n_ca - IDEAL_BOND["N_CA"]) < 0.05
    assert abs(ca_c - IDEAL_BOND["CA_C"]) < 0.02
    assert abs(c_o - IDEAL_BOND["C_O"]) < 0.02


def test_geometry_summary_ideal_backbone_all_valid():
    ca = _ideal_helix(80)
    bb, _, _ = lift_ca_to_backbone(ca[None])
    mask = torch.ones(1, 80, dtype=torch.bool)
    g = backbone_geometry_summary(bb[0], mask[0])
    assert g["bond_N_CA_valid_frac"] == 1.0
    assert g["bond_CA_C_valid_frac"] == 1.0
    assert g["bond_C_O_valid_frac"] == 1.0
    assert g["angle_N_CA_C_valid_frac"] == 1.0
    assert g["ra_allowed_frac"] == 1.0
    assert g["omega_trans_frac"] > 0.9


def test_geometry_summary_scrambled_backbone_invalid():
    ca = _ideal_helix(40)
    bb, _, _ = lift_ca_to_backbone(ca[None])
    mask = torch.ones(1, 40, dtype=torch.bool)
    scrambled = bb.clone()
    scrambled[0, :, 0] += 3.0 * torch.randn(40, 3)   # break N positions
    g = backbone_geometry_summary(scrambled[0], mask[0])
    assert g["bond_N_CA_valid_frac"] < 0.5
