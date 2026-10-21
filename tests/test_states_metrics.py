"""Tests: metastable-state ensemble metrics (TASK 2A).

Analytic checks: identical distributions -> zero free-energy/population error and
full coverage; missing a well-populated state -> lower coverage.
"""
from __future__ import annotations

import math

import torch

from evaluation.states_metrics import (
    free_energy, free_energy_mae, state_populations, reference_state_coverage,
    internal_distance_features, assign_to_centroids, _kmeans,
)


def test_populations_and_free_energy_analytic():
    pop = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    F = free_energy(pop, 300.0)
    # F relative to max pop (0.5) -> [0, -kT ln(0.5), -kT ln(0.5)]
    expect = -1.9872041e-3 * 300.0 * torch.log(torch.tensor([1.0, 0.5, 0.5], dtype=torch.float64))
    assert torch.allclose(F, expect, atol=1e-10)


def test_free_energy_mae_identical_is_zero():
    pop = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float64)
    out = free_energy_mae(pop, pop, 348.0)
    assert out["free_energy_mae_kcal"] < 1e-10
    assert out["population_tv"] < 1e-10
    assert out["population_jsd"] < 1e-10


def test_coverage_oracle_full():
    torch.manual_seed(0)
    L = 12
    base = torch.zeros(L, 3)
    base[:, 0] = torch.arange(L, dtype=torch.float32) * 3.8
    # Three *conformational* modes (internal distances differ): bend the chain in
    # y as a function of residue so pairwise distances change.
    ramp = torch.linspace(-1.0, 1.0, L)[:, None]
    modes = []
    for m in (-6.0, 0.0, 6.0):
        bend = (ramp * m).expand(L, 1) * torch.tensor([[0.0, 1.0, 0.0]])
        modes.append(base + bend + 0.05 * torch.randn(40, L, 3))
    ref = torch.cat(modes, 0)
    mask = torch.ones(L, dtype=torch.bool)
    out = reference_state_coverage(ref, ref, mask, k_states=3, eps_angstrom=0.5, seed=0)
    assert out["coverage"] == 1.0
    # model only samples the middle (unbent) conformation -> missing outer states
    model = base + 0.05 * torch.randn(60, L, 3)
    out2 = reference_state_coverage(model, ref, mask, k_states=3, eps_angstrom=0.5, seed=0)
    assert out2["coverage"] < 1.0


def test_kmeans_recovers_separated_clusters():
    torch.manual_seed(1)
    c = torch.tensor([[0.0, 0.0], [10.0, 10.0], [-10.0, 10.0]])
    x = torch.cat([c[i] + 0.01 * torch.randn(50, 2) for i in range(3)], 0)
    C, labels = _kmeans(x, 3, seed=0)
    # each of the 3 true clusters should map to a distinct centroid
    assert len(set(labels[:50].tolist())) == 1
    assert len(set(labels[50:100].tolist())) == 1
    assert len(set(labels[100:].tolist())) == 1
