"""Curvature estimates must reproduce a known dense symmetric operator."""
from types import SimpleNamespace

import torch

from scripts.lasp_curvature import mobility_curvature


def test_mobility_curvature_matches_dense_extreme_eigenvalues():
    class Kernel:
        cfg = SimpleNamespace(force_clip=None, step_ps=1000.)

        def bath(self, anchor, temperature):
            return torch.tensor([[[0.2], [0.7]]], dtype=anchor.dtype), None, None

        def compute_forces(self, H, mask, temperature, create_graph):
            hessian = torch.tensor([[2., 0.3], [0.3, -4.]], dtype=H.dtype)
            return -(hessian @ H.flatten()).reshape_as(H)

    H = torch.ones(1, 2, 1, dtype=torch.float64)
    factor = torch.tensor([[[1., 0.2], [0.1, 0.8]]], dtype=H.dtype)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = mobility_curvature(Kernel(), H, H, None, mask, factor, iterations=2)
    N = torch.diag(torch.tensor([0.2, 0.7], dtype=H.dtype).sqrt()) @ factor[0]
    hessian = torch.tensor([[2., 0.3], [0.3, -4.]], dtype=H.dtype)
    eigenvalues = torch.linalg.eigvalsh(N.T @ hessian @ N)
    assert abs(result["ritz_min_per_ps"] - float(eigenvalues[0])) < 1e-12
    assert abs(result["ritz_max_per_ps"] - float(eigenvalues[-1])) < 1e-12
