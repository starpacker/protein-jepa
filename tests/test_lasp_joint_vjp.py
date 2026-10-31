from types import SimpleNamespace

import torch

from scripts.lasp_joint_vjp import joint_path_vjp


class LinearKernel:
    cfg = SimpleNamespace(bath_coupling="velocity", step_ps=0.2)

    def bath(self, anchor, temperature):
        return torch.full_like(anchor, 0.7), torch.full((*anchor.shape, 1), 0.3, dtype=anchor.dtype), torch.full_like(anchor, 0.4)

    def compute_forces(self, H, mask, temperature, create_graph):
        return -1.2 * H


def test_joint_vjp_matches_dense_coupled_jacobian():
    kernel = LinearKernel()
    H = torch.ones(1, 1, 1, dtype=torch.float64)
    path = H[:, None].expand(-1, 5, -1, -1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    # K=4 exercises both spatial factors in the frozen-anchor mobility.
    factor = torch.full_like(H, 2.)
    result = joint_path_vjp(kernel, path, path, H, None, mask, factor, H)
    mobility = 0.7 * 4
    dt, c, gamma, curvature = 0.2, 0.3, 0.4, 1.2
    J = torch.tensor([
        [1 - dt * mobility * curvature, -dt * mobility * c],
        [-dt * c * mobility * curvature, 1 - dt * gamma - dt * c * mobility * c],
    ], dtype=torch.float64)
    expected = torch.tensor([1., 0.], dtype=torch.float64)
    for row in reversed(result["steps"]):
        expected = J.T @ expected
        assert abs(row["input_covector_norm64"] - float(expected.norm())) < 1e-12
        assert row["finite"]
