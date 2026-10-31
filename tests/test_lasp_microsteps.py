"""Physical-time and regression contracts for the opt-in microstep path."""
from types import SimpleNamespace

import torch

from protein_jepa_frame.models.stage_b1b.model import StageB1b
from scripts.diagnose_lasp_microsteps import _coarsen
from scripts.lasp_microstate_vjp import microstate_vjp


class LinearKernel:
    sample_step = StageB1b.sample_step
    rollout = StageB1b.rollout
    rollout_microstep_v1 = StageB1b.rollout_microstep_v1
    _collective_velocity_step_fixed = StageB1b._collective_velocity_step_fixed

    def __init__(self, force_scale=0.3):
        self.cfg = SimpleNamespace(use_bath=True, bath_coupling="velocity", bath_dim=2, step_ps=0.2)
        self.force_scale = force_scale

    def bath(self, anchor, temperature):
        return (torch.full_like(anchor, 0.7),
                torch.full((*anchor.shape, 2), 0.1, dtype=anchor.dtype),
                torch.full((*anchor.shape[:2], 2), 0.4, dtype=anchor.dtype))

    def compute_forces(self, H, mask, temperature, create_graph):
        return -self.force_scale * H


def test_one_microstep_matches_original_path_and_gradient():
    kernel = LinearKernel()
    H = torch.randn(2, 3, 4, dtype=torch.float64).requires_grad_()
    S = torch.randn(2, 3, 2, dtype=torch.float64)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    factor = torch.eye(3, dtype=H.dtype).expand(2, -1, -1) * 1.3
    kwargs = dict(S0=S, detach=False, return_bath=True, noise_factor=factor,
                  collective_mobility=True, seq_feature=H.detach())
    old = kernel.rollout(H, mask, None, 4,
                         generator=torch.Generator().manual_seed(19), **kwargs)
    new = kernel.rollout_microstep_v1(H, mask, None, 4,
                                    generator=torch.Generator().manual_seed(19), **kwargs)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = torch.autograd.grad(sum(x.square().sum() for x in old), H)[0]
    b = torch.autograd.grad(sum(x.square().sum() for x in new), H)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_brownian_aggregation_preserves_physical_increment():
    fine = torch.randn(2, 4, 8, 3, 2, dtype=torch.float64)
    reference = (0.2 / 8) ** 0.5 * fine.sum(2)
    for count in (1, 2, 4, 8):
        coarse = _coarsen(fine, count)
        actual = (0.2 / count) ** 0.5 * coarse.sum(2)
        torch.testing.assert_close(actual, reference, rtol=1e-14, atol=1e-14)


def test_refinement_stabilizes_stiff_coupled_linear_recurrence():
    kernel = LinearKernel(force_scale=30.)
    H = torch.ones(1, 1, 1, dtype=torch.float64).requires_grad_()
    S = torch.zeros(1, 1, 2, dtype=H.dtype)
    mask = torch.ones(1, 1, dtype=torch.bool)
    gains = []
    for count in (1, 2, 4, 8):
        out, _ = kernel.rollout_microstep_v1(
            H, mask, None, 8, S0=S, detach=False, return_bath=True,
            collective_mobility=True, microsteps=count,
            brownian_H=torch.zeros(1, 8, count, 1, 1, dtype=H.dtype),
            brownian_S=torch.zeros(1, 8, count, 1, 2, dtype=H.dtype))
        gains.append(float(torch.autograd.grad(out[:, -1].sum(), H)[0].abs().max()))
    assert gains[0] > 1000
    assert max(gains[2:]) < 1
    assert gains[1] < gains[0]


def test_microstate_replay_matches_dense_joint_adjoint():
    for integrator in ("explicit", "coupled_midpoint_v1"):
        torch.manual_seed(23)
        kernel = LinearKernel()
        kernel.cfg.force_clip = None
        kernel.energy = SimpleNamespace(log_tail=torch.zeros(2, dtype=torch.float64))
        H = torch.randn(1, 2, 2, dtype=torch.float64).requires_grad_()
        S = torch.randn(1, 2, 2, dtype=H.dtype).requires_grad_()
        mask = torch.ones(1, 2, dtype=torch.bool)
        factor = torch.tensor([[[1., 0.2], [0.1, 0.8]]], dtype=H.dtype)
        anchor = H.detach()
        z_h = torch.randn(1, 2, 2, 2, 2, dtype=H.dtype)
        z_s = torch.randn_like(z_h)
        trace = []
        trajectory, _ = kernel.rollout_microstep_v1(
            H, mask, None, 2, S0=S, seq_feature=anchor, detach=False,
            return_bath=True, collective_mobility=True, microsteps=2,
            brownian_H=z_h, brownian_S=z_s, noise_factor=factor,
            integrator=integrator, microstate_trace=trace)
        endpoint = torch.randn_like(H)
        full = torch.autograd.grad(trajectory[:, -1], (H, S), endpoint,
                                   retain_graph=True)
        jacobian = torch.stack([
            torch.cat([g.flatten() for g in torch.autograd.grad(
                trajectory[:, -1].flatten()[i], (H, S), retain_graph=True)])
            for i in range(H.numel())])
        dense = jacobian.T @ endpoint.flatten()
        result = microstate_vjp(
            kernel, trace, anchor, None, mask, factor, z_h, z_s,
            endpoint, integrator, curvature_iterations=0)
        local = torch.cat((result["initial_H_covector"].flatten(),
                           result["initial_S_covector"].flatten()))
        # Midpoint replay uses the production 1e-6 CG residual tolerance.
        torch.testing.assert_close(local, dense, rtol=2e-5, atol=1e-6)
        for actual, expected in zip((result["initial_H_covector"],
                                     result["initial_S_covector"]), full):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
        assert len(result["steps"]) == 4
        assert result["forward_max_abs_difference"] == 0.
