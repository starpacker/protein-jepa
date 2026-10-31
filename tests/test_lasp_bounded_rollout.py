"""Full parameter-backward contract through bounded energy and midpoint bath."""
import torch
from protein_jepa_frame.models.lasp_bounded_energy import BoundedRidgeEnergy
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig


def test_bounded_rollout_full_parameter_backward():
    torch.manual_seed(83)
    cfg = StageB1bConfig(latent_dim=4, bath_dim=2, energy_hidden=8,
                        energy_heads=2, energy_depth=1, step_ps=1000.)
    kernel = StageB1b(cfg).double()
    kernel.energy = BoundedRidgeEnergy(4, features=8).double()
    H = torch.randn(2, 3, 4, dtype=torch.float64)
    mask = torch.ones(2, 3, dtype=torch.bool)
    temp = torch.tensor([348., 413.], dtype=H.dtype)
    trajectory = kernel.rollout_microstep_v1(
        H, mask, temp, 4, seq_feature=H.detach(), detach=False,
        collective_mobility=True, microsteps=4, integrator='coupled_midpoint_v1',
        generator=torch.Generator().manual_seed(17))
    trajectory[:, -1].square().mean().backward()
    for module in (kernel.energy, kernel.bath):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0
