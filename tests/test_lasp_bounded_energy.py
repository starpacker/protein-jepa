"""Dense derivative and size-independent curvature contracts."""
import torch
from protein_jepa_frame.models.lasp_bounded_energy import BoundedRidgeEnergy


def test_residual_hessian_bound_and_padding():
    torch.manual_seed(41)
    energy = BoundedRidgeEnergy(3, features=5, curvature_budget=0.8).double()
    with torch.no_grad():
        energy.amplitude.normal_(0, 4)
        energy.weight.mul_(10)
        energy.temperature_phase.normal_()
    for length in (1, 3, 7):
        H = torch.randn(2, length, 3, dtype=torch.float64)
        mask = torch.ones(2, length, dtype=torch.bool)
        if length > 1:
            mask[0, -1] = False
        temp = torch.tensor([320., 450.], dtype=H.dtype)
        hessian = torch.autograd.functional.hessian(lambda x: energy(x, mask, temp), H)
        dense = hessian.reshape(H.numel(), H.numel())
        tail = torch.nn.functional.softplus(energy.log_tail).expand_as(H)
        tail = torch.diag((tail * mask[..., None]).flatten())
        norm = torch.linalg.eigvalsh(dense - tail).abs().max()
        bound = energy.residual_hessian_bound()
        assert norm <= bound + 1e-12
        assert bound <= 0.8
        changed = H.clone()
        changed[~mask] = 1e5
        torch.testing.assert_close(energy(changed, mask, temp), energy(H, mask, temp))


def test_force_parameter_double_backward_and_nonconvexity():
    energy = BoundedRidgeEnergy(2, features=3, curvature_budget=2., tail_init=.01).double()
    with torch.no_grad():
        energy.amplitude.fill_(3)
        energy.weight.fill_(1)
    H = torch.zeros(1, 2, 2, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    hessian = torch.autograd.functional.hessian(lambda x: energy(x, mask), H)
    assert torch.linalg.eigvalsh(hessian.reshape(4, 4)).min() < 0
    H = torch.randn_like(H).requires_grad_()
    force = -torch.autograd.grad(energy(H, mask), H, create_graph=True)[0]
    grads = torch.autograd.grad(force.square().sum(), tuple(energy.parameters()), allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    assert grads[0].abs().max() > 0
    assert grads[3].abs().max() > 0


def test_mobility_scaled_curvature_bound():
    torch.manual_seed(71)
    energy = BoundedRidgeEnergy(2, features=4, curvature_budget=.7).double()
    with torch.no_grad():
        energy.amplitude.normal_(0, 3)
    H = torch.randn(1, 3, 2, dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    dense = torch.autograd.functional.hessian(lambda x: energy(x, mask), H).reshape(6, 6)
    tail = torch.diag(torch.nn.functional.softplus(energy.log_tail).repeat(3))
    factor = torch.tensor([[1., .2, .1], [.1, .8, .2], [0., .2, 1.1]], dtype=H.dtype)
    diffusion = torch.tensor([.1, .3, .2, .4, .1, .2], dtype=H.dtype)
    N = diffusion.sqrt().diag() @ torch.kron(factor, torch.eye(2, dtype=H.dtype))
    scaled = N.T @ (dense-tail) @ N
    bound = energy.residual_hessian_bound() * torch.linalg.matrix_norm(N, ord=2).square()
    assert torch.linalg.eigvalsh(scaled).abs().max() <= bound + 1e-12
