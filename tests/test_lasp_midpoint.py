"""Dense-reference and covariance contracts for coupled midpoint LASP."""
import torch

from protein_jepa_frame.models.lasp_midpoint import CoupledMidpoint


def _setup():
    dtype = torch.float64
    D = torch.tensor([[[0.7, 0.4], [0.6, 0.9]]], dtype=dtype)
    C = torch.tensor([[[[0.3], [-0.2]], [[0.1], [0.4]]]], dtype=dtype)
    Gamma = torch.full((1, 2, 1), 0.5, dtype=dtype)
    tail = torch.tensor([2., 4.], dtype=dtype)
    factor = torch.tensor([[[1., 0.2], [0.1, 0.8]]], dtype=dtype)
    solver = CoupledMidpoint(D, C, Gamma, tail, torch.ones(1, 2, dtype=torch.bool),
                             3., factor, tolerance=1e-12, max_iterations=40)
    return solver


def test_coupled_midpoint_matches_dense_and_stationary_covariance():
    solver = _setup()
    eye = torch.eye(6, dtype=torch.float64)
    B = torch.stack([solver.block(e.reshape(1, 2, 3)).flatten() for e in eye], 1)
    torch.testing.assert_close(B, B.T, rtol=1e-13, atol=1e-13)
    assert torch.linalg.eigvalsh(B).min() > 0
    left = eye + solver.dt / 2 * B
    right = eye - solver.dt / 2 * B
    transition = torch.linalg.solve(left, right)
    covariance = torch.linalg.solve(left, 2 * solver.dt * B)
    covariance = torch.linalg.solve(left, covariance.T).T
    torch.testing.assert_close(transition @ transition.T + covariance, eye,
                               rtol=1e-12, atol=1e-12)
    assert torch.linalg.svdvals(transition).max() < 1
    rhs = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    actual = solver.solve(rhs)
    expected = torch.linalg.solve(left, rhs.flatten()).reshape_as(rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    # Apply each independent noise basis through the actual step primitive.
    columns = []
    for e in eye:
        z = e.reshape(1, 2, 3)
        H = torch.zeros(1, 2, 2, dtype=torch.float64)
        S = torch.zeros(1, 2, 1, dtype=torch.float64)
        h, s = solver.step(H, S, H, z[..., :2], z[..., 2:])
        columns.append(solver.scale(torch.cat((h, s), -1)).flatten())
    noise_map = torch.stack(columns, 1)
    torch.testing.assert_close(noise_map @ noise_map.T, covariance, rtol=1e-11, atol=1e-11)


def test_coupled_midpoint_gradients_match_dense_solve():
    solver = _setup()
    rhs = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    actual = solver.solve(rhs)
    eye = torch.eye(6, dtype=torch.float64)
    A = torch.stack([solver.system(e.reshape_as(rhs)).flatten() for e in eye], 1)
    expected = torch.linalg.solve(A, rhs.flatten()).reshape_as(rhs)
    a = torch.autograd.grad(actual.square().sum(), rhs)[0]
    b = torch.autograd.grad(expected.square().sum(), rhs)[0]
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_coupled_midpoint_zero_rhs_has_correct_gradient():
    solver = _setup()
    rhs = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
    actual = solver.solve(rhs)
    eye = torch.eye(6, dtype=torch.float64)
    A = torch.stack([solver.system(e.reshape_as(rhs)).flatten() for e in eye], 1)
    gradient = torch.autograd.grad(actual.sum(), rhs)[0]
    expected = torch.linalg.solve(A.T, torch.ones(6, dtype=rhs.dtype)).reshape_as(rhs)
    torch.testing.assert_close(gradient, expected, rtol=1e-11, atol=1e-11)


def test_coupled_midpoint_coefficient_gradients_match_dense():
    solver = _setup()
    names = ("root_D", "C", "root_Gamma", "root_tail", "factor")
    for name in names:
        setattr(solver, name, getattr(solver, name).detach().requires_grad_())
    rhs = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3).requires_grad_()
    actual = solver.solve(rhs)
    eye = torch.eye(6, dtype=rhs.dtype)
    A = torch.stack([solver.system(e.reshape_as(rhs)).flatten() for e in eye], 1)
    expected = torch.linalg.solve(A, rhs.flatten()).reshape_as(rhs)
    parameters = [rhs] + [getattr(solver, name) for name in names]
    a = torch.autograd.grad(actual.square().sum(), parameters)
    b = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, dense_gradient in zip(a, b):
        torch.testing.assert_close(actual_gradient, dense_gradient, rtol=1e-10, atol=1e-10)


def test_coupled_midpoint_masks_padded_residues():
    solver = _setup()
    solver.mask[:, 1] = 0
    H = torch.ones(1, 2, 2, dtype=torch.float64)
    S = torch.ones(1, 2, 1, dtype=torch.float64)
    h, s = solver.step(H, S, -solver.root_tail.square() * H, H, S)
    assert torch.equal(h[:, 1], torch.zeros_like(h[:, 1]))
    assert torch.equal(s[:, 1], torch.zeros_like(s[:, 1]))
