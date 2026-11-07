"""Synthetic numerical checks only; no model checkpoints or trajectory access."""
import itertools

import pytest
import torch

from protein_jepa_frame.training.contact_ensemble_objective_v1 import contact_ensemble_brier


def cross_member_score(z, q):
    # Independent explicit ordered-pair expression, including target terms.
    products = [(z[i]-q)*(z[j]-q) for i in range(len(z)) for j in range(len(z)) if i != j]
    return torch.stack(products).mean()


@pytest.mark.parametrize('members', [2, 3, 12])
@pytest.mark.parametrize('kind', ['random', 'zero', 'constant', 'boundaries'])
def test_cross_member_values_and_gradients(members, kind):
    generator = torch.Generator().manual_seed(48)
    z = torch.rand(members, 5, dtype=torch.float64, generator=generator)
    if kind == 'zero':
        z.zero_()
    elif kind == 'constant':
        z[:] = torch.tensor([0., .2, .5, .8, 1.], dtype=z.dtype)
    elif kind == 'boundaries':
        z = (z >= .5).to(z.dtype)
    z.requires_grad_()
    q = torch.tensor([0., .3, .5, .7, 1.], dtype=z.dtype, requires_grad=True)
    before_z, before_q = z.detach().clone(), q.detach().clone()
    scores = contact_ensemble_brier(z, q)
    explicit = cross_member_score(z, q)
    torch.testing.assert_close(scores.population_brier_u, explicit, atol=2e-16, rtol=2e-15)
    gradients = torch.autograd.grad(scores.population_brier_u, (z, q), retain_graph=True)
    explicit_gradients = torch.autograd.grad(explicit, (z, q), retain_graph=True)
    for actual, expected in zip(gradients, explicit_gradients):
        # The explicit reference accumulates M*(M-1) target-gradient terms.
        torch.testing.assert_close(actual, expected, atol=3e-15, rtol=2e-14)
    analytical_z = 2*((z.sum(0)-z)/(members-1)-q)/(members*z.shape[1])
    analytical_q = 2*(q-z.mean(0))/z.shape[1]
    torch.testing.assert_close(gradients[0], analytical_z, atol=2e-16, rtol=2e-15)
    torch.testing.assert_close(gradients[1], analytical_q, atol=2e-16, rtol=2e-15)
    plugin_grad = torch.autograd.grad(scores.plug_in_brier, (z, q), retain_graph=True)
    penalty_grad = torch.autograd.grad(scores.variance_penalty, z)[0]
    torch.testing.assert_close(gradients[0]+penalty_grad, plugin_grad[0], atol=2e-16, rtol=2e-15)
    torch.testing.assert_close(gradients[1], plugin_grad[1], atol=0., rtol=0.)
    assert torch.equal(z, before_z) and torch.equal(q, before_q)


def test_negative_finite_member_score_is_retained():
    scores = contact_ensemble_brier(torch.tensor([[0.], [1.]], dtype=torch.float64),
                                    torch.tensor([.5], dtype=torch.float64))
    assert scores.plug_in_brier.item() == 0.
    assert scores.variance_penalty.item() == .25
    assert scores.population_brier_u.item() == -.25


def test_member_permutation_preserves_values_and_permutes_gradients():
    z = torch.linspace(0., 1., 36, dtype=torch.float64).reshape(12, 3).requires_grad_()
    q = torch.tensor([.1, .4, .9], dtype=z.dtype)
    permutation = torch.tensor([11, 2, 7, 0, 1, 6, 4, 8, 10, 9, 5, 3])
    permuted = z.detach()[permutation].requires_grad_()
    original, reordered = contact_ensemble_brier(z, q), contact_ensemble_brier(permuted, q)
    for a, b in zip(original, reordered):
        torch.testing.assert_close(a, b, atol=2e-16, rtol=2e-15)
        ga = torch.autograd.grad(a, z, retain_graph=True)[0]
        gb = torch.autograd.grad(b, permuted, retain_graph=True)[0]
        torch.testing.assert_close(ga[permutation], gb, atol=2e-16, rtol=2e-15)


@pytest.mark.parametrize('members', [2, 3, 4])
@pytest.mark.parametrize('probability', [0., .05, .5, .8, 1.])
def test_exhaustive_bernoulli_expected_bias_and_correction(members, probability):
    # Sum all 2**M member realizations exactly, without Monte Carlo tolerance.
    q = torch.tensor([.3], dtype=torch.float64)
    expected = torch.zeros(3, dtype=torch.float64)
    for bits in itertools.product((0., 1.), repeat=members):
        successes = int(sum(bits))
        weight = probability**successes * (1-probability)**(members-successes)
        z = torch.tensor(bits, dtype=q.dtype)[:, None]
        expected += weight*torch.stack(contact_ensemble_brier(z, q))
    population = (probability-q[0]).square()
    bias = probability*(1-probability)/members
    desired = torch.stack((population+bias, population.new_tensor(bias), population))
    torch.testing.assert_close(expected, desired, atol=2e-16, rtol=2e-15)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_actual_member_budget_dtype_device_and_noncontiguous_inputs(dtype):
    z = torch.linspace(0., 1., 48, dtype=dtype).reshape(4, 12).T.requires_grad_()
    q = torch.tensor([.2, .4, .6, .8], dtype=dtype)
    assert not z.is_contiguous()
    for score in contact_ensemble_brier(z, q):
        assert score.shape == () and score.dtype == dtype and score.device == z.device
        assert score.requires_grad


@pytest.mark.parametrize('z,q', [
    (torch.zeros(1, 3), torch.zeros(3)),
    (torch.zeros(0, 3), torch.zeros(3)),
    (torch.zeros(2, 0), torch.zeros(0)),
    (torch.zeros(2, 3), torch.zeros(4)),
    (torch.zeros(2, 3), torch.zeros(1, 3)),
    (torch.zeros(2, 3, 1), torch.zeros(3)),
    (torch.zeros(2, 3), torch.zeros(3, dtype=torch.float64)),
])
def test_invalid_shapes_or_dtype_mismatch_rejected(z, q):
    with pytest.raises(ValueError):
        contact_ensemble_brier(z, q)


@pytest.mark.parametrize('which', ['predictions', 'target'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -.01, 1.01])
def test_invalid_probabilities_rejected(which, value):
    z, q = torch.full((12, 3), .5), torch.full((3,), .5)
    if which == 'predictions':
        z[0, 0] = value
    else:
        q[0] = value
    with pytest.raises(ValueError):
        contact_ensemble_brier(z, q)


@pytest.mark.parametrize('z,q', [
    ([[.2], [.3]], torch.tensor([.5])),
    (torch.zeros(2, 1), [.5]),
    (torch.zeros(2, 1, dtype=torch.int64), torch.zeros(1)),
    (torch.zeros(2, 1), torch.zeros(1, dtype=torch.bool)),
    (torch.zeros(2, 1, dtype=torch.complex64), torch.zeros(1)),
])
def test_nontensors_and_nonfloating_tensors_rejected(z, q):
    with pytest.raises(TypeError):
        contact_ensemble_brier(z, q)


def test_device_mismatch_rejected_before_value_operations():
    with pytest.raises(ValueError, match='device'):
        contact_ensemble_brier(torch.zeros(2, 3), torch.zeros(3, device='meta'))
