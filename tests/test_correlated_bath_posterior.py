"""Independent oracles for the experimental correlated-mobility bath filter."""
import pytest
import torch

from protein_jepa_frame.models.correlated_bath_posterior import (
    apply_spatial_mobility,
    bath_mobility_gram,
    correlated_bath_filter_step,
)


def fixture(batch=1, length=2, latent=3, bath=2):
    rng = torch.Generator().manual_seed(84)
    def normal(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float64)
    mean = .2 * normal(batch, length, bath)
    raw_prior = .15 * normal(batch, length * bath, length * bath)
    prior = torch.linalg.cholesky(
        raw_prior @ raw_prior.transpose(-1, -2) + torch.eye(length * bath))
    delta = .3 * normal(batch, length, latent)
    force = .2 * normal(batch, length, latent)
    diffusion = normal(batch, length, latent).square() * .1 + .2
    coupling = .4 * normal(batch, length, latent, bath)
    gamma = normal(batch, length, bath).square() * .1 + .1
    kernel = .65 * torch.ones(batch, length, length, dtype=torch.float64)
    kernel += .35 * torch.eye(length, dtype=torch.float64)
    return mean, prior, delta, force, diffusion, coupling, gamma, kernel, .4


def dense_matrices(diffusion, coupling, kernel):
    length, latent, bath = coupling.shape
    root = torch.diag(diffusion.flatten().sqrt())
    mobility = root @ torch.kron(kernel.contiguous(), torch.eye(latent)) @ root
    global_coupling = torch.block_diag(*coupling.unbind())
    assert global_coupling.shape == (length * latent, length * bath)
    return mobility, global_coupling


def test_structured_contractions_match_full_mobility():
    args = fixture(batch=2)
    _, _, delta, _, diffusion, coupling, _, kernel, _ = args
    applied = apply_spatial_mobility(delta, diffusion, kernel)
    gram = bath_mobility_gram(diffusion, coupling, kernel)
    for i in range(len(delta)):
        mobility, c = dense_matrices(diffusion[i], coupling[i], kernel[i])
        torch.testing.assert_close(applied[i].flatten(), mobility @ delta[i].flatten())
        torch.testing.assert_close(gram[i], c.T @ mobility @ c)


def test_filter_matches_direct_joint_gaussian_conditioning():
    args = fixture(batch=2)
    mean, prior, delta, force, diffusion, coupling, gamma, kernel, h = args
    actual_mean, actual_chol = correlated_bath_filter_step(*args)
    for i in range(len(mean)):
        m, g, f, observed = [x[i].flatten() for x in (mean, gamma, force, delta)]
        p = prior[i] @ prior[i].T
        mobility, c = dense_matrices(diffusion[i], coupling[i], kernel[i])
        b = -h * mobility @ c
        r = 2. * h * mobility
        a = torch.diag(1. - h * g)
        # Build the joint (dH,S_next) before conditioning, including shared noise.
        transition = a + c.T @ b
        observation_mean = h * mobility @ f + b @ m
        next_mean = a @ m + c.T @ observation_mean
        observation_covariance = b @ p @ b.T + r
        cross = transition @ p @ b.T + c.T @ r
        next_covariance = transition @ p @ transition.T + c.T @ r @ c \
            + torch.diag(2. * h * g)
        expected_mean = next_mean + cross @ torch.linalg.solve(
            observation_covariance, observed - observation_mean)
        expected_covariance = next_covariance - cross @ torch.linalg.solve(
            observation_covariance, cross.T)
        torch.testing.assert_close(actual_mean[i].flatten(), expected_mean, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual_chol[i] @ actual_chol[i].T,
                                   expected_covariance, atol=1e-12, rtol=1e-12)


def test_sampler_monte_carlo_residual_covariance_and_observation_independence():
    args = fixture()
    mean, prior, _, force, diffusion, coupling, gamma, kernel, h = args
    mobility, c = dense_matrices(diffusion[0], coupling[0], kernel[0])
    hidden, observed = mean.numel(), force.numel()
    # Evaluate the filter's affine map on a basis, independently of its algebra.
    probes = torch.cat((torch.zeros(1, observed, dtype=torch.float64),
                        torch.eye(observed, dtype=torch.float64)))
    probe_args = [x.expand(len(probes), *x.shape[1:]) for x in args[:-1]]
    probe_args[2] = probes.reshape(len(probes), *force.shape[1:])
    probe_mean, probe_chol = correlated_bath_filter_step(*probe_args, h)
    intercept = probe_mean[0].flatten()
    slope = probe_mean[1:].flatten(1) - intercept
    posterior_chol = probe_chol[0]

    count = 60000
    rng = torch.Generator().manual_seed(59)
    initial = mean.flatten() + torch.randn(count, hidden, generator=rng,
                                          dtype=torch.float64) @ prior[0].T
    noise_h = torch.randn(count, observed, generator=rng, dtype=torch.float64) \
        @ torch.linalg.cholesky(2. * h * mobility).T
    noise_s = torch.randn(count, hidden, generator=rng, dtype=torch.float64) \
        * (2. * h * gamma.flatten()).sqrt()
    delta = h * (force.flatten() - initial @ c.T) @ mobility.T + noise_h
    next_bath = (1. - h * gamma.flatten()) * initial + delta @ c + noise_s
    predicted = intercept + delta @ slope
    residual = torch.linalg.solve_triangular(
        posterior_chol, (next_bath - predicted).T, upper=False).T
    assert residual.mean(0).abs().max() < .02
    torch.testing.assert_close(torch.cov(residual.T), torch.eye(hidden, dtype=torch.float64),
                               atol=.025, rtol=0)
    standardized_observation = (delta - delta.mean(0)) / delta.std(0)
    cross = residual.T @ standardized_observation / count
    assert cross.abs().max() < .02
    # Nonzero posterior cross-residue covariance must survive into sampling.
    covariance = posterior_chol @ posterior_chol.T
    assert covariance[:2, 2:].abs().max() > 1e-4


def test_identity_spatial_kernel_recovers_independent_residue_filters():
    args = list(fixture())
    args[1] = torch.eye(4, dtype=torch.float64)[None]
    args[7] = torch.eye(2, dtype=torch.float64)[None]
    mean, chol = correlated_bath_filter_step(*args)
    covariance = chol @ chol.transpose(-1, -2)
    torch.testing.assert_close(covariance[0, :2, 2:], torch.zeros(2, 2, dtype=torch.float64))
    for residue in range(2):
        single = [x[:, residue:residue + 1] for x in args[:-1]]
        single[1] = torch.eye(2, dtype=torch.float64)[None]
        single[7] = torch.ones(1, 1, 1, dtype=torch.float64)
        local_mean, local_chol = correlated_bath_filter_step(*single, args[-1])
        torch.testing.assert_close(mean[:, residue:residue + 1], local_mean)
        block = covariance[:, 2 * residue:2 * residue + 2, 2 * residue:2 * residue + 2]
        torch.testing.assert_close(block, local_chol @ local_chol.transpose(-1, -2))


def test_filter_gradients_include_mobility_history_force_and_prior():
    base = fixture(latent=2, bath=1)
    inputs = tuple(x.clone().requires_grad_() for x in base[:-1])

    def objective(mean, raw_prior, delta, force, raw_diffusion, coupling,
                  raw_gamma, raw_kernel):
        prior = raw_prior.tril(-1) + torch.diag_embed(
            torch.nn.functional.softplus(raw_prior.diagonal(dim1=-2, dim2=-1)) + .2)
        kernel = raw_kernel @ raw_kernel.transpose(-1, -2) \
            + .2 * torch.eye(raw_kernel.shape[-1])
        return correlated_bath_filter_step(
            mean, prior, delta, force, raw_diffusion.exp(), coupling,
            raw_gamma.exp(), kernel, base[-1], validate=False)

    assert torch.autograd.gradcheck(objective, inputs, eps=1e-6, atol=2e-6, rtol=1e-4)


def test_singular_spatial_noise_is_rejected_for_arbitrary_observations():
    args = list(fixture())
    args[7] = torch.ones_like(args[7])
    with pytest.raises(ValueError, match='symmetric positive definite'):
        correlated_bath_filter_step(*args)
