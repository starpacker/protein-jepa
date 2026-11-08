"""StageB integration of the full spatial posterior, independent of training."""
import pytest
import torch

from scripts.debug_kalman_stepwise import build


def setup_model():
    model = build(2, 2, 3, .4, .7, torch.tensor([.3, .2]),
                  torch.tensor([.25, .4]), torch.tensor([[.8, -.3], [.2, .7]])).double()
    model.s0_mode = 'exact_spatial'
    return model


def inputs():
    generator = torch.Generator().manual_seed(47)
    history = torch.randn(2, 4, 3, 2, dtype=torch.float64, generator=generator)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    history = history * mask[:, None, :, None]
    dt = torch.tensor([[.2, .4, .7, 0.], [.5, .3, .6, 0.]], dtype=torch.float64)
    # Non-triangular and non-prefix masks make dropping padded noise columns wrong.
    factor = torch.tensor([[[1., .4, .2], [.1, .9, -.2], [.3, -.4, 1.1]],
                           [[.8, -.3, .1], [.5, 1., .2], [-.2, .4, 1.]]], dtype=torch.float64)
    return history, dt, mask, factor


def dense_reference(model, history, dt, mask, factor):
    """Observation-space Kalman recursion, independent of information update."""
    diffusion, coupling, gamma = model.bath(history[:, -1], None)
    results = []
    for b in range(len(history)):
        valid = mask[b].nonzero(as_tuple=True)[0]
        n, dH, dS = len(valid), model.cfg.latent_dim, model.cfg.bath_dim
        K = (factor[b] @ factor[b].T)[valid][:, valid]
        root_D = torch.diag(diffusion[b, valid].flatten().sqrt())
        mobility = root_D @ torch.kron(K.contiguous(), torch.eye(dH, dtype=history.dtype)) @ root_D
        C = torch.block_diag(*coupling[b, valid])
        G = gamma[b, valid].flatten()
        m = history.new_zeros(n * dS)
        P = torch.eye(n * dS, dtype=history.dtype)
        for t in range(history.shape[1] - 1):
            h = dt[b, t]
            delta = (history[b, t + 1, valid] - history[b, t, valid]).flatten()
            force = -.7 * history[b, t, valid].flatten()
            observation = -h * mobility @ C
            noise = 2 * h * mobility
            innovation = delta - h * mobility @ force - observation @ m
            gain = torch.linalg.solve(observation @ P @ observation.T + noise,
                                      observation @ P).T
            updated_m = m + gain @ innovation
            I_KA = torch.eye(n * dS, dtype=history.dtype) - gain @ observation
            updated_P = I_KA @ P @ I_KA.T + gain @ noise @ gain.T
            decay = torch.diag(1 - h * G)
            m = decay @ updated_m + C.T @ delta
            P = decay @ updated_P @ decay.T + torch.diag(2 * h * G)
        results.append((valid, m, P))
    return results


def test_packed_spatial_posterior_matches_dense_reference_without_jitter():
    model = setup_model()
    model.cfg.jitter = .37  # Historical jitter must not leak into this new mode.
    history, dt, mask, factor = inputs()
    mean, chol = model.exact_spatial_bath_posterior(history, dt, None, mask, noise_factor=factor)
    assert mean.shape == (2, 3, 2) and chol.shape == (2, 6, 6)
    assert torch.count_nonzero(chol.triu(1)) == 0
    for b, (valid, expected_m, expected_P) in enumerate(dense_reference(model, history, dt, mask, factor)):
        flat = (valid[:, None] * 2 + torch.arange(2)).flatten()
        covariance = chol[b] @ chol[b].T
        torch.testing.assert_close(mean[b, valid].flatten(), expected_m, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(covariance[flat][:, flat], expected_P, atol=2e-12, rtol=2e-12)
        assert expected_P[:2, 2:].abs().max() > 1e-4
        invalid = (~mask[b]).repeat_interleave(2)
        assert not mean[b, ~mask[b]].count_nonzero()
        assert not chol[b, invalid].count_nonzero() and not chol[b, :, invalid].count_nonzero()


def test_identity_spatial_posterior_matches_legacy_moments_except_legacy_jitter():
    model = setup_model()
    model.cfg.jitter = .013
    history, dt, mask, _ = inputs()
    old_mean, old_factor = model.exact_bath_posterior(history, dt, None, mask)
    mean, factor = model.initial_bath_state(history, dt, None, mask)
    torch.testing.assert_close(mean, old_mean, atol=2e-12, rtol=2e-12)
    for b in range(len(history)):
        blocks = []
        for i in range(mask.shape[1]):
            block = old_factor[b, i] @ old_factor[b, i].T
            if mask[b, i]:
                block = block - model.cfg.jitter * torch.eye(2, dtype=history.dtype)
            blocks.append(block)
        torch.testing.assert_close(factor[b] @ factor[b].T, torch.block_diag(*blocks),
                                   atol=2e-12, rtol=2e-12)


def test_one_frame_prior_and_valid_only_spd_contract():
    model = setup_model()
    history, dt, mask, factor = inputs()
    factor = factor * mask[:, :, None]  # Full covariance singular; valid part SPD.
    mean, chol = model.exact_spatial_bath_posterior(
        history[:, -1:], dt[:, :1], None, mask, noise_factor=factor)
    assert not mean.count_nonzero()
    torch.testing.assert_close(chol, torch.diag_embed(mask.repeat_interleave(2, -1).double()))
    factor[:, :, :] = 0
    with pytest.raises(ValueError, match='positive definite'):
        model.exact_spatial_bath_posterior(history, dt, None, mask, noise_factor=factor)
    dt[0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite and positive'):
        model.exact_spatial_bath_posterior(history, dt, None, mask)


@pytest.mark.parametrize('path', ['rollout', 'rollout_microstep_v1'])
def test_rollout_draws_full_bath_factor_and_matches_explicit_initial_state(path):
    model = setup_model()
    history, dt, mask, noise_factor = inputs()
    mean, factor = model.initial_bath_state(history, dt, None, mask, noise_factor=noise_factor)
    rng = torch.Generator().manual_seed(91)
    eps = torch.randn(mean.shape, dtype=mean.dtype, generator=rng)
    S0 = mean + (factor @ eps.flatten(1)[..., None]).reshape_as(mean)
    rollout = getattr(model, path)
    common = dict(noise_factor=noise_factor, collective_mobility=True, return_bath=True)
    explicit = rollout(history[:, -1], mask, None, 2, S0=S0, generator=rng, **common)
    explicit_rng = rng.get_state()
    rng.manual_seed(91)
    inferred = rollout(history[:, -1], mask, None, 2, H_hist=history, dt_hist=dt,
                       generator=rng, **common)
    for actual, expected in zip(inferred, explicit):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(rng.get_state(), explicit_rng)
    for b in range(len(mask)):
        assert not inferred[1][b, :, ~mask[b]].count_nonzero()


def test_spatial_rollout_guards_and_explicit_s0_bypass():
    model = setup_model()
    history, dt, mask, factor = inputs()
    with pytest.raises(ValueError, match='observed H history'):
        model.rollout(history[:, -1], mask, None, 1, collective_mobility=True)
    with pytest.raises(ValueError, match='collective_mobility=True'):
        model.rollout(history[:, -1], mask, None, 1, H_hist=history, dt_hist=dt)
    for kwargs in ({'microsteps': 2}, {'integrator': 'coupled_midpoint_v1'}):
        with pytest.raises(ValueError, match='single-step explicit'):
            model.rollout_microstep_v1(history[:, -1], mask, None, 1, H_hist=history,
                                       dt_hist=dt, collective_mobility=True, **kwargs)
    S0 = torch.zeros(2, 3, 2, dtype=history.dtype)
    def sample(mode):
        model.s0_mode = mode
        return model.rollout_microstep_v1(history[:, -1], mask, None, 2, S0=S0,
            noise_factor=factor, collective_mobility=True, microsteps=3,
            generator=torch.Generator().manual_seed(16), return_bath=True)
    for spatial, prior in zip(sample('exact_spatial'), sample('prior')):
        torch.testing.assert_close(spatial, prior, atol=0, rtol=0)
    model.s0_mode = 'exact'  # Historical identity guard remains in force.
    with pytest.raises(ValueError, match='identity spatial noise covariance'):
        model.rollout(history[:, -1], mask, None, 1, H_hist=history, dt_hist=dt,
                      collective_mobility=True, noise_factor=factor)
    model.s0_mode, model.cfg.bath_dim = 'exact_spatial', 0
    for rollout in (model.rollout, model.rollout_microstep_v1):
        with pytest.raises(ValueError, match='requires a bath'):
            rollout(history[:, -1], mask, None, 1, H_hist=history, dt_hist=dt,
                    collective_mobility=True)


class TrainableEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.curvature = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))

    def forward(self, H, mask, temperature_k=None):
        return .5 * self.curvature * (H.square() * mask[..., None]).sum()


class TrainableBath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64))

    def forward(self, anchor, temperature_k):
        return (.3 * (self.weight * anchor).exp(),
                .4 + self.weight * anchor[..., None].expand(-1, -1, -1, 2),
                .2 + .1 * (self.weight * anchor).sigmoid())


@pytest.mark.parametrize('path', ['posterior', 'rollout'])
def test_spatial_history_energy_coefficients_and_noise_gradients(path):
    model = setup_model()
    model.energy, model.bath = TrainableEnergy(), TrainableBath()
    history, dt, mask, factor = inputs()
    history = history[:1].clone().requires_grad_()
    factor = factor[:1].clone().requires_grad_()
    mask, dt = mask[:1], dt[:1]
    def objective():
        if path == 'posterior':
            mean, chol = model.exact_spatial_bath_posterior(
                history, dt, None, mask, noise_factor=factor, create_graph=True)
            return mean.square().sum() + .3 * chol.square().sum()
        H, S = model.rollout(history[:, -1], mask, None, 2, H_hist=history, dt_hist=dt,
            noise_factor=factor, collective_mobility=True, detach=False, return_bath=True,
            generator=torch.Generator().manual_seed(34))
        return H[:, -1].square().sum() + .3 * S[:, -1].square().sum()
    variables = (history, model.energy.curvature, model.bath.weight, factor)
    gradients = torch.autograd.grad(objective(), variables)
    generator = torch.Generator().manual_seed(5)
    for variable, gradient in zip(variables, gradients):
        direction = torch.randn(variable.shape, dtype=variable.dtype, generator=generator)
        direction = direction / direction.norm()
        expected = (gradient * direction).sum()
        assert expected.abs() > 1e-6
        eps = 1e-5
        with torch.no_grad():
            variable.add_(eps * direction)
        plus = objective().detach()
        with torch.no_grad():
            variable.sub_(2 * eps * direction)
        minus = objective().detach()
        with torch.no_grad():
            variable.add_(eps * direction)
        torch.testing.assert_close(expected, (plus - minus) / (2 * eps), atol=3e-7, rtol=3e-5)


def test_spatial_filter_matches_actual_sampler_conditional_residual_covariance():
    """Actual collective draws, including shared H noise, pin the filter contract."""
    torch.manual_seed(33)
    model = setup_model()
    n = 40000
    mask = torch.ones(n, 2, dtype=torch.bool)
    factor = torch.tensor([[1., .3], [.2, .8]], dtype=torch.float64).expand(n, -1, -1)
    H = torch.zeros(n, 2, 2, dtype=torch.float64)
    S = torch.randn(n, 2, 2, dtype=torch.float64)
    states = [H]
    with torch.no_grad():
        for _ in range(2):
            H, S = model.sample_step(H, S, mask, None, states[0],
                                     noise_factor=factor, collective_mobility=True)
            states.append(H)
    observations = torch.cat([x.flatten(1) for x in states[1:]], -1)
    centered = observations - observations.mean(0)
    targets = S.flatten(1)
    centered_targets = targets - targets.mean(0)
    beta = torch.linalg.lstsq(centered, centered_targets).solution
    residuals = centered_targets - centered @ beta
    covariance = residuals.T @ residuals / n
    selected = torch.tensor([7, 19, 33])
    history = torch.stack(states, 1)[selected]
    mean, chol = model.exact_spatial_bath_posterior(
        history, torch.full((3, 3), .4, dtype=torch.float64), None, mask[selected],
        seq_feature=states[0][selected], noise_factor=factor[selected])
    expected_mean = targets.mean(0) + centered[selected] @ beta
    torch.testing.assert_close(mean.flatten(1), expected_mean, atol=.035, rtol=0)
    torch.testing.assert_close(chol[0] @ chol[0].T, covariance, atol=.035, rtol=0)
