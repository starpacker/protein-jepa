"""CPU contracts for retaining the existing sampler's continuous H/S path."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from evaluation.ca_trajectory_adapter import sample_ca_trajectory
from test_ca_joint_encoder import fixture


def setup(mode='exact_spatial', temperature=True):
    model, batch = fixture()
    model.enable_encoder_training()  # Eval must suppress even enabled encoder gradients.
    model.kernel.s0_mode = mode
    model.noise_mix = .6 if mode == 'exact_spatial' else 0.
    model.noise_kernel, model.noise_length = 'gaussian', 8.
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in batch.items()}
    batch['coords'][1] += .3
    batch['residue_mask'][1, :, 1] = False  # Non-prefix padding.
    model.eval()
    inputs = (batch['coords'][:, :4], batch['residue_mask'][:, :4].all(1),
              *[batch[key] for key in ('aatype', 'residue_index', 'chain_id')],
              batch['temperature_k'] if temperature else None)
    return model, inputs


@pytest.mark.parametrize('mode', ['exact', 'exact_spatial', 'prior'])
@pytest.mark.parametrize('temperature', [True, False])
def test_terminal_rng_fixed_anchor_decoding_and_no_grad(mode, temperature, monkeypatch):
    model, inputs = setup(mode, temperature)
    generator = torch.Generator().manual_seed(73)
    initial_global = torch.get_rng_state()
    with torch.no_grad():
        expected_ca, expected_h = model.sample(*inputs, 3, 3, generator)
    expected_rng, expected_global = generator.get_state(), torch.get_rng_state()
    calls = []
    original_encode = model.bundle.encode
    def encode(*args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return original_encode(*args, **kwargs)
    monkeypatch.setattr(model.bundle, 'encode', encode)
    torch.set_rng_state(initial_global)
    generator = torch.Generator().manual_seed(73)
    result = sample_ca_trajectory(model, *inputs, 3, 3, generator,
                                  decode_chunk_size=2, context_dt_ps=.1)
    assert calls == [False, False]  # Only observed anchor/history, never predictions.
    assert result.ca.shape == (2, 3, 4, 6, 3)
    assert result.latent.shape == (2, 3, 4, 6, 8)
    assert result.bath.shape == (2, 3, 4, 6, 2)
    assert result.step_ps == .1
    assert torch.equal(result.ca[:, :, -1], expected_ca)
    assert torch.equal(result.latent[:, :, -1], expected_h)
    assert torch.equal(generator.get_state(), expected_rng)
    assert torch.equal(torch.get_rng_state(), expected_global)
    assert 'rollout' not in model.kernel.__dict__
    anchor = inputs[0][:, -1, :, 1]
    assert torch.equal(result.ca[:, :, 0], anchor[:, None].expand(2, 3, 6, 3))
    direct = anchor[:, None, None] + model.bundle.decoder(
        result.latent - result.latent[:, :, :1]) * inputs[1][:, None, None, :, None]
    torch.testing.assert_close(result.ca, direct, atol=1e-14, rtol=1e-14)
    assert all(not value.requires_grad for value in (result.ca, result.latent, result.bath))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_single_rollout_preserves_initial_and_continuous_bath_and_prefix(monkeypatch):
    model, inputs = setup()
    transitions, filters = [], []
    step, posterior = model.kernel.sample_step, model.kernel.exact_spatial_bath_posterior
    def track_step(h, s, *args, **kwargs):
        result = step(h, s, *args, **kwargs)
        transitions.append((h.clone(), s.clone(), result[0].clone(), result[1].clone()))
        return result
    def track_filter(history, *args, **kwargs):
        filters.append(history.shape[0])
        return posterior(history, *args, **kwargs)
    monkeypatch.setattr(model.kernel, 'sample_step', track_step)
    monkeypatch.setattr(model.kernel, 'exact_spatial_bath_posterior', track_filter)
    result = sample_ca_trajectory(model, *inputs, 4, 3, torch.Generator().manual_seed(73))
    assert filters == [2] and len(transitions) == 4
    h, s = result.latent.flatten(0, 1), result.bath.flatten(0, 1)
    for time, (h0, s0, h1, s1) in enumerate(transitions):
        assert torch.equal(h[:, time], h0) and torch.equal(s[:, time], s0)
        assert torch.equal(h[:, time + 1], h1) and torch.equal(s[:, time + 1], s1)
    shorter = sample_ca_trajectory(model, *inputs, 2, 3, torch.Generator().manual_seed(73))
    assert torch.equal(result.latent[:, :, :3], shorter.latent)
    assert torch.equal(result.bath[:, :, :3], shorter.bath)
    torch.testing.assert_close(result.ca[:, :, :3], shorter.ca, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize('where', ['rollout', 'decode'])
def test_exception_restores_original_method_and_releases_guard(where, monkeypatch):
    model, inputs = setup()
    original_rollout, original_decode = model.kernel.rollout, model.bundle.decode_ca
    if where == 'rollout':
        def fail(*args, **kwargs):
            raise RuntimeError('deliberate rollout failure')
        monkeypatch.setattr(model.kernel, 'rollout', fail)
    else:
        calls = 0
        def fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError('deliberate decode failure')
            return original_decode(*args, **kwargs)
        monkeypatch.setattr(model.bundle, 'decode_ca', fail)
    previous = model.kernel.rollout
    with pytest.raises(RuntimeError, match='deliberate'):
        sample_ca_trajectory(model, *inputs, 3, 2, torch.Generator().manual_seed(2))
    assert model.kernel.rollout == previous
    monkeypatch.setattr(model.kernel, 'rollout', original_rollout)
    monkeypatch.setattr(model.bundle, 'decode_ca', original_decode)
    result = sample_ca_trajectory(model, *inputs, 1, 2, torch.Generator().manual_seed(2))
    assert result.ca.shape[2] == 2


def test_reentrant_and_concurrent_calls_are_rejected_and_restored(monkeypatch):
    model, inputs = setup()
    original = model.kernel.rollout
    entered, release = threading.Event(), threading.Event()
    def blocking(*args, **kwargs):
        with pytest.raises(RuntimeError, match='recursive or concurrent'):
            sample_ca_trajectory(model, *inputs, 1, 2)
        entered.set()
        assert release.wait(timeout=10)
        return original(*args, **kwargs)
    monkeypatch.setattr(model.kernel, 'rollout', blocking)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sample_ca_trajectory, model, *inputs, 1, 2,
                             torch.Generator().manual_seed(2))
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(RuntimeError, match='recursive or concurrent'):
                sample_ca_trajectory(model, *inputs, 1, 2)
            with torch.no_grad(), pytest.raises(RuntimeError, match='exclusive rollout'):
                model.sample(*inputs, 1, 2)
        finally:
            release.set()
        assert future.result(timeout=10).bath is not None
    assert model.kernel.rollout is blocking


def test_eval_time_spacing_and_prediction_contracts():
    model, inputs = setup()
    model.train()
    with pytest.raises(ValueError, match='model.eval'):
        sample_ca_trajectory(model, *inputs, 2, 3)
    model.eval()
    with torch.inference_mode(), pytest.raises(ValueError, match='inference_mode'):
        sample_ca_trajectory(model, *inputs, 2, 3)
    with pytest.raises(ValueError, match='spacing'):
        sample_ca_trajectory(model, *inputs, 2, 3, context_dt_ps=1000.)
    model.prediction_kind = 'time_mixture'
    with pytest.raises(ValueError, match='endpoint'):
        sample_ca_trajectory(model, *inputs, 2, 3)
