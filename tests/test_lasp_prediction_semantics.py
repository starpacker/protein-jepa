"""Observation semantics must not change path generation or legacy defaults."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics


class Bundle(nn.Module):
    def encode(self, coords, *args):
        return coords[:, :, 1]

    def decode_ca(self, generated, start, aa, ri, ci, rotation, anchor, mask):
        return anchor + generated-start


class Kernel(nn.Module):
    cfg = SimpleNamespace(step_ps=1000.)

    def rollout(self, start, mask, temp, steps, generator=None, **kw):
        eps = torch.randn(start.shape, generator=generator, device=start.device)
        self.path = torch.stack([start+i+eps*i*.01 for i in range(steps+1)], 1)
        return self.path


def model():
    result = CADynamics.__new__(CADynamics)
    nn.Module.__init__(result)
    result.bundle = Bundle()
    result.kernel = Kernel()
    result.history_conditioning = False
    return result.eval()


def inputs():
    return (torch.zeros(1, 8, 5, 4, 3), torch.ones(1, 5, dtype=torch.bool),
            torch.zeros(1, 5, dtype=torch.long), torch.arange(5)[None],
            torch.zeros(1, 5, dtype=torch.long), torch.tensor([348.]))


def test_default_and_explicit_endpoint_are_bitwise_equal():
    m = model()
    a = m.sample(*inputs(), 4, 12, torch.Generator().manual_seed(7))
    m.prediction_kind = 'endpoint'
    b = m.sample(*inputs(), 4, 12, torch.Generator().manual_seed(7))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_time_selection_preserves_noise_rng_and_selects_one_state_per_path():
    m = model()
    endpoint_rng = torch.Generator().manual_seed(9)
    m.sample(*inputs(), 16, 12, endpoint_rng)
    path = m.kernel.path.clone()
    mixture_rng = torch.Generator().manual_seed(9)
    m.prediction_kind = 'time_mixture'
    _, h = m.sample(*inputs(), 16, 12, mixture_rng)
    assert torch.equal(endpoint_rng.get_state(), mixture_rng.get_state())
    torch.testing.assert_close(m.kernel.path, path, rtol=0, atol=0)
    times = torch.tensor(m.last_observation_steps)
    assert times.shape == (12,) and times.min() >= 1 and times.max() <= 16
    assert times.unique().numel() > 1
    torch.testing.assert_close(h[0], path[torch.arange(12), times])


def test_horizon_one_mixture_equals_endpoint():
    m = model()
    a = m.sample(*inputs(), 1, 12, torch.Generator().manual_seed(9))
    m.prediction_kind = 'time_mixture'
    b = m.sample(*inputs(), 1, 12, torch.Generator().manual_seed(9))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_unknown_observation_semantics_fail_closed():
    m = model()
    m.prediction_kind = 'unknown'
    with pytest.raises(ValueError, match='prediction_kind'):
        m.sample(*inputs(), 1, 12, torch.Generator().manual_seed(9))
