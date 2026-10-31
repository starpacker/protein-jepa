"""Observed history and physical time must reach the GLE initial posterior."""
import pytest
import torch
from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer


class Encoder(torch.nn.Module):
    def forward(self, coords, *args, **kwargs):
        return coords[..., 1, :]


class CaptureProducer(HSEnsembleProducer):
    def _rollout_decode(self, *args):
        return {"history": args[-2], "dt": args[-1], "anchor": args[0]}


def sample(producer, times):
    coords = torch.arange(2 * 3 * 4 * 4 * 3).float().reshape(2, 3, 4, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    seq = torch.zeros(2, 4, dtype=torch.long)
    return producer.sample_future_ca(coords, mask, seq, seq, seq, None, None,
                                     2, context_time_ps=times)


def test_observed_history_and_dt_replicated_in_sample_order():
    p = CaptureProducer(Encoder(), torch.nn.Identity(), torch.nn.Identity(),
                        history_conditioning=True)
    out = sample(p, torch.tensor([[0., 2., 5.], [10., 14., 20.]]))
    assert out['history'].shape == (4, 3, 4, 3)
    assert torch.equal(out['history'][:, -1], out['anchor'])
    assert torch.equal(out['dt'], torch.tensor([[2., 2., 3.], [2., 2., 3.],
                                               [4., 4., 6.], [4., 4., 6.]]))
    assert not torch.equal(out['history'][0, 0], out['history'][0, -1])


def test_history_rejects_missing_or_noncausal_time():
    p = CaptureProducer(Encoder(), torch.nn.Identity(), torch.nn.Identity(),
                        history_conditioning=True)
    for times in (None, torch.tensor([[0., 2., 1.], [0., 1., 2.]])):
        with pytest.raises(ValueError):
            sample(p, times)


def test_legacy_anchor_only_does_not_require_history_time():
    p = CaptureProducer(Encoder(), torch.nn.Identity(), torch.nn.Identity())
    out = sample(p, None)
    assert out['history'] is None and out['dt'] is None
