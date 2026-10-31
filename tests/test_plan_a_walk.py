"""Contract tests for the Plan-A additions: decoder unfreeze + walk metrics."""
import argparse

import torch

from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer


class _Enc(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__(); self.lin = torch.nn.Linear(d, d)
    def forward(self, *a, **k):
        raise NotImplementedError


class _Dec(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__(); self.lin = torch.nn.Linear(d, d)


class _Kern(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__(); self.lin = torch.nn.Linear(d, d)


def _producer(train_encoder, train_decoder):
    return HSEnsembleProducer(_Enc(), _Dec(), _Kern(),
                              train_encoder=train_encoder,
                              train_decoder=train_decoder)


def test_decoder_frozen_by_default():
    """Default contract: encoder and decoder both frozen, only kernel trains."""
    p = _producer(False, False)
    assert all(not q.requires_grad for q in p.encoder.parameters())
    assert all(not q.requires_grad for q in p.decoder.parameters())
    assert p._encoder_grad is False and p._decoder_grad is False
    assert all(q.requires_grad for q in p.kernel.parameters())


def test_decoder_unfreeze_is_independent_of_encoder():
    """--train-decoder must NOT require --train-encoder (groups are additive)."""
    p = _producer(train_encoder=False, train_decoder=True)
    assert all(not q.requires_grad for q in p.encoder.parameters())
    assert all(q.requires_grad for q in p.decoder.parameters())
    assert p._decoder_grad is True


def test_walk_metrics_persistence_shape():
    """A no-motion walk must have zero path length and the worst per-step RMSD."""
    from evaluation.trajectory_metrics import walk_summary
    torch.manual_seed(0)
    M, T, L = 4, 6, 12
    mask = torch.ones(L, dtype=torch.bool)
    anchor = torch.randn(L, 3)
    true = anchor[None] + 0.2 * torch.randn(T, L, 3).cumsum(0)
    pers = anchor[None, None].expand(M, T, L, 3)
    s = walk_summary(pers, true[1:1 + T], anchor, mask)
    assert abs(s["path_length"]) < 1e-6
    moving = walk_summary(true[None].expand(M, T, L, 3), true[1:1 + T], anchor, mask)
    assert moving["per_step_rmsd_mean"] < s["per_step_rmsd_mean"]


def test_acf_lag1_is_not_the_useless_integral():
    """The lag-1 ACF must be a real per-window number, not the -0.5 identity."""
    from evaluation.trajectory_metrics import autocorr_time
    torch.manual_seed(0)
    L = 12
    mask = torch.ones(L, dtype=torch.bool)
    x = torch.randn(8, L, 3)
    r = autocorr_time(x, mask)
    assert -0.6 <= r["acf_lag1"] <= 1.0
    assert abs(r["acf_lag1"] + 0.5) > 1e-6   # NOT the degenerate identity
