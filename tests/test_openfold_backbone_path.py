"""Tests: the OpenFold backbone path surfaced by the ensemble producers (TASK 1).

Guards the T1.1 contract: the full mainchain returned by the producers is a
strict superset of the Cα ensemble and comes from the frozen decoder's official
OpenFold reconstruction (keys ``backbone_samples`` / ``atom37_samples`` /
``torsion_samples``), with the Cα column bit-identical to ``ca_samples``.
"""
from __future__ import annotations

import torch
import pytest

from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.models.stage_c.openfold_atom37 import OPENFOLD_ATOM37_NAMES


class _DummyEncoder(torch.nn.Module):
    def __init__(self, dH=16):
        super().__init__()
        self.linear = torch.nn.Linear(4 * 3, dH)

    def forward(self, coords, m3, aatype, residue_index, chain_id, token_mask=None):
        B, K, L, _, _ = coords.shape
        return self.linear(coords.reshape(B * K, L, 12)).reshape(B, K, L, -1)


def _make_ensemble_producer(L, dH=16):
    from protein_jepa_frame.models.stage_c.structure_decoder import StructureDecoder, DecoderConfig
    # Build a real StructureDecoder and a thin wrapper exposing decode_structure.
    dec = StructureDecoder(DecoderConfig(c_hidden=32, c_ipa=16, n_heads=2, n_ipa_blocks=1,
                                         q_dim=dH))
    dec.eval()

    class _Wrapper(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def decode_structure(self, H, aatype, R_anchor, t_anchor, mask):
            return self.d(aatype=aatype, q_dynamics=H, R_anchor=R_anchor,
                          t_anchor=t_anchor, residue_mask=mask)

        def parameters(self):
            return self.d.parameters()

    enc = _DummyEncoder(dH)
    head = GaussianHHead(dH, hidden=16, cond_dim=0)
    enc.eval(); head.eval()
    return EnsembleProducer(enc, _Wrapper(dec), head)


def test_producer_surfaces_openfold_backbone():
    torch.manual_seed(0)
    B, K, L = 2, 4, 20
    prod = _make_ensemble_producer(L).eval()
    ctx = torch.randn(B, K, L, 4, 3) * 8
    mask = torch.ones(B, L, dtype=torch.bool)
    aa = torch.randint(0, 20, (B, L))
    ri = torch.arange(L)[None].expand(B, L)
    ci = torch.zeros(B, L, dtype=torch.long)
    bb = ctx[:, -1]
    R, t = RigidBodyFrame.from_backbone(bb[..., 0, :], bb[..., 1, :], bb[..., 2, :])
    with torch.no_grad():
        out = prod.sample_future_ca(ctx, mask, aa, ri, ci, R, t, n_samples=3)

    assert out["backbone_samples"].shape == (B, 3, L, 4, 3)
    assert out["atom37_samples"].shape[-2] == len(OPENFOLD_ATOM37_NAMES)
    assert out["torsion_samples"].shape[-1] == 7
    # CA column is bit-identical to the CA ensemble (same atoms)
    assert torch.equal(out["ca_samples"], out["backbone_samples"][..., 1, :])
    assert not torch.isnan(out["backbone_samples"]).any()
