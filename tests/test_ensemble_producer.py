"""Tests for the model-side ensemble producer (T3)."""
from __future__ import annotations

import torch

from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer


class _DummyEncoder(torch.nn.Module):
    """Deterministic toy encoder: H = per-residue CA position lifted to dH."""
    def __init__(self, dH=6):
        super().__init__()
        self.dH = dH
        self.proj = torch.nn.Linear(3, dH)

    def forward(self, coords, mask, aatype, residue_index, chain_id, token_mask=None):
        ca = coords[:, :, :, 1, :]          # [B,T,L,3]
        return self.proj(ca)                # [B,T,L,dH]


class _DummyDecoder(torch.nn.Module):
    """Frozen-ish decoder: CA = anchor CA + small function of H."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)

    def decode_structure(self, H, aatype, R_anchor, t_anchor, residue_mask):
        B, L, dH = H.shape
        delta = H[:, :, :3] * self.scale
        ca = t_anchor + delta
        backbone = torch.zeros(B, L, 4, 3, device=H.device, dtype=H.dtype)
        backbone[:, :, 1, :] = ca
        return {"backbone_coords": backbone}


def _batch(B=2, K=4, L=6, dH=6):
    g = torch.Generator().manual_seed(0)
    coords = torch.randn(B, K, L, 4, 3, generator=g)
    mask = torch.ones(B, L, dtype=torch.bool)
    aa = torch.zeros(B, L, dtype=torch.long)
    ri = torch.arange(L).expand(B, L)
    ci = torch.zeros(B, L, dtype=torch.long)
    R = torch.eye(3).expand(B, L, 3, 3).clone()
    t = torch.randn(B, L, 3, generator=g)
    return coords, mask, aa, ri, ci, R, t


def _producer(dH=6):
    enc = _DummyEncoder(dH)
    for p in enc.parameters():
        p.requires_grad_(False)
    head = GaussianHHead(dH, hidden=16)
    return EnsembleProducer(enc, _DummyDecoder(), head)


def test_sample_shapes():
    prod = _producer()
    coords, mask, aa, ri, ci, R, t = _batch()
    out = prod.sample_future_ca(coords, mask, aa, ri, ci, R, t, n_samples=5)
    assert out["ca_samples"].shape == (2, 5, 6, 3)
    assert out["ca_mean"].shape == (2, 6, 3)
    assert out["H_samples"].shape == (2, 5, 6, 6)
    assert out["mu"].shape == (2, 6, 6)
    assert out["sigma"].shape == (2, 6, 6)


def test_zero_epsilon_reproduces_mu_deterministically():
    prod = _producer()
    coords, mask, aa, ri, ci, R, t = _batch()
    eps = torch.zeros(2, 3, 6, 6)
    out = prod.sample_future_ca(coords, mask, aa, ri, ci, R, t, n_samples=3, eps=eps)
    # all samples identical and equal to decoding mu
    assert torch.allclose(out["ca_samples"][:, 0], out["ca_samples"][:, -1])
    H_mu = out["mu"]
    dec = prod.decoder.decode_structure(H_mu, aa, R, t, mask)
    assert torch.allclose(out["ca_samples"][:, 0], dec["backbone_coords"][:, :, 1, :], atol=1e-5)


def test_grad_flows_to_head_only():
    prod = _producer()
    coords, mask, aa, ri, ci, R, t = _batch()
    out = prod.sample_future_ca(coords, mask, aa, ri, ci, R, t, n_samples=4)
    loss = out["ca_samples"].mean()
    loss.backward()
    head_grads = [p.grad for p in prod.head.parameters() if p.grad is not None]
    assert len(head_grads) > 0, "head received no gradient"
    assert any(g.abs().sum() > 0 for g in head_grads)
    # frozen encoder: no grads
    assert all(p.grad is None for p in prod.encoder.parameters())


class _FullDummyDecoder(_DummyDecoder):
    """Decoder that also exposes the OpenFold-style atom37/torsion views."""

    def decode_structure(self, H, aatype, R_anchor, t_anchor, residue_mask):
        out = super().decode_structure(H, aatype, R_anchor, t_anchor, residue_mask)
        B, L, _ = H.shape
        atom37 = torch.zeros(B, L, 37, 3, device=H.device, dtype=H.dtype)
        atom37[:, :, 1, :] = out["backbone_coords"][:, :, 1, :]     # CA column
        out["atom37_coords"] = atom37
        out["atom37_mask"] = torch.ones(B, L, 37, dtype=torch.bool, device=H.device)
        out["torsions"] = torch.zeros(B, L, 7, device=H.device, dtype=H.dtype)
        return out


def test_optional_backbone_outputs_surface_when_available():
    """The new backbone/atom37 keys appear iff the decoder provides them, and
    the CA column of ``backbone_samples`` equals ``ca_samples`` exactly."""
    enc = _DummyEncoder(6)
    for p in enc.parameters():
        p.requires_grad_(False)
    prod = EnsembleProducer(enc, _FullDummyDecoder(), GaussianHHead(6, hidden=16))
    coords, mask, aa, ri, ci, R, t = _batch()
    out = prod.sample_future_ca(coords, mask, aa, ri, ci, R, t, n_samples=5)
    assert out["backbone_samples"].shape == (2, 5, 6, 4, 3)
    assert out["atom37_samples"].shape == (2, 5, 6, 37, 3)
    assert out["torsion_samples"].shape == (2, 5, 6, 7)
    # lossless: the CA column of the backbone is exactly ca_samples
    assert torch.equal(out["backbone_samples"][:, :, :, 1, :], out["ca_samples"])
    # a minimal decoder still yields backbone_samples but no atom37 key
    minimal = _producer().sample_future_ca(coords, mask, aa, ri, ci, R, t, n_samples=5)
    assert "backbone_samples" in minimal
    assert "atom37_samples" not in minimal


def test_sigma_is_bounded():
    head = GaussianHHead(6, hidden=16, sigma_min=1e-3, sigma_max=5.0)
    h = torch.randn(2, 6, 6) * 100      # extreme input
    mu, sigma = head.forward(h)
    assert (sigma >= 1e-3 - 1e-6).all()
    assert (sigma <= 5.0 + 1e-6).all()


def test_samples_spread_scales_with_sigma():
    prod = _producer()
    coords, mask, aa, ri, ci, R, t = _batch()
    # force large sigma
    with torch.no_grad():
        prod.head.net[-1].bias.data[6:] = 2.0     # log_sigma ~ 2 -> sigma ~ 7 (clamped 5)
    g = torch.Generator().manual_seed(1)
    out = prod.sample_future_ca(coords, mask, aa, ri, ci, R, t, n_samples=64, generator=g)
    spread = out["ca_samples"].detach().std(dim=1)
    assert float(spread.mean()) > 1e-3, "ensemble should have non-trivial spread"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
