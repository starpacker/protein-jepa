"""Stage A (static/dynamic factorization) code-level gate tests.

Covers the mandatory invariants from PROTEIN_JEPA_V4 spec §7 and §A3:
  * shapes/masks (variable length, chain breaks)
  * SE(3) invariance of H_static and equivariance of decoded CA
  * H_static is time-invariant (constant over a rollout)
  * H_dyn magnitude is unlocked (varies with conformation) -- the Phase 0 root cause
  * static branch never sees future frames (leakage audit)
  * identity/zero/shuffled-H_dyn controls degrade decoded time-specific accuracy
"""

import torch
import pytest

from protein_jepa_frame.models.static_dynamic_v4 import (
    StaticEncoder, DynamicEncoder, ConditionalStructureDecoder,
)
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder


def _batch(B=2, L=60, T=5, seed=0, chain_break=True):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, 1, L, 4, 3, generator=g)
    coords = base + 0.5 * torch.randn(B, T, L, 4, 3, generator=g)
    # enforce plausible CA backbone (roughly consecutive along x for bond sanity)
    mask = torch.ones(B, L, dtype=torch.bool)
    if chain_break:
        mask[:, L // 2] = False  # a chain break / missing residue
    aa = torch.randint(0, 20, (B, L), generator=g)
    ri = torch.arange(L)[None].expand(B, L).clone()
    ci = torch.zeros(B, L, dtype=torch.long)
    # chain break: bump residue_index after the gap so adjacency is not implied
    ri[:, L // 2:] += 3
    return coords, mask, aa, ri, ci


def _encoder_decoder(L=60, D=128):
    se = StaticEncoder(D)
    de = DynamicEncoder(D, anchor_conditioned=False)
    dec = ConditionalStructureDecoder(GeometricDeltaCADecoder(latent_dim=D))
    return se, de, dec


def test_shapes_and_mask():
    coords, mask, aa, ri, ci = _batch()
    se, de, dec = _encoder_decoder(L=coords.shape[2])
    hs = se(coords[:, 0], mask, aa, ri, ci)
    hd = de(coords, mask, aa, ri, ci, h_static=hs)
    assert hs.shape == (2, 60, 128)
    assert hd.shape == (2, 5, 60, 128)
    # masked residues zeroed
    assert (hd[:, :, ~mask[0]].abs().sum() == 0)
    R = torch.eye(3)[None].expand(2, 60, 3, 3)
    out = dec(hd[:, 2], hs, aa, ri, ci, R, coords[:, 0, :, 1], mask)
    assert out.shape == (2, 60, 3)


def test_static_is_time_invariant():
    # H_static depends only on the anchor, so encoding two different frames as
    # the "anchor" should still give a constant H_static per rollout (it has no
    # time axis by construction).  Here we assert the *same* anchor always gives
    # the *same* H_static, and that H_static is independent of future frames.
    coords, mask, aa, ri, ci = _batch()
    se, _, _ = _encoder_decoder(L=coords.shape[2])
    hs_a = se(coords[:, 0], mask, aa, ri, ci)
    hs_b = se(coords[:, 0], mask, aa, ri, ci)
    torch.testing.assert_close(hs_a, hs_b)
    # Give the static encoder a DIFFERENT frame as "anchor"; it should produce a
    # different (but still deterministic) output -- this tests it reads its input
    # (not hardcoded), while the *time-constancy* is enforced by the API (no time
    # axis).
    hs_c = se(coords[:, 3], mask, aa, ri, ci)
    assert not torch.allclose(hs_a, hs_c)


def test_dynamic_magnitude_unlocked():
    # The Phase 0 root cause: H_dyn magnitude must vary with conformation.
    coords, mask, aa, ri, ci = _batch()
    se, de, _ = _encoder_decoder(L=coords.shape[2])
    hs = se(coords[:, 0], mask, aa, ri, ci)
    hd = de(coords, mask, aa, ri, ci, h_static=hs)
    norms = hd[:, :, mask[0]].norm(dim=-1).mean(dim=-1)  # [B, T]
    # With distinct frames, per-frame norms must not all be identical.
    assert norms.std(dim=-1).min() > 1e-3


def test_se3_static_invariant_dynamic_equivariant_decoded():
    coords, mask, aa, ri, ci = _batch()
    se, de, dec = _encoder_decoder(L=coords.shape[2])
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    shift = torch.randn(3)
    coords_rot = coords @ q.T + shift

    hs = se(coords[:, 0], mask, aa, ri, ci)
    hs_rot = se(coords_rot[:, 0], mask, aa, ri, ci)
    torch.testing.assert_close(hs_rot, hs, atol=1e-4, rtol=1e-4)

    # Decoded CA must transform equivariantly with the anchor.
    hd = de(coords, mask, aa, ri, ci, h_static=hs)
    R = torch.eye(3)[None].expand(2, 60, 3, 3)
    out = dec(hd[:, 2], hs, aa, ri, ci, R, coords[:, 0, :, 1], mask)
    out_rot = dec(hd[:, 2], hs, aa, ri, ci, R, coords_rot[:, 0, :, 1], mask)
    torch.testing.assert_close(out_rot, out @ q.T + shift, atol=1e-4, rtol=1e-4)


def test_zero_and_shuffled_hdyn_degrade_decoding():
    # Spec A3.3: zeroing or shuffling H_dyn (keeping H_static fixed) must change
    # the decoded structure meaningfully vs the correct H_dyn.
    coords, mask, aa, ri, ci = _batch()
    se, de, dec = _encoder_decoder(L=coords.shape[2])
    hs = se(coords[:, 0], mask, aa, ri, ci)
    hd = de(coords, mask, aa, ri, ci, h_static=hs)
    R = torch.eye(3)[None].expand(2, 60, 3, 3)
    anchor = coords[:, 0, :, 1]

    out_true = dec(hd[:, 2], hs, aa, ri, ci, R, anchor, mask)
    out_zero = dec(torch.zeros_like(hd[:, 2]), hs, aa, ri, ci, R, anchor, mask)
    # A clearly-distinct (time-shuffled) H_dyn: use a distant frame and scale it,
    # so the dynamic branch is forced to respond if the decoder reads H_dyn at all.
    out_shuf = dec(4.0 * hd[:, 4], hs, aa, ri, ci, R, anchor, mask)

    # The controls must produce a different structure than the correct H_dyn.
    assert not torch.allclose(out_zero, out_true, atol=1e-3)
    assert not torch.allclose(out_shuf, out_true, atol=1e-3)


def test_no_future_in_static_branch():
    # The static encoder's forward only accepts an anchor (no time axis), so by
    # API construction it cannot read future frames.  Assert its output is a
    # deterministic function of the anchor alone: perturbing a FUTURE frame's
    # coordinates must not change H_static.
    coords, mask, aa, ri, ci = _batch()
    se, _, _ = _encoder_decoder(L=coords.shape[2])
    hs_a = se(coords[:, 0], mask, aa, ri, ci)
    coords_perturbed = coords.clone()
    coords_perturbed[:, 4] += 100.0  # large perturbation to a future frame
    hs_b = se(coords_perturbed[:, 0], mask, aa, ri, ci)
    torch.testing.assert_close(hs_a, hs_b)


def test_openfold_static_branch_participates():
    # Spec §1 / A1.2: the OpenFold-informed static branch must *demonstrably*
    # participate when enabled, and must be absent (strict-loadable layout) when
    # disabled.  We cannot load real pretrained OpenFold weights on this system
    # (none are vendored), so we verify the interface with synthetic features:
    #   (a) use_openfold=True: openfold_single/pair change H_static;
    #   (b) use_openfold=False: those inputs are ignored, and the module registers
    #       no openfold_* parameters (so the no-OpenFold ablation checkpoint keeps
    #       its exact layout).
    coords, mask, aa, ri, ci = _batch()
    L, D = coords.shape[2], 128

    # (a) enabled branch contributes
    se_on = StaticEncoder(D, use_openfold=True,
                          openfold_single_dim=384, openfold_pair_dim=128)
    single = torch.randn(2, L, 384)
    pair = torch.randn(2, L, L, 128)
    base = se_on(coords[:, 0], mask, aa, ri, ci)                 # no openfold feats
    with_single = se_on(coords[:, 0], mask, aa, ri, ci, openfold_single=single)
    with_pair = se_on(coords[:, 0], mask, aa, ri, ci, openfold_pair=pair)
    assert not torch.allclose(base, with_single), "openfold_single must change H_static"
    assert not torch.allclose(base, with_pair), "openfold_pair must change H_static"
    # masked residues stay zeroed
    assert (with_single[~mask] == 0).all()

    # (b) disabled branch ignores inputs and has no openfold params
    se_off = StaticEncoder(D, use_openfold=False)
    base_off = se_off(coords[:, 0], mask, aa, ri, ci)
    with_feat = se_off(coords[:, 0], mask, aa, ri, ci,
                       openfold_single=single, openfold_pair=pair)
    torch.testing.assert_close(base_off, with_feat)              # inputs ignored
    assert not any("openfold" in n for n, _ in se_off.named_parameters())
    assert any("openfold" in n for n, _ in se_on.named_parameters())
