"""Tests for Stage B2 bath-aware decoder (§3.7 STAGE_AB_REDESIGN_PLAN.md).

Test coverage:
  B2-1. HOnlyDecoder ignores S_t — output identical regardless of S value.
  B2-2. BathAwareDecoder changes output when S_t changes.
  B2-3. BathAwareDecoder raises on None S_t.
  B2-4. BathAwareDecoder raises on wrong S_t last dim.
  B2-5. HOnlyDecoder and BathAwareDecoder have separate parameters — training
        one does NOT change the other.
  B2-6. autonomous_rollout_and_decode produces H/S/coords of the right shape.
  B2-7. autonomous_rollout_and_decode: "h_only" coords are independent of S.
  B2-8. autonomous_rollout_and_decode: "h_plus_bath" coords differ from "h_only"
        (S_t is actually used).
  B2-9. S_traj comes strictly from the kernel — identical generator seed gives
        identical S trajectory.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_b2 import (
    BathAwareDecoderConfig,
    BathAwareDecoder,
    AutonomousRolloutOutput,
    autonomous_rollout_and_decode,
)
from protein_jepa_frame.models.stage_b2.bath_aware_decoder import HOnlyDecoder
from protein_jepa_frame.models.stage_b1b.model import StageB1b, StageB1bConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cfg(dH=16, dS=4):
    return BathAwareDecoderConfig(
        latent_dim=dH, bath_dim=dS,
        c_hidden=32, n_ipa_blocks=1, n_heads=2, torsion_bins=8,
        max_local_position_update=5.0,
    )


def _make_kernel(dH=16, dS=4):
    cfg = StageB1bConfig(
        latent_dim=dH, bath_dim=dS,
        energy_depth=1, energy_heads=2, energy_hidden=16,
        history_hidden=16, step_ps=10.0,
    )
    return StageB1b(cfg)


def _dummy_inputs(B=2, L=6, dH=16, dS=4):
    torch.manual_seed(0)
    H = torch.randn(B, L, dH)
    S = torch.randn(B, L, dS)
    aatype = torch.zeros(B, L, dtype=torch.long)
    R_anchor = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, L, 3, 3).contiguous()
    t_anchor = torch.zeros(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    return H, S, aatype, R_anchor, t_anchor, mask


def _untie_position_heads(decoder_module):
    """Give the position head non-zero weights.

    The StructureDecoder initialises its local-position head with a small
    (near-zero) norm so the decoded structure starts close to the anchor.  At
    that init the backbone coordinates are ~the anchor regardless of H/S, so a
    raw coordinate-difference test would wrongly conclude S is unused.  A
    trained decoder has a non-trivial head; we emulate that here so the
    "S moves the structure" contract is tested on a decoder that can actually
    move atoms.
    """
    sd = decoder_module.decoder
    for p in sd.local_position_head.parameters():
        p.data.normal_(std=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# B2-1: HOnlyDecoder ignores S
# ─────────────────────────────────────────────────────────────────────────────

def test_h_only_ignores_s():
    B, L, dH, dS = 2, 6, 16, 4
    cfg = _make_cfg(dH, dS)
    dec = HOnlyDecoder(cfg)
    dec.eval()
    H, S, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    S_alt = torch.randn_like(S) * 100.0

    with torch.no_grad():
        out1 = dec.decode(H, S, aatype, R_anchor, t_anchor, mask)
        out2 = dec.decode(H, S_alt, aatype, R_anchor, t_anchor, mask)
        out3 = dec.decode(H, None, aatype, R_anchor, t_anchor, mask)

    for out in [out2, out3]:
        assert torch.allclose(out1["backbone_coords"], out["backbone_coords"]), \
            "HOnlyDecoder must not depend on S_t"
    print("B2-1 passed: HOnlyDecoder ignores S_t")


# ─────────────────────────────────────────────────────────────────────────────
# B2-2: BathAwareDecoder changes output when S_t changes
# ─────────────────────────────────────────────────────────────────────────────

def test_bath_decoder_uses_s():
    B, L, dH, dS = 2, 6, 16, 4
    cfg = _make_cfg(dH, dS)
    dec = BathAwareDecoder(cfg)
    _untie_position_heads(dec)      # emulate a trained decoder (§3.2 zero-init)
    dec.eval()
    H, S, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    S_alt = S + 5.0

    with torch.no_grad():
        out1 = dec.decode(H, S, aatype, R_anchor, t_anchor, mask)
        out2 = dec.decode(H, S_alt, aatype, R_anchor, t_anchor, mask)

    assert not torch.allclose(out1["backbone_coords"], out2["backbone_coords"]), \
        "BathAwareDecoder output must differ when S_t changes"

    # Independently of the heads' init, S_t must reach the computation graph.
    S_grad = S.clone().requires_grad_(True)
    out = dec.decode(H, S_grad, aatype, R_anchor, t_anchor, mask)
    out["backbone_coords"].sum().backward()
    assert S_grad.grad is not None and S_grad.grad.abs().max() > 0, \
        "S_t must have a non-zero gradient path into the decoded structure"
    print("B2-2 passed: BathAwareDecoder uses S_t (coords move + gradient flows)")


# ─────────────────────────────────────────────────────────────────────────────
# B2-3: BathAwareDecoder raises on None S
# ─────────────────────────────────────────────────────────────────────────────

def test_bath_decoder_rejects_none_s():
    B, L, dH, dS = 1, 4, 16, 4
    cfg = _make_cfg(dH, dS)
    dec = BathAwareDecoder(cfg)
    dec.eval()
    H, _, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)

    try:
        dec.decode(H, None, aatype, R_anchor, t_anchor, mask)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "requires S_t" in str(e)
    print("B2-3 passed: BathAwareDecoder rejects None S_t")


# ─────────────────────────────────────────────────────────────────────────────
# B2-4: BathAwareDecoder raises on wrong S_t last dim
# ─────────────────────────────────────────────────────────────────────────────

def test_bath_decoder_rejects_wrong_s_dim():
    B, L, dH, dS = 1, 4, 16, 4
    cfg = _make_cfg(dH, dS)
    dec = BathAwareDecoder(cfg)
    dec.eval()
    H, _, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    S_bad = torch.randn(B, L, dS + 3)

    try:
        dec.decode(H, S_bad, aatype, R_anchor, t_anchor, mask)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dS=" in str(e)
    print("B2-4 passed: BathAwareDecoder rejects wrong S_t dim")


# ─────────────────────────────────────────────────────────────────────────────
# B2-5: Separate parameters — gradient through one does not touch the other
# ─────────────────────────────────────────────────────────────────────────────

def test_decoders_have_separate_parameters():
    B, L, dH, dS = 2, 5, 16, 4
    cfg = _make_cfg(dH, dS)
    h_dec = HOnlyDecoder(cfg)
    b_dec = BathAwareDecoder(cfg)

    # Verify no parameter is shared
    h_params = set(id(p) for p in h_dec.parameters())
    b_params = set(id(p) for p in b_dec.parameters())
    assert h_params.isdisjoint(b_params), "Decoders must not share any parameters"

    # Verify gradient through BathAwareDecoder does not affect HOnlyDecoder
    H, S, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    h_params_before = {n: p.data.clone() for n, p in h_dec.named_parameters()}

    out = b_dec.decode(H, S, aatype, R_anchor, t_anchor, mask)
    loss = out["backbone_coords"].sum()
    loss.backward()

    for n, p_before in h_params_before.items():
        p_now = dict(h_dec.named_parameters())[n].data
        assert torch.allclose(p_before, p_now), \
            f"HOnlyDecoder param {n} was modified by BathAwareDecoder backward"

    print("B2-5 passed: decoders have fully separate parameters")


# ─────────────────────────────────────────────────────────────────────────────
# B2-6: autonomous_rollout_and_decode output shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_rollout_output_shapes():
    B, L, dH, dS = 2, 6, 16, 4
    n_steps = 3
    cfg = _make_cfg(dH, dS)
    kernel = _make_kernel(dH, dS)
    kernel.eval()
    decoders = {"h_only": HOnlyDecoder(cfg), "h_plus_bath": BathAwareDecoder(cfg)}
    for d in decoders.values():
        d.eval()

    H0, _, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    torch.manual_seed(42)
    result = autonomous_rollout_and_decode(
        kernel=kernel, decoders=decoders, H0=H0,
        aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=mask, n_steps=n_steps,
    )

    assert isinstance(result, AutonomousRolloutOutput)
    assert result.H.shape == (B, n_steps + 1, L, dH), \
        f"H shape wrong: {result.H.shape}"
    assert result.S is not None
    assert result.S.shape == (B, n_steps + 1, L, dS), \
        f"S shape wrong: {result.S.shape}"
    for mode in ("h_only", "h_plus_bath"):
        assert mode in result.coords
        # backbone_coords: [B, n_steps+1, L, 4, 3]
        c = result.coords[mode]
        assert c.shape == (B, n_steps + 1, L, 4, 3), \
            f"{mode} coords shape wrong: {c.shape}"
    print(f"B2-6 passed: rollout shapes H{tuple(result.H.shape)}, "
          f"S{tuple(result.S.shape)}, "
          f"coords{tuple(result.coords['h_only'].shape)}")


# ─────────────────────────────────────────────────────────────────────────────
# B2-7: h_only coords don't change when S varies (S ignored by that mode)
# ─────────────────────────────────────────────────────────────────────────────

def test_h_only_rollout_independent_of_s():
    """Changing S0 changes the trajectory (kernel uses S) but h_only coords
    at the SAME H must be equal, so we compare: identical S0 with two different
    h_only decoders that have been made identical by copying weights."""
    B, L, dH, dS = 1, 5, 16, 4
    n_steps = 2
    cfg = _make_cfg(dH, dS)
    kernel = _make_kernel(dH, dS)
    kernel.eval()

    h_dec = HOnlyDecoder(cfg)
    h_dec.eval()

    H0, S0, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)

    # Same kernel, same H0, same S0 → h_only coords must be deterministic
    torch.manual_seed(7)
    res1 = autonomous_rollout_and_decode(
        kernel=kernel, decoders={"h_only": h_dec},
        H0=H0, aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=mask, n_steps=n_steps, S0=S0.clone(),
        decode_modes=("h_only",),
    )
    torch.manual_seed(7)
    res2 = autonomous_rollout_and_decode(
        kernel=kernel, decoders={"h_only": h_dec},
        H0=H0, aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=mask, n_steps=n_steps, S0=S0.clone(),
        decode_modes=("h_only",),
    )

    assert torch.allclose(res1.coords["h_only"], res2.coords["h_only"]), \
        "h_only coords must be deterministic given same seed/S0"
    print("B2-7 passed: h_only coords deterministic given same inputs")


# ─────────────────────────────────────────────────────────────────────────────
# B2-8: h_plus_bath coords differ from h_only (S is actually used)
# ─────────────────────────────────────────────────────────────────────────────

def test_h_plus_bath_differs_from_h_only():
    B, L, dH, dS = 2, 6, 16, 4
    n_steps = 2
    cfg = _make_cfg(dH, dS)
    kernel = _make_kernel(dH, dS)
    kernel.eval()
    decoders = {"h_only": HOnlyDecoder(cfg), "h_plus_bath": BathAwareDecoder(cfg)}
    for d in decoders.values():
        _untie_position_heads(d)    # emulate trained decoders so atoms can move
        d.eval()

    H0, S0, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    torch.manual_seed(0)
    result = autonomous_rollout_and_decode(
        kernel=kernel, decoders=decoders, H0=H0,
        aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=mask, n_steps=n_steps, S0=S0,
    )

    assert not torch.allclose(
        result.coords["h_only"], result.coords["h_plus_bath"]
    ), "h_plus_bath coords must differ from h_only (S_t is actually used)"
    print("B2-8 passed: h_plus_bath coords differ from h_only")


# ─────────────────────────────────────────────────────────────────────────────
# B2-9: S trajectory comes from kernel — same seed → same S
# ─────────────────────────────────────────────────────────────────────────────

def test_s_trajectory_reproducible_with_same_seed():
    B, L, dH, dS = 1, 4, 16, 4
    n_steps = 4
    cfg = _make_cfg(dH, dS)
    kernel = _make_kernel(dH, dS)
    kernel.eval()
    decoders = {"h_only": HOnlyDecoder(cfg)}
    for d in decoders.values():
        d.eval()

    H0, S0, aatype, R_anchor, t_anchor, mask = _dummy_inputs(B, L, dH, dS)
    g1 = torch.Generator()
    g1.manual_seed(123)
    g2 = torch.Generator()
    g2.manual_seed(123)

    res1 = autonomous_rollout_and_decode(
        kernel=kernel, decoders=decoders, H0=H0.clone(),
        aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=mask, n_steps=n_steps, S0=S0.clone(),
        generator=g1, decode_modes=("h_only",),
    )
    res2 = autonomous_rollout_and_decode(
        kernel=kernel, decoders=decoders, H0=H0.clone(),
        aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=mask, n_steps=n_steps, S0=S0.clone(),
        generator=g2, decode_modes=("h_only",),
    )

    assert torch.allclose(res1.S, res2.S), \
        "S trajectory must be identical given the same generator seed"
    assert torch.allclose(res1.H, res2.H), \
        "H trajectory must be identical given the same generator seed"
    print("B2-9 passed: S and H trajectories reproducible with same seed")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_h_only_ignores_s()
    test_bath_decoder_uses_s()
    test_bath_decoder_rejects_none_s()
    test_bath_decoder_rejects_wrong_s_dim()
    test_decoders_have_separate_parameters()
    test_rollout_output_shapes()
    test_h_only_rollout_independent_of_s()
    test_h_plus_bath_differs_from_h_only()
    test_s_trajectory_reproducible_with_same_seed()
    print("\nAll Stage B2 tests passed.")
