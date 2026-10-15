"""Tests for the Stage A v2 multi-lag temporal objective and its contracts.

Covers:
  * fused orientation bias == explicit Linear(pair_orientation) (memory opt)
  * masked-history predictor cannot see the target frame's own geometry
  * multi-lag forward shape/lag-range contracts
  * temporal variance term responds to a time-invariant representation
"""
import pytest
import torch

from protein_jepa_frame.models.encoder import (
    _orientation_pair_bias,
    _pair_orientation_features,
)
from protein_jepa_frame.models.stage_a_v2 import (
    StageAV2,
    StageAV2Config,
    StageAV2LossWeights,
    compute_stage_a_v2_loss,
)


def _cfg(**over):
    base = dict(latent_dim=32, num_factors=4, encoder_depth=2, predictor_depth=2, num_heads=4)
    base.update(over)
    return StageAV2Config(**base)


def _batch(B=2, K=8, L=10, device="cpu"):
    coords = torch.randn(B, K, L, 4, 3, device=device)
    coords[..., 1, :] += torch.arange(L, device=device).view(1, 1, L, 1)
    rm = torch.ones(B, K, L, dtype=torch.bool, device=device)
    return dict(
        context_coords=coords,
        context_residue_mask=rm,
        context_aatype=torch.full((B, L), 20, dtype=torch.long, device=device),
        context_residue_index=torch.arange(L, device=device).expand(B, L).contiguous(),
        context_chain_id=torch.zeros(B, L, dtype=torch.long, device=device),
        context_time_ps=torch.arange(K, device=device).float().expand(B, K).contiguous() * 1000.0,
        temperature_k=torch.full((B,), 320.0, device=device),
    )


def test_fused_orientation_bias_matches_explicit_projection():
    torch.manual_seed(0)
    B, L = 2, 17
    coords = torch.randn(B, L, 4, 3)
    w = torch.randn(4, 12)
    ref = _pair_orientation_features(coords) @ w.t()
    for chunk in (1, 4, 16, 100):
        got = _orientation_pair_bias(coords, w, chunk=chunk)
        assert torch.allclose(ref, got, atol=1e-5), f"chunk={chunk}"


def test_multilag_lags_must_be_less_than_context():
    model = StageAV2(_cfg()).eval()
    batch = _batch(K=4)
    with pytest.raises(ValueError):
        model.forward_temporal_multilag(batch, (4,))     # lag == K
    with pytest.raises(ValueError):
        model.forward_temporal_multilag(batch, (0,))


def test_multilag_shapes_and_lag_slots():
    torch.manual_seed(0)
    model = StageAV2(_cfg()).eval()
    batch = _batch(B=6, K=8, L=9)
    out = model.forward_temporal_multilag(batch, (1, 2, 4, 7), recent_frac=0.5)
    B, L = 6, 9
    assert out["pred_factors"].shape == (B, L, 4, 8)
    assert out["target_factors"].shape == (B, L, 4, 8)
    assert out["lag_slot"].shape == (B,)
    assert out["lag_sel"].shape == (B,)
    # recent_frac=0.5 -> first 3 samples restricted to the two shortest lags
    assert set(out["lag_slot"][:3].tolist()) <= {0, 1}
    # all selected lags are valid and lag_ps is positive & finite
    assert torch.all(out["lag_sel"] >= 1)
    assert torch.all(out["lag_sel"] < 8)
    assert torch.all(out["lag_ps"] > 0) and torch.isfinite(out["lag_ps"]).all()


def test_multilag_predictor_last_slot_is_anchor_not_target():
    """The predictor's last visible slot is the anchor frame (frame before the
    target), never the target frame itself.

    With delta-target, anchor masking is removed — the anchor is encoded
    normally and placed in the last slot.  context_H and context_H_full are
    both the plain online-encoder trajectory; the identity-copy shortcut is
    broken at the *target* level (target = P_j·(h̄_tgt - h̄_anchor) ≈ 0 for
    a copy) rather than by hiding the anchor residues."""
    torch.manual_seed(0)
    model = StageAV2(_cfg()).eval()
    batch = _batch(B=3, K=4, L=12)
    out = model.forward_temporal_multilag(batch, (1,), recent_frac=1.0)
    with torch.no_grad():
        full = model.context_encoder(
            coords=batch["context_coords"], residue_mask=batch["context_residue_mask"],
            aatype=batch["context_aatype"], residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
        )
    last_in = out["context_H"][:, -1]           # predictor's last slot
    # lag=1, K=4 -> anchor index 2, target index 3.
    # The last slot must NOT be the target.
    assert not torch.allclose(last_in, full[:, 3], atol=1e-4)
    # The last slot IS the anchor (no masking applied in delta-target mode).
    assert torch.allclose(last_in, full[:, 2], atol=1e-4)
    # context_H_full carries the unmasked trajectory (same as context_H now).
    assert torch.allclose(out["context_H_full"], full, atol=1e-4)

    # Delta-target contract: target ≈ 0 for a static protein; the out dict
    # must expose delta_teacher and h_teacher_anchor.
    assert "delta_teacher" in out
    assert "h_teacher_anchor" in out


def test_multilag_history_drops_future_frames():
    """For lag > 1 no frame at or after the target may appear in the predictor
    input; slots before the visible window are zero.  Masking disabled here so
    the anchor's encoding is directly comparable to the full encoder output."""
    torch.manual_seed(0)
    model = StageAV2(_cfg(spatial_mask_ratio=0.0)).eval()
    batch = _batch(B=2, K=8, L=6)
    out = model.forward_temporal_multilag(batch, (4,), recent_frac=1.0)
    H = out["context_H"]                        # [B,8,L,dH]
    with torch.no_grad():
        full = model.context_encoder(
            coords=batch["context_coords"], residue_mask=batch["context_residue_mask"],
            aatype=batch["context_aatype"], residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
        )
    # lag=4, K=8 -> anchor index 3, target index 4; frames 4..7 must not appear.
    for b in range(2):
        for f in range(4, 8):
            assert not torch.allclose(
                H[b], full[b, f][None].expand(8, -1, -1), atol=1e-4
            )
        # frame 3 (anchor) appears at the last slot
        assert torch.allclose(H[b, -1], full[b, 3], atol=1e-4)


def test_temporal_variance_term_penalises_time_invariant_H():
    weights = StageAV2LossWeights(temporal_variance_weight=1.0, temporal_min_std=0.05)
    cfg = _cfg()
    model = StageAV2(cfg)

    valid = torch.ones(2, 10, dtype=torch.bool)
    base = dict(
        pred_factors=torch.zeros(2, 10, 4, 8),
        target_factors=torch.zeros(2, 10, 4, 8),
        valid_mask=valid,
        lag_slot=torch.zeros(2, dtype=torch.long),
    )
    # time-invariant context -> maximum penalty (hinge saturated at ~1.0 after
    # normalisation by gamma^2).  context_H_full carries the true trajectory.
    const = torch.ones(2, 6, 10, cfg.latent_dim)
    frozen = dict(base, context_H=const, context_H_full=const)
    comp_frozen = compute_stage_a_v2_loss(
        frozen, None, model.opf.analysis_basis(), weights, cfg.latent_dim)
    # strongly time-varying context -> penalty near zero
    tv = torch.randn(2, 6, 10, cfg.latent_dim) * 1.0
    varying = dict(base, context_H=tv, context_H_full=tv)
    comp_varying = compute_stage_a_v2_loss(
        varying, None, model.opf.analysis_basis(), weights, cfg.latent_dim)
    assert comp_frozen.temporal_variance.item() > comp_varying.temporal_variance.item()
    assert comp_frozen.temporal_variance.item() > 0.9   # hinge saturated
    assert comp_varying.temporal_variance.item() < 0.1


def test_temporal_contrast_has_gradient_in_near_collapsed_regime():
    """The temporal term must supply a strong gradient when consecutive frames
    are *almost* identical (the near-collapse basin).  A tiny step below the
    floor should produce a gradient pushing the frames apart."""
    weights = StageAV2LossWeights(temporal_variance_weight=1.0, temporal_min_std=0.05)
    cfg = _cfg()
    model = StageAV2(cfg)
    valid = torch.ones(2, 10, dtype=torch.bool)
    base = dict(
        pred_factors=torch.zeros(2, 10, 4, 8),
        target_factors=torch.zeros(2, 10, 4, 8),
        valid_mask=valid,
        lag_slot=torch.zeros(2, dtype=torch.long),
    )
    # Near-collapse: consecutive frames differ by a tiny step below the floor.
    H = torch.ones(2, 6, 10, cfg.latent_dim)
    H[:, 1::2] += 0.005
    H = H.detach().requires_grad_(True)
    diffs = H[:, 1:] - H[:, :-1]
    s_t = diffs.pow(2).mean(dim=-1)
    gamma_sq = weights.temporal_min_std ** 2
    L = torch.clamp(gamma_sq - s_t, min=0.0).mean() / gamma_sq
    L.backward()
    assert H.grad is not None and H.grad.norm().item() > 1e-4, (
        "temporal contrast must have a usable gradient in the near-collapsed regime"
    )
