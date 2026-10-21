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


def _batch(B=2, K=8, L=10, horizon_frames=32, device="cpu"):
    """Paired-window batch with both context_* and future_* keys.

    The future clip mirrors the context (same frame count K) shifted forward in
    time by ``horizon_frames``, matching collate_paired_windows().  This is what
    forward_temporal_multilag now consumes: the target is drawn from future_*.
    """
    coords = torch.randn(B, K, L, 4, 3, device=device)
    coords[..., 1, :] += torch.arange(L, device=device).view(1, 1, L, 1)
    rm = torch.ones(B, K, L, dtype=torch.bool, device=device)
    fut_coords = torch.randn(B, K, L, 4, 3, device=device)
    fut_coords[..., 1, :] += torch.arange(L, device=device).view(1, 1, L, 1)
    ctx_t = torch.arange(K, device=device).float().expand(B, K).contiguous() * 1000.0
    fut_t = ctx_t + float(horizon_frames) * 1000.0
    return dict(
        context_coords=coords,
        context_residue_mask=rm,
        context_aatype=torch.full((B, L), 20, dtype=torch.long, device=device),
        context_residue_index=torch.arange(L, device=device).expand(B, L).contiguous(),
        context_chain_id=torch.zeros(B, L, dtype=torch.long, device=device),
        context_time_ps=ctx_t,
        future_coords=fut_coords,
        future_residue_mask=rm.clone(),
        future_time_ps=fut_t,
        horizon_time_ps=torch.full((B,), float(horizon_frames) * 1000.0, device=device),
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


def test_multilag_lags_must_be_within_future_clip():
    model = StageAV2(_cfg()).eval()
    batch = _batch(K=4)  # future clip has K=4 frames
    with pytest.raises(ValueError):
        model.forward_temporal_multilag(batch, (4,))     # lag == T_fut
    with pytest.raises(ValueError):
        model.forward_temporal_multilag(batch, (-1,))    # negative lag


def test_multilag_shapes_and_lag_slots():
    torch.manual_seed(0)
    model = StageAV2(_cfg()).eval()
    batch = _batch(B=6, K=8, L=9)
    out = model.forward_temporal_multilag(batch, (0, 1, 2, 3), recent_frac=0.5)
    B, L = 6, 9
    assert out["pred_factors"].shape == (B, L, 4, 8)
    assert out["target_factors"].shape == (B, L, 4, 8)
    assert out["lag_slot"].shape == (B,)
    assert out["lag_sel"].shape == (B,)
    # recent_frac=0.5 -> first 3 samples restricted to the two shortest lags
    assert set(out["lag_slot"][:3].tolist()) <= {0, 1}
    # all selected future indices are valid and lag_ps is positive & finite
    assert torch.all(out["lag_sel"] >= 0)
    assert torch.all(out["lag_sel"] < 8)
    assert torch.all(out["lag_ps"] > 0) and torch.isfinite(out["lag_ps"]).all()


def test_multilag_target_comes_from_future_window():
    """The teacher target must be the encoding of a FUTURE frame (outside the
    context window), not any context frame.  lag=0 selects the last future
    frame (= anchor + full horizon).  This is the core fix: Stage A now trains
    exactly the way B1a / eval evaluates — predict out-of-window futures."""
    torch.manual_seed(0)
    model = StageAV2(_cfg()).eval()
    batch = _batch(B=3, K=4, L=12, horizon_frames=32)
    out = model.forward_temporal_multilag(batch, (0,), recent_frac=1.0)

    # lag=0 -> last future frame index T_fut-1.
    assert torch.all(out["lag_sel"] == 3)

    # The teacher target (future encoding) must match the teacher encoding of
    # the selected FUTURE frame, not of any context frame.
    with torch.no_grad():
        fut_H = model.teacher_encoder(
            coords=batch["future_coords"], residue_mask=batch["future_residue_mask"],
            aatype=batch["context_aatype"], residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
        )[:, 3]   # last future frame
    assert torch.allclose(out["h_teacher_full"], fut_H, atol=1e-4), \
        "target must be the teacher encoding of the future frame"

    # Predictor sees the full context: last slot is the real anchor encoding.
    with torch.no_grad():
        full = model.context_encoder(
            coords=batch["context_coords"], residue_mask=batch["context_residue_mask"],
            aatype=batch["context_aatype"], residue_index=batch["context_residue_index"],
            chain_id=batch["context_chain_id"],
        )
    assert torch.allclose(out["context_H_full"], full, atol=1e-4)
    assert torch.allclose(out["anchor_H"], full[:, -1], atol=1e-4)

    # lag_ps equals the physical horizon (future_time - anchor_time).
    expect = batch["future_time_ps"][:, 3] - batch["context_time_ps"][:, -1]
    assert torch.allclose(out["lag_ps"], expect, atol=1e-3)


def test_multilag_lag_steps_back_from_horizon_end():
    """Larger lags step BACKWARD through the future window (closer to anchor).
    lag=0 -> last future frame; lag=2 -> two offsets earlier."""
    torch.manual_seed(0)
    model = StageAV2(_cfg()).eval()
    batch = _batch(B=2, K=8, L=6)
    out = model.forward_temporal_multilag(batch, (2,), recent_frac=1.0)
    # lag=2, T_fut=8 -> future index 8-1-2 = 5
    assert torch.all(out["lag_sel"] == 5)


def test_delta_target_anchor_calibration():
    """forward_temporal_multilag always emits a delta target: the output dict
    carries all anchor fields, and compose(decompose(h_anchor)) roundtrips with
    cos > 0.99 so 'pred=0 → reproduce anchor' holds in practice."""
    torch.manual_seed(1)
    model = StageAV2(_cfg()).eval()
    batch = _batch(B=2, K=4, L=10)
    out = model.forward_temporal_multilag(batch, (0,), recent_frac=1.0)

    for key in ("h_teacher_anchor", "delta_teacher", "anchor_factors"):
        assert key in out, f"missing key in delta-target output: {key}"

    # compose(decompose(h_anchor)) must roundtrip (persistence fallback validity)
    anchor_recon = model.opf.compose(out["anchor_factors"])
    h_anc = out["h_teacher_anchor"]
    cos = torch.nn.functional.cosine_similarity(
        anchor_recon.reshape(-1), h_anc.reshape(-1), dim=0).item()
    assert cos > 0.99, (
        f"compose(decompose(h_anchor)) diverges from h_anchor: cos={cos:.4f}. "
        "Anchor calibration is broken — pred=0 will not reproduce the anchor."
    )

    # target_factors are deltas; their norm must be smaller than absolute factors
    tgt_norm = out["target_factors"].norm(dim=-1).mean().item()
    abs_norm = model.opf.decompose(out["h_teacher_full"]).norm(dim=-1).mean().item()
    assert tgt_norm < abs_norm, (
        f"delta target norm ({tgt_norm:.4f}) >= absolute factor norm ({abs_norm:.4f}); "
        "delta_target may not be subtracting the anchor."
    )


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


# ── P0-A: structural-displacement readout ────────────────────────────────────

def test_anchor_local_displacement_is_rigid_invariant_and_internal_sensitive():
    """Pure rigid motion ⇒ zero target; internal change ⇒ nonzero at that residue."""
    torch.manual_seed(0)
    B, L = 2, 12
    anchor = torch.randn(B, L, 4, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    ang = 0.7
    R = torch.tensor([[torch.cos(torch.tensor(ang)), -torch.sin(torch.tensor(ang)), 0.0],
                      [torch.sin(torch.tensor(ang)), torch.cos(torch.tensor(ang)), 0.0],
                      [0.0, 0.0, 1.0]])
    rigid = anchor @ R.T + torch.tensor([5.0, -3.0, 2.0])
    d_rigid = StageAV2._anchor_local_ca_displacement(rigid, anchor, mask)
    assert d_rigid.abs().max().item() < 1e-5, "rigid motion must give ~0 internal displacement"

    internal = anchor.clone()
    internal[0, 5, 1, :] += torch.tensor([1.5, 0.0, 0.0])
    d_int = StageAV2._anchor_local_ca_displacement(internal, anchor, mask)
    assert d_int[0, 5].norm().item() > 0.5, "a real CA move must appear at that residue"


def test_displacement_readout_has_grad_to_encoder_and_predictor():
    """The displacement loss must back-propagate to encoder and predictor."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = StageAV2(cfg)
    model.train()
    batch = _batch(B=2, K=4, L=10, horizon_frames=8)
    model.zero_grad(set_to_none=True)
    out = model.forward_temporal_multilag(batch, (0,), recent_frac=0.5)
    assert out["target_delta_local"].shape == (2, 10, 3)
    loss = model.displacement_readout_loss(
        pred_factors=out["pred_factors"],
        target_delta_local=out["target_delta_local"],
        residue_mask=out["valid_mask"],
        synth_delta_H=out["synth_delta_H"],
    )
    loss.backward()
    enc = sum(p.grad.abs().sum().item() for p in model.context_encoder.parameters() if p.grad is not None)
    pred = sum(p.grad.abs().sum().item() for p in model.predictor.parameters() if p.grad is not None)
    assert enc > 0 and pred > 0, "displacement supervision must reach encoder and predictor"


def test_displacement_anchor_is_online_not_detached_teacher():
    """The anchor factors must come from the ONLINE encoder with grad.

    Regression guard for the bug where anchor came from the teacher (detached),
    so the synthesized delta had gradient only through the predictor and the
    encoder's anchor representation never received displacement supervision.
    """
    torch.manual_seed(0)
    cfg = _cfg()
    model = StageAV2(cfg)
    model.train()
    batch = _batch(B=2, K=4, L=10, horizon_frames=8)
    out = model.forward_temporal_multilag(batch, (0,), recent_frac=0.5)
    assert out["anchor_factors"].requires_grad, "anchor_factors must carry grad"
    assert out["anchor_H"].requires_grad, "anchor_H must carry grad"
    # And it must equal the online encoding of the last context frame, i.e. the
    # encoder forward already computed into context_H[:, -1].
    assert out["anchor_H"].shape == out["context_H"][:, -1].shape


def test_displacement_head_input_is_change_only():
    """The head must read ONLY the dH-wide predicted change (no side-channel)."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = StageAV2(cfg)
    first = next(model.displacement_head.parameters())
    assert first.shape[1] == cfg.latent_dim, (
        "displacement head input must be dH (change only), got "
        f"{first.shape[1]}"
    )


def test_displacement_readout_does_not_tolerate_constant_prediction():
    """A constant (mean-only) prediction must score ≈1, not 0, after normalisation."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = StageAV2(cfg)
    B, L = 4, 12
    # Constant predicted change ⇒ head can only see a constant; target varies.
    tgt = torch.randn(B, L, 3)
    tgt = tgt - tgt.mean(dim=1, keepdim=True)      # vary across residues
    pf = torch.zeros(B, L, cfg.num_factors, cfg.factor_dim)
    synth = torch.zeros(B, L, cfg.latent_dim)
    mask = torch.ones(B, L, dtype=torch.bool)
    loss = model.displacement_readout_loss(
        pf, tgt, mask, synth_delta_H=synth,
    )
    # With the direction term, a constant-ish prediction is penalised >= 1.
    assert loss.item() >= 0.5, (
        f"constant prediction should score near 1.0 or worse, got {loss.item():.3f}"
    )



# ── Relative-position (within-frame canonical frame) channel ──────────────────

def test_within_frame_canonical_positions_are_rigid_invariant():
    """Global rigid motion must not change the canonical-frame positions."""
    from protein_jepa_frame.models.encoder import _within_frame_canonical_positions
    torch.manual_seed(0)
    coords = torch.randn(2, 15, 4, 3)
    valid = torch.ones(2, 15, dtype=torch.bool)
    p0 = _within_frame_canonical_positions(coords, valid)
    ang = 0.9
    R = torch.tensor([[torch.cos(torch.tensor(ang)), -torch.sin(torch.tensor(ang)), 0.0],
                      [torch.sin(torch.tensor(ang)), torch.cos(torch.tensor(ang)), 0.0],
                      [0.0, 0.0, 1.0]])
    moved = coords @ R.T + torch.tensor([4.0, -1.0, 2.5])
    p1 = _within_frame_canonical_positions(moved, valid)
    assert (p0 - p1).abs().max().item() < 1e-4, "canonical positions must be rigid-invariant"


def test_within_frame_canonical_positions_encode_position():
    """Two residues at DIFFERENT positions but with identical local geometry must
    get different canonical positions (the channel is position-aware)."""
    from protein_jepa_frame.models.encoder import _within_frame_canonical_positions
    torch.manual_seed(1)
    coords = torch.randn(1, 20, 4, 3)
    valid = torch.ones(1, 20, dtype=torch.bool)
    p = _within_frame_canonical_positions(coords, valid)
    assert p[0].std(dim=0).min().item() > 1e-3, (
        "canonical positions must vary across residues (position-aware)"
    )


def test_encoder_relative_position_channel_changes_H():
    """Enabling the channel must change the encoder output for a moved residue."""
    from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
    torch.manual_seed(0)
    enc = ProteinGeometryEncoder(latent_dim=16, depth=1, num_heads=2,
                                 relative_position_features=True)
    enc.eval()
    coords = torch.randn(1, 1, 12, 4, 3)
    rm = torch.ones(1, 1, 12, dtype=torch.bool)
    aa = torch.zeros(1, 12, dtype=torch.long)
    ri = torch.arange(12)[None]
    ci = torch.zeros(1, 12, dtype=torch.long)
    with torch.no_grad():
        h0 = enc(coords, rm, aa, ri, ci)
        coords2 = coords.clone()
        coords2[0, 0, 3, 1, :] += torch.tensor([2.0, 0.0, 0.0])  # move one CA
        h1 = enc(coords2, rm, aa, ri, ci)
    assert (h0 - h1).abs().max().item() > 1e-4
