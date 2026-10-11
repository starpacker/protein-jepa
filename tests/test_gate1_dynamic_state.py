"""Gate 1 validation tests: static/dynamic separation and coordinate-derived dynamics.

These tests ensure:
1. Static coordinates → zero/constant dynamic features
2. Coordinate changes → non-zero dynamic features
3. Dynamic state is driven by geometry, not timestamps
"""

import torch
import pytest
from protein_jepa_frame.models.dynamic_encoder import DynamicStateEncoder


@pytest.fixture
def simple_coords():
    """Create simple test coordinates: a short helix-like structure."""
    B, T, L = 2, 8, 10
    coords = torch.zeros(B, T, L, 4, 3)

    # Simple helix: CA positions along a spiral
    for i in range(L):
        angle = i * 0.5
        radius = 2.0
        ca_x = radius * torch.cos(torch.tensor(angle))
        ca_y = radius * torch.sin(torch.tensor(angle))
        ca_z = i * 1.5  # rise along z

        coords[:, :, i, 1, :] = torch.tensor([ca_x, ca_y, ca_z])  # CA
        coords[:, :, i, 0, :] = coords[:, :, i, 1, :] + torch.tensor([-0.5, 0.0, -0.5])  # N
        coords[:, :, i, 2, :] = coords[:, :, i, 1, :] + torch.tensor([0.5, 0.0, 0.5])   # C
        coords[:, :, i, 3, :] = coords[:, :, i, 2, :] + torch.tensor([0.0, 0.5, 0.5])   # O

    return coords


def test_static_coordinate_control(simple_coords):
    """Gate 1.3: Static coordinates should produce zero/constant dynamic state."""
    B, T, L = simple_coords.shape[:3]

    # Use identical coordinates across all frames (no motion)
    static_coords = simple_coords[:, 0:1].expand(B, T, L, 4, 3).contiguous()

    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_id = torch.zeros(B, L, dtype=torch.long)

    encoder = DynamicStateEncoder(latent_dim=64, depth=1, num_heads=2, k_nearest=4)
    encoder.eval()

    with torch.no_grad():
        output = encoder(static_coords, residue_mask, residue_index, chain_id)

    dynamic_states = output["dynamic_frame_states"]  # [B, T, D]

    # Dynamic states should be nearly identical across time (no coordinate change)
    # Check that variance across time is much smaller than mean magnitude
    temporal_variance = dynamic_states.var(dim=1).mean()  # variance across T
    mean_magnitude = dynamic_states.abs().mean()

    # With static coords, temporal variance should be << mean magnitude
    # (some variance ok due to position encoding, but should be small)
    assert temporal_variance < 0.1 * mean_magnitude, \
        f"Static coords should have low temporal variance: {temporal_variance:.4f} vs {mean_magnitude:.4f}"

    # More direct test: frame-to-frame differences should be very small
    frame_diffs = (dynamic_states[:, 1:] - dynamic_states[:, :-1]).abs().mean()
    assert frame_diffs < 0.5, \
        f"Static coords should have small frame-to-frame differences: {frame_diffs:.4f}"


def test_moving_coordinate_sensitivity():
    """Gate 1.3: Internal conformational changes should produce non-zero dynamic features.

    Note: pure rigid-body motion (translation/rotation) is intentionally removed by
    Kabsch alignment and does NOT register as dynamics. We therefore use genuine
    *internal deformation* here.
    """
    B, T, L = 2, 8, 10

    # Create coordinates with internal deformation: residues progressively bend/stretch
    coords = torch.zeros(B, T, L, 4, 3)
    for t in range(T):
        for i in range(L):
            # Internal deformation grows over time: residue i shifts by an
            # amount that depends on both i and t (non-rigid)
            bend = t * 0.15 * torch.sin(torch.tensor(i * 0.7))
            stretch = 1.0 + t * 0.05
            ca_pos = torch.tensor([bend, i * 0.5 * stretch, 0.0])
            coords[:, t, i, 1, :] = ca_pos  # CA
            coords[:, t, i, 0, :] = ca_pos + torch.tensor([-0.5, 0.0, -0.5])  # N
            coords[:, t, i, 2, :] = ca_pos + torch.tensor([0.5, 0.0, 0.5])   # C
            coords[:, t, i, 3, :] = coords[:, t, i, 2, :] + torch.tensor([0.0, 0.5, 0.5])  # O

    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_id = torch.zeros(B, L, dtype=torch.long)

    encoder = DynamicStateEncoder(latent_dim=64, depth=1, num_heads=2, k_nearest=4)
    encoder.eval()

    with torch.no_grad():
        output = encoder(coords, residue_mask, residue_index, chain_id)

    dynamic_states = output["dynamic_frame_states"]  # [B, T, D]

    # With moving coords, frame-to-frame differences should be substantial
    frame_diffs = (dynamic_states[:, 1:] - dynamic_states[:, :-1]).abs().mean()
    assert frame_diffs > 0.01, \
        f"Moving coords should have substantial frame-to-frame differences: {frame_diffs:.4f}"

    # Temporal variance should be non-negligible
    temporal_variance = dynamic_states.var(dim=1).mean()
    assert temporal_variance > 1e-4, \
        f"Moving coords should have non-negligible temporal variance: {temporal_variance:.4f}"


def test_static_vs_dynamic_comparison(simple_coords):
    """Compare static vs dynamic coordinates directly."""
    B, T, L = simple_coords.shape[:3]

    # Static: repeat first frame
    static_coords = simple_coords[:, 0:1].expand(B, T, L, 4, 3).contiguous()

    # Dynamic: add progressive displacement
    dynamic_coords = simple_coords.clone()
    for t in range(T):
        dynamic_coords[:, t, :, :, 0] += t * 0.3  # shift along x

    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_id = torch.zeros(B, L, dtype=torch.long)

    encoder = DynamicStateEncoder(latent_dim=64, depth=1, num_heads=2, k_nearest=4)
    encoder.eval()

    with torch.no_grad():
        static_output = encoder(static_coords, residue_mask, residue_index, chain_id)
        dynamic_output = encoder(dynamic_coords, residue_mask, residue_index, chain_id)

    static_states = static_output["dynamic_frame_states"]
    dynamic_states = dynamic_output["dynamic_frame_states"]

    # Static should have lower temporal variance
    static_var = static_states.var(dim=1).mean()
    dynamic_var = dynamic_states.var(dim=1).mean()

    assert dynamic_var > static_var * 2, \
        f"Dynamic coords should have higher temporal variance: dynamic={dynamic_var:.4f} vs static={static_var:.4f}"


def test_displacement_feature_computation(simple_coords):
    """Test that displacement features are computed correctly."""
    from protein_jepa_frame.models.dynamic_encoder import _ca_displacement_features

    B, T, L = simple_coords.shape[:3]
    ca_coords = simple_coords[:, :, :, 1, :]  # [B, T, L, 3]

    # Create a known displacement: shift by [1, 0, 0] each frame
    ca_coords = torch.zeros(B, T, L, 3)
    for t in range(T):
        ca_coords[:, t, :, 0] = t * 1.0  # shift along x by t

    valid_mask = torch.ones(B, T, L, dtype=torch.bool)

    displacements = _ca_displacement_features(ca_coords, valid_mask)

    # First frame should have zero displacement
    assert displacements[:, 0].abs().max() < 1e-5, "First frame displacement should be zero"

    # Subsequent frames should have displacement magnitude ~1.0 (after alignment)
    # Note: after Kabsch alignment, rigid translation is removed, so displacement
    # should be close to zero for uniform translation
    # Let's test with a non-uniform motion instead
    ca_coords_nonuniform = torch.zeros(B, T, L, 3)
    for t in range(T):
        for i in range(L):
            ca_coords_nonuniform[:, t, i, 0] = t * (1.0 + 0.1 * i)  # differential motion

    displacements_nonuniform = _ca_displacement_features(ca_coords_nonuniform, valid_mask)

    # Should have non-zero displacements (differential motion can't be removed by rigid alignment)
    assert displacements_nonuniform[:, 1:].abs().max() > 0.01, \
        "Non-uniform motion should produce non-zero displacements"


def test_pair_distance_changes():
    """Test that pair distance changes are computed correctly."""
    from protein_jepa_frame.models.dynamic_encoder import _pair_distance_changes

    B, T, L = 2, 4, 6
    ca_coords = torch.randn(B, T, L, 3)

    # Static coords: all frames identical
    ca_static = ca_coords[:, 0:1].expand(B, T, L, 3).contiguous()
    dist_changes_static = _pair_distance_changes(ca_static, k_nearest=4)

    # All distance changes should be zero
    assert dist_changes_static[:, 1:].abs().max() < 1e-5, \
        "Static coords should have zero distance changes"

    # Dynamic coords: expand/contract
    ca_dynamic = ca_coords.clone()
    for t in range(T):
        scale = 1.0 + t * 0.1
        ca_dynamic[:, t] = ca_dynamic[:, t] * scale

    dist_changes_dynamic = _pair_distance_changes(ca_dynamic, k_nearest=4)

    # Should have non-zero distance changes
    assert dist_changes_dynamic[:, 1:].abs().max() > 0.01, \
        "Expanding coords should have non-zero distance changes"


def test_forward_shape_and_validity(simple_coords):
    """Test basic forward pass shape and validity."""
    B, T, L = simple_coords.shape[:3]

    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_id = torch.zeros(B, L, dtype=torch.long)

    latent_dim = 64
    encoder = DynamicStateEncoder(latent_dim=latent_dim, depth=2, num_heads=2, k_nearest=4)

    output = encoder(simple_coords, residue_mask, residue_index, chain_id)

    assert "dynamic_frame_states" in output
    assert "dynamic_clip_state" in output

    assert output["dynamic_frame_states"].shape == (B, T, latent_dim)
    assert output["dynamic_clip_state"].shape == (B, latent_dim)

    # Check for NaN/Inf
    assert not torch.isnan(output["dynamic_frame_states"]).any()
    assert not torch.isinf(output["dynamic_frame_states"]).any()
    assert not torch.isnan(output["dynamic_clip_state"]).any()
    assert not torch.isinf(output["dynamic_clip_state"]).any()


def test_with_static_frame_latents(simple_coords):
    """Test combining with static frame latents (residual connection)."""
    B, T, L = simple_coords.shape[:3]
    latent_dim = 64

    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_id = torch.zeros(B, L, dtype=torch.long)

    static_latents = torch.randn(B, T, latent_dim)

    encoder = DynamicStateEncoder(latent_dim=latent_dim, depth=1, num_heads=2, k_nearest=4)
    encoder.eval()

    with torch.no_grad():
        output = encoder(simple_coords, residue_mask, residue_index, chain_id, static_frame_latents=static_latents)

    assert output["dynamic_frame_states"].shape == (B, T, latent_dim)

    # Output should incorporate static latents (different from no-static case)
    output_no_static = encoder(simple_coords, residue_mask, residue_index, chain_id, static_frame_latents=None)

    diff = (output["dynamic_frame_states"] - output_no_static["dynamic_frame_states"]).abs().mean()
    assert diff > 0.1, "Adding static latents should change output"


def test_timestamp_shuffle_control(simple_coords):
    """Gate 1.3: dt (timestamp) conditioning must not fabricate dynamics from static coords.

    With static coordinates, varying dt should still yield ~constant dynamic state
    (no motion means no dynamics, regardless of physical time labels).
    """
    B, T, L = simple_coords.shape[:3]
    static_coords = simple_coords[:, 0:1].expand(B, T, L, 4, 3).contiguous()

    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L)
    chain_id = torch.zeros(B, L, dtype=torch.long)

    encoder = DynamicStateEncoder(latent_dim=64, depth=1, num_heads=2, k_nearest=4)
    encoder.eval()

    dt_a = torch.ones(B, T)
    dt_b = torch.linspace(0.1, 5.0, T).unsqueeze(0).expand(B, T)  # shuffled/varied dt

    with torch.no_grad():
        out_a = encoder(static_coords, residue_mask, residue_index, chain_id, dt=dt_a)
        out_b = encoder(static_coords, residue_mask, residue_index, chain_id, dt=dt_b)

    # Static coords → both should be near-constant across time
    var_a = out_a["dynamic_frame_states"].var(dim=1).mean()
    var_b = out_b["dynamic_frame_states"].var(dim=1).mean()
    mag = out_a["dynamic_frame_states"].abs().mean()

    assert var_a < 0.1 * mag, f"dt=uniform static temporal variance too high: {var_a:.4f}"
    assert var_b < 0.1 * mag, f"dt=varied static temporal variance too high: {var_b:.4f}"


def test_stage_a_integration_with_dynamic_encoder():
    """Gate 1: dynamic encoder integrates into Stage A and receives gradients."""
    from protein_jepa_frame.config import ModelConfig
    from protein_jepa_frame.models.stage_a.model import StageAJEPA

    cfg = ModelConfig(
        latent_dim=64, encoder_depth=1, predictor_depth=1, temporal_depth=1,
        state_predictor_depth=1, num_heads=2, max_frames=16,
        use_dynamic_encoder=True, dynamic_encoder_depth=1, dynamic_k_nearest=4,
        dynamic_fusion="gate",
    )
    model = StageAJEPA(cfg)
    model.train()
    assert model.dynamic_encoder is not None
    assert model.target_dynamic_encoder is not None

    # Target encoder must be frozen
    for p in model.target_dynamic_encoder.parameters():
        assert not p.requires_grad

    B, T, L = 2, 6, 10
    coords = torch.randn(B, T, L, 4, 3)
    residue_mask = torch.ones(B, T, L, dtype=torch.bool)
    aatype = torch.zeros(B, L, dtype=torch.long)
    residue_index = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    chain_id = torch.zeros(B, L, dtype=torch.long)
    batch = dict(
        coords=coords, residue_mask=residue_mask, aatype=aatype,
        residue_index=residue_index, chain_id=chain_id,
    )
    ctx_mask = residue_mask.clone()
    ctx_mask[:, :, :3] = False
    tgt_mask = residue_mask.clone()

    out = model.forward_mask_prediction(batch, ctx_mask, tgt_mask)
    loss = out["frame_states"].sum() + out["predictions"].sum()
    loss.backward()

    grad_sum = sum(
        p.grad.abs().sum().item()
        for p in model.dynamic_encoder.parameters()
        if p.grad is not None
    )
    assert grad_sum > 0, "Dynamic encoder should receive gradients"

    # Target encoder must NOT receive gradients
    target_grads = [p.grad for p in model.target_dynamic_encoder.parameters() if p.grad is not None]
    assert len(target_grads) == 0, "Frozen target dynamic encoder must not receive gradients"


def test_stage_a_ema_sync_dynamic_encoder():
    """Gate 1: EMA sync and update propagate to target dynamic encoder."""
    from protein_jepa_frame.config import ModelConfig
    from protein_jepa_frame.models.stage_a.model import StageAJEPA

    cfg = ModelConfig(
        latent_dim=32, encoder_depth=1, predictor_depth=1, temporal_depth=1,
        state_predictor_depth=1, num_heads=2, max_frames=16,
        use_dynamic_encoder=True, dynamic_encoder_depth=1, dynamic_k_nearest=4,
    )
    model = StageAJEPA(cfg)

    # Perturb online encoder so it differs from target
    with torch.no_grad():
        for p in model.dynamic_encoder.parameters():
            p.add_(torch.randn_like(p))

    # sync should make them identical
    model.sync_target_encoders()
    for po, pt in zip(model.dynamic_encoder.parameters(), model.target_dynamic_encoder.parameters()):
        assert torch.allclose(po, pt), "sync should copy online -> target"

    # perturb and EMA update
    with torch.no_grad():
        for p in model.dynamic_encoder.parameters():
            p.add_(torch.randn_like(p))
    before = [pt.clone() for pt in model.target_dynamic_encoder.parameters()]
    model.update_target_encoder(momentum=0.9)
    moved = any(
        not torch.allclose(b, pt)
        for b, pt in zip(before, model.target_dynamic_encoder.parameters())
    )
    assert moved, "EMA update should move target dynamic encoder toward online"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
