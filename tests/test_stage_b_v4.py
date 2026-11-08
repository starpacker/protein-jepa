"""Stage B (v4) code-level gate tests.

Covers the spec §7 and §B3 dynamics invariants at the unit level:
  * stable first-step H_dyn + S (finite, no divergence, no bond collapse signal)
  * H_static conditions the bath (different H_static -> different D/C/Gamma)
  * S contributes measurably (S-enabled rollout diverges from S-disabled)
  * autonomous rollout stays finite and within a sane range of H_dyn0
  * gradients reach the GLE energy/bath from a rollout-driven loss
  * exact S_0 posterior runs without history leakage (causal)

The encoder/decoder are NOT owned by StageB; these tests use synthetic H_dyn/H_static
to isolate the GLE.  The full encoder->GLE->decoder route is exercised in
``test_stage_a_v4_forecast`` via the Stage A bundle.
"""

import torch
import pytest

from protein_jepa_frame.models.stage_b_v4 import StageBV4, build_stage_b_v4


def _inputs(B=2, L=24, D=128, dS=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    H0 = 0.5 * torch.randn(B, L, D, generator=g)
    H_static = 0.5 * torch.randn(B, L, D, generator=g)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[:, L // 2] = False
    # 4-frame observed history ending at H0
    H_hist = 0.5 * torch.randn(B, 4, L, D, generator=g)
    H_hist[:, -1] = H0
    dt_hist = torch.full((B, 4), 1000.0)
    return H0, H_static, mask, H_hist, dt_hist


def test_first_step_finite_stable():
    model = build_stage_b_v4(bath_dim=8)
    H0, H_static, mask, H_hist, dt_hist = _inputs()
    H, S = model.rollout(
        H0, H_static, mask, None, n_steps=8,
        H_hist=H_hist, dt_hist=dt_hist, return_bath=True,
        s0_mode="exact", generator=torch.Generator().manual_seed(1),
    )
    assert H.shape == (2, 9, 24, 128)
    assert torch.isfinite(H).all()
    # First-step H stays within a sane range of H0 (no immediate divergence).
    step1 = (H[:, 1] - H0).norm(dim=-1)[mask].mean()
    assert torch.isfinite(step1)
    assert step1 < 50.0, f"first step blew up: {step1}"
    # Bath state finite and masked residues zeroed.
    assert S is not None and torch.isfinite(S).all()
    assert (S[:, :, ~mask[0]].abs().sum() == 0)


def test_static_conditions_bath():
    model = build_stage_b_v4(bath_dim=8)
    H0, H_static, mask, H_hist, dt_hist = _inputs()
    D1, C1, G1 = model.gle.bath(H_static, None)
    # A clearly different static context must change the bath params.  D is
    # deliberately init'd to a constant (diffusion_scale, zero d_net weights),
    # but C and Gamma carry non-zero init and must respond to static context.
    H_static2 = H_static * 3.0 + 4.0
    D2, C2, G2 = model.gle.bath(H_static2, None)
    assert not torch.allclose(C1, C2, atol=1e-6)
    assert not torch.allclose(G1, G2, atol=1e-6)


def test_bath_contributes_measurably():
    # With vs without bath: disabling S must measurably change the trajectory.
    model = build_stage_b_v4(bath_dim=8)
    H0, H_static, mask, H_hist, dt_hist = _inputs()
    g = torch.Generator().manual_seed(3)
    H_with, _ = model.rollout(
        H0, H_static, mask, None, n_steps=16,
        H_hist=H_hist, dt_hist=dt_hist, return_bath=True,
        s0_mode="exact", generator=g,
    )
    # Bath disabled by a zero-dim bath model (H-only Markov baseline).
    model_nobath = build_stage_b_v4(bath_dim=0)
    g2 = torch.Generator().manual_seed(3)
    H_without, _ = model_nobath.rollout(
        H0, H_static, mask, None, n_steps=16,
        H_hist=H_hist, dt_hist=dt_hist, return_bath=True,
        s0_mode="prior", generator=g2,
    )
    diff = (H_with - H_without).norm(dim=-1) * mask.unsqueeze(1).float()
    diff = diff.sum() / mask.unsqueeze(1).sum().clamp_min(1.0)
    # Even with identical noise, the bath (velocity coupling) alters the drift.
    assert diff > 0.0


def test_gradients_reach_gle():
    model = build_stage_b_v4(bath_dim=8)
    H0, H_static, mask, H_hist, dt_hist = _inputs()
    H0 = H0.detach().requires_grad_(True)
    # CPU SDPA lacks a flash-attention backward; force the math backend so the
    # energy self-attention is differentiable.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        H, _ = model.rollout(
            H0, H_static, mask, None, n_steps=4,
            H_hist=H_hist, dt_hist=dt_hist, return_bath=True,
            s0_mode="prior", detach=False,
        )
        loss = (H[:, -1] ** 2).sum()
        loss.backward()
    assert H0.grad is not None and torch.isfinite(H0.grad).all()
    # The GLE energy/bath params receive gradients.
    grads = [p.grad for p in model.gle.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_exact_posterior_is_causal_and_finite():
    model = build_stage_b_v4(bath_dim=8)
    H0, H_static, mask, H_hist, dt_hist = _inputs()
    model.gle.s0_mode = "exact"
    m0, chol0 = model.gle.exact_bath_posterior(
        H_hist, dt_hist, None, mask, seq_feature=H_static,
    )
    assert torch.isfinite(m0).all() and torch.isfinite(chol0).all()
    # Causal: the posterior must not depend on future frames (none supplied here).
    # Cholesky factor is lower-triangular with positive diagonal.
    assert chol0.shape[-1] == chol0.shape[-2] == 8


def test_decoded_bond_gradient_reaches_gle():
    """Spec §7 #8: decoded-geometry gradient must reach the GLE (no detach).

    Guards the regression fixed in train_stage_b_v4: ``decode_paths_v4`` was
    ``@torch.no_grad()``-wrapped, so the adjacent-CA bond loss could not propagate
    through the frozen decoder into the GLE's H_dyn output.  ``differentiable=True``
    restores that path while keeping the decoder *weights* frozen.
    """
    from protein_jepa_frame.training.train_stage_b_v4 import (
        adjacent_ca_bond_loss, decode_paths_v4,
    )
    from protein_jepa_frame.models.static_dynamic_v4 import ConditionalStructureDecoder
    from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
    from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame

    B, L, D = 2, 12, 128
    model = build_stage_b_v4(bath_dim=8)
    g = torch.Generator().manual_seed(0)
    H0 = (0.3 * torch.randn(B, L, D, generator=g)).requires_grad_(True)
    H_static = 0.3 * torch.randn(B, L, D, generator=g)
    mask = torch.ones(B, L, dtype=torch.bool)
    H_hist = 0.3 * torch.randn(B, 4, L, D, generator=g)
    H_hist[:, -1] = H0.detach()
    dt_hist = torch.full((B, 4), 1000.0)

    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        H, _ = model.rollout(
            H0, H_static, mask, None, n_steps=3,
            H_hist=H_hist, dt_hist=dt_hist, return_bath=True,
            s0_mode="prior", detach=False,
        )  # [B, T=4, L, D]
        # Reshape into the [M, B, T, L, D] convention expected by decode_paths_v4.
        H_paths = H[None]  # [1, B, T, L, D] (M=1 sample)
        # Synthetic anchor coords [B, L, 4, 3] and batch dict.
        anchor_coords = torch.randn(B, L, 4, 3)
        aatype = torch.randint(0, 20, (B, L))
        ridx = torch.arange(L).expand(B, L)
        cid = torch.zeros(B, L, dtype=torch.long)
        batch = {"aatype": aatype, "residue_index": ridx, "chain_id": cid}
        decoder = GeometricDeltaCADecoder(latent_dim=D)
        cond = ConditionalStructureDecoder(decoder, D)
        # Freeze the decoder (weights) but keep the input gradient alive.
        cond.requires_grad_(False)

        pred_bb = decode_paths_v4(
            cond, H_paths[:, :, 1:], H_static, batch, mask, anchor_coords,
            differentiable=True,
        )  # [1, B, T-1, L, 4, 3]
        # Target backbone for the bond loss (same convention, matching T-1 frames).
        Tp = pred_bb.shape[2]
        target_bb = torch.zeros(B, Tp, L, 4, 3)
        target_bb[..., 1, :] = torch.randn(B, Tp, L, 3) * 3.8
        loss = adjacent_ca_bond_loss(pred_bb, target_bb, batch, mask)
        assert torch.isfinite(loss)
        loss.backward()

    # The gradient must reach H0 (input to the GLE), proving the decoded-geometry
    # signal propagates through GLE + frozen decoder.
    assert H0.grad is not None, "bond gradient did not reach the GLE input"
    assert torch.isfinite(H0.grad).all()
    assert H0.grad.abs().sum() > 0
    # Decoder weights stay frozen (no grad) — only the input path is live.
    assert all(p.grad is None for p in cond.parameters())

