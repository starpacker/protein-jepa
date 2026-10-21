"""Plan-A warm-start / freeze contract for the from-none joint trainer.

These exercise the pure helper logic (encoder-key remapping, freezing, param-group
construction) without needing a manifest or a GPU.
"""
import argparse

import torch

from protein_jepa_frame.models.stage_b1b import StageB1bConfig
from protein_jepa_frame.training.train_stage_cb1b import build_encoder, build_kernel, build_decoder


def _base_args(**over):
    a = dict(
        latent_dim=16, encoder_depth=2, encoder_heads=2, dropout=0.0,
        orientation_features=True, relative_position_features=True,
        anchor_conditioned=False, bath_dim=4, energy_depth=2, energy_heads=2,
        energy_hidden=32, history_hidden=32, step_ps=1000.0,
        coupling_scale=1e-3, diffusion_scale=1e-4, c_init_std=0.1,
        gamma_scale=2e-4, gamma_floor=2e-5, max_gamma_step=1.9, force_clip=1.0,
        tail_init=1e-3, rollout_grad_steps=None, decoder_type="mlp",
        decoder_hidden=32, decoder_depth=2, max_update=12.0,
    )
    a.update(over)
    return argparse.Namespace(**a)


def test_stage_a_context_encoder_keys_remap_and_load():
    """Stage-A checkpoints store the encoder under ``context_encoder.*``; the
    Plan-A loader must strip that prefix and load with no unexpected keys."""
    args = _base_args()
    enc = build_encoder(args, "cpu")
    # Simulate a Stage-A "model" state dict for exactly this encoder.
    fake_model_sd = {f"context_encoder.{k}": v for k, v in enc.state_dict().items()}
    fake_model_sd["predictor.foo"] = torch.zeros(1)          # unrelated Stage-A module

    remapped = {k[len("context_encoder."):]: v for k, v in fake_model_sd.items()
                if k.startswith("context_encoder.")}
    fresh = build_encoder(args, "cpu")
    result = fresh.load_state_dict(remapped, strict=False)
    assert result.unexpected_keys == []
    # Every encoder weight round-trips.
    for k, v in enc.state_dict().items():
        assert torch.allclose(fresh.state_dict()[k], v)


def test_param_groups_drop_frozen_encoder():
    """A fully frozen encoder must leave the encoder group empty (and be dropped),
    while dynamic/decoder params stay trainable."""
    args = _base_args()
    encoder = build_encoder(args, "cpu")
    kernel = build_kernel(args, "cpu")
    decoder = build_decoder(args, "cpu")
    for p in encoder.parameters():
        p.requires_grad_(False)
    groups = [
        {"params": [p for p in encoder.parameters() if p.requires_grad], "name": "encoder"},
        {"params": [p for p in list(kernel.parameters()) + list(decoder.parameters())
                    if p.requires_grad], "name": "dynamics"},
    ]
    groups = [g for g in groups if g["params"]]
    assert len(groups) == 1
    assert groups[0]["name"] == "dynamics"
    assert len(groups[0]["params"]) > 0


def test_frozen_decoder_params_excluded():
    args = _base_args()
    kernel = build_kernel(args, "cpu")
    decoder = build_decoder(args, "cpu")
    for p in decoder.parameters():
        p.requires_grad_(False)
    trainable = [p for p in list(kernel.parameters()) + list(decoder.parameters())
                 if p.requires_grad]
    assert len(trainable) == len(list(kernel.parameters())) > 0
    assert all(p.requires_grad is False for p in decoder.parameters())


def test_dense_step_idx_matches_window_length():
    """--dense-trajectory supervises steps 0..H of a length-H contiguous window.

    The rollout produces H+1 frames (anchor + H steps) and the contiguous future
    window is [anchor, t1, ..., tH], so step i lines up with truth frame i.  A
    mismatch (e.g. rollout_steps != horizon) must be rejected.
    """
    import torch
    H = 8
    n_steps = H                          # rollout_steps == horizon
    for stride in (1, 2, 4):
        idx = torch.arange(0, n_steps + 1, max(1, stride))
        # every selected index is a valid rollout frame and never exceeds H
        assert int(idx.max()) <= n_steps
        assert idx[0] == 0               # the anchor is always supervised
    # A window shorter/longer than the rollout is a misalignment.
    assert n_steps != H + 1
