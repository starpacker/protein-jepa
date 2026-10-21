"""Regression gates for real information flow and calibrated H coordinates."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.stage_b1a.model import freeze_encoder_partial
from protein_jepa_frame.training.train_stage_b1a import _encode_frame, parse_args
from protein_jepa_frame.utils.contracts import validate_representation_contract


def small_model(**kwargs):
    return StageAV2(StageAV2Config(latent_dim=16, num_factors=4, num_heads=4,
                                  encoder_depth=1, predictor_depth=1,
                                  predictor_spatial_depth=1, **kwargs))


def frame_batch():
    torch.manual_seed(7)
    return dict(coords=torch.randn(1, 1, 8, 4, 3),
                residue_mask=torch.ones(1, 1, 8, dtype=torch.bool),
                aatype=torch.arange(8)[None], residue_index=torch.arange(8)[None],
                chain_id=torch.zeros(1, 8, dtype=torch.long), temperature_k=torch.tensor([320.]))


def test_masked_queries_keep_identity_and_reject_hidden_geometry():
    model = small_model().eval()
    batch = frame_batch()
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[:, [2, 4, 6]] = True
    first = model.forward_spatial(batch, spatial_mask=mask)["pred_factors"]
    assert (first - first[:1]).abs().max() > 1e-4
    perturbed = dict(batch, coords=batch["coords"].clone())
    perturbed["coords"][:, :, mask[0]] += 1000 * torch.randn(1, 1, 3, 4, 3)
    second = model.forward_spatial(perturbed, spatial_mask=mask)["pred_factors"]
    torch.testing.assert_close(first, second)


def test_temporal_encoder_is_deterministic_and_differentiable_in_train_mode():
    model = small_model(dropout=0.5).train()
    batch = frame_batch()
    kwargs = {k: batch[k] for k in ("coords", "residue_mask", "aatype", "residue_index", "chain_id")}
    kwargs["coords"] = kwargs["coords"].expand(1, 4, 8, 4, 3).contiguous()
    kwargs["residue_mask"] = kwargs["residue_mask"].expand(1, 4, 8).contiguous()
    h = model.encode_temporal(**kwargs)
    torch.testing.assert_close(h[:, 1:], h[:, :-1], atol=1e-6, rtol=0)
    assert model.context_encoder.training
    (h * torch.randn_like(h)).sum().backward()
    assert model.context_encoder.geom_proj.weight.grad.abs().sum() > 0


def test_default_opf_is_fixed_and_qr_mode_retains_good_condition():
    model = small_model()
    assert not any(p.requires_grad for p in model.opf.parameters())
    assert model.opf_diagnostics()["opf/condition"] == 1
    learned = small_model(opf_mode="qr_retraction")
    with torch.no_grad():
        learned.opf.raw_basis.add_(torch.randn_like(learned.opf.raw_basis) * 0.2)
    learned.after_optimizer_step()
    assert learned.opf_diagnostics()["opf/condition"] < 1.00001
    with pytest.raises(ValueError, match="delta_factors"):
        StageAV2Config(delta_target=False)


def test_predictor_rejects_missing_or_invalid_physical_conditions():
    model = small_model().eval()
    h = torch.randn(1, 4, 8, 16)
    with pytest.raises(ValueError, match="finite"):
        model.predictor(h, torch.tensor([1000.]), torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="strictly positive"):
        model.predictor(h, torch.tensor([0.]), torch.tensor([320.]))
    batch = frame_batch()
    batch.pop("temperature_k")
    with pytest.raises(ValueError, match="explicit temperature"):
        model.forward_spatial(batch)


def test_oracle_default_and_explicit_predicted_adaptation_rejects_unfreezing():
    assert not parse_args(["--stage-a-ckpt", "dummy"]).use_predicted_h
    with pytest.raises(SystemExit):
        parse_args(["--stage-a-ckpt", "dummy", "--mode", "predicted_adaptation",
                    "--unfreeze-top-n-blocks", "1"])


def test_calibration_encoding_keeps_unfrozen_block_gradients():
    encoder = small_model(dropout=0.5).context_encoder
    params = freeze_encoder_partial(encoder, 1)
    batch = frame_batch()
    h = _encode_frame(encoder, batch["coords"][:, 0], batch["residue_mask"][:, 0],
                      batch["aatype"], batch["residue_index"], batch["chain_id"])
    (h * torch.randn_like(h)).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in params if p.grad is not None) > 0


def test_normalizer_refit_preserves_probe_raw_h_mapping():
    oracle = StageBOracleDecoder(StageBOracleConfig(latent_dim=16, aux_dim=8,
                                decoder_c_hidden=16, decoder_n_ipa_blocks=1))
    oracle.h_normalizer.fit(torch.randn(20, 16) * 2 + 1)
    # Nonzero heads test the full affine compensation, not a zero-output shortcut.
    torch.nn.init.normal_(oracle.probe.torsion_head[-1].weight)
    torch.nn.init.normal_(oracle.probe.dist_head[-1].weight)
    h, pair, valid = torch.randn(1, 8, 16), torch.tensor([[[0, 2], [3, 5]]]), torch.ones(1, 8, dtype=torch.bool)
    before = oracle.probe_future(h, pair, valid)
    oracle.refit_normalizer(torch.randn(30, 16) * 4 - 3)
    after = oracle.probe_future(h, pair, valid)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=1e-5, rtol=1e-5)


def test_cpu_cli_handoff_has_best_checkpoint_and_valid_contract(tmp_path: Path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    commands = [
        ["protein_jepa_frame.training.train_stage_a_v2", "--output", str(a_dir),
         "--latent-dim", "16", "--num-factors", "4", "--encoder-depth", "1",
         "--predictor-depth", "1", "--predictor-spatial-depth", "1",
         "--temporal-lags", "0", "1", "--struct-recon-dim", "16", "--struct-recon-pairs", "12"],
        ["protein_jepa_frame.training.train_stage_b1a", "--output", str(b_dir),
         "--stage-a-ckpt", str(a_dir / "checkpoint_latest.pt"),
         "--decoder-hidden", "16", "--decoder-blocks", "1", "--num-distance-pairs", "4",
         "--unfreeze-top-n-blocks", "1", "--fit-normalizer-batches", "1"],
    ]
    common = ["--synthetic", "--device", "cpu", "--steps", "2", "--warmup-steps", "1",
              "--batch-size", "1", "--num-workers", "0", "--history-frames", "4",
              "--paired-horizons-frames", "4", "8", "--max-train-samples", "2",
              "--val-every", "2", "--val-batches", "1", "--save-every", "2", "--log-every", "1"]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", *command, *common],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (a_dir / "checkpoint_best.pt").is_file()
    assert (b_dir / "checkpoint_best.pt").is_file()
    ck = torch.load(b_dir / "checkpoint_latest.pt", weights_only=False)
    oracle = StageBOracleDecoder(StageBOracleConfig(**ck["stage_b1a_config"]))
    oracle.load_state_dict(ck["oracle"])
    validate_representation_contract(ck["representation_contract"], encoder_state=ck["encoder"],
        encoder_config=ck["stage_a_v2_config"], normalizer_state=oracle.h_normalizer.state_dict(),
        decoder_state=oracle.decoder.state_dict(), decoder_config=ck["stage_b1a_config"])
    assert ck["representation_contract"]["calibration_mode"] == "oracle"
    assert ck["normalizer_fitted_split"] == "train"
    stage_a = torch.load(a_dir / "checkpoint_latest.pt", weights_only=False)
    initial = {k.removeprefix("context_encoder."): v for k, v in stage_a["model"].items()
               if k.startswith("context_encoder.")}
    assert any(not torch.equal(value, initial[key]) for key, value in ck["encoder"].items())
