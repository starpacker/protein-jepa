"""End-to-end contracts for autonomous, history-only Stage B1b training."""
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.stage_a_v2 import StageAV2Config
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.stage_b1b import (
    StageB1b, StageB1bConfig, ResidualCADecoder, ResidualDecoderConfig,
)
from protein_jepa_frame.training.train_stage_b1b import (
    autonomous_samples, checkpoint_payload, forecast_objective, freeze_module,
    load_frozen_representation, parse_args, resume_training, split_contiguous_batch,
    validate_oracle_gate, _loader, EpochPermutation,
)
from protein_jepa_frame.utils.contracts import build_representation_contract


@pytest.fixture
def tiny(tmp_path):
    torch.manual_seed(19)
    ds = build_dataset(DataConfig(num_frames=4, synthetic_size=2, synthetic_residues=6))
    batch = collate_clips([ds[0]])
    cfg = StageAV2Config(latent_dim=8, encoder_depth=1, num_heads=1, num_rbf=4,
                        num_factors=2, dropout=0.0, max_residues=32)
    enc = ProteinGeometryEncoder(
        latent_dim=8, depth=1, num_heads=1, num_rbf=4, dropout=0.0,
        max_residues=32, orientation_features=True, identity_embed_scale=cfg.identity_embed_scale,
    )
    ocfg = StageBOracleConfig(latent_dim=8, aux_dim=8, num_distance_pairs=3,
                             decoder_c_hidden=8, decoder_n_ipa_blocks=1,
                             decoder_n_heads=1, decoder_torsion_bins=8,
                             decoder_anchor_free=False)
    oracle = StageBOracleDecoder(ocfg)
    oracle.h_normalizer.fit(torch.randn(30, 8)*0.7+0.2)
    freeze_module(enc)
    freeze_module(oracle)
    contract = build_representation_contract(
        enc.state_dict(), asdict(cfg), oracle.h_normalizer.state_dict(),
        oracle.decoder.state_dict(), asdict(ocfg), calibration_mode="oracle",
    )
    bundle = {"encoder": enc.state_dict(), "stage_a_v2_config": asdict(cfg),
              "oracle": oracle.state_dict(), "stage_b1a_config": asdict(ocfg),
              "representation_contract": contract}
    path = tmp_path/"b1a.pt"
    torch.save(bundle, path)
    args = parse_args([
        "--b1a-ckpt", str(path), "--device", "cpu", "--history-frames", "2",
        "--horizon-steps", "2", "--step-ps", "1", "--es-num-samples", "2",
        "--physical-es-weight", "0.1", "--rmsd-weight", "0.1",
        "--dry-run",
    ])
    mcfg = StageB1bConfig(
        latent_dim=8, bath_dim=2, energy_depth=1, energy_heads=1,
        energy_hidden=8, history_hidden=8, step_ps=1.0,
        diffusion_scale=0.01, coupling_scale=0.03,
    )
    return enc, oracle, batch, args, bundle, StageB1b(mcfg), path


def test_future_targets_cannot_change_autonomous_prediction(tiny):
    enc, oracle, batch, args, _, model, _ = tiny
    changed = deepcopy(batch)
    changed["coords"][:, 2:] += torch.randn_like(changed["coords"][:, 2:])*0.2
    with sdpa_kernel([SDPBackend.MATH]):
        loss, _, a = forecast_objective(
            model, enc, oracle, batch, args, generator=torch.Generator().manual_seed(31),
        )
        other, _, b = forecast_objective(
            model, enc, oracle, changed, args, generator=torch.Generator().manual_seed(31),
        )
    assert torch.equal(a["H_pred"], b["H_pred"])
    assert torch.equal(a["S_pred"], b["S_pred"])
    assert torch.equal(a["predicted_backbone"], b["predicted_backbone"])
    assert not torch.equal(a["H_target"], b["H_target"])
    assert loss.item() != other.item()


def test_decoder_reads_only_autonomous_predictions_and_observed_anchor(tiny, monkeypatch):
    enc, oracle, batch, args, _, model, _ = tiny
    decoded_inputs = []
    original = oracle.decode_structure

    def record(H, *positional, **kwargs):
        decoded_inputs.append(H.detach().clone())
        return original(H, *positional, **kwargs)

    monkeypatch.setattr(oracle, "decode_structure", record)
    with sdpa_kernel([SDPBackend.MATH]):
        _, _, forecast = forecast_objective(model, enc, oracle, batch, args)
    M, B, _, L, d = forecast["H_pred"].shape
    # Future H is a detached latent label. The only two structure readouts are
    # the autonomous forecast and the observed anchor used for nuisance pose.
    assert not forecast["H_target"].requires_grad
    assert len(decoded_inputs) == 2
    assert torch.equal(decoded_inputs[0], oracle.h_normalizer.inverse(
        forecast["H_pred"][:, :, 1:],
    ).reshape(M*B*args.horizon_steps, L, d))
    assert torch.equal(decoded_inputs[1], oracle.h_normalizer.inverse(forecast["H_anchor"]))


def test_autonomous_loss_trains_energy_bath_posterior_only(tiny):
    enc, oracle, batch, args, _, model, _ = tiny
    with sdpa_kernel([SDPBackend.MATH]):
        loss, _, _ = forecast_objective(model, enc, oracle, batch, args)
        loss.backward()
    for module in (model.energy, model.bath, model.history_posterior):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    assert all(p.grad is None for p in enc.parameters())
    assert all(p.grad is None for p in oracle.parameters())


def test_contiguous_history_has_one_shared_anchor(tiny):
    _, _, batch, args, *_ = tiny
    history, sequence, mask, dt = split_contiguous_batch(batch, 2, 2, 1.0)
    assert history.shape[1] == 2 and sequence.shape[1] == 3
    assert torch.equal(history[:, -1], sequence[:, 0])
    assert dt.tolist() == [[1.0, 1.0]]
    bad = deepcopy(batch)
    bad["time_ps"][:, 2] = bad["time_ps"][:, 1]
    with pytest.raises(ValueError, match="increasing"):
        split_contiguous_batch(bad, 2, 2, 1.0)


def test_zero_bath_autonomous_training(tiny):
    enc, oracle, batch, args, _, _, _ = tiny
    cfg = StageB1bConfig(
        latent_dim=8, bath_dim=0, energy_depth=1, energy_heads=1,
        energy_hidden=8, step_ps=1.0,
    )
    model = StageB1b(cfg)
    with sdpa_kernel([SDPBackend.MATH]):
        loss, _, forecast = forecast_objective(model, enc, oracle, batch, args)
        loss.backward()
    assert forecast["S_pred"] is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.energy.parameters())


def test_local_residual_rotates_and_translates_with_anchor():
    torch.manual_seed(7)
    model = ResidualCADecoder(ResidualDecoderConfig(latent_dim=4, hidden=8, depth=1))
    hf, hc, x = torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.randn(2, 6, 3)
    m = torch.ones(2, 6, dtype=torch.bool)
    R = torch.eye(3).expand(2, 6, 3, 3)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = torch.tensor([2., -3., 1.])
    base = model(hf, hc, x, m, R_anchor=R)
    moved = model(hf, hc, x@rotation.T+shift, m, R_anchor=rotation@R)
    assert torch.allclose(moved["ca_pred"], base["ca_pred"]@rotation.T+shift, atol=1e-6)
    with pytest.raises(ValueError, match="R_anchor"):
        model(hf, hc, x, m)


def test_frozen_bundle_and_exact_step_resume(tiny, tmp_path):
    enc, oracle, batch, args, bundle, model, b1a_path = tiny
    loaded_enc, loaded_oracle, _ = load_frozen_representation(b1a_path, torch.device("cpu"))
    assert not loaded_enc.training and not loaded_oracle.training
    assert not any(p.requires_grad for p in loaded_enc.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    with sdpa_kernel([SDPBackend.MATH]):
        loss, _, _ = forecast_objective(model, enc, oracle, batch, args)
        loss.backward()
        optimizer.step()
    checkpoint = tmp_path/"resume.pt"
    torch.save(checkpoint_payload(model, optimizer, 1, bundle, args, {}, 1.0, 0), checkpoint)
    optimizer.zero_grad()
    with sdpa_kernel([SDPBackend.MATH]):
        next_loss, _, expected = forecast_objective(model, enc, oracle, batch, args)
        next_loss.backward()
        optimizer.step()
    expected_state = deepcopy(model.state_dict())
    restored = StageB1b(model.cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    state = resume_training(checkpoint, restored, restored_optimizer, bundle, args, torch.device("cpu"))
    assert state["step"] == 1
    with sdpa_kernel([SDPBackend.MATH]):
        actual_loss, _, actual = forecast_objective(restored, enc, oracle, batch, args)
        actual_loss.backward()
        restored_optimizer.step()
    assert torch.equal(expected["H_pred"], actual["H_pred"])
    assert next_loss.item() == actual_loss.item()
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in expected_state.items())


def test_teacher_forced_training_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--b1a-ckpt", "unused", "--nll-weight", "1"])


def test_resume_rejects_changed_sampling_seed_and_manifest_contents(tiny, tmp_path):
    _, _, _, args, bundle, model, _ = tiny
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.manifest_split_record = {"train_manifest_sha256": "original"}
    path = tmp_path/"resume.pt"
    torch.save(checkpoint_payload(model, optimizer, 1, bundle, args, {}, 1.0, 0), path)
    changed = deepcopy(args)
    changed.seed += 1
    with pytest.raises(ValueError, match="contract mismatch: seed"):
        resume_training(path, model, optimizer, bundle, changed, torch.device("cpu"))
    changed = deepcopy(args)
    changed.manifest_split_record["train_manifest_sha256"] = "modified"
    with pytest.raises(ValueError, match="manifest split or content mismatch"):
        resume_training(path, model, optimizer, bundle, changed, torch.device("cpu"))


def test_validation_budget_spans_full_dataset_instead_of_only_its_prefix(tiny):
    _, _, _, args, *_ = tiny
    args.synthetic_size, args.batch_size, args.validation_batches = 20, 1, 3
    validation = _loader(args, None, training=False)
    assert validation.dataset.indices == [0, 4, 9]
    assert list(validation.dataset.dataset.indices) == list(range(10, 20))
    sampling = validation.validation_sampling
    assert sampling["available_windows"] == 10
    assert sampling["selected_windows"] == 3
    assert sampling["indices_in_validation_dataset"] == [0, 4, 9]
    assert sampling["unit"] == "trajectory_windows_not_independent_proteins"
    training = _loader(args, None, training=True)
    assert len(training.dataset) == 10
    assert isinstance(training.sampler, EpochPermutation)
    assert training.validation_sampling is None
    args.validation_batches = 20
    complete = _loader(args, None, training=False)
    assert complete.dataset.indices == list(range(10))


def test_formal_training_requires_matching_scientific_gate(tiny, tmp_path):
    import json
    _, _, _, args, bundle, _, _ = tiny
    args.dry_run = False
    with pytest.raises(ValueError, match="requires"):
        validate_oracle_gate(None, bundle, args, None)
    report = {"schema_version": 2, "passed": True, "scope": "heldout_protein",
              "representation_id": bundle["representation_contract"]["representation_id"],
              "protocol": {"history_frames": 2, "frame_stride": 1,
                           "frame_interval_ps": 1.0, "horizons_frames": [2]},
              "by_horizon": {"2": {"passed": True}},
              "train_manifest_sha256": "train", "val_manifest_sha256": "val"}
    path = tmp_path/"gate.json"
    path.write_text(json.dumps(report))
    audit = {"train_manifest_sha256": "train", "val_manifest_sha256": "val"}
    assert validate_oracle_gate(path, bundle, args, audit)["passed"]
    report["representation_id"] = "another_encoder"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="representation"):
        validate_oracle_gate(path, bundle, args, audit)


def test_gate_must_cover_training_and_long_validation_horizons(tiny, tmp_path):
    import json
    _, _, _, args, bundle, _, _ = tiny
    args.dry_run = False
    args.validation_horizon_steps = 4
    audit = {"train_manifest_sha256": "train", "val_manifest_sha256": "val"}
    report = {"schema_version": 2, "passed": True, "scope": "heldout_protein",
              "representation_id": bundle["representation_contract"]["representation_id"],
              "protocol": {"history_frames": 2, "frame_stride": 1,
                           "frame_interval_ps": 1.0, "horizons_frames": [1, 2]},
              "by_horizon": {"1": {"passed": True}, "2": {"passed": True}}, **audit}
    path = tmp_path/"gate.json"

    def check():
        path.write_text(json.dumps(report))
        return validate_oracle_gate(path, bundle, args, audit)

    with pytest.raises(ValueError, match="cover requested horizon: 4"):
        check()
    report["protocol"]["horizons_frames"].append(4)
    report["by_horizon"]["4"] = {"passed": False}
    with pytest.raises(ValueError, match="did not pass requested horizon: 4"):
        check()
    report["by_horizon"]["4"]["passed"] = True
    assert check()["passed"]
    report["protocol"]["frame_interval_ps"] = 2.0
    with pytest.raises(ValueError, match="physical step mismatch"):
        check()
    report["protocol"].pop("frame_interval_ps")
    with pytest.raises(ValueError, match="requires a positive frame_interval_ps"):
        check()


def test_gate_horizons_are_raw_offsets_at_the_requested_stride(tiny, tmp_path):
    import json
    _, _, _, args, bundle, _, _ = tiny
    args.frame_stride, args.step_ps = 2, 2.0
    audit = {"train_manifest_sha256": "train", "val_manifest_sha256": "val"}
    report = {"schema_version": 2, "passed": True, "scope": "heldout_protein",
              "representation_id": bundle["representation_contract"]["representation_id"],
              "protocol": {"history_frames": 2, "frame_stride": 2,
                           "frame_interval_ps": 1.0, "horizons_frames": [2]},
              "by_horizon": {"2": {"passed": True}}, **audit}
    path = tmp_path/"gate.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cover requested horizon: 4"):
        validate_oracle_gate(path, bundle, args, audit)
    report["protocol"]["horizons_frames"].append(4)
    report["by_horizon"]["4"] = {"passed": True}
    path.write_text(json.dumps(report))
    assert validate_oracle_gate(path, bundle, args, audit)["passed"]

