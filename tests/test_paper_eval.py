import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig, ModelConfig, model_config_from_dict
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import SyntheticProteinDynamicsDataset
from protein_jepa_frame.data.manifest import TrajectoryRecord
from protein_jepa_frame.eval.paper_eval import (
    _data_cfg_from_artifact,
    _kinetic_normalized_vamp2,
    _default_nonoverlap_lags,
    _ridge_probe,
    _ridge_future_score,
    _stratified_trajectories,
    _window_overlap_ratio,
)
from protein_jepa_frame.eval.paper_eval import _model_readout
from protein_jepa_frame.models.jepa import ProteinJEPA


def _record(protein: str, trajectory: str) -> TrajectoryRecord:
    return TrajectoryRecord(path=Path("unused.npz"), protein_id=protein, trajectory_id=trajectory)


def test_trajectory_selection_round_robins_across_proteins():
    records = [
        _record("p1", "p1-r1"),
        _record("p1", "p1-r2"),
        _record("p2", "p2-r1"),
        _record("p2", "p2-r2"),
        _record("p3", "p3-r1"),
    ]

    selected = _stratified_trajectories(records, 4)

    assert [record.trajectory_id for record in selected] == ["p1-r1", "p2-r1", "p3-r1", "p1-r2"]
    assert len({record.protein_id for record in selected[:3]}) == 3


def test_eval_defaults_to_training_sampling_and_nonoverlap_forecast_lags(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"data": {"frame_stride": 2, "window_stride": 8, "max_residues": 384, "coordinate_scale": 1.0}}
        )
    )

    cfg = _data_cfg_from_artifact(tmp_path, batch_size=4, num_frames=24, window_stride=None)

    assert cfg.frame_stride == 2
    assert cfg.window_stride == 8
    assert np.isclose(_window_overlap_ratio(24, cfg.frame_stride, cfg.window_stride), 39.0 / 47.0)
    assert _default_nonoverlap_lags(24, cfg.frame_stride, cfg.window_stride) == [6, 12, 24]
    assert np.isclose(_window_overlap_ratio(24, 1, 8), 2.0 / 3.0)


def test_vamp_normalization_removes_constant_mode():
    assert _kinetic_normalized_vamp2(1.0, 5) == 0.0
    assert _kinetic_normalized_vamp2(6.0, 5) == 1.0


def test_msm_lambda_selection_does_not_use_outer_validation_labels():
    rng = np.random.RandomState(3)
    train_x = rng.randn(60, 4)
    train_y = (train_x[:, 0] > 0).astype(int)
    val_x = rng.randn(20, 4)
    val_y = (val_x[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(6), 10)
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]

    original = _ridge_probe(train_x, train_y, val_x, val_y, lambdas, train_groups=groups)
    permuted = _ridge_probe(train_x, train_y, val_x, val_y[::-1], lambdas, train_groups=groups)

    assert original["lambda"] == permuted["lambda"]
    assert original["lambda_selection"] == "train_group_holdout"


def test_collate_preserves_thermodynamic_metadata():
    dataset = SyntheticProteinDynamicsDataset(
        DataConfig(batch_size=2, num_frames=4, synthetic_size=2, synthetic_residues=8)
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))

    assert torch.equal(batch["temperature_k"], torch.tensor([300.0, 300.0]))
    assert torch.equal(batch["dataset_index"], torch.tensor([3, 3]))
    assert batch["dataset"] == ["synthetic", "synthetic"]


def test_small_linear_fold_uses_fixed_train_only_lambda():
    rng = np.random.RandomState(9)
    x0_train = rng.randn(10, 3)
    x1_train = x0_train + 0.1 * rng.randn(10, 3)
    x0_val = rng.randn(5, 3)
    x1_val = x0_val + 0.1 * rng.randn(5, 3)
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]

    result = _ridge_future_score(
        x0_train, x1_train, x0_val, x1_val, lambdas, max_pre_dim=8, lambda_selection="train_holdout"
    )

    assert result["lambda"] == 1.0
    assert np.isfinite(result["r2"])


def test_v22_readouts_use_causal_endpoint_and_slow_projection():
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            slow_state_dim=5,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            slow_state_enabled=True,
            num_heads=4,
            num_rbf=8,
            dropout=0.0,
        )
    )
    latents = torch.randn(2, 3, 6, 16)
    residue_mask = torch.ones(2, 3, 6, dtype=torch.bool)
    endpoint = _model_readout(model, latents, residue_mask, "endpoint")
    slow = _model_readout(model, latents, residue_mask, "slow_endpoint")

    assert endpoint.shape == (2, 16)
    assert slow.shape == (2, 5)
    assert torch.allclose(endpoint, _model_readout(model, latents, residue_mask, "state"))


def test_v23_readouts_keep_offline_and_causal_towers_explicit():
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            num_heads=4,
            num_rbf=8,
            dropout=0.0,
            temporal_position_encoding="none",
        )
    )
    latents = torch.randn(2, 3, 6, 16)
    residue_mask = torch.ones(2, 3, 6, dtype=torch.bool)

    offline = _model_readout(model, latents, residue_mask, "endpoint")
    causal = _model_readout(model, latents, residue_mask, "causal_endpoint")

    assert torch.allclose(offline, model.encode_states(latents, residue_mask)["frame_states"][:, -1])
    assert torch.allclose(causal, model.encode_causal_states(latents, residue_mask)["frame_states"][:, -1])
    assert not torch.allclose(offline, causal)


def test_historical_checkpoint_keys_restore_missing_model_flags():
    config = model_config_from_dict(
        {"latent_dim": 16},
        {
            "paired_delta_predictor.delta.weight": torch.empty(16, 32),
            "slow_state_projector.net.0.weight": torch.empty(16),
        },
    )

    assert config.paired_forecast
    assert config.slow_state_enabled
