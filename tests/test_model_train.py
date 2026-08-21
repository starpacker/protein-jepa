import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig, MaskConfig, ModelConfig, OptimizationConfig, TrainConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import SyntheticProteinDynamicsDataset
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.models.jepa import ProteinJEPA, jepa_loss, state_jepa_loss
from protein_jepa_frame.training.train import train


def test_model_forward_loss_is_finite():
    data_cfg = DataConfig(batch_size=2, num_frames=4, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(MaskConfig(num_blocks=1))(batch["residue_mask"])
    model = ProteinJEPA(ModelConfig(latent_dim=32, encoder_depth=1, predictor_depth=1, num_heads=4, num_rbf=8))

    outputs = model(batch, context, target)
    loss = jepa_loss(outputs)

    assert torch.isfinite(loss)
    assert outputs["predictions"].shape == outputs["targets"].shape


def test_temporal_state_encoder_forward_loss_is_finite():
    data_cfg = DataConfig(batch_size=2, num_frames=6, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(latent_dim=32, encoder_depth=1, temporal_depth=1, predictor_depth=1, num_heads=4, num_rbf=8)
    )

    outputs = model(batch, context, target)
    loss = jepa_loss(outputs)
    state_loss = state_jepa_loss(outputs)

    assert torch.isfinite(loss)
    assert torch.isfinite(state_loss)
    assert outputs["frame_states"].shape == (2, 6, 32)
    assert outputs["clip_state"].shape == (2, 32)
    assert outputs["state_predictions"].shape == outputs["target_frame_states"].shape
    assert outputs["state_target_mask"].shape == (2, 6)
    assert outputs["state_target_mask"].sum() > 0


def test_state_jepa_trains_temporal_modules():
    data_cfg = DataConfig(batch_size=2, num_frames=6, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(latent_dim=32, encoder_depth=1, temporal_depth=1, predictor_depth=1, num_heads=4, num_rbf=8)
    )

    outputs = model(batch, context, target)
    loss = state_jepa_loss(outputs)
    loss.backward()

    temporal_grads = [p.grad for p in model.temporal_encoder.parameters() if p.requires_grad]
    predictor_grads = [p.grad for p in model.state_predictor.parameters() if p.requires_grad]
    target_grads = [p.grad for p in model.target_temporal_encoder.parameters()]

    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in temporal_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in predictor_grads)
    assert all(g is None for g in target_grads)


def test_smoke_training_writes_checkpoint(tmp_path):
    cfg = TrainConfig(
        output_dir=str(tmp_path),
        data=DataConfig(batch_size=2, num_workers=0, num_frames=4, synthetic_size=8, synthetic_residues=16),
        model=ModelConfig(latent_dim=32, encoder_depth=1, temporal_depth=1, predictor_depth=1, num_heads=4, num_rbf=8),
        mask=MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.5),
        optimization=OptimizationConfig(
            epochs=1,
            steps_per_epoch=2,
            warmup_steps=1,
            residue_jepa_weight=0.25,
            state_jepa_weight=1.0,
        ),
        log_every=1,
    )
    ckpt = train(cfg)

    assert ckpt.exists()
