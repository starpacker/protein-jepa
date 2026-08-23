import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig, MaskConfig, ModelConfig, OptimizationConfig, TrainConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import SyntheticProteinDynamicsDataset
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.models.jepa import (
    kinetic_vampnet_loss,
    ProteinJEPA,
    jepa_loss,
    lagged_state_jepa_loss,
    lagged_state_semigroup_loss,
    linear_dynamics_jepa_loss,
    physchem_prediction_loss,
    state_jepa_loss,
    state_transition_jepa_loss,
)
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


def test_kinetic_query_pooling_trains_pooler():
    data_cfg = DataConfig(batch_size=2, num_frames=6, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            frame_pooling="kinetic_query",
            frame_pool_queries=2,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = state_jepa_loss(outputs)
    loss.backward()
    pooler_grads = [p.grad for p in model.temporal_encoder.frame_pooler.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert outputs["frame_states"].shape == (2, 6, 32)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in pooler_grads)


def test_residual_kinetic_pooling_starts_as_mean_pooling():
    data_cfg = DataConfig(batch_size=2, num_frames=6, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    mean_model = ProteinJEPA(
        ModelConfig(
            latent_dim=32, encoder_depth=1, temporal_depth=1, predictor_depth=1, num_heads=4, num_rbf=8, dropout=0.0
        )
    )
    residual_model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            frame_pooling="mean_kinetic_query",
            frame_pool_queries=2,
            num_heads=4,
            num_rbf=8,
            dropout=0.0,
        )
    )
    mean_model.eval()
    residual_model.eval()
    residual_model.context_encoder.load_state_dict(mean_model.context_encoder.state_dict())
    residual_model.temporal_encoder.time_embed.load_state_dict(mean_model.temporal_encoder.time_embed.state_dict())
    residual_model.temporal_encoder.blocks.load_state_dict(mean_model.temporal_encoder.blocks.state_dict())
    residual_model.temporal_encoder.norm.load_state_dict(mean_model.temporal_encoder.norm.state_dict())
    residual_model.temporal_encoder.cls_token.data.copy_(mean_model.temporal_encoder.cls_token.data)

    latents = mean_model.context_encoder(
        coords=batch["coords"],
        residue_mask=batch["residue_mask"],
        aatype=batch["aatype"],
        residue_index=batch["residue_index"],
        chain_id=batch["chain_id"],
        token_mask=None,
    )
    mean_states = mean_model.encode_states(latents, batch["residue_mask"])["frame_states"]
    residual_states = residual_model.encode_states(latents, batch["residue_mask"])["frame_states"]

    assert torch.allclose(mean_states, residual_states, atol=1e-6)
    assert torch.count_nonzero(residual_model.temporal_encoder.frame_pool_gate).item() == 0


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


def test_lagged_state_jepa_trains_lag_predictor():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 4),
            lag_predictor_depth=1,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = lagged_state_jepa_loss(outputs)
    loss.backward()
    lag_grads = [p.grad for p in model.lagged_state_predictor.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in lag_grads)


def test_state_transition_jepa_trains_state_predictor():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = state_transition_jepa_loss(outputs)
    loss.backward()
    predictor_grads = [p.grad for p in model.state_predictor.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in predictor_grads)


def test_lagged_semigroup_loss_trains_lag_predictor():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 3, 4),
            lag_predictor_depth=1,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = lagged_state_semigroup_loss(outputs, model.lagged_state_predictor)
    loss.backward()
    lag_grads = [p.grad for p in model.lagged_state_predictor.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in lag_grads)


def test_physchem_prediction_loss_trains_physchem_head():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            physchem_prediction_lags=(1, 2),
            physchem_predictor_depth=1,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = physchem_prediction_loss(outputs)
    loss.backward()
    physchem_grads = [p.grad for p in model.physchem_predictor.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in physchem_grads)


def test_linear_dynamics_jepa_trains_linear_transition_heads():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            linear_dynamics_lags=(1, 2, 4),
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = linear_dynamics_jepa_loss(outputs)
    loss.backward()
    dyn_grads = [p.grad for p in model.linear_dynamics_predictor.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert loss.item() > 0
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in dyn_grads)


def test_kinetic_vampnet_loss_trains_readout_head():
    data_cfg = DataConfig(batch_size=3, num_frames=8, synthetic_size=6, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.75)
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            kinetic_readout_dim=4,
            kinetic_readout_depth=1,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = kinetic_vampnet_loss(outputs, lag=1)
    loss.backward()
    readout_grads = [p.grad for p in model.kinetic_readout.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in readout_grads)


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
            auxiliary_warmup_steps=2,
            residue_jepa_weight=0.25,
            state_jepa_weight=1.0,
            semigroup_weight=0.01,
            state_transition_weight=0.05,
        ),
        log_every=1,
    )
    ckpt = train(cfg)

    assert ckpt.exists()
