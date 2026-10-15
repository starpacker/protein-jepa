import pytest
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import (
    DataConfig,
    MaskConfig,
    ModelConfig,
    OptimizationConfig,
    TrainConfig,
    _coerce_dataclass,
)
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import SyntheticProteinDynamicsDataset
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.models.jepa import (
    kinetic_vampnet_loss,
    ProteinJEPA,
    jepa_loss,
    lagged_state_jepa_loss,
    lagged_state_jepa_terms,
    lagged_state_semigroup_loss,
    lagged_state_semigroup_terms,
    linear_dynamics_jepa_loss,
    physchem_prediction_loss,
    state_jepa_loss,
    state_covariance_regularizer,
    state_feature_variance_regularizer,
    state_transition_jepa_loss,
)
from protein_jepa_frame.training.train import (
    _load_init_checkpoint,
    _load_resume_checkpoint,
    _save_checkpoint,
    _validate_init_freeze_contract,
    train,
)
from protein_jepa_frame.models.temporal import TemporalStateEncoder


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


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown TrainConfig fields"):
        _coerce_dataclass(TrainConfig, {"deterministc": True})


def test_ema_target_encoders_remain_deterministic_in_train_mode():
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
            dropout=0.5,
            num_heads=4,
            num_rbf=8,
        )
    )
    model.train()

    first = model(batch, context, target)
    second = model(batch, context, target)

    assert not model.target_encoder.training
    assert not model.target_temporal_encoder.training
    assert torch.equal(first["targets"], second["targets"])
    assert torch.equal(first["state_targets"], second["state_targets"])


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


def test_lagged_state_jepa_supervises_short_lags_across_future_gap():
    data_cfg = DataConfig(batch_size=2, num_frames=24, synthetic_size=4, synthetic_residues=8)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(
            mode="future",
            temporal_scale=(0.25, 0.5),
            future_context_ratio=0.5,
            future_gap_frames=4,
        ),
        required_future_horizon=12,
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 4, 8, 12),
            lag_predictor_depth=1,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    terms = lagged_state_jepa_terms(outputs)
    loss = lagged_state_jepa_loss(outputs)
    loss.backward()

    assert set(terms) == {1, 2, 4, 8, 12}
    assert all(term["pairs"] == 2 for term in terms.values())
    for lag in terms:
        grads = [p.grad for p in model.lagged_state_predictor.heads[str(lag)].parameters()]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_time_conditioned_lagged_state_jepa_trains_lag_embedding():
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
            lag_conditioning="embedding",
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = lagged_state_jepa_loss(outputs)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.lagged_state_predictor.lag_embed.weight.grad is not None
    assert torch.isfinite(model.lagged_state_predictor.lag_embed.weight.grad).all()
    assert model.lagged_state_predictor.lag_embed.weight.grad.abs().sum() > 0


def test_physical_time_conditioning_trains_time_embeddings():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    batch["time_ps"][0] = batch["time_ps"][0] * 10.0
    batch["time_ps"][1] = batch["time_ps"][1] * 1000.0
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
            time_conditioning="physical",
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = state_jepa_loss(outputs) + lagged_state_jepa_loss(outputs)
    loss.backward()

    temporal_grad = model.temporal_encoder.physical_time_embed.proj[0].weight.grad
    state_predictor_grad = model.state_predictor.physical_time_embed.proj[0].weight.grad
    lag_predictor_grad = model.lagged_state_predictor.physical_lag_embed.proj[0].weight.grad

    assert torch.isfinite(loss)
    assert temporal_grad is not None and torch.isfinite(temporal_grad).all() and temporal_grad.abs().sum() > 0
    assert (
        state_predictor_grad is not None
        and torch.isfinite(state_predictor_grad).all()
        and state_predictor_grad.abs().sum() > 0
    )
    assert lag_predictor_grad is not None and torch.isfinite(lag_predictor_grad).all()
    assert lag_predictor_grad.abs().sum() > 0


def test_thermodynamic_conditioning_trains_temperature_and_dataset_embeddings():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    batch["temperature_k"] = torch.tensor([300.0, 450.0])
    batch["dataset_index"] = torch.tensor([1, 2])
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
            thermodynamic_conditioning=True,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = state_jepa_loss(outputs)
    loss.backward()

    condition = model.temporal_encoder.condition_embed
    assert torch.isfinite(loss)
    assert condition.temperature_proj[0].weight.grad is not None
    assert condition.temperature_proj[0].weight.grad.abs().sum() > 0
    assert condition.dataset_embed.weight.grad is not None
    assert condition.dataset_embed.weight.grad.abs().sum() > 0


def test_causal_temporal_states_do_not_depend_on_future_frames():
    encoder = TemporalStateEncoder(
        latent_dim=16,
        depth=2,
        num_heads=4,
        dropout=0.0,
        causal=True,
        delta_features=True,
    )
    encoder.eval()
    latents = torch.randn(2, 8, 6, 16)
    residue_mask = torch.ones(2, 8, 6, dtype=torch.bool)
    perturbed = latents.clone()
    perturbed[:, 4:] += 100.0 * torch.randn_like(perturbed[:, 4:])

    with torch.no_grad():
        original = encoder(latents, residue_mask)["frame_states"]
        changed = encoder(perturbed, residue_mask)["frame_states"]

    assert torch.allclose(original[:, :4], changed[:, :4], atol=1e-5)
    assert not torch.allclose(original[:, 4:], changed[:, 4:], atol=1e-5)


def test_windowed_causal_temporal_state_has_bounded_receptive_field():
    encoder = TemporalStateEncoder(
        latent_dim=16,
        depth=2,
        num_heads=4,
        dropout=0.0,
        causal=True,
        causal_window=4,
        position_encoding="none",
    ).eval()
    latents = torch.randn(2, 8, 6, 16)
    residue_mask = torch.ones(2, 8, 6, dtype=torch.bool)
    perturbed = latents.clone()
    perturbed[:, 0] += 100.0 * torch.randn_like(perturbed[:, 0])

    with torch.no_grad():
        original = encoder(latents, residue_mask)["frame_states"]
        changed = encoder(perturbed, residue_mask)["frame_states"]

    assert not torch.allclose(original[:, :7], changed[:, :7], atol=1e-5)
    assert torch.allclose(original[:, 7], changed[:, 7], atol=1e-5)


def test_position_free_temporal_state_does_not_encode_clip_index_for_static_frames():
    encoder = TemporalStateEncoder(
        latent_dim=16,
        depth=2,
        num_heads=4,
        dropout=0.0,
        causal=True,
        position_encoding="none",
    )
    encoder.eval()
    frame = torch.randn(2, 1, 6, 16)
    latents = frame.expand(-1, 8, -1, -1).clone()
    residue_mask = torch.ones(2, 8, 6, dtype=torch.bool)

    with torch.no_grad():
        states = encoder(latents, residue_mask)["frame_states"]

    assert torch.allclose(states, states[:, :1].expand_as(states), atol=1e-5)


def test_state_feature_regularizers_train_causal_temporal_encoder():
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
            temporal_depth=2,
            predictor_depth=1,
            state_predictor_depth=1,
            temporal_causal=True,
            temporal_delta_features=True,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    variance = state_feature_variance_regularizer(outputs)
    covariance = state_covariance_regularizer(outputs)
    (variance + covariance).backward()

    assert torch.isfinite(variance)
    assert torch.isfinite(covariance)
    assert model.temporal_encoder.delta_proj[1].weight.grad is not None
    assert model.temporal_encoder.delta_proj[1].weight.grad.abs().sum() > 0


def test_motion_transition_is_decoupled_from_slow_state_encoder():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    batch["temperature_k"] = torch.tensor([320.0, 450.0])
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5),
        required_future_horizon=4,
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=2,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 4),
            motion_depth=1,
            motion_stopgrad_input=True,
            transition_source="motion",
            transition_thermodynamic_conditioning=True,
            temporal_causal=True,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = lagged_state_jepa_loss(outputs)
    loss.backward()

    motion_grads = [p.grad for p in model.temporal_encoder.motion_blocks.parameters() if p.requires_grad]
    slow_grads = [p.grad for p in model.temporal_encoder.blocks.parameters() if p.requires_grad]
    spatial_grads = [p.grad for p in model.context_encoder.parameters() if p.requires_grad]
    condition_grad = model.lagged_state_predictor.condition_embed.temperature_proj[0].weight.grad
    assert outputs["motion_states"].shape == outputs["frame_states"].shape
    assert any(g is not None and g.abs().sum() > 0 for g in motion_grads)
    assert all(g is None or g.abs().sum() == 0 for g in slow_grads)
    assert all(g is None or g.abs().sum() == 0 for g in spatial_grads)
    assert condition_grad is not None and condition_grad.abs().sum() > 0


def test_state_motion_residual_transition_starts_as_identity_and_preserves_gradient_split():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5),
        required_future_horizon=4,
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=2,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 4),
            lag_predictor_depth=1,
            lag_prediction_mode="residual",
            motion_depth=1,
            motion_stopgrad_input=True,
            transition_source="state_motion",
            temporal_causal=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    assert torch.isfinite(outputs["target_state_motion_states"]).all()
    assert torch.equal(outputs["lagged_state_targets"], outputs["target_state_motion_states"])
    for prediction in outputs["lagged_state_predictions"].values():
        assert torch.allclose(prediction, outputs["transition_states"])
    assert lagged_state_semigroup_loss(outputs, model.lagged_state_predictor).item() == 0.0

    loss = lagged_state_jepa_loss(outputs)
    loss.backward()
    motion_grad = model.temporal_encoder.motion_gate.grad
    slow_grads = [p.grad for p in model.temporal_encoder.blocks.parameters() if p.requires_grad]
    spatial_grads = [p.grad for p in model.context_encoder.parameters() if p.requires_grad]
    assert motion_grad is not None and motion_grad.abs().sum() > 0
    assert all(g is None or g.abs().sum() == 0 for g in slow_grads)
    assert all(g is None or g.abs().sum() == 0 for g in spatial_grads)


def test_state_transition_uses_normalized_closed_state_and_backpropagates():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5),
        required_future_horizon=4,
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=2,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 4),
            lag_predictor_depth=1,
            lag_prediction_mode="residual",
            transition_source="state",
            temporal_causal=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    expected_source = torch.nn.functional.layer_norm(
        outputs["frame_states"], (outputs["frame_states"].shape[-1],)
    )
    assert torch.allclose(outputs["transition_states"], expected_source, atol=1.0e-6)
    assert torch.equal(outputs["lagged_state_targets"], outputs["state_targets"])

    loss = lagged_state_jepa_loss(outputs)
    loss.backward()
    temporal_grads = [p.grad for p in model.temporal_encoder.blocks.parameters() if p.requires_grad]
    spatial_grads = [p.grad for p in model.context_encoder.parameters() if p.requires_grad]
    assert torch.isfinite(loss)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in temporal_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in spatial_grads)


def test_semigroup_is_disabled_for_motion_to_state_predictor():
    data_cfg = DataConfig(batch_size=2, num_frames=8, synthetic_size=4, synthetic_residues=16)
    dataset = SyntheticProteinDynamicsDataset(data_cfg)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_clips)))
    context, target = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.5, 0.5), future_context_ratio=0.5),
        required_future_horizon=4,
    )(batch["residue_mask"])
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2, 4),
            motion_depth=1,
            motion_stopgrad_input=True,
            transition_source="motion",
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    assert lagged_state_semigroup_terms(outputs, model.lagged_state_predictor) == {}
    assert lagged_state_semigroup_loss(outputs, model.lagged_state_predictor).item() == 0.0


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


def test_stopgrad_kinetic_vampnet_loss_only_trains_readout_head():
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
            kinetic_readout_stopgrad=True,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model(batch, context, target)
    loss = kinetic_vampnet_loss(outputs, lag=1)
    loss.backward()
    readout_grads = [p.grad for p in model.kinetic_readout.parameters() if p.requires_grad]
    temporal_grads = [p.grad for p in model.temporal_encoder.parameters() if p.requires_grad]

    assert torch.isfinite(loss)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in readout_grads)
    assert all(g is None or g.abs().sum() == 0 for g in temporal_grads)


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


def test_checkpoint_helpers_preserve_resume_step(tmp_path):
    cfg = TrainConfig(
        output_dir=str(tmp_path),
        device="cpu",
        model=ModelConfig(latent_dim=16, encoder_depth=1, predictor_depth=1, num_heads=2, num_rbf=4),
        checkpoint_every_steps=2,
    )
    model = ProteinJEPA(cfg.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimization.lr)

    _save_checkpoint(tmp_path, model, optimizer, epoch=1, step=7, config=cfg, write_step_file=True)

    restored_model = ProteinJEPA(cfg.model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=cfg.optimization.lr)
    step = _load_resume_checkpoint(restored_model, restored_optimizer, str(tmp_path / "checkpoint_latest.pt"), torch.device("cpu"))

    assert step == 7
    assert (tmp_path / "checkpoint_step7.pt").exists()


def test_init_checkpoint_can_exclude_reinitialized_modules(tmp_path):
    cfg = TrainConfig(
        output_dir=str(tmp_path),
        device="cpu",
        model=ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            state_prediction_lags=(1, 2),
            num_heads=2,
            num_rbf=4,
        ),
    )
    source = ProteinJEPA(cfg.model)
    optimizer = torch.optim.AdamW(source.parameters(), lr=cfg.optimization.lr)
    with torch.no_grad():
        source.context_encoder.geom_proj.weight.fill_(0.25)
        source.lagged_state_predictor.heads["1"].weight.fill_(0.75)
    _save_checkpoint(tmp_path, source, optimizer, epoch=1, step=3, config=cfg, write_step_file=False)

    restored = ProteinJEPA(cfg.model)
    initial_lag_weight = restored.lagged_state_predictor.heads["1"].weight.detach().clone()
    _load_init_checkpoint(
        restored,
        str(tmp_path / "checkpoint_latest.pt"),
        torch.device("cpu"),
        strict=False,
        exclude_prefixes=("lagged_state_predictor.",),
    )

    assert torch.all(restored.context_encoder.geom_proj.weight == 0.25)
    assert torch.equal(restored.lagged_state_predictor.heads["1"].weight, initial_lag_weight)


def test_init_checkpoint_cannot_exclude_frozen_tmjepa_potential():
    cfg = TrainConfig(
        init_checkpoint="pretrained.pt",
        init_exclude_prefixes=("tmjepa.",),
        model=ModelConfig(
            paired_forecast=True,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_freeze_potential=True,
        ),
    )

    with pytest.raises(ValueError, match="exclusions overlap frozen TM-JEPA parameters"):
        _validate_init_freeze_contract(cfg)


def test_init_checkpoint_may_reinitialize_trainable_tmjepa_kinetics():
    cfg = TrainConfig(
        init_checkpoint="pretrained.pt",
        init_exclude_prefixes=("tmjepa.dynamics.kinetics.",),
        model=ModelConfig(
            paired_forecast=True,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_freeze_potential=True,
            tmjepa_freeze_kinetics=False,
        ),
    )

    _validate_init_freeze_contract(cfg)


def test_pmf_stage_may_ignore_reinitialized_frozen_kinetics():
    cfg = TrainConfig(
        init_checkpoint="pretrained.pt",
        init_exclude_prefixes=("tmjepa.dynamics.kinetics.",),
        model=ModelConfig(
            paired_forecast=True,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_freeze_potential=False,
            tmjepa_freeze_kinetics=True,
        ),
        optimization=OptimizationConfig(tmjepa_equilibrium_score_weight=0.001),
    )

    _validate_init_freeze_contract(cfg)


def test_frozen_learned_tmjepa_representation_requires_checkpoint():
    cfg = TrainConfig(
        model=ModelConfig(
            paired_forecast=True,
            temporal_depth=1,
            temporal_causal=True,
            temporal_causal_window=4,
            freeze_context_encoder=True,
            freeze_temporal_encoder=True,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_learned_cv_dim=2,
            tmjepa_learned_cv_burn_in_frames=3,
            tmjepa_freeze_learned_projection=True,
        )
    )

    with pytest.raises(ValueError, match="requires an init or resume checkpoint"):
        _validate_init_freeze_contract(cfg)
