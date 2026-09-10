import pytest
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import (
    ProteinJEPA,
    gaussian_transition_nll,
    metastable_vamp_terms,
    local_observable_prediction_loss,
    closed_local_state_terms,
    compute_v3_closed_local_state,
    compute_v3_closed_local_state_sequence,
    compute_v3_closed_local_velocity,
    compute_v4c_force_state,
    observable_prediction_loss,
    offline_forecast_terms,
    paired_endpoint_variance_regularizer,
    paired_forecast_terms,
    reversible_kinetic_terms,
    slow_forecast_terms,
    v4_slow_kinetic_terms,
    v4_physics_terms,
    v5_contextual_dynamics_terms,
    v51_sparse_motion_terms,
    _per_trajectory_vamp_terms,
)
from protein_jepa_frame.models.predictor import (
    ClosedLocalStateOperator,
    ContextualAdaptiveDynamics,
    ContextualSparseMotionDynamics,
    ReversibleKineticOUDynamics,
    V4PhysicsSlowModel,
    V4SlowKineticModel,
)
from scripts.evaluate_paired_dynamics import (
    _closed_local_metrics,
    _closed_local_pair_metrics,
    _v4_slow_metrics,
)


def _paired_batch():
    config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(2, 4),
        synthetic_size=4,
        synthetic_residues=12,
    )
    dataset = build_dataset(config)
    return next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_paired_windows)))


def test_paired_dataset_preserves_trajectory_and_horizon():
    batch = _paired_batch()

    assert batch["context_trajectory_id"] == batch["future_trajectory_id"]
    assert torch.all(torch.isin(batch["horizon_frames"], torch.tensor([2, 4])))
    assert torch.allclose(
        batch["future_time_ps"][:, 0] - batch["context_time_ps"][:, 0],
        batch["horizon_time_ps"],
    )
    assert batch["context_coords"].shape == batch["future_coords"].shape


def test_grouped_paired_dataset_returns_all_horizons_for_same_context():
    config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8, 12, 16),
        paired_nonoverlap_only=True,
        paired_all_horizons=True,
        synthetic_size=4,
        synthetic_residues=12,
    )
    dataset = build_dataset(config)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_paired_windows)))

    assert batch["context_coords"].shape[0] == 6
    assert batch["context_group_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert batch["context_group_size"].tolist() == [3, 3]
    assert batch["horizon_frames"].tolist() == [8, 12, 16, 8, 12, 16]
    assert torch.equal(batch["context_coords"][0], batch["context_coords"][1])
    assert torch.equal(batch["context_coords"][1], batch["context_coords"][2])


def test_paired_dataset_collates_optional_forces():
    config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=4,
        synthetic_residues=12,
        load_forces=True,
    )
    batch = next(iter(DataLoader(build_dataset(config), batch_size=2, collate_fn=collate_paired_windows)))

    assert batch["context_forces"].shape == batch["context_coords"].shape
    assert batch["future_forces"].shape == batch["future_coords"].shape
    assert torch.isfinite(batch["context_forces"]).all()


def test_nonoverlap_contract_rejects_overlapping_windows():
    config = DataConfig(
        training_mode="paired_window",
        num_frames=8,
        frame_stride=2,
        paired_horizons_frames=(8, 16),
        paired_nonoverlap_only=True,
    )
    with pytest.raises(ValueError, match="raw clip span"):
        build_dataset(config)


def test_paired_forward_trains_endpoint_and_delta_predictor():
    batch = _paired_batch()
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=2,
            predictor_depth=1,
            paired_forecast=True,
            learned_state_training=True,
            slow_state_enabled=True,
            slow_state_dim=16,
            slow_vamp_lags=(1, 2),
            paired_predictor_depth=1,
            thermodynamic_conditioning=True,
            transition_thermodynamic_conditioning=True,
            temporal_causal=True,
            temporal_position_encoding="none",
            temporal_delta_features=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
        )
    )
    outputs = model.forward_paired(batch)
    terms = paired_forecast_terms(outputs)
    slow_terms = slow_forecast_terms(outputs, vamp_lags=(1, 2))
    loss = (
        terms["delta_loss"]
        + 0.25 * terms["state_loss"]
        + slow_terms["delta_loss"]
        + 0.25 * slow_terms["state_loss"]
        + 0.1 * slow_terms["contrastive_loss"]
        + 0.05 * slow_terms["relative_delta_loss"]
        + 0.05 * slow_terms["alignment_loss"]
        + 0.05 * slow_terms["vamp_loss"]
        + paired_endpoint_variance_regularizer(outputs)
    )
    loss.backward()

    assert outputs["context_endpoint"].shape == (2, 32)
    assert outputs["delta_prediction"].shape == (2, 32)
    assert outputs["context_slow_state"].shape == (2, 16)
    assert outputs["slow_delta_prediction"].shape == (2, 16)
    assert outputs["context_slow_sequence"].shape == (2, 8, 16)
    assert torch.isfinite(slow_terms["vamp2_score"])
    assert torch.isfinite(loss)
    assert torch.allclose(outputs["delta_prediction"], torch.zeros_like(outputs["delta_prediction"]), atol=1e-5)
    assert torch.allclose(outputs["slow_delta_prediction"], torch.zeros_like(outputs["slow_delta_prediction"]), atol=1e-5)
    temporal_grads = [p.grad for p in model.temporal_encoder.parameters() if p.requires_grad]
    predictor_grads = [p.grad for p in model.paired_delta_predictor.parameters() if p.requires_grad]
    slow_grads = [p.grad for p in model.slow_state_projector.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in temporal_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in predictor_grads)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in slow_grads)
    assert all(p.grad is None for p in model.target_slow_state_projector.parameters())


def test_learned_state_training_rejects_frozen_or_fixed_cv_paths():
    common = dict(
        latent_dim=32,
        encoder_depth=1,
        temporal_depth=1,
        predictor_depth=1,
        paired_forecast=True,
        learned_state_training=True,
        slow_state_enabled=True,
        temporal_causal=True,
        num_heads=4,
        num_rbf=8,
    )
    with pytest.raises(ValueError, match="trainable online encoders"):
        ProteinJEPA(ModelConfig(**common, freeze_context_encoder=True))
    with pytest.raises(ValueError, match="fixed-CV TM-JEPA branch"):
        ProteinJEPA(
            ModelConfig(
                **common,
                tmjepa_enabled=True,
                tmjepa_version=3,
                tmjepa_learned_cv_dim=2,
                temporal_causal_window=2,
                tmjepa_learned_cv_burn_in_frames=1,
            )
        )


def test_ls2_reversible_generator_has_exact_balance_stationarity_and_semigroup():
    torch.manual_seed(11)
    dynamics = ReversibleKineticOUDynamics(
        state_dim=8,
        num_states=6,
        base_step_ps=1000.0,
        assignment_temperature=0.5,
        stationary_floor=1.0e-4,
        rate_scale=0.1,
        dropout=0.0,
    ).eval()
    aatype = torch.randint(0, 20, (2, 12))
    residue_mask = torch.ones(2, 12, dtype=torch.bool)
    temperature = torch.tensor([320.0, 379.0])
    dataset_index = torch.ones(2, dtype=torch.long)
    generator, stationary, detailed_balance = dynamics.generator(
        aatype, residue_mask, temperature, dataset_index
    )
    assert torch.allclose(generator.sum(dim=-1), torch.zeros(2, 6), atol=1.0e-7)
    assert torch.all(generator.diagonal(dim1=-2, dim2=-1) <= 0.0)
    assert detailed_balance.abs().max() < 1.0e-7

    initial = torch.softmax(torch.randn(2, 6), dim=-1)
    residual = torch.randn(2, 8)
    first_time = torch.tensor([2000.0, 3000.0])
    second_time = torch.tensor([4000.0, 2000.0])
    first_mean, _, first_assignment, _ = dynamics.propagate(
        initial, residual, generator, first_time
    )
    first_residual = first_mean - first_assignment @ dynamics.prototypes
    composed_mean, _, composed_assignment, _ = dynamics.propagate(
        first_assignment, first_residual, generator, second_time
    )
    direct_mean, _, direct_assignment, direct_transition = dynamics.propagate(
        initial, residual, generator, first_time + second_time
    )
    assert torch.allclose(composed_assignment, direct_assignment, atol=2.0e-6)
    assert torch.allclose(composed_mean, direct_mean, atol=2.0e-6)
    stationary_after = torch.bmm(
        stationary.unsqueeze(1), direct_transition
    ).squeeze(1)
    assert torch.allclose(stationary_after, stationary, atol=2.0e-6)


def test_ls2_joint_vjepa_path_trains_online_modules_only():
    batch = _paired_batch()
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            learned_state_training=True,
            slow_state_enabled=True,
            slow_state_dim=8,
            slow_vamp_lags=(1, 2),
            ls2_reversible_enabled=True,
            ls2_num_states=6,
            paired_predictor_depth=1,
            temporal_causal=True,
            temporal_position_encoding="none",
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
        )
    )
    model.sync_target_encoders()
    outputs = model.forward_paired(batch)
    terms = reversible_kinetic_terms(outputs, vamp_lags=(1, 2))
    loss = (
        terms["state_loss"]
        + terms["assignment_loss"]
        + 0.05 * terms["nll"]
        + 0.01 * terms["balance_loss"]
        + 0.01 * terms["vamp_loss"]
    )
    loss.backward()

    assert outputs["ls2_state_prediction"].shape == (2, 8)
    assert outputs["ls2_assignment_prediction"].shape == (2, 6)
    assert torch.isfinite(loss)
    assert terms["detailed_balance_max"] < 1.0e-6
    assert terms["stationarity_error"] < 1.0e-5
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.context_encoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.slow_state_projector.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.ls2_reversible_dynamics.parameters()
        if parameter.requires_grad
    )
    assert all(
        parameter.grad is None
        for parameter in model.target_slow_state_projector.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.ls2_reversible_dynamics.target_assignment_head.parameters()
    )


def test_v23_dual_towers_gaussian_and_observables_train_independently():
    config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=4,
        synthetic_residues=12,
    )
    batch = next(iter(DataLoader(build_dataset(config), batch_size=2, collate_fn=collate_paired_windows)))
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            probabilistic_transition=True,
            observable_head=True,
            thermodynamic_conditioning=True,
            transition_thermodynamic_conditioning=True,
            temporal_causal=True,
            temporal_position_encoding="none",
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
        )
    )

    outputs = model.forward_paired(batch)
    offline_terms = offline_forecast_terms(outputs)
    loss = gaussian_transition_nll(outputs) + observable_prediction_loss(outputs) + offline_terms["delta_loss"]
    loss.backward()

    assert outputs["observable_target"].shape == (2, 4)
    assert torch.isfinite(outputs["observable_target"]).all()
    assert outputs["gaussian_log_sigma"].min() >= model.config.transition_log_sigma_min
    assert outputs["gaussian_log_sigma"].max() <= model.config.transition_log_sigma_max
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.offline_temporal_encoder.parameters())
    assert all(p.grad is None for p in model.target_temporal_encoder.parameters())
    assert all(p.grad is None for p in model.target_offline_temporal_encoder.parameters())


def test_v231_canonical_delta_and_persistence_share_teacher_space():
    config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=4,
        synthetic_residues=12,
    )
    batch = next(iter(DataLoader(build_dataset(config), batch_size=2, collate_fn=collate_paired_windows)))
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            paired_teacher_space_delta=True,
            paired_target_tower="offline",
            freeze_context_encoder=True,
            freeze_offline_temporal_encoder=True,
            temporal_causal=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
            feature_norm=False,
        )
    )
    model.train()
    outputs = model.forward_paired(batch)

    assert torch.allclose(outputs["persistence_endpoint"], outputs["context_teacher_endpoint"])
    assert torch.allclose(
        outputs["delta_target"],
        outputs["future_endpoint"] - outputs["context_teacher_endpoint"],
    )
    assert torch.allclose(
        outputs["state_prediction"],
        outputs["persistence_endpoint"] + outputs["delta_prediction"],
    )
    assert not model.context_encoder.training
    assert not model.offline_temporal_encoder.training
    assert all(not parameter.requires_grad for parameter in model.context_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.offline_temporal_encoder.parameters())


def test_v3_history_transition_metastable_and_physical_targets_train():
    config = DataConfig(
        training_mode="paired_window",
        batch_size=3,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=6,
        synthetic_residues=12,
    )
    batch = next(iter(DataLoader(build_dataset(config), batch_size=3, collate_fn=collate_paired_windows)))
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            paired_teacher_space_delta=True,
            paired_target_tower="offline",
            history_conditioned_transition=True,
            history_transition_strides=(1, 2, 4),
            metastable_state_dim=4,
            local_dynamics_enabled=True,
            local_dynamics_tokens=4,
            local_dynamics_dim=8,
            observable_head=True,
            observable_dim=12,
            observable_target_version="v3",
            freeze_context_encoder=True,
            freeze_offline_temporal_encoder=True,
            temporal_causal=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
            feature_norm=False,
        )
    )
    outputs = model.forward_paired(batch)
    metastable_terms = metastable_vamp_terms(outputs, lags=(1, 2, 4))
    loss = (
        paired_forecast_terms(outputs)["delta_loss"]
        + observable_prediction_loss(outputs)
        + metastable_terms["vamp_loss"]
        + 0.1 * metastable_terms["balance_loss"]
        + local_observable_prediction_loss(outputs)
    )
    loss.backward()

    assert outputs["observable_target"].shape == (3, 12)
    assert torch.isfinite(outputs["observable_target"]).all()
    assert outputs["local_observable_target"].shape == (3, 4, 12)
    assert outputs["local_dynamics_tokens"].shape == (3, 4, 8)
    assert torch.isfinite(outputs["local_observable_target"]).all()
    assert torch.allclose(outputs["delta_prediction"], torch.zeros_like(outputs["delta_prediction"]), atol=1e-6)
    assert torch.allclose(outputs["context_metastable_states"].sum(dim=-1), torch.ones(3, 8))
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.paired_delta_predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.metastable_state_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.observable_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.local_dynamics_head.parameters())
    assert all(p.grad is None for p in model.context_encoder.parameters())
    assert all(p.grad is None for p in model.offline_temporal_encoder.parameters())


def test_v32_grouped_horizons_train_closed_state_operator_only():
    data_config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8, 12, 16),
        paired_nonoverlap_only=True,
        paired_all_horizons=True,
        synthetic_size=4,
        synthetic_residues=16,
    )
    batch = next(iter(DataLoader(build_dataset(data_config), batch_size=2, collate_fn=collate_paired_windows)))
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            paired_teacher_space_delta=True,
            paired_target_tower="offline",
            history_conditioned_transition=True,
            local_dynamics_enabled=True,
            local_observable_targets=False,
            local_dynamics_tokens=4,
            local_dynamics_dim=8,
            closed_local_state_enabled=True,
            closed_local_state_dim=12,
            closed_local_horizons=(8, 12, 16),
            freeze_context_encoder=True,
            freeze_offline_temporal_encoder=True,
            temporal_causal=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
            feature_norm=False,
        )
    )
    outputs = model.forward_paired(batch)
    terms = closed_local_state_terms(outputs, margin=0.001)
    loss = terms["state_loss"] + 0.1 * terms["ranking_loss"]
    loss.backward()

    assert outputs["closed_local_prediction"].shape == (6, 4, 12)
    assert torch.allclose(
        outputs["closed_local_prediction"], outputs["context_closed_local_state"], atol=1e-6
    )
    assert torch.isfinite(loss)
    assert terms["ranking_loss"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.closed_local_operator.parameters())
    assert all(p.grad is None for p in model.local_dynamics_head.parameters())
    assert all(p.grad is None for p in model.context_encoder.parameters())


def test_closed_local_state_operator_validates_channel_scales():
    with pytest.raises(ValueError, match="one positive value per channel"):
        ClosedLocalStateOperator(state_dim=3, state_scale=(1.0, 2.0))
    with pytest.raises(ValueError, match="one positive value per channel"):
        ClosedLocalStateOperator(state_dim=3, state_scale=(1.0, 0.0, 2.0))


def test_continuous_closed_operator_shares_context_and_has_zero_time_origin():
    operator = ClosedLocalStateOperator(
        latent_dim=8,
        token_dim=4,
        state_dim=3,
        num_tokens=2,
        horizons=(8, 12, 16),
        state_scale=(1.0, 1.0, 1.0),
        operator_mode="continuous_relaxation",
        relaxation_scales=(0.5, 1.0),
        depth=1,
        dropout=0.0,
    )
    local = torch.randn(1, 2, 4).expand(3, -1, -1).clone()
    current = torch.randn(1, 2, 3).expand(3, -1, -1).clone()
    history = torch.randn(1, 8).expand(3, -1).clone()
    horizons = torch.tensor([8, 12, 16])

    prediction, representation = operator(
        local, current, history, horizons, horizons.float()
    )

    assert torch.allclose(prediction, current, atol=1.0e-7)
    assert torch.allclose(representation[0], representation[1], atol=1.0e-7)
    with torch.no_grad():
        operator.relaxation_head.bias.view(3, 3)[0, 0] = 1.0
    prediction, _ = operator(local, current, history, horizons, horizons.float())
    expected_linear_drift = horizons.float() / 16.0
    assert torch.allclose(prediction[:, 0, 0] - current[:, 0, 0], expected_linear_drift)


def test_velocity_probabilistic_closed_operator_uses_standardized_deltas():
    operator = ClosedLocalStateOperator(
        latent_dim=8,
        token_dim=4,
        state_dim=2,
        num_tokens=1,
        horizons=(8, 16),
        state_scale=(0.1, 10.0),
        operator_mode="velocity_probabilistic",
        relaxation_scales=(1.0,),
        velocity_strides=(1, 2),
        depth=1,
        dropout=0.0,
    )
    local = torch.randn(2, 1, 4)
    current = torch.tensor([[[1.0, 20.0]], [[1.0, 20.0]]])
    history = torch.randn(2, 8)
    horizons = torch.tensor([8, 16])
    velocity = torch.randn(2, 1, 2, 2)

    prediction, representation, log_sigma = operator(
        local, current, history, horizons, horizons.float(), current_velocity=velocity
    )

    assert torch.allclose(prediction, current, atol=1.0e-7)
    assert representation.shape == (2, 1, 8)
    assert torch.allclose(log_sigma, torch.zeros_like(log_sigma))
    with torch.no_grad():
        operator.relaxation_head.bias.view(2, 2)[0] = 1.0
    prediction, _, _ = operator(
        local, current, history, horizons, horizons.float(), current_velocity=velocity
    )
    expected = horizons.float() / 16.0
    assert torch.allclose((prediction[:, 0] - current[:, 0])[:, 0], expected * 0.1)
    assert torch.allclose((prediction[:, 0] - current[:, 0])[:, 1], expected * 10.0)


def test_compositional_closed_operator_reuses_one_stable_step():
    operator = ClosedLocalStateOperator(
        latent_dim=8,
        token_dim=4,
        state_dim=2,
        num_tokens=2,
        horizons=(8, 16),
        state_scale=(0.5, 2.0),
        state_mean=(1.0, 10.0),
        operator_mode="compositional_rollout",
        base_step_frames=4,
        cross_segment_heads=2,
        depth=1,
        dropout=0.0,
    )
    current = torch.tensor([[[1.0, 10.0], [1.0, 10.0]]]).expand(2, -1, -1).clone()
    sequence = current[:, None].expand(-1, 3, -1, -1).clone()
    local = torch.zeros(2, 2, 4)
    history = torch.zeros(2, 8)
    horizons = torch.tensor([8, 16])
    with torch.no_grad():
        operator.equilibrium_head.bias.fill_(1.0)
        operator.rate_head.bias.zero_()

    prediction, representation = operator(
        local, current, history, horizons, horizons.float(),
        context_state_sequence=sequence,
        context_sequence_mask=torch.ones(2, 3, 2, dtype=torch.bool),
    )

    decay = torch.exp(-torch.nn.functional.softplus(torch.tensor(0.0)))
    expected_standardized = torch.tensor([1.0 - decay**2, 1.0 - decay**4])
    assert representation.shape == (2, 2, 4)
    assert torch.allclose((prediction[:, 0, 0] - 1.0) / 0.5, expected_standardized, atol=1.0e-5)
    assert torch.allclose((prediction[:, 0, 1] - 10.0) / 2.0, expected_standardized, atol=1.0e-5)


def test_v4_compositional_forward_trains_base_and_rollout_steps():
    data_config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        frame_stride=1,
        paired_horizons_frames=(8, 12, 16),
        paired_nonoverlap_only=True,
        paired_all_horizons=True,
        synthetic_size=4,
        synthetic_residues=16,
    )
    batch = next(iter(DataLoader(build_dataset(data_config), batch_size=2, collate_fn=collate_paired_windows)))
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=32,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            paired_teacher_space_delta=True,
            paired_target_tower="offline",
            history_conditioned_transition=True,
            local_dynamics_enabled=True,
            local_observable_targets=False,
            local_dynamics_tokens=4,
            local_dynamics_dim=8,
            closed_local_state_enabled=True,
            closed_local_horizons=(8, 12, 16),
            closed_local_operator_mode="compositional_rollout",
            closed_local_base_step_frames=4,
            closed_local_history_frame_stride=1,
            closed_local_cross_segment_heads=2,
            v4_slow_state_enabled=True,
            v4_slow_latent_dim=8,
            v4_slow_num_states=4,
            v4_slow_vamp_lags=(1, 2, 4),
            temporal_causal=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
            feature_norm=False,
        )
    )
    outputs = model.forward_paired(batch)
    terms = closed_local_state_terms(outputs)
    kinetic_terms = v4_slow_kinetic_terms(outputs, lags=(1, 2, 4))
    (
        terms["state_loss"]
        + terms["base_state_loss"]
        + kinetic_terms["reconstruction_loss"]
        + kinetic_terms["vamp_loss"]
        + kinetic_terms["base_transition_loss"]
        + kinetic_terms["rollout_transition_loss"]
    ).backward()

    assert outputs["context_closed_local_sequence"].shape == (6, 8, 4, 12)
    assert outputs["closed_local_base_prediction"].shape == (6, 4, 12)
    assert torch.allclose(outputs["closed_local_prediction"], outputs["context_closed_local_state"], atol=1e-6)
    assert torch.allclose(outputs["closed_local_base_prediction"], outputs["closed_local_base_current"], atol=1e-6)
    assert torch.isfinite(terms["base_state_loss"])
    assert outputs["v4_slow_context_assignment"].shape == (6, 8, 4)
    assert torch.allclose(
        outputs["v4_slow_context_assignment"].sum(dim=-1), torch.ones(6, 8), atol=1e-6
    )
    assert torch.allclose(
        outputs["v4_slow_transition_matrix"].sum(dim=-1), torch.ones(6, 4), atol=1e-6
    )
    assert torch.isfinite(kinetic_terms["vamp_loss"])
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.closed_local_operator.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.v4_slow_kinetic_model.parameters()
    )


def test_v4_slow_kinetic_model_composes_row_stochastic_transition():
    model = V4SlowKineticModel(
        num_tokens=2,
        state_dim=3,
        latent_dim=8,
        num_states=4,
        base_step_frames=4,
        history_frame_stride=1,
        state_scale=(1.0, 2.0, 3.0),
        dropout=0.0,
    )
    context = torch.randn(2, 6, 2, 3)
    future = torch.randn(2, 6, 2, 3)
    mask = torch.ones(2, 6, 2, dtype=torch.bool)
    outputs = model(context, future, mask, mask, torch.tensor([8, 16]))

    assert outputs["v4_slow_prediction"].shape == (2, 4)
    assert torch.allclose(outputs["v4_slow_prediction"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(
        outputs["v4_slow_transition_matrix"].sum(dim=-1), torch.ones(2, 4), atol=1e-6
    )
    assert outputs["v4_slow_context_normalized"].mean(dim=1).abs().max() < 1e-5


def test_v4c_model_is_reversible_and_separates_slow_residual():
    model = V4PhysicsSlowModel(
        offline_dim=8,
        num_tokens=2,
        state_dim=3,
        latent_dim=8,
        num_states=4,
        base_step_frames=4,
        history_frame_stride=1,
        state_scale=(1.0, 2.0, 3.0),
        lowpass_width=3,
        dropout=0.0,
    )
    context = torch.randn(2, 6, 2, 3)
    future = torch.randn(2, 6, 2, 3)
    mask = torch.ones(2, 6, 2, dtype=torch.bool)
    offline_context = torch.randn(2, 6, 8)
    offline_future = torch.randn(2, 6, 8)
    outputs = model(
        context, future, mask, mask, offline_context, offline_future,
        torch.tensor([8, 16]), temperature_k=torch.tensor([320.0, 450.0]),
    )
    outputs["delta_target"] = torch.zeros(2, 8)
    terms = v4_physics_terms(outputs, lags=(1, 2, 4))
    loss = (
        terms["reconstruction_loss"] + terms["vamp_loss"]
        + terms["cross_covariance_loss"] + terms["increment_covariance_loss"]
        + terms["drift_loss"] + terms["stationary_loss"]
        + terms["base_transition_loss"] + terms["rollout_transition_loss"]
    )
    loss.backward()

    assert outputs["v4c_context_residual"].shape == (2, 6, 6)
    assert torch.allclose(outputs["v4c_transition_matrix"].sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(outputs["v4c_stationary_distribution"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert outputs["v4c_detailed_balance_residual"].abs().max() < 1e-6
    assert terms["mean_lambda2"] > 0.9
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_v4c_conservative_force_path_supports_second_derivatives():
    batch, residues = 1, 8
    coordinates = torch.randn(batch, 1, residues, 4, 3, requires_grad=True)
    atom_mask = torch.ones(batch, 1, residues, 4, dtype=torch.bool)
    residue_mask = torch.ones(batch, 1, residues, dtype=torch.bool)
    chain_id = torch.zeros(batch, residues, dtype=torch.long)
    residue_index = torch.arange(residues).repeat(batch, 1)
    physical_state, state_mask = compute_v4c_force_state(
        coordinates, atom_mask, residue_mask, chain_id, residue_index, num_tokens=2
    )
    model = V4PhysicsSlowModel(
        offline_dim=8,
        num_tokens=2,
        state_dim=12,
        latent_dim=8,
        num_states=4,
        base_step_frames=2,
        history_frame_stride=1,
        state_scale=(1.0,) * 12,
        lowpass_width=3,
    )
    energy = model.force_energy(
        physical_state[:, None], physical_state[:, None].detach(), torch.randn(batch, 8)
    )
    predicted_force = -torch.autograd.grad(
        energy.sum(), coordinates, create_graph=True, retain_graph=True
    )[0]
    force_loss = predicted_force.square().mean()
    force_loss.backward()

    assert state_mask.all()
    assert predicted_force.shape == coordinates.shape
    assert torch.isfinite(predicted_force).all()
    assert any(p.grad is not None for p in model.free_energy_head.parameters())


def test_v4c_per_trajectory_balance_rejects_static_identity_states():
    states = torch.zeros(4, 8, 4)
    for index in range(4):
        states[index, :, index] = 1.0
    mask = torch.ones(4, 8, dtype=torch.bool)
    terms = _per_trajectory_vamp_terms(states, states, mask, mask, lags=(1, 2))

    global_occupancy = states.mean(dim=(0, 1))
    global_balance = (global_occupancy * (global_occupancy / 0.25).log()).sum()
    assert global_balance.abs() < 1e-6
    assert terms["trajectory_balance_loss"] > 1.0


def test_v4c_dynamic_only_latent_ignores_offline_identity_features():
    model = V4PhysicsSlowModel(
        offline_dim=8,
        num_tokens=2,
        state_dim=3,
        latent_dim=8,
        num_states=4,
        base_step_frames=4,
        history_frame_stride=1,
        state_scale=(1.0, 2.0, 3.0),
        lowpass_width=3,
        dynamic_only=True,
        condition_on_equilibrium=True,
        dropout=0.0,
    )
    context = torch.randn(2, 6, 2, 3)
    future = torch.randn(2, 6, 2, 3)
    mask = torch.ones(2, 6, 2, dtype=torch.bool)
    offline_a = torch.randn(2, 6, 8)
    offline_b = torch.randn(2, 6, 8) * 10.0
    first = model(context, future, mask, mask, offline_a, offline_a, torch.tensor([8, 16]))
    second = model(context, future, mask, mask, offline_b, offline_b, torch.tensor([8, 16]))

    assert torch.allclose(first["v4c_context_latent"], second["v4c_context_latent"])
    assert torch.allclose(first["v4c_context_assignment"], second["v4c_context_assignment"])
    assert torch.allclose(first["v4c_transition_matrix"], second["v4c_transition_matrix"])


def test_v4c_d2_emits_fixed_controls_and_horizon_losses():
    model = V4PhysicsSlowModel(
        offline_dim=8,
        num_tokens=2,
        state_dim=3,
        latent_dim=8,
        num_states=4,
        base_step_frames=4,
        history_frame_stride=1,
        state_scale=(1.0, 2.0, 3.0),
        lowpass_width=3,
        dynamic_only=True,
        condition_on_equilibrium=True,
        training_fixed_controls=True,
        dropout=0.0,
    )
    context = torch.randn(3, 8, 2, 3)
    future = torch.randn(3, 8, 2, 3)
    mask = torch.ones(3, 8, 2, dtype=torch.bool)
    offline = torch.randn(3, 8, 8)
    outputs = model(
        context, future, mask, mask, offline, offline,
        torch.tensor([48, 64, 96]), temperature_k=torch.tensor([320.0] * 3),
    )
    outputs["delta_target"] = torch.zeros(3, 8)
    outputs["horizon_frames"] = torch.tensor([48, 64, 96])
    outputs["context_group_index"] = torch.zeros(3, dtype=torch.long)
    terms = v4_physics_terms(outputs, lags=(1, 2, 4), horizon_ranking_margin=0.01)
    loss = (
        terms["base_transition_loss"] + terms["rollout_transition_loss"]
        + terms["transition_nll"] + terms["horizon_ranking_loss"]
        + terms["horizon_difference_loss"]
    )
    loss.backward()

    assert outputs["v4c_fixed_predictions"].shape == (3, 5, 4)
    assert terms["horizon_ranking_loss"] >= 0
    assert terms["horizon_difference_loss"] >= 0
    assert any(p.grad is not None for p in model.symmetric_flux_head.parameters())


def test_v4c_d2_freezes_representation_not_reversible_generator():
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            offline_temporal_depth=1,
            causal_temporal_depth=1,
            dual_temporal_towers=True,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            paired_target_tower="offline",
            history_conditioned_transition=True,
            local_dynamics_enabled=True,
            local_observable_targets=False,
            local_dynamics_tokens=2,
            local_dynamics_dim=8,
            closed_local_state_enabled=True,
            closed_local_horizons=(48, 64, 96),
            closed_local_operator_mode="compositional_rollout",
            closed_local_base_step_frames=16,
            closed_local_history_frame_stride=2,
            v4_physics_state_enabled=True,
            v4_physics_dynamic_only=True,
            freeze_v4_physics_representation=True,
            dropout=0.0,
            num_heads=4,
            num_rbf=8,
        )
    )
    trainable = {
        name for name, parameter in model.v4_physics_slow_model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable
    assert all(
        name.startswith((
            "condition_trunk.", "equilibrium_condition_trunk.", "condition_embed.",
            "state_free_energy_head.", "symmetric_flux_head.",
        ))
        for name in trainable
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.v4_physics_slow_model.frame_encoder.parameters()
    )


def test_v5_assignment_and_generator_depend_on_history_with_same_endpoint():
    torch.manual_seed(7)
    model = ContextualAdaptiveDynamics(
        latent_dim=8,
        slot_dim=6,
        max_slots=4,
        base_step_ps=4.0,
        max_rollout_steps=4,
        assignment_temperature=0.5,
        thermodynamic_conditioning=False,
        dropout=0.0,
    )
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, :-1] = -first[:, :-1]
    context = torch.cat([first, second], dim=0)
    future = torch.randn_like(context)
    mask = torch.ones(2, 6, dtype=torch.bool)
    outputs = model(
        context,
        future,
        mask,
        mask,
        torch.tensor([4.0, 4.0]),
        torch.tensor([4, 4]),
    )

    assert torch.equal(context[0, -1], context[1, -1])
    assert not torch.allclose(outputs["v5_context_slots"][0], outputs["v5_context_slots"][1])
    assert not torch.allclose(outputs["v5_current_assignment"][0], outputs["v5_current_assignment"][1])
    assert not torch.allclose(outputs["v5_generator"][0], outputs["v5_generator"][1])
    assert outputs["v5_detailed_balance_residual"].abs().max() < 1.0e-5
    assert outputs["v5_generator"].sum(dim=-1).abs().max() < 1.0e-5


def test_v5_rollout_uses_physical_time_and_emits_fixed_controls():
    torch.manual_seed(11)
    model = ContextualAdaptiveDynamics(
        latent_dim=8,
        slot_dim=6,
        max_slots=4,
        base_step_ps=4.0,
        max_rollout_steps=12,
        assignment_temperature=0.5,
        control_horizons_frames=(4, 12),
        training_fixed_controls=True,
        thermodynamic_conditioning=False,
        dropout=0.0,
    )
    model.log_rate_scale.data.fill_(-2.0)
    context = torch.randn(1, 6, 8).expand(2, -1, -1).clone()
    future = torch.randn(1, 6, 8).expand(2, -1, -1).clone()
    mask = torch.ones(2, 6, dtype=torch.bool)
    outputs = model(
        context,
        future,
        mask,
        mask,
        torch.tensor([4.0, 12.0]),
        torch.tensor([4, 4]),
    )

    assert torch.allclose(outputs["v5_current_assignment"][0], outputs["v5_current_assignment"][1])
    assert not torch.allclose(outputs["v5_prediction"][0], outputs["v5_prediction"][1])
    assert outputs["v5_fixed_predictions"].shape == (2, 2, 4)
    assert torch.allclose(outputs["v5_prediction"].sum(dim=-1), torch.ones(2), atol=1.0e-5)


def test_v5_joint_loss_trains_geometry_temporal_slots_and_generator():
    batch = _paired_batch()
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            temporal_causal=True,
            temporal_position_encoding="none",
            v5_contextual_dynamics_enabled=True,
            v5_slot_dim=8,
            v5_max_slots=4,
            v5_base_step_ps=1.0,
            v5_max_rollout_steps=8,
            v5_control_horizons_frames=(2, 4),
            transition_thermodynamic_conditioning=False,
            num_heads=2,
            num_rbf=4,
            dropout=0.0,
            feature_norm=False,
        )
    )
    outputs = model.forward_paired(batch)
    terms = v5_contextual_dynamics_terms(outputs)
    loss = (
        terms["transition_loss"]
        + terms["state_reconstruction_loss"]
        + terms["feature_forecast_loss"]
        + terms["observable_loss"]
        + terms["observable_horizon_difference_loss"]
        + terms["transition_nll"]
        + terms["slot_diversity_loss"]
        + terms["gate_complexity_loss"]
    )
    loss.backward()

    assert outputs["v5_prediction"].shape == (2, 4)
    assert outputs["v5_observable_prediction"].shape == (2, 12)
    assert outputs["v5_observable_target"].shape == (2, 12)
    assert torch.isfinite(loss)
    for module in (
        model.context_encoder,
        model.temporal_encoder,
        model.v5_contextual_dynamics,
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_temporal_encoder.parameters())


def test_v51_sparse_modes_depend_on_history_and_decoder_is_anchored():
    torch.manual_seed(19)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=6,
        num_modes=5,
        active_modes=2,
        base_step_ps=1.0,
        max_rollout_steps=8,
        control_horizons_frames=(2, 4),
        training_controls=True,
        dropout=0.0,
    )
    torch.nn.init.normal_(model.motion_expert[-1].weight, std=0.02)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, :-1] = -first[:, :-1]
    context = torch.cat([first, second], dim=0)
    future = torch.randn_like(context)
    mask = torch.ones(2, 6, dtype=torch.bool)
    time_ps = torch.arange(6).float().unsqueeze(0).expand(2, -1)
    outputs = model(
        context,
        context,
        future,
        mask,
        mask,
        time_ps,
        time_ps,
        torch.tensor([2.0, 4.0]),
        torch.tensor([2, 2]),
    )

    assert torch.equal(context[0, -1], context[1, -1])
    assert not torch.allclose(
        outputs["v51_router_probabilities"][0],
        outputs["v51_router_probabilities"][1],
    )
    assert torch.equal(
        outputs["v51_active_mode_mask"].sum(dim=-1), torch.tensor([2, 2])
    )
    assert torch.allclose(
        outputs["v51_no_rollout_observable"],
        model.zero_motion_observable.unsqueeze(0).expand(2, -1),
        atol=1.0e-7,
    )
    assert outputs["v51_fixed_predictions"].shape == (2, 2, 6)
    assert outputs["v51_fixed_observable_predictions"].shape == (2, 2, 12)


def test_v51_joint_loss_trains_representation_motion_and_decoder():
    batch = _paired_batch()
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            temporal_causal=True,
            temporal_position_encoding="none",
            v51_contextual_dynamics_enabled=True,
            v51_state_dim=8,
            v51_num_modes=4,
            v51_active_modes=2,
            v51_base_step_ps=1.0,
            v51_max_rollout_steps=8,
            v51_control_horizons_frames=(2, 4),
            v51_training_controls=True,
            v51_causal_prediction_enabled=True,
            transition_thermodynamic_conditioning=False,
            num_heads=2,
            num_rbf=4,
            dropout=0.0,
            feature_norm=False,
        )
    )
    outputs = model.forward_paired(batch)
    terms = v51_sparse_motion_terms(outputs)
    loss = (
        terms["delta_loss"]
        + 0.1 * terms["nll_loss"]
        + terms["current_alignment_loss"]
        + terms["observable_loss"]
        + terms["horizon_ranking_loss"]
        + terms["history_ranking_loss"]
        + 0.1 * terms["causal_next_loss"]
        + 0.1 * terms["horizon_consistency_loss"]
        + terms["horizon_calibration_loss"]
        + 0.1 * terms["variance_loss"]
        + 0.01 * terms["covariance_loss"]
        + terms["load_balance_loss"]
        + terms["mode_diversity_loss"]
        + terms["mode_orthogonality_loss"]
    )
    loss.backward()

    assert outputs["v51_predicted_feature"].shape == (2, 8)
    assert outputs["v51_observable_prediction"].shape == (2, 12)
    assert outputs["v51_observable_target"].shape == (2, 12)
    assert terms["mean_active_modes"] == 2
    assert outputs["v51_causal_next_prediction"].shape[-1] == 8
    assert torch.isfinite(terms["causal_next_loss"])
    assert torch.isfinite(terms["horizon_calibration_loss"])
    assert torch.isfinite(terms["mode_diversity_loss"])
    assert torch.isfinite(terms["mode_orthogonality_loss"])
    assert torch.isfinite(loss)
    for module in (
        model.context_encoder,
        model.temporal_encoder,
        model.v51_contextual_dynamics,
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_temporal_encoder.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.v51_contextual_dynamics.target_state_encoder.parameters()
    )


def test_v53_autonomous_rollout_drift_is_independent_of_requested_horizon():
    torch.manual_seed(23)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=6,
        num_modes=5,
        active_modes=2,
        autonomous_rollout_enabled=True,
        dropout=0.0,
    ).eval()
    state = torch.randn(2, 6)
    memory = torch.randn(2, 6)
    static = torch.randn(2, 6)
    horizon = torch.randn(2, 6)
    first = model._motion(state, memory, static, horizon)[0]
    second = model._motion(state, memory, static, horizon * 7.0 + 2.0)[0]
    assert torch.allclose(first, second)


def test_v53_autonomous_rollout_has_no_terminal_horizon_shortcut():
    torch.manual_seed(29)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=6,
        num_modes=5,
        active_modes=2,
        base_step_ps=1.0,
        max_rollout_steps=8,
        autonomous_rollout_enabled=True,
        horizon_calibration_enabled=False,
        dropout=0.0,
    ).eval()
    state = torch.randn(1, 6)
    memory = torch.randn(1, 6)
    static = torch.randn(1, 6)
    short = model._rollout(state, memory, static, torch.tensor([2.0]))[0]
    long = model._rollout(state, memory, static, torch.tensor([4.0]))[0]
    assert torch.isfinite(short).all()
    assert torch.isfinite(long).all()
    assert not torch.allclose(short, long)


def test_v53_frame_local_target_ignores_temporal_segment_offset():
    torch.manual_seed(31)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=6,
        num_modes=5,
        active_modes=2,
        base_step_ps=1.0,
        max_rollout_steps=8,
        frame_local_target_enabled=True,
        dropout=0.0,
    ).eval()
    context_states = torch.randn(2, 5, 8)
    future_states = context_states + 100.0
    local_states = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    time_ps = torch.arange(5).float().unsqueeze(0).expand(2, -1)
    outputs = model(
        context_states,
        context_states,
        future_states,
        mask,
        mask,
        time_ps,
        time_ps,
        torch.tensor([2.0, 4.0]),
        torch.tensor([2, 4]),
        context_local_states=local_states,
        teacher_context_local_states=local_states,
        future_local_states=local_states,
    )

    assert torch.allclose(outputs["v51_online_target_delta"], torch.zeros(2, 6), atol=1.0e-7)
    assert not torch.allclose(outputs["v51_history_state"], outputs["v51_current_feature"])


def test_v53_frame_local_target_uses_one_ema_coordinate_system():
    torch.manual_seed(37)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=6,
        num_modes=5,
        active_modes=2,
        base_step_ps=1.0,
        max_rollout_steps=8,
        frame_local_target_enabled=True,
        dropout=0.0,
    ).eval()
    temporal_states = torch.randn(2, 5, 8)
    teacher_local = torch.randn(2, 5, 8)
    tower_offset = torch.linspace(-0.75, 0.75, 8).view(1, 1, 8)
    online_local = teacher_local + tower_offset
    mask = torch.ones(2, 5, dtype=torch.bool)
    time_ps = torch.arange(5).float().unsqueeze(0).expand(2, -1)
    outputs = model(
        temporal_states,
        temporal_states,
        temporal_states,
        mask,
        mask,
        time_ps,
        time_ps,
        torch.tensor([2.0, 4.0]),
        torch.tensor([2, 4]),
        context_local_states=online_local,
        teacher_context_local_states=teacher_local,
        future_local_states=teacher_local,
    )

    assert outputs["v51_online_target_delta"].abs().max() > 1.0e-3
    assert torch.allclose(outputs["v51_target_delta"], torch.zeros(2, 6), atol=1.0e-7)
    assert torch.allclose(
        outputs["v51_target_feature"], outputs["v51_teacher_current_feature"], atol=1.0e-7
    )


def test_v53_history_reads_frame_local_motion_before_temporal_smoothing():
    torch.manual_seed(41)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=6,
        num_modes=5,
        active_modes=2,
        base_step_ps=1.0,
        max_rollout_steps=8,
        frame_local_target_enabled=True,
        dropout=0.0,
    ).eval()
    temporal_states = torch.zeros(2, 6, 8)
    first_history = torch.randn(1, 6, 8)
    second_history = first_history.clone()
    second_history[:, :-1] = first_history[:, :-1].flip(1)
    local_history = torch.cat([first_history, second_history], dim=0)
    future_local = local_history.clone()
    mask = torch.ones(2, 6, dtype=torch.bool)
    time_ps = torch.arange(6).float().unsqueeze(0).expand(2, -1)
    outputs = model(
        temporal_states,
        temporal_states,
        temporal_states,
        mask,
        mask,
        time_ps,
        time_ps,
        torch.tensor([2.0, 2.0]),
        torch.tensor([2, 2]),
        context_local_states=local_history,
        teacher_context_local_states=local_history,
        future_local_states=future_local,
    )

    assert torch.equal(local_history[0, -1], local_history[1, -1])
    assert not torch.allclose(
        outputs["v51_context_memory"][0], outputs["v51_context_memory"][1]
    )
    assert not torch.allclose(
        outputs["v51_router_probabilities"][0], outputs["v51_router_probabilities"][1]
    )


def test_v53_physical_anchor_is_a_stable_continuous_state():
    torch.manual_seed(43)
    model = ContextualSparseMotionDynamics(
        latent_dim=8,
        state_dim=10,
        num_modes=5,
        active_modes=2,
        base_step_ps=1.0,
        max_rollout_steps=8,
        frame_local_target_enabled=True,
        physical_state_anchor_enabled=True,
        learned_state_residual_scale=0.1,
        dropout=0.0,
    ).eval()
    local = torch.randn(2, 5, 8)
    physical_context = torch.randn(2, 5, 10)
    physical_future = physical_context.clone()
    physical_future[:, -1] += torch.linspace(-0.2, 0.2, 10)
    mask = torch.ones(2, 5, dtype=torch.bool)
    time_ps = torch.arange(5).float().unsqueeze(0).expand(2, -1)
    outputs = model(
        torch.zeros_like(local),
        torch.zeros_like(local),
        torch.zeros_like(local),
        mask,
        mask,
        time_ps,
        time_ps,
        torch.tensor([2.0, 4.0]),
        torch.tensor([2, 4]),
        context_local_states=local,
        teacher_context_local_states=local,
        future_local_states=local,
        context_physical_states=physical_context,
        future_physical_states=physical_future,
    )

    expected_delta = physical_future[:, -1] - physical_context[:, -1]
    assert torch.allclose(outputs["v51_target_delta"], expected_delta, atol=1.0e-6)
    assert torch.allclose(
        outputs["v51_teacher_target_feature"] - outputs["v51_teacher_current_feature"],
        expected_delta,
        atol=1.0e-6,
    )


def test_closed_local_state_sequence_matches_endpoint_state():
    batch = _paired_batch()
    sequence, sequence_mask = compute_v3_closed_local_state_sequence(
        batch["context_coords"], batch["context_atom_mask"], batch["context_residue_mask"],
        batch["context_chain_id"], batch["context_residue_index"], num_tokens=2,
    )
    endpoint, endpoint_mask = compute_v3_closed_local_state(
        batch["context_coords"], batch["context_atom_mask"], batch["context_residue_mask"],
        batch["context_chain_id"], batch["context_residue_index"], num_tokens=2,
    )

    assert sequence.shape == (2, 8, 2, 12)
    assert torch.allclose(sequence[:, -1], endpoint)
    assert torch.equal(sequence_mask[:, -1], endpoint_mask)


def test_closed_local_velocity_is_causal_standardized_and_multiscale():
    batch = _paired_batch()
    current, _ = compute_v3_closed_local_state(
        batch["context_coords"], batch["context_atom_mask"], batch["context_residue_mask"],
        batch["context_chain_id"], batch["context_residue_index"], num_tokens=2,
    )
    velocity, mask = compute_v3_closed_local_velocity(
        batch["context_coords"], batch["context_atom_mask"], batch["context_residue_mask"],
        batch["context_chain_id"], batch["context_residue_index"], current,
        torch.ones(12), strides=(1, 2), num_tokens=2,
    )

    assert velocity.shape == (2, 2, 2, 12)
    assert mask.shape == (2, 2)
    assert torch.isfinite(velocity).all()


def test_closed_local_horizon_difference_cancels_shared_state_offset():
    target = torch.tensor([[[1.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]])
    prediction = (target + torch.tensor([[[5.0, -2.0]]])).requires_grad_()
    outputs = {
        "closed_local_prediction": prediction,
        "future_closed_local_state": target,
        "context_closed_local_state": torch.zeros_like(target),
        "closed_local_mask": torch.ones(3, 1, dtype=torch.bool),
        "closed_local_state_scale": torch.tensor([1.0, 2.0]),
        "context_group_index": torch.tensor([0, 0, 0]),
        "horizon_frames": torch.tensor([8, 12, 16]),
    }

    terms = closed_local_state_terms(outputs, difference_min_gap=8)
    terms["difference_loss"].backward()

    assert terms["state_loss"] > 0
    assert terms["difference_loss"].item() == pytest.approx(0.0)
    assert terms["target_difference_mse"] > 0
    assert prediction.grad is not None


def test_closed_local_robust_terms_limit_target_energy_and_train_uncertainty():
    target = torch.tensor([[[0.0]], [[100.0]]])
    prediction = torch.zeros_like(target, requires_grad=True)
    log_sigma = torch.zeros_like(target, requires_grad=True)
    outputs = {
        "closed_local_prediction": prediction,
        "future_closed_local_state": target,
        "context_closed_local_state": torch.zeros_like(target),
        "closed_local_mask": torch.ones(2, 1, dtype=torch.bool),
        "closed_local_state_scale": torch.ones(1),
        "closed_local_log_sigma": log_sigma,
        "context_group_index": torch.tensor([0, 0]),
        "horizon_frames": torch.tensor([8, 16]),
    }

    terms = closed_local_state_terms(outputs, difference_min_gap=1, robust_beta=0.5)
    (terms["robust_state_loss"] + terms["robust_difference_loss"] + terms["student_t_nll"]).backward()

    assert terms["difference_loss"].item() == pytest.approx(10000.0)
    assert terms["robust_difference_loss"].item() < 1.0
    assert torch.isfinite(terms["student_t_nll"])
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert log_sigma.grad is not None and log_sigma.grad.abs().sum() > 0


def test_closed_local_state_terms_standardize_channels_and_grouped_ranking():
    prediction = torch.tensor(
        [[[2.0, 20.0]], [[3.0, 10.0]]], requires_grad=True
    )
    target = torch.tensor([[[1.0, 10.0]], [[2.0, 20.0]]])
    current = torch.zeros_like(target)
    outputs = {
        "closed_local_prediction": prediction,
        "future_closed_local_state": target,
        "context_closed_local_state": current,
        "closed_local_mask": torch.ones(2, 1, dtype=torch.bool),
        "closed_local_state_scale": torch.tensor([1.0, 10.0]),
        "context_group_index": torch.tensor([0, 0]),
    }

    terms = closed_local_state_terms(outputs, margin=0.05)
    (terms["state_loss"] + terms["ranking_loss"]).backward()

    assert terms["state_loss"].item() == pytest.approx(1.0)
    assert torch.nn.functional.mse_loss(prediction.detach(), target).item() == pytest.approx(50.5)
    assert torch.isfinite(terms["ranking_loss"])
    assert prediction.grad is not None
    assert prediction.grad.abs().sum() > 0


def test_v32_checkpoint_without_state_scale_loads_non_strictly():
    config = ModelConfig(
        latent_dim=16,
        encoder_depth=1,
        temporal_depth=1,
        paired_forecast=True,
        local_dynamics_enabled=True,
        local_dynamics_dim=8,
        closed_local_state_enabled=True,
        closed_local_horizons=(8, 12, 16),
        num_heads=2,
        num_rbf=4,
    )
    historical_state = ProteinJEPA(config).state_dict()
    historical_state.pop("closed_local_operator.state_scale")

    restored = ProteinJEPA(config)
    incompatible = restored.load_state_dict(historical_state, strict=False)

    assert incompatible.missing_keys == ["closed_local_operator.state_scale"]
    assert not incompatible.unexpected_keys
    assert torch.allclose(restored.closed_local_operator.state_scale, torch.ones(12))


def test_closed_local_eval_reports_raw_and_standardized_controls():
    metrics = _closed_local_metrics({
        "values": 4.0,
        "pred_sse": 8.0,
        "persistence_sse": 10.0,
        "normalized_pred_sse": 2.0,
        "normalized_persistence_sse": 4.0,
        "shuffled_history_sse": 12.0,
        "normalized_shuffled_history_sse": 6.0,
        "shuffled_horizon_sse": 9.0,
        "normalized_shuffled_horizon_sse": 3.0,
        "student_t_nll_sum": 8.0,
        "persistence_student_t_nll_sum": 10.0,
        "normalized_scale_sum": 6.0,
        "coverage_1scale": 3.0,
        "coverage_2scale": 4.0,
        "uncertainty_values": 4.0,
        "normalized_fixed_48_sse": 2.4,
        "normalized_fixed_96_sse": 3.2,
    })

    assert metrics["mse"] == pytest.approx(2.0)
    assert metrics["persistence_relative_skill"] == pytest.approx(0.2)
    assert metrics["normalized_mse"] == pytest.approx(0.5)
    assert metrics["normalized_persistence_relative_skill"] == pytest.approx(0.5)
    assert metrics["normalized_shuffled_history_mse"] == pytest.approx(1.5)
    assert metrics["normalized_shuffled_horizon_mse"] == pytest.approx(0.75)
    assert metrics["student_t_nll"] == pytest.approx(2.0)
    assert metrics["mean_normalized_scale"] == pytest.approx(1.5)
    assert metrics["coverage_1scale"] == pytest.approx(0.75)
    assert metrics["best_fixed_horizon"] == 48
    assert metrics["correct_vs_best_fixed_skill"] == pytest.approx(1.0 - 2.0 / 2.4)


def test_closed_local_pair_eval_scores_against_zero_difference():
    metrics = _closed_local_pair_metrics({
        "values": 4.0,
        "error_sse": 2.0,
        "target_sse": 8.0,
        "normalized_error_sse": 1.0,
        "normalized_target_sse": 4.0,
    })

    assert metrics["difference_relative_skill"] == pytest.approx(0.75)
    assert metrics["normalized_difference_relative_skill"] == pytest.approx(0.75)


def test_v4_slow_eval_reports_fixed_depth_control():
    metrics = _v4_slow_metrics({
        "values": 8.0,
        "rows": 2.0,
        "prediction_sse": 4.0,
        "persistence_sse": 8.0,
        "lambda2_sum": 1.6,
        "shuffled_history_sse": 6.0,
        "shuffled_horizon_sse": 5.0,
        "fixed_48_sse": 4.8,
        "fixed_96_sse": 5.6,
    })

    assert metrics["persistence_relative_skill"] == pytest.approx(0.5)
    assert metrics["mean_lambda2"] == pytest.approx(0.8)
    assert metrics["best_fixed_horizon"] == 48
    assert metrics["correct_vs_best_fixed_skill"] == pytest.approx(1.0 - 4.0 / 4.8)
