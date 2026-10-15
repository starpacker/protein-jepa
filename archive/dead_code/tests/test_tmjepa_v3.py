import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import (
    ProteinJEPA,
    _trajectory_centered_learned_cv_terms,
    tmjepa_terms,
    upgrade_tmjepa_v3_state_dict,
)
from protein_jepa_frame.models.thermodynamic_v3 import (
    EyringEinsteinKinetics,
    HeatCapacityFreeEnergy,
    ProjectedForceCorrector,
    ThermodynamicMoriDynamicsV3,
    _latent_history_features,
)
from protein_jepa_frame.training.train import _tmjepa_optimizer_groups
from scripts.evaluate_tmjepa_v3 import _diffusion_gates, _first_context_rows


def test_v3_diffusion_promotion_rejects_positive_but_degenerate_tensor():
    degenerate = _diffusion_gates(9.0e-8, 243.0, 0.005, 0.01, 100.0)
    practical = _diffusion_gates(1.3e-6, 25.0, 0.02, 0.01, 100.0)

    assert degenerate["diffusion_psd"]
    assert not degenerate["diffusion_nondegenerate"]
    assert not degenerate["diffusion_well_conditioned"]
    assert all(practical.values())


def test_v3_latent_history_summary_respects_valid_frames():
    states = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    mask = torch.tensor([[True, True, True, True, False]])

    features = _latent_history_features(states, mask)

    last = states[:, 3]
    mean = states[:, :4].mean(dim=1)
    expected = torch.cat(
        [last, last - mean, torch.zeros_like(last), last - states[:, 2]],
        dim=-1,
    )
    assert features.shape == (1, 8)
    assert torch.equal(features, expected)


def test_v3_projected_force_has_quadratic_short_time_gate():
    corrector = ProjectedForceCorrector(
        cv_dim=2,
        condition_dim=4,
        auxiliary_dim=1,
        enabled=True,
        max_shift=1.0,
        horizon_gate_steps=2.0,
    )
    with torch.no_grad():
        corrector.net[-1].bias.fill_(0.5)
    q = torch.zeros(3, 2)
    shift = corrector(
        q,
        torch.zeros(3, 1),
        torch.zeros(3, 4),
        torch.full((3,), 348.0),
        torch.tensor([0.0, 1000.0, 8000.0]),
    )

    assert torch.count_nonzero(shift[0]) == 0
    assert torch.all(shift[1].abs() < shift[2].abs())


def test_v3_heat_capacity_basis_reduces_at_reference_temperature():
    model = HeatCapacityFreeEnergy(
        cv_dim=3,
        condition_dim=8,
        reference_temperature_k=348.0,
        heat_capacity_scale_over_kb=5.0,
    )
    with torch.no_grad():
        model.reference_enthalpy_over_kbt_head.bias.fill_(2.0)
        model.reference_entropy_over_kb_head.bias.fill_(0.5)
        model.heat_capacity_over_kb_head.bias.fill_(torch.atanh(torch.tensor(0.4)))
    q = torch.zeros(2, 3)
    condition = torch.zeros(2, 8)
    temperature = torch.tensor([348.0, 413.0])
    values = model(q, condition, temperature)

    assert torch.allclose(values["beta_free_energy"][0], torch.tensor(1.5))
    assert torch.allclose(values["heat_capacity_over_kb"], torch.tensor([2.0, 2.0]))
    assert values["beta_free_energy"][1] != values["beta_free_energy"][0]


def test_v3_eyring_clock_and_exact_ito_divergence():
    torch.manual_seed(2)
    kinetics = EyringEinsteinKinetics(
        cv_dim=3,
        condition_dim=8,
        auxiliary_dim=2,
        diffusion_rank=2,
        memory_enabled=True,
        diffusion_floor=1.0e-8,
        diffusion_scale=1.0e-5,
        coupling_scale=1.0e-3,
        reference_temperature_k=348.0,
        activation_energy_kcal_per_mol=3.0,
        activation_deviation_kcal_per_mol=0.0,
        state_mobility_scale=0.5,
        memory_times_ps=(1000.0, 4000.0),
    )
    with torch.no_grad():
        kinetics.mobility_head[-1].bias[3:6].fill_(0.4)
    q = torch.tensor([[0.2, -0.3, 0.1]], requires_grad=True)
    condition = torch.zeros(1, 8)
    low = kinetics(q, condition, torch.tensor([320.0]))
    high = kinetics(q, condition, torch.tensor([413.0]))
    automatic = torch.autograd.grad(
        torch.diagonal(low["diffusion"], dim1=-2, dim2=-1).sum(), q
    )[0]

    assert torch.allclose(low["diffusion_divergence"], automatic, atol=1.0e-10)
    assert high["diagonal_diffusion_rates"].mean() > 2.0 * low[
        "diagonal_diffusion_rates"
    ].mean()
    assert high["relaxation_rates"].mean() > 2.0 * low[
        "relaxation_rates"
    ].mean()
    assert high["memory_clock"].shape == (1, 2)
    gram = torch.einsum(
        "bqr,bqs->brs", low["coupling_basis"], low["coupling_basis"]
    )
    assert gram[0, 0, 1].abs() > 1.0e-3


def test_v3_coupled_bath_preserves_fdt_and_diagonal_limit():
    torch.manual_seed(12)
    diagonal = ThermodynamicMoriDynamicsV3(
        cv_dim=2,
        condition_dim=8,
        auxiliary_dim=2,
        diffusion_rank=1,
        memory_enabled=True,
        coupled_bath_enabled=False,
        memory_times_ps=(1000.0, 4000.0),
        base_step_ps=1000.0,
        ensemble_size=3,
    ).eval()
    coupled = ThermodynamicMoriDynamicsV3(
        cv_dim=2,
        condition_dim=8,
        auxiliary_dim=2,
        diffusion_rank=1,
        memory_enabled=True,
        coupled_bath_enabled=True,
        memory_times_ps=(1000.0, 4000.0),
        base_step_ps=1000.0,
        ensemble_size=3,
    ).eval()
    coupled.load_state_dict(diagonal.state_dict(), strict=True)
    with torch.no_grad():
        coupled.kinetics.memory_rotation_raw.zero_()
    q = torch.tensor([[0.2, -0.1]])
    auxiliary = torch.tensor([[0.3, -0.2]])
    condition = torch.zeros(1, 8)
    temperature = torch.tensor([348.0])
    dt = torch.tensor([1000.0])
    diagonal_step = diagonal._local_transition(
        q, auxiliary, condition, temperature, dt, create_graph=False
    )
    coupled_step = coupled._local_transition(
        q, auxiliary, condition, temperature, dt, create_graph=False
    )
    for key in (
        "q_mean_increment",
        "auxiliary_mean",
        "q_covariance",
        "q_auxiliary_covariance",
        "auxiliary_covariance",
    ):
        assert torch.allclose(diagonal_step[key], coupled_step[key], atol=2.0e-5)

    with torch.no_grad():
        coupled.kinetics.memory_rotation_raw[0, 1] = 0.8
    kinetic = coupled.kinetic_parameters(condition, temperature, q)
    generator = kinetic["auxiliary_generator"]
    noise_rate = kinetic["auxiliary_noise_rate"]
    assert kinetic["auxiliary_rotation"].abs().max() > 0.0
    assert torch.allclose(
        generator + generator.transpose(-1, -2), noise_rate, atol=1.0e-10
    )


def test_v3_physical_filter_whitens_matching_gle_process():
    torch.manual_seed(9)
    batch = 384
    steps = 8
    model = ThermodynamicMoriDynamicsV3(
        cv_dim=2,
        condition_dim=8,
        auxiliary_dim=1,
        diffusion_rank=1,
        memory_enabled=True,
        memory_times_ps=(3000.0,),
        base_step_ps=1000.0,
        max_rollout_steps=16,
        ensemble_size=3,
    ).eval()
    q = torch.zeros(batch, 2)
    auxiliary = torch.randn(batch, 1)
    condition = torch.zeros(batch, 8)
    temperature = torch.full((batch,), 348.0)
    dt = torch.full((batch,), 1000.0)
    sequence = [q]
    identity = torch.eye(3).unsqueeze(0)
    for _ in range(steps - 1):
        local = model._local_transition(
            q, auxiliary, condition, temperature, dt, create_graph=False
        )
        joint_covariance = torch.cat(
            [
                torch.cat(
                    [local["q_covariance"], local["q_auxiliary_covariance"]], -1
                ),
                torch.cat(
                    [
                        local["q_auxiliary_covariance"].transpose(-1, -2),
                        local["auxiliary_covariance"],
                    ],
                    -1,
                ),
            ],
            -2,
        )
        noise = torch.einsum(
            "bij,bj->bi",
            torch.linalg.cholesky(joint_covariance + 1.0e-8 * identity),
            torch.randn(batch, 3),
        )
        q = q + local["q_mean_increment"] + noise[:, :2]
        auxiliary = local["auxiliary_mean"] + noise[:, 2:]
        sequence.append(q)
    sequence = torch.stack(sequence, dim=1)
    filtered = model.filter(
        sequence,
        torch.ones(batch, steps, dtype=torch.bool),
        torch.arange(steps, dtype=torch.float32).expand(batch, -1) * 1000.0,
        condition,
        temperature,
    )
    innovation = filtered["innovation"].reshape(-1, 2)
    centered = innovation - innovation.mean(0)
    covariance = centered.T @ centered / (innovation.shape[0] - 1)
    lagged = filtered["innovation"][:, 1:].reshape(-1, 2)
    previous = filtered["innovation"][:, :-1].reshape(-1, 2)
    lag_covariance = (lagged - lagged.mean(0)).T @ (
        previous - previous.mean(0)
    ) / (lagged.shape[0] - 1)

    assert torch.allclose(covariance, torch.eye(2), atol=0.08)
    assert lag_covariance.abs().max() < 0.08


def test_v3_observation_error_filter_whitens_ma1_increments():
    torch.manual_seed(21)
    batch, steps = 512, 8
    model = ThermodynamicMoriDynamicsV3(
        cv_dim=2,
        condition_dim=8,
        auxiliary_dim=1,
        diffusion_rank=1,
        memory_enabled=False,
        coupled_bath_enabled=True,
        observation_noise_enabled=True,
        observation_noise_scale=0.1,
        memory_times_ps=(3000.0,),
        base_step_ps=1000.0,
        ensemble_size=3,
        diffusion_floor=1.0e-12,
        diffusion_scale=1.0e-12,
    ).eval()
    with torch.no_grad():
        for parameter in model.potential.parameters():
            parameter.zero_()
        model.kinetics.observation_noise_raw.fill_(
            torch.log(torch.expm1(torch.tensor(0.1)))
        )
    condition = torch.zeros(batch, 8)
    temperature = torch.full((batch,), 348.0)
    observation_std = 0.1
    sequence = observation_std * torch.randn(batch, steps, 2)
    filtered = model.filter(
        sequence,
        torch.ones(batch, steps, dtype=torch.bool),
        torch.arange(steps, dtype=torch.float32).expand(batch, -1) * 1000.0,
        condition,
        temperature,
    )
    innovation = filtered["innovation"]
    values = innovation.reshape(-1, 2)
    covariance = values.T @ values / values.shape[0]
    lagged = innovation[:, 1:].reshape(-1, 2)
    previous = innovation[:, :-1].reshape(-1, 2)
    lag_covariance = lagged.T @ previous / lagged.shape[0]
    assert torch.allclose(covariance, torch.eye(2), atol=0.1)
    assert lag_covariance.abs().max() < 0.08


def test_v3_correlated_observation_noise_is_psd_and_temperature_scaled():
    torch.manual_seed(22)
    kinetics = EyringEinsteinKinetics(
        cv_dim=4,
        condition_dim=8,
        auxiliary_dim=1,
        diffusion_rank=1,
        memory_enabled=False,
        diffusion_floor=1.0e-8,
        diffusion_scale=1.0e-5,
        coupling_scale=1.0e-3,
        reference_temperature_k=348.0,
        activation_energy_kcal_per_mol=3.0,
        activation_deviation_kcal_per_mol=1.0,
        state_mobility_scale=0.5,
        memory_times_ps=(1000.0,),
        observation_noise_enabled=True,
        observation_noise_scale=0.1,
        observation_noise_rank=2,
        observation_noise_condition_scale=0.5,
    )
    with torch.no_grad():
        kinetics.observation_noise_mode_raw.fill_(
            torch.log(torch.expm1(torch.tensor(0.2)))
        )
        kinetics.observation_noise_head[-1].bias.fill_(0.5)
    q = torch.zeros(2, 4)
    condition = torch.zeros(2, 8)
    values = kinetics(q, condition, torch.tensor([320.0, 413.0]))
    covariance = values["observation_noise_covariance"]

    assert torch.linalg.eigvalsh(covariance).min() > 0.0
    off_diagonal = covariance - torch.diag_embed(
        torch.diagonal(covariance, dim1=-2, dim2=-1)
    )
    assert off_diagonal.abs().max() > 0.0
    assert covariance[1].trace() > covariance[0].trace()


def test_v3_integrated_protein_jepa_backward():
    data_config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=4,
        synthetic_residues=12,
    )
    batch = next(
        iter(
            DataLoader(
                build_dataset(data_config),
                batch_size=2,
                collate_fn=collate_paired_windows,
            )
        )
    )
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_memory_times_ps=(1.0, 4.0),
            tmjepa_memory_intervention_enabled=True,
            tmjepa_base_step_ps=1.0,
            tmjepa_max_rollout_steps=16,
            tmjepa_ensemble_size=3,
        )
    )
    outputs = model.forward_tmjepa(batch)
    terms = tmjepa_terms(outputs)
    loss = terms["energy_score"] + 0.01 * terms["path_nll"]
    loss.backward()

    assert torch.all(outputs["tmjepa_v3"] == 1.0)
    assert torch.isfinite(loss)
    assert torch.linalg.eigvalsh(outputs["tmjepa_diffusion"]).min() > 0.0
    assert outputs["tmjepa_heat_capacity_over_kb"].shape == (2,)
    assert outputs["tmjepa_memory_activation_energy_kcal_per_mol"].shape == (2, 2)
    assert torch.isfinite(terms["memory_intervention_loss"])
    assert torch.isfinite(terms["memory_intervention_advantage"])
    assert torch.isfinite(terms["memory_residual_loss"])
    assert torch.isfinite(terms["memory_long_horizon_residual_loss"])
    assert torch.isfinite(terms["memory_ratio_band_loss"])
    assert torch.isfinite(terms["memory_log_ratio_loss"])
    assert torch.isfinite(terms["long_horizon_energy_score"])
    expected_persistence = torch.linalg.vector_norm(
        outputs["tmjepa_observed_current_cv"] - outputs["tmjepa_target_cv"],
        dim=-1,
    ).mean()
    assert torch.allclose(terms["persistence_energy_score"], expected_persistence)
    assert torch.equal(
        outputs["tmjepa_score_base_cv"], outputs["tmjepa_observed_current_cv"]
    )


def test_v3_learned_cv_uses_jepa_encoders_and_fixed_history_gauge():
    data_config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=4,
        synthetic_residues=12,
    )
    batch = next(
        iter(
            DataLoader(
                build_dataset(data_config),
                batch_size=2,
                collate_fn=collate_paired_windows,
            )
        )
    )
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            temporal_causal=True,
            temporal_causal_window=4,
            temporal_position_encoding="none",
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_learned_cv_dim=2,
            tmjepa_learned_cv_burn_in_frames=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_enabled=False,
            tmjepa_memory_times_ps=(1.0, 4.0),
            tmjepa_base_step_ps=1.0,
            tmjepa_max_rollout_steps=16,
            tmjepa_ensemble_size=3,
            tmjepa_cv_mean=(0.0,) * 16,
            tmjepa_cv_scale=(1.0,) * 16,
        )
    )

    outputs = model.forward_tmjepa(batch)
    terms = tmjepa_terms(outputs, vamp_lags=(1, 2))
    loss = terms["learned_cv_variance_loss"] + 0.01 * terms["learned_cv_covariance_loss"] + terms["vamp_loss"]
    loss.backward()

    assert outputs["tmjepa_current_cv"].shape == (2, 18)
    assert outputs["tmjepa_context_cv_sequence"].shape[1:] == (5, 18)
    assert outputs["tmjepa_current_physical_cv"].shape == (2, 16)
    assert model.context_encoder.geom_proj.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.temporal_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.tmjepa.learned_projection.parameters())
    assert all(parameter.grad is None for parameter in model.tmjepa.target_learned_projection.parameters())


def test_learned_cv_vamp_does_not_reward_static_protein_identity():
    learned = torch.tensor(
        [
            [[-0.8, 0.4]] * 8,
            [[0.9, -0.3]] * 8,
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(2, 8, dtype=torch.bool)

    variance, covariance, vamp_loss, vamp2 = _trajectory_centered_learned_cv_terms(
        learned, mask, (1, 2, 4)
    )

    assert variance > 0.2
    assert covariance == 0.0
    assert vamp_loss == 0.0
    assert vamp2 == 0.0


def test_v3_endpoint_observation_model_is_reusable_for_interventions():
    torch.manual_seed(31)
    wrapper = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=1,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_times_ps=(3000.0,),
            tmjepa_observation_noise_enabled=True,
        )
    ).tmjepa
    samples = torch.zeros(1024, 3, wrapper.cv_dim)
    covariance = torch.eye(wrapper.cv_dim).expand(1024, -1, -1) * 0.04
    observed = wrapper.observe_samples(samples, covariance)

    assert observed.std(unbiased=False) > 0.15
    assert observed.std(unbiased=False) < 0.25


def test_v3_evaluation_deduplicates_all_horizon_contexts():
    batch = {"context_group_index": torch.tensor([0, 0, 0, 1, 1, 1])}

    rows = _first_context_rows(batch, row_count=6, device=torch.device("cpu"))

    assert torch.equal(rows, torch.tensor([0, 3]))


def test_v3_diffusion_and_memory_freezing_are_independent():
    diffusion_frozen = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_times_ps=(1000.0, 4000.0),
            tmjepa_freeze_diffusion=True,
        )
    ).tmjepa.dynamics.kinetics
    assert not diffusion_frozen.activation_energy_raw.requires_grad
    assert diffusion_frozen.memory_activation_energy_raw.requires_grad
    assert diffusion_frozen.memory_activation_deviation.requires_grad

    memory_frozen = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_times_ps=(1000.0, 4000.0),
            tmjepa_freeze_memory=True,
        )
    ).tmjepa.dynamics.kinetics
    assert memory_frozen.activation_energy_raw.requires_grad
    assert not memory_frozen.memory_activation_energy_raw.requires_grad
    assert not memory_frozen.memory_activation_deviation.requires_grad

    observation_frozen = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_times_ps=(1000.0, 4000.0),
            tmjepa_observation_noise_rank=2,
            tmjepa_freeze_observation_noise=True,
        )
    ).tmjepa.dynamics.kinetics
    assert observation_frozen.activation_energy_raw.requires_grad
    assert not observation_frozen.observation_noise_raw.requires_grad
    assert not observation_frozen.observation_noise_basis.requires_grad
    assert all(
        not parameter.requires_grad
        for parameter in observation_frozen.observation_noise_head.parameters()
    )


def test_v3_qr_optimizer_groups_are_disjoint_and_scaled():
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_times_ps=(1000.0, 4000.0),
            tmjepa_observation_noise_rank=2,
        )
    )
    groups = _tmjepa_optimizer_groups(
        model,
        1.0e-3,
        0.05,
        0.5,
        projected_force_lr_scale=4.0,
    )
    by_name = {group["group_name"]: group for group in groups}
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]

    assert len(parameter_ids) == len(set(parameter_ids))
    assert by_name["tmjepa_diffusion"]["lr"] == 5.0e-5
    assert by_name["tmjepa_observation_noise"]["lr"] == 5.0e-4
    assert by_name["tmjepa_projected_force"]["lr"] == 4.0e-3
    assert id(model.tmjepa.projected_force_corrector.net[-1].weight) in {
        id(parameter) for parameter in by_name["tmjepa_projected_force"]["params"]
    }
    assert id(model.tmjepa.dynamics.kinetics.activation_energy_raw) in {
        id(parameter) for parameter in by_name["tmjepa_diffusion"]["params"]
    }
    assert id(model.tmjepa.dynamics.kinetics.observation_noise_raw) in {
        id(parameter) for parameter in by_name["tmjepa_observation_noise"]["params"]
    }


def test_v3_learned_cv_has_an_independent_diffusion_floor():
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            temporal_causal=True,
            temporal_causal_window=4,
            predictor_depth=1,
            paired_forecast=True,
            num_heads=4,
            num_rbf=8,
            tmjepa_enabled=True,
            tmjepa_version=3,
            tmjepa_learned_cv_dim=2,
            tmjepa_learned_cv_burn_in_frames=3,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_diffusion_rank=1,
            tmjepa_memory_times_ps=(1000.0, 4000.0),
            tmjepa_learned_cv_diffusion_floor_scale=0.2,
        )
    )
    scale = model.tmjepa.dynamics.kinetics.diffusion_floor_scale

    assert torch.all(scale[: model.tmjepa.fixed_cv_dim] == 1.0)
    assert torch.all(scale[model.tmjepa.fixed_cv_dim :] == 0.2)


def test_v3_checkpoint_upgrade_adds_independent_memory_clock():
    config = ModelConfig(
        latent_dim=16,
        encoder_depth=1,
        temporal_depth=1,
        predictor_depth=1,
        paired_forecast=True,
        num_heads=4,
        num_rbf=8,
        tmjepa_enabled=True,
        tmjepa_version=3,
        tmjepa_condition_dim=16,
        tmjepa_auxiliary_dim=2,
        tmjepa_diffusion_rank=1,
        tmjepa_memory_times_ps=(1000.0, 4000.0),
    )
    source = ProteinJEPA(config)
    old_state = {
        key: value
        for key, value in source.state_dict().items()
        if "memory_activation_" not in key and not key.endswith("_memory_coupling_ramp")
    }
    target = ProteinJEPA(config)
    upgraded = upgrade_tmjepa_v3_state_dict(target, old_state)
    target.load_state_dict(upgraded, strict=True)
    kinetics = target.tmjepa.dynamics.kinetics
    assert torch.equal(
        kinetics.memory_activation_energy_raw,
        kinetics.activation_energy_raw,
    )
    assert torch.count_nonzero(kinetics.memory_activation_deviation) == 0
    assert torch.equal(kinetics._memory_coupling_ramp, torch.ones(1))


def test_v3_checkpoint_upgrade_adds_endpoint_calibrator_identity():
    config = ModelConfig(
        latent_dim=16,
        encoder_depth=1,
        temporal_depth=1,
        predictor_depth=1,
        paired_forecast=True,
        num_heads=4,
        num_rbf=8,
        tmjepa_enabled=True,
        tmjepa_version=3,
        tmjepa_condition_dim=16,
        tmjepa_auxiliary_dim=2,
        tmjepa_diffusion_rank=1,
        tmjepa_memory_times_ps=(1000.0, 4000.0),
        tmjepa_endpoint_calibration_enabled=True,
        tmjepa_endpoint_calibration_scale=0.25,
        tmjepa_endpoint_calibration_temperature_scale=1.0,
    )
    target = ProteinJEPA(config)
    old_state = {
        key: value
        for key, value in target.state_dict().items()
        if not key.startswith("tmjepa.endpoint_calibration.")
    }

    upgraded = upgrade_tmjepa_v3_state_dict(target, old_state)
    target.load_state_dict(upgraded, strict=True)
    log_scale = target.tmjepa.endpoint_calibration(
        torch.tensor([1000.0, 16000.0]),
        torch.tensor([320.0, 413.0]),
        torch.zeros(2, 3, target.tmjepa.cv_dim),
    )

    assert torch.allclose(log_scale, torch.zeros_like(log_scale))


def test_v3_checkpoint_upgrade_adds_correlated_observation_model():
    config = ModelConfig(
        latent_dim=16,
        encoder_depth=1,
        temporal_depth=1,
        predictor_depth=1,
        paired_forecast=True,
        num_heads=4,
        num_rbf=8,
        tmjepa_enabled=True,
        tmjepa_version=3,
        tmjepa_condition_dim=16,
        tmjepa_auxiliary_dim=2,
        tmjepa_diffusion_rank=1,
        tmjepa_memory_times_ps=(1000.0, 4000.0),
        tmjepa_observation_noise_enabled=True,
        tmjepa_observation_noise_rank=2,
    )
    target = ProteinJEPA(config)
    old_state = {
        key: value
        for key, value in target.state_dict().items()
        if not key.startswith("tmjepa.dynamics.kinetics.observation_noise_")
    }

    upgraded = upgrade_tmjepa_v3_state_dict(target, old_state)
    target.load_state_dict(upgraded, strict=True)
    covariance = target.tmjepa.dynamics.kinetic_parameters(
        torch.zeros(2, 16), torch.tensor([320.0, 413.0])
    )["observation_noise_covariance"]

    assert covariance.shape == (2, target.tmjepa.cv_dim, target.tmjepa.cv_dim)
    assert torch.linalg.eigvalsh(covariance).min() > 0.0


def test_v3_endpoint_calibration_rescales_samples_without_mean_shift():
    torch.manual_seed(44)
    data_config = DataConfig(
        training_mode="paired_window",
        batch_size=2,
        num_frames=8,
        paired_horizons_frames=(8,),
        paired_nonoverlap_only=True,
        synthetic_size=4,
        synthetic_residues=12,
    )
    batch = next(
        iter(
            DataLoader(
                build_dataset(data_config),
                batch_size=2,
                collate_fn=collate_paired_windows,
            )
        )
    )
    config = ModelConfig(
        latent_dim=16,
        encoder_depth=1,
        temporal_depth=1,
        predictor_depth=1,
        paired_forecast=True,
        num_heads=4,
        num_rbf=8,
        tmjepa_enabled=True,
        tmjepa_version=3,
        tmjepa_condition_dim=16,
        tmjepa_auxiliary_dim=2,
        tmjepa_diffusion_rank=1,
        tmjepa_memory_enabled=False,
        tmjepa_memory_times_ps=(1.0, 4.0),
        tmjepa_base_step_ps=1.0,
        tmjepa_max_rollout_steps=16,
        tmjepa_ensemble_size=8,
        tmjepa_endpoint_calibration_enabled=True,
        tmjepa_endpoint_calibration_scale=0.5,
    )
    model = ProteinJEPA(config).eval()
    baseline = ProteinJEPA(config).eval()
    baseline.load_state_dict(model.state_dict(), strict=True)
    baseline.tmjepa.endpoint_calibration.enabled = False
    with torch.no_grad():
        model.tmjepa.endpoint_calibration.bias.fill_(1.0)

    torch.manual_seed(123)
    calibrated = model.forward_tmjepa(batch)
    torch.manual_seed(123)
    uncalibrated = baseline.forward_tmjepa(batch)

    assert torch.allclose(
        calibrated["tmjepa_prediction"],
        uncalibrated["tmjepa_prediction"],
        atol=1.0e-6,
    )
    assert calibrated["tmjepa_sample_variance"].mean() > uncalibrated[
        "tmjepa_sample_variance"
    ].mean()
    assert calibrated["tmjepa_endpoint_calibration_log_scale"].mean() > 0.0
