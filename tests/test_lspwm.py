from __future__ import annotations

import torch

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.jepa import ProteinJEPA
from protein_jepa_frame.models.lspwm import (
    ConstantDiffusion,
    GaugeFixedBath,
    LSPWMConfig,
    LSPWMWorldModel,
    LSPWMProteinWorldModel,
    MeanForcePotential,
    PhysicalTransitionCalibration,
    FlowMatchObservationModel,
    PoissonOperator,
    DissipationOperator,
    fit_koopman_operator,
    empirical_fdt_residual,
    fdt_residual,
    energy_loop_circulation,
    kde_mean_force,
    koopman_diagnostics,
    mean_force_matching_loss,
    msm_consistency_loss,
    overdamped_transition_nll,
    calibrated_overdamped_transition_nll,
    calibrated_mean_force_matching_loss,
    vamp2_loss,
    asymmetric_cycle_loss,
    local_density_weights,
)


def test_constant_diffusion_is_state_independent_and_temperature_scaled():
    diffusion = ConstantDiffusion(
        3,
        reference_temperature_k=300.0,
        diffusion_scale=0.01,
    )
    q_a = torch.randn(4, 3)
    q_b = torch.randn(4, 3)
    low = diffusion(4, torch.full((4,), 300.0), reference=q_a)
    high = diffusion(4, torch.full((4,), 600.0), reference=q_b)
    assert torch.allclose(low[0], low[1])
    assert torch.allclose(high, 2.0 * low, rtol=1.0e-5, atol=1.0e-7)
    assert torch.linalg.eigvalsh(low).min() > 0.0


def test_gauge_fixed_bath_has_orthogonal_modes_and_ordered_rates():
    bath = GaugeFixedBath(
        5,
        3,
        relaxation_times_ps=(1.0, 2.0, 4.0),
    )
    params = bath.parameters_at(dtype=torch.float32, device=torch.device("cpu"))
    gram = params["basis"].T @ params["basis"]
    assert torch.allclose(gram, torch.eye(3), atol=1.0e-5)
    assert torch.all(params["rates"][1:] < params["rates"][:-1])
    kernel = bath(torch.tensor([0.0, 1.0, 10.0]))
    assert kernel.shape == (3, 5, 5)
    assert torch.linalg.eigvalsh(kernel).min() >= -1.0e-7


def test_kde_mean_force_has_gaussian_score_direction():
    torch.manual_seed(3)
    reference = torch.randn(4000, 2) * 0.5
    query = torch.tensor([[0.5, -0.25], [-0.5, 0.25]])
    force = kde_mean_force(query, reference, bandwidth=0.08)
    expected = -query / 0.25
    assert torch.allclose(force, expected, atol=0.45, rtol=0.2)


def test_adaptive_kde_bandwidth_ignores_exact_self_matches():
    reference = torch.tensor([[-2.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [2.0, 0.0]])
    force = kde_mean_force(reference[:3], reference, bandwidth=0.0)
    assert torch.isfinite(force).all()
    assert torch.linalg.vector_norm(force, dim=-1).sum() > 1.0e-4


def test_scalar_potential_force_matching_is_differentiable():
    torch.manual_seed(4)
    potential = MeanForcePotential(2, condition_dim=1, hidden_dim=16)
    q = torch.randn(8, 2)
    condition = torch.randn(8, 1)
    target = torch.randn(8, 2)
    loss = mean_force_matching_loss(potential, q, target, condition)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in potential.parameters())


def test_fdt_and_integrability_contracts_are_explicit():
    potential = MeanForcePotential(2, hidden_dim=16)
    loop = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )
    assert energy_loop_circulation(potential, loop) < 1.0e-6
    diffusion = torch.eye(2).expand(3, -1, -1) * 0.2
    covariance = 2.0 * diffusion * 0.5
    assert fdt_residual(diffusion, covariance, 0.5) < 1.0e-6


def test_overdamped_transition_nll_is_finite_and_trainable():
    torch.manual_seed(7)
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=3,
            condition_dim=0,
            auxiliary_dim=1,
            relaxation_times_ps=(2.0,),
            diffusion_scale=1.0e-2,
        )
    )
    q = torch.randn(6, 3)
    target = q + 0.01 * torch.randn_like(q)
    loss = overdamped_transition_nll(model, q, target, None, 0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_calibrated_transition_preserves_fdt_and_is_trainable():
    torch.manual_seed(17)
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=3,
            condition_dim=2,
            auxiliary_dim=1,
            relaxation_times_ps=(2.0,),
            diffusion_scale=1.0e-2,
            physical_calibration_enabled=True,
            physical_mobility_init=0.25,
            physical_free_energy_init=4.0,
            conservative_correction_enabled=True,
            conservative_correction_rank=2,
        )
    )
    q = torch.randn(6, 3)
    condition = torch.randn(6, 2)
    target = q + 0.01 * torch.randn_like(q)
    loss = calibrated_overdamped_transition_nll(
        model, q, target, condition, 0.1
    )
    loss.backward()
    calibration = model.physical_calibration
    assert torch.isfinite(loss)
    assert torch.allclose(
        calibration.mobility_scale, torch.tensor(0.25), atol=1.0e-6
    )
    assert torch.allclose(
        calibration.free_energy_scale, torch.tensor(4.0), atol=1.0e-6
    )
    assert calibration.log_mobility_scale.grad is not None
    assert calibration.log_free_energy_scale.grad is not None
    assert calibration.conservative_correction.bias.grad is not None
    parameters = model.calibrated_transition(
        q,
        condition,
        0.1,
    )
    base_diffusion = model.diffusion(
        q.shape[0], reference=q
    )
    assert torch.allclose(
        parameters["diffusion"],
        0.25 * base_diffusion,
        atol=1.0e-6,
    )


def test_conservative_correction_is_zero_at_initialization():
    correction = PhysicalTransitionCalibration(
        state_dim=4,
        condition_dim=3,
        enabled=True,
        conservative_enabled=True,
        conservative_rank=2,
    )
    q = torch.randn(5, 4)
    condition = torch.randn(5, 3)
    energy = correction.conservative_correction(q, condition)
    gradient = correction.correction_gradient(q, condition)
    assert torch.allclose(energy, torch.zeros_like(energy))
    assert torch.allclose(gradient, torch.zeros_like(gradient))


def test_calibrated_force_matching_uses_total_conservative_force():
    torch.manual_seed(19)
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=3,
            condition_dim=2,
            auxiliary_dim=1,
            relaxation_times_ps=(2.0,),
            physical_calibration_enabled=True,
            physical_mobility_init=0.5,
            physical_free_energy_init=3.0,
            conservative_correction_enabled=True,
            conservative_correction_rank=2,
        )
    )
    with torch.no_grad():
        model.physical_calibration.conservative_correction.bias.fill_(0.2)
    q = torch.randn(5, 3)
    condition = torch.randn(5, 2)
    target = torch.randn(5, 3)
    force = model.calibrated_mean_force(q, condition, 348.0, create_graph=True)
    loss = calibrated_mean_force_matching_loss(
        model, q, target, condition, 348.0
    )
    loss.backward()
    assert torch.isfinite(force).all()
    assert torch.isfinite(loss)
    assert model.physical_calibration.conservative_correction.bias.grad is not None


def test_conservative_correction_can_share_free_energy_scale():
    torch.manual_seed(23)
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=3,
            condition_dim=2,
            auxiliary_dim=1,
            relaxation_times_ps=(2.0,),
            physical_calibration_enabled=True,
            physical_mobility_init=0.5,
            physical_free_energy_init=3.0,
            conservative_correction_enabled=True,
            conservative_correction_rank=2,
            conservative_correction_rescale_with_free_energy=True,
        )
    )
    with torch.no_grad():
        model.physical_calibration.conservative_correction.bias.fill_(0.2)
    q = torch.randn(5, 3)
    condition = torch.randn(5, 2)
    base_force = model.potential.mean_force(
        q, condition, 348.0, create_graph=True
    )
    correction_gradient = model.physical_calibration.correction_gradient(
        q, condition, create_graph=True
    )
    expected = model.physical_calibration.free_energy_scale * (
        base_force - correction_gradient
    )
    actual = model.calibrated_mean_force(
        q, condition, 348.0, create_graph=True
    )
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_force_field_stage_can_freeze_validated_calibration_scales():
    model = LSPWMProteinWorldModel(
        latent_dim=4,
        learned_cv_dim=4,
        num_segments=3,
        condition_dim=8,
        auxiliary_dim=1,
        relaxation_times_ps=(2.0,),
        physical_calibration_enabled=True,
        conservative_correction_enabled=True,
        freeze_base_dynamics=True,
        freeze_calibration_scales=True,
    )
    calibration = model.world_model.physical_calibration
    assert not calibration.log_mobility_scale.requires_grad
    assert not calibration.log_free_energy_scale.requires_grad
    assert calibration.conservative_correction.bias.requires_grad
    assert model.calibration_parameters() == list(
        calibration.conservative_correction.parameters()
    )


def test_proteinjepa_freeze_contract_keeps_calibration_scales_frozen():
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=8,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            num_heads=1,
            max_frames=8,
            paired_forecast=True,
            temporal_causal=True,
            lspwm_enabled=True,
            lspwm_learned_cv_dim=8,
            lspwm_condition_dim=8,
            lspwm_koopman_role_dim=2,
            lspwm_msm_role_dim=2,
            lspwm_geometric_role_dim=2,
            lspwm_auxiliary_role_dim=2,
            lspwm_role_hidden_dim=8,
            lspwm_physical_calibration_enabled=True,
            lspwm_conservative_correction_enabled=True,
            lspwm_conservative_correction_rank=2,
            lspwm_freeze_base_dynamics=True,
            lspwm_freeze_calibration_scales=True,
        )
    )
    calibration = model.lspwm.world_model.physical_calibration
    assert not calibration.log_mobility_scale.requires_grad
    assert not calibration.log_free_energy_scale.requires_grad
    assert calibration.conservative_correction.bias.requires_grad


def test_koopman_diagnostics_recovers_linear_operator():
    torch.manual_seed(5)
    source = torch.randn(200, 2)
    operator = torch.tensor([[0.9, 0.1], [-0.05, 0.8]])
    target = source @ operator.T
    fit = fit_koopman_operator(source, target, ridge=1.0e-8)
    diagnostics = koopman_diagnostics(source, target, lag_time=2.0)
    assert torch.allclose(fit["operator"], operator, atol=2.0e-3)
    assert fit["residual_mse"] < 1.0e-6
    assert diagnostics["vamp2_score"] > 0.0
    assert diagnostics["eigenvalues"].shape == (2,)


def test_world_model_rollout_is_finite_and_has_expected_shapes():
    torch.manual_seed(6)
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=4,
            condition_dim=3,
            auxiliary_dim=2,
            relaxation_times_ps=(1.0, 4.0),
            diffusion_scale=1.0e-4,
        )
    )
    q = torch.randn(3, 4)
    condition = torch.randn(3, 3)
    result = model.rollout(q, condition, dt_ps=0.5, steps=5, stochastic=True)
    assert result["q"].shape == (3, 6, 4)
    assert result["auxiliary"].shape == (3, 6, 2)
    assert torch.isfinite(result["q"]).all()
    assert torch.isfinite(result["auxiliary"]).all()


def test_protein_bridge_builds_a_unified_256d_coarse_state():
    torch.manual_seed(8)
    model = LSPWMProteinWorldModel(
        latent_dim=256,
        learned_cv_dim=256,
        num_segments=3,
        condition_dim=8,
        auxiliary_dim=2,
        relaxation_times_ps=(1.0, 4.0),
        cv_mean=(0.0,) * 16,
        cv_scale=(1.0,) * 16,
        role_hidden_dim=64,
    )
    batch, frames, residues = 2, 5, 10
    coords = torch.randn(batch, frames, residues, 4, 3)
    atom_mask = torch.ones(batch, frames, residues, 4, dtype=torch.bool)
    residue_mask = torch.ones(batch, frames, residues, dtype=torch.bool)
    chain_id = torch.zeros(batch, residues, dtype=torch.long)
    residue_index = torch.arange(residues).expand(batch, -1)
    frame_states = torch.randn(batch, frames, 256)
    outputs = model(
        context_coords=coords,
        future_coords=coords + 0.01,
        context_atom_mask=atom_mask,
        future_atom_mask=atom_mask,
        context_residue_mask=residue_mask,
        future_residue_mask=residue_mask,
        context_chain_id=chain_id,
        future_chain_id=chain_id,
        context_residue_index=residue_index,
        future_residue_index=residue_index,
        context_frame_states=frame_states,
        future_frame_states=frame_states,
        context_time_ps=torch.arange(frames).expand(batch, -1).float(),
        horizon_time_ps=torch.tensor([1.0, 4.0]),
        aatype=torch.zeros(batch, residues, dtype=torch.long),
        temperature_k=torch.tensor([320.0, 348.0]),
        dataset_index=torch.ones(batch, dtype=torch.long),
    )
    assert outputs["lspwm_context_state"].shape == (batch, 256)
    assert outputs["lspwm_history_features"].shape == (batch, 768)
    assert outputs["lspwm_koopman_features"].shape == (batch, 64)
    assert outputs["lspwm_msm_logits"].shape == (batch, 32)
    assert outputs["lspwm_geometric_features"].shape == (batch, 96)
    assert outputs["lspwm_auxiliary_features"].shape == (batch, 64)
    for name in (
        "lspwm_msm_loss",
        "lspwm_geometric_loss",
        "lspwm_auxiliary_loss",
    ):
        assert torch.isfinite(outputs[name])
    assert torch.isfinite(outputs["lspwm_dynamic_nll"])
    assert torch.isfinite(outputs["lspwm_force_loss"])


def test_rollout_losses_include_observed_endpoint_and_variance_calibration():
    torch.manual_seed(18)
    model = LSPWMProteinWorldModel(
        latent_dim=16,
        learned_cv_dim=16,
        num_segments=3,
        condition_dim=8,
        auxiliary_dim=2,
        relaxation_times_ps=(1.0, 4.0),
        cv_mean=(0.0,) * 16,
        cv_scale=(1.0,) * 16,
        role_hidden_dim=32,
        rollout_enabled=True,
        rollout_steps=2,
        rollout_ensemble_size=2,
        rollout_intermediate_supervision=False,
    )
    batch, frames, residues = 2, 4, 8
    coords = torch.randn(batch, frames, residues, 4, 3)
    atom_mask = torch.ones(batch, frames, residues, 4, dtype=torch.bool)
    residue_mask = torch.ones(batch, frames, residues, dtype=torch.bool)
    chain_id = torch.zeros(batch, residues, dtype=torch.long)
    residue_index = torch.arange(residues).expand(batch, -1)
    frame_states = torch.randn(batch, frames, 16)
    outputs = model(
        context_coords=coords,
        future_coords=coords + 0.05,
        context_atom_mask=atom_mask,
        future_atom_mask=atom_mask,
        context_residue_mask=residue_mask,
        future_residue_mask=residue_mask,
        context_chain_id=chain_id,
        future_chain_id=chain_id,
        context_residue_index=residue_index,
        future_residue_index=residue_index,
        context_frame_states=frame_states,
        future_frame_states=frame_states + 0.03,
        context_time_ps=torch.arange(frames).expand(batch, -1).float(),
        horizon_time_ps=torch.tensor([4.0, 4.0]),
        aatype=torch.zeros(batch, residues, dtype=torch.long),
        temperature_k=torch.tensor([320.0, 348.0]),
        dataset_index=torch.ones(batch, dtype=torch.long),
        curriculum_step=1,
        curriculum_steps=(0, 0, 0, 0, 100),
    )
    for name in ("lspwm_rollout_endpoint_loss", "lspwm_rollout_gaussian_nll"):
        assert torch.isfinite(outputs[name])
        assert outputs[name].ndim == 0


def test_flow_match_observation_model_shapes_and_loss():
    torch.manual_seed(10)
    model = FlowMatchObservationModel(latent_dim=4, coord_dim=4, hidden_dim=32, num_layers=2)
    z = torch.randn(8, 4)
    x = model(z)
    assert x.shape == (8, 4)
    loss = model.flow_matching_loss(z, x)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_asymmetric_cycle_loss_is_finite():
    torch.manual_seed(11)
    decoder = FlowMatchObservationModel(latent_dim=3, coord_dim=3, hidden_dim=32, num_layers=2)
    z_context = torch.randn(6, 3)
    z_future = z_context + 0.1 * torch.randn_like(z_context)
    loss = asymmetric_cycle_loss(decoder, z_context, z_future, z_context, z_future)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_local_density_weights_are_normalised():
    torch.manual_seed(12)
    z = torch.randn(20, 4)
    weights = local_density_weights(z, alpha=0.5, bandwidth=1.0)
    assert weights.shape == (20,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1.0e-6)
    assert torch.all(weights >= 0.0)


def test_poisson_operator_is_antisymmetric():
    torch.manual_seed(13)
    poisson = PoissonOperator(3, 2)
    gradient = torch.randn(4, 5)
    result = poisson(gradient)
    assert result.shape == (4, 5)
    q_grad = gradient[:, :3]
    s_grad = gradient[:, 3:]
    coupling = poisson.coupling
    q_part = torch.einsum("qa,ba->bq", coupling, s_grad)
    s_part = -torch.einsum("qa,bq->ba", coupling, q_grad)
    reconstructed = torch.cat([q_part, s_part], dim=-1)
    assert torch.allclose(result, reconstructed, atol=1.0e-6)


def test_vamp2_loss_is_negative_and_trainable():
    torch.manual_seed(14)
    source = torch.randn(50, 4)
    target = source @ torch.eye(4) * 0.9 + 0.1 * torch.randn_like(source)
    loss = vamp2_loss(source, target, ridge=1.0e-4)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
    vamp2_score = -loss
    assert vamp2_score > 0.0


def test_msm_consistency_is_finite_and_differentiable():
    torch.manual_seed(15)
    source = torch.randn(12, 32, requires_grad=True)
    target = torch.randn(12, 32)
    loss = msm_consistency_loss(source, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert source.grad is not None


def test_generic_step_matches_legacy_in_overdamped_limit():
    torch.manual_seed(15)
    config = LSPWMConfig(
        state_dim=3,
        condition_dim=0,
        auxiliary_dim=1,
        relaxation_times_ps=(2.0,),
        diffusion_scale=1.0e-2,
    )
    model = LSPWMWorldModel(config)
    q = torch.randn(4, 3)
    aux = torch.randn(4, 1)
    result_generic = model.step(q, aux, None, 0.1, stochastic=False, use_generic=True)
    result_legacy = model.step(q, aux, None, 0.1, stochastic=False, use_generic=False)
    assert torch.isfinite(result_generic["q"]).all()
    assert torch.isfinite(result_legacy["q"]).all()
    assert torch.allclose(result_generic["q"], result_legacy["q"])
    assert torch.allclose(result_generic["auxiliary"], result_legacy["auxiliary"])


def test_empirical_fdt_gate_samples_the_executed_noise_law():
    torch.manual_seed(16)
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=2,
            condition_dim=0,
            auxiliary_dim=1,
            relaxation_times_ps=(2.0,),
            diffusion_scale=1.0e-2,
        )
    ).eval()
    q = torch.randn(3, 2)
    result = empirical_fdt_residual(
        model,
        q,
        None,
        torch.full((3,), 0.1),
        torch.full((3,), 348.0),
        samples=64,
    )
    assert torch.isfinite(result["fdt_residual"])
    assert result["q_covariance"].shape == (3, 2, 2)
    assert result["auxiliary_covariance"].shape == (3, 1, 1)
    assert torch.allclose(
        result["q_fdt_sampling_baseline"],
        torch.tensor((3.0 / 64.0) ** 0.5),
    )
    assert result["analytic_fdt_residual"] < 1.0e-6


def test_temperature_is_required_to_be_finite_and_positive():
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=2,
            condition_dim=0,
            auxiliary_dim=1,
            relaxation_times_ps=(2.0,),
        )
    )
    q = torch.zeros(2, 2)
    auxiliary = torch.zeros(2, 1)
    try:
        model.step(q, auxiliary, None, 0.1, torch.tensor([348.0, float("nan")]))
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("non-finite temperature must be rejected")


def test_curriculum_phase_progression():
    config = LSPWMConfig(
        state_dim=4,
        condition_dim=0,
        auxiliary_dim=1,
        relaxation_times_ps=(2.0,),
        curriculum_phase_steps=(0, 100, 300, 600, 1000),
    )
    model = LSPWMWorldModel(config)
    assert model.get_active_phase(0) == "representation"
    assert model.get_active_phase(150) == "koopman"
    assert model.get_active_phase(400) == "thermodynamic_identification"
    assert model.get_active_phase(800) == "dynamics"
    assert model.get_active_phase(1500) == "observation"
