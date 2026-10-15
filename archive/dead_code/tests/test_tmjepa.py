import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig, ModelConfig, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, tmjepa_terms
from protein_jepa_frame.models.thermodynamic import (
    CausalAuxiliaryFilter,
    FixedBackboneCollectiveVariables,
    K_B_KCAL_PER_MOL_K,
    ThermodynamicMoriDynamics,
)
from scripts.build_tmjepa_pilot_manifests import build_splits
from scripts.evaluate_tmjepa import _ck_error, _two_sample_energy_distance


def test_energy_distance_u_statistic_has_no_same_distribution_bias():
    torch.manual_seed(123)
    left = torch.randn(4096, 8, 3)
    right = torch.randn(4096, 8, 3)
    shifted = right + 1.0

    same = _two_sample_energy_distance(left, right).mean()
    different = _two_sample_energy_distance(left, shifted).mean()

    assert same.abs() < 0.02
    assert different > 0.3
from scripts.compare_tmjepa_ablation import compare


def _batch() -> dict:
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
    return next(
        iter(
            DataLoader(
                build_dataset(config), batch_size=2, collate_fn=collate_paired_windows
            )
        )
    )


def test_fixed_collective_variables_are_se3_invariant_and_smooth():
    batch = _batch()
    extractor = FixedBackboneCollectiveVariables(num_segments=3)
    coords = batch["context_coords"]
    angle = torch.tensor(0.71)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = coords @ rotation.T + torch.tensor([4.0, -3.0, 2.0])
    original_cv = extractor(
        coords,
        batch["context_atom_mask"],
        batch["context_residue_mask"],
        batch["context_chain_id"],
        batch["context_residue_index"],
    )
    transformed_cv = extractor(
        transformed,
        batch["context_atom_mask"],
        batch["context_residue_mask"],
        batch["context_chain_id"],
        batch["context_residue_index"],
    )

    assert original_cv.shape == (2, 8, 16)
    assert torch.isfinite(original_cv).all()
    assert torch.allclose(original_cv, transformed_cv, atol=2.0e-5)
    assert torch.allclose(
        extractor.to_unconstrained(extractor.to_physical(original_cv)),
        original_cv,
        atol=2.0e-5,
    )
    assert torch.isfinite(
        extractor.log_abs_det_to_unconstrained(extractor.to_physical(original_cv))
    ).all()
    perturbed = coords.clone()
    perturbed[:, :, 5, 1, 0] += 1.0e-3
    perturbed_cv = extractor(
        perturbed,
        batch["context_atom_mask"],
        batch["context_residue_mask"],
        batch["context_chain_id"],
        batch["context_residue_index"],
    )
    assert torch.linalg.vector_norm(perturbed_cv - original_cv) < 0.1


def test_free_energy_has_explicit_temperature_and_psd_diffusion():
    model = ThermodynamicMoriDynamics(
        cv_dim=8,
        condition_dim=16,
        auxiliary_dim=2,
        base_step_ps=1000.0,
        ensemble_size=3,
    )
    with torch.no_grad():
        model.enthalpy_head.bias.fill_(3.0)
        model.entropy_over_kb_head.bias.fill_(2.0)
    q = torch.zeros(2, 8)
    condition = torch.zeros(2, 16)
    temperature = torch.tensor([300.0, 450.0])
    free_energy, beta_free_energy, entropy = model.free_energy(q, condition, temperature)
    expected = 3.0 - K_B_KCAL_PER_MOL_K * temperature * 2.0
    kinetic = model.kinetic_parameters(condition, temperature)

    assert torch.allclose(free_energy, expected)
    assert torch.allclose(beta_free_energy, expected / (K_B_KCAL_PER_MOL_K * temperature))
    assert torch.allclose(entropy, torch.full_like(entropy, 2.0))
    assert torch.linalg.eigvalsh(kinetic["diffusion"]).min() > 0.0
    assert torch.all(kinetic["diffusion_rates"] > 0.0)
    assert torch.all(kinetic["relaxation_rates"] > 0.0)
    assert not torch.allclose(kinetic["basis"], kinetic["coupling_basis"])

    no_memory = ThermodynamicMoriDynamics(
        cv_dim=8,
        condition_dim=16,
        auxiliary_dim=2,
        memory_enabled=False,
    )
    assert torch.count_nonzero(
        no_memory.kinetic_parameters(condition, temperature)["coupling_rates"]
    ).item() == 0


def test_auxiliary_filter_is_strictly_prefix_causal():
    torch.manual_seed(4)
    observer = CausalAuxiliaryFilter(
        cv_dim=5, auxiliary_dim=2, base_step_ps=1000.0, hidden_dim=12
    ).eval()
    sequence = torch.randn(2, 7, 5)
    mask = torch.ones(2, 7, dtype=torch.bool)
    time_ps = torch.arange(7, dtype=torch.float32).expand(2, -1) * 1000.0

    full_mean, full_logvar = observer(sequence, mask, time_ps)
    prefix_mean, prefix_logvar = observer(
        sequence[:, :4], mask[:, :4], time_ps[:, :4]
    )

    assert torch.allclose(full_mean[:, :4], prefix_mean, atol=1.0e-6)
    assert torch.allclose(full_logvar[:, :4], prefix_logvar, atol=1.0e-6)


def test_heun_rollout_preserves_auxiliary_ou_equilibrium_variance():
    torch.manual_seed(7)
    model = ThermodynamicMoriDynamics(
        cv_dim=2,
        condition_dim=8,
        auxiliary_dim=1,
        memory_enabled=False,
        base_step_ps=250.0,
        max_rollout_steps=8,
        ensemble_size=12000,
    ).eval()
    _, auxiliary = model.rollout(
        torch.zeros(1, 2),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 8),
        torch.tensor([300.0]),
        torch.tensor([1000.0]),
    )

    variance = auxiliary[0, :, 0].var(unbiased=False)
    assert 0.9 < variance < 1.1


def test_tmjepa_integrated_forward_distribution_and_backward():
    batch = _batch()
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            temporal_causal=True,
            num_heads=4,
            num_rbf=8,
            dropout=0.0,
            tmjepa_enabled=True,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_base_step_ps=1.0,
            tmjepa_max_rollout_steps=16,
            tmjepa_ensemble_size=3,
        )
    )
    outputs = model.forward_paired(batch)
    terms = tmjepa_terms(outputs)
    loss = (
        terms["energy_score"]
        + 1.0e-3 * terms["equilibrium_score_loss"]
        + 1.0e-3 * terms["auxiliary_kl_loss"]
    )
    loss.backward()

    assert outputs["tmjepa_samples"].shape == (2, 3, 16)
    assert outputs["tmjepa_prediction"].shape == (2, 16)
    assert outputs["tmjepa_temperature_k"].tolist() == [300.0, 300.0]
    assert torch.isfinite(terms["path_nll"])
    assert torch.isfinite(terms["auxiliary_transition_kl_loss"])
    assert all(torch.isfinite(value) for value in terms.values())
    assert terms["minimum_diffusion_eigenvalue"] > 0.0
    gradients = [
        parameter.grad
        for parameter in model.tmjepa.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_tmjepa_checkpoint_config_is_recovered():
    config = ModelConfig(tmjepa_enabled=True)
    model = ProteinJEPA(
        ModelConfig(
            paired_forecast=True,
            temporal_depth=1,
            tmjepa_enabled=True,
        )
    )
    recovered = model_config_from_dict(config.__dict__, model.state_dict())

    assert recovered.tmjepa_enabled
    assert recovered.tmjepa_num_segments == 3


def test_tmjepa_learned_cv_vamp_and_augmented_ck_are_finite():
    batch = _batch()
    model = ProteinJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            paired_forecast=True,
            paired_predictor_depth=1,
            temporal_causal=True,
            num_heads=4,
            num_rbf=8,
            dropout=0.0,
            tmjepa_enabled=True,
            tmjepa_learned_cv_dim=2,
            tmjepa_condition_dim=16,
            tmjepa_auxiliary_dim=2,
            tmjepa_base_step_ps=1.0,
            tmjepa_max_rollout_steps=16,
            tmjepa_ensemble_size=3,
        )
    )
    outputs = model.forward_paired(batch)
    terms = tmjepa_terms(outputs)
    model.eval()
    eval_outputs = model.forward_paired(batch)
    ck = _ck_error(model, eval_outputs, samples=2)

    assert outputs["tmjepa_samples"].shape[-1] == 18
    assert outputs["tmjepa_learned_cv"].shape == (2, 8, 2)
    assert torch.isfinite(terms["vamp_loss"])
    assert torch.isfinite(terms["vamp2_score"])
    assert ck.shape == (2,)
    assert torch.isfinite(ck).all()


def test_tmjepa_manifest_split_holds_out_replica_and_temperature():
    rows = []
    for protein in ("p0", "p1"):
        for temperature in (320.0, 450.0):
            for replica in range(5):
                rows.append(
                    {
                        "protein_id": protein,
                        "temperature_k": temperature,
                        "replica_id": str(replica),
                    }
                )
    split = build_splits(
        rows, num_proteins=2, heldout_temperature_k=450.0, seed=1
    )

    assert {row["replica_id"] for row in split["train"]} == {"0", "1", "2"}
    assert {row["replica_id"] for row in split["validation"]} == {"3"}
    assert {row["replica_id"] for row in split["test"]} == {"4"}
    assert {row["temperature_k"] for row in split["temperature_test"]} == {450.0}
    assert all(row["temperature_k"] != 450.0 for row in split["train"])


def test_tmjepa_ablation_requires_real_memory_improvement():
    baseline = {
        "horizons": {"1": {"energy_skill": 0.2, "augmented_ck_energy_distance": 0.1}},
        "closure": {"innovation_lag1_rms": 0.3},
        "physics": {
            "mean_absolute_memory_coupling_per_ps": 0.0,
            "minimum_diffusion_eigenvalue": 1.0e-6,
        },
    }
    improved = {
        "horizons": {"1": {"energy_skill": 0.2, "augmented_ck_energy_distance": 0.08}},
        "closure": {
            "innovation_lag1_rms": 0.2,
            "path_nll_per_dimension": -1.1,
        },
        "physics": {
            "mean_absolute_memory_coupling_per_ps": 1.0e-5,
            "minimum_diffusion_eigenvalue": 1.0e-6,
        },
    }
    failed = {
        **improved,
        "closure": {
            "innovation_lag1_rms": 0.2,
            "path_nll_per_dimension": -0.9,
        },
    }

    baseline["closure"]["path_nll_per_dimension"] = -1.0

    assert compare(baseline, improved, 0.05)["memory_promoted"]
    assert not compare(baseline, failed, 0.05)["memory_promoted"]


def test_tmjepa_rejects_legacy_v5_coactivation():
    try:
        ProteinJEPA(
            ModelConfig(
                paired_forecast=True,
                temporal_depth=1,
                tmjepa_enabled=True,
                v51_contextual_dynamics_enabled=True,
            )
        )
    except ValueError as error:
        assert "legacy V5" in str(error)
    else:
        raise AssertionError("TM-JEPA must not share a training path with legacy V5")
