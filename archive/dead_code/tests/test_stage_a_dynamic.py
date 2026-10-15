import torch

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.dynamic_encoder import DynamicStateEncoder
from protein_jepa_frame.models.stage_a import StageAJEPA
from protein_jepa_frame.models.stage_a.objectives import (
    coordinate_activity_regularizer,
    coordinate_covariance_regularizer,
    stage_a_total_loss,
)


def _backbone(batch: int = 2, frames: int = 5, residues: int = 8) -> torch.Tensor:
    atoms = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    out = atoms.view(1, 1, 1, 4, 3).repeat(batch, frames, residues, 1, 1)
    out = out + torch.arange(residues).view(1, 1, residues, 1, 1) * torch.tensor(
        [1.5, 0.2, 0.1]
    )
    return out


def test_dynamic_encoder_static_coordinates_are_zero_and_motion_has_gradient():
    coords = _backbone()
    mask = torch.ones(coords.shape[:3], dtype=torch.bool)
    residue_index = torch.arange(coords.shape[2]).repeat(coords.shape[0], 1)
    chain_id = torch.zeros_like(residue_index)
    encoder = DynamicStateEncoder(latent_dim=16, depth=2, k_nearest=3, dropout=0.0)

    static = encoder(coords, mask, residue_index, chain_id, dt=torch.ones(coords.shape[:2]))
    assert static["dynamic_residue_states"].abs().max() < 1.0e-5

    moving = coords.clone()
    moving[:, 2:, 3, 1, :] += torch.tensor([0.2, -0.1, 0.3])
    output = encoder(moving, mask, residue_index, chain_id, dt=torch.ones(coords.shape[:2]))
    assert output["dynamic_residue_states"].abs().mean() > 1.0e-7
    output["dynamic_residue_states"].square().mean().backward()
    assert encoder.dynamic_proj[0].weight.grad.abs().sum() > 0


def test_dynamic_activity_and_covariance_losses_are_finite_with_gradients():
    features = torch.randn(2, 4, 5, 12, requires_grad=True)
    valid = torch.ones(2, 4, 5, dtype=torch.bool)
    activity = coordinate_activity_regularizer(features, valid, target_std=0.02)
    covariance = coordinate_covariance_regularizer(features, valid)
    total = activity + covariance
    assert torch.isfinite(total)
    total.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_paired_online_dynamic_prediction_does_not_read_future():
    batch_size, frames, residues = 2, 4, 8
    context_coords = _backbone(batch_size, frames, residues)
    future_coords = _backbone(batch_size, frames, residues).clone()
    mask = torch.ones(batch_size, frames, residues, dtype=torch.bool)
    residue_index = torch.arange(residues).repeat(batch_size, 1)
    chain_id = torch.zeros_like(residue_index)

    def view(prefix, coords):
        return {
            f"{prefix}coords": coords,
            f"{prefix}residue_mask": mask,
            f"{prefix}aatype": torch.zeros(batch_size, residues, dtype=torch.long),
            f"{prefix}residue_index": residue_index,
            f"{prefix}chain_id": chain_id,
            f"{prefix}time_ps": torch.arange(frames).repeat(batch_size, 1).float(),
            f"{prefix}dt_ps": torch.ones(batch_size, frames),
        }

    batch = {
        **view("context_", context_coords),
        **view("future_", future_coords),
        "horizon_time_ps": torch.full((batch_size,), 32000.0),
        "horizon_frames": torch.full((batch_size,), 32, dtype=torch.long),
        "temperature_k": None,
        "dataset_index": torch.zeros(batch_size, dtype=torch.long),
    }
    model = StageAJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            paired_predictor_depth=1,
            num_heads=4,
            num_rbf=4,
            max_frames=16,
            max_residues=16,
            dropout=0.0,
            thermodynamic_conditioning=False,
            paired_forecast=True,
            use_dynamic_encoder=True,
            dynamic_k_nearest=3,
        )
    ).eval()
    with torch.no_grad():
        first = model.forward_paired(batch)["dynamic_prediction"]
        batch["future_coords"] = future_coords + 100.0 * torch.randn_like(future_coords)
        second = model.forward_paired(batch)["dynamic_prediction"]
    assert torch.allclose(first, second, atol=1.0e-6, rtol=1.0e-6)
    assert all(not parameter.requires_grad for parameter in model.target_dynamic_encoder.parameters())


def test_dynamic_jepa_loss_is_active_and_target_is_detached():
    coords = _backbone()
    mask = torch.ones(coords.shape[:3], dtype=torch.bool)
    residue_index = torch.arange(coords.shape[2]).repeat(coords.shape[0], 1)
    chain_id = torch.zeros_like(residue_index)
    model = StageAJEPA(
        ModelConfig(
            latent_dim=16,
            encoder_depth=1,
            temporal_depth=1,
            predictor_depth=1,
            state_predictor_depth=1,
            paired_predictor_depth=1,
            num_heads=4,
            num_rbf=4,
            max_frames=16,
            max_residues=16,
            dropout=0.0,
            paired_forecast=True,
            use_dynamic_encoder=True,
            dynamic_k_nearest=3,
        )
    )
    batch = {
        "context_coords": coords,
        "context_residue_mask": mask,
        "context_aatype": torch.zeros(coords.shape[0], coords.shape[2], dtype=torch.long),
        "context_residue_index": residue_index,
        "context_chain_id": chain_id,
        "context_time_ps": torch.arange(coords.shape[1]).repeat(coords.shape[0], 1).float(),
        "context_dt_ps": torch.ones(coords.shape[0], coords.shape[1]),
        "future_coords": coords.clone(),
        "future_residue_mask": mask.clone(),
        "future_aatype": torch.zeros(coords.shape[0], coords.shape[2], dtype=torch.long),
        "future_residue_index": residue_index,
        "future_chain_id": chain_id,
        "future_time_ps": torch.arange(coords.shape[1]).repeat(coords.shape[0], 1).float(),
        "future_dt_ps": torch.ones(coords.shape[0], coords.shape[1]),
        "horizon_time_ps": torch.full((coords.shape[0],), 32000.0),
        "horizon_frames": torch.full((coords.shape[0],), 32, dtype=torch.long),
        "temperature_k": None,
        "dataset_index": torch.zeros(coords.shape[0], dtype=torch.long),
    }
    outputs = model.forward_paired(batch)
    assert not outputs["dynamic_target"].requires_grad
    loss, log = stage_a_total_loss(
        None,
        outputs,
        dynamic_jepa_weight=1.0,
        dynamic_activity_weight=0.05,
        dynamic_covariance_weight=0.01,
        dynamic_variance_floor=0.02,
    )
    assert torch.isfinite(loss)
    assert log["dynamic_jepa_loss"] >= 0.0
    loss.backward()
    assert model.dynamic_encoder.dynamic_proj[0].weight.grad is not None
