import torch

from protein_jepa_frame.models.stage_b1a.objectives import kabsch_ca_mse_loss
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.models.stage_c.objectives import fape_loss
from protein_jepa_frame.utils.backbone_geometry import backbone_torsion_targets


def _backbone(batch=1, length=6):
    ca = torch.stack(
        [torch.arange(length, dtype=torch.float32) * 3.8,
         torch.zeros(length), torch.zeros(length)], dim=-1
    )
    return torch.stack(
        [ca + torch.tensor([-0.5, 0.2, 0.0]), ca,
         ca + torch.tensor([0.6, 0.1, 0.0]),
         ca + torch.tensor([0.1, -0.4, 0.2])], dim=1
    ).unsqueeze(0).repeat(batch, 1, 1, 1)


def test_kabsch_ca_loss_ignores_padding_in_rotation_fit():
    target = torch.randn(2, 6, 3)
    pred = target.clone()
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    pred[1, 3:] = 1000.0
    target[1, 3:] = -1000.0
    loss = kabsch_ca_mse_loss(pred, target, mask)
    assert loss.item() < 1.0e-8


def test_torsion_targets_respect_chain_gaps_and_atom_masks():
    coords = _backbone(length=6)
    residue_mask = torch.ones(1, 6, dtype=torch.bool)
    residue_index = torch.arange(6).view(1, 6)
    chain_id = torch.zeros(1, 6, dtype=torch.long)
    chain_id[:, 3:] = 1
    residue_index[:, 3:] = torch.arange(3)
    atom_mask = torch.ones(1, 6, 4, dtype=torch.bool)
    atom_mask[:, 1, 3] = False
    _, valid = backbone_torsion_targets(
        coords, residue_mask, residue_index, chain_id, atom_mask
    )
    assert not valid[:, 1:3, :3].any()
    assert not valid[:, 3:4, 0].any()  # phi crosses the chain boundary
    assert valid[:, 4:, :3].any()


def test_frame_aligned_fape_uses_canonical_frames_and_atom_mask():
    coords = _backbone(length=5)
    R, t = RigidBodyFrame.from_backbone(
        coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    )
    atom_mask = torch.ones(1, 5, 4, dtype=torch.bool)
    atom_mask[:, 2, 3] = False
    value = fape_loss(
        coords, coords, pred_R=R, pred_t=t, target_R=R, target_t=t,
        mask=torch.ones(1, 5, dtype=torch.bool), atom_mask=atom_mask,
    )
    assert value.item() < 1.0e-3


def test_b1b_temperature_energy_and_rollout_share_transition_mean():
    torch.manual_seed(7)
    cfg = StageB1bConfig(latent_dim=8, bath_dim=2, energy_depth=0)
    model = StageB1b(cfg).eval()
    H = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    cold = model.energy(H, mask, torch.tensor([300.0]))
    hot = model.energy(H, mask, torch.tensor([450.0]))
    assert not torch.allclose(cold, hot)

    m0, P0 = model.init_bath(H, mask)
    chol = torch.linalg.cholesky(P0 + 1.0e-6 * torch.eye(2))
    s0 = m0 + torch.einsum("blij,blj->bli", chol, torch.zeros_like(m0))
    mean, s_mean = model.deterministic_transition_mean(
        H, s0, mask, 1000.0, temperature_k=torch.tensor([348.0])
    )
    rolled, s_roll = model.rollout(
        H, s0, mask, n_steps=1, dt_ps=1000.0,
        temperature_k=torch.tensor([348.0]), stochastic=False,
    )
    assert torch.allclose(mean, rolled, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(s_mean, s_roll, atol=1.0e-6, rtol=1.0e-6)


def test_b1b_history_posterior_receives_transition_gradient():
    torch.manual_seed(11)
    cfg = StageB1bConfig(latent_dim=8, bath_dim=2, energy_depth=0)
    model = StageB1b(cfg).train()
    H = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    m0, P0 = model.init_bath(H, mask)
    chol = torch.linalg.cholesky(P0 + 1.0e-6 * torch.eye(2))
    s0 = m0 + torch.einsum("blij,blj->bli", chol, torch.zeros_like(m0))
    mean, _ = model.deterministic_transition_mean(
        H, s0, mask, torch.tensor([1000.0]), temperature_k=torch.tensor([348.0])
    )
    mean.square().mean().backward()
    grads = [p.grad for p in model.history_posterior.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_b1b_nll_mean_override_has_a_defined_shared_covariance():
    torch.manual_seed(19)
    cfg = StageB1bConfig(latent_dim=8, bath_dim=2, energy_depth=0)
    model = StageB1b(cfg).eval()
    H = torch.randn(1, 4, 8)
    H_next = torch.randn_like(H)
    mask = torch.ones(1, 4, dtype=torch.bool)
    bath, _ = model.init_bath(H, mask)
    dt = torch.tensor([500.0])
    temperature = torch.tensor([348.0])
    mean, _, covariance = model.deterministic_transition_statistics(
        H, bath, mask, dt, temperature_k=temperature
    )
    diffusion = model.mobility(temperature, H.shape[0], H.device, H.dtype)

    # This is the exact call shape used by training after the shared-kernel
    # repair.  Also exercise the fallback that recomputes covariance when a
    # caller supplies a mean override but omits covariance_override.
    nll_shared = model.transition_nll(
        H, H_next, None, diffusion, dt, mask,
        temperature_k=temperature, bath_state=bath,
        mean_override=mean, covariance_override=covariance,
    )
    nll_recomputed = model.transition_nll(
        H, H_next, None, diffusion, dt, mask,
        temperature_k=temperature, bath_state=bath,
        mean_override=mean,
    )
    assert torch.isfinite(nll_shared)
    assert torch.allclose(nll_shared, nll_recomputed, atol=1.0e-6, rtol=1.0e-6)
