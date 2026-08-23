from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.predictor import (
    LaggedLinearDynamicsPredictor,
    LaggedPhyschemPredictor,
    LaggedStatePredictor,
    SpatiotemporalPredictor,
    StatePredictor,
)
from protein_jepa_frame.models.temporal import TemporalStateEncoder


class ProteinJEPA(nn.Module):
    """V-JEPA mechanics adapted to protein trajectory clips."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.context_encoder = ProteinGeometryEncoder(
            latent_dim=config.latent_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            num_rbf=config.num_rbf,
            dropout=config.dropout,
            max_residues=config.max_residues,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = SpatiotemporalPredictor(
            latent_dim=config.latent_dim,
            depth=config.predictor_depth,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_frames=config.max_frames,
            max_residues=config.max_residues,
        )
        self.temporal_encoder = (
            TemporalStateEncoder(
                latent_dim=config.latent_dim,
                depth=config.temporal_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
            )
            if config.temporal_depth > 0
            else None
        )
        self.target_temporal_encoder = copy.deepcopy(self.temporal_encoder) if self.temporal_encoder is not None else None
        if self.target_temporal_encoder is not None:
            for param in self.target_temporal_encoder.parameters():
                param.requires_grad = False
        self.state_predictor = (
            StatePredictor(
                latent_dim=config.latent_dim,
                depth=config.state_predictor_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
            )
            if self.temporal_encoder is not None
            else None
        )
        self.lagged_state_predictor = (
            LaggedStatePredictor(
                latent_dim=config.latent_dim,
                lags=config.state_prediction_lags,
                depth=config.lag_predictor_depth,
                dropout=config.dropout,
            )
            if self.temporal_encoder is not None and len(config.state_prediction_lags) > 0
            else None
        )
        self.linear_dynamics_predictor = (
            LaggedLinearDynamicsPredictor(latent_dim=config.latent_dim, lags=config.linear_dynamics_lags)
            if self.temporal_encoder is not None and len(config.linear_dynamics_lags) > 0
            else None
        )
        self.physchem_predictor = (
            LaggedPhyschemPredictor(
                latent_dim=config.latent_dim,
                lags=config.physchem_prediction_lags,
                depth=config.physchem_predictor_depth,
                dropout=config.dropout,
            )
            if self.temporal_encoder is not None and len(config.physchem_prediction_lags) > 0
            else None
        )
        self.state_classifiers: nn.ModuleDict = nn.ModuleDict()

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)
        if self.target_temporal_encoder is not None and self.temporal_encoder is not None:
            for target, context in zip(self.target_temporal_encoder.parameters(), self.temporal_encoder.parameters()):
                target.data.mul_(momentum).add_(context.detach().data, alpha=1.0 - momentum)

    def forward(self, batch: dict, context_mask: torch.Tensor, target_mask: torch.Tensor) -> dict:
        coords = batch["coords"]
        residue_mask = batch["residue_mask"]
        aatype = batch["aatype"]
        residue_index = batch["residue_index"]
        chain_id = batch["chain_id"]
        context_latents = self.context_encoder(
            coords=coords,
            residue_mask=residue_mask,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
            token_mask=context_mask,
        )
        with torch.no_grad():
            target_latents = self.target_encoder(
                coords=coords,
                residue_mask=residue_mask,
                aatype=aatype,
                residue_index=residue_index,
                chain_id=chain_id,
                token_mask=residue_mask,
            )
            if self.config.feature_norm:
                target_latents = F.layer_norm(target_latents, (target_latents.shape[-1],))
        predictions = self.predictor(
            context_latents=context_latents,
            context_mask=context_mask,
            target_mask=target_mask,
            residue_index=residue_index,
            chain_id=chain_id,
        )
        outputs = {
            "predictions": predictions,
            "targets": target_latents,
            "target_mask": target_mask,
            "context_latents": context_latents,
        }
        if self.temporal_encoder is not None:
            context_states = self.temporal_encoder(context_latents, context_mask)
            outputs.update(context_states)
            with torch.no_grad():
                target_states = self.target_temporal_encoder(target_latents, residue_mask)
                target_frame_states = target_states["frame_states"]
                if self.config.feature_norm:
                    target_frame_states = F.layer_norm(target_frame_states, (target_frame_states.shape[-1],))
            context_frame_mask = context_mask.any(dim=2)
            target_frame_mask = target_mask.any(dim=2)
            state_predictions = self.state_predictor(
                context_states=context_states["frame_states"],
                context_frame_mask=context_frame_mask,
                target_frame_mask=target_frame_mask,
            )
            outputs.update(
                {
                    "state_predictions": state_predictions,
                    "state_targets": target_frame_states,
                    "state_target_mask": target_frame_mask,
                    "context_frame_mask": context_frame_mask,
                    "target_frame_states": target_frame_states,
                }
            )
            if self.lagged_state_predictor is not None:
                outputs["lagged_state_predictions"] = self.lagged_state_predictor(context_states["frame_states"])
                outputs["frame_valid_mask"] = residue_mask.any(dim=2)
            if self.linear_dynamics_predictor is not None:
                outputs["linear_dynamics_predictions"] = self.linear_dynamics_predictor(context_states["frame_states"])
                outputs["frame_valid_mask"] = residue_mask.any(dim=2)
            if self.physchem_predictor is not None:
                outputs["physchem_predictions"] = self.physchem_predictor(context_states["frame_states"])
                targets, target_masks = compute_physchem_targets(
                    coords=coords,
                    atom_mask=batch["atom_mask"],
                    residue_mask=residue_mask,
                    residue_index=residue_index,
                    chain_id=chain_id,
                    lags=self.config.physchem_prediction_lags,
                    contact_cutoff=self.config.physchem_contact_cutoff,
                )
                outputs["physchem_targets"] = targets
                outputs["physchem_target_masks"] = target_masks
                outputs["frame_valid_mask"] = residue_mask.any(dim=2)
        return outputs

    def encode_states(self, residue_latents: torch.Tensor, residue_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return frame-level and clip-level state latents for dynamics readouts."""
        if self.temporal_encoder is not None:
            return self.temporal_encoder(residue_latents, residue_mask)
        valid = residue_mask.unsqueeze(-1).float()
        frame_states = (residue_latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
        clip_valid = residue_mask.any(dim=2).unsqueeze(-1).float()
        clip_state = (frame_states * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)
        return {"frame_states": frame_states, "clip_state": clip_state}

    def compute_msm_state_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        """Per-protein MSM state classification loss on context latents."""
        if "msm_state_labels" not in batch or batch["msm_state_labels"] is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        pooled = outputs.get("frame_states")
        if pooled is None:
            pooled = self.encode_states(outputs["context_latents"], batch["residue_mask"])["frame_states"]
        labels = batch["msm_state_labels"]  # [B, T], -1 = unlabeled
        protein_ids = batch["protein_id"]  # list of str
        D = pooled.shape[-1]
        total_loss = torch.tensor(0.0, device=pooled.device)
        count = 0
        for b in range(pooled.shape[0]):
            pid = protein_ids[b]
            if pid not in self.state_classifiers:
                self.state_classifiers[pid] = nn.Linear(D, 6).to(pooled.device)
            classifier = self.state_classifiers[pid]
            logits = classifier(pooled[b])  # [T, 6]
            mask = labels[b] >= 0
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], labels[b][mask])
            total_loss = total_loss + loss
            count += 1
        return total_loss / max(count, 1)

    def compute_vamp2_loss(self, outputs: dict, batch: dict, lag: int = 1) -> torch.Tensor:
        """VAMP-2 maximization loss per clip (maximise -> minimise negative)."""
        _ = outputs  # unused, we need full-structure latents
        coords = batch["coords"]
        residue_mask = batch["residue_mask"]
        aatype = batch["aatype"]
        residue_index = batch["residue_index"]
        chain_id = batch["chain_id"]
        full_latents = self.context_encoder(
            coords=coords, residue_mask=residue_mask,
            aatype=aatype, residue_index=residue_index, chain_id=chain_id,
            token_mask=residue_mask,
        )
        pooled = self.encode_states(full_latents, residue_mask)["frame_states"]  # [B, T, D]
        D = pooled.shape[-1]
        eps = 1e-3
        total_vamp2 = torch.tensor(0.0, device=pooled.device)
        n_clips = 0
        for b in range(pooled.shape[0]):
            seq = pooled[b]  # [T, D]
            if len(seq) < lag + 2:
                continue
            # Use a fixed random projection to a low-dimensional space
            # (T=8 is too few frames for stable 128D covariance estimation)
            if not hasattr(self, '_vamp2_projector') or self._vamp2_projector.shape[0] != D:
                proj = torch.randn(D, min(D, 8), device=seq.device)
                proj = proj / proj.norm(dim=0, keepdim=True)
                self.register_buffer('_vamp2_projector', proj)
            seq = seq @ self._vamp2_projector  # [T, d] with d = min(D, 8)
            d = seq.shape[-1]
            X0 = seq[:-lag]
            X1 = seq[lag:]
            X0 = X0 - X0.mean(dim=0, keepdim=True)
            X1 = X1 - X1.mean(dim=0, keepdim=True)
            N = X0.shape[0]
            try:
                # Compute whitening matrices without backprop (eigh gradients are unstable)
                with torch.no_grad():
                    C00 = X0.T @ X0 / (N - 1) + eps * torch.eye(d, device=X0.device)
                    C01 = X0.T @ X1 / (N - 1)
                    C11 = X1.T @ X1 / (N - 1) + eps * torch.eye(d, device=X1.device)
                    s0, U0 = torch.linalg.eigh(C00)
                    s1, U1 = torch.linalg.eigh(C11)
                    cutoff = 1e-6
                    s0_inv_sqrt = torch.where(s0 > cutoff, 1.0 / s0.sqrt(), torch.zeros_like(s0))
                    s1_inv_sqrt = torch.where(s1 > cutoff, 1.0 / s1.sqrt(), torch.zeros_like(s1))
                    C00_inv_half = U0 @ torch.diag(s0_inv_sqrt) @ U0.T
                    C11_inv_half = U1 @ torch.diag(s1_inv_sqrt) @ U1.T
                # Gradients flow through X0 and X1, NOT through the whitening matrices
                K = (X0 @ C00_inv_half).T @ (X1 @ C11_inv_half) / (N - 1)
                vamp2 = (K ** 2).sum()
                total_vamp2 = total_vamp2 + vamp2
                n_clips += 1
            except torch._C._LinAlgError:
                continue
        vamp2_mean = total_vamp2 / max(n_clips, 1)
        # Normalise to [0, 1]: 1 = trivial, proj_dim = perfect
        proj_dim = self._vamp2_projector.shape[-1]
        loss = 1.0 - (vamp2_mean - 1.0) / (proj_dim - 1.0)
        return loss


def _latent_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if pred.numel() == 0:
        raise ValueError("empty target mask")
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target)


def residue_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    pred = outputs["predictions"][outputs["target_mask"]]
    target = outputs["targets"][outputs["target_mask"]]
    return _latent_loss(pred, target, loss_type)


def state_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    pred = outputs["state_predictions"][outputs["state_target_mask"]]
    target = outputs["state_targets"][outputs["state_target_mask"]]
    return _latent_loss(pred, target, loss_type)


def state_transition_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    pred_delta = outputs["state_predictions"][:, 1:] - outputs["state_predictions"][:, :-1]
    target_delta = outputs["state_targets"][:, 1:] - outputs["state_targets"][:, :-1]
    frame_mask = outputs["state_target_mask"]
    valid_mask = outputs.get("frame_valid_mask", frame_mask)
    pair_mask = frame_mask[:, 1:] & frame_mask[:, :-1] & valid_mask[:, 1:] & valid_mask[:, :-1]
    if not pair_mask.any():
        return outputs["predictions"].new_tensor(0.0)
    return _latent_loss(pred_delta[pair_mask], target_delta[pair_mask], loss_type)


def lagged_state_semigroup_loss(
    outputs: dict,
    lagged_state_predictor: LaggedStatePredictor | None,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    if "lagged_state_predictions" not in outputs or lagged_state_predictor is None:
        return outputs["predictions"].new_tensor(0.0)
    predictions = outputs["lagged_state_predictions"]
    lags = sorted(predictions)
    available = set(lags)
    context_mask = outputs["context_frame_mask"]
    target_mask = outputs["state_target_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | target_mask)
    losses = []
    for first_lag in lags:
        first_pred = predictions[first_lag]
        composed_predictions = lagged_state_predictor(first_pred)
        for second_lag in lags:
            total_lag = first_lag + second_lag
            if total_lag not in available or total_lag >= first_pred.shape[1]:
                continue
            direct = predictions[total_lag][:, :-total_lag].detach()
            composed = composed_predictions[second_lag][:, :-total_lag]
            pair_mask = (
                context_mask[:, :-total_lag]
                & target_mask[:, total_lag:]
                & valid_mask[:, first_lag:-second_lag]
                & valid_mask[:, total_lag:]
            )
            if pair_mask.any():
                losses.append(_latent_loss(composed[pair_mask], direct[pair_mask], loss_type))
    if not losses:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack(losses).mean()


def lagged_state_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "lagged_state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    context_mask = outputs["context_frame_mask"]
    target_mask = outputs["state_target_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | target_mask)
    target_states = outputs["state_targets"]
    losses = []
    for lag, pred_all in outputs["lagged_state_predictions"].items():
        if lag >= pred_all.shape[1]:
            continue
        pred = pred_all[:, :-lag]
        target = target_states[:, lag:]
        pair_mask = context_mask[:, :-lag] & target_mask[:, lag:] & valid_mask[:, lag:]
        if pair_mask.any():
            losses.append(_latent_loss(pred[pair_mask], target[pair_mask], loss_type))
    if not losses:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack(losses).mean()


def physchem_prediction_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "physchem_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    context_mask = outputs["context_frame_mask"]
    target_mask = outputs["state_target_mask"]
    losses = []
    for lag, pred_all in outputs["physchem_predictions"].items():
        if lag >= pred_all.shape[1]:
            continue
        target_all = outputs["physchem_targets"][lag]
        physchem_mask = outputs["physchem_target_masks"][lag]
        pair_mask = context_mask[:, :-lag] & target_mask[:, lag:] & physchem_mask[:, :-lag]
        if pair_mask.any():
            losses.append(_latent_loss(pred_all[:, :-lag][pair_mask], target_all[:, :-lag][pair_mask], loss_type))
    if not losses:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack(losses).mean()


def linear_dynamics_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    if "linear_dynamics_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    context_mask = outputs["context_frame_mask"]
    target_mask = outputs["state_target_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | target_mask)
    target_states = outputs["state_targets"]
    losses = []
    for lag, pred_all in outputs["linear_dynamics_predictions"].items():
        if lag >= pred_all.shape[1]:
            continue
        pred = pred_all[:, :-lag]
        target = target_states[:, lag:]
        pair_mask = context_mask[:, :-lag] & target_mask[:, lag:] & valid_mask[:, lag:]
        if pair_mask.any():
            losses.append(_latent_loss(pred[pair_mask], target[pair_mask], loss_type))
    if not losses:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack(losses).mean()


def jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    return residue_jepa_loss(outputs, loss_type=loss_type)


def variance_regularizer(outputs: dict) -> torch.Tensor:
    pred = outputs["predictions"][outputs["target_mask"]]
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    std = torch.sqrt(pred.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def state_variance_regularizer(outputs: dict) -> torch.Tensor:
    if "state_predictions" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    pred = outputs["state_predictions"][outputs["state_target_mask"]]
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    std = torch.sqrt(pred.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = F.normalize(b1, dim=-1, eps=1.0e-8)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


@torch.no_grad()
def _backbone_torsions(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, L = residue_mask.shape
    angles = coords.new_zeros(B, T, L, 3)
    masks = torch.zeros(B, T, L, 3, dtype=torch.bool, device=coords.device)
    if L < 2:
        return angles, masks

    n = coords[:, :, :, 0]
    ca = coords[:, :, :, 1]
    c = coords[:, :, :, 2]
    valid_n = atom_mask[:, :, :, 0] & residue_mask
    valid_ca = atom_mask[:, :, :, 1] & residue_mask
    valid_c = atom_mask[:, :, :, 2] & residue_mask
    same_chain_prev = chain_id[:, 1:] == chain_id[:, :-1]
    same_chain_next = same_chain_prev

    phi_mask = valid_c[:, :, :-1] & valid_n[:, :, 1:] & valid_ca[:, :, 1:] & valid_c[:, :, 1:]
    phi_mask = phi_mask & same_chain_prev[:, None, :]
    angles[:, :, 1:, 0] = _dihedral(c[:, :, :-1], n[:, :, 1:], ca[:, :, 1:], c[:, :, 1:])
    masks[:, :, 1:, 0] = phi_mask

    psi_mask = valid_n[:, :, :-1] & valid_ca[:, :, :-1] & valid_c[:, :, :-1] & valid_n[:, :, 1:]
    psi_mask = psi_mask & same_chain_next[:, None, :]
    angles[:, :, :-1, 1] = _dihedral(n[:, :, :-1], ca[:, :, :-1], c[:, :, :-1], n[:, :, 1:])
    masks[:, :, :-1, 1] = psi_mask

    omega_mask = valid_ca[:, :, :-1] & valid_c[:, :, :-1] & valid_n[:, :, 1:] & valid_ca[:, :, 1:]
    omega_mask = omega_mask & same_chain_next[:, None, :]
    angles[:, :, :-1, 2] = _dihedral(ca[:, :, :-1], c[:, :, :-1], n[:, :, 1:], ca[:, :, 1:])
    masks[:, :, :-1, 2] = omega_mask
    return angles, masks


@torch.no_grad()
def compute_physchem_targets(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    lags: tuple[int, ...],
    contact_cutoff: float = 8.0,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    B, T, L = residue_mask.shape
    device = coords.device
    ca = coords[:, :, :, 1]
    ca_valid = atom_mask[:, :, :, 1] & residue_mask
    pair_valid = ca_valid[:, :, :, None] & ca_valid[:, :, None, :]
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    seq_sep = (residue_index[:, :, None] - residue_index[:, None, :]).abs() > 2
    pair_valid = pair_valid & same_chain[:, None] & seq_sep[:, None]
    flat_ca = ca.reshape(B * T, L, 3)
    distances = torch.cdist(flat_ca, flat_ca).reshape(B, T, L, L)
    contacts = (distances < contact_cutoff) & pair_valid

    torsions, torsion_mask = _backbone_torsions(coords, atom_mask, residue_mask, chain_id)
    targets: dict[int, torch.Tensor] = {}
    masks: dict[int, torch.Tensor] = {}
    for lag in lags:
        target = coords.new_zeros(B, T, 4)
        valid = torch.zeros(B, T, dtype=torch.bool, device=device)
        if lag >= T:
            targets[int(lag)] = target
            masks[int(lag)] = valid
            continue

        current_contacts = contacts[:, :-lag]
        future_contacts = contacts[:, lag:]
        contact_count = current_contacts.sum(dim=(-1, -2))
        persisted = (current_contacts & future_contacts).sum(dim=(-1, -2))
        contact_persistence = persisted.float() / contact_count.float().clamp(min=1.0)
        contact_valid = contact_count > 0

        torsion_current = torsions[:, :-lag]
        torsion_future = torsions[:, lag:]
        torsion_pair_mask = torsion_mask[:, :-lag] & torsion_mask[:, lag:]
        torsion_delta = (
            (torch.sin(torsion_future) - torch.sin(torsion_current)).pow(2)
            + (torch.cos(torsion_future) - torch.cos(torsion_current)).pow(2)
        )
        torsion_sum = (torsion_delta * torsion_pair_mask.float()).sum(dim=2)
        torsion_count = torsion_pair_mask.sum(dim=2)
        torsion_mean = torsion_sum / torsion_count.float().clamp(min=1.0)
        torsion_valid = torsion_count > 0

        target[:, :-lag, 0] = contact_persistence
        target[:, :-lag, 1:] = torsion_mean
        valid[:, :-lag] = contact_valid & torsion_valid.all(dim=-1)
        targets[int(lag)] = target
        masks[int(lag)] = valid
    return targets, masks
