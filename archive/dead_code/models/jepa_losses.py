"""Basic loss functions and target computation for ProteinJEPA.

Split from jepa.py to keep the core model class under 2000 lines.
Contains: latent losses, forecast terms, variance regularizers,
physchem targets, closed_local_state, metastable_vamp, v4_slow_kinetic.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from protein_jepa_frame.models.lspwm import vamp2_loss as lspwm_vamp2_loss

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
    terms = lagged_state_semigroup_terms(outputs, lagged_state_predictor, loss_type=loss_type)
    if not terms:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack([term["loss"] for term in terms.values()]).mean()


def lagged_state_semigroup_terms(
    outputs: dict,
    lagged_state_predictor: LaggedStatePredictor | None,
    loss_type: str = "smooth_l1",
) -> dict[str, dict[str, torch.Tensor | int]]:
    """Compare direct and composed transitions from the final causal state.

    Composition is only valid when predictor inputs and outputs inhabit the
    same state space. A motion-only source predicts an absolute state and is
    therefore intentionally excluded.
    """
    if (
        "lagged_state_predictions" not in outputs
        or lagged_state_predictor is None
        or not outputs.get("transition_source_is_state_space", True)
    ):
        return {}
    predictions = outputs["lagged_state_predictions"]
    lags = sorted(predictions)
    available = set(lags)
    context_mask = outputs["context_frame_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | outputs["state_target_mask"])
    time_ps = outputs.get("time_ps")
    B, T = context_mask.shape
    frame_index = torch.arange(T, device=context_mask.device).view(1, T).expand(B, -1)
    source_index = torch.where(context_mask, frame_index, -1).max(dim=1).values
    terms: dict[str, dict[str, torch.Tensor | int]] = {}
    for first_lag in lags:
        first_pred = predictions[first_lag]
        if time_ps is not None and first_lag < time_ps.shape[1]:
            shifted_time = time_ps[:, first_lag:]
            if shifted_time.shape[1] < first_pred.shape[1]:
                pad = shifted_time[:, -1:].expand(-1, first_pred.shape[1] - shifted_time.shape[1])
                shifted_time = torch.cat([shifted_time, pad], dim=1)
            shifted_time = shifted_time[:, : first_pred.shape[1]]
        else:
            shifted_time = None
        composed_predictions = lagged_state_predictor(
            first_pred,
            time_ps=shifted_time,
            temperature_k=outputs.get("temperature_k"),
            dataset_index=outputs.get("dataset_index"),
        )
        for second_lag in lags:
            total_lag = first_lag + second_lag
            if total_lag not in available or total_lag >= T:
                continue
            intermediate_index = source_index + first_lag
            target_index = source_index + total_lag
            pair_mask = (source_index >= 0) & (target_index < T)
            if pair_mask.any():
                eligible = torch.nonzero(pair_mask, as_tuple=False).flatten()
                eligible = eligible[
                    valid_mask[eligible, intermediate_index[eligible]]
                    & valid_mask[eligible, target_index[eligible]]
                ]
                pair_mask = torch.zeros_like(pair_mask)
                pair_mask[eligible] = True
            if not pair_mask.any():
                continue
            batch_index = torch.arange(B, device=context_mask.device)[pair_mask]
            source = source_index[pair_mask]
            direct = predictions[total_lag][batch_index, source].detach()
            composed = composed_predictions[second_lag][batch_index, source]
            key = f"{first_lag}+{second_lag}={total_lag}"
            terms[key] = {
                "loss": _latent_loss(composed, direct, loss_type),
                "pairs": int(pair_mask.sum().item()),
            }
    return terms


def lagged_state_jepa_loss(outputs: dict, loss_type: str = "smooth_l1") -> torch.Tensor:
    terms = lagged_state_jepa_terms(outputs, loss_type=loss_type)
    if not terms:
        return outputs["predictions"].new_tensor(0.0)
    return torch.stack([term["loss"] for term in terms.values()]).mean()


def paired_forecast_terms(outputs: dict, loss_type: str = "smooth_l1") -> dict[str, torch.Tensor]:
    """Return endpoint forecast losses and persistence-relative diagnostics."""
    prediction = outputs["delta_prediction"]
    target = outputs["delta_target"]
    zero = torch.zeros_like(target)
    delta_loss = _latent_loss(prediction, target, loss_type)
    persistence_loss = _latent_loss(zero, target, loss_type)
    state_loss = _latent_loss(outputs["state_prediction"], outputs["future_endpoint"], loss_type)
    relative_delta_loss = delta_loss / persistence_loss.detach().clamp(min=1.0e-6)
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
        "relative_delta_loss": relative_delta_loss,
        "persistence_delta_loss": persistence_loss,
        "skill": 1.0 - delta_loss / persistence_loss.clamp(min=1.0e-8),
        "pairs": torch.tensor(target.shape[0], device=target.device),
    }


def _slow_cross_entropy(prediction: torch.Tensor, target: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """In-batch temporal contrastive loss that prevents the slow head collapsing."""
    if prediction.shape[0] < 2:
        return prediction.new_tensor(0.0)
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(prediction.shape[0], device=prediction.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _continuous_slow_vamp2(
    sequence: torch.Tensor,
    mask: torch.Tensor,
    lags: tuple[int, ...],
    burn_in_frames: int = 0,
    eps: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAMP-2 on within-trajectory fluctuations of a continuous learned state."""
    if burn_in_frames:
        sequence = sequence[:, burn_in_frames:]
        mask = mask[:, burn_in_frames:]
    zero = sequence.new_tensor(0.0)
    dim = sequence.shape[-1]
    scores = []
    eye = torch.eye(dim, device=sequence.device, dtype=sequence.dtype)
    for lag in sorted(set(int(value) for value in lags)):
        if lag <= 0 or lag >= sequence.shape[1]:
            continue
        x_parts = []
        y_parts = []
        for row in range(sequence.shape[0]):
            pair_mask = mask[row, :-lag] & mask[row, lag:]
            if not pair_mask.any():
                continue
            left = sequence[row, :-lag][pair_mask]
            right = sequence[row, lag:][pair_mask]
            # Remove static protein identity while retaining temporal changes.
            center = torch.cat([left, right], dim=0).mean(dim=0, keepdim=True)
            x_parts.append(left - center)
            y_parts.append(right - center)
        if not x_parts:
            continue
        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        if x.shape[0] <= dim + 1:
            continue
        denominator = max(x.shape[0] - 1, 1)
        c00 = x.T @ x / denominator + eps * eye
        c11 = y.T @ y / denominator + eps * eye
        c01 = x.T @ y / denominator
        with torch.no_grad():
            def inverse_sqrt(covariance: torch.Tensor) -> torch.Tensor:
                values, vectors = torch.linalg.eigh(covariance)
                return (vectors * values.clamp(min=eps).rsqrt().unsqueeze(0)) @ vectors.T

            w0 = inverse_sqrt(c00)
            w1 = inverse_sqrt(c11)
        koopman = w0 @ c01 @ w1
        scores.append(koopman.square().sum())
    score = torch.stack(scores).mean() if scores else zero
    return -score / float(dim), score


def slow_forecast_terms(
    outputs: dict,
    loss_type: str = "smooth_l1",
    vamp_lags: tuple[int, ...] = (1, 2, 4, 8),
    vamp_burn_in_frames: int = 0,
) -> dict[str, torch.Tensor]:
    """Losses for a low-dimensional slow/metastable endpoint representation."""
    prediction = outputs["slow_delta_prediction"]
    target = outputs["slow_delta_target"]
    delta_loss = _latent_loss(prediction, target, loss_type)
    state_loss = _latent_loss(outputs["slow_state_prediction"], outputs["future_slow_state"], loss_type)
    persistence_loss = _latent_loss(torch.zeros_like(target), target, loss_type)
    relative_delta_loss = (
        (prediction - target).square().mean(dim=-1)
        / target.detach().square().mean(dim=-1).clamp(min=1.0e-4)
    ).mean()
    alignment_loss = _latent_loss(
        outputs["context_slow_state"], outputs["teacher_context_slow_state"].detach(), loss_type
    )
    context = outputs["context_slow_state"]
    std = torch.sqrt(context.var(dim=0, unbiased=False) + 1.0e-4)
    variance_loss = torch.relu(1.0 - std).mean() if context.shape[0] > 1 else context.new_tensor(0.0)
    if context.shape[0] > 2:
        normalized = (context - context.mean(dim=0, keepdim=True)) / std.detach().clamp(min=1.0e-3)
        correlation = normalized.T @ normalized / max(context.shape[0] - 1, 1)
        off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
        covariance_loss = off_diagonal.square().mean()
    else:
        covariance_loss = context.new_tensor(0.0)
    vamp_loss, vamp2_score = _continuous_slow_vamp2(
        outputs["context_slow_sequence"],
        outputs["context_slow_mask"],
        vamp_lags,
        burn_in_frames=vamp_burn_in_frames,
    )
    return {
        "delta_loss": delta_loss,
        "state_loss": state_loss,
        "relative_delta_loss": relative_delta_loss,
        "alignment_loss": alignment_loss,
        "persistence_loss": persistence_loss,
        "skill": 1.0 - delta_loss / persistence_loss.clamp(min=1.0e-8),
        "contrastive_loss": _slow_cross_entropy(outputs["slow_state_prediction"], outputs["future_slow_state"]),
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "vamp_loss": vamp_loss,
        "vamp2_score": vamp2_score,
    }


def reversible_kinetic_terms(
    outputs: dict,
    vamp_lags: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, torch.Tensor]:
    """Losses and diagnostics for the LS2 reversible kinetic-OU predictor."""
    if "ls2_state_prediction" not in outputs:
        reference = outputs.get("delta_target", outputs.get("context_endpoint"))
        zero = reference.new_tensor(0.0)
        return {
            "state_loss": zero,
            "persistence_loss": zero,
            "state_skill": zero,
            "assignment_loss": zero,
            "assignment_persistence_loss": zero,
            "assignment_skill": zero,
            "nll": zero,
            "balance_loss": zero,
            "entropy_loss": zero,
            "vamp_loss": zero,
            "vamp2_score": zero,
            "detailed_balance_max": zero,
            "stationarity_error": zero,
        }
    prediction = outputs["ls2_state_prediction"]
    target = outputs["ls2_future_state"].detach()
    persistence = outputs["ls2_persistence_state"].detach()
    state_loss = F.smooth_l1_loss(prediction, target)
    persistence_loss = F.smooth_l1_loss(persistence, target)

    assignment_prediction = outputs["ls2_assignment_prediction"].clamp(min=1.0e-8)
    assignment_target = outputs["ls2_future_assignment"].detach()
    assignment_persistence = outputs["ls2_persistence_assignment"].detach()
    assignment_loss = (
        assignment_target
        * (
            assignment_target.clamp(min=1.0e-8).log()
            - assignment_prediction.log()
        )
    ).sum(dim=-1).mean()
    assignment_brier = (assignment_prediction - assignment_target).square().sum(dim=-1).mean()
    persistence_brier = (assignment_persistence - assignment_target).square().sum(dim=-1).mean()

    residual = target - prediction
    log_sigma = outputs["ls2_log_sigma"]
    nll = 0.5 * (
        residual.square() * torch.exp(-2.0 * log_sigma) + 2.0 * log_sigma
    ).mean()
    usage = outputs["ls2_context_assignment"].mean(dim=0).clamp(min=1.0e-8)
    balance_loss = (usage * (usage * usage.numel()).log()).sum()
    entropy_loss = -(
        outputs["ls2_context_assignment"].clamp(min=1.0e-8)
        * outputs["ls2_context_assignment"].clamp(min=1.0e-8).log()
    ).sum(dim=-1).mean()
    vamp_loss, vamp2_score = _continuous_slow_vamp2(
        outputs["ls2_context_assignment_sequence"],
        outputs["context_slow_mask"],
        vamp_lags,
    )
    stationary = outputs["ls2_stationary_distribution"]
    transition = outputs["ls2_transition_matrix"]
    stationary_after = torch.bmm(stationary.unsqueeze(1), transition).squeeze(1)
    stationarity_error = (stationary_after - stationary).abs().max()
    return {
        "state_loss": state_loss,
        "persistence_loss": persistence_loss,
        "state_skill": 1.0 - state_loss / persistence_loss.clamp(min=1.0e-8),
        "assignment_loss": assignment_loss,
        "assignment_persistence_loss": persistence_brier,
        "assignment_skill": 1.0 - assignment_brier / persistence_brier.clamp(min=1.0e-8),
        "nll": nll,
        "balance_loss": balance_loss,
        "entropy_loss": entropy_loss,
        "vamp_loss": vamp_loss,
        "vamp2_score": vamp2_score,
        "detailed_balance_max": outputs["ls2_detailed_balance_residual"].abs().max(),
        "stationarity_error": stationarity_error,
    }


def paired_endpoint_variance_regularizer(outputs: dict) -> torch.Tensor:
    endpoint = outputs["context_endpoint"]
    if endpoint.shape[0] < 2:
        return endpoint.new_tensor(0.0)
    std = torch.sqrt(endpoint.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def paired_causal_alignment_loss(outputs: dict) -> torch.Tensor:
    if "context_teacher_endpoint" not in outputs:
        return outputs["context_endpoint"].new_tensor(0.0)
    return F.smooth_l1_loss(outputs["context_endpoint"], outputs["context_teacher_endpoint"].detach())


def lagged_state_jepa_terms(outputs: dict, loss_type: str = "smooth_l1") -> dict[int, dict[str, torch.Tensor | int]]:
    """Return per-lag losses from the final visible context state.

    Future masking can leave a gap that makes short lags miss the explicit
    target mask entirely. The EMA target encoder nevertheless provides a
    target for every valid frame. Anchoring each transition at the final
    visible context frame gives every configured horizon a causal source and
    prevents short-lag pairs from silently disappearing.
    """
    if "lagged_state_predictions" not in outputs:
        return {}
    context_mask = outputs["context_frame_mask"]
    valid_mask = outputs.get("frame_valid_mask", context_mask | outputs["state_target_mask"])
    target_states = outputs.get("lagged_state_targets", outputs["state_targets"])
    B, T = context_mask.shape
    frame_index = torch.arange(T, device=context_mask.device).view(1, T).expand(B, -1)
    source_index = torch.where(context_mask, frame_index, -1).max(dim=1).values
    terms: dict[int, dict[str, torch.Tensor | int]] = {}
    for lag, pred_all in outputs["lagged_state_predictions"].items():
        target_index = source_index + lag
        pair_mask = (source_index >= 0) & (target_index < T)
        if pair_mask.any():
            eligible = torch.nonzero(pair_mask, as_tuple=False).flatten()
            eligible = eligible[valid_mask[eligible, target_index[eligible]]]
            pair_mask = torch.zeros_like(pair_mask)
            pair_mask[eligible] = True
        if not pair_mask.any():
            continue
        batch_index = torch.arange(B, device=context_mask.device)[pair_mask]
        source = source_index[pair_mask]
        target = target_index[pair_mask]
        loss = _latent_loss(pred_all[batch_index, source], target_states[batch_index, target], loss_type)
        intrinsic_persistence = _latent_loss(
            target_states[batch_index, source], target_states[batch_index, target], loss_type
        )
        transition_states = outputs.get("transition_states")
        persistence = (
            _latent_loss(transition_states[batch_index, source], target_states[batch_index, target], loss_type)
            if transition_states is not None
            else intrinsic_persistence
        )
        source_alignment = (
            _latent_loss(transition_states[batch_index, source], target_states[batch_index, source], loss_type)
            if transition_states is not None
            else loss.new_tensor(0.0)
        )
        terms[int(lag)] = {
            "loss": loss,
            "persistence_loss": persistence,
            "intrinsic_persistence_loss": intrinsic_persistence,
            "source_alignment_loss": source_alignment,
            "pairs": int(pair_mask.sum().item()),
        }
    return terms


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


def kinetic_vampnet_loss(outputs: dict, lag: int = 1, eps: float = 1.0e-3) -> torch.Tensor:
    if "kinetic_readout" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    if lag <= 0:
        raise ValueError("lag must be positive")
    z = outputs["kinetic_readout"]
    mask = outputs.get("kinetic_readout_mask")
    if mask is None:
        mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
    if lag >= z.shape[1]:
        return z.new_tensor(0.0)
    pair_mask = mask[:, :-lag] & mask[:, lag:]
    if not pair_mask.any():
        return z.new_tensor(0.0)
    x0 = z[:, :-lag][pair_mask]
    x1 = z[:, lag:][pair_mask]
    n, dim = x0.shape
    if n < dim + 2 or dim < 2:
        return z.new_tensor(0.0)

    x0 = x0 - x0.mean(dim=0, keepdim=True)
    x1 = x1 - x1.mean(dim=0, keepdim=True)
    eye = torch.eye(dim, device=z.device, dtype=z.dtype)
    with torch.no_grad():
        c00 = x0.T @ x0 / (n - 1) + eps * eye
        c11 = x1.T @ x1 / (n - 1) + eps * eye
        s0, u0 = torch.linalg.eigh(c00)
        s1, u1 = torch.linalg.eigh(c11)
        inv0 = torch.where(s0 > eps, s0.rsqrt(), torch.zeros_like(s0))
        inv1 = torch.where(s1 > eps, s1.rsqrt(), torch.zeros_like(s1))
        c00_inv_half = (u0 * inv0[None, :]) @ u0.T
        c11_inv_half = (u1 * inv1[None, :]) @ u1.T
    koopman = (x0 @ c00_inv_half).T @ (x1 @ c11_inv_half) / (n - 1)
    vamp2 = koopman.pow(2).sum()
    return 1.0 - vamp2 / float(dim)


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


def state_feature_variance_regularizer(outputs: dict) -> torch.Tensor:
    if "frame_states" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    mask = outputs.get("context_frame_mask")
    states = outputs["frame_states"] if mask is None else outputs["frame_states"][mask]
    if states.shape[0] < 2:
        return outputs["predictions"].new_tensor(0.0)
    std = torch.sqrt(states.var(dim=0, unbiased=False) + 1.0e-4)
    return torch.relu(1.0 - std).mean()


def state_covariance_regularizer(outputs: dict) -> torch.Tensor:
    if "frame_states" not in outputs:
        return outputs["predictions"].new_tensor(0.0)
    mask = outputs.get("context_frame_mask")
    states = outputs["frame_states"] if mask is None else outputs["frame_states"][mask]
    if states.shape[0] < 2:
        return outputs["predictions"].new_tensor(0.0)
    states = states - states.mean(dim=0, keepdim=True)
    states = states / torch.sqrt(states.var(dim=0, unbiased=False, keepdim=True) + 1.0e-4)
    covariance = states.T @ states / max(1, states.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off_diagonal.square().sum() / states.shape[1]


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


@torch.no_grad()
def compute_paired_observable_targets(
    context_coords: torch.Tensor,
    future_coords: torch.Tensor,
    context_atom_mask: torch.Tensor,
    future_atom_mask: torch.Tensor,
    context_residue_mask: torch.Tensor,
    future_residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    contact_cutoff: float = 8.0,
) -> torch.Tensor:
    """Compute stable motion targets directly from the two endpoint frames.

    The four channels are contact persistence, mean backbone torsion change,
    radius-of-gyration change, and local CA-distance change. They are
    deliberately low-dimensional so they anchor the latent transition
    without turning MSM labels into a training target.
    """
    c = context_coords[:, -1]
    f = future_coords[:, -1]
    c_atoms = context_atom_mask[:, -1]
    f_atoms = future_atom_mask[:, -1]
    c_valid = context_residue_mask[:, -1] & c_atoms[:, :, 1]
    f_valid = future_residue_mask[:, -1] & f_atoms[:, :, 1]
    pair_valid = c_valid[:, :, None] & c_valid[:, None, :] & f_valid[:, :, None] & f_valid[:, None, :]
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    seq_sep = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
    nonlocal_pair = pair_valid & same_chain & (seq_sep > 2)
    c_dist = torch.cdist(c[:, :, 1], c[:, :, 1])
    f_dist = torch.cdist(f[:, :, 1], f[:, :, 1])
    c_contact = (c_dist < contact_cutoff) & nonlocal_pair
    f_contact = (f_dist < contact_cutoff) & nonlocal_pair
    contact_count = c_contact.sum(dim=(-1, -2)).float()
    contact_persistence = (c_contact & f_contact).sum(dim=(-1, -2)).float() / contact_count.clamp(min=1.0)

    c_torsion, c_torsion_mask = _backbone_torsions(
        context_coords[:, -1:], context_atom_mask[:, -1:], context_residue_mask[:, -1:], chain_id
    )
    f_torsion, f_torsion_mask = _backbone_torsions(
        future_coords[:, -1:], future_atom_mask[:, -1:], future_residue_mask[:, -1:], chain_id
    )
    torsion_mask = c_torsion_mask[:, 0] & f_torsion_mask[:, 0]
    torsion_delta = (
        (torch.sin(f_torsion[:, 0]) - torch.sin(c_torsion[:, 0])).square()
        + (torch.cos(f_torsion[:, 0]) - torch.cos(c_torsion[:, 0])).square()
    ).sqrt()
    torsion_change = (torsion_delta * torsion_mask.float()).sum(dim=(-1, -2)) / torsion_mask.sum(dim=(-1, -2)).float().clamp(min=1.0)

    def radius_of_gyration(ca: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.float().unsqueeze(-1)
        center = (ca * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        count = weights.sum(dim=(1, 2)).clamp(min=1.0)
        return torch.sqrt((((ca - center[:, None]) ** 2) * weights).sum(dim=(1, 2)) / count).clamp(min=1.0e-6)

    c_rg = radius_of_gyration(c[:, :, 1], c_valid)
    f_rg = radius_of_gyration(f[:, :, 1], f_valid)
    rg_change = (f_rg - c_rg) / 10.0

    local_pair = pair_valid & same_chain & (seq_sep > 0) & (seq_sep <= 8)
    local_count = local_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
    local_change = ((f_dist - c_dist).abs() * local_pair.float()).sum(dim=(-1, -2)) / local_count / 10.0
    return torch.stack([contact_persistence, torsion_change, rg_change, local_change], dim=-1).clamp(-5.0, 5.0)


@torch.no_grad()
def compute_v3_paired_observable_targets(
    context_coords: torch.Tensor,
    future_coords: torch.Tensor,
    context_atom_mask: torch.Tensor,
    future_atom_mask: torch.Tensor,
    context_residue_mask: torch.Tensor,
    future_residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    contact_cutoff: float = 8.0,
) -> torch.Tensor:
    """Twelve signed and event-resolved physical transition targets."""
    c = context_coords[:, -1]
    f = future_coords[:, -1]
    c_valid = context_residue_mask[:, -1] & context_atom_mask[:, -1, :, 1]
    f_valid = future_residue_mask[:, -1] & future_atom_mask[:, -1, :, 1]
    valid = c_valid & f_valid
    pair_valid = valid[:, :, None] & valid[:, None, :]
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    sequence_separation = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
    nonlocal_pair = pair_valid & same_chain & (sequence_separation > 2)
    c_dist = torch.cdist(c[:, :, 1], c[:, :, 1])
    f_dist = torch.cdist(f[:, :, 1], f[:, :, 1])
    c_contact = (c_dist < contact_cutoff) & nonlocal_pair
    f_contact = (f_dist < contact_cutoff) & nonlocal_pair
    contact_union = (c_contact | f_contact).sum(dim=(-1, -2)).float().clamp(min=1.0)
    retained = (c_contact & f_contact).sum(dim=(-1, -2)).float() / contact_union
    gained = ((~c_contact) & f_contact & nonlocal_pair).sum(dim=(-1, -2)).float() / contact_union
    lost = (c_contact & (~f_contact) & nonlocal_pair).sum(dim=(-1, -2)).float() / contact_union

    c_torsion, c_torsion_mask = _backbone_torsions(
        context_coords[:, -1:], context_atom_mask[:, -1:], context_residue_mask[:, -1:], chain_id
    )
    f_torsion, f_torsion_mask = _backbone_torsions(
        future_coords[:, -1:], future_atom_mask[:, -1:], future_residue_mask[:, -1:], chain_id
    )
    torsion_mask = c_torsion_mask[:, 0] & f_torsion_mask[:, 0]
    angle_delta = torch.atan2(
        torch.sin(f_torsion[:, 0] - c_torsion[:, 0]),
        torch.cos(f_torsion[:, 0] - c_torsion[:, 0]),
    )
    torsion_count = torsion_mask.sum(dim=1).float().clamp(min=1.0)
    signed_torsion = (angle_delta * torsion_mask.float()).sum(dim=1) / torsion_count
    absolute_torsion = (angle_delta.abs() * torsion_mask.float()).sum(dim=1) / torsion_count

    def radius_of_gyration(ca: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float().unsqueeze(-1)
        center = (ca * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        return torch.sqrt(
            (((ca - center[:, None]) ** 2) * weight).sum(dim=(1, 2))
            / weight.sum(dim=(1, 2)).clamp(min=1.0)
        ).clamp(min=1.0e-6)

    c_rg = radius_of_gyration(c[:, :, 1], valid)
    f_rg = radius_of_gyration(f[:, :, 1], valid)
    rg_change = (f_rg - c_rg) / 10.0

    aligned_rmsd = c_rg.new_zeros(c_rg.shape)
    for batch_index in range(c.shape[0]):
        mask = valid[batch_index]
        if int(mask.sum()) < 3:
            continue
        x = c[batch_index, mask, 1]
        y = f[batch_index, mask, 1]
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        u, _, vh = torch.linalg.svd(x.T @ y)
        correction = torch.ones(3, device=x.device, dtype=x.dtype)
        correction[-1] = torch.sign(torch.det(u @ vh)).clamp(min=-1.0, max=1.0)
        rotation = u @ torch.diag(correction) @ vh
        aligned_rmsd[batch_index] = torch.sqrt(((x @ rotation - y) ** 2).sum(dim=-1).mean()) / 10.0

    local_pair = pair_valid & same_chain & (sequence_separation > 0) & (sequence_separation <= 8)
    local_count = local_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
    signed_local_distance = ((f_dist - c_dist) * local_pair.float()).sum(dim=(-1, -2)) / local_count / 10.0
    return torch.cat(
        [
            torch.stack([retained, gained, lost], dim=-1),
            signed_torsion,
            absolute_torsion,
            rg_change[:, None],
            aligned_rmsd[:, None],
            signed_local_distance[:, None],
        ],
        dim=-1,
    ).clamp(-5.0, 5.0)


@torch.no_grad()
def compute_v3_local_observable_targets(
    context_coords: torch.Tensor,
    future_coords: torch.Tensor,
    context_atom_mask: torch.Tensor,
    future_atom_mask: torch.Tensor,
    context_residue_mask: torch.Tensor,
    future_residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the V3 physical transition vector within sequence segments."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    B, _, L = context_residue_mask.shape
    common = context_residue_mask[:, -1] & future_residue_mask[:, -1]
    position = torch.arange(L, device=common.device).view(1, L).expand(B, -1)
    length = common.sum(dim=1, keepdim=True).clamp(min=1)
    segment_id = torch.div(position * num_tokens, length, rounding_mode="floor").clamp(max=num_tokens - 1)
    targets = []
    masks = []
    for token_index in range(num_tokens):
        segment = (segment_id == token_index) & common
        context_segment_mask = context_residue_mask & segment[:, None]
        future_segment_mask = future_residue_mask & segment[:, None]
        targets.append(
            compute_v3_paired_observable_targets(
                context_coords, future_coords,
                context_atom_mask, future_atom_mask,
                context_segment_mask, future_segment_mask,
                chain_id, residue_index,
                contact_cutoff=contact_cutoff,
            )
        )
        masks.append(segment.sum(dim=1) >= 3)
    return torch.stack(targets, dim=1), torch.stack(masks, dim=1)


def _compute_v3_closed_local_state(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
    differentiable_distances: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one endpoint into composable, SE(3)-invariant local states."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    endpoint = coords[:, -1, :, 1]
    valid = residue_mask[:, -1] & atom_mask[:, -1, :, 1]
    B, L = valid.shape
    position = torch.arange(L, device=coords.device).view(1, L).expand(B, -1)
    length = valid.sum(dim=1, keepdim=True).clamp(min=1)
    segment_id = torch.div(position * num_tokens, length, rounding_mode="floor").clamp(max=num_tokens - 1)
    if differentiable_distances:
        delta = endpoint[:, :, None] - endpoint[:, None, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
    else:
        distances = torch.cdist(endpoint, endpoint)
    sequence_separation = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
    same_chain = chain_id[:, :, None] == chain_id[:, None, :]
    torsions, torsion_mask = _backbone_torsions(coords[:, -1:], atom_mask[:, -1:], residue_mask[:, -1:], chain_id)
    torsions = torsions[:, 0]
    torsion_mask = torsion_mask[:, 0]
    all_weight = valid.float().unsqueeze(-1)
    protein_center = (endpoint * all_weight).sum(dim=1) / all_weight.sum(dim=1).clamp(min=1.0)
    states = []
    masks = []
    for token_index in range(num_tokens):
        segment = (segment_id == token_index) & valid
        segment_pair = segment[:, :, None] & segment[:, None, :] & same_chain
        nonlocal_pair = segment_pair & (sequence_separation > 2)
        possible = nonlocal_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
        contacts = (distances < contact_cutoff) & nonlocal_pair
        contact_density = contacts.sum(dim=(-1, -2)).float() / possible

        segment_torsion_mask = torsion_mask & segment.unsqueeze(-1)
        torsion_count = segment_torsion_mask.sum(dim=1).float().clamp(min=1.0)
        torsion_sin = (torch.sin(torsions) * segment_torsion_mask.float()).sum(dim=1) / torsion_count
        torsion_cos = (torch.cos(torsions) * segment_torsion_mask.float()).sum(dim=1) / torsion_count

        weight = segment.float().unsqueeze(-1)
        count = weight.sum(dim=1).clamp(min=1.0)
        center = (endpoint * weight).sum(dim=1) / count
        rg = torch.sqrt((((endpoint - center[:, None]) ** 2) * weight).sum(dim=(1, 2)) / count.squeeze(-1)).clamp(min=1.0e-6) / 10.0

        local_pair = segment_pair & (sequence_separation > 0) & (sequence_separation <= 8)
        local_count = local_pair.sum(dim=(-1, -2)).float().clamp(min=1.0)
        local_mean = (distances * local_pair.float()).sum(dim=(-1, -2)) / local_count / 10.0
        local_variance = (
            ((distances - local_mean[:, None, None] * 10.0).square() * local_pair.float()).sum(dim=(-1, -2))
            / local_count
        )
        local_std = torch.sqrt(local_variance.clamp(min=1.0e-6)) / 10.0
        center_distance = torch.linalg.vector_norm(center - protein_center, dim=-1) / 10.0
        contact_count = contacts.sum(dim=(-1, -2)).float().clamp(min=1.0)
        contact_order = (
            (sequence_separation.float() * contacts.float()).sum(dim=(-1, -2))
            / contact_count
            / length.squeeze(1).float().clamp(min=1.0)
        )
        states.append(
            torch.cat(
                [
                    contact_density[:, None], torsion_sin, torsion_cos,
                    rg[:, None], local_mean[:, None], local_std[:, None],
                    center_distance[:, None], contact_order[:, None],
                ],
                dim=-1,
            )
        )
        masks.append(segment.sum(dim=1) >= 3)
    return torch.stack(states, dim=1).clamp(-5.0, 5.0), torch.stack(masks, dim=1)


@torch.no_grad()
def compute_v3_closed_local_state(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compute_v3_closed_local_state(
        coords, atom_mask, residue_mask, chain_id, residue_index,
        num_tokens=num_tokens, contact_cutoff=contact_cutoff,
        differentiable_distances=False,
    )


def compute_v4c_force_state(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable endpoint state used only by conservative force matching."""
    return _compute_v3_closed_local_state(
        coords, atom_mask, residue_mask, chain_id, residue_index,
        num_tokens=num_tokens, contact_cutoff=contact_cutoff,
        differentiable_distances=True,
    )


@torch.no_grad()
def compute_v3_closed_local_state_sequence(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    num_tokens: int = 4,
    contact_cutoff: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute closed local physical states at every observed frame."""
    B, T, L, A, C = coords.shape
    states, masks = compute_v3_closed_local_state(
        coords.reshape(B * T, 1, L, A, C),
        atom_mask.reshape(B * T, 1, L, A),
        residue_mask.reshape(B * T, 1, L),
        chain_id[:, None].expand(-1, T, -1).reshape(B * T, L),
        residue_index[:, None].expand(-1, T, -1).reshape(B * T, L),
        num_tokens=num_tokens,
        contact_cutoff=contact_cutoff,
    )
    return states.reshape(B, T, num_tokens, -1), masks.reshape(B, T, num_tokens)


@torch.no_grad()
def compute_v3_closed_local_velocity(
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    current_state: torch.Tensor,
    state_scale: torch.Tensor,
    strides: tuple[int, ...] = (1, 2, 4, 8),
    num_tokens: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardized finite-difference rates from observed context frames only."""
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("closed local velocity strides must be positive")
    scale = state_scale.to(device=current_state.device, dtype=current_state.dtype)
    velocities = []
    masks = []
    for stride in strides:
        if stride >= coords.shape[1]:
            velocities.append(torch.zeros_like(current_state))
            masks.append(torch.zeros_like(current_state[..., 0], dtype=torch.bool))
            continue
        previous_state, previous_mask = compute_v3_closed_local_state(
            coords[:, :-stride], atom_mask[:, :-stride], residue_mask[:, :-stride],
            chain_id, residue_index, num_tokens=num_tokens,
        )
        velocities.append((current_state - previous_state) / scale / float(stride))
        masks.append(previous_mask)
    velocity = torch.stack(velocities, dim=2)
    velocity_mask = torch.stack(masks, dim=-1).all(dim=-1)
    return velocity, velocity_mask


def metastable_vamp_terms(
    outputs: dict,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    eps: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    """VAMP-2 and occupancy controls for the V3 metastable readout."""
    if "context_metastable_states" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {"vamp_loss": zero, "balance_loss": zero, "assignment_entropy": zero}
    sequences = [outputs["context_metastable_states"], outputs["future_metastable_states"]]
    masks = [outputs["context_metastable_mask"], outputs["future_metastable_mask"]]
    scores = []
    for lag in lags:
        x_parts = []
        y_parts = []
        for sequence, mask in zip(sequences, masks):
            if lag >= sequence.shape[1]:
                continue
            pair_mask = mask[:, :-lag] & mask[:, lag:]
            if pair_mask.any():
                x_parts.append(sequence[:, :-lag][pair_mask])
                y_parts.append(sequence[:, lag:][pair_mask])
        if not x_parts:
            continue
        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        dim = x.shape[-1]
        if x.shape[0] < dim + 2:
            continue
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            c00 = x.T @ x / max(x.shape[0] - 1, 1) + eps * eye
            c11 = y.T @ y / max(y.shape[0] - 1, 1) + eps * eye
            s0, u0 = torch.linalg.eigh(c00)
            s1, u1 = torch.linalg.eigh(c11)
            inv0 = torch.where(s0 > eps, s0.rsqrt(), torch.zeros_like(s0))
            inv1 = torch.where(s1 > eps, s1.rsqrt(), torch.zeros_like(s1))
            w0 = (u0 * inv0[None, :]) @ u0.T
            w1 = (u1 * inv1[None, :]) @ u1.T
        koopman = (x @ w0).T @ (y @ w1) / max(x.shape[0] - 1, 1)
        scores.append(koopman.square().sum())
    all_states = torch.cat(
        [sequence[mask] for sequence, mask in zip(sequences, masks)], dim=0
    )
    occupancy = all_states.mean(dim=0).clamp(min=1.0e-8)
    uniform = 1.0 / float(occupancy.numel())
    balance_loss = (occupancy * torch.log(occupancy / uniform)).sum()
    assignment_entropy = -(all_states.clamp(min=1.0e-8) * torch.log(all_states.clamp(min=1.0e-8))).sum(dim=-1).mean()
    if not scores:
        vamp_loss = all_states.new_tensor(0.0)
    else:
        score = torch.stack(scores).mean()
        vamp_loss = 1.0 - (score - 1.0) / max(all_states.shape[-1] - 1, 1)
    return {
        "vamp_loss": vamp_loss,
        "balance_loss": balance_loss,
        "assignment_entropy": assignment_entropy,
    }


def v4_slow_kinetic_terms(
    outputs: dict,
    lags: tuple[int, ...] = (4, 8, 16),
) -> dict[str, torch.Tensor]:
    """VAMP, reconstruction, and compositional transition terms for V4-B."""
    if "v4_slow_prediction" not in outputs:
        zero = outputs["delta_target"].new_tensor(0.0)
        return {
            "vamp_loss": zero,
            "balance_loss": zero,
            "assignment_entropy": zero,
            "reconstruction_loss": zero,
            "base_transition_loss": zero,
            "base_transition_skill": zero,
            "rollout_transition_loss": zero,
            "rollout_transition_skill": zero,
            "mean_lambda2": zero,
        }
    vamp = metastable_vamp_terms(
        {
            "delta_target": outputs["delta_target"],
            "context_metastable_states": outputs["v4_slow_context_assignment"],
            "future_metastable_states": outputs["v4_slow_future_assignment"],
            "context_metastable_mask": outputs["v4_slow_context_mask"],
            "future_metastable_mask": outputs["v4_slow_future_mask"],
        },
        lags=lags,
    )
    context_mask = outputs["v4_slow_context_mask"]
    future_mask = outputs["v4_slow_future_mask"]
    reconstruction_loss = 0.5 * (
        F.smooth_l1_loss(
            outputs["v4_slow_context_reconstruction"][context_mask],
            outputs["v4_slow_context_normalized"][context_mask],
        )
        + F.smooth_l1_loss(
            outputs["v4_slow_future_reconstruction"][future_mask],
            outputs["v4_slow_future_normalized"][future_mask],
        )
    )
    base_target = outputs["v4_slow_base_target"].detach()
    base_prediction = outputs["v4_slow_base_prediction"]
    base_current = outputs["v4_slow_base_current"]
    base_transition_loss = F.mse_loss(base_prediction, base_target)
    base_persistence = F.mse_loss(base_current, base_target)
    rollout_target = outputs["v4_slow_target_assignment"].detach()
    rollout_prediction = outputs["v4_slow_prediction"]
    rollout_current = outputs["v4_slow_current_assignment"]
    rollout_transition_loss = F.mse_loss(rollout_prediction, rollout_target)
    rollout_persistence = F.mse_loss(rollout_current, rollout_target)
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvals(outputs["v4_slow_transition_matrix"]).abs()
        mean_lambda2 = eigenvalues.sort(dim=-1, descending=True).values[:, 1].mean().real
    return {
        "vamp_loss": vamp["vamp_loss"],
        "balance_loss": vamp["balance_loss"],
        "assignment_entropy": vamp["assignment_entropy"],
        "reconstruction_loss": reconstruction_loss,
        "base_transition_loss": base_transition_loss,
        "base_transition_skill": 1.0 - base_transition_loss / base_persistence.clamp(min=1.0e-8),
        "rollout_transition_loss": rollout_transition_loss,
        "rollout_transition_skill": 1.0
        - rollout_transition_loss / rollout_persistence.clamp(min=1.0e-8),
        "mean_lambda2": mean_lambda2,
    }


def _per_trajectory_vamp_terms(
    context_states: torch.Tensor,
    future_states: torch.Tensor,
    context_mask: torch.Tensor,
    future_mask: torch.Tensor,
    lags: tuple[int, ...],
    eps: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    """VAMP and occupancy computed after centering every trajectory independently."""
    scores = []
    occupancies = []
    entropies = []
    dim = context_states.shape[-1]
    eye = torch.eye(dim, device=context_states.device, dtype=context_states.dtype)
    for batch_index in range(context_states.shape[0]):
        sample_states = []
        for sequence, mask in (
            (context_states, context_mask), (future_states, future_mask)
        ):
            valid_states = sequence[batch_index][mask[batch_index]]
            if valid_states.numel():
                sample_states.append(valid_states)
                entropies.append(
                    -(valid_states.clamp(min=1.0e-8) * valid_states.clamp(min=1.0e-8).log())
                    .sum(dim=-1)
                    .mean()
                )
        if sample_states:
            occupancy = torch.cat(sample_states, dim=0).mean(dim=0).clamp(min=1.0e-8)
            occupancies.append(occupancy)
        for lag in lags:
            x_parts = []
            y_parts = []
            for sequence, mask in (
                (context_states, context_mask), (future_states, future_mask)
            ):
                if lag >= sequence.shape[1]:
                    continue
                pair_mask = mask[batch_index, :-lag] & mask[batch_index, lag:]
                if pair_mask.any():
                    x_parts.append(sequence[batch_index, :-lag][pair_mask])
                    y_parts.append(sequence[batch_index, lag:][pair_mask])
            if not x_parts:
                continue
            x = torch.cat(x_parts, dim=0)
            y = torch.cat(y_parts, dim=0)
            if x.shape[0] < dim + 2:
                continue
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)
            denominator = max(x.shape[0] - 1, 1)
            with torch.no_grad():
                c00 = x.T @ x / denominator + eps * eye
                c11 = y.T @ y / denominator + eps * eye
                s0, u0 = torch.linalg.eigh(c00)
                s1, u1 = torch.linalg.eigh(c11)
                w0 = (u0 * s0.clamp(min=eps).rsqrt().unsqueeze(0)) @ u0.T
                w1 = (u1 * s1.clamp(min=eps).rsqrt().unsqueeze(0)) @ u1.T
            koopman = (x @ w0).T @ (y @ w1) / denominator
            scores.append(koopman.square().sum())
    zero = context_states.new_tensor(0.0)
    if scores:
        score = torch.stack(scores).mean()
        vamp_loss = 1.0 - (score - 1.0) / max(dim - 1, 1)
    else:
        vamp_loss = zero
    if occupancies:
        occupancy = torch.stack(occupancies)
        uniform = 1.0 / float(dim)
        trajectory_balance = (
            occupancy * (occupancy / uniform).clamp(min=1.0e-8).log()
        ).sum(dim=-1).mean()
    else:
        trajectory_balance = zero
    assignment_entropy = torch.stack(entropies).mean() if entropies else zero
    return {
        "vamp_loss": vamp_loss,
        "trajectory_balance_loss": trajectory_balance,
        "assignment_entropy": assignment_entropy,
    }


def _whitened_cross_covariance_loss(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-3
) -> torch.Tensor:
    if x.shape[0] < 4:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    denominator = max(x.shape[0] - 1, 1)
    with torch.no_grad():
        def inverse_sqrt_covariance(values: torch.Tensor) -> torch.Tensor:
            covariance = values.T @ values / denominator
            covariance = covariance + eps * torch.eye(
                covariance.shape[0], device=values.device, dtype=values.dtype
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            inverse = eigenvalues.clamp(min=eps).rsqrt()
            return (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T

        x_whitener = inverse_sqrt_covariance(x)
        y_whitener = inverse_sqrt_covariance(y)
    cross = (x @ x_whitener).T @ (y @ y_whitener) / denominator
    return cross.square().mean()


def _lag_autocorrelation(values: torch.Tensor, mask: torch.Tensor, lag: int) -> torch.Tensor:
    if lag <= 0 or lag >= values.shape[1]:
        return values.new_tensor(0.0)
    pair_mask = mask[:, :-lag] & mask[:, lag:]
    if not pair_mask.any():
        return values.new_tensor(0.0)
    left = values[:, :-lag][pair_mask]
    right = values[:, lag:][pair_mask]
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    numerator = (left * right).mean()
    denominator = torch.sqrt(left.square().mean() * right.square().mean()).clamp(min=1.0e-8)
    return numerator / denominator


