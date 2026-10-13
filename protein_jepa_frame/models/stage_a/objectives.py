"""Stage A loss functions: masked JEPA prediction + multi-horizon endpoint forecast.

All losses operate on either mask-prediction outputs or paired-endpoint outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="none")


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average loss over valid (masked) positions."""
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def stage_a_mask_loss(
    outputs: dict,
    residue_weight: float = 1.0,
    state_weight: float = 1.0,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Joint masked prediction loss: residue JEPA + state JEPA.
    
    Args:
        outputs: from StageAJEPA.forward_mask_prediction()
        residue_weight: weight for residue-level loss
        state_weight: weight for frame-level state loss
        loss_type: "smooth_l1" or "mse"
    """
    loss_fn = _smooth_l1 if loss_type == "smooth_l1" else F.mse_loss
    if loss_type == "mse":
        # mse_loss uses kwargs pattern
        def _mse(pred, target):
            return F.mse_loss(pred, target, reduction="none")
        loss_fn = _mse

    total = torch.tensor(0.0, device=outputs["predictions"].device)

    # Residue-level JEPA loss
    if residue_weight > 0:
        pred = outputs["predictions"]
        target = outputs["targets"]
        mask = outputs["target_mask"].unsqueeze(-1).float()
        raw = loss_fn(pred, target)
        total += residue_weight * _masked_mean(raw, mask)

    # Frame-level state prediction loss
    if state_weight > 0 and "state_predictions" in outputs:
        pred = outputs["state_predictions"]
        target = outputs["state_targets"]
        mask = outputs["state_target_mask"].unsqueeze(-1).float()
        raw = loss_fn(pred, target)
        total += state_weight * _masked_mean(raw, mask)

    return total


def stage_a_horizon_loss(
    outputs: dict,
    loss_type: str = "smooth_l1",
) -> dict[str, torch.Tensor]:
    """Multi-horizon endpoint forecast losses.
    
    Args:
        outputs: from StageAJEPA.forward_paired()
        loss_type: "smooth_l1" or "mse"
        
    Returns:
        dict with individual loss components
    """
    loss_fn = _smooth_l1 if loss_type == "smooth_l1" else F.mse_loss
    if loss_type == "mse":
        def _mse(p, t):
            return F.mse_loss(p, t, reduction="none")
        loss_fn = _mse

    losses = {}
    
    # Delta (endpoint change) prediction loss
    delta_pred = outputs["delta_prediction"]
    delta_target = outputs["delta_target"]
    raw_delta = loss_fn(delta_pred, delta_target)
    losses["delta_loss"] = raw_delta.mean()

    # State prediction loss (full endpoint)
    if "state_prediction" in outputs:
        state_pred = outputs["state_prediction"]
        future = outputs["future_endpoint"]
        raw_state = loss_fn(state_pred, future)
        losses["state_prediction_loss"] = raw_state.mean()

    # Persistence baseline (for monitoring, no gradient)
    if "context_endpoint" in outputs:
        with torch.no_grad():
            persistence = loss_fn(
                outputs["context_endpoint"], outputs["future_endpoint"]
            ).mean()
        losses["persistence_baseline"] = persistence

    return losses


def stage_a_displacement_loss(
    outputs: dict,
    dir_weight: float = 1.0,
    mag_weight: float = 1.0,
    target_scale: float = 8.0,
) -> torch.Tensor:
    """Per-residue displacement prediction loss (iter_009 / iter_010).

    Two terms:
      direction: 1 - cosine_similarity(pred, target), over valid residues with
                 non-trivial target displacement (|target| > 0.1 Å).
      magnitude: smooth_l1 on predicted vs target displacement magnitude,
                 with both sides divided by ``target_scale`` first so the
                 regression lives in O(1) units (raw targets are tens of Å and
                 make the gradient explode / the head collapse).

    Args:
        outputs: from StageAJEPA.forward_displacement()
        dir_weight: weight for direction cosine loss
        mag_weight: weight for magnitude loss
        target_scale: Å scale to normalise the magnitude term

    Returns:
        scalar loss
    """
    pred   = outputs["disp_pred"]    # [B, L, 3]
    target = outputs["disp_target"]  # [B, L, 3]
    valid  = outputs["valid_mask"]   # [B, L] bool

    # magnitude of target [B, L]
    target_mag = target.norm(dim=-1)
    # only supervise direction for residues with non-trivial motion
    dir_valid = valid & (target_mag > 0.1)

    total = torch.tensor(0.0, device=pred.device)

    if dir_weight > 0 and dir_valid.any():
        cos = F.cosine_similarity(
            pred[dir_valid],    # [N, 3]
            target[dir_valid],  # [N, 3]
            dim=-1,
        )
        dir_loss = (1.0 - cos).mean()
        total = total + dir_weight * dir_loss

    if mag_weight > 0 and valid.any():
        s = float(max(target_scale, 1e-6))
        pred_mag = pred[valid].norm(dim=-1) / s   # [N]
        tgt_mag  = target_mag[valid] / s           # [N]
        mag_loss = F.smooth_l1_loss(pred_mag, tgt_mag, reduction="mean")
        total = total + mag_weight * mag_loss

    return total


def temporal_variance_regularizer(
    frame_states: torch.Tensor,
    frame_mask: torch.Tensor,
    target_std: float = 1.0,
) -> torch.Tensor:
    """VICReg-style hinge on the per-dimension temporal spread of frame states.

    Penalises only *deficient* spread (std < target_std); it does NOT reward
    unbounded growth, so it cannot be gamed by amplitude inflation. Operates over
    the (B, T) sample axis so it targets the TIME axis specifically: a temporally
    frozen H (identical vector every frame) has zero std and takes full penalty.

    Args:
        frame_states: [B, T, D]
        frame_mask:   [B, T] bool valid frames
        target_std:   spread floor (std below this is penalised)
    """
    if frame_states is None or frame_states.numel() == 0:
        return torch.zeros((), device=frame_states.device if frame_states is not None else "cpu")
    m = frame_mask.bool()
    flat = frame_states[m]                       # [N, D]
    if flat.shape[0] < 2:
        return frame_states.new_tensor(0.0)
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1.0e-6)
    return torch.relu(target_std - std).mean()


def temporal_covariance_regularizer(
    frame_states: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Decorrelate latent dimensions across time (prevents dimensional redundancy).

    Off-diagonal squared covariance of mean-centred, unit-variance frame states
    over the (B, T) axis, normalised by D. Encourages H's temporal variation to
    occupy many independent directions rather than one dominant mode.
    """
    m = frame_mask.bool()
    flat = frame_states[m]                       # [N, D]
    if flat.shape[0] < 2:
        return frame_states.new_tensor(0.0)
    x = flat - flat.mean(dim=0, keepdim=True)
    x = x / torch.sqrt(x.var(dim=0, unbiased=False, keepdim=True) + 1.0e-4)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return off.square().sum() / x.shape[1]


def temporal_velocity_loss(
    frame_states: torch.Tensor,
    frame_mask: torch.Tensor,
    step_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Self-supervised within-context motion objective (no future leakage).

    Forces consecutive frame states to differ, i.e. H must encode a *velocity*,
    not a constant. Uses smooth-L1 on the normalised frame-to-frame delta so the
    encoder is rewarded for carrying per-frame motion information. A static code
    (all frames identical) makes the delta zero and is penalised.
    """
    if frame_states.shape[1] < 2:
        return frame_states.new_tensor(0.0)
    d = frame_states[:, 1:] - frame_states[:, :-1]           # [B, T-1, D]
    valid = frame_mask[:, 1:] & frame_mask[:, :-1]           # [B, T-1]
    if not valid.any():
        return frame_states.new_tensor(0.0)
    # normalise by the per-sample state norm so the term measures RELATIVE motion
    scale = frame_states.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1e-3)
    d = d / scale
    loss = F.smooth_l1_loss(d, torch.zeros_like(d), reduction="none")
    w = valid.unsqueeze(-1).float()
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def coordinate_activity_regularizer(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
    target_std: float = 0.02,
) -> torch.Tensor:
    """VICReg variance floor over valid residue/time tokens.

    This is deliberately a floor rather than a target variance: it prevents
    temporal/per-residue H from becoming constant without rewarding unlimited
    amplitude.  The mask is kept separate from the feature tensor so padded
    residues cannot manufacture activity.
    """
    if features is None or features.numel() == 0:
        return torch.zeros((), device=features.device if features is not None else "cpu")
    flat = features[valid_mask.bool()]
    if flat.shape[0] < 2:
        return features.new_tensor(0.0)
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + 1.0e-6)
    return F.relu(float(target_std) - std).mean()


def coordinate_covariance_regularizer(
    features: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize off-diagonal covariance without requiring a task label."""
    if features is None or features.numel() == 0:
        return torch.zeros((), device=features.device if features is not None else "cpu")
    flat = features[valid_mask.bool()]
    if flat.shape[0] < 2:
        return features.new_tensor(0.0)
    centered = flat - flat.mean(dim=0, keepdim=True)
    centered = centered / torch.sqrt(centered.var(dim=0, unbiased=False, keepdim=True) + 1.0e-4)
    cov = centered.transpose(0, 1) @ centered / max(1, centered.shape[0] - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return off.square().mean()


def stage_a_dynamic_loss(
    outputs: dict,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """Self-supervised future-view dynamic JEPA loss at residue resolution."""
    if "dynamic_prediction" not in outputs:
        return outputs["context_endpoint"].new_tensor(0.0)
    pred = outputs["dynamic_prediction"]
    target = outputs["dynamic_target"]
    mask = outputs["dynamic_target_mask"].bool()
    if not mask.any():
        return pred.new_tensor(0.0)
    if loss_type == "mse":
        raw = F.mse_loss(pred, target, reduction="none")
    else:
        raw = F.smooth_l1_loss(pred, target, reduction="none")
    return raw[mask].mean()



def stage_a_total_loss(
    mask_outputs: dict | None,
    paired_outputs: dict | None,
    mask_weight: float = 1.0,
    paired_weight: float = 1.0,
    residue_weight: float = 1.0,
    state_weight: float = 1.0,
    loss_type: str = "smooth_l1",
    dynamic_jepa_weight: float = 0.0,
    dynamic_activity_weight: float = 0.0,
    dynamic_covariance_weight: float = 0.0,
    dynamic_variance_floor: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total Stage A loss as weighted combination of mask + paired objectives.
    
    Args:
        mask_outputs: from mask prediction forward (can be None)
        paired_outputs: from paired forward (can be None)
        mask_weight: overall weight for mask loss
        paired_weight: overall weight for paired endpoint loss
        residue_weight: residue-level weight within mask loss
        state_weight: state-level weight within mask loss
        loss_type: "smooth_l1" or "mse"
        
    Returns:
        (total_loss, log_dict with all components)
    """
    device = (
        mask_outputs["predictions"].device
        if mask_outputs is not None
        else paired_outputs["delta_prediction"].device
    )
    total = torch.tensor(0.0, device=device)
    log = {}

    if mask_outputs is not None:
        mask_loss = stage_a_mask_loss(
            mask_outputs, residue_weight=residue_weight,
            state_weight=state_weight, loss_type=loss_type,
        )
        total += mask_weight * mask_loss
        log["mask_loss"] = float(mask_loss.detach().cpu())

    if paired_outputs is not None:
        hlosses = stage_a_horizon_loss(paired_outputs, loss_type=loss_type)
        paired_loss = hlosses["delta_loss"]
        total += paired_weight * paired_loss
        log["paired_delta_loss"] = float(paired_loss.detach().cpu())
        if "state_prediction_loss" in hlosses:
            log["paired_state_loss"] = float(
                hlosses["state_prediction_loss"].detach().cpu()
            )
        if "persistence_baseline" in hlosses:
            log["persistence_baseline"] = float(
                hlosses["persistence_baseline"].detach().cpu()
            )

        if dynamic_jepa_weight > 0 and "dynamic_prediction" in paired_outputs:
            dynamic_loss = stage_a_dynamic_loss(paired_outputs, loss_type=loss_type)
            total += paired_weight * dynamic_jepa_weight * dynamic_loss
            log["dynamic_jepa_loss"] = float(dynamic_loss.detach().cpu())

    # The activity/covariance terms are applied to online H only.  They are
    # structural anti-collapse terms and remain valid on both masked clips and
    # paired context windows.
    activity_source = None
    activity_mask = None
    if mask_outputs is not None and "dynamic_residue_states" in mask_outputs:
        activity_source = mask_outputs["dynamic_residue_states"]
        activity_mask = mask_outputs.get("context_mask", mask_outputs.get("residue_mask"))
    elif paired_outputs is not None and "context_dynamic_endpoint" in paired_outputs:
        activity_source = paired_outputs["context_dynamic_endpoint"]
        activity_mask = paired_outputs.get("dynamic_context_mask")
    if activity_source is not None and activity_mask is not None:
        activity_loss = coordinate_activity_regularizer(
            activity_source, activity_mask, target_std=dynamic_variance_floor
        )
        covariance_loss = coordinate_covariance_regularizer(activity_source, activity_mask)
        total = total + dynamic_activity_weight * activity_loss
        total = total + dynamic_covariance_weight * covariance_loss
        log["dynamic_activity_loss"] = float(activity_loss.detach().cpu())
        log["dynamic_covariance_loss"] = float(covariance_loss.detach().cpu())

    log["total_loss"] = float(total.detach().cpu())
    return total, log
