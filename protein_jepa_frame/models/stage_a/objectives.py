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


def stage_a_total_loss(
    mask_outputs: dict | None,
    paired_outputs: dict | None,
    mask_weight: float = 1.0,
    paired_weight: float = 1.0,
    residue_weight: float = 1.0,
    state_weight: float = 1.0,
    loss_type: str = "smooth_l1",
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

    log["total_loss"] = float(total.detach().cpu())
    return total, log
