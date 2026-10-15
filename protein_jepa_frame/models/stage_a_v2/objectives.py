"""Stage A V2 objectives: wires jepa_anything_core losses to StageAV2 forward outputs.

Loss composition (per STAGE_ABC_REDESIGN_PLAN.md §1.6):

    L_A = L_future + 0.5·L_masked + λ_o·L_orth/dH² + λ_f·V(y) + λ_h·V(H)

Where:
    L_future   = factor_prediction_loss(pred, target) for temporal forward pass
    L_masked   = factor_prediction_loss(pred, target) for spatial forward pass
    L_orth     = projector_orthogonality_loss(basis)    (Gram: within + cross)
    V(y)       = factor_activity_loss(target_factors)   (std floor on factors)
    V(H)       = encoder_variance_loss(context_H)       (std floor on encoder)

All losses are computed on valid (non-padding) positions only.
Padding and masked tokens are excluded from all variance statistics.

stop_grad contract:
    target_factors = P_j · stop_grad(h̄)  [in model.py forward]
    P_j is NOT detached — projectors get gradients from L_pred and L_orth.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

def _find_jepa_anything_src() -> Path:
    """Locate jepa-anything-core/src by walking up from this file."""
    rel = Path("jepa_anything") / "JEPA-Anything-main" / "jepa-anything-core" / "src"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / rel
        if candidate.is_dir():
            return candidate
    raise ImportError(
        f"could not locate {rel} in any parent of {Path(__file__).resolve()}"
    )


_ja_path = _find_jepa_anything_src()
if str(_ja_path) not in sys.path:
    sys.path.insert(0, str(_ja_path))

from jepa_anything_core.losses import (  # noqa: E402
    encoder_variance_loss,
    factor_activity_loss,
    factor_prediction_loss,
    projector_orthogonality_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Loss weights dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageAV2LossWeights:
    """All scalar loss weights for Stage A V2.

    Unused weights should be 0.0, not omitted — config fields must be consumed.
    """
    # Prediction losses
    future_weight: float = 1.0        # temporal mask prediction
    masked_weight: float = 0.5        # spatial mask prediction

    # Regularization
    orth_weight: float = 0.01         # λ_o; divided by dH² inside loss
    factor_activity_weight: float = 0.01   # λ_f; V(y)
    encoder_variance_weight: float = 0.01  # λ_h; V(H)

    # Temporal diversity: penalise H being constant along a trajectory.
    # encoder_variance_loss only checks variance ACROSS samples; without a
    # temporal term nothing stops the encoder collapsing to a time-invariant
    # identity code (observed cos(H_t, H_{t+Δ}) ≈ 0.999 at Δ = 400 ns).
    #
    # Implemented as a *squared hinge on the per-channel temporal variance*,
    # relu(γ² - Var_t)² / γ⁴  (see compute_stage_a_v2_loss).  Unlike the old
    # linear hinge max(0, γ - std) — which has zero gradient at std=0, exactly
    # the collapsed state — this keeps a finite, bounded penalty (≤ 1/channel)
    # and a usable gradient throughout the near-collapsed regime.
    #
    # NOTE: this is only a floor.  Its magnitude must stay small relative to
    # L_future, otherwise the encoder buys variance by *injecting noise* into H
    # rather than by encoding the conformational change that actually predicts
    # the target (a run with temporal_variance_weight=1.0 drove L_future to 0
    # while adding nothing decodable).  It is also applied only to the *near-
    # consecutive* frame differences, where real MD carries signal.
    temporal_variance_weight: float = 0.10  # λ_τ; temporal variance floor
    temporal_min_std: float = 0.40         # γ; std floor (squared internally)
    # γ=0.40 → γ²=0.16 >> natural per-step variance ~0.05, so the hinge
    # fires immediately and pushes H to vary along the trajectory.

    # Activity floor thresholds
    factor_min_std: float = 0.1       # γ for factor activity
    encoder_min_std: float = 0.1      # γ for encoder variance

    # Multi-lag weighting: if True, each sampled lag contributes equally to
    # L_future (per-lag means averaged with equal weight) instead of being
    # weighted by element count.  Kept False by default so the objective is the
    # plain mean the plan specifies; exposed for ablation.
    lag_balanced: bool = False

    eps: float = 1e-6

    def validate(self) -> None:
        """Raise if any weight is negative or non-finite."""
        for name, val in self.__dict__.items():
            if not isinstance(val, float):
                continue
            if not (val >= 0.0 and val < float("inf")):
                raise ValueError(
                    f"StageAV2LossWeights.{name} must be finite and non-negative, got {val}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Loss computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageAV2LossComponents:
    total: Tensor
    future: Tensor
    masked: Tensor
    orth: Tensor
    factor_activity: Tensor
    encoder_variance: Tensor
    temporal_variance: Tensor

    def as_loggable_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total.item(),
            "loss/future": self.future.item(),
            "loss/masked": self.masked.item(),
            "loss/orth": self.orth.item(),
            "loss/factor_activity": self.factor_activity.item(),
            "loss/encoder_variance": self.encoder_variance.item(),
            "loss/temporal_variance": self.temporal_variance.item(),
        }


def compute_stage_a_v2_loss(
    temporal_out: dict,
    spatial_out: dict | None,
    opf_basis: Tensor,
    weights: StageAV2LossWeights,
    dH: int,
) -> StageAV2LossComponents:
    """Compute the composite Stage A V2 loss.

    Args:
        temporal_out:  output of StageAV2.forward_temporal(); must contain
                       pred_factors [B,L,J,r], target_factors [B,L,J,r],
                       context_H [B,K,L,dH], valid_mask [B,L].
        spatial_out:   output of StageAV2.forward_spatial(); same keys but
                       pred/target are already gathered to [N_masked, J, r].
                       Pass None to skip the masked term (e.g. temporal-only step).
        opf_basis:     analysis_basis() from OrthogonalFactorProjection, shape
                       (J, r, dH).
        weights:       StageAV2LossWeights instance.
        dH:            latent_dim (used to normalise L_orth by dH²).

    Returns:
        StageAV2LossComponents with .total and each named component.
    """
    weights.validate()

    device = temporal_out["pred_factors"].device

    # ── Temporal (future) prediction loss ────────────────────────────────────
    # pred/target: [B, L, J, r]  —  all positions, including padding
    # We must exclude padding from the mean.
    valid = temporal_out["valid_mask"]   # [B, L]
    pred_t = temporal_out["pred_factors"]    # [B, L, J, r]
    tgt_t = temporal_out["target_factors"]   # [B, L, J, r]

    # Gather only valid residues to keep padding out of statistics
    pred_t_valid = pred_t[valid]   # [N_valid, J, r]
    tgt_t_valid = tgt_t[valid]     # [N_valid, J, r]

    if pred_t_valid.numel() == 0:
        L_future = torch.zeros((), device=device, requires_grad=True)
    else:
        L_future = factor_prediction_loss(pred_t_valid, tgt_t_valid)

    # ── Multi-step per-lag aggregation (when available) ──────────────────────
    # The predictor is time-conditioned, so a single batch holds samples drawn
    # from several lags.  `factor_prediction_loss` averages their squared errors
    # with equal weight.  Long lags are intrinsically less predictable, so we
    # additionally report (and optionally re-weight) per-lag means.  The default
    # `lag_balanced=False` keeps the plain mean so no signal is lost; the per-lag
    # breakdown is logged for diagnosis.
    lag_slot = temporal_out.get("lag_slot")
    lag_balanced = weights.lag_balanced
    lag_losses: list[tuple[float, Tensor]] = []
    if lag_slot is not None and pred_t_valid.numel() > 0:
        per_slot = pred_t[valid]                       # [N_valid, J, r]
        tgt_slot = tgt_t[valid]
        lag_slot_valid = lag_slot.unsqueeze(1).expand(-1, valid.shape[1])[valid]
        for s in lag_slot_valid.unique().tolist():
            sel = lag_slot_valid == s
            if sel.any():
                lag_losses.append(
                    (int(s), factor_prediction_loss(per_slot[sel], tgt_slot[sel]))
                )
        if lag_balanced and lag_losses:
            n_lags = lag_slot.unique().numel()
            # `factor_prediction_loss` returns a mean over elements; scale each
            # lag term by (elements_in_lag / total_elements) after normalising
            # per-lag means so every lag contributes equally.
            counts = torch.tensor(
                [float((lag_slot_valid == s).sum()) for s, _ in lag_losses],
                device=device,
            )
            fractions = counts / counts.sum().clamp_min(1.0)
            L_future = sum(
                fractions[i] * loss for i, (_, loss) in enumerate(lag_losses)
            )

    # ── Spatial (masked) prediction loss ─────────────────────────────────────
    if spatial_out is not None and weights.masked_weight > 0.0:
        pred_s = spatial_out["pred_factors"]    # [N_masked, J, r]
        tgt_s = spatial_out["target_factors"]   # [N_masked, J, r]
        if pred_s.numel() == 0:
            L_masked = torch.zeros((), device=device, requires_grad=True)
        else:
            L_masked = factor_prediction_loss(pred_s, tgt_s)
    else:
        L_masked = torch.zeros((), device=device)

    # ── OPF Gram orthogonality loss ───────────────────────────────────────────
    # basis: (J, r, dH) — matches (K, r, d) convention in jepa_anything_core
    L_orth_raw = projector_orthogonality_loss(opf_basis, reduction="sum")
    L_orth = L_orth_raw / (dH * dH)   # normalise by dH² per plan §1.6

    # ── Factor activity loss V(y) ─────────────────────────────────────────────
    # Use temporal target factors on valid residues as the batch of samples.
    # These are stop_grad (P_j · sg(h̄)); gradients through P_j are preserved.
    if tgt_t_valid.numel() == 0:
        L_factor_act = torch.zeros((), device=device)
    else:
        # tgt_t_valid: [N_valid, J, r] — leading dim is samples, last two are (K, r)
        L_factor_act = factor_activity_loss(
            tgt_t_valid,
            min_std=weights.factor_min_std,
            eps=weights.eps,
        )

    # ── Encoder variance loss V(H) ────────────────────────────────────────────
    # context_H: [B, K, L, dH] or [B, L, dH] depending on forward path.
    context_H = temporal_out["context_H"]   # [B, K, L, dH]
    # valid_mask [B, L] needs to expand to cover all K frames if K>1
    if context_H.dim() == 4:
        B, K, L, D = context_H.shape
        H_flat = context_H.reshape(B * K, L, D)    # treat B*K as samples
        valid_flat = valid.unsqueeze(1).expand(B, K, L).reshape(B * K, L)
    else:
        H_flat = context_H        # [B, L, D]
        valid_flat = valid        # [B, L]

    if H_flat[valid_flat].numel() == 0:
        L_enc_var = torch.zeros((), device=device)
    else:
        L_enc_var = encoder_variance_loss(
            H_flat,
            valid_mask=valid_flat,
            min_std=weights.encoder_min_std,
            eps=weights.eps,
        )

    # ── Temporal contrast loss — hinge on per-step squared H distance ────────
    # Penalise H being constant ALONG a trajectory.
    #
    # IMPORTANT: no penalty that is a pure function of (H_t - H_{t-1}) can break
    # the collapse symmetry on its own — H_t = H_{t-1} is a stationary point of
    # every such term (the gradient ∝ (H_t - H_{t-1}) vanishes there).  The
    # actual symmetry-breaking signal is the prediction-task asymmetry: the
    # anchor frame is now spatially masked before entering the predictor (see
    # forward_temporal_multilag), so the predictor cannot copy the anchor's H to
    # satisfy the objective and the encoder must make H frame-sensitive to be
    # predictable at all.  This term is only a *floor* that keeps the near-
    # collapsed regime from being a flat basin: as soon as the steps are
    # nonzero it supplies a strong gradient proportional to -(H_t - H_{t-1}),
    # pushing consecutive frames apart until the floor is met.
    #
    # Form (squared-distance hinge, differentiable everywhere incl. a step of 0,
    # where the gradient is exactly 0 by symmetry but the penalty is at its
    # maximum so any perturbation is rewarded):
    #
    #   s_t        = ||H_t - H_{t-1}||^2 / dH          (mean-square per channel)
    #   L_temporal = mean_{t,i valid} relu(γ² - s_t)
    #
    # γ (temporal_min_std) is the floor on the per-step RMS channel distance.
    #
    # Use context_H_full (unmasked full trajectory) when available so the
    # contrast is on the real H values, not the spatially-masked predictor input.
    ctx_full = temporal_out.get("context_H_full", temporal_out.get("context_H"))
    if ctx_full is not None and ctx_full.dim() == 4 and ctx_full.shape[1] > 1:
        H_seq = ctx_full  # [B, K, L, dH]
        diffs = H_seq[:, 1:] - H_seq[:, :-1]                # [B, K-1, L, dH]
        valid3 = valid.unsqueeze(1).expand(-1, diffs.shape[1], -1)  # [B, K-1, L]
        dH = H_seq.shape[-1]
        # Per-step mean-square channel distance: ||H_t - H_{t-1}||² / dH
        s_t = diffs.pow(2).mean(dim=-1)                     # [B, K-1, L]
        gamma_sq = weights.temporal_min_std ** 2
        deficit = torch.clamp(gamma_sq - s_t, min=0.0)      # hinge on s_t
        vf = valid3.float()
        n_valid = vf.sum().clamp(min=1.0)
        # Normalise by γ² so the term is in [0,1] and comparable across γ.
        L_temporal = (deficit * vf).sum() / (n_valid * gamma_sq + weights.eps)
    else:
        L_temporal = torch.zeros((), device=device)

    # ── Composite ─────────────────────────────────────────────────────────────
    total = (
        weights.future_weight * L_future
        + weights.masked_weight * L_masked
        + weights.orth_weight * L_orth
        + weights.factor_activity_weight * L_factor_act
        + weights.encoder_variance_weight * L_enc_var
        + weights.temporal_variance_weight * L_temporal
    )

    return StageAV2LossComponents(
        total=total,
        future=L_future.detach(),
        masked=L_masked.detach(),
        orth=L_orth.detach(),
        factor_activity=L_factor_act.detach(),
        encoder_variance=L_enc_var.detach(),
        temporal_variance=L_temporal.detach(),
    )
