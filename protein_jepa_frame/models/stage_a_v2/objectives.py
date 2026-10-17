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
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch-aligned CA RMSD helper
# ─────────────────────────────────────────────────────────────────────────────

def _kabsch_rmsd(P: Tensor, Q: Tensor, mask: Tensor) -> Tensor:
    """Kabsch-aligned per-sample CA RMSD.

    Args:
        P:    [B, L, 3]  anchor CA coords
        Q:    [B, L, 3]  target CA coords
        mask: [B, L]     float or bool; 1 = valid residue

    Returns:
        rmsd: [B]  RMSD in Å after optimal rigid alignment
    """
    m = mask.float()  # [B, L]
    n = m.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]

    # Centre on centroid of valid residues
    p_c = (P * m.unsqueeze(-1)).sum(1, keepdim=True) / n.unsqueeze(-1)  # [B,1,3]
    q_c = (Q * m.unsqueeze(-1)).sum(1, keepdim=True) / n.unsqueeze(-1)
    p = (P - p_c) * m.unsqueeze(-1)  # [B, L, 3]
    q = (Q - q_c) * m.unsqueeze(-1)

    # Covariance H = P^T Q  (sum over valid residues)
    H = torch.bmm(p.transpose(1, 2), q)  # [B, 3, 3]
    U, S, Vh = torch.linalg.svd(H)       # U [B,3,3], S [B,3], Vh [B,3,3]

    # Ensure proper rotation (det = +1)
    d = torch.linalg.det(torch.bmm(Vh.transpose(-1, -2), U.transpose(-1, -2)))  # [B]
    diag = torch.stack([
        torch.ones_like(d), torch.ones_like(d), d
    ], dim=-1)  # [B, 3]
    R = torch.bmm(
        Vh.transpose(-1, -2) * diag.unsqueeze(-2),
        U.transpose(-1, -2),
    )  # [B, 3, 3]

    # Rotate P, compute RMSD
    p_rot = torch.bmm(p, R.transpose(-1, -2))  # [B, L, 3]
    diff_sq = ((p_rot - q) ** 2).sum(-1) * m   # [B, L]
    rmsd = (diff_sq.sum(-1) / n.squeeze(-1)).sqrt()  # [B]
    return rmsd


def compute_temporal_contrast_loss(
    H_seq: Tensor,
    coords_seq: Tensor,
    valid: Tensor,
    rmsd_scale: float = 2.0,
) -> Tensor:
    """Supervised temporal contrastive loss — ranking + regression hybrid.

    For every consecutive frame pair (i, i+1) in the context window:

    1. MSE regression: regress normalised H-distance to tanh(CA_RMSD / rmsd_scale).
    2. Ranking (triplet) loss: for every pair of frame-pairs within a batch,
       enforce that the pair with higher CA RMSD has larger H-distance by at
       least a margin of 0.05.  This prevents the trivial solution of shrinking
       all H-distances toward zero, which satisfies regression when most
       consecutive RMSD values are small (tanh(small)≈0) but doesn't teach
       the encoder to scale H-distance with structural change.

    L = L_regression + L_ranking

    Args:
        H_seq:      [B, K, L, dH]  full (unmasked) context H
        coords_seq: [B, K, L, 4, 3]  backbone coords; CA is index 1
        valid:      [B, L]  bool/float residue mask
        rmsd_scale: Å; tanh saturation point (~2 Å maps to tanh≈0.76)

    Returns:
        scalar loss averaged over all valid consecutive pairs
    """
    B, K, L, dH = H_seq.shape

    ca = coords_seq[:, :, :, 1, :]  # [B, K, L, 3]  CA only
    m = valid.float()                # [B, L]
    L_valid = m.sum(-1).clamp(min=1.0)  # [B]

    reg_losses = []
    rank_losses = []
    margin = 0.05  # minimum H-distance gap between a far pair and a near pair

    for k in range(K - 1):
        h_i = H_seq[:, k]       # [B, L, dH]
        h_j = H_seq[:, k + 1]   # [B, L, dH]
        ca_i = ca[:, k]
        ca_j = ca[:, k + 1]

        rmsd = _kabsch_rmsd(ca_i, ca_j, valid).detach()  # [B]

        dH_dist = ((h_i - h_j) * m.unsqueeze(-1)).pow(2).sum((-1, -2))  # [B]
        h_dist = (dH_dist / (L_valid * dH)).sqrt()  # [B]

        target = torch.tanh(rmsd / rmsd_scale)  # [B]
        reg_losses.append(F.mse_loss(h_dist, target))

        # Ranking: all pairs (a, b) where rmsd[a] > rmsd[b] + margin_rmsd.
        # Enforce h_dist[a] > h_dist[b] + margin.
        # Use margin_rmsd = rmsd_scale * 0.25 (0.5 Å for default scale=2.0).
        if B >= 2:
            margin_rmsd = rmsd_scale * 0.25
            # [B, B] pairwise RMSD difference: rmsd[a] - rmsd[b]
            rmsd_diff = rmsd.unsqueeze(1) - rmsd.unsqueeze(0)  # [B, B]
            # positive where a has higher RMSD than b by enough
            pos_pairs = (rmsd_diff > margin_rmsd).float()  # [B, B]
            n_pos = pos_pairs.sum().clamp(min=1.0)
            # h_dist[a] should exceed h_dist[b] by margin
            h_diff = h_dist.unsqueeze(1) - h_dist.unsqueeze(0)  # [B, B]
            rank_viol = F.relu(margin - h_diff) * pos_pairs  # hinge
            rank_losses.append(rank_viol.sum() / n_pos)

    if not reg_losses:
        return H_seq.new_zeros(())

    L_reg = torch.stack(reg_losses).mean()
    L_rank = torch.stack(rank_losses).mean() if rank_losses else L_reg.new_zeros(())
    return L_reg + L_rank

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
    temporal_variance_weight: float = 0.05  # λ_τ; temporal variance floor
    temporal_min_std: float = 0.10         # γ; std floor (squared internally)
    # Fix 5: lowered from 0.40 → 0.10 so the floor is a gentle backstop, not
    # a dominant gradient forcing noise injection.  γ=0.10 → γ²=0.01, still
    # well above the natural per-step variance ~0.001 so the hinge fires, but
    # no longer so aggressive that it overwhelms the prediction signal.

    # Supervised temporal contrastive loss: regress normalised H-distance onto
    # tanh(CA_RMSD / rmsd_scale) so the encoder embeds conformational distance
    # directly into H-space distance.  This is a direct supervised signal that
    # breaks time-invariance without injecting unstructured noise.
    #
    # Loss = mean over frame pairs (i,j):
    #   (||H_i - H_j||_F / sqrt(L·dH)  -  tanh(RMSD_ij / rmsd_scale))²
    #
    # rmsd_scale=2.0 Å maps the 0–4 Å range to tanh 0–0.96,
    # so even small motions (1 Å) get a nonzero target (tanh≈0.46).
    temporal_contrast_weight: float = 0.0    # λ_c; set >0 to enable
    temporal_contrast_rmsd_scale: float = 2.0  # Å; controls target saturation

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
    temporal_contrast: Tensor

    def as_loggable_dict(self) -> dict[str, float]:
        return {
            "loss/total": self.total.item(),
            "loss/future": self.future.item(),
            "loss/masked": self.masked.item(),
            "loss/orth": self.orth.item(),
            "loss/factor_activity": self.factor_activity.item(),
            "loss/encoder_variance": self.encoder_variance.item(),
            "loss/temporal_variance": self.temporal_variance.item(),
            "loss/temporal_contrast": self.temporal_contrast.item(),
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
    # Use context_H_full (the unmasked online encoder output for all K frames)
    # rather than context_H (H_target_in), which contains zero-filled slots for
    # frames beyond the visible history.  Zero-fill artificially deflates the
    # cross-sample variance and causes the loss to over-estimate collapse.
    context_H = temporal_out.get("context_H_full", temporal_out["context_H"])
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

    # ── Supervised temporal contrastive loss ─────────────────────────────────
    # Regress normalised H-distance onto tanh(CA_RMSD / rmsd_scale) for each
    # consecutive frame pair in the context window.  This gives the encoder a
    # direct structural supervision signal: pairs that move more (higher RMSD)
    # must produce more distant H embeddings.  Unlike the unsupervised floor
    # above, this loss specifies *which* directions in H-space should vary
    # (those correlated with structural change) and cannot be satisfied by noise.
    L_contrast = torch.zeros((), device=device)
    if weights.temporal_contrast_weight > 0.0:
        ctx_full_c = temporal_out.get("context_H_full")
        ctx_coords = temporal_out.get("context_coords")   # injected by training loop
        if (ctx_full_c is not None and ctx_coords is not None
                and ctx_full_c.dim() == 4 and ctx_full_c.shape[1] > 1):
            L_contrast = compute_temporal_contrast_loss(
                H_seq=ctx_full_c,
                coords_seq=ctx_coords,
                valid=valid.float(),
                rmsd_scale=weights.temporal_contrast_rmsd_scale,
            )

    # ── Composite ─────────────────────────────────────────────────────────────
    total = (
        weights.future_weight * L_future
        + weights.masked_weight * L_masked
        + weights.orth_weight * L_orth
        + weights.factor_activity_weight * L_factor_act
        + weights.encoder_variance_weight * L_enc_var
        + weights.temporal_variance_weight * L_temporal
        + weights.temporal_contrast_weight * L_contrast
    )

    return StageAV2LossComponents(
        total=total,
        future=L_future.detach(),
        masked=L_masked.detach(),
        orth=L_orth.detach(),
        factor_activity=L_factor_act.detach(),
        encoder_variance=L_enc_var.detach(),
        temporal_variance=L_temporal.detach(),
        temporal_contrast=L_contrast.detach(),
    )
