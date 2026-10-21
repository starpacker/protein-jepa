"""Objectives for the from-none Stage B pipeline.

The career of these losses is to force a *from-scratch* encoder to produce an H
whose inter-frame change is geometrically decodable, while the H/S kernel learns
an autonomous multi-lag law.  Three terms:

  energy_score        : distributional term on the rollout H path (proper ES).
  ca_displacement     : differentiable Kabsch-aligned CA point loss (decodability).
  physical_energy     : energy score on the decoded backbone physical features.

The CA term is the one Stage A never had; without it a same-frame invariant H
optimises the ES by moving in a direction that is not aligned with real motion
(measured R^2 < 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# -----------------------------------------------------------------------------
# Differentiable Kabsch alignment (row-vector convention: x @ R + t)
# -----------------------------------------------------------------------------

def kabsch_rotation(pred: Tensor, true: Tensor) -> tuple[Tensor, Tensor]:
    """Least-squares rotation/translation mapping ``pred`` onto ``true``.

    ``pred`` / ``true``: [..., N, 3] (last two dims are residues and xyz; the
    leading dims are batch).  Returns ``(R, t)`` with ``x @ R + t`` convention,
    differentiable through the SVD.  Fully batched (no python loop).
    """
    pmean = pred.mean(-2, keepdim=True)
    qmean = true.mean(-2, keepdim=True)
    pc = pred - pmean
    qc = true - qmean
    H = pc.transpose(-1, -2) @ qc                     # [..., 3, 3]
    U, _, Vh = torch.linalg.svd(H)
    sgn = torch.sign(torch.det(U @ Vh))               # [...,] proper-rotation sign
    D = torch.diag_embed(torch.stack([
        torch.ones_like(sgn), torch.ones_like(sgn), sgn], dim=-1))
    R = U @ D @ Vh                                    # maps centered pred -> centered true
    # Full affine map so pred @ R + t == true (translation absorbs both means).
    t = qmean - pmean @ R
    return R, t


def aligned_ca_rmsd(pred_ca: Tensor, true_ca: Tensor, mask: Tensor) -> Tensor:
    """Batched Kabsch CA RMSD (A); differentiable through the rotation.

    ``pred_ca``/``true_ca``: [B, L, 3]; ``mask``: [B, L].  Uses a masked
    Kabsch: invalid residues are replaced by the masked centroid so they
    contribute exactly zero error and do not bias the rotation.
    """
    m = mask.unsqueeze(-1).to(pred_ca.dtype)                 # [B, L, 1]
    valid = mask.sum(-1)                                     # [B]
    if int((valid < 3).sum()) == pred_ca.shape[0]:
        return pred_ca.new_zeros(())
    # Fill invalid residues with the per-sample valid centroid so they cancel.
    def fill(x):
        c = (x * m).sum(-2, keepdim=True) / m.sum(-2, keepdim=True).clamp_min(1.0)
        return x * m + c * (1.0 - m)
    p, q = fill(pred_ca), fill(true_ca)
    R, t = kabsch_rotation(p, q)
    aligned = p @ R + t
    diff = (aligned - q) * m
    # squared error is already summed over xyz; the mean is over valid residues.
    per_b = diff.square().sum(-1).sum(-1) / m.sum(-2).squeeze(-1).clamp_min(1e-8)
    good = (valid >= 3)
    if not bool(good.any()):
        return pred_ca.new_zeros(())
    return per_b[good].clamp_min(1e-8).sqrt().mean()


def persistence_ca_rmsd(true_ca: Tensor, anchor_ca: Tensor, mask: Tensor) -> Tensor:
    """Kabsch CA RMSD of the no-op forecast (pred = anchor), for skill logging."""
    return aligned_ca_rmsd(anchor_ca, true_ca, mask)


# -----------------------------------------------------------------------------
# Energy score (proper scoring rule, as in stage_b1b.objectives)
# -----------------------------------------------------------------------------

def energy_score(samples: Tensor, target: Tensor) -> Tensor:
    """ES_hat = (1/M) sum_m ||v_m - v*|| - (1/(2M(M-1))) sum_{m!=m'} ||v_m - v_m'||.

    ``samples``: [M, D]; ``target``: [D].
    """
    if samples.dim() != 2:
        raise ValueError(f"samples must be [M, D], got {tuple(samples.shape)}")
    M = samples.shape[0]
    if M < 2:
        raise ValueError("energy_score needs M >= 2 samples")
    d_target = (samples - target.unsqueeze(0)).norm(dim=-1)
    if M < 2:
        return d_target.mean()
    diff = samples.unsqueeze(0) - samples.unsqueeze(1)
    dist = diff.norm(dim=-1)
    off = dist.sum() / (M * (M - 1)) if M > 1 else d_target.new_zeros(())
    return d_target.mean() - 0.5 * off


# -----------------------------------------------------------------------------
# Ensemble (distributional) losses on the DECODED CA cloud
# -----------------------------------------------------------------------------
# The point CA-RMSD term alone is satisfied by a collapsed, zero-variance
# decoder: on any trajectory the minimum of E||x_hat - x|| is the conditional
# mean, and a point estimator has no incentive to spread its M rollouts.  The
# ensemble metrics that matter (rmsf_mae, variance_w2, root_mean_w2) all compare
# a *distribution* of samples against the true trajectory's spread, so a
# collapsed model can score worse than persistence despite a lower point error.
# These three terms give the rollout that missing incentive.

def _ensemble_stats(ca_pred: Tensor, true_ca: Tensor, mask: Tensor):
    """Model per-time sample variance and true cumulative-in-time variance.

    ``ca_pred``: [M, B, T, L, 3] (M stochastic rollout samples at each step).
    ``true_ca``: [B, T, L, 3]  (the true frames at the same steps).
    ``mask``:    [B, L].

    Model side: spread *across the M samples* at each step -- the uncertainty
    the rollout actually expresses.
    True side:  spread *across steps 0..t* of the real trajectory -- how far the
    system has genuinely explored by step t.

    The spread is measured *after* Kabsch-aligning every cloud onto a common
    reference, exactly as ``evaluation.ensemble_metrics.align_ensemble`` does.
    This matters: absolute MD frames carry global rigid-body drift, and raw
    coordinate variance over 8 ns spans 5-7 A of unpredictable translation and
    rotation.  Asking the model to reproduce that drift is impossible -- it
    cannot know the global frame of the future -- so an unaligned target injects
    huge, unlearnable spread and destroys the mean prediction (observed: v4
    val/skill fell to -1.06 with root_mean_w2 6.3 vs persistence 1.24).
    Aligning first makes the target the *internal* fluctuation (1-2 A), which is
    the quantity the eval actually scores and the model can actually learn.
    """
    M, B, T, L, _ = ca_pred.shape
    mf = mask[:, None, :].expand(B, T, L).float()                  # [B,T,L]
    model_var = ca_pred.new_zeros(B, T, L)
    true_var = ca_pred.new_zeros(B, T, L)

    for b in range(B):
        mb = mask[b].to(ca_pred.dtype)                             # [L]
        if float(mb.sum()) < 3.0:
            continue
        # --- model: align the M samples at each step onto their own sample 0 ---
        pred_b = ca_pred[:, b]                                     # [M,T,L,3]
        pred_flat = pred_b.permute(1, 0, 2, 3).reshape(T * M, L, 3)
        ref_b = pred_b[0]                                          # [T,L,3]
        ref_flat = ref_b.unsqueeze(1).expand(T, M, L, 3).reshape(T * M, L, 3)
        Rm, tm = kabsch_rotation(pred_flat, ref_flat)              # tm: [T*M,1,3]
        am = (pred_flat @ Rm + tm).reshape(T, M, L, 3).permute(1, 0, 2, 3)
        model_var[b] = ((am - am.mean(0, keepdim=True)) ** 2).sum(-1).mean(0)

        # --- true: align every future frame onto frame 0, then cumulative var ---
        true_b = true_ca[b]                                        # [T,L,3]
        ref_true = true_b[0:1].expand(T, L, 3)
        Rt, tt = kabsch_rotation(true_b, ref_true)                 # tt: [T,1,3]
        at = true_b @ Rt + tt                                      # [T,L,3]
        tc = at * mb[None, :, None]
        cnt = mb[None, :].expand(T, L).cumsum(0).clamp_min(1.0)    # [T,L]
        csum = tc.cumsum(0)
        csum2 = (tc ** 2).sum(-1).cumsum(0)
        tmean = csum / cnt.unsqueeze(-1)
        true_var[b] = (csum2 / cnt - tmean.square().sum(-1)).clamp_min(0.0)

    return model_var, true_var, mf


def ensemble_rmsf_loss(ca_pred: Tensor, true_ca: Tensor, mask: Tensor,
                       eps: float = 1e-4) -> Tensor:
    """Per-residue RMSF of the M-sample ensemble vs the true temporal RMSF.

    Differentiable surrogate for the ``rmsf_mae`` evaluation metric.  Matches the
    *amplitude* of per-residue fluctuation, which is the thing persistence (zero
    variance) and a collapsed decoder both get wrong.
    ``eps`` floors the sqrt to keep gradients finite when the model variance is
    exactly zero (collapsed decoder at initialisation).
    """
    model_var, true_var, mf = _ensemble_stats(ca_pred, true_ca, mask)
    diff = ((model_var + eps).sqrt()
            - (true_var + eps).sqrt()).abs() * mf
    return diff.sum() / mf.sum().clamp_min(1.0)


def _kabsch_align_flat(x: Tensor, ref: Tensor, mask_1d: Tensor) -> Tensor:
    """Kabsch-align a batch of frames ``x`` [N, L, 3] onto ``ref`` [L, 3].

    Only valid residues (``mask_1d`` [L] bool) participate in the rotation
    estimate; the returned frames keep all L residues so downstream RMSF
    indexing stays consistent.  Differentiable through the SVD.

    This mirrors ``evaluation.ensemble_metrics.align_ensemble`` exactly so the
    training surrogate and the eval metric share the same alignment definition.
    """
    m = mask_1d.to(x.dtype).unsqueeze(-1)           # [L, 1]
    n_valid = m.sum().clamp_min(1.0)
    ref_c = (ref * m).sum(0, keepdim=True) / n_valid  # [1, 3]
    refc = (ref - ref_c) * m                         # [L, 3]
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        xi = x[i]
        xi_c = (xi * m).sum(0, keepdim=True) / n_valid
        p = (xi - xi_c) * m                          # [L, 3]
        H = p.transpose(0, 1) @ refc                 # [3, 3]
        U, _, Vh = torch.linalg.svd(H)
        sgn = torch.sign(torch.det(U @ Vh))
        D = torch.diag(torch.stack([
            torch.ones_like(sgn), torch.ones_like(sgn), sgn]))
        R = U @ D @ Vh                               # [3, 3]
        out[i] = p @ R + ref_c
    return out


def _rmsf_from_aligned(aligned: Tensor, mask_1d: Tensor, eps: float) -> Tensor:
    """Per-residue RMSF [L] of a Kabsch-aligned cloud [N, L, 3].

    Matches ``evaluation.ensemble_metrics.rmsf_profile``: mean over N, then
    sqrt of mean-squared displacement, zeroed on invalid residues.
    """
    mean = aligned.mean(0, keepdim=True)             # [1, L, 3]
    sq = (aligned - mean).square().sum(-1).mean(0)   # [L]
    rmsf = (sq.clamp_min(eps)).sqrt()
    return rmsf * mask_1d.to(rmsf.dtype)


def ensemble_rmsf_corr_loss(ca_pred: Tensor, true_ca: Tensor, mask: Tensor,
                            eps: float = 1e-4) -> Tensor:
    """Pearson-r loss between model and true per-residue RMSF profiles.

    Differentiable surrogate for the ``rmsf_corr`` (RMSF-r) evaluation metric.
    This version matches the eval pipeline exactly:

    1. Model ensemble  : flatten M×T decoded frames per protein → [M*T, L, 3],
       Kabsch-align all frames onto the first decoded frame (same as
       ``align_ensemble`` in the evaluator, which aligns onto frames[0]).
    2. True ensemble   : T future CA frames → [T, L, 3], Kabsch-align onto
       true frame 0.
    3. Compute per-residue RMSF on each aligned cloud (``rmsf_profile``
       equivalent).
    4. Return mean over batch of ``1 - Pearson(model_rmsf, true_rmsf)``
       over valid residues.

    The old implementation used ``_ensemble_stats`` cumulative variance, which
    measures M-sample spatial spread at each step separately — structurally
    different from the evaluator's single-cloud RMSF over all frames at once.
    This version is equivalent to what the evaluator computes, so the training
    gradient actually points in the direction that improves the scored metric.
    """
    # ca_pred: [M, B, T, L, 3];  true_ca: [B, T, L, 3]
    M, B, T, L, _ = ca_pred.shape
    losses = []

    for b in range(B):
        mb = mask[b].bool()
        if int(mb.sum()) < 3:
            continue

        # --- model RMSF: align M*T frames onto decoded frame (0, 0) ----------
        # Use all M stochastic rollouts and all T time steps as the "ensemble"
        # to maximise the effective N, exactly as the evaluator does when it
        # gets [N, L, 3] from n_samples independent rollouts rolled to n_steps.
        pred_b = ca_pred[:, b]                       # [M, T, L, 3]
        pred_flat = pred_b.reshape(M * T, L, 3)      # [M*T, L, 3]
        ref_model = pred_flat[0]                     # [L, 3]  first frame
        pred_aligned = _kabsch_align_flat(pred_flat, ref_model, mb)
        model_rmsf = _rmsf_from_aligned(pred_aligned, mb, eps)  # [L]

        # --- true RMSF: align T frames onto true frame 0 ---------------------
        true_b = true_ca[b]                          # [T, L, 3]
        ref_true = true_b[0]                         # [L, 3]
        true_aligned = _kabsch_align_flat(true_b, ref_true, mb)
        true_rmsf = _rmsf_from_aligned(true_aligned, mb, eps)   # [L]

        # --- Pearson r over valid residues -----------------------------------
        a = model_rmsf[mb]
        t = true_rmsf[mb]
        a = a - a.mean()
        t = t - t.mean()
        denom = (a.norm() * t.norm()).clamp_min(eps)
        r = (a * t).sum() / denom
        losses.append(1.0 - r)

    if not losses:
        return ca_pred.new_zeros(())
    return torch.stack(losses).mean()


def ensemble_variance_loss(ca_pred: Tensor, true_ca: Tensor, mask: Tensor,
                           eps: float = 1e-4) -> Tensor:
    """Log-domain L1 between model and true per-residue fluctuation variance.

    The log makes the term scale-aware: being 10x too small is penalised as hard
    as being 10x too large, which is exactly the failure mode (samples clustered
    in 0.3 A while the true motion spans several A).
    """
    model_var, true_var, mf = _ensemble_stats(ca_pred, true_ca, mask)
    lv_p = torch.log(model_var + eps)
    lv_t = torch.log(true_var + eps)
    diff = (lv_p - lv_t).abs() * mf
    return diff.sum() / mf.sum().clamp_min(1.0)


def ensemble_spread_floor(ca_pred: Tensor, true_ca: Tensor, mask: Tensor,
                          eps: float = 1e-4) -> Tensor:
    """One-sided hinge penalising under-dispersion (model spread < true spread).

    Complements ``ensemble_rmsf_loss`` with an asymmetric floor: the symmetric L1
    can be reduced by shrinking *toward* the truth when the model over-disperses,
    but a collapsed model needs a term that only ever pushes outward.  The true
    spread is detached -- this is a constraint on the model, not a target for the
    data.  ``eps`` keeps the sqrt gradient finite at zero.
    """
    model_var, true_var, mf = _ensemble_stats(ca_pred, true_ca, mask)
    hinge = F.relu((true_var.detach() + eps).sqrt()
                   - (model_var + eps).sqrt()) * mf
    return hinge.sum() / mf.sum().clamp_min(1.0)


# -----------------------------------------------------------------------------
# Composite
# -----------------------------------------------------------------------------

@dataclass
class FromNoneWeights:
    """Loss weights for the from-none Stage B objective.

    ``ca_weight`` dominates: it is the decodability term that shapes the encoder.
    ``es_weight`` keeps the rollout distributional; ``phys_weight`` ties the
    decoded backbone to physical features.
    """
    es_weight: float = 0.1
    ca_weight: float = 1.0
    phys_weight: float = 0.1
    anchor_weight: float = 0.0    # optional same-frame H->anchor-CA probe (0 = off)
    variance_weight: float = 0.0  # optional temporal-variance floor (0 = off)
    # Ensemble supervision on the decoded cloud (v4).  These are the terms that
    # make the rollout distribution -- not just its mean -- match the trajectory.
    ens_rmsf_weight: float = 0.0   # per-residue fluctuation amplitude (RMSF MAE)
    ens_rmsf_corr_weight: float = 0.0  # per-residue fluctuation shape (RMSF-r / Pearson)
    ens_var_weight: float = 0.0    # log-variance match (variance_w2)
    ens_spread_weight: float = 0.0 # one-sided anti-collapse hinge

    def validate(self) -> None:
        for k, v in self.__dict__.items():
            if not (v >= 0.0 and v < float("inf")):
                raise ValueError(f"FromNoneWeights.{k} must be finite and non-negative, got {v}")


def temporal_variance_floor(H_seq: Tensor, residue_mask: Tensor, floor: float) -> Tensor:
    """Penalty pushing per-(residue,channel) temporal std toward ``floor``.

    ``H_seq``: [B, T, L, dH].  Returns a scalar hinge penalty
    ``mean_relu(floor - std_t)``.  Prevents H from collapsing to a constant,
    which is the trivial way to make the ES term small.
    """
    if floor <= 0:
        return H_seq.new_zeros(())
    std = H_seq.std(dim=1)                          # [B, L, dH]
    mask = residue_mask.unsqueeze(-1).float()        # [B, L, 1]
    pen = F.relu(floor - std) * mask
    return pen.sum() / mask.sum().clamp_min(1.0) / H_seq.shape[-1]


@dataclass
class FromNoneLossComponents:
    total: Tensor
    es: Tensor
    ca: Tensor
    phys: Tensor
    anchor: Tensor
    variance: Tensor
    persist: Tensor
    ens_rmsf: Tensor | None = None
    ens_rmsf_corr: Tensor | None = None
    ens_var: Tensor | None = None
    ens_spread: Tensor | None = None

    def as_loggable_dict(self) -> dict[str, float]:
        d = {
            "loss/total": float(self.total.detach()),
            "loss/es": float(self.es.detach()),
            "loss/ca": float(self.ca.detach()),
            "loss/phys": float(self.phys.detach()),
            "loss/anchor": float(self.anchor.detach()),
            "loss/variance": float(self.variance.detach()),
        }
        for name in ("ens_rmsf", "ens_rmsf_corr", "ens_var", "ens_spread"):
            v = getattr(self, name)
            if v is not None:
                d[f"loss/{name}"] = float(v.detach())
        if torch.isfinite(self.persist):
            d["loss/persist_ref"] = float(self.persist.detach())
        return d
