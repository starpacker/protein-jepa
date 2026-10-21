"""Stage B1a calibration loss.

L_cal = L_oracle + λ_pred · L_phys_future + λ_anchor · L_anchor

- L_oracle:      structure decoder loss (FAPE + torsion + distance) on current frame
- L_phys_future: physical future probe (torsion sin/cos + distance + contact)
- L_anchor:      soft L2 distillation from frozen Stage A H reference
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from protein_jepa_frame.models.stage_c.objectives import (
    fape_loss,
    torsion_loss,
    pairwise_distance_loss,
)


@dataclass
class StageBOracleLossWeights:
    # Oracle decoder sub-weights
    fape_weight: float = 1.0
    torsion_weight: float = 0.5
    distance_weight: float = 0.2
    # MAIN positional loss: anchor-local internal displacement (SE(3) quotient).
    # The decoder's only positional path is X_hat = R_anchor @ d_local + t_anchor,
    # so this supervises exactly what the state (H_t) is responsible for.
    relative_disp_weight: float = 0.0
    # Main anchor-free positional target: aligned CA MSE removes rigid pose.
    ca_mse_weight: float = 1.0

    # Physical future probe
    phys_future_weight: float = 1.0     # λ_pred
    phys_torsion_weight: float = 1.0
    phys_dist_weight: float = 1.0
    phys_contact_weight: float = 0.5
    phys_dist_scale: float = 10.0       # Å; normalise distance regression to O(1)

    # Anchor distillation
    anchor_weight: float = 0.1          # λ_anchor

    eps: float = 1e-6

    def validate(self) -> None:
        for name, val in self.__dict__.items():
            if not isinstance(val, float):
                continue
            if not (val >= 0.0 and val < float("inf")):
                raise ValueError(f"StageBOracleLossWeights.{name} must be finite non-negative, got {val}")


@dataclass
class StageBOracleLossComponents:
    total: Tensor
    oracle_fape: Tensor
    oracle_ca_mse: Tensor
    oracle_torsion: Tensor
    oracle_dist: Tensor
    oracle_relative_disp: Tensor
    phys_torsion: Tensor
    phys_dist: Tensor
    phys_contact: Tensor
    anchor: Tensor

    def as_loggable_dict(self) -> dict[str, float]:
        return {
            "b1a/total": self.total.item(),
            "b1a/oracle_fape": self.oracle_fape.item(),
            "b1a/oracle_ca_mse": self.oracle_ca_mse.item(),
            "b1a/oracle_torsion": self.oracle_torsion.item(),
            "b1a/oracle_dist": self.oracle_dist.item(),
            "b1a/oracle_relative_disp": self.oracle_relative_disp.item(),
            "b1a/phys_torsion": self.phys_torsion.item(),
            "b1a/phys_dist": self.phys_dist.item(),
            "b1a/phys_contact": self.phys_contact.item(),
            "b1a/anchor": self.anchor.item(),
        }


def _default_torsion_mask(torsion_logits: Tensor, target_torsions: Tensor) -> Tensor:
    """Mask slots whose target is an all-zero placeholder.

    B1a fills only phi/psi/omega (slots 0,1,2) with real dihedrals and leaves
    chi1-4 as zeros.  Supervising a constant-zero slot is pure noise; detect
    those slots by zero variance across the batch and drop them.
    """
    nt = torsion_logits.shape[-2]
    perm = target_torsions.reshape(-1, target_torsions.shape[-1])  # [B*L, nt]
    slot_std = perm.std(dim=0)                                     # [nt]
    return (slot_std > 1e-6).to(torsion_logits.device)


def _kabsch_rotation(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Batched differentiable Kabsch rotation aligning ``pred`` onto ``target``.

    Args:
        pred:   [B, N, 3]
        target: [B, N, 3]
    Returns:
        R: [B, 3, 3]  such that pred @ R is optimally aligned to target.
    """
    if mask is None:
        m = torch.ones(pred.shape[:-1], device=pred.device, dtype=pred.dtype)
    else:
        if mask.shape != pred.shape[:-1]:
            raise ValueError("Kabsch mask must match [B, N]")
        m = mask.to(device=pred.device, dtype=pred.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    p = pred - (pred * m.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
    t = target - (target * m.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
    p = p * m.unsqueeze(-1)
    t = t * m.unsqueeze(-1)
    # Cross-covariance H = P^T T  -> [B, 3, 3]
    H = torch.einsum("bni,bnj->bij", p, t)
    # The optimal alignment is a nuisance fit. Its detached optimum gives the
    # exact envelope gradient of squared aligned error, and avoids undefined
    # SVD-vector derivatives for repeated singular values at initialization.
    U, S, Vh = torch.linalg.svd(H.detach())
    # Coordinates are row vectors and the loss applies ``p @ R``.  Therefore
    # the Kabsch rotation is U @ D @ Vh (the V @ U^T form is the column-vector
    # convention and leaves a large rigid-body residual here).
    # Correct for reflection so det(R) == +1.
    d = torch.sign(torch.linalg.det(torch.bmm(U, Vh)))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).expand_as(H).clone()
    D[:, 2, 2] = d
    R = torch.bmm(U, torch.bmm(D, Vh))
    return R


def kabsch_ca_mse_loss(
    pred_ca: Tensor,
    target_ca: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Masked, Kabsch-aligned per-residue CA MSE (Å²).

    Unlike ``fape_loss`` this is not normalised by L², so each residue
    contributes a full-magnitude gradient.  This gives the oracle decoder a
    clean, direct H→CA signal instead of the diluted FAPE path.

    This is the main positional objective for anchor-free oracle calibration.
    Detaching the optimal nuisance rotation preserves its envelope gradient.
    """
    if pred_ca.shape != target_ca.shape or pred_ca.shape[-1] != 3:
        raise ValueError("pred_ca and target_ca must both be [B, L, 3]")
    B, L, _ = pred_ca.shape
    if mask is None:
        m = torch.ones(B, L, device=pred_ca.device, dtype=pred_ca.dtype)
    else:
        m = mask.to(device=pred_ca.device, dtype=pred_ca.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

    # Masked centroids.
    pc = (pred_ca * m.unsqueeze(-1)).sum(1, keepdim=True) / denom.unsqueeze(-1)
    tc = (target_ca * m.unsqueeze(-1)).sum(1, keepdim=True) / denom.unsqueeze(-1)
    p0 = (pred_ca - pc) * m.unsqueeze(-1)

    # Per-sample Kabsch (loop is fine: B is small).  Pass the validity mask
    # into the fit; re-centering a zero-padded p0/t0 over the full padded L
    # would otherwise bias the rotation toward padding.
    aligned = []
    for b in range(B):
        R = _kabsch_rotation(
            pred_ca[b:b + 1], target_ca[b:b + 1], m[b:b + 1]
        )[0]  # [3, 3]
        aligned.append(p0[b] @ R + tc[b])
    aligned = torch.stack(aligned, dim=0)  # [B, L, 3]

    diff = (aligned - target_ca) ** 2
    return (diff * m.unsqueeze(-1)).sum() / (denom.sum() * 3.0 + 1e-6)


def compute_stage_b1a_loss(
    # Oracle decoder outputs for the current frame
    decoder_out: dict,               # from StageBOracleDecoder.decode_structure()
    target_backbone_coords: Tensor,  # [B, L, 4, 3]  N, CA, C, O backbone atoms
    target_R: Tensor,                # [B, L, 3, 3]
    target_t: Tensor,                # [B, L, 3]
    target_torsions: Tensor,         # [B, L, 7]
    residue_mask: Tensor,            # [B, L]

    # Physical future probe
    probe_out: dict,                   # from StageBOracleDecoder.probe_future()
    future_torsion_target: Tensor,     # [B, L, 6]  phi/psi/omega sin/cos pairs
    future_dist_target: Tensor,        # [B, n_pairs]
    future_contact_target: Tensor,     # [B, n_pairs]  float (0/1)

    # Anchor distillation
    H_current: Tensor,            # [B, L, dH]  current H from online encoder
    H_anchor: Tensor,             # [B, L, dH]  frozen Stage A reference H (stop_grad)

    weights: StageBOracleLossWeights,
    target_delta_local: Tensor | None = None,  # [B, L, 3], Kabsch-aligned
    target_atom_mask: Tensor | None = None,  # [B, L, 4]
    future_torsion_valid_mask: Tensor | None = None,  # [B,L,7] shared angle mask
    torsion_mask: Tensor | None = None,  # [3] bool; slots with real targets
    torsion_valid_mask: Tensor | None = None,  # [B,L,7] atom/chain validity
) -> StageBOracleLossComponents:
    """Compute the composite Stage B1a calibration loss.

    Main positional term is the anchor-local internal displacement
    ``L_relative_disp`` (SE(3) quotient): the decoder's only positional path is
    ``X_hat = R_anchor @ d_local + t_anchor``, so this supervises exactly the
    invariant, H-owned part.  ``L_ca_mse`` (Kabsch-aligned absolute CA) is a
    DIAGNOSTIC logged for reference; its training weight defaults to 0.
    """
    weights.validate()
    dev = H_current.device

    def _zero() -> Tensor:
        return torch.zeros((), device=dev)

    # ── Oracle structure decoder losses ──────────────────────────────────────
    pred_R = decoder_out.get("R")
    pred_t = decoder_out.get("t")
    torsion_logits = decoder_out.get("torsion_logits")
    pred_backbone = decoder_out.get("backbone_coords")   # [B, L, 4, 3]

    if pred_backbone is not None:
        L_fape = fape_loss(
            pred_backbone, target_backbone_coords,
            pred_R=pred_R, pred_t=pred_t,
            target_R=target_R, target_t=target_t,
            mask=residue_mask,
            atom_mask=target_atom_mask,
        )
    else:
        L_fape = _zero()

    if torsion_logits is not None:
        # B1a only supplies real phi/psi/omega targets (torsion slots 0,1,2);
        # chi1-4 are placeholders, so supervise only the slots with real data.
        tm = torsion_mask if torsion_mask is not None else _default_torsion_mask(
            torsion_logits, target_torsions
        )
        L_torsion = torsion_loss(torsion_logits, target_torsions, mask=residue_mask,
                                 torsion_mask=tm, angle_mask=torsion_valid_mask)
    else:
        L_torsion = _zero()

    if pred_backbone is not None:
        # index 1 = Cα in N,CA,C,O ordering; target Cα from target_backbone_coords
        pred_ca_pos = pred_backbone[:, :, 1, :]          # [B, L, 3]
        target_ca_pos = target_backbone_coords[:, :, 1, :]  # [B, L, 3]
        L_dist = pairwise_distance_loss(pred_ca_pos, target_ca_pos, mask=residue_mask)
        # Kabsch-aligned absolute CA MSE.  This is the natural positional target
        # for the ANCHOR-FREE decoder (H predicts the whole internal cloud, the
        # anchor is only a post-hoc rigid placement, so a Kabsch-aligned absolute
        # CA loss supervises exactly H's output and is invariant to the anchor's
        # rigid pose).  When ca_mse_weight == 0 it stays a detached diagnostic,
        # preserving the old anchor-anchored behaviour.
        if weights.ca_mse_weight > 0.0:
            L_ca_mse = kabsch_ca_mse_loss(pred_ca_pos, target_ca_pos, mask=residue_mask)
        else:
            with torch.no_grad():
                L_ca_mse = kabsch_ca_mse_loss(pred_ca_pos, target_ca_pos, mask=residue_mask)
    else:
        L_dist = _zero()
        L_ca_mse = _zero()

    # ── MAIN positional loss: anchor-local internal displacement ─────────────
    # The decoder's local displacement is applied as R_anchor @ d_local.  The
    # target is Kabsch-aligned to the anchor first, so this asks H for internal
    # conformational change only — never the global rigid-body motion (which is
    # the anchor's job).
    pred_delta_local = decoder_out.get("local_position_update_local")
    if pred_delta_local is None:
        pred_delta_local = decoder_out.get("translation_update_local")
    if pred_delta_local is not None and target_delta_local is not None:
        m = residue_mask.to(pred_delta_local.dtype).unsqueeze(-1)
        L_relative_disp = F.smooth_l1_loss(
            pred_delta_local,
            target_delta_local.detach(),
            reduction="none",
        )
        L_relative_disp = (L_relative_disp * m).sum() / (
            m.sum() * 3.0 + weights.eps
        )
    else:
        L_relative_disp = _zero()

    L_oracle = (
        weights.fape_weight * L_fape
        + weights.torsion_weight * L_torsion
        + weights.distance_weight * L_dist
        + weights.relative_disp_weight * L_relative_disp
        + weights.ca_mse_weight * L_ca_mse
    )

    # ── Physical future probe losses ─────────────────────────────────────────
    torsion_pred = probe_out.get("torsion_pred")   # [B, L, 6]
    dist_pred = probe_out.get("dist_pred")         # [B, n_pairs]
    contact_logit = probe_out.get("contact_logit") # [B, n_pairs]

    if torsion_pred is not None:
        valid = residue_mask.unsqueeze(-1).expand_as(torsion_pred).float()
        if future_torsion_valid_mask is not None:
            backbone_valid = future_torsion_valid_mask[..., :3]
            valid = valid * torch.cat([backbone_valid, backbone_valid], dim=-1).float()
        L_phys_tor = (
            ((torsion_pred - future_torsion_target) ** 2) * valid
        ).sum() / (valid.sum() + weights.eps)
    else:
        L_phys_tor = _zero()

    if dist_pred is not None:
        # Normalise distances (Å) by a characteristic length so the MSE is O(1)
        # rather than dominated by large absolute CA–CA separations.
        scale = weights.phys_dist_scale + weights.eps
        L_phys_dist = ((dist_pred - future_dist_target) / scale).pow(2).mean()
    else:
        L_phys_dist = _zero()

    if contact_logit is not None:
        L_phys_contact = torch.nn.functional.binary_cross_entropy_with_logits(
            contact_logit, future_contact_target
        )
    else:
        L_phys_contact = _zero()

    L_phys = weights.phys_future_weight * (
        weights.phys_torsion_weight * L_phys_tor
        + weights.phys_dist_weight * L_phys_dist
        + weights.phys_contact_weight * L_phys_contact
    )

    # ── Anchor distillation ───────────────────────────────────────────────────
    # Soft L2 toward frozen Stage A H; normalise by valid residue count
    H_ref = H_anchor.detach()   # stop_grad on reference
    diff = H_current - H_ref
    valid_res = residue_mask.float().unsqueeze(-1)   # [B, L, 1]
    L_anchor = (diff.pow(2) * valid_res).sum() / (valid_res.sum() * diff.shape[-1] + weights.eps)

    # ── Total ─────────────────────────────────────────────────────────────────
    total = L_oracle + L_phys + weights.anchor_weight * L_anchor

    return StageBOracleLossComponents(
        total=total,
        oracle_fape=L_fape.detach(),
        oracle_ca_mse=L_ca_mse.detach(),
        oracle_torsion=L_torsion.detach(),
        oracle_dist=L_dist.detach(),
        oracle_relative_disp=L_relative_disp.detach(),
        phys_torsion=L_phys_tor.detach(),
        phys_dist=L_phys_dist.detach(),
        phys_contact=L_phys_contact.detach(),
        anchor=L_anchor.detach(),
    )
