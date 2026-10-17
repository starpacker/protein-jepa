"""Stage B1a training loop: representation calibration with an oracle decoder.

Implements STAGE_AB_REDESIGN_PLAN.md §2.0:

  - Encoder f_theta produces H_t from backbone coords (loaded from Stage A V2,
    frozen by default; optionally top-N blocks unfrozen with a tiny LR).
  - Oracle decoder D_chi(Q:=H_t, anchor, c) decodes the invariant x equivariant
    decomposition

        X_hat = anchor_frame  x  d_local(Q:=H_t, c)

    where anchor = the last observed CONTEXT frame's N/CA/C rigid-body basis
    (absolute placement only) and d_local is the internal displacement H owns.
    The MAIN loss supervises d_local in the anchor-local frame (SE(3) quotient);
    Kabsch-aligned absolute CA is logged as a diagnostic only.
  - Physical future probe checks observable info-content of H_t.
  - Anchor distillation keeps H close to the frozen Stage A reference so the
    representation does not drift when the encoder is partially unfrozen.
  - Diagnostics (plan §2.0 item 4): every log interval the same anchor-local
    displacement is also decoded from ZERO H and from the wrong-time (context)
    H.  H is "used" only if real-H clearly beats zero-H.

L_cal = L_oracle + lambda_pred * L_phys_future + lambda_anchor * L_anchor

Usage:
    python -m protein_jepa_frame.training.train_stage_b1a \
        --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
        --stage-a-ckpt artifacts/stage_a_v2_run/checkpoint_latest.pt \
        --output artifacts/stage_b1a_run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_b1a import (
    StageBOracleConfig,
    StageBOracleDecoder,
    StageBOracleLossWeights,
    compute_stage_b1a_loss,
)
from protein_jepa_frame.models.stage_b1a.model import freeze_encoder_partial
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.utils.backbone_geometry import backbone_torsion_targets
from protein_jepa_frame.training.train_utils import (
    _device,
    _lr_scale,
    _move_batch,
    _set_reproducibility,
)


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage B1a (oracle calibration)")
    p.add_argument("--manifest", type=str, default="artifacts/mdcath_manifest_200dom/train.jsonl")
    p.add_argument("--stage-a-ckpt", type=str, required=True,
                   help="Stage A V2 checkpoint providing the pretrained encoder.")
    p.add_argument("--output", type=str, default="artifacts/stage_b1a_run")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--unfreeze-top-n-blocks", type=int, default=0)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--num-distance-pairs", type=int, default=20)
    p.add_argument("--fape-weight", type=float, default=1.0)
    # The oracle decoder's anchor is ALWAYS the last observed context frame
    # (its N/CA/C rigid-body basis).  Per STAGE_AB_REDESIGN_PLAN.md §2.0.0 the
    # anchor supplies absolute placement only; the internal displacement is
    # predicted from H and supervised in the anchor-local frame.  There is no
    # target-anchor / noisy-anchor mode: those are degenerate (zero-init
    # residuals reproduce the target exactly at step 0, independent of H).
    p.add_argument("--local-position-budget", type=float, default=12.0,
                   help="max |anchor-local displacement| update in Å (the decoder's "
                        "only positional path; see §2.0.0).")
    p.add_argument("--torsion-weight", type=float, default=0.5)
    p.add_argument("--distance-weight", type=float, default=0.2)
    p.add_argument("--relative-disp-weight", type=float, default=5.0,
                   help="weight on the MAIN anchor-local internal displacement loss")
    p.add_argument("--phys-future-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=0.1)
    p.add_argument("--diag-zero-h", action="store_true", default=True,
                   help="Each log interval, additionally decode with H=0 and with a "
                        "wrong-time H, and log the same anchor-local displacement "
                        "metric.  H is 'used' only if real-H beats zero-H (plan §2.0).")
    p.add_argument("--no-diag-zero-h", dest="diag_zero_h", action="store_false")
    p.add_argument("--use-predicted-h", action="store_true", default=True,
                   help="Change C: decode the PREDICTED future H from the frozen "
                        "Stage A predictor (predictor(context_H) run forward), not "
                        "the encoding of the true future frame.  This is what makes "
                        "B1a calibrate the real H→H_future→structure pipeline.")
    p.add_argument("--no-use-predicted-h", dest="use_predicted_h",
                   action="store_false",
                   help="Legacy oracle-reconstruction mode: decode encode(target).")
    p.add_argument("--ca-mse-weight", type=float, default=1.0,
                   help="Weight on the Kabsch-aligned absolute CA MSE.  With the "
                        "anchor-free decoder this is the natural positional target "
                        "(H predicts the whole internal cloud; anchor is a post-hoc "
                        "rigid placement).")
    p.add_argument("--anchor-free", action="store_true", default=True,
                   help="Decoder predicts the whole internal CA cloud from H and "
                        "applies the anchor only as a post-hoc rigid placement.")
    p.add_argument("--no-anchor-free", dest="anchor_free", action="store_false",
                   help="Legacy anchored decoder (anchor base + bounded d_local).")
    p.add_argument("--coord-scale", type=float, default=40.0,
                   help="Bound (Å) on the anchor-free decoder's predicted internal "
                        "coordinates.")
    p.add_argument("--fit-normalizer-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Target derivation helpers
# -----------------------------------------------------------------------------

def _extract_backbone_frames(backbone_coords: torch.Tensor):
    """[B, L, 4, 3] (N,CA,C,O) -> (R [B,L,3,3], t [B,L,3])."""
    N = backbone_coords[..., 0, :]
    CA = backbone_coords[..., 1, :]
    C = backbone_coords[..., 2, :]
    return RigidBodyFrame.from_backbone(N, CA, C)


def _kabsch_target_delta_local(
    target_ca: torch.Tensor,
    anchor_ca: torch.Tensor,
    R_anchor: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Internal target displacement in each anchor residue-local frame.

    The target CA cloud is Kabsch-aligned onto the anchor CA cloud before the
    displacement is formed.  Consequently a pure global translation/rotation
    gives zero target displacement; only internal conformational change is
    supervised.  The final rotation converts the global displacement into the
    same local basis used by ``StructureDecoder``.
    """
    m = residue_mask.to(target_ca.dtype).unsqueeze(-1)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_center = (target_ca * m).sum(dim=1, keepdim=True) / denom
    anchor_center = (anchor_ca * m).sum(dim=1, keepdim=True) / denom
    P = (target_ca - target_center) * m
    Q = (anchor_ca - anchor_center) * m
    cov = P.transpose(1, 2) @ Q
    U, _, Vh = torch.linalg.svd(cov.to(torch.float64))
    # Coordinates are stored as row vectors here: target_centered @ R_align
    # should equal anchor_centered.  Therefore the proper Kabsch rotation is
    # U @ Vh (not V @ U^T, which is the column-vector convention).
    d = torch.sign(torch.linalg.det(U @ Vh))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.eye(3, device=target_ca.device, dtype=torch.float64)
    D = D.unsqueeze(0).expand(target_ca.shape[0], -1, -1).clone()
    D[:, 2, 2] = d
    R_align = (U @ D @ Vh).to(target_ca.dtype)
    target_aligned = (target_ca - target_center) @ R_align + anchor_center
    delta_global = target_aligned - anchor_ca
    delta_local = (
        R_anchor.transpose(-1, -2) @ delta_global.unsqueeze(-1)
    ).squeeze(-1)
    return delta_local * m


def _dihedral(p1, p2, p3, p4):
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    b2_n = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b1_p = b1 - (b1 * b2_n).sum(dim=-1, keepdim=True) * b2_n
    b3_p = b3 - (b3 * b2_n).sum(dim=-1, keepdim=True) * b2_n
    n1 = torch.cross(b2_n, b1_p / (b1_p.norm(dim=-1, keepdim=True) + 1e-8), dim=-1)
    n2 = torch.cross(b2_n, b3_p / (b3_p.norm(dim=-1, keepdim=True) + 1e-8), dim=-1)
    return torch.atan2(
        (torch.cross(n1, n2, dim=-1) * b2_n).sum(dim=-1),
        (n1 * n2).sum(dim=-1),
    )


def _future_torsion_sincos(torsions: torch.Tensor) -> torch.Tensor:
    """Return sin/cos for the three valid backbone torsions."""
    ang = torsions[..., :3]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def _build_pair_idx(residue_mask: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Sample distinct valid (i,j) pairs per batch element."""
    B, L = residue_mask.shape
    device = residue_mask.device
    pairs = torch.zeros(B, n_pairs, 2, dtype=torch.long, device=device)
    for b in range(B):
        valid = torch.nonzero(residue_mask[b], as_tuple=False).flatten()
        if valid.numel() < 2:
            continue
        first = torch.randint(valid.numel(), (n_pairs,), device=device)
        second = torch.randint(valid.numel() - 1, (n_pairs,), device=device)
        second = second + (second >= first).long()
        pairs[b, :, 0] = valid[first]
        pairs[b, :, 1] = valid[second]
    return pairs


def _pair_distances(ca: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    """ca [B, L, 3], pair_idx [B, n_pairs, 2] -> [B, n_pairs]."""
    B = ca.shape[0]
    ar = torch.arange(B, device=ca.device)[:, None]
    return (ca[ar, pair_idx[:, :, 0]] - ca[ar, pair_idx[:, :, 1]]).norm(dim=-1)


# -----------------------------------------------------------------------------
# Batch / encoder utilities
# -----------------------------------------------------------------------------

def _truncate_history(batch: dict, K: int) -> dict:
    out = dict(batch)
    for key in ("context_coords", "context_residue_mask", "context_time_ps"):
        v = out.get(key)
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] > K:
            out[key] = v[:, -K:].contiguous()
    return out


def build_paired_batch(samples: list[dict], K: int) -> dict:
    return _truncate_history(collate_paired_windows(samples), K)


def _encode_frame(encoder, coords_t, residue_mask_t, aatype, residue_index,
                  chain_id) -> torch.Tensor:
    """Encode a single frame [B, L, 4, 3] -> H [B, L, dH]."""
    coords_5d = coords_t.unsqueeze(1)      # [B, 1, L, 4, 3]
    mask_3d = residue_mask_t.unsqueeze(1)  # [B, 1, L]
    H = encoder(
        coords_5d, mask_3d, aatype, residue_index, chain_id,
        token_mask=mask_3d,
    )                                       # [B, 1, L, dH]
    return H[:, 0]                          # [B, L, dH]


def _predict_future_h(stage_a, paired, device) -> torch.Tensor:
    """Predicted future H from the frozen Stage A predictor (change C).

    This is the key B1a fix: the oracle must decode the PREDICTED future H
    (predictor(context_H) run forward in time), NOT the encoding of the true
    future frame.  Otherwise B1a trains a reconstruction decoder that never
    sees the predictor's output and the whole H→H_future→structure pipeline is
    bypassed.

    The Stage A temporal objective is anchor-calibrated:
        H_future = compose( decompose(H_anchor) + pred_factors )
    so pred_factors = 0 reproduces the anchor (persistence), and the predictor's
    job is to emit the displacement toward the future.  We reproduce exactly
    that contract here using the SAME predictor + OPF the Stage A objective used.

    Returns H_future_pred: [B, L, dH] (detached; Stage A is frozen in B1a).
    """
    ctx_coords = paired["context_coords"]            # [B, K, L, 4, 3]
    ctx_rmask  = paired["context_residue_mask"]      # [B, K, L]
    aatype         = paired["context_aatype"]
    residue_index  = paired["context_residue_index"]
    chain_id       = paired["context_chain_id"]
    ctx_t      = paired.get("context_time_ps")
    fut_t      = paired.get("future_time_ps")
    temp_all   = paired.get("temperature_k")

    B, K, L = ctx_coords.shape[:3]
    valid = ctx_rmask[:, 0]

    with torch.no_grad():
        context_H = stage_a.context_encoder(
            coords=ctx_coords, residue_mask=ctx_rmask,
            aatype=aatype, residue_index=residue_index, chain_id=chain_id,
        )                                             # [B, K, L, dH]

        if ctx_t is not None and fut_t is not None:
            lag_ps = (fut_t[:, -1] - ctx_t[:, -1]).to(torch.float32)
            bad = ~torch.isfinite(lag_ps) | (lag_ps <= 0)
            hz = paired.get("horizon_time_ps")
            fallback = (hz.to(torch.float32) if hz is not None
                        else torch.full_like(lag_ps, 1000.0))
            lag_ps = torch.where(bad, fallback, lag_ps)
        else:
            hz = paired.get("horizon_time_ps")
            lag_ps = (hz.to(torch.float32) if hz is not None
                      else torch.full((B,), 1000.0, device=device))

        temp_k = (temp_all if temp_all is not None
                  else torch.full((B,), 348.0, device=device))

        pred_factors = stage_a.predictor(
            context_H, lag_ps, temp_k, valid_mask=valid, task_id=1,
        )                                             # [B, L, J, r]

        anchor_H = context_H[:, -1]                   # [B, L, dH]
        anchor_factors = stage_a.opf.decompose(anchor_H)   # [B, L, J, r]
        H_future_pred = stage_a.opf.compose(anchor_factors + pred_factors)
    return H_future_pred.detach()


# -----------------------------------------------------------------------------
# H normalizer fit (run once before training)
# -----------------------------------------------------------------------------

@torch.no_grad()
def _fit_normalizer(oracle, encoder, loader, device, n_batches):
    encoder.eval()
    samples = []
    for i, paired in enumerate(loader):
        if i >= n_batches:
            break
        paired = _move_batch(paired, device)
        tgt_coords = paired["future_coords"][:, -1]
        tgt_mask = paired["future_residue_mask"][:, -1]
        H = _encode_frame(
            encoder, tgt_coords, tgt_mask,
            paired["future_aatype"], paired["future_residue_index"],
            paired["future_chain_id"],
        )
        samples.append(H[tgt_mask].detach().cpu())
    if samples:
        oracle.h_normalizer.fit(torch.cat(samples, dim=0).to(device))
        print(f"H normalizer fitted on {sum(s.shape[0] for s in samples)} residues",
              flush=True)
    else:
        print("WARNING: no batches for H normalizer; using identity", flush=True)


# -----------------------------------------------------------------------------
# Single training step
# -----------------------------------------------------------------------------

def _decode_anchor_local_metrics(oracle, H, aatype, R_anchor, t_anchor,
                                 residue_mask, target_delta_local):
    """Anchor-local displacement error for a given H (diagnostic).

    Returns (smooth_l1, rmse) both in Å, over valid residues.  Used to compare
    real-H vs zero-H vs wrong-time-H: H is 'used' only if real-H is clearly
    better (plan §2.0 item 4).
    """
    with torch.no_grad():
        out = oracle.decode_structure(
            H=H, aatype=aatype, R_anchor=R_anchor, t_anchor=t_anchor,
            residue_mask=residue_mask,
        )
        pred = out.get("local_position_update_local")
        if pred is None or target_delta_local is None:
            return None, None
        m = residue_mask.to(pred.dtype).unsqueeze(-1)
        se = (pred - target_delta_local) ** 2
        rmse = ((se * m).sum() / (m.sum() * 3.0 + 1e-6)).sqrt()
        l1 = (F.smooth_l1_loss(pred, target_delta_local, reduction="none") * m).sum() / (
            m.sum() * 3.0 + 1e-6)
        return float(l1), float(rmse)


def _step(paired, encoder, anchor_encoder, oracle, weights, cfg, device,
          encoder_trainable, diag_zero_h: bool = True, stage_a=None,
          use_predicted_h: bool = True):
    future_coords = paired["future_coords"]          # [B, T_fut, L, 4, 3]
    future_mask   = paired["future_residue_mask"]    # [B, T_fut, L]
    aatype         = paired["future_aatype"]          # [B, L]
    residue_index  = paired["future_residue_index"]
    chain_id       = paired["future_chain_id"]

    tgt_coords = future_coords[:, -1]                # [B, L, 4, 3]
    tgt_mask   = future_mask[:, -1]                  # [B, L]
    ctx_coords = paired["context_coords"][:, -1]     # [B, L, 4, 3]

    # H at the future frame.  Change C: decode the PREDICTED future H from the
    # frozen Stage A predictor, not the encoding of the true future frame.
    # Falls back to the true-frame encoding only when no Stage A model / the
    # flag is off (legacy oracle-reconstruction mode).
    if use_predicted_h and stage_a is not None:
        H_future = _predict_future_h(stage_a, paired, device)   # [B, L, dH]
    else:
        H_future = _encode_frame(encoder, tgt_coords, tgt_mask, aatype,
                                 residue_index, chain_id)

    # Frozen anchor reference H (stop_grad) — only used for the anchor
    # distillation term toward the frozen Stage-A representation.
    with torch.no_grad():
        H_anchor = _encode_frame(anchor_encoder, tgt_coords, tgt_mask, aatype,
                                 residue_index, chain_id)

    R_target, t_target = _extract_backbone_frames(tgt_coords)

    # Anchor = the last observed CONTEXT frame's rigid-body basis.  Per
    # plan §2.0.0 it supplies absolute placement only; the internal
    # displacement is H's job and is supervised in the anchor-local frame.
    R_anchor, t_anchor = _extract_backbone_frames(ctx_coords)

    # Main target: internal displacement of the future CA cloud relative to the
    # anchor, in the anchor-local residue frames (SE(3) quotient — no global
    # rigid-body motion).
    target_delta_local = _kabsch_target_delta_local(
        target_ca=tgt_coords[:, :, 1, :],
        anchor_ca=ctx_coords[:, :, 1, :],
        R_anchor=R_anchor,
        residue_mask=tgt_mask,
    )
    target_torsions, torsion_valid_mask = backbone_torsion_targets(
        tgt_coords, tgt_mask, residue_index, chain_id,
        atom_mask=paired.get("future_atom_mask", None)[:, -1]
        if paired.get("future_atom_mask", None) is not None else None,
    )

    decoder_out = oracle.decode_structure(
        H=H_future, aatype=aatype,
        R_anchor=R_anchor, t_anchor=t_anchor,
        residue_mask=tgt_mask,
    )

    pair_idx            = _build_pair_idx(tgt_mask, cfg.num_distance_pairs)
    probe_out           = oracle.probe_future(H_future, pair_idx, tgt_mask)
    future_tor_tgt      = _future_torsion_sincos(target_torsions)
    ca                  = tgt_coords[:, :, 1, :]
    future_dist_tgt     = _pair_distances(ca, pair_idx)
    future_contact_tgt  = (future_dist_tgt < 8.0).float()

    comp = compute_stage_b1a_loss(
        decoder_out=decoder_out,
        target_backbone_coords=tgt_coords,
        target_R=R_target,
        target_t=t_target,
        target_torsions=target_torsions,
        residue_mask=tgt_mask,
        target_delta_local=target_delta_local,
        target_atom_mask=(paired.get("future_atom_mask", None)[:, -1]
                          if paired.get("future_atom_mask", None) is not None else None),
        probe_out=probe_out,
        future_torsion_target=future_tor_tgt,
        future_torsion_valid_mask=torsion_valid_mask,
        future_dist_target=future_dist_tgt,
        future_contact_target=future_contact_tgt,
        H_current=H_future,
        H_anchor=H_anchor,
        weights=weights,
        torsion_valid_mask=torsion_valid_mask,
    )

    # ── Diagnostic: is H actually used? (plan §2.0 item 4) ───────────────────
    # Same anchor-local target, decoded from real H / zero H / the *context*
    # (wrong-time) H.  H is informative only if real-H clearly beats zero-H.
    diag: dict[str, float] = {}
    if diag_zero_h:
        _, rmse_real = _decode_anchor_local_metrics(
            oracle, H_future, aatype, R_anchor, t_anchor, tgt_mask, target_delta_local)
        _, rmse_zero = _decode_anchor_local_metrics(
            oracle, torch.zeros_like(H_future), aatype, R_anchor, t_anchor,
            tgt_mask, target_delta_local)
        H_wrong = _encode_frame(encoder, ctx_coords, tgt_mask, aatype,
                                residue_index, chain_id)
        _, rmse_wrong = _decode_anchor_local_metrics(
            oracle, H_wrong, aatype, R_anchor, t_anchor, tgt_mask, target_delta_local)
        if rmse_real is not None:
            diag["b1a/diag_disp_rmse_real"] = rmse_real
            diag["b1a/diag_disp_rmse_zeroH"] = rmse_zero
            diag["b1a/diag_disp_rmse_wrongH"] = rmse_wrong

    return comp, diag


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------

def _save(out_dir, oracle, encoder, optimizer, step, cfg, weights, encoder_trainable):
    payload = {
        "oracle":            oracle.state_dict(),
        "optimizer":         optimizer.state_dict(),
        "step":              step,
        "stage_b1a_config":  cfg.__dict__,
        "loss_weights":      weights.__dict__,
    }
    if encoder_trainable:
        payload["encoder"] = encoder.state_dict()
    torch.save(payload, out_dir / "checkpoint_latest.pt")
    print(f"checkpoint saved at step {step}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _set_reproducibility(args.seed, False)
    device = _device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage A V2 and extract its encoder
    ckpt      = torch.load(args.stage_a_ckpt, map_location=device)
    a_cfg_d   = ckpt.get("stage_a_v2_config", {})
    a_cfg     = StageAV2Config(**a_cfg_d) if a_cfg_d else StageAV2Config()
    stage_a   = StageAV2(a_cfg).to(device)
    missing_keys, unexpected_keys = stage_a.load_state_dict(ckpt["model"], strict=False)
    if missing_keys:
        print(f"[B1a] Stage A ckpt missing keys (ok if struct_recon_head): {missing_keys}", flush=True)
    stage_a.eval()
    encoder = stage_a.context_encoder

    # Frozen deep-copy for anchor distillation
    anchor_encoder = copy.deepcopy(encoder).to(device).eval()
    for p in anchor_encoder.parameters():
        p.requires_grad_(False)

    # Oracle decoder.  Anchoring is always the last observed context frame;
    # the only positional path is the anchor-local displacement (§2.0.0).
    cfg = StageBOracleConfig(
        latent_dim=args.latent_dim,
        num_distance_pairs=args.num_distance_pairs,
        unfreeze_top_n_blocks=args.unfreeze_top_n_blocks,
        encoder_lr_scale=args.encoder_lr_scale,
        decoder_max_local_position_update=args.local_position_budget,
        decoder_anchor_free=args.anchor_free,
        decoder_coord_scale=args.coord_scale,
    )
    oracle = StageBOracleDecoder(cfg).to(device)

    weights = StageBOracleLossWeights(
        fape_weight=args.fape_weight,
        torsion_weight=args.torsion_weight,
        distance_weight=args.distance_weight,
        relative_disp_weight=args.relative_disp_weight,
        phys_future_weight=args.phys_future_weight,
        anchor_weight=args.anchor_weight,
        ca_mse_weight=args.ca_mse_weight,
    )
    weights.validate()

    encoder_trainable = freeze_encoder_partial(encoder, args.unfreeze_top_n_blocks)
    print(f"encoder trainable params: {sum(p.numel() for p in encoder_trainable)}",
          flush=True)

    # Data
    data_cfg = DataConfig(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training_mode="paired_window",
        num_frames=max(args.history_frames, 8),
        paired_all_horizons=False,
    )
    dataset    = build_dataset(data_cfg)
    collate_fn = lambda s: build_paired_batch(s, args.history_frames)  # noqa: E731
    loader     = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )

    _fit_normalizer(oracle, encoder, loader, device, args.fit_normalizer_batches)

    # Optimizer
    groups: list[dict] = [
        {"params": [p for p in oracle.parameters() if p.requires_grad],
         "lr": args.lr, "name": "oracle"},
    ]
    if encoder_trainable:
        groups.append({"params": encoder_trainable,
                       "lr": args.lr * args.encoder_lr_scale, "name": "encoder"})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    for pg in optimizer.param_groups:
        pg["base_lr"] = pg["lr"]

    start_step = 0
    if args.resume:
        rck = torch.load(args.resume, map_location=device)
        oracle.load_state_dict(rck["oracle"])
        if "encoder" in rck and encoder_trainable:
            # The checkpoint encoder was saved from a possibly different freeze
            # configuration; only load it when this run also trains the encoder,
            # otherwise keep the freshly loaded Stage A encoder.
            missing, unexpected = encoder.load_state_dict(
                rck["encoder"], strict=False)
            print(f"resume: loaded encoder (missing={len(missing)} "
                  f"unexpected={len(unexpected)})", flush=True)
        # Optimizer state: the param-group layout differs between frozen (1
        # group) and unfrozen (2 groups) runs.  Load it only when the groups
        # match; otherwise start the optimizer fresh (the decoder weights, which
        # are what we actually want to warm-start, are already loaded above).
        try:
            optimizer.load_state_dict(rck["optimizer"])
            start_step = int(rck.get("step", 0))
            print(f"resumed optimizer from {args.resume} at step {start_step}",
                  flush=True)
        except ValueError:
            print("resume: optimizer param groups differ (frozen->unfrozen); "
                  "starting optimizer fresh, decoder weights loaded", flush=True)
            start_step = 0

    with (out_dir / "config.json").open("w") as f:
        json.dump({"stage_b1a_config": cfg.__dict__,
                   "loss_weights": weights.__dict__,
                   "train_args": vars(args)}, f, indent=2)
    # Training loop
    oracle.train()
    encoder.train() if encoder_trainable else encoder.eval()

    step    = start_step
    t0      = time.time()
    running: dict[str, float] = {}
    seen    = 0

    while step < args.steps:
        for paired in loader:
            if step >= args.steps:
                break
            paired = _move_batch(paired, device)

            scale = _lr_scale(step, args.steps, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["base_lr"] * scale

            comp, diag = _step(paired, encoder, anchor_encoder, oracle, weights,
                               cfg, device, encoder_trainable,
                               diag_zero_h=args.diag_zero_h,
                               stage_a=stage_a,
                               use_predicted_h=args.use_predicted_h)

            optimizer.zero_grad(set_to_none=True)
            comp.total.backward()
            if args.grad_clip > 0:
                params = [p for g in optimizer.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            seen += 1
            for k, v in comp.as_loggable_dict().items():
                running[k] = running.get(k, 0.0) + v
            for k, v in diag.items():
                running[k] = running.get(k, 0.0) + v

            if step % args.log_every == 0:
                avg     = {k: v / seen for k, v in running.items()}
                elapsed = time.time() - t0
                print(
                    f"step {step:6d} | lr {optimizer.param_groups[0]['lr']:.2e} "
                    f"| {elapsed:.1f}s | "
                    + " ".join(f"{k.replace('b1a/','')}={v:.4f}"
                               for k, v in avg.items()),
                    flush=True,
                )
                running = {}
                seen    = 0
                t0      = time.time()

            if step > 0 and step % args.save_every == 0:
                _save(out_dir, oracle, encoder, optimizer, step, cfg, weights,
                      encoder_trainable)

            if args.dry_run and step >= 3:
                print("dry-run complete", flush=True)
                return
            step += 1

    _save(out_dir, oracle, encoder, optimizer, step, cfg, weights, encoder_trainable)
    print(f"training complete at step {step}", flush=True)


if __name__ == "__main__":
    main()
