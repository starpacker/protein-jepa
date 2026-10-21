"""Stage A V2 training loop (clean JEPA-Anything-aligned rewrite).

Trains the `StageAV2` model described in STAGE_ABC_REDESIGN_PLAN.md §1:

  - online encoder f_θ        (gradient)
  - EMA teacher f_θ̄           (never gradient; updated after every step)
  - fixed orthogonal OPF coordinates (explicit learnable ablations)
  - causal Transformer predictor over K history frames

Each step samples BOTH objectives from the same batches:
  - temporal: paired context/future windows  → L_future
  - spatial:  within-frame contiguous mask   → L_masked

EMA follows the model's configured warmup and linear increase schedule.

Usage:
    python -m protein_jepa_frame.training.train_stage_a_v2 \
        --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
        --output   artifacts/stage_a_v2_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args, validate_manifest_split
from protein_jepa_frame.models.stage_a_v2 import (
    StageAV2,
    StageAV2Config,
    StageAV2LossWeights,
    compute_stage_a_v2_loss,
)
from protein_jepa_frame.training.train_utils import (
    _device,
    _lr_scale,
    _move_batch,
    _set_reproducibility,
)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage A V2")
    p.add_argument("--manifest", type=str, default="artifacts/mdcath_manifest/train.jsonl",
                   help="Path to training manifest JSONL.  Default is the full mdCATH dataset "
                        "(5398 domains / 107947 trajectories).  Use "
                        "artifacts/mdcath_manifest_1000dom/train.jsonl for quick smoke tests.")
    p.add_argument("--output", type=str, default="artifacts/stage_a_v2_run")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--val-manifest", type=str)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--synthetic", action="store_true", help="Explicit synthetic implementation smoke only.")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--max-train-samples", type=int, default=0)

    # Schedule
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Stride between consecutive context frames in raw-frame units. "
                        "Kept at 1: the multiscale mode (paired_slow_stride) already "
                        "provides the physical reach back to ~28 ns, and raising "
                        "frame_stride would grow the raw clip span past the minimum "
                        "paired horizon, breaking the nonoverlap contract.  The "
                        "conformational-regime fix lives in --temporal-lags instead.")
    p.add_argument("--paired-history-mode", type=str, default="uniform",
                   choices=["uniform", "multiscale"],
                   help="'uniform': all context frames at frame_stride. "
                        "'multiscale': recent frames at frame_stride, older frames at paired_slow_stride.")
    p.add_argument("--paired-recent-frames", type=int, default=0,
                   help="Number of recent frames in multiscale mode (at frame_stride).")
    p.add_argument("--paired-slow-stride", type=int, default=0,
                   help="Stride for older frames in multiscale mode.")
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Micro-batches accumulated per optimizer step; effective batch = "
             "batch-size × grad-accum.  Needed because the multi-lag objective "
             "stacks K context frames, so the pairwise attention activation is "
             "K× larger than a single-frame forward.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-residues", type=int, default=256,
                   help="Crop each domain to at most this many residues before batching. "
                        "Prevents OOM on the few large mdCATH domains (>256 residues) that "
                        "would otherwise pad the entire batch. 256 covers >90%% of mdCATH.")

    # Model
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--num-factors", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--opf-mode", choices=["fixed_orthogonal", "soft_gram", "qr_retraction"], default="fixed_orthogonal")
    p.add_argument("--lag-balanced", action="store_true")
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--predictor-depth", type=int, default=4)
    p.add_argument("--predictor-spatial-depth", type=int, default=2,
                   help="Depth of the spatial residue-attention stage in the predictor. "
                        "Fix 1: adds inter-residue interaction to capture collective motion.")
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--spatial-mask-ratio", type=float, default=0.40)
    p.add_argument(
        "--temporal-lags",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Future-clip offsets supervised by the temporal objective, counted "
             "BACKWARD from the end of the future window (0 = last future frame "
             "= anchor + full horizon; 1 = one offset earlier, etc.).  The "
             "future window mirrors the context offsets shifted by the horizon, "
             "so lag 0 is the cleanest out-of-window target (exactly one horizon "
             "ahead of the anchor).  Each must be in [0, future_frames-1].",
    )
    p.add_argument(
        "--recent-frac",
        type=float,
        default=0.5,
        help="Fraction of each batch restricted to the two shortest lags, so "
             "long (intrinsically unpredictable) horizons cannot dominate.",
    )
    p.add_argument(
        "--paired-horizons-frames",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32, 48, 64, 96],
        help="Legacy list of horizon lengths in frames — retained for the "
             "paired dataset builder (which sizes its window by the max).  The "
             "temporal objective itself now draws its lags from --temporal-lags.",
    )

    # Loss weights
    p.add_argument("--orth-weight", type=float, default=0.01)
    p.add_argument("--factor-activity-weight", type=float, default=0.01)
    p.add_argument("--encoder-variance-weight", type=float, default=0.01)
    p.add_argument("--encoder-min-std", type=float, default=0.1,
                   help="Per-channel std floor for cross-batch H variance floor V(H).")
    p.add_argument("--temporal-variance-weight", type=float, default=0.05)
    p.add_argument("--temporal-min-std", type=float, default=0.10,
                   help="Per-step H variance floor γ (Fix 5: lowered from 0.40 so "
                        "the floor is a gentle backstop, not a dominant gradient "
                        "that pushes the encoder to inject noise).")
    p.add_argument("--temporal-contrast-weight", type=float, default=1.0,
                   help="Fix 4: supervised temporal contrastive loss ON by default. "
                        "Regresses norm(H_i-H_j)/sqrt(L*dH) onto tanh(CA_RMSD/scale), "
                        "the only loss that ties H-distance to structural distance. "
                        "Set 0 to disable.")
    p.add_argument("--temporal-contrast-rmsd-scale", type=float, default=2.0,
                   help="RMSD scale (Å) for temporal contrast target; tanh saturation point.")
    p.add_argument(
        "--struct-recon-weight",
        type=float,
        default=1.0,
        help="Weight on the CA-CA distance reconstruction auxiliary loss. "
             "Forces H to retain absolute internal geometry so the downstream "
             "oracle decoder has a structural signal to read.  0 disables it.",
    )
    p.add_argument("--struct-recon-pairs", type=int, default=256,
                   help="CA-CA pairs sampled per (batch, frame) per step.")
    p.add_argument("--struct-recon-dim", type=int, default=256,
                   help="Hidden width of the distance reconstruction head.")
    p.add_argument(
        "--displacement-weight",
        type=float,
        default=0.5,
        help="Weight on the structural-displacement readout loss (P0-A, "
             "2026-10-18).  A small head reads ONLY the predicted H change "
             "synth_delta_H and is supervised by the true future frame's "
             "Kabsch-aligned internal CA displacement, so the temporal JEPA "
             "objective is forced to carry real structural change rather than a "
             "self-referential latent delta.  This is the minimal added "
             "supervision (one head, discarded after Stage A).  0 disables it.",
    )
    p.add_argument(
        "--displacement-direction-weight",
        type=float,
        default=1.0,
        help="λ_dir on the per-residue direction (cosine) term inside the "
             "displacement readout loss.  Supervises 'which residue moves "
             "where' directly, since the pattern-MSE alone tolerates correct "
             "magnitude with wrong direction.",
    )
    p.add_argument(
        "--delta-target",
        action="store_true",
        default=True,
        help="Compatibility spelling: Stage A always predicts teacher delta_factors.",
    )
    p.add_argument("--identity-embed-scale", type=float, default=0.1)
    p.add_argument(
        "--relative-position-features",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y", "t"),
        default=True,
        help="Enable the within-frame canonical-frame position channel in the "
             "encoder (2026-10-18).  Gives H a reference frame so the anchor->"
             "future displacement is expressible; without it H is position-blind "
             "and the readout cannot generalise (R2 0.09 held-out).",
    )
    p.add_argument("--ema-start", type=float, default=0.998)
    p.add_argument("--ema-end", type=float, default=0.9999)
    p.add_argument("--ema-warmup-steps", type=int, default=2500)

    # Logging
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer parameter groups
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_groups(model: StageAV2, lr: float, encoder_lr_scale: float,
                           weight_decay: float) -> list[dict]:
    """Encoder gets a scaled LR; OPF + predictor get the full LR.

    The teacher encoder holds no trainable parameters (requires_grad=False),
    so it is never added to the optimizer.
    """
    encoder_params = [p for p in model.context_encoder.parameters() if p.requires_grad]
    head_params = (
        [p for p in model.predictor.parameters() if p.requires_grad]
        + [p for p in model.opf.parameters() if p.requires_grad]
        + [p for p in model.struct_recon_head.parameters() if p.requires_grad]
        + [p for p in model.displacement_head.parameters() if p.requires_grad]
    )
    if not encoder_params:
        raise RuntimeError("online encoder has no trainable parameters")
    if not head_params:
        raise RuntimeError("predictor/OPF have no trainable parameters")
    return [
        {"params": encoder_params, "lr": lr * encoder_lr_scale, "name": "encoder"},
        {"params": head_params, "lr": lr, "name": "predictor_opf"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Batch construction
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_history(batch: dict, K: int) -> dict:
    """Keep only the last K context frames so the predictor sees a fixed window.

    Keeps chronological order (oldest first); selected targets are validated
    separately against the last real context timestamp.
    """
    out = dict(batch)
    for key in list(out.keys()):
        if not key.startswith("context_"):
            continue
        value = out[key]
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] > K:
            if key in ("context_coords", "context_residue_mask", "context_time_ps"):
                out[key] = value[:, -K:].contiguous()
    # context_dt_ps / context_time_ps must stay frame-aligned with coords
    if "context_time_ps" in out and out["context_time_ps"].shape[1] > K:
        out["context_time_ps"] = out["context_time_ps"][:, -K:].contiguous()
    return out


def build_paired_batch(samples: list[dict], K: int) -> dict:
    return _truncate_history(collate_paired_windows(samples), K)


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.train_manifest_sha256 = (hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
                                  if not args.synthetic else None)
    args.split_audit = (validate_manifest_split(args.manifest, args.val_manifest)
                        if args.val_manifest and not args.synthetic else None)
    if args.latent_dim % args.num_factors != 0:
        raise SystemExit("--latent-dim must be divisible by --num-factors")

    _set_reproducibility(args.seed, False)
    torch.set_num_threads(args.torch_threads)
    device = _device(args.device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    cfg = StageAV2Config(
        latent_dim=args.latent_dim,
        num_factors=args.num_factors,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        predictor_spatial_depth=args.predictor_spatial_depth,
        num_heads=args.num_heads,
        predictor_heads=args.num_heads,
        dropout=args.dropout,
        opf_mode=args.opf_mode,
        opf_learnable=args.opf_mode != "fixed_orthogonal",
        spatial_mask_ratio=args.spatial_mask_ratio,
        identity_embed_scale=args.identity_embed_scale,
        relative_position_features=args.relative_position_features,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        ema_warmup_steps=args.ema_warmup_steps,
        struct_recon_dim=args.struct_recon_dim,
        struct_recon_pairs=args.struct_recon_pairs,
        delta_target=args.delta_target,
    )
    model = StageAV2(cfg).to(device)
    model.sync_teacher()          # start teacher == online

    weights = StageAV2LossWeights(
        orth_weight=args.orth_weight,
        factor_activity_weight=args.factor_activity_weight,
        encoder_variance_weight=args.encoder_variance_weight,
        encoder_min_std=args.encoder_min_std,
        temporal_variance_weight=args.temporal_variance_weight,
        temporal_min_std=args.temporal_min_std,
        temporal_contrast_weight=args.temporal_contrast_weight,
        temporal_contrast_rmsd_scale=args.temporal_contrast_rmsd_scale,
        lag_balanced=args.lag_balanced,
    )
    weights.validate()

    # ── Data ─────────────────────────────────────────────────────────────────
    horizons = tuple(sorted(set(int(h) for h in args.paired_horizons_frames)))
    # The paired dataset needs a context span and the largest horizon; the
    # temporal objective only reads `context_*`, but the dataset builder sizes
    # its windows by max(horizon) so we keep the horizon list for window sizing.
    num_frames = args.history_frames
    protocol = protocol_from_args(args)

    lags = tuple(sorted(set(int(l) for l in args.temporal_lags)))
    # lags are future-clip offsets counted back from the end; valid range is
    # [0, future_frames-1].  The future clip has the same frame count as the
    # context window (num_frames), so bound by that.
    if not lags or any(l < 0 or l >= num_frames for l in lags):
        raise SystemExit(
            f"--temporal-lags must be in [0, future_frames-1]={0, num_frames - 1}; got {lags}"
        )
    data_cfg = DataConfig(
        manifest_path=None if args.synthetic else args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        training_mode="paired_window",
        num_frames=num_frames,
        frame_stride=args.frame_stride,
        paired_history_mode=args.paired_history_mode,
        paired_recent_frames=args.paired_recent_frames,
        paired_slow_stride=args.paired_slow_stride,
        paired_horizons_frames=horizons,
        # All horizons are sampled from each context window every step.
        # collate_paired_windows flattens the grouped pairs so the effective
        # batch is batch_size × len(horizons) — set batch_size accordingly
        # (batch_size=4 with 8 horizons → 32 pairs per step).
        paired_all_horizons=True,
        # Hard guard: reject any horizon that would share frames with context.
        # With num_frames=K this enforces horizon >= K (gap >= 1 frame).
        paired_nonoverlap_only=True,
        # Cap sequence length to prevent OOM on the few large mdCATH domains.
        # Without this, collate pads to the longest sequence in the batch and
        # BK × n_pairs × 3dH activations in struct_recon can exceed 50 GB on
        # an 80 GB GPU.
        max_residues=args.max_residues,
    )
    apply_protocol(data_cfg, protocol)
    dataset = build_dataset(data_cfg)
    if args.max_train_samples:
        from torch.utils.data import Subset
        indices = torch.linspace(0, len(dataset) - 1, min(len(dataset), args.max_train_samples)).long().tolist()
        dataset = Subset(dataset, indices)
    collate_fn = lambda s: build_paired_batch(s, args.history_frames)  # noqa: E731
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    if not len(loader):
        raise ValueError("training loader is empty; lower batch size or increase samples")
    val_loader = None
    if args.val_manifest or args.synthetic:
        val_cfg = replace(data_cfg, manifest_path=args.val_manifest if not args.synthetic else None,
                          synthetic_size=8)
        from torch.utils.data import Subset
        val_dataset = build_dataset(val_cfg)
        n_val = min(len(val_dataset), args.val_batches * args.batch_size)
        val_indices = torch.linspace(0, len(val_dataset) - 1, n_val).long().tolist()
        val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=args.batch_size,
                                shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ── Optimizer / schedule ─────────────────────────────────────────────────
    groups = build_optimizer_groups(model, args.lr, args.encoder_lr_scale, args.weight_decay)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    for pg in optimizer.param_groups:
        pg["base_lr"] = pg["lr"]

    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("stage_a_v2_config") != cfg.__dict__:
            raise ValueError("resume model configuration differs from checkpoint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_validation_loss", float("inf")))
        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"])
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    # ── Save config ──────────────────────────────────────────────────────────
    with (out_dir / "config.json").open("w") as f:
        json.dump(
            {
                "stage_a_v2_config": cfg.__dict__,
                "loss_weights": weights.__dict__,
                "train_args": vars(args),
            },
            f,
            indent=2,
        )

    # ── Loop ─────────────────────────────────────────────────────────────────
    model.train()
    step = start_step
    t0 = time.time()
    running: dict[str, float] = {}
    seen = 0
    micro = 0
    momentum = cfg.ema_start
    optimizer.zero_grad(set_to_none=True)

    while step < args.steps:
        for paired in loader:
            if step >= args.steps:
                break

            paired = _move_batch(paired, device)
            clips = build_clip_batch_from_pair(paired, device)

            # LR schedule
            scale = _lr_scale(step, args.steps, args.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["base_lr"] * scale

            # NOTE: no autocast.  jepa_anything_core.factor_prediction_loss
            # requires predicted/target factors to share a dtype exactly, and the
            # OPF analysis basis is kept in fp32.  Half-precision activations
            # would violate that contract.  The model is small enough that fp32
            # is affordable.
            temporal_out = model.forward_temporal_multilag(
                paired, lags, recent_frac=args.recent_frac,
            )
            # Inject context_coords so the contrastive loss can compute CA RMSD.
            temporal_out["context_coords"] = paired["context_coords"]
            spatial_out = model.forward_spatial(clips)
            comp = compute_stage_a_v2_loss(
                temporal_out,
                spatial_out,
                model.opf.analysis_basis(),
                weights,
                dH=cfg.latent_dim,
            )

            # Structure-reconstruction auxiliary: force H to retain the
            # absolute internal geometry (CA-CA distances) that the delta-target
            # temporal objective would otherwise let the encoder discard.
            total = comp.total
            recon_val = 0.0
            if args.struct_recon_weight > 0.0:
                recon = model.struct_recon_loss(
                    context_H=temporal_out["context_H_full"],
                    coords=paired["context_coords"],
                    residue_mask=paired["context_residue_mask"],
                    n_pairs=args.struct_recon_pairs,
                )
                total = total + args.struct_recon_weight * recon
                recon_val = float(recon.detach())

            # P0-A structural-displacement readout: the head reads ONLY the
            # predicted H change synth_delta_H (no anchor/H_mean side-channel) so
            # the real internal CA displacement of the target frame must be
            # carried by the predicted H change itself.  One small head,
            # discarded after Stage A.
            disp_val = 0.0
            if args.displacement_weight > 0.0:
                disp = model.displacement_readout_loss(
                    pred_factors=temporal_out["pred_factors"],
                    target_delta_local=temporal_out["target_delta_local"],
                    residue_mask=temporal_out["valid_mask"],
                    synth_delta_H=temporal_out["synth_delta_H"],
                    direction_weight=args.displacement_direction_weight,
                )
                total = total + args.displacement_weight * disp
                disp_val = float(disp.detach())

            # Gradient accumulation: divide each micro-batch loss so the summed
            # gradient equals the mean over the effective batch.  The optimizer
            # step (and EMA update) fires only on the last micro-batch.
            if not torch.isfinite(total):
                raise FloatingPointError("nonfinite Stage A total loss")
            (total / args.grad_accum).backward()

            micro += 1
            seen += 1
            for k, v in comp.as_loggable_dict().items():
                running[k] = running.get(k, 0.0) + v
            running["loss/struct_recon"] = running.get("loss/struct_recon", 0.0) + recon_val
            running["loss/displacement"] = running.get("loss/displacement", 0.0) + disp_val
            running["loss/total"] += float(total.detach()) - float(comp.total.detach())

            if micro % args.grad_accum == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                model.after_optimizer_step()
                optimizer.zero_grad(set_to_none=True)

                momentum = model.update_ema()
                step += 1

                # Logging (one line per optimizer step)
                if step % args.log_every == 0:
                    avg = {k: v / seen for k, v in running.items()}
                    avg.update(model.opf_diagnostics())
                    elapsed = time.time() - t0
                    print(
                        f"step {step:6d} | lr {optimizer.param_groups[0]['lr']:.2e} "
                        f"| ema {momentum:.5f} | {elapsed:.1f}s | "
                        + " ".join(f"{k.replace('loss/','')}={v:.4f}" for k, v in avg.items()),
                        flush=True,
                    )
                    with (out_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": step, "split": "train", **avg}) + "\n")
                    running = {}
                    seen = 0
                    t0 = time.time()

                if step > 0 and step % args.save_every == 0:
                    _save(out_dir, model, optimizer, step, cfg, weights, args, best_validation_loss=best_val)
                if val_loader is not None and step % args.val_every == 0:
                    val = _validate(model, val_loader, device, lags, weights, args)
                    print(f"validation step {step}: total={val:.6f}", flush=True)
                    with (out_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": step, "split": "validation", "loss/total": val}) + "\n")
                    if val < best_val:
                        best_val = val
                        _save(out_dir, model, optimizer, step, cfg, weights, args,
                              filename="checkpoint_best.pt", validation_loss=val, best_validation_loss=best_val)
                    _save(out_dir, model, optimizer, step, cfg, weights, args, best_validation_loss=best_val)

                if args.dry_run and step >= 3:
                    _save(out_dir, model, optimizer, step, cfg, weights, args)
                    print("dry-run complete", flush=True)
                    return

    if val_loader is not None:
        val = _validate(model, val_loader, device, lags, weights, args)
        if val < best_val:
            best_val = val
            _save(out_dir, model, optimizer, step, cfg, weights, args,
                  filename="checkpoint_best.pt", validation_loss=val, best_validation_loss=best_val)
    _save(out_dir, model, optimizer, step, cfg, weights, args, best_validation_loss=best_val)
    print(f"training complete at step {step}", flush=True)


def build_clip_batch_from_pair(paired: dict, device: torch.device) -> dict:
    """Recover a single-frame clip batch for the spatial objective.

    The paired batch already holds context frames; the spatial objective reuses
    the *last* context frame of each sample (the frame most consistent with the
    temporal target).
    """
    clip = {
        "coords": paired["context_coords"],
        "residue_mask": paired["context_residue_mask"],
        "aatype": paired["context_aatype"],
        "residue_index": paired["context_residue_index"],
        "chain_id": paired["context_chain_id"],
        "temperature_k": paired["temperature_k"],
    }
    # forward_spatial only reads the last frame internally, so passing the whole
    # context window is valid and keeps the two forward paths consistent.
    return clip


@torch.no_grad()
def _validate(model, loader, device, lags, weights, args):
    # Validation sampling does not perturb the training RNG stream.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.seed + 10000)
        model.eval()
        values = []
        for i, paired in enumerate(loader):
            if i >= args.val_batches:
                break
            paired = _move_batch(paired, device)
            temporal = model.forward_temporal_multilag(paired, lags, recent_frac=args.recent_frac)
            temporal["context_coords"] = paired["context_coords"]
            spatial = model.forward_spatial(build_clip_batch_from_pair(paired, device))
            comp = compute_stage_a_v2_loss(temporal, spatial, model.opf.analysis_basis(), weights,
                                          dH=model.config.latent_dim)
            total = comp.total
            if args.struct_recon_weight:
                total = total + args.struct_recon_weight * model.struct_recon_loss(
                    temporal["context_H_full"], paired["context_coords"], paired["context_residue_mask"])
            if getattr(args, "displacement_weight", 0.0):
                total = total + args.displacement_weight * model.displacement_readout_loss(
                    pred_factors=temporal["pred_factors"],
                    target_delta_local=temporal["target_delta_local"],
                    residue_mask=temporal["valid_mask"],
                    synth_delta_H=temporal["synth_delta_H"],
                    direction_weight=getattr(args, "displacement_direction_weight", 1.0),
                )
            values.append(float(total))
        model.train()
    if not values:
        raise ValueError("validation loader is empty")
    return sum(values) / len(values)


def _save(out_dir: Path, model: StageAV2, optimizer, step: int,
          cfg: StageAV2Config, weights: StageAV2LossWeights, args=None,
          filename="checkpoint_latest.pt", validation_loss=None, best_validation_loss=float("inf")) -> None:
    from protein_jepa_frame.utils.contracts import build_prediction_contract
    protocol = protocol_from_args(args).to_dict() if args else None
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "stage_a_v2_config": cfg.__dict__,
        "loss_weights": weights.__dict__,
        "train_args": vars(args) if args else {},
        "split_audit": getattr(args, "split_audit", None),
        "train_manifest_sha256": getattr(args, "train_manifest_sha256", None),
        "synthetic": bool(getattr(args, "synthetic", False)),
        "rng_state": torch.get_rng_state(),
        "output_contract": "delta_factors",
        "prediction_contract": build_prediction_contract(
            model.context_encoder.state_dict(), cfg.__dict__, predictor_output="delta_factors",
            window_protocol=protocol),
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "resume_reproducibility": "RNG restored; loader shuffle cursor is not restored; resume is not byte-exact",
    }
    torch.save(payload, out_dir / filename)
    print(f"checkpoint saved at step {step}", flush=True)


if __name__ == "__main__":
    main()
