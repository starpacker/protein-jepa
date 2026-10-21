"""T5 — Train the ensemble-level Gaussian-H producer (NEW, 2026-10-18).

Loads a VERIFIED Stage-A encoder and a VERIFIED B1a anchored decoder (both
frozen), adds a small trainable `GaussianHHead`, and trains it to match the
ensemble statistics (RMSF / contact-prob / internal-distance distribution) of the
TRUE future window against the model ensemble.

Nothing in the precise-coordinate pipeline is modified: this script constructs
its own optimizer over the new head only, and reads the existing checkpoints
read-only (never overwrites them).

Usage:
    python -m protein_jepa_frame.training.train_stage_ensemble \
        --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
        --b1a-ckpt     artifacts/stage_b1a_relpos_anchored/checkpoint_best.pt \
        --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
        --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
        --output artifacts/stage_ensemble_relpos
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.training.ensemble_targets import (
    build_ensemble_targets, persistence_ensemble,
)
from protein_jepa_frame.training.ensemble_losses import (
    EnsembleLossWeights, compute_ensemble_loss,
)
from protein_jepa_frame.training.train_utils import (
    _device, _move_batch, _set_reproducibility,
)
from protein_jepa_frame.training.train_stage_b1a import (
    _extract_backbone_frames, build_paired_batch, _truncate_history,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ensemble-level Gaussian-H producer")
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/train.jsonl")
    p.add_argument("--val-manifest", default=None)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--output", default="artifacts/stage_ensemble_run")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--history-frames", type=int, default=4)
    # ensemble window: horizons must produce >= 2 contiguous future frames
    p.add_argument("--paired-horizons-frames", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--head-hidden", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--dist-bins", type=int, default=40)
    p.add_argument("--dist-lo", type=float, default=0.0)
    p.add_argument("--dist-hi", type=float, default=40.0)
    p.add_argument("--max-pairs", type=int, default=256)
    p.add_argument("--min-seq-sep", type=int, default=3)
    p.add_argument("--rmsf-weight", type=float, default=1.0)
    p.add_argument("--contact-weight", type=float, default=1.0)
    p.add_argument("--dist-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=0.0,
                   help="D2 atom-position guide weight (0 disables)")
    p.add_argument("--cond-dims", type=int, default=0,
                   help="D1 conditioning dims injected into the head (0 disables; "
                        "1=log horizon time, 2=+temperature)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Model / checkpoint loading (read-only)
# ---------------------------------------------------------------------------

def load_frozen_stack(stage_a_ckpt: str, b1a_ckpt: str, device):
    """Return (encoder, decoder) frozen, from verified Stage-A + B1a ckpts."""
    a_ck = torch.load(stage_a_ckpt, map_location=device, weights_only=False)
    a_cfg_d = a_ck.get("stage_a_v2_config", {})
    a_cfg = StageAV2Config(**a_cfg_d) if a_cfg_d else StageAV2Config()
    stage_a = StageAV2(a_cfg).to(device)
    stage_a.load_state_dict(a_ck["model"], strict=True)
    stage_a.eval()
    encoder = stage_a.context_encoder
    for p in encoder.parameters():
        p.requires_grad_(False)

    b_ck = torch.load(b1a_ckpt, map_location=device, weights_only=False)
    b_cfg_d = b_ck.get("stage_b1a_config", {})
    b_cfg = StageBOracleConfig(**b_cfg_d) if b_cfg_d else StageBOracleConfig(
        latent_dim=a_cfg.latent_dim)
    decoder = StageBOracleDecoder(b_cfg).to(device)
    decoder.load_state_dict(b_ck["oracle"], strict=True)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    if b_cfg.latent_dim != a_cfg.latent_dim:
        raise ValueError("Stage-A latent_dim must equal B1a latent_dim")
    return encoder, decoder, a_cfg.latent_dim


# ---------------------------------------------------------------------------
# Step / validate
# ---------------------------------------------------------------------------

def _targets_from(paired, args, device):
    fut = paired["future_coords"].to(device)
    fmask = paired["future_residue_mask"].to(device)
    return build_ensemble_targets(
        fut, fmask, cutoff=args.cutoff, dist_lo=args.dist_lo,
        dist_hi=args.dist_hi, dist_bins=args.dist_bins,
        min_seq_sep=args.min_seq_sep, max_pairs=args.max_pairs,
    ), fut, fmask


def _anchors(paired, device):
    ctx = paired["context_coords"].to(device)[:, -1]         # [B,L,4,3]
    return _extract_backbone_frames(ctx)


def _conditioning(paired, args, device):
    """D1: build per-sample conditioning [B, cond_dim] from horizon time and T.

    cond = [log(horizon_time_ps)/log(1e4), (T-300)/150]  (each ~ O(1)).
    Returns None when --cond-dims == 0 (original behaviour).
    """
    cond_dim = int(getattr(args, "cond_dims", 0))
    if cond_dim <= 0:
        return None
    ht = paired.get("horizon_time_ps")
    B = paired["context_coords"].shape[0]
    if ht is None:
        t_norm = torch.zeros(B, device=device)
    else:
        t_norm = torch.log(ht.to(device).clamp_min(1.0)) / torch.log(torch.tensor(1e4, device=device))
    temp = paired.get("temperature_k")
    if temp is None:
        T_norm = torch.zeros(B, device=device)
    else:
        T_norm = (temp.to(device).float() - 300.0) / 150.0
    parts = [t_norm, T_norm]
    cond = torch.stack(parts[:cond_dim], dim=-1) if cond_dim <= 2 else torch.stack(
        parts + [torch.zeros_like(t_norm)] * (cond_dim - 2), dim=-1)
    return cond


def _run_batch(producer, paired, args, weights, device, generator=None):
    targets, fut, fmask = _targets_from(paired, args, device)
    aatype = paired["future_aatype"].to(device)
    ri = paired["future_residue_index"].to(device)
    ci = paired["future_chain_id"].to(device)
    ctx = paired["context_coords"].to(device)
    ctx_mask = paired["context_residue_mask"].to(device)[:, -1]
    R_a, t_a = _anchors(paired, device)
    cond = _conditioning(paired, args, device)

    out = producer.sample_future_ca(
        ctx, ctx_mask, aatype, ri, ci, R_a, t_a,
        n_samples=args.n_samples, generator=generator, cond=cond,
    )
    ca = out["ca_samples"]                                    # [B,n,L,3]
    pred_hist = targets.dist_hist_from_ca(ca)
    true_hist = targets.dist_hist_true()
    bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)
    # D2: atom-position guide uses the TRUE future frame (last) vs deterministic decode.
    true_ca_pos = None
    pred_ca_pos = None
    if weights.pos_weight > 0.0:
        true_ca_pos = fut[:, -1, :, 1, :]                     # [B,L,3]
        pred_ca_pos = out["ca_mean_decoded"]                  # [B,L,3]
    comp = compute_ensemble_loss(
        ca, targets.rmsf, targets.contact, pred_hist, true_hist,
        targets.mask, bin_w, weights, cutoff=args.cutoff,
        true_ca_pos=true_ca_pos, pred_ca_pos=pred_ca_pos,
    )
    return comp, targets, out


@torch.no_grad()
def _validate(producer, loader, args, weights, device, max_batches):
    producer.eval()
    from collections import defaultdict
    agg = defaultdict(float)
    pers = defaultdict(float)
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        paired = _move_batch(batch, device)
        comp, targets, _ = _run_batch(producer, paired, args, weights, device)
        for k, v in comp.as_loggable_dict().items():
            agg[k] += v
        # persistence ensemble
        ctx = paired["context_coords"].to(device)
        pmask = paired["context_residue_mask"].to(device)
        pca = persistence_ensemble(ctx, args.n_samples, pmask)
        phist = targets.dist_hist_from_ca(pca)
        thist = targets.dist_hist_true()
        bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)
        if weights.pos_weight > 0.0:
            # persistence position baseline (same convention as D2 guide)
            true_ca_pos = paired["future_coords"].to(device)[:, -1, :, 1, :]
            pers_ca_pos = ctx[:, -1, :, 1, :]
            pcomp = compute_ensemble_loss(
                pca, targets.rmsf, targets.contact, phist, thist,
                targets.mask, bin_w, weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=pers_ca_pos)
        else:
            pcomp = compute_ensemble_loss(
                pca, targets.rmsf, targets.contact, phist, thist,
                targets.mask, bin_w, weights, cutoff=args.cutoff)
        for k, v in pcomp.as_loggable_dict().items():
            pers[k] += v
        n += 1
    if n == 0:
        return {}
    out = {f"val/{k}": v / n for k, v in agg.items()}
    out.update({f"val_persist/{k}": v / n for k, v in pers.items()})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    _set_reproducibility(args.seed, False)
    torch.set_num_threads(args.torch_threads)
    device = _device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, device)
    head = GaussianHHead(dH, hidden=args.head_hidden, cond_dim=args.cond_dims)
    producer = EnsembleProducer(encoder, decoder, head).to(device)

    weights = EnsembleLossWeights(
        rmsf_weight=args.rmsf_weight, contact_weight=args.contact_weight,
        dist_weight=args.dist_weight, pos_weight=args.pos_weight)
    weights.validate()

    data_cfg = DataConfig(
        manifest_path=None if args.synthetic else args.manifest,
        batch_size=args.batch_size, num_workers=args.num_workers,
        training_mode="paired_window", num_frames=args.history_frames,
        paired_all_horizons=False, max_residues=args.max_residues,
    )
    protocol = protocol_from_args(args)
    apply_protocol(data_cfg, protocol)
    dataset = build_dataset(data_cfg)
    if args.max_train_samples:
        from torch.utils.data import Subset
        # NOTE: `torch.linspace(0, len-1, N)` picks N EVENLY-SPACED indices, not
        # the first N; the last one rounds to `len` and raises IndexError once
        # enough batches are drawn (this crashed the 1000dom run).  Subsample the
        # first `max_train_samples` windows uniformly instead, guaranteed in-range.
        n_sub = min(len(dataset), args.max_train_samples)
        stride = max(1, len(dataset) // n_sub)
        idx = list(range(0, len(dataset), stride))[:n_sub]
        dataset = Subset(dataset, idx)
    collate_fn = lambda s: _truncate_history(collate_paired_windows(s), args.history_frames)  # noqa: E731
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        drop_last=True, pin_memory=(device.type == "cuda"))
    if not len(loader):
        raise ValueError("training loader is empty")

    val_loader = None
    if args.val_manifest or args.synthetic:
        val_cfg = replace(data_cfg,
                          manifest_path=args.val_manifest if not args.synthetic else None,
                          synthetic_size=8)
        vd = build_dataset(val_cfg)
        n_val = min(len(vd), args.val_batches * args.batch_size)
        vidx = torch.linspace(0, len(vd) - 1, n_val).long().tolist()
        from torch.utils.data import Subset
        val_loader = DataLoader(Subset(vd, vidx), batch_size=args.batch_size,
                                num_workers=0, shuffle=False, collate_fn=collate_fn)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for pg in opt.param_groups:
        pg["base_lr"] = pg["lr"]
    start_step, best_val = 0, float("inf")
    if args.resume:
        rck = torch.load(args.resume, map_location=device, weights_only=False)
        head.load_state_dict(rck["head"])
        try:
            opt.load_state_dict(rck["optimizer"])
        except Exception:  # noqa: BLE001
            pass
        start_step = int(rck.get("step", 0))
        best_val = float(rck.get("best_val", best_val))

    def save(tag, step, val):
        torch.save({
            "head": head.state_dict(), "optimizer": opt.state_dict(),
            "step": step, "best_val": val, "args": vars(args),
            "stage_a_ckpt": args.stage_a_ckpt, "b1a_ckpt": args.b1a_ckpt,
            "ensemble_weights": vars(weights),
        }, out_dir / f"checkpoint_{tag}.pt")

    t0 = time.time()
    step = start_step
    while step < args.steps:
        for batch in loader:
            paired = _move_batch(batch, device)
            producer.train()
            comp, _, out = _run_batch(producer, paired, args, weights, device)
            opt.zero_grad(set_to_none=True)
            comp.total.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            lr_scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * lr_scale
            opt.step()
            step += 1
            if step % args.log_every == 0:
                d = comp.as_loggable_dict()
                print(f"step {step:6d} | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"{time.time()-t0:6.1f}s | ens/total={d['ens/total']:.4f} "
                      f"rmsf={d['ens/rmsf']:.4f} contact={d['ens/contact']:.4f} "
                      f"dist={d['ens/dist']:.4f}", flush=True)
            if val_loader is not None and step % args.val_every == 0:
                v = _validate(producer, val_loader, args, weights, device, args.val_batches)
                print(f"  [val {step}] {json.dumps({k: round(x,4) for k,x in v.items()})}",
                      flush=True)
                if v.get("val/ens/total", float("inf")) < best_val:
                    best_val = v["val/ens/total"]
                    save("best", step, best_val)
            if step % args.save_every == 0:
                save("latest", step, best_val)
            if step >= args.steps:
                break
    save("latest", step, best_val)
    print(f"done: {step} steps, best val ens/total = {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
