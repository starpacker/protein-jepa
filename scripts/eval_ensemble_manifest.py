"""T6 — Ensemble-level evaluation on an arbitrary manifest (NEW, 2026-10-18).

Loads a trained ensemble run (GaussianHHead ckpt) on top of a frozen
Stage-A encoder + B1a anchored decoder, and reports the ensemble diagnostics
(model / persistence / oracle) on a chosen val manifest (mdCATH or ATLAS).

Reports per metric: model, persistence baseline, oracle upper bound, plus a
bootstrap 95% CI (over samples) for the model-vs-persistence difference.

Reads checkpoints read-only; never modifies the precise-coordinate pipeline.

Usage:
    python -m scripts.eval_ensemble_manifest \
        --ensemble-ckpt artifacts/stage_ensemble_relpos/checkpoint_best.pt \
        --stage-a-ckpt  artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
        --b1a-ckpt      artifacts/stage_b1a_relpos_anchored/checkpoint_best.pt \
        --manifest      artifacts/atlas_manifest/val.jsonl \
        --device cuda:5 --batches 40 --batch-size 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.training.ensemble_targets import (
    build_ensemble_targets, persistence_ensemble,
)
from protein_jepa_frame.training.ensemble_losses import (
    EnsembleLossWeights, compute_ensemble_loss,
)
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _anchors, _targets_from,
)
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch


def _bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=0):
    """95% CI of the mean of `deltas` (list of floats)."""
    if not deltas:
        return [float("nan"), float("nan")]
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    means = torch.empty(n_boot, dtype=torch.float64)
    N = t.numel()
    for b in range(n_boot):
        idx = torch.randint(0, N, (N,), generator=g)
        means[b] = t[idx].mean()
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return [lo, hi]


def _build_cond(paired, cond_dim, dev):
    """Mirror train_stage_ensemble._conditioning for eval (D1)."""
    if cond_dim <= 0:
        return None
    ht = paired.get("horizon_time_ps")
    B = paired["context_coords"].shape[0]
    t_norm = (torch.log(ht.to(dev).clamp_min(1.0)) / torch.log(torch.tensor(1e4, device=dev))
              if ht is not None else torch.zeros(B, device=dev))
    temp = paired.get("temperature_k")
    T_norm = ((temp.to(dev).float() - 300.0) / 150.0
              if temp is not None else torch.zeros(B, device=dev))
    parts = [t_norm, T_norm]
    cond = torch.stack(parts[:cond_dim], dim=-1) if cond_dim <= 2 else torch.stack(
        parts + [torch.zeros_like(t_norm)] * (cond_dim - 2), dim=-1)
    return cond


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-ckpt", required=True)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batches", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--dist-bins", type=int, default=40)
    p.add_argument("--dist-lo", type=float, default=0.0)
    p.add_argument("--dist-hi", type=float, default=40.0)
    p.add_argument("--max-pairs", type=int, default=256)
    p.add_argument("--min-seq-sep", type=int, default=3)
    p.add_argument("--rmsf-weight", type=float, default=1.0)
    p.add_argument("--contact-weight", type=float, default=10.0)
    p.add_argument("--dist-weight", type=float, default=0.3)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="default: use the value stored in the ensemble checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = _device(args.device)
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    ck_args = ck.get("args", {})
    cond_dim = int(ck_args.get("cond_dims", 0))
    head = GaussianHHead(dH, hidden=int(ck_args.get("head_hidden", 128)),
                         cond_dim=cond_dim)
    head.load_state_dict(ck["head"])
    head.to(dev).eval()
    producer = EnsembleProducer(encoder, decoder, head).to(dev).eval()

    # pos guide: use the checkpoint's trained weight if the CLI left the default
    pos_weight = args.pos_weight if args.pos_weight is not None else \
        float(ck_args.get("pos_weight", 0.0))
    weights = EnsembleLossWeights(
        rmsf_weight=args.rmsf_weight, contact_weight=args.contact_weight,
        dist_weight=args.dist_weight, pos_weight=pos_weight)
    print(f"[eval] cond_dim={cond_dim} pos_weight={pos_weight}")

    class P: pass
    pr = P()
    pr.history_frames = args.history_frames
    pr.frame_stride = args.frame_stride
    pr.paired_horizons_frames = args.horizons
    cfg = DataConfig(manifest_path=args.manifest, batch_size=args.batch_size,
                     num_workers=args.num_workers, training_mode="paired_window",
                     num_frames=args.history_frames, paired_all_horizons=False,
                     max_residues=args.max_residues)
    apply_protocol(cfg, protocol_from_args(pr))
    ds = build_dataset(cfg)
    n_take = min(len(ds), args.batches * args.batch_size)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n_take].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=lambda s: _truncate_history(collate_paired_windows(s),
                                                           args.history_frames))

    keys = ("total", "rmsf", "contact", "dist", "pos")
    model_v = {k: [] for k in keys}
    pers_v = {k: [] for k in keys}
    orac_v = {k: [] for k in keys}
    protos = set()
    hz_all = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            paired = _move_batch(batch, dev)
            targets, fut, fmask = _targets_from(paired, args, dev)
            bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)

            # model
            ctx = paired["context_coords"].to(dev)
            ctx_mask = paired["context_residue_mask"].to(dev)[:, -1]
            aatype = paired["future_aatype"].to(dev)
            ri = paired["future_residue_index"].to(dev)
            ci = paired["future_chain_id"].to(dev)
            R_a, t_a = _anchors(paired, dev)
            cond = _build_cond(paired, cond_dim, dev)
            out = producer.sample_future_ca(ctx, ctx_mask, aatype, ri, ci,
                                            R_a, t_a, n_samples=args.n_samples,
                                            cond=cond)
            ca = out["ca_samples"]
            true_ca_pos = fut[:, -1, :, 1, :]
            comp = compute_ensemble_loss(
                ca, targets.rmsf, targets.contact,
                targets.dist_hist_from_ca(ca), targets.dist_hist_true(),
                targets.mask, bin_w, weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=out["ca_mean_decoded"])
            d = comp.as_loggable_dict()

            # persistence ensemble
            pca = persistence_ensemble(ctx, args.n_samples, paired["context_residue_mask"].to(dev))
            pcomp = compute_ensemble_loss(
                pca, targets.rmsf, targets.contact,
                targets.dist_hist_from_ca(pca), targets.dist_hist_true(),
                targets.mask, bin_w, weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=ctx[:, -1, :, 1, :])
            pd = pcomp.as_loggable_dict()

            # oracle = true future window as the ensemble
            ocomp = compute_ensemble_loss(
                targets.ca_window, targets.rmsf, targets.contact,
                targets.dist_hist_true(), targets.dist_hist_true(),
                targets.mask, bin_w, weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=true_ca_pos)
            od = ocomp.as_loggable_dict()

            B = ca.shape[0]
            for name, dd, sink in (("model", d, model_v), ("pers", pd, pers_v),
                                   ("orac", od, orac_v)):
                for k in keys:
                    sink[k].append(dd.get(f"ens/{k}", 0.0))
            for t in paired["trajectory_id"]:
                protos.add(t)
            if "horizon_frames" in paired:
                hz_all += paired["horizon_frames"].tolist()
            _ = (fut, fmask, B)

    def agg(v):
        return {k: float(torch.tensor(x).mean()) for k, x in v.items()}

    model_a, pers_a, orac_a = agg(model_v), agg(pers_v), agg(orac_v)
    # model-vs-persistence win / CI on each metric (persist error - model error; >0 => model better)
    win = {}
    for k in keys:
        delta = [p - m for p, m in zip(pers_v[k], model_v[k])]
        win[k] = {"mean_improvement": float(torch.tensor(delta).mean()),
                  "ci95": _bootstrap_ci(delta),
                  "model_better": bool(torch.tensor(delta).mean() > 0)}

    res = {
        "manifest": args.manifest,
        "n_batches": min(args.batches, i + 1),
        "n_trajectories": len(protos),
        "n_samples": args.n_samples,
        "cond_dim": cond_dim, "pos_weight": pos_weight,
        "weights": {"rmsf": args.rmsf_weight, "contact": args.contact_weight,
                    "dist": args.dist_weight},
        "model": model_a,
        "persistence": pers_a,
        "oracle": orac_a,
        "model_vs_persistence": win,
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
