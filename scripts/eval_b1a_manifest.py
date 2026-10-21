"""Standalone B1a evaluation on an arbitrary manifest (NEW, 2026-10-18).

Loads a verified Stage-A encoder + a verified B1a anchored decoder (both frozen),
and evaluates the SAME diagnostic the B1a trainer reports:

    diag_ca_rmsd_{real,zeroH,wrongH,persist}

on a val manifest of choice (e.g. ATLAS unseen proteins).  Nothing in the
training or precise-coordinate code is modified; this only reads checkpoints.

Usage:
    python -m scripts.eval_b1a_manifest \
        --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
        --b1a-ckpt    artifacts/stage_b1a_relpos_anchored/checkpoint_best.pt \
        --manifest    artifacts/atlas_manifest/val.jsonl \
        --device cuda:5 --batches 12 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.training.train_stage_b1a import (
    build_paired_batch, _truncate_history, _encode_frame, _extract_backbone_frames,
)
from protein_jepa_frame.models.stage_b1a.objectives import kabsch_ca_mse_loss
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack
from protein_jepa_frame.training.train_utils import _device, _move_batch


def _per_sample_ca_rmsd(pred_ca, target_ca, mask):
    """Return per-sample Kabsch CA RMSD (Å) as a [B] tensor (mirrors
    kabsch_ca_mse_loss but keeps the batch dimension for stratification)."""
    B, L, _ = pred_ca.shape
    m = mask.to(device=pred_ca.device, dtype=pred_ca.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    pc = (pred_ca * m.unsqueeze(-1)).sum(1, keepdim=True) / denom.unsqueeze(-1)
    tc = (target_ca * m.unsqueeze(-1)).sum(1, keepdim=True) / denom.unsqueeze(-1)
    p0 = (pred_ca - pc) * m.unsqueeze(-1)
    t0 = (target_ca - tc) * m.unsqueeze(-1)
    out = torch.empty(B, device=pred_ca.device, dtype=torch.float64)
    for b in range(B):
        Hm = p0[b].T.to(torch.float64) @ t0[b].to(torch.float64)
        U, S, Vh = torch.linalg.svd(Hm)
        d = torch.sign(torch.det(U @ Vh))
        D = torch.diag(torch.tensor([1.0, 1.0, float(d)], device=pred_ca.device, dtype=torch.float64))
        R = U @ D @ Vh
        diff = p0[b].to(torch.float64) @ R - t0[b].to(torch.float64)
        mse = (diff.square().sum(-1) * m[b].to(torch.float64)).sum() / denom[b, 0].to(torch.float64)
        out[b] = torch.sqrt(3.0 * mse.clamp_min(0.0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batches", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = _device(args.device)
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)

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
    # Sample spread across the WHOLE dataset (all proteins and all horizons),
    # not just the first entries, so the metric is not dominated by one protein.
    n_take = min(len(ds), args.batches * args.batch_size)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n_take].tolist()
    from torch.utils.data import Subset
    ds = Subset(ds, idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=lambda s: _truncate_history(collate_paired_windows(s),
                                                           args.history_frames))

    acc = {"real": 0.0, "zeroH": 0.0, "wrongH": 0.0, "persist": 0.0}
    by_h: dict[int, dict[str, list[float]]] = {}
    n = 0
    protos = set()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            paired = _move_batch(batch, dev)
            tgt = paired["future_coords"][:, -1].to(dev)
            tmask = paired["future_residue_mask"][:, -1].to(dev)
            ctx = paired["context_coords"][:, -1].to(dev)
            aatype = paired["future_aatype"].to(dev)
            ri = paired["future_residue_index"].to(dev)
            ci = paired["future_chain_id"].to(dev)
            R_a, t_a = _extract_backbone_frames(ctx)
            ca = tgt[:, :, 1, :]
            B = tgt.shape[0]
            hz = paired["horizon_frames"].to(dev) if "horizon_frames" in paired else None
            vals = {}
            for name, coords in (("real", tgt), ("wrongH", ctx)):
                H = _encode_frame(encoder, coords, tmask, aatype, ri, ci)
                dec = decoder.decode_structure(H, aatype, R_a, t_a, tmask)
                vals[name] = _per_sample_ca_rmsd(dec["backbone_coords"][:, :, 1], ca, tmask)
            H_zero = torch.zeros(B, tgt.shape[1], dH, device=dev)
            dec = decoder.decode_structure(H_zero, aatype, R_a, t_a, tmask)
            vals["zeroH"] = _per_sample_ca_rmsd(dec["backbone_coords"][:, :, 1], ca, tmask)
            vals["persist"] = _per_sample_ca_rmsd(ctx[:, :, 1], ca, tmask)
            for name, v in vals.items():
                acc[name] += float(v.mean())
                if hz is not None:
                    for h, vv in zip(hz.tolist(), v.tolist()):
                        by_h.setdefault(int(h), {}).setdefault(name, []).append(vv)
            n += 1
            for t in paired["trajectory_id"]:
                protos.add(t)
    res = {k: v / max(n, 1) for k, v in acc.items()}
    res["n_batches"] = n
    res["n_trajectories"] = len(protos)
    res["by_horizon"] = {
        int(h): {k: float(sum(v) / len(v)) for k, v in d.items()}
        for h, d in sorted(by_h.items())
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
