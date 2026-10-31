"""Order-aware trajectory ("walk") comparison: Model 1 vs Stage-B GLE (H+S).

The ensemble metrics score a *bag* of conformers and are blind to ordering.  A
GLE rollout is a *walk*, so here both models produce a full future trajectory and
we score it with ``evaluation.trajectory_metrics`` (per-step RMSD, path shape,
Rg, ACF, path energy score, walk-RMSF).  Everything is a proper *paired* test:
same frozen stack, same windows, same sample count; per-window
delta = (H+S - Model1) with a bootstrap 95% CI.

Two trajectory producers, deliberately matching each model's own mechanism:
  * Model 1 (``--mode model1``) : autoregressive static-head rollout
    (``scripts/eval_ours_trajectory.rollout_trajectory``) -- the SAME rollout used
    in the project report for Model 1.
  * H+S     (``--mode hs``)     : the GLE kernel rollout path, ALREADY produced
    internally (``H_traj``), decoded once per step with the frozen decoder.
    This is the dynamics the GLE actually models; terminal-only training only
    scored its last frame, so the walk metrics are exactly the "whole-trajectory"
    signal that was missing.
"""
from __future__ import annotations

import argparse, copy, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack, _anchors
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch
from scripts.eval_ours_trajectory import rollout_trajectory, _build_cond
from evaluation.trajectory_metrics import walk_summary

METRICS = ["per_step_rmsd_mean", "path_length", "end_to_end", "rg_mean",
           "traj_energy_score", "walk_rmsf_mae", "walk_rmsf_corr",
           "acf_lag1", "acf_corr"]
# direction: True == higher is better.
HIGHER_BETTER = {"walk_rmsf_corr", "acf_corr"}
# acf_lag1 sits between -0.5 and 1; higher == a smoother, less jittery walk.
ACF_LAG1_HIGHER = True


def build(args, dev):
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    enc_m1 = copy.deepcopy(encoder); dec_m1 = copy.deepcopy(decoder)
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    ca = ck.get("args", {})
    head = GaussianHHead(dH, hidden=int(ca.get("head_hidden", 128)),
                         cond_dim=int(ca.get("cond_dims", 0)))
    head.load_state_dict(ck["head"]); head.to(dev).eval()
    m1 = EnsembleProducer(enc_m1, dec_m1, head).to(dev).eval()
    cond_dim = int(ca.get("cond_dims", 0))

    hs = None
    if args.kernel_ckpt:
        kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
        k_cfg = kck.get("kernel_config") or {}; k_args = kck.get("args", {})
        k_cfg["step_ps"] = k_args.get("step_ps", 1000.0); k_cfg["rollout_grad_steps"] = None
        kern = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
        kern.load_state_dict(kck["kernel"]); kern.eval()
        te = bool(kck.get("train_encoder", False)) and kck.get("encoder") is not None
        td = bool(kck.get("train_decoder", False)) and kck.get("decoder") is not None
        hs = HSEnsembleProducer(encoder, decoder, kern, train_encoder=te,
                                train_decoder=td).to(dev).eval()
        if te:
            hs.encoder.load_state_dict(kck["encoder"], strict=False)
        if td:
            hs.decoder.load_state_dict(kck["decoder"], strict=False)
        hs._rollout_steps = int(k_args.get("rollout_steps", 8))
        hs._rollout_grad_steps = None
        hs._temperature_k = float(k_args.get("temperature_k", 320.0))
    return m1, hs, cond_dim


@torch.no_grad()
def hs_walk(producer, paired, args, dev):
    """Decode the ENTIRE GLE rollout path -> [B, M, T+1, L, 3] CA (incl. anchor)."""
    ctx = paired["context_coords"].to(dev)
    ctx_mask = paired["context_residue_mask"].to(dev)[:, -1]
    aatype = paired["future_aatype"].to(dev)
    ri = paired["future_residue_index"].to(dev)
    ci = paired["future_chain_id"].to(dev)
    R_a, t_a = _anchors(paired, dev)
    tk = paired.get("temperature_k")
    tkv = float(tk.to(dev).float().mean()) if tk is not None else producer._temperature_k
    out = producer.sample_future_ca(ctx, ctx_mask, aatype, ri, ci, R_a, t_a,
                                    n_samples=args.n_samples, temperature_k=tkv)
    H_traj = out["H_traj"]                                   # [B*M, n+1, L, dH]
    B = ctx.shape[0]; M = args.n_samples; L = ctx.shape[2]
    Tn = H_traj.shape[1]
    flat = H_traj.reshape(B * M * Tn, L, -1)
    R = R_a[:, None, None].expand(B, M, Tn, L, 3, 3).reshape(B * M * Tn, L, 3, 3)
    t = t_a[:, None, None].expand(B, M, Tn, L, 3).reshape(B * M * Tn, L, 3)
    aa = aatype[:, None, None].expand(B, M, Tn, L).reshape(B * M * Tn, L)
    rm = ctx_mask[:, None, None].expand(B, M, Tn, L).reshape(B * M * Tn, L)
    dec = producer.decoder.decode_structure(flat, aa, R, t, rm)
    ca = dec["backbone_coords"][:, :, 1, :].reshape(B, M, Tn, L, 3)
    return ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["model1", "hs"], required=True)
    ap.add_argument("--ensemble-ckpt", required=True)
    ap.add_argument("--kernel-ckpt", default=None)
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--b1a-ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--history-frames", type=int, default=4)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--horizons", type=int, nargs="+", default=[8])
    ap.add_argument("--max-residues", type=int, default=192)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", choices=["contiguous", "paired"], default="contiguous",
                    help="contiguous = K+H clip split by split_contiguous_window (exact "
                         "step-i<->frame-i time alignment; use for path/ACF metrics); "
                         "paired = the paired_window loader (time-gapped future window).")
    ap.add_argument("--step-ps", type=float, default=1000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = _device(args.device)
    m1, hs, cond_dim = build(args, dev)

    class P: pass
    pr = P(); pr.history_frames = args.history_frames; pr.frame_stride = args.frame_stride
    pr.paired_horizons_frames = args.horizons
    if args.protocol == "contiguous":
        # One clip of K + H frames, split K history / H+1 contiguous future.
        # rollout step i then lines up EXACTLY (in physical time) with truth
        # frame i -- required for path_length / ACF / per-step curves.
        from protein_jepa_frame.data.collate import collate_clips
        from protein_jepa_frame.data.protocol import split_contiguous_window
        K, H = args.history_frames, args.horizons[0]
        cfg = DataConfig(manifest_path=args.manifest, batch_size=args.batch_size,
                         num_workers=args.num_workers, training_mode="clip",
                         num_frames=K + H, frame_stride=args.frame_stride,
                         max_residues=args.max_residues)
        ds = build_dataset(cfg)
        n_take = min(len(ds), args.batches * args.batch_size)
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(ds), generator=g)[:n_take].tolist()
        dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=lambda s: split_contiguous_window(
                            collate_clips(s), K, H, step_ps=args.step_ps))
    else:
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

    acc = defaultdict(list)                                  # per-window metric dicts
    both = args.kernel_ckpt is not None                      # true paired mode
    m1_acc = defaultdict(list); hs_acc = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            paired = _move_batch(batch, dev)
            fut = paired["future_coords"].to(dev)
            true = fut[:, :, :, 1, :][0]                     # [T, L, 3] (bs=1)
            mask = paired["future_residue_mask"].to(dev)[0, -1]
            anchor = paired["context_coords"].to(dev)[0, -1, :, 1, :]   # [L, 3]

            # rollout step 0 replays the anchor, so the aligned future window is
            # true[0:] (which starts AT the anchor under the contiguous protocol).
            cond = _build_cond(paired, cond_dim, dev)
            # rollout_trajectory switches its sample_future_ca signature on
            # args.kernel_ckpt; for Model 1 we must force the static-head path.
            class _A: pass
            a1 = _A(); a1.__dict__.update(vars(args)); a1.kernel_ckpt = None
            t1 = rollout_trajectory(m1, paired, a1, cond)[0]        # [M, T1, L, 3]
            if both:
                ca = hs_walk(hs, paired, args, dev)[0]              # [M, T+1, L, 3]
                th = ca[:, 1:]                                  # step 1..H -> true[1..H]
            else:
                th = None
            if args.mode == "hs" and th is not None:
                pred = th
            else:
                pred = t1
            T = min(pred.shape[1], true.shape[0] - 1)
            if th is not None:
                T = min(T, th.shape[1])
            tw = true[1:1 + T]                                  # true[1..T] aligns with steps 1..T
            d1 = walk_summary(t1[:, :T], tw, anchor, mask)
            dh = walk_summary(th[:, :T], tw, anchor, mask) if th is not None else d1
            for k in set(d1) | set(dh):
                m1_acc[k].append(d1.get(k, np.nan))
                hs_acc[k].append(dh.get(k, np.nan))
    n = len(m1_acc[METRICS[0]])
    res = {"mode": args.mode, "n_windows": n, "metrics": {}}
    print(f"=== {args.mode}: {n} windows "
          f"({'paired Model1 vs H+S' if both else 'single'}) ===")
    print(f"{'metric':22}{'Model1':>10}{'H+S':>10}{'Δ':>10}{'95% CI':>22}{'better':>8}")
    for k in METRICS:
        a = np.array(m1_acc[k], float); b = np.array(hs_acc[k], float)
        d = b - a
        lo, hi = _boot_ci(d)
        if k == "acf_lag1":
            hb = ACF_LAG1_HIGHER
        else:
            hb = k in HIGHER_BETTER
        better = (d.mean() > 0) if hb else (d.mean() < 0)
        sig = (lo > 0) if hb else (hi < 0)
        res["metrics"][k] = dict(model1=float(np.nanmean(a)), hs=float(np.nanmean(b)),
                                 delta=float(np.nanmean(d)), ci=[float(lo), float(hi)],
                                 higher_better=hb, hs_better=bool(better),
                                 significant=bool(sig))
        print(f"{k:22}{np.nanmean(a):>10.4f}{np.nanmean(b):>10.4f}"
              f"{np.nanmean(d):>+10.4f}[{lo:>+8.4f},{hi:>+8.4f}]"
              f"{'YES' if better else 'no ':>8}{' *' if sig else ''}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")


def _boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if d.size < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, d.size, size=(n, d.size))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


if __name__ == "__main__":
    main()
