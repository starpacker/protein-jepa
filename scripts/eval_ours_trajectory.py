"""Our-model trajectory rollout evaluation on mdCATH (NEW, 2026-10-20).

Unlike ``eval_ensemble_metrics_suite.py`` (which compares *single-horizon*
ensembles), this script makes our model produce a **future-frame trajectory**
of length ``--num-frames`` starting from the window's anchor frame, exactly the
way MDGen's forward-simulation task does.  That trajectory is then scored on the
same window set with the same metric family:

    RMSF magnitude / RMSF-r / ROOT-MEAN W2 / VARIANCE W2 / MD-PCA W2 /
    WEAK-CONTACT J / TRANSIENT-CONTACTS J
plus Kabsch CA RMSD (common-anchor) and aligned CA RMSD.

Rollout: the encoder is fed the observed context (anchor = last frame); the head
samples ``--n-samples`` latent perturbations; the frozen anchored decoder maps
each to a CA cloud for the *first* future frame.  For an ergodic rollout we feed
the per-sample mean CA back as the next frame and repeat, so the produced
trajectory is [T_future, L, 3] per sample (an ensemble over ``--n-samples``).
If ``--n-samples 1`` the run is a single deterministic rollout.

Both Stage-A ensemble checkpoints and the later from-none (Stage B) checkpoints
use the same frozen encoder/decoder anchor-basis interface, so this script is
shared between the two.

It never touches the precise-atom-coordinate decoding path; only the frozen
decoder + a new head are read.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _anchors, _targets_from,
)
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch

from evaluation.ensemble_metrics_ext import (
    extended_ensemble_summary, rmsf_profile_aligned,
)
from evaluation.geometry import common_anchor_rmsd, aligned_rmsd

METRIC_DIRS = {
    "rmsf_mae": False, "rmsf_corr": True, "root_mean_w2": False,
    "variance_w2": False, "md_pca_w2": False,
    "weak_contact_j": True, "transient_contacts_j": True,
    "kabsch_ca_rmsd": False, "aligned_ca_rmsd": False,
}


def _build_cond(paired, cond_dim, dev):
    if cond_dim <= 0:
        return None
    ht = paired.get("horizon_time_ps")
    B = paired["context_coords"].shape[0]
    t_norm = (torch.log(ht.to(dev).clamp_min(1.0)) /
              torch.log(torch.tensor(1e4, device=dev))
              if ht is not None else torch.zeros(B, device=dev))
    temp = paired.get("temperature_k")
    T_norm = ((temp.to(dev).float() - 300.0) / 150.0
              if temp is not None else torch.zeros(B, device=dev))
    parts = [t_norm, T_norm]
    if cond_dim <= 2:
        return torch.stack(parts[:cond_dim], dim=-1)
    return torch.stack(parts + [torch.zeros_like(t_norm)] * (cond_dim - 2), dim=-1)


def rollout_trajectory(producer, paired, args, cond):
    """Return ca_traj [B, n_samples, T_fut, L, 3] for the future window.

    Autoregressive: each step decodes the CA cloud for the next frame from the
    frozen anchored decoder, then the *mean* of the sampled clouds is appended
    to the context (all samples share the same observed context path, so they
    form an ensemble about a common mean rollout).
    """
    ctx = paired["context_coords"].to(next(producer.parameters()).device)
    ctx_mask = paired["context_residue_mask"]
    aatype = paired["future_aatype"]
    ri = paired["future_residue_index"]
    ci = paired["future_chain_id"]
    T_fut = paired["future_coords"].shape[1]
    dev = ctx.device
    aatype = aatype.to(dev); ri = ri.to(dev); ci = ci.to(dev)
    ctx_mask = ctx_mask.to(dev)

    # rolling context buffer keeps the last K history frames
    K = ctx.shape[1]
    buf = ctx.clone()                                    # [B,K,L,4,3]
    per_step = []                                        # list of [B,n,L,3]
    for _ in range(T_fut):
        R_a, t_a = _anchors_like(buf)
        if getattr(args, "kernel_ckpt", None):
            tk = getattr(producer, "_temperature_k", 320.0)
            _tk = paired.get("temperature_k")
            if _tk is not None:
                tk = float(_tk.to(dev).float().mean().item())
            out = producer.sample_future_ca(
                buf, ctx_mask[:, -1], aatype, ri, ci, R_a, t_a,
                n_samples=args.n_samples, temperature_k=tk)
        else:
            out = producer.sample_future_ca(
                buf, ctx_mask[:, -1], aatype, ri, ci, R_a, t_a,
                n_samples=args.n_samples, cond=cond)
        ca = out["ca_samples"]                           # [B,n,L,3]
        per_step.append(ca)
        # feedback: build a synthetic backbone frame from the mean CA and slide
        mean_ca = ca.mean(dim=1)                         # [B,L,3]
        new_frame = _synthetic_backbone(mean_ca, buf[:, -1])   # [B,L,4,3]
        buf = torch.cat([buf[:, 1:], new_frame[:, None]], dim=1)
    return torch.stack(per_step, dim=2)                  # [B,n,T,L,3]


def _pad_profile(p, L):
    """Right-pad a [l] RMSF profile with NaN to a fixed length ``L``.

    Aligning every window's profile to the same residue-index axis makes the
    pooled ("global") correlation well defined even though windows have
    different sequence lengths; NaN entries are dropped by ``_corr``.
    """
    p = p.detach().cpu()          # ALWAYS return CPU so pooling never mixes devices
    if p.shape[0] == L:
        return p
    if p.shape[0] > L:
        return p[:L]
    pad = torch.full((L - p.shape[0],), float("nan"), dtype=p.dtype)
    return torch.cat([p, pad])


def _anchors_like(ctx):
    """Backbone frames (R,t) of the current last context frame."""
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    return _extract_backbone_frames(ctx[:, -1])


def _synthetic_backbone(ca, prev_frame):
    """Place N/C/O around the predicted CA using the previous frame's offsets.

    This is a *rollout plumbing* detail, not a decoding change: the frozen
    decoder consumes only R_anchor/t_anchor (built from N/CA/C) so we just need
    a consistent backbone whose CA equals ``ca`` and whose local geometry is
    inherited from ``prev_frame``.
    """
    off = prev_frame - prev_frame[:, :, 1:2, :]          # [B,L,4,3] offsets from CA
    return ca[:, :, None, :] + off


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-ckpt", default=None)
    p.add_argument("--kernel-ckpt", default=None,
                   help="Stage B H+S (GLE) kernel ckpt; uses frozen Model-1 stack")
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
    p.add_argument("--num-frames", type=int, default=100,
                   help="rollout length (future frames)")
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--pca-k", type=int, default=3)
    p.add_argument("--weak-lo", type=float, default=0.05)
    p.add_argument("--weak-hi", type=float, default=0.5)
    p.add_argument("--trans-lo", type=float, default=0.01)
    p.add_argument("--trans-hi", type=float, default=0.99)
    p.add_argument("--min-seq-sep", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=256)
    p.add_argument("--dist-bins", type=int, default=40)
    p.add_argument("--dist-lo", type=float, default=0.0)
    p.add_argument("--dist-hi", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = _device(args.device)
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    _kernel_mode = bool(getattr(args, "kernel_ckpt", None))
    if _kernel_mode:
        from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
        from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
        kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
        k_cfg = kck.get("kernel_config") or {}
        k_args = kck.get("args", {})
        k_cfg["step_ps"] = k_args.get("step_ps", k_cfg.get("step_ps", 1000.0))
        k_cfg["rollout_grad_steps"] = None
        kernel = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
        kernel.load_state_dict(kck["kernel"])
        kernel.eval()
        producer = HSEnsembleProducer(encoder, decoder, kernel).to(dev).eval()
        producer._rollout_steps = int(k_args.get("rollout_steps", 1))
        producer._rollout_grad_steps = None
        producer._temperature_k = float(k_args.get("temperature_k", 320.0))
        cond_dim = 0
        for q in producer.parameters():
            q.requires_grad_(False)
        print(f"[ours-traj] H+S kernel rollout_steps={producer._rollout_steps} "
              f"n_samples={args.n_samples} num_frames={args.num_frames}")
    else:
        ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
        ck_args = ck.get("args", {})
        cond_dim = int(ck_args.get("cond_dims", 0))
        head = GaussianHHead(dH, hidden=int(ck_args.get("head_hidden", 128)),
                             cond_dim=cond_dim)
        head.load_state_dict(ck["head"])
        head.to(dev).eval()
        producer = EnsembleProducer(encoder, decoder, head).to(dev).eval()
        for q in producer.parameters():
            q.requires_grad_(False)
        print(f"[ours-traj] cond_dim={cond_dim} n_samples={args.n_samples} "
              f"num_frames={args.num_frames}")

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

    sink = defaultdict(list)
    pers_sink = defaultdict(list)
    pred_glob, pred_tgt = [], defaultdict(list)
    true_glob, true_tgt = [], defaultdict(list)
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            n_batches = i + 1
            paired = _move_batch(batch, dev)
            fut = paired["future_coords"].to(dev)
            fn = min(args.num_frames, fut.shape[1])
            true_win = fut[:, :fn, :, 1, :]                 # [B,T,L,3] CA
            mask = paired["future_residue_mask"].to(dev)[:, -1]   # [B,L]
            cond = _build_cond(paired, cond_dim, dev)

            traj = rollout_trajectory(producer, paired, args, cond)  # [B,n,T,L,3]
            T = traj.shape[2]

            # persistence ensemble = the anchor frame copied to every future frame
            anchor_ca = paired["context_coords"].to(dev)[:, -1, :, 1, :]  # [B,L,3]
            pers = anchor_ca[:, None, None].expand(-1, args.n_samples, T, -1, -1)

            dom_ids = paired.get("protein_id")
            for b in range(traj.shape[0]):
                mb = mask[b]
                tw = true_win[b, :T]
                sc = extended_ensemble_summary(
                    traj[b].reshape(-1, traj.shape[-2], 3), tw.reshape(-1, tw.shape[-2], 3),
                    mb, cutoff=args.cutoff, pca_k=args.pca_k,
                    weak_lo=args.weak_lo, weak_hi=args.weak_hi,
                    trans_lo=args.trans_lo, trans_hi=args.trans_hi)
                for k, v in sc.items():
                    sink[k].append(v)
                ps = extended_ensemble_summary(
                    pers[b].reshape(-1, pers.shape[-2], 3), tw.reshape(-1, tw.shape[-2], 3),
                    mb, cutoff=args.cutoff, pca_k=args.pca_k,
                    weak_lo=args.weak_lo, weak_hi=args.weak_hi,
                    trans_lo=args.trans_lo, trans_hi=args.trans_hi)
                for k, v in ps.items():
                    pers_sink[k].append(v)

                # Kabsch / aligned CA RMSD on the FINAL rolled-out frame vs true
                ka = [common_anchor_rmsd(traj[b, s, -1], anchor_ca[b], tw[-1], mb)
                      for s in range(traj.shape[1])]
                al = [aligned_rmsd(traj[b, s, -1], tw[-1], mb)
                      for s in range(traj.shape[1])]
                sink["kabsch_ca_rmsd"].append(float(torch.tensor(ka).nanmean()))
                sink["aligned_ca_rmsd"].append(float(torch.tensor(al).nanmean()))
                # persistence Kabsch: anchor displaced onto the true final frame
                pers_sink["kabsch_ca_rmsd"].append(
                    common_anchor_rmsd(anchor_ca[b], anchor_ca[b], tw[-1], mb))
                pers_sink["aligned_ca_rmsd"].append(
                    aligned_rmsd(anchor_ca[b], tw[-1], mb))

                tp = _pad_profile(rmsf_profile_aligned(tw, mb), args.max_residues)
                true_glob.append(tp)
                if dom_ids is not None:
                    true_tgt[dom_ids[b]].append(tp)
                # one averaged predicted profile per window (same length as tp)
                pp_s = torch.stack([
                    _pad_profile(rmsf_profile_aligned(traj[b, s], mb), args.max_residues)
                    for s in range(traj.shape[1])]).nanmean(dim=0)
                pred_glob.append(pp_s)
                if dom_ids is not None:
                    pred_tgt[dom_ids[b]].append(pp_s)
            if n_batches % 5 == 0:
                print(f"  [{n_batches}] rmsf_mae={torch.tensor(sink['rmsf_mae']).mean():.3f} "
                      f"kabsch={torch.tensor(sink['kabsch_ca_rmsd']).mean():.3f}")

    def _corr(a, b):
        a, b = a.double(), b.double()
        fin = torch.isfinite(a) & torch.isfinite(b)
        a, b = a[fin], b[fin]
        if a.numel() < 2:
            return float("nan")
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))

    agg = {k: float(torch.tensor(v).mean()) for k, v in sink.items()}
    pers_agg = {k: float(torch.tensor(v).mean()) for k, v in pers_sink.items()}

    tg = torch.cat(true_glob)
    rmsf_r = {"global": _corr(torch.cat(pred_glob), tg)}
    per_t = {}
    for dom, vecs in pred_tgt.items():
        # same protein -> same length; average the window profiles
        pp = torch.stack(vecs).nanmean(dim=0) if len(vecs) > 1 else vecs[0]
        tv = true_tgt[dom]
        tt = torch.stack(tv).nanmean(dim=0) if len(tv) > 1 else tv[0]
        per_t[dom] = _corr(pp, tt)
    vals = [v for v in per_t.values() if v == v]
    rmsf_r["per_target"] = per_t
    rmsf_r["per_target_stats"] = (
        {"n_targets": len(vals),
         "mean": float(torch.tensor(vals).mean()),
         "median": float(torch.tensor(vals).median()),
         "min": float(torch.tensor(vals).min()),
         "max": float(torch.tensor(vals).max())} if vals else {"n_targets": 0})

    res = {
        "model": "ours", "checkpoint": args.ensemble_ckpt,
        "manifest": args.manifest, "n_batches": n_batches,
        "n_samples_total": len(sink["rmsf_mae"]),
        "n_samples_ensemble": args.n_samples, "num_frames": args.num_frames,
        "cond_dim": cond_dim,
        "metrics": agg, "persistence": pers_agg, "rmsf_r": rmsf_r,
        "config": {"cutoff": args.cutoff, "pca_k": args.pca_k,
                   "weak_band": [args.weak_lo, args.weak_hi],
                   "transient_band": [args.trans_lo, args.trans_hi]},
    }
    print(json.dumps({k: agg[k] for k in METRIC_DIRS}, indent=2))
    print("persistence:", json.dumps({k: pers_agg.get(k) for k in METRIC_DIRS}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[ours-traj] wrote {args.out}")


if __name__ == "__main__":
    main()
