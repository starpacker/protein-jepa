"""Direct paired comparison: Stage-B H+S (GLE) vs Model 1 (static Gaussian head).

Both share the SAME frozen Stage-A encoder + B1a decoder and are evaluated on the
SAME windows with the SAME ensemble metrics.  We compute, per window, the
difference (H+S - Model1) for each metric and bootstrap a 95% CI so we can say
whether H+S is significantly better -- a proper paired test, not a table peek.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import sys

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
from protein_jepa_frame.training.ensemble_targets import persistence_ensemble
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack, _anchors, _targets_from
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch
from evaluation.ensemble_metrics_ext import extended_ensemble_summary, rmsf_profile_aligned

METRICS = ["rmsf_mae","rmsf_corr","root_mean_w2","variance_w2","md_pca_w2",
           "weak_contact_j","transient_contacts_j"]
HIGHER_BETTER = {"rmsf_corr","weak_contact_j","transient_contacts_j"}


def build(args, dev):
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    # Model 1 static head
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    ca = ck.get("args", {})
    head = GaussianHHead(dH, hidden=int(ca.get("head_hidden", 128)),
                         cond_dim=int(ca.get("cond_dims", 0)))
    head.load_state_dict(ck["head"]); head.to(dev).eval()
    m1 = EnsembleProducer(encoder, decoder, head).to(dev).eval()
    # H+S kernel
    kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
    k_cfg = kck.get("kernel_config") or {}
    k_args = kck.get("args", {})
    k_cfg["step_ps"] = k_args.get("step_ps", 1000.0); k_cfg["rollout_grad_steps"] = None
    kern = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
    kern.load_state_dict(kck["kernel"]); kern.eval()
    hs = HSEnsembleProducer(encoder, decoder, kern).to(dev).eval()
    hs._rollout_steps = int(k_args.get("rollout_steps", 1))
    hs._rollout_grad_steps = None
    hs._temperature_k = float(k_args.get("temperature_k", 320.0))
    return m1, hs, int(ca.get("cond_dims", 0))


def score(ens, true_win, mask, args):
    # Mirrors eval_ensemble_metrics_suite._score: score per batch sample, with
    # the true future WINDOW [B,T,L,3] as the reference ensemble.
    ens = ens.detach().cpu(); true_win = true_win.detach().cpu(); mask = mask.detach().cpu()
    out = defaultdict(list)
    for b in range(ens.shape[0]):
        s = extended_ensemble_summary(ens[b], true_win[b], mask[b],
                                      pca_k=args.pca_k, cutoff=args.cutoff)
        for k in METRICS:
            if k in s:
                out[k].append(float(s[k]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-ckpt", required=True)
    ap.add_argument("--kernel-ckpt", required=True)
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--b1a-ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--history-frames", type=int, default=4)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1,2,4,8])
    ap.add_argument("--max-residues", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--pca-k", type=int, default=3)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--min-seq-sep", type=int, default=3)
    ap.add_argument("--dist-lo", type=float, default=0.0)
    ap.add_argument("--max-pairs", type=int, default=4096)
    ap.add_argument("--dist-hi", type=float, default=40.0)
    ap.add_argument("--dist-bins", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = _device(args.device)
    m1, hs, cond_dim = build(args, dev)

    class P: pass
    pr = P(); pr.history_frames = args.history_frames; pr.frame_stride = args.frame_stride
    pr.paired_horizons_frames = args.horizons
    cfg = DataConfig(manifest_path=args.manifest, batch_size=args.batch_size,
                     num_workers=args.num_workers, training_mode="paired_window",
                     num_frames=args.history_frames, paired_all_horizons=False,
                     max_residues=args.max_residues)
    apply_protocol(cfg, protocol_from_args(pr))
    ds = build_dataset(cfg)
    n_take = min(len(ds), args.batches*args.batch_size)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n_take].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=lambda s: _truncate_history(collate_paired_windows(s), args.history_frames))

    diffs = defaultdict(list)      # per-window (H+S - Model1)
    m1_all = defaultdict(list); hs_all = defaultdict(list)
    rmsf_r_diffs = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches: break
            paired = _move_batch(batch, dev)
            _t, fut, fmask = _targets_from(paired, args, dev)
            true_win = fut[:, :, :, 1, :]; mask = fmask[:, -1]
            ctx = paired["context_coords"].to(dev)
            cm = paired["context_residue_mask"].to(dev)[:, -1]
            aa = paired["future_aatype"].to(dev); ri = paired["future_residue_index"].to(dev)
            ci = paired["future_chain_id"].to(dev)
            R_a, t_a = _anchors(paired, dev)
            tk = paired.get("temperature_k")
            tkv = float(tk.to(dev).float().mean().item()) if tk is not None else 320.0

            o1 = m1.sample_future_ca(ctx, cm, aa, ri, ci, R_a, t_a, n_samples=args.n_samples,
                                     cond=None)["ca_samples"]
            oh = hs.sample_future_ca(ctx, cm, aa, ri, ci, R_a, t_a, n_samples=args.n_samples,
                                     temperature_k=tkv)["ca_samples"]
            s1 = score(o1, true_win, mask, args); sh = score(oh, true_win, mask, args)
            for k in s1:
                m1_all[k].append(s1[k]); hs_all[k].append(sh[k])
                diffs[k].append(sh[k]-s1[k])
            # RMSF-r per batch (profile correlation of the ensembles)
            o1c = o1.detach().cpu(); ohc = oh.detach().cpu()
            twc = true_win.detach().cpu(); mc = mask.detach().cpu()
            for b in range(twc.shape[0]):
                tp = rmsf_profile_aligned(twc[b], mc[b])      # [L]
                p1 = rmsf_profile_aligned(o1c[b], mc[b])      # [L]
                ph = rmsf_profile_aligned(ohc[b], mc[b])      # [L]
                rmsf_r_diffs.append((_corr(ph, tp) - _corr(p1, tp)))

    res = {"n_windows": len(diffs[METRICS[0]])}
    print(f"\n{'metric':22} {'Model1':>10} {'H+S':>10} {'Δ(H+S−M1)':>11} {'95% CI':>22} {'better':>7}")
    rows = {}
    for k in METRICS:
        a = np.array(m1_all[k]); b = np.array(hs_all[k]); d = b-a
        lo, hi = _boot_ci(d)
        hb = k in HIGHER_BETTER
        better = (d.mean() > 0) if hb else (d.mean() < 0)
        sig = (lo > 0) if hb else (hi < 0)
        rows[k] = dict(model1=float(a.mean()), hs=float(b.mean()),
                       delta=float(d.mean()), ci=[float(lo),float(hi)],
                       higher_better=hb, hs_better=bool(better), significant=bool(sig))
        print(f"{k:22} {a.mean():>10.4f} {b.mean():>10.4f} {d.mean():>+11.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {'YES' if better else 'no ':>7}{' *' if sig else ''}")
    rd = np.array(rmsf_r_diffs); lo,hi = _boot_ci(rd)
    rows["rmsf_r"] = dict(model1=None, hs=None, delta=float(rd.mean()), ci=[float(lo),float(hi)],
                          higher_better=True, hs_better=bool(rd.mean()>0), significant=bool(lo>0))
    print(f"{'rmsf_r(profile)':22} {'':>10} {'':>10} {rd.mean():>+11.4f} "
          f"[{lo:>+8.4f},{hi:>+8.4f}] {'YES' if rd.mean()>0 else 'no ':>7}{' *' if lo>0 else ''}")
    res["metrics"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {args.out}")


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0,1])


def _boot_ci(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 2: return (np.nan, np.nan)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == "__main__":
    main()
