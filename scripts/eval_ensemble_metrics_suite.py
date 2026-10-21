"""Unified ensemble-metric suite (NEW, 2026-10-19).

Computes the six requested ensemble-level metrics *together* for three
ensembles -- model / persistence / oracle -- on any val manifest
(mdCATH, ATLAS-1ns, ATLAS-100ps):

    RMSF                     (rmsf_mae, from evaluation.ensemble_metrics)
    ROOT MEAN W2             (root_mean_w2)
    VARIANCE W2              (variance_w2)
    MD-PCA W2                (md_pca_w2)
    WEAK CONTACT J           (weak_contact_j)
    TRANSIENT CONTACTS J     (transient_contacts_j)

All W2 metrics are lower-better (0 = perfect); Jaccard are higher-better
(1 = perfect).  Reports per metric the model value, the persistence baseline,
the oracle ceiling, and a bootstrap 95% CI of (persist - model).

This script never touches the precise-coordinate decoding path; it only reads
checkpoints and frozen models.
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
from protein_jepa_frame.training.ensemble_targets import persistence_ensemble
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _anchors, _targets_from,
)
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch

from evaluation.ensemble_metrics_ext import (
    extended_ensemble_summary, rmsf_profile_aligned,
)

# metric name -> direction; True = higher is better
METRIC_DIRS = {
    "rmsf_mae": False,       # RMSF magnitude error (A), lower better
    "rmsf_corr": True,       # RMSF-r: profile Pearson correlation, higher better
    "root_mean_w2": False,
    "variance_w2": False,
    "md_pca_w2": False,
    "weak_contact_j": True,
    "transient_contacts_j": True,
}

# extra diagnostic keys reported but not scored as "model vs persistence"
DIAG_KEYS = ("weak_n_true", "weak_n_pred", "transient_n_true", "transient_n_pred")


def _bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=0):
    if not deltas:
        return [float("nan"), float("nan")]
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    means = torch.empty(n_boot, dtype=torch.float64)
    N = t.numel()
    for b in range(n_boot):
        idx = torch.randint(0, N, (N,), generator=g)
        means[b] = t[idx].mean()
    return [torch.quantile(means, alpha / 2).item(),
            torch.quantile(means, 1 - alpha / 2).item()]


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
        cond = torch.stack(parts[:cond_dim], dim=-1)
    else:
        cond = torch.stack(parts + [torch.zeros_like(t_norm)] * (cond_dim - 2), dim=-1)
    return cond


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two 1-D profiles (nan-safe)."""
    a = a.double(); b = b.double()
    fin = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[fin], b[fin]
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def _rmsf_r_stats(per_target_corr: dict) -> dict:
    """Distribution of per-target RMSF-r values (ignoring nan targets)."""
    vals = [v for v in per_target_corr.values() if v == v]      # drop nan
    if not vals:
        return {"n_targets": 0}
    t = torch.tensor(vals, dtype=torch.float64)
    return {
        "n_targets": len(vals),
        "mean": float(t.mean()),
        "median": float(t.median()),
        "std": float(t.std(unbiased=True)) if len(vals) > 1 else 0.0,
        "min": float(t.min()),
        "max": float(t.max()),
        "q25": float(torch.quantile(t, 0.25)),
        "q75": float(torch.quantile(t, 0.75)),
        "frac_gt0": float((t > 0).double().mean()),
        "frac_gt0_3": float((t > 0.3).double().mean()),
    }


def _score(pred, true, mask, args, n_true=None):
    """Return the 6-metric dict for one ensemble pair, per-sample (mask [B,L]).

    ``pred``  is [B, P, L, 3]  (an ensemble of P conformers per sample).
    ``true``  is [B, L, 3]     (the single true future frame); it is broadcast to
    the same ensemble size so that the true *window* (used for the oracle and for
    the distributional W2 metrics) has P copies.  Where a true *window* is
    available the caller passes ``true`` already as [B, P, L, 3].
    """
    if true.dim() == 3:
        P = pred.shape[1]
        true = true[:, None].expand(-1, P, -1, -1)
    out = defaultdict(list)
    for b in range(pred.shape[0]):
        m = mask[b] if mask is not None else None
        s = extended_ensemble_summary(
            pred[b], true[b], m,
            cutoff=args.cutoff, pca_k=args.pca_k,
            weak_lo=args.weak_lo, weak_hi=args.weak_hi,
            trans_lo=args.trans_lo, trans_hi=args.trans_hi)
        for k, v in s.items():
            out[k].append(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-ckpt", default=None,
                   help="Gaussian-head ckpt (Model 1); ignored if --kernel-ckpt set")
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
    if getattr(args, "kernel_ckpt", None):
        # Stage B "H+S (GLE) iteration": Model-1 frozen stack + trainable StageB1b
        # GLE kernel.  Same producer interface as the Gaussian head, so the rest
        # of this suite is unchanged.
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
        print(f"[suite] H+S kernel rollout_steps={producer._rollout_steps} "
              f"T={producer._temperature_k} pca_k={args.pca_k} n_samples={args.n_samples}")
    else:
        ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
        ck_args = ck.get("args", {})
        cond_dim = int(ck_args.get("cond_dims", 0))
        head = GaussianHHead(dH, hidden=int(ck_args.get("head_hidden", 128)),
                             cond_dim=cond_dim)
        head.load_state_dict(ck["head"])
        head.to(dev).eval()
        producer = EnsembleProducer(encoder, decoder, head).to(dev).eval()
        print(f"[suite] cond_dim={cond_dim} pca_k={args.pca_k} n_samples={args.n_samples}")

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

    model_v = defaultdict(list)
    pers_v = defaultdict(list)
    orac_v = defaultdict(list)
    # RMSF-r accumulators:
    #   *_glob : list of per-residue profile vectors, pooled over ALL windows (global r)
    #   *_tgt  : {domain_id: [profile vectors]} for per-target r
    prof_glob = {"model": [], "pers": [], "orac": []}
    prof_tgt = {"model": defaultdict(list), "pers": defaultdict(list),
                "orac": defaultdict(list)}
    true_glob = []
    true_tgt = defaultdict(list)
    protos, hz_all = set(), []
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            n_batches = i + 1
            paired = _move_batch(batch, dev)
            _targets, fut, fmask = _targets_from(paired, args, dev)
            true_ca = fut[:, -1, :, 1, :]                          # [B, L, 3] CA@horizon
            true_win = fut[:, :, :, 1, :]                          # [B, T, L, 3] CA window
            mask_win = fmask                                       # [B, T, L]
            mask = mask_win[:, -1]                                 # [B, L] horizon mask

            ctx = paired["context_coords"].to(dev)
            ctx_mask = paired["context_residue_mask"].to(dev)[:, -1]
            aatype = paired["future_aatype"].to(dev)
            ri = paired["future_residue_index"].to(dev)
            ci = paired["future_chain_id"].to(dev)
            R_a, t_a = _anchors(paired, dev)
            cond = _build_cond(paired, cond_dim, dev)
            if getattr(args, "kernel_ckpt", None):
                tk = paired.get("temperature_k")
                tk_val = float(producer._temperature_k)
                if tk is not None:
                    tk_val = float(tk.to(dev).float().mean().item())
                out = producer.sample_future_ca(ctx, ctx_mask, aatype, ri, ci,
                                                R_a, t_a, n_samples=args.n_samples,
                                                temperature_k=tk_val)
            else:
                out = producer.sample_future_ca(ctx, ctx_mask, aatype, ri, ci,
                                                R_a, t_a, n_samples=args.n_samples,
                                                cond=cond)
            ca_pred = out["ca_samples"]

            pca = persistence_ensemble(ctx, args.n_samples,
                                       paired["context_residue_mask"].to(dev))

            # For the distributional metrics the *true* reference is the true
            # future WINDOW (all its frames), not a single copied frame.
            for name, ens in (("model", ca_pred), ("pers", pca), ("orac", true_win)):
                sc = _score(ens, true_win, mask, args)
                sink = {"model": model_v, "pers": pers_v, "orac": orac_v}[name]
                for k, v in sc.items():
                    sink[k].extend(v)

            # ---- RMSF-r: collect per-residue RMSF profiles (global + per-target) ----
            dom_ids = paired.get("protein_id")
            for b in range(true_win.shape[0]):
                mb = mask[b]
                tp = rmsf_profile_aligned(true_win[b], mb)        # [L]  (nan where masked)
                true_glob.append(tp)
                if dom_ids is not None:
                    true_tgt[dom_ids[b]].append(tp)
                for name, ens in (("model", ca_pred), ("pers", pca), ("orac", true_win)):
                    pp = rmsf_profile_aligned(ens[b], mb)
                    prof_glob[name].append(pp)
                    if dom_ids is not None:
                        prof_tgt[name][dom_ids[b]].append(pp)

            for t in paired["trajectory_id"]:
                protos.add(t)
            if "horizon_frames" in paired:
                hz_all += paired["horizon_frames"].tolist()

    def agg(v):
        return {k: float(torch.tensor(x).mean()) for k, x in v.items()}

    model_a, pers_a, orac_a = agg(model_v), agg(pers_v), agg(orac_v)
    win = {}
    for k in METRIC_DIRS:
        better = METRIC_DIRS[k]
        # improvement = (pers - model) for lower-better, (model - pers) for higher-better
        if better:
            delta = [m - p for m, p in zip(model_v[k], pers_v[k])]
        else:
            delta = [p - m for p, m in zip(pers_v[k], model_v[k])]
        dmean = float(torch.tensor(delta).mean())
        win[k] = {"mean_improvement": dmean,
                  "ci95": _bootstrap_ci(delta),
                  "model_better": bool(dmean > 0),
                  "win_rate": float(torch.tensor([1.0 if d > 0 else 0.0
                                                  for d in delta]).mean())}

    # ---- RMSF-r: global (pooled over all windows) + per-target distribution ----
    def _pool(vecs):
        if not vecs:
            return None
        return torch.cat(vecs)

    tg = _pool(true_glob)
    rmsf_r = {}
    if tg is not None:
        for name in ("model", "pers", "orac"):
            pg = _pool(prof_glob[name])
            rmsf_r[name] = {"global": _corr(pg, tg) if pg is not None else float("nan")}
    # per-target correlation: concatenate all windows of a target, then correlate
    true_tgt_pool = {k: _pool(v) for k, v in true_tgt.items()}
    for name in ("model", "pers", "orac"):
        per_t = {}
        for dom, vecs in prof_tgt[name].items():
            tv = true_tgt_pool.get(dom)
            if tv is None:
                continue
            per_t[dom] = _corr(_pool(vecs), tv)
        rmsf_r.setdefault(name, {})["per_target"] = per_t
        rmsf_r[name]["per_target_stats"] = _rmsf_r_stats(per_t)
    # also keep the window-mean correlation (the scalar already in model_v)
    for name, v in (("model", model_v), ("pers", pers_v), ("orac", orac_v)):
        rmsf_r.setdefault(name, {})["per_window_mean"] = float(
            torch.tensor(v["rmsf_corr"], dtype=torch.float64).nanmean())

    res = {
        "manifest": args.manifest,
        "rmsf_r": rmsf_r,
        "n_batches": n_batches,
        "n_samples_total": len(model_v["rmsf_mae"]),
        "n_trajectories": len(protos),
        "n_samples_ensemble": args.n_samples,
        "cond_dim": cond_dim,
        "config": {"cutoff": args.cutoff, "pca_k": args.pca_k,
                   "weak_band": [args.weak_lo, args.weak_hi],
                   "transient_band": [args.trans_lo, args.trans_hi]},
        "metric_direction": METRIC_DIRS,
        "model": model_a,
        "persistence": pers_a,
        "oracle": orac_a,
        "model_vs_persistence": win,
        "diagnostics": {name: {k: float(d.get(k, 0.0)) for k in DIAG_KEYS}
                        for name, d in (("model", model_a), ("persistence", pers_a),
                                        ("oracle", orac_a))},
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"[suite] wrote {args.out}")


if __name__ == "__main__":
    main()
