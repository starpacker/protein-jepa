"""Lasp final evaluation: Lasp (Stage A + GLE) vs Model 1 vs bioemu2.

Apples-to-apples by construction: same manifest, same ``EvalProtocol`` window
iterator, same true MD future ensemble, same persistence, same
``extended_ensemble_summary`` metric code as ``run_bioemu2_zeroshot.py`` and
``run_eval_fromnone_ensemble.py``.  So the Lasp numbers here can be compared
directly against ``results/bioemu2_reps/summary.json``.

Models scored per window (horizon H frames):
  * Lasp   : HSEnsembleProducer.sample_future_ca with the GLE kernel rollout;
  * Model1 : EnsembleProducer (static GaussianHHead);
  * persist: anchor CA repeated.
"""
from __future__ import annotations

import argparse, copy, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack
from protein_jepa_frame.training.train_utils import _device
from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary

METRICS = ["rmsf_mae", "rmsf_corr", "root_mean_w2", "variance_w2", "md_pca_w2",
           "weak_contact_j", "transient_contacts_j"]
HIGHER_BETTER = {"rmsf_corr", "weak_contact_j", "transient_contacts_j"}


def build(args, dev):
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    enc_m1 = copy.deepcopy(encoder); dec_m1 = copy.deepcopy(decoder)
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    ca = ck.get("args", {})
    head = GaussianHHead(dH, hidden=int(ca.get("head_hidden", 128)),
                         cond_dim=int(ca.get("cond_dims", 0)))
    head.load_state_dict(ck["head"]); head.to(dev).eval()
    m1 = EnsembleProducer(enc_m1, dec_m1, head).to(dev).eval()
    kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
    k_cfg = kck.get("kernel_config") or {}; k_args = kck.get("args", {})
    k_cfg = dict(k_cfg)
    k_cfg.setdefault("bath_coupling", k_args.get("bath_coupling", "force"))
    k_cfg["step_ps"] = k_args.get("step_ps", 1000.0); k_cfg["rollout_grad_steps"] = None
    kern = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
    kern.load_state_dict(kck["kernel"]); kern.eval()
    te = bool(kck.get("train_encoder", False)) and kck.get("encoder") is not None
    td = bool(kck.get("train_decoder", False)) and kck.get("decoder") is not None
    hs = HSEnsembleProducer(encoder, decoder, kern, train_encoder=te,
                            train_decoder=td,
                            history_conditioning=bool(k_args.get("history_conditioning", False))).to(dev).eval()
    if te:
        hs.encoder.load_state_dict(kck["encoder"], strict=False)
    if td:
        hs.decoder.load_state_dict(kck["decoder"], strict=False)
    hs._rollout_steps = int(k_args.get("rollout_steps", 8))
    hs._eval_path_ensemble = False
    hs._rollout_grad_steps = None
    hs._temperature_k = float(k_args.get("temperature_k", 320.0))
    return m1, hs, int(ca.get("cond_dims", 0)), k_args


@torch.no_grad()
def _ctx_tensors(ctx, rmask, dev):
    coords = ctx["coords"].to(dev).unsqueeze(0)              # [1,K,L,4,3]
    K, L = coords.shape[1], coords.shape[2]
    rm = rmask.to(dev).unsqueeze(0)                          # [1,L]
    aat = ctx["aatype"].to(dev).unsqueeze(0)
    ridx = ctx["residue_index"].to(dev).unsqueeze(0)
    cid = ctx["chain_id"].to(dev).unsqueeze(0)
    rmK = rm.unsqueeze(1).expand(1, K, L)
    return coords, rm, aat, ridx, cid, rmK


@torch.no_grad()
def score_hs(hs, ctx, rmask, n_samples, dev, temperature_k):
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    coords, rm, aat, ridx, cid, _ = _ctx_tensors(ctx, rmask, dev)
    anchor = coords[:, -1]                                   # [1,L,4,3]
    R, t = _extract_backbone_frames(anchor)
    out = hs.sample_future_ca(coords, rm, aat, ridx, cid, R, t,
                              n_samples=n_samples, temperature_k=temperature_k,
                              context_time_ps=ctx["time_ps"].to(dev).unsqueeze(0))
    if getattr(hs, "_eval_path_ensemble", False):
        from protein_jepa_frame.training.train_stage_b_hs import _decode_path
        idx = torch.arange(out["n_steps"] + 1, device=dev)
        path = _decode_path(hs, out["H_traj"], R, t, aat, rm,
                            1, n_samples, coords.shape[2], None, idx)
        return path[0].flatten(0, 1)
    return out["ca_samples"][0]                              # [N,L,3]


@torch.no_grad()
def score_m1(m1, ctx, rmask, n_samples, cond_dim, dev):
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    coords, rm, aat, ridx, cid, _ = _ctx_tensors(ctx, rmask, dev)
    anchor = coords[:, -1]
    R, t = _extract_backbone_frames(anchor)
    cond = None
    if cond_dim > 0:
        cond = torch.zeros(1, cond_dim, device=dev)
    out = m1.sample_future_ca(coords, rm, aat, ridx, cid, R, t,
                              n_samples=n_samples, cond=cond)
    return out["ca_samples"][0]


def _cluster_boot_ci(values, clusters, n=2000, seed=0):
    """Protein-clustered bootstrap CI for the mean of ``values``.

    Windows from one protein share a fold and a trajectory family, so they are
    NOT independent.  Resampling _windows_ understates the CI -- catastrophically
    so when the whole run is a single protein, where it reports a tight interval
    around one domain's idiosyncrasy.

    Here we resample PROTEINS with replacement and, inside each drawn protein,
    average its windows.  The CI is that of the mean over proteins.
    """
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if v.size < 2:
        return [float("nan"), float("nan")]
    uniq = np.unique(c)
    if uniq.size < 2:                        # single protein: CI is undefined
        return [float("nan"), float("nan")]
    # per-protein means and counts
    sums = np.zeros(uniq.size); cnts = np.zeros(uniq.size)
    pos = {u: i for i, u in enumerate(uniq)}
    for val, cl in zip(v, c):
        i = pos[cl]; sums[i] += val; cnts[i] += 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, uniq.size, size=(n, uniq.size))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-ckpt", required=True)
    ap.add_argument("--ensemble-ckpt", required=True)
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--b1a-ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--horizons-ns", type=float, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--history-frames", type=int, default=8)
    ap.add_argument("--max-residues", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--pca-k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    # --- benchmark coverage -------------------------------------------------
    # Default True: the manifest-order walk scores only the first few trajectories,
    # which on this val split is ONE 60-residue protein.  Round-robin makes any
    # window budget cover all proteins x temperatures x replicas x horizons.
    ap.add_argument("--manifest-order", action="store_true",
                    help="Use the legacy manifest-order window walk (reproduces the "
                         "old single-protein numbers; do NOT use for reported scores).")
    ap.add_argument("--max-windows-per-protein", type=int, default=None,
                    help="Cap how many windows one protein contributes (round-robin "
                         "only). Use with a large --max-windows for wide coverage.")
    # Temperature conditioning. Default: use each window's OWN temperature.
    ap.add_argument("--fixed-temperature", type=float, default=None,
                    help="Force one temperature for every window (legacy behaviour was "
                         "the checkpoint's static temperature, which is wrong for 25%% "
                         "of the val windows).")
    ap.add_argument("--path-ensemble", action="store_true",
                    help="Score the pooled decoded path; report separately from endpoint ensembles.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.horizons_ns = [1.0]
        args.max_windows = 2
        args.n_samples = 2

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = _device(args.device)
    m1, hs, cond_dim, _ = build(args, dev)
    hs._eval_path_ensemble = args.path_ensemble
    # Legacy behaviour was the CHECKPOINT's static temperature, which silently
    # evaluates every 348/379/413/450 K window at 320 K.  Each window carries its
    # own temperature_k; use it unless the caller pins one explicitly.
    per_hz = defaultdict(lambda: defaultdict(list))
    # Per-window records for protein-clustered bootstrap: (protein, hz, metric, value)
    per_win: list[dict] = []
    n_win = 0
    Hs = tuple(max(1, int(round(ns))) for ns in args.horizons_ns)
    proto = EvalProtocol(manifest=args.manifest, horizons_frames=Hs,
                         context_frames=args.history_frames, context_stride=1,
                         frame_stride=1, max_residues=args.max_residues,
                         max_anchors_per_trajectory=4, anchor_stride=1, seed=args.seed,
                         round_robin_proteins=not args.manifest_order,
                         round_robin_conditions=not args.manifest_order,
                         max_windows_per_protein=args.max_windows_per_protein)
    for w in iter_windows(proto):
        if args.max_windows is not None and n_win >= args.max_windows:
            break
        rmask = w.context["residue_mask"].all(dim=0).bool()
        if int(rmask.sum()) < 4:
            continue
        fp = getattr(w, "future_path", None)
        if fp is None:
            continue
        true = fp["coords"][:, :, 1, :][:, rmask.cpu()].to(dev)     # [T,L,3]
        if true.shape[0] < 2:
            continue
        hz_ns = float(w.horizon)                                     # frames == ns
        tk = args.fixed_temperature if args.fixed_temperature is not None else float(w.temperature_k)
        hs._rollout_steps = int(w.horizon)
        m_e = score_hs(hs, w.context, rmask, args.n_samples, dev, tk)
        m1_e = score_m1(m1, w.context, rmask, args.n_samples, cond_dim, dev)
        anchor = w.context["coords"][-1, :, 1, :][rmask.cpu()].to(dev)
        pers = anchor[None].expand(true.shape[0], -1, -1)
        kw = dict(cutoff=args.cutoff, pca_k=args.pca_k)
        s_hs = extended_ensemble_summary(m_e.float(), true.float(), None, **kw)
        s_m1 = extended_ensemble_summary(m1_e.float(), true.float(), None, **kw)
        s_p = extended_ensemble_summary(pers.float(), true.float(), None, **kw)
        for m in METRICS:
            per_hz[hz_ns]["hs_" + m].append(float(s_hs[m]))
            per_hz[hz_ns]["m1_" + m].append(float(s_m1[m]))
            per_hz[hz_ns]["pers_" + m].append(float(s_p[m]))
        per_win.append({"protein": w.protein_id, "traj": w.trajectory_id, "anchor_frame": int(w.anchor_frame),
                        "temperature_k": float(w.temperature_k), "horizon": hz_ns,
                        "lasp": {m: float(s_hs[m]) for m in METRICS},
                        "model1": {m: float(s_m1[m]) for m in METRICS},
                        "persist": {m: float(s_p[m]) for m in METRICS},
                        # Contact-set sizes: a Jaccard of 1.0 with n_true == 0 is
                        # degenerate (the truth has NO fractional contacts at this
                        # horizon), not a win.  Recorded so the table can flag it.
                        "weak_n_true": float(s_hs.get("weak_n_true", float("nan"))),
                        "weak_n_pred": float(s_hs.get("weak_n_pred", float("nan"))),
                        "transient_n_true": float(s_hs.get("transient_n_true", float("nan"))),
                        "transient_n_pred": float(s_hs.get("transient_n_pred", float("nan")))})
        n_win += 1
        if n_win % 10 == 0:
            print(f"  scored {n_win} windows (last hz={hz_ns}, "
                  f"{len({r['protein'] for r in per_win})} proteins so far)", flush=True)

    prot = sorted({r["protein"] for r in per_win})
    temps = sorted({r["temperature_k"] for r in per_win})
    res = {"metric_version": "empirical_w2_v2_pca_projection",
           "lasp_sampling": "pooled_path" if args.path_ensemble else "endpoint",
           "kernel_ckpt": args.kernel_ckpt, "seed": args.seed,
           "n_windows": n_win, "n_samples": args.n_samples,
           "n_proteins": len(prot), "proteins": prot,
           "n_temperatures": len(temps), "temperatures": temps,
           "protocol": {"round_robin_proteins": not args.manifest_order,
                        "round_robin_conditions": not args.manifest_order,
                        "max_windows_per_protein": args.max_windows_per_protein,
                        "manifest": str(args.manifest),
                        "horizons_ns": list(map(float, args.horizons_ns)),
                        "ci": "protein-clustered bootstrap"},
           "by_horizon": {}}
    if len(prot) < 2:
        # Refuse to emit a "result" whose CI is a single domain's idiosyncrasy.
        print(f"\n*** WARNING: only {len(prot)} protein(s) scored ({prot or 'none'}). "
              f"Confidence intervals are reported as NaN because windows within a "
              f"protein are not independent. Use --max-windows-per-protein to widen "
              f"coverage, or drop --manifest-order.", flush=True)
    for hz_ns, d in sorted(per_hz.items()):
        rows = {}
        recs = [r for r in per_win if r["horizon"] == hz_ns]
        clus = np.array([r["protein"] for r in recs])
        print(f"\n===== horizon {hz_ns} ns  (H={int(hz_ns)} frames)  "
              f"n={len(recs)} windows over {len(set(clus))} proteins =====")
        print(f"{'metric':22}{'persist':>10}{'Model1':>10}{'Lasp':>10}"
              f"{'Δ(Lasp-M1)':>12}{'95% CI (clustered)':>22}{'better':>8}")
        for m in METRICS:
            a = np.array(d["m1_" + m], float); b = np.array(d["hs_" + m], float)
            p = np.array(d["pers_" + m], float)
            delta = b - a
            lo, hi = _cluster_boot_ci(delta, clus)
            hb = m in HIGHER_BETTER
            better = (delta.mean() > 0) if hb else (delta.mean() < 0)
            sig = bool(np.isfinite(lo) and np.isfinite(hi) and
                       ((lo > 0) if hb else (hi < 0)))
            rows[m] = dict(persist=float(np.nanmean(p)), model1=float(np.nanmean(a)),
                           lasp=float(np.nanmean(b)), delta=float(np.nanmean(delta)),
                           ci=[lo, hi], higher_better=hb, lasp_better=bool(better),
                           significant=sig, n_windows=len(recs),
                           n_proteins=int(len(set(clus))))
            # Degenerate-cell guard: with n_true == 0 the truth has no fractional
            # contacts, so BOTH real models score 0 and persistence scores 1 purely
            # by empty-set convention.  Flag it instead of reporting a "loss".
            key = "weak" if m == "weak_contact_j" else (
                "transient" if m == "transient_contacts_j" else None)
            if key is not None:
                ntr = float(np.nanmean([r[f"{key}_n_true"] for r in recs]))
                npr = float(np.nanmean([r[f"{key}_n_pred"] for r in recs]))
                rows[m]["mean_n_true"] = ntr
                rows[m]["mean_n_pred"] = npr
                rows[m]["degenerate"] = bool(ntr < 1.0)
                if ntr < 1.0:
                    rows[m]["lasp_better"] = False
                    rows[m]["significant"] = False
            flag = " [degenerate: n_true=0]" if rows[m].get("degenerate") else ""
            print(f"{m:22}{np.nanmean(p):>10.4f}{np.nanmean(a):>10.4f}{np.nanmean(b):>10.4f}"
                  f"{np.nanmean(delta):>+12.4f}[{lo:>+8.4f},{hi:>+8.4f}]"
                  f"{'YES' if rows[m]['lasp_better'] else 'no ':>8}"
                  f"{' *' if rows[m]['significant'] else ''}{flag}")
        res["by_horizon"][str(hz_ns)] = rows
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(res, indent=1))
        # Per-window rows next to the summary: lets any downstream table be
        # rebuilt with a different clustering (protein / temperature / horizon).
        pw = out.with_suffix(".per_window.json")
        pw.write_text(json.dumps(per_win))
        print(f"\nwrote {out}\nwrote {pw}")


if __name__ == "__main__":
    main()
