"""bioemu2 zero-shot dynamics baseline on the mdCATH ensemble protocol.

Purpose
-------
Measure the *out-of-the-box* (zero-shot, no fine-tuning) dynamics skill of the
Microsoft Research ``bioemu2`` ``bioemu2-preview`` checkpoint on exactly the
same windows / horizons / metrics that this repository uses for its own
from-none GLE models (``evaluation/run_eval_fromnone_ensemble.py``).

The comparison is apples-to-apples by construction:

* same manifest (default ``artifacts/mdcath_manifest_1000dom/val.jsonl``),
* same window iterator (``evaluation.data.iter_windows`` through ``EvalProtocol``),
* same true MD ensemble (``w.future_path`` future CA frames),
* same persistence baseline (anchor CA repeated T times),
* same metric code (``evaluation.ensemble_metrics_ext.extended_ensemble_summary``),
* same aggregation (mean over windows + bootstrap CI over windows).

What bioemu2 is asked to do (per window)
----------------------------------------
Given the *full-all-atom* anchor frame that ends the history window, draw N
independent ``lag_time_ns``-forward endpoints with the propagation (dynamics)
model:

    bioemu2.propagate(system, model=model, x0=anchor_allatom,
                      lag_time_ns=horizon_ns, num_samples=N)

Each endpoint is rigidly superposed on its own input frame by bioemu2 itself,
which matches the ``align_ensemble`` convention used by our metrics.  We then
extract CA and evaluate the six distributional metrics against the real MD
future frames.

The MSA is computed once per *sequence* and reused for every window of that
protein/temperature (via ``get_msa`` + ``model.embed`` + ``propagate_with_embeddings``).

Caveats (documented, not hidden)
--------------------------------
* bioemu2 predicts an all-atom structure; only CA is compared here, to stay on
  the identical metric footing as the from-none models.
* bioemu2 is conditioned on a single anchor structure, exactly like the
  from-none models' anchor, but it does NOT see the K-frame history.
* ``lag_time_ns`` beyond the training range (1 ns .. 1 us) degrades conditioning.

Usage
-----
  python -m evaluation.baselines.run_bioemu2_zeroshot \\
      --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \\
      --horizons-ns 1 4 16 64 --n-samples 12 --max-windows 200 \\
      --device cuda --out results/bioemu2_zeroshot.json

  # smoke test (2 windows, 2 samples, 1 ns only):
  python -m evaluation.baselines.run_bioemu2_zeroshot --max-anchors-per-trajectory 2 \\
      --max-proteins 1 --max-windows 2 --n-samples 2 --horizons-ns 1 --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics_ext import extended_ensemble_summary

METRICS = ["rmsf_corr", "rmsf_mae", "root_mean_w2", "variance_w2", "md_pca_w2",
           "weak_contact_j", "transient_contacts_j"]
METRIC_DIR = {
    "rmsf_corr": True,
    "rmsf_mae": False, "root_mean_w2": False, "variance_w2": False,
    "md_pca_w2": False, "weak_contact_j": True, "transient_contacts_j": True,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    p.add_argument("--horizons-ns", type=float, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--n-samples", type=int, default=12,
                   help="bioemu2 endpoints per window (ensemble size).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Endpoints per denoiser call (default: all n_samples at once).")
    p.add_argument("--num-steps", type=int, default=128,
                   help="Denoising steps per endpoint (bioemu2 default is 200; "
                        "128 is used here to keep the 200-window sweep tractable).")
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument("--max-anchors-per-trajectory", type=int, default=4)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--temperatures", type=float, nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sampler", default="edm",
                   choices=["edm", "edm_gd", "edm_fk", "edm_fkc"])
    p.add_argument("--checkpoint", default="bioemu2-preview")
    p.add_argument("--checkpoint-path", default=None,
                   help="Local checkpoint .ckpt (config.yaml must sit beside it).")
    p.add_argument("--cache-dir", default=os.environ.get("BIOEMU2_CACHE_DIR"),
                   help="Hugging Face cache directory holding the checkpoint.")
    p.add_argument("--msa-cache", default=None,
                   help="BIOEMU2_MSA_CACHE override (ColabFold A3M cache dir).")
    p.add_argument("--seed", type=int, default=0,
                   help="Run index. Distinct values give independent sampling "
                        "replicates; the same value reproduces a run exactly "
                        "(with --deterministic).")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Force deterministic cuDNN/matmul kernels. Required for "
                        "bit-identical reruns of the same --seed.")
    p.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                   help="Disable deterministic kernels (faster, not reproducible).")
    p.add_argument("--precision", default="32-true",
                   choices=["32-true", "bf16-mixed"],
                   help="Inference precision. '32-true' (default here) is required "
                        "for bit-identical reruns; bioemu2's own default is "
                        "'bf16-mixed', which is NOT reproducible.")
    p.add_argument("--matmul-precision", default="highest",
                   choices=["highest", "high", "medium"],
                   help="torch float32 matmul precision. 'highest' avoids TF32 "
                        "non-determinism (bioemu2's own default is 'high').")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--pca-k", type=int, default=3)
    p.add_argument("--weak-lo", type=float, default=0.05)
    p.add_argument("--weak-hi", type=float, default=0.50)
    p.add_argument("--trans-lo", type=float, default=0.01)
    p.add_argument("--trans-hi", type=float, default=0.99)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# metrics helpers
# --------------------------------------------------------------------------- #
def _bootstrap_ci(deltas, n_boot=2000, alpha=0.05, seed=0):
    if not deltas:
        return [float("nan"), float("nan")]
    t = torch.tensor(deltas, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    N = t.numel()
    means = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(0, N, (N,), generator=g)
        means[b] = t[idx].mean()
    return [torch.quantile(means, alpha / 2).item(),
            torch.quantile(means, 1 - alpha / 2).item()]


def _ensure_hash_seed() -> None:
    """Re-exec with a fixed PYTHONHASHSEED.

    PYTHONHASHSEED must be set before the interpreter starts, and Python's
    per-process string-hash randomization otherwise leaks into bioemu2, which
    uses ``hash()`` / set / dict iteration in several code paths (bond
    bookkeeping, physicality counters).  Empirically that is the difference
    between bit-identical and non-identical reruns of the *same* seed, so we
    re-exec ourselves once with the variable pinned.
    """
    if os.environ.get("_BIOEMU2_REEXEC") == "1":
        return
    if os.environ.get("PYTHONHASHSEED") == "0":
        return
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["_BIOEMU2_REEXEC"] = "1"
    print("[bioemu2] re-exec with PYTHONHASHSEED=0 for reproducibility", flush=True)
    os.execve(sys.executable, [sys.executable, "-m",
                               "evaluation.baselines.run_bioemu2_zeroshot"] + sys.argv[1:], env)


def main(argv=None):
    args = parse_args(argv)
    _ensure_hash_seed()
    if args.smoke:
        args.horizons_ns = [1.0]
        args.n_samples = 2
        args.max_windows = args.max_windows or 2
        args.max_proteins = args.max_proteins or 1
        args.max_anchors_per_trajectory = 2

    if args.msa_cache:
        os.environ["BIOEMU2_MSA_CACHE"] = args.msa_cache

    # ------------------------------------------------------------------ #
    # Reproducibility (see BIOEMU2_ZEROSHOT_BASELINE_*.md):
    # bioemu2's propagate path draws EDM initial noise and per-step random
    # augmentations from the *global* torch RNG, and its own seed is not
    # forwarded through propagate_with_embeddings.  Two things are needed
    # for bit-identical reruns:
    #   1. deterministic cuDNN kernels (cuDNN reduction order otherwise
    #      varies between runs), and
    #   2. explicit per-(window, horizon, run) seeding of the global RNG.
    # ------------------------------------------------------------------ #
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[bioemu2] deterministic mode ON (cuDNN deterministic + "
              "use_deterministic_algorithms)", flush=True)

    dev = args.device
    print(f"[bioemu2] device={dev} horizons={args.horizons_ns} ns "
          f"n_samples={args.n_samples} manifest={args.manifest}", flush=True)

    from bioemu2 import load_model, get_msa
    from bioemu2.propagation import propagate_with_embeddings
    from bioemu2.sampling import sampler_from_preset

    model = load_model(
        device=dev,
        checkpoint=None if args.checkpoint_path else args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        cache_dir=args.cache_dir,
        precision=args.precision,
        matmul_precision=args.matmul_precision,
    )
    print(f"[bioemu2] checkpoint loaded; supports_propagation={model.supports_propagation}",
          flush=True)

    proto = EvalProtocol(
        manifest=args.manifest,
        horizons_frames=tuple(max(1, int(round(ns))) for ns in args.horizons_ns),
        context_frames=args.history_frames,
        context_stride=1, frame_stride=1,
        max_residues=args.max_residues,
        max_proteins=args.max_proteins,
        temperatures=tuple(args.temperatures) if args.temperatures else None,
        max_anchors_per_trajectory=args.max_anchors_per_trajectory,
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False,
        seed=args.seed,
    )

    per_hz = defaultdict(lambda: defaultdict(list))
    n_windows = 0
    n_skipped = 0
    # cache bioemu2 embeddings per (protein, temperature) key
    embed_cache: dict[str, object] = {}

    t_start = time.time()
    for w in iter_windows(proto):
        if args.max_windows is not None and n_windows >= args.max_windows:
            break
        rmask = w.context["residue_mask"].all(dim=0).bool()
        if rmask.sum() < 4:
            n_skipped += 1
            continue
        fp = getattr(w, "future_path", None)
        if fp is None:
            continue
        true_ca_path = fp["coords"][:, :, 1, :]              # [T, L, 3]
        L_val = int(rmask.sum())
        true_ens = true_ca_path[:, rmask.cpu()].to(dev)      # [T, L_val, 3]
        if true_ens.shape[0] < 2:
            continue

        # --- anchor frame -> bioemu2 all-atom x0 ---
        anchor_ca = w.context["coords"][-1, :, 1, :][rmask.cpu()]        # [L_val,3] A

        # bioemu2 is conditioned on the anchor only (no K-frame history), so all
        # horizons share one system/embedding per (protein, temperature).
        key = f"{w.protein_id}_T{int(w.temperature_k)}"
        try:
            cached = embed_cache.get(key)
            if cached is None:
                from evaluation.baselines.bioemu2_io import (
                    make_system, ca_indices)
                system = make_system(w, rmask)
                msa = get_msa(system)
                trunk = model.embed(system, msa=msa, mode="equilibrium")
                ca_idx = ca_indices(system)
                seq_len = system.topology.n_residues
                if seq_len != L_val:
                    raise ValueError(f"sequence length {seq_len} != evaluated residues {L_val}")
                embed_cache[key] = (system, trunk, ca_idx)
                cached = embed_cache[key]
                print(f"  [embed] {key} L={L_val} len(MSA)={len(msa.a3m())}", flush=True)
            system, trunk, ca_idx = cached

            from evaluation.baselines.bioemu2_io import window_anchor_x0
            x0 = window_anchor_x0(system, w, rmask)

            # N independent endpoints from the (identical) anchor frame, drawn in
            # one denoiser call: this is exactly what bioemu2's `propagate(...,
            # num_samples=N)` does, but we keep the embeddings for reuse.
            #
            # REPRODUCIBILITY: bioemu2's propagate_with_embeddings() calls
            # generate_samples() WITHOUT forwarding its seed, so rng_seed(None)
            # leaves the global RNG untouched and the EDM initial noise
            # (init_sigma * torch.randn) differs on every run.  We therefore
            # seed the global CPU/CUDA RNG ourselves, deterministically per
            # (window, horizon, run), and restore state afterwards.
            sampler = sampler_from_preset(args.sampler, mode="equilibrium",
                                          num_steps=args.num_steps)
            # Seed from the window's *identity* (not its loop position), so a
            # replicate has a reproducible seed for every window, and reordering
            # the manifest cannot silently change what was simulated.
            win_key = (f"{w.protein_id}|{w.trajectory_id}|{w.anchor_frame}|"
                       f"{w.horizon}")
            win_hash = int(hashlib.sha256(win_key.encode()).hexdigest()[:12], 16)
            win_seed = (int(args.seed) * 1_000_003 + win_hash) % (2 ** 31 - 1)
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(win_seed)
                torch.cuda.manual_seed_all(win_seed)
                traj = propagate_with_embeddings(
                    system, trunk,
                    x0=x0[[0] * args.n_samples],
                    lag_time_ns=float(w.horizon),   # mdCATH frame_interval = 1 ns
                    model=model,
                    sampler=sampler,
                    batch_size=args.batch_size,
                    seed=None,      # seeding is handled explicitly above
                    progress=False,
                )
            pred_ens = torch.as_tensor(traj.xyz[:, ca_idx] * 10.0,
                                       dtype=torch.float32, device=dev)  # [N, L_val, 3] A
        except Exception as e:  # noqa: BLE001  -- a baseline must not crash the sweep
            print(f"  skip {w.protein_id} h={w.horizon}: {type(e).__name__}: {e}",
                  flush=True)
            n_skipped += 1
            continue

        pers_ens = anchor_ca.to(dev).unsqueeze(0).expand(true_ens.shape[0], -1, -1)

        s_model = extended_ensemble_summary(
            pred_ens.float(), true_ens.float(), None,
            cutoff=args.cutoff, pca_k=args.pca_k,
            weak_lo=args.weak_lo, weak_hi=args.weak_hi,
            trans_lo=args.trans_lo, trans_hi=args.trans_hi,
        )
        s_pers = extended_ensemble_summary(
            pers_ens.float(), true_ens.float(), None,
            cutoff=args.cutoff, pca_k=args.pca_k,
            weak_lo=args.weak_lo, weak_hi=args.weak_hi,
            trans_lo=args.trans_lo, trans_hi=args.trans_hi,
        )
        hz = w.horizon
        for m in METRICS:
            per_hz[hz]["model_" + m].append(s_model[m])
            per_hz[hz]["pers_" + m].append(s_pers[m])

        n_windows += 1
        if n_windows % 10 == 0:
            print(f"  processed {n_windows} windows  ({time.time()-t_start:.0f}s)",
                  flush=True)

    print(f"[bioemu2] windows={n_windows} skipped={n_skipped}", flush=True)

    result = {
        "baseline": "bioemu2",
        "checkpoint": args.checkpoint_path or args.checkpoint,
        "manifest": args.manifest,
        "n_samples": args.n_samples, "n_windows": n_windows,
        "horizons_ns": args.horizons_ns,
        "sampler": args.sampler,
        "by_horizon": {},
    }
    rows = []
    for hz in sorted(per_hz):
        hz_ns = float(hz)
        d = per_hz[hz]
        hz_res = {"horizon_ns": hz_ns, "n_windows": len(d["model_" + METRICS[0]]),
                  "metrics": {}}
        for m in METRICS:
            mv, pv = d["model_" + m], d["pers_" + m]
            up = METRIC_DIR[m]
            deltas = [(a - b if up else b - a) for a, b in zip(mv, pv)]
            dmean = float(np.nanmean(deltas))
            hz_res["metrics"][m] = {
                "model": float(np.nanmean(mv)),
                "persistence": float(np.nanmean(pv)),
                "improvement": dmean,
                "ci95": _bootstrap_ci([x for x in deltas if not math.isnan(x)]),
                "model_better": bool(dmean > 0),
                "win_rate": float(np.nanmean([1.0 if x > 0 else 0.0 for x in deltas])),
            }
            rows.append((hz_ns, m, float(np.nanmean(mv)), float(np.nanmean(pv)),
                         dmean, hz_res["metrics"][m]["win_rate"]))
        result["by_horizon"][str(hz)] = hz_res

    print()
    print(f"{'horizon':>8}  {'metric':>22}  {'model':>9}  {'persist':>9}  {'impr':>9}  {'win':>5}  ok")
    print("-" * 80)
    for hz_ns, m, mv, pv, imp, wr in rows:
        print(f"{hz_ns:>7.1f}  {m:>22}  {mv:>9.4f}  {pv:>9.4f}  {imp:>+9.4f}  "
              f"{wr*100:>4.0f}%  {'OK' if imp > 0 else 'X'}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\n[bioemu2] wrote {args.out}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
