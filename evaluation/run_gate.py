"""Stage-B1 §2.7 gate: held-out Kalman-innovation NLL ablations.

This is the likelihood-side companion to the CA-RMSD harness.  It is *not* a
CA-RMSD metric and does not import geometry.py's Kabsch (no coordinates decoded).

Reports, per residue·dim (lower is better):
  full     - trained model (bath + history posterior)
  nohist   - history posterior forced to N(0,I)  (same weights)
  nobath   - bath disabled -> diagonal Gaussian path (same D)
  persist  - H_{t+1}=H_t diagonal Gaussian with the model's R = 2hD
plus paired bootstrap CIs and whitened-innovation diagnostics.

Usage:
  python -m evaluation.run_gate --ckpt artifacts/abc_v3/stage_b1b_mdcath/checkpoint_latest.pt \
      --b1a-ckpt artifacts/abc_v3/stage_b1a_fix4_unfrozen/checkpoint_latest.pt \
      --manifest artifacts/mdcath_manifest_200dom/val.jsonl --n-prot 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.data import EvalProtocol, iter_windows
from evaluation.loaders import build_bundle
from evaluation.metrics import paired_bootstrap_ci


def persist_nll(H_seq, rmask, D, h):
    N = H_seq.shape[1] - 1
    d = H_seq.shape[-1]
    R = (2.0 * h * D).clamp_min(1e-6)
    tot, cnt = 0.0, 0
    for n in range(N):
        e = H_seq[:, n + 1] - H_seq[:, n]
        nll = 0.5 * (e * e / R + torch.log(R) + math.log(2 * math.pi)).sum(-1)
        tot += float((nll * rmask.float()).sum())
        cnt += int(rmask.sum()) * d
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="B1b checkpoint")
    ap.add_argument("--b1a-ckpt", required=True, help="B1a checkpoint (encoder + normalizer)")
    ap.add_argument("--manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--n-prot", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_bundle(variant="h_plus_s", device=dev,
                          b1a_ckpt=args.b1a_ckpt, b1b_ckpt=args.ckpt)
    model = bundle.b1b
    b_cfg = bundle.b1b_cfg

    proto = EvalProtocol(manifest=args.manifest, horizons_frames=(args.horizon,),
                         context_frames=4, max_proteins=args.n_prot)
    rows = {"full": [], "nohist": [], "nobath": [], "persist": []}
    innov = []
    h = b_cfg.step_ps

    for w in iter_windows(proto):
        ctx = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in w.context.items()}
        tgt = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else v) for k, v in w.target.items()}
        rmask = ctx["residue_mask"][:, -1].bool()
        aat = ctx["aatype"]
        with torch.no_grad():
            H_hist = bundle.encode(ctx["coords"], ctx["residue_mask"], aat,
                                   ctx["residue_index"], ctx["chain_id"])
            H_fut = bundle.encode(tgt["coords"], tgt["residue_mask"], tgt["aatype"],
                                  tgt["residue_index"], tgt["chain_id"])
            H_seq = torch.cat([H_hist[:, -1:], H_fut], dim=1)
            K = H_hist.shape[1]
            dt_hist = torch.full((1, K), h, device=dev)
            T = ctx.get("temperature_k")
            if T is not None:
                T = T.to(dev)
            d = H_seq.shape[-1]
            n_dim = int(rmask.sum()) * args.horizon * d
            if n_dim == 0:
                continue

            nll, diag = model.kalman_nll(H_hist, dt_hist, H_seq, rmask, T, return_diagnostics=True)
            if "innov_whitened" in diag:
                innov.append(diag["innov_whitened"].cpu())
            rows["full"].append(float(nll) / n_dim)

            orig = model.history_posterior.forward
            def zero_hp(*a, **kw):
                B_, K_, L_, _ = a[0].shape
                dS = b_cfg.bath_dim
                return (torch.zeros(B_, L_, dS, device=dev, dtype=a[0].dtype),
                        torch.eye(dS, device=dev, dtype=a[0].dtype).expand(B_, L_, dS, dS).contiguous())
            model.history_posterior.forward = zero_hp
            rows["nohist"].append(float(model.kalman_nll(H_hist, dt_hist, H_seq, rmask, T)) / n_dim)
            model.history_posterior.forward = orig

            old_bd = model.cfg.bath_dim
            model.cfg.bath_dim = 0
            rows["nobath"].append(float(model.kalman_nll(H_hist, dt_hist, H_seq, rmask, T)) / n_dim)
            model.cfg.bath_dim = old_bd

            D, _, _ = model.bath(H_seq[:, 0], T)
            rows["persist"].append(persist_nll(H_seq, rmask, D, h))

    print("\n=== §2.7 gate — held-out NLL per residue·dim (lower better) ===")
    for k in ["full", "persist", "nohist", "nobath"]:
        v = np.array(rows[k])
        print(f"{k:9s} {v.mean():8.4f} ± {v.std()/math.sqrt(max(len(v),1)):.4f} (n={len(v)})")
    print("\nPaired differences (95% bootstrap CI):")
    for name, other in [("full - persist", "persist"), ("full - nohist", "nohist"),
                        ("full - nobath", "nobath")]:
        m, lo, hi = paired_bootstrap_ci(rows["full"], rows[other])
        print(f"  {name:16s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIGNIFICANT' if (lo > 0 or hi < 0) else 'not significant'}")

    outp = args.out or os.path.join(os.path.dirname(args.ckpt), "eval_gate.json")
    json.dump({k: [float(x) for x in v] for k, v in rows.items()}, open(outp, "w"), indent=2)
    print(f"\nsaved {outp}")


if __name__ == "__main__":
    main()
