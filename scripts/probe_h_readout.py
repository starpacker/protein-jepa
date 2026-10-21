#!/usr/bin/env python3
"""Independent H readout probe (representation acceptance test).

Measures how well H_t linearly/weakly decodes to the future internal CA
displacement (the SE(3)-quotient quantity the decoder must reproduce), and
compares against persistence.  This isolates the *representation* question from
the project decoder: it uses a fresh probe head, not B1a's decoder.

Acceptance target: within-protein held-out true-H error <= 0.5x persistence.

Usage:
  python scripts/probe_h_readout.py --stage-a-ckpt <ckpt.pt> --manifest <val.jsonl> \
     --npz-dir <dir with <protein>.npz> --max-proteins 16
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
from protein_jepa_frame.models.stage_a_v2.model import deterministic_module


def kabsch_align(P, Q):
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return Pc @ R + Q.mean(0)


def shape_rmsd(p, q):
    return float(np.sqrt(((kabsch_align(p, q) - q) ** 2).sum(1).mean()))


def load_encoder(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = StageAV2Config(**ck["stage_a_v2_config"])
    model = StageAV2(cfg)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return cfg, model.context_encoder, int(ck.get("step", -1))


def encode_protein(encoder, coords, rmask, aatype, ridx, cid, Lmax, device="cpu"):
    F, L = coords.shape[:2]
    L = min(L, Lmax)
    cp = coords[:, :L]; rm = rmask[:, :L]
    aa = torch.tensor(aatype[:L])[None].to(device); rx = torch.tensor(ridx[:L])[None].to(device); ch = torch.tensor(cid[:L])[None].to(device)
    with torch.no_grad(), deterministic_module(encoder):
        H = encoder(torch.tensor(cp)[:, None].to(device), torch.tensor(rm)[:, None].to(device),
                    aa, rx, ch, token_mask=torch.tensor(rm)[:, None].to(device))[:, 0].float().cpu().numpy()
    return H, cp, rm, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-ckpt", required=True)
    ap.add_argument("--npz-glob", required=True, help="glob of npz files for the probe proteins")
    ap.add_argument("--max-proteins", type=int, default=16)
    ap.add_argument("--max-residues", type=int, default=128)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ridge-lambda", type=float, default=10.0)
    ap.add_argument("--no-cross", action="store_true", help="skip the (expensive) cross-protein branch")
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg, enc, step = load_encoder(args.stage_a_ckpt)
    enc = enc.to(args.device)
    files = sorted(glob.glob(args.npz_glob))[: args.max_proteins]
    if not files:
        raise SystemExit(f"no npz files matched {args.npz_glob}")

    data = []
    for f in files:
        d = np.load(f)
        H, cp, rm, L = encode_protein(enc, d["coords"].astype(np.float32),
                                      d["residue_mask"].astype(bool), d["aatype"],
                                      d["residue_index"], d["chain_id"], args.max_residues,
                                      device=args.device)
        data.append({"H": H, "cp": cp, "rm": rm, "L": L, "name": Path(f).stem})

    results = {}
    for h in args.horizons:
        within_probe, within_persist = [], []
        cross_probe, cross_persist = [], []
        for i, d in enumerate(data):
            H, cp, rm, L = d["H"], d["cp"], d["rm"], d["L"]
            ca = cp[:, :, 1, :]
            F = H.shape[0]
            idx = list(range(0, F - h, 3))
            if len(idx) < 6:
                continue
            n = int(len(idx) * 0.6)
            def build(ixs):
                X, Y = [], []
                for t in ixs:
                    a = ca[t]; tg = kabsch_align(ca[t + h], a)
                    X.append(H[t][:L].reshape(-1)); Y.append((tg[:L] - a[:L]).reshape(-1))
                return np.array(X), np.array(Y)
            Xtr, Ytr = build(idx[:n]); Xte, Yte = build(idx[n:])
            if len(Xtr) < 3 or len(Xte) < 2:
                continue
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
            Xtr_n = (Xtr - mu) / sd; Xte_n = (Xte - mu) / sd
            # Solve ridge via the (small) sample-space dual: W = X^T (K + l I)^-1 Y.
            K = Xtr_n @ Xtr_n.T
            A = np.linalg.solve(K + args.ridge_lambda * np.eye(K.shape[0]), Ytr)
            P = (Xte_n @ (Xtr_n.T @ A)).reshape(-1, L, 3); Q = Yte.reshape(-1, L, 3)
            within_probe.append(np.mean([shape_rmsd(P[j], Q[j]) for j in range(len(P))]))
            within_persist.append(np.mean([np.sqrt((Q[j] ** 2).sum(1).mean()) for j in range(len(Q))]))

            # cross-protein (train on all *same-length* others; feature layout is
            # tied to L, so mixing different lengths is not well-defined)
            Xo, Yo = [], []
            if not args.no_cross:
                for k, d2 in enumerate(data):
                    if k == i:
                        continue
                    H2, cp2, rm2, L2 = d2["H"], d2["cp"], d2["rm"], d2["L"]
                    if L2 != L:
                        continue
                    ca2 = cp2[:, :, 1, :]
                    for t in range(0, H2.shape[0] - h, 5):
                        a = ca2[t]; tg = kabsch_align(ca2[t + h], a)
                        Xo.append(H2[t][:L2].reshape(-1)); Yo.append((tg[:L2] - a[:L2]).reshape(-1))
            if len(Xo) > 5 and len(set(len(x) for x in Xo)) == 1:
                Xo_a = np.array(Xo)
                mu2 = Xo_a.mean(0); sd2 = Xo_a.std(0) + 1e-6
                Xo_n = (Xo_a - mu2) / sd2; Xe_n = (Xte - mu2) / sd2
                Yo_a = np.array(Yo)
                K2 = Xo_n @ Xo_n.T
                A2 = np.linalg.solve(K2 + 10.0 * np.eye(K2.shape[0]), Yo_a)
                P2 = (Xe_n @ (Xo_n.T @ A2)).reshape(-1, L, 3)
                cross_probe.append(np.mean([shape_rmsd(P2[j], Q[j]) for j in range(len(P2))]))
                cross_persist.append(within_persist[-1])
        results[h] = dict(
            within_probe=float(np.mean(within_probe)), within_persist=float(np.mean(within_persist)),
            within_ratio=float(np.mean(within_probe) / max(np.mean(within_persist), 1e-6)),
            cross_probe=float(np.mean(cross_probe)) if cross_probe else None,
            cross_persist=float(np.mean(cross_persist)) if cross_persist else None,
            cross_ratio=float(np.mean(cross_probe) / max(np.mean(cross_persist), 1e-6)) if cross_probe else None,
        )
        r = results[h]
        print(f"h={h}ns  within: probe {r['within_probe']:.3f} persist {r['within_persist']:.3f} "
              f"ratio {r['within_ratio']:.3f}   cross: "
              + (f"probe {r['cross_probe']:.3f} persist {r['cross_persist']:.3f} ratio {r['cross_ratio']:.3f}"
                 if r['cross_probe'] else "n/a"))

    out = dict(stage_a_ckpt=args.stage_a_ckpt, step=step, n_proteins=len(data), results=results,
               acceptance="within-protein held-out true-H <= 0.5x persistence")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
