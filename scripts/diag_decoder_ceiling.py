"""Is the bottleneck the H->CA decoder, the GLE rollout, or the encoder?

The mean-collapse hypothesis is testable directly: take the TRUE future frames,
push them through the frozen encoder and then the decoder, and see how much of
the true conformational variation survives the round trip.  Compare that with

  * true  : per-residue RMSF of the ground-truth future window;
  * enc->dec: same, after Enc(true X_t) -> Dec -> X_t (the ceiling);
  * GLE   : same, after the deployed rollout (what the benchmark actually scores).

If enc->dec is already below GLE, the decoder is throwing away information and no
amount of kernel training can recover it.  RMSF-r (Pearson correlation of the
per-residue RMSF profile) is the metric bioemu2 wins by the largest margin, so it
is reported alongside the magnitude ratio.

Usage:
  python scripts/diag_decoder_ceiling.py \
    --kernel-ckpt artifacts/lasp_mh_1000dom/checkpoint_best.pt --max-windows 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.data import EvalProtocol, iter_windows
from evaluation.ensemble_metrics import align_ensemble
from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
from protein_jepa_frame.training.train_utils import _device
from scripts.eval_lasp_final import build


def _rmsf_profile(x, mask):
    a = align_ensemble(x.float(), mask)
    return (a - a.mean(0)).pow(2).mean(0).sum(-1).sqrt()


def _pearson(a, b, mask):
    a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-ckpt", required=True)
    ap.add_argument("--ensemble-ckpt", default="artifacts/stage_ensemble_1000dom/checkpoint_best.pt")
    ap.add_argument("--stage-a-ckpt", default="artifacts/stage_a_disp_relpos/checkpoint_latest.pt")
    ap.add_argument("--b1a-ckpt",
                    default="artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt")
    ap.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--horizons-ns", type=float, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--history-frames", type=int, default=8)
    ap.add_argument("--max-residues", type=int, default=256)
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--manifest-order", action="store_true",
                    help="legacy single-trajectory walk (do NOT use: one protein)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = _device(args.device)
    _, hs, _, _ = build(args, dev)

    proto = EvalProtocol(
        manifest=args.manifest,
        horizons_frames=tuple(max(1, int(round(n))) for n in args.horizons_ns),
        context_frames=args.history_frames, context_stride=1, frame_stride=1,
        max_residues=args.max_residues, max_anchors_per_trajectory=4,
        anchor_stride=1, seed=args.seed,
        round_robin_proteins=not args.manifest_order,
        round_robin_conditions=not args.manifest_order,
    )

    rows = []
    for w in iter_windows(proto):
        if len(rows) >= args.max_windows:
            break
        rm = w.context["residue_mask"].all(dim=0).bool()
        if int(rm.sum()) < 4:
            continue
        m = rm.to(dev)
        coords = w.context["coords"].to(dev).unsqueeze(0)          # [1,K,L,4,3]
        K, L = coords.shape[1], coords.shape[2]
        rmK = m.unsqueeze(0).unsqueeze(1).expand(1, K, L)
        aat = w.context["aatype"].to(dev).unsqueeze(0)
        ridx = w.context["residue_index"].to(dev).unsqueeze(0)
        cid = w.context["chain_id"].to(dev).unsqueeze(0)
        R, t = _extract_backbone_frames(coords[:, -1])

        full = w.future_path["coords"].to(dev)[:, m]                # [T,L,4,3] TRUE
        if full.shape[0] < 2:
            continue
        T_ = full.shape[0]
        with torch.no_grad():
            H = hs.encoder(full.unsqueeze(1).float(),
                           m.unsqueeze(0).unsqueeze(1).expand(T_, 1, -1),
                           aat[0].unsqueeze(0).expand(T_, -1),
                           ridx[0].unsqueeze(0).expand(T_, -1),
                           cid[0].unsqueeze(0).expand(T_, -1),
                           token_mask=m.unsqueeze(0).unsqueeze(1).expand(T_, 1, -1))
            Ht = H[:, 0]                                            # [T,L,dH]
            dec = hs.decoder.decode_structure(
                Ht, aat[0].unsqueeze(0).expand(T_, -1),
                R[0].unsqueeze(0).expand(T_, -1, -1, -1),
                t[0].unsqueeze(0).expand(T_, -1, -1),
                m.unsqueeze(0).expand(T_, -1))
            ca_rt = dec["backbone_coords"][:, :, 1, :]              # [T,L,3]
            hs._rollout_steps = int(w.horizon)
            gle = hs.sample_future_ca(coords, m.unsqueeze(0), aat, ridx, cid, R, t,
                                      n_samples=args.n_samples,
                                      temperature_k=float(w.temperature_k))
            ca_gle = gle["ca_samples"][0].float()
            lat = Ht.reshape(T_, -1)
            lat_spread = float(torch.cdist(lat, lat)[torch.triu_indices(T_, T_, 1)].mean())

        prof_t = _rmsf_profile(full[:, :, 1, :], m)
        prof_e = _rmsf_profile(ca_rt, m)
        prof_g = _rmsf_profile(ca_gle, m)
        rows.append(dict(horizon=float(w.horizon), protein=w.protein_id,
                         temperature_k=float(w.temperature_k),
                         true_rmsf=float(prof_t[m].mean()),
                         encdec_rmsf=float(prof_e[m].mean()),
                         gle_rmsf=float(prof_g[m].mean()),
                         encdec_rmsf_r=_pearson(prof_t, prof_e, m),
                         gle_rmsf_r=_pearson(prof_t, prof_g, m),
                         latent_spread=lat_spread))

    if not rows:
        print("no windows scored")
        return
    print(f"\nn={len(rows)} windows over {len({r['protein'] for r in rows})} proteins")
    print(f"{'H':>4}{'trueRMSF':>10}{'enc>dec':>9}{'GLE':>8}"
          f"{'r(enc>dec)':>12}{'r(GLE)':>8}{'latent':>8}")
    summary = {}
    for h in sorted({r["horizon"] for r in rows}):
        g = [r for r in rows if r["horizon"] == h]
        rec = {k: float(np.mean([r[k] for r in g]))
               for k in ("true_rmsf", "encdec_rmsf", "gle_rmsf",
                         "encdec_rmsf_r", "gle_rmsf_r", "latent_spread")}
        rec["n_windows"] = len(g)
        rec["encdec_captures"] = rec["encdec_rmsf"] / max(rec["true_rmsf"], 1e-9)
        summary[str(h)] = rec
        print(f"{h:>4.0f}{rec['true_rmsf']:>10.2f}{rec['encdec_rmsf']:>9.2f}"
              f"{rec['gle_rmsf']:>8.2f}{rec['encdec_rmsf_r']:>12.2f}"
              f"{rec['gle_rmsf_r']:>8.2f}{rec['latent_spread']:>8.1f}")
        if rec["encdec_rmsf_r"] < rec["gle_rmsf_r"]:
            print(f"      ^ DECODER CEILING: enc->dec (r={rec['encdec_rmsf_r']:.2f}) is "
                  f"worse than the rollout it renders (r={rec['gle_rmsf_r']:.2f}); "
                  f"train with --roundtrip-weight.")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n_windows": len(rows), "by_horizon": summary, "windows": rows}, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
