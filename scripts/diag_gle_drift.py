"""Does the GLE rollout leave the encoder's manifold?

The multi-protein benchmark shows the rollout losing to a STATIC Gaussian head at
every horizon, and the ceiling diagnostic shows it over-dispersing (RMSF 1.17 vs
a true 0.82 at 1 ns) while correlating poorly.  A single mechanism explains both:
the kernel drifts H away from the manifold the frozen decoder was trained on, so
the decoder faithfully renders a state that never occurs in MD.

This measures that directly.  For each window we have, in the same latent space:

    H_anchor      = Enc(ctx[-1])                 (on-manifold by construction)
    H_true(t)     = Enc(true frame t)            (on-manifold, the target)
    H_roll(t)     = GLE rollout step t            (what the decoder is fed)

and report, per horizon, the ratio

    drift = ||H_roll(t) - H_true(t)|| / ||H_true(t) - H_anchor||

A value > 1 means the rollout overshoots the true conformational change; if drift
also grows without a corresponding gain in decoded RMSF, the extra latent motion
is not conformational, it is noise the decoder cannot express.

Usage:
  python scripts/diag_gle_drift.py --kernel-ckpt artifacts/lasp_mh_rc_1000dom/checkpoint_best.pt
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
from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
from protein_jepa_frame.training.train_utils import _device
from scripts.eval_lasp_final import build


@torch.no_grad()
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
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--manifest-order", action="store_true")
    ap.add_argument("--fixed-temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    # None = report only.  Set e.g. 1.5 to make this a training gate: exit
    # non-zero if any horizon's drift/true-step exceeds the tolerance.
    ap.add_argument("--max-drift", type=float, default=None,
                    help="Fail (exit 1) if any horizon's drift/true-step exceeds this.")
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
        round_robin_conditions=not args.manifest_order)

    rows = []
    for w in iter_windows(proto):
        if len(rows) >= args.max_windows:
            break
        rm = w.context["residue_mask"].all(dim=0).bool()
        if int(rm.sum()) < 4:
            continue
        m = rm.to(dev)
        ctx = w.context["coords"].to(dev).unsqueeze(0)
        K, L = ctx.shape[1], ctx.shape[2]
        aat = w.context["aatype"].to(dev).unsqueeze(0)
        ridx = w.context["residue_index"].to(dev).unsqueeze(0)
        cid = w.context["chain_id"].to(dev).unsqueeze(0)
        R, t = _extract_backbone_frames(ctx[:, -1])
        full = w.future_path["coords"].to(dev)[:, m]
        T_ = full.shape[0]
        if T_ < 2:
            continue

        H_anchor = hs.encode_frame(ctx[:, -1], m.unsqueeze(0), aat, ridx, cid)[0]  # [L,dH]
        H_true = hs.encoder(full.unsqueeze(1).float(),
                            m.unsqueeze(0).unsqueeze(1).expand(T_, 1, -1),
                            aat[0].unsqueeze(0).expand(T_, -1),
                            ridx[0].unsqueeze(0).expand(T_, -1),
                            cid[0].unsqueeze(0).expand(T_, -1),
                            token_mask=m.unsqueeze(0).unsqueeze(1).expand(T_, 1, -1))[:, 0]

        tk = args.fixed_temperature if args.fixed_temperature is not None else float(w.temperature_k)
        hs._rollout_steps = int(w.horizon)
        out = hs.sample_future_ca(ctx, m.unsqueeze(0), aat, ridx, cid, R, t,
                                  n_samples=args.n_samples, temperature_k=tk)
        traj = out["H_traj"]                       # [n, T_roll+1, L, dH]
        H_roll = traj[:, -1].mean(0)               # ensemble-mean terminal latent [L,dH]

        vm = m
        def _d(a, b):
            return float((a - b)[vm].norm(dim=-1).mean())

        step_true = _d(H_true[-1], H_anchor)
        drift = _d(H_roll, H_true[-1])
        spread_roll = float(H_roll[vm].std(0).norm())
        spread_true = float(H_true[-1][vm].std(0).norm())

        # Decompose the drift: how much of it is the rollout going TOO FAR along
        # the true conformational direction, versus pointing the WRONG WAY?
        #   true_vec   = H_true(t)  - H_anchor      (the real conformational step)
        #   roll_vec   = H_roll     - H_anchor      (what the GLE did)
        # along = roll_vec . true_unit ; perp = ||roll_vec - along*true_unit||
        # A rollout that is merely over-dispersed has along/|true_vec| > 1 and
        # perp ~ 0; one that is mis-shaped has perp comparable to |true_vec|.
        tv = (H_true[-1] - H_anchor)[vm]          # [Lm, dH]
        rv = (H_roll - H_anchor)[vm]
        tu = tv / tv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        along = (rv * tu).sum(-1)                 # signed, per residue
        perp = (rv - along.unsqueeze(-1) * tu).norm(dim=-1)
        rows.append(dict(horizon=float(w.horizon), protein=w.protein_id,
                         temperature_k=float(w.temperature_k),
                         true_step=step_true, drift=drift,
                         ratio=drift / max(step_true, 1e-9),
                         spread_roll=spread_roll, spread_true=spread_true,
                         along=float(along.mean()),
                         perp=float(perp.mean()),
                         along_ratio=float(along.mean() / max(step_true, 1e-9)),
                         perp_ratio=float(perp.mean() / max(step_true, 1e-9))))

    if not rows:
        print("no windows scored")
        return
    print(f"\nn={len(rows)} windows over {len({r['protein'] for r in rows})} proteins")
    print(f"{'H':>4}{'|dH true|':>12}{'|dH drift|':>12}{'drift/true':>12}"
          f"{'roll spread':>13}{'true spread':>13}{'along/true':>12}{'perp/true':>11}")
    summary = {}
    keys = ("true_step", "drift", "ratio", "spread_roll", "spread_true",
            "along", "perp", "along_ratio", "perp_ratio")
    for h in sorted({r["horizon"] for r in rows}):
        g = [r for r in rows if r["horizon"] == h]
        rec = {k: float(np.mean([r[k] for r in g])) for k in keys}
        rec["n_windows"] = len(g)
        summary[str(h)] = rec
        print(f"{h:>4.0f}{rec['true_step']:>12.1f}{rec['drift']:>12.1f}"
              f"{rec['ratio']:>12.2f}{rec['spread_roll']:>13.2f}{rec['spread_true']:>13.2f}"
              f"{rec['along_ratio']:>12.2f}{rec['perp_ratio']:>11.2f}")
    med = float(np.median([summary[k]["ratio"] for k in summary]))
    print(f"\nmedian drift/true-step = {med:.2f}  "
          f"({'ROLLOUT OVERSHOOTS the true conformational change' if med > 1.15 else 'rollout is within the true step size'})")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n_windows": len(rows), "by_horizon": summary, "windows": rows,
             "median_ratio": med}, indent=1))
        print(f"wrote {args.out}")

    # Gate: a kernel whose rollouts are stable must track the true conformational
    # step to within the --max-drift tolerance at EVERY horizon, not just on
    # average.  The legacy force-bath kernel scored 152.8 here, and the
    # `artifacts/lasp_v2_full` run reached 1.80 but still overshot 1.12 -> 2.40
    # as the horizon grew, which is the signature that remains to be fixed.
    if args.max_drift is not None:
        worst = max(summary[k]["ratio"] for k in summary)
        if worst > args.max_drift:
            print(f"FAIL: worst drift/true-step = {worst:.2f} > {args.max_drift}")
            raise SystemExit(1)
        print(f"PASS: worst drift/true-step = {worst:.2f} <= {args.max_drift}")


if __name__ == "__main__":
    main()
