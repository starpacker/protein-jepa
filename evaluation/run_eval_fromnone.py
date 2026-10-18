"""Held-out evaluation for the from-none Stage B pipeline.

Evaluates a from-none checkpoint (fresh encoder + H/S kernel + CA decoder) with a
protocol identical to training: uniform K-frame history, one rollout of
``horizon`` frames, decode CA, Kabsch-align to the true future, report RMSD.

Reported per horizon (in ns):
  - persist : no-op forecast (anchor CA)
  - fromnone: autonomous H/S rollout -> decoder -> CA
  - skill   : 1 - rmsd/persist  (>0 beats persistence)

Splits (matching the user's requirement):
  --split protein -> held-out proteins (val/test manifest)
  --split temp    -> held-out temperature (trajectories at --heldout-temperature)

The rollout is autonomous and history-only: true future coordinates never enter
the prediction path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protein_jepa_frame.models.stage_b1b_fromnone.decoder import FromNoneConfig, SimpleCADecoder
from protein_jepa_frame.training.train_stage_cb1b import load_models_for_eval
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.utils.backbone_geometry import backbone_torsion_targets  # noqa: F401 (parity)
from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import ca_from_backbone, common_anchor_rmsd, contact_jaccard
from evaluation.metrics import paired_bootstrap_ci


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest", required=True, help="JSONL to evaluate (held-out proteins).")
    p.add_argument("--split", choices=["protein", "temp"], default="protein")
    p.add_argument("--heldout-temperature", type=float, default=None)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 24, 32, 48, 64, 96])
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--max-proteins", type=int, default=None)
    p.add_argument("--max-anchors-per-trajectory", type=int, default=8)
    p.add_argument("--n-rollout-reps", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def encode_history(encoder, normalizer, ctx, device):
    coords = ctx["coords"].to(device)                      # [B,K,L,4,3]
    B, K, L = coords.shape[:3]
    rmask = ctx["residue_mask"].to(device)                 # [B,K,L]
    H = encoder(
        coords.reshape(B * K, 1, L, 4, 3), rmask.reshape(B * K, 1, L),
        ctx["aatype"].to(device)[:, None].expand(B, K, L).reshape(B * K, L),
        ctx["residue_index"].to(device)[:, None].expand(B, K, L).reshape(B * K, L),
        ctx["chain_id"].to(device)[:, None].expand(B, K, L).reshape(B * K, L),
        token_mask=rmask.reshape(B * K, 1, L),
    )[:, 0].reshape(B, K, L, -1)
    if normalizer is not None:
        H = normalizer(H)
    return H


@torch.no_grad()
def evaluate(args):
    device = _device(args.device)
    encoder, kernel, decoder, normalizer, ck = load_models_for_eval(args.ckpt, device)
    step_ps = kernel.cfg.step_ps

    proto = EvalProtocol(
        manifest=args.manifest, horizons_frames=tuple(args.horizons),
        context_frames=args.history_frames, context_stride=1, frame_stride=args.frame_stride,
        max_residues=args.max_residues, max_proteins=args.max_proteins,
        max_anchors_per_trajectory=args.max_anchors_per_trajectory,
        temperatures=(args.heldout_temperature,) if args.split == "temp" and args.heldout_temperature else None,
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False, seed=args.seed,
    )

    # accumulate per (horizon, variant) window records, grouped by protein for CI
    rows = []
    n_windows = 0
    for w in iter_windows(proto):
        ctx = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in w.context.items()}
        tgt = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in w.target.items()}
        ctx = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ctx.items()}
        tgt = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in tgt.items()}
        rmask = ctx["residue_mask"].all(dim=1).bool()       # [1,L]
        anchor_bb = ctx["coords"][:, -1]                    # [1,L,4,3]
        anchor_ca = ca_from_backbone(anchor_bb)
        true_ca = ca_from_backbone(tgt["coords"][:, -1])   # [1,L,3]
        R, t = RigidBodyFrame.from_backbone(anchor_bb[..., 0, :], anchor_bb[..., 1, :], anchor_bb[..., 2, :])

        H_hist = encode_history(encoder, normalizer, ctx, device)   # [1,K,L,d]
        n_steps = int(round(w.horizon_ps / step_ps))
        if n_steps < 1:
            continue
        dt = ctx["time_ps"].diff(dim=-1)
        dt = torch.cat([dt[:, :1], dt], dim=-1)             # [1,K]
        temp = ctx.get("temperature_k")
        preds = []
        for rep in range(args.n_rollout_reps):
            gen = torch.Generator(device=device).manual_seed(args.seed + rep)
            traj = kernel.rollout(
                H_hist[:, -1], rmask, temp, n_steps=n_steps, H_hist=H_hist, dt_hist=dt,
                seq_feature=H_hist[:, -1], generator=gen, detach=True, return_bath=True,
            )
            Hf, Sf = traj[0][:, -1], traj[1][:, -1]
            out = decoder(H_future=Hf, H_context=H_hist[:, -1], X_context=anchor_ca,
                          residue_mask=rmask, R_anchor=R, S_future=Sf)
            preds.append(out["ca_pred"][0])
        samples = torch.stack(preds)                        # [R,L,3]
        ensemble = samples[0] + (samples - samples[0]).mean(0)
        pred_ca = ensemble                                   # [L,3]

        # metrics (window-level)
        rmsd = common_anchor_rmsd(pred_ca, anchor_ca[0], true_ca[0], rmask[0])
        p_rmsd = common_anchor_rmsd(anchor_ca[0], anchor_ca[0], true_ca[0], rmask[0])
        try:
            jac = float(contact_jaccard(pred_ca, true_ca[0], rmask[0]))
            pjac = float(contact_jaccard(anchor_ca[0], true_ca[0], rmask[0]))
        except Exception:
            jac = pjac = float("nan")
        rows.append(dict(protein_id=w.protein_id, temperature_k=w.temperature_k,
                         horizon=w.horizon, rmsd=rmsd, persist_rmsd=p_rmsd,
                         contact_jaccard=jac, persist_contact_jaccard=pjac))
        n_windows += 1
        if args.max_proteins is not None:
            pass

    # aggregate: per-protein mean per horizon, then mean + paired bootstrap
    horizons = sorted({r["horizon"] for r in rows})
    out = {"ckpt": args.ckpt, "split": args.split, "manifest": args.manifest,
           "heldout_temperature": args.heldout_temperature, "n_windows": n_windows,
           "step_ps": step_ps, "n_rollout_reps": args.n_rollout_reps, "by_horizon": {}}
    for h in horizons:
        rs = [r for r in rows if r["horizon"] == h]
        per_protein = {}
        for r in rs:
            per_protein.setdefault(r["protein_id"], []).append(r)
        prot_rmsd = [np.mean([x["rmsd"] for x in v]) for v in per_protein.values()]
        prot_persist = [np.mean([x["persist_rmsd"] for x in v]) for v in per_protein.values()]
        prot_jac = [np.mean([x["contact_jaccard"] for x in v]) for v in per_protein.values()]
        prot_pjac = [np.mean([x["persist_contact_jaccard"] for x in v]) for v in per_protein.values()]
        dm, lo, hi = paired_bootstrap_ci(prot_rmsd, prot_persist, seed=args.seed)
        djm, jlo, jhi = paired_bootstrap_ci(prot_jac, prot_pjac, seed=args.seed)
        out["by_horizon"][str(h)] = {
            "horizon_ns": h * (step_ps / 1000.0),
            "n_proteins": len(per_protein), "n_windows": len(rs),
            "rmsd": float(np.nanmean(prot_rmsd)),
            "persist_rmsd": float(np.nanmean(prot_persist)),
            "skill": float(np.nanmean(prot_persist) - np.nanmean(prot_rmsd)),
            "delta_vs_persist": dm, "delta_ci95": [lo, hi],
            "contact_jaccard": float(np.nanmean(prot_jac)),
            "persist_contact_jaccard": float(np.nanmean(prot_pjac)),
            "contact_delta": djm, "contact_ci95": [jlo, jhi],
        }
    return out


def main(argv=None):
    args = parse_args(argv)
    result = evaluate(args)
    text = json.dumps(result, indent=2, sort_keys=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    # compact per-horizon table
    print("\nhorizon_ns  persist  fromnone  skill(+better)  contact_jac  persist_jac")
    for h, d in result["by_horizon"].items():
        print(f"{d['horizon_ns']:>9.0f}  {d['persist_rmsd']:>7.3f}  {d['rmsd']:>8.3f}  "
              f"{d['skill']:>+8.3f}        {d['contact_jaccard']:>7.3f}  {d['persist_contact_jaccard']:>7.3f}")


if __name__ == "__main__":
    main()
