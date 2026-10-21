"""TASK 3 — Temperature-generalization evaluation (NEW, 2026-10-21).

mdCATH spans 5 temperatures {320, 348, 379, 413, 450} K.  We measure how well a
trained predictor transfers when the evaluation temperature was NOT seen in
training (leave-one-temperature-out), and how the prediction DEGRADES when the
wrong temperature is fed to a temperature-conditioned model.

Design (the matrix the reviewer asked for):
    rows  : each held-out temperature T* in {320,348,379,413,450}
    cols  : {seen-protein, unseen-protein} x {correct-T, wrong-T, no-T}
    metric: ensemble RMSF L1 (lower better) + model-vs-persistence skill.

Two conditioning regimes:
    * Model 1 (ensemble head, cond_dim>=2): the head receives
      [log horizon time, (T-300)/150].  We can probe correct/wrong/no T.
    * cond_dim==0 checkpoints CANNOT take a temperature -> only "no T" is
      meaningful, which is exactly the finding Task 3 must report.

The temperature manifests:
    artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl  (T in 320..413)
    artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl    (held-out proteins)
    artifacts/mdcath_manifest_1000dom_temp/val_t450.jsonl      (T=450, unseen T)

Usage:
    python -m scripts.eval_temperature_generalization \
        --ensemble-ckpt artifacts/stage_ensemble_1000dom_temp_cond/checkpoint_best.pt \
        --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
        --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
        --manifest artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
        --out artifacts/eval_temperature_generalization.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.training.ensemble_targets import (
    build_ensemble_targets, persistence_ensemble,
)
from protein_jepa_frame.training.ensemble_losses import (
    EnsembleLossWeights, compute_ensemble_loss,
)
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _anchors, _targets_from,
)
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch

# mdCATH temperatures
MDCATH_TEMPS = (320.0, 348.0, 379.0, 413.0, 450.0)


def _build_cond(paired, cond_dim, dev, *, temp_override=None, temp_drop=False):
    """Mirror train_stage_ensemble._conditioning with T override / drop."""
    if cond_dim <= 0:
        return None
    ht = paired.get("horizon_time_ps")
    B = paired["context_coords"].shape[0]
    t_norm = (torch.log(ht.to(dev).clamp_min(1.0)) / torch.log(torch.tensor(1e4, device=dev))
              if ht is not None else torch.zeros(B, device=dev))
    temp = paired.get("temperature_k")
    if temp_drop:
        T_norm = torch.zeros(B, device=dev)
    elif temp_override is not None:
        T_norm = torch.full((B,), (float(temp_override) - 300.0) / 150.0, device=dev)
    else:
        T_norm = ((temp.to(dev).float() - 300.0) / 150.0
                  if temp is not None else torch.zeros(B, device=dev))
    parts = [t_norm, T_norm]
    return torch.stack(parts[:cond_dim], dim=-1) if cond_dim <= 2 else torch.stack(
        parts + [torch.zeros_like(t_norm)] * (cond_dim - 2), dim=-1)


def _rmsf_corr(pred_ca, true_rmsf, mask):
    """Per-residue RMSF Pearson r between model ensemble and true RMSF."""
    pred_r = pred_ca.std(dim=1).mean(-1)                       # [B,L]
    m = mask.float()
    n = m.sum(-1, keepdim=True).clamp_min(1.0)
    pm = (pred_r * m).sum(-1, keepdim=True) / n
    tm = (true_rmsf * m).sum(-1, keepdim=True) / n
    pc = (pred_r - pm) * m
    tc = (true_rmsf - tm) * m
    num = (pc * tc).sum(-1)
    den = torch.sqrt((pc * pc).sum(-1) * (tc * tc).sum(-1)).clamp_min(1e-9)
    return (num / den)


def _eval_conditions(producer, pairs, args, dev, cond_dim, cond_mode, cond_temp):
    """Run one (dataset, condition) cell; return mean rmsf + skill + rmsf_r."""
    rmsf_m, rmsf_p, rmsf_o, corr_m, n = [], [], [], [], 0
    for batch in pairs:
        paired = _move_batch(batch, dev)
        targets, fut, fmask = _targets_from(paired, args, dev)
        bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)
        ctx = paired["context_coords"].to(dev)
        ctx_mask = paired["context_residue_mask"].to(dev)[:, -1]
        aatype = paired["future_aatype"].to(dev)
        ri = paired["future_residue_index"].to(dev)
        ci = paired["future_chain_id"].to(dev)
        R_a, t_a = _anchors(paired, dev)

        if cond_mode == "correct":
            cond = _build_cond(paired, cond_dim, dev)
        elif cond_mode == "wrong":
            cond = _build_cond(paired, cond_dim, dev, temp_override=cond_temp)
        else:  # "none"
            cond = _build_cond(paired, cond_dim, dev, temp_drop=True)

        with torch.no_grad():
            out = producer.sample_future_ca(ctx, ctx_mask, aatype, ri, ci, R_a, t_a,
                                            n_samples=args.n_samples, cond=cond)
            ca = out["ca_samples"]
            true_ca_pos = fut[:, -1, :, 1, :]
            m = compute_ensemble_loss(
                ca, targets.rmsf, targets.contact,
                targets.dist_hist_from_ca(ca), targets.dist_hist_true(),
                targets.mask, bin_w, args.weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=out["ca_mean_decoded"])
            pca = persistence_ensemble(ctx, args.n_samples,
                                       paired["context_residue_mask"].to(dev))
            p = compute_ensemble_loss(
                pca, targets.rmsf, targets.contact,
                targets.dist_hist_from_ca(pca), targets.dist_hist_true(),
                targets.mask, bin_w, args.weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=ctx[:, -1, :, 1, :])
            o = compute_ensemble_loss(
                targets.ca_window, targets.rmsf, targets.contact,
                targets.dist_hist_true(), targets.dist_hist_true(),
                targets.mask, bin_w, args.weights, cutoff=args.cutoff,
                true_ca_pos=true_ca_pos, pred_ca_pos=true_ca_pos)
        rmsf_m.append(float(m.rmsf)); rmsf_p.append(float(p.rmsf)); rmsf_o.append(float(o.rmsf))
        corr_m.append(float(_rmsf_corr(ca, targets.rmsf, targets.mask).mean()))
        n += ca.shape[0]
    if not n:
        return None
    rmsf_m = float(np.mean(rmsf_m)); rmsf_p = float(np.mean(rmsf_p)); rmsf_o = float(np.mean(rmsf_o))
    return {
        "rmsf_model": rmsf_m, "rmsf_persistence": rmsf_p, "rmsf_oracle": rmsf_o,
        "rmsf_skill_vs_persist": (rmsf_p - rmsf_m) / max(rmsf_p, 1e-9),
        "rmsf_corr_model": float(np.mean(corr_m)),
        "n_windows": int(n),
    }


def _subset_by_temp(ds, temp, max_items, seed):
    """Window indices whose underlying trajectory record has the given T (Kelvin).

    Works for PairedTrajectoryDataset (uses `_locate`) and TrajectoryClipDataset
    (record_index == index).
    """
    idx = []
    locate = getattr(ds, "_locate", None)
    for i in range(len(ds)):
        if locate is not None:
            rec = locate(i)[0]
        else:
            rec_idx = getattr(ds, "_record_index", None)
            rec = ds.records[i] if rec_idx is None else ds.records[rec_idx(i)]
        tk = getattr(rec, "temperature_k", None)
        if tk is None:
            continue
        if abs(float(tk) - float(temp)) < 1e-3:
            idx.append(i)
    g = torch.Generator().manual_seed(seed)
    if max_items and len(idx) > max_items:
        sel = torch.randperm(len(idx), generator=g)[:max_items].tolist()
        idx = [idx[j] for j in sel]
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-ckpt", required=True)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--manifest", required=True,
                   help="held-out-protein manifest (val_not450)")
    p.add_argument("--manifest-unseen-temp", default=None,
                   help="T=450 unseen-temperature manifest (val_t450)")
    p.add_argument("--manifest-seen", default=None,
                   help="training-protein manifest (e.g. train_not450) to fill the "
                        "'seen protein' cells; optional")
    p.add_argument("--manifest-seen-t450", default=None,
                   help="TRAIN proteins at the held-out T=450 (train_t450) so the "
                        "'seen protein, unseen temperature' cell can be filled; optional")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-windows-per-temp", type=int, default=40)
    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--dist-bins", type=int, default=40)
    p.add_argument("--dist-lo", type=float, default=0.0)
    p.add_argument("--dist-hi", type=float, default=40.0)
    p.add_argument("--max-pairs", type=int, default=256)
    p.add_argument("--min-seq-sep", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = _device(args.device)
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    ck_args = ck.get("args", {})
    cond_dim = int(ck_args.get("cond_dims", 0))
    head = GaussianHHead(dH, hidden=int(ck_args.get("head_hidden", 128)), cond_dim=cond_dim)
    head.load_state_dict(ck["head"]); head.to(dev).eval()
    producer = EnsembleProducer(encoder, decoder, head).to(dev).eval()
    args.weights = EnsembleLossWeights(
        rmsf_weight=1.0, contact_weight=10.0, dist_weight=0.3,
        pos_weight=float(ck_args.get("pos_weight", 0.0)))
    print(f"[temp-gen] cond_dim={cond_dim} (0 => model cannot be temperature-conditioned)",
          flush=True)

    class P: pass
    pr = P()
    pr.history_frames = args.history_frames
    pr.frame_stride = args.frame_stride
    pr.paired_horizons_frames = args.horizons

    def build_ds(path):
        cfg = DataConfig(manifest_path=path, batch_size=args.batch_size,
                         num_workers=args.num_workers, training_mode="paired_window",
                         num_frames=args.history_frames, paired_all_horizons=False,
                         max_residues=args.max_residues)
        apply_protocol(cfg, protocol_from_args(pr))
        return build_dataset(cfg)

    def loader_for(ds, idx):
        return DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=lambda s: _truncate_history(
                              collate_paired_windows(s), args.history_frames))

    # ---- datasets ----
    # ds_eval      : held-out PROTEINS, T in {320..413} (seen T, unseen protein)
    # ds_unseen_T  : held-out PROTEINS at T=450   (unseen T, unseen protein)
    # ds_trainprots: TRAIN proteins at T in {320..413} (seen T, seen protein)
    ds_eval = build_ds(args.manifest)
    ds_unseen_T = build_ds(args.manifest_unseen_temp) if args.manifest_unseen_temp else None
    ds_seen_T450 = build_ds(args.manifest_seen_t450) if args.manifest_seen_t450 else None
    ds_trainprots = build_ds(args.manifest_seen) if args.manifest_seen else None

    results = {"cond_dim": cond_dim, "n_samples": args.n_samples,
               "horizons": args.horizons, "cells": {}}

    temps_seen = (320.0, 348.0, 379.0, 413.0)
    # (protein_set_label, dataset) pairs to populate.
    sets = [("unseen_protein", ds_eval)]
    if ds_trainprots is not None:
        sets.append(("seen_protein", ds_trainprots))

    for T in MDCATH_TEMPS:
        unseen_T = (T == 450.0)
        for ps_label, ds_base in sets:
            # T=450 uses its own manifests: unseen-temperature+unseen-protein is
            # `val_t450`; seen-protein+unseen-T is `train_t450`.
            if unseen_T:
                ds = ds_seen_T450 if ps_label == "seen_protein" else ds_unseen_T
            else:
                ds = ds_base
            if ds is None:
                continue
            idx = _subset_by_temp(ds, T, args.max_windows_per_temp * args.batch_size, args.seed)
            if not idx:
                continue
            plist = list(loader_for(ds, idx))
            key = f"T{T:.0f}__{ps_label}"
            cell = {"T": T, "unseen_temperature": unseen_T, "protein_set": ps_label,
                    "n_trajectories": len({pid for b in plist
                                           for pid in b.get("protein_id", [])})}
            if cond_dim <= 0:
                cell["none"] = _eval_conditions(producer, plist, args, dev, cond_dim, "none", None)
                cell["note"] = "cond_dim=0: temperature input is inert"
            else:
                cell["correct"] = _eval_conditions(producer, plist, args, dev, cond_dim, "correct", None)
                cell["none"] = _eval_conditions(producer, plist, args, dev, cond_dim, "none", None)
                wrong = 320.0 if abs(T - 320.0) > 1e-3 else 450.0
                cell["wrong"] = _eval_conditions(producer, plist, args, dev, cond_dim, "wrong", wrong)
                cell["wrong_temp_used"] = wrong
            results["cells"][key] = cell
            print(f"[temp-gen] {key} done", flush=True)

    # ---- temperature-response sanity: does rmsf rise with T? (correct cond) ----
    resp = {}
    for T in temps_seen:
        key = f"T{T:.0f}__unseen_protein"
        c = results["cells"].get(key, {})
        cell = c.get("correct") or c.get("none")
        if cell:
            resp[f"T{T:.0f}"] = cell["rmsf_model"]
    results["temperature_response_rmsf_model"] = resp
    results["temperature_monotone"] = (
        all(resp.get(f"T{a:.0f}", 0) <= resp.get(f"T{b:.0f}", 1) + 1e-6
            for a, b in zip(temps_seen[:-1], temps_seen[1:]))
        if len(resp) == len(temps_seen) else None)

    print(json.dumps(results, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"[temp-gen] wrote {args.out}")


if __name__ == "__main__":
    main()
