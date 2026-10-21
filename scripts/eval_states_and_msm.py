"""Static (BioEmu-style) + MSM state-prediction evaluation (TASK 2, NEW 2026-10-21).

Two families of metrics, reported SEPARATELY (per the source doc §5):

Static / ensemble (evaluation.states_metrics):
    reference-state coverage, metastable free-energy MAE, population MAE/TV/JSD

Dynamics / MSM (evaluation.msm_metrics):
    lagged state accuracy / macro-F1 / balanced accuracy (vs persistence),
    transition-matrix error (TV/JSD/Frobenius), implied-timescale error.

Comparison set: model (Model-1 single-step ensemble), persistence, oracle.
The MSM (TICA + KMeans + MSM) is fit on the TRUE trajectories only and reused to
label both true and predicted frames, so the label space is model-independent.

This script only reads frozen models; it never modifies the decoding path.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _anchors, _targets_from,
)
from protein_jepa_frame.training.train_utils import _device

from evaluation.data import EvalProtocol, iter_windows
from evaluation.geometry import ca_from_backbone
from evaluation.states_metrics import (
    internal_distance_features, reference_state_coverage, free_energy_mae,
    state_populations, _kmeans,
)
from evaluation.msm_metrics import (
    fit_msm, lagged_state_prediction, transition_matrix_error,
    implied_timescales,
)


def _load_full_trajectories(manifest, max_proteins, max_residues=192, stride=1):
    """Load full time-ordered CA trajectories per (protein, T, replica).

    Returns dict protein_id -> list of [T, L, 3] numpy arrays (one per trajectory),
    for the FIRST temperature/replica of each protein (deterministic).
    """
    from protein_jepa_frame.data.dataset import TrajectoryClipDataset
    from protein_jepa_frame.data.manifest import load_manifest
    from protein_jepa_frame.config import DataConfig

    recs = load_manifest(manifest)
    seen, out = {}, {}
    for r in recs:
        p = r.protein_id
        if p not in seen:
            if len(seen) >= max_proteins:
                continue
            seen[p] = True
        if seen.get(p) is not None and p in out:
            continue  # already have one trajectory for this protein
        n = int(r.num_frames or TrajectoryClipDataset._read_num_frames(r))
        frames = np.arange(0, n, stride)
        ds = TrajectoryClipDataset([r], DataConfig(manifest_path=None, num_frames=len(frames),
                                                   max_residues=max_residues))
        clip = ds._read_clip(r, frames)
        ca = ca_from_backbone(torch.as_tensor(clip["coords"])).numpy()  # [T,L,3]
        m = np.asarray(clip["residue_mask"])
        m = m.all(0) if m.ndim == 2 else m
        out[p] = [(ca, m.astype(bool))]
    return out


def _cls_metrics(y_true, y_pred, n_states):
    """Accuracy, macro-F1, balanced accuracy for integer state labels."""
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan")
    labels = list(range(n_states))
    return (float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            float(balanced_accuracy_score(y_true, y_pred)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--ensemble-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-proteins", type=int, default=40)
    p.add_argument("--max-trajectories-per-protein", type=int, default=1)
    p.add_argument("--max-anchors-per-trajectory", type=int, default=4)
    p.add_argument("--context-frames", type=int, default=4)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--max-residues", type=int, default=192)
    p.add_argument("--k-states", type=int, default=5)
    p.add_argument("--n-micro", type=int, default=50)
    p.add_argument("--eps-angstrom", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def _sample_model(producer, ctx, ctx_mask, aatype, ri, ci, R_a, t_a, n_samples, cond):
    return producer.sample_future_ca(ctx, ctx_mask, aatype, ri, ci, R_a, t_a,
                                     n_samples=n_samples, cond=cond)


def main():
    args = parse_args()
    dev = _device(args.device)
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    cond_dim = int(ck.get("args", {}).get("cond_dims", 0))
    head = GaussianHHead(dH, hidden=int(ck.get("args", {}).get("head_hidden", 128)),
                         cond_dim=cond_dim)
    head.load_state_dict(ck["head"]); head.to(dev).eval()
    producer = EnsembleProducer(encoder, decoder, head).to(dev).eval()

    proto = EvalProtocol(
        manifest=args.manifest,
        horizons_frames=tuple(args.horizons),
        context_frames=args.context_frames, context_stride=1,
        frame_stride=args.frame_stride, max_residues=args.max_residues,
        max_proteins=args.max_proteins,
        max_trajectories_per_protein=args.max_trajectories_per_protein,
        max_anchors_per_trajectory=args.max_anchors_per_trajectory,
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False, seed=args.seed,
    )

    # Collect per-protein true future frames, and model/persistence samples.
    # Keyed by (protein, horizon).
    true_by_ph: dict[tuple, list] = defaultdict(list)
    model_by_ph: dict[tuple, list] = defaultdict(list)
    pers_by_ph: dict[tuple, list] = defaultdict(list)
    prot_ca: dict[str, list] = defaultdict(list)   # protein -> list of true CA frames (for MSM)
    prot_feat: dict[str, np.ndarray] = {}

    n_win = 0
    windows_meta: list[dict] = []   # carries (protein, frames, horizon, model feats)
    for w in iter_windows(proto):
        ctx = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in w.context.items()}
        tgt = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in w.target.items()}
        ctx = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ctx.items()}
        tgt = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in tgt.items()}
        rmask = ctx["residue_mask"].all(dim=1).bool()
        anchor_bb = ctx["coords"][:, -1]
        anchor_ca = ca_from_backbone(anchor_bb)[0]                      # [L,3]
        true_ca = ca_from_backbone(tgt["coords"][:, -1])[0]             # [L,3]
        aatype = ctx["aatype"][:, -1] if ctx["aatype"].dim() == 3 else ctx["aatype"]
        ri = ctx.get("residue_index")
        ri = (ri[:, -1] if ri is not None and ri.dim() == 3 else ri)
        ci = ctx.get("chain_id")
        ci = (ci[:, -1] if ci is not None and ci.dim() == 3 else ci)
        from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
        R_a, t_a = _extract_backbone_frames(anchor_bb)
        cond = None
        out = _sample_model(producer, ctx["coords"], rmask, aatype, ri, ci, R_a, t_a,
                            args.n_samples, cond)
        model_ca = out["ca_samples"][0]                                 # [n,L,3]

        key = (w.protein_id, w.horizon)
        true_by_ph[key].append(true_ca.cpu().numpy())
        model_by_ph[key].append(model_ca.cpu().numpy())
        pers_by_ph[key].append(np.repeat(anchor_ca.cpu().numpy()[None], args.n_samples, 0))
        prot_ca[w.protein_id].append(true_ca.cpu().numpy())

        # model-sample features for MSM state labelling (same recipe as the MSM)
        mc = torch.tensor(model_ca.cpu().numpy()[0], dtype=torch.float32)  # [L,3]
        mmask = rmask[0].cpu()
        mf, _ = internal_distance_features(mc.unsqueeze(0), mmask, min_sep=3,
                                           max_pairs=256, seed=args.seed)
        windows_meta.append({
            "protein_id": w.protein_id, "horizon": int(w.horizon),
            "anchor_frame": int(w.anchor_frame), "target_frame": int(w.target_frame),
            "model_feats": mf.numpy(),
        })
        n_win += 1
        if n_win % 5 == 0:
            print(f"[states-msm] windows={n_win} proteins={len(prot_ca)}", flush=True)

    # ---------- MSM fit on TRUE time-ordered trajectories ----------
    # The MSM must be fit on REAL temporal trajectories (not shuffled anchor
    # windows), otherwise the transition matrix is degenerate. We load the full
    # NA-frame trajectory for each protein and fit a joint MSM over proteins
    # (TICA + KMeans on internal-distance features + reversible MSM).
    full_traj = {}
    try:
        full_traj = _load_full_trajectories(args.manifest, args.max_proteins,
                                            max_residues=args.max_residues, stride=1)
    except Exception as e:
        print(f"[states-msm] full-trajectory load failed: {e}", flush=True)

    msm_model = None
    msm_traj_feats: dict[str, np.ndarray] = {}
    msm_traj_lab: dict[str, np.ndarray] = {}
    if full_traj:
        feats_list, ids = [], []
        for p, arrs in full_traj.items():
            for ca, m in arrs:
                ca_t = torch.tensor(np.asarray(ca), dtype=torch.float32)   # [T,L,3]
                mt = torch.tensor(np.asarray(m), dtype=torch.bool)
                if int(mt.sum()) < 4:
                    continue
                f, _ = internal_distance_features(ca_t, mt, min_sep=3,
                                                  max_pairs=256, seed=args.seed)
                msm_traj_feats[p] = f.numpy()
                feats_list.append(f.numpy())
                ids.append(p)
        if feats_list:
            try:
                msm_model = fit_msm(feats_list, lag=1, n_micro=args.n_micro,
                                    n_states=args.k_states, seed=args.seed)
                for p in msm_traj_feats:
                    msm_traj_lab[p] = msm_model.label(msm_traj_feats[p])
                print(f"[states-msm] fit MSM on {len(feats_list)} true trajectories "
                      f"({msm_model.n_states} states)", flush=True)
            except Exception as e:
                print(f"[states-msm] MSM fit failed: {e}", flush=True)

    # ---------- Static metrics (coverage + free energy) ----------
    # Prefer the true full trajectories as the reference ensemble; fall back to
    # the pooled future frames when unavailable.
    static_ref = {p: nparr for p, arrs in full_traj.items() for nparr, _m in [arrs[0]]} \
        if full_traj else {}
    ref_pool = {p: np.stack(v) for p, v in prot_ca.items() if len(v) >= 3}
    cov_rows, fe_rows = [], []
    ref_source = static_ref if static_ref else ref_pool
    for p, arr in ref_source.items():
        ca_t = torch.tensor(np.asarray(arr), dtype=torch.float32)
        m = torch.ones(ca_t.shape[-2], dtype=torch.bool)
        model_all = []
        for (pp, h), v in model_by_ph.items():
            if pp == p:
                model_all.append(np.concatenate(v, 0))
        if not model_all:
            continue
        model_cat = torch.tensor(np.concatenate(model_all, 0), dtype=torch.float32)
        cov = reference_state_coverage(model_cat, ca_t, m, k_states=args.k_states,
                                       eps_angstrom=args.eps_angstrom, seed=args.seed)
        cov_rows.append(cov["coverage"])
        ref_feat, _ = internal_distance_features(ca_t, m, min_sep=3, max_pairs=256, seed=args.seed)
        mod_feat, _ = internal_distance_features(model_cat, m, min_sep=3, max_pairs=256, seed=args.seed)
        C, ref_lab = _kmeans(ref_feat, args.k_states, seed=args.seed)
        mod_lab = torch.cdist(mod_feat, C).argmin(-1)
        pop_ref = state_populations(ref_lab, C.shape[0])
        pop_mod = state_populations(mod_lab, C.shape[0])
        fe = free_energy_mae(pop_mod, pop_ref, 348.0)
        fe_rows.append(fe)

    static = {
        "coverage_mean": float(np.mean(cov_rows)) if cov_rows else None,
        "n_proteins": len(cov_rows),
        "free_energy_mae_kcal_mean": float(np.mean([r["free_energy_mae_kcal"] for r in fe_rows])) if fe_rows else None,
        "population_tv_mean": float(np.mean([r["population_tv"] for r in fe_rows])) if fe_rows else None,
        "population_jsd_mean": float(np.mean([r["population_jsd"] for r in fe_rows])) if fe_rows else None,
    }

    # ---------- Dynamics metrics (lagged MSM state prediction) ----------
    # For each anchor window we know the true target frame index in the real
    # trajectory (w.* metadata). We map that frame to its true MSM state, and
    # map each model sample to a predicted state. Accuracy is measured with a
    # lag tau = horizon (the model predicts tau frames ahead).
    dynamics = {}
    if msm_model is not None:
        # accumulate per-horizon (true_label, pred_label) pairs across windows
        # for BOTH the fine microstate discretization and PCCA macrostates.
        per_h: dict[int, list] = defaultdict(list)        # micro
        per_h_macro: dict[int, list] = defaultdict(list)  # macro
        per_h_pers: dict[int, list] = defaultdict(list)   # micro persistence
        per_h_pers_macro: dict[int, list] = defaultdict(list)
        per_h_pers_macro_anchor: dict[int, list] = defaultdict(list)  # true macrostate at anchor
        per_h_change: dict[int, list] = defaultdict(list)  # true state change flag
        n_macro = int(msm_model.n_macro or msm_model.n_states)
        macro_lab = {p: (msm_model.macro_map[np.clip(v, 0, len(msm_model.macro_map) - 1)]
                         if msm_model.macro_map is not None else v)
                     for p, v in msm_traj_lab.items()}
        for w in windows_meta:
            p = w["protein_id"]
            if p not in msm_traj_lab:
                continue
            lab = msm_traj_lab[p]
            t_tgt = int(w["target_frame"])
            if t_tgt < 0 or t_tgt >= len(lab):
                continue
            h = int(w["horizon"])
            fm = w["model_feats"]
            if fm is None or len(fm) == 0:
                continue
            y_true = int(lab[t_tgt])
            pred = msm_model.label(np.asarray(fm)).astype(int)
            per_h[h].extend([(y_true, int(pp)) for pp in pred])

            lab_m = macro_lab[p]
            y_true_m = int(lab_m[t_tgt])
            pred_m = msm_model.label_macro(np.asarray(fm)).astype(int) if msm_model.macro_map is not None else pred
            per_h_macro[h].extend([(y_true_m, int(pp)) for pp in pred_m])

            # persistence baseline: predict the anchor frame's own state
            t_anc = int(w["anchor_frame"])
            y_pers = int(lab[t_anc]) if 0 <= t_anc < len(lab) else y_true
            y_pers_m = int(lab_m[t_anc]) if 0 <= t_anc < len(lab_m) else y_true_m
            per_h_pers[h].extend([(y_true, y_pers) for _ in pred])
            per_h_pers_macro[h].extend([(y_true_m, y_pers_m) for _ in pred])
            # true macrostate at the anchor (t), expanded to one entry per model
            # sample, so it aligns with the per-sample predicted labels.
            per_h_pers_macro_anchor[h].extend([y_pers_m for _ in pred])
            per_h_change[h].append(int(y_true_m != y_pers_m))

        for h in args.horizons:
            pairs = per_h.get(int(h), [])
            if not pairs:
                continue
            yt = np.array([a for a, _ in pairs]); yp = np.array([b for _, b in pairs])
            acc, mf1, bacc = _cls_metrics(yt, yp, msm_model.n_states)
            pt = np.array([a for a, _ in per_h_pers.get(int(h), [])])
            pp_ = np.array([b for _, b in per_h_pers.get(int(h), [])])
            pacc = _cls_metrics(pt, pp_, msm_model.n_states)[0] if len(pt) else float("nan")

            mac = per_h_macro.get(int(h), [])
            ytm = np.array([a for a, _ in mac]); ypm = np.array([b for _, b in mac])
            macc, mmf1, mbacc = _cls_metrics(ytm, ypm, n_macro) if len(mac) else (np.nan,)*3
            ptm = np.array([a for a, _ in per_h_pers_macro.get(int(h), [])])
            ppm = np.array([b for _, b in per_h_pers_macro.get(int(h), [])])
            pmacc = _cls_metrics(ptm, ppm, n_macro)[0] if len(ptm) else float("nan")

            # true transition rate: fraction of windows where the macrostate
            # changed between anchor and target (model-independent reference).
            changed = per_h_change.get(int(h), [])

            # ---- binary transition-vs-non-transition detection (key ask) ----
            # macrostate at anchor (t) and at target (t+tau), model's predicted
            # macrostate at the target: "did the model predict a jump?".
            td = {}
            if len(mac):
                y_from = np.array([a for a, _ in per_h_pers_macro_anchor.get(int(h), [])])
                if len(y_from) == len(ytm):
                    from evaluation.msm_metrics import transition_detection
                    td = transition_detection(y_from, ytm, ypm)

            dynamics[str(h)] = {
                "micro_accuracy": acc, "micro_macro_f1": mf1,
                "micro_balanced_accuracy": bacc, "micro_persist_accuracy": pacc,
                "macro_accuracy": float(macc), "macro_macro_f1": float(mmf1),
                "macro_balanced_accuracy": float(mbacc), "macro_persist_accuracy": float(pmacc),
                "n_macro_states": int(n_macro),
                "true_transition_rate": float(np.mean(changed)) if changed else float("nan"),
                "transition_detection": td,
                "n_eval": int(len(yt)),
            }
        if msm_model.msm is not None:
            dynamics["true_msm_timescales"] = implied_timescales(msm_model.msm, 5).tolist()
            dynamics["true_msm_n_states"] = int(msm_model.n_states)

    res = {"manifest": args.manifest, "n_windows": n_win, "n_samples": args.n_samples,
           "horizons": args.horizons, "static": static, "dynamics": dynamics}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"[states-msm] wrote {args.out}")


if __name__ == "__main__":
    main()
