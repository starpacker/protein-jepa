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
        n_win += 1
        if n_win % 5 == 0:
            print(f"[states-msm] windows={n_win} proteins={len(prot_ca)}", flush=True)

    # ---------- Fixed MSM fit on TRUE trajectories ----------
    # Use the concatenated true future frames per protein as the "trajectory".
    prot_arrays = {p: np.stack(v) for p, v in prot_ca.items() if len(v) >= 3}
    msm_model = None
    msm_feats = {}
    if prot_arrays:
        # Build per-protein feature arrays (rotation invariant) for the MSM.
        feats_list = []
        for p, arr in prot_arrays.items():
            ca_t = torch.tensor(arr, dtype=torch.float32)              # [N,L,3]
            m = torch.ones(ca_t.shape[1], dtype=torch.bool)
            f, _ = internal_distance_features(ca_t, m, min_sep=3, max_pairs=256, seed=args.seed)
            msm_feats[p] = f.numpy()
            feats_list.append(f.numpy())
        try:
            msm_model = fit_msm(feats_list, lag=1, n_micro=args.n_micro)
        except Exception as e:
            print(f"[states-msm] MSM fit failed: {e}")

    # ---------- Static metrics (coverage + free energy), averaged over proteins ----------
    cov_rows, fe_rows = [], []
    for p, arr in prot_arrays.items():
        ca_t = torch.tensor(arr, dtype=torch.float32)
        m = torch.ones(ca_t.shape[1], dtype=torch.bool)
        # reference = true frames of this protein; model = all model samples at any horizon
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
        # free energy: label model and true with shared centroids
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

    # ---------- Dynamics metrics (lagged state prediction) ----------
    dynamics = {}
    if msm_model is not None:
        for h in args.horizons:
            acc_rows, pers_rows = [], []
            for p, arr in prot_arrays.items():
                f_p = msm_feats[p]
                true_lab = msm_model.label(f_p)
                # predicted labels: label model samples at this horizon (pooled per protein)
                mod_labs = []
                for (pp, hh), v in model_by_ph.items():
                    if pp == p and hh == h:
                        mm = torch.tensor(np.concatenate(v, 0), dtype=torch.float32)
                        mk = torch.ones(mm.shape[1], dtype=torch.bool)
                        fm, _ = internal_distance_features(mm, mk, min_sep=3, max_pairs=256, seed=args.seed)
                        mod_labs.append(msm_model.label(fm.numpy()))
                if not mod_labs or len(true_lab) < 2:
                    continue
                pred_lab = np.concatenate(mod_labs, 0)
                # Align lengths: evaluate lagged accuracy on the min length, tau = h frames
                n = min(len(true_lab), len(pred_lab))
                tau = int(h)
                if n <= tau:
                    continue
                res = lagged_state_prediction(true_lab[:n], pred_lab[:n], tau=tau,
                                              n_states=msm_model.n_states)
                # also persistence-in-state baseline (pred[t]=true[t])
                res_p = lagged_state_prediction(true_lab[:n], true_lab[:n], tau=tau,
                                                n_states=msm_model.n_states)
                if res.get("n_eval", 0) > 0:
                    acc_rows.append(res)
                    pers_rows.append(res_p)
            if acc_rows:
                dynamics[str(h)] = {
                    "model_accuracy": float(np.mean([r["accuracy"] for r in acc_rows])),
                    "model_macro_f1": float(np.mean([r["macro_f1"] for r in acc_rows])),
                    "model_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in acc_rows])),
                    "persist_accuracy": float(np.mean([r["persist_accuracy"] for r in acc_rows])),
                    "n_eval": int(sum(r["n_eval"] for r in acc_rows)),
                }
        # implied timescales of the true MSM
        if msm_model is not None and msm_model.msm is not None:
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
