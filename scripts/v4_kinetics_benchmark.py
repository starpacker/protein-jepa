"""v4 Stage B kinetics + microscopic benchmark (autonomous rollout vs GT vs persistence).

Spec §3 B3.7 / §6 "Dynamics" and the expanded goal ("微观指标" against BioEmu2 and
persistence): the ensemble harness (``eval_lasp_final.py``) reports only *static*
ensemble metrics (RMSF/W2/contacts).  This script reports the *dynamical / microscopic*
half on the same held-out data:

  * multi-step autonomous CA-RMSD of the decoded rollout (per horizon), vs persistence
  * adjacent-CA bond validity of the autonomous rollout (the structural-validity gate)
  * CA-displacement autocorrelation decay time (does the model reproduce the GT
    memory/relaxation envelope?)
  * implied timescales (deeptime TICA + reversible MSM) of the decoded path vs GT

It reuses the *exact* Stage B forecast path from ``train_stage_b_v4``
(``load_v4_bundle`` + ``StageBV4`` + ``forecast_objective_v4``) so the numbers are
consistent with training-time validation, and reuses ``_load_full_trajectories`` from
``eval_states_and_msm.py`` for the full GT trajectories the MSM must be fit on
(clip-mode windows are too short for meaningful timescales).

Two checkpoints can be compared (full-S vs no-S) to test the S bath's effect on
held-out kinetics beyond a matched Markov baseline (spec §8 "no bath claim unless
disabling S measurably changes forecasts AND improves held-out kinetics").

Usage:
    python scripts/v4_kinetics_benchmark.py \
        --stage-a-v4-ckpt artifacts/v4_stage_a/run_400/checkpoint_best.pt \
        --stage-b-ckpt  artifacts/v4_stage_b/run_bond_grad/checkpoint_latest.pt \
        --manifest artifacts/mdcath_manifest_full/val.jsonl \
        --out results/v4_kinetics_fullS.json
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.models.stage_b_v4 import StageBV4
from protein_jepa_frame.models.stage_b1b import StageB1bConfig
from protein_jepa_frame.training.train_stage_b_v4 import (
    load_v4_bundle, forecast_objective_v4, _move_batch, _loader,
)
from protein_jepa_frame.training.train_utils import _device


def _ckpt_args(path: str) -> argparse.Namespace:
    """Defaults mirroring train_stage_b_v4's training args (the Stage B checkpoint
    stores only the GLE ``cfg``, not the forecast/training hyperparameters, so those
    are supplied explicitly via CLI flags and merged here)."""
    return argparse.Namespace(
        history_frames=8, horizon_steps=8, frame_stride=1, max_residues=96,
        latent_dim=128, bath_dim=8, energy_depth=3, energy_heads=4, energy_hidden=128,
        history_hidden=128, step_ps=1000.0, coupling_scale=1e-3, diffusion_scale=1e-4,
        c_init_std=0.1, gamma_scale=2e-4, gamma_floor=2e-5, max_gamma_step=1.9,
        tail_init=1.0, es_weight=1.0, physical_es_weight=0.3, bond_weight=1.0,
        es_num_samples=2, rmsd_weight=0.0, manifold_weight=0.0, seed=42,
        # _loader / forecast_objective_v4 also read these:
        validation_horizon_steps=None, es_horizon_steps=None, rmsd_horizon_steps=None,
        validation_batches=16, batch_size=2, num_workers=0,
    )


def _load_stage_b(path: str, args: argparse.Namespace, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = StageB1bConfig(**ck["cfg"])
    model = StageBV4(cfg).to(device)
    model.load_state_dict(ck["gle"], strict=True)
    model.eval()
    return model


def _autocorr_decay(ca_path: torch.Tensor, mask: torch.Tensor) -> dict:
    """CA-displacement autocorrelation of a decoded [T, L, 3] path.

    Returns the lag-1..4 autocorrelation of per-residue CA displacement and a
    fitted exponential decay time (in frames).  A model that reproduces GT memory
    has a decay time close to the GT path's; persistence has decay time 0 (no
    displacement at all) and white noise has ~1 (delta-correlated).
    """
    ca = ca_path[:, mask, :]                      # [T, L', 3]
    disp = (ca[1:] - ca[:-1]).norm(dim=-1)        # [T-1, L']
    disp = disp - disp.mean(0, keepdim=True)
    def acf(lag):
        a = disp[:-lag] if lag else disp
        b = disp[lag:]
        num = (a * b).sum()
        den = ((a * a).sum() * (b * b).sum()).clamp_min(1e-12).sqrt()
        return float((num / den).item()) if den.item() > 0 else float("nan")
    lags = [acf(k) for k in (1, 2, 3, 4)]
    # Decay time from lag-1 autocorr: rho(k) ~ exp(-k/tau) -> tau = -1/ln(rho1)
    rho1 = lags[0] if np.isfinite(lags[0]) else 0.0
    if 0.0 < rho1 < 1.0:
        tau = -1.0 / np.log(rho1)
    elif rho1 >= 1.0:
        tau = float("inf")
    else:
        tau = 0.0
    return {"acf_lag1": lags[0], "acf_lag2": lags[1], "acf_lag3": lags[2],
            "acf_lag4": lags[3], "decay_time_frames": tau}


def _implied_timescales(paths: list[np.ndarray], lag: int = 1, n_states: int = 4) -> dict:
    """Fit a reversible MSM over a list of [T,L,3] CA paths and return top-n ITS.

    ``paths`` is a list of *long* decoded/true CA trajectories (one per protein,
    concatenated frames).  Returns {} if deeptime is unavailable or fitting fails,
    so the caller can degrade to autocorrelation-only kinetics.
    """
    try:
        from evaluation.msm_metrics import fit_msm, implied_timescales
        from evaluation.states_metrics import internal_distance_features
    except Exception as e:
        return {"error": f"kinetic deps unavailable: {e}"}
    feats = []
    for ca in paths:
        t = torch.tensor(np.asarray(ca, dtype=np.float32))          # [T,L,3]
        m = torch.ones(t.shape[1], dtype=torch.bool)
        f, _ = internal_distance_features(t, m, min_sep=3, max_pairs=256, seed=0)
        feats.append(f.numpy())
    if len(feats) < 2:
        return {"error": "need >=2 trajectories to fit an MSM"}
    try:
        msm = fit_msm(feats, lag=lag, n_micro=min(30, min(f.shape[0] for f in feats)),
                      n_states=n_states, seed=0)
        its = implied_timescales(msm, n_states).tolist()
        return {"n_states": int(msm.n_states), "implied_timescales": its}
    except Exception as e:
        return {"error": f"MSM fit failed: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a-v4-ckpt", required=True)
    ap.add_argument("--stage-b-ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--validation-batches", type=int, default=16)
    ap.add_argument("--horizon-steps", type=int, default=8)
    ap.add_argument("--history-frames", type=int, default=8)
    ap.add_argument("--bath-dim", type=int, default=8)
    ap.add_argument("--physical-es-weight", type=float, default=0.3)
    ap.add_argument("--bond-weight", type=float, default=1.0)
    ap.add_argument("--es-num-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kinetic-trajectories", type=int, default=0,
                    help="number of full trajectories to load for implied-timescales "
                         "(0 = autocorrelation-only kinetics, no deeptime MSM)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(_device(a.device))
    targs = _ckpt_args(a.stage_b_ckpt)
    # CLI overrides (bath-dim / physical-es / bond-weight / horizon / history).
    targs.bath_dim = a.bath_dim
    targs.physical_es_weight = a.physical_es_weight
    targs.bond_weight = a.bond_weight
    targs.horizon_steps = a.horizon_steps
    targs.history_frames = a.history_frames
    targs.es_num_samples = a.es_num_samples
    targs.validation_batches = a.validation_batches
    targs.batch_size = a.batch_size

    static_encoder, dynamic_encoder, conditional_decoder, _ = load_v4_bundle(
        a.stage_a_v4_ckpt, device, targs.latent_dim)
    model = _load_stage_b(a.stage_b_ckpt, targs, device)
    val = _loader(targs, a.manifest, False)

    # --- autonomous rollout, decoded path -> microscopic + kinetic per window ---
    gen = torch.Generator(device=device).manual_seed(a.seed + 1000003)
    per_win = []
    ca_rmsd = []          # multi-step autonomous CA-RMSD (model vs GT)
    pers_rmsd = []        # persistence baseline
    bond_viol = []        # bond validity of autonomous rollout
    acf_model, acf_true = [], []
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        for idx, batch in enumerate(val):
            if idx >= a.validation_batches:
                break
            batch = _move_batch(batch, device)
            _, _, fc = forecast_objective_v4(
                model, static_encoder, dynamic_encoder, conditional_decoder, batch,
                targs, generator=gen, training=False)
            H_pred = fc["H_pred"]                       # [M, B, T, L, D]
            H_static = fc["H_static"]
            mask = batch["residue_mask"][:, targs.history_frames - 1]
            pred_bb = fc["predicted_bb"]
            if pred_bb is None:
                from protein_jepa_frame.training.train_stage_b_v4 import decode_paths_v4
                pred_bb = decode_paths_v4(conditional_decoder, H_pred[:, :, 1:], H_static,
                                          batch, mask, batch["coords"][:, targs.history_frames - 1])
            # pred_bb: [M, B, T, L, 4, 3], CA at ch 1; target GT path at [B, T, L, 4, 3]
            gt_bb = batch["coords"][:, targs.history_frames:]      # [B, T, L, 4, 3]
            pred_ca = pred_bb[..., 1, :]                           # [M, B, T, L, 3]
            gt_ca = gt_bb[..., 1, :]                               # [B, T, L, 3]
            anchor_ca = batch["coords"][:, targs.history_frames - 1, :, 1, :]  # [B, L, 3] (CA atom=1)

            # --- CA-RMSD (multi-step, Kabsch-aligned to anchor frame) vs persistence ---
            # Raw coordinates are in the trajectory's global frame, so an unaligned
            # RMSD is dominated by rigid-body drift, not structural error.  Align
            # every model/GT frame onto the anchor CA cloud (same _align_to the
            # ensemble harness uses) before computing the per-residue error.
            from evaluation.ensemble_metrics_ext import _align_to
            for b in range(pred_ca.shape[1]):
                m = mask[b].bool()
                if int(m.sum()) < 4:
                    continue
                anc = anchor_ca[b, m, :]                          # [L', 3] reference cloud
                pc = pred_ca[:, b, :, m, :]                       # [M, T, L', 3]
                gc = gt_ca[b, :, m, :]                            # [T, L', 3]
                # flatten [M,T,L',3] -> [M*T, L',3] and [T,L',3] -> align to anc.
                pc_a = _align_to(pc.reshape(-1, pc.shape[-2], 3), anc, None).reshape(pc.shape)
                gc_a = _align_to(gc, anc, None)
                # per-sample per-frame RMSD vs GT, averaged over samples
                rmsd = ((pc_a - gc_a[None]).norm(dim=-1).mean(dim=-1)).mean(dim=0)  # [T]
                prmsd = ((anc[None].expand_as(gc_a) - gc_a).norm(dim=-1).mean(dim=-1))  # [T]
                ca_rmsd.extend(rmsd.tolist())
                pers_rmsd.extend(prmsd.tolist())
                # bond validity on the (first-sample) autonomous path
                ca = pred_bb[0, b, :, m, 1, :]                   # [T, L', 3]
                bonds = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)    # [T, L'-1]
                viol = ((bonds < 3.0) | (bonds > 4.5)).float().mean()
                bond_viol.append(float(viol))
                # autocorrelation of the decoded path (first sample) vs GT path
                acf_model.append(_autocorr_decay(pred_bb[0, b, :, :, 1, :], mask[b]))
                acf_true.append(_autocorr_decay(gt_bb[b, :, :, 1, :], mask[b]))
                per_win.append({"batch": idx, "sample": b,
                                "ca_rmsd": float(rmsd.mean()), "pers_rmsd": float(prmsd.mean()),
                                "bond_violation": float(viol)})
        if idx % 4 == 0 and idx:
            print(f"  processed {idx} batches", flush=True)

    # --- kinetic: implied timescales on full trajectories (optional) ---
    kinetics = {"autocorr_model": {}, "autocorr_true": {}, "implied_timescales": {}}
    if acf_model:
        kinetics["autocorr_model"] = {k: float(np.nanmean([r[k] for r in acf_model]))
                                      for k in ("acf_lag1", "decay_time_frames")}
        kinetics["autocorr_true"] = {k: float(np.nanmean([r[k] for r in acf_true]))
                                     for k in ("acf_lag1", "decay_time_frames")}
    if a.kinetic_trajectories > 0:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_states_and_msm import _load_full_trajectories
        full = _load_full_trajectories(a.manifest, a.kinetic_trajectories, max_residues=192)
        true_paths = [arrs[0][0] for arrs in full.values()]     # [T,L,3] each
        kinetics["implied_timescales"]["true"] = _implied_timescales(true_paths)
        # Note: a model implied-timescale fit would require rolling out full-length
        # autonomous trajectories; the clip-mode forecast gives 8-step paths only.
        # We report GT ITS as the reference and rely on autocorr for the model's
        # memory envelope (8-step windows are too short for a stable model MSM).
        kinetics["implied_timescales"]["note"] = (
            "GT-only: model ITS needs full-length autonomous rollouts (not clip windows).")

    res = {
        "metric_version": "v4_kinetics_microscopic_v1",
        "stage_a_ckpt": a.stage_a_v4_ckpt, "stage_b_ckpt": a.stage_b_ckpt,
        "manifest": a.manifest, "seed": a.seed,
        "n_windows": len(per_win),
        "ca_rmsd_model": float(np.nanmean(ca_rmsd)) if ca_rmsd else float("nan"),
        "ca_rmsd_persistence": float(np.nanmean(pers_rmsd)) if pers_rmsd else float("nan"),
        "ca_rmsd_win_vs_persist": float(np.nanmean(ca_rmsd) - np.nanmean(pers_rmsd)) if ca_rmsd else float("nan"),
        "bond_violation_fraction": float(np.nanmean(bond_viol)) if bond_viol else float("nan"),
        "kinetics": kinetics,
        "per_window": per_win,
    }
    print(f"\nmodel CA-RMSD (multi-step)   = {res['ca_rmsd_model']:.3f} Å")
    print(f"persistence CA-RMSD          = {res['ca_rmsd_persistence']:.3f} Å")
    print(f"bond violation fraction      = {res['bond_violation_fraction']:.4f}")
    print(f"model acf lag1 / decay tau   = {kinetics['autocorr_model'].get('acf_lag1', float('nan')):.3f} / "
          f"{kinetics['autocorr_model'].get('decay_time_frames', float('nan')):.2f} frames")
    print(f"true  acf lag1 / decay tau   = {kinetics['autocorr_true'].get('acf_lag1', float('nan')):.3f} / "
          f"{kinetics['autocorr_true'].get('decay_time_frames', float('nan')):.2f} frames")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
