#!/usr/bin/env python3
"""Export evaluation trajectories as PDB files (NEW file, 2026-10-20).

Produces, for a small set of fixed mdCATH validation windows, the CA (and, where
available, the original backbone N/CA/C/O) coordinates of

  * ``mdgen_zeroshot``   MDGen ``atlas.ckpt`` forward-sim rollout  (1 ns span)
  * ``mdgen_mdcath``     MDGen mdCATH short-train forward-sim rollout
  * ``ours_stageA``      our Model 1 (Stage-A ensemble producer, autoregressive)
  * ``ours_fromnone``    our Model 2 (from-none GLE, autoregressive)
  * ``ground_truth``     the real MD future frames for the same window
  * ``persistence``      the anchor frame repeated T times

so that the numbers in the report can be inspected by eye / in PyMOL.

This script ONLY reads frozen models and writes PDBs.  It never touches or
modifies the precise-atom-coordinate decoding path.

PDB convention
--------------
* one MODEL/ENDMDL block per frame (a real multi-model NMR-style PDB),
* only CA atoms are written (``ATOM  ... CA  ALA A``); predictions do not carry
  side chains, and writing only CA keeps GT / pred on the exact same footing,
* residue numbering is the window's `residue_index`, chain id from `chain_id`.

Usage
-----
    python scripts/export_eval_trajectories_pdb.py \
        --outdir artifacts/pdbs_20261020 \
        --n-windows 3 --n-frames 16 --n-samples 4

Only the pieces named via ``--which`` are run, so the slow MDGen rollouts can be
skipped (``--which ours_stageA ours_fromnone ground_truth persistence``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# MDGen source lives outside the repo; put it on the path so mdgen_baseline shims
# (which import `mdgen.*`) can be applied.  Harmless if the dir is absent.
if Path("/liuyutian/mdgen_src/mdgen-master").exists():
    sys.path.insert(0, "/liuyutian/mdgen_src/mdgen-master")

# 20 standard three-letter codes, indexed by the repo one-letter alphabet
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR", "X": "GLY",
}

ALL_MODELS = ["ours_stageA", "ours_hs", "ours_fromnone", "mdgen_zeroshot", "mdgen_mdcath",
              "ground_truth", "persistence"]


# --------------------------------------------------------------------------- #
# PDB writer
# --------------------------------------------------------------------------- #
def kabsch_align_to_ref(frames_ca, ref_ca):
    """Superpose every frame in ``frames_ca`` [T,L,3] onto ``ref_ca`` [L,3].

    Returns aligned frames, all sharing the reference frame: each frame gets the
    optimal rotation+translation (Kabsch on the CA coordinates) that maps it onto
    ``ref_ca``.  This is the right visualisation for "does the shape change, in a
    common frame" -- but it *deliberately removes* global translation/rotation, so
    it must NOT be used for metrics that care about absolute placement.
    """
    P = np.asarray(ref_ca, dtype=np.float64)
    Pc = P - P.mean(0)
    out = np.empty_like(np.asarray(frames_ca, dtype=np.float64))
    for t, X in enumerate(np.asarray(frames_ca, dtype=np.float64)):
        Xc = X - X.mean(0)
        H = Xc.T @ Pc
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T                      # maps Xc -> Pc
        # superpose onto the anchor AND move the anchor centroid to the origin,
        # so the molecule sits centred in the viewer.
        out[t] = (Xc @ R.T)
    return out


def write_ca_pdb(path: Path, frames_ca, aatype, residue_index, chain_id,
                 origin: str):
    """Write ``frames_ca`` [T, L, 3] as a multi-MODEL CA-only PDB.

    ``aatype`` is the repo's one-letter-index array [L]; ``residue_index`` [L];
    ``chain_id`` [L].
    """
    frames_ca = np.asarray(frames_ca, dtype=np.float64)
    T, L, _ = frames_ca.shape
    lines = [f"REMARK 100 ORIGIN {origin}",
             f"REMARK 100 NFRAMES {T}", f"REMARK 100 NRESIDUES {L}"]
    serial = 1
    for t in range(T):
        lines.append(f"MODEL     {t + 1:4d}")
        for i in range(L):
            aa = AA_ALPHABET[int(aatype[i])] if 0 <= int(aatype[i]) < len(AA_ALPHABET) else "X"
            resn = THREE.get(aa, "GLY")
            ch = "A"
            if chain_id is not None:
                c = str(int(chain_id[i]))
                ch = c if c.isalpha() else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:26][int(chain_id[i]) % 26]
            resi = int(residue_index[i]) if residue_index is not None else i + 1
            x, y, z = frames_ca[t, i]
            lines.append(
                "ATOM  {ser:5d}  CA  {resn:>3s} {ch:1s}{resi:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C".format(
                    ser=serial, resn=resn, ch=ch, resi=resi, x=x, y=y, z=z))
            serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# window selection (shared, deterministic)
# --------------------------------------------------------------------------- #
def build_protocol(args):
    from evaluation.data import EvalProtocol
    return EvalProtocol(
        manifest=args.manifest,
        horizons_frames=(args.horizon,),
        context_frames=args.history_frames,
        context_stride=1, frame_stride=1,
        max_residues=args.max_residues,
        max_proteins=args.n_windows,
        max_anchors_per_trajectory=1,
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False,
        seed=args.seed,
    )


def collect_windows(args):
    """Return a small fixed list of windows (one anchor per protein)."""
    from evaluation.data import iter_windows
    proto = build_protocol(args)
    wins = []
    seen = set()
    for w in iter_windows(proto):
        if w.protein_id in seen:
            continue
        seen.add(w.protein_id)
        wins.append(w)
        if len(wins) >= args.n_windows:
            break
    return wins


def window_meta(w, dev):
    rmask = w.context["residue_mask"].all(dim=0).bool()
    idx = np.where(rmask.cpu().numpy())[0]
    aatype = w.context["aatype"]
    aatype = aatype[-1] if aatype.ndim == 2 else aatype
    aatype = aatype.cpu().numpy()[idx]
    ri = w.context.get("residue_index")
    ri = ri.cpu().numpy()[idx] if ri is not None else np.arange(len(idx))
    ci = w.context.get("chain_id")
    ci = ci.cpu().numpy()[idx] if ci is not None else None
    anchor_ca = w.context["coords"][-1, :, 1, :].cpu().numpy()[idx]   # [L,3]
    true_ca = w.future_path["coords"][:, :, 1, :].cpu().numpy()[:, idx]  # [T,L,3]
    return rmask, idx, aatype, ri, ci, anchor_ca, true_ca


# --------------------------------------------------------------------------- #
# our models
# --------------------------------------------------------------------------- #
def load_ours_stageA(args, dev):
    from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
    from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    ck = torch.load(args.ensemble_ckpt, map_location=dev, weights_only=False)
    ck_args = ck.get("args", {})
    head = GaussianHHead(dH, hidden=int(ck_args.get("head_hidden", 128)),
                         cond_dim=int(ck_args.get("cond_dims", 0)))
    head.load_state_dict(ck["head"])
    head.to(dev).eval()
    producer = EnsembleProducer(encoder, decoder, head).to(dev).eval()
    for p in producer.parameters():
        p.requires_grad_(False)
    return producer, int(ck_args.get("cond_dims", 0))


def load_ours_hs(args, dev):
    """Model-1 frozen stack + Stage-B H+S (GLE) kernel producer."""
    from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
    from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
    from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, dev)
    kck = torch.load(args.kernel_ckpt, map_location=dev, weights_only=False)
    k_cfg = kck.get("kernel_config") or {}
    k_args = kck.get("args", {})
    k_cfg["step_ps"] = k_args.get("step_ps", k_cfg.get("step_ps", 1000.0))
    k_cfg["rollout_grad_steps"] = None
    kernel = StageB1b(StageB1bConfig(**k_cfg)).to(dev)
    kernel.load_state_dict(kck["kernel"])
    kernel.eval()
    producer = HSEnsembleProducer(encoder, decoder, kernel).to(dev).eval()
    producer._rollout_steps = int(k_args.get("rollout_steps", 1))
    producer._rollout_grad_steps = None
    producer._temperature_k = float(k_args.get("temperature_k", 320.0))
    for p in producer.parameters():
        p.requires_grad_(False)
    return producer


def load_ours_fromnone(args, dev):
    from protein_jepa_frame.training.train_stage_cb1b import load_models_for_eval
    return load_models_for_eval(args.fromnone_ckpt, dev)


@torch.no_grad()
def rollout_ours_stageA(producer, w, args, dev, cond_dim):
    """Autoregressive CA rollout for the Stage-A producer. Returns [T,L,3]."""
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    rmask, idx, aatype, ri, ci, anchor_ca, _ = window_meta(w, dev)
    ctx = w.context["coords"][None].to(dev)             # [1,K,L,4,3]
    aat = torch.from_numpy(aatype)[None].to(dev)
    rit = torch.from_numpy(ri)[None].to(dev)
    cit = torch.from_numpy(ci)[None].to(dev) if ci is not None else torch.zeros_like(rit)
    m = torch.from_numpy(rmask.cpu().numpy())[None].to(dev)
    cond = None
    if cond_dim > 0:
        cond = torch.zeros(1, cond_dim, device=dev)
    K = ctx.shape[1]
    buf = ctx.clone()
    out_frames = []
    for _ in range(args.n_frames):
        R_a, t_a = _extract_backbone_frames(buf[:, -1])
        res = producer.sample_future_ca(buf, m, aat, rit, cit, R_a, t_a,
                                        n_samples=args.n_samples, cond=cond)
        ca = res["ca_samples"].mean(dim=1)               # [1,L,3]
        out_frames.append(ca[0].cpu())
        off = buf[:, -1] - buf[:, -1, :, 1:2, :]
        new_frame = ca[:, :, None, :] + off
        buf = torch.cat([buf[:, 1:], new_frame[:, None]], dim=1)
    traj = torch.stack(out_frames, dim=0).numpy()        # [T,L_all,3]
    return traj[:, idx]


@torch.no_grad()
def ensemble_ours_stageA(producer, w, args, dev, cond_dim, n_samples):
    """Single-step ensemble from Stage-A producer (= the reported metric input).

    Returns [n_samples, L, 3]: N independent future CA conformers decoded from the
    anchor context in ONE forward pass.  No self-feedback -> no rollout drift.
    """
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    rmask, idx, aatype, ri, ci, anchor_ca, _ = window_meta(w, dev)
    ctx = w.context["coords"][None].to(dev)
    aat = torch.from_numpy(aatype)[None].to(dev)
    rit = torch.from_numpy(ri)[None].to(dev)
    cit = torch.from_numpy(ci)[None].to(dev) if ci is not None else torch.zeros_like(rit)
    m = torch.from_numpy(rmask.cpu().numpy())[None].to(dev)
    cond = torch.zeros(1, cond_dim, device=dev) if cond_dim > 0 else None
    R_a, t_a = _extract_backbone_frames(ctx[:, -1])
    res = producer.sample_future_ca(ctx, m, aat, rit, cit, R_a, t_a,
                                    n_samples=n_samples, cond=cond)
    ca = res["ca_samples"][0].cpu().numpy()              # [n_samples, L_all, 3]
    return ca[:, idx]


@torch.no_grad()
def ensemble_ours_hs(producer, w, args, dev, n_samples):
    """Single-step ensemble from the Stage-B H+S (GLE) producer."""
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    rmask, idx, aatype, ri, ci, anchor_ca, _ = window_meta(w, dev)
    ctx = w.context["coords"][None].to(dev)
    aat = torch.from_numpy(aatype)[None].to(dev)
    rit = torch.from_numpy(ri)[None].to(dev)
    cit = torch.from_numpy(ci)[None].to(dev) if ci is not None else torch.zeros_like(rit)
    m = torch.from_numpy(rmask.cpu().numpy())[None].to(dev)
    tk = float(getattr(producer, "_temperature_k", 320.0))
    R_a, t_a = _extract_backbone_frames(ctx[:, -1])
    res = producer.sample_future_ca(ctx, m, aat, rit, cit, R_a, t_a,
                                    n_samples=n_samples, temperature_k=tk)
    ca = res["ca_samples"][0].cpu().numpy()
    return ca[:, idx]


@torch.no_grad()
def rollout_ours_hs(producer, w, args, dev):
    """Autoregressive CA rollout for the Stage-B H+S (GLE) producer. [T,L,3]."""
    from protein_jepa_frame.training.train_stage_ensemble import _extract_backbone_frames
    rmask, idx, aatype, ri, ci, anchor_ca, _ = window_meta(w, dev)
    ctx = w.context["coords"][None].to(dev)
    aat = torch.from_numpy(aatype)[None].to(dev)
    rit = torch.from_numpy(ri)[None].to(dev)
    cit = torch.from_numpy(ci)[None].to(dev) if ci is not None else torch.zeros_like(rit)
    m = torch.from_numpy(rmask.cpu().numpy())[None].to(dev)
    tkw = float(getattr(producer, "_temperature_k", 320.0))
    buf = ctx.clone()
    out_frames = []
    for _ in range(args.n_frames):
        R_a, t_a = _extract_backbone_frames(buf[:, -1])
        res = producer.sample_future_ca(buf, m, aat, rit, cit, R_a, t_a,
                                        n_samples=args.n_samples, temperature_k=tkw)
        ca = res["ca_samples"].mean(dim=1)
        out_frames.append(ca[0].cpu())
        off = buf[:, -1] - buf[:, -1, :, 1:2, :]
        new_frame = ca[:, :, None, :] + off
        buf = torch.cat([buf[:, 1:], new_frame[:, None]], dim=1)
    traj = torch.stack(out_frames, dim=0).numpy()
    return traj[:, idx]


@torch.no_grad()
def rollout_ours_fromnone(models, w, args, dev):
    """Autoregressive CA rollout for the from-none GLE model. Returns [T,L,3]."""
    from evaluation.run_eval_fromnone_ensemble import _encode_history
    encoder, kernel, decoder, normalizer, ck = models
    step_ps = float(kernel.cfg.step_ps)
    rmask, idx, _, _, _, _, _ = window_meta(w, dev)
    ctx = w.context["coords"][None].to(dev)              # [1,K,L,4,3]
    aat = w.context["aatype"]
    aat = (aat[-1] if aat.ndim == 2 else aat)[None].to(dev)
    ri = w.context.get("residue_index")
    ri = (ri if ri is not None else torch.arange(ctx.shape[2]))[None].to(dev)
    ci = w.context.get("chain_id")
    ci = (ci if ci is not None else torch.zeros(ctx.shape[2], dtype=torch.long))[None].to(dev)
    m1 = rmask[None].to(dev)
    K, L = ctx.shape[1], ctx.shape[2]
    mK = m1[:, None].expand(1, K, L)
    temp = w.context.get("temperature_k")
    if torch.is_tensor(temp):
        temp = temp.to(dev)[None]
    tp = w.context.get("time_ps")
    dt_hist = None
    if tp is not None:
        tp = tp.to(dev)[None]
        dt = tp.diff(dim=-1)
        dt_hist = torch.cat([dt[:, :1], dt], dim=-1)
    H_hist = _encode_history(encoder, normalizer, ctx, mK, aat, ri, ci, dev)
    H0 = H_hist[:, -1]
    from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
    anchor_bb = ctx[:, -1]
    R_anc, t_anc = RigidBodyFrame.from_backbone(
        anchor_bb[..., 0, :], anchor_bb[..., 1, :], anchor_bb[..., 2, :])
    # one rollout, but with distinct latent noise per sample
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    traj = kernel.rollout(H0, m1, temp, n_steps=args.n_frames,
                          H_hist=H_hist, dt_hist=dt_hist,
                          seq_feature=H0, generator=gen, detach=True, return_bath=True)
    Hf = traj[0]                                         # [1, Tf, L, d]
    Sf = traj[1]                                         # [1, Tf, L, ds] or None
    Tf = Hf.shape[1]
    X_ctx = ctx[:, -1, :, 1, :]                           # anchor CA [1,L,3]
    ca_all = []
    for t in range(Tf):
        Ht = Hf[:, t]
        St = Sf[:, t] if Sf is not None else None
        out = decoder(H_future=Ht, H_context=H0, X_context=X_ctx,
                      residue_mask=m1, R_anchor=R_anc, S_future=St)
        ca_all.append(out["ca_pred"][0].cpu())
    return torch.stack(ca_all, dim=0).numpy()[:, idx]


# --------------------------------------------------------------------------- #
# MDGen
# --------------------------------------------------------------------------- #
def mdgen_rollout(ckpt, w, args, dev):
    """MDGen forward-sim over the window; returns last-frame CA [L,3] plus the
    full trajectory [T,L,3] is NOT produced (MDGen returns atom14 per frame, we
    keep CA of the final frame, matching the metric script)."""
    import mdgen_baseline.mdgen_compat as mdgen_compat
    mdgen_compat.apply()
    from mdgen.wrapper import NewMDGenWrapper
    from mdgen.residue_constants import restype_order as RO
    from mdgen.geometry import atom14_to_frames
    from mdgen.dataset import restype_order as RO2  # noqa

    model = NewMDGenWrapper.load_from_checkpoint(ckpt, map_location=dev)
    model = model.to(dev).eval()
    m_args = model.args
    m_args.sim_condition = True
    if args.num_frames_mdgen:
        m_args.num_frames = int(args.num_frames_mdgen)
    if args.sampling_method:
        m_args.sampling_method = args.sampling_method
    if args.sampling_steps:
        mdgen_compat.set_ode_steps(args.sampling_steps)

    rmask, idx, _, _, _, _, _ = window_meta(w, dev)
    # sequence from aatype + atom14 anchor from N/CA/C/O (place side chains on CA)
    aat = w.context["aatype"]
    aat = (aat[-1] if aat.ndim == 2 else aat).cpu().numpy()[idx]
    seq = np.array([RO.get(AA_ALPHABET[int(a)], 20) if 0 <= int(a) < 20 else 20
                    for a in aat], dtype=np.int64)
    bb = w.context["coords"][-1].cpu().numpy()[idx]       # [L,4,3]
    L0 = bb.shape[0]
    a14 = np.zeros((L0, 14, 3), dtype=np.float32)
    a14[:, 0] = bb[:, 0]; a14[:, 1] = bb[:, 1]
    a14[:, 2] = bb[:, 2]; a14[:, 3] = bb[:, 3]
    for k in range(4, 14):
        a14[:, k] = bb[:, 1]

    T = int(m_args.num_frames)
    crop = args.crop
    a14b = a14[None].repeat(args.n_samples, axis=0)
    seqb = seq.copy()
    if L0 > crop:
        a14b, seqb = a14b[:, :crop], seqb[:crop]
    elif L0 < crop:
        pad = crop - L0
        a14b = np.pad(a14b, ((0, 0), (0, pad), (0, 0), (0, 0)))
        seqb = np.concatenate([seqb, np.zeros(pad, dtype=np.int64)])
    Lc = crop
    a14_t = torch.from_numpy(a14b).float()
    a14_T = a14_t[:, None].expand(-1, T, -1, -1, -1).contiguous()
    flat = a14_T.reshape(-1, Lc, 14, 3)
    frames = atom14_to_frames(flat)
    seqres = torch.from_numpy(seqb).long()
    torsion = torch.zeros(flat.shape[0], Lc, 7, 2)
    batch = {
        "name": [f"exp_{i}" for i in range(args.n_samples)],
        "frame_start": [0] * args.n_samples,
        "trans": frames._trans.reshape(args.n_samples, T, Lc, 3).to(dev),
        "rots": frames._rots._rot_mats.reshape(args.n_samples, T, Lc, 3, 3).to(dev),
        "torsions": torsion.reshape(args.n_samples, T, Lc, 7, 2).to(dev),
        "torsion_mask": torch.ones(args.n_samples, Lc, 7).to(dev),
        "seqres": seqres[None].expand(args.n_samples, -1).to(dev),
        "mask": torch.ones(args.n_samples, Lc, dtype=torch.float32).to(dev),
    }
    with torch.no_grad():
        atom14_out, _ = model.inference(batch)             # [B,T,Lc,14,3]
    # mean over samples -> a single trajectory [T, L0, 3]
    ca_traj = atom14_out[:, :, :L0, 1, :].detach().float().cpu().numpy()  # [B,T,L0,3]
    return ca_traj.mean(axis=0)                            # [T, L0, 3]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="artifacts/pdbs_20261020")
    ap.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    ap.add_argument("--which", nargs="+", default=ALL_MODELS)
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--n-frames", type=int, default=16,
                    help="frames to write per trajectory")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--align-to-anchor", action="store_true",
                    help="Kabsch-superpose each frame onto the anchor CA and "
                         "centre at the origin (visual comparison only; removes "
                         "global translation/rotation)")
    ap.add_argument("--ensemble", action="store_true",
                    help="export the SINGLE-STEP ensemble (what the reported SOTA "
                         "metrics measure) instead of the autoregressive rollout")
    ap.add_argument("--ensemble-samples", type=int, default=32,
                    help="number of ensemble members per window in --ensemble mode")
    ap.add_argument("--horizon", type=int, default=64,
                    help="future horizon (frames) whose full future_path is saved as GT")
    ap.add_argument("--history-frames", type=int, default=8)
    ap.add_argument("--max-residues", type=int, default=256)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--num-frames-mdgen", type=int, default=100)
    ap.add_argument("--sampling-method", default="euler")
    ap.add_argument("--sampling-steps", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage-a-ckpt", default="artifacts/stage_a_disp_relpos/checkpoint_latest.pt")
    ap.add_argument("--b1a-ckpt", default="artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt")
    ap.add_argument("--ensemble-ckpt", default="artifacts/stage_ensemble_1000dom/checkpoint_best.pt")
    ap.add_argument("--kernel-ckpt", default="artifacts/stage_hs_1000dom/checkpoint_best.pt",
                    help="Stage-B H+S (GLE) kernel ckpt for the ours_hs trajectory")
    ap.add_argument("--fromnone-ckpt", default="artifacts/fromnone/stage_cb1b_v4e/checkpoint_latest.pt")
    ap.add_argument("--mdgen-zeroshot-ckpt", default="/liuyutian/mdgen_run/hf_mdgen/atlas.ckpt")
    ap.add_argument("--mdgen-mdcath-ckpt",
                    default="/liuyutian/mdgen_src/mdgen-master/workdir/mdcath_short/epoch=1-step=2000.ckpt")
    args = ap.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    wins = collect_windows(args)
    print(f"[export] {len(wins)} windows: {[w.protein_id for w in wins]}")

    # lazily construct models only if requested
    oursA = oursHS = oursF = None
    if "ours_stageA" in args.which:
        oursA = load_ours_stageA(args, dev)
    if "ours_hs" in args.which:
        oursHS = load_ours_hs(args, dev)
    if "ours_fromnone" in args.which:
        oursF = load_ours_fromnone(args, dev)

    index = []
    for wi, w in enumerate(wins):
        rmask, idx, aatype, ri, ci, anchor_ca, true_ca = window_meta(w, dev)
        tag = f"{w.protein_id}_{w.trajectory_id.split('_T')[-1]}_anchor{w.anchor_frame}"
        wdir = outdir / tag
        wdir.mkdir(exist_ok=True, parents=True)
        Lval = len(idx)
        print(f"[export] window {wi+1}/{len(wins)} {tag} L={Lval}")

        def _save(name, frames_ca, origin):
            arr = np.asarray(frames_ca)[:args.n_frames]
            if getattr(args, "align_to_anchor", False):
                # Superpose every frame onto the anchor CA and show in a shared
                # frame (also re-centres at the anchor centroid).  Makes all
                # trajectory types mutually overlayable in PyMOL/VMD.
                arr = kabsch_align_to_ref(arr, anchor_ca)
                origin = origin + " [Kabsch-aligned to anchor]"
            write_ca_pdb(wdir / f"{name}.pdb", arr, aatype, ri, ci, origin)

        def _save_ens(name, members_ca, origin):
            arr = np.asarray(members_ca)
            if getattr(args, "align_to_anchor", False):
                arr = kabsch_align_to_ref(arr, anchor_ca)
                origin = origin + " [Kabsch-aligned to anchor]"
            write_ca_pdb(wdir / f"{name}.pdb", arr, aatype, ri, ci, origin)

        # ground truth + persistence always cheap
        if "ground_truth" in args.which:
            _save("ground_truth", true_ca, "real MD future frames")
        if "persistence" in args.which:
            T = min(args.n_frames, true_ca.shape[0])
            _save("persistence", np.repeat(anchor_ca[None], T, axis=0),
                  "anchor frame repeated")

        # ---- ENSEMBLE mode (single-step, = the reported SOTA metrics) ----
        # The report's metrics are computed from ONE forward `sample_future_ca`
        # call per window: an ensemble of N future conformers conditioned on the
        # anchor.  This is NOT a rollout and has no self-feedback drift.  When
        # `--ensemble` is set we write exactly those conformers as a multi-MODEL
        # PDB so the SOTA numbers become directly visualisable.
        if getattr(args, "ensemble", False):
            N = args.ensemble_samples
            if "ground_truth" in args.which:
                _save_ens("ground_truth", true_ca,
                          "real MD future frames (oracle ensemble)")
            if "persistence" in args.which:
                _save_ens("persistence", np.repeat(anchor_ca[None], N, axis=0),
                          "anchor frame repeated (persistence ensemble)")
            if oursA is not None:
                ens = ensemble_ours_stageA(oursA[0], w, args, dev, oursA[1], N)
                _save_ens("ours_stageA", ens,
                          "Stage-A ensemble producer (single-step ensemble)")
            if oursHS is not None:
                ens = ensemble_ours_hs(oursHS, w, args, dev, N)
                _save_ens("ours_hs", ens,
                          "Stage-B H+S (GLE) producer (single-step ensemble)")
            index.append({"tag": tag, "protein_id": w.protein_id,
                          "trajectory_id": w.trajectory_id,
                          "anchor_frame": int(w.anchor_frame), "L": int(Lval),
                          "temperature_k": float(w.temperature_k),
                          "mode": "ensemble"})
            continue

        # ---- ROLLOUT mode (autoregressive self-feedback; NOT the SOTA metric) ----
        if "ours_stageA" in args.which:
            tr = rollout_ours_stageA(oursA[0], w, args, dev, oursA[1])
            _save("ours_stageA", tr, "Stage-A ensemble producer (autoregressive)")
        if "ours_hs" in args.which:
            tr = rollout_ours_hs(oursHS, w, args, dev)
            _save("ours_hs", tr, "Stage-B H+S (GLE) producer (autoregressive)")
        if "ours_fromnone" in args.which:
            tr = rollout_ours_fromnone(oursF, w, args, dev)
            _save("ours_fromnone", tr, "from-none GLE (autoregressive)")
        if "mdgen_zeroshot" in args.which:
            tr = mdgen_rollout(args.mdgen_zeroshot_ckpt, w, args, dev)
            _save("mdgen_zeroshot", tr, "MDGen atlas.ckpt forward-sim")
        if "mdgen_mdcath" in args.which:
            tr = mdgen_rollout(args.mdgen_mdcath_ckpt, w, args, dev)
            _save("mdgen_mdcath", tr, "MDGen mdCATH short-train forward-sim")

        index.append({"tag": tag, "protein_id": w.protein_id,
                      "trajectory_id": w.trajectory_id,
                      "anchor_frame": int(w.anchor_frame), "L": int(Lval),
                      "temperature_k": float(w.temperature_k)})

    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[export] wrote {len(index)} windows to {outdir}")


if __name__ == "__main__":
    main()
