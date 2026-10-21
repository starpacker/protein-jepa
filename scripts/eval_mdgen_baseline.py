"""MDGen baseline evaluation on mdCATH (NEW, 2026-10-20).

Treats a trained MDGen forward-simulation model as a *future-frame ensemble
generator*: given the anchor frame of a val window it samples ``--n-samples``
future frames, extracts CA traces, and scores them with the SAME metric family
used for our model in ``scripts/eval_ensemble_metrics_suite.py``:

    RMSF magnitude (L1)          rmsf_mae        lower better
    RMSF-r (profile corr)        rmsf_corr       higher better
    ROOT MEAN W2                 root_mean_w2    lower better
    VARIANCE W2                  variance_w2     lower better
    MD-PCA W2                    md_pca_w2       lower better
    WEAK CONTACT J               weak_contact_j  higher better
    TRANSIENT CONTACTS J         transient_contacts_j  higher better

plus two position metrics on the CA trace:

    Kabsch CA RMSD               common-anchor RMSD (the fitted anchor->target
                                 transform is computed from REAL coords only);
    aligned CA RMSD              independently superposed shape RMSD.

mdCATH units: the HDF5 stores Angstrom coordinates; we feed MDGen a
``[B, T, L, 14, 3]`` atom14 array whose frame 0 is the anchor and whose later
frames duplicate the anchor.  MDGen's internal nm conversion is a global length
scale which cancels in the Kabsch fits; positions are compared in Angstrom
after sampling.

This script only *reads* a checkpoint and frozen data; it never touches the
precise-atom-coordinate decoding path.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path("/liuyutian/mdgen_src/mdgen-master")))

import mdgen_baseline.mdgen_compat as mdgen_compat  # noqa: E402  (apply shims first)

from evaluation.ensemble_metrics_ext import (  # noqa: E402
    extended_ensemble_summary, rmsf_profile_aligned,
)
from evaluation.geometry import common_anchor_rmsd, aligned_rmsd  # noqa: E402

METRIC_DIRS = {
    "rmsf_mae": False, "rmsf_corr": True, "root_mean_w2": False,
    "variance_w2": False, "md_pca_w2": False,
    "weak_contact_j": True, "transient_contacts_j": True,
    "kabsch_ca_rmsd": False, "aligned_ca_rmsd": False,
}
_CA = 1  # atom14 index of CA


def _seq_to_indices(seq: str, restype_order: dict):
    return np.array([restype_order.get(c, 20) for c in seq], dtype=np.int64)


def _sample_manifest(manifest_path, split_csv, n_windows, horizon, history,
                     stride, seed, max_residues, data_dir):
    """Pick val windows whose ``<trajectory_id>.npy`` atom14 files exist."""
    seqs = {}
    if split_csv:
        import pandas as pd
        df = pd.read_csv(split_csv, index_col="name")
        seqs = df["seqres"].to_dict()

    recs = [json.loads(l) for l in open(manifest_path) if l.strip()]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    data_dir = Path(data_dir)
    out = []
    for k in order:
        r = recs[int(k)]
        tid = r["trajectory_id"]
        npy = data_dir / f"{tid}.npy"
        if not npy.exists():
            continue
        arr = np.load(npy, mmap_mode="r")                # [F, L, 14, 3]
        F, L = arr.shape[0], arr.shape[1]
        need = (history - 1) * stride + horizon + 1
        if F < need or L > max_residues or L < 3:
            continue
        start = int(rng.integers(0, F - need + 1))
        f0 = start + (history - 1) * stride
        tgt = f0 + horizon
        anchor = np.asarray(arr[f0], dtype=np.float32)   # [L,14,3]
        true_ca = np.asarray(arr[tgt][:, _CA, :], dtype=np.float32)
        seq = seqs.get(tid)
        if seq is None or len(seq) != L:
            seq = "A" * L                                # fallback
        out.append({"trajectory_id": tid, "domain_id": r["domain_id"],
                    "anchor": anchor, "true_ca": true_ca, "seq": seq,
                    "mask": np.ones(L, dtype=bool)})
        if len(out) >= n_windows:
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mdgen-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split-csv", default=None)
    p.add_argument("--data-dir", default="/liuyutian/mdgen_run/data_mdcath")
    p.add_argument("--device", default="auto")
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--history-frames", type=int, default=1)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--num-frames", type=int, default=None,
                   help="override T (default: from the checkpoint)")
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--pca-k", type=int, default=3)
    p.add_argument("--weak-lo", type=float, default=0.05)
    p.add_argument("--weak-hi", type=float, default=0.5)
    p.add_argument("--trans-lo", type=float, default=0.01)
    p.add_argument("--trans-hi", type=float, default=0.99)
    p.add_argument("--sampling-steps", type=int, default=0,
                   help="number of ODE steps for fixed-step solvers (euler)")
    p.add_argument("--sampling-method", default=None,
                   help="dopri5 (default, slow) | euler (fast)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)

    from mdgen.wrapper import NewMDGenWrapper
    from mdgen.residue_constants import restype_order
    model = NewMDGenWrapper.load_from_checkpoint(args.mdgen_ckpt, map_location=dev)
    model = model.to(dev).eval()
    m_args = model.args
    # force the sim_condition path even if the checkpoint default differs
    m_args.sim_condition = True
    m_args.num_frames = int(args.num_frames or m_args.num_frames)
    if args.sampling_method:
        m_args.sampling_method = args.sampling_method
    if args.sampling_steps:
        m_args.inference_steps = args.sampling_steps
        mdgen_compat.set_ode_steps(args.sampling_steps)
    print(f"[mdgen-eval] ckpt={args.mdgen_ckpt} T={m_args.num_frames} "
          f"crop={getattr(m_args,'crop',None)} sim_condition={m_args.sim_condition} "
          f"sampler={getattr(m_args,'sampling_method',None)}")

    wins = _sample_manifest(
        args.manifest, args.split_csv, args.n_windows, args.horizon,
        args.history_frames, args.frame_stride, args.seed,
        args.max_residues, args.data_dir)
    print(f"[mdgen-eval] {len(wins)} windows")

    from mdgen.dataset import restype_order as RO
    from mdgen.geometry import atom14_to_frames
    from mdgen.rigid_utils import Rigid

    sink = defaultdict(list)
    pred_glob, pred_tgt = [], defaultdict(list)
    true_glob, true_tgt = [], defaultdict(list)

    T = int(m_args.num_frames)
    crop = args.crop
    for wi, win in enumerate(wins):
        L0 = win["anchor"].shape[0]
        a14 = np.repeat(win["anchor"][None], args.n_samples, axis=0)
        seq = _seq_to_indices(win["seq"], RO)
        if L0 > crop:
            a14, seq = a14[:, :crop], seq[:crop]
        elif L0 < crop:
            pad = crop - L0
            a14 = np.pad(a14, ((0, 0), (0, pad), (0, 0), (0, 0)))
            seq = np.concatenate([seq, np.zeros(pad, dtype=np.int64)])
        Lc = crop

        a14_t = torch.from_numpy(a14).float()                       # [B,Lc,14,3]
        # MDGen wants Angstrom -> frames; duplicate the anchor over T
        a14_T = a14_t[:, None].expand(-1, T, -1, -1, -1).contiguous()
        flat = a14_T.reshape(-1, Lc, 14, 3)
        frames = atom14_to_frames(flat)                             # Rigid [B*T,Lc]
        seqres = torch.from_numpy(seq).long()
        aatype = seqres[None].expand(T, -1)
        torsion = torch.zeros(flat.shape[0], Lc, 7, 2)
        batch = {
            "name": [f"{win['trajectory_id']}_{i}" for i in range(args.n_samples)],
            "frame_start": [0] * args.n_samples,
            "trans": frames._trans.reshape(args.n_samples, T, Lc, 3).to(dev),
            "rots": frames._rots._rot_mats.reshape(args.n_samples, T, Lc, 3, 3).to(dev),
            "torsions": torsion.reshape(args.n_samples, T, Lc, 7, 2).to(dev),
            "torsion_mask": torch.ones(args.n_samples, Lc, 7).to(dev),
            "seqres": seqres[None].expand(args.n_samples, -1).to(dev),
            "mask": torch.ones(args.n_samples, Lc, dtype=torch.float32).to(dev),
        }
        with torch.no_grad():
            atom14_out, _ = model.inference(batch)                  # [B,T,Lc,14,3]
        ca = atom14_out[:, -1, :, _CA, :].detach().float()          # [B,Lc,3]
        ca = ca[:, :L0].cpu()                                       # drop padding
        mask = torch.from_numpy(win["mask"])
        true_ca = torch.from_numpy(win["true_ca"])                  # [L0,3]

        s = extended_ensemble_summary(
            ca, true_ca[None].expand(args.n_samples, -1, -1), mask,
            cutoff=args.cutoff, pca_k=args.pca_k,
            weak_lo=args.weak_lo, weak_hi=args.weak_hi,
            trans_lo=args.trans_lo, trans_hi=args.trans_hi)
        for k, v in s.items():
            sink[k].append(v)

        anchor_ca = torch.from_numpy(
            win["anchor"][:, _CA, :].astype(np.float32))
        # Kabsch CA RMSD (common-anchor) + aligned shape RMSD, averaged over samples
        ka = [common_anchor_rmsd(ca[i], anchor_ca, true_ca, mask)
              for i in range(ca.shape[0])]
        al = [aligned_rmsd(ca[i], true_ca, mask) for i in range(ca.shape[0])]
        sink["kabsch_ca_rmsd"].append(float(np.nanmean(ka)))
        sink["aligned_ca_rmsd"].append(float(np.nanmean(al)))

        tp = rmsf_profile_aligned(true_ca[None], mask)              # [L0]
        true_glob.append(tp); true_tgt[win["domain_id"]].append(tp)
        pp = rmsf_profile_aligned(ca, mask)
        pred_glob.append(pp); pred_tgt[win["domain_id"]].append(pp)

        if (wi + 1) % 8 == 0:
            print(f"  [{wi+1}/{len(wins)}] rmsf_mae={np.mean(sink['rmsf_mae']):.3f} "
                  f"kabsch={np.mean(sink['kabsch_ca_rmsd']):.3f}")

    def _corr(a, b):
        a, b = a.double(), b.double()
        fin = torch.isfinite(a) & torch.isfinite(b)
        a, b = a[fin], b[fin]
        if a.numel() < 2:
            return float("nan")
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))

    tg = torch.cat(true_glob)
    rmsf_r = {"global": _corr(torch.cat(pred_glob), tg)}
    per_t = {}
    for dom, vecs in pred_tgt.items():
        per_t[dom] = _corr(torch.cat(vecs), torch.cat(true_tgt[dom]))
    vals = [v for v in per_t.values() if v == v]
    rmsf_r["per_target"] = per_t
    rmsf_r["per_target_stats"] = (
        {"n_targets": len(vals), "mean": float(np.mean(vals)),
         "median": float(np.median(vals)), "min": float(np.min(vals)),
         "max": float(np.max(vals))} if vals else {"n_targets": 0})

    agg = {k: float(np.mean(v)) for k, v in sink.items()}
    res = {
        "model": "mdgen", "checkpoint": args.mdgen_ckpt,
        "manifest": args.manifest, "n_windows": len(wins),
        "n_samples_ensemble": args.n_samples, "num_frames": T,
        "metrics": agg, "rmsf_r": rmsf_r,
        "config": {"cutoff": args.cutoff, "pca_k": args.pca_k,
                   "weak_band": [args.weak_lo, args.weak_hi],
                   "transient_band": [args.trans_lo, args.trans_hi]},
    }
    print(json.dumps({k: agg[k] for k in METRIC_DIRS}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[mdgen-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
