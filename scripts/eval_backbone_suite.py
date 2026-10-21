"""Backbone (N, CA, C, O) evaluation suite (TASK 1, NEW 2026-10-21).

Mirror of ``scripts/eval_ensemble_metrics_suite.py`` but scores the **full
mainchain** instead of only Cα, for three ensembles (model / persistence /
oracle):

    backbone RMSD (all-atom Kabsch)        -> lower better
    per-atom RMSD (N / CA / C / O)          -> lower better
    bond/angle/dihedral validity fractions  -> higher better (from OpenFold ideals)
    Ramachandran allowed fraction           -> higher better
    omega trans fraction                    -> diagnostic
    nonlocal Cα clash fraction              -> lower better

Model-1 backbone comes from the frozen decoder's **official OpenFold**
reconstruction (``backbone_samples``).  The *true* future backbone is lifted from
the true Cα future frames via the same OpenFold lifter, so model and truth are on
the exact same footing (both OpenFold-placed).  Persistence = anchor backbone
repeated; oracle = the true future window's backbone.

This script never modifies the frozen decoding path.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.models.stage_c.openfold_ca_lifter import lift_ca_to_backbone
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _anchors, _targets_from,
)
from protein_jepa_frame.training.train_stage_b1a import _truncate_history
from protein_jepa_frame.training.train_utils import _device, _move_batch

from evaluation.backbone_metrics import (
    backbone_geometry_summary, backbone_rmsd, backbone_per_atom_rmsd,
)


def _lift_true_backbone(true_ca: torch.Tensor, aatype: torch.Tensor, mask: torch.Tensor):
    """Lift true future Cα [B,T,L,3] to backbone [B,T,L,4,3] via OpenFold.

    Returns backbone in a flattened [B*T, L, 4, 3] view and matching aatype/mask.
    """
    B, T, L, _ = true_ca.shape
    ca = true_ca.reshape(B * T, L, 3)
    aa = aatype[:, None].expand(B, T, L).reshape(B * T, L)
    bb, _, _ = lift_ca_to_backbone(ca, aatype=aa)
    return bb, aa


def _score_backbone(pred_bb, true_bb, mask, meta):
    """One (pred [P,L,4,3], true [P,L,4,3]) pair -> scalar dict."""
    out = {}
    out["backbone_rmsd"] = backbone_rmsd(pred_bb, true_bb, mask)
    out.update(backbone_per_atom_rmsd(pred_bb, true_bb, mask))
    g = backbone_geometry_summary(pred_bb[0], mask,
                                  chain_id=meta.get("chain_id"),
                                  residue_index=meta.get("residue_index"))
    for k in ("bond_N_CA_mean", "bond_CA_C_mean", "bond_C_O_mean",
              "bond_N_CA_valid_frac", "bond_CA_C_valid_frac", "bond_C_O_valid_frac",
              "bond_C_N_mean", "bond_C_N_valid_frac",
              "angle_N_CA_C_mean_deg", "angle_N_CA_C_valid_frac",
              "ra_allowed_frac", "omega_trans_frac", "ca_nonlocal_clash_frac"):
        out.update({f"geo_{k}": g[k]})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-ckpt", default=None)
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--dist-bins", type=int, default=40)
    p.add_argument("--dist-lo", type=float, default=0.0)
    p.add_argument("--dist-hi", type=float, default=40.0)
    p.add_argument("--min-seq-sep", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=256)
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
    print(f"[backbone-suite] cond_dim={cond_dim} n_samples={args.n_samples}")

    class P: pass
    pr = P()
    pr.history_frames = args.history_frames
    pr.frame_stride = args.frame_stride
    pr.paired_horizons_frames = args.horizons
    cfg = DataConfig(manifest_path=args.manifest, batch_size=args.batch_size,
                     num_workers=args.num_workers, training_mode="paired_window",
                     num_frames=args.history_frames, paired_all_horizons=False,
                     max_residues=args.max_residues)
    apply_protocol(cfg, protocol_from_args(pr))
    ds = build_dataset(cfg)
    n_take = min(len(ds), args.batches * args.batch_size)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:n_take].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=lambda s: _truncate_history(collate_paired_windows(s),
                                                           args.history_frames))

    model_v, pers_v, orac_v = defaultdict(list), defaultdict(list), defaultdict(list)
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.batches:
                break
            n_batches = i + 1
            paired = _move_batch(batch, dev)
            _targets, fut, fmask = _targets_from(paired, args, dev)
            true_ca = fut[:, -1, :, 1, :]                       # [B,L,3] CA@horizon
            mask = fmask[:, -1]                                 # [B,L]
            aatype = paired["future_aatype"].to(dev)
            ri = paired.get("future_residue_index")
            ci = paired.get("future_chain_id")
            ctx = paired["context_coords"].to(dev)
            ctx_mask = paired["context_residue_mask"].to(dev)[:, -1]
            ri_c = paired["future_residue_index"].to(dev)
            ci_c = paired["future_chain_id"].to(dev)
            R_a, t_a = _anchors(paired, dev)
            cond = None
            if cond_dim > 0:
                ht = paired.get("horizon_time_ps")
                B = ctx.shape[0]
                t_norm = (torch.log(ht.to(dev).clamp_min(1.0)) /
                          torch.log(torch.tensor(1e4, device=dev))) if ht is not None \
                    else torch.zeros(B, device=dev)
                temp = paired.get("temperature_k")
                T_norm = ((temp.to(dev).float() - 300.0) / 150.0) if temp is not None \
                    else torch.zeros(B, device=dev)
                parts = [t_norm, T_norm]
                cond = torch.stack(parts[:cond_dim], dim=-1) if cond_dim <= 2 else \
                    torch.stack(parts + [torch.zeros_like(t_norm)] * (cond_dim - 2), dim=-1)
            out = producer.sample_future_ca(ctx, ctx_mask, aatype, ri_c, ci_c,
                                            R_a, t_a, n_samples=args.n_samples, cond=cond)
            model_bb = out["backbone_samples"]                  # [B,n,L,4,3]

            # True backbone (lifted) for the horizon frame.
            true_bb, _ = _lift_true_backbone(true_ca[:, None], aatype, mask[:, None])
            true_bb = true_bb.reshape(true_ca.shape[0], 1, true_ca.shape[1], 4, 3)

            # Persistence backbone: anchor backbone lifted from anchor CA, repeated.
            anchor_ca = ctx[:, -1, :, 1, :]                     # [B,L,3]
            pers_bb, _, _ = lift_ca_to_backbone(anchor_ca, aatype=aatype)
            pers_bb = pers_bb[:, None].expand(-1, args.n_samples, -1, -1, -1)

            for b in range(ctx.shape[0]):
                m = mask[b]
                meta = {"chain_id": ci_c[b], "residue_index": ri_c[b]}
                tbb = true_bb[b].expand(args.n_samples, -1, -1, -1)
                sc = _score_backbone(model_bb[b], tbb, m, meta)
                for k, v in sc.items():
                    if v is not None:
                        model_v[k].append(v)
                sc = _score_backbone(pers_bb[b], tbb, m, meta)
                for k, v in sc.items():
                    if v is not None:
                        pers_v[k].append(v)
                sc = _score_backbone(tbb, tbb, m, meta)
                for k, v in sc.items():
                    if v is not None:
                        orac_v[k].append(v)

    def agg(v):
        return {k: float(torch.tensor(x, dtype=torch.float64).mean()) for k, x in v.items()}

    res = {"manifest": args.manifest, "n_batches": n_batches,
           "n_samples_ensemble": args.n_samples, "cond_dim": cond_dim,
           "model": agg(model_v), "persistence": agg(pers_v), "oracle": agg(orac_v)}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"[backbone-suite] wrote {args.out}")


if __name__ == "__main__":
    main()
