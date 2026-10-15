"""Cross-PROTEIN held-out probe: does Delta_H predict Delta_CA at each lag band?

The random-split ridge in eval_temporal_info.py lets the same protein leak into
train and test, so it flatters the encoder (abs: R^2=0.31) while a cross-protein
MLP fails outright (11 A CA-RMSD).  B1a/B2 need a decoder that works on UNSEEN
proteins, so the honest metric is a protein-held-out ridge.

For each lag l we form pairs (frame 0, frame l) per protein, Kabsch-align frame l
onto frame 0, and use:
    x = H_l - H_0        (dH)          ->  y = CA_l_aligned - CA_0   (dCA)
Ridge fit on train proteins, R^2 scored on held-out proteins.
R^2 = 0 corresponds to always predicting the test mean; R^2 < 0 is worse than
the persistence baseline (predict zero displacement).

Usage: python probe_dh_dca_holdout.py <ckpt> [--label NAME] [--n-proteins 60]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config

MANIFEST = "artifacts/mdcath_manifest_200dom/train.jsonl"
LAGS = [8, 16, 32, 64, 96, 192, 384]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(ckpt):
    ck = torch.load(ckpt, map_location=dev)
    cfg = StageAV2Config(**ck.get("stage_a_v2_config", {}))
    sa = StageAV2(cfg).to(dev)
    sa.load_state_dict(ck["model"], strict=False)
    sa.eval()
    enc = sa.context_encoder
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg


@torch.no_grad()
def encode_traj(enc, coords, rmask, aatype, ridx, cid, chunk=16):
    Hs = []
    T = coords.shape[0]
    for t0 in range(0, T, chunk):
        ct = coords[t0:t0 + chunk]
        mt = rmask[t0:t0 + chunk]
        Hs.append(enc(ct[None], mt[None], aatype[None], ridx[None], cid[None],
                      token_mask=mt[None])[0])
    return torch.cat(Hs, 0)


def kabsch_align(P, Q, m):
    n = m.sum().clamp(min=1.0)
    cP = (P * m.unsqueeze(-1)).sum(0, keepdim=True) / n
    cQ = (Q * m.unsqueeze(-1)).sum(0, keepdim=True) / n
    Pc, Qc = P - cP, Q - cQ
    Hm = (Qc * m.unsqueeze(-1)).T @ (Pc * m.unsqueeze(-1))
    U, S, Vt = torch.linalg.svd(Hm)
    d = torch.det(U @ Vt)
    D = torch.eye(3, device=P.device); D[2, 2] = d
    R = Vt.T @ D @ U.T
    return (Qc @ R.T), Pc, cP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--label", default="model")
    ap.add_argument("--n-proteins", type=int, default=60)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()

    print(f"=== Cross-protein DeltaH->DeltaCA probe: {args.label} ===")
    enc, cfg = load_encoder(args.checkpoint)
    dH = cfg.latent_dim

    dc = DataConfig(manifest_path=MANIFEST, batch_size=1, num_workers=0,
                    training_mode="clip", num_frames=440)
    ds = build_dataset(dc)
    n_prot = min(args.n_proteins, len(ds))
    print(f"Dataset: {len(ds)} traj; using {n_prot}; dH={dH}\n")

    # per-lag: lists of (x, y) with a protein id
    X = {l: [] for l in LAGS}
    Y = {l: [] for l in LAGS}
    PID = {l: [] for l in LAGS}

    for pi in range(n_prot):
        s = ds[pi]
        coords = s["coords"].to(dev)
        rmask = s["residue_mask"].to(dev)
        if int(rmask[0].sum().item()) < 8:
            continue
        aatype = s["aatype"].to(dev)
        ridx = s["residue_index"].to(dev)
        cid = s["chain_id"].to(dev)
        T = coords.shape[0]
        H = encode_traj(enc, coords, rmask, aatype, ridx, cid)   # [T,L,dH]
        ca = coords[:, :, 1, :]                                   # [T,L,3]
        m0 = rmask[0].bool()

        for l in LAGS:
            if T <= l + 1:
                continue
            cal_al, ca0c, _ = kabsch_align(ca[l][m0], ca[0][m0], m0.float())
            x = (H[l][m0] - H[0][m0]).cpu()
            y = (cal_al - ca0c).cpu()
            X[l].append(x); Y[l].append(y)
            PID[l].append(torch.full((x.shape[0],), pi, dtype=torch.long))
        if (pi + 1) % 10 == 0:
            print(f"  encoded {pi+1}/{n_prot}", flush=True)

    heldout_pids = set(range(0, n_prot, 5))  # every 5th protein held out (~20%)
    print(f"\n{'lag(ns)':>8} {'n_tr_res':>9} {'n_te_res':>9} {'R^2_holdout':>12} {'R^2_random':>11}")
    print("-" * 58)
    for l in LAGS:
        if not X[l]:
            continue
        xall = torch.cat(X[l], 0).float()
        yall = torch.cat(Y[l], 0).float()
        pid = torch.cat(PID[l], 0).numpy()
        te_mask = np.array([p in heldout_pids for p in pid])
        tr_mask = ~te_mask
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        Xtr, Ytr = xall[tr_mask], yall[tr_mask]
        Xte, Yte = xall[te_mask], yall[te_mask]
        Xm, Xs = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
        Ym, Ys = Ytr.mean(0, keepdim=True), Ytr.std(0, keepdim=True).clamp(min=1e-6)
        Xtr_n, Xte_n = (Xtr - Xm) / Xs, (Xte - Xm) / Xs
        Ytr_n, Yte_n = (Ytr - Ym) / Ys, (Yte - Ym) / Ys
        XtX = Xtr_n.T @ Xtr_n
        XtY = Xtr_n.T @ Ytr_n
        W = torch.linalg.solve(XtX + args.lam * torch.eye(XtX.shape[0], device=XtX.device), XtY)
        pred = Xte_n @ W
        r2_ho = (1 - ((Yte_n - pred) ** 2).sum() / (Yte_n ** 2).sum()).item()
        # random-split comparison (same protocol as eval_temporal_info — inflated by leakage)
        n = xall.shape[0]; perm = torch.randperm(n); ntr = int(n * 0.8)
        tr2, te2 = perm[:ntr], perm[ntr:]
        Xm2 = xall[tr2].mean(0, keepdim=True)
        Xs2 = xall[tr2].std(0, keepdim=True).clamp(min=1e-6)
        Ym2 = yall[tr2].mean(0, keepdim=True)
        Ys2 = yall[tr2].std(0, keepdim=True).clamp(min=1e-6)
        Xtr2_n = (xall[tr2] - Xm2) / Xs2
        Xte2_n = (xall[te2] - Xm2) / Xs2
        Ytr2_n = (yall[tr2] - Ym2) / Ys2
        Yte2_n = (yall[te2] - Ym2) / Ys2
        W2 = torch.linalg.solve(Xtr2_n.T @ Xtr2_n + args.lam * torch.eye(dH), Xtr2_n.T @ Ytr2_n)
        p2 = Xte2_n @ W2
        r2_rd = (1 - ((Yte2_n - p2) ** 2).sum() / (Yte2_n ** 2).sum()).item()
        print(f"{l:>8} {int(tr_mask.sum()):>9} {int(te_mask.sum()):>9} {r2_ho:>+12.4f} {r2_rd:>+11.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
