"""H->delta information-ceiling probe using the project's iter_windows."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from protein_jepa_frame.training.train_stage_cb1b import load_models_for_eval, encode_frames
from protein_jepa_frame.models.stage_b1b_fromnone.objectives import kabsch_rotation
from evaluation.data import EvalProtocol, iter_windows


def kabsch_rmsd(pred, true, mask):
    idx = np.where(mask)[0]
    if len(idx) < 3: return np.nan
    p, q = pred[idx].astype(float), true[idx].astype(float)
    pc, qc = p - p.mean(0), q - q.mean(0)
    H = pc.T @ qc
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    al = pc @ (U @ D @ Vt) + q.mean(0)
    return float(np.sqrt(((al - q)**2).sum(-1).mean()))


@torch.no_grad()
def collect(encoder, normalizer, windows, h_frames, device, want):
    recs = []
    for w in windows:
        ctx = w.context
        fp  = w.future_path
        rmask = ctx["residue_mask"].all(dim=0).bool()
        if rmask.sum() < 4: continue
        fut = fp["coords"]
        if fut.shape[0] < h_frames + 1: continue

        coords = ctx["coords"].to(device)          # [K,L,4,3]
        anc_bb = coords[-1]                        # [L,4,3]
        anc_ca = anc_bb[:, 1, :]                   # [L,3]
        fut_bb = fut[h_frames].to(device)          # [L,4,3]  (index 0 == anchor)
        fut_ca = fut_bb[:, 1, :]                   # [L,3]

        aat = ctx["aatype"].to(device).unsqueeze(0)
        ridx = ctx["residue_index"].to(device).unsqueeze(0)
        cid = ctx["chain_id"].to(device).unsqueeze(0)
        rmask_d = rmask.to(device)
        rm_b = rmask_d[None, None].expand(1, 1, -1)   # [1,1,L]

        # both encoded with anchor conditioning from anc_bb
        H_anc = encode_frames(encoder, anc_bb[None, None], rm_b, aat, ridx, cid,
                              normalizer, anchor_coords=anc_bb[None], anchor_mask=rmask_d[None])[0, 0]
        H_fut = encode_frames(encoder, fut_bb[None, None], rm_b, aat, ridx, cid,
                              normalizer, anchor_coords=anc_bb[None], anchor_mask=rmask_d[None])[0, 0]

        m = rmask_d.float().unsqueeze(-1)
        def fill(x):
            c = (x * m).sum(0, keepdim=True) / m.sum(0, keepdim=True).clamp_min(1)
            return x * m + c * (1 - m)
        R, t = kabsch_rotation(fill(fut_ca).unsqueeze(0), fill(anc_ca).unsqueeze(0))
        delta = (fill(fut_ca).unsqueeze(0) @ R + t - anc_ca.unsqueeze(0)).squeeze(0) * m

        recs.append(dict(H_anc=H_anc.cpu(), H_fut=H_fut.cpu(), delta=delta.cpu(),
                         anc_ca=anc_ca.cpu(), fut_ca=fut_ca.cpu(), mask=rmask.cpu(),
                         anc_bb=anc_bb.cpu()))
        if len(recs) >= want: break
    return recs


class Prober(nn.Module):
    def __init__(self, d, mode):
        super().__init__(); self.mode = mode
        if mode == "linear":
            self.w = nn.Linear(d, 3); nn.init.zeros_(self.w.weight); nn.init.zeros_(self.w.bias)
        else:
            self.net = nn.Sequential(nn.LayerNorm(3*d), nn.Linear(3*d,256), nn.GELU(),
                                     nn.Linear(256,128), nn.GELU(), nn.Linear(128,3))
    def forward(self, hf, ha):
        if self.mode == "linear": return self.w(hf)
        return self.net(torch.cat([hf, ha, hf-ha], -1))


def fit(recs, mode, epochs=300, lr=3e-3, split=0.7):
    n = len(recs); ntr = max(2, int(n*split))
    tr, va = recs[:ntr], recs[ntr:]
    p = Prober(recs[0]["H_fut"].shape[-1], mode)
    opt = torch.optim.Adam(p.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        p.train()
        for r in tr:
            m = r["mask"].float().unsqueeze(-1)
            loss = ((p(r["H_fut"], r["H_anc"]) - r["delta"])**2 * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
    p.eval()
    def ev(rs, use):
        out = []
        with torch.no_grad():
            for r in rs:
                mk = r["mask"].numpy()
                if mk.sum() < 3: continue
                pred = (r["anc_ca"] + p(r["H_fut"], r["H_anc"])).numpy() if use else r["anc_ca"].numpy()
                out.append(kabsch_rmsd(pred, r["fut_ca"].numpy(), mk))
        return float(np.nanmean(out))
    return ev(tr,True), ev(va,True), ev(tr,False), ev(va,False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/fromnone/stage_cb1b_anchor/checkpoint_latest.pt")
    ap.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    ap.add_argument("--horizon-ns", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    enc, ker, dec, norm, _ = load_models_for_eval(a.ckpt, dev)
    step_ps = float(ker.cfg.step_ps)
    h = max(1, int(round(a.horizon_ns*1000.0/step_ps)))
    print(f"h_frames={h} step_ps={step_ps}")

    proto = EvalProtocol(manifest=a.manifest, horizons_frames=(h,), context_frames=8,
        context_stride=1, frame_stride=1, max_residues=192, max_proteins=None,
        max_anchors_per_trajectory=2, anchor_stride=1, temporal_lags=(),
        emit_multi_lag=False, seed=0)
    wins = [w for w in iter_windows(proto) if w.horizon == h][:a.n]
    print(f"windows: {len(wins)}")
    recs = collect(enc, norm, wins, h, dev, a.n)
    print(f"records: {len(recs)}")

    # trained decoder on true-H
    dec_r, per_r = [], []
    with torch.no_grad():
        from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
        for r in recs:
            mk = r["mask"].numpy()
            if mk.sum() < 3: continue
            bb = r["anc_bb"].unsqueeze(0).to(dev)
            R, t = RigidBodyFrame.from_backbone(bb[...,0,:], bb[...,1,:], bb[...,2,:])
            out = dec(H_future=r["H_fut"][None].to(dev), H_context=r["H_anc"][None].to(dev),
                      X_context=r["anc_ca"][None].to(dev), residue_mask=r["mask"][None].to(dev),
                      R_anchor=R, S_future=torch.zeros(r["mask"].shape[0], dec.cfg.bath_dim, device=dev)[None])
            dec_r.append(kabsch_rmsd(out["ca_pred"][0].cpu().numpy(), r["fut_ca"].numpy(), mk))
            per_r.append(kabsch_rmsd(r["anc_ca"].numpy(), r["fut_ca"].numpy(), mk))
    print(f"\ndecoder(true-H): {np.nanmean(dec_r):.4f} Å   persist: {np.nanmean(per_r):.4f} Å   "
          f"skill: {1-np.nanmean(dec_r)/np.nanmean(per_r):.4f}")

    print("\nlinear probe H_fut->delta:")
    tr,va,ptr,pva = fit(recs,"linear",epochs=400,lr=3e-3)
    print(f"  train {tr:.4f} (pers {ptr:.4f})  val {va:.4f} (pers {pva:.4f})  skill {1-va/pva:.4f}")
    print("\nMLP probe (H_fut,H_anc,dH)->delta:")
    tr,va,ptr,pva = fit(recs,"mlp",epochs=400,lr=1e-3)
    print(f"  train {tr:.4f} (pers {ptr:.4f})  val {va:.4f} (pers {pva:.4f})  skill {1-va/pva:.4f}")

if __name__ == "__main__":
    main()
