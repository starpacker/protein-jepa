"""§1.7 gate item 3 (honest): does H predict SE(3)-INVARIANT internal structure
on HELD-OUT proteins?  Target = pairwise CA distance matrix (invariant), which
a frozen decoder must reproduce.  We fit a small MLP:  H_t[i] (+) H_t[j] -> d_ij.

Two conditions per lag:
  (a) absolute: predict d_ij at frame t=l           -> can H place the fold?
  (b) delta:    predict d_ij(l) - d_ij(0)           -> can H encode the CHANGE?

Protein-held-out split (train 80% proteins, score 20%).  Reports MSE and the
fraction of distance variance explained (R^2) on held-out proteins.
Baseline for (b): predict zero change (persistence) -> R^2 = 0.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config

MANIFEST = "artifacts/mdcath_manifest_200dom/train.jsonl"
LAGS = [16, 64, 192]
NPAIR = 400           # random residue pairs per protein
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(ckpt):
    ck = torch.load(ckpt, map_location=dev)
    cfg = StageAV2Config(**ck.get("stage_a_v2_config", {}))
    sa = StageAV2(cfg).to(dev); sa.load_state_dict(ck["model"], strict=False); sa.eval()
    enc = sa.context_encoder
    for p in enc.parameters(): p.requires_grad_(False)
    return enc, cfg


@torch.no_grad()
def encode_traj(enc, coords, rmask, aa, ri, cid, chunk=16):
    Hs = []
    for t0 in range(0, coords.shape[0], chunk):
        Hs.append(enc(coords[t0:t0+chunk][None], rmask[t0:t0+chunk][None],
                      aa[None], ri[None], cid[None], token_mask=rmask[t0:t0+chunk][None])[0])
    return torch.cat(Hs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--label", default="model")
    ap.add_argument("--n-proteins", type=int, default=60)
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    print(f"=== H internal-distance probe (held-out): {a.label} ===")
    enc, cfg = load_encoder(a.checkpoint); dH = cfg.latent_dim

    ds = build_dataset(DataConfig(manifest_path=MANIFEST, batch_size=1, num_workers=0,
                                  training_mode="clip", num_frames=440))
    n_prot = min(a.n_proteins, len(ds))
    heldout = set(range(0, n_prot, 5))

    # collect features per (protein, lag)
    feats = {l: {"tr": [], "te": []} for l in LAGS}
    rng = np.random.default_rng(0)
    for pi in range(n_prot):
        s = ds[pi]
        coords = s["coords"].to(dev); rmask = s["residue_mask"].to(dev)
        if int(rmask[0].sum().item()) < 16: continue
        aa = s["aatype"].to(dev); ri = s["residue_index"].to(dev); cid = s["chain_id"].to(dev)
        H = encode_traj(enc, coords, rmask, aa, ri, cid)
        ca = coords[:, :, 1, :]; m0 = rmask[0].bool()
        idx = torch.nonzero(m0).squeeze(-1).cpu().numpy()
        if len(idx) < 4: continue
        ii = rng.choice(idx, NPAIR); jj = rng.choice(idx, NPAIR)
        keep = ii != jj; ii, jj = ii[keep], jj[keep]
        bucket = "te" if pi in heldout else "tr"
        for l in LAGS:
            if coords.shape[0] <= l + 1: continue
            d_l = (ca[l][ii] - ca[l][jj]).norm(dim=-1).cpu()
            d_0 = (ca[0][ii] - ca[0][jj]).norm(dim=-1).cpu()
            hi = H[l][ii].cpu(); hj = H[l][jj].cpu()
            x = torch.cat([hi + hj, (hi - hj).abs()], -1)   # symmetric pair feature
            feats[l][bucket].append((x, d_l, d_l - d_0))
        if (pi+1) % 15 == 0: print(f"  encoded {pi+1}/{n_prot}", flush=True)

    print(f"\n{'lag':>5} {'cond':>8} {'R2_heldout':>11}")
    print("-"*28)
    for l in LAGS:
        if not feats[l]["tr"] or not feats[l]["te"]: continue
        Xtr = torch.cat([f[0] for f in feats[l]["tr"]], 0)
        Xte = torch.cat([f[0] for f in feats[l]["te"]], 0)
        for ci, name in [(1, "abs_dij"), (2, "delta_dij")]:
            ytr = torch.cat([f[ci] for f in feats[l]["tr"]], 0).unsqueeze(-1)
            yte = torch.cat([f[ci] for f in feats[l]["te"]], 0).unsqueeze(-1)
            xm, xs = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
            ym, ys = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True).clamp(min=1e-6)
            Xa, Xb = ((Xtr-xm)/xs).to(dev), ((Xte-xm)/xs).to(dev)
            Ya, Yb = ((ytr-ym)/ys).to(dev), ((yte-ym)/ys).to(dev)
            mlp = nn.Sequential(nn.Linear(2*dH, 256), nn.SiLU(),
                                nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1)).to(dev)
            opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
            n = Xa.shape[0]
            for step in range(a.steps):
                bi = torch.randint(0, n, (4096,), device=dev)
                loss = ((mlp(Xa[bi]) - Ya[bi])**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred = mlp(Xb)
                r2 = (1 - ((Yb-pred)**2).sum()/(Yb**2).sum()).item()
            print(f"{l:>5} {name:>8} {r2:>+11.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
