"""Fast H->delta probe: measure information ceiling in 60s."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from protein_jepa_frame.training.train_stage_cb1b import (
    load_models_for_eval, encode_frames,
)
from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
    aligned_ca_rmsd, kabsch_rotation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/fromnone/stage_cb1b_anchor/checkpoint_latest.pt")
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    p.add_argument("--horizon-ns", type=float, default=1.0)
    p.add_argument("--n-records", type=int, default=40)
    p.add_argument("--device", default="cuda:1")
    return p.parse_args()


def kabsch_rmsd(pred, true, mask):
    """pred/true: [L,3] numpy, mask: [L] bool. Returns scalar Å."""
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return np.nan
    p, q = pred[idx].astype(float), true[idx].astype(float)
    pc, qc = p - p.mean(0), q - q.mean(0)
    H = pc.T @ qc
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    al = pc @ R + q.mean(0)
    return float(np.sqrt(((al - q)**2).sum(-1).mean()))


@torch.no_grad()
def collect(ckpt_path, manifest, horizon_ns, n_records, device):
    from protein_jepa_frame.models.encoder import _canonical_frame
    import json

    encoder, kernel, decoder, normalizer, ck = load_models_for_eval(ckpt_path, device)
    step_ps = float(kernel.cfg.step_ps)
    h_frames = max(1, int(round(horizon_ns * 1000.0 / step_ps)))
    print(f"step_ps={step_ps}  h_frames={h_frames}")

    records = []
    K = 8

    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = rec.get("path") or rec.get("h5_path")
        if path is None:
            continue
        try:
            import h5py
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                dom = keys[0] if len(keys) == 1 else [k for k in keys if not k.startswith("_")][0]
                coords_all = f[dom]["coordinates"][:]   # [T, L, 4, 3] or [T, N, 3]
        except Exception as e:
            continue

        # We need [T, L, 4, 3] backbone
        if coords_all.ndim != 4 or coords_all.shape[-2] != 4:
            continue
        T_tot = coords_all.shape[0]
        L = coords_all.shape[1]
        if T_tot < K + h_frames:
            continue

        # Build aatype/ridx/cid tensors (dummy but consistent)
        aat  = torch.zeros(L, dtype=torch.long, device=device)
        ridx = torch.arange(L, dtype=torch.long, device=device)
        cid  = torch.zeros(L, dtype=torch.long, device=device)

        try:
            # load residue mask from hdf5 if available
            with h5py.File(path, "r") as f:
                if "residue_mask" in f[dom]:
                    rmask_np = f[dom]["residue_mask"][:].astype(bool)  # [L]
                else:
                    rmask_np = np.ones(L, dtype=bool)
        except:
            rmask_np = np.ones(L, dtype=bool)

        rmask = torch.from_numpy(rmask_np).to(device)
        if rmask.sum() < 4:
            continue

        # Sample a few anchors from this trajectory
        for t0 in range(0, min(T_tot - K - h_frames, 3 * (K + h_frames)), K + h_frames):
            coords_np = coords_all[t0:t0 + K + h_frames]   # [K+h_frames, L, 4, 3]
            coords_t = torch.from_numpy(coords_np).float().to(device)

            hist_coords = coords_t[:K]                     # [K, L, 4, 3]
            fut_coords  = coords_t[K + h_frames - 1]      # [L, 4, 3]  (the target frame)

            anc_coords = hist_coords[-1]   # [L, 4, 3]
            anc_ca = anc_coords[:, 1, :]   # [L, 3]
            fut_ca = fut_coords[:, 1, :]   # [L, 3]

            # Encode anchor frame with anchor conditioning
            anc_b = anc_coords.unsqueeze(0).unsqueeze(0)   # [1,1,L,4,3]
            rm_b  = rmask.unsqueeze(0).unsqueeze(0)         # [1,1,L]
            H_anc = encode_frames(encoder, anc_b, rm_b,
                                  aat.unsqueeze(0), ridx.unsqueeze(0), cid.unsqueeze(0),
                                  normalizer,
                                  anchor_coords=anc_coords.unsqueeze(0),
                                  anchor_mask=rmask.unsqueeze(0))  # [1,1,L,d]
            H_anc = H_anc[0, 0]   # [L, d]

            # Encode TRUE future frame with same anchor conditioning
            fut_b = fut_coords.unsqueeze(0).unsqueeze(0)
            H_fut = encode_frames(encoder, fut_b, rm_b,
                                  aat.unsqueeze(0), ridx.unsqueeze(0), cid.unsqueeze(0),
                                  normalizer,
                                  anchor_coords=anc_coords.unsqueeze(0),
                                  anchor_mask=rmask.unsqueeze(0))  # [1,1,L,d]
            H_fut = H_fut[0, 0]   # [L, d]

            # Ground-truth Kabsch-aligned displacement (anchor -> future)
            m = rmask.float().unsqueeze(-1)  # [L,1]
            def fill(x):
                c = (x * m).sum(0, keepdim=True) / m.sum(0, keepdim=True).clamp_min(1)
                return x * m + c * (1 - m)
            R, t = kabsch_rotation(fill(fut_ca).unsqueeze(0), fill(anc_ca).unsqueeze(0))
            delta = (fill(fut_ca).unsqueeze(0) @ R + t - anc_ca.unsqueeze(0)).squeeze(0) * m  # [L,3]

            # Decode true future H through trained decoder
            R_anc = torch.eye(3, device=device).unsqueeze(0).expand(L, 3, 3)  # identity (no local frame)
            out = decoder(
                H_future=H_fut.unsqueeze(0),
                H_context=H_anc.unsqueeze(0),
                X_context=anc_ca.unsqueeze(0),
                residue_mask=rmask.unsqueeze(0),
                R_anchor=R_anc.unsqueeze(0),   # [1,L,3,3]
            )
            pred_dec = out["ca_pred"][0]   # [L,3]

            records.append({
                "H_anc": H_anc.cpu(), "H_fut": H_fut.cpu(),
                "delta": delta.cpu(),
                "anc_ca": anc_ca.cpu(), "fut_ca": fut_ca.cpu(),
                "pred_dec": pred_dec.cpu(),
                "mask": rmask.cpu(),
            })
            if len(records) >= n_records:
                break
        if len(records) >= n_records:
            break

    return records


class MLPProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )
    def forward(self, h_f, h_a):
        return self.net(torch.cat([h_f, h_a, h_f - h_a], dim=-1))


def train_probe(records, n_epochs=200, lr=1e-3, split=0.7):
    n = len(records); n_tr = max(2, int(n * split))
    tr, va = records[:n_tr], records[n_tr:]
    d = records[0]["H_fut"].shape[-1]
    probe = MLPProbe(d); opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(n_epochs):
        probe.train()
        for r in tr:
            m = r["mask"].float().unsqueeze(-1)
            loss = ((probe(r["H_fut"], r["H_anc"]) - r["delta"])**2 * m).sum() / m.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
    probe.eval()
    def rmsd_set(recs, use_probe):
        rs = []
        with torch.no_grad():
            for r in recs:
                mask = r["mask"].numpy()
                if mask.sum() < 3: continue
                if use_probe:
                    pred_d = probe(r["H_fut"], r["H_anc"]).numpy()
                    pred_ca = r["anc_ca"].numpy() + pred_d
                else:
                    pred_ca = r["anc_ca"].numpy()   # persistence
                rs.append(kabsch_rmsd(pred_ca, r["fut_ca"].numpy(), mask))
        return float(np.nanmean(rs)) if rs else float("nan")
    return (rmsd_set(tr, True), rmsd_set(va, True),
            rmsd_set(tr, False), rmsd_set(va, False))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records = collect(args.ckpt, args.manifest, args.horizon_ns, args.n_records, device)
    print(f"\nCollected {len(records)} records")
    if len(records) < 6:
        print("Too few records."); return

    # Report decoder(true H) vs persistence
    dec_rmsds, pers_rmsds = [], []
    for r in records:
        mask = r["mask"].numpy()
        if mask.sum() < 3: continue
        dec_rmsds.append(kabsch_rmsd(r["pred_dec"].numpy(), r["fut_ca"].numpy(), mask))
        pers_rmsds.append(kabsch_rmsd(r["anc_ca"].numpy(), r["fut_ca"].numpy(), mask))
    print(f"\nDecoder(true-H, identity R_anc): {np.nanmean(dec_rmsds):.4f} Å")
    print(f"Persistence:                      {np.nanmean(pers_rmsds):.4f} Å")
    print(f"Decoder skill: {1 - np.nanmean(dec_rmsds)/np.nanmean(pers_rmsds):.4f}")

    print("\nTraining MLP probe (H_fut,H_anc,dH -> delta_CA)...")
    tr_r, va_r, tr_p, va_p = train_probe(records, n_epochs=300)
    print(f"Probe train: {tr_r:.4f} Å  persist: {tr_p:.4f} Å  skill: {1-tr_r/tr_p:.4f}")
    print(f"Probe val:   {va_r:.4f} Å  persist: {va_p:.4f} Å  skill: {1-va_r/va_p:.4f}")

    print("\n--- Diagnosis ---")
    avg_pers = np.nanmean(pers_rmsds)
    if va_r < avg_pers * 0.85:
        print(f"H carries learnable displacement (probe val {va_r:.3f} < {avg_pers*0.85:.3f}).")
        print("Bottleneck is decoder/loss — v4 ensemble losses should help.")
    elif va_r < avg_pers * 0.98:
        print(f"H carries weak signal (probe val {va_r:.3f} vs persist {avg_pers:.3f}).")
        print("Partial bottleneck in encoder AND decoder. v4 helps but encoder also needs work.")
    else:
        print(f"H carries near-zero displacement signal (probe {va_r:.3f} vs persist {avg_pers:.3f}).")
        print("Bottleneck is the encoder / training objective. Adding ensemble losses alone won't fix CA RMSD.")


if __name__ == "__main__":
    main()
