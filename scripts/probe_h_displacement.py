"""Probe: how much CA displacement is linearly decodable from H?

Loads the from-none checkpoint, encodes true future frames with the trained
encoder, fits a per-residue ridge probe H->delta_CA, measures RMSD vs
persistence.  This gives the information ceiling: if the ceiling is near
persistence, ensemble losses won't fix CA RMSD -- the encoder is the bottleneck.

Usage:
    python scripts/probe_h_displacement.py \
        --ckpt artifacts/fromnone/stage_cb1b_anchor/checkpoint_latest.pt \
        --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
        --n-windows 60 --horizon-ns 1 --device cuda:0
"""
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

from evaluation.data import EvalProtocol, iter_windows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/fromnone/stage_cb1b_anchor/checkpoint_latest.pt")
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    p.add_argument("--horizon-ns", type=float, default=1.0)
    p.add_argument("--n-windows", type=int, default=60)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--max-residues", type=int, default=192)
    return p.parse_args()


def kabsch_rmsd_np(pred, true, mask):
    """Kabsch-aligned RMSD, numpy.  pred/true: [L,3], mask: [L]."""
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return np.nan
    p, q = pred[idx], true[idx]
    pc = p - p.mean(0); qc = q - q.mean(0)
    H = pc.T @ qc
    U, _, Vt = np.linalg.svd(H)
    sgn = np.linalg.det(U @ Vt)
    D = np.diag([1, 1, sgn])
    R = U @ D @ Vt
    aligned = pc @ R + q.mean(0)
    diff = aligned - q
    return float(np.sqrt((diff**2).sum(-1).mean()))


@torch.no_grad()
def collect_samples(encoder, normalizer, data, device, step_ps, h_frames, history_frames):
    """Returns (H_anc, H_fut, delta_CA_local, anchor_ca, future_ca, mask) lists."""
    from protein_jepa_frame.models.encoder import _canonical_frame
    from protein_jepa_frame.models.stage_b1b_fromnone.objectives import kabsch_rotation

    records = []
    for w in data:
        ctx = w.context
        fp  = getattr(w, "future_path", None)
        if fp is None:
            continue
        rmask = ctx["residue_mask"].all(dim=0).bool()  # [L]
        if rmask.sum() < 4:
            continue

        # coords: [K, L, 4, 3]
        coords = ctx["coords"].to(device)              # [K, L, 4, 3]
        K, L = coords.shape[:2]
        aat  = ctx["aatype"].to(device)                # [L]
        ridx = ctx["residue_index"].to(device)
        cid  = ctx["chain_id"].to(device)
        rmask_d = rmask.to(device)

        # --- encode anchor (last history frame) ---
        anc_coords  = coords[-1]       # [L, 4, 3]
        anc_mask    = rmask_d          # [L]
        anc_ca      = anc_coords[:, 1, :]  # [L, 3]

        anc_coords_b  = anc_coords.unsqueeze(0).unsqueeze(0)   # [1,1,L,4,3]
        anc_rmask_b   = rmask_d.unsqueeze(0).unsqueeze(0)      # [1,1,L]
        H_anc = encode_frames(encoder,
                              anc_coords_b,
                              anc_rmask_b,
                              aat.unsqueeze(0),
                              ridx.unsqueeze(0),
                              cid.unsqueeze(0),
                              normalizer,
                              anchor_coords=anc_coords.unsqueeze(0),
                              anchor_mask=anc_mask.unsqueeze(0))  # [1,1,L,d]
        H_anc = H_anc[0, 0]   # [L, d]

        # --- encode true future frame (first h_frames step) ---
        fut_coords_np = fp["coords"]  # [T, L, 4, 3]
        if isinstance(fut_coords_np, torch.Tensor):
            fut_coords = fut_coords_np.to(device)
        else:
            fut_coords = torch.from_numpy(fut_coords_np).float().to(device)

        if fut_coords.shape[0] < h_frames:
            continue
        fut_frame = fut_coords[h_frames - 1]   # [L, 4, 3]
        fut_ca    = fut_frame[:, 1, :]          # [L, 3]

        fut_b = fut_frame.unsqueeze(0).unsqueeze(0)        # [1,1,L,4,3]
        fut_rmask_b = rmask_d.unsqueeze(0).unsqueeze(0)   # [1,1,L]
        H_fut = encode_frames(encoder,
                              fut_b,
                              fut_rmask_b,
                              aat.unsqueeze(0),
                              ridx.unsqueeze(0),
                              cid.unsqueeze(0),
                              normalizer,
                              anchor_coords=anc_coords.unsqueeze(0),
                              anchor_mask=anc_mask.unsqueeze(0))  # [1,1,L,d]
        H_fut = H_fut[0, 0]   # [L, d]

        # --- compute Kabsch-aligned displacement (ground truth for probe) ---
        m = rmask_d.float().unsqueeze(-1)   # [L, 1]
        def fill(x):
            c = (x * m).sum(0, keepdim=True) / m.sum(0, keepdim=True).clamp_min(1)
            return x * m + c * (1 - m)

        # Kabsch: align future onto anchor
        from protein_jepa_frame.models.stage_b1b_fromnone.objectives import kabsch_rotation
        p_fill = fill(fut_ca)
        q_fill = fill(anc_ca)
        R, t = kabsch_rotation(p_fill.unsqueeze(0), q_fill.unsqueeze(0))
        delta = (p_fill.unsqueeze(0) @ R + t - anc_ca.unsqueeze(0)).squeeze(0)  # [L, 3]
        delta = delta * m  # zero out masked

        records.append({
            "H_anc": H_anc.cpu(),        # [L, d]
            "H_fut": H_fut.cpu(),        # [L, d]
            "delta": delta.cpu(),        # [L, 3]
            "anc_ca": anc_ca.cpu(),      # [L, 3]
            "fut_ca": fut_ca.cpu(),      # [L, 3]
            "mask": rmask.cpu(),         # [L]
        })
    return records


class RidgeProbe(nn.Module):
    """Per-residue linear map: H_fut [d] -> delta_CA [3]."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 3, bias=True)
        nn.init.zeros_(self.w.weight); nn.init.zeros_(self.w.bias)

    def forward(self, h):
        return self.w(h)   # [L, 3]


class MLPProbe(nn.Module):
    """Per-residue MLP: (H_fut, H_anc, dH) [3d] -> delta_CA [3]."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, h_fut, h_anc):
        dh = h_fut - h_anc
        return self.net(torch.cat([h_fut, h_anc, dh], dim=-1))  # [L, 3]


def train_probe(records, probe_type, n_epochs=200, lr=1e-3, split=0.7):
    """Train ridge/MLP probe on records, return (train_rmsd, val_rmsd, persist_rmsd)."""
    n = len(records)
    n_train = max(1, int(n * split))
    train_recs = records[:n_train]
    val_recs   = records[n_train:]

    d = records[0]["H_fut"].shape[-1]
    if probe_type == "ridge":
        probe = RidgeProbe(d)
    else:
        probe = MLPProbe(d)

    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(n_epochs):
        probe.train()
        total_loss = 0.0
        for r in train_recs:
            h_fut = r["H_fut"]  # [L, d]
            h_anc = r["H_anc"]
            delta = r["delta"]  # [L, 3]
            mask  = r["mask"].float().unsqueeze(-1)  # [L, 1]

            if probe_type == "ridge":
                pred = probe(h_fut)
            else:
                pred = probe(h_fut, h_anc)

            loss = ((pred - delta)**2 * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss)

    # Evaluate
    probe.eval()
    def rmsd_on(recs):
        rmsds = []
        with torch.no_grad():
            for r in recs:
                mask = r["mask"].numpy()
                if mask.sum() < 3: continue
                h_fut = r["H_fut"]; h_anc = r["H_anc"]
                if probe_type == "ridge":
                    pred_delta = probe(h_fut).numpy()
                else:
                    pred_delta = probe(h_fut, h_anc).numpy()
                pred_ca = r["anc_ca"].numpy() + pred_delta
                true_ca = r["fut_ca"].numpy()
                rmsds.append(kabsch_rmsd_np(pred_ca, true_ca, mask))
        return float(np.nanmean(rmsds)) if rmsds else float("nan")

    def pers_rmsd_on(recs):
        rmsds = []
        for r in recs:
            mask = r["mask"].numpy()
            if mask.sum() < 3: continue
            rmsds.append(kabsch_rmsd_np(r["anc_ca"].numpy(), r["fut_ca"].numpy(), mask))
        return float(np.nanmean(rmsds)) if rmsds else float("nan")

    train_rmsd = rmsd_on(train_recs)
    val_rmsd   = rmsd_on(val_recs)
    pers_train = pers_rmsd_on(train_recs)
    pers_val   = pers_rmsd_on(val_recs)
    return train_rmsd, val_rmsd, pers_train, pers_val


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder, kernel, decoder, normalizer, ck = load_models_for_eval(args.ckpt, device)
    step_ps = float(kernel.cfg.step_ps)
    h_frames = max(1, int(round(args.horizon_ns * 1000.0 / step_ps)))
    print(f"ckpt={args.ckpt}  horizon={args.horizon_ns}ns -> {h_frames} frames  step_ps={step_ps}")

    proto = EvalProtocol(
        manifest=args.manifest,
        horizons_frames=(h_frames,),
        context_frames=args.history_frames,
        context_stride=1, frame_stride=1,
        max_residues=args.max_residues,
        max_proteins=None,
        max_anchors_per_trajectory=2,
        anchor_stride=1, temporal_lags=(), emit_multi_lag=False,
        seed=0,
    )

    print(f"Collecting {args.n_windows} windows...")
    windows = []
    for w in iter_windows(proto):
        if w.horizon != h_frames:
            continue
        windows.append(w)
        if len(windows) >= args.n_windows:
            break
    print(f"Got {len(windows)} windows")

    records = collect_samples(encoder, normalizer, windows, device,
                              step_ps, h_frames, args.history_frames)
    print(f"Encoded {len(records)} records")

    if len(records) < 4:
        print("Too few records; aborting."); return

    print("\n--- Ridge probe (H_fut -> delta_CA) ---")
    tr, vr, tp, vp = train_probe(records, "ridge", n_epochs=300, lr=3e-3)
    print(f"  train RMSD: probe={tr:.4f}  persist={tp:.4f}  skill={1-tr/tp:.4f}")
    print(f"  val   RMSD: probe={vr:.4f}  persist={vp:.4f}  skill={1-vr/vp:.4f}")

    print("\n--- MLP probe (H_fut, H_anc, dH -> delta_CA) ---")
    tr2, vr2, tp2, vp2 = train_probe(records, "mlp", n_epochs=400, lr=1e-3)
    print(f"  train RMSD: probe={tr2:.4f}  persist={tp2:.4f}  skill={1-tr2/tp2:.4f}")
    print(f"  val   RMSD: probe={vr2:.4f}  persist={vp2:.4f}  skill={1-vr2/vp2:.4f}")

    print("\n--- Decoder (trained, teacher-forced, first val record) ---")
    # decode true future H through the trained decoder to confirm tf_ca result
    with torch.no_grad():
        r = records[int(len(records) * 0.7)]
        h_fut = r["H_fut"].unsqueeze(0).to(device)   # [1, L, d]
        h_anc = r["H_anc"].unsqueeze(0).to(device)
        mask  = r["mask"].unsqueeze(0).to(device)
        anc_ca = r["anc_ca"].unsqueeze(0).to(device)
        # need R_anchor
        from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
        # load anchor bb from window (we stored anc_ca but not full bb)
        # just pass identity R for the probe check
        R_anc = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, h_fut.shape[1], 3, 3)
        out = decoder(H_future=h_fut, H_context=h_anc,
                      X_context=anc_ca, residue_mask=mask,
                      R_anchor=R_anc)
        pred_ca = out["ca_pred"][0].cpu().numpy()
        true_ca = r["fut_ca"].numpy()
        m = r["mask"].numpy()
        dec_rmsd = kabsch_rmsd_np(pred_ca, true_ca, m)
        pers_rmsd = kabsch_rmsd_np(r["anc_ca"].numpy(), true_ca, m)
        print(f"  decoder(identity R_anc) RMSD: {dec_rmsd:.4f}  persist={pers_rmsd:.4f}")


if __name__ == "__main__":
    main()
