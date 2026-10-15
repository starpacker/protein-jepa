"""Delta-decodability probe: does ΔH = H_future - H_anchor predict the per-residue
CA displacement (in the Kabsch-aligned anchor frame)?

This is the TRUE test of whether H carries conformational-change info -- unlike
cos_sim, which is dominated by the large shared static-fold component.

For each (protein, lag), we collect per-residue:
  X = [H_anchor_i, H_future_i]  (or ΔH_i)   feature
  y = R·ca_future_i - ca_anchor_i  (displacement after optimal rigid alignment)

Then fit a held-out ridge regression (train on 70% of residue samples pooled
across proteins, test on 30%) and report R² per lag. Compared to a persistence
baseline (predict zero displacement) and basin baseline (predict mean displacement).
"""
import sys, json, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/liuyutian/protein_jepa_frame')
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset, TrajectoryRecord
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config
import h5py

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/liuyutian/protein_jepa_frame/artifacts/abc_v3/stage_a_fix2/checkpoint_latest.pt"
MANIFEST = "/liuyutian/protein_jepa_frame/artifacts/mdcath_manifest_200dom/train.jsonl"
LAGS_NS = [1, 4, 16, 64, 96]
N_PROTEINS = 40

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location=dev)
cfg = StageAV2Config(**ck.get("stage_a_v2_config", {}))
sa = StageAV2(cfg).to(dev)
sa.load_state_dict(ck["model"], strict=False); sa.eval()
enc = sa.context_encoder
for p in enc.parameters(): p.requires_grad_(False)
dH = cfg.latent_dim

entries = [json.loads(l) for l in open(MANIFEST)]
seen = {}
for e in entries:
    d = e["domain_id"]
    if d not in seen: seen[d] = e
domains = list(seen.values())
random.seed(0)
domains = [e for e in domains if e["temperature_k"] == 320] or domains
random.shuffle(domains)

dc = DataConfig(manifest_path=MANIFEST)
ds = build_dataset(dc)
_topo = ds._mdcath_topology
_gather = ds._gather_backbone

def kabsch_align(P, Q):
    """Return R, t aligning P onto Q (both [N,3], centered internally)."""
    pc = P - P.mean(0); qc = Q - Q.mean(0)
    H = pc.T @ qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1., 1., d])
    R = Vt.T @ D @ U.T
    return R, pc, qc

# Collect samples per lag
samples = {lag: {"X": [], "y": []} for lag in LAGS_NS}
used = 0
for e in domains:
    if used >= N_PROTEINS: break
    try:
        frame_idx = np.array([0] + LAGS_NS, dtype=np.int64)
        with h5py.File(e["h5_path"], "r") as h:
            grp = h[e["metadata"]["hdf5_group"]]
            if grp["coords"].shape[0] <= max(LAGS_NS): continue
            all_atom = np.asarray(grp["coords"][frame_idx.tolist()], dtype=np.float32)
        rec = TrajectoryRecord(
            path=Path(e["h5_path"]), protein_id=e["protein_id"],
            trajectory_id=e["trajectory_id"], domain_id=e["domain_id"],
            temperature_k=e["temperature_k"], replica_id=e["replica_id"],
            frame_interval_ps=e["frame_interval_ps"],
            first_frame_time_ps=0.0, metadata=e["metadata"],
        )
        topo = _topo(rec)
        coords, atom_mask = _gather(all_atom, topo)   # [F, L, 4, 3]
        rmask = atom_mask.all(axis=-1)
        aatype = torch.tensor(topo.aatype)
        residue_index = torch.tensor(topo.residue_index)
        chain_id = torch.tensor(topo.chain_id)
    except Exception:
        continue
    v = rmask[0]
    if v.sum() < 10: continue

    ct = torch.tensor(coords).to(dev)
    rm = torch.tensor(rmask).to(dev)
    with torch.no_grad():
        H = enc(ct[None], rm[None], aatype[None].to(dev),
                residue_index[None].to(dev), chain_id[None].to(dev))[0]  # [F, L, dH]
    Hv = H[:, v, :].cpu().numpy()        # [F, Lv, dH]
    cav = coords[:, :, 1, :][:, v, :]    # [F, Lv, 3]

    H_anchor = Hv[0]; ca_anchor = cav[0]
    for k, lag in enumerate(LAGS_NS):
        H_fut = Hv[k+1]; ca_fut = cav[k+1]
        # Align future onto anchor; displacement = aligned_future - anchor
        R, pc, qc = kabsch_align(ca_fut, ca_anchor)   # align future(P) to anchor(Q)
        ca_fut_aligned = (ca_fut - ca_fut.mean(0)) @ R.T
        disp = ca_fut_aligned - (ca_anchor - ca_anchor.mean(0))   # [Lv, 3]
        dH_vec = H_fut - H_anchor                      # [Lv, dH]
        samples[lag]["X"].append(dH_vec)
        samples[lag]["y"].append(disp)
    used += 1

print(f"[probe_delta_decode] {used} proteins, dH={dH}, ckpt={Path(CKPT).parent.name}")
print("="*78)
print(f"  Ridge R² of (ΔH -> per-residue CA displacement) in anchor frame")
print(f"  baseline = predict mean displacement (basin);  persistence = predict 0")
print("="*78)
print(f"  {'lag(ns)':>8}  {'n_res':>8}  {'ridge R²':>9}  {'basin R²':>9}  {'disp_rms(Å)':>11}")
print(f"  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*11}")

out = {"ckpt": CKPT, "dH": dH, "by_lag": {}}
for lag in LAGS_NS:
    X = np.concatenate(samples[lag]["X"], 0)   # [N, dH]
    y = np.concatenate(samples[lag]["y"], 0)   # [N, 3]
    N = X.shape[0]
    idx = np.random.RandomState(0).permutation(N)
    ntr = int(0.7*N)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
    # standardize X
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr-mu)/sd; Xte = (Xte-mu)/sd
    # ridge closed-form
    lam = 10.0
    A = Xtr.T @ Xtr + lam*np.eye(dH)
    W = np.linalg.solve(A, Xtr.T @ ytr)   # [dH, 3]
    pred = Xte @ W
    ss_res = ((yte - pred)**2).sum()
    ss_tot = ((yte - yte.mean(0))**2).sum()
    r2 = 1 - ss_res/ss_tot
    # basin baseline: predict train-mean displacement
    basin_pred = np.tile(ytr.mean(0), (yte.shape[0],1))
    ss_res_b = ((yte - basin_pred)**2).sum()
    r2_basin = 1 - ss_res_b/ss_tot
    disp_rms = np.sqrt((y**2).sum(-1).mean())
    print(f"  {lag:>8}  {N:>8}  {r2:>9.4f}  {r2_basin:>9.4f}  {disp_rms:>11.3f}")
    out["by_lag"][str(lag)] = {"n_res": int(N), "ridge_r2": float(r2),
                               "basin_r2": float(r2_basin), "disp_rms": float(disp_rms)}

Path("/liuyutian/protein_jepa_frame/tmp_claude/delta_decode.json").write_text(json.dumps(out, indent=2))
print("\n[saved] tmp_claude/delta_decode.json")
