"""Measure Stage A H cosine similarity at lags 1/4/8/16/32/64/96 ns.

Reads the FULL mdCATH trajectory (440 frames @ 1 ns) directly from HDF5,
encodes frames [0, 1, 4, 8, 16, 32, 64, 96], and reports:
  - trajectory-level cos_sim(H_anchor, H_future) pooled over residues
  - per-residue cos_sim distribution
  - Kabsch CA RMSD (reference: how much the structure actually moved)
"""
import sys, json, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/liuyutian/protein_jepa_frame')
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.stage_a_v2 import StageAV2, StageAV2Config

CKPT = "/liuyutian/protein_jepa_frame/artifacts/abc_v3/stage_a_fix2/checkpoint_latest.pt"
MANIFEST = "/liuyutian/protein_jepa_frame/artifacts/mdcath_manifest_200dom/train.jsonl"
LAGS_NS = [1, 4, 8, 16, 32, 64, 96]
N_PROTEINS = 20

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location=dev)
cfg = StageAV2Config(**ck.get("stage_a_v2_config", {}))
sa = StageAV2(cfg).to(dev)
sa.load_state_dict(ck["model"], strict=False); sa.eval()
enc = sa.context_encoder
for p in enc.parameters(): p.requires_grad_(False)
dH = cfg.latent_dim

# ── pick distinct domains from the manifest ─────────────────────────────────
entries = [json.loads(l) for l in open(MANIFEST)]
seen = {}
for e in entries:
    d = e["domain_id"]
    if d not in seen:
        seen[d] = e
domains = list(seen.values())
random.seed(0)
# prefer temperature 320 replica 0 for comparability
domains = [e for e in domains if e["temperature_k"] == 320] or domains
random.shuffle(domains)
domains = domains[:N_PROTEINS*3]   # oversample, skip failures

# ── reuse dataset internals for topology + backbone gather ──────────────────
dc = DataConfig(manifest_path=MANIFEST)
ds = build_dataset(dc)
_read = ds._read_mdcath_h5
_topo = ds._mdcath_topology
_gather = ds._gather_backbone
from protein_jepa_frame.data.dataset import TrajectoryRecord
import h5py

results = defaultdict(lambda: {"cos": [], "rmsd": [], "res_cos": []})
used = 0
for e in domains:
    if used >= N_PROTEINS: break
    try:
        frame_idx = np.array([0] + LAGS_NS, dtype=np.int64)
        with h5py.File(e["h5_path"], "r") as h:
            grp = h[e["metadata"]["hdf5_group"]]
            n_frames = grp["coords"].shape[0]
            if n_frames <= max(LAGS_NS): continue
            all_atom = np.asarray(grp["coords"][frame_idx.tolist()], dtype=np.float32)
        # topology
        rec = TrajectoryRecord(
            path=Path(e["h5_path"]), protein_id=e["protein_id"],
            trajectory_id=e["trajectory_id"], domain_id=e["domain_id"],
            temperature_k=e["temperature_k"], replica_id=e["replica_id"],
            frame_interval_ps=e["frame_interval_ps"],
            first_frame_time_ps=0.0, metadata=e["metadata"],
        )
        topo = _topo(rec)
        coords, atom_mask = _gather(all_atom, topo)   # [8, L, 4, 3]
        rmask = atom_mask.all(axis=-1)                # [8, L]
        aatype = torch.tensor(topo.aatype)
        residue_index = torch.tensor(topo.residue_index)
        chain_id = torch.tensor(topo.chain_id)
    except Exception as ex:
        continue

    v = torch.tensor(rmask[0])
    if v.sum() < 10: continue

    coords_t = torch.tensor(coords[:1]).to(dev)   # just anchor shape ref
    ct = torch.tensor(coords).to(dev)             # [8, L, 4, 3]
    rm = torch.tensor(rmask).to(dev)
    with torch.no_grad():
        H = enc(ct[None], rm[None], aatype[None].to(dev),
                residue_index[None].to(dev), chain_id[None].to(dev))[0]  # [8, L, dH]
    Hv = H[:, v, :].cpu()        # [8, Lv, dH]
    cav = ct[:, :, 1, :][:, v, :].cpu()  # [8, Lv, 3]

    H_anchor = Hv[0]             # [Lv, dH]
    ca_anchor = cav[0]           # [Lv, 3]
    ca_ac = ca_anchor - ca_anchor.mean(0)

    for k, lag in enumerate(LAGS_NS):
        H_fut = Hv[k+1]          # [Lv, dH]
        # trajectory-level: pooled cosine over all residues
        c = torch.nn.functional.cosine_similarity(
            H_anchor.reshape(-1)[None], H_fut.reshape(-1)[None]
        ).item()
        results[lag]["cos"].append(c)
        # per-residue cosine
        rc = torch.nn.functional.cosine_similarity(H_anchor, H_fut, dim=-1)
        results[lag]["res_cos"].append(rc)
        # Kabsch CA RMSD
        ca_fc = cav[k+1] - cav[k+1].mean(0)
        Hk = ca_ac.T @ ca_fc
        U, S, Vt = torch.linalg.svd(Hk)
        det = torch.det(Vt.T @ U.T)
        D = torch.diag(torch.tensor([1., 1., det.item()]))
        R = Vt.T @ D @ U.T
        rmsd = ((ca_ac @ R.T - ca_fc).pow(2).sum(-1).mean().sqrt()).item()
        results[lag]["rmsd"].append(rmsd)
    used += 1

print(f"Encoded {used} proteins from full 440 ns trajectories, dH={dH}, T=320K R0")
print()
print("="*86)
print("METRIC 1: Trajectory-level H cosine similarity vs lag (1–96 ns)")
print("  cos_sim(H_anchor, H_future) pooled over all residues; across-protein stats")
print("="*86)
print(f"  {'lag(ns)':>8}  {'cos mean':>9}  {'cos std':>8}  {'cos min':>8}  {'cos max':>8}  {'CA RMSD mean(Å)':>16}")
print(f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*16}")
for lag in LAGS_NS:
    cs = np.array(results[lag]["cos"]); rs = np.array(results[lag]["rmsd"])
    print(f"  {lag:>8}  {cs.mean():>9.4f}  {cs.std():>8.4f}  {cs.min():>8.4f}  {cs.max():>8.4f}  {rs.mean():>16.3f}")

print()
print("="*86)
print("METRIC 2: Per-residue H cosine similarity vs lag")
print("  mean over (protein, residue) of cos_sim(H_anchor_i, H_future_i)")
print("="*86)
print(f"  {'lag(ns)':>8}  {'res_cos mean':>13}  {'res_cos std':>12}  {'frac < 0.9':>11}  {'frac < 0.99':>12}")
print(f"  {'-'*8}  {'-'*13}  {'-'*12}  {'-'*11}  {'-'*12}")
frac_ns = defaultdict(list)
for lag in LAGS_NS:
    allrc = torch.cat(results[lag]["res_cos"])
    f90 = (allrc < 0.9).float().mean().item()
    f99 = (allrc < 0.99).float().mean().item()
    print(f"  {lag:>8}  {allrc.mean():>13.4f}  {allrc.std():>12.4f}  {f90:>11.3f}  {f99:>12.3f}")

# ── persist a machine-readable copy ─────────────────────────────────────────
out = {"ckpt": CKPT, "dH": dH, "n_proteins": used, "lags_ns": LAGS_NS, "by_lag": {}}
for lag in LAGS_NS:
    cs = np.array(results[lag]["cos"]); rs = np.array(results[lag]["rmsd"])
    allrc = torch.cat(results[lag]["res_cos"])
    out["by_lag"][str(lag)] = {
        "cos_mean": float(cs.mean()), "cos_std": float(cs.std()),
        "cos_min": float(cs.min()), "cos_max": float(cs.max()),
        "ca_rmsd_mean": float(rs.mean()),
        "res_cos_mean": float(allrc.mean()), "res_cos_std": float(allrc.std()),
        "frac_res_cos_lt_0.9": float((allrc < 0.9).float().mean()),
        "frac_res_cos_lt_0.99": float((allrc < 0.99).float().mean()),
    }
Path("/liuyutian/protein_jepa_frame/tmp_claude/lag_metrics_fix2.json").write_text(json.dumps(out, indent=2))
print("\n[saved] tmp_claude/lag_metrics_fix2.json")
