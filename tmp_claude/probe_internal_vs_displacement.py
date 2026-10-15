"""How much does INTERNAL geometry change vs how much do CA positions move?

If internal geometry (CA-CA distance matrix, torsions) is nearly constant while
CA positions move several Å, then an SE(3)-invariant encoder CANNOT recover the
displacement -- the info is literally not in its input. This would prove the
Stage A collapse is architectural (invariant encoder), not a training problem.

Reports per lag:
  - CA RMSD (rigid-aligned) : how far atoms move in space
  - distance-matrix RMSD    : how much the INVARIANT internal geometry changes
  - ratio                   : internal change / spatial change
"""
import sys, json, random
from pathlib import Path
import numpy as np
import torch, h5py

sys.path.insert(0, '/liuyutian/protein_jepa_frame')
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset, TrajectoryRecord

MANIFEST = "/liuyutian/protein_jepa_frame/artifacts/mdcath_manifest_200dom/train.jsonl"
LAGS_NS = [1, 4, 16, 64, 96]
N_PROTEINS = 40

entries = [json.loads(l) for l in open(MANIFEST)]
seen = {}
for e in entries:
    if e["domain_id"] not in seen: seen[e["domain_id"]] = e
domains = [e for e in seen.values() if e["temperature_k"] == 320]
random.seed(0); random.shuffle(domains)

dc = DataConfig(manifest_path=MANIFEST)
ds = build_dataset(dc)
_topo = ds._mdcath_topology; _gather = ds._gather_backbone

def kabsch_rmsd(P, Q):
    pc = P - P.mean(0); qc = Q - Q.mean(0)
    H = pc.T @ qc
    U,S,Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1,1,d]) @ U.T
    return np.sqrt(((pc @ R.T - qc)**2).sum(-1).mean())

res = {lag: {"ca": [], "dist": [], "dist_rel": []} for lag in LAGS_NS}
used = 0
for e in domains:
    if used >= N_PROTEINS: break
    try:
        fi = np.array([0]+LAGS_NS)
        with h5py.File(e["h5_path"],"r") as h:
            grp = h[e["metadata"]["hdf5_group"]]
            if grp["coords"].shape[0] <= max(LAGS_NS): continue
            aa = np.asarray(grp["coords"][fi.tolist()], dtype=np.float32)
        rec = TrajectoryRecord(path=Path(e["h5_path"]), protein_id=e["protein_id"],
            trajectory_id=e["trajectory_id"], domain_id=e["domain_id"],
            temperature_k=e["temperature_k"], replica_id=e["replica_id"],
            frame_interval_ps=e["frame_interval_ps"], first_frame_time_ps=0.0,
            metadata=e["metadata"])
        topo = _topo(rec)
        coords, am = _gather(aa, topo)
        rmask = am.all(-1)
    except Exception: continue
    v = rmask[0]
    if v.sum() < 10: continue
    ca = coords[:, :, 1, :][:, v, :]   # [F, Lv, 3]
    ca0 = ca[0]
    D0 = np.linalg.norm(ca0[:,None]-ca0[None], axis=-1)  # [Lv,Lv]
    for k, lag in enumerate(LAGS_NS):
        caf = ca[k+1]
        res[lag]["ca"].append(kabsch_rmsd(caf, ca0))
        Df = np.linalg.norm(caf[:,None]-caf[None], axis=-1)
        drms = np.sqrt(((Df-D0)**2).mean())
        res[lag]["dist"].append(drms)
        # relative: normalise distance change by mean pairwise distance
        res[lag]["dist_rel"].append(drms / (D0.mean()+1e-6))
    used += 1

print(f"[internal_vs_displacement] {used} proteins")
print("="*72)
print("  CA RMSD = spatial motion (Å); dist-mat RMSD = INVARIANT internal change")
print("="*72)
print(f"  {'lag(ns)':>8}  {'CA RMSD(Å)':>11}  {'distmat RMSD(Å)':>15}  {'internal/spatial':>16}")
print(f"  {'-'*8}  {'-'*11}  {'-'*15}  {'-'*16}")
for lag in LAGS_NS:
    ca = np.mean(res[lag]["ca"]); dm = np.mean(res[lag]["dist"])
    print(f"  {lag:>8}  {ca:>11.3f}  {dm:>15.3f}  {dm/ca:>16.3f}")
