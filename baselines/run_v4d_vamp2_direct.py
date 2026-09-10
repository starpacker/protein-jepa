#!/usr/bin/env python3
"""VAMP-2 on V4-D D2 test set — direct HDF5 read, no TrajectoryClipDataset."""
from __future__ import annotations

import json, time, sys
from collections import OrderedDict, defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import h5py
from deeptime.decomposition import VAMP

from protein_jepa_frame.config import model_config_from_dict
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

CHECKPOINT = Path("/liuyutian/protein_jepa_frame/artifacts/phase4_v4_d2_reversible_kinetics_1k/checkpoint_step250.pt")
CONFIG_DIR = Path("/liuyutian/protein_jepa_frame/artifacts/phase4_v4_d2_reversible_kinetics_1k")
TEST_MANIFEST = Path("/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.test.filtered.jsonl")
OUTPUT_DIR = Path("/liuyutian/protein_jepa_frame/baselines/results/v4d_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(OUTPUT_DIR / "v4d_vamp2.log", "a") as f:
        f.write(f"{msg}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")

log("Loading model...")
t0 = time.time()
with open(CONFIG_DIR / "config.json") as f:
    cfg = json.load(f)
model_cfg = model_config_from_dict(cfg["model"])
model = ProteinJEPA(model_cfg).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model.eval()
log(f"Model: {model_cfg.latent_dim}D, step={ckpt.get('step', '?')}, loaded in {time.time()-t0:.1f}s")

test_records = load_manifest(str(TEST_MANIFEST))
log(f"Test records: {len(test_records)}")

tid_counts = Counter(r.trajectory_id for r in test_records)
longest = sorted(tid_counts.keys(), key=lambda x: tid_counts[x], reverse=True)[:50]
traj_map = OrderedDict()
for r in test_records:
    if r.trajectory_id in longest and r.trajectory_id not in traj_map:
        traj_map[r.trajectory_id] = r
log(f"Using {len(traj_map)} trajectories")

log("\nProcessing trajectories...")
t_start = time.time()
max_residues = cfg["data"]["max_residues"]
traj_latents = {}

for traj_idx, (tid, record) in enumerate(traj_map.items()):
    t1 = time.time()
    group_name = record.metadata.get("hdf5_group", "")

    with h5py.File(record.path, "r") as handle:
        grp = handle[str(group_name)]
        coords = np.asarray(grp["coords"], dtype=np.float32)
        n_frames = coords.shape[0]

    stride = 5
    frame_idx = np.arange(0, n_frames, stride, dtype=np.int64)
    n_sampled = len(frame_idx)
    if n_sampled < 30:
        log(f"  [{traj_idx+1}/{len(traj_map)}] {tid}: skip, {n_sampled} frames")
        continue

    batch_size = 32
    all_pooled = []
    n_atoms = coords.shape[1]

    for bs in range(0, n_sampled, batch_size):
        be = min(bs + batch_size, n_sampled)
        batch_frames = frame_idx[bs:be].tolist()

        with h5py.File(record.path, "r") as handle:
            grp = handle[str(group_name)]
            batch_coords = np.asarray(grp["coords"][batch_frames], dtype=np.float32)

        B = batch_coords.shape[0]
        L = n_atoms // 4
        if max_residues is not None:
            L = min(L, max_residues)

        coords_4 = batch_coords[:, :L*4].reshape(B, L, 4, 3)
        res_mask = np.ones((B, L), dtype=bool)
        aatype = np.zeros(L, dtype=np.int64)
        res_idx = np.arange(L, dtype=np.int64)
        chain = np.zeros(L, dtype=np.int64)

        with torch.no_grad():
            latents = model.context_encoder(
                coords=torch.from_numpy(coords_4).to(device),
                residue_mask=torch.from_numpy(res_mask).to(device),
                aatype=torch.from_numpy(aatype).to(device),
                residue_index=torch.from_numpy(res_idx).to(device),
                chain_id=torch.from_numpy(chain).to(device),
                token_mask=None,
            )
            valid = torch.from_numpy(res_mask).to(device).unsqueeze(-1).float()
            pooled = (latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)
            all_pooled.append(pooled[:, 0].cpu().numpy())

    if all_pooled:
        traj_latents[tid] = np.concatenate(all_pooled, axis=0)

    t2 = time.time()
    log(f"  [{traj_idx+1}/{len(traj_map)}] {tid}: {traj_latents[tid].shape}, {t2-t1:.1f}s")

log(f"\nExtracted {len(traj_latents)} trajectories in {time.time()-t_start:.1f}s")

log(f"\n{'='*60}")
log("VAMP-2 Scores (V4-D D2, Test Set, 64D latents)")
log(f"{'='*60}")

vamp_results = {}
for lag in [1, 20]:
    scores = []
    for tid, L in traj_latents.items():
        if len(L) < lag + 30:
            continue
        try:
            v = VAMP(lagtime=lag, dim=None, scaling=None)
            v = v.fit(L).fetch_model()
            s = v.score(r="VAMP2")
            scores.append(s)
        except Exception as e:
            log(f"  VAMP lag={lag} {tid}: {e}")
    if scores:
        vamp_results[f"lag_{lag}"] = {
            "mean": float(np.mean(scores)), "std": float(np.std(scores)), "n_trajs": len(scores),
        }
        log(f"  Lag={lag}: VAMP2={np.mean(scores):.4f}+-{np.std(scores):.4f} (n={len(scores)})")
    else:
        vamp_results[f"lag_{lag}"] = {"mean": None, "n_trajs": 0}
        log(f"  Lag={lag}: no data")

results = {
    "vamp2": vamp_results, "n_trajs": len(traj_latents),
    "latent_dim": model_cfg.latent_dim, "checkpoint_step": ckpt.get("step", "?"),
    "manifest": str(TEST_MANIFEST), "checkpoint": str(CHECKPOINT),
    "representation": "64D latents (mean pooled, direct HDF5 read)",
}
out_path = OUTPUT_DIR / "v4d_test_vamp2_latents.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nSaved to {out_path}")
log("Done!")
