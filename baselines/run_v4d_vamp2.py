#!/usr/bin/env python3
"""VAMP-2 scoring on V4-D D2 test set.

Adapted from run_vamp2_eval.py — takes 50 longest test trajectories,
extracts 64D latents, scores with deeptime VAMP.
"""
from __future__ import annotations

import json, time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import deeptime
from deeptime.decomposition import VAMP
print(f"deeptime {deeptime.__version__}")

from protein_jepa_frame.config import DataConfig, model_config_from_dict
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

# ========= Paths =========
CHECKPOINT = Path("/liuyutian/protein_jepa_frame/artifacts/phase4_v4_d2_reversible_kinetics_1k/checkpoint_step250.pt")
CONFIG_DIR = Path("/liuyutian/protein_jepa_frame/artifacts/phase4_v4_d2_reversible_kinetics_1k")
TEST_MANIFEST = Path("/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.test.filtered.jsonl")
OUTPUT_DIR = Path("/liuyutian/protein_jepa_frame/baselines/results/v4d_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
with open(CONFIG_DIR / "config.json") as f:
    cfg = json.load(f)
model_cfg = model_config_from_dict(cfg["model"])
model = ProteinJEPA(model_cfg).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model.eval()
print(f"Model: {model_cfg.latent_dim}D, step={ckpt.get('step', '?')}")

# Load test records
test_records = load_manifest(str(TEST_MANIFEST))

# Take 50 trajectories with the most frames
traj_map = OrderedDict()
for r in test_records:
    if r.trajectory_id not in traj_map and len(traj_map) < 50:
        traj_map[r.trajectory_id] = r
print(f"Using {len(traj_map)} trajectories")

# Build dataset with single-frame clips, stride=5
sparse_cfg = DataConfig(
    manifest_path=None, batch_size=16, num_workers=0,
    num_frames=1, frame_stride=1, window_stride=5,
    max_residues=cfg["data"]["max_residues"],
)

ds = TrajectoryClipDataset(list(traj_map.values()), sparse_cfg)
print(f"Dataset: {len(ds)} clips, {len(traj_map)} trajs")

# Build index map
idx_map = {}
for i in range(len(ds)):
    record, frames = ds._locate(i)
    idx_map[i] = {"trajectory_id": record.trajectory_id, "frame_idx": int(frames[0])}

loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_clips)

# Extract per-frame latents
print("Extracting latents...")
t0 = time.time()
traj_buffers = defaultdict(list)
batch_offset = 0
for batch in loader:
    B = batch["coords"].shape[0]
    coords = batch["coords"].to(device)
    residue_mask = batch["residue_mask"].to(device)
    aatype = batch["aatype"].to(device)
    residue_index = batch["residue_index"].to(device)
    chain_id = batch["chain_id"].to(device)

    latents = model.context_encoder(
        coords=coords, residue_mask=residue_mask,
        aatype=aatype, residue_index=residue_index, chain_id=chain_id,
        token_mask=None,
    )

    valid = residue_mask.unsqueeze(-1).float()
    pooled = (latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)

    for b in range(B):
        info = idx_map.get(batch_offset + b)
        if info:
            traj_buffers[info["trajectory_id"]].append(
                (info["frame_idx"], pooled[b, 0].detach().cpu().numpy())
            )
    batch_offset += B

# Build sorted sequences
traj_latents = {}
for tid, items in traj_buffers.items():
    items.sort(key=lambda x: x[0])
    traj_latents[tid] = np.stack([v[1] for v in items], axis=0)

print(f"Extracted {len(traj_latents)} trajectories in {time.time()-t0:.1f}s")
for tid, L in list(traj_latents.items())[:3]:
    print(f"  {tid}: {L.shape}")

# ========= VAMP-2 =========
print(f"\n{'='*60}")
print("VAMP-2 Scores (V4-D D2, Test Set, 64D latents)")
print(f"{'='*60}")

lags = [1, 20]
vamp_results = {}
for lag in lags:
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
            print(f"  VAMP lag={lag} {tid}: {e}")

    if scores:
        vamp_results[f"lag_{lag}"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_trajs": len(scores),
        }
        print(f"  Lag={lag}: VAMP2={np.mean(scores):.4f}±{np.std(scores):.4f} (n={len(scores)})")
    else:
        vamp_results[f"lag_{lag}"] = {"mean": None, "n_trajs": 0}
        print(f"  Lag={lag}: no data")

# Save
results = {
    "vamp2": vamp_results,
    "n_trajs": len(traj_latents),
    "latent_dim": model_cfg.latent_dim,
    "checkpoint_step": ckpt.get("step", "?"),
    "manifest": str(TEST_MANIFEST),
    "checkpoint": str(CHECKPOINT),
    "representation": "64D latents (mean pooled)",
}
out_path = OUTPUT_DIR / "v4d_test_vamp2_latents.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
print("Done!")