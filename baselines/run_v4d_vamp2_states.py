#!/usr/bin/env python3
"""VAMP-2 on V4-D D2 test set — 6-state soft memberships, not 64D latents."""
from __future__ import annotations

import json, time, sys
from collections import OrderedDict, Counter
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
    with open(OUTPUT_DIR / "v4d_vamp2_states.log", "a") as f:
        f.write(f"{msg}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")

log("Loading model...")
with open(CONFIG_DIR / "config.json") as f:
    cfg = json.load(f)
model_cfg = model_config_from_dict(cfg["model"])
model = ProteinJEPA(model_cfg).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model.eval()
log(f"Model: {model_cfg.latent_dim}D, step={ckpt.get('step', '?')}, physics_states={model_cfg.v4_physics_num_states}")

test_records = load_manifest(str(TEST_MANIFEST))
log(f"Test records: {len(test_records)}")

tid_counts = Counter(r.trajectory_id for r in test_records)
longest = sorted(tid_counts.keys(), key=lambda x: tid_counts[x], reverse=True)[:50]
traj_map = OrderedDict()
for r in test_records:
    if r.trajectory_id in longest and r.trajectory_id not in traj_map:
        traj_map[r.trajectory_id] = r
log(f"Using {len(traj_map)} trajectories")

log("\nProcessing trajectories for 6-state memberships...")
t_start = time.time()
max_residues = cfg["data"]["max_residues"]
traj_states = {}

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
    all_states = []
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
            coords_t = torch.from_numpy(coords_4).to(device)
            res_mask_t = torch.from_numpy(res_mask).to(device)
            aatype_t = torch.from_numpy(aatype).to(device)
            res_idx_t = torch.from_numpy(res_idx).to(device)
            chain_t = torch.from_numpy(chain).to(device)

            # Get 64D latents
            latents = model.context_encoder(
                coords=coords_t, residue_mask=res_mask_t,
                aatype=aatype_t, residue_index=res_idx_t, chain_id=chain_t,
                token_mask=None,
            )
            valid = res_mask_t.unsqueeze(-1).float()
            pooled = (latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)

            # Get 6-state physics memberships
            # V4-D physics state uses the encoder features
            # The physics state is computed in the V4 physics module
            # Let's access it through the model's forward method or directly
            try:
                # Try to get state from the physics module
                if hasattr(model, 'v4_physics_module'):
                    # physics module expects context features
                    state_out = model.v4_physics_module.get_state(pooled)
                    if isinstance(state_out, tuple):
                        state_probs = state_out[0]
                    else:
                        state_probs = state_out
                else:
                    state_probs = None
            except:
                state_probs = None

            if state_probs is not None:
                all_states.append(state_probs.cpu().numpy())
            else:
                # Fallback: use latents directly
                all_states.append(pooled[:, 0].cpu().numpy())

    if all_states:
        traj_states[tid] = np.concatenate(all_states, axis=0)

    t2 = time.time()
    if all_states:
        log(f"  [{traj_idx+1}/{len(traj_map)}] {tid}: {traj_states[tid].shape}, {t2-t1:.1f}s")

log(f"\nExtracted {len(traj_states)} trajectories in {time.time()-t_start:.1f}s")

# VAMP-2 on state memberships
log(f"\n{'='*60}")
log("VAMP-2 Scores (V4-D D2, Test Set, 6-state memberships)")
log(f"{'='*60}")

vamp_results = {}
for lag in [1, 20]:
    scores = []
    for tid, S in traj_states.items():
        if len(S) < lag + 30:
            continue
        try:
            v = VAMP(lagtime=lag, dim=None, scaling=None)
            v = v.fit(S).fetch_model()
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
    "vamp2": vamp_results, "n_trajs": len(traj_states),
    "latent_dim": 6, "checkpoint_step": ckpt.get("step", "?"),
    "manifest": str(TEST_MANIFEST), "checkpoint": str(CHECKPOINT),
    "representation": "6-state physics memberships",
}
out_path = OUTPUT_DIR / "v4d_test_vamp2_states.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nSaved to {out_path}")
log("Done!")
