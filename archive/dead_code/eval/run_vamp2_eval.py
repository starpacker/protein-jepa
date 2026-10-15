#!/usr/bin/env python3
"""VAMP-2 / TICA evaluation with proper per-trajectory sequences.

Uses single-frame clips (num_frames=1, frame_stride=1, window_stride=5)
on 50 validation trajectories to build long sequences for VAMP-2 scoring.
"""
from __future__ import annotations

import json, time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import deeptime
from deeptime.decomposition import VAMP, TICA
print(f"deeptime {deeptime.__version__}")

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

# ========= Paths =========
ARTIFACTS = Path("/liuyutian/protein_jepa_frame/artifacts/mdcath_full_dynamics")
CHECKPOINT = ARTIFACTS / "checkpoint_latest.pt"
EVAL_DIR = ARTIFACTS / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
MSM_MANIFESTS = Path("/liuyutian/protein_jepa/artifacts/manifests")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
with open(ARTIFACTS / "config.json") as f:
    cfg = json.load(f)
model_cfg = ModelConfig(**cfg["model"])
model = ProteinJEPA(model_cfg).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model.eval()
print(f"Model: {model_cfg.latent_dim}D, step={ckpt.get('step', '?')}")

# Load validation records
val_records = load_manifest(
    MSM_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl"
)

# Take 50 trajectories with the most frames
traj_map = OrderedDict()
for r in val_records:
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
print("VAMP-2 Scores")
print(f"{'='*60}")

lags = [1, 2, 5, 10, 20]
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

# ========= Implied Timescales =========
print(f"\n{'='*60}")
print("Implied Timescales (TICA)")
print(f"{'='*60}")

ts_lags = [2, 5, 10, 20]
ts_results = {}
for lag in ts_lags:
    all_ts = []
    for tid, L in traj_latents.items():
        if len(L) < lag * 3:
            continue
        try:
            n_comp = min(5, L.shape[1], len(L) // 3)
            tica = TICA(lagtime=lag, dim=n_comp)
            tica = tica.fit(L).fetch_model()
            evals = tica.singular_values
            ts = []
            for e in evals:
                if e >= 1.0:
                    continue  # skip inf
                elif e <= 0.0:
                    continue
                else:
                    ts.append(-lag * 1000.0 / np.log(e))
            if ts:
                all_ts.append(ts)
        except Exception:
            pass

    if all_ts:
        max_len = min(5, len(all_ts[0]))
        means = [np.mean([t[i] for t in all_ts if len(t) > i]) for i in range(max_len)]
        ts_results[f"lag_{lag}"] = {"mean_timescales_ps": [float(m) for m in means], "n_trajs": len(all_ts)}
        ts_str = ", ".join([f"ts{i+1}={m:.0f}ps" for i, m in enumerate(means[:3])])
        print(f"  Lag={lag}: {ts_str}, n={len(all_ts)}")
    else:
        ts_results[f"lag_{lag}"] = {"n_trajs": 0}
        print(f"  Lag={lag}: no data")

# ========= Baseline comparison =========
print(f"\n{'='*60}")
print("COMPARISON WITH BASELINE (mdcath_full_run)")
print(f"{'='*60}")
print(f"  {'Metric':<25} {'Baseline':>12} {'Current':>12} {'Improvement':>12}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
baseline_means = {1: 1.172, 2: 1.141, 5: 1.115, 10: 1.091, 20: 1.090}
for lag in lags:
    k = f"lag_{lag}"
    if k in vamp_results and vamp_results[k].get("mean"):
        bm = baseline_means.get(lag, 0)
        cm = vamp_results[k]["mean"]
        stdv = vamp_results[k]["std"]
        imp = cm / bm if bm > 0 else 0
        print(f"  VAMP-2 lag={lag:<5} {bm:>8.4f}±0.357 {cm:>8.4f}±{stdv:.4f} {imp:>6.2f}x")

# Save
results = {
    "vamp2": vamp_results,
    "implied_timescales": ts_results,
    "n_trajs": len(traj_latents),
    "latent_dim": model_cfg.latent_dim,
    "checkpoint_step": ckpt.get("step", "?"),
}
with open(EVAL_DIR / "vamp2_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {EVAL_DIR / 'vamp2_summary.json'}")
print("Done!")