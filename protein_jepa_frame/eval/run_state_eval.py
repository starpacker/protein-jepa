#!/usr/bin/env python3
"""Per-protein MSM state linear probing — single-pass extraction."""
from __future__ import annotations

import json, time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

ARTIFACTS = Path("/liuyutian/protein_jepa_frame/artifacts/mdcath_full_dynamics")
CHECKPOINT = ARTIFACTS / "checkpoint_latest.pt"
EVAL_DIR = ARTIFACTS / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
MSM_LABELS = Path("/liuyutian/protein_jepa/artifacts/msm_state_labels")
MSM_MANIFESTS = Path("/liuyutian/protein_jepa/artifacts/manifests")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open(ARTIFACTS / "config.json") as f:
    cfg = json.load(f)
model_cfg = ModelConfig(**cfg["model"])
model = ProteinJEPA(model_cfg).to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model", ckpt), strict=False)
model.eval()
print(f"Model: {model_cfg.latent_dim}D, step={ckpt.get('step', '?')}")

def load_msm_labels(file: Path) -> dict:
    with open(file) as f:
        data = json.load(f)
    return data["trajectories"]

val_msm = load_msm_labels(MSM_LABELS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered_labels.json")

data_cfg = DataConfig(
    manifest_path=None, batch_size=8, num_workers=0,
    num_frames=cfg["data"]["num_frames"],
    frame_stride=cfg["data"]["frame_stride"],
    window_stride=cfg["data"]["window_stride"],
    max_residues=cfg["data"]["max_residues"],
)

val_records = load_manifest(MSM_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl")

def assign_clip_label(record, frames, msm_trajs):
    tid = record.trajectory_id
    if tid not in msm_trajs:
        return -1
    traj_labels = msm_trajs[tid]["state_labels"]
    states = []
    for f in frames:
        f_idx = int(f)
        if f_idx < len(traj_labels) and traj_labels[f_idx] >= 0:
            states.append(traj_labels[f_idx])
    if not states:
        return -1
    return Counter(states).most_common(1)[0][0]

# Single pass: extract all val clips, group by protein_id
print("Building dataset...")
t0 = time.time()
dataset = TrajectoryClipDataset(val_records, data_cfg)
print(f"  {len(dataset)} total clips, built in {time.time()-t0:.1f}s")

# Limit to 8000 clips for speed
MAX_CLIPS = 8000
if len(dataset) > MAX_CLIPS:
    idx = np.random.RandomState(42).choice(len(dataset), MAX_CLIPS, replace=False)
    idx = sorted(idx.tolist())
    dataset = Subset(dataset, idx)

# Build index map (one pass)
inner = dataset.dataset if isinstance(dataset, Subset) else dataset
index_map = {}
for i in range(len(dataset)):
    if isinstance(dataset, Subset):
        orig_idx = dataset.indices[i]
        record, frames = inner._locate(orig_idx)
    else:
        record, frames = dataset._locate(i)
    index_map[i] = (record, frames)

# Extract latents (one pass)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_clips)

print("Extracting latents (single pass)...")
t0 = time.time()
all_latents, all_labels, all_pids = [], [], []
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
    clip_valid = valid.any(dim=2).float()
    clip_pooled = (pooled * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)
    for b in range(B):
        record, frames = index_map[batch_offset + b]
        label = assign_clip_label(record, frames, val_msm)
        if label >= 0:
            all_latents.append(clip_pooled[b].detach().cpu().numpy())
            all_labels.append(label)
            all_pids.append(record.protein_id)
    batch_offset += B

print(f"  {len(all_latents)} labeled clips, {time.time()-t0:.1f}s")

# Group by protein_id
protein_data = defaultdict(lambda: {"latents": [], "labels": []})
for lat, lbl, pid in zip(all_latents, all_labels, all_pids):
    protein_data[pid]["latents"].append(lat)
    protein_data[pid]["labels"].append(lbl)

print(f"  {len(protein_data)} proteins with data")
for pid in sorted(protein_data)[:5]:
    n = len(protein_data[pid]["latents"])
    states = set(protein_data[pid]["labels"])
    print(f"    {pid}: {n} clips, {len(states)} states")

# Per-protein evaluation (70/30 split within each protein)
print(f"\n{'='*60}")
print("PER-PROTEIN LINEAR PROBING (70/30 within validation set)")
print(f"{'='*60}")

all_accuracies, baselines, protein_details = [], [], {}
eval_proteins = [p for p in sorted(protein_data) if len(protein_data[p]["latents"]) >= 20]

for pid in eval_proteins:
    t0 = time.time()
    latents = np.array(protein_data[pid]["latents"])
    labels = np.array(protein_data[pid]["labels"])

    if len(set(labels)) < 2:
        continue

    n = len(latents)
    idx = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.7)
    train_lat, train_lbl = latents[idx[:n_train]], labels[idx[:n_train]]
    val_lat, val_lbl = latents[idx[n_train:]], labels[idx[n_train:]]

    if len(set(train_lbl)) < 2 or len(set(val_lbl)) < 2:
        continue

    majority = Counter(train_lbl.tolist()).most_common(1)[0][0]
    baseline = (val_lbl == majority).mean()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_lat)
    X_va = scaler.transform(val_lat)

    best_acc, best_C = 0, 1.0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(solver="lbfgs", max_iter=5000, C=C, random_state=42)
        clf.fit(X_tr, train_lbl)
        acc = accuracy_score(val_lbl, clf.predict(X_va))
        if acc > best_acc:
            best_acc, best_C = acc, C

    n_states = len(set(train_lbl))
    all_accuracies.append(best_acc)
    baselines.append(baseline)
    protein_details[pid] = {
        "accuracy": float(best_acc), "baseline": float(baseline),
        "n_train": int(len(train_lat)), "n_val": int(len(val_lat)),
        "n_states": n_states, "best_C": best_C, "time_s": round(time.time()-t0, 1),
    }
    print(f"  {pid:12s}: acc={best_acc:.4f} (base={baseline:.4f}), "
          f"n={len(train_lat)}/{len(val_lat)}, {n_states}s, "
          f"Δ={best_acc-baseline:+.4f}")

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
if all_accuracies:
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    median_acc = np.median(all_accuracies)
    mean_baseline = np.mean(baselines)
    improvements = [a - b for a, b in zip(all_accuracies, baselines)]
    n_above = sum(1 for a, b in zip(all_accuracies, baselines) if a > b)
    print(f"  Proteins evaluated: {len(all_accuracies)}")
    print(f"  Per-protein accuracy: {mean_acc:.4f} ± {std_acc:.4f}  (median: {median_acc:.4f})")
    print(f"  Mean baseline (majority class): {mean_baseline:.4f}")
    print(f"  Mean improvement: {np.mean(improvements):+.4f}")
    print(f"  Above baseline: {n_above}/{len(all_accuracies)} ({100*n_above/len(all_accuracies):.1f}%)")
    print(f"  Range: [{min(all_accuracies):.4f}, {max(all_accuracies):.4f}]")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_accuracies, bins=20, alpha=0.7, label="Per-protein accuracy", color="steelblue")
    ax.axvline(mean_acc, color="r", linestyle="--", label=f"Mean: {mean_acc:.4f}")
    ax.axvline(mean_baseline, color="g", linestyle=":", label=f"Baseline: {mean_baseline:.4f}")
    ax.set_xlabel("Accuracy"); ax.set_ylabel("Count")
    ax.set_title(f"Per-Protein MSM State Classification\n{model_cfg.latent_dim}D, step={ckpt.get('step', '?')}")
    ax.legend(); fig.tight_layout()
    fig.savefig(EVAL_DIR / "per_protein_accuracy.png", dpi=150)
    plt.close(fig); print(f"  Saved per_protein_accuracy.png")

results = {"per_protein_accuracy": {
    "mean": float(mean_acc) if all_accuracies else None,
    "std": float(std_acc) if all_accuracies else None,
    "median": float(median_acc) if all_accuracies else None,
    "mean_baseline": float(mean_baseline) if all_accuracies else None,
    "n_above_baseline": n_above if all_accuracies else 0,
    "n_total": len(all_accuracies) if all_accuracies else 0,
}, "protein_details": protein_details,
   "latent_dim": model_cfg.latent_dim,
   "checkpoint_step": ckpt.get("step", "?"),}
with open(EVAL_DIR / "state_eval_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {EVAL_DIR / 'state_eval_summary.json'}")
print("Done!")
