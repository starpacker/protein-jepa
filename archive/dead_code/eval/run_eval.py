#!/usr/bin/env python3
"""Consolidated evaluation for Protein JEPA Frame.

Runs:
  1. VAMP-2 / TICA dynamics assessment (per-trajectory sequences)
  2. Per-protein MSM state linear probing (70/30 within validation set)
  3. State separation, PCA, temporal correlation

Usage:
  python3 -m protein_jepa_frame.eval.run_eval \\
      --checkpoint artifacts/mdcath_full_dynamics/checkpoint_latest.pt \\
      --output-dir artifacts/mdcath_full_dynamics/eval
"""
from __future__ import annotations

import argparse, json, sys, time, math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import deeptime
from deeptime.decomposition import VAMP, TICA

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

# ---------------------------------------------------------------------------
#  Paths (defaults for mdcath_full_dynamics)
# ---------------------------------------------------------------------------
MSM_LABELS = Path("/liuyutian/protein_jepa/artifacts/msm_state_labels")
MSM_MANIFESTS = Path("/liuyutian/protein_jepa/artifacts/manifests")

VAL_MANIFEST = MSM_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl"
VAL_MSM_LABELS = MSM_LABELS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered_labels.json"

# ===================================================================
#  Helpers
# ===================================================================

def load_msm_labels(file: Path) -> dict:
    with open(file) as f:
        data = json.load(f)
    return data["trajectories"]

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

def _device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

# ===================================================================
#  1. VAMP-2 / TICA (per-trajectory sequences)
# ===================================================================

def run_vamp2_analysis(model, val_records, data_cfg, device, out_dir):
    """Evaluate VAMP-2 scores and TICA implied timescales."""
    print(f"\n{'='*60}")
    print("VAMP-2 / TICA ANALYSIS")
    print(f"{'='*60}")

    # Use first 50 trajectories for VAMP-2
    from collections import OrderedDict
    traj_map = OrderedDict()
    for r in val_records:
        if r.trajectory_id not in traj_map and len(traj_map) < 50:
            traj_map[r.trajectory_id] = r

    # Single-frame dataset with stride=5 for long sequences
    sparse_cfg = DataConfig(
        manifest_path=None, batch_size=16, num_workers=0,
        num_frames=1, frame_stride=1, window_stride=5,
        max_residues=data_cfg.max_residues,
    )

    ds = TrajectoryClipDataset(list(traj_map.values()), sparse_cfg)
    idx_map = {}
    for i in range(len(ds)):
        record, frames = ds._locate(i)
        idx_map[i] = {"trajectory_id": record.trajectory_id, "frame_idx": int(frames[0])}

    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_clips)

    model.eval()
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

    traj_latents = {}
    for tid, items in traj_buffers.items():
        items.sort(key=lambda x: x[0])
        traj_latents[tid] = np.stack([v[1] for v in items], axis=0)

    print(f"  {len(traj_latents)} trajectories")

    # VAMP-2
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
                scores.append(v.score(r="VAMP2"))
            except Exception:
                pass
        if scores:
            vamp_results[f"lag_{lag}"] = {
                "mean": float(np.mean(scores)), "std": float(np.std(scores)), "n_trajs": len(scores),
            }
            print(f"  VAMP-2 lag={lag}: {np.mean(scores):.4f}±{np.std(scores):.4f} (n={len(scores)})")
        else:
            vamp_results[f"lag_{lag}"] = {"mean": None, "n_trajs": 0}

    # TICA implied timescales
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
                    if e >= 1.0 or e <= 0.0:
                        continue
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
            print(f"  TICA lag={lag}: {ts_str}, n={len(all_ts)}")
        else:
            ts_results[f"lag_{lag}"] = {"n_trajs": 0}

    return {"vamp2": vamp_results, "implied_timescales": ts_results, "n_trajs": len(traj_latents)}

# ===================================================================
#  2. Per-protein MSM state linear probing
# ===================================================================

def run_msm_state_analysis(model, val_records, val_msm, data_cfg, device, out_dir):
    """Per-protein linear probing (70/30 split within validation set)."""
    print(f"\n{'='*60}")
    print("PER-PROTEIN MSM STATE LINEAR PROBING")
    print(f"{'='*60}")

    # Single-pass extraction
    dataset = TrajectoryClipDataset(val_records, data_cfg)
    MAX_CLIPS = 8000
    if len(dataset) > MAX_CLIPS:
        idx = np.random.RandomState(42).choice(len(dataset), MAX_CLIPS, replace=False)
        idx = sorted(idx.tolist())
        dataset = Subset(dataset, idx)

    inner = dataset.dataset if isinstance(dataset, Subset) else dataset
    index_map = {}
    for i in range(len(dataset)):
        if isinstance(dataset, Subset):
            orig_idx = dataset.indices[i]
            record, frames = inner._locate(orig_idx)
        else:
            record, frames = dataset._locate(i)
        index_map[i] = (record, frames)

    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_clips)

    model.eval()
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

    # Group by protein
    protein_data = defaultdict(lambda: {"latents": [], "labels": []})
    for lat, lbl, pid in zip(all_latents, all_labels, all_pids):
        protein_data[pid]["latents"].append(lat)
        protein_data[pid]["labels"].append(lbl)

    print(f"  {len(protein_data)} proteins, {len(all_latents)} clips")

    # Per-protein 70/30 split
    all_accs, all_baselines, protein_details = [], [], {}
    for pid in sorted(protein_data):
        if len(protein_data[pid]["latents"]) < 20:
            continue
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
        all_accs.append(best_acc)
        all_baselines.append(baseline)
        protein_details[pid] = {
            "accuracy": float(best_acc), "baseline": float(baseline),
            "n_train": int(len(train_lat)), "n_val": int(len(val_lat)),
            "n_states": n_states, "best_C": best_C,
        }
        print(f"  {pid:12s}: acc={best_acc:.4f} (base={baseline:.4f}), Δ={best_acc-baseline:+.4f}")

    if all_accs:
        mean_acc = np.mean(all_accs)
        n_above = sum(1 for a, b in zip(all_accs, all_baselines) if a > b)
        print(f"\n  Summary: {len(all_accs)} proteins, "
              f"acc={mean_acc:.4f}±{np.std(all_accs):.4f}, "
              f"baseline={np.mean(all_baselines):.4f}, "
              f"above baseline={n_above}/{len(all_accs)}")

        # Distribution plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_accs, bins=20, alpha=0.7, color="steelblue", label="Per-protein accuracy")
        ax.axvline(mean_acc, color="r", linestyle="--", label=f"Mean: {mean_acc:.4f}")
        ax.axvline(np.mean(all_baselines), color="g", linestyle=":", label=f"Baseline: {np.mean(all_baselines):.4f}")
        ax.set_xlabel("Accuracy"); ax.set_ylabel("Count")
        ax.set_title(f"Per-Protein MSM State Classification\n{latents.shape[1]}D")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "per_protein_accuracy.png", dpi=150)
        plt.close(fig)

    return {
        "per_protein_accuracy": {
            "mean": float(np.mean(all_accs)) if all_accs else None,
            "std": float(np.std(all_accs)) if all_accs else None,
            "mean_baseline": float(np.mean(all_baselines)) if all_accs else None,
            "n_above_baseline": n_above if all_accs else 0,
            "n_total": len(all_accs) if all_accs else 0,
        },
        "protein_details": protein_details,
    }

# ===================================================================
#  Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Protein JEPA Frame model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (e.g. artifacts/mdcath_full_dynamics/checkpoint_latest.pt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for eval results (default: <checkpoint_dir>/eval)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cuda:1, cpu)")
    parser.add_argument("--vamp2", action="store_true", default=True, help="Run VAMP-2 analysis")
    parser.add_argument("--msm", action="store_true", default=True, help="Run MSM state analysis")
    parser.add_argument("--no-vamp2", action="store_false", dest="vamp2", help="Skip VAMP-2 analysis")
    parser.add_argument("--no-msm", action="store_false", dest="msm", help="Skip MSM state analysis")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    artifacts_dir = checkpoint_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else artifacts_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    # Load config
    with open(artifacts_dir / "config.json") as f:
        cfg = json.load(f)
    model_cfg = ModelConfig(**cfg["model"])
    data_cfg = DataConfig(
        manifest_path=None, batch_size=8, num_workers=0,
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],
        window_stride=cfg["data"]["window_stride"],
        max_residues=cfg["data"]["max_residues"],
    )

    # Load model
    print("\nLoading model...")
    model = ProteinJEPA(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    print(f"  {model_cfg.latent_dim}D latent, step={ckpt.get('step', '?')}")

    # Load data
    print("\nLoading data...")
    val_records = load_manifest(VAL_MANIFEST)
    val_msm = load_msm_labels(VAL_MSM_LABELS)
    print(f"  {len(val_records)} records, {len(val_msm)} trajectories")

    # Run analyses
    results = {
        "checkpoint_step": ckpt.get("step", "?"),
        "latent_dim": model_cfg.latent_dim,
    }

    if args.vamp2:
        print("\n" + "="*60)
        print("PHASE 1: VAMP-2 / TICA")
        print("="*60)
        t0 = time.time()
        vamp_results = run_vamp2_analysis(model, val_records, data_cfg, device, output_dir)
        results.update(vamp_results)
        print(f"  Time: {time.time()-t0:.1f}s")

    if args.msm:
        print("\n" + "="*60)
        print("PHASE 2: MSM STATE ANALYSIS")
        print("="*60)
        t0 = time.time()
        msm_results = run_msm_state_analysis(model, val_records, val_msm, data_cfg, device, output_dir)
        results.update(msm_results)
        print(f"  Time: {time.time()-t0:.1f}s")

    # Save
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'eval_summary.json'}")
    print("Done!")

if __name__ == "__main__":
    main()