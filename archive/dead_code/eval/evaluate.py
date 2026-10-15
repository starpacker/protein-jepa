#!/usr/bin/env python3
"""Evaluation suite for Protein JEPA frame model.

Measures whether the model's latent representations encode:
  1. MSM kinetic states (linear probing)
  2. Temporal dynamics/ordering
  3. Meaningful latent structure (visualization)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# -- project imports --
from protein_jepa_frame.config import DataConfig, ModelConfig, load_config
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import TrajectoryRecord, load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

# sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("WARNING: sklearn not available — classification & PCA disabled")

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
ARTIFACTS = Path("/liuyutian/protein_jepa_frame/artifacts/mdcath_full_run")
CHECKPOINT = ARTIFACTS / "checkpoint_latest.pt"
EVAL_DIR = ARTIFACTS / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MSM_LABELS = Path("/liuyutian/protein_jepa/artifacts/msm_state_labels")
MSM_MANIFESTS = Path("/liuyutian/protein_jepa/artifacts/manifests")

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: Path, model: ProteinJEPA, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    info = {"epoch": ckpt.get("epoch", "?"), "step": ckpt.get("step", "?")}
    print(f"  Loaded checkpoint: epoch={info['epoch']}, step={info['step']}")
    return model


def load_msm_labels(labels_file: Path) -> dict:
    """Return the 'trajectories' dict keyed by trajectory_id."""
    with open(labels_file) as f:
        data = json.load(f)
    meta = data["_metadata"]
    print(f"  MSM labels: {meta['num_states']} states, "
          f"{data['summary']['n_labeled_frames']} frames, "
          f"{len(data['trajectories'])} trajectories")
    return data["trajectories"]


# ---------------------------------------------------------------------------
#  Wrapped dataset that also returns frame indices
# ---------------------------------------------------------------------------

class FrameIndexedDataset(Dataset):
    """Wraps TrajectoryClipDataset and also returns the raw frame indices per clip."""

    def __init__(self, records: list[TrajectoryRecord], config: DataConfig):
        self.inner = TrajectoryClipDataset(records, config)
        self.config = config
        self.records = records

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index: int):
        sample = self.inner[index]
        # Recover frame indices through _locate
        # (We need access to the inner's _locate — we'll compute it separately)
        return sample


# Actually, let's just build a sidecar mapping: index -> trajectory_id, frame_indices
def build_index_map(dataset) -> dict:
    """Build a mapping from dataset index to {'trajectory_id', 'frame_indices'}.

    Handles both TrajectoryClipDataset and Subset wrapping one.
    """
    if isinstance(dataset, Subset):
        inner = dataset.dataset
        if not isinstance(inner, TrajectoryClipDataset):
            return {}
        mapping = {}
        for subset_idx, orig_idx in enumerate(dataset.indices):
            record, frames = inner._locate(orig_idx)
            mapping[subset_idx] = {
                "trajectory_id": record.trajectory_id,
                "frame_indices": frames.tolist(),
            }
        return mapping
    elif isinstance(dataset, TrajectoryClipDataset):
        mapping = {}
        for i in range(len(dataset)):
            record, frames = dataset._locate(i)
            mapping[i] = {
                "trajectory_id": record.trajectory_id,
                "frame_indices": frames.tolist(),
            }
        return mapping
    return {}


# ---------------------------------------------------------------------------
#  Latent extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latents(
    model: ProteinJEPA,
    loader: DataLoader,
    index_map: dict,
    device: torch.device,
    batch_offset: int = 0,
) -> dict:
    """Extract context_encoder latents.

    Returns:
      latents: [N, D]  mean-pooled per clip
      trajectory_ids: list of str
      frame_indices: list of lists
    """
    model.eval()
    all_latents = []
    all_traj_ids = []
    all_frames = []

    for batch_idx, batch in enumerate(loader):
        B = batch["coords"].shape[0]
        coords = batch["coords"].to(device)
        residue_mask = batch["residue_mask"].to(device)
        aatype = batch["aatype"].to(device)
        residue_index = batch["residue_index"].to(device)
        chain_id = batch["chain_id"].to(device)

        # Encode all tokens
        latents = model.context_encoder(
            coords=coords, residue_mask=residue_mask,
            aatype=aatype, residue_index=residue_index, chain_id=chain_id,
            token_mask=None,
        )  # [B, T, L, D]

        # Mean-pool over valid residues per frame
        valid = residue_mask.unsqueeze(-1).float()  # [B, T, L, 1]
        pooled = (latents * valid).sum(dim=2)  # [B, T, D]
        counts = valid.sum(dim=2).clamp(min=1)
        pooled = pooled / counts  # [B, T, D]

        # Mean-pool over all frames in the clip
        clip_valid = valid.any(dim=2).float()
        clip_pooled = (pooled * clip_valid).sum(dim=1) / clip_valid.sum(dim=1).clamp(min=1)

        all_latents.append(clip_pooled.cpu().numpy())

        # Sidecar info
        for b in range(B):
            idx = batch_offset + batch_idx * loader.batch_size + b
            info = index_map.get(idx, {})
            all_traj_ids.append(info.get("trajectory_id", "unknown"))
            all_frames.append(info.get("frame_indices", []))

    return {
        "latents": np.concatenate(all_latents, axis=0),
        "trajectory_ids": all_traj_ids,
        "frame_indices": all_frames,
    }


# ---------------------------------------------------------------------------
#  Label assignment
# ---------------------------------------------------------------------------

def assign_msm_labels(
    traj_ids: list[str],
    frame_indices_list: list[list[int]],
    msm_trajs: dict,
    n_states: int = 6,
) -> np.ndarray:
    """Assign each clip an MSM state label (majority vote)."""
    labels = np.full(len(traj_ids), -1, dtype=np.int32)
    for i in range(len(traj_ids)):
        tid = traj_ids[i]
        if tid not in msm_trajs:
            continue
        traj_labels = msm_trajs[tid]["state_labels"]  # list of ints
        clip_frames = frame_indices_list[i]
        if not clip_frames:
            continue
        states = [traj_labels[f] for f in clip_frames if f < len(traj_labels)]
        if states:
            labels[i] = Counter(states).most_common(1)[0][0]
    return labels


# ---------------------------------------------------------------------------
#  Analysis functions
# ---------------------------------------------------------------------------

def run_pca_analysis(latents: np.ndarray, traj_ids: list[str], out_dir: Path) -> dict:
    """PCA of latent space, colored by trajectory."""
    if not SKLEARN_OK:
        return {"note": "sklearn unavailable"}

    print(f"\n{'='*60}")
    print("Latent Space PCA")
    print(f"  Samples: {len(latents)}, Dim: {latents.shape[1]}")
    print(f"{'='*60}")

    scaler = StandardScaler()
    X_s = scaler.fit_transform(latents)

    n_comp = min(10, X_s.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_s)
    evr = pca.explained_variance_ratio_

    print("  Explained variance:")
    for i in range(min(5, len(evr))):
        print(f"    PC{i+1}: {evr[i]:.4f}  (cum: {evr[:i+1].sum():.4f})")

    # --- Scatter colored by trajectory (up to 20) ---
    unique_trajs = list(set(traj_ids))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_trajs))))
    for idx, tid in enumerate(unique_trajs[:20]):
        mask = np.array([t == tid for t in traj_ids])
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[idx]], label=tid[:14], alpha=0.5, s=10)
    ax.set_title(f"PCA of Latent Space\n{len(unique_trajs)} trajectories, {len(latents)} clips")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.legend(markerscale=3, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_by_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"  Saved pca_by_trajectory.png")

    # --- Variance explained bar ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(range(1, len(evr) + 1), evr, alpha=0.7)
    ax1.set_xlabel("PC")
    ax1.set_ylabel("Explained Variance")
    cum = np.cumsum(evr)
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(evr) + 1), cum, "r-o", markersize=4)
    ax2.set_ylabel("Cumulative", color="r")
    ax2.axhline(0.9, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=150)
    plt.close(fig)

    return {
        "explained_variance_ratio": evr.tolist(),
        "cumulative_5": float(evr[:5].sum()) if len(evr) >= 5 else float(evr.sum()),
        "n_trajectories": len(unique_trajs),
    }


def run_dynamics_analysis(latents: np.ndarray, traj_ids: list[str],
                          frame_indices_list: list, out_dir: Path) -> dict:
    """Correlate latent distance vs temporal separation within each trajectory."""

    print(f"\n{'='*60}")
    print("Dynamics Encoding Analysis")
    print(f"  Latent distance vs temporal separation")
    print(f"{'='*60}")

    # Group indices by trajectory
    traj_groups = defaultdict(list)
    for i, tid in enumerate(traj_ids):
        traj_groups[tid].append(i)

    deltas = []
    for tid, indices in traj_groups.items():
        if len(indices) < 2:
            continue
        n_pairs = min(100, len(indices) * (len(indices) - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(len(indices), 2, replace=False)
            idx_i, idx_j = indices[i], indices[j]
            dist = float(np.linalg.norm(latents[idx_i] - latents[idx_j]))
            # Temporal separation: use first frame of each clip
            fi = frame_indices_list[idx_i]
            fj = frame_indices_list[idx_j]
            if fi and fj:
                td = abs(fi[0] - fj[0])
            else:
                continue
            deltas.append({"latent_distance": dist, "time_diff_frames": td})

    if len(deltas) < 10:
        print("  WARNING: too few pairs")
        return {"n_pairs": len(deltas), "note": "insufficient data"}

    time_diffs = np.array([d["time_diff_frames"] for d in deltas])
    latent_dists = np.array([d["latent_distance"] for d in deltas])

    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(time_diffs, latent_dists)
    sr, sp = spearmanr(time_diffs, latent_dists)
    print(f"  Pairs: {len(deltas)}")
    print(f"  Pearson r: {pr:.4f}  (p={pp:.2e})")
    print(f"  Spearman r: {sr:.4f}  (p={sp:.2e})")

    # Binned plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if time_diffs.max() > time_diffs.min():
        bins = np.logspace(np.log10(max(time_diffs.min(), 1)),
                           np.log10(max(time_diffs.max(), 2)), 20)
        centers, means, stds = [], [], []
        for k in range(len(bins) - 1):
            mask = (time_diffs >= bins[k]) & (time_diffs < bins[k + 1])
            if mask.sum() >= 3:
                centers.append(np.sqrt(bins[k] * bins[k + 1]))
                means.append(latent_dists[mask].mean())
                stds.append(latent_dists[mask].std())
        ax.errorbar(centers, means, yerr=stds, fmt="o-", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Temporal Separation (frames)")
        ax.set_ylabel("Mean Latent Distance")
        ax.set_title(f"Latent Distance vs Time\nPearson r={pr:.4f}, Spearman r={sr:.4f}")
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved temporal_correlation.png")

    return {"n_pairs": len(deltas), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def run_linear_probing(train_latents: np.ndarray, train_labels: np.ndarray,
                       val_latents: np.ndarray, val_labels: np.ndarray,
                       out_dir: Path, n_states: int = 6) -> dict:
    """Train linear classifiers on latents to predict MSM states."""
    if not SKLEARN_OK:
        return {"note": "sklearn unavailable"}

    print(f"\n{'='*60}")
    print("Linear Probing: latents → MSM state")
    print(f"  Train: {len(train_latents)}, Val: {len(val_latents)}")
    print(f"  Latent dim: {train_latents.shape[1]}, States: {n_states}")
    print(f"{'='*60}")

    # Filter labeled
    tm = train_labels >= 0
    vm = val_labels >= 0
    X_tr, y_tr = train_latents[tm], train_labels[tm]
    X_va, y_va = val_latents[vm], val_labels[vm]

    if len(X_tr) < 10 or len(X_va) < 10:
        return {"accuracy": 0.0, "note": f"too few labeled: train={len(X_tr)}, val={len(X_va)}"}

    print(f"  Labeled: train={len(X_tr)}, val={len(X_va)}")
    print(f"  State distribution (train): {dict(Counter(y_tr.tolist()))}")
    print(f"  State distribution (val):   {dict(Counter(y_va.tolist()))}")

    # Baseline: most-frequent class
    majority = Counter(y_tr.tolist()).most_common(1)[0][0]
    baseline_acc = (y_va == majority).mean()
    print(f"  Baseline (majority class): {baseline_acc:.4f}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    results = {}
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(solver="lbfgs",
                                 max_iter=5000, C=C, random_state=42)
        clf.fit(X_tr_s, y_tr)
        pred = clf.predict(X_va_s)
        acc = accuracy_score(y_va, pred)
        f1 = f1_score(y_va, pred, average="weighted")
        results[f"LR_C={C}"] = {"accuracy": float(acc), "f1": float(f1)}
        print(f"  LR C={C:.2f}:  acc={acc:.4f},  f1={f1:.4f}")

    best = max(results, key=lambda k: results[k]["accuracy"])
    best_C = float(best.split("=")[1])
    best_acc = results[best]["accuracy"]

    # Confusion matrix for best model
    clf = LogisticRegression(solver="lbfgs",
                             max_iter=5000, C=best_C, random_state=42)
    clf.fit(X_tr_s, y_tr)
    pred = clf.predict(X_va_s)
    cm = confusion_matrix(y_va, pred, labels=range(n_states))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Confusion Matrix (LR C={best_C}, acc={best_acc:.4f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    tick = range(n_states)
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved confusion_matrix.png")

    return {
        "best_accuracy": float(best_acc),
        "best_f1": float(results[best]["f1"]),
        "best_C": best_C,
        "baseline_accuracy": float(baseline_acc),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "all_results": results,
    }


def run_state_separation(latents: np.ndarray, labels: np.ndarray,
                         traj_ids: list[str], out_dir: Path, n_states: int = 6) -> dict:
    """Within vs between state distance in latent space."""
    if not SKLEARN_OK:
        return {"note": "sklearn unavailable"}

    print(f"\n{'='*60}")
    print("State Separation Analysis")
    print(f"{'='*60}")

    # Group by trajectory
    groups = defaultdict(lambda: {"latents": [], "labels": []})
    for i in range(len(latents)):
        if labels[i] >= 0:
            groups[traj_ids[i]]["latents"].append(latents[i])
            groups[traj_ids[i]]["labels"].append(labels[i])

    within = []
    between = []
    for tid, g in groups.items():
        L = np.array(g["latents"])
        lbls = np.array(g["labels"])
        uniq = set(lbls)
        if len(uniq) < 2 or len(L) < 5:
            continue
        for s in uniq:
            sm = L[lbls == s]
            if len(sm) < 2:
                continue
            for _ in range(min(50, len(sm) - 1)):
                i, j = np.random.choice(len(sm), 2, replace=False)
                within.append(float(np.linalg.norm(sm[i] - sm[j])))
            om = L[lbls != s]
            if len(om) > 0:
                for _ in range(min(50, len(sm), len(om))):
                    i = np.random.randint(len(sm))
                    j = np.random.randint(len(om))
                    between.append(float(np.linalg.norm(sm[i] - om[j])))

    if len(within) < 5 or len(between) < 5:
        return {"note": "insufficient data"}

    mw, mb = np.mean(within), np.mean(between)
    ratio = mb / max(mw, 1e-8)
    print(f"  Within-state mean distance:  {mw:.4f}")
    print(f"  Between-state mean distance: {mb:.4f}")
    print(f"  Separation ratio:            {ratio:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(within, bins=50, alpha=0.6, label="Within-state", density=True)
    ax.hist(between, bins=50, alpha=0.6, label="Between-state", density=True)
    ax.set_xlabel("Latent Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"State Separation in Latent Space\nRatio = {ratio:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "state_separation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved state_separation.png")

    return {"mean_within": float(mw), "mean_between": float(mb), "ratio": float(ratio)}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Protein JEPA Frame — Evaluation Suite")
    print("=" * 60)
    print(f"Output: {EVAL_DIR}")

    device = _device("auto")
    print(f"Device: {device}")

    # Load config
    with open(ARTIFACTS / "config.json") as f:
        cfg = json.load(f)
    model_cfg = ModelConfig(**cfg["model"])

    # Load model
    print("\nLoading model...")
    model = ProteinJEPA(model_cfg).to(device)
    load_checkpoint(CHECKPOINT, model, device)
    model.eval()

    # Load MSM labels
    print("\nLoading MSM labels...")
    val_msm = load_msm_labels(MSM_LABELS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered_labels.json")
    train_msm = load_msm_labels(MSM_LABELS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered_labels.json")

    # Build datasets
    print("\nBuilding datasets...")
    MAX_CLIPS = 500
    BATCH_SIZE = 8
    data_cfg = DataConfig(
        manifest_path=None,
        batch_size=BATCH_SIZE, num_workers=0,
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],  # use original stride
        window_stride=cfg["data"]["window_stride"],
        max_residues=cfg["data"]["max_residues"],
    )

    train_records = load_manifest(MSM_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.train.filtered.jsonl")
    val_records = load_manifest(MSM_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl")

    # Use a subset for speed
    train_dataset = TrajectoryClipDataset(train_records, data_cfg)
    val_dataset = TrajectoryClipDataset(val_records, data_cfg)
    print(f"  Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")

    if len(train_dataset) > MAX_CLIPS:
        idx = np.random.RandomState(42).choice(len(train_dataset), MAX_CLIPS, replace=False)
        train_dataset = Subset(train_dataset, idx.tolist())
    if len(val_dataset) > MAX_CLIPS:
        idx = np.random.RandomState(42).choice(len(val_dataset), MAX_CLIPS, replace=False)
        val_dataset = Subset(val_dataset, idx.tolist())

    # Build index maps
    train_map = build_index_map(train_dataset)
    val_map = build_index_map(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_clips)
    val_loader = DataLoader(val_dataset, batch_size=data_cfg.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_clips)

    # Extract latents
    print("\nExtracting latents (this may take a while)...")
    t0 = time.time()

    print("  Train...")
    train_out = extract_latents(model, train_loader, train_map, device, batch_offset=0)
    print(f"    {len(train_out['latents'])} clips")

    print("  Val...")
    val_out = extract_latents(model, val_loader, val_map, device, batch_offset=0)
    print(f"    {len(val_out['latents'])} clips")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Assign MSM labels
    print("\nAssigning MSM labels...")
    n_states = max(max(v["state_labels"]) for v in val_msm.values()) + 1 if val_msm else 6
    train_labels = assign_msm_labels(train_out["trajectory_ids"], train_out["frame_indices"], train_msm, n_states)
    val_labels = assign_msm_labels(val_out["trajectory_ids"], val_out["frame_indices"], val_msm, n_states)
    print(f"  Labeled: train={(train_labels >= 0).sum()}, val={(val_labels >= 0).sum()}")

    # Run analyses
    results = {}

    # 1. PCA
    results["pca"] = run_pca_analysis(val_out["latents"], val_out["trajectory_ids"], EVAL_DIR)

    # 2. Dynamics
    results["dynamics"] = run_dynamics_analysis(val_out["latents"], val_out["trajectory_ids"],
                                                 val_out["frame_indices"], EVAL_DIR)

    # 3. Linear probing
    lp = run_linear_probing(train_out["latents"], train_labels, val_out["latents"], val_labels, EVAL_DIR, n_states)
    results["linear_probing"] = lp

    # 4. State separation
    ss = run_state_separation(val_out["latents"], val_labels, val_out["trajectory_ids"], EVAL_DIR, n_states)
    results["state_separation"] = ss

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if "best_accuracy" in lp:
        print(f"  Linear probing accuracy:  {lp['best_accuracy']:.4f}  "
              f"(baseline: {lp.get('baseline_accuracy', 0):.4f})")
    if "pearson_r" in results["dynamics"]:
        print(f"  Temporal correlation:     Pearson r={results['dynamics']['pearson_r']:.4f}")
    if "ratio" in ss:
        print(f"  State separation ratio:   {ss['ratio']:.4f}")
    print(f"  PCA explained var (PC1-5): {results['pca'].get('cumulative_5', 'N/A'):.4f}")

    # Save
    with open(EVAL_DIR / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {EVAL_DIR / 'summary.json'}")
    print("Done!")


if __name__ == "__main__":
    main()