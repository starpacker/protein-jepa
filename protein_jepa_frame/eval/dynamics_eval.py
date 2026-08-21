#!/usr/bin/env python3
"""Enhanced dynamics evaluation using deeptime.

Uses the official deeptime library for:
  - VAMP-2 score (variational approach for Markov processes)
  - Implied timescales from latent space
  - TICA decomposition of latent trajectories
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import deeptime
from deeptime.decomposition import VAMP, TICA

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.jepa import ProteinJEPA

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
ARTIFACTS = Path("/liuyutian/protein_jepa_frame/artifacts/mdcath_full_run")
CHECKPOINT = ARTIFACTS / "checkpoint_latest.pt"
EVAL_DIR = ARTIFACTS / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
MSM_MANIFESTS = Path("/liuyutian/protein_jepa/artifacts/manifests")


# ---------------------------------------------------------------------------
#  Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_trajectory_latents(
    model: ProteinJEPA,
    records: list,
    data_cfg: DataConfig,
    device: torch.device,
    max_trajs: int = 50,
) -> dict[str, np.ndarray]:
    """Extract latents frame-by-frame, preserving temporal order.

    Returns dict: trajectory_id -> [T, D] latents (sorted by time)
    """
    model.eval()

    # Deduplicate records by trajectory_id (first 50)
    traj_map = OrderedDict()
    for r in records:
        if r.trajectory_id not in traj_map and len(traj_map) < max_trajs:
            traj_map[r.trajectory_id] = r

    # Subsample frames: every 5th frame
    sparse_cfg = DataConfig(
        manifest_path=None, batch_size=16, num_workers=0,
        num_frames=1, frame_stride=1, window_stride=5,
        max_residues=data_cfg.max_residues,
    )

    ds = TrajectoryClipDataset(list(traj_map.values()), sparse_cfg)
    print(f"  Dataset: {len(ds)} clips, {len(traj_map)} trajectories (stride=5)")

    # Map each dataset index to trajectory_id + frame_idx
    idx_map = {}
    for i in range(len(ds)):
        record, frames = ds._locate(i)
        idx_map[i] = {
            "trajectory_id": record.trajectory_id,
            "frame_idx": int(frames[0]),
        }

    loader = DataLoader(ds, batch_size=sparse_cfg.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_clips)

    # Extract
    traj_buffers: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
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
        )  # [B, 1, L, D]

        valid = residue_mask.unsqueeze(-1).float()
        pooled = (latents * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)  # [B, 1, D]

        for b in range(B):
            info = idx_map.get(batch_offset + b)
            if info:
                traj_buffers[info["trajectory_id"]].append(
                    (info["frame_idx"], pooled[b, 0].cpu().numpy())
                )
        batch_offset += B

    # Sort by frame index and concatenate
    result = {}
    for tid, items in traj_buffers.items():
        items.sort(key=lambda x: x[0])
        result[tid] = np.stack([v[1] for v in items], axis=0)
    return result


# ---------------------------------------------------------------------------
#  VAMP-2 scoring using deeptime
# ---------------------------------------------------------------------------

def compute_vamp2_scores(
    traj_latents: dict[str, np.ndarray],
    lags: list[int] = None,
    n_components: int = 10,
) -> dict:
    """Compute VAMP-2 score per trajectory using deeptime."""
    if lags is None:
        lags = [1, 2, 5, 10, 20]

    results = {}
    for lag in lags:
        scores = []
        for tid, L in traj_latents.items():
            if len(L) < lag + 30:
                continue
            try:
                v = VAMP(lagtime=lag, n_components=n_components, scaling=None)
                v = v.fit(L).fetch_model()
                vamp2 = v.score(L, method="VAMP2")
                scores.append(vamp2)
            except Exception:
                continue

        if scores:
            results[f"lag_{lag}"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "n_trajs": len(scores),
                "lag_frames": lag,
            }
        else:
            results[f"lag_{lag}"] = {"mean": None, "n_trajs": 0, "lag_frames": lag}

    return results


# ---------------------------------------------------------------------------
#  Implied timescales via deeptime TICA
# ---------------------------------------------------------------------------

def compute_implied_timescales(
    traj_latents: dict[str, np.ndarray],
    lags: list[int] = None,
    dt_ps: float = 1000.0,
    n_timescales: int = 5,
) -> dict:
    """Compute implied timescales from latent trajectories using deeptime TICA.

    TICA eigenvalues λ_i give implied timescales: t_i = -τ / ln|λ_i|.
    """
    if lags is None:
        lags = [2, 5, 10, 20]

    results = {}
    for lag in lags:
        all_ts = []
        for tid, L in traj_latents.items():
            if len(L) < lag * 3:
                continue
            try:
                tica = TICA(lagtime=lag, dim=n_timescales)
                tica = tica.fit(L).fetch_model()
                # Get eigenvalues
                evals = tica.eigenvalues_
                evals = evals[evals > 0]  # only positive for log
                ts = []
                for e in evals[:n_timescales]:
                    if e >= 1.0:
                        ts.append(float("inf"))
                    else:
                        ts.append(-lag * dt_ps / np.log(e))
                all_ts.append(ts)
            except Exception:
                continue

        if all_ts:
            arr = np.array([t[:n_timescales] for t in all_ts if len(t) >= n_timescales])
            if len(arr) > 0:
                finite = arr[~np.isinf(arr) & ~np.isnan(arr)]
                results[f"lag_{lag}"] = {
                    "mean_timescales_ps": [float(np.mean(arr[:, i])) for i in range(min(n_timescales, arr.shape[1]))],
                    "n_trajs": len(arr),
                    "lag_frames": lag,
                    "dt_ps": dt_ps,
                }
                continue
        results[f"lag_{lag}"] = {"n_trajs": 0, "lag_frames": lag}

    return results


# ---------------------------------------------------------------------------
#  Latent-space TICA projections (visualization)
# ---------------------------------------------------------------------------

def compute_tica_projection(
    traj_latents: dict[str, np.ndarray],
    lag: int = 5,
    out_dir: Path = EVAL_DIR,
) -> dict:
    """Fit TICA on all trajectories concatenated, then project each one."""
    # Concatenate all trajectories (with NaN separators)
    all_data = []
    boundaries = []
    for tid, L in traj_latents.items():
        if len(L) < 20:
            continue
        all_data.append(L)
        boundaries.append(len(L))

    if len(all_data) < 3:
        return {"note": "too few trajectories"}

    X = np.concatenate(all_data, axis=0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    tica = TICA(lagtime=lag, dim=5)
    tica = tica.fit(X_s).fetch_model()

    evals = tica.eigenvalues_
    print(f"\nTICA on all trajectories (lag={lag}):")
    for i in range(min(5, len(evals))):
        print(f"  IC{i+1} eigenvalue: {evals[i]:.6f}")

    # Project
    Y = tica.transform(X_s)

    # Plot: first 2 ICs, colored by trajectory
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(all_data))))
    offset = 0
    for idx in range(min(10, len(all_data))):
        n = boundaries[idx]
        ax.scatter(Y[offset:offset+n, 0], Y[offset:offset+n, 1],
                   c=[colors[idx]], alpha=0.5, s=5, label=f"Traj {idx}")
        offset += n
    ax.set_title(f"TICA Projection of Latent Space (lag={lag})")
    ax.set_xlabel("IC1")
    ax.set_ylabel("IC2")
    ax.legend(markerscale=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "tica_projection.png", dpi=150)
    plt.close(fig)
    print(f"  Saved tica_projection.png")

    return {
        "eigenvalues": [float(e) for e in evals[:5]],
        "lag_frames": lag,
        "n_trajs": len(all_data),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Dynamics Evaluation (deeptime)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config and model
    with open(ARTIFACTS / "config.json") as f:
        cfg = json.load(f)
    model_cfg = ModelConfig(**cfg["model"])
    model = ProteinJEPA(model_cfg).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    print(f"Model: {model_cfg.latent_dim}D latent, {cfg['model']['encoder_depth']} layers")

    # Load validation records
    val_records = load_manifest(
        MSM_MANIFESTS / "mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl"
    )
    data_cfg = DataConfig(
        manifest_path=None, batch_size=4, num_workers=0,
        num_frames=cfg["data"]["num_frames"],
        frame_stride=cfg["data"]["frame_stride"],
        window_stride=cfg["data"]["window_stride"],
        max_residues=cfg["data"]["max_residues"],
    )

    # Extract per-frame latents
    print("\nExtracting per-frame latents...")
    t0 = time.time()
    traj_latents = extract_trajectory_latents(model, val_records, data_cfg, device, max_trajs=50)
    print(f"  {len(traj_latents)} trajectories, {time.time()-t0:.1f}s")
    for tid, L in list(traj_latents.items())[:3]:
        print(f"    {tid}: {L.shape[0]} frames")

    # -----------------------------------------------------------------------
    #  VAMP-2
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("VAMP-2 Scores")
    print(f"{'='*60}")

    vamp_results = compute_vamp2_scores(traj_latents, lags=[1, 2, 5, 10, 20])
    for lag_key, vr in vamp_results.items():
        if vr.get("mean") is not None:
            print(f"  {lag_key}:  VAMP-2 = {vr['mean']:.4f} ± {vr['std']:.4f}  "
                  f"(n_trajs={vr['n_trajs']})")
        else:
            print(f"  {lag_key}:  no data")

    # -----------------------------------------------------------------------
    #  Implied timescales
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Implied Timescales (via deeptime TICA)")
    print(f"{'='*60}")

    ts_results = compute_implied_timescales(traj_latents, lags=[2, 5, 10, 20])
    for lag_key, tr in ts_results.items():
        if tr.get("n_trajs", 0) > 0 and "mean_timescales_ps" in tr:
            mt = tr["mean_timescales_ps"]
            print(f"  {lag_key}:  ts1={mt[0]:.0f}ps, ts2={mt[1]:.0f}ps, "
                  f"ts3={mt[2]:.0f}ps, n_trajs={tr['n_trajs']}")

    # -----------------------------------------------------------------------
    #  TICA projection
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("TICA Projection")
    print(f"{'='*60}")
    tica_result = compute_tica_projection(traj_latents, lag=5, out_dir=EVAL_DIR)

    # -----------------------------------------------------------------------
    #  Save
    # -----------------------------------------------------------------------
    results = {
        "vamp2": vamp_results,
        "implied_timescales": ts_results,
        "tica_projection": tica_result,
        "n_trajs": len(traj_latents),
        "latent_dim": model_cfg.latent_dim,
    }
    (EVAL_DIR / "dynamics_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {EVAL_DIR / 'dynamics_summary.json'}")
    print("Done!")


if __name__ == "__main__":
    main()