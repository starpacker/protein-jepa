"""Held-out evaluation for Stage A: report per-horizon skill vs persistence.

Usage:
    python scripts/eval_stage_a_heldout.py \
        --checkpoint artifacts/mdcath_stage_a_nonoverlap/checkpoint_latest.pt \
        --config configs/mdcath_stage_a_nonoverlap.yaml \
        --split val \
        --output artifacts/eval_stage_a_heldout.json
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config, DataConfig
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.manifest import load_manifest
from protein_jepa_frame.models.stage_a import StageAJEPA


def _move_batch(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        elif isinstance(v, list):
            moved[k] = v
        else:
            moved[k] = v
    return moved


def evaluate_heldout(checkpoint_path: str, config_path: str, split: str,
                     max_samples: int = 500, device: str = "cuda",
                     output: str | None = None):
    cfg = load_config(config_path)
    
    # Load manifest for the requested split
    manifest_path = str(Path(cfg.data.manifest_path).parent / f"{split}.jsonl")
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    
    data_cfg = DataConfig(
        manifest_path=manifest_path,
        num_frames=cfg.data.num_frames,
        frame_stride=cfg.data.frame_stride,
        window_stride=cfg.data.window_stride,
        max_residues=cfg.data.max_residues,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        coordinate_scale=cfg.data.coordinate_scale,
        training_mode="paired_window",
        paired_horizons_frames=cfg.data.paired_horizons_frames,
        paired_all_horizons=cfg.data.paired_all_horizons,
    )
    
    # Build held-out dataset
    records = load_manifest(manifest_path)
    ds = PairedTrajectoryDataset(records, data_cfg)
    print(f"Split {split}: {len(ds)} samples, {len(records)} trajectories")
    
    loader = DataLoader(
        ds, batch_size=min(16, cfg.data.batch_size),
        shuffle=False, num_workers=cfg.data.num_workers,
        collate_fn=collate_paired_windows, drop_last=False,
    )
    
    # Build model
    dev = torch.device(device)
    model = StageAJEPA(cfg.model).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # Accumulate per-horizon metrics
    horizon_skill: dict[int, list[float]] = defaultdict(list)
    horizon_loss: dict[int, list[float]] = defaultdict(list)
    horizon_persistence: dict[int, list[float]] = defaultdict(list)
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_samples and i * loader.batch_size >= max_samples:
                break
            batch = _move_batch(batch, dev)
            
            try:
                outputs = model.forward_paired(batch)
            except Exception as e:
                print(f"  Skipping batch {i}: {e}", file=sys.stderr)
                continue
            
            pred = outputs["delta_prediction"] + outputs["context_endpoint"]
            target = outputs["future_endpoint"]
            context = outputs["context_endpoint"]
            
            # Per-sample horizon_time_ps
            if "horizon_time_ps" in batch:
                horizon_ps = batch["horizon_time_ps"].cpu().numpy()
            else:
                horizon_ps = np.full(pred.shape[0], np.nan)
            
            # Compute loss and persistence per sample
            for j in range(pred.shape[0]):
                pred_j = pred[j]
                target_j = target[j]
                ctx_j = context[j]
                
                # Model forecast error
                mse_model = F.mse_loss(pred_j, target_j).item()
                # Persistence error (just use context endpoint)
                mse_pers = F.mse_loss(ctx_j, target_j).item()
                
                skill = 1.0 - mse_model / (mse_pers + 1e-8)
                
                h = int(round(horizon_ps[j] / 1000.0)) if not np.isnan(horizon_ps[j]) else -1
                horizon_skill[h].append(skill)
                horizon_loss[h].append(mse_model)
                horizon_persistence[h].append(mse_pers)
            
            if i % 10 == 0:
                sys.stdout.write(f"\r  Batch {i}/{min(len(loader), max_samples//loader.batch_size)}")
                sys.stdout.flush()
    
    print()
    
    # Aggregate per horizon
    results = {"checkpoint": checkpoint_path, "split": split, "config": config_path}
    horizon_tables = []
    for h in sorted([k for k in horizon_skill if k > 0]):
        skills = np.array(horizon_skill[h])
        losses = np.array(horizon_loss[h])
        pers = np.array(horizon_persistence[h])
        row = {
            "horizon_ns": h,
            "n_samples": int(len(skills)),
            "mean_skill": float(np.mean(skills)),
            "median_skill": float(np.median(skills)),
            "skill_gt_zero_frac": float((skills > 0).mean()),
            "mean_model_mse": float(np.mean(losses)),
            "mean_pers_mse": float(np.mean(pers)),
        }
        horizon_tables.append(row)
        results[f"horizon_{h}ns"] = row
    
    results["horizon_summary"] = horizon_tables
    
    # Overall paired skill
    all_skills = np.concatenate([np.array(v) for v in horizon_skill.values() if len(v) > 0])
    results["overall"] = {
        "n_total": int(len(all_skills)),
        "mean_skill": float(np.mean(all_skills)),
        "median_skill": float(np.median(all_skills)),
        "skill_gt_zero_frac": float((all_skills > 0).mean()),
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Stage A Held-out Evaluation ({split} split)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    print(f"{'Horizon (ns)':>14s}  {'N':>6s}  {'Mean Skill':>12s}  {'>0 Frac':>10s}  {'Model MSE':>12s}  {'Pers MSE':>12s}")
    print("-" * 70)
    for row in horizon_tables:
        print(f"{row['horizon_ns']:>14d}  {row['n_samples']:>6d}  {row['mean_skill']:>12.4f}  {row['skill_gt_zero_frac']:>10.3f}  {row['mean_model_mse']:>12.6f}  {row['mean_pers_mse']:>12.6f}")
    print(f"{'='*70}")
    print(f"Overall: mean skill = {results['overall']['mean_skill']:.4f}, "
          f">0 fraction = {results['overall']['skill_gt_zero_frac']:.3f} "
          f"({results['overall']['n_total']} samples)")
    
    # Save
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(output, "w"), indent=2)
        print(f"\nSaved to {output}")
    
    # Gate exit condition: all horizons must have mean_skill > 0
    all_positive = all(row["mean_skill"] > 0 for row in horizon_tables)
    if all_positive:
        print("\n✅ GATE 1 EXIT: Mean skill > 0 on ALL horizons (held-out proteins)")
    else:
        negative_horizons = [row["horizon_ns"] for row in horizon_tables if row["mean_skill"] <= 0]
        print(f"\n❌ GATE 1 NOT SATISFIED: Negative skill on horizons {negative_horizons}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    evaluate_heldout(args.checkpoint, args.config, args.split,
                     max_samples=args.max_samples, device=args.device,
                     output=args.output)
