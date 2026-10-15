from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import compute_v3_closed_local_state


CHANNEL_NAMES = (
    "contact_density",
    "torsion_sin_phi",
    "torsion_sin_psi",
    "torsion_sin_omega",
    "torsion_cos_phi",
    "torsion_cos_psi",
    "torsion_cos_omega",
    "radius_of_gyration",
    "local_distance_mean",
    "local_distance_std",
    "segment_center_distance",
    "contact_order",
)


class RunningChannelStats:
    def __init__(self, channels: int):
        self.count = 0
        self.sum = torch.zeros(channels, dtype=torch.float64)
        self.sum_sq = torch.zeros(channels, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if values.numel() == 0:
            return
        if values.dim() != 2:
            raise ValueError("channel statistics require [N,C] values")
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.sum_sq += values.square().sum(dim=0)

    def result(self, minimum_scale: float) -> dict:
        if self.count < 2:
            raise ValueError("at least two valid local-state deltas are required")
        mean = self.sum / self.count
        variance = (self.sum_sq - self.count * mean.square()) / (self.count - 1)
        std = variance.clamp(min=0.0).sqrt()
        scale = std.clamp(min=minimum_scale)
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "scale": scale.tolist(),
        }


@torch.no_grad()
def estimate(
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
    max_batches: int,
    batch_size: int,
    seed: int,
    minimum_scale: float,
) -> dict:
    config = load_config(config_path)
    horizons = tuple(config.data.paired_horizons_frames)
    data_config = replace(
        config.data,
        manifest_path=str(manifest_path),
        batch_size=batch_size,
        num_workers=0,
        training_mode="paired_window",
        sampling_mode="hierarchical",
        paired_all_horizons=True,
    )
    dataset = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=max_batches * batch_size,
        seed=seed,
        dataset_weights=data_config.dataset_weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    channels = config.model.closed_local_state_dim
    global_stats = RunningChannelStats(channels)
    horizon_stats = defaultdict(lambda: RunningChannelStats(channels))

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        context_state, context_mask = compute_v3_closed_local_state(
            batch["context_coords"],
            batch["context_atom_mask"],
            batch["context_residue_mask"],
            batch["context_chain_id"],
            batch["context_residue_index"],
            num_tokens=config.model.local_dynamics_tokens,
        )
        future_state, future_mask = compute_v3_closed_local_state(
            batch["future_coords"],
            batch["future_atom_mask"],
            batch["future_residue_mask"],
            batch["future_chain_id"],
            batch["future_residue_index"],
            num_tokens=config.model.local_dynamics_tokens,
        )
        valid = context_mask & future_mask
        delta = future_state - context_state
        global_stats.update(delta[valid])
        for horizon in horizons:
            rows = batch["horizon_frames"] == horizon
            horizon_stats[int(horizon)].update(delta[rows][valid[rows]])

    global_result = global_stats.result(minimum_scale)
    result = {
        "manifest": str(manifest_path),
        "config": str(config_path),
        "seed": seed,
        "max_batches": max_batches,
        "context_batch_size": batch_size,
        "horizons_frames": list(horizons),
        "channel_names": list(CHANNEL_NAMES),
        "minimum_scale": minimum_scale,
        "global": global_result,
        "by_horizon": {
            str(horizon): horizon_stats[horizon].result(minimum_scale)
            for horizon in sorted(horizon_stats)
        },
    }
    if channels != len(CHANNEL_NAMES):
        raise ValueError(f"expected {len(CHANNEL_NAMES)} closed-state channels, got {channels}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate train-only V3 closed-state delta scales")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--minimum-scale", type=float, default=1.0e-3)
    args = parser.parse_args()
    print(json.dumps(estimate(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        minimum_scale=args.minimum_scale,
    ), indent=2))


if __name__ == "__main__":
    main()
