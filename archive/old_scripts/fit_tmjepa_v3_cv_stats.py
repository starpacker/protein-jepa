#!/usr/bin/env python3
"""Fit a fixed robust affine gauge for V3 collective variables."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.thermodynamic import FixedBackboneCollectiveVariables


@torch.no_grad()
def fit(args: argparse.Namespace) -> dict:
    config = DataConfig(
        manifest_path=args.manifest,
        num_frames=args.num_frames,
        frame_stride=1,
        window_stride=args.num_frames,
        max_residues=args.max_residues,
        training_mode="clip",
    )
    dataset = build_dataset(config)
    rng = random.Random(args.seed)
    entry_indices = list(range(len(dataset.entries)))
    rng.shuffle(entry_indices)
    requested_samples = args.max_batches * args.batch_size
    if requested_samples < len(entry_indices):
        raise ValueError(
            "CV statistics must cover every trajectory entry; increase --max-batches "
            f"to at least {(len(entry_indices) + args.batch_size - 1) // args.batch_size}"
        )
    sampled_indices = []
    while len(sampled_indices) < requested_samples:
        rng.shuffle(entry_indices)
        for entry_index in entry_indices:
            entry = dataset.entries[entry_index]
            previous = dataset.cumulative[entry_index - 1] if entry_index else 0
            sampled_indices.append(previous + rng.randrange(entry.count))
            if len(sampled_indices) == requested_samples:
                break
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampled_indices,
        num_workers=0,
        collate_fn=collate_clips,
    )
    extractor = FixedBackboneCollectiveVariables(
        num_segments=args.num_segments
    ).to(args.device)
    values = []
    proteins = set()
    temperatures = set()
    for index, batch in enumerate(loader):
        if index >= args.max_batches:
            break
        tensors = {
            key: value.to(args.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        proteins.update(batch["protein_id"])
        temperatures.update(float(value) for value in batch["temperature_k"])
        q = extractor(
            tensors["coords"],
            tensors["atom_mask"],
            tensors["residue_mask"],
            tensors["chain_id"],
            tensors["residue_index"],
        )
        frame_mask = tensors["residue_mask"].any(dim=2)
        values.append(q[frame_mask].cpu())
    if not values:
        raise ValueError("no CV values were collected")
    values = torch.cat(values)
    median = values.median(dim=0).values
    lower = torch.quantile(values, 0.25, dim=0)
    upper = torch.quantile(values, 0.75, dim=0)
    robust_scale = ((upper - lower) / 1.3489795).clamp(min=args.minimum_scale)
    return {
        "manifest": args.manifest,
        "samples": values.shape[0],
        "trajectory_clips": len(sampled_indices),
        "proteins": len(proteins),
        "temperatures_k": sorted(temperatures),
        "seed": args.seed,
        "channel_names": list(extractor.channel_names),
        "cv_mean": median.tolist(),
        "cv_scale": robust_scale.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=384)
    parser.add_argument("--minimum-scale", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = fit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
