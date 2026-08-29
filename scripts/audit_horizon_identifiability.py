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


@torch.no_grad()
def audit(
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
    max_batches: int,
    batch_size: int,
    seed: int,
    device_name: str,
) -> dict:
    config = load_config(config_path)
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
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    scale = torch.tensor(config.model.closed_local_state_scale, device=device)
    pair_sum: dict[tuple[int, int], float] = defaultdict(float)
    pair_count: dict[tuple[int, int], int] = defaultdict(int)
    features: list[torch.Tensor] = []
    labels: list[int] = []
    context_ids: list[int] = []

    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        context_state, context_mask = compute_v3_closed_local_state(
            batch["context_coords"], batch["context_atom_mask"], batch["context_residue_mask"],
            batch["context_chain_id"], batch["context_residue_index"],
            num_tokens=config.model.local_dynamics_tokens,
        )
        future_state, future_mask = compute_v3_closed_local_state(
            batch["future_coords"], batch["future_atom_mask"], batch["future_residue_mask"],
            batch["future_chain_id"], batch["future_residue_index"],
            num_tokens=config.model.local_dynamics_tokens,
        )
        normalized_delta = (future_state - context_state) / scale
        valid = context_mask & future_mask
        for group in batch["context_group_index"].unique(sorted=True):
            members = torch.nonzero(batch["context_group_index"] == group, as_tuple=False).flatten()
            horizons = batch["horizon_frames"][members]
            context_id = batch_index * batch_size + int(group)
            for position, member in enumerate(members):
                for token in torch.nonzero(valid[member], as_tuple=False).flatten():
                    features.append(normalized_delta[member, token].cpu())
                    labels.append(int(horizons[position]))
                    context_ids.append(context_id)
                for right_position in range(position + 1, members.numel()):
                    right = members[right_position]
                    common = valid[member] & valid[right]
                    if not common.any():
                        continue
                    key = tuple(sorted((int(horizons[position]), int(horizons[right_position]))))
                    distance = (
                        normalized_delta[right, common] - normalized_delta[member, common]
                    ).square().mean()
                    pair_sum[key] += float(distance)
                    pair_count[key] += 1

    feature_tensor = torch.stack(features)
    label_tensor = torch.tensor(labels)
    context_tensor = torch.tensor(context_ids)
    split_context = max_batches * batch_size // 2
    train = context_tensor < split_context
    test = ~train
    horizons = sorted(set(labels))
    centroids = torch.stack([
        feature_tensor[train & (label_tensor == horizon)].mean(dim=0) for horizon in horizons
    ])
    distances = (feature_tensor[test, None] - centroids[None]).square().mean(dim=-1)
    prediction = torch.tensor(horizons)[distances.argmin(dim=-1)]
    accuracy = float((prediction == label_tensor[test]).float().mean())
    confusion = {
        str(target): {
            str(predicted): int(((label_tensor[test] == target) & (prediction == predicted)).sum())
            for predicted in horizons
        }
        for target in horizons
    }
    result = {
        "config": str(config_path),
        "manifest": str(manifest_path),
        "max_batches": max_batches,
        "context_batch_size": batch_size,
        "seed": seed,
        "horizons_frames": horizons,
        "pairwise_normalized_target_mse": {
            f"{left}-{right}": pair_sum[(left, right)] / pair_count[(left, right)]
            for left, right in sorted(pair_sum)
        },
        "centroid_horizon_accuracy": accuracy,
        "chance_accuracy": 1.0 / len(horizons),
        "train_segments": int(train.sum()),
        "test_segments": int(test.sum()),
        "confusion": confusion,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit closed-state horizon identifiability")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = audit(
        config_path=args.config,
        manifest_path=args.manifest,
        output_path=args.output,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
