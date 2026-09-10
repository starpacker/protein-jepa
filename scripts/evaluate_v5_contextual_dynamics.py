#!/usr/bin/env python3
"""Evaluate V5 contextual dynamics with history and horizon controls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _shuffle_history_keep_endpoint(batch: dict) -> dict:
    """Reverse observed history while preserving its endpoint and protein."""
    shuffled = dict(batch)
    time_length = batch["context_coords"].shape[1]
    permutation = torch.arange(time_length, device=batch["context_coords"].device)
    permutation[:-1] = torch.flip(permutation[:-1], dims=(0,))
    for key in (
        "context_coords", "context_atom_mask", "context_residue_mask", "context_forces"
    ):
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[1] == time_length:
            shuffled[key] = value[:, permutation]
    # Keep time_ps ordered: frames are moved to the original observation times.
    return shuffled


def _shuffle_horizons_within_context(batch: dict) -> dict:
    shuffled = dict(batch)
    rows = torch.arange(batch["horizon_frames"].shape[0], device=batch["horizon_frames"].device)
    group_index = batch.get("context_group_index")
    if group_index is None:
        permutation = torch.roll(rows, shifts=1)
    else:
        permutation = rows.clone()
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            permutation[members] = torch.roll(members, shifts=1)
    shuffled["horizon_frames"] = batch["horizon_frames"][permutation]
    shuffled["horizon_time_ps"] = batch["horizon_time_ps"][permutation]
    return shuffled


def _decoded_current(outputs: dict) -> torch.Tensor:
    return torch.bmm(
        outputs["v5_current_assignment"].unsqueeze(1), outputs["v5_context_slots"]
    ).squeeze(1)


def _effective_slots(gates: torch.Tensor) -> torch.Tensor:
    probabilities = gates / gates.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
    entropy = -(probabilities * probabilities.clamp(min=1.0e-8).log()).sum(dim=-1)
    return entropy.exp()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    data_config = replace(
        config.data,
        manifest_path=str(args.manifest),
        batch_size=args.batch_size,
        num_workers=0,
        training_mode="paired_window",
        paired_all_horizons=True,
        sampling_mode="hierarchical",
        load_forces=False,
    )
    dataset = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=args.max_batches * args.batch_size,
        seed=args.seed,
        dataset_weights=data_config.dataset_weights,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    checkpoint_config = checkpoint.get("config", {})
    model_values = checkpoint_config.get("model", config.model.__dict__)
    model = ProteinJEPA(model_config_from_dict(model_values, state)).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.v5_contextual_dynamics is None:
        raise ValueError("checkpoint does not contain V5 contextual dynamics")

    totals: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch = _move_batch(batch, device)
        outputs = model.forward_paired(batch)
        history_outputs = model.forward_paired(_shuffle_history_keep_endpoint(batch))
        horizon_outputs = model.forward_paired(_shuffle_horizons_within_context(batch))
        target = outputs["v5_target_feature"]
        prediction = outputs["v5_predicted_feature"]
        persistence = _decoded_current(outputs)
        history_prediction = history_outputs["v5_predicted_feature"]
        horizon_prediction = horizon_outputs["v5_predicted_feature"]
        observable_target = outputs["v5_observable_target"]
        observable_prediction = outputs["v5_observable_prediction"]
        history_observable = history_outputs["v5_observable_prediction"]
        horizon_observable = horizon_outputs["v5_observable_prediction"]
        observable_persistence = torch.zeros_like(observable_target)
        observable_persistence[:, 0] = 1.0
        effective = _effective_slots(outputs["v5_slot_gates"])
        fixed = outputs.get("v5_fixed_predictions")
        fixed_horizons = outputs.get("v5_fixed_horizons")
        fixed_features = None
        if fixed is not None:
            fixed_features = torch.einsum("bfk,bkd->bfd", fixed, outputs["v5_context_slots"])

        for horizon in outputs["horizon_frames"].unique(sorted=True):
            mask = outputs["horizon_frames"] == horizon
            item = totals[int(horizon)]
            item["values"] += float(target[mask].numel())
            item["rows"] += float(mask.sum())
            item["prediction_sse"] += float((prediction[mask] - target[mask]).square().sum())
            item["persistence_sse"] += float((persistence[mask] - target[mask]).square().sum())
            item["history_sse"] += float((history_prediction[mask] - target[mask]).square().sum())
            item["horizon_sse"] += float((horizon_prediction[mask] - target[mask]).square().sum())
            item["observable_values"] += float(observable_target[mask].numel())
            item["observable_prediction_sse"] += float(
                (observable_prediction[mask] - observable_target[mask]).square().sum()
            )
            item["observable_persistence_sse"] += float(
                (observable_persistence[mask] - observable_target[mask]).square().sum()
            )
            item["observable_history_sse"] += float(
                (history_observable[mask] - observable_target[mask]).square().sum()
            )
            item["observable_horizon_sse"] += float(
                (horizon_observable[mask] - observable_target[mask]).square().sum()
            )
            item["active_slots"] += float(outputs["v5_slot_gates"][mask].sum())
            item["effective_slots"] += float(effective[mask].sum())
            item["detailed_balance"] += float(
                outputs["v5_detailed_balance_residual"][mask].abs().sum()
            )
            item["balance_values"] += float(
                outputs["v5_detailed_balance_residual"][mask].numel()
            )
            if fixed_features is not None and fixed_horizons is not None:
                for index, fixed_horizon in enumerate(fixed_horizons.tolist()):
                    item[f"fixed_{fixed_horizon}"] += float(
                        (fixed_features[mask, index] - target[mask]).square().sum()
                    )

    horizons = {}
    control_horizons = tuple(int(value) for value in config.model.v5_control_horizons_frames)
    for horizon, item in sorted(totals.items()):
        values = max(item["values"], 1.0)
        rows = max(item["rows"], 1.0)
        prediction_mse = item["prediction_sse"] / values
        persistence_mse = item["persistence_sse"] / values
        history_mse = item["history_sse"] / values
        horizon_mse = item["horizon_sse"] / values
        fixed_mse = {
            str(fixed_horizon): item[f"fixed_{fixed_horizon}"] / values
            for fixed_horizon in control_horizons
            if f"fixed_{fixed_horizon}" in item
        }
        wrong_fixed_mse = {
            fixed_horizon: mse
            for fixed_horizon, mse in fixed_mse.items()
            if int(fixed_horizon) != horizon
        }
        best_wrong_fixed = (
            min(wrong_fixed_mse.values()) if wrong_fixed_mse else float("nan")
        )
        observable_values = max(item["observable_values"], 1.0)
        observable_mse = item["observable_prediction_sse"] / observable_values
        observable_persistence_mse = item["observable_persistence_sse"] / observable_values
        observable_history_mse = item["observable_history_sse"] / observable_values
        observable_horizon_mse = item["observable_horizon_sse"] / observable_values
        horizons[str(horizon)] = {
            "feature_mse": prediction_mse,
            "persistence_feature_mse": persistence_mse,
            "persistence_relative_skill": 1.0 - prediction_mse / max(persistence_mse, 1.0e-12),
            "history_order_advantage": 1.0 - prediction_mse / max(history_mse, 1.0e-12),
            "horizon_advantage": 1.0 - prediction_mse / max(horizon_mse, 1.0e-12),
            "correct_vs_best_wrong_fixed_skill": (
                1.0 - prediction_mse / max(best_wrong_fixed, 1.0e-12)
                if wrong_fixed_mse
                else None
            ),
            "physical_observable_mse": observable_mse,
            "physical_observable_zero_motion_mse": observable_persistence_mse,
            "physical_observable_zero_motion_skill": 1.0
            - observable_mse / max(observable_persistence_mse, 1.0e-12),
            "physical_history_order_advantage": 1.0
            - observable_mse / max(observable_history_mse, 1.0e-12),
            "physical_horizon_advantage": 1.0
            - observable_mse / max(observable_horizon_mse, 1.0e-12),
            "mean_active_slots": item["active_slots"] / rows,
            "mean_effective_slots": item["effective_slots"] / rows,
            "detailed_balance_residual": item["detailed_balance"]
            / max(item["balance_values"], 1.0),
            "fixed_horizon_feature_mse": fixed_mse,
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", "?"),
        "manifest": str(args.manifest),
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "history_control": "reverse all context frames except the endpoint within each sample",
        "score_space": "shared normalized V5 feature space; slot indices are not compared across contexts",
        "horizons": horizons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
