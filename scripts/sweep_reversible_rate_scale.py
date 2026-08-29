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


def _prediction(outputs: dict, horizons: torch.Tensor, scale: float, base_step: int) -> torch.Tensor:
    steps = horizons.to(dtype=outputs["v4c_generator"].dtype) / float(base_step)
    transition = torch.matrix_exp(
        outputs["v4c_generator"] * steps.view(-1, 1, 1) * float(scale)
    )
    return torch.bmm(outputs["v4c_current_assignment"].unsqueeze(1), transition).squeeze(1)


def _history_permutation(batch: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(batch_size, device=device)
    group_index = batch.get("context_group_index")
    if group_index is None or group_index.unique().numel() <= 1:
        return torch.roll(rows, shifts=1)
    groups = group_index.unique(sorted=True)
    source_group = {
        int(group): int(groups[(position - 1) % len(groups)])
        for position, group in enumerate(groups)
    }
    history_rows = []
    for row in rows:
        group = int(group_index[row])
        horizon = batch["horizon_frames"][row]
        candidates = torch.nonzero(
            (group_index == source_group[group]) & (batch["horizon_frames"] == horizon),
            as_tuple=False,
        ).flatten()
        history_rows.append(candidates[0])
    return torch.stack(history_rows)


def _horizon_permutation(batch: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(batch_size, device=device)
    group_index = batch.get("context_group_index")
    if group_index is None:
        return torch.roll(rows, shifts=1)
    permutation = rows.clone()
    for group in group_index.unique(sorted=True):
        members = torch.nonzero(group_index == group, as_tuple=False).flatten()
        permutation[members] = torch.roll(members, shifts=1)
    return permutation


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    scales: tuple[float, ...],
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
        load_forces=False,
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
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    artifact_config = json.loads((checkpoint_path.parent / "config.json").read_text())
    model = ProteinJEPA(model_config_from_dict(artifact_config["model"], state)).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    sums = {
        scale: defaultdict(lambda: defaultdict(float))
        for scale in scales
    }
    base_step = int(config.model.closed_local_base_step_frames)
    fixed_horizons = tuple(int(value) for value in config.model.closed_local_horizons)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model.forward_paired(batch)
            rows = int(outputs["horizon_frames"].shape[0])
            history_permutation = _history_permutation(batch, rows, device)
            horizon_permutation = _horizon_permutation(batch, rows, device)
            shuffled_batch = dict(batch)
            for key, value in batch.items():
                if not (key.startswith("context_") or key in {
                    "temperature_k", "dataset_index", "dataset", "protein_id", "trajectory_id"
                }):
                    continue
                if torch.is_tensor(value) and value.shape[0] == rows:
                    shuffled_batch[key] = value[history_permutation]
                elif isinstance(value, list) and len(value) == rows:
                    indices = history_permutation.detach().cpu().tolist()
                    shuffled_batch[key] = [value[index] for index in indices]
            shuffled_outputs = model.forward_paired(shuffled_batch)
            target = outputs["v4c_target_assignment"]
            current = outputs["v4c_current_assignment"]
            shuffled_horizons = outputs["horizon_frames"][horizon_permutation]
            for scale in scales:
                prediction = _prediction(outputs, outputs["horizon_frames"], scale, base_step)
                history_prediction = _prediction(
                    shuffled_outputs, outputs["horizon_frames"], scale, base_step
                )
                horizon_prediction = _prediction(outputs, shuffled_horizons, scale, base_step)
                fixed_predictions = {
                    horizon: _prediction(
                        outputs,
                        torch.full_like(outputs["horizon_frames"], horizon),
                        scale,
                        base_step,
                    )
                    for horizon in fixed_horizons
                }
                base_transition = torch.matrix_exp(outputs["v4c_generator"] * float(scale))
                eigenvalues = torch.linalg.eigvals(base_transition).abs().sort(
                    dim=-1, descending=True
                ).values
                for horizon in outputs["horizon_frames"].unique(sorted=True):
                    mask = outputs["horizon_frames"] == horizon
                    item = sums[scale][int(horizon)]
                    prediction_labels = prediction[mask].argmax(dim=-1)
                    target_labels = target[mask].argmax(dim=-1)
                    current_labels = current[mask].argmax(dim=-1)
                    top2_labels = prediction[mask].topk(
                        k=min(2, prediction.shape[-1]), dim=-1
                    ).indices
                    target_changed = target_labels != current_labels
                    predicted_changed = prediction_labels != current_labels
                    item["values"] += float(target[mask].numel())
                    item["rows"] += float(mask.sum())
                    item["prediction_sse"] += float((prediction[mask] - target[mask]).square().sum())
                    item["persistence_sse"] += float((current[mask] - target[mask]).square().sum())
                    item["history_sse"] += float((history_prediction[mask] - target[mask]).square().sum())
                    item["horizon_sse"] += float((horizon_prediction[mask] - target[mask]).square().sum())
                    item["lambda2"] += float(eigenvalues[mask, 1].sum())
                    item["top1_correct"] += float((prediction_labels == target_labels).sum())
                    item["top2_correct"] += float(
                        (top2_labels == target_labels.unsqueeze(-1)).any(dim=-1).sum()
                    )
                    item["persistence_correct"] += float((current_labels == target_labels).sum())
                    item["changed_rows"] += float(target_changed.sum())
                    item["changed_correct"] += float(
                        ((prediction_labels == target_labels) & target_changed).sum()
                    )
                    item["unchanged_rows"] += float((~target_changed).sum())
                    item["unchanged_correct"] += float(
                        ((prediction_labels == target_labels) & ~target_changed).sum()
                    )
                    item["transition_tp"] += float((predicted_changed & target_changed).sum())
                    item["transition_fp"] += float((predicted_changed & ~target_changed).sum())
                    item["transition_fn"] += float((~predicted_changed & target_changed).sum())
                    for target_state, predicted_state in zip(
                        target_labels.detach().cpu().tolist(),
                        prediction_labels.detach().cpu().tolist(),
                    ):
                        item[f"confusion_{target_state}_{predicted_state}"] += 1.0
                    for fixed_horizon, fixed_prediction in fixed_predictions.items():
                        item[f"fixed_{fixed_horizon}"] += float(
                            (fixed_prediction[mask] - target[mask]).square().sum()
                        )
    result = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "checkpoint_step": checkpoint.get("step", "?"),
        "max_batches": max_batches,
        "batch_size": batch_size,
        "seed": seed,
        "scales": {},
    }
    for scale in scales:
        horizons = {}
        for horizon, item in sorted(sums[scale].items()):
            values = max(item["values"], 1.0)
            prediction_mse = item["prediction_sse"] / values
            persistence_mse = item["persistence_sse"] / values
            history_mse = item["history_sse"] / values
            horizon_mse = item["horizon_sse"] / values
            fixed = {
                str(fixed_horizon): item[f"fixed_{fixed_horizon}"] / values
                for fixed_horizon in fixed_horizons
            }
            _, best_fixed_mse = min(fixed.items(), key=lambda pair: pair[1])
            class_recalls = []
            num_states = int(round(item["values"] / max(item["rows"], 1.0)))
            for target_state in range(num_states):
                support = sum(
                    item[f"confusion_{target_state}_{predicted_state}"]
                    for predicted_state in range(num_states)
                )
                if support > 0:
                    class_recalls.append(
                        item[f"confusion_{target_state}_{target_state}"] / support
                    )
            transition_precision = item["transition_tp"] / max(
                item["transition_tp"] + item["transition_fp"], 1.0
            )
            transition_recall = item["transition_tp"] / max(
                item["transition_tp"] + item["transition_fn"], 1.0
            )
            horizons[str(horizon)] = {
                "brier_mse": prediction_mse,
                "persistence_brier_mse": persistence_mse,
                "persistence_relative_skill": 1.0 - prediction_mse / max(persistence_mse, 1.0e-12),
                "history_advantage": 1.0 - prediction_mse / max(history_mse, 1.0e-12),
                "horizon_advantage": 1.0 - prediction_mse / max(horizon_mse, 1.0e-12),
                "correct_vs_best_fixed_skill": 1.0 - prediction_mse / max(best_fixed_mse, 1.0e-12),
                "mean_lambda2": item["lambda2"] / max(item["rows"], 1.0),
                "top1_accuracy": item["top1_correct"] / max(item["rows"], 1.0),
                "top2_accuracy": item["top2_correct"] / max(item["rows"], 1.0),
                "balanced_accuracy": sum(class_recalls) / max(len(class_recalls), 1),
                "persistence_accuracy": item["persistence_correct"] / max(item["rows"], 1.0),
                "changed_fraction": item["changed_rows"] / max(item["rows"], 1.0),
                "changed_state_accuracy": item["changed_correct"] / max(item["changed_rows"], 1.0),
                "unchanged_state_accuracy": item["unchanged_correct"] / max(item["unchanged_rows"], 1.0),
                "transition_precision": transition_precision,
                "transition_recall": transition_recall,
                "fixed_horizon_brier_mse": fixed,
            }
        aggregate_keys = (
            "persistence_relative_skill", "history_advantage", "horizon_advantage",
            "correct_vs_best_fixed_skill", "mean_lambda2",
            "top1_accuracy", "top2_accuracy", "balanced_accuracy",
            "persistence_accuracy", "changed_fraction", "changed_state_accuracy",
            "unchanged_state_accuracy", "transition_precision", "transition_recall",
        )
        aggregate = {
            key: sum(values[key] for values in horizons.values()) / max(len(horizons), 1)
            for key in aggregate_keys
        }
        result["scales"][str(scale)] = {"aggregate": aggregate, "horizons": horizons}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(
        args.config, args.checkpoint, args.manifest, args.output,
        tuple(args.scales), args.max_batches, args.batch_size, args.seed, args.device,
    )
    print(json.dumps({scale: value["aggregate"] for scale, value in result["scales"].items()}, indent=2))


if __name__ == "__main__":
    main()
