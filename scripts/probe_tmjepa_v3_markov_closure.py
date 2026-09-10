#!/usr/bin/env python3
"""Test whether raw history predicts innovations beyond the resolved V3 state."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_tmjepa_v3_state_dict


def _ridge(features: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    design = torch.cat([features, torch.ones(features.shape[0], 1)], dim=-1)
    regularizer = (
        torch.eye(design.shape[1], dtype=design.dtype)
        * ridge
        * features.shape[0]
    )
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(
        design.T @ design + regularizer, design.T @ target
    )


def _predict(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.cat([features, torch.ones(features.shape[0], 1)], dim=-1) @ weights


@torch.no_grad()
def _collect(
    model: ProteinJEPA,
    config,
    manifest: str,
    max_batches: int,
    batch_size: int,
    history_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_config = replace(
        config.data,
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=0,
        paired_all_horizons=False,
        paired_horizons_frames=(1,),
    )
    loader = DataLoader(
        build_dataset(data_config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_paired_windows,
    )
    history_features = []
    targets = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        values = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model.forward_tmjepa(values)
        q = outputs["tmjepa_context_cv_sequence"]
        innovation = outputs["tmjepa_innovation"]
        innovation_mask = outputs["tmjepa_innovation_mask"]
        for step in range(history_steps, innovation.shape[1]):
            valid = innovation_mask[:, step - history_steps : step + 1].all(dim=-1)
            if not bool(valid.any()):
                continue
            increments = [
                q[:, step - offset + 1] - q[:, step - offset]
                for offset in range(1, history_steps + 1)
            ]
            history_features.append(
                torch.cat(increments, dim=-1)[valid].cpu().double()
            )
            targets.append(innovation[:, step][valid].cpu().double())
    return (
        torch.cat(history_features),
        torch.cat(targets),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    model = ProteinJEPA(
        model_config_from_dict(checkpoint["config"]["model"], state)
    ).to(device)
    model.load_state_dict(upgrade_tmjepa_v3_state_dict(model, state), strict=True)
    model.eval()
    train_history, train_target = _collect(
        model,
        config,
        args.train_manifest,
        args.max_train_batches,
        args.batch_size,
        args.history_steps,
        device,
    )
    eval_history, eval_target = _collect(
        model,
        config,
        args.eval_manifest,
        args.max_eval_batches,
        args.batch_size,
        args.history_steps,
        device,
    )
    history_mean = train_history.mean(dim=0)
    history_scale = train_history.std(dim=0).clamp(min=1.0e-6)
    train_history = (train_history - history_mean) / history_scale
    eval_history = (eval_history - history_mean) / history_scale
    history_weights = _ridge(train_history, train_target, args.ridge)
    history_mse = (_predict(eval_history, history_weights) - eval_target).square().mean()
    zero_mse = eval_target.square().mean()
    history_skill = 1.0 - history_mse / zero_mse.clamp(min=1.0e-12)
    stable_probe = torch.isfinite(history_mse) & (history_mse <= 10.0 * zero_mse)
    result = {
        "target": "standardized_one_step_innovation",
        "train_samples": int(train_target.shape[0]),
        "eval_samples": int(eval_target.shape[0]),
        "zero_predictor_mse": float(zero_mse),
        "state_mse": float(zero_mse),
        "state_plus_history_mse": float(history_mse),
        "residual_history_skill": float(history_skill),
        "stable_probe": bool(stable_probe),
        "markov_closed": bool(stable_probe and history_skill < 0.01),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
