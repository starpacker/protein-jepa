#!/usr/bin/env python3
"""Model-free audit of whether fixed CV dynamics need finite history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.thermodynamic import FixedBackboneCollectiveVariables


def _ridge_fit(features: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
    design = torch.cat([features, ones], dim=-1)
    gram = design.T @ design
    regularizer = torch.eye(gram.shape[0], dtype=gram.dtype) * ridge
    regularizer[-1, -1] = 0.0
    return torch.linalg.solve(gram + regularizer, design.T @ targets)


def _predict(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
    return torch.cat([features, ones], dim=-1) @ weights


@torch.no_grad()
def _collect(
    manifest: str,
    *,
    num_frames: int,
    max_batches: int,
    batch_size: int,
    history_steps: int,
    max_residues: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config = DataConfig(
        manifest_path=manifest,
        batch_size=batch_size,
        num_workers=0,
        num_frames=num_frames,
        frame_stride=1,
        window_stride=max(1, num_frames // 2),
        max_residues=max_residues,
        training_mode="clip",
    )
    loader = DataLoader(
        build_dataset(config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clips,
    )
    extractor = FixedBackboneCollectiveVariables(num_segments=3).to(device)
    q_only = []
    history = []
    target = []
    groups = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        values = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        q = extractor(
            values["coords"],
            values["atom_mask"],
            values["residue_mask"],
            values["chain_id"],
            values["residue_index"],
        ).to(device="cpu", dtype=torch.float64)
        for step in range(history_steps, q.shape[1] - 1):
            current = q[:, step]
            increments = [q[:, step - offset + 1] - q[:, step - offset] for offset in range(1, history_steps + 1)]
            q_only.append(current)
            history.append(torch.cat([current] + increments, dim=-1))
            target.append(q[:, step + 1] - current)
            groups.extend(batch["trajectory_id"])
    if not q_only:
        raise ValueError("audit found no valid transitions")
    return torch.cat(q_only), torch.cat(history), torch.cat(target), torch.tensor(
        [hash(value) % (2**31) for value in groups], dtype=torch.long
    )


def _innovation_lag1(residual: torch.Tensor, groups: torch.Tensor) -> float:
    products = []
    for group in groups.unique():
        values = residual[groups == group]
        if values.shape[0] > 1:
            products.append(values[1:] * values[:-1])
    if not products:
        return float("nan")
    product = torch.cat(products).mean(dim=0)
    variance = residual.var(dim=0, unbiased=False).clamp(min=1.0e-12)
    return float((product / variance).square().mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--max-residues", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    train_q, train_history, train_target, _ = _collect(
        args.train_manifest,
        num_frames=args.num_frames,
        max_batches=args.max_train_batches,
        batch_size=args.batch_size,
        history_steps=args.history_steps,
        max_residues=args.max_residues,
        device=device,
    )
    eval_q, eval_history, eval_target, eval_groups = _collect(
        args.eval_manifest,
        num_frames=args.num_frames,
        max_batches=args.max_eval_batches,
        batch_size=args.batch_size,
        history_steps=args.history_steps,
        max_residues=args.max_residues,
        device=device,
    )
    q_weights = _ridge_fit(train_q, train_target, args.ridge)
    history_weights = _ridge_fit(train_history, train_target, args.ridge)
    q_residual = eval_target - _predict(eval_q, q_weights)
    history_residual = eval_target - _predict(eval_history, history_weights)
    q_mse_by_dim = q_residual.square().mean(dim=0).clamp(min=1.0e-12)
    history_mse_by_dim = history_residual.square().mean(dim=0).clamp(min=1.0e-12)
    q_mse = float(q_mse_by_dim.mean())
    history_mse = float(history_mse_by_dim.mean())
    gaussian_cmi_nats = float(0.5 * torch.log(q_mse_by_dim / history_mse_by_dim).sum())
    result = {
        "train_manifest": args.train_manifest,
        "eval_manifest": args.eval_manifest,
        "cv_dim": int(eval_target.shape[1]),
        "train_transitions": int(train_target.shape[0]),
        "eval_transitions": int(eval_target.shape[0]),
        "q_only_mse": q_mse,
        "history_mse": history_mse,
        "history_skill": 1.0 - history_mse / q_mse,
        "gaussian_conditional_mutual_information_nats": gaussian_cmi_nats,
        "q_only_innovation_lag1_rms": _innovation_lag1(q_residual, eval_groups),
        "history_innovation_lag1_rms": _innovation_lag1(history_residual, eval_groups),
        "memory_supported": history_mse < q_mse and gaussian_cmi_nats > 0.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
