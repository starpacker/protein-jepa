#!/usr/bin/env python3
"""Identify Mori rank and relaxation poles from overdamped innovations."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_tmjepa_v3_state_dict


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def _collect(args: argparse.Namespace) -> tuple[torch.Tensor, float]:
    config = load_config(args.config)
    data_config = replace(
        config.data,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=0,
        paired_all_horizons=True,
    )
    data = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        data,
        num_samples=args.max_batches * args.batch_size,
        seed=args.seed,
        dataset_weights=data_config.dataset_weights,
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_paired_windows,
    )
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    model = ProteinJEPA(
        model_config_from_dict(checkpoint["config"]["model"], state)
    ).to(device)
    model.load_state_dict(upgrade_tmjepa_v3_state_dict(model, state), strict=True)
    model.eval()
    sequences = []
    time_step = None
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        outputs = model.forward_tmjepa(values)
        minimum_horizon = outputs["horizon_frames"].min()
        unique_context = outputs["horizon_frames"] == minimum_horizon
        sequences.append(outputs["tmjepa_innovation"][unique_context].cpu())
        observed_dt = (
            values["context_time_ps"][:, 1:] - values["context_time_ps"][:, :-1]
        ).abs()
        time_step = float(observed_dt.median())
    if not sequences or time_step is None:
        raise ValueError("memory identification found no innovations")
    return torch.cat(sequences), time_step


def _fit_positive_prony(
    correlation: torch.Tensor,
    dt_ps: float,
    rank: int,
    maximum_time_ps: float,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    raw_weight = torch.zeros(rank, requires_grad=True)
    initial_times = torch.logspace(
        math.log10(dt_ps), math.log10(maximum_time_ps), rank
    )
    fraction = ((initial_times - dt_ps) / (maximum_time_ps - dt_ps)).clamp(0.01, 0.99)
    raw_time = torch.logit(fraction).requires_grad_(True)
    optimizer = torch.optim.Adam([raw_weight, raw_time], lr=0.03)
    lag = torch.arange(1, correlation.numel() + 1, dtype=correlation.dtype) * dt_ps
    for _ in range(steps):
        optimizer.zero_grad()
        weight = F.softmax(raw_weight, dim=0)
        times = dt_ps + (maximum_time_ps - dt_ps) * torch.sigmoid(raw_time)
        prediction = torch.einsum("r,lr->l", weight, torch.exp(-lag[:, None] / times))
        loss = (prediction - correlation).square().mean()
        loss.backward()
        optimizer.step()
    return weight.detach(), times.detach().sort().values, float(loss.detach())


def identify(args: argparse.Namespace) -> dict:
    innovation, dt_ps = _collect(args)
    innovation = innovation - innovation.mean(dim=(0, 1), keepdim=True)
    covariances = []
    for lag in range(1, args.max_lag + 1):
        left = innovation[:, lag:].reshape(-1, innovation.shape[-1])
        right = innovation[:, :-lag].reshape(-1, innovation.shape[-1])
        covariances.append(left.T @ right / max(left.shape[0] - 1, 1))
    block_rows = min(args.hankel_blocks, len(covariances) // 2)
    blocks = [
        torch.cat(covariances[row : row + block_rows], dim=1)
        for row in range(block_rows)
    ]
    hankel = torch.cat(blocks, dim=0)
    singular_values = torch.linalg.svdvals(hankel)
    cumulative = singular_values.square().cumsum(0) / singular_values.square().sum()
    rank_for_target = int(
        torch.nonzero(cumulative >= args.explained_variance, as_tuple=False)[0]
    ) + 1
    recommended_rank = min(max(rank_for_target, 1), args.maximum_rank)
    strengths = torch.stack([value.square().mean().sqrt() for value in covariances])
    compatibility = []
    for covariance in covariances:
        symmetric = 0.5 * (covariance + covariance.T)
        antisymmetric = 0.5 * (covariance - covariance.T)
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        total_energy = covariance.square().sum().clamp(min=1.0e-12)
        compatibility.append(
            {
                "symmetric_energy_fraction": float(
                    symmetric.square().sum() / total_energy
                ),
                "antisymmetric_energy_fraction": float(
                    antisymmetric.square().sum() / total_energy
                ),
                "negative_symmetric_spectral_fraction": float(
                    eigenvalues.clamp(max=0.0).square().sum()
                    / eigenvalues.square().sum().clamp(min=1.0e-12)
                ),
                "minimum_symmetric_eigenvalue": float(eigenvalues.min()),
                "maximum_symmetric_eigenvalue": float(eigenvalues.max()),
            }
        )
    target = (strengths / strengths[0].clamp(min=1.0e-8)).clamp(min=0.0, max=1.0)
    weight, times, fit_loss = _fit_positive_prony(
        target,
        dt_ps,
        recommended_rank,
        args.maximum_memory_time_ps,
        args.optimization_steps,
    )
    return {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "observation_step_ps": dt_ps,
        "lagged_covariance_rms": strengths.tolist(),
        "hankel_singular_values": singular_values.tolist(),
        "explained_variance": cumulative.tolist(),
        "rank_for_explained_variance_target": rank_for_target,
        "rank_was_capped": rank_for_target > args.maximum_rank,
        "recommended_rank": recommended_rank,
        "fdt_ou_compatibility_by_lag": compatibility,
        "prony_weights": weight.tolist(),
        "recommended_memory_times_ps": times.tolist(),
        "prony_fit_mse": fit_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-lag", type=int, default=16)
    parser.add_argument("--hankel-blocks", type=int, default=8)
    parser.add_argument("--maximum-rank", type=int, default=8)
    parser.add_argument("--explained-variance", type=float, default=0.9)
    parser.add_argument("--maximum-memory-time-ps", type=float, default=16000.0)
    parser.add_argument("--optimization-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = identify(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
