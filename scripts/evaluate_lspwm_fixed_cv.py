"""Evaluate a fixed-CV LSP-WM pilot with the matched endpoint energy metric."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.lspwm import LSPWMConfig, LSPWMWorldModel
from protein_jepa_frame.models.thermodynamic import FixedBackboneCollectiveVariables


def _last_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = mask.long().sum(dim=1).clamp(min=1) - 1
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, index]


def _nth_last(values: torch.Tensor, mask: torch.Tensor, offset: int) -> torch.Tensor:
    count = mask.long().sum(dim=1)
    index = (count - 1 - offset).clamp(min=0)
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, index]


def _history_features(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    last = _nth_last(values, mask, 0)
    previous = _nth_last(values, mask, 1)
    rank = mask.long().cumsum(dim=1)
    count = mask.long().sum(dim=1, keepdim=True)
    recent = mask & (rank > (count - 4).clamp(min=0))
    recent_mean = (values * recent.unsqueeze(-1)).sum(dim=1) / recent.sum(dim=1, keepdim=True).clamp(min=1)
    global_mean = (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    return torch.cat([last - recent_mean, recent_mean - global_mean, last - previous], dim=-1)


def _energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    first = torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(dim=1)
    pairwise = torch.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], dim=-1
    )
    count = samples.shape[1]
    return first - 0.5 * pairwise.sum(dim=(1, 2)) / max(1, count * count)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6303)
    args = parser.parse_args()
    if args.ensemble_size < 2:
        parser.error("--ensemble-size must be at least 2")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LSPWMWorldModel(LSPWMConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    source_config = load_config(args.config)
    data_config = replace(
        source_config.data,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=0,
        paired_all_horizons=True,
        sampling_mode="hierarchical",
        training_mode="paired_window",
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
    cv = FixedBackboneCollectiveVariables(
        num_segments=source_config.model.tmjepa_num_segments,
        contact_cutoff_angstrom=source_config.model.physchem_contact_cutoff,
        contact_width_angstrom=source_config.model.tmjepa_contact_width_angstrom,
    ).to(device)
    mean = checkpoint["cv_mean"].to(device)
    scale = checkpoint["cv_scale"].to(device)
    totals: dict[str, float] = defaultdict(float)
    by_horizon: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rows = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch = _move(batch, device)
        context_raw = cv(
            batch["context_coords"],
            batch["context_atom_mask"],
            batch["context_residue_mask"],
            batch["context_chain_id"],
            batch["context_residue_index"],
        )
        future_raw = cv(
            batch["future_coords"],
            batch["future_atom_mask"],
            batch["future_residue_mask"],
            batch["future_chain_id"],
            batch["future_residue_index"],
        )
        context_q = (context_raw - mean) / scale.clamp(min=1.0e-6)
        future_q = (future_raw - mean) / scale.clamp(min=1.0e-6)
        context_mask = batch["context_residue_mask"].any(dim=2)
        future_mask = batch["future_residue_mask"].any(dim=2)
        current = _last_valid(context_q, context_mask)
        target = _last_valid(future_q, future_mask)
        history_features = _history_features(context_q, context_mask)
        ensemble_current = current.repeat_interleave(args.ensemble_size, dim=0)
        ensemble_target = target.repeat_interleave(args.ensemble_size, dim=0)
        dt = batch["horizon_time_ps"].repeat_interleave(args.ensemble_size)
        temperature = batch["temperature_k"].repeat_interleave(args.ensemble_size)
        auxiliary = torch.zeros(
            ensemble_current.shape[0], model.config.auxiliary_dim, device=device
        )
        result = model.step(
            ensemble_current,
            auxiliary,
            None,
            dt,
            temperature,
            stochastic=True,
        )
        samples = result["q"].reshape(current.shape[0], args.ensemble_size, -1)
        correction = model.history_correction(
            current,
            history_features,
            batch["horizon_time_ps"],
            batch["temperature_k"],
        )
        samples = samples + correction[:, None, :]
        prediction_score = _energy_score(samples, target)
        persistence_score = torch.linalg.vector_norm(current - target, dim=-1)
        skill = 1.0 - prediction_score / persistence_score.clamp(min=1.0e-6)
        for horizon in batch["horizon_frames"].unique(sorted=True):
            mask = batch["horizon_frames"] == horizon
            key = str(int(horizon))
            by_horizon[key]["score"] += float(prediction_score[mask].sum().detach())
            by_horizon[key]["persistence"] += float(persistence_score[mask].sum().detach())
            by_horizon[key]["rows"] += float(mask.sum())
        totals["score"] += float(prediction_score.sum().detach())
        totals["persistence"] += float(persistence_score.sum().detach())
        totals["skill_sum"] += float(skill.sum().detach())
        rows += current.shape[0]
    result = {
        "framework": "LSP-WM v4.1 fixed-CV pilot",
        "protocol": "TM-JEPA V3 matched endpoint energy score",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "rows": rows,
        "macro_skill_percent": 100.0 * (1.0 - totals["score"] / max(totals["persistence"], 1.0e-8)),
        "mean_row_skill_percent": 100.0 * totals["skill_sum"] / max(rows, 1),
        "by_horizon": {
            key: {
                **values,
                "skill_percent": 100.0 * (1.0 - values["score"] / max(values["persistence"], 1.0e-8)),
            }
            for key, values in by_horizon.items()
        },
        "claim_boundary": "pilot comparison only; promotion requires physical gates and held-out test",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
