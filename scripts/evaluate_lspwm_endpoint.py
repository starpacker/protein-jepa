"""Evaluate observed one-step LSP-WM endpoint prediction.

This is the primary V4.4 dynamics benchmark. It compares a deterministic
world-model step at the declared physical horizon against the observed paired
endpoint. It does not use bridge interpolation or generated intermediate
targets, and it is independent of rollout-training objectives.
"""

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

from protein_jepa_frame.config import load_config, model_config_from_dict
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
from protein_jepa_frame.models.lspwm import overdamped_transition_nll


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_term = torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(dim=1)
    pairwise = torch.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], dim=-1
    )
    return target_term - 0.5 * pairwise.mean(dim=(1, 2))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"]
    model_config = model_config_from_dict(checkpoint["config"]["model"], state)
    model = ProteinJEPA(model_config).to(device)
    model.load_state_dict(upgrade_lspwm_state_dict(model, state), strict=True)
    model.eval()
    if model.lspwm is None:
        raise ValueError("checkpoint is not an LSP-WM checkpoint")

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

    totals = defaultdict(float)
    rows = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        outputs = model.forward_paired(values)
        current = outputs["lspwm_context_state"]
        target = outputs["lspwm_target_state"]
        condition = outputs["lspwm_condition"]
        temperature = outputs["lspwm_temperature_k"]
        dt = values["horizon_time_ps"]
        auxiliary = current.new_zeros(current.shape[0], model.lspwm.world_model.config.auxiliary_dim)

        deterministic = model.lspwm.world_model.step(
            current,
            auxiliary,
            condition,
            dt,
            temperature,
            stochastic=False,
        )["q"]
        samples = model.lspwm.sample_endpoint(
            current,
            condition,
            outputs["lspwm_history_features"],
            dt,
            temperature,
            ensemble_size=args.ensemble_size,
        )
        prediction_mse = (deterministic - target).square().mean(dim=-1)
        persistence_mse = (current - target).square().mean(dim=-1)
        energy_score = _energy_score(samples, target)
        persistence_energy = torch.linalg.vector_norm(current - target, dim=-1)
        nll = overdamped_transition_nll(
            model.lspwm.world_model,
            current,
            target,
            condition,
            dt,
            temperature,
            history_features=outputs["lspwm_history_features"],
        )

        predicted_roles = model.lspwm.role_readout(deterministic)
        target_roles = model.lspwm.role_readout(target)
        predicted_state = predicted_roles["msm_logits"].argmax(dim=-1)
        target_state = target_roles["msm_logits"].argmax(dim=-1)
        msm_accuracy = (predicted_state == target_state).float()
        msm_overlap = (
            predicted_roles["msm_logits"].softmax(dim=-1)
            * target_roles["msm_logits"].softmax(dim=-1)
        ).sum(dim=-1)

        for horizon in values["horizon_frames"].unique(sorted=True):
            mask = values["horizon_frames"] == horizon
            key = str(int(horizon))
            count = float(mask.sum())
            totals[f"{key}.rows"] += count
            totals[f"{key}.prediction_mse"] += float(prediction_mse[mask].sum())
            totals[f"{key}.persistence_mse"] += float(persistence_mse[mask].sum())
            totals[f"{key}.energy_score"] += float(energy_score[mask].sum())
            totals[f"{key}.persistence_energy"] += float(persistence_energy[mask].sum())
            totals[f"{key}.msm_accuracy"] += float(msm_accuracy[mask].sum())
            totals[f"{key}.msm_overlap"] += float(msm_overlap[mask].sum())
        totals["nll"] += float(nll) * current.shape[0]
        rows += current.shape[0]

    by_horizon = {}
    for key in sorted({name.split(".")[0] for name in totals if "." in name}, key=int):
        count = max(totals[f"{key}.rows"], 1.0)
        pred_mse = totals[f"{key}.prediction_mse"] / count
        persistence_mse = totals[f"{key}.persistence_mse"] / count
        score = totals[f"{key}.energy_score"] / count
        persistence_energy = totals[f"{key}.persistence_energy"] / count
        by_horizon[key] = {
            "rows": int(count),
            "deterministic_endpoint_mse": pred_mse,
            "persistence_mse": persistence_mse,
            "endpoint_skill_vs_persistence": 1.0 - pred_mse / max(persistence_mse, 1.0e-12),
            "ensemble_energy_score": score,
            "persistence_energy": persistence_energy,
            "energy_skill_vs_persistence": 1.0 - score / max(persistence_energy, 1.0e-12),
            "msm_endpoint_accuracy": totals[f"{key}.msm_accuracy"] / count,
            "msm_soft_overlap": totals[f"{key}.msm_overlap"] / count,
        }

    return {
        "framework": "LSP-WM v4.4",
        "protocol": "observed one-step endpoint prediction",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "rows": rows,
        "mean_transition_nll": totals["nll"] / max(rows, 1),
        "by_horizon": by_horizon,
        "claim_boundary": "direct endpoint and MSM diagnostics; no claim about unobserved intermediate paths",
    }


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
    parser.add_argument("--seed", type=int, default=6442)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
