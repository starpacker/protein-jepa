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


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
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
        paired_horizons_frames=(1,),
        paired_all_horizons=False,
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
    device = torch.device(
        device_name
        if device_name != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    artifact_config = json.loads((checkpoint_path.parent / "config.json").read_text())
    model = ProteinJEPA(
        model_config_from_dict(artifact_config["model"], state)
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    totals: dict[str, float] = defaultdict(float)
    by_temperature: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    proteins: set[str] = set()
    torch.manual_seed(seed)
    for batch in loader:
        proteins.update(str(value) for value in batch["protein_id"])
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            outputs = model.forward_paired(batch)
        prediction = outputs["tmjepa_score_prediction"]
        target = outputs["tmjepa_score_target"]
        row_error = (prediction - target).square().mean(dim=(-1, -2))
        row_baseline = target.square().mean(dim=(-1, -2))
        row_prediction = prediction.square().mean(dim=(-1, -2))
        temperatures = outputs["tmjepa_temperature_k"].round().to(torch.int64)
        for index in range(row_error.shape[0]):
            values = by_temperature[int(temperatures[index])]
            values["score_squared_error"] += float(row_error[index].cpu())
            values["zero_score_squared_error"] += float(row_baseline[index].cpu())
            values["prediction_squared"] += float(row_prediction[index].cpu())
            values["samples"] += 1.0
        totals["score_squared_error"] += float(row_error.sum().cpu())
        totals["zero_score_squared_error"] += float(row_baseline.sum().cpu())
        totals["prediction_squared"] += float(row_prediction.sum().cpu())
        totals["samples"] += float(row_error.shape[0])

    def summarize(values: dict[str, float]) -> dict[str, float | int]:
        count = max(values["samples"], 1.0)
        score_mse = values["score_squared_error"] / count
        baseline_mse = values["zero_score_squared_error"] / count
        return {
            "samples": int(values["samples"]),
            "score_mse": score_mse,
            "zero_score_mse": baseline_mse,
            "score_skill": 1.0 - score_mse / max(baseline_mse, 1.0e-12),
            "prediction_score_rms": (values["prediction_squared"] / count) ** 0.5,
        }

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "manifest": str(manifest_path),
        "seed": seed,
        "noise_std": model.tmjepa.dynamics.equilibrium_noise_std,
        "noise_samples_per_structure": model.tmjepa.dynamics.equilibrium_noise_samples,
        "protein_coverage": len(proteins),
        "overall": summarize(totals),
        "by_temperature_k": {
            str(temperature): summarize(values)
            for temperature, values in sorted(by_temperature.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TM-JEPA PMF score matching")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6405)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate(
        args.config,
        args.checkpoint,
        args.manifest,
        args.output,
        args.max_batches,
        args.batch_size,
        args.seed,
        args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
