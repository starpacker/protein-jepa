from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker
from protein_jepa_frame.data.sampler import HierarchicalClipSampler
from protein_jepa_frame.models.jepa import ProteinJEPA, lagged_state_jepa_terms, lagged_state_semigroup_terms


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "positive_fraction": float((array > 0.0).mean()),
        "count": int(array.size),
    }


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
    torch.manual_seed(seed)
    config = load_config(config_path)
    data_config = replace(
        config.data,
        manifest_path=str(manifest_path),
        batch_size=batch_size,
        num_workers=0,
        sampling_mode="hierarchical",
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
        collate_fn=collate_clips,
        drop_last=True,
    )
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = ProteinJEPA(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    model.eval()
    masker = SpatiotemporalBlockMasker(
        config.mask,
        required_future_horizon=max(config.model.state_prediction_lags, default=0),
    )

    lag_values: dict[int, dict[str, list[float]]] = {}
    composition_values: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            context_mask, target_mask = masker(batch["residue_mask"])
            outputs = model(batch, context_mask, target_mask)
            full_latents = model.context_encoder(
                coords=batch["coords"],
                residue_mask=batch["residue_mask"],
                aatype=batch["aatype"],
                residue_index=batch["residue_index"],
                chain_id=batch["chain_id"],
                token_mask=batch["residue_mask"],
            )
            online_outputs = model.encode_states(
                full_latents,
                batch["residue_mask"],
                time_ps=batch.get("time_ps"),
                temperature_k=batch.get("temperature_k"),
                dataset_index=batch.get("dataset_index"),
            )
            online_states = online_outputs.get("state_motion_states", online_outputs["frame_states"])
            context_frames = outputs["context_frame_mask"]
            batch_count, frame_count = context_frames.shape
            frame_index = torch.arange(frame_count, device=device).view(1, -1).expand(batch_count, -1)
            source_index = torch.where(context_frames, frame_index, -1).max(dim=1).values
            for lag, term in lagged_state_jepa_terms(outputs).items():
                values = lag_values.setdefault(
                    lag,
                    {
                        "loss": [],
                        "persistence": [],
                        "intrinsic_persistence": [],
                        "skill": [],
                        "alignment": [],
                        "online_loss": [],
                        "online_persistence": [],
                        "online_intrinsic_persistence": [],
                        "online_skill": [],
                        "online_alignment": [],
                    },
                )
                loss = float(term["loss"].cpu())
                persistence = float(term["persistence_loss"].cpu())
                values["loss"].append(loss)
                values["persistence"].append(persistence)
                values["intrinsic_persistence"].append(float(term["intrinsic_persistence_loss"].cpu()))
                values["skill"].append(1.0 - loss / max(persistence, 1.0e-8))
                values["alignment"].append(float(term["source_alignment_loss"].cpu()))
                target_index = source_index + lag
                eligible = (source_index >= 0) & (target_index < frame_count)
                selected_batch = torch.arange(batch_count, device=device)[eligible]
                selected_source = source_index[eligible]
                selected_target = target_index[eligible]
                prediction = outputs["lagged_state_predictions"][lag][selected_batch, selected_source]
                transition_source = outputs["transition_states"][selected_batch, selected_source]
                online_source = online_states[selected_batch, selected_source]
                online_target = online_states[selected_batch, selected_target]
                online_loss = float(F.smooth_l1_loss(prediction, online_target).cpu())
                online_persistence = float(F.smooth_l1_loss(transition_source, online_target).cpu())
                values["online_loss"].append(online_loss)
                values["online_persistence"].append(online_persistence)
                values["online_intrinsic_persistence"].append(
                    float(F.smooth_l1_loss(online_source, online_target).cpu())
                )
                values["online_skill"].append(
                    1.0 - online_loss / max(online_persistence, 1.0e-8)
                )
                values["online_alignment"].append(
                    float(F.smooth_l1_loss(transition_source, online_source).cpu())
                )
            for key, term in lagged_state_semigroup_terms(outputs, model.lagged_state_predictor).items():
                composition_values.setdefault(key, []).append(float(term["loss"].cpu()))

    result = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "max_batches": max_batches,
        "batch_size": batch_size,
        "seed": seed,
        "lags": {
            str(lag): {name: _summary(values) for name, values in metrics.items()}
            for lag, metrics in sorted(lag_values.items())
        },
        "semigroup_compositions": {
            key: _summary(values) for key, values in sorted(composition_values.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned lag transitions on held-out trajectory clips")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
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
