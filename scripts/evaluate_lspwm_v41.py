"""Evaluate the independent LSP-WM v4.1 protein framework."""

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
from protein_jepa_frame.models.lspwm import (
    empirical_fdt_residual,
    koopman_diagnostics,
)


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
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint["model"]
    model_config = model_config_from_dict(checkpoint["config"]["model"], state)
    model = ProteinJEPA(model_config).to(device)
    state = upgrade_lspwm_state_dict(model, state)
    model.load_state_dict(state, strict=True)
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
    per_horizon: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    score_sum = 0.0
    persistence_sum = 0.0
    rows = 0
    koopman_rows = []
    diffusion_minimum = float("inf")
    diffusion_condition = 0.0
    fdt_residual_max = 0.0
    q_fdt_residual_max = 0.0
    auxiliary_fdt_residual_max = 0.0
    fdt_sampling_normalized_max = 0.0
    analytic_fdt_residual_max = 0.0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        outputs = model.forward_paired(values)
        current = outputs["lspwm_context_state"]
        target = outputs["lspwm_target_state"]
        samples = model.lspwm.sample_endpoint(
            current,
            outputs["lspwm_condition"],
            outputs["lspwm_history_features"],
            values["horizon_time_ps"],
            outputs["lspwm_temperature_k"],
            ensemble_size=args.ensemble_size,
        )
        score = _energy_score(samples, target)
        persistence = torch.linalg.vector_norm(current - target, dim=-1)
        score_sum += float(score.sum())
        persistence_sum += float(persistence.sum())
        rows += current.shape[0]
        for horizon in values["horizon_frames"].unique(sorted=True):
            mask = values["horizon_frames"] == horizon
            key = str(int(horizon))
            per_horizon[key]["score"] += float(score[mask].sum())
            per_horizon[key]["persistence"] += float(persistence[mask].sum())
            per_horizon[key]["rows"] += float(mask.sum())
        sequence = outputs["lspwm_context_sequence"]
        mask = outputs["lspwm_context_mask"]
        pair_mask = mask[:, 1:] & mask[:, :-1]
        if pair_mask.any():
            koopman_rows.append((sequence[:, :-1][pair_mask], sequence[:, 1:][pair_mask]))
        diffusion = model.lspwm.world_model.diffusion(
            current.shape[0], outputs["lspwm_temperature_k"], reference=current
        )
        eigenvalues = torch.linalg.eigvalsh(diffusion)
        diffusion_minimum = min(diffusion_minimum, float(eigenvalues.min()))
        diffusion_condition = max(
            diffusion_condition,
            float((eigenvalues[:, -1] / eigenvalues[:, 0].clamp(min=1.0e-12)).max()),
        )
        fdt = empirical_fdt_residual(
            model.lspwm.world_model,
            current,
            outputs["lspwm_condition"],
            values["horizon_time_ps"],
            outputs["lspwm_temperature_k"],
            samples=args.fdt_samples,
        )
        fdt_residual_max = max(fdt_residual_max, float(fdt["fdt_residual"]))
        q_fdt_residual_max = max(q_fdt_residual_max, float(fdt["q_fdt_residual"]))
        auxiliary_fdt_residual_max = max(
            auxiliary_fdt_residual_max,
            float(fdt["auxiliary_fdt_residual"]),
        )
        fdt_sampling_normalized_max = max(
            fdt_sampling_normalized_max,
            float(fdt["fdt_sampling_normalized"]),
        )
        analytic_fdt_residual_max = max(
            analytic_fdt_residual_max,
            float(fdt["analytic_fdt_residual"]),
        )
    horizon_results = {}
    horizon_skills = []
    for key, values in sorted(per_horizon.items(), key=lambda item: int(item[0])):
        skill = 1.0 - values["score"] / max(values["persistence"], 1.0e-8)
        horizon_skills.append(skill)
        horizon_results[key] = {**values, "energy_skill_percent": 100.0 * skill}
    koopman = {}
    if koopman_rows:
        source = torch.cat([item[0] for item in koopman_rows])
        target = torch.cat([item[1] for item in koopman_rows])
        diagnostics = koopman_diagnostics(source, target, lag_time=1.0)
        koopman = {
            "residual_mse": float(diagnostics["residual_mse"]),
            "vamp2_score": float(diagnostics["vamp2_score"]),
            "max_implied_timescale": float(diagnostics["implied_timescales"].real.max()),
        }
    return {
        "framework": "LSP-WM v4.1",
        "protocol": "matched TM-JEPA endpoint-energy validation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "rows": rows,
        "macro_skill_percent": 100.0 * sum(horizon_skills) / max(len(horizon_skills), 1),
        "ratio_skill_percent": 100.0 * (1.0 - score_sum / max(persistence_sum, 1.0e-8)),
        "by_horizon": horizon_results,
        "koopman": koopman,
        "physical_gates": {
            "diffusion_psd": diffusion_minimum > 0.0,
            "minimum_diffusion_eigenvalue": diffusion_minimum,
            "maximum_diffusion_condition_number": diffusion_condition,
            "analytic_fdt_residual": analytic_fdt_residual_max,
            "fdt_integrator_identity": analytic_fdt_residual_max < 1.0e-6,
            "fdt_sampling_normalized_residual": fdt_sampling_normalized_max,
            "fdt_raw_residual": fdt_residual_max,
            "q_fdt_residual": q_fdt_residual_max,
            "auxiliary_fdt_residual": auxiliary_fdt_residual_max,
            "fdt_sample_count_per_batch": args.fdt_samples,
            "energy_force_is_conservative_by_construction": True,
        },
        "claim_boundary": "independent LSP-WM result; compare against EVA5C only after matched held-out evaluation",
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
    parser.add_argument("--fdt-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6303)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
