"""Evaluate the LSP-WM v4.1 framework with multi-step rollout diagnostics.

This script performs a multi-step rollout evaluation:
    1. Encode context states from the validation set
    2. Run a multi-step GENERIC Langevin rollout from each context state
    3. At each rollout step, compute:
       a. Energy score against a declared linear bridge diagnostic target
       b. Distribution matching (KDE density under reference)
       c. Koopman VAMP-2 score (from rollout trajectory)
       d. Diffusion PSD check
       e. Trajectory divergence check (NaN, state norm explosion)
    4. Aggregate all metrics across steps and batches

The bridge target is not an observed intermediate trajectory. This script is
therefore a numerical stability and self-consistency diagnostic, not a
transition-path benchmark. A path benchmark must provide observed intermediate
states at matching physical times.
"""

from __future__ import annotations

import argparse
import json
import math
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
    vamp2_loss,
    kde_mean_force,
    mean_force_matching_loss,
)


def _energy_score_per_sample(
    samples: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Energy score for one sample: ||s - t|| - 0.5 * pairwise mean."""
    first = torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(dim=1)
    pairwise = torch.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], dim=-1
    )
    count = max(1, samples.shape[1])
    return first - 0.5 * pairwise.sum(dim=(1, 2)) / (count * count)


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate_rollout(args: argparse.Namespace) -> dict:
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

    rollout_steps = max(1, args.rollout_steps)
    ensemble_size = max(2, args.ensemble_size)

    # Per-step accumulators
    step_energy_score: dict[int, float] = defaultdict(float)
    step_persistence: dict[int, float] = defaultdict(float)
    step_count: dict[int, int] = defaultdict(int)
    step_koopman_vamp2: dict[int, float] = defaultdict(float)
    step_koopman_rows: dict[int, int] = defaultdict(int)
    step_distribution_log_density: dict[int, float] = defaultdict(float)
    step_divergence_count: dict[int, int] = defaultdict(int)

    trajectory_energy_score = 0.0
    trajectory_persistence = 0.0
    trajectory_rows = 0
    all_koopman_sources = []
    all_koopman_targets = []
    total_divergences = 0
    total_steps_sampled = 0

    diffusion_minimum = float("inf")
    diffusion_condition = 0.0
    fdt_residual_max = 0.0
    q_fdt_residual_max = 0.0
    auxiliary_fdt_residual_max = 0.0
    fdt_sampling_normalized_max = 0.0
    analytic_fdt_residual_max = 0.0
    max_state_norm = 0.0
    divergence_norm_limit = 0.0

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        outputs = model.forward_paired(values)
        current = outputs["lspwm_context_state"]
        target = outputs["lspwm_target_state"]
        sequence = outputs["lspwm_context_sequence"]
        mask = outputs["lspwm_context_mask"]
        condition = outputs["lspwm_condition"]
        temperature = outputs["lspwm_temperature_k"]
        horizon_time_ps = values["horizon_time_ps"]
        history = outputs["lspwm_history_features"]
        reference = sequence[mask].detach()
        batch_size = current.shape[0]

        # Run multi-step rollout: divide horizon into equal steps
        dt_per_step = horizon_time_ps / rollout_steps

        # Expand for ensemble
        exp_current = current.repeat_interleave(ensemble_size, dim=0)
        exp_condition = condition.repeat_interleave(ensemble_size, dim=0)
        exp_temperature = temperature.repeat_interleave(ensemble_size)
        exp_dt = dt_per_step.repeat_interleave(ensemble_size)
        auxiliary = current.new_zeros(
            exp_current.shape[0], model.lspwm.world_model.config.auxiliary_dim
        )

        # Full rollout trajectory
        trajectory = model.lspwm.world_model.rollout(
            exp_current,
            exp_condition,
            exp_dt,
            rollout_steps,
            exp_temperature,
            stochastic=True,
            auxiliary=auxiliary,
        )
        states = trajectory["q"]  # [batch*ensemble, steps+1, state_dim]
        states = states.reshape(batch_size, ensemble_size, rollout_steps + 1, -1)

        # At each step, compute diagnostics
        for step_idx in range(1, rollout_steps + 1):
            q_step = states[:, :, step_idx, :]  # [batch, ensemble, state_dim]
            # Target at this step: interpolate between current and target
            step_frac = step_idx / rollout_steps
            step_target = current + step_frac * (target - current)

            # Energy score
            score = _energy_score_per_sample(q_step, step_target)
            persistence = torch.linalg.vector_norm(current - step_target, dim=-1)

            # Distribution matching
            flat_step = q_step.reshape(-1, current.shape[-1])
            if reference.shape[0] >= 2:
                bandwidth = model.lspwm.mean_force_bandwidth
                bw = bandwidth * bandwidth
                delta = flat_step[:, None, :] - reference[None, :, :]
                log_kernel = -0.5 * delta.square().sum(dim=-1) / bw
                log_density = torch.logsumexp(log_kernel, dim=1) - math.log(
                    max(reference.shape[0], 1)
                )
                mean_log_density = float(log_density.mean().detach().cpu())
            else:
                mean_log_density = 0.0

            # Numerical stability diagnostic. This is not a physical validity
            # claim and is reported separately from the path metrics.
            nan_count = int(torch.isnan(q_step).any(dim=-1).any(dim=-1).sum().detach().cpu())
            inf_count = int(torch.isinf(q_step).any(dim=-1).any(dim=-1).sum().detach().cpu())
            state_norm = torch.linalg.vector_norm(q_step, dim=-1)
            initial_norm = torch.linalg.vector_norm(current, dim=-1)
            norm_limit = torch.maximum(
                initial_norm * 100.0,
                current.new_full(initial_norm.shape, 100.0),
            )
            norm_count = int((state_norm > norm_limit[:, None]).sum().detach().cpu())
            divergence = nan_count + inf_count + norm_count
            max_state_norm = max(max_state_norm, float(state_norm.max().detach().cpu()))
            divergence_norm_limit = max(
                divergence_norm_limit,
                float(norm_limit.max().detach().cpu()),
            )

            step_energy_score[step_idx] += float(score.sum().detach().cpu())
            step_persistence[step_idx] += float(persistence.sum().detach().cpu())
            step_count[step_idx] += batch_size
            step_distribution_log_density[step_idx] += mean_log_density * batch_size
            if divergence > 0:
                step_divergence_count[step_idx] += divergence
                total_divergences += divergence

            # Koopman diagnostics on consecutive rollout steps
            if step_idx >= 1:
                source_step = states[:, :, step_idx - 1, :].reshape(-1, current.shape[-1])
                target_step = states[:, :, step_idx, :].reshape(-1, current.shape[-1])
                if source_step.shape[0] >= 2 and source_step.shape[0] == target_step.shape[0]:
                    vamp2 = float(
                        (-vamp2_loss(source_step, target_step, ridge=1.0e-6))
                        .detach()
                        .cpu()
                    )
                    step_koopman_vamp2[step_idx] += vamp2 * batch_size
                    step_koopman_rows[step_idx] += batch_size

            # Collect for aggregate Koopman
            if step_idx == rollout_steps:
                all_koopman_sources.append(current)
                all_koopman_targets.append(
                    states[:, :, -1, :].mean(dim=1)  # mean over ensemble
                )

        # Trajectory-level energy score (whole rollout)
        q_final = states[:, :, -1, :]
        final_score = _energy_score_per_sample(q_final, target)
        final_persistence = torch.linalg.vector_norm(current - target, dim=-1)
        trajectory_energy_score += float(final_score.sum().detach().cpu())
        trajectory_persistence += float(final_persistence.sum().detach().cpu())
        trajectory_rows += batch_size
        total_steps_sampled += batch_size * rollout_steps

        # Diffusion diagnostics
        diffusion = model.lspwm.world_model.diffusion(
            current.shape[0], temperature, reference=current
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
            condition,
            dt_per_step,
            temperature,
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

    # Aggregate per-step results
    by_step = {}
    macro_skill_sum = 0.0
    n_horizons = 0
    for step_idx in sorted(step_count.keys()):
        cnt = max(1, step_count[step_idx])
        mean_score = step_energy_score[step_idx] / cnt
        mean_persistence = step_persistence[step_idx] / cnt
        skill = 1.0 - mean_score / max(mean_persistence, 1.0e-8)
        macro_skill_sum += skill
        n_horizons += 1
        koopman_vamp2 = (
            step_koopman_vamp2[step_idx] / max(1, step_koopman_rows[step_idx])
            if step_koopman_rows[step_idx] > 0
            else 0.0
        )
        dist_log_density = step_distribution_log_density[step_idx] / cnt
        by_step[str(step_idx)] = {
            "energy_score": round(mean_score, 4),
            "persistence": round(mean_persistence, 4),
            "energy_skill_percent": round(100.0 * skill, 4),
            "koopman_vamp2_score": round(koopman_vamp2, 4),
            "mean_log_density": round(dist_log_density, 4),
            "divergence_count": step_divergence_count[step_idx],
            "rows": cnt,
        }

    # Aggregate Koopman diagnostics
    koopman_result = {}
    if all_koopman_sources:
        koop_source = torch.cat(all_koopman_sources, dim=0)
        koop_target = torch.cat(all_koopman_targets, dim=0)
        if koop_source.shape[0] >= 2:
            diagnostics = koopman_diagnostics(koop_source, koop_target, lag_time=1.0)
            koopman_result = {
                "residual_mse": float(diagnostics["residual_mse"]),
                "vamp2_score": float(diagnostics["vamp2_score"]),
                "max_implied_timescale": float(
                    diagnostics["implied_timescales"].real.max()
                ),
            }

    # Trajectory-level energy skill
    traj_skill = (
        1.0
        - trajectory_energy_score / max(trajectory_persistence, 1.0e-8)
    )

    result = {
        "framework": args.framework_label,
        "protocol": "multi-step rollout self-consistency",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "rollout_steps": rollout_steps,
        "ensemble_size": ensemble_size,
        "max_batches": args.max_batches,
        "total_rows": trajectory_rows,
        "total_steps_sampled": total_steps_sampled,
        "total_divergences": total_divergences,
        "macro_skill_percent": round(100.0 * macro_skill_sum / max(n_horizons, 1), 4),
        "trajectory_energy_skill_percent": round(100.0 * traj_skill, 4),
        "by_step": by_step,
        "koopman": koopman_result,
        "physical_gates": {
            "diffusion_psd": diffusion_minimum > 0.0,
            "minimum_diffusion_eigenvalue": round(diffusion_minimum, 8),
            "maximum_diffusion_condition_number": round(diffusion_condition, 4),
            "analytic_fdt_residual": round(analytic_fdt_residual_max, 6),
            "fdt_integrator_identity": bool(analytic_fdt_residual_max < 1.0e-6),
            "fdt_sampling_normalized_residual": round(fdt_sampling_normalized_max, 6),
            "fdt_raw_residual": round(fdt_residual_max, 6),
            "q_fdt_residual": round(q_fdt_residual_max, 6),
            "auxiliary_fdt_residual": round(auxiliary_fdt_residual_max, 6),
            "fdt_sample_count_per_batch": args.fdt_samples,
            "max_state_norm": round(max_state_norm, 6),
            "divergence_norm_limit": round(divergence_norm_limit, 6),
            "energy_force_is_conservative_by_construction": True,
        },
        "intermediate_target_is_observed": False,
        "claim_boundary": "multi-step numerical rollout diagnostic; intermediate "
        "linear targets are not a transition-path benchmark",
    }
    return result


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
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--fdt-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--framework-label", default="LSP-WM v4.1")
    args = parser.parse_args()
    result = evaluate_rollout(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
