"""Train the first protein-scale LSP-WM pilot on fixed physical CVs.

This is intentionally a 16D pilot. It validates the thermodynamic core before
the learned four-dimensional JEPA projection is promoted into the canonical
20D model.
"""

from __future__ import annotations

import argparse
import json
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
from protein_jepa_frame.models.lspwm import (
    LSPWMConfig,
    LSPWMWorldModel,
    kde_mean_force,
    mean_force_matching_loss,
    overdamped_transition_nll,
)
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
    return (first - 0.5 * pairwise.sum(dim=(1, 2)) / max(1, count * count)).mean()


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _fixed_cv(
    batch: dict,
    cv: FixedBackboneCollectiveVariables,
    mean: torch.Tensor,
    scale: torch.Tensor,
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = cv(
        batch[f"{prefix}_coords"],
        batch[f"{prefix}_atom_mask"],
        batch[f"{prefix}_residue_mask"],
        batch[f"{prefix}_chain_id"],
        batch[f"{prefix}_residue_index"],
    )
    q = (raw - mean) / scale.clamp(min=1.0e-6)
    mask = batch[f"{prefix}_residue_mask"].any(dim=2)
    return q, mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--diffusion-scale", type=float, default=1.0e-6)
    parser.add_argument("--bandwidth", type=float, default=1.5)
    parser.add_argument("--force-weight", type=float, default=0.25)
    parser.add_argument("--nll-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.0)
    parser.add_argument("--training-ensemble-size", type=int, default=8)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_config = load_config(args.config)
    data_config = replace(
        source_config.data,
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=0,
        paired_all_horizons=False,
        sampling_mode="hierarchical",
        training_mode="paired_window",
    )
    dataset = build_dataset(data_config)
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=args.steps * args.batch_size,
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
    cv_mean = torch.tensor(source_config.model.tmjepa_cv_mean, device=device)
    cv_scale = torch.tensor(source_config.model.tmjepa_cv_scale, device=device)
    if cv_mean.numel() != cv.output_dim:
        raise ValueError("the source config must contain fixed-CV statistics")
    model = LSPWMWorldModel(
        LSPWMConfig(
            state_dim=cv.output_dim,
            condition_dim=0,
            auxiliary_dim=source_config.model.tmjepa_auxiliary_dim,
            reference_temperature_k=source_config.model.tmjepa_reference_temperature_k,
            diffusion_scale=args.diffusion_scale,
            relaxation_times_ps=source_config.model.tmjepa_memory_times_ps,
            coupling_scale=source_config.model.tmjepa_coupling_scale,
            history_feature_dim=cv.output_dim * 3,
            history_closure_hidden_dim=256,
            history_closure_max_shift=0.55,
            history_closure_horizon_gate_steps=2.0,
        )
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5)
    model.train()
    log = []
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = _move(next(iterator), device)
        except StopIteration:
            iterator = iter(loader)
            batch = _move(next(iterator), device)
        context_q, context_mask = _fixed_cv(batch, cv, cv_mean, cv_scale, "context")
        future_q, future_mask = _fixed_cv(batch, cv, cv_mean, cv_scale, "future")
        current = _last_valid(context_q, context_mask)
        target = _last_valid(future_q, future_mask).detach()
        history_features = _history_features(context_q, context_mask)
        reference = context_q[context_mask].detach()
        force_target = kde_mean_force(current.detach(), reference, args.bandwidth)
        force_loss = mean_force_matching_loss(model.potential, current, force_target)
        dynamic_loss = overdamped_transition_nll(
            model,
            current,
            target,
            None,
            batch["horizon_time_ps"],
            batch.get("temperature_k"),
            history_features=history_features,
        )
        energy_loss = current.new_zeros(())
        if args.energy_weight > 0.0:
            ensemble = args.training_ensemble_size
            expanded_current = current.repeat_interleave(ensemble, dim=0)
            expanded_target = target.repeat_interleave(ensemble, dim=0)
            expanded_dt = batch["horizon_time_ps"].repeat_interleave(ensemble)
            expanded_temperature = batch["temperature_k"].repeat_interleave(ensemble)
            auxiliary = current.new_zeros(
                expanded_current.shape[0], model.config.auxiliary_dim
            )
            stochastic = model.step(
                expanded_current,
                auxiliary,
                None,
                expanded_dt,
                expanded_temperature,
                stochastic=True,
            )["q"]
            correction = model.history_correction(
                current,
                history_features,
                batch["horizon_time_ps"],
                batch["temperature_k"],
            )
            samples = stochastic.reshape(current.shape[0], ensemble, -1) + correction[:, None, :]
            energy_loss = _energy_score(samples, target)
        loss = (
            args.nll_weight * dynamic_loss
            + args.force_weight * force_loss
            + args.energy_weight * energy_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "dynamic_nll": float(dynamic_loss.detach()),
            "force_loss": float(force_loss.detach()),
            "energy_loss": float(energy_loss.detach()),
        }
        log.append(record)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(record))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "framework": "LSP-WM-v4.1-fixed-cv-pilot",
            "model": model.state_dict(),
            "config": model.config.__dict__,
            "cv_mean": cv_mean.detach().cpu(),
            "cv_scale": cv_scale.detach().cpu(),
            "source_config": str(Path(args.config).resolve()),
            "manifest": str(Path(args.manifest).resolve()),
            "seed": args.seed,
            "steps": args.steps,
            "log": log,
        },
        output,
    )
    print(json.dumps({"checkpoint": str(output), "steps": args.steps}, indent=2))


if __name__ == "__main__":
    main()
