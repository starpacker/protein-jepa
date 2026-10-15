#!/usr/bin/env python3
"""Evaluate V5.1 sparse motion dynamics with strict causal controls."""

from __future__ import annotations

import argparse
import json
import math
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


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _reverse_raw_history_keep_endpoint(batch: dict) -> dict:
    """Reverse raw context frames before temporal encoding while preserving endpoint."""
    reversed_batch = dict(batch)
    time_length = batch["context_coords"].shape[1]
    permutation = torch.arange(time_length, device=batch["context_coords"].device)
    permutation[:-1] = permutation[:-1].flip(0)
    for key in (
        "context_coords",
        "context_atom_mask",
        "context_residue_mask",
        "context_forces",
    ):
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[1] == time_length:
            reversed_batch[key] = value[:, permutation]
    return reversed_batch


def _shuffle_horizons_within_context(batch: dict) -> dict:
    shuffled = dict(batch)
    rows = torch.arange(
        batch["horizon_frames"].shape[0], device=batch["horizon_frames"].device
    )
    group_index = batch.get("context_group_index")
    if group_index is None:
        permutation = torch.roll(rows, shifts=1)
    else:
        permutation = rows.clone()
        for group in group_index.unique(sorted=True):
            members = torch.nonzero(group_index == group, as_tuple=False).flatten()
            permutation[members] = torch.roll(members, shifts=1)
    shuffled["horizon_frames"] = batch["horizon_frames"][permutation]
    shuffled["horizon_time_ps"] = batch["horizon_time_ps"][permutation]
    return shuffled


def _routing_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp(min=1.0e-8)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def _safe_skill(numerator: float, denominator: float) -> float:
    return 1.0 - numerator / max(denominator, 1.0e-12)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    data_config = replace(
        config.data,
        manifest_path=str(args.manifest),
        batch_size=args.batch_size,
        num_workers=0,
        training_mode="paired_window",
        paired_all_horizons=True,
        sampling_mode="hierarchical",
        load_forces=False,
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
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=collate_paired_windows,
        drop_last=True,
    )

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get("model", checkpoint)
    checkpoint_config = checkpoint.get("config", {})
    model_values = checkpoint_config.get("model", config.model.__dict__)
    model = ProteinJEPA(model_config_from_dict(model_values, state)).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.v51_contextual_dynamics is None:
        raise ValueError("checkpoint does not contain V5.1 sparse motion dynamics")

    totals: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    global_totals: defaultdict[str, float] = defaultdict(float)
    mode_probability_sum: torch.Tensor | None = None
    mode_active_sum: torch.Tensor | None = None

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        batch = _move_batch(batch, device)
        outputs = model.forward_paired(batch)
        history_outputs = model.forward_paired(_reverse_raw_history_keep_endpoint(batch))
        horizon_outputs = model.forward_paired(_shuffle_horizons_within_context(batch))

        target = outputs["v51_target_feature"]
        prediction = outputs["v51_predicted_feature"]
        persistence = outputs["v51_current_feature"]
        history_prediction = history_outputs["v51_predicted_feature"]
        horizon_prediction = horizon_outputs["v51_predicted_feature"]
        short_history_prediction = outputs.get("v51_short_history_prediction")

        observable_target = outputs["v51_observable_target"]
        observable_prediction = outputs["v51_observable_prediction"]
        observable_no_rollout = outputs["v51_no_rollout_observable"]
        observable_oracle = outputs["v51_oracle_observable"]
        observable_ema_oracle = outputs.get("v51_ema_oracle_observable", observable_oracle)
        observable_online_oracle = outputs["v51_online_oracle_observable"]
        observable_history = history_outputs["v51_observable_prediction"]
        observable_horizon = horizon_outputs["v51_observable_prediction"]
        short_history_observable = outputs.get("v51_short_history_observable_prediction")

        target_delta = outputs["v51_target_delta"]
        predicted_delta = outputs["v51_predicted_delta"]
        routing = outputs["v51_router_probabilities"]
        active = outputs["v51_active_mode_mask"].to(routing.dtype)
        entropy = _routing_entropy(routing)
        active_count = active.sum(dim=-1)

        mode_probability = routing.detach().sum(dim=0).cpu()
        mode_active = active.detach().sum(dim=0).cpu()
        mode_probability_sum = (
            mode_probability
            if mode_probability_sum is None
            else mode_probability_sum + mode_probability
        )
        mode_active_sum = (
            mode_active if mode_active_sum is None else mode_active_sum + mode_active
        )

        fixed = outputs.get("v51_fixed_predictions")
        fixed_observable = outputs.get("v51_fixed_observable_predictions")
        fixed_horizons = outputs.get("v51_fixed_horizons")

        for horizon in outputs["horizon_frames"].unique(sorted=True):
            mask = outputs["horizon_frames"] == horizon
            item = totals[int(horizon)]
            item["rows"] += float(mask.sum())
            item["feature_values"] += float(target[mask].numel())
            item["feature_sse"] += float((prediction[mask] - target[mask]).square().sum())
            item["feature_persistence_sse"] += float(
                (persistence[mask] - target[mask]).square().sum()
            )
            item["feature_history_sse"] += float(
                (history_prediction[mask] - target[mask]).square().sum()
            )
            if short_history_prediction is not None:
                short_error = (short_history_prediction[mask] - target[mask]).square()
                if torch.isfinite(short_error).all():
                    item["feature_short_history_sse"] += float(short_error.sum())
            item["feature_horizon_sse"] += float(
                (horizon_prediction[mask] - target[mask]).square().sum()
            )
            item["observable_values"] += float(observable_target[mask].numel())
            item["observable_sse"] += float(
                (observable_prediction[mask] - observable_target[mask]).square().sum()
            )
            item["observable_no_rollout_sse"] += float(
                (observable_no_rollout[mask] - observable_target[mask]).square().sum()
            )
            item["observable_oracle_sse"] += float(
                (observable_oracle[mask] - observable_target[mask]).square().sum()
            )
            item["observable_ema_oracle_sse"] += float(
                (observable_ema_oracle[mask] - observable_target[mask]).square().sum()
            )
            item["observable_online_oracle_sse"] += float(
                (
                    observable_online_oracle[mask] - observable_target[mask]
                ).square().sum()
            )
            item["observable_history_sse"] += float(
                (observable_history[mask] - observable_target[mask]).square().sum()
            )
            if short_history_observable is not None:
                short_observable_error = (
                    short_history_observable[mask] - observable_target[mask]
                ).square()
                if torch.isfinite(short_observable_error).all():
                    item["observable_short_history_sse"] += float(short_observable_error.sum())
            item["observable_horizon_sse"] += float(
                (observable_horizon[mask] - observable_target[mask]).square().sum()
            )
            item["target_delta_norm"] += float(target_delta[mask].square().sum())
            item["predicted_delta_norm"] += float(predicted_delta[mask].square().sum())
            item["delta_values"] += float(target_delta[mask].numel())
            item["active_count"] += float(active_count[mask].sum())
            item["routing_entropy"] += float(entropy[mask].sum())

            if fixed is not None and fixed_horizons is not None:
                for index, fixed_horizon in enumerate(fixed_horizons.tolist()):
                    item[f"fixed_feature_{int(fixed_horizon)}"] += float(
                        (fixed[mask, index] - target[mask]).square().sum()
                    )
                    if fixed_observable is not None:
                        item[f"fixed_observable_{int(fixed_horizon)}"] += float(
                            (
                                fixed_observable[mask, index]
                                - observable_target[mask]
                            ).square().sum()
                        )

        global_totals["rows"] += float(target.shape[0])

    control_horizons = tuple(int(value) for value in config.model.v51_control_horizons_frames)
    horizons: dict[str, dict] = {}
    aggregate_lists: defaultdict[str, list[float]] = defaultdict(list)
    for horizon, item in sorted(totals.items()):
        rows = max(item["rows"], 1.0)
        feature_values = max(item["feature_values"], 1.0)
        observable_values = max(item["observable_values"], 1.0)
        delta_values = max(item["delta_values"], 1.0)

        feature_mse = item["feature_sse"] / feature_values
        feature_persistence_mse = item["feature_persistence_sse"] / feature_values
        feature_history_mse = item["feature_history_sse"] / feature_values
        feature_short_history_mse = (
            item.get("feature_short_history_sse", 0.0) / feature_values
            if "feature_short_history_sse" in item else None
        )
        feature_horizon_mse = item["feature_horizon_sse"] / feature_values

        observable_mse = item["observable_sse"] / observable_values
        observable_no_rollout_mse = item["observable_no_rollout_sse"] / observable_values
        observable_oracle_mse = item["observable_oracle_sse"] / observable_values
        observable_ema_oracle_mse = item["observable_ema_oracle_sse"] / observable_values
        observable_online_oracle_mse = (
            item["observable_online_oracle_sse"] / observable_values
        )
        observable_history_mse = item["observable_history_sse"] / observable_values
        observable_short_history_mse = (
            item.get("observable_short_history_sse", 0.0) / observable_values
            if "observable_short_history_sse" in item else None
        )
        observable_horizon_mse = item["observable_horizon_sse"] / observable_values

        fixed_feature_mse = {
            str(fixed_horizon): item[f"fixed_feature_{fixed_horizon}"] / feature_values
            for fixed_horizon in control_horizons
            if f"fixed_feature_{fixed_horizon}" in item
        }
        fixed_observable_mse = {
            str(fixed_horizon): item[f"fixed_observable_{fixed_horizon}"]
            / observable_values
            for fixed_horizon in control_horizons
            if f"fixed_observable_{fixed_horizon}" in item
        }
        wrong_fixed_feature = {
            fixed_horizon: mse
            for fixed_horizon, mse in fixed_feature_mse.items()
            if int(fixed_horizon) != horizon
        }
        wrong_fixed_observable = {
            fixed_horizon: mse
            for fixed_horizon, mse in fixed_observable_mse.items()
            if int(fixed_horizon) != horizon
        }
        best_wrong_feature = (
            min(wrong_fixed_feature.values()) if wrong_fixed_feature else None
        )
        best_wrong_observable = (
            min(wrong_fixed_observable.values()) if wrong_fixed_observable else None
        )

        target_displacement = (item["target_delta_norm"] / delta_values) ** 0.5
        predicted_displacement = (item["predicted_delta_norm"] / delta_values) ** 0.5
        displacement_ratio = predicted_displacement / max(target_displacement, 1.0e-12)

        horizon_result = {
            "rows": int(item["rows"]),
            "feature_mse": feature_mse,
            "feature_persistence_mse": feature_persistence_mse,
            "feature_vs_persistence_skill": _safe_skill(
                feature_mse, feature_persistence_mse
            ),
            "strict_raw_history_advantage": _safe_skill(
                feature_mse, feature_history_mse
            ),
            "history_truncation_advantage": (
                _safe_skill(feature_mse, feature_short_history_mse)
                if feature_short_history_mse is not None else None
            ),
            "horizon_shuffle_advantage": _safe_skill(feature_mse, feature_horizon_mse),
            "correct_vs_best_wrong_fixed_skill": (
                _safe_skill(feature_mse, best_wrong_feature)
                if best_wrong_feature is not None
                else None
            ),
            "physical_observable_mse": observable_mse,
            "physical_no_rollout_mse": observable_no_rollout_mse,
            "physical_oracle_mse": observable_oracle_mse,
            "physical_ema_oracle_mse": observable_ema_oracle_mse,
            "physical_online_oracle_mse": observable_online_oracle_mse,
            "physical_vs_no_rollout_skill": _safe_skill(
                observable_mse, observable_no_rollout_mse
            ),
            "physical_oracle_vs_no_rollout_skill": _safe_skill(
                observable_oracle_mse, observable_no_rollout_mse
            ),
            "physical_ema_oracle_vs_no_rollout_skill": _safe_skill(
                observable_ema_oracle_mse, observable_no_rollout_mse
            ),
            "physical_online_oracle_vs_no_rollout_skill": _safe_skill(
                observable_online_oracle_mse, observable_no_rollout_mse
            ),
            "physical_history_advantage": _safe_skill(
                observable_mse, observable_history_mse
            ),
            "physical_history_truncation_advantage": (
                _safe_skill(observable_mse, observable_short_history_mse)
                if observable_short_history_mse is not None else None
            ),
            "physical_horizon_advantage": _safe_skill(
                observable_mse, observable_horizon_mse
            ),
            "physical_correct_vs_best_wrong_fixed_skill": (
                _safe_skill(observable_mse, best_wrong_observable)
                if best_wrong_observable is not None
                else None
            ),
            "target_displacement": target_displacement,
            "predicted_displacement": predicted_displacement,
            "predicted_target_displacement_ratio": displacement_ratio,
            "mean_active_modes": item["active_count"] / rows,
            "mean_routing_entropy": item["routing_entropy"] / rows,
            "fixed_horizon_feature_mse": fixed_feature_mse,
            "fixed_horizon_observable_mse": fixed_observable_mse,
        }
        horizons[str(horizon)] = horizon_result
        for key in (
            "feature_vs_persistence_skill",
            "strict_raw_history_advantage",
            "history_truncation_advantage",
            "horizon_shuffle_advantage",
            "physical_vs_no_rollout_skill",
            "physical_oracle_vs_no_rollout_skill",
            "physical_ema_oracle_vs_no_rollout_skill",
            "physical_online_oracle_vs_no_rollout_skill",
            "physical_history_advantage",
            "physical_history_truncation_advantage",
            "physical_horizon_advantage",
            "predicted_target_displacement_ratio",
            "mean_active_modes",
        ):
            value = horizon_result[key]
            if value is not None and math.isfinite(float(value)):
                aggregate_lists[key].append(float(value))
        for key in (
            "correct_vs_best_wrong_fixed_skill",
            "physical_correct_vs_best_wrong_fixed_skill",
        ):
            if horizon_result[key] is not None and math.isfinite(float(horizon_result[key])):
                aggregate_lists[key].append(float(horizon_result[key]))

    rows = max(global_totals["rows"], 1.0)
    if mode_probability_sum is None or mode_active_sum is None:
        mode_probability_mean = []
        mode_active_fraction = []
    else:
        mode_probability_mean = (mode_probability_sum / rows).tolist()
        mode_active_fraction = (mode_active_sum / rows).tolist()

    aggregate = {
        key: _mean_or_none(values) for key, values in sorted(aggregate_lists.items())
    }
    expected_active = int(model.v51_contextual_dynamics.active_modes)
    utilized_modes = sum(1 for value in mode_active_fraction if value >= args.min_mode_active_fraction)
    active_error = max(
        abs(value - expected_active)
        for value in aggregate_lists.get("mean_active_modes", [float("inf")])
    )
    promotion_gates = {
        "feature_beats_persistence": (
            aggregate.get("feature_vs_persistence_skill") is not None
            and aggregate["feature_vs_persistence_skill"] > args.min_feature_skill
        ),
        "physical_beats_no_rollout": (
            aggregate.get("physical_vs_no_rollout_skill") is not None
            and aggregate["physical_vs_no_rollout_skill"] > args.min_physical_skill
        ),
        "oracle_decoder_is_useful": (
            aggregate.get("physical_oracle_vs_no_rollout_skill") is not None
            and aggregate["physical_oracle_vs_no_rollout_skill"] > args.min_oracle_skill
        ),
        "history_control_matters": (
            (
                aggregate.get("strict_raw_history_advantage") is not None
                and aggregate["strict_raw_history_advantage"] > args.min_history_advantage
            )
            or (
                config.model.v51_history_truncation_enabled
                and aggregate.get("history_truncation_advantage") is not None
                and aggregate["history_truncation_advantage"]
                > args.min_history_truncation_advantage
            )
        ),
        "history_truncation_matters": (
            not config.model.v51_history_truncation_enabled
            or (
                aggregate.get("history_truncation_advantage") is not None
                and aggregate["history_truncation_advantage"]
                > args.min_history_truncation_advantage
            )
        ),
        "horizon_control_matters": (
            aggregate.get("horizon_shuffle_advantage") is not None
            and aggregate["horizon_shuffle_advantage"] > args.min_horizon_advantage
            and all(
                result.get("horizon_shuffle_advantage") is not None
                and result["horizon_shuffle_advantage"] > args.min_horizon_advantage
                for result in horizons.values()
                if int(result["rows"]) > 0
            )
        ),
        "same_order_displacement": (
            aggregate.get("predicted_target_displacement_ratio") is not None
            and args.min_displacement_ratio
            <= aggregate["predicted_target_displacement_ratio"]
            <= args.max_displacement_ratio
        ),
        "exact_active_mode_count": active_error <= 1.0e-5,
        "hard_mode_utilization": utilized_modes >= args.min_active_utilized_modes,
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", "?"),
        "manifest": str(args.manifest),
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "score_space": "V5.1 online physics-supervised dynamic-state feature space plus anchored physical observables; EMA state is diagnostic",
        "history_control": "strict raw context-frame reversal before temporal encoding, endpoint preserved",
        "aggregate": aggregate,
        "promotion_gates": promotion_gates,
        "promotion_pass": all(promotion_gates.values()),
        "routing": {
            "expected_active_modes": expected_active,
            "mode_probability_mean": mode_probability_mean,
            "mode_active_fraction": mode_active_fraction,
            "utilized_modes": utilized_modes,
            "min_mode_active_fraction": args.min_mode_active_fraction,
        },
        "horizons": horizons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-feature-skill", type=float, default=0.0)
    parser.add_argument("--min-physical-skill", type=float, default=0.0)
    parser.add_argument("--min-oracle-skill", type=float, default=0.05)
    parser.add_argument("--min-history-advantage", type=float, default=0.0)
    parser.add_argument("--min-history-truncation-advantage", type=float, default=0.0)
    parser.add_argument("--min-horizon-advantage", type=float, default=0.0)
    parser.add_argument("--min-displacement-ratio", type=float, default=0.1)
    parser.add_argument("--max-displacement-ratio", type=float, default=10.0)
    parser.add_argument("--min-active-utilized-modes", type=int, default=8)
    parser.add_argument("--min-mode-active-fraction", type=float, default=0.05)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
