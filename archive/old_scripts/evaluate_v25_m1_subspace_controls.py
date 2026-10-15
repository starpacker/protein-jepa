#!/usr/bin/env python3
"""V25 M1 subspace controls under the original paired-window protocol.

This supplement keeps the existing V25 M1 directional/subspace diagnostic, but
adds time-alignment controls for the memory direction. It is deliberately not
the strict temporal-holdout evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v25_m1_fair import (
    _cosine,
    _paired_bootstrap,
    _per_sample_skill,
    _single_direction_projection,
    _two_direction_projection,
)


EPS = 1.0e-12


def _collect_arrays(model: ProteinJEPA, loader: DataLoader, device: torch.device, max_batches: int) -> dict:
    rows = {"current": [], "future": [], "force": [], "memory": [], "dt": [], "horizon": []}
    model.eval()
    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_paired(batch)
        required = [
            "lspwm_m1_global_current",
            "lspwm_m1_global_future",
            "lspwm_m1_dissipative_drift",
            "lspwm_m1_bath_drift",
            "lspwm_m1_dt",
        ]
        if any(out.get(k) is None for k in required):
            n_batches += 1
            continue
        rows["current"].append(out["lspwm_m1_global_current"].detach().cpu().numpy())
        rows["future"].append(out["lspwm_m1_global_future"].detach().cpu().numpy())
        rows["force"].append(out["lspwm_m1_dissipative_drift"].detach().cpu().numpy())
        rows["memory"].append(out["lspwm_m1_bath_drift"].detach().cpu().numpy())
        rows["dt"].append(out["lspwm_m1_dt"].detach().cpu().numpy().reshape(-1))
        rows["horizon"].append(out["horizon_frames"].detach().cpu().numpy().reshape(-1))
        n_batches += 1
    if not rows["current"]:
        raise RuntimeError("no valid V25 M1 batches were collected")
    return {k: np.concatenate(v, axis=0) for k, v in rows.items()}


def _random_subspace_expected(delta: np.ndarray, seed: int, draws: int) -> dict:
    rng = np.random.default_rng(seed)
    n, d = delta.shape
    one = []
    two = []
    for _ in range(draws):
        r1 = rng.normal(size=(n, d))
        r2a = rng.normal(size=(n, d))
        r2b = rng.normal(size=(n, d))
        one.append(_per_sample_skill(delta, _single_direction_projection(delta, r1)))
        two.append(_per_sample_skill(delta, _two_direction_projection(delta, r2a, r2b)))
    return {
        "random_1d_expected": _paired_bootstrap(np.stack(one, axis=0).mean(axis=0)),
        "random_2d_expected": _paired_bootstrap(np.stack(two, axis=0).mean(axis=0)),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    dataset = build_dataset(config.data)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
        generator=generator,
    )
    arrays = _collect_arrays(model, loader, device, args.max_batches)
    delta = arrays["future"] - arrays["current"]
    force = arrays["force"].astype(np.float64)
    memory = arrays["memory"].astype(np.float64)
    rng = np.random.default_rng(args.seed + 222)
    perm = rng.permutation(len(memory))
    memory_shuffled = memory[perm]
    force_shuffled = force[perm]
    memory_random = rng.normal(
        loc=memory.mean(axis=0, keepdims=True),
        scale=np.maximum(memory.std(axis=0, keepdims=True), 1.0e-6),
        size=memory.shape,
    )
    force_random = rng.normal(
        loc=force.mean(axis=0, keepdims=True),
        scale=np.maximum(force.std(axis=0, keepdims=True), 1.0e-6),
        size=force.shape,
    )

    projections = {
        "force_1d_correct": _single_direction_projection(delta, force),
        "memory_1d_correct": _single_direction_projection(delta, memory),
        "memory_1d_shuffled": _single_direction_projection(delta, memory_shuffled),
        "memory_1d_random_matched": _single_direction_projection(delta, memory_random),
        "full_2d_correct_force_memory": _two_direction_projection(delta, force, memory),
        "force_plus_shuffled_memory": _two_direction_projection(delta, force, memory_shuffled),
        "force_plus_random_memory": _two_direction_projection(delta, force, memory_random),
        "shuffled_force_plus_memory": _two_direction_projection(delta, force_shuffled, memory),
        "random_force_plus_memory": _two_direction_projection(delta, force_random, memory),
    }
    skills = {name: _per_sample_skill(delta, pred) for name, pred in projections.items()}
    random_expected = _random_subspace_expected(delta, args.seed + 707, args.random_draws)
    differences = {
        "correct_memory_minus_shuffled_memory_1d": skills["memory_1d_correct"] - skills["memory_1d_shuffled"],
        "correct_memory_minus_random_memory_1d": skills["memory_1d_correct"] - skills["memory_1d_random_matched"],
        "correct_full_minus_force_plus_shuffled_memory": (
            skills["full_2d_correct_force_memory"] - skills["force_plus_shuffled_memory"]
        ),
        "correct_full_minus_force_plus_random_memory": (
            skills["full_2d_correct_force_memory"] - skills["force_plus_random_memory"]
        ),
        "correct_full_minus_shuffled_force_plus_memory": (
            skills["full_2d_correct_force_memory"] - skills["shuffled_force_plus_memory"]
        ),
        "correct_full_minus_random_force_plus_memory": (
            skills["full_2d_correct_force_memory"] - skills["random_force_plus_memory"]
        ),
    }
    return {
        "experiment_id": "V25_M1_subspace_time_alignment_controls",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "existing V25 paired-window M1 subspace diagnostic",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "seed": args.seed,
            "n_pairs": int(delta.shape[0]),
            "cv_dim": int(delta.shape[1]),
            "uses_m1_strict": False,
        },
        "skills": {name: _paired_bootstrap(value) for name, value in skills.items()},
        "differences": {name: _paired_bootstrap(value) for name, value in differences.items()},
        "random_expected": random_expected,
        "force_memory_cosine": _paired_bootstrap(_cosine(force, memory)),
        "claim_boundary": (
            "This is a decomposition subspace diagnostic, not a calibrated drift-vector metric. "
            "The controls test whether the correct time-aligned memory direction is more informative "
            "than shuffled or distribution-matched random directions under the same 2D span capacity."
        ),
    }


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# V25 M1 Subspace Time-Alignment Controls",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"seed: {result['protocol']['seed']}",
        "",
        "## Subspace Skills",
        "",
        "| Diagnostic | mean | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, stats in result["skills"].items():
        lines.append(f"| {name} | {stats['mean']:.4f} | [{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}] |")
    for name, stats in result["random_expected"].items():
        lines.append(f"| {name} | {stats['mean']:.4f} | [{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}] |")
    lines.extend([
        "",
        "## Time-Alignment Differences",
        "",
        "| Difference | mean | 95% CI |",
        "|---|---:|---:|",
    ])
    for name, stats in result["differences"].items():
        lines.append(f"| {name} | {stats['mean']:.4f} | [{stats['ci_lo']:.4f}, {stats['ci_hi']:.4f}] |")
    cos = result["force_memory_cosine"]
    lines.extend([
        "",
        f"Force-memory cosine: {cos['mean']:.4f} [{cos['ci_lo']:.4f}, {cos['ci_hi']:.4f}].",
        "",
        result["claim_boundary"],
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate V25 M1 subspace time-alignment controls")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--random-draws", type=int, default=256)
    args = p.parse_args()
    result = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        write_summary(result, Path(args.summary))
    print(json.dumps({
        "output": str(out),
        "summary": args.summary,
        "n_pairs": result["protocol"]["n_pairs"],
        "full_correct": result["skills"]["full_2d_correct_force_memory"]["mean"],
        "correct_full_minus_force_plus_shuffled_memory": result["differences"]["correct_full_minus_force_plus_shuffled_memory"]["mean"],
        "correct_full_minus_force_plus_random_memory": result["differences"]["correct_full_minus_force_plus_random_memory"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
