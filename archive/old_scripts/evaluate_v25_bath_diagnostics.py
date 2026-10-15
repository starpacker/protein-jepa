#!/usr/bin/env python3
"""Numerical diagnostics for V25 multi-timescale Mori bath.

No figures; this emits table-ready values:

* learned fast/mid/slow relaxation times;
* coupling matrix norms by group;
* memory-kernel Frobenius norm ||K(t)||_F for K(t)=C exp(-t/tau) C^T;
* history-inferred bath RMS by group on paired-window batches.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


def _stats(x: np.ndarray) -> dict:
    return {
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
    }


def _kernel_norms(taus: np.ndarray, C: np.ndarray, times: list[float]) -> dict:
    out = {}
    K0 = C @ C.T
    K0_norm = max(float(np.linalg.norm(K0)), 1.0e-30)
    for t in times:
        decay = np.exp(-float(t) / np.maximum(taus, 1.0e-30))
        K = C @ np.diag(decay) @ C.T
        norm = float(np.linalg.norm(K))
        out[str(t)] = {
            "fro_norm": norm,
            "relative_to_t0": norm / K0_norm,
            "trace": float(np.trace(K)),
            "spectral_norm": float(np.linalg.norm(K, ord=2)),
        }
    return out


def _collect_bath_states(model: ProteinJEPA, config, device: torch.device, max_batches: int) -> np.ndarray:
    dataset = build_dataset(config.data)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=True,
    )
    rows = []
    model.eval()
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            out = model.forward_paired(batch)
            hist = out["lspwm_history_features"]
            bath = model.lspwm.history_to_bath(hist).detach().cpu().numpy()
        rows.append(bath)
    return np.concatenate(rows, axis=0)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    model.load_state_dict(state, strict=False)
    if model.lspwm is None or not hasattr(model.lspwm, "multi_bath"):
        raise RuntimeError("checkpoint does not expose lspwm.multi_bath")

    taus_t, C_t = model.lspwm.multi_bath.parameters_at(dtype=torch.float64, device=device)
    taus = taus_t.detach().cpu().numpy()
    C = C_t.detach().cpu().numpy()
    group_sizes = tuple(int(x) for x in model.lspwm.multi_bath.group_sizes)
    names = ["fast", "mid", "slow"]
    slices = {}
    start = 0
    for name, size in zip(names, group_sizes):
        slices[name] = slice(start, start + size)
        start += size

    groups = {}
    for name, sl in slices.items():
        Cg = C[:, sl]
        groups[name] = {
            "n_dim": int(sl.stop - sl.start),
            "tau_ps": _stats(taus[sl]),
            "tau_ns": _stats(taus[sl] / 1000.0),
            "coupling_fro_norm": float(np.linalg.norm(Cg)),
            "coupling_spectral_norm": float(np.linalg.norm(Cg, ord=2)),
        }

    bath = _collect_bath_states(model, config, device, args.max_batches)
    bath_groups = {}
    for name, sl in slices.items():
        bg = bath[:, sl]
        bath_groups[name] = {
            "rms": float(np.sqrt(np.mean(bg * bg))),
            "mean_abs": float(np.mean(np.abs(bg))),
            "std": float(np.std(bg)),
        }

    times = args.times_ps
    kernel = _kernel_norms(taus, C, times)
    result = {
        "experiment_id": "V25_bath_memory_diagnostics",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "max_batches_for_history_bath": args.max_batches,
            "n_bath_samples": int(len(bath)),
            "kernel_times_ps": times,
        },
        "groups": groups,
        "history_inferred_bath_groups": bath_groups,
        "kernel_norms": kernel,
        "all_tau_ps": [float(x) for x in taus],
        "all_tau_ns": [float(x / 1000.0) for x in taus],
        "claim_boundary": (
            "Numerical bath diagnostics only. These show learned timescales and "
            "kernel decay, but do not by themselves prove unique physical identifiability."
        ),
    }
    return result


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# V25 Bath Memory Diagnostics",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"n bath samples: {result['protocol']['n_bath_samples']}",
        "",
        "## Learned Timescales And Couplings",
        "",
        "| Group | dims | tau ns min | tau ns mean | tau ns max | coupling Fro norm | bath RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["fast", "mid", "slow"]:
        g = result["groups"][name]
        b = result["history_inferred_bath_groups"][name]
        lines.append(
            f"| {name} | {g['n_dim']} | {g['tau_ns']['min']:.4f} | "
            f"{g['tau_ns']['mean']:.4f} | {g['tau_ns']['max']:.4f} | "
            f"{g['coupling_fro_norm']:.4f} | {b['rms']:.4f} |"
        )
    lines.extend([
        "",
        "## Memory Kernel Decay",
        "",
        "| t ps | ||K(t)||F | relative to K(0) | trace | spectral norm |",
        "|---:|---:|---:|---:|---:|",
    ])
    for t, row in result["kernel_norms"].items():
        lines.append(
            f"| {float(t):.1f} | {row['fro_norm']:.6g} | "
            f"{row['relative_to_t0']:.6g} | {row['trace']:.6g} | {row['spectral_norm']:.6g} |"
        )
    lines.extend(["", result["claim_boundary"]])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate V25 bath memory diagnostics")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument(
        "--times-ps",
        type=float,
        nargs="+",
        default=[0.0, 1000.0, 4000.0, 16000.0, 64000.0, 128000.0],
    )
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
        "tau_ns_mean_by_group": {
            k: v["tau_ns"]["mean"] for k, v in result["groups"].items()
        },
        "kernel_relative_64ns": result["kernel_norms"].get("64000.0", {}).get("relative_to_t0"),
    }, indent=2))


if __name__ == "__main__":
    main()
