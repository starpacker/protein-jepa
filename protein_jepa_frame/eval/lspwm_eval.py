#!/usr/bin/env python3
"""Comprehensive LSP-WM evaluation suite for protein dynamics.

Evaluates the world model on multiple physical fidelity axes:

  1.  Deterministic & ensemble endpoint skill
  2.  Force matching accuracy
  3.  Energy conservation (loop test)
  4.  VAMP-2 / Koopman diagnostics
  5.  LSPWM component diagnostics

Usage:
  python3 -m protein_jepa_frame.eval.lspwm_eval \\
      --checkpoint <path> \\
      --config <config-path> \\
      --output-dir <eval-dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from protein_jepa_frame.config import TrainConfig, load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA
from protein_jepa_frame.models.lspwm import (
    LSPWMWorldModel,
    LSPWMProteinWorldModel,
    MeanForcePotential,
    kde_mean_force,
    mean_force_matching_loss,
    vamp2_loss,
    koopman_diagnostics,
    empirical_fdt_residual,
    energy_loop_circulation,
)



def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_checkpoint(path: Path, model: ProteinJEPA, device: torch.device, strict: bool = False):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    # Apply the same upgrade/migration as training's _load_init_checkpoint
    from protein_jepa_frame.models.jepa import upgrade_lspwm_state_dict
    state = upgrade_lspwm_state_dict(model, state)
    incompatible = model.load_state_dict(state, strict=strict)
    info = {"epoch": ckpt.get("epoch", "?"), "step": ckpt.get("step", "?")}
    print(f"  Loaded checkpoint: epoch={info['epoch']}, step={info['step']}")
    if incompatible.missing_keys:
        print(f"    missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"    unexpected keys: {incompatible.unexpected_keys}")
    return model


# ===================================================================
#  1. Endpoint skill metrics
# ===================================================================

@torch.no_grad()
def eval_endpoint_skill(
    model: ProteinJEPA,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """Evaluate deterministic and ensemble endpoint prediction skill."""
    model.eval()
    if model.lspwm is None:
        return {"error": "LSP-WM not enabled"}

    all_delta_pred = []
    all_delta_target = []
    all_ensemble_pred = []
    all_horizons = []

    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model.forward_paired(batch)

        delta_target = outputs.get("delta_target")
        delta_pred = outputs.get("delta_prediction")
        if delta_target is not None and delta_pred is not None:
            all_delta_pred.append(delta_pred.cpu())
            all_delta_target.append(delta_target.cpu())
            all_horizons.append(batch.get("horizon_frames", batch.get("horizon_time_ps")).cpu())

        lspwm_rollout = outputs.get("lspwm_rollout_trajectory")
        if lspwm_rollout is not None:
            ensemble_endpoint = lspwm_rollout[:, :, -1, :]
            all_ensemble_pred.append(ensemble_endpoint.cpu())

        n_batches += 1

    if not all_delta_pred:
        return {"error": "no predictions collected"}

    delta_pred = torch.cat(all_delta_pred, dim=0)
    delta_target = torch.cat(all_delta_target, dim=0)
    horizons = torch.cat(all_horizons, dim=0) if all_horizons else None

    mse = (delta_pred - delta_target).square().mean().item()
    persistence_mse = delta_target.square().mean().item()
    skill = 1.0 - mse / max(persistence_mse, 1e-12)

    per_horizon = {}
    if horizons is not None:
        unique_h = torch.unique(horizons)
        for h in unique_h.cpu().tolist():
            mask = (horizons == h)
            h_pred = delta_pred[mask]
            h_target = delta_target[mask]
            h_mse = (h_pred - h_target).square().mean().item()
            h_pers = h_target.square().mean().item()
            per_horizon[str(int(h))] = {
                "mse": h_mse,
                "skill": 1.0 - h_mse / max(h_pers, 1e-12),
                "n": int(mask.sum().item()),
            }

    results = {
        "deterministic_mse": mse,
        "deterministic_persistence_mse": persistence_mse,
        "deterministic_skill": skill,
        "n_predictions": delta_pred.shape[0],
        "per_horizon": per_horizon,
    }

    if all_ensemble_pred:
        ensemble = torch.cat(all_ensemble_pred, dim=0)
        ensemble_mean = ensemble.mean(dim=1)
        ensemble_var = ensemble.var(dim=1, unbiased=False).mean(dim=-1)
        ensemble_mse = (ensemble_mean.cpu() - delta_target).square().mean().item()
        results["ensemble_mse"] = ensemble_mse
        results["ensemble_mean_var"] = ensemble_var.mean().item()
        results["ensemble_calibration_ratio"] = ensemble_mse / max(ensemble_var.mean().item(), 1e-12)

    return results



# ===================================================================
#  2. Force matching evaluation
# ===================================================================

@torch.no_grad()
def eval_force_matching(
    model: ProteinJEPA,
    loader: DataLoader,
    device: torch.device,
    bandwidth: float = 0.1,
    max_batches: int = 20,
) -> dict:
    """Evaluate PMF mean-force gradient fidelity against KDE target."""
    model.eval()
    if model.lspwm is None:
        return {"error": "LSP-WM not enabled"}

    world_model = model.lspwm.world_model
    potential = world_model.potential

    all_force_pred = []
    all_force_target = []
    all_force_cosine = []

    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model.forward_paired(batch)

        context_q = outputs.get("lspwm_context_state")
        if context_q is None:
            continue

        # Use full context sequence as KDE reference (all valid latent states)
        context_sequence = outputs.get("lspwm_context_sequence")
        context_mask = outputs.get("lspwm_context_mask")
        if context_sequence is not None and context_mask is not None:
            kde_reference = context_sequence[context_mask].detach()
        else:
            kde_reference = context_q.detach()

        condition = outputs.get("lspwm_condition")
        temperature_k = batch.get("temperature_k")

        q_flat = context_q.reshape(-1, context_q.shape[-1])
        condition_flat = condition.reshape(-1, condition.shape[-1]) if condition is not None else None
        temp_flat = temperature_k.reshape(-1) if temperature_k is not None else None

        if q_flat.shape[0] < 2 or kde_reference.shape[0] < 2:
            continue

        force_target = kde_mean_force(q_flat, kde_reference, bandwidth=bandwidth)
        force_pred = potential.mean_force(q_flat, condition_flat, temp_flat, create_graph=False)

        mse = (force_pred - force_target).square().mean().item()
        force_norm = force_pred.norm(dim=-1).mean().item()
        target_norm = force_target.norm(dim=-1).mean().item()
        cosine = F.cosine_similarity(force_pred, force_target, dim=-1).mean().item()

        all_force_pred.append(force_pred.cpu())
        all_force_target.append(force_target.cpu())
        all_force_cosine.append(cosine)

        n_batches += 1

    if not all_force_pred:
        return {"error": "no force data collected"}

    force_pred = torch.cat(all_force_pred, dim=0)
    force_target = torch.cat(all_force_target, dim=0)

    n_samples = min(1000, force_pred.shape[0])
    indices = torch.randperm(force_pred.shape[0])[:n_samples]
    f_pred = force_pred[indices]
    f_target = force_target[indices]

    cosine_mean = np.mean(all_force_cosine)
    mse = (f_pred - f_target).square().mean().item()
    norm_pred = f_pred.norm(dim=-1).mean().item()
    norm_target = f_target.norm(dim=-1).mean().item()

    return {
        "force_mse": mse,
        "force_cosine": cosine_mean,
        "force_pred_norm": norm_pred,
        "force_target_norm": norm_target,
        "n_force_samples": force_pred.shape[0],
    }


# ===================================================================
#  3. Energy conservation (loop circulation)
# ===================================================================

@torch.no_grad()
def eval_energy_conservation(
    model: ProteinJEPA,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Check PMF conservativeness via numerical loop circulation."""
    model.eval()
    if model.lspwm is None:
        return {"error": "LSP-WM not enabled"}

    world_model = model.lspwm.world_model
    potential = world_model.potential
    circulations = []

    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model.forward_paired(batch)

        context_q = outputs.get("lspwm_context_state")
        condition = outputs.get("lspwm_condition")
        temperature_k = batch.get("temperature_k")

        if context_q is None or context_q.shape[0] < 2:
            continue

        q_flat = context_q.reshape(-1, context_q.shape[-1])
        condition_flat = condition.reshape(-1, condition.shape[-1]) if condition is not None else None

        for _ in range(min(5, q_flat.shape[0] // 2)):
            idx = torch.randperm(q_flat.shape[0])[:4]
            loop_q = q_flat[idx]
            loop_cond = condition_flat[idx] if condition_flat is not None else None

            try:
                circulation = energy_loop_circulation(potential, loop_q, loop_cond)
                circulations.append(circulation.item())
            except Exception:
                continue

        n_batches += 1

    if not circulations:
        return {"error": "no circulation data"}

    return {
        "mean_circulation": float(np.mean(circulations)),
        "max_circulation": float(np.max(circulations)),
        "std_circulation": float(np.std(circulations)),
        "n_loops": len(circulations),
    }



# ===================================================================
#  4. VAMP-2 / Koopman diagnostics
# ===================================================================

@torch.no_grad()
def eval_koopman(
    model: ProteinJEPA,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """Evaluate VAMP-2 scores and Koopman implied timescales."""
    model.eval()
    if model.lspwm is None:
        return {"error": "LSP-WM not enabled"}

    all_source = []
    all_target = []

    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model.forward_paired(batch)

        context_q = outputs.get("lspwm_context_state")
        future_q = outputs.get("lspwm_target_state")

        if context_q is not None and future_q is not None:
            all_source.append(context_q.cpu())
            all_target.append(future_q.cpu())

        n_batches += 1

    if not all_source:
        return {"error": "no trajectory data"}

    source = torch.cat(all_source, dim=0)
    target = torch.cat(all_target, dim=0)

    n = source.shape[0]
    if n > 5000:
        idx = torch.randperm(n)[:5000]
        source = source[idx]
        target = target[idx]

    vamp_score = vamp2_loss(source, target, ridge=1e-3).item()
    vamp_low = vamp2_loss(source[:, :min(16, source.shape[-1])],
                          target[:, :min(16, target.shape[-1])],
                          ridge=1e-3).item()

    koopman = koopman_diagnostics(source, target, lag_time=1.0, ridge=1e-5)
    implied_timescales = koopman["implied_timescales"]

    timescales = implied_timescales[~torch.isnan(implied_timescales) &
                                     ~torch.isinf(implied_timescales)]
    timescales = timescales[timescales > 0]
    top_timescales = timescales[:min(10, len(timescales))].tolist() if len(timescales) > 0 else []

    return {
        "vamp2_score": -vamp_score,
        "vamp2_score_lowd": -vamp_low,
        "koopman_residual_mse": koopman["residual_mse"].item(),
        "implied_timescales_top10": top_timescales,
        "n_samples": n,
    }


# ===================================================================
#  5. Component diagnostics
# ===================================================================

@torch.no_grad()
def eval_lspwm_components(model: ProteinJEPA, device: torch.device) -> dict:
    """Diagnose LSPWM component health."""
    if model.lspwm is None:
        return {"error": "LSP-WM not enabled"}

    lspwm = model.lspwm
    wm = lspwm.world_model
    diag = {}

    if hasattr(wm, 'potential') and wm.potential is not None:
        pot = wm.potential
        diag["potential_params"] = sum(p.numel() for p in pot.parameters())
        depth = 0
        for m in pot.network.modules():
            if isinstance(m, torch.nn.Linear):
                depth += 1
        diag["potential_depth"] = depth

    if hasattr(wm, 'diffusion') and wm.diffusion is not None:
        diag["diffusion_params"] = sum(p.numel() for p in wm.diffusion.parameters())
        diag["diffusion_state_dependent"] = hasattr(wm.diffusion, 'net')

    if hasattr(wm, 'bath') and wm.bath is not None:
        diag["bath_params"] = sum(p.numel() for p in wm.bath.parameters())
        diag["auxiliary_dim"] = wm.bath.auxiliary_dim

    diag["state_dim"] = wm.potential.state_dim if hasattr(wm.potential, 'state_dim') else lspwm.state_dim
    diag["condition_dim"] = wm.potential.condition_dim if hasattr(wm.potential, 'condition_dim') else wm.potential.state_dim

    # Quick forward pass
    dummy_q = torch.randn(4, diag["state_dim"], device=device)
    dummy_cond = torch.randn(4, diag["condition_dim"], device=device) if diag["condition_dim"] > 0 else None

    try:
        energy = wm.potential(dummy_q, dummy_cond)
        diag["energy_healthy"] = bool(torch.isfinite(energy).all())
        diag["energy_mean"] = energy.mean().item()
        diag["energy_std"] = energy.std().item()

        force = wm.potential.mean_force(dummy_q, dummy_cond, create_graph=False)
        diag["force_healthy"] = bool(torch.isfinite(force).all())
        diag["force_norm"] = force.norm(dim=-1).mean().item()

        diff_matrix = wm.diffusion(reference=dummy_q)
        diag["diffusion_healthy"] = bool(torch.isfinite(diff_matrix).all())
        diag["diffusion_mean_diag"] = torch.diagonal(diff_matrix, dim1=-2, dim2=-1).mean().item()
    except Exception as e:
        diag["diagnostic_error"] = str(e)

    return diag



# ===================================================================
#  6. Comprehensive evaluation
# ===================================================================

def evaluate_lspwm(
    checkpoint_path: Path,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    max_batches: int = 50,
    bandwidth: float = 0.1,
) -> dict:
    """Run all LSP-WM evaluations and write results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ProteinJEPA(config.model).to(device)
    _load_checkpoint(checkpoint_path, model, device, strict=False)
    model.eval()

    dataset = build_dataset(config.data)
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_paired_windows,
        drop_last=False,
    )

    results = {"checkpoint": str(checkpoint_path)}

    print("\n[1/5] Endpoint skill evaluation...")
    sys.stdout.flush()
    try:
        skill = eval_endpoint_skill(model, loader, device, max_batches=max_batches)
        results["endpoint_skill"] = skill
        print(f"  Deterministic skill: {skill.get('deterministic_skill', 'N/A'):.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["endpoint_skill"] = {"error": str(e)}

    print("\n[2/5] Force matching evaluation...")
    sys.stdout.flush()
    try:
        force = eval_force_matching(model, loader, device, bandwidth=bandwidth,
                                     max_batches=max_batches // 2)
        results["force_matching"] = force
        print(f"  Force MSE: {force.get('force_mse', 'N/A'):.6f}")
        print(f"  Force cosine: {force.get('force_cosine', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["force_matching"] = {"error": str(e)}

    print("\n[3/5] Energy conservation (loop test)...")
    sys.stdout.flush()
    try:
        energy = eval_energy_conservation(model, loader, device,
                                           max_batches=max_batches // 5)
        results["energy_conservation"] = energy
        print(f"  Mean circulation: {energy.get('mean_circulation', 'N/A'):.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["energy_conservation"] = {"error": str(e)}

    print("\n[4/5] Koopman / VAMP-2 evaluation...")
    sys.stdout.flush()
    try:
        koopman = eval_koopman(model, loader, device, max_batches=max_batches)
        results["koopman"] = koopman
        print(f"  VAMP-2 score: {koopman.get('vamp2_score', 'N/A'):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["koopman"] = {"error": str(e)}

    print("\n[5/5] LSPWM component diagnostics...")
    sys.stdout.flush()
    try:
        diag = eval_lspwm_components(model, device)
        results["components"] = diag
        print(f"  Potential params: {diag.get('potential_params', 'N/A')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["components"] = {"error": str(e)}

    results_path = output_dir / "lspwm_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="LSP-WM comprehensive evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, help="Path to training config YAML")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = _device(args.device)
    checkpoint = Path(args.checkpoint)

    if args.config:
        config = load_config(args.config)
    else:
        config_path = checkpoint.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                raw = json.load(f)
            config = TrainConfig(**raw)
        else:
            raise ValueError("Must provide --config or have config.json beside checkpoint")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent / "eval"

    print(f"Evaluating LSP-WM from {checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Max batches: {args.max_batches}")
    print(f"KDE bandwidth: {args.bandwidth}")
    print()

    results = evaluate_lspwm(
        checkpoint, config, device, output_dir,
        max_batches=args.max_batches, bandwidth=args.bandwidth,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    skill = results.get("endpoint_skill", {})
    if "deterministic_skill" in skill:
        print(f"  Deterministic skill:     {skill['deterministic_skill']:.6f}")
    force = results.get("force_matching", {})
    if "force_cosine" in force:
        print(f"  Force cosine similarity:  {force['force_cosine']:.4f}")
    if "force_mse" in force:
        print(f"  Force MSE:                {force['force_mse']:.6f}")
    koop = results.get("koopman", {})
    if "vamp2_score" in koop:
        print(f"  VAMP-2 score:             {koop['vamp2_score']:.4f}")
    energy = results.get("energy_conservation", {})
    if "mean_circulation" in energy:
        print(f"  Mean loop circulation:    {energy['mean_circulation']:.6f}")


if __name__ == "__main__":
    main()
