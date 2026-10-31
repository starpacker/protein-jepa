"""Compare fixed-physical-time LASP microstep paths without optimizer updates.

This diagnostic is versioned separately from the v1 one-step sampler. It uses
one shared fine Brownian path per window and aggregates it for 1/2/4/8
microsteps, while holding anchor-conditioned D/C/Gamma fixed over each path.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evaluation.data import EvalWindow
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from scripts.diagnose_lasp_stability import (
    _as_device,
    _decode,
    _finite_stats,
    _minimum_bond,
    _panel,
    _repeat,
    _target_and_terms,
)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return float(value)


def _coarsen(fine: torch.Tensor, microsteps: int) -> torch.Tensor:
    factor = fine.shape[2] // microsteps
    if fine.shape[2] != microsteps * factor:
        raise ValueError("fine Brownian path length must divide evenly")
    shape = (*fine.shape[:2], microsteps, factor, *fine.shape[3:])
    return fine.reshape(shape).sum(dim=3) / math.sqrt(factor)


def _panel_windows(manifest, temperatures, horizons, max_residues, max_proteins, windows_per_horizon):
    return _panel(manifest, temperatures, horizons, max_residues, max_proteins, windows_per_horizon)


def _static_diagnostics(kernel, seq_feature, temperature, mask, noise_factor=None):
    with torch.no_grad():
        D, C, Gamma = kernel.bath(seq_feature, temperature)
        valid = mask[0]
        K = torch.eye(valid.shape[0], device=valid.device, dtype=seq_feature.dtype)
        if noise_factor is not None:
            K = noise_factor[0] @ noise_factor[0].transpose(-1, -2)
        spectrum = torch.linalg.eigvalsh(K[valid][:, valid].double())
        return {
            "D_min": float(D[:, valid].min()),
            "D_max": float(D[:, valid].max()),
            "Gamma_min": float(Gamma[:, valid].min()) if Gamma is not None else None,
            "Gamma_max": float(Gamma[:, valid].max()) if Gamma is not None else None,
            "C_max_abs": float(C[:, valid].abs().max()) if C is not None else None,
            "step_ps": float(kernel.cfg.step_ps),
            "microstep_dt_ps": None,
            "K_trace": float(spectrum.sum()),
            "K_eigenvalue_min": float(spectrum.min()),
            "K_eigenvalue_max": float(spectrum.max()),
        }


def _run_window(model, window: EvalWindow, n_samples, microstep_values, seed, device, checkpoint_args,
                integrator="explicit", curvature_iterations=0, trace_microstates=False):
    context_coords = _as_device(window.context["coords"], device)[None]
    mask = _as_device(window.context["residue_mask"], device).all(0).bool()[None]
    aa = _as_device(window.context["aatype"], device).long()[None]
    ri = _as_device(window.context["residue_index"], device).long()[None]
    ci = _as_device(window.context["chain_id"], device).long()[None]
    temperature = torch.tensor([float(window.temperature_k)], device=device)
    with torch.no_grad():
        start_one = model.bundle.encode(context_coords[:, -1], mask, aa, ri, ci)
    start = _repeat(start_one, n_samples)
    rm = _repeat(mask, n_samples)
    temp_tensor = _repeat(temperature[:, None], n_samples).flatten()
    anchor_ca = context_coords[:, -1, :, 1]
    noise_factor = None
    if model.noise_mix:
        from protein_jepa_frame.training.train_lasp_ca_dynamics import anchor_noise_factor
        noise_factor = _repeat(
            anchor_noise_factor(anchor_ca, mask, model.noise_mix, model.noise_length), n_samples
        )
    rotation, anchor = _extract_backbone_frames(context_coords[:, -1])
    generator = torch.Generator(device=device).manual_seed(int(seed))
    d_s = model.kernel.cfg.bath_dim
    s0 = torch.randn((n_samples, start.shape[1], d_s), device=device, dtype=start.dtype, generator=generator)
    max_microsteps = max(microstep_values)
    fine_h = torch.randn(
        (n_samples, int(window.horizon), max_microsteps, start.shape[1], start.shape[2]),
        device=device, dtype=start.dtype, generator=generator,
    )
    fine_s = torch.randn(
        (n_samples, int(window.horizon), max_microsteps, start.shape[1], d_s),
        device=device, dtype=start.dtype, generator=generator,
    )

    outputs = {}
    endpoint_tensors = {}
    for microsteps in microstep_values:
        brownian_h = _coarsen(fine_h, microsteps)
        brownian_s = _coarsen(fine_s, microsteps)
        start_for_grad = start.detach().requires_grad_(True)
        bath_for_grad = s0.detach().requires_grad_(True)
        trace = [] if trace_microstates else None
        with sdpa_kernel(SDPBackend.MATH):
            trajectory, bath = model.kernel.rollout_microstep_v1(
                start_for_grad, rm, temp_tensor, int(window.horizon),
                seq_feature=start.detach(), generator=None, S0=bath_for_grad,
                detach=False, return_bath=True, grad_steps=None,
                noise_factor=noise_factor, collective_mobility=True,
                microsteps=microsteps, brownian_H=brownian_h, brownian_S=brownian_s,
                integrator=integrator, microstate_trace=trace,
            )
        h_final = trajectory[:, -1]
        with sdpa_kernel(SDPBackend.MATH):
            raw_ca, ca = _decode(model, h_final, start_one, aa, ri, ci, rotation, anchor, mask)
        terms = _target_and_terms(
            model, window,
            ca.reshape(1, n_samples, ca.shape[-2], 3)[0],
            h_final.reshape(1, n_samples, h_final.shape[-2], h_final.shape[-1])[0],
            mask[0], checkpoint_args, device,
        )
        endpoint = torch.autograd.grad(terms["loss"], h_final, retain_graph=True)[0]
        initial_grad, initial_bath_grad = torch.autograd.grad(
            terms["loss"], (start_for_grad, bath_for_grad))
        endpoint_norm = endpoint.detach().double().norm().clamp_min(1e-30)
        joint_norm = torch.sqrt(initial_grad.detach().double().square().sum()
                                + initial_bath_grad.detach().double().square().sum())
        endpoint_tensors[microsteps] = ca.detach().cpu()
        static = _static_diagnostics(model.kernel, start.detach(), temp_tensor, rm, noise_factor)
        static["microstep_dt_ps"] = float(model.kernel.cfg.step_ps) / microsteps
        outputs[microsteps] = {
            "microsteps": microsteps,
            "terms": {k: float(v.detach()) for k, v in terms.items()},
            "trajectory": _finite_stats(trajectory),
            "bath": _finite_stats(bath),
            "ca_raw": _finite_stats(raw_ca),
            "ca_projected": _finite_stats(ca),
            "min_bond_raw": _minimum_bond(raw_ca, rm, _repeat(ri, n_samples), _repeat(ci, n_samples)),
            "min_bond_projected": _minimum_bond(ca, rm, _repeat(ri, n_samples), _repeat(ci, n_samples)),
            "endpoint_gradient_rms": float(endpoint.detach().square().mean().sqrt()),
            "initial_gradient_rms": float(initial_grad.detach().square().mean().sqrt()),
            "backward_gain": float(initial_grad.detach().double().norm() / endpoint_norm),
            "joint_backward_gain": float(joint_norm / endpoint_norm),
            "gradient_elements_finite": bool(torch.isfinite(initial_grad).all()
                                               and torch.isfinite(initial_bath_grad).all()
                                               and torch.isfinite(endpoint).all()),
            "forward_finite": bool(torch.isfinite(trajectory).all() and torch.isfinite(bath).all()),
            "static": static,
        }
        if trace is not None:
            from scripts.lasp_microstate_vjp import microstate_vjp
            local = microstate_vjp(
                model.kernel, trace, start.detach(), temp_tensor, rm, noise_factor,
                brownian_h, brownian_s, endpoint.detach(), integrator,
                curvature_iterations or 20)
            torch.testing.assert_close(local.pop("initial_H_covector"), initial_grad, rtol=2e-3, atol=1e-7)
            torch.testing.assert_close(local.pop("initial_S_covector"), initial_bath_grad, rtol=2e-3, atol=1e-7)
            outputs[microsteps]["microstate_vjp"] = local
        if curvature_iterations:
            from scripts.lasp_curvature import mobility_curvature
            positions = sorted({0, int(window.horizon) // 2, int(window.horizon)})
            outputs[microsteps]["mobility_curvature"] = [
                {"macro_step": position, **mobility_curvature(
                    model.kernel, trajectory[:, position].detach(), start.detach(),
                    temp_tensor, rm, noise_factor, curvature_iterations)}
                for position in positions]
        del trajectory, bath, h_final, raw_ca, ca
        del terms, endpoint, initial_grad, initial_bath_grad, start_for_grad, bath_for_grad
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference = endpoint_tensors[max_microsteps]
    for microsteps, row in outputs.items():
        delta = endpoint_tensors[microsteps] - reference
        row["endpoint_ca_rms_to_microsteps_max"] = float(delta.square().mean().sqrt())
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--integrator", choices=["explicit", "coupled_midpoint_v1"], default="explicit")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[348., 413.])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--microsteps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--windows-per-horizon", type=int, default=1)
    parser.add_argument("--max-proteins", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=96)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--curvature-iterations", type=int, default=0)
    parser.add_argument("--trace-microstates", action="store_true")
    args = parser.parse_args()
    if sorted(set(args.microsteps)) != args.microsteps or args.microsteps[0] < 1:
        raise ValueError("microsteps must be sorted, unique and positive")
    if any(max(args.microsteps) % count for count in args.microsteps):
        raise ValueError("each microstep count must divide the finest path length")
    device = torch.device(args.device)
    panel = _panel_windows(args.manifest, args.temperatures, args.horizons,
                           args.max_residues, args.max_proteins, args.windows_per_horizon)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    component_dir = output.parent / "microstep_components"
    component_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "development fixed-physical-time microstep no-update diagnostic",
        "protocol": {
            "integrator": args.integrator, "curvature_iterations": args.curvature_iterations,
            "trace_microstates": args.trace_microstates,
            "manifest": args.manifest, "temperatures": args.temperatures,
            "horizons": args.horizons, "microsteps": args.microsteps,
            "windows_per_horizon": args.windows_per_horizon,
            "n_samples": args.n_samples, "shared_fine_brownian": True,
            "anchor_fixed_mobility": True, "optimizer_steps": 0,
            "test_opened": False,
        },
        "panel": [dict(protein=w.protein_id, trajectory=w.trajectory_id,
                         temperature_k=float(w.temperature_k), anchor_frame=int(w.anchor_frame),
                         horizon=int(w.horizon)) for w in panel],
        "checkpoints": [],
    }
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        component_path = component_dir / f"seed{int(checkpoint['args']['seed'])}.component.pt"
        torch.save(checkpoint["component_checkpoint"], component_path)
        model = CADynamics(
            str(component_path), device, checkpoint_path,
            history_conditioning=checkpoint["args"].get("history_conditioning", False),
            noise_mix=checkpoint["args"].get("noise_mix", 0.),
            noise_length=checkpoint["args"].get("noise_length", 8.),
            collective_mobility=checkpoint["args"].get("collective_mobility", False),
        ).to(device).eval()
        checkpoint_result = {"seed": int(checkpoint["args"]["seed"]), "checkpoint": checkpoint_path, "windows": []}
        result["checkpoints"].append(checkpoint_result)
        for index, window in enumerate(panel):
            window_seed = (args.seed * 1000003 + int.from_bytes(
                f"{window.protein_id}|{window.trajectory_id}|{window.anchor_frame}|{window.horizon}|{window.temperature_k}".encode(),
                "little")) % (2**31 - 1)
            modes = _run_window(model, window, args.n_samples, args.microsteps,
                                window_seed, device, type("Args", (), checkpoint["args"])(), args.integrator,
                                args.curvature_iterations, args.trace_microstates)
            checkpoint_result["windows"].append({
                "index": index, "identity": result["panel"][index], "modes": _jsonable(modes),
            })
            output.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(f"seed={checkpoint_result['seed']} window={index + 1}/{len(panel)} "
                  f"{window.temperature_k:g}K h={window.horizon}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(output)


if __name__ == "__main__":
    main()
