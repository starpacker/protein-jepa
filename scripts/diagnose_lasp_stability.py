"""Fixed-path, no-update diagnostics for LASP rollout backward stability.

This is a development-only probe. It restores frozen stopped-best checkpoints,
uses deterministic validation windows and shared per-window noise, and never
calls an optimizer. It compares full BPTT with truncated backward windows while
keeping forward samples identical. The output is diagnostic evidence, not a
training result or a forecast benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import math

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evaluation.data import EvalProtocol, EvalWindow, iter_windows
from evaluation.ensemble_metrics_ext import _align_to
from protein_jepa_frame.models.ca_geometry import project_ca_bonds
from protein_jepa_frame.training.train_lasp_ca_dynamics import (
    CADynamics,
    contact_probability_loss,
    energy_score,
    slow_mode_distribution_loss,
    slow_mode_scale_loss,
)
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames


ENSEMBLE_GRAD_MODES = (64, 8, 1)


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _as_device(value, device):
    return torch.as_tensor(value, device=device)


def _repeat(x: torch.Tensor, n: int) -> torch.Tensor:
    return x[:, None].expand(x.shape[0], n, *x.shape[1:]).reshape(
        x.shape[0] * n, *x.shape[1:]
    )


def _finite_stats(values: torch.Tensor) -> dict[str, object]:
    x = values.detach()
    finite = torch.isfinite(x)
    result = {
        "finite": bool(finite.all()),
        "finite_fraction": float(finite.float().mean()),
        "max_abs": _safe_float(torch.nan_to_num(x.abs(), nan=float("inf")).max()),
        "rms": _safe_float(torch.nan_to_num(x.square().mean().sqrt(), nan=float("inf"))),
    }
    if finite.any():
        result["min"] = _safe_float(x[finite].min())
        result["max"] = _safe_float(x[finite].max())
    else:
        result["min"] = None
        result["max"] = None
    return result


def _gradient_stats(parameters, gradients) -> dict[str, object]:
    entries = []
    total_sq64 = 0.0
    total_sq32 = torch.zeros((), dtype=torch.float32)
    total_elements = 0
    finite_elements = 0
    groups = defaultdict(lambda: {"sq64": 0.0, "sq32": torch.zeros((), dtype=torch.float32), "max_abs": 0.0})
    for (name, parameter), grad in zip(parameters, gradients):
        if grad is None:
            continue
        g = grad.detach()
        finite = torch.isfinite(g)
        finite_sq64 = torch.where(finite, g.to(torch.float64).square(), torch.zeros_like(g, dtype=torch.float64)).sum()
        # This deliberately mirrors the vulnerable float32 accumulation domain.
        sq32 = g.square().sum()
        total_elements += g.numel()
        finite_elements += int(finite.sum().item())
        total_sq64 += float(finite_sq64)
        total_sq32 = total_sq32 + sq32.cpu()
        group = name.split(".", 1)[0]
        groups[group]["sq64"] += float(finite_sq64)
        groups[group]["sq32"] = groups[group]["sq32"] + sq32.cpu()
        max_abs = _safe_float(torch.nan_to_num(g.abs(), nan=float("inf")).max())
        groups[group]["max_abs"] = max(groups[group]["max_abs"], max_abs or 0.0)
        entries.append(
            dict(
                name=name,
                numel=g.numel(),
                finite=bool(finite.all()),
                finite_fraction=float(finite.float().mean()),
                max_abs=max_abs,
                norm64=_safe_float(finite_sq64.sqrt()),
                norm32=_safe_float(g.norm()),
            )
        )
    group_stats = {
        name: dict(norm64=_safe_float(math.sqrt(v["sq64"])), norm32=_safe_float(v["sq32"].sqrt()), max_abs=v["max_abs"])
        for name, v in groups.items()
    }
    return dict(
        norm64=_safe_float(math.sqrt(total_sq64)),
        norm32=_safe_float(total_sq32.sqrt()),
        norm32_sq=_safe_float(total_sq32),
        finite=finite_elements == total_elements,
        finite_fraction=(finite_elements / total_elements if total_elements else 1.0),
        max_abs=max((entry["max_abs"] or 0.0 for entry in entries), default=0.0),
        parameter_groups=group_stats,
        parameters=entries,
    )


def _bond_mask(mask, residue_index, chain_id):
    return (
        mask[:, 1:]
        & mask[:, :-1]
        & (chain_id[:, 1:] == chain_id[:, :-1])
        & ((residue_index[:, 1:] - residue_index[:, :-1]) == 1)
    )


def _minimum_bond(ca, mask, residue_index, chain_id):
    valid = _bond_mask(mask, residue_index, chain_id)
    distance = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)
    if not valid.any():
        return None
    return float(distance[valid].detach().min())


def _decode(model, h_final, start_one, aa, ri, ci, rotation, anchor, mask):
    n = h_final.shape[0] // start_one.shape[0]
    start = _repeat(start_one, n)
    raw = model.bundle.decoder(
        h_final,
        start,
        _repeat(aa, n),
        _repeat(ri, n),
        _repeat(ci, n),
        _repeat(rotation, n),
        _repeat(anchor, n),
        _repeat(mask, n),
    )
    projected = raw
    if model.bundle.projection_steps:
        projected = project_ca_bonds(
            raw,
            _repeat(anchor, n),
            _repeat(mask, n),
            _repeat(ri, n),
            _repeat(ci, n),
            model.bundle.projection_steps,
        )
    return raw, projected


def _target_and_terms(model, window: EvalWindow, ca, h_final, mask, args, device):
    """Build the active path-mixture objective and its component terms."""
    context_coords = _as_device(window.context["coords"], device)
    future_coords = _as_device(window.future_path["coords"], device)
    aa = _as_device(window.context["aatype"], device).long()
    ri = _as_device(window.context["residue_index"], device).long()
    ci = _as_device(window.context["chain_id"], device).long()
    anchor = context_coords[-1, :, 1]
    truth = _align_to(future_coords[:, :, 1], anchor, mask)
    target_x = truth[1:]
    with torch.no_grad():
        future_mask = mask[None].expand(future_coords.shape[0] - 1, -1)
        encoded_future = model.bundle.encode(
            future_coords[1:],
            future_mask,
            aa[None].expand(future_coords.shape[0] - 1, -1),
            ri[None].expand(future_coords.shape[0] - 1, -1),
            ci[None].expand(future_coords.shape[0] - 1, -1),
        )
    valid = mask
    target_h = encoded_future
    pred_h = h_final[:, valid].flatten(1)
    target_h = target_h[:, valid].flatten(1)
    pred_ca = ca[:, valid]
    target_ca = target_x[:, valid]
    terms = {}
    terms["latent_es"] = energy_score(pred_h, target_h)
    terms["coord_es"] = energy_score(pred_ca.flatten(1), target_ca.flatten(1))

    pred_dist = torch.cdist(ca[None], ca[None])[0]
    true_dist = torch.cdist(target_x[None], target_x[None])[0]
    pairs = valid[:, None] & valid[None, :]
    sep = (ri[:, None] - ri[None, :]).abs()
    same = ci[:, None] == ci[None, :]
    pairs = pairs & ((sep >= 3) | ~same)
    pairs = pairs & torch.triu(torch.ones_like(pairs), diagonal=1)
    if not pairs.any():
        raise ValueError("fixed diagnostic window has no valid nonlocal residue pairs")
    terms["distance_es"] = energy_score(pred_dist[:, pairs], true_dist[:, pairs])
    terms["clash"] = torch.relu(3.5 - pred_dist[:, pairs]).square().mean()
    terms["contact"] = contact_probability_loss(
        torch.sigmoid((8 - pred_dist) / 0.5).mean(0)[pairs],
        torch.sigmoid((8 - true_dist) / 0.5).mean(0)[pairs],
        args.contact_objective,
    )
    predicted_rmsf = (ca[:, valid] - ca[:, valid].mean(0)).square().sum(-1).mean(0).clamp_min(1e-8).sqrt()
    target_rmsf = (target_x[:, valid] - target_x[:, valid].mean(0)).square().sum(-1).mean(0).clamp_min(1e-8).sqrt()
    terms["rmsf"] = (predicted_rmsf - target_rmsf).abs().mean()
    terms["slow_mode_scale"] = slow_mode_scale_loss(ca[:, valid], target_x[:, valid])
    terms["slow_mode_distribution"] = slow_mode_distribution_loss(ca[:, valid], target_x[:, valid])
    terms["loss"] = (
        args.latent_weight * terms["latent_es"]
        + terms["coord_es"]
        + args.contact_weight * terms["contact"]
        + args.rmsf_weight * terms["rmsf"]
        + args.distance_weight * terms["distance_es"]
        + args.clash_weight * terms["clash"]
        + args.mode_weight * terms["slow_mode_scale"]
        + args.mode_distribution_weight * terms["slow_mode_distribution"]
    )
    return terms


def _state_diagnostics(model, H, S, start, temperature, mask, noise_factor):
    """Record forward state and local drift quantities without building a graph."""
    cfg = model.kernel.cfg
    temp = temperature
    with torch.no_grad():
        D, C, Gamma = model.kernel.bath(start.detach(), temp)
        K = noise_factor[0] @ noise_factor[0].transpose(-1, -2)
        valid = mask[0]
        k_lambda = float(torch.linalg.eigvalsh(K[valid][:, valid]).max())
        c_singular = None
        if C is not None and valid.any():
            c_singular = float(torch.linalg.svdvals(C[0, valid]).amax())
        static = dict(
            D_min=float(D[:, valid].min()),
            D_max=float(D[:, valid].max()),
            Gamma_min=float(Gamma[:, valid].min()) if Gamma is not None else None,
            Gamma_max=float(Gamma[:, valid].max()) if Gamma is not None else None,
            C_singular_max=c_singular,
            hGamma_max=float((cfg.step_ps * Gamma[:, valid]).max()) if Gamma is not None else None,
            K_lambda_max=k_lambda,
        )
    rows = []
    for step in range(H.shape[1]):
        h_state = H[:, step]
        s_state = None if S is None else S[:, step]
        with torch.no_grad():
            force = model.kernel.compute_forces(
                h_state.detach(), mask, temp, create_graph=False
            ).detach()
            if s_state is None or C is None:
                p = force
            else:
                p = force - torch.einsum("blij,blj->bli", C, s_state.detach())
            root_D = D.sqrt()
            drive = root_D * p
            if noise_factor is not None:
                drive = torch.einsum("bij,bjd->bid", noise_factor.transpose(-1, -2), drive)
                drive = torch.einsum("bij,bjd->bid", noise_factor, drive)
            drift = cfg.step_ps * root_D * drive
            values = {
                "step": step,
                "H": _finite_stats(h_state),
                "force": _finite_stats(force),
                "drift": _finite_stats(drift),
            }
            if s_state is not None:
                values["S"] = _finite_stats(s_state)
            rows.append(values)
    return dict(static=static, steps=rows)


def _run_mode(model, window, grad_steps, n_samples, generator_state, args, device, joint_vjp=False):
    model.train()
    model.bundle.eval()
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
    generator = torch.Generator(device=device)
    generator.set_state(generator_state)
    with sdpa_kernel(SDPBackend.MATH):
        trajectory, bath = model.kernel.rollout(
            start,
            rm,
            temp_tensor,
            int(window.horizon),
            seq_feature=start,
            generator=generator,
            detach=False,
            return_bath=True,
            grad_steps=grad_steps,
            noise_factor=noise_factor,
            collective_mobility=model.collective_mobility,
        )
    h_final = trajectory[:, -1]
    with sdpa_kernel(SDPBackend.MATH):
        raw_ca, ca = _decode(model, h_final, start_one, aa, ri, ci, rotation, anchor, mask)
    terms = _target_and_terms(
        model,
        window,
        ca.reshape(1, n_samples, ca.shape[-2], 3)[0],
        h_final.reshape(1, n_samples, h_final.shape[-2], h_final.shape[-1])[0],
        mask[0],
        args,
        device,
    )
    parameters = list(model.kernel.named_parameters())
    gradients = {}
    for name, value in terms.items():
        if not value.requires_grad:
            gradients[name] = _gradient_stats(parameters, [None] * len(parameters))
            continue
        gradients[name] = _gradient_stats(
            parameters,
            torch.autograd.grad(
                value,
                [parameter for _, parameter in parameters],
                retain_graph=True,
                allow_unused=True,
            ),
        )
    endpoint_grad = torch.autograd.grad(
        terms["loss"], h_final, retain_graph=True, allow_unused=True
    )[0]
    endpoint = _finite_stats(endpoint_grad) if endpoint_grad is not None else None
    recurrence = None
    if joint_vjp and grad_steps == 64 and endpoint_grad is not None:
        from scripts.lasp_joint_vjp import joint_path_vjp
        recurrence = joint_path_vjp(
            model.kernel, trajectory.detach(), bath.detach(), start.detach(),
            temp_tensor, rm, noise_factor, endpoint_grad,
        )
    with torch.no_grad():
        truth = _align_to(
            _as_device(window.future_path['coords'], device)[:, :, 1], anchor[0], mask[0]
        )[1:, mask[0]].flatten(1)
    raw_score = energy_score(raw_ca[:, mask[0]].flatten(1), truth)
    raw_gradient = torch.autograd.grad(raw_score, h_final, retain_graph=True)[0]
    projected_gradient = torch.autograd.grad(terms['coord_es'], h_final, retain_graph=True)[0]
    decoder_gradients = dict(
        raw_coord_es=_finite_stats(raw_gradient),
        projected_coord_es=_finite_stats(projected_gradient),
        raw_coord_es_value=float(raw_score.detach()),
    )
    forward = dict(
        H=_finite_stats(trajectory),
        S=_finite_stats(bath) if bath is not None else None,
        ca_raw=_finite_stats(raw_ca),
        ca_projected=_finite_stats(ca),
        min_bond_raw=_safe_float(_minimum_bond(raw_ca, rm, _repeat(ri, n_samples), _repeat(ci, n_samples))),
        min_bond_projected=_safe_float(_minimum_bond(ca, rm, _repeat(ri, n_samples), _repeat(ci, n_samples))),
    )
    state = _state_diagnostics(model, trajectory.detach(), None if bath is None else bath.detach(), start, temp_tensor, rm, noise_factor)
    return dict(
        grad_steps=grad_steps,
        joint_recurrence_vjp=recurrence,
        effective_grad_steps=min(int(grad_steps), int(window.horizon)),
        terms={k: float(v.detach()) for k, v in terms.items()},
        gradients=gradients,
        endpoint_gradient=endpoint,
        decoder_gradients=decoder_gradients,
        forward=forward,
        state=state,
        trajectory=trajectory.detach().cpu(),
        ca=ca.detach().cpu(),
        generator_state=generator_state,
    )


def _panel(manifest, temperatures, horizons, max_residues, max_proteins, windows_per_horizon):
    selected = []
    for temperature in temperatures:
        protocol = EvalProtocol(
            manifest=manifest,
            horizons_frames=tuple(horizons),
            context_frames=8,
            frame_stride=1,
            max_residues=max_residues,
            max_proteins=max_proteins,
            max_anchors_per_trajectory=1,
            temperatures=(float(temperature),),
            seed=0,
            round_robin_proteins=True,
            round_robin_conditions=True,
        )
        counts = defaultdict(int)
        proteins = defaultdict(set)
        for window in iter_windows(protocol):
            horizon = int(window.horizon)
            if counts[horizon] >= windows_per_horizon:
                continue
            # A one-anchor group contains every horizon. Prefer distinct proteins
            # before reusing one, so a small panel does not become one-protein.
            if (
                len(proteins[horizon]) < min(windows_per_horizon, max_proteins)
                and window.protein_id in proteins[horizon]
            ):
                continue
            selected.append(window)
            counts[horizon] += 1
            proteins[horizon].add(window.protein_id)
            if all(counts[h] >= windows_per_horizon for h in horizons):
                break
        if any(counts[h] < windows_per_horizon for h in horizons):
            raise RuntimeError(
                f"could not fill diagnostic panel at {temperature} K: "
                f"counts={dict(counts)} proteins={dict(proteins)}"
            )
    return selected


def _jsonable_window(window):
    return dict(
        protein=window.protein_id,
        trajectory=window.trajectory_id,
        temperature_k=float(window.temperature_k),
        anchor_frame=int(window.anchor_frame),
        horizon=int(window.horizon),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="optional cap for quick screening; default audits all supplied checkpoints")
    parser.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[320., 348., 450.])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--windows-per-horizon", type=int, default=1)
    parser.add_argument("--max-proteins", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=96)
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--joint-vjp", action="store_true")
    parser.add_argument("--seed", type=int, default=20261024)
    args_cli = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args_cli.device)
    panel = _panel(
        args_cli.manifest,
        args_cli.temperatures,
        args_cli.horizons,
        args_cli.max_residues,
        args_cli.max_proteins,
        args_cli.windows_per_horizon,
    )
    if len({(_jsonable_window(w)["protein"], _jsonable_window(w)["trajectory"], _jsonable_window(w)["anchor_frame"], _jsonable_window(w)["horizon"], _jsonable_window(w)["temperature_k"]) for w in panel}) != len(panel):
        raise RuntimeError("diagnostic panel contains duplicate window identities")
    output = Path(args_cli.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    component_dir = output.parent / "stability_components"
    component_dir.mkdir(parents=True, exist_ok=True)
    result = dict(
        scope="development fixed-path no-update stability diagnostic",
        protocol=dict(
            manifest=args_cli.manifest,
            temperatures=args_cli.temperatures,
            horizons=args_cli.horizons,
            windows_per_horizon=args_cli.windows_per_horizon,
            max_proteins=args_cli.max_proteins,
            max_residues=args_cli.max_residues,
            n_samples=args_cli.n_samples,
            grad_modes=list(ENSEMBLE_GRAD_MODES),
            shared_noise=True,
            optimizer_steps=0,
        ),
        panel=[_jsonable_window(w) for w in panel],
        checkpoints=[],
    )
    checkpoint_paths = args_cli.checkpoint
    if args_cli.max_checkpoints is not None:
        if args_cli.max_checkpoints < 1:
            raise ValueError("max-checkpoints must be positive")
        checkpoint_paths = checkpoint_paths[:args_cli.max_checkpoints]
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_kind") != "lasp_ca_dynamics":
            raise ValueError(f"unexpected checkpoint kind: {checkpoint_path}")
        seed = int(checkpoint["args"]["seed"])
        component_path = component_dir / f"seed{seed}.component.pt"
        torch.save(checkpoint["component_checkpoint"], component_path)
        model = CADynamics(
            str(component_path),
            device,
            checkpoint_path,
            history_conditioning=checkpoint["args"].get("history_conditioning", False),
            noise_mix=checkpoint["args"].get("noise_mix", 0.),
            noise_length=checkpoint["args"].get("noise_length", 8.),
            collective_mobility=checkpoint["args"].get("collective_mobility", False),
            noise_kernel=checkpoint["args"].get("noise_kernel", "gaussian"),
            noise_cutoff=checkpoint["args"].get("noise_cutoff", 10.),
        ).to(device)
        model.bundle.eval()
        model.kernel.train()
        checkpoint_result = dict(seed=seed, checkpoint=checkpoint_path, kernel_config=checkpoint["kernel_config"], windows=[])
        for window_index, window in enumerate(panel):
            seed_value = (
                args_cli.seed * 1000003
                + int.from_bytes(
                    f"{window.protein_id}|{window.trajectory_id}|{window.anchor_frame}|{window.horizon}|{window.temperature_k}".encode(),
                    "little",
                )
            ) % (2**31 - 1)
            base_generator = torch.Generator(device=device).manual_seed(int(seed_value))
            generator_state = base_generator.get_state()
            modes = []
            reference = None
            for grad_steps in ENSEMBLE_GRAD_MODES:
                mode = _run_mode(
                    model,
                    window,
                    grad_steps,
                    args_cli.n_samples,
                    generator_state,
                    SimpleNamespace(**checkpoint["args"]),
                    device,
                    joint_vjp=args_cli.joint_vjp,
                )
                if reference is None:
                    reference = {"trajectory": mode["trajectory"], "ca": mode["ca"]}
                    mode["forward_differences"] = dict(H=0.0, ca=0.0)
                else:
                    mode["forward_differences"] = dict(
                        H=float((mode["trajectory"] - reference["trajectory"]).abs().max()),
                        ca=float((mode["ca"] - reference["ca"]).abs().max()),
                    )
                if any(not math.isfinite(v) or v > 1e-6 for v in mode["forward_differences"].values()):
                    raise RuntimeError(f"forward disagreement: {mode['forward_differences']}")
                mode.pop("trajectory", None)
                mode.pop("ca", None)
                mode.pop("generator_state", None)
                modes.append(mode)
                torch.cuda.empty_cache() if device.type == "cuda" else None
            checkpoint_result["windows"].append(dict(index=window_index, identity=_jsonable_window(window), modes=modes))
            print(f"seed={seed} window={window_index + 1}/{len(panel)} {window.temperature_k:g}K h={window.horizon}", flush=True)
        result["checkpoints"].append(checkpoint_result)
        output.write_text(json.dumps(result, indent=2, allow_nan=False))
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    output.write_text(json.dumps(result, indent=2, allow_nan=False))

    aggregate = defaultdict(lambda: defaultdict(list))
    for checkpoint in result["checkpoints"]:
        for window in checkpoint["windows"]:
            for mode in window["modes"]:
                key = (checkpoint["checkpoint"], checkpoint["seed"], window["identity"]["temperature_k"], window["identity"]["horizon"], mode["grad_steps"])
                aggregate[key]["norm64"].append(mode["gradients"]["loss"]["norm64"])
                aggregate[key]["norm32"].append(mode["gradients"]["loss"]["norm32"])
                aggregate[key]["endpoint_norm64"].append((mode["endpoint_gradient"] or {}).get("rms", float("nan")))
    lines = [
        "# LASP fixed-path stability diagnostic",
        "",
        "Development-only, no optimizer updates. Forward samples are shared across full64, last8 and last1 backward windows.",
        "",
        "| checkpoint | seed | T (K) | h | grad window | loss grad norm64 | loss grad norm32 | endpoint grad rms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(aggregate):
        checkpoint_path, seed, temp, horizon, grad_steps = key
        values = aggregate[key]
        lines.append(
            f"| {checkpoint_path} | {seed} | {temp:g} | {horizon} | {grad_steps} | "
            f"{np.nanmean(values['norm64']):.6e} | {np.nanmean(values['norm32']):.6e} | "
            f"{np.nanmean(values['endpoint_norm64']):.6e} |"
        )
    lines += [
        "",
        "Interpretation guardrails:",
        "",
        "- `norm64` is a robust float64 accumulation of finite gradient elements; `norm32` is retained to expose norm-accumulation overflow.",
        "- A non-finite element is stronger evidence than a non-finite aggregate norm. This probe does not classify a terminal trainer exception by itself.",
        "- `grad_steps` changes only backward graph retention; forward values must match the reported `forward_differences` before interpreting gradient differences.",
        "- Results are development diagnostics and do not establish a new trained model or independent test performance.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
