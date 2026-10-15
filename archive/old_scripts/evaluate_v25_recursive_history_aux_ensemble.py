#!/usr/bin/env python3
"""Reproducible V25 recursive history-aux ensemble rollout evaluation.

This script evaluates the paper-facing ``recursive_history_aux_ensemble`` row
using the V25 per-residue / history-bath path, rather than the older global
4D ``world_model.rollout`` path.  The rollout is autonomous after the context
window: future frames are used only to score the generated endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_jepa_frame.config import load_config
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


EPS = 1.0e-12


def _move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _bootstrap(values: np.ndarray, *, n_resamples: int = 2000, seed: int = 17) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    if values.size == 1:
        value = float(values[0])
        return {"mean": value, "ci_lo": value, "ci_hi": value, "n": 1}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, values.size, size=values.size)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(values.size),
    }


def _jsd_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    p = p / max(p.sum(), EPS)
    q = q / max(q.sum(), EPS)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], EPS, None))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _stationary_jsd(pred: np.ndarray, target: np.ndarray, *, bins: int) -> float:
    if pred.shape[0] < 4 or target.shape[0] < 4:
        return float("nan")
    scores = []
    for dim in range(pred.shape[1]):
        lo = float(min(pred[:, dim].min(), target[:, dim].min()))
        hi = float(max(pred[:, dim].max(), target[:, dim].max()))
        if not np.isfinite(lo + hi) or hi <= lo:
            continue
        p_hist, edges = np.histogram(pred[:, dim], bins=bins, range=(lo, hi), density=False)
        q_hist, _ = np.histogram(target[:, dim], bins=edges, density=False)
        scores.append(_jsd_1d(p_hist, q_hist))
    return float(np.mean(scores)) if scores else float("nan")


def _acf_rmse_proxy(pred_delta: np.ndarray, target_delta: np.ndarray) -> float:
    if pred_delta.shape[0] < pred_delta.shape[1] + 2:
        return float("nan")
    pred_c = pred_delta - pred_delta.mean(axis=0, keepdims=True)
    target_c = target_delta - target_delta.mean(axis=0, keepdims=True)
    pred_cov = np.corrcoef(pred_c, rowvar=False)
    target_cov = np.corrcoef(target_c, rowvar=False)
    return float(np.sqrt(np.nanmean((pred_cov - target_cov) ** 2)))


def _energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_term = torch.linalg.vector_norm(samples - target[:, None, :], dim=-1).mean(dim=1)
    pairwise = torch.linalg.vector_norm(
        samples[:, :, None, :] - samples[:, None, :, :], dim=-1
    )
    return target_term - 0.5 * pairwise.mean(dim=(1, 2))


@torch.no_grad()
def _encode_per_residue(model: ProteinJEPA, batch: dict) -> dict[str, torch.Tensor]:
    """Reconstruct the per-residue tensors used by the V25 LSP-WM branch."""
    lspwm = model.lspwm
    if lspwm is None:
        raise RuntimeError("LSP-WM is not enabled")

    context_latents = model.context_encoder(
        coords=batch["context_coords"],
        residue_mask=batch["context_residue_mask"],
        aatype=batch["context_aatype"],
        residue_index=batch["context_residue_index"],
        chain_id=batch["context_chain_id"],
        token_mask=batch["context_residue_mask"],
    )
    future_latents = model.target_encoder(
        coords=batch["future_coords"],
        residue_mask=batch["future_residue_mask"],
        aatype=batch["future_aatype"],
        residue_index=batch["future_residue_index"],
        chain_id=batch["future_chain_id"],
        token_mask=batch["future_residue_mask"],
    )
    if model.config.feature_norm:
        future_latents = F.layer_norm(future_latents, (future_latents.shape[-1],))

    B, T, L = batch["context_coords"].shape[:3]
    Tf = batch["future_coords"].shape[1]
    q_context = lspwm.residue_projection(context_latents.reshape(B, T, L, -1))
    q_future = lspwm.residue_projection(future_latents.reshape(B, Tf, L, -1))

    context_frame_mask = batch["context_residue_mask"].any(dim=-1)
    future_frame_mask = batch["future_residue_mask"].any(dim=-1)
    context_last = (context_frame_mask.long().sum(dim=1) - 1).clamp(min=0)
    future_last = (future_frame_mask.long().sum(dim=1) - 1).clamp(min=0)
    rows = torch.arange(B, device=batch["context_coords"].device)

    q_current = q_context[rows, context_last]
    q_target = q_future[rows, future_last]
    residue_mask = batch["context_residue_mask"][rows, context_last].to(dtype=q_current.dtype)
    future_residue_mask = batch["future_residue_mask"][rows, future_last].to(dtype=q_current.dtype)
    common_mask = (residue_mask.bool() & future_residue_mask.bool()).to(dtype=q_current.dtype)
    ca_last = batch["context_coords"][rows, context_last, :, 1, :]
    ca_distances = torch.cdist(ca_last, ca_last)

    global_current = (
        q_current * residue_mask.unsqueeze(-1)
    ).sum(dim=1) / residue_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    global_target = (
        q_target * common_mask.unsqueeze(-1)
    ).sum(dim=1) / common_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    frames = []
    frame_mask_float = batch["context_residue_mask"].to(dtype=q_context.dtype)
    for offset in range(3):
        idx = (context_frame_mask.long().sum(dim=1) - 1 - offset).clamp(min=0)
        q_frame = q_context[rows, idx]
        rmask = frame_mask_float[rows, idx]
        pooled = (q_frame * rmask.unsqueeze(-1)).sum(dim=1) / rmask.sum(dim=1, keepdim=True).clamp_min(1.0)
        frames.append(pooled)
    aux_input = torch.cat(frames, dim=-1)
    history_aux = lspwm.history_to_bath(aux_input) * 2.0

    return {
        "q_current": q_current.detach(),
        "q_target": q_target.detach(),
        "residue_mask": residue_mask.detach(),
        "common_mask": common_mask.detach(),
        "ca_distances": ca_distances.detach(),
        "global_current": global_current.detach(),
        "global_target": global_target.detach(),
        "history_aux": history_aux.detach(),
    }


def _diffusion(lspwm, G: torch.Tensor, condition: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    diffusion = lspwm.mori_diffusion(G, condition, temperature)
    if hasattr(lspwm, "diffusion_scale_correction"):
        corr = torch.exp(lspwm.diffusion_scale_correction) * torch.relu(
            lspwm._diffusion_scale_correction_scale
        )
        diffusion = diffusion @ torch.diag_embed(corr.to(dtype=diffusion.dtype, device=diffusion.device))
    return diffusion


def _residue_force(
    lspwm,
    q: torch.Tensor,
    residue_mask: torch.Tensor,
    ca_distances: torch.Tensor,
    condition: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    with torch.enable_grad():
        q_req = q.detach().clone().requires_grad_(True)
        mask = residue_mask.to(dtype=q_req.dtype)
        G_req = (q_req * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        energy = lspwm.unified_energy(q_req, G_req, ca_distances, mask, condition, temperature)
        force = -torch.autograd.grad(
            energy["energy_total"].sum(), q_req, create_graph=False, retain_graph=False
        )[0]
    return force.detach()


def _rollout_history_aux_ensemble(
    model: ProteinJEPA,
    encoded: dict[str, torch.Tensor],
    condition: torch.Tensor,
    temperature: torch.Tensor,
    horizon_time_ps: torch.Tensor,
    *,
    rollout_steps: int,
    ensemble_size: int,
    time_scale: float,
    stochastic: bool,
    seed: int,
) -> torch.Tensor:
    lspwm = model.lspwm
    assert lspwm is not None
    B = encoded["q_current"].shape[0]
    E = ensemble_size
    torch.manual_seed(seed)

    q = encoded["q_current"].repeat_interleave(E, dim=0)
    residue_mask = encoded["residue_mask"].repeat_interleave(E, dim=0)
    ca_distances = encoded["ca_distances"].repeat_interleave(E, dim=0)
    condition = condition.repeat_interleave(E, dim=0)
    temperature = temperature.repeat_interleave(E, dim=0)
    horizon_time_ps = horizon_time_ps.repeat_interleave(E) * float(time_scale)
    aux = encoded["history_aux"].repeat_interleave(E, dim=0)

    taus, C_bath = lspwm.multi_bath.parameters_at(dtype=q.dtype, device=q.device)
    rates = taus.reciprocal()
    dt = horizon_time_ps / float(max(1, rollout_steps))
    invalid = (~residue_mask.bool()).unsqueeze(-1)

    for _ in range(max(1, rollout_steps)):
        mask = residue_mask.to(dtype=q.dtype)
        G = (q * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        force = _residue_force(lspwm, q, residue_mask, ca_distances, condition, temperature)
        diffusion = _diffusion(lspwm, G, condition, temperature)
        coupling = C_bath.unsqueeze(0).expand(q.shape[0], -1, -1)
        dissipative = -torch.einsum("bdd,bld->bld", diffusion, force)
        bath = torch.einsum("bda,ba->bd", coupling, aux).unsqueeze(1)
        drift = dissipative + bath

        q_noise = torch.zeros_like(q)
        aux_noise = torch.zeros_like(aux)
        if stochastic:
            eye = torch.eye(diffusion.shape[-1], dtype=diffusion.dtype, device=diffusion.device)
            cov = 2.0 * diffusion * dt[:, None, None]
            cov = 0.5 * (cov + cov.transpose(-1, -2)) + eye.unsqueeze(0) * 1.0e-8
            chol = torch.linalg.cholesky(cov)
            eps = torch.randn_like(q)
            q_noise = torch.einsum("bdd,bld->bld", chol, eps)
            aux_std = (2.0 * rates.unsqueeze(0) * dt[:, None]).clamp_min(0.0).sqrt()
            aux_noise = torch.randn_like(aux) * aux_std

        q_next = q + drift * dt[:, None, None] + q_noise
        q = torch.where(invalid, q, q_next)

        aux_drift = -aux * rates.unsqueeze(0)
        aux = aux + aux_drift * dt[:, None] + aux_noise

    mask = residue_mask.to(dtype=q.dtype)
    G_final = (q * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return G_final.reshape(B, E, -1)


@torch.no_grad()
def _role_overlap(model: ProteinJEPA, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert model.lspwm is not None
    target_prob = model.lspwm.role_readout(target)["msm_logits"].softmax(dim=-1)
    pred_prob = model.lspwm.role_readout(pred)["msm_logits"].softmax(dim=-1)
    return (pred_prob * target_prob).sum(dim=-1)


def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    config = load_config(args.config)
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = upgrade_lspwm_state_dict(model, ckpt.get("model", {}))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    if model.lspwm is None or not getattr(model.lspwm, "per_residue_enabled", False):
        raise RuntimeError("V25 per-residue LSP-WM path is required")

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

    rows = {
        "current": [],
        "target": [],
        "pred_mean": [],
        "energy": [],
        "persistence_energy": [],
        "msm_overlap": [],
        "horizon": [],
        "protein_id": [],
        "trajectory_id": [],
    }
    divergence_count = 0

    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        out = model.forward_paired(values)
        encoded = _encode_per_residue(model, values)
        samples = _rollout_history_aux_ensemble(
            model,
            encoded,
            out["lspwm_condition"].detach(),
            out["lspwm_temperature_k"].detach(),
            values["horizon_time_ps"].detach(),
            rollout_steps=args.rollout_steps,
            ensemble_size=args.ensemble_size,
            time_scale=args.time_scale,
            stochastic=True,
            seed=args.seed + batch_index,
        )
        target = encoded["global_target"]
        current = encoded["global_current"]
        pred_mean = samples.mean(dim=1)
        bad = ~torch.isfinite(pred_mean).all(dim=-1)
        divergence_count += int(bad.sum().detach().cpu())
        energy = _energy_score(samples, target)
        persistence_energy = torch.linalg.vector_norm(current - target, dim=-1)
        msm_overlap = _role_overlap(model, pred_mean, target)

        rows["current"].append(current.detach().cpu().numpy())
        rows["target"].append(target.detach().cpu().numpy())
        rows["pred_mean"].append(pred_mean.detach().cpu().numpy())
        rows["energy"].append(energy.detach().cpu().numpy())
        rows["persistence_energy"].append(persistence_energy.detach().cpu().numpy())
        rows["msm_overlap"].append(msm_overlap.detach().cpu().numpy())
        rows["horizon"].append(values["horizon_frames"].detach().cpu().numpy().reshape(-1))
        rows["protein_id"].extend(str(x) for x in batch.get("protein_id", []))
        rows["trajectory_id"].extend(str(x) for x in batch.get("trajectory_id", []))

    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in rows.items()
        if key not in {"protein_id", "trajectory_id"} and value
    }
    if not arrays:
        raise RuntimeError("no batches were evaluated")

    current = arrays["current"]
    target = arrays["target"]
    pred = arrays["pred_mean"]
    persistence_mse = np.mean((current - target) ** 2, axis=1)
    mse_rows = np.mean((pred - target) ** 2, axis=1)
    skill_rows = 1.0 - mse_rows / np.maximum(persistence_mse, EPS)
    energy_skill = 1.0 - arrays["energy"] / np.maximum(arrays["persistence_energy"], EPS)

    aggregate = {
        "recursive_history_aux_ensemble": {
            "mse": _bootstrap(mse_rows, n_resamples=args.bootstrap_resamples, seed=args.seed + 101),
            "skill_vs_persistence": _bootstrap(
                skill_rows, n_resamples=args.bootstrap_resamples, seed=args.seed + 102
            ),
            "stationary_jsd": _stationary_jsd(pred, target, bins=args.bins),
            "displacement_correlation_rmse_proxy": _acf_rmse_proxy(pred - current, target - current),
            "energy_score": _bootstrap(arrays["energy"], n_resamples=args.bootstrap_resamples, seed=args.seed + 103),
            "energy_skill_vs_persistence": _bootstrap(
                energy_skill, n_resamples=args.bootstrap_resamples, seed=args.seed + 104
            ),
            "msm_soft_overlap": _bootstrap(
                arrays["msm_overlap"], n_resamples=args.bootstrap_resamples, seed=args.seed + 105
            ),
        }
    }

    per_horizon = {}
    for horizon in sorted(set(int(x) for x in arrays["horizon"].tolist())):
        mask = arrays["horizon"] == horizon
        per_horizon[str(horizon)] = {
            "rows": int(mask.sum()),
            "mse": _bootstrap(mse_rows[mask], n_resamples=args.bootstrap_resamples, seed=args.seed + horizon),
            "skill_vs_persistence": _bootstrap(
                skill_rows[mask], n_resamples=args.bootstrap_resamples, seed=args.seed + horizon + 1
            ),
        }

    return {
        "experiment_id": "V25_recursive_history_aux_ensemble_reproduction",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "protocol": {
            "base": "V25 paired-window sample; per-residue autonomous rollout",
            "manifest_path": config.data.manifest_path,
            "max_batches": args.max_batches,
            "batch_size": config.data.batch_size,
            "n_pairs": int(target.shape[0]),
            "n_proteins": int(len(set(rows["protein_id"]))),
            "n_trajectories": int(len(set(rows["trajectory_id"]))),
            "horizon_counts": {
                str(int(h)): int(np.sum(arrays["horizon"] == h))
                for h in sorted(set(arrays["horizon"].tolist()))
            },
            "rollout_steps": args.rollout_steps,
            "ensemble_size": args.ensemble_size,
            "time_scale": args.time_scale,
            "seed": args.seed,
            "target_space": "per_residue_pooled_G8",
            "auxiliary_initialization": "history_to_bath(last3_pooled_per_residue_CVs)*2",
            "bath_evolution": "V25 per_residue_step OU bath evolution",
            "uses_global_4d_world_model_rollout": False,
        },
        "load_state": {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        },
        "divergence_count": divergence_count,
        "aggregate": aggregate,
        "per_horizon": {"recursive_history_aux_ensemble": per_horizon},
        "claim_boundary": (
            "This is a direct reproduction script for the recursive history-aux ensemble row. "
            "It uses observed context only for initialization; target/future states are used only for scoring."
        ),
    }


def _fmt_ci(row: dict, digits: int = 6) -> str:
    return f"{row['mean']:.{digits}f} [{row['ci_lo']:.{digits}f}, {row['ci_hi']:.{digits}f}]"


def write_summary(result: dict, path: Path) -> None:
    row = result["aggregate"]["recursive_history_aux_ensemble"]
    lines = [
        "# V25 Recursive History-Aux Ensemble Reproduction",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Config: `{result['config']}`",
        f"n pairs: {result['protocol']['n_pairs']}",
        f"n proteins: {result['protocol']['n_proteins']}",
        f"n trajectories: {result['protocol']['n_trajectories']}",
        f"horizon counts: {result['protocol']['horizon_counts']}",
        f"rollout steps: {result['protocol']['rollout_steps']}",
        f"ensemble size: {result['protocol']['ensemble_size']}",
        f"time scale: {result['protocol']['time_scale']}",
        f"divergence count: {result['divergence_count']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Method | MSE | Skill vs persistence | Stationary JSD | Disp-corr RMSE proxy | Energy skill | MSM overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            "| recursive_history_aux_ensemble | "
            f"{_fmt_ci(row['mse'])} | "
            f"{_fmt_ci(row['skill_vs_persistence'])} | "
            f"{row['stationary_jsd']:.5f} | "
            f"{row['displacement_correlation_rmse_proxy']:.5f} | "
            f"{_fmt_ci(row['energy_skill_vs_persistence'])} | "
            f"{_fmt_ci(row['msm_soft_overlap'])} |"
        ),
        "",
        "## Protocol",
        "",
        f"- target space: `{result['protocol']['target_space']}`",
        f"- auxiliary initialization: `{result['protocol']['auxiliary_initialization']}`",
        f"- bath evolution: `{result['protocol']['bath_evolution']}`",
        f"- uses global 4D world-model rollout: `{result['protocol']['uses_global_4d_world_model_rollout']}`",
        "",
        result["claim_boundary"],
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lspwm_v44_stage_b_v25_comprehensive.yaml")
    parser.add_argument("--checkpoint", default="artifacts/lspwm_v44_stage_b_v25_comprehensive/checkpoint_step5400.pt")
    parser.add_argument("--output", default="artifacts/lspwm_v44_stage_b_v25_comprehensive/eval/recursive_history_aux_ensemble_repro.json")
    parser.add_argument("--summary", default="artifacts/lspwm_v44_stage_b_v25_comprehensive/eval/recursive_history_aux_ensemble_repro_summary.md")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to horizon_time_ps before integration; 0.001 evaluates ns-scaled steps.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9225)
    args = parser.parse_args()
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        write_summary(result, Path(args.summary))
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
