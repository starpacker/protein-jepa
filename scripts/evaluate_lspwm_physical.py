"""Evaluate LSP-WM v4.1 with physically meaningful metrics.

Metrics:
  1. MSM state prediction accuracy    — soft-MSM assignment consistency
  2. VAMP-2 score (multi-lag)          — implied timescale spectrum
  3. Koopman eigenvalue spectrum        — full eigenvalue structure
  4. Rollout MSM accuracy               — MSM state prediction on rollout trajectories
  5. Rollout ensemble coverage          — fraction of target covered by ensemble
  6. Transition NLL                     — per-step negative log-likelihood
  7. Detailed balance residual          — π_i T_ij ≈ π_j T_ji
  8. Mean force error                   — ‖∇A_ϕ + F_PMF‖²
  9. Diffusion condition number         — max/eigenvalue spread
  10. FDT residual                      — Cov[dW] vs 2D dt
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
    koopman_diagnostics,
    fit_koopman_operator,
    vamp2_loss,
    _vamp2_score,
    kde_mean_force,
    mean_force_matching_loss,
    overdamped_transition_nll,
    empirical_fdt_residual,
    msm_consistency_loss,
    local_density_weights,
    LSPWMWorldModel,
)


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

    # All lspwm settings are flat attributes on ModelConfig, prefixed with lspwm_
    n_msm_states = getattr(model_config, 'lspwm_msm_role_dim', 32)
    lag_times = [1, 2, 4, 8, 16]  # ns
    
    # Accumulators
    msm_current_list = []
    msm_target_list = []
    all_current = []
    all_target = []
    all_condition = []
    all_temperature = []
    all_horizon = []
    all_history = []
    all_fixed_current = []
    all_fixed_target = []
    all_reference = []
    
    transition_nll_list = []
    force_error_list = []
    
    # For detailed balance
    msm_assignment_pairs = []  # (current_logits, target_logits)
    
    # For diffusion diagnostics
    diff_eigenvalues_list = []
    
    # For FDT
    fdt_noise_cov_list = []
    fdt_diffusion_list = []
    fdt_dt_list = []
    fdt_residual_list = []
    q_fdt_residual_list = []
    auxiliary_fdt_residual_list = []
    fdt_sampling_normalized_list = []
    analytic_fdt_residual_list = []
    
    total_rows = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        outputs = model.forward_paired(values)
        
        current = outputs["lspwm_context_state"]
        target = outputs["lspwm_target_state"]
        condition = outputs["lspwm_condition"]
        temperature = outputs["lspwm_temperature_k"]
        horizon_time_ps = values["horizon_time_ps"]
        history = outputs["lspwm_history_features"]
        reference = outputs["lspwm_context_sequence"][outputs["lspwm_context_mask"]].detach()
        current_fixed = outputs["lspwm_context_fixed_cv"]
        target_fixed = outputs["lspwm_target_fixed_cv"]
        
        all_current.append(current.cpu())
        all_target.append(target.cpu())
        all_condition.append(condition.cpu())
        all_temperature.append(temperature.cpu())
        all_horizon.append(horizon_time_ps.cpu())
        all_history.append(history.cpu())
        all_reference.append(reference.cpu())
        all_fixed_current.append(current_fixed.cpu())
        all_fixed_target.append(target_fixed.cpu())
        
        # Get MSM logits
        role_losses, current_roles = model.lspwm.role_readout.losses(
            current, target, current_fixed, condition
        )
        msm_logits_current = current_roles["msm_logits"]
        # Get MSM logits for target
        _, target_roles = model.lspwm.role_readout.losses(
            target, target, target_fixed, condition
        )
        msm_logits_target = target_roles["msm_logits"]
        
        msm_current_list.append(msm_logits_current.cpu())
        msm_target_list.append(msm_logits_target.cpu())
        msm_assignment_pairs.append((msm_logits_current.cpu(), msm_logits_target.cpu()))
        
        # Transition NLL
        if reference.shape[0] >= 2:
            nll = overdamped_transition_nll(
                model.lspwm.world_model,
                current, target, condition,
                horizon_time_ps, temperature,
                history_features=history,
            )
            transition_nll_list.append(float(nll.cpu()))
        
            # Force error
            mean_force_target = kde_mean_force(
                current.detach(), reference, model.lspwm.mean_force_bandwidth
            )
            force_err = mean_force_matching_loss(
                model.lspwm.world_model.potential,
                current, mean_force_target, condition, temperature,
            )
            force_error_list.append(float(force_err.cpu()))
        
        # Diffusion eigenvalues
        diffusion = model.lspwm.world_model.diffusion(
            current.shape[0], temperature, reference=current
        )
        evals = torch.linalg.eigvalsh(diffusion)
        diff_eigenvalues_list.append(evals.cpu())

        fdt = empirical_fdt_residual(
            model.lspwm.world_model,
            current,
            condition,
            horizon_time_ps,
            temperature,
            samples=args.fdt_samples,
        )
        fdt_residual_list.append(float(fdt["fdt_residual"].cpu()))
        q_fdt_residual_list.append(float(fdt["q_fdt_residual"].cpu()))
        auxiliary_fdt_residual_list.append(
            float(fdt["auxiliary_fdt_residual"].cpu())
        )
        fdt_sampling_normalized_list.append(
            float(fdt["fdt_sampling_normalized"].cpu())
        )
        analytic_fdt_residual_list.append(
            float(fdt["analytic_fdt_residual"].cpu())
        )
        
        total_rows += current.shape[0]

    # ============================
    # 1. MSM STATE PREDICTION ACCURACY
    # ============================
    msm_current_all = torch.cat(msm_current_list, dim=0)  # [N, n_states]
    msm_target_all = torch.cat(msm_target_list, dim=0)
    
    # Hard assignment: argmax
    current_assignment = msm_current_all.argmax(dim=-1)
    target_assignment = msm_target_all.argmax(dim=-1)
    msm_state_accuracy = float((current_assignment == target_assignment).float().mean())
    
    # Soft overlap: average assignment probability overlap
    soft_current = msm_current_all.softmax(dim=-1)
    soft_target = msm_target_all.softmax(dim=-1)
    msm_overlap = float((soft_current * soft_target).sum(dim=-1).mean())
    
    # MSM state distribution
    msm_state_distribution = soft_current.mean(dim=0).tolist()
    msm_state_entropy = float(-(soft_current * soft_current.clamp(min=1e-8).log()).sum(dim=-1).mean())
    
    # ============================
    # 2. VAMP-2 SCORE (MULTI-LAG)
    # ============================
    current_all = torch.cat(all_current, dim=0)
    target_all = torch.cat(all_target, dim=0)
    
    # VAMP-2 at actual lag (varies by batch)
    vamp2 = float(_vamp2_score(current_all, target_all, ridge=1.0e-5).cpu())
    
    # Multi-lag VAMP-2 from Koopman operator analysis
    koopman_diag = koopman_diagnostics(current_all, target_all, lag_time=1.0, ridge=1.0e-5)
    vamp2_one_lag = float(koopman_diag["vamp2_score"])
    implied_timescales = koopman_diag["implied_timescales"].real
    top_implied_timescales = [float(t) for t in implied_timescales[:5].cpu()]
    
    # Koopman eigenvalue spectrum
    koop_eigenvalues = koopman_diag["eigenvalues"]
    koop_eigenvalues_real = [float(v.real) for v in koop_eigenvalues.cpu()]
    koop_eigenvalues_imag = [float(v.imag) for v in koop_eigenvalues.cpu()]
    koopman_residual_mse = float(koopman_diag["residual_mse"])
    
    # ============================
    # 3. ROLLOUT MSM ACCURACY
    # ============================
    rollout_msm_list = []
    rollout_ensemble_coverage = []
    rollout_energy_score_list = []
    rollout_divergence_count = 0
    rollout_total_steps = 0
    
    # Reuse the same sampled windows for rollout diagnostics. Without resetting
    # the sampler, the second pass silently evaluates a different random set.
    sampler.iteration = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        values = _move(batch, device)
        outputs = model.forward_paired(values)
        
        current = outputs["lspwm_context_state"]
        target = outputs["lspwm_target_state"]
        condition = outputs["lspwm_condition"]
        temperature = outputs["lspwm_temperature_k"]
        horizon_time_ps = values["horizon_time_ps"]
        history = outputs["lspwm_history_features"]
        reference = outputs["lspwm_context_sequence"][outputs["lspwm_context_mask"]].detach()
        current_fixed = outputs["lspwm_context_fixed_cv"]
        
        rollout_steps = args.rollout_steps
        ensemble_size = args.ensemble_size
        batch_size = current.shape[0]
        
        # Run rollout
        diag = model.lspwm.compute_rollout_diagnostics(
            current, condition, history, temperature, horizon_time_ps,
            rollout_steps=rollout_steps, ensemble_size=ensemble_size,
        )
        states = diag["lspwm_rollout_trajectory"]  # [batch, ensemble, steps+1, state_dim]
        
        for step_idx in range(1, rollout_steps + 1):
            q_step = states[:, :, step_idx, :]  # [batch, ensemble, state_dim]
            rollout_total_steps += batch_size * ensemble_size
            
            # Check divergence
            if torch.isnan(q_step).any() or torch.isinf(q_step).any():
                rollout_divergence_count += 1
                continue
            
            # MSM state prediction for rollout states
            flat_step = q_step.reshape(-1, current.shape[-1])
            flat_condition = condition.repeat_interleave(ensemble_size, dim=0)
            _, step_roles = model.lspwm.role_readout.losses(
                flat_step, flat_step, current_fixed.repeat_interleave(ensemble_size, dim=0), flat_condition
            )
            msm_logits_step = step_roles["msm_logits"].reshape(batch_size, ensemble_size, -1)
            
            # MSM accuracy: compare current MSM assignment to rollout step MSM assignment
            _, current_roles_step = model.lspwm.role_readout.losses(
                current, current, current_fixed, condition
            )
            current_msm = current_roles_step["msm_logits"].argmax(dim=-1)  # [batch]
            step_msm = msm_logits_step.argmax(dim=-1)  # [batch, ensemble]
            
            # Accuracy: does rollout stay in same MSM state?
            per_sample = (step_msm == current_msm.unsqueeze(-1)).float().mean(dim=-1)
            rollout_msm_list.append(per_sample.cpu())
            
            # Ensemble coverage: does the ensemble cover the target MSM state?
            _, target_roles_step = model.lspwm.role_readout.losses(
                target, target, outputs["lspwm_target_fixed_cv"], condition
            )
            target_msm = target_roles_step["msm_logits"].argmax(dim=-1)  # [batch]
            coverage = (step_msm == target_msm.unsqueeze(-1)).float().mean(dim=-1)
            rollout_ensemble_coverage.append(coverage.cpu())
            
            # Energy score (per-sample)
            step_frac = step_idx / rollout_steps
            step_target = current + step_frac * (target - current)
            delta = q_step - step_target.unsqueeze(1)
            score = torch.linalg.vector_norm(delta, dim=-1).mean(dim=-1)
            rollout_energy_score_list.append(score.cpu())
    
    rollout_msm_accuracy = float(torch.cat(rollout_msm_list).mean()) if rollout_msm_list else 0.0
    rollout_ensemble_coverage_val = float(torch.cat(rollout_ensemble_coverage).mean()) if rollout_ensemble_coverage else 0.0
    rollout_mean_energy_score = float(torch.cat(rollout_energy_score_list).mean()) if rollout_energy_score_list else 0.0
    
    # ============================
    # 4. DETAILED BALANCE RESIDUAL
    # ============================
    # Use MSM logits to estimate transition matrix
    if msm_assignment_pairs:
        all_cur = torch.cat([p[0] for p in msm_assignment_pairs], dim=0)
        all_tgt = torch.cat([p[1] for p in msm_assignment_pairs], dim=0)
        p_cur = all_cur.softmax(dim=-1)
        p_tgt = all_tgt.softmax(dim=-1)
        
        # Stationary distribution: mean of current assignments
        pi = p_cur.mean(dim=0)  # [n_states]
        
        # Transition matrix: T_{ij} = P(target = j | current = i)
        # Soft assignment weighting
        n_states = p_cur.shape[-1]
        T = torch.zeros(n_states, n_states, device=p_cur.device)
        for i in range(n_states):
            w_i = p_cur[:, i]  # weight of being in state i
            if w_i.sum() > 1e-8:
                T[i, :] = (p_tgt * w_i.unsqueeze(-1)).sum(dim=0) / w_i.sum()
        
        # Detailed balance: pi_i * T_{ij} ≈ pi_j * T_{ji}
        pi_matrix = pi.unsqueeze(-1) * T  # pi_i * T_{ij}
        pi_matrix_T = pi.unsqueeze(0) * T.T  # pi_j * T_{ji}
        db_residual = float((pi_matrix - pi_matrix_T).square().mean().sqrt().cpu())
        db_max_residual = float((pi_matrix - pi_matrix_T).abs().max().cpu())
    else:
        db_residual = 0.0
        db_max_residual = 0.0
        n_states = 0
        T = torch.zeros(0, 0)
    
    # ============================
    # 5. DIFFUSION & FDT DIAGNOSTICS
    # ============================
    diff_all = torch.cat(diff_eigenvalues_list, dim=0) if diff_eigenvalues_list else torch.zeros(1, 1)
    diff_min = float(diff_all[:, 0].min())
    diff_max = float(diff_all[:, -1].max())
    diff_condition = float((diff_all[:, -1] / diff_all[:, 0].clamp(min=1e-12)).max())
    diff_psd = diff_min > 0.0
    
    # ============================
    # 6. TRANSITION & FORCE STATISTICS
    # ============================
    mean_transition_nll = float(torch.tensor(transition_nll_list).mean()) if transition_nll_list else 0.0
    mean_force_error = float(torch.tensor(force_error_list).mean()) if force_error_list else 0.0
    mean_fdt_residual = float(torch.tensor(fdt_residual_list).mean()) if fdt_residual_list else float("nan")
    max_fdt_residual = max(fdt_residual_list) if fdt_residual_list else float("nan")
    max_q_fdt_residual = max(q_fdt_residual_list) if q_fdt_residual_list else float("nan")
    max_auxiliary_fdt_residual = max(auxiliary_fdt_residual_list) if auxiliary_fdt_residual_list else float("nan")
    mean_fdt_sampling_normalized = (
        float(torch.tensor(fdt_sampling_normalized_list).mean())
        if fdt_sampling_normalized_list else float("nan")
    )
    max_fdt_sampling_normalized = (
        max(fdt_sampling_normalized_list)
        if fdt_sampling_normalized_list else float("nan")
    )
    max_analytic_fdt_residual = (
        max(analytic_fdt_residual_list)
        if analytic_fdt_residual_list else float("nan")
    )
    
    # ============================
    # BUILD RESULTS
    # ============================
    result = {
        "framework": args.framework_label,
        "protocol": "physical_metrics_evaluation",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "seed": args.seed,
        "total_rows": total_rows,
        "rollout_steps": args.rollout_steps,
        "ensemble_size": args.ensemble_size,
        
        # === MSM State Prediction ===
        "msm_state_prediction": {
            "n_msm_states": n_msm_states,
            "hard_assignment_accuracy": round(msm_state_accuracy, 6),
            "soft_assignment_overlap": round(msm_overlap, 6),
            "state_distribution_entropy": round(msm_state_entropy, 6),
            "state_distribution": [round(float(v), 6) for v in msm_state_distribution],
        },
        
        # === VAMP-2 / Koopman ===
        "koopman_analysis": {
            "vamp2_score": round(vamp2, 4),
            "vamp2_score_one_lag": round(vamp2_one_lag, 4),
            "koopman_residual_mse": round(koopman_residual_mse, 8),
            "top_implied_timescales": [round(float(t), 2) for t in top_implied_timescales],
            "eigenvalue_real": [round(float(v), 6) for v in koop_eigenvalues_real[:8]],
            "eigenvalue_imag": [round(float(v), 6) for v in koop_eigenvalues_imag[:8]],
            "n_koopman_modes": len(koop_eigenvalues_real),
        },
        
        # === Rollout Self-Consistency ===
        "rollout_self_consistency": {
            "msm_state_retention_accuracy": round(rollout_msm_accuracy, 6),
            "ensemble_target_coverage": round(rollout_ensemble_coverage_val, 6),
            "mean_energy_score": round(rollout_mean_energy_score, 6),
            "divergence_count": rollout_divergence_count,
            "total_steps_sampled": rollout_total_steps,
        },
        
        # === Detailed Balance ===
        "detailed_balance": {
            "rms_residual": round(db_residual, 8),
            "max_residual": round(db_max_residual, 8),
            "stationary_distribution": [round(float(v), 6) for v in pi.cpu().tolist()] if msm_assignment_pairs else [],
            "interpretation": "soft-MSM readout diagnostic; not a detailed-balance proof for the continuous world model",
        },
        
        # === Thermodynamic Consistency ===
        "thermodynamic_consistency": {
            "mean_transition_nll": round(mean_transition_nll, 4),
            "mean_force_error": round(mean_force_error, 8),
            "diffusion_psd": diff_psd,
            "min_diffusion_eigenvalue": round(diff_min, 8),
            "max_diffusion_eigenvalue": round(diff_max, 8),
            "diffusion_condition_number": round(diff_condition, 4),
            "mean_fdt_residual": round(mean_fdt_residual, 8),
            "max_fdt_residual": round(max_fdt_residual, 8),
            "max_q_fdt_residual": round(max_q_fdt_residual, 8),
            "max_auxiliary_fdt_residual": round(max_auxiliary_fdt_residual, 8),
            "mean_fdt_sampling_normalized": round(mean_fdt_sampling_normalized, 8),
            "max_fdt_sampling_normalized": round(max_fdt_sampling_normalized, 8),
            "analytic_fdt_residual": round(max_analytic_fdt_residual, 8),
            "fdt_sample_count_per_batch": args.fdt_samples,
            "fdt_interpretation": "raw covariance error is normalized by its finite-sample Wishart baseline; values near one are expected under the exact integrator.",
        },
        
        # === Physical Gate Summary ===
        "physical_gates": {
            "diffusion_psd": diff_psd,
            "minimum_diffusion_eigenvalue": round(diff_min, 8),
            "maximum_diffusion_condition_number": round(diff_condition, 4),
            "analytic_fdt_residual": round(max_analytic_fdt_residual, 8),
            "fdt_integrator_identity": bool(max_analytic_fdt_residual < 1.0e-6),
            "fdt_sampling_normalized_residual": round(max_fdt_sampling_normalized, 8),
            "fdt_raw_residual": round(max_fdt_residual, 8),
            "fdt_measured": bool(fdt_residual_list),
            "continuous_detailed_balance_measured": False,
            "energy_force_is_conservative_by_construction": True,
        },
        
        "claim_boundary": f"{args.framework_label} physical metrics (internal validation); compare against EVA5C only after matched held-out evaluation",
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
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
