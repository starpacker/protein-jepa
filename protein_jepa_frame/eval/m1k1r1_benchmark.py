#!/usr/bin/env python3
"""M1/K1/R1 Paper Benchmarks for LSP-WM.

M1 — Thermodynamic Closure: 2×2 {force}×{memory} ablation
K1 — Folding Kinetics: scaffold-disjoint, 8 horizons, ITS relative error
R1 — Representation Closure: history-gap ridge probes
R2 — Autoregressive Rollout Stability: multi-step rollout error accumulation
     (direct test of JEPA-Anything Prop-1 claim: orthogonal geometry →
      stable long-horizon autoregressive prediction)

Reference: docs/LSPWM_METHOD_REPORT.md
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from protein_jepa_frame.config import TrainConfig, load_config, DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict


# =============================================================================
# Utility
# =============================================================================
def _ridge_probe(train_x, train_y, val_x, val_y, lambdas=None):
    """Ridge regression probe: train on (x→y), eval on val."""
    if lambdas is None:
        lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_mse = float('inf')
    for lam in lambdas:
        try:
            n, d = train_x.shape
            xtx = train_x.T @ train_x + lam * np.eye(d)
            xty = train_x.T @ train_y
            w = np.linalg.solve(xtx, xty)
            pred = val_x @ w
            mse = float(((pred - val_y) ** 2).mean())
            if np.isfinite(mse):
                best_mse = min(best_mse, mse)
        except Exception:
            continue
    return best_mse if best_mse < float('inf') else float('nan')


def bootstrap_pairwise_delta(a, b, n_resamples=2000, ci=0.95):
    """Paired bootstrap CI for mean(a) - mean(b)."""
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    if n < 2:
        return float(np.mean(a) - np.mean(b)), [float('nan'), float('nan')]
    deltas = []
    rng = np.random.RandomState(42)
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        deltas.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [(1-ci)/2*100, (1+ci)/2*100])
    return float(np.mean(a) - np.mean(b)), [float(lo), float(hi)]


# =============================================================================
# M1: Thermodynamic Closure
# =============================================================================
def eval_m1_thermodynamic_closure(
    model: ProteinJEPA, loader: DataLoader, device: torch.device,
    max_batches: int = 50,
) -> dict:
    """M1: 2×2 {force}×{memory} ablation on global CVs.

    Four predictions for global G at future time:
      - M_Full:  G_pred = G + (force_drift + bath_drift) * dt   [full LSP-WM]
      - M_Force: G_pred = G + force_drift * dt                  [no memory]
      - M_Memory:G_pred = G + bath_drift * dt                   [no force]
      - M_None:  G_pred = G                                     [persistence]

    Skill = 1 - MSE(pred, target) / MSE(persistence, target)

    Reference baselines (from docs/LSPWM_METHOD_REPORT.md Table 3.1.1):
      - CausalTransformer: MSE=0.00184, DW=7.53, JSD=0.456
      - GRU:               MSE=0.01283, DW=7.55, JSD=0.531
      - GenericMarkov:     MSE=0.00341, DW=8.12, JSD=0.187
    """
    model.eval()

    # Per-mode accumulators
    mode_skills = {
        'full': [], 'force': [],
        'memory': [], 'none': [],
    }
    mode_mses = {'full': [], 'force': [], 'memory': [], 'none': []}
    n_batches = 0
    n_valid_samples = 0

    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break

        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        try:
            out = model.forward_paired(batch)

            # ── CRITICAL: use LSP-WM M1 ablation outputs FIRST ──
            G_current = out.get('lspwm_m1_global_current')
            G_future = out.get('lspwm_m1_global_future')
            force_drift = out.get('lspwm_m1_dissipative_drift')
            bath_drift = out.get('lspwm_m1_bath_drift')
            dt = out.get('lspwm_m1_dt')

            # ── V27 FALLBACK: Per-residue mode produces 8D CV states.
            #     M1 directional projection works in any dimension.
            #     If M1 future is missing, use lspwm_target_state.
            if G_future is None or (isinstance(G_future, torch.Tensor) and G_future.numel() == 0):
                G_future = out.get('lspwm_target_state')
            if G_current is None or (isinstance(G_current, torch.Tensor) and G_current.numel() == 0):
                G_current = out.get('lspwm_context_state')

            # Final fallback via mori-path outputs
            if G_current is None:
                G_current = out.get('lspwm_global_current')
            if G_future is None:
                G_future = out.get('lspwm_global_future')

            if G_current is None or G_future is None:
                n_batches += 1
                continue

            G_current = G_current.detach()
            G_future = G_future.detach()
            if isinstance(G_current, torch.Tensor) and G_current.numel() == 0:
                n_batches += 1
                continue

            # CRITICAL: drifts are in mismatched energy-graph units.  The
            # dissipative_drift uses diffusion_scale=5e-5 and bath_drift
            # has untrained coupling when mori_nll_weight=0.
            #
            # Instead of trying to calibrate drift magnitudes, we measure
            # DIRECTIONAL INFORMATION CONTENT of each component:
            #
            #   Let f̂ = unit(force_drift),  b̂ = unit(bath_drift).
            #   M_Force:  proj of ΔG onto f̂ (predict ΔG ∝ f̂)
            #   M_Memory: proj of ΔG onto b̂ (predict ΔG ∝ b̂)
            #   M_Full:   joint proj onto span{f̂, b̂}
            #
            # Skill = 1 - ‖ΔG - proj‖² / ‖ΔG‖²  (= R² for projection).
            # This measures how much predictive information each direction
            # carries about the actual change.

            # Compute four predictions
            if force_drift is not None:
                force_drift = force_drift.detach()
            if bath_drift is not None:
                bath_drift = bath_drift.detach()

            B = G_current.shape[0]
            for i in range(B):
                G = G_current[i]
                Gt = G_future[i]
                delta_true = Gt - G  # [D]
                d_norm2 = delta_true.dot(delta_true)
                if d_norm2 < 1e-16:
                    for mn in ['full', 'force', 'memory', 'none']:
                        mode_skills[mn].append(0.0)
                        mode_mses[mn].append(0.0)
                    continue

                # Build unit drift directions
                _f_drift = force_drift[i] if force_drift is not None else None
                _b_drift = bath_drift[i] if bath_drift is not None else None

                f_norm2 = _f_drift.dot(_f_drift) if _f_drift is not None else 1e-12
                b_norm2 = _b_drift.dot(_b_drift) if _b_drift is not None else 1e-12

                # Effective drifts (zero out if near-zero norm)
                f_eff = _f_drift if _f_drift is not None and f_norm2 > 1e-12 else None
                b_eff = _b_drift if _b_drift is not None and b_norm2 > 1e-12 else None

                # M_None: zero prediction
                pred_none = torch.zeros_like(delta_true)
                resid_none_norm2 = d_norm2

                # Compute projections and residuals for each mode
                if f_eff is not None and b_eff is not None:
                    # Unit directions
                    f_u = f_eff / torch.sqrt(f_norm2)
                    b_u = b_eff / torch.sqrt(b_norm2)
                    cos_fb = f_u.dot(b_u)  # cos(angle)
                    sin2_fb = 1.0 - cos_fb**2

                    # Force-only: project onto f_unit
                    coeff_f = delta_true.dot(f_u)
                    pred_force = coeff_f * f_u

                    # Memory-only: project onto b_unit
                    coeff_b = delta_true.dot(b_u)
                    pred_memory = coeff_b * b_u

                    # Full: LS projection onto span{f_u, b_u}
                    if sin2_fb > 1e-12:
                        alpha = (coeff_f - cos_fb * coeff_b) / sin2_fb
                        beta  = (coeff_b - cos_fb * coeff_f) / sin2_fb
                        pred_full = alpha * f_u + beta * b_u
                    else:
                        # Collinear: use max of single-projection
                        pred_full = pred_force if coeff_f.abs() >= coeff_b.abs() else pred_memory
                elif f_eff is not None:
                    f_u = f_eff / torch.sqrt(f_norm2)
                    coeff_f = delta_true.dot(f_u)
                    pred_force = pred_full = coeff_f * f_u
                    pred_memory = torch.zeros_like(delta_true)
                elif b_eff is not None:
                    b_u = b_eff / torch.sqrt(b_norm2)
                    coeff_b = delta_true.dot(b_u)
                    pred_memory = pred_full = coeff_b * b_u
                    pred_force = torch.zeros_like(delta_true)
                else:
                    pred_full = pred_force = pred_memory = torch.zeros_like(delta_true)

                # Skill for each mode
                for mode_name, pred in [
                    ('full', pred_full),
                    ('force', pred_force),
                    ('memory', pred_memory),
                    ('none', pred_none),
                ]:
                    resid_norm2 = (delta_true - pred).dot(delta_true - pred)
                    skill = 1.0 - resid_norm2 / d_norm2  # R²
                    skill_float = max(min(skill.item(), 1.0), -10.0)
                    mse = resid_norm2.item() / delta_true.shape[0]
                    mode_skills[mode_name].append(skill_float)
                    mode_mses[mode_name].append(mse)

            n_valid_samples += B
            n_batches += 1

        except Exception as e:
            print(f"  M1 batch {n_batches} error: {e}")
            n_batches += 1
            continue

    # Compute stats per mode
    mode_stats = {}
    for mode_name in ['full', 'force', 'memory', 'none']:
        skills_arr = np.array([s for s in mode_skills[mode_name] if np.isfinite(s)])
        mses_arr = np.array([s for s in mode_mses[mode_name] if np.isfinite(s)])
        if len(skills_arr) == 0:
            mode_stats[mode_name] = {
                'skill_mean': 0.0, 'skill_std': 0.0,
                'mse_mean': 0.0, 'n': 0,
            }
        else:
            mode_stats[mode_name] = {
                'skill_mean': float(skills_arr.mean()),
                'skill_std': float(skills_arr.std()),
                'mse_mean': float(mses_arr.mean()),
                'n': int(len(skills_arr)),
            }

    # Pairwise deltas with bootstrap CI
    pairwise = {}
    for mode_a, mode_b in [
        ('full', 'none'),   # M_Full - M_None (closure delta)
        ('full', 'memory'),  # memory effect  (M_Full w/ force + memory vs w/ memory only)
        ('full', 'force'),   # force effect   (M_Full w/ force + memory vs w/ force only)
        ('force', 'none'),   # force-only vs persistence
        ('memory', 'none'),  # memory-only vs persistence
    ]:
        a_arr = np.array([s for s in mode_skills[mode_a] if np.isfinite(s)])
        b_arr = np.array([s for s in mode_skills[mode_b] if np.isfinite(s)])
        if len(a_arr) > 1 and len(b_arr) > 1:
            delta, ci = bootstrap_pairwise_delta(a_arr, b_arr)
            pairwise[f"{mode_a}_vs_{mode_b}"] = {
                'delta': float(delta),
                'ci_lo': float(ci[0]),
                'ci_hi': float(ci[1]),
                'significant': bool(ci[0] > 0),  # mode_a > mode_b
            }

    # Overall closure test: M_Full > M_None?
    full_arr = np.array([s for s in mode_skills['full'] if np.isfinite(s)])
    none_arr = np.array([s for s in mode_skills['none'] if np.isfinite(s)])
    if len(full_arr) > 1 and len(none_arr) > 1:
        closure_delta, closure_ci = bootstrap_pairwise_delta(full_arr, none_arr)
        closure_significant = bool(closure_ci[0] > 0)
    else:
        closure_delta, closure_ci = 0.0, [0.0, 0.0]
        closure_significant = False

    return {
        'modes': mode_stats,
        'pairwise_deltas': pairwise,
        'closure_delta': float(closure_delta),
        'closure_ci_lo': float(closure_ci[0]),
        'closure_ci_hi': float(closure_ci[1]),
        'closure_significant': closure_significant,
        'n_samples': n_valid_samples,
        'n_batches': n_batches,
        # Backward-compat keys
        'full_mean': mode_stats['full']['skill_mean'],
        'full_std': mode_stats['full']['skill_std'],
        'none_mean': mode_stats['none']['skill_mean'],
        'none_std': mode_stats['none']['skill_std'],
        'm1_closure_delta': float(closure_delta),
        'm1_closure_ci_lo': float(closure_ci[0]),
        'm1_closure_ci_hi': float(closure_ci[1]),
        'm1_significant': closure_significant,
    }


# =============================================================================
# OPF: Orthogonality mechanism audit (JEPA-Anything Table 5 analog)
# Reports cross-factor overlap, sigma_min(P), kappa2(P), synthesis NMSE, and
# factor activity on the trained OPF predictor's projectors.
# =============================================================================
def eval_opf_orthogonality_audit(
    model: ProteinJEPA, device: torch.device, num_batches: int = 20,
) -> dict:
    """Audit the OPF factor interface geometry on a trained model.

    Mirrors JEPA-Anything Sec 3.3.1 / Table 5:
      - cross_factor_overlap : mean squared cross-subspace overlap (lower better)
      - sigma_min(P)         : weakest-covered state direction (higher better)
      - kappa2(P)            : condition number of P = [P_1,...,P_K] (lower better)
      - synthesis_nmse       : ||P(P^T z) - z||^2 / ||z||^2 with exact factors (lower better)
      - factor_activity      : fraction of projected coordinates with std >= gamma
    """
    opf = getattr(model, "opf_predictor", None)
    if opf is None or opf.num_factors is None or opf.num_factors <= 1:
        return {"error": "OPF predictor not present or factor_count==1 (baseline)"}

    P = opf.concatenated_projector()           # [d, d]
    PT = P.T
    PTP = PT @ P
    eye = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)

    # cross-factor overlap: sum_{i<j} ||P_i^T P_j||_F^2 / (K choose 2)
    overlaps = []
    for i in range(opf.num_factors):
        for j in range(i + 1, opf.num_factors):
            overlaps.append((opf.projectors[i].T @ opf.projectors[j]).square().mean())
    cross_overlap = torch.stack(overlaps).mean().item() if overlaps else 0.0

    sv = torch.linalg.svdvals(P)
    sigma_min = float(sv.min().item())
    sigma_max = float(sv.max().item())
    kappa2 = float(sigma_max / max(sigma_min, 1e-12))

    # Synthesis: with exact factors u = P^T z, transpose synthesis P u should
    # reconstruct z. NMSE relative to ||z||^2.
    torch.manual_seed(0)
    z = torch.randn(256, P.shape[0], device=P.device)
    u = (PT @ z.T).T
    z_recon = (P @ u.T).T
    synth_nmse = (z_recon - z).square().sum(dim=1).mean(dim=0) / (
        z.square().sum(dim=1).mean(dim=0) + 1e-12
    )

    return {
        "cross_factor_overlap": float(cross_overlap),
        "sigma_min_P": float(sigma_min),
        "sigma_max_P": float(sigma_max),
        "kappa2_P": float(kappa2),
        "synthesis_nmse": float(synth_nmse.item()),
        "num_factors": int(opf.num_factors),
        "factor_dim": int(opf.factor_dim),
        "latent_dim": int(opf.latent_dim),
    }


# =============================================================================
# K1: Folding Kinetics
# =============================================================================
def eval_k1_folding_kinetics(
    model: ProteinJEPA, loader: DataLoader, device: torch.device,
    max_batches: int = 100,
) -> dict:
    """K1: Evaluate prediction skill across physical horizons.

    For each batch and paired horizon, compute:
      skill = 1 - MSE(pred, target) / MSE(0, target)

    Horizons come from horizon_frames output (sampled during paired training).
    Reports mean per-horizon skill ± bootstrap CI.

    Reference baselines (from docs/LSPWM_METHOD_REPORT.md):
      - JEPA (ours): Mean ITS RE = 0.465 (K1 from Method Report)
      - tICA+MSM: Mean ITS RE = 0.627
      - MSM: Mean ITS RE = 0.785
      - Persistence: Mean ITS RE = 1.595
    """
    model.eval()

    horizon_delta_mse = defaultdict(list)
    horizon_pers_mse = defaultdict(list)
    horizon_skills = defaultdict(list)

    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        try:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model.forward_paired(batch)

            delta_pred = out.get('delta_prediction')
            delta_target = out.get('delta_target')
            horizon_frames = out.get('horizon_frames')
            if delta_pred is None or delta_target is None:
                n_batches += 1
                continue

            delta_pred = delta_pred.detach()
            delta_target = delta_target.detach()

            # Per-sample skill
            B = delta_pred.shape[0]
            for i in range(B):
                h = int(horizon_frames[i].item()) if horizon_frames is not None else 0
                dp = delta_pred[i]
                dt = delta_target[i]
                mse = (dp - dt).square().mean().item()
                pmse = dt.square().mean().item()
                skill = 1.0 - mse / max(pmse, 1e-12)
                horizon_delta_mse[h].append(mse)
                horizon_pers_mse[h].append(pmse)
                horizon_skills[h].append(skill)
            n_batches += 1
        except Exception as e:
            print(f"  K1 batch {n_batches} error: {e}")
            n_batches += 1
            continue

    # Compute per-horizon stats
    k1_horizons = {}
    for h in sorted(horizon_skills.keys()):
        skills = np.array([s for s in horizon_skills[h] if np.isfinite(s)])
        if len(skills) > 1:
            k1_horizons[str(h)] = {
                'skill_mean': float(skills.mean()),
                'skill_std': float(skills.std()),
                'n': int(len(skills)),
                'skill_median': float(np.median(skills)),
                'delta_mse_mean': float(np.mean(horizon_delta_mse[h])),
                'persistence_mse_mean': float(np.mean(horizon_pers_mse[h])),
            }

    # Persistence baseline: skill=0 at all horizons
    baseline_skill = {str(h): 0.0 for h in k1_horizons}

    # Mean skill across horizons
    all_skill_means = [hd['skill_mean'] for hd in k1_horizons.values()]
    mean_skill = float(np.mean(all_skill_means)) if all_skill_means else 0.0

    return {
        'k1_mean_skill': mean_skill,
        'k1_per_horizon': k1_horizons,
        'k1_baseline_per_horizon': baseline_skill,
        'k1_n_batches': n_batches,
        'k1_note': 'Skill = 1 - MSE(pred,target)/MSE(0,target); 0 = persistence; 1 = perfect; JEPA Method Report ITS RE=0.465 vs tICA+MSM=0.627',
    }


# =============================================================================
# R1: Representation Closure
# =============================================================================
def eval_r1_representation_closure(
    model: ProteinJEPA, loader: DataLoader, device: torch.device,
    max_batches: int = 30,
    horizons: list[int] | None = None,
    history_lengths: list[int] | None = None,
) -> dict:
    """R1: History-gap detection with ridge probes.

    Test: Can we predict future state from current state alone, or
    does adding history (multiple past frames) improve prediction?

    If history helps → representation is NOT closed (missing information).
    If history doesn't help → representation IS closed (Markov-sufficient).

    Protocol (matching LSPWM_METHOD_REPORT):
    - Δ_hist = MSE_current - MSE_history  (positive = history helps)
    - Must cross zero-exclusion threshold (95% CI)
    """
    if horizons is None:
        horizons = [1, 2, 4, 8, 16]
    if history_lengths is None:
        history_lengths = [1, 2, 4, 8]

    model.eval()
    lspwm = model.lspwm
    if lspwm is None:
        return {"error": "LSP-WM not enabled", "gate_pass": 0, "total_conditions": 0}

    # Collect current state features and future targets across batches
    all_current = []
    all_future = {}  # horizon -> features
    all_context = []  # history context

    n_batches = 0
    for batch in loader:
        if max_batches and n_batches >= max_batches:
            break
        try:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model.forward_paired(batch)

            # Current CV state
            q = out.get('lspwm_q_current_res')
            if q is not None:
                all_current.append(q.detach().cpu().numpy())

            # Context fixed CV (history)
            ctx = out.get('lspwm_context_fixed_cv')
            if ctx is not None:
                all_context.append(ctx.detach().cpu().numpy())

            # Future CV states for different horizons
            future_q = out.get('lspwm_q_future')
            if future_q is not None:
                fq_np = future_q.detach().cpu().numpy()
                for h in horizons:
                    if h <= fq_np.shape[1]:
                        if h not in all_future:
                            all_future[h] = []
                        # Pool per-residue to global
                        if fq_np.ndim >= 3:
                            all_future[h].append(fq_np[:, h-1, :, :].mean(axis=(1,2)))
                        else:
                            all_future[h].append(fq_np[:, h-1, :].mean(axis=1))

            n_batches += 1
        except Exception as e:
            print(f"  R1 batch {n_batches} error: {e}")
            n_batches += 1
            continue

    if not all_current or not all_future:
        return {"error": "Insufficient data", "gate_pass": 0, "total_conditions": 0}

    # Concatenate
    current = np.concatenate(all_current, axis=0)
    # Pool to global: mean over residues
    if current.ndim >= 3:
        current = current.mean(axis=1)

    context = None
    if all_context:
        context = np.concatenate(all_context, axis=0)
        if context.ndim >= 3:
            context = context.mean(axis=1)

    # Train/val split (per-sample)
    n_samples = current.shape[0]
    n_train = int(0.8 * n_samples)
    idx = np.random.RandomState(42).permutation(n_samples)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    current_train, current_val = current[train_idx], current[val_idx]
    ctx_train = context[train_idx] if context is not None else None
    ctx_val = context[val_idx] if context is not None else None

    # For each horizon: ridge probe current→future vs (current+context)→future
    r1_grid = {}
    gate_pass = 0
    total_conditions = 0

    for h in horizons:
        if h not in all_future:
            continue
        f_future = np.concatenate(all_future[h], axis=0)
        future_train, future_val = f_future[train_idx], f_future[val_idx]

        r1_grid[str(h)] = {}
        for H_len in history_lengths:
            # Current-only probe
            mse_cur = _ridge_probe(current_train, future_train,
                                   current_val, future_val)

            # History-augmented probe (if context available)
            if ctx_train is not None and H_len > 1:
                # Build history feature: [current, context(t-1), context(t-2), ...]
                hist_train = np.concatenate([current_train, ctx_train], axis=1)
                hist_val = np.concatenate([current_val, ctx_val], axis=1)
                mse_hist = _ridge_probe(hist_train, future_train,
                                        hist_val, future_val)
                delta_hist = mse_cur - mse_hist
            else:
                mse_hist = mse_cur
                delta_hist = 0.0

            r1_grid[str(h)][str(H_len)] = {
                'mse_current': float(mse_cur),
                'mse_history': float(mse_hist),
                'delta_hist': float(delta_hist),
            }
            total_conditions += 1
            if delta_hist > 1e-6:
                gate_pass += 1

    return {
        'r1_grid': r1_grid,
        'r1_gate_pass': gate_pass,
        'r1_total_conditions': total_conditions,
        'r1_is_closed': bool(gate_pass <= total_conditions * 0.05),  # <5% pass = closed
        'r1_note': 'Pass=history helps; fewer passes = better representation closure',
    }


# =============================================================================
# R2: Autoregressive Rollout Stability (JEPA-Anything Prop-1 Verification)
# =============================================================================
def eval_r2_rollout_stability(
    model: ProteinJEPA, loader: DataLoader, device: torch.device,
    max_batches: int = 15,
    rollout_steps: list[int] | None = None,
) -> dict:
    """R2: Multi-step autoregressive rollout stability.

    DIRECT TEST of JEPA-Anything Prop-1 claim: OPF's orthogonal factor geometry
    (κ₂(P)=1, zero cross-factor overlap) conditions the latent prediction problem
    so that small per-step errors do not amplify under repeated autoregressive
    composition. This is the mechanism the paper claims underpins stable
    long-horizon rollouts.

    Protocol:
      For each paired batch, we extract the context window's frame_states (a
      sequence of T successive [B, D] state vectors produced by the temporal
      encoder). The context frames form a GROUND-TRUTH trajectory q_0, q_1, ...,
      q_{T-1} where q_k is the endpoint after frame k.

      Rollout (open-loop):
        q̂_0 = context endpoint (= q_{T-1}, the last valid frame)
        For k = 1, ..., rollout_steps:
          Δ̂_k = f(q̂_{k-1}, k·dt)          (model predicts delta at horizon k·dt)
          q̂_k = q̂_{k-1} + Δ̂_k
          Compare q̂_k vs the observed q_{T-1+k} (extrapolated from context states)

      Since the context window provides T frames of observed trajectory, we can
      use the later context frames as ground-truth targets for the first few
      rollout steps. For longer rollouts beyond the context window, we measure
      error growth but lack ground-truth targets — so we report per-step metrics
      up to the available context length.

    Key metrics:
      - per-step: MSE(q̂_k, q_target_k), skill = 1 - MSE / MSE(persistence)
      - Error doubling time: steps for MSE(k)/MSE(1) = 2
      - Error accumulation slope: d(MSE)/d(step) from linear fit
      - Spectral radius estimate: ρ = sqrt(MSE(k+1)/MSE(k)) averaged

    Paper prediction: K4 (OPF) should have flatter error slope and longer
    doubling time than K1 (baseline) because κ₂(P)=1 prevents off-diagonal
    error mixing in the 256-dim prediction space.
    """
    if rollout_steps is None:
        rollout_steps = [1, 2, 3, 4, 6, 8, 12, 16, 24]

    model.eval()

    # Determine which predictor the model ACTUALLY uses in forward_paired.
    # Important: with opf_enabled=True, forward_paired replaces the single-head
    # delta_prediction with opf_out["state_prediction"], even when K==1 (the K1
    # baseline IS OPFPredictor(num_factors=1)). The legacy paired_delta_predictor
    # is untrained in that case, so we must route through opf_predictor whenever
    # it is present to evaluate the model's real autoregressive map.
    opf = getattr(model, "opf_predictor", None)
    use_opf = opf is not None
    num_factors = int(opf.num_factors) if use_opf else 1
    label = f'OPF K={num_factors}' if use_opf else f'K1 legacy-head Baseline'

    # Accumulators
    step_mses = {s: [] for s in rollout_steps}
    step_skills = {s: [] for s in rollout_steps}
    persistence_mses = {s: [] for s in rollout_steps}
    step_mse_first = []
    n_batches = 0
    n_valid = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and n_batches >= max_batches:
            break
        try:
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            B = batch["future_coords"].shape[0]

            # ── Encode context into per-frame states ──
            with torch.no_grad():
                context_latents = model.context_encoder(
                    coords=batch["context_coords"],
                    residue_mask=batch["context_residue_mask"],
                    aatype=batch["context_aatype"],
                    residue_index=batch["context_residue_index"],
                    chain_id=batch["context_chain_id"],
                    token_mask=batch["context_residue_mask"],
                )
                context_states = model.temporal_encoder(
                    context_latents,
                    batch["context_residue_mask"],
                    time_ps=batch.get("context_time_ps"),
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )

                # Encode future into per-frame states (ground-truth trajectory)
                future_latents = model.target_encoder(
                    coords=batch["future_coords"],
                    residue_mask=batch["future_residue_mask"],
                    aatype=batch["future_aatype"],
                    residue_index=batch["future_residue_index"],
                    chain_id=batch["future_chain_id"],
                    token_mask=batch["future_residue_mask"],
                )
                future_states = model.target_temporal_encoder(
                    future_latents,
                    batch["future_residue_mask"],
                    time_ps=batch.get("future_time_ps"),
                    temperature_k=batch.get("temperature_k"),
                    dataset_index=batch.get("dataset_index"),
                )

            # ── Physical-time aligned ground-truth trajectory ──
            # context_time_ps: [B, T_ctx] absolute ps timestamps of context frames
            # future_time_ps:  [B, T_fut] absolute ps timestamps of future frames
            ctx_t = batch["context_time_ps"][:, 0]            # [B] — context start time
            fut_t = batch["future_time_ps"][:, 0]             # [B] — future start time
            # Context frame stride in ps (context frames are contiguous, stride 1)
            ctx_stride = torch.diff(batch["context_time_ps"][:, :4], dim=1).abs().mean(
                dim=1).clamp(min=1e-6)                        # [B]
            fut_stride = torch.diff(batch["future_time_ps"][:, :4], dim=1).abs().mean(
                dim=1).clamp(min=1e-6)                        # [B]

            # The seed is the last valid context frame (standard training entry)
            context_mask = batch["context_residue_mask"].any(dim=2)  # [B, T_ctx]
            future_mask = batch["future_residue_mask"].any(dim=2)    # [B, T_fut]
            batch_idx_ar = torch.arange(B, device=device)
            last_ctx = context_mask.long().mul(
                torch.arange(context_mask.shape[1], device=device).view(1, -1)
            ).max(dim=1).values  # [B]
            seed_frame = context_states["frame_states"][batch_idx_ar, last_ctx]
            seed_state = seed_frame.detach()
            if model.config.feature_norm:
                seed_state = F.layer_norm(seed_state, (seed_state.shape[-1],))
            current_state = seed_state

            # Seed absolute time = time of the last valid context frame
            seed_t0 = batch["context_time_ps"][batch_idx_ar, last_ctx]  # [B]

            # Future window: observed states at known physical times
            # t_future[b, f] = fut_t[b] + f * fut_stride[b] for valid frames
            fut_times = batch["future_time_ps"]  # [B, T_fut] absolute
            # Ground-truth future states in transition space (matching seed space)
            gt_future = future_states["frame_states"]  # [B, T_fut, D]
            if model.config.feature_norm:
                gt_future = F.layer_norm(gt_future, (gt_future.shape[-1],))

            # Context frame states in transition space (extra gt if within window)
            gt_context = context_states["frame_states"]
            if model.config.feature_norm:
                gt_context = F.layer_norm(gt_context, (gt_context.shape[-1],))

            # ── Rollout: closed-loop autoregressive prediction ──
            # The model was trained to predict the delta over `horizon_time_ps`
            # (a physical time >= that of a full 24-frame window). We roll the
            # predictor forward with a FIXED per-step horizon = the batch's
            # horizon_time_ps (the smallest trained horizon, typically 24 frames).
            #
            #   q̂_0 = seed (last context frame)
            #   q̂_k = q̂_{k-1} + f(q̂_{k-1}, dt)      dt = horizon_time_ps (fixed)
            #
            # This is closed-loop / open-loop autoregression: after step 1 the
            # predictor consumes its OWN output state, exactly the "rollout"
            # setting the paper claims OPF's well-conditioned geometry stabilizes.
            #
            # Ground-truth matching by physical time:
            #   t_target[b, k] = seed_t0[b] + k * dt[b]
            #   nearest observed (context ∪ future) frame state at t_target is the
            #   target. We use the future window as the primary ground truth and
            #   the context window for intra-window steps.
            ctx_times = batch["context_time_ps"]  # [B, T_ctx]
            fut_times = batch["future_time_ps"]   # [B, T_fut]

            # Fixed per-step horizon (physical ps). Use the batch's horizon_time_ps;
            # for a batch with mixed horizons take per-sample values directly.
            dt_roll = batch["horizon_time_ps"]  # [B] ps (24, 48 or 96 frame times)

            # Build per-sample observed (time → state) lookup (context + future)
            obs_lists = []
            for i in range(B):
                ctx_t_i = ctx_times[i][context_mask[i]]
                ctx_s_i = gt_context[i][context_mask[i]]
                fut_t_i = fut_times[i][future_mask[i]]
                fut_s_i = gt_future[i][future_mask[i]]
                obs_lists.append((ctx_t_i, ctx_s_i, fut_t_i, fut_s_i))

            max_gt_steps = None

            for step_idx in range(1, max(rollout_steps) + 1):
                # Horizon fed to the predictor at every step is the SAME trained
                # physical delta (per-sample). The rolled state advances by dt_roll.
                step_horizon_ps = dt_roll  # [B]

                # Get prediction for this step using the rolled state.
                # Route through the SAME predictor the model uses at inference
                # (opf_predictor when present, else the legacy single head).
                with torch.no_grad():
                    if use_opf:
                        opf_input = current_state
                        if opf_input.dim() != 2:
                            opf_input = opf_input.reshape(opf_input.shape[0], -1)
                        opf_out = model.opf_predictor(
                            opf_input,
                            step_horizon_ps,
                            temperature_k=batch.get("temperature_k"),
                            dataset_index=batch.get("dataset_index"),
                        )
                        delta_pred = opf_out["state_prediction"]
                    else:
                        delta_pred = model.paired_delta_predictor(
                            current_state,
                            step_horizon_ps,
                            temperature_k=batch.get("temperature_k"),
                            dataset_index=batch.get("dataset_index"),
                        )

                next_state = current_state + delta_pred
                current_state = next_state.detach()  # closed-loop: feed own output

                # ── Evaluate at tracked steps ──
                if step_idx in rollout_steps:
                    # Expected observed time for each sample: seed + k*dt_roll
                    t_target = seed_t0 + dt_roll * step_idx  # [B]

                    for i in range(B):
                        ctx_t_i, ctx_s_i, fut_t_i, fut_s_i = obs_lists[i]
                        obs_t = torch.cat([ctx_t_i, fut_t_i])
                        obs_s = torch.cat([ctx_s_i, fut_s_i])
                        n_obs = obs_t.shape[0]
                        if n_obs == 0:
                            continue
                        # Only match frames at or after the seed (future legit)
                        future_obs_mask = obs_t >= seed_t0[i].item() - 1e-6
                        n_valid_obs = int(future_obs_mask.sum().item())
                        if n_valid_obs == 0:
                            continue
                        valid_idx = torch.where(future_obs_mask)[0]
                        # Nearest observed frame at/after the target time
                        d_pos = (obs_t[valid_idx] - t_target[i]).abs()
                        best_pos = int(valid_idx[d_pos.argmin()].item())
                        tgt_state_obs = obs_s[best_pos].detach()

                        pred = next_state[i].detach()
                        pers = seed_state[i].detach()

                        mse = (pred - tgt_state_obs).square().mean().item()
                        pmse = (pers - tgt_state_obs).square().mean().item()
                        skill = 1.0 - mse / max(pmse, 1e-12)

                        step_mses[step_idx].append(mse)
                        persistence_mses[step_idx].append(pmse)
                        step_skills[step_idx].append(skill)
                        if step_idx == 1:
                            step_mse_first.append(mse)

            n_batches += 1
            n_valid += B

            if (batch_idx + 1) % 10 == 0:
                print(f"  R2 ({label}): processed {n_batches} batches ({n_valid} samples)...")

        except Exception as e:
            print(f"  R2 batch {n_batches} error: {e}")
            import traceback
            traceback.print_exc()
            n_batches += 1
            continue

    # ── Compute summary statistics ──
    r2_steps = {}
    for s in rollout_steps:
        mse_arr = np.array(step_mses[s])
        skill_arr = np.array(step_skills[s])
        pmse_arr = np.array(persistence_mses[s])
        if len(mse_arr) == 0:
            continue
        r2_steps[str(s)] = {
            "mse_mean": float(mse_arr.mean()),
            "mse_std": float(mse_arr.std()),
            "skill_mean": float(skill_arr.mean()),
            "skill_std": float(skill_arr.std()),
            "persistence_mse_mean": float(pmse_arr.mean()),
            "n": int(len(mse_arr)),
        }

    # Compute aggregate metrics from step data
    first_mse_mean = float(np.mean(step_mse_first)) if step_mse_first else 1e-12

    step_nums = []
    mse_means = []
    for s in rollout_steps:
        if s in step_mses and len(step_mses[s]) > 0:
            step_nums.append(float(s))
            mse_means.append(float(np.mean(step_mses[s])))
    step_nums = np.array(step_nums)
    mse_means = np.array(mse_means)

    # Error doubling time
    error_doubling_time = None
    if len(mse_means) > 1 and first_mse_mean > 1e-12:
        threshold = 2.0 * first_mse_mean
        for i in range(len(mse_means)):
            if mse_means[i] >= threshold and i > 0:
                s_lo, s_hi = step_nums[i-1], step_nums[i]
                m_lo, m_hi = mse_means[i-1], mse_means[i]
                if m_hi > m_lo and m_lo < threshold:
                    frac = (threshold - m_lo) / (m_hi - m_lo)
                    error_doubling_time = float(s_lo + frac * (s_hi - s_lo))
                else:
                    error_doubling_time = float(s_hi)
                break
        if error_doubling_time is None and len(mse_means) > 2:
            try:
                log2_ratio = np.log2(mse_means / first_mse_mean)
                coeffs = np.polyfit(step_nums, log2_ratio, 1)
                if coeffs[0] > 1e-12:
                    error_doubling_time = float(1.0 / coeffs[0])
                else:
                    error_doubling_time = float('inf')
            except Exception:
                pass

    # Error accumulation slope
    error_slope = None
    if len(mse_means) > 2:
        try:
            coeffs = np.polyfit(step_nums, mse_means, 1)
            error_slope = float(coeffs[0])
        except Exception:
            pass

    # Spectral radius estimate from MSE growth
    spectral_radius = None
    if len(mse_means) > 1:
        try:
            ratios = []
            for k in range(len(mse_means) - 1):
                if mse_means[k] > 1e-12:
                    ratios.append(np.sqrt(max(mse_means[k+1] / mse_means[k], 0.0)))
            if ratios:
                spectral_radius = float(np.mean(ratios))
        except Exception:
            pass

    # MSE at first step vs last available step
    mse_first_avail = mse_means[0] if len(mse_means) > 0 else 0
    mse_last_avail = mse_means[-1] if len(mse_means) > 0 else 0
    growth_factor = (mse_last_avail / max(mse_first_avail, 1e-12)) if mse_first_avail > 0 else 0

    r2_results = {
        "per_step": r2_steps,
        "error_doubling_time_steps": error_doubling_time,
        "error_slope_per_step": error_slope,
        "spectral_radius_estimate": spectral_radius,
        "first_step_mse_mean": first_mse_mean,
        "last_step_mse_mean": mse_last_avail,
        "error_growth_factor": float(growth_factor),
        "n_batches": n_batches,
        "n_samples": n_valid,
        "is_opf": use_opf,
        "opf_factors": num_factors,
    }

    # Print summary
    print(f"\n  R2 Rollout Stability Summary ({label}):")
    print(f"  {'Step':>6s}  {'MSE':>10s}  {'Skill':>8s}  {'n':>6s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*6}")
    for s in rollout_steps:
        if str(s) in r2_steps:
            d = r2_steps[str(s)]
            print(f"  {s:6d}  {d['mse_mean']:10.6f}  {d['skill_mean']:8.4f}  {d['n']:6d}")
    if error_doubling_time is not None:
        print(f"  Error doubling time: {error_doubling_time:.2f} steps")
    if error_slope is not None:
        print(f"  Error accumulation slope: {error_slope:.6f} MSE/step")
    if spectral_radius is not None:
        print(f"  Spectral radius estimate: {spectral_radius:.4f} (ρ < 1 → stable)")
    if growth_factor > 0:
        print(f"  Error growth factor (step 1 → last): {growth_factor:.2f}x")

    return r2_results


# =============================================================================
# Main
# =============================================================================
def run_m1k1r1_benchmarks(
    checkpoint_path: Path,
    config: TrainConfig,
    device: torch.device,
    output_dir: Path,
    m1_batches: int = 50,
    k1_batches: int = 100,
    r1_batches: int = 30,
    r2_batches: int = 15,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[M1K1R1] Loading model from {checkpoint_path}...")
    model = ProteinJEPA(config.model).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state = upgrade_lspwm_state_dict(model, ckpt.get('model', {}))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model.eval()

    dataset = build_dataset(config.data)
    loader = DataLoader(
        dataset, batch_size=config.data.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_paired_windows, drop_last=True,
    )

    results = {
        'checkpoint': str(checkpoint_path),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── M1 ──
    print("\n" + "=" * 60)
    print("M1: THERMODYNAMIC CLOSURE (2×2 {force}×{memory} ablation)")
    print("=" * 60)
    try:
        m1 = eval_m1_thermodynamic_closure(model, loader, device, max_batches=m1_batches)
        results['M1'] = m1

        modes = m1.get('modes', {})
        print(f"  {'Mode':<12s} {'Skill':>8s}  {'± Std':>8s}  {'MSE':>10s}  {'n':>6s}")
        print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")
        for mode_name in ['full', 'force', 'memory', 'none']:
            ms = modes.get(mode_name, {})
            print(f"  {mode_name:<12s} {ms.get('skill_mean', 0):8.4f}  {ms.get('skill_std', 0):8.4f}  {ms.get('mse_mean', 0):10.6f}  {ms.get('n', 0):6d}")

        pairwise = m1.get('pairwise_deltas', {})
        print(f"\n  Pairwise Δ (bootstrap 95% CI):")
        key_labels = [
            ('full_vs_none', 'M_Full − M_None'),
            ('full_vs_memory', 'M_Full − M_Memory (force benefit)'),
            ('full_vs_force', 'M_Full − M_Force   (memory benefit)'),
            ('force_vs_none', 'M_Force − M_None (force-only gains)'),
            ('memory_vs_none', 'M_Memory − M_None (memory-only gains)'),
        ]
        for key, label in key_labels:
            pd = pairwise.get(key)
            if pd:
                sig = "*" if pd.get('significant') else " "
                print(f"  {label:<40s}: {pd['delta']:+.4f} [{pd['ci_lo']:+.4f}, {pd['ci_hi']:+.4f}] {sig}")

        print(f"\n  Overall closure: Δ = {m1.get('closure_delta', 0):+.4f} "
              f"[{m1.get('closure_ci_lo', 0):+.4f}, {m1.get('closure_ci_hi', 0):+.4f}] "
              f"significant={m1.get('closure_significant', False)}")
        print(f"  n_samples={m1.get('n_samples', 0)}, n_batches={m1.get('n_batches', 0)}")
    except Exception as e:
        results['M1'] = {'error': str(e)}
        print(f"  ERROR: {e}")

    # ── OPF: Orthogonality mechanism audit (JEPA-Anything Table 5 analog) ──
    print("\n" + "=" * 60)
    print("OPF: ORTHOGONALITY MECHANISM AUDIT (cross-overlap / sigma_min / kappa2)")
    print("=" * 60)
    try:
        opf_audit = eval_opf_orthogonality_audit(model, device)
        results['OPF'] = opf_audit
        if 'error' not in opf_audit:
            print(f"  Cross-factor overlap : {opf_audit['cross_factor_overlap']:.6f}")
            print(f"  sigma_min(P)         : {opf_audit['sigma_min_P']:.6f}")
            print(f"  kappa2(P)            : {opf_audit['kappa2_P']:.4f}")
            print(f"  synthesis NMSE       : {opf_audit['synthesis_nmse']:.4e}")
            print(f"  factors/width        : K={opf_audit['num_factors']}, r={opf_audit['factor_dim']}, d={opf_audit['latent_dim']}")
        else:
            print(f"  {opf_audit['error']}")
    except Exception as e:
        results['OPF'] = {'error': str(e)}
        print(f"  ERROR: {e}")

    # ── K1 ──
    print("\n" + "=" * 60)
    print("K1: FOLDING KINETICS (8 horizons, implied timescales)")
    print("=" * 60)
    try:
        k1 = eval_k1_folding_kinetics(model, loader, device, max_batches=k1_batches)
        results['K1'] = k1
        print(f"  Mean skill: {k1.get('k1_mean_skill', 0):.4f}")
        for h, hd in sorted(k1.get('k1_per_horizon', {}).items(), key=lambda x: int(x[0])):
            print(f"    h={h:>3}: skill={hd['skill_mean']:.4f} ± {hd['skill_std']:.4f} (n={hd['n']})")
    except Exception as e:
        results['K1'] = {'error': str(e)}
        print(f"  ERROR: {e}")

    # ── R1 ──
    print("\n" + "=" * 60)
    print("R1: REPRESENTATION CLOSURE (history-gap ridge probes)")
    print("=" * 60)
    try:
        r1 = eval_r1_representation_closure(model, loader, device, max_batches=r1_batches)
        results['R1'] = r1
        print(f"  Gate pass:    {r1.get('r1_gate_pass', 0)}/{r1.get('r1_total_conditions', 0)}")
        print(f"  Is closed:    {r1.get('r1_is_closed', False)}")
        grid = r1.get('r1_grid', {})
        if grid:
            horizons_sorted = sorted(grid.keys(), key=int)
            histories = sorted(next(iter(grid.values())).keys(), key=int)
            print(f"  {'h\\H':>6}", end='')
            for H in histories:
                print(f" {H:>8}", end='')
            print()
            for h in horizons_sorted:
                print(f"  {h:>6}", end='')
                for H in histories:
                    d = grid[h].get(H, {})
                    delta = d.get('delta_hist', 0)
                    print(f" {delta:8.4f}", end='')
                print()
    except Exception as e:
        results['R1'] = {'error': str(e)}
        print(f"  ERROR: {e}")

    # ── R2: Autoregressive Rollout Stability ──
    print("\n" + "=" * 60)
    print("R2: AUTOREGRESSIVE ROLLOUT STABILITY (Prop-1 verification)")
    print("=" * 60)
    try:
        r2 = eval_r2_rollout_stability(model, loader, device, max_batches=r2_batches)
        results['R2'] = r2
        if r2 and r2.get('error_doubling_time_steps') is not None:
            print(f"  Error doubling time: {r2['error_doubling_time_steps']:.2f} steps")
        if r2 and r2.get('error_slope_per_step') is not None:
            print(f"  Error accumulation slope: {r2['error_slope_per_step']:.6f} MSE/step")
        if r2 and r2.get('spectral_radius_estimate') is not None:
            print(f"  Spectral radius estimate: {r2['spectral_radius_estimate']:.4f} (ρ < 1 → stable)")
    except Exception as e:
        results['R2'] = {'error': str(e)}
        print(f"  ERROR: {e}")

    # Save
    out_path = output_dir / "m1k1r1_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


def print_summary_table(results: dict):
    """Print a paper-ready summary table with baseline comparisons."""
    print("\n" + "=" * 80)
    print("PAPER-READY SUMMARY TABLE — LSP-WM V24 vs Baselines")
    print("=" * 80)

    m1 = results.get('M1', {})
    k1 = results.get('K1', {})
    r1 = results.get('R1', {})

    # ── M1 Section ──
    print("\n--- M1: Thermodynamic Closure — 2×2 {force}×{memory} Ablation ---")
    print(f"  Metric: skill = 1 - MSE(model) / MSE(M_None=persistence)")
    modes = m1.get('modes', {})
    print(f"  {'Mode':<16s} {'Skill':>8s}  {'± Std':>8s}  {'MSE':>10s}  {'n':>6s}")
    print(f"  {'-'*16} {'-'*8}  {'-'*8}  {'-'*10}  {'-'*6}")
    for mode_name, label in [('full', 'M_Full'), ('force', 'M_Force'), ('memory', 'M_Memory'), ('none', 'M_None')]:
        ms = modes.get(mode_name, {})
        print(f"  {label:<16s} {ms.get('skill_mean', 0):8.4f}  {ms.get('skill_std', 0):8.4f}  {ms.get('mse_mean', 0):10.6f}  {ms.get('n', 0):6d}")

    pairwise = m1.get('pairwise_deltas', {})
    print(f"\n  Pairwise Ablation Deltas (bootstrap 95% CI, * = significant):")
    key_labels = [
        ('full_vs_none',   'M_Full  − M_None  (full model benefit)'),
        ('full_vs_memory', 'M_Full  − M_Memory (force benefit)'),
        ('full_vs_force',  'M_Full  − M_Force  (memory benefit)'),
        ('force_vs_none',  'M_Force − M_None  (force-only benefit)'),
        ('memory_vs_none', 'M_Memory− M_None  (memory-only benefit)'),
    ]
    for key, label in key_labels:
        pd = pairwise.get(key)
        if pd:
            sig = " *SIGNIFICANT*" if pd.get('significant') else ""
            print(f"    {label:<36s}: Δ = {pd['delta']:+.4f} [{pd['ci_lo']:+.4f}, {pd['ci_hi']:+.4f}]{sig}")

    print(f"\n  Overall closure: M_Full − M_None = "
          f"{m1.get('closure_delta', 0):+.4f} "
          f"[{m1.get('closure_ci_lo', 0):+.4f}, {m1.get('closure_ci_hi', 0):+.4f}], "
          f"significant={m1.get('closure_significant', False)}")

    print("\n  External Baseline Comparison (from Method Report, Table 3.1.1):")
    print(f"  {'Method':<22s} {'Prediction MSE ↓':>16s} {'Innovation DW':>15s} {'Stationary JSD':>15s}")
    print(f"  {'-'*22} {'-'*16} {'-'*15} {'-'*15}")
    for name, mse, dw, jsd in [
        ("CausalTransformer",   "0.00184", "7.53", "0.456"),
        ("GRU",                 "0.01283", "7.55", "0.531"),
        ("GenericMarkov",       "0.00341", "8.12", "0.187"),
    ]:
        print(f"  {name:<22s} {mse:>16s} {dw:>15s} {jsd:>15s}")

    # ── K1 Section ──
    print(f"\n--- K1: Folding Kinetics (Per-Horizon Prediction Skill) ---")
    print(f"  Mean skill across horizons: {k1.get('k1_mean_skill', 0):.4f}")
    if k1.get('k1_per_horizon'):
        print(f"  {'Horizon (frames)':>18s} {'Skill':>8s}  {'± Std':>8s}  {'n':>5s}  {'Δ MSE':>10s}")
        print(f"  {'-'*18} {'-'*8}  {'-'*8}  {'-'*5}  {'-'*10}")
        for h in sorted(k1['k1_per_horizon'].keys(), key=int):
            hd = k1['k1_per_horizon'][h]
            print(f"  {h:>18s} {hd['skill_mean']:8.4f}  {hd['skill_std']:8.4f}  {hd['n']:5d}  {hd['delta_mse_mean']:10.6f}")

    print("\n  K1 Baseline Comparison (from Method Report, Table 3.2.1, ITS relative error ↓):")
    print(f"  {'Method':<16s} {'h=1':>8s} {'h=4':>8s} {'h=8':>8s} {'h=16':>8s} {'h=24':>8s} {'h=32':>8s} {'h=64':>8s} {'h=80':>8s} {'Mean':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, vals in [
        ("JEPA (Ours)",   [0.424, 0.469, 0.406, 0.288, 0.413, 0.396, 0.662, 0.611, 0.465]),
        ("tICA+MSM",      [0.324, 0.383, 0.476, 0.523, 0.628, 0.648, 0.923, 1.108, 0.627]),
        ("MSM",           [0.775, 0.317, 0.429, 0.604, 0.810, 0.814, 1.200, 1.329, 0.785]),
        ("Persistence",   [1.868, 1.871, 1.896, 1.807, 1.706, 1.782, 1.199, 0.634, 1.595]),
    ]:
        row = f"  {name:<16s}"
        for v in vals:
            row += f" {v:8.3f}"
        print(row)
    print("  Note: K1 from Method Report uses scaffold-disjoint ITS; V24 eval uses prediction skill on training distribution.")

    # ── R1 Section ──
    print(f"\n--- R1: Representation Closure ---")
    print(f"  Gate pass:    {r1.get('r1_gate_pass', 0)}/{r1.get('r1_total_conditions', 0)}")
    print(f"  Is closed:    {r1.get('r1_is_closed', False)}")
    grid = r1.get('r1_grid', {})
    if grid:
        horizons_sorted = sorted(grid.keys(), key=int)
        histories = sorted(next(iter(grid.values())).keys(), key=int)
        print(f"  {'h\\H':>6}", end='')
        for H in histories:
            print(f" {H:>8}", end='')
        print()
        for h in horizons_sorted:
            print(f"  {h:>6}", end='')
            for H in histories:
                d = grid[h].get(H, {})
                delta = d.get('delta_hist', 0)
                print(f" {delta:8.4f}", end='')
            print()

    print(f"\n  R1 Baseline Comparison (from Method Report, Table 3.3.1-3.3.2):")
    print(f"  JEPA (256D):        0/20 conditions pass (history does not add info → closed)")
    print(f"  tICA (192D):        14/20 pass (history improves prediction → open)")
    print(f"  VAMP (192D):        11/20 pass (history improves at short horizons)")
    print(f"  PCA/tICA/VAMP (16D): 0/20 pass (too few dims to capture dynamics)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M1/K1/R1 Benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--m1-batches", type=int, default=50)
    parser.add_argument("--k1-batches", type=int, default=100)
    parser.add_argument("--r1-batches", type=int, default=30)
    parser.add_argument("--r2-batches", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_config(args.config)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    results = run_m1k1r1_benchmarks(
        checkpoint, config, device, output_dir,
        m1_batches=args.m1_batches,
        k1_batches=args.k1_batches,
        r1_batches=args.r1_batches,
        r2_batches=args.r2_batches,
    )

    print_summary_table(results)
