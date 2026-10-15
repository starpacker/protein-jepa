# M05: Training Objective Audit & Simplification Plan

**Date:** 2026-09-29  |  **Per:** CVPR Section 4.5  
**Priority:** P0

## Current Objective Inventory

From `config.py` and `train.py`, the following loss terms are active:

### Primary Prediction Losses

| Name | Config Key | Weight Default | Purpose |
|------|-----------|---------------|---------|
| Transition NLL | `state_transition_weight` | 0.0 (disabled) | -log p(q_next | q, F, D, dt) |
| Lagged State JEPA | `lagged_state_jepa_weight` | 0.0 (disabled) | Contrastive across time |
| Residue JEPA | `residue_jepa_weight` | 1.0 (active) | Masked residue prediction (Stage A) |
| State JEPA | `state_jepa_weight` | 0.1 (active) | Masked state prediction (Stage B) |

### Dynamics Regularization

| Name | Config Key | Weight Default | Purpose |
|------|-----------|---------------|---------|
| Semigroup | `semigroup_weight` | 0.0 (disabled) | Consistency: step(step(q,dt1),dt2) ≈ step(q,dt1+dt2) |
| Linear Dynamics | `linear_dynamics_weight` | 0.0 (disabled) | Approximate Koopman / linear evolution |

### Structural Regularization

| Name | Config Key | Weight Default | Purpose |
|------|-----------|---------------|---------|
| PhysChem | `physchem_weight` | 0.0 (disabled) | Physics-chemistry constraints (bond, angle, etc.) |

### Statistical Regularization

| Name | Config Key | Weight Default | Purpose |
|------|-----------|---------------|---------|
| State Variance | `state_variance_reg` | 0.0 (disabled) | Prevent Q collapse to zero |
| Feature Variance | `state_feature_variance_reg` | 0.0 (disabled) | Per-dimension diversity |
| Covariance | `state_covariance_reg` | 0.0 (disabled) | Decorrelate Q dimensions |

### MSM/Kinetic

| Name | Config Key | Weight Default | Purpose |
|------|-----------|---------------|---------|
| MSM State | `msm_state_weight` | 0.0 (disabled) | Markov state model alignment |
| VAMP-2 | `vamp2_weight` | 0.0 (disabled) | Variational approach for Markov processes |
| Kinetic VAMP | `kinetic_vamp_weight` | 0.0 (disabled) | Time-lagged kinetic score |

### Rollout Losses (LSPWM)

| Name | Config Key | Weight Default | Purpose |
|------|-----------|---------------|---------|
| Energy | `lspwm_rollout_energy_weight` | 0.0 (disabled) | Energy conservation in rollout |
| Endpoint | `lspwm_rollout_endpoint_weight` | 0.0 (disabled) | Endpoint matching |
| Gaussian NLL | `lspwm_rollout_gaussian_nll_weight` | 0.0 (disabled) | Trajectory likelihood |
| Distribution | `lspwm_rollout_distribution_weight` | 0.0 (disabled) | Distribution matching |
| Koopman | `lspwm_rollout_koopman_weight` | 0.0 (disabled) | Spectral temporal structure |
| Force | `lspwm_rollout_force_weight` | 0.0 (disabled) | Force matching along trajectory |
| Trajectory NLL | `lspwm_rollout_trajectory_nll_weight` | 0.0 (disabled) | Full trajectory NLL |
| Dynamic | `lspwm_dynamic_weight` | 0.0 (disabled) | Dynamic consistency |
| Dynamic NLL | `lspwm_dynamic_nll_enabled` | false | Dynamic NLL mode |

### E01 Motivated Findings

The E01 experiment revealed that **the current training configuration produces a model where Q grows without bound**. The root cause is that only short-horizon endpoint losses are active (JEPA residue/state), with no trajectory-level regularization to constrain dynamics.

## Simplified Objective Hierarchy

Per CVPR Section 4.5, regroup into:

```
L = L_prediction + λ_rollout L_rollout + λ_struct L_structure
```

### Tier 1: Core Prediction (always active)

| Component | Code | Weight | Justification |
|-----------|------|--------|---------------|
| Transition NLL | `_transition_nll()` | 1.0 | Direct dynamics supervision |
| Residue JEPA | Existing | 1.0 | Stage A representation quality |
| State JEPA | Existing | 0.1 | Q projection learning |

### Tier 2: Rollout Stability (MUST ADD)

| Component | Implementation Needed | Weight | Justification |
|-----------|----------------------|--------|---------------|
| **Semigroup Consistency** | `L_semigroup = ||step(step(q,dt),dt) - step(q,2dt)||²` | 0.5 | Bounds ODE error; proven by M02 |
| **Rollout Endpoint** | Match rollout predictions to true future Q | 1.0 | Directly trains what we evaluate |
| **State Bounding** | Soft penalty on ||Q||² beyond threshold | 0.1 | Prevents drift; M02 diagnostic |

### Tier 3: Physics Regularization (optional, add if beneficial)

| Component | Condition to Enable | Weight |
|-----------|-------------------|--------|
| Energy Conservation | ΔE over rollout << tolerance | 0.01 |
| Koopman Spectral | Clear ITS signal present | 0.05 |
| Force Matching | Physical forces available | 0.1 |

### Tier 4: Structural (Stage C only)

| Component | When | Weight |
|-----------|------|--------|
| PhysChem | When structure decoder active | 0.01 |
| Bond/Clash | Structure output quality | 0.001 |

## Implementation Plan

### Immediate (blocking E04-E06 quality):

1. **Enable semigroup loss** — simple: two-step consistency check
   - File: `models/stage_b/model.py` or `training/train.py`
   - Formula: forward(q, dt1), forward(., dt2) vs forward(q, dt1+dt2)

2. **Enable rollout endpoint matching** — requires trajectory sampling during training
   - Run: 5-10 steps of deterministic rollout
   - Loss: MSE between rollout q and encoded true future q

3. **Add state bound penalty** — soft hinge on Q magnitude
   - `L_bound = softplus(|Q| - bound) * mask`

### This Week (if needed):

4. **Ablation**: Compare with/without each Tier 2 component on E01 metrics
5. **Metric-driven** choice: keep only losses that improve T1a/T1b

## Latent Space Issues from M02/E01

**Finding:** Q magnitude grows ~linearly in rollouts for B4/B5 (17.9 at τ=1000ps).  
**Root cause:** Energy landscape has only local minima from short-horizon JEPA training.  
**Fix needed for CVPR:** Trajectory-level objectives (Tier 2) to create a proper energy basin.

**Without M05 fixes:**
- B0 (persistence) will dominate at τ > 200ps
- B4/B5 degradation will be exponential in time
- CVPR claim "autonomous prediction works" cannot be supported

**With M05 fixes:**
- Semigroup loss constrains integrator accuracy → reduces sub-stepping error
- Rollout endpoint directly optimizes what we measure → closes gap vs B0
- State bound prevents runaway → maintains prediction quality at long horizons

## Decision Point

The current smoke config uses ONLY Tier 1 losses (JEPA residue + state). M05 states we should:
1. First audit (DONE - above)
2. Group (DONE - Tier 1-4 above)
3. Verify with ablation
4. Keep only what helps

**Recommendation:** Implement semigroup loss + rollout endpoint this week (before E04/E06 evaluation). Both are O(10 lines) additions to the training loop and have clear theoretical justification from the M02 stability report.
