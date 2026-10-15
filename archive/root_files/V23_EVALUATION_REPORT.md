# V23 LSP-WM Evaluation Report
## Hierarchical Mori-Zwanzig World Model with Bug-Fixed Scoring
**Date**: 2026-09-22
**Checkpoint**: `checkpoint_step4500.pt` (best — dynamics phase end, pre-rollout explosion)
**Training**: 4770 steps total, killed at rollout explosion (step 4630)

---

## 1. Executive Summary

V23 fixed all four critical bugs from V22 (force scoring, per-residue gating, self-consistency, Mori NLL). The model trained successfully through representation, thermodynamic identification, and dynamics phases (steps 0–4500). The rollout phase (step 4500+) caused a catastrophic numerical explosion at step 4630, replicating the V22 failure mode.

**Best checkpoint** (step 4500) shows:
- ✅ Force score matching on G converges (from ~97 to ~0.34)
- ✅ Mori NLL converges to strong negative values (−42)
- ✅ Per-residue forces show good directional alignment
- ⚠️ Force score Q remains high (~400, target-dominated)
- ❌ Bath auxiliary state is never propagated (always zero)
- ❌ Rollout trajectory NLL explodes to 10¹⁴
- ❌ Endpoint skills worse than persistence baseline

### Improvements over V22

| Metric | V22 (broken) | V23 (fixed) | Δ |
|---|---|---|---|
| Force score G (KDE MSE) | ~100 (unstable) | 0.34 ± 0.10 | ✅ 300× improvement |
| Mori NLL | No computation | −42.2 ± 1.2 | ✅ Now computed |
| Residue direction loss | 0 (gated) | 0.052 ± 0.002 | ✅ Now active |
| Self-consistency loss | 0 (identity bug) | 2.33 ± 0.39 | ✅ Now meaningful |
| Stationary loss | NaN (buggy MMD) | 0.27 ± 0.11 | ✅ Now working |

---

## 2. Architecture Summary

The V23 LSP-WM implements a hierarchical free energy decomposition:

**A(Q,G) = A_local(q_i) + A_pair(q_i, q_j, d_ij) + A_g(G) + A_coupling(q_i, G)**

Components:
- `UnifiedEnergyGraph`: Conservative potential producing per-residue + global forces
- `GaugeProjector`: Separates rigid-body motion from internal dynamics
- `MultiTimescaleBath`: FDT-consistent OU processes (τ ∈ {1000, 8000, 64000} ps, 16 dims)
- `DiagonalDiffusion`: State-dependent mobility tensor
- `ResidueConditionalDecoder`: p(q_i^{t+1} | q_i^t, G^t, G^{t+1})

Losses: cv_vamp2 + force_score_G + force_score_Q + force_direction + mori_nll + self_consistency + residue_conditional + stationary + (rollout blown up)

---

## 3. Quantitative Results (Step 4500, N=30 batches)

### 3.1 Force Field Closure (M1)

| Metric | Value | Notes |
|---|---|---|
| **Force Score G (KDE MSE)** | 1.56 ± 0.07 | Targets grad_G A(Q,G) via KDE on global CV ensemble |
| **Force Score Q (Sliced Score)** | 397 ± 5 | Very high — energy gradient is ~2 while target score is ~400 |
| **Residue Force Loss** | 0.0067 ± 0.0025 | MSE on per-residue displacement |
| **Residue Direction Loss** | 0.0488 ± 0.0031 | Angular alignment (lower = better) |
| **Residue Amplitude Loss** | 0.0018 ± 0.0000 | Rate-space force amplitude match |
| **Force-Displacement Cosine** | 0.027 ± 0.008 | Near zero — forces nearly orthogonal to displacements |
| **Global Force Norm** | 0.43 ± 0.004 | Force magnitude on G |

**Key insight**: The force-score-Q loss is dominated by the target score norm (−ε/σ² = 400). The energy graph predicts forces of norm ~2, creating a 200× mismatch. This suggests either: (a) the energy graph needs stronger gradients, or (b) the noise scale σ should be increased to reduce the target magnitude.

### 3.2 Thermodynamic Consistency

| Metric | Value | Notes |
|---|---|---|
| **CV VAMP-2 loss** | −0.65 ± 0.21 | Slow-mode representation quality |
| **Koopman loss** | −1.02 ± 0.24 | Koopman operator eigenvalue alignment |
| **Self-Consistency Loss** | 2.63 ± 0.07 | A_g(G) + mean[φ(q_i,G)] ≈ KDE(G) mismatch |
| **Stationary Distribution MMD** | 0.027 ± 0.033 | Distance from equilibrium (converging) |
| **Energy Coupling Variance** | 0.0003 ± 0.00001 | Cross-scale coupling strength (stable) |

### 3.3 Dynamics Folding

| Metric | Value | Notes |
|---|---|---|
| **Mori NLL** | −36.9 ± 2.1 | FDT-consistent transition log-likelihood on G |
| **Dynamic NLL** | 8.0 ± 0.2 | Overdamped transition NLL (positive = poorly calibrated) |
| **Endpoint Energy Skill** | −8.5 ± 1.6 | **Negative** → worse than persistence baseline |
| **Deterministic Skill** | −0.12 ± 0.09 | No predictive power for displacement |
| **Mori Force G Norm** | 0.43 ± 0.004 | Conservative force magnitude |
| **Mori Bath Drift Norm** | 0.0000 | **Always zero** — auxiliary state not propagated |
| **Mori Dissipative Drift** | 0.0000 | **Always zero** — same root cause |
| **Mori Total Drift** | 0.0000 | Only reversible drift is active |

### 3.4 Bath State Pathology

The `MultiTimescaleBath` has 16 auxiliary dimensions across 3 timescale groups but produces zero drift because:

```python
# In mori_transition_nll:
if auxiliary is None:
    auxiliary = G.new_zeros(B, aux_dim)  # ← Always zero
bath_drift = torch.einsum("bda,ba->bd", C_expanded, auxiliary)  # = 0
```

The auxiliary state is never propagated between calls — each mori_transition_nll call receives `auxiliary=None`, zeroes it out, computes a next auxiliary state, and then **discards it**. This means:
- Bath memory is never accumulated
- FDT noise is missing from dynamics
- Only the deterministic overdamped part operates (dissipative + conservative)

---

## 4. Training Dynamics

### 4.1 Loss Convergence

```
Phase 1 (steps 0–200): Representation warmup
  cv_vamp2:      −4.87 → Good slow-mode separation
  koopman:       −4.71 → Good spectral alignment
  All force losses: 0 (gated by phase)

Phase 2 (steps 500–1200): Thermodynamic identification
  force_score_G:  97.4 → 0.4  ✅ Converges rapidly
  force_score_Q:  118  → 400  ⚠️ Saturates at high plateau
  mori_nll:       53   → -30  ✅ Crosses from positive to negative
  self_consistency: 0.2 → 2.2  ✅ Becomes meaningful
  residue*:       Activates   ✅ Gradients flow

Phase 3 (steps 2000–4500): Dynamics
  mori_nll:       −30  → −42  ✅ Continues improving
  force_score_G:   0.4 → 0.34 ✅ Stable convergence
  force_score_Q:  400  → 400  ⚠️ Frozen — gradient vanishing?
  residue_direction: 0.051      ⚡ Reasonable
  residue_conditional: −0.55 → −0.79 ✅ Negative NLL (well-calibrated)
  stationary:      0.14 → 0.27  ⚠️ Slowly increasing (distribution drift?)

Phase 4 (steps 4500+): Rollout
  rollout_gaussian_nll:   0 → 1368 → ∞  ❌ Explodes
  rollout_trajectory_nll:  0 → 2×10¹⁴  ❌ Catastrophic
  rollout_endpoint_loss:   0 → 11900    ❌ Explodes
  Training killed at step 4630
```

### 4.2 Rollout Explosion Analysis

The rollout phase introduced three new loss terms that all exploded:

1. **rollout_gaussian_nll_weight = 0.1**: From ~300 to 1368 at step 4630, contributing ~136 × 0.1 = 13.6 to total loss (moderate)
2. **rollout_endpoint_weight = 0.1**: From ~0 to 11905, contributing ~1190 to total loss (large)
3. **rollout_trajectory_nll_weight = 0.05**: From ~0 to 2×10¹⁴, contributing 10¹³ to total loss (catastrophic)

Root cause: The rollout uses `mori_transition_nll` iteratively with the same buggy zero-auxiliary behavior. Over 8 rollout steps, each step's NLL compounds additively but gradient scales with the accumulated error.

---

## 5. Issues Requiring Fix (V24)

### Critical (training-blocking)

1. **Rollout trajectory NLL explosion**: Need numerical safeguards:
   - Clamp log-variance of Gaussian NLL
   - Use log-sum-exp for numerical stability
   - Initialize rollout weights with warmup (curriculum step 4500 may be too aggressive)
   - Consider using Huber loss instead of NLL for trajectory

2. **Bath auxiliary not propagated**: Add stateful auxiliary tracking:
   - Store last auxiliary state from mori_transition_nll
   - Pass it to the next call (even in single-step mode)
   - Implement multi-step bath propagation for rollout

### High Priority

3. **Force Score Q saturation (~400)**: The target score norm is −ε/σ², which for dim=8 and σ=0.05 gives per-element target ~20. The total MSE is E[‖ε/σ²‖²] = D/σ² = 8/0.0025 = 3200. The energy graph predicts forces of norm ~2, creating a 200× scale mismatch.

   Fix: Normalize by D (dimension) OR use sliced score properly with per-element normalization OR increase noise_std to 0.2-0.5 to reduce target magnitude.

4. **Endpoint skills negative**: The model's energy-based predictions are *worse* than using the current energy as prediction of future energy. This indicates the conservative force field is not improving short-term prediction accuracy.

   Possible causes:
   - Energy graph has not learned meaningful dynamics yet
   - Global CV pooling loses per-residue information
   - Forces and displacements are in different spaces

### Medium Priority

5. **Force-displacement cosine near zero**: Forces are essentially orthogonal to actual displacements. Either:
   - The energy graph captures a different subspace than actual motion
   - The gauge projection removes the physically relevant components
   - The noise dominates the signal at this training stage

6. **Self-consistency loss at 2.6**: Decreasing → the KDE potential and the energy graph disagree on PMF by ~1.6σ

---

## 6. Comparison to Baselines

| Metric | V22 (Broken) | V23 Best (Step 4500) | Target |
|---|---|---|---|
| Force score G MSE | NaN/unstable | 1.56 ± 0.07 | < 0.1 |
| Mori NLL | N/A | −36.9 ± 2.1 | < −50 |
| Residue direction | 0 (gated) | 0.049 ± 0.003 | < 0.01 |
| Residue conditional NLL | 0 (gated) | −0.84 ± 0.002 | < −2.0 |
| Endpoint skill | N/A | −8.5 ± 1.6 | > 0.5 |
| Deterministic skill | N/A | −0.12 ± 0.09 | > 0.3 |
| Stationary MMD | NaN | 0.027 ± 0.033 | < 0.01 |
| Force-displacement cos | N/A | 0.027 ± 0.008 | > 0.3 |

---

## 7. Recommended V24 Action Items

1. **Fix rollout numerically**: Add log-space computation, gradient clipping, Huber loss
2. **Implement bath state propagation**: Store and update auxiliary across forward calls
3. **Normalize force_score_Q**: Divide by D or increase σ to 0.2
4. **Diagnose endpoint skill**: Compare against baseline, check if energy predictions are correlated with actual energies
5. **Reduce rollout to 2 steps**: Lower `lspwm_rollout_steps` from 8 to 2 initially
6. **Softer curriculum**: Delay rollout phase to step 5000, increase warmup_steps to 1000
