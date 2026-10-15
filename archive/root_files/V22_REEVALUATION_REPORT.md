# V22 Hierarchical Mori World Model — Evaluation Report

**Date:** 2026-09-22  
**Artifact:** `artifacts/lspwm_v44_stage_b_v22_hierarchical_mori`  
**Checkpoint:** `checkpoint_latest.pt` (step 5170, epoch 5)  
**Status:** ❌ EFFECTIVELY FAILED — Training dominated by a formula bug; new architectural components untrained

---

## Executive Summary

The v22 run implements the hierarchical world model with explicit energy graph from the plan, but **4 critical code-level bugs** prevented most of the new architecture from contributing. The training was dominated by an exploding `force_score_Q_loss` (25k–152k), causing the total loss to be ~17,266 compared to ~1.0 for the meaningful signal from all other losses combined. The run completed 5170 steps (of planned 6000) and produces acceptable paired-skill (0.80), but the core novelties — coupling energy, FDT bath, residue conditional decoder, and self-consistency constraint — were essentially unused or untrained.

**Bottom line: this checkpoint is not usable.** Fix the bugs below, re-run, and re-evaluate.

---

## 1. Core Metrics vs. Baseline

| Metric | v27 (KDE CV8) | v22 (this run) | Delta | Notes |
|--------|---------------|----------------|-------|-------|
| `paired_skill` (final) | 0.835 | 0.804 | −0.031 | Same frozen JEPA backbone |
| `paired_skill` (peak) | — | 0.916 | — | At step 480, during Phase 1 |
| `lspwm_dynamic_nll` | 8.34 | 8.00 | −0.34 | Slight improvement, overdamped NLL |
| `lspwm_force_loss` | 0.041 | 0.042 | +0.001 | Identical |
| `lspwm_koopman_loss` | −5.0 (est.) | −4.88 | — | Similar |

**Verdict:** No measurable improvement over v27. The two runs are statistically indistinguishable on the legacy metrics.

---

## 2. Bug #1: `force_score_Q_loss` Formula Error (CRITICAL)

**Location:** `protein_jepa_frame/models/lspwm.py`, lines 877–903  
**Severity:** Blocks all meaningful training

### The Bug

The sliced score matching formula mixes up the noise term:

```python
# Line 902 — BUGGY CODE:
loss = loss + ((proj_noisy + (q_noisy * v).sum(-1) / (noise_std ** 2)).square() * residue_mask).sum()
```

The correct formula should use only the noise (ε = q_noisy − q_clean):

```python
# CORRECT:
noise = (q_noisy - q.detach())  # = ε, should be saved from line 890
loss = loss + ((proj_noisy + (noise * v).sum(-1) / noise_std**2).square() * residue_mask).sum()
```

### Quantitative Impact

- `noise_std = 0.05` → `noise_std² = 0.0025`
- The correct term `(ε·v)/σ²` has variance `1/σ² = 400`
- The buggy term `(q_noisy·v)/σ²` includes the full `q` (~O(1)) divided by 0.0025 → ~20,000
- This makes the loss 50× too large
- Multiplying by weight 0.3 gives contribution ≈ 17,265 to total loss
- All other losses sum to ~1.0 → **force_score_Q dominates training by 4 orders of magnitude**

### Training Impact

The model is effectively being trained only by a broken force_score_Q loss. The `score_noisy = unified_energy.force_q(q_noisy, ...)` computed at line 898 does compute the correct gradient through `UnifiedEnergyGraph`, so there IS some signal propagating through the energy graph. But the noise term in the loss is wrong, making the loss magnitude wild and the optimization unstable.

---

## 3. Bug #2: Force Direction & Amplitude Losses Masked Off

**Location:** `protein_jepa_frame/models/lspwm.py`, line 3620  
**Severity:** Per-residue force supervision disabled

### The Bug

```python
# Line 3620:
if (self.residue_force_direction_weight > 0
        and phase not in ("representation",)
        and future_latents is not None
        and future_latents.ndim == 4):
```

`self.residue_force_direction_weight` is set from **`config.model.lspwm_force_direction_weight`** (via jepa.py line 849-850), which defaults to `0.0` in `ModelConfig` (config.py line 329).

The YAML config sets this as a loss weight in `optimization` section:
```yaml
optimization:
  lspwm_force_direction_weight: 0.3
```

But the gate checks the model-level config, which is:
```json
"model.lspwm_force_direction_weight": 0.0
```

### Consequences

This `if` block is never entered, so the following losses are **always zero**:
- `lspwm_residue_direction_loss = 0.0` — hinge penalty on force-displacement alignment
- `lspwm_residue_amplitude_loss = 0.0` — amplitude matching of force to displacement
- `lspwm_residue_conditional_loss = 0.0` — conditional p(Q_local_{t+1} | Q_local_t, G_t, G_{t+1})

The `ResidueConditionalDecoder` is created but never receives any gradient.

### Fix

The gate at line 3620 should check `self.residue_force_direction_weight > 0` OR the optimization-section weight, or the YAML should set `model.lspwm_force_direction_weight: 0.3` directly.

---

## 4. Bug #3: `self_consistency_loss` is Identical to `force_score_G_loss`

**Location:** `protein_jepa_frame/models/lspwm.py`, lines 906–940  
**Severity:** Self-consistency constraint provides no additional signal

### The Bug

```python
# force_score_G_loss (line 854-874):
global_force = unified_energy.force_G(q, G, ...)        # = -∇_G A(Q,G)
kde_target = kde_mean_force(G, reference_G, bandwidth)  # = -∇_G log p(G)
return MSE(global_force - kde_target.detach())

# self_consistency_loss (line 906-940):
grad_A_g = ∇_G A_g(G)                    # part 1
grad_coupling = mean_i[∇_G φ(q_i, G)]    # part 2
# grad_A_g + grad_coupling = ∇_G A(Q,G) = global_force (by construction)
return MSE((-grad_A_g - grad_coupling) - kde_target.detach())
```

Both compute `MSE(-∇_G A(Q,G) − KDE_target)`. They are **mathematically identical**. The plan intended for `self_consistency_loss` to provide an orthogonal constraint between `A_g` and `φ`, but the implementation just recomputes the same quantity.

### Fix

The self-consistency loss should measure the **difference between a standalone `A_g(G)` force and the full `A(Q,G)` force**, e.g.:

```python
force_Ag_only = ∇_G A_g(G)                              # only global PMF
force_coupling = mean_i[∇_G φ(q_i, G)]                  # coupling correction
force_full = ∇_G A(Q, G)                                # total
kde_target = KDE(G)                                     # empirical

# Ag should explain the slow component, φ should explain the residual
L_consistency = MSE(force_Ag_only - kde_target)          # ← A_g alone vs KDE
              + MSE(force_full - force_Ag_only - force_coupling)  # ← decomposition check
```

---

## 5. Bug #4: MultiTimescaleBath Never Used for Dynamics

**Location:** `protein_jepa_frame/models/lspwm.py`, lines 3851–3907  
**Severity:** FDT bath trained but never contributes to dynamics

### The Problem

The `MultiTimescaleBath` with 16 modes (4 fast + 8 mid + 4 slow) is instantiated at line 3271 but:

1. **Not used in transition NLL:** The dynamics use `lspwm_dynamic_nll` which is the legacy overdamped transition — no bath coupling.
2. **Not used in rollout:** `lspwm_rollout_enabled: false` → no multi-step simulation.
3. **Diagnostics only:** The bath drift is computed at line 3871–3892 but only for logging, gated by `skew_operator is not None` (which is None because `reversible_dynamics_enabled: false`).

All bath diagnostics show `0.0`:
```
lspwm_dissipative_force_norm = 0.0
lspwm_reversible_force_norm = 0.0
lspwm_bath_drift_norm = 0.0
lspwm_total_drift_norm = 0.0
lspwm_drift_incr_cosine = 0.0
```

The `lspwm_mori_nll_weight` in config is also `0.0`, confirming the Mori NLL was never implemented.

---

## 6. What Actually Worked

Despite the bugs, some signals are visible:

| Component | Evidence | Assessment |
|-----------|----------|------------|
| JEPA backbone | `paired_skill` reaches 0.92 peak, 0.80 final | ✅ Frozen encoder works fine |
| `force_score_G_loss` | Decreases 842 → 0.21 | ✅ KDE PMF on G learns |
| `UnifiedEnergyGraph` | `energy_global` stabilizes at −0.06, `coupling_var` ≈ 0.007 | ⚠️ Weak but non-degenerate |
| `cv_vamp2_loss` | −4.0 to −5.0 | ✅ CV projection maintains structure |
| `stationary_loss` | 0.14–0.65 | ⚠️ Measures batch partition similarity, not true stationary MMD |
| `lspwm_dynamic_nll` | Decreases 12.5 → 8.0 | ✅ Overdamped transition learns |
| `gauge_norm` | Stable at ~0.19 | ✅ Gauge projector doesn't diverge |
| `global_force_norm` | 23 → 1.3 | ✅ Force converges to finite value |
| `energy_coupling_var` | 0.005–0.011 | ⚠️ Small but non-zero variance suggests coupling has some structure |

---

## 7. Training Dynamics

### Phase Progression

| Phase | Steps | Duration |
|-------|-------|----------|
| representation | 0–990 | 990 steps |
| thermodynamic_identification | 1000–2490 | 1490 steps |
| dynamics | 2500–4490 | 1990 steps |
| rollout | 4500–5170 | 670 steps (truncated) |

Training stopped at step 5170 (epoch 5 of 6 planned) — the run was either killed or crashed.

### Loss Composition at Step 5170

| Loss Term | Value | Weight | Contribution |
|-----------|-------|--------|-------------|
| `force_score_Q_loss` | 57,551.1 | 0.3 | **17,265** (99.99%) |
| `force_loss` | 0.042 | 1.0 | 0.04 |
| `force_score_G_loss` | 0.209 | 1.0 | 0.21 |
| `dynamic_nll` | 7.997 | 0.1 | 0.80 |
| `stationary_loss` | 0.301 | 0.02 | 0.006 |
| `self_consistency_loss` | 0.209 | 0.05 | 0.01 |
| `koopman_loss` | −4.875 | 0.05 | −0.24 |
| `cv_vamp2_loss` | −4.867 | 0.01 | −0.05 |
| `paired_delta_loss` | 0.015 | 0.02 | 0.0003 |
| All others | ~0 | — | **~0** |
| **TOTAL** | | | **17,266** |

---

## 8. Required Fixes for v23

### Fix 1: Correct `force_score_Q_loss` formula (MUST FIX)

```python
# In force_score_Q_loss(), line 902:
# OLD (buggy):
loss = loss + ((proj_noisy + (q_noisy * v).sum(-1) / (noise_std ** 2)).square() * residue_mask).sum()
# NEW:
noise = (q_noisy - q.detach())  # the epsilon perturbation
loss = loss + ((proj_noisy + (noise * v).sum(-1) / noise_std**2).square() * residue_mask).sum()
```

Alternatively, adopt the standard denoising score matching form:
```python
score_noisy = unified_energy.force_q(q_noisy, ...)
true_score_approx = -noise / noise_std**2  # Stein's identity
loss = ((score_noisy - true_score_approx)**2 * mask).sum(dim=-1).mean()
```

### Fix 2: Enable per-residue force supervision

In YAML config, set `model.lspwm_force_direction_weight: 0.3` (not just `optimization.lspwm_force_direction_weight: 0.3`).

### Fix 3: Implement distinct self-consistency loss

Rewrite `self_consistency_loss` to provide orthogonal constraint between `A_g(G)` and `φ(q_i, G)`:

```python
# Ag-only force vs KDE target
force_Ag = -torch.autograd.grad(E_global.sum(), G, create_graph=True)[0]
# Coupling contribution to force on G
force_phi = -torch.autograd.grad(E_coupling_sum, G, create_graph=True)[0]  
# Full force (should match KDE)
force_full = unified_energy.force_G(q, G, ...)
# Target
kde = kde_mean_force(G, reference_G, bandwidth)

# Triplet loss: Ag explains global, coupling explains residual
L_sc = ((force_Ag + force_phi - force_full)**2).mean()  # decomposition check
     + ((force_Ag - kde.detach())**2).mean()             # Ag baseline
     + ((force_phi)**2).mean() * 0.01                    # φ regularization
```

### Fix 4: Wire FDT bath into dynamics

The `MultiTimescaleBath` should contribute to the Mori transition NLL. This requires implementing `mori_transition_nll()` as described in the plan (Section "Bath Design"). Without this, the bath is dead code.

### Fix 5: Enable rollout with bath coupling

Set `lspwm_rollout_enabled: true` and `lspwm_rollout_steps: 16`. The rollout should use the Mori dynamics with bath coupling, not just overdamped.

---

## 9. Comparison to Plan Requirements

| Plan Requirement | Implemented? | Working? | Notes |
|-----------------|-------------|----------|-------|
| Hierarchical decomposition (Q_local → G) | ✅ | ✅ | Single-frame state works |
| UnifiedEnergyGraph with 4 terms | ✅ | ⚠️ | Present but trained by buggy loss |
| A_coupling(Q, G) | ✅ | ⚠️ | Small but non-degenerate signal |
| MultiTimescaleBath (FDT) | ✅ | ❌ | Instantiated but unused |
| DiagonalDiffusion | ✅ | ⚠️ | State-dependent, but never coupled to bath |
| GaugeProjector | ✅ | ✅ | Stable, non-diverging |
| ResidueConditionalDecoder | ✅ | ❌ | Masked by Bug #2, never trained |
| Mori transition NLL | ❌ | ❌ | Not implemented |
| Self-consistency loss | ✅ | ❌ | Identical to force_score_G (Bug #3) |
| Stationary MMD | ⚠️ | ⚠️ | Splits MD batch, no true rollout comparison |
| Rollout + semigroup | ❌ | ❌ | `rollout_enabled: false` |
| Curriculum with gradient in Phase 1 | ✅ | ✅ | `cv_vamp2` provides gradient |

---

## 10. Recommendation

**Do not use this checkpoint.** The training was dominated by Bug #1, and Bugs #2–4 prevented the novel components from contributing. The resulting model is essentially a v27 with a slightly degraded paired-skill (0.804 vs 0.835).

**For v23:**
1. Fix Bug #1 (force_score_Q formula)
2. Fix Bug #2 (force_direction weight wiring)
3. Fix Bug #3 (implement distinct self-consistency loss)
4. Fix Bug #4 (wire Mori bath into dynamics)
5. Enable rollout with bath coupling
6. Re-train from scratch with all fixes

The architecture itself (UnifiedEnergyGraph, coupling, MultiTimescaleBath, gauge) appears correctly structured — it's the training code that prevented it from working.

