# LSP-WM v28+ Improvement Plan

**Date:** 2026-09-20
**Branch:** `v21_thermo_helmholtz_mori` (pushed to GitHub)
**Current best model:** v27 (KDE force matching, CV=8, bw=1.0)

---

## Executive Summary

After thorough code, document, and memory review, I've identified the following root causes holding back the model:

1. **Force field is still degenerate** — KDE score-matching trains ∇U(q) to match the log-density gradient, but the learned direction is near-orthogonal to true dynamics. M1's OLS refit absorbs any residual signal, giving force_scale = 3.15e-7 (effectively zero).

2. **CV=8 is too small for dynamics** — KDE forces dimensionality ≤ 8, but protein conformational dynamics likely need more degrees of freedom. The trade-off between KDE statistical stability and dynamical expressivity is unresolved.

3. **Memory kernel is post-hoc, not learned** — the M1 ablation uses ridge-fitted linear AR kernels, not the learned Mori-Zwanzig bath. The native bath has kernel_norm=0.000477 but is never evaluated in a meaningful force+memory combination where both are active.

4. **Scale decoupling remains** — diffusion (~1.13e-7), force (~3e-7), and observed dynamics operate at incompatible scales, preventing the physical terms from contributing meaningfully.

---

## Phase 1: Fix the Fundamental Architecture (v28-v30)

### v28 — Joint Score+Dynamics Matching (HIGHEST PRIORITY)

**Problem:** KDE score matching trains ∇U to match ∇log p(q), but this only constrains the equilibrium distribution shape — NOT the dynamics direction. The force that produces the right steady state may be orthogonal to the force that drives actual transitions.

**Solution:** Replace pure KDE score matching with a **joint loss** that simultaneously matches:
1. **Score** (steady-state): ∇U(q) ≈ -kT · ∇log p(q) — KDE target
2. **Dynamics** (transitions): ∇U(q) · (q_{t+1} - q_t) ≈ observed displacement direction — cosine alignment

This is implemented as:

```
L_joint = L_score + λ · L_direction
```

where `L_direction = 1 - cos(∇U(q), q_{t+1} - q_t - memory_contribution)`, subtracting the memory kernel's contribution so the force learns what memory CANNOT explain.

**Implementation:**
- New loss function in `lspwm.py`: `joint_score_dynamics_loss()`
- Subtract fitted memory contribution from the target displacement before computing force alignment
- This makes the force learn the *residual* dynamics that memory doesn't capture

**Config:** `configs/lspwm_v44_stage_b_v28_joint_score_dynamics.yaml`
- Keep CV=8 for now (validate the approach first)
- `force_direction_weight`: 0.3 (start conservative)

### v29 — Lift the CV Dimension Constraint

**Problem:** CV=8 is forced by KDE curse of dimensionality. But sliced score matching (already in config as `sliced_score_matching_enabled`) scales to >16D.

**Solution:** Enable sliced score matching with CV=32 or CV=64:

```
config.sliced_score_matching_enabled = True
config.sliced_score_matching_noise = 0.05
config.learned_cv_dim = 32  # up from 8
```

Sliced score matching projects ∇U onto random directions and matches the directional derivative, avoiding the pairwise distance computation that kills KDE in high dimensions. This is Rev-6 in the LSPWMConfig, already implemented but never used.

**Advantages:**
- CV=32 captures more dynamical degrees of freedom
- No bandwidth selection needed
- Scales linearly with dimension, not quadratically
- Better representation for K1 folding kinetics

**Config:** `configs/lspwm_v44_stage_b_v29_sliced_score_cv32.yaml`

### v30 — Learn the Memory Kernel End-to-End

**Problem:** The M1 evaluation uses ridge-fitted linear AR kernels. The native Mori-Zwanzig bath (GaugeFixedBath) exists but its kernel_norm is small (0.000477) and it's never properly trained because:
1. Bath relaxation times (4000-128000 ps) exceed training window
2. The auxiliary state has no effective training signal
3. Memory training is gated behind physical_calibration_enabled which is usually False

**Solution:** Train the memory kernel end-to-end during Stage B:

1. **Enable memory kernel training in the dynamics phase:**
   - Set `history_closure_enabled = True`
   - Use `ProjectedForceClosure` (already implemented in lspwm.py:521) with bounded, short-time-gated correction
   - Train the memory kernel matrix directly (replace the fixed GaugeFixedBath relaxation times with learned parameters)

2. **Trainable bath times:**
   - Replace `register_buffer` relaxation rates with `nn.Parameter` (log-space)
   - Add a spectral radius constraint to ensure stability
   - Train coupling_scale as a parameter

3. **Auxiliary state initialization from context:**
   - The `auxiliary_initializer` (lspwm.py:2683-2689) already exists — a 3-frame context → auxiliary state MLP
   - Enable it during training (currently only used during rollout)

**Config:** `configs/lspwm_v44_stage_b_v30_learned_memory.yaml`

---

## Phase 2: Structural Innovations (v31-v33)

### v31 — Stationary-Preserving Memory Projection

**Problem (from ICLR_PAPER_PLAN.md):** Memory kernel changes BOTH dynamics and steady-state. We need to decouple them so memory only affects time correlations, not the equilibrium distribution.

**Solution:** Project the memory correction onto the orthogonal complement of ∇U:

```
m_⊥(q, h_t) = m(q, h_t) - α · ∇U(q)
```

where `α = ⟨m, ∇U⟩ / ‖∇U‖²` removes the component parallel to the potential gradient.

This is the **key methodological innovation** from the ICLR paper plan. The projected memory term, by construction, cannot change the steady-state distribution (since its expectation under π vanishes when projected orthogonal to ∇log π).

**Implementation:**
- New method `projected_history_correction()` in `lspwm.py`
- Optional `stationary_preserving_projection` flag with weight
- Diagnostic: `projection_leakage = ‖⟨m_⊥, ∇U⟩‖` (should be ~0)

### v32 — State-Dependent Diffusion with Physical Constraints

**Problem:** Diffusion is constant (1.13e-7) across all states. Real protein dynamics have state-dependent noise — folded states are rigid (low diffusion), unfolded states are flexible (high diffusion).

**Solution:** Enable `state_dependent_diffusion_enabled = True` with:
1. Physical constraint: diffusion ∝ local density (high in sparse regions, low in dense regions)
2. FDT consistency check: verify that the learned D(q) satisfies Einstein relation with mobility

### v33 — Multi-Timescale Conditioning

**Problem:** The model is trained at 348K and evaluated at 348K. The JEPA latent space already encodes temperature implicitly, but the dynamics model doesn't exploit this for temperature extrapolation.

**Solution:** Explicit temperature conditioning in the potential and diffusion:
1. Train with temperature as input (already supported in code via `temperature_k` parameter)
2. Add multi-temperature training data (348K, 379K, 413K available in `manifests/tmjepa_v3.temperature_cv_*.jsonl`)
3. Verify that the potential correctly scales with temperature (∇U ∝ kT)

This unlocks the temperature extrapolation experiment (P4 from ICLR plan).

---

## Phase 3: Evaluation and Benchmarking Improvements

### E1 — Fix the 2×2 Ablation to Use Learned (Not Fitted) Components

**Current:** M_Force and M_Memory use `fitted_ridge` kernel — a post-hoc linear fit that has nothing to do with the learned model.

**Fix:** 
- `M_Force` should use the **learned** ∇U(q) directly (already does, but force_scale=3e-7 makes it irrelevant)
- `M_Memory` should use the **learned** memory kernel from the native bath or ProjectedForceClosure
- Add a new ablation cell `M_Learned` that uses both learned force AND learned memory WITHOUT OLS refit

### E2 — Add Ground-Truth Recovery Test (Synthetic GLE)

**From ICLR plan:** The strongest defense against "how do we know your model works?" is showing parameter recovery on known ground truth.

The synthetic GLE framework already exists in:
- `scripts/generate_synthetic_gle.py` — generates data from known A(z), D, K(t)
- `scripts/evaluate_synthetic_gle.py` — evaluates recovery

Run this to validate that the architecture CAN recover known parameters before claiming it does on real proteins.

### E3 — Open the Temperature Blind Test

`manifests/tmjepa_v3.temperature_test.jsonl` — 20 proteins at 450K, never evaluated. This is a one-shot blind test that can provide strong evidence for physical consistency.

---

## Priority Order and Timeline

| Priority | Experiment | Est. Time | Risk | Impact |
|----------|-----------|-----------|------|--------|
| **P0** | v28: Joint Score+Dynamics | 1 day | Low | 🔴 Critical — fixes force degeneracy |
| **P0** | E2: Synthetic GLE Recovery | 0.5 day | Low | 🔴 Critical — validates method |
| **P1** | v29: Sliced Score CV=32 | 1 day | Medium | 🟡 High — lifts CV constraint |
| **P1** | v30: Learned Memory | 2 days | Medium | 🟡 High — end-to-end training |
| **P2** | v31: Stationary-Preserving | 2 days | High | 🟢 Very High — key innovation |
| **P2** | E1: Fix Ablation | 0.5 day | Low | 🟡 High — honest evaluation |
| **P3** | v32: State-Dep Diffusion | 1 day | Low | 🟢 Medium |
| **P3** | v33: Multi-Temperature | 1 day | Medium | 🟢 Medium |
| **P4** | E3: Temperature Blind Test | 1 day | Low | 🟢 High — one-shot only |

**Total:** ~10 days of focused work for the full pipeline.

---

## Immediate Next Step: Launch v28

v28 is the highest-priority experiment because it directly attacks the force degeneracy that has persisted through ALL versions (v5-v27). The key insight is:

> KDE score matching learns the correct equilibrium distribution, but the force direction that maintains that distribution may be orthogonal to the force that drives actual transitions. We need to jointly constrain both.

### v28 Config Summary

```yaml
# Based on v27 but with joint score+dynamics loss
lspwm_learned_cv_dim: 8  # keep CV=8 for v28
lspwm_mean_force_bandwidth: 1.0
lspwm_force_weight: 1.0
lspwm_force_direction_weight: 0.3  # NEW
sliced_score_matching_enabled: false  # save for v29
# Copy rest from v27
```

### Success Criteria for v28

1. `M_Force` prediction_mse differs from `M_None` (statistically significant)
2. `force_scale > 0.01` (not 3e-7)
3. `degenerate_zero_force = false` in M1 evaluation
4. Force direction cosine with observed displacement > 0.1 (not ~0.008)

---

## Appendix: What We've Learned from v5-v27

### What Works
- **JEPA representation (256D):** Markov-sufficient, kinetically informative, beats tICA+MSM on K1 at intermediate/long horizons
- **Memory kernel:** Even a simple ridge-fitted linear AR kernel significantly improves innovation whiteness and stationary JSD
- **KDE with CV≤8:** Stable score matching that learns the equilibrium distribution shape
- **4-phase curriculum:** representation → thermo_id → dynamics → rollout is effective

### What Doesn't Work
- **Force field alone:** Never shown benefit over M_None in any version
- **KDE in high dimensions:** CV≥16 collapses, CV≥32 completely fails
- **Native bath:** kernel_norm ~1e-4 is too small; bath times don't match training window
- **Physical calibration:** mobility_scale/free_energy_scale are structurally unidentifiable from NLL

### The Critical Missing Piece
The force field needs a **dynamics-direction signal** — knowing the equilibrium distribution shape is not enough. The joint score+dynamics loss (v28) is the most direct path to solving this.