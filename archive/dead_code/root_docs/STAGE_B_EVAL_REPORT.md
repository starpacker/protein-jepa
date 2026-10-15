# Stage B Full Training Evaluation Report

## Training Summary

| Metric | Value |
|--------|-------|
| Config | `full_abc_stage_b_from_scratch.yaml` |
| Steps | 8990/9000 (epoch 30 — process exited at 99.9%, checkpoint saved) |
| GPU time | ~5 hours |
| Init checkpoint | `full_abc_stage_a/checkpoint_latest.pt` |
| transition_nll | 27 → 13 (−52%) |
| rollout_mse | 11 → 0.007 (−99.9%) |
| energy_reg | → 0 by step ~200 |
| semigroup | **0.0** throughout (never activated) |

## Evaluation Results (n=100 val samples, all 1b4uB00, L=256)

### Core Finding: 100% Divergence Rate

**Every single autonomous rollout diverges.** The model produces NaN/inf on all 100
validation trajectories. The failure occurs in step 1 (|Q| 4.5 → 29) and hits
infinity by step ~41.

### Root Cause: Flat Energy Landscape

```
|Q|:       4.5 (initial) → 3.0 (oracle target) → 29 (step 1!) → nan (step 41)
|F|:       ~0.0003  (mean per-residue force magnitude)
noise/step: ~0.59   (FDT noise per dimension, dt=100ps)
ratio:     noise is 10,000× larger than drift
```

The model learned to minimize teacher-forced NLL by driving forces → 0, making
the energy landscape essentially flat. FDT noise then causes an unbounded
random walk — the exact opposite of what a Langevin model should do.

### What Works

| Check | Status | Detail |
|-------|--------|--------|
| Q encoding | ✓ | |Q| ≈ 4.5, stable, no NaN in encoding |
| Temperature → Energy | ✓ | E varies from −0.10 (320K) to +0.05 (413K) |
| Temperature → Diffusion | ✓ | D scales 0.001737 → 0.002242 (+29% over 93K span) |
| Energy minimum exists | ✓ | Minimum near 0.5× Q scale |

### What's Broken

| Problem | Evidence | Severity |
|---------|----------|----------|
| Flat energy landscape | |F| = 3e-4 — zero restoring force | **Fatal** |
| Noise/drift ratio 10,000× | Random walk dominates completely | **Fatal** |
| 100% rollout divergence | All 100 samples NaN/inf | **Fatal** |
| Semigroup loss always 0.0 | `step_ps` not in scope (bug in training code) | Moderate |
| Single protein eval | All 100 samples from 1b4uB00 | Known |

## Training Loss Interpretation

The NLL reduction from 27→13 was achieved by the model learning to predict
`q_next ≈ q_current` (small drift + noise matching), NOT by learning a
meaningful potential energy surface. The teacher-forced loss rewards small
one-step changes, which is optimal when the model's forces are negligible.

The rollout MSE reduction from 11→0.007 is similarly misleading — with
`stochastic=False`, deterministic rollout produces near-identity due to
negligible forces, and `q_target ≈ q0` after 4 steps.

## Immediate Next Steps

1. **Fix force magnitude**: Add explicit force-norm target (e.g., from slice
   matching or KDE). Current force is ~1e-4 — needs to be at least 0.1 to
   overcome FDT noise.

2. **Add energy regularization**: Penalize `|F| < threshold` to prevent
   flat-landscape collapse.

3. **Train with stochastic rollouts**: The deterministic-only rollout
   (`stochastic=False`) never exposes the model to FDT noise during training.

4. **Fix semigroup loss**: Bug where `step_ps` variable is undefined in
   semigroup section of `training/stage_b.py` line 236.

5. **Multi-protein eval**: Stratified sampling across all validation proteins.
