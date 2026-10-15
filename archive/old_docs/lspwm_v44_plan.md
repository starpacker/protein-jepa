# LSP-WM V4.4 Plan

## Objective

Build a reliable long-horizon dynamics model from a frozen, kinetically
informative latent state. V4.4 treats thermodynamic consistency and predictive
accuracy as separate promotion gates.

## Training design

1. **Stage A: kinetic representation**
   - Train paired endpoint prediction at 24, 48, and 96 frames.
   - Retain sequence-level VAMP, MSM, geometric, and latent-variance terms.
   - Keep all rollout losses disabled.

2. **Stage B: single-step thermodynamic fitting**
   - Freeze context encoder, temporal encoder, EMA teacher towers, and learned
     LSP projection.
   - Fit the world model only with the observed paired endpoint transition.
   - Optimize exact overdamped transition NLL and conservative KDE mean-force
     matching.
   - Keep rollout training and Gaussian ensemble NLL disabled.

The new `freeze_target_encoders` contract prevents a frozen-latent stage from
silently changing its target representation through EMA updates. Configuration
validation rejects any nonzero rollout weight when rollout training is off.

## Promotion gates

The primary dynamics gate is `evaluate_lspwm_endpoint.py`, which reports
deterministic endpoint skill, ensemble energy skill, transition NLL, and MSM
state accuracy against the observed endpoint at each horizon. Physical checks
remain secondary gates: diffusion PSD, diffusion conditioning, and empirical
FDT sampling residual.

The old multi-step rollout evaluator is retained only as a diagnostic. It does
not observe intermediate target frames and cannot be used as evidence of a
correct transition path.

## Launch

```bash
/liuyutian/protein_jepa_frame/scripts/run_lspwm_v44_single_step.sh
```

Artifacts are written to `artifacts/lspwm_v44_stage_a_representation` and
`artifacts/lspwm_v44_stage_b_single_step_thermodynamics`.
