# Current framework supervision report

Date: 2026-10-07

## Verdict

The current full ABC pipeline has **not** completed successfully.

| Component | Status | Evidence |
|---|---|---|
| Stage A | Engineering-complete | `full_abc_stage_a` has checkpoints through step 230000; its log continues to step 231800 |
| Stage B v1 | Complete but failed quality gate | 9000/9000 steps; semigroup is exactly 0.0 throughout; autonomous rollout diverges 100% |
| Stage B v2 | Incomplete and failed interim quality gate | checkpoint through step 2500, log through ~2870 of 25000 planned; stochastic Q norm grows 6.69 → 23.4 on step 1 and 101 by step 10 |
| Stage C full | Not trained | `full_abc_stage_c` contains only config and run manifest; no checkpoint |
| Full ABC evaluation | Not complete | existing quick evaluation has invalid rollouts and NaN temperature summaries |

## Stage B diagnosis

The v1 failure was not a normal under-training fluctuation. Its `|F|` was about
`3e-4`, while FDT noise was about `0.59` per dimension, so the energy surface
collapsed toward a nearly flat landscape and the stochastic process performed
an unbounded random walk. The v1 semigroup term was not active despite being
listed in the intended experiment configuration.

The v2 force-floor experiment did activate semigroup loss (`~0.04–0.12`) and
raised the mean force to about `0.276`, but the stochastic rollout remained
unstable. A deterministic rollout shrank Q, while the stochastic rollout
expanded Q from `6.69` to `23.4` after one step and `101` after ten steps.
Therefore v2 is not a passing energy landscape.

## Fixes applied

1. `lspwm_diffusion_scale` is now used as the physical mobility scale instead
   of being silently ignored. Fresh models start at the configured scale.
2. `lspwm_coupling_scale` is now used for bath coupling instead of a hard-coded
   `0.01`. This is important because bath forcing was otherwise multiplied by
   long physical horizons without the configured scale.
3. Stage B now exposes `q_next` and includes an optional observed-displacement
   force-direction loss, preventing the force-floor term from producing an
   arbitrary constant-norm vector field.
4. Stage B now supports checkpoint resume; previously the v2 interruption
   could not be continued through the Stage-B entry point.
5. The energy graph force helper now uses the same temperature-aware forward
   path as the main energy calculation; its old private path had stale input
   dimensions and ignored temperature.

## Verification

- Full test suite: **280 passed**, 62 warnings.
- Gate 2 / synthetic / Stage C rollout subset: **26 passed**.
- Repaired Stage B real-data smoke: completed two steps without NaN/Inf.
- Resume smoke: resumed checkpoint at step 2 and exited cleanly at the target.
- Fresh diffusion check: `D ~= 1e-6` for the default configuration.
- Temperature-aware force check: finite forces and different energies/force
  norms at 320 K and 413 K.

## Required next run

The old v2 checkpoint must not be promoted: its state was trained under the
old absolute-D semantics. The repaired configuration is
`configs/full_abc_stage_b_v3_repaired.yaml`. Run Stage B v3 from the completed
Stage A checkpoint, then evaluate autonomous stochastic rollouts before
starting Stage C. Stage C should only be launched after Stage B satisfies:

- finite rollouts at short and long horizons;
- no systematic Q-norm growth;
- semigroup loss is nonzero during training and decreases or stabilizes;
- force direction is aligned with observed displacement, not only above a
  magnitude floor;
- temperature intervention changes both mobility and energy/force response.
