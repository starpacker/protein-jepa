# V5 Contextual Adaptive Dynamics: Formal 2K Result

Date: 2026-09-01

## Run

- Config: `configs/phase5_v5_contextual_adaptive_2k.yaml`
- Initialization: Phase-1 multi-lag JEPA step 20000
- Training: 2000 steps on mdCATH, all five horizons per context
- Validation checkpoints: 250-step intervals
- Device: NVIDIA A800 GPU 1
- Training and all eight checkpoint evaluations completed successfully

## Promotion Decision

**V5 does not pass the promotion gate.** No checkpoint is promoted.

The local reversible generators remained numerically valid, and training loss
improved substantially. These gains did not produce held-out history-dependent,
horizon-identifiable dynamics.

| Step | Feature skill mean | History advantage mean | Correct vs fixed mean | Physical history mean | Physical horizon mean | Effective slots |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | -5.45% | 0.008% | -2.14% | 0.014% | 0.008% | 15.42 |
| 500 | -3.51% | 0.025% | -1.14% | 0.010% | 0.010% | 15.32 |
| 750 | -2.58% | 0.038% | -0.76% | -0.0005% | 0.005% | 15.26 |
| 1000 | **-2.00%** | 0.037% | **-0.64%** | 0.009% | 0.008% | 15.22 |
| 1250 | -2.05% | 0.038% | -0.66% | 0.016% | 0.008% | 15.12 |
| 1500 | -2.30% | 0.035% | -0.73% | **0.032%** | 0.007% | 15.03 |
| 1750 | -2.12% | **0.038%** | -0.70% | 0.026% | **0.008%** | 15.01 |
| 2000 | -2.15% | 0.034% | -0.71% | 0.027% | 0.008% | 15.02 |

At step 2000, the worst-horizon feature skill was `-3.28%` and the worst
correct-versus-fixed skill was `-1.83%`. Local detailed-balance residuals were
below `2.2e-13`, so the failure is not numerical instability.

Physical zero-motion skill reached about `87%`, but this is not evidence of
learned dynamics. A history-blind conditional mean can perform well on these
low-dimensional observables. The decisive controls remained tiny: physical
history advantage was about `0.027%` and physical horizon advantage about
`0.008%` at step 2000.

## Diagnosis

1. The observable decoder can predict the conditional mean directly from the
   history embedding and largely ignore the rolled-out posterior.
2. Context-specific slots remained effectively all active. The sigmoid gate
   penalty reduced total gate mass but did not create a sparse state basis.
3. The future state target is still produced by the learned V5 projection.
   Joint optimization can contract that feature space faster than the
   generator learns identifiable dynamics.
4. Horizon-difference regression is dominated by small target differences and
   does not force correct-horizon predictions to beat wrong-depth controls.

## Next Structural Change

Do not extend this run or tune its loss weights. The next version should:

- remove the direct history-to-observable shortcut; decode observables only
  from rolled-out state plus static protein condition;
- replace sigmoid gates with discrete sparse routing such as hard-concrete or
  top-k slots, including an explicit capacity budget;
- stop target contraction with a separately initialized/frozen physical target
  representation or an EMA V5 target inference network;
- train wrong-horizon ranking in physical observable space from the first step;
- include matched-history counterfactual pairs where endpoint geometry is held
  fixed but incoming motion differs.

All checkpoint controls are in
`artifacts/paper_eval/phase5_v5_contextual_step*_controls.json`.
