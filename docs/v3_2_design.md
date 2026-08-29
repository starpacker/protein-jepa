# V3.2 Counterfactual Multi-Horizon Closed Local Operator

Date: 2026-08-26

## Evidence

V3.1 learned useful local physical prediction and improved MSM/linear R2, but
horizon shuffling barely changed its MSE. A held-out target audit showed real
but weak horizon separation: contact retention means were
`0.586/0.565/0.553`, RMSD means were `0.456/0.507/0.520`, and pairwise horizon
mean distances were only `0.08-0.12` pooled standard deviations.

## Changes

- Every sampled context is paired with all horizons `48/96/192`.
- A `context_group_index` preserves counterfactual groups through collation.
- Each sequence segment is encoded as a closed 12-dimensional state containing
  contact density, torsion sin/cos, Rg, local-distance moments, center distance
  and contact order.
- A FiLM-modulated shared trunk and horizon-specific residual experts predict
  the future state in the same space.
- A ranking loss requires the correct-horizon prediction to beat predictions
  from the other horizons of the same context.
- V3.1 champion, metastable and local-token parameters remain unchanged.
- Checkpoints are saved every 500 steps through the 2k gate.

## Gate

Continue only if closed-state prediction beats persistence at every horizon
and correct-horizon MSE is lower than shuffled-horizon MSE. The 500/1000/1500/
2000 checkpoints are selected using held-out validation, not training loss.
Semigroup, Gaussian and long rollout objectives remain disabled until this
one-step closed-state gate passes.

## Result

The 1500-step checkpoint was selected. Closed-state persistence-relative skill
at horizons `48/96/192` was `0.83%/1.10%/1.91%`, and correct-horizon prediction
was slightly better than shuffled horizon at all three buckets. The horizon
advantage was only `0.09%/0.28%/0.49%`, so the required 5% gate was not met.
Training beyond 1500 regressed the horizon-48 control. V3.2 is retained as the
first positive closed-state time diagnostic, not a promoted rollout model.
