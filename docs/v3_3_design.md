# V3.3 Dense Standardized Horizon Identification

Date: 2026-08-27

## Diagnosis

V3.2 was the first version in which the correct horizon beat a shuffled
horizon at all three selected buckets, but the advantage was only
`0.09%/0.28%/0.49%`. Two implementation choices limited that signal:

1. The 12 closed-state channels have very different natural delta scales, so
   raw MSE and raw ranking gradients are dominated by high-variance channels.
2. Only three horizons (`48/96/192`) supervise the time-conditioned trunk,
   leaving sparse support for learning a smooth time dependence.

The V3.2 ranking implementation also compared correct and wrong predictions
over different valid-token supports in some masked batches. V3.3 fixes that
comparison before increasing the ranking weight.

## Changes

- Use dense non-overlapping horizons `48/64/96/128/192` for every sampled
  context.
- Estimate one 12-channel delta standard deviation from mdCATH train only,
  pooled across all five horizons. Do not normalize separately by horizon.
- Normalize state, persistence, correct-horizon and wrong-horizon residuals by
  the frozen train-only scales.
- Compare correct and wrong predictions on exactly the same valid local-token
  intersection.
- Increase the normalized ranking margin to `0.05` and its loss weight to
  `1.0`.
- Preserve all V3.2 representation heads and initialize from its selected
  step-1500 checkpoint. Only the closed local operator remains trainable
  through the active objectives.
- Save checkpoints at steps `500/1000/1500/2000`.

For prediction `p`, target `y`, persistence state `x`, channel scale `s`, and
valid support `M`, the principal losses are:

```text
L_state = mean_M(((p - y) / s)^2)
L_rank  = mean_pairs(max(0, e_correct - e_wrong + 0.05))
```

The scale is used only as a loss/evaluation unit conversion. Predictions and
reported raw physical MSE remain in the original closed-state units.

The frozen statistics were estimated from 500 mdCATH train contexts, with all
five horizons for each context (`10,000` valid segment deltas). Channel scales
range from `0.00507` to `0.21938`, a `43.3x` spread. The full global and
per-horizon audit is stored at
`artifacts/phase3_v3_3_dense_standardized_2k/closed_local_state_scale_train.json`.

## Controls And Gate

Evaluate every saved checkpoint on held-out mdCATH with group-aware controls:

- shuffle history across context groups while preserving horizon;
- shuffle horizon within the same context group;
- report raw and train-standardized MSE;
- report persistence-relative skill for every horizon.

The initial diagnostic gate requires the correct horizon to beat the shuffled
horizon by at least `2%` in normalized MSE at every horizon. The promotion
target remains `5%`, together with positive normalized persistence-relative
skill at every horizon. Training loss alone cannot promote the model.

Semigroup, Gaussian and rollout losses remain disabled until this gate passes.
If the 2% gate fails, the next change must alter the state or operator rather
than extending this run beyond 2k.

## Result

The 2k run completed and all four checkpoints were evaluated with the same
held-out manifest, seed, 100 pairs per horizon, and group-aware controls used
for V3.2. Step 2000 had positive normalized persistence-relative skill at all
five horizons: `0.60%/0.81%/1.22%/1.47%/1.90%`. Its normalized correct-versus-
shuffled horizon advantage was only `-0.06%/0.16%/0.21%/0.21%/0.16%`.

Step 1500 was closest to the minimax horizon criterion, but its worst horizon
advantage was still `-0.03%`. No checkpoint passed the 2% diagnostic gate.
On the common `48/96/192` horizons, even step 2000 raw persistence skills
(`0.64%/0.83%/1.41%`) were below selected V3.2
(`0.83%/1.10%/1.91%`). V3.3 is therefore not promoted and V3.2 remains the
closed-state time baseline.

The training ranking loss stayed near its `0.05` margin throughout the run,
while shuffled-history normalized MSE was about `8.5` versus roughly `1.0`
for correct history. The operator uses structural history, but the five
horizon predictions remain nearly interchangeable. More training on this
objective is not justified.
