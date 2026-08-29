# Phase 2 Execution Status

Date: 2026-08-24

## Implemented

- Evaluation inherits checkpoint frame/window stride and reports the actual
  saved-frame clip span.
- Default forecast horizons begin at the first lag with non-overlapping input
  windows.
- Quick VAMP/linear records are selected round-robin across proteins.
- VAMP kinetic normalization removes the constant mode.
- VAMP paired bootstrap comparisons use proteins, not correlated folds.
- MSM ridge regularization uses a nested train-trajectory holdout and reports
  balanced accuracy and macro-F1.
- Multi-lag JEPA predicts from the final visible context state and the masker
  reserves the maximum requested horizon.
- Training logs per-lag loss and pair count. Lags 1, 2, 4, 8, and 12 are all
  active in the real-data smoke and T0 run.
- Temperature and dataset identity pass from manifest through collate into an
  optional thermodynamic state condition.
- Hierarchical sampling supports explicit dataset weights, then samples
  protein, trajectory, and clip separately.

## Honest E0 Baseline

Artifact: `artifacts/paper_eval/phase2_e0_honest_baseline`

Protocol: 24 frames, frame stride 2, window stride 8, first non-overlap lag 6,
ten protein-stratified trajectories, and nested MSM lambda selection.

| Metric | Phase 1 checkpoint | Strong comparison |
| --- | ---: | ---: |
| VAMP kinetic norm, lag6 dim5 | 0.4065 | centered CA 0.4291 |
| MSM accuracy | 0.4328 | majority 0.4341 |
| MSM balanced accuracy | 0.3882 | - |
| MSM macro-F1 | 0.3304 | - |
| Linear dynamics R2, lag6 | -0.8443 | geometry stats -0.1836 |

The checkpoint is significantly better than the random encoder on lag-6 VAMP
(mean paired protein delta 0.0452, 95% bootstrap CI 0.0145 to 0.0727), but it
does not beat centered CA and has no non-overlapping linear forecast skill.

## T0 Training Outcome

- Config: `configs/phase2_t0_physical_conditioned.yaml`
- Output: `artifacts/phase2_t0_physical_conditioned`
- Launcher log: `artifacts/phase2_t0_physical_conditioned/launcher.log`
- Training log: `artifacts/phase2_t0_physical_conditioned/train.jsonl`
- Device: GPU 0
- Status: completed 20,000 steps without NaN, OOM, or runtime errors

T0 trained from scratch with physical time, temperature/domain conditioning,
75/25 mdCATH/ATLAS source weights, hierarchical sampling, and all five repaired
lag objectives. All lags retained four valid pairs per batch in the sampled
training logs.

Honest evaluations:

| Checkpoint | VAMP lag6 dim5 | MSM accuracy | Linear lag6 R2 |
| --- | ---: | ---: | ---: |
| Phase 1 E0 | 0.4065 | 0.4328 | -0.8443 |
| T0 10k | 0.3731 | 0.3740 | -0.5330 |
| T0 20k | 0.3991 | 0.3719 | -0.5923 |

T0 improves the failed non-overlapping linear diagnostic but remains negative,
regresses VAMP and MSM, and does not beat geometry baselines. Neither T0
checkpoint is promoted. More steps on this objective are not justified.

Completed-run audit commands:

```bash
pgrep -af phase2_t0_physical_conditioned.yaml
tail -f artifacts/phase2_t0_physical_conditioned/launcher.log
nvidia-smi
```

Promotion requires beating E0 on protein-paired VAMP, nested MSM balanced
accuracy, and non-overlapping lag-6 forecast skill. Positive lag-6 skill is a
hard requirement; overlapping lag-1 metrics are not used for promotion.

## Active A1 mdCATH-Only Training

- Config: `configs/phase2_a1_mdcath_causal_geometry.yaml`
- Output: `artifacts/phase2_a1_mdcath_causal_geometry`
- Dataset: mdCATH only, hierarchical protein/trajectory/clip sampling
- Device: GPU 0
- Budget: 20,000 steps from scratch

A1 adds chirality-sensitive backbone torsions, local-frame pair orientations,
causal temporal attention, frame-difference motion features, physical time,
temperature conditioning, and state variance/covariance control. It does not
use ATLAS or dataset-domain conditioning.

The real-data batch-4 start completed with about 9.9 GB GPU memory and all five
lags reporting four valid pairs. Evaluate 10k and 20k with the E0 honest
protocol; do not use training loss for promotion.

### A1 10k interim evaluation

Artifact: `artifacts/paper_eval/phase2_a1_10k_honest_eval`

| Model | VAMP lag6 dim5 | MSM accuracy | MSM balanced | Linear lag6 R2 |
| --- | ---: | ---: | ---: | ---: |
| Phase 1 E0 | 0.4065 | 0.4328 | 0.3882 | -0.8443 |
| T0 10k | 0.3731 | 0.3740 | 0.3228 | -0.5330 |
| A1 10k | 0.3723 | 0.3393 | 0.2965 | -0.4045 |

A1 improves the non-overlapping linear diagnostic but regresses VAMP and MSM.
Its VAMP score is significantly below the same-architecture random encoder:
mean paired protein delta -0.0334, 95% bootstrap CI -0.0495 to -0.0176. The 10k
checkpoint is not promoted. Training was resumed to retain the preplanned 20k
diagnostic, not because the 10k result met a gate.

A1 was subsequently stopped at approximately 12.8k after the 10k hard failure;
continuing to 20k was judged lower-value than testing a gradient-decoupled
architecture.

## Active A2 Split-State/Motion Training

- Config: `configs/phase2_a2_mdcath_split_state_motion.yaml`
- Output: `artifacts/phase2_a2_mdcath_split_state_motion`
- Dataset: mdCATH only
- Device: GPU 0

A2 keeps `frame_states` as the formal slow/conformational representation and
introduces an independent causal `motion_states` stream. Multi-lag prediction
uses the motion stream; its spatial input is stop-gradient, so the lag objective
cannot update the slow temporal transformer or spatial encoder. Temperature is
provided only to the lag transition predictor. A1's state variance/covariance
weights are all zero.

The evaluator exposes `motion` and `state_motion` as diagnostics, while `state`
remains the promotion readout. Full tests pass and the real-data batch-4 start
is stable with all five lags reporting four valid pairs.

### A2 training-loss check and 5k gate

As of 2026-08-25 08:59 UTC, the original A2 process was still running:

- Original PID: `684643`
- Last observed training step before the watcher was launched: about `3480`
- Loss curve artifact: `artifacts/phase2_a2_mdcath_split_state_motion/loss_curve_latest.png`
- Loss curve summary: `artifacts/phase2_a2_mdcath_split_state_motion/loss_curve_latest.summary.json`

The curve shows fast early optimization, but it should not be interpreted as a
solved dynamics model. `state_jepa_loss` is low, while `lagged_state_jepa_loss`
has nontrivial post-warmup spikes and the unweighted semigroup diagnostic rises
through the first 0.6 epoch. The next honest decision point is therefore a 5k
checkpoint evaluation, not continued reliance on training loss.

Checkpointing was changed from a hard-coded 10k interval to configurable
`checkpoint_every_steps`; A2 is configured for 2k step checkpoints after resume.
Because the already-running Python process cannot hot-load that code, watcher
PID `688824` was launched:

```bash
scripts/phase2_a2_eval_resume_at_5k.sh 684643
```

The watcher waits for the old process to write `checkpoint_latest.pt` at the 5k
epoch boundary, snapshots it as `checkpoint_step5000.pt` if the checkpoint step
is 5000, pauses the old process, runs the honest quick evaluation into
`artifacts/paper_eval/phase2_a2_5k_honest_eval`, terminates the old process, and
then resumes training with the new 2k checkpoint interval.

### A2 5k interim evaluation

Artifact: `artifacts/paper_eval/phase2_a2_5k_honest_eval`

| Model | VAMP lag6 dim5 | MSM accuracy | Linear lag6 R2 |
| --- | ---: | ---: | ---: |
| Phase 1 E0 | 0.4065 | 0.4328 | -0.8443 |
| A1 10k | 0.3723 | 0.3393 | -0.4045 |
| A2 5k | 0.3850 | 0.4974 | -0.7574 |
| A2 random encoder | 0.4147 | n/a | n/a |
| geometry_stats | 0.3729 | n/a | -0.1836 |
| centered_ca | 0.4291 | n/a | -2.5174 |

A2 at 5k improves MSM accuracy relative to E0/A1, but the state readout fails
the main promotion gates: VAMP is below E0, below the same-architecture random
encoder, and below centered-CA; non-overlapping lag-6 linear forecast skill is
still negative and worse than the geometry-stat baseline. This checkpoint is not
promoted.

The resumed A2 process is running from `checkpoint_latest.pt` with 2k step
checkpointing active. The refreshed aggregate and per-horizon curves are:

- `artifacts/phase2_a2_mdcath_split_state_motion/loss_curve_latest.png`
- `artifacts/phase2_a2_mdcath_split_state_motion/loss_curve_latest.dynamics.png`

### A2 8k evaluation and objective diagnosis

Artifact: `artifacts/paper_eval/phase2_a2_8k_honest_eval`

| Model | VAMP lag6 dim5 | MSM accuracy | Linear lag6 R2 |
| --- | ---: | ---: | ---: |
| A2 5k | 0.3850 | 0.4974 | -0.7574 |
| A2 8k | 0.4199 | 0.5062 | -0.5751 |
| A2 10k | 0.3976 | 0.4944 | -0.5457 |
| A2 8k random encoder | 0.4139 | n/a | n/a |
| geometry stats | 0.3729 | n/a | -0.1836 |
| centered CA | 0.4291 | n/a | -2.5174 |

Additional training from 5k to 8k helps all three diagnostics, but A2 still
does not beat centered CA on VAMP and its non-overlapping forecast R2 remains
negative and substantially below geometry statistics. This is evidence for a
limited training-duration benefit, not a reason to continue the same objective
indefinitely. At 10k, VAMP and MSM regress from their 8k values while the
negative R2 improves only slightly. A2 was therefore stopped at 10k.

The loss audit found two concrete issues:

1. A2 logs `semigroup_loss`, but `semigroup_weight` defaults to zero. It never
   receives a gradient.
2. More importantly, A2's operator maps `motion_states` to absolute EMA state
   targets. Reapplying that operator to its own absolute-state output, as the
   old semigroup diagnostic does, is semantically invalid because its declared
   input space is motion, not state. Predictor dropout also adds noise between
   the direct and composed paths.

The lagged-state loss is optimized, but its motion-only input omits the absolute
starting conformation. It falls after its 4k--6k peak, while lag 4 remains the
hardest horizon, but the operator is solving an unnecessarily ill-posed
motion-to-absolute-state task.

## A3 State-Space Residual Operator

- Config: `configs/phase2_a3_mdcath_state_motion_semigroup.yaml`
- Dataset: mdCATH only
- Checkpoint interval: 2,000 steps

A3 changes the transition source to a normalized detached absolute state plus
a gated causal motion residual. The lag predictor is a residual state-space
operator initialized to identity, so both its input and output represent the
same state and semigroup composition is well-defined. Lag-predictor dropout is
zero while the rest of the model retains dropout 0.1. Semigroup loss is enabled
at weight 0.02 with a 3,000-step ramp.

Training now logs, per lag, the supervised prediction loss, persistence loss,
and skill relative to persistence. It also logs each valid semigroup
composition independently (for example `4+8=12`) and skips semigroup entirely
for the invalid motion-only source. Focused and full related test suites pass
(the latest focused suite passes 9/9 and the full related suite passes 40/40).

Two further audits were required before the final A3 launch:

- The EMA target encoders inherited training mode, so dropout generated a
  different random target on every forward pass. `ProteinJEPA.train()` now
  keeps both EMA teachers in eval mode, and warm starts can explicitly sync
  target weights to context weights.
- Learned absolute frame-position embeddings dominated the temporal state.
  Before removal, the EMA target persistence loss between adjacent frames was
  about 0.78 even though the same-time context-to-target alignment loss was
  only 0.004. A3 disables absolute temporal and motion position embeddings;
  causal ordering and explicit lag conditioning remain. The intrinsic
  persistence loss then falls to roughly `1e-4`--`1e-3`.

The final A3 launch selectively warm-starts the A2 8k geometry and slow-state
weights, resets state/motion/lag heads, synchronizes EMA teachers, uses no
dropout in A3, and sets `state_jepa_weight=0` so the obsolete position-dependent
state predictor cannot perturb the protected slow representation. Early at
steps 120--140, all five lag skills versus online persistence are positive
(approximately 0.43--0.71), lag loss is around `7e-4`, and semigroup loss is
around `1.4e-3` with all five valid compositions populated. These are training
diagnostics only; the first promotion gate is the honest 2k evaluation.

### A3 2k gate

Artifacts:

- `artifacts/paper_eval/phase2_a3_2k_honest_eval`
- `artifacts/paper_eval/phase2_a3_2k_state_motion_eval`
- `artifacts/paper_eval/phase2_a3_2k_transition_operator_eval.json`

| Readout | VAMP lag6 dim5 | MSM accuracy | Linear lag6 R2 |
| --- | ---: | ---: | ---: |
| A3 state | **0.4449** | 0.4751 | -0.7840 |
| A3 state+motion | 0.3806 | 0.4751 | -0.7199 |
| random encoder | 0.4192 | n/a | n/a |
| centered CA | 0.4291 | n/a | -2.5174 |

A3 is the first Phase 2 checkpoint whose state VAMP exceeds both its random
encoder and centered CA in the selected quick metric. Its MSM remains above the
majority baseline but below A2 8k, and its non-overlapping linear R2 remains
negative. The state+motion evaluation is diagnostic only and is not promoted.

The first held-out transition audit initially showed positive EMA-target skill
on all 100/100 validation batches (mean 0.39--0.46 across lags). A stricter
online-state audit then showed negative skill at every lag. This exposed a
remaining target mismatch: A3 maps a `state_motion` source to a raw EMA
`frame_state` target, so its low supervised loss primarily learns the teacher
coordinate conversion. The output is not the same state type as the input even
though semigroup composition assumes that it is.

## A4 Closed State Operator

- Config: `configs/phase2_a4_mdcath_closed_state_operator.yaml`
- Warm start: A3 2k state representation
- Dataset: mdCATH only

A4 changes the lag target to the EMA teacher's `state_motion_states`, making
source, direct target, composed prediction, persistence baseline, and rollout
state the same semantic type. Dynamics modules are reset while the A3 slow
state representation is retained. A separate bug found during this change was
fixed in the causal motion transformer: frame zero was both causally isolated
and padding-masked, creating an all-masked attention row and NaN. It now serves
as a finite zero anchor.

At steps 40--80, all lag skills are positive (about 0.12--0.45), lag loss is
roughly `2e-4`--`6e-4`, and semigroup loss is roughly `4e-6`--`1e-4`. A4 is
scheduled for the same honest and strict online-transition evaluation at 2k.
