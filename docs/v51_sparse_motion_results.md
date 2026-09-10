# V5.1 Sparse Motion Dynamics Results

## Implementation

V5.1 replaces global conformational slots with history-conditioned sparse motion
experts. The active route is an exact straight-through top-k mask over motion
modes, and rollout is continuous in the learned dynamic state. The observable
decoder is anchored so that zero rollout is exactly the zero-motion physical
baseline.

Main files:

- `protein_jepa_frame/models/predictor.py`: `DynamicSequenceStateEncoder` and
  `ContextualSparseMotionDynamics`.
- `protein_jepa_frame/models/jepa.py`: V5.1 forward wiring, strict raw-history
  reversal control, and V5.1 losses.
- `protein_jepa_frame/training/train.py`: V5.1 optimizer and logging terms.
- `scripts/evaluate_v51_sparse_motion.py`: held-out evaluator with strict
  history, horizon, fixed-horizon, oracle, displacement, and routing gates.
- `configs/phase5_v51_sparse_motion_pilot.yaml`: current 200-step pilot config.
- `configs/phase5_v51_sparse_motion_2k.yaml`: promoted 2K action-state config.

## Why V5 Failed

The V5 held-out ablation showed the generator had effectively learned near-zero
motion while the observable decoder relied on static/history shortcuts:

- feature skill vs persistence: `-2.15%`
- correct vs fixed horizon: `-0.71%`
- physical history advantage: `0.027%`
- physical horizon advantage: `0.008%`
- rollout displacement: `8.66e-7`
- true feature displacement: `1.07e-3`

This is not a training-length issue. The old state assignment and generator were
not forced to represent history-dependent physical motion.

## V5.1 Pilot Results

### Initial V5.1, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot/eval_v51_50b.json`

- feature vs persistence skill: `-37.979`
- physical vs no-rollout skill: `0.786`
- physical oracle vs no-rollout skill: `0.018`
- strict raw-history advantage: `0.0006`
- horizon shuffle advantage: `-0.013`
- predicted/target displacement ratio: `6.17`
- active modes: `4.0 / 4`

Diagnosis: the observable loss made physical metrics look good while the latent
generator overshot target motion by about 6x. This is a decoder/generator
collusion failure.

### Calibrated delta pilot, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot_calibrated/eval_v51_50b.json`

- feature vs persistence skill: `-0.180`
- physical vs no-rollout skill: `0.0056`
- physical oracle vs no-rollout skill: `0.0039`
- strict raw-history advantage: `0.075`
- horizon shuffle advantage: `0.014`
- predicted/target displacement ratio: `0.495`
- active modes: `4.0 / 4`

Diagnosis: normalized delta and displacement calibration fixed the amplitude
failure, but the teacher dynamic state still did not encode physical observable
changes.

### State-observable pilot, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot_state_observable/eval_v51_50b.json`

- feature vs persistence skill: `-0.162`
- physical vs no-rollout skill: `0.0016`
- EMA oracle vs no-rollout skill: `-0.0004`
- online oracle vs no-rollout skill: `0.776`
- strict raw-history advantage: `0.0755`
- horizon shuffle advantage: `0.0186`
- predicted/target displacement ratio: `0.493`
- active modes: `4.0 / 4`

Diagnosis: the online state encoder can carry physical future changes, but the
EMA teacher and rollout objective do not yet transfer that information into the
teacher target space strongly enough. This run should not be promoted to 2K.

### Online-target pilot, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot_online_target/eval_v51_50b.json`

- feature vs persistence skill: `0.882`
- physical vs no-rollout skill: `0.785`
- physical oracle vs no-rollout skill: `0.780`
- horizon shuffle advantage: `-0.0016`
- predicted/target displacement ratio: `0.945`
- hard utilized modes: `11 / 16`

Diagnosis: replacing the EMA target with online physics-supervised target space
fixes the main representation mismatch. The remaining failure is horizon
identifiability: same-context target displacement is almost constant across
horizons, so the generator can win without really learning horizon-specific
future actions.

### Horizon-conditioned pilot, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot_horizon_conditioned/eval_v51_50b.json`

- feature vs persistence skill: `0.907`
- physical vs no-rollout skill: `0.786`
- horizon shuffle advantage: `-0.00024`
- predicted/target displacement ratio: `0.953`
- hard utilized modes: `15 / 16`

Diagnosis: adding horizon embeddings to the router and experts improves fit and
mode usage, but does not by itself make the learned target state
horizon-discriminative.

### Amplitude-state pilot, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot_amplitude_state/eval_v51_50b.json`

- feature vs persistence skill: `0.918`
- physical vs no-rollout skill: `0.787`
- horizon shuffle advantage: `-0.00051`
- predicted/target displacement ratio: `1.013`
- hard utilized modes: `16 / 16`

Diagnosis: amplitude conditioning fixes rollout scale and sparse mode usage, but
the target displacement remains nearly flat across horizons. The model still has
too weak a reason to predict different futures for 48, 64, 96, 128, and 192
frames.

### Action-state pilot, 200 steps

Output:
`artifacts/phase5_v51_sparse_motion_pilot_action_state/eval_v51_50b.json`

- feature vs persistence skill: `0.918`
- physical vs no-rollout skill: `0.785`
- physical oracle vs no-rollout skill: `0.781`
- online oracle vs no-rollout skill: `0.768`
- EMA oracle vs no-rollout skill: `-0.0021`
- strict raw-history advantage: `0.000081`
- horizon shuffle advantage: `0.000031`
- physical horizon advantage: `0.0036`
- predicted/target displacement ratio: `0.997`
- active modes: `4.0 / 4`
- hard utilized modes: `16 / 16`
- promotion pass: `true`

Diagnosis: action-observable target injection makes the 200-step gate pass. This
is the first V5.1 run where rollout displacement, physical observables, and hard
mode utilization are simultaneously sane. The caveat is still sharp: horizon
specificity is only marginal in aggregate, and the 192-frame fixed-horizon
control remains worse than shorter wrong-horizon controls in feature space.

## Promotion Decision

Promote the action-state V5.1 pilot to formal 2K training, but treat horizon
identifiability as the main watch item. The promotion is justified because the
latest held-out pilot passes every hard gate in `evaluate_v51_sparse_motion.py`:

- feature rollout beats persistence by a large margin.
- physical rollout beats no-rollout by a large margin.
- online/oracle physical decoders are useful.
- predicted and target displacement are same-order.
- all 16 hard modes are used with exact top-k active count.

The non-negotiable 2K failure condition is horizon collapse. If the 500-step or
2K checkpoint keeps `correct_vs_best_wrong_fixed_skill <= 0` or shows near-zero
horizon shuffle advantage, the next architecture must add explicit
time-contrastive, same-context multi-horizon state learning rather than only
making the generator stronger.

## Next Required Change

The next change after this 2K run should target same-context horizon
discrimination directly:

1. Evaluate `phase5_v51_sparse_motion_2k.yaml` at 500, 1000, and 2000 steps.
2. Keep the action/online target; do not return to EMA-only teacher targets.
3. If horizon controls fail, implement paired same-context multi-horizon
   contrastive targets so future states for the same context are ordered by
   horizon and cannot collapse to one amplitude bucket.
4. Only compare against baselines after held-out horizon controls are positive,
   not just after physical observable MSE improves.

## Verification

- `python -m py_compile` on modified Python files: passed.
- targeted V5/V5.1 tests: `5 passed`.
- full test suite: `82 passed`.
