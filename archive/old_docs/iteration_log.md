# Iteration Log

This file records each dynamics-world-model iteration and its promotion status.
Metrics use the honest Phase 2 protocol unless explicitly stated otherwise.

## V0: A4 closed-state operator, 2026-08-25

- Config: `configs/phase2_a4_mdcath_closed_state_operator.yaml`
- Checkpoint: `artifacts/phase2_a4_mdcath_closed_state_operator/checkpoint_step2000.pt`
- Evaluation: `artifacts/paper_eval/phase2_a4_2k_honest_eval`
- Initialization: warm-started from A3 2k; diagnostic only, not a scratch promotion run.

Result: VAMP `0.4360`, MSM accuracy `0.4994`, MSM balanced accuracy `0.4465`,
linear lag-6 R2 `-0.8512`. VAMP and MSM improved over E0, but non-overlapping
forecast skill remained negative and the candidate was not promoted.

### V0 diagnosis

The lag objective trained `state_motion_states`, while the paper evaluation
read out `frame_states`. In addition, `feature_norm` normalized the EMA target
state but not the online `state` transition source. The loss therefore did not
describe the latent space used by the forecast evaluation.

## V1: normalized state-space operator

- Config: `configs/phase2_v1_state_space_operator.yaml`
- Output: `artifacts/phase2_v1_state_space_operator`
- Initialization: A4 2k, with transition heads and motion/state auxiliary heads reset.

Changes:

- `transition_source=state`, so the lag operator and evaluation use the same slow state.
- Online state transition inputs are layer-normalized when `feature_norm=true`, matching EMA targets.
- Semigroup weight reduced from `0.02` to `0.005` while validating the repaired forecast path.
- Added a regression test that checks both source/target space alignment and gradient flow into temporal/spatial encoders.

Promotion gate: positive non-overlapping lag-6 R2 and improvement on at least two
of VAMP, nested MSM balanced accuracy, and forecast R2 without a large regression.

### V1 2k interim result

- Evaluation: `artifacts/paper_eval/phase2_v1_2k_honest_eval`
- VAMP lag-6 dim-5: `0.4511`
- MSM accuracy: `0.5128`
- Linear lag-6 R2: `-0.9039`

V1 improves VAMP and MSM over E0, but the forecast R2 is worse than both E0
(`-0.8443`) and V0 A4 2k (`-0.8512`). The normalization mismatch was real but
was not the only forecast failure. V1 remains an unpromoted diagnostic while
the run is monitored to the next checkpoint.

### V1 latest saved-checkpoint re-evaluation

- Checkpoint: `artifacts/phase2_v1_state_space_operator/checkpoint_latest.pt`
- Evaluation: `artifacts/paper_eval/phase2_v1_latest_eval`
- Transition evaluation: `artifacts/paper_eval/phase2_v1_latest_transition_operator_eval.json`
- VAMP lag-6 dim-5: `0.4381`
- MSM accuracy: `0.5201`
- Linear lag-6 R2: `-0.8194`
- Direct transition skill: lag-1 `0.0374`, lag-2 `0.0249`, lag-4 `0.0057`, lag-8 `-0.0238`, lag-12 `-0.0334`
- Semigroup composition losses: approximately `0.9e-6` to `1.3e-6`

The saved checkpoint improves over the earlier V1 2k evaluation on lag-6 R2,
but remains below the geometry forecast baseline (`-0.1836`) and fails the
positive-R2 promotion gate. Low semigroup loss is therefore not sufficient
evidence of useful long-horizon prediction.

### V1 12k checkpoint evaluation

- Checkpoint: `artifacts/phase2_v1_state_space_operator/checkpoint_step12000.pt`
- Evaluation: `artifacts/paper_eval/phase2_v1_12k_honest_eval`
- VAMP lag-6 dim-5: `0.4381`
- MSM accuracy: `0.5201`
- Linear lag-6 R2: `-0.8194`

Training beyond 2k did not produce positive non-overlapping forecast skill. V1
is therefore not promoted; the current run is useful as a diagnostic for the
loss/representation mismatch but should not be treated as a successful world
model.

## V2.1: paired-window endpoint delta, launched 2026-08-25

- Design: `docs/v2_1_design.md`
- Config: `configs/phase2_v2_1_paired_window_10k.yaml`
- Output: `artifacts/phase2_v2_1_paired_window_10k`
- Training: mdCATH-only, scratch, 10,000 steps, checkpoints every 2,000 steps.
- Horizons: raw frame offsets `8/16/24/48`.
- Semigroup gradient: disabled.

The V2.1 trainer uses same-trajectory paired windows, causal endpoint states,
physical-time-conditioned residual delta prediction, and persistence-relative
per-horizon logging. Initial smoke and real mdCATH batch checks passed; the
10k run is active. Promotion will require positive non-overlapping horizon-48
forecast skill and the existing multi-metric gate.

### V2.1 10k evaluation

- Paired evaluation: `artifacts/paper_eval/phase2_v2_1_10k_paired_eval.json`
- Common pipeline: `artifacts/paper_eval/phase2_v2_1_10k_honest_eval`
- Horizon-8 paired skill: `0.3846`
- Horizon-16 paired skill: `0.3733`
- Horizon-24 paired skill: `0.3687`
- Horizon-48 paired skill: `0.3675`
- Common-pipeline VAMP lag-6 dim-5: `0.3799`
- Common-pipeline MSM accuracy: `0.3821`
- Common-pipeline MSM balanced accuracy: `0.3056`
- Common-pipeline linear lag-6 R2: `-0.4539`

V2.1 clearly learns the paired endpoint transition and beats persistence at
all four trained horizons, including the 48-frame horizon. It also improves
the common-pipeline linear R2 substantially over V1, but remains negative;
VAMP is near the geometry baseline and MSM regresses. V2.1 is therefore a
successful direct-transition diagnostic, not yet a promoted general-purpose
representation.

## V2.2: dual endpoint/slow state, launched 2026-08-26

- Design: `docs/v2_2_design.md`
- Config: `configs/phase2_v2_2_dual_state_10k.yaml`
- Output: `artifacts/phase2_v2_2_dual_state_10k`
- Training: mdCATH-only, scratch, 10,000 steps, checkpoints every 2,000 steps.
- Horizons: raw frame offsets `8/16/24/48`.
- Semigroup gradient: disabled.

Changes:

- Retained the V2.1 full-dimensional causal endpoint delta/state forecast.
- Added a 32-dimensional slow-state projector and EMA target projector.
- Added slow-state forecast losses, temporal in-batch contrastive alignment,
  variance floor, and covariance decorrelation.
- Added explicit `endpoint` and `slow_endpoint` paper evaluation readouts;
  legacy window-averaged `state` remains available for comparison.

Initial step-100 training diagnostics: endpoint persistence-relative skill
`0.0117`, slow-state skill `0.0006`. These are training diagnostics only;
promotion awaits held-out paired evaluation and the common VAMP/MSM/forecast
scoreboard at the 2k and 10k checkpoints.

### V2.2 2k checkpoint evaluation

- Paired validation: `artifacts/paper_eval/phase2_v2_2_2k_paired_eval.json`
- Endpoint paired skill, horizons 8/16/24/48: `0.584/0.543/0.639/0.605`
- Slow endpoint paired skill, horizons 8/16/24/48: `0.776/0.709/0.737/0.764`
- Endpoint common pipeline: VAMP normalized `0.3682`, MSM accuracy `0.4018`,
  linear R2 `-0.4484`.
- Slow endpoint common pipeline: VAMP normalized `0.3557`, MSM accuracy
  `0.3532`, linear R2 `-0.3802`.

At 2k both paired contracts generalize strongly beyond persistence. The slow
head improves linear forecast R2 relative to the full endpoint, but neither
readout yet clears the geometry VAMP/MSM/R2 baselines. A 10k continuation was
initially planned; no promotion decision is made from this early checkpoint.

### V2.2 early-stop audit

- `checkpoint_step4000.pt` paired validation endpoint skill 8/16/24/48:
  `0.177/0.138/0.211/0.209`.
- `checkpoint_step4000.pt` slow endpoint skill 8/16/24/48:
  `0.318/0.410/0.360/0.411`.
- The 4k checkpoint is materially worse than 2k on both paired contracts.
- Training was stopped at approximately step 4,520; the 2k checkpoint is the
  current V2.2 candidate and the 4k checkpoint is retained as a negative
  continuation-control artifact.

Interpretation: the current dual objective still permits the shared endpoint
representation to become overly time-invariant after its early optimum. A
future V2.2 revision needs an explicit representation-preservation or
physical-observable anchor before attempting another 10k run.

## V2.3: dual-timescale path prepared, 2026-08-26

- Design: `docs/v2_3_design.md`
- Status: architecture and gates prepared; no V2.3 training launched yet.
- Mandatory preflight fix: orientation-aware torsion features now use the
  visible token mask, with a masked-neighbor regression test.

The V2.3 proposal separates a bidirectional offline representation tower from
an independent causal dynamics tower, prioritizes non-overlap horizons
`48/96/192`, and introduces a bounded Gaussian residual transition only after
the deterministic 2k gate. It defines both a scratch fair track and a clearly
labeled champion-anchored engineering track. Promotion requires both the
aligned Phase-1 champion comparison and the honest non-overlap forecast gates.

### V2.3 implementation preflight

- Added independent bidirectional offline and causal temporal towers, each with
  its own EMA teacher; causal gradients into the shared geometry trunk are
  explicitly scaled.
- Added a bounded diagonal Gaussian residual transition, proper NLL training,
  and evaluator outputs for NLL, CRPS and one-/two-sigma coverage against a
  Gaussian persistence mean.
- Added coordinate-derived contact persistence, torsion change,
  radius-of-gyration change and local-distance-change targets.
- Enforced non-overlap at the dataset boundary for primary raw horizons
  `48/96/192` and retained physical tau, temperature and domain conditioning.
- Added explicit offline `endpoint` and `causal_endpoint` paper readouts.
- Added scratch deterministic, scratch Gaussian and champion-anchored 2k gate
  configs. The anchored loader copies the champion temporal tower only into
  the offline branch and leaves the causal branch random.
- Real mdCATH batch forward/backward passed with finite loss and nonzero causal
  and offline gradients at 48k/192k ps. Full test suite passed before launch.

Status: code-ready, not promoted, and no V2.3 long run launched. The next
authorized experiment is the scratch deterministic 2k gate; 10k continuation
remains prohibited until held-out representation and non-overlap forecast
metrics pass the checkpoint gate.

### V2.3-S deterministic 2k launch

- Launched: 2026-08-26 UTC
- Config: `configs/phase2_v2_3_s_deterministic_2k.yaml`
- Output: `artifacts/phase2_v2_3_s_deterministic_2k`
- Process at launch: PID `794526`, GPU 0; the existing A4 process was left untouched.
- Step-100 training diagnostic: aggregate paired skill `0.0848`; horizon
  48/96/192 skills `0.1140/0.0637/0.0805`; observable loss `0.00948`.

These are in-sample minibatch diagnostics only. Promotion remains pending the
held-out paired evaluation and both offline/causal representation scoreboards
at checkpoint step 2,000.

### V2.3-S deterministic 2k evaluation

- Paired result: `artifacts/paper_eval/phase2_v2_3_s_deterministic_2k_paired.json`
- Offline quick pipeline:
  `artifacts/paper_eval/phase2_v2_3_s_deterministic_2k_offline_quick`
- Causal quick pipeline:
  `artifacts/paper_eval/phase2_v2_3_s_deterministic_2k_causal_quick`

The original paired evaluator reported causal persistence-relative skills of
`0.9255/0.9263/0.9253` at raw horizons `48/96/192`. A same-space audit found
that this comparison mixed the online context state with the EMA future state.
After recomputing persistence from the EMA context state, canonical causal
skill was only `0.1066/0.0962/0.1341`; canonical offline skill was
`0.2627/0.2619/0.2492`. The transition learned a real but modest increment,
while most of the original score was online/teacher coordinate alignment.

Under the identical honest quick protocol:

| Metric | Champion | V2.3 offline | V2.3 causal | Geometry |
| --- | ---: | ---: | ---: | ---: |
| VAMP normalized, lag 6 dim 5 | 0.4049 | 0.3560 | 0.3672 | 0.3729 |
| MSM accuracy | 0.4154 | 0.3263 | not rerun | majority 0.4341 |
| Linear R2, lag 6 | -0.8434 | -0.5712 | -0.4124 | -0.1836 |

V2.3-S deterministic improves linear predictability over the learned champion
but loses slow-state and MSM information and remains below geometry. It is not
promoted. The Gaussian scratch gate is blocked until the canonical-state bug
is fixed and the offline branch regains an explicit future-mask State-JEPA
objective.

## V2.3.1-A: canonical champion-space transition

- Config: `configs/phase2_v2_3_1_a_canonical_champion_2k.yaml`
- Output: `artifacts/phase2_v2_3_1_a_canonical_champion_2k`
- Launched: 2026-08-26 UTC, PID `811340`, GPU 0.

Corrections relative to V2.3-S:

- context and future deltas, predictions and persistence now share one frozen
  champion teacher space;
- the Phase-1 spatial encoder and bidirectional offline tower are loaded
  exactly and frozen in evaluation mode;
- only the causal temporal tower, transition predictor and coordinate
  observable head train during the first gate;
- a same-time causal-to-champion alignment objective prevents an arbitrary
  online/teacher coordinate offset from being counted as dynamics.

Preflight verified that the frozen V2.3.1 offline output is bit-identical to
the Phase-1 champion (`max_abs_error=0`) and that no gradient reaches either
frozen module. Full tests passed (`52 passed`). Initial canonical paired skill
is zero by construction; checkpoint selection will use held-out aggregate SSE,
not unstable minibatch ratios for near-static champion states.

### V2.3.1-A 2k evaluation

- Paired evaluation:
  `artifacts/paper_eval/phase2_v2_3_1_a_canonical_champion_2k_paired.json`
- Causal quick pipeline:
  `artifacts/paper_eval/phase2_v2_3_1_a_canonical_champion_2k_causal_quick`

Held-out canonical persistence-relative skill at raw horizon `48/96/192` was
`0.0137/0.0364/0.0122`. All three are positive, confirming a small real
dynamics increment, but the gain is too small for promotion. Causal VAMP at
honest lag 6 dim 5 was `0.3897`, near the champion `0.4049` and above geometry
`0.3729`. Causal linear R2 was `-0.8835`, worse than the champion `-0.8434`
and far below geometry `-0.1836`.

The exact frozen offline branch retains the champion representation by
construction, but the learned causal transition does not improve the combined
scoreboard. V2.3.1-A is retained as a corrected state-space control and is not
continued or promoted. Gaussian training remains blocked.

## V3.0: protected history and metastable model, prepared 2026-08-26

- Design: `docs/v3_design.md`
- Config: `configs/phase3_v3_0_history_metastable_2k.yaml`
- Dataset: mdCATH only; non-overlap horizons `48/96/192`.
- Initialization: exact frozen Phase-1 champion geometry/offline state tower.

V3.0 replaces the global endpoint-only transition with a zero-initialized
multiscale history transition over champion states. It adds a separate
VAMPNet-style metastable probability readout and expands physical targets to
12 event-resolved channels, including contact gain/loss, signed torsions and
aligned RMSD. Observable evaluation now reports MSE and persistence-relative
skill. The first run is a 2k deterministic gate; Gaussian and semigroup losses
remain blocked until deterministic held-out skill passes the documented gates.

### V3.0 2k evaluation

- Checkpoint: `artifacts/phase3_v3_0_history_metastable_2k/checkpoint_step2000.pt`
- Paired evaluation:
  `artifacts/paper_eval/phase3_v3_0_history_metastable_2k_paired.json`
- Metastable quick pipeline:
  `artifacts/paper_eval/phase3_v3_0_history_metastable_2k_metastable_quick`

Held-out canonical latent persistence-relative skill at horizons `48/96/192`
was `0.0078/0.0102/-0.0015`. This is substantially below the horizon-48
continuation gate of `0.10`; multiscale history statistics do not make the
champion endpoint a sufficiently closed dynamical state.

The 12-channel physical head reported MSE skill over the fixed physical
persistence vector of `0.8744/0.8758/0.8837`. This remains a diagnostic, not a
world-model success: a horizon-conditioned mean can beat that fixed baseline,
so shuffled-history and train-mean controls are still required.

Under the honest quick representation protocol:

| Metric | Champion | V3 metastable | Geometry/reference |
| --- | ---: | ---: | ---: |
| VAMP normalized, lag 6 dim 5 | 0.4049 | 0.4104 | 0.3729 |
| MSM accuracy | 0.4154 | 0.4098 | majority 0.4341 |
| Linear R2, lag 6 | -0.8434 | -0.7561 | -0.1836 |

The metastable readout gives small VAMP and linear-R2 gains over the champion
but loses MSM and remains far below geometry for forecast. Its assignment
entropy fell to about `0.001` versus `log(8)=2.079`, indicating an overconfident
hard partition; the VAMP training loss also plateaued near `0.704`. V3.0 is not
continued beyond 2k. The protected champion branch remains available exactly,
and Gaussian/semigroup training remains blocked.

### V3.0 control audit and V3.1 preparation

- Control artifact:
  `artifacts/paper_eval/phase3_v3_0_history_metastable_2k_controls.json`
- V3.1 config: `configs/phase3_v3_1_local_dynamics_2k.yaml`

On the held-out split, the global physical model MSE was `0.021-0.027`, versus
`0.033-0.044` for a mean fitted on the first half and `0.042-0.062` after
shuffling context history. Thus the physical path contains real structural
history information. Horizon shuffling changed little, and latent-delta MSE
was also nearly unchanged by history shuffling, confirming that the global
champion endpoint remains the wrong closed state.

V3.1 adds four sequence-local physical tokens and a `local_dynamics` readout.
Only the local token/head objective is active in its 2k gate; champion,
metastable, global transition and global observable paths receive no update.

### V3.1 2k launch

- Launched: 2026-08-26 UTC
- Output: `artifacts/phase3_v3_1_local_dynamics_2k`
- Python PID at launch: `837334`, GPU 0.
- Existing A4 PID `718720` was left untouched.

Preflight passed all tests (`53 passed`) and a real mdCATH backward audit:
shared V3.0 weights had max error `0.0`, all local segments were valid, the
local prediction initialized exactly at physical persistence, and gradients
were nonzero only for the local head. Early local loss moved from `0.0530` at
step 0 to `0.0121` at step 140; this is a training diagnostic only.

### V3.1 2k evaluation

- Checkpoint: `artifacts/phase3_v3_1_local_dynamics_2k/checkpoint_step2000.pt`
- Paired controls:
  `artifacts/paper_eval/phase3_v3_1_local_dynamics_2k_controls.json`
- Local quick pipeline:
  `artifacts/paper_eval/phase3_v3_1_local_dynamics_2k_quick`
- Local plus metastable pipeline:
  `artifacts/paper_eval/phase3_v3_1_local_metastable_2k_quick`

Held-out local physical MSE at horizons `48/96/192` was
`0.0240/0.0247/0.0275`, corresponding to persistence-relative skill
`0.8230/0.8376/0.8237`. Shuffling context history worsened MSE to
`0.0432/0.0505/0.0463`, so the local head uses real context structure rather
than only a target mean. Horizon shuffling remained almost unchanged, showing
that physical time scale is still not identified.

Honest quick representation results:

| Readout | VAMP lag6 dim5 | MSM accuracy | Linear lag6 R2 |
| --- | ---: | ---: | ---: |
| Champion reference | 0.4049 | 0.4154 | -0.8434 |
| V3.1 local dynamics | 0.3787 | 0.4261 | -0.4101 |
| V3.1 local plus metastable | 0.3855 | 0.4337 | -0.4159 |
| Geometry/reference | 0.3729 | majority 0.4341 | -0.1836 |

V3.1 substantially improves MSM and linear predictability over the champion,
and local physical prediction passes both persistence and shuffled-history
controls. A single readout still loses champion VAMP, so this is the strongest
multi-head engineering candidate rather than a formally promoted universal
readout. The frozen champion state remains the VAMP path. The next evaluation
must pre-register these task-specific readouts and run aligned plus strict
protocols before a promotion claim.

Training local-loss means were `0.0266` over steps 0-400, `0.00881` over
800-1200, and `0.01005` over 1600-2000. Future gates should validate and save
every 500 steps in addition to the required 2k checkpoint because the optimum
may occur before the end of the gate.

## V3.2: counterfactual multi-horizon closed local operator

- Design: `docs/v3_2_design.md`
- Config: `configs/phase3_v3_2_closed_multihorizon_2k.yaml`
- Initialization: V3.1 2k; all existing representation heads preserved.

V3.2 groups horizons `48/96/192` under the same context, predicts an absolute
12-dimensional local state with FiLM plus horizon experts, and applies a
correct-versus-wrong horizon ranking loss. Only the new closed operator trains.
The gate saves every 500 steps and blocks semigroup/Gaussian training until
closed-state persistence and shuffled-horizon controls pass.

### V3.2 2k launch

- Launched: 2026-08-26 UTC
- Python PID: `856749`, GPU 0.
- Checkpoints: every 500 steps.
- Existing A4 PID `718720` was left untouched.

Preflight passed `55` tests and a real grouped mdCATH backward audit. The
three horizon rows had an identical context, all inherited V3.1 weights had
max error `0.0`, closed predictions initialized exactly at persistence, and
only the new closed operator received gradients. Step 0 started with closed
skill `0.0` and ranking loss equal to the configured `0.0005` margin.

### V3.2 checkpoint evaluation

Held-out controls used 100 pairs per horizon and a group-aware shuffle:
history was exchanged across context groups at the same horizon, while horizon
was exchanged within each context group.

| Step | Skill 48/96/192 | Correct better than shuffled horizon |
| --- | --- | --- |
| 500 | -0.05% / 0.01% / 0.55% | no / no / no |
| 1000 | 0.28% / 0.73% / 1.11% | no / yes / yes |
| 1500 | 0.83% / 1.10% / 1.91% | yes / yes / yes |
| 2000 | 0.89% / 1.29% / 1.84% | no / yes / yes |

Step 1500 is selected because it is the only checkpoint with the correct
horizon better than a shuffled horizon for all three buckets. Its advantage
was nevertheless only about `0.09%/0.28%/0.49%`, far below the predeclared 5%
horizon gate. V3.2 demonstrates a real but weak closed-state time signal and is
not continued. Semigroup, Gaussian and long rollout objectives remain blocked.

- Selected checkpoint:
  `artifacts/phase3_v3_2_closed_multihorizon_2k/checkpoint_step1500.pt`
- Step-1500 controls:
  `artifacts/paper_eval/phase3_v3_2_closed_multihorizon_step1500_controls.json`

## V3.3: dense standardized horizon identification

- Design: `docs/v3_3_design.md`
- Config: `configs/phase3_v3_3_dense_standardized_2k.yaml`
- Initialization: selected V3.2 step 1500.
- Scale audit:
  `artifacts/phase3_v3_3_dense_standardized_2k/closed_local_state_scale_train.json`

V3.3 adds same-context horizons `48/64/96/128/192`, standardizes all
closed-state residual losses with one 12-channel scale fitted on mdCATH train
only, and fixes correct-versus-wrong ranking to use the same valid-token
support. The train-only audit contains `10,000` valid segment deltas and finds
a `43.3x` spread between the smallest and largest channel standard deviation.
The normalized ranking weight is `1.0` with margin `0.05`.

### V3.3 2k launch

- Launched: 2026-08-27 UTC.
- Python PID: `911706`, GPU 0.
- Checkpoints: every 500 steps.
- Existing A4 PID `718720` was left untouched.

Preflight passed `58` tests and a real five-horizon mdCATH backward audit.
All inherited V3.2 tensors had max error `0.0`, all five experts had nonzero
gradients, and parameters outside the closed local operator had zero gradient.
Initial normalized state/persistence/ranking losses on the audit batch were
`1.0472/1.0541/0.0487`. Evaluation and promotion still depend on held-out raw
and normalized controls, not these training diagnostics.

### V3.3 2k evaluation

- Checkpoints: `500/1000/1500/2000`, all complete.
- Controls: `artifacts/paper_eval/phase3_v3_3_dense_standardized_step*_controls.json`.
- Curve: `artifacts/phase3_v3_3_dense_standardized_2k/training_curve.png`.

At step 2000, normalized persistence-relative skill for horizons
`48/64/96/128/192` was `0.60%/0.81%/1.22%/1.47%/1.90%`. Normalized horizon-
shuffle advantage was `-0.06%/0.16%/0.21%/0.21%/0.16%`, below the 2% gate at
every horizon and negative at horizon 48. No checkpoint passed the gate.

The ranking loss remained near the `0.05` margin in every 500-step block.
Correct history was strongly useful (normalized shuffled-history MSE about
`8.5` versus model MSE near `1.0`), but horizon identity remained weak. Common-
horizon raw skill was lower than selected V3.2, so V3.3 is not promoted and
should not be extended beyond 2k.

## V3.4: continuous context-conditioned dynamics

- Design: `docs/v3_4_design.md`.
- Config: `configs/phase3_v3_4_continuous_difference_2k.yaml`.
- Target audit:
  `artifacts/phase3_v3_4_continuous_difference_2k/horizon_identifiability_audit.json`.
- Initialization: selected V3.2 step 1500, with the old closed operator
  intentionally excluded.

The held-out target audit found same-context normalized horizon-difference MSE
from `0.66` to `1.18`, but unconditional horizon-centroid accuracy was `20.2%`
for five classes (`20%` chance). V3.4 therefore replaces independent experts
with shared continuous drift/relaxation modes and directly supervises
same-context horizon differences with minimum raw-frame gap 48.

### V3.4 2k launch

- Launched: 2026-08-27 UTC.
- Python PID: `927183`, GPU 0.
- Checkpoints: every 500 steps.
- Existing A4 PID `718720` was left untouched.

Preflight passed `62` tests and a real mdCATH five-horizon backward audit.
All inherited non-operator tensors had max error `0.0`; initial prediction was
exactly persistence; same-context representations differed by at most
`4.5e-8`; and non-operator gradient sum was `0`. Step 0 state/persistence loss
was `0.24167/0.24167`, while pairwise difference loss equaled its zero-change
baseline at `0.26731` as required by zero initialization.

### V3.4 10k resume

- Resume config: `configs/phase3_v3_4_continuous_difference_10k_resume.yaml`.
- Resume checkpoint:
  `artifacts/phase3_v3_4_continuous_difference_2k/checkpoint_step2000.pt`.
- Resume artifact:
  `artifacts/phase3_v3_4_continuous_difference_10k_resume`.
- Loss curve:
  `artifacts/phase3_v3_4_continuous_difference_10k_resume/v3_4_10k_closed_local_losses.png`.
- Held-out controls:
  `artifacts/paper_eval/phase3_v3_4_continuous_difference_step{500,1000,1500,2000,2500,3000,4000,6000,8000,10000}_controls.json`.

Held-out normalized summary:

| step | horizon mean skill | worst horizon skill | horizon shuffle mean | trained-pair mean skill | worst trained-pair skill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.011% | -0.099% | 0.194% | 0.075% | -0.102% |
| 1000 | 0.549% | 0.391% | 0.172% | 0.067% | -0.063% |
| 1500 | 0.655% | 0.363% | 0.096% | 0.085% | -0.144% |
| 2000 | 1.121% | 0.531% | 0.153% | 0.273% | 0.081% |
| 2500 | 0.627% | 0.361% | 0.195% | 0.143% | -0.007% |
| 3000 | 0.860% | 0.665% | 0.216% | 0.034% | -0.017% |
| 4000 | 0.753% | 0.337% | 0.127% | 0.181% | 0.042% |
| 6000 | 1.079% | 0.634% | 0.144% | 0.108% | -0.187% |
| 8000 | 1.665% | 0.901% | 0.168% | 0.308% | 0.001% |
| 10000 | 1.701% | 0.977% | 0.155% | 0.299% | 0.066% |

V3.4 10k is the strongest closed-state persistence result so far under this
paired-dynamics protocol: it slightly exceeds V3.2 step 2000 (`1.34%` mean,
`0.89%` worst). It is not a solved world model. The horizon-shuffle signal is
still only about `0.16%`, and pairwise difference skill remains near zero. More
steps improved absolute state MSE modestly but did not fix deterministic
horizon-conditioned dynamics.

## V3.5: velocity-conditioned robust probabilistic dynamics

- Design: `docs/v3_5_design.md`.
- Config: `configs/phase3_v3_5_velocity_student_t_2k.yaml`.
- Initialization: V3.4 step 10000, with `closed_local_operator.*` excluded.

The V3.4 log audit showed that target difference energy had coefficient of
variation `1.45`, correlated `1.000` with difference loss, while predicted
difference energy was only about `0.47%` of target energy at the median. V3.5
adds causal multiscale closed-state velocity, predicts deltas in standardized
channel coordinates, and adds a Student-t uncertainty head. Optimization uses
per-pair energy-normalized Huber losses; unchanged raw MSE diagnostics remain
the basis of comparisons with V3.2/V3.4.

### V3.5 2k launch

- Launched: 2026-08-27 UTC.
- Training PID: `965746`, GPU 0.
- Checkpoints and held-out controls: every 500 steps.
- Existing stopped A4 PID `718720` was left untouched.

Preflight passed `65` tests and a real five-horizon mdCATH backward audit.
The initial prediction was exactly persistence, normalized scale was exactly
one, operator gradient L1 was `6.14`, and all non-operator gradients were zero.
The real-batch peak allocation was about `3.6 GiB` before optimizer state.

## V4-A: compositional mean-reverting dynamics

- Design: `docs/v4_design.md`.
- Config: `configs/phase4_v4_a_compositional_2k.yaml`.
- Initialization: V3.5 step 2000 with the closed operator excluded.

V4-A replaces direct continuous-basis horizon decoding with one shared stable
raw-frame-16 transition composed 3/4/6/8/12 times. It adds centered per-frame
physical history, recurrent memory, cross-segment coupling, explicit physical
step time and temperature, and a causal context-internal base-step objective.
The low-rank stochastic transition is deliberately deferred until this
deterministic composition passes the correct-depth held-out gate.

### V4-A 2k launch

- Launched: 2026-08-27 UTC.
- Training PID: `985532`, GPU 0.
- Evaluation watcher PID: `985717`.
- Checkpoints and held-out controls: every 500 steps.
- Existing stopped A4 PID `718720` was left untouched.

Preflight passed `68` tests and a real mdCATH backward audit. Initial long
rollout and internal T16 predictions matched persistence within `3e-8`; the
new operator gradient L1 was `66.2`, all non-operator gradients were zero, and
peak allocated memory was about `3.6 GiB`.

## V4-B1: fast/slow kinetic decomposition

- Design: `docs/v4_b_design.md`.
- Config: `configs/phase4_v4_b1_slow_kinetic_2k.yaml`.
- Fast initialization: V4-A step 2000, frozen.

V4-B1 adds a trajectory-centered 16-dimensional slow latent, eight soft
metastable states, physical reconstruction, and one context-conditioned
row-stochastic T16 matrix composed across all five horizons. Multi-lag VAMP,
base transition, rollout transition, occupancy, entropy and fixed-depth
controls form a strict gate before adding low-rank Student-t innovations.

### V4-B1 2k launch

- Launched: 2026-08-28 UTC.
- Training PID: `1041284`, GPU 0.
- Evaluation watcher PID: `1041454`.
- Checkpoints and held-out controls: every 500 steps.
- Existing stopped A4 PID `718720` was left untouched.

Preflight passed `70` tests. Every inherited fast-operator tensor matched
V4-A step 2000 exactly, the fast branch had zero trainable parameters and zero
gradient, assignment and transition normalization errors were below `1.2e-7`,
and only the new slow model received gradients. Initial eight-state occupancy
was non-collapsed and real-batch peak allocation was about `3.6 GiB` before
optimizer state.

### V4-B1 held-out decision

All step `500/1000/1500/2000` evaluations completed. Step 2000 retained the
V4-A fast result (about `10.09%` mean normalized skill and `8.96%` worst
horizon), but B1 failed the slow-kinetic gate. Its apparent mean Brier skill
was `47.55%`, while horizon shuffle advantage was `-0.0014%`, history shuffle
advantage was only `0.18%`, and correct depth was about `-0.0026%` relative to
the best fixed depth. Mean `lambda2` collapsed from `0.225` at step 500 to
`0.075` at step 2000. The transition learned horizon-independent rapid mixing,
not a slow process, so B1 is not extended or promoted to Student-t B2.

## V4-C: physics-constrained fast/slow design

- Design: `docs/v4_c_physics_design.md`.
- Status: design and data audit complete; implementation not started.

The held-out representation audit found excessive fast/slow overlap (mean
state CCA about `0.682`, increment CCA about `0.663`, and `21.5%` linearly
predictable fast variance) and no evidence that the learned slow latent was
slower. V4-C uses additive slow reconstruction plus a fast residual and
whitened conditional cross-covariance penalties, rather than pointwise
orthogonality.

The mdCATH audit also found aligned per-frame all-atom forces in every sampled
trajectory group. V4-C will use backbone force matching to shape a conditional
free-energy latent, then fit protein/temperature-conditional reversible
generators. Representation learning, reversible kinetics, and stochastic
rollout are separated into gated stages.

### V4-C1 implementation and launch

- Config: `configs/phase4_v4_c1_physics_representation_2k.yaml`.
- Training PID: `1072444`, GPU 0.
- Checkpoints and held-out controls: every 500 steps.
- Existing stopped A4 PID `718720` was left untouched.

V4-C1 loads aligned mdCATH backbone forces, combines frozen offline temporal
features with closed physical states, decodes only a temporal low-pass state,
and defines the fast residual explicitly. Training adds whitened state and
increment cross-covariance, a slow-versus-residual autocorrelation margin, a
latent free-energy drift, and conservative atom-force matching through
coordinate gradients. The kinetic head uses a symmetric-flux reversible
continuous-time generator and `matrix_exp`, but transition fitting remains
disabled during C1 representation learning.

Preflight passed `73` tests and a real mdCATH second-order backward audit. The
audit had V4-C gradient L1 `145.52`, frozen-fast gradient `0`, transition row
error `1.2e-7`, detailed-balance maximum residual `2.9e-11`, and peak allocated
memory `1.82 GiB` for one grouped context. At training step 20, total loss fell
from `1.163` to `1.129`, low-pass reconstruction from `0.172` to `0.103`, and
force loss from `0.360` to `0.332`; base `lambda2` remained `0.988` rather than
collapsing toward zero.

### V4-C1 step-500 held-out evaluation

Training remained active after the checkpoint. On 50 held-out batches, VAMP
loss was `0.410`, latent autocorrelation was `0.464`, and fast-residual
autocorrelation was `-0.100`; the timescale-separation margin was therefore
inactive. Cross-covariance and increment cross-covariance losses were
`0.0163/0.0179`, assignment entropy was `0.649`, and occupancy balance loss was
`0.0108`. This is the first V4 fast/slow experiment with clear held-out
timescale separation.

The force objective has not yet succeeded: training force cosine continues to
oscillate around zero and normalized force loss remains near its initial
baseline. This is treated as a failed sub-objective, not hidden by the VAMP
gain.

C1 intentionally assigns zero weight to stationary, base-transition, and
rollout-transition fitting. Consequently its held-out rollout Brier skill is
strongly negative and correct depth loses to the shortest fixed depth. Those
metrics are expected controls for an unfitted generator and are not a C1
promotion signal. C2 may only fit reversible kinetics after selecting a C1
representation checkpoint; stochastic rollout remains deferred.

### V4-C1 final evaluation

Training and held-out evaluations completed at steps `500/1000/1500/2000`.
The frozen V4-A branch remained unchanged at about `10.10%` mean normalized
skill and `8.96%` worst-horizon skill. C1 reduced held-out VAMP loss from
`0.410` at step 500 to `0.407` at step 2000 and maintained latent versus
residual autocorrelation near `0.462/-0.106`. Whitened state and increment
cross-covariance also fell to `0.0135/0.0156`.

These values do not pass the representation gate. Assignment entropy
collapsed from `0.649` at step 500 to `0.153` at step 2000 while global
occupancy remained balanced. At the same time, assignment persistence error
became almost zero within trajectories and Brier skill relative to persistence
became catastrophically negative. The model learned nearly static, globally
balanced identity clusters rather than metastable transitions. Frozen offline
features supplied to every frame are the likely protein-identity leakage path.

The physics objective also failed: force cosine averaged approximately zero
throughout training and force loss did not improve over its initial baseline.
The current no-gradient evaluator reports a zero-drift baseline rather than a
valid energy-gradient drift metric, so that value is not used for promotion.
No C1 checkpoint is promoted to C2. The next representation must residualize
static protein context, compute variational statistics within each
protein/trajectory, require nontrivial within-trajectory switching, and replace
raw instantaneous force matching with a coarse or time-averaged projected
force target.

## V4-D1: scratch dynamic-state gate

- Design: `docs/v4_d_scratch_design.md`.
- Config: `configs/phase4_v4_d1_scratch_dynamic_1k.yaml`.
- Initialization: none; every learned parameter starts from its random
  initialization.

D1 removes offline temporal features from the slow encoder and feeds only
trajectory-centered, fixed-low-pass physical dynamics. Static equilibrium
structure and temperature condition the reversible generator but cannot alter
state assignment. VAMP covariance is computed per trajectory, and new losses
require within-trajectory state occupancy, nonzero latent temporal variance,
and correlation between assignment changes and physical changes. An explicit
between- versus within-trajectory assignment variance ratio audits identity
leakage. Raw instantaneous force matching is disabled.

Preflight passed `75` tests and a real mdCATH one-step backward. Initial
assignment switching was `0.045`, trajectory balance loss was `0.015`, and
physical activity was `0.907`; the starting solution was not a constant state.
The run uses a 1000-step gate with checkpoints every 250 steps rather than an
unconditional long training schedule.

## V4-D2: reversible kinetics

- Design: `docs/v4_d2_kinetics_design.md`.
- Config: `configs/phase4_v4_d2_reversible_kinetics_1k.yaml`.
- Representation source: D1 scratch step 750.

D2 freezes all dynamic-state representation parameters and trains only the
equilibrium/temperature conditioner, stationary state energies, and symmetric
flux rates. It adds stationary occupancy, base and rollout transition losses,
soft transition NLL, correct-depth ranking against every fixed-depth control,
and same-context horizon-difference loss. All transitions remain reversible
matrix exponentials of one continuous-time generator.

Preflight passed `77` tests and a real mdCATH backward. Trainable generator
gradient L1 was `1.36`, frozen representation gradient was exactly `0`, row
normalization error was `1.2e-7`, detailed-balance error was `0`, and peak
allocated memory was `1.12 GiB` for one grouped context.

### V4-D2 step-250 gate

D2 was stopped after the first held-out gate rather than spending the full
1000-step budget. Mean held-out Brier skill reached `44.47%`, with positive
skill at every horizon (`42.38/44.59/46.73/43.78/44.86%`). This gain was again
mostly rapid mixing: history-shuffle advantage was only `0.37%`, horizon-
shuffle advantage `0.16%`, and correct depth was `-0.24%` relative to the best
fixed depth. Base `lambda2` collapsed from the D1 initialization near `0.993`
to `0.414` at step 250 and had reached about `0.245` in training by step 400.

The reversible parameterization continued to satisfy detailed balance, but it
correctly exposed that D1 assignments are not slow enough to support the
requested horizons. D2 training and its watcher were terminated after saving
step 250. No D2 checkpoint is promoted. A hard spectral floor alone would
hide, rather than fix, the missing slow representation.

### D2 rate calibration and independent test

A post-training reversible rate sweep was run in one-pass evaluation without
changing assignments or generator geometry. The scale was selected on the
validation manifest before test evaluation. Scale `0.25` was selected as the
Pareto point with validation Brier skill `43.32%`, history-shuffle advantage
`5.97%`, horizon-shuffle advantage `0.81%`, and base `lambda2=0.802`.

On the independent test manifest, the locked scale achieved mean Brier skill
`41.37%`, history-shuffle advantage `5.47%`, and base `lambda2=0.806`.
Horizon-specific Brier skill was positive at every horizon
(`39.49/40.40/42.79/44.21/39.96%`). Horizon-shuffle advantage was positive at
64/96/128 frames but negative at 48/192, giving a `-0.38%` mean. Correct depth
remained `-2.60%` relative to the best fixed depth. This is a defensible
nonstationary-history result, not evidence that full time calibration is
solved.

- Rate sweep: `scripts/sweep_reversible_rate_scale.py`.
- Pareto figure: `artifacts/paper_eval/v4d_rate_pareto.png`.
- Submission summary: `docs/submission_rescue_results.md`.

### Intuitive test metrics

The locked scale-0.25 test evaluation was extended with hard six-state
classification controls. Top-1 accuracy was `23.20%` versus `25.60%` for
persistence; top-2 accuracy was `38.20%` versus a `33.33%` random reference,
and balanced accuracy was `24.05%` versus `16.67%` random. Target states changed
in `74.4%` of examples. Transition detection precision/recall were
`67.52/25.85%`, changed-state exact accuracy was `5.11%`, and unchanged-state
accuracy was `76.47%`.

Thus the large Brier gain is a soft-probability improvement, not a hard-label
accuracy improvement. The accuracy figure is
`artifacts/paper_eval/v4d_test_accuracy.png`.
