# LSP-WM Revise Plan: Phase 0, Phase 1, Phase 2, And Phase 3

Date: 2026-09-11

Repository: `/liuyutian/protein_jepa_frame`

## 1. Objective

The immediate objective is to identify the limiting component of the current
protein latent world model before changing its architecture or launching a new
training run.

The working hypothesis is that the representation contains useful kinetic
signal, while the learned transition law, scale calibration, and uncertainty
calibration are not yet closed. Phase 0 and Phase 1 are designed to test that
hypothesis with one reproducible protocol.

This document is executable. Every result must be traceable to:

- one checkpoint;
- one immutable config;
- one explicit manifest;
- one evaluation seed;
- one output directory;
- one machine-readable JSON result.

No checkpoint is promoted by a training-log peak alone.

## 2. Scope And Claim Boundaries

There are two independent experiment tracks:

| Track | Model | Primary claim |
|---|---|---|
| LSP-WM v4.4 | Stage B v3 and DetSkill Fix | held-out paired endpoint and transition closure |
| TM-JEPA V3 / EVA5C | EVA5C quadratic latent closure | endpoint-energy validation under the TM-JEPA protocol |

The two tracks must not share a leaderboard. A positive DetSkill score is not
evidence that the EVA5C thermodynamic endpoint model is improved, and a
positive EVA5C endpoint-energy score is not evidence of a closed LSP-WM
transition operator.

The following claims remain disabled until their own gates pass:

- complete protein-environment world model;
- action-conditioned or counterfactual response;
- autonomous long-rollout equilibrium correctness;
- external literature SOTA;
- held-out test-set SOTA when the test set has been used for selection.

## 3. Frozen Evaluation Protocol

### 3.1 Splits

The LSP-WM v4.4 primary protocol uses:

| Purpose | Manifest | Usage |
|---|---|---|
| Training | `manifests/tmjepa_v3.train.jsonl` | training only |
| Validation | `manifests/tmjepa_v3.validation.jsonl` | checkpoint selection and diagnosis |
| Test | `manifests/tmjepa_v3.test.jsonl` | opened once after selection |
| Temperature extrapolation | `manifests/tmjepa_v3.temperature_test.jsonl` | separate transfer diagnostic |

The split is protein-disjoint. The validation and test manifests must never be
replaced by the training manifest through a config default.

FastFolding remains a separate scaffold-disjoint downstream benchmark:

- horizons: `1, 4, 8, 16`;
- state labels: `unfolded, transition, folded`;
- selection is based on the LSP-WM validation protocol, not FastFolding test
  accuracy.

### 3.2 LSP-WM endpoint metrics

The primary endpoint evaluator is
`scripts/evaluate_lspwm_endpoint.py`.

For horizons `24, 48, 96` frames, report:

- deterministic endpoint MSE;
- persistence MSE;
- endpoint skill versus persistence;
- ensemble energy score;
- energy skill versus persistence;
- transition NLL;
- MSM readout accuracy and soft overlap as diagnostics only.

The baseline is persistence:

```text
q_hat(t + tau) = q(t)
```

Skill is:

```text
skill = 1 - model_error / persistence_error
```

All horizon cells must be reported. A macro average cannot hide a negative
long-horizon cell.

### 3.3 Closure and calibration metrics

After endpoint screening, run the held-out comprehensive evaluator and report:

- observed endpoint skill;
- force target norm and force target degeneracy flag;
- VAMP-2 / Koopman residual as representation diagnostics;
- state forecast NLL, Brier score, ECE, and macro-F1;
- transition row JS / total variation;
- stationary distribution L1 distance;
- MFPT relative error;
- rollout skill versus persistence;
- covariance closure and innovation whiteness where the evaluator supports it;
- diffusion PSD and conditioning as parameterization checks.

The following are not sufficient by themselves:

- VAMP-2;
- same-space MSM assignment accuracy;
- a training-batch paired skill;
- PSD diffusion;
- numerical absence of NaN or Inf.

### 3.4 Force matching validity

The force target is a KDE estimate. A force cosine is invalid when the target
is numerically degenerate. The evaluator must record:

```text
force_target_norm
force_target_degenerate = force_target_norm < 1e-6
force_cosine_valid = not force_target_degenerate
```

If the target is degenerate, force cosine and force MSE are diagnostics of the
estimator failure, not model promotion metrics. The bandwidth must be recorded
for every run. Phase 0 checks the fixed bandwidths `0.05` and `0.10`, plus the
adaptive setting `0`.

## 4. Phase 0: Protocol Freeze And Integrity Audit

### Goal

Make accidental train-set evaluation, non-reproducible sampling, invalid force
diagnostics, and ambiguous artifact provenance impossible to miss.

### Deliverables

1. `scripts/audit_lspwm_protocol.py`
   - verifies manifest existence and JSONL readability;
   - reports records, proteins, trajectories, temperatures, and frame intervals;
   - checks train/validation/test protein overlap;
   - records the active config weights and freeze flags;
   - writes a machine-readable audit and a Markdown summary.
2. `protein_jepa_frame/eval/comprehensive_eval.py`
   - accepts an explicit `--manifest`;
   - accepts `--seed` and `--batch-size`;
   - uses a deterministic hierarchical sampler;
   - records protocol provenance in `combined_results.json`;
   - marks degenerate force targets.
3. A clean audit artifact:

```text
artifacts/phase0_protocol_audit/protocol_audit.json
artifacts/phase0_protocol_audit/protocol_audit.md
```

### Phase 0 command

```bash
cd /liuyutian/protein_jepa_frame
PYTHONPATH=$PWD /liuyutian/protein_jepa/.venv/bin/python \
  scripts/audit_lspwm_protocol.py \
  --output-dir artifacts/phase0_protocol_audit
```

### Phase 0 acceptance criteria

- all four canonical manifests exist;
- train/validation/test protein intersections are empty;
- every evaluated checkpoint names an explicit validation or test manifest;
- the active config is copied into the result provenance;
- `paired_delta_weight`, freeze flags, rollout flags, and KDE bandwidth are
  visible in the audit;
- no force cosine is promoted when `force_target_degenerate=true`.

## 5. Phase 1: Checkpoint Screening And Bottleneck Attribution

### Goal

Separate checkpoint selection effects from architecture effects, then determine
whether the primary bottleneck is:

1. representation information;
2. paired predictor capacity/training;
3. one-step thermodynamic calibration;
4. multi-step transition closure;
5. uncertainty/operator calibration; or
6. an invalid evaluation target.

### Candidate set

The initial validation screening includes:

| Label | Checkpoint family | Candidate checkpoints |
|---|---|---|
| `stage_b_v3` | original Stage B v3 | steps `1500, 2000, 2500, 3000` |
| `detskill_v1` | DetSkill Fix, seed 6444 | steps `1500, 2000, 2100, 2500, 3000` |
| `detskill_v2` | KDE bandwidth 0.05, seed 6445 | step `3000` |
| `detskill_v3` | KDE buffer, seed 6446 | steps `2000, 2500, 3000` |

The exact available files are checked before execution. Missing checkpoints
are recorded, not silently substituted.

### Phase 1A: held-out endpoint screen

For every available candidate:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python \
scripts/evaluate_lspwm_endpoint.py \
  --config <candidate-config> \
  --checkpoint <candidate-checkpoint> \
  --manifest manifests/tmjepa_v3.validation.jsonl \
  --output <candidate-output>/endpoint_validation.json \
  --batch-size 4 \
  --max-batches 80 \
  --ensemble-size 8 \
  --seed 6303 \
  --device cuda:0
```

Selection uses validation only. The ranking key is:

1. positive endpoint skill at all three horizons;
2. highest minimum horizon skill;
3. lower transition NLL;
4. higher energy skill;
5. stable result across seeds.

The final checkpoint is not automatically preferred. A mid-training checkpoint
can be selected when late training degrades held-out skill.

### Phase 1B: held-out comprehensive diagnosis

Run the selected checkpoint from each family on validation with an explicit
manifest:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python \
-m protein_jepa_frame.eval.comprehensive_eval \
  --checkpoint <checkpoint> \
  --config <config> \
  --manifest manifests/tmjepa_v3.validation.jsonl \
  --output-dir <output>/comprehensive_validation \
  --max-batches 20 \
  --batch-size 4 \
  --seed 6303 \
  --bandwidth 0.10 \
  --device cuda:0 \
  --skip-fastfolding
```

Repeat the force-only diagnostic with `--bandwidth 0.05` and `--bandwidth 0`
when the target degeneracy flag is raised.

### Phase 1C: confirmatory test evaluation

After validation ranking is frozen, evaluate at most the selected candidate
from each family on:

```text
manifests/tmjepa_v3.test.jsonl
```

The test result is confirmatory and cannot be used to change the checkpoint
selection rule.

### Bottleneck decision rules

| Observation | Attribution |
|---|---|
| training paired skill high, held-out endpoint skill low | overfitting, leakage, or checkpoint selection failure |
| DetSkill candidates improve held-out endpoint but not closure metrics | paired predictor was a real local fix; transition law remains the bottleneck |
| VAMP/MSM representation diagnostics remain strong while rollout/operator fail | transition closure and scale calibration are the bottleneck |
| force target norm is near zero across bandwidths | force evaluator/KDE target is invalid; do not optimize force loss yet |
| endpoint MSE beats persistence but energy skill/NLL fail | probabilistic scale or covariance calibration is the bottleneck |
| all horizons fail, including one-step | state representation or one-step transition mapping is insufficient |
| only long horizons fail after one-step passes | multi-step path likelihood, semigroup consistency, or memory is the bottleneck |

## 6. Phase 1 Promotion Gates

Phase 1 does not promote a new SOTA model. It only produces a defensible
diagnosis. A candidate can enter Phase 2 only if:

- validation endpoint skill is positive at `24, 48, 96`;
- validation endpoint is above persistence at the macro level;
- no metric relies on a degenerate force target;
- checkpoint selection is frozen before test evaluation;
- at least one independent evaluator records the explicit held-out manifest;
- the claim boundary in the result JSON matches the actual metrics.

The full model promotion gates remain stricter:

- held-out endpoint skill positive at every horizon and temperature;
- rollout skill above persistence;
- H16 state forecast NLL/ECE/Brier improved;
- transition operator stationary L1, MFPT, and Chapman-Kolmogorov errors pass;
- force target nondegenerate;
- covariance closure and innovation whiteness pass;
- three-seed stability;
- synchronized README, docs, configs, and artifacts.

## 7. Phase 2+ Execution Order

Only after Phase 0/1 identify the bottleneck:

1. repair the direct observed transition objective;
2. calibrate latent state scale, physical time, drift, and diffusion jointly;
3. add observed multi-step path likelihood;
4. add probabilistic state/operator prediction and calibration;
5. audit fixed CV versus learned representation;
6. test memory/Mori residuals only after Markov closure passes;
7. run three seeds and promote only after all gates pass.

No architecture change, encoder unfreezing, memory module, or new training run
is authorized by this document before the Phase 1 bottleneck report is written.

## 8. Current Status

- Phase 0 implementation: complete on 2026-09-11.
- Phase 0 audit: passed on the frozen manifests.
- Phase 1 endpoint screening: complete on 2026-09-11.
- Phase 1 bottleneck attribution: complete on 2026-09-11.
- Training changes: not started.
- Existing untracked file `scripts/run_training.py`: preserved.

## 9. Phase 0 Execution Result

The protocol audit is stored in:

```text
artifacts/phase0_protocol_audit/protocol_audit.json
artifacts/phase0_protocol_audit/protocol_audit.md
artifacts/phase0_protocol_audit/lspwm_contract.json
```

The canonical manifests passed the protein-disjoint check:

| Check | Result |
|---|---:|
| train proteins | 80 |
| validation proteins | 20 |
| test proteins | 20 |
| train/validation overlap | 0 |
| train/test overlap | 0 |
| validation/test overlap | 0 |
| frame interval | `1000 ps` |
| validation temperatures | `320, 348, 379, 413 K` |

The expected test/temperature-test protein overlap is `20`, because
`temperature_test` is a second-temperature view of the blind test proteins.
It is not used for checkpoint selection.

Phase 0 also found and repaired three reproducibility hazards:

1. `comprehensive_eval.py` could silently use the training manifest from the
   config. It now accepts an explicit `--manifest`, `--seed`, and
   `--batch-size`, and uses a seeded hierarchical sampler.
2. Historical checkpoints could fail to load because of the later-added
   `auxiliary_initializer` or DetSkill v3 KDE runtime buffers. The migration
   function now initializes only the optional missing module and drops only
   the non-canonical runtime buffers.
3. Adaptive KDE bandwidth could select an exact self-match and collapse to the
   clamp floor. The nearest-neighbor estimate now excludes exact self/duplicate
   matches. The regression test is in `tests/test_lspwm.py`.

The targeted regression suite passed:

```text
54 passed, 28 warnings
```

## 10. Phase 1 Execution Result

All endpoint screening outputs are under:

```text
artifacts/phase1_checkpoint_screening/
```

Each endpoint screen used:

- explicit `manifests/tmjepa_v3.validation.jsonl`;
- seed `6303`;
- batch size `4`;
- `80` batches;
- ensemble size `8`;
- horizons `24, 48, 96` frames.

### 10.1 Training-log skill versus held-out world-model skill

The training log measures the JEPA paired predictor. The endpoint evaluator
measures the LSP-WM thermodynamic transition step. They are intentionally
reported separately:

| Candidate | Training paired peak | Held-out paired diagnostic | Held-out LSP-WM endpoint macro skill |
|---|---:|---:|---:|
| Stage B v3 step 3000 | `0.8360` at step 1620 | `0.7377` | `0.00581` |
| DetSkill v1 step 3000 | `0.8892` at step 2090 | `0.8215` | `0.00636` |
| DetSkill v2 step 3000 | `0.8899` at step 2350 | `0.8463` | `0.00683` |
| DetSkill v3 step 3000 | `0.8967` at step 2860 | `0.8360` | `0.00726` |

For the direct LSP-WM endpoint macro skill, DetSkill v3 is the validation
winner, but its advantage over DetSkill v2 is small. The maximum endpoint
skill remains only about `0.7%`.

The held-out per-horizon endpoint skill for DetSkill v3 step 3000 is:

| Horizon | Deterministic skill | Energy skill |
|---:|---:|---:|
| 24 frames | `0.00316` | `-1.4286` |
| 48 frames | `0.00615` | `-2.3259` |
| 96 frames | `0.01247` | `-3.7544` |

The confirmatory test result for DetSkill v3 step 3000 is similar:

| Horizon | Deterministic skill | Energy skill |
|---:|---:|---:|
| 24 frames | `0.00320` | `-1.581` |
| 48 frames | `0.00625` | `-2.535` |
| 96 frames | `0.01223` | `-3.944` |

### 10.2 Scale-calibration evidence

For DetSkill v3 step 3000 on validation, the direct world-model evaluator
reports:

| Horizon | Target displacement norm | Deterministic displacement norm | Ensemble mean displacement norm | Ensemble spread norm |
|---:|---:|---:|---:|---:|
| 24 | `3.3809` | `0.0100` | `7.3298` | `19.4037` |
| 48 | `3.4252` | `0.0200` | `10.3914` | `27.3905` |
| 96 | `3.3614` | `0.0400` | `14.6990` | `38.8087` |

This is the decisive Phase 1 result:

- the deterministic drift is approximately `0.30%`, `0.58%`, and `1.19%` of
  the observed displacement at the three horizons;
- the stochastic ensemble is already much broader than the target at 24
  frames and becomes increasingly over-dispersed;
- the model is numerically finite, but its transition distribution is not
  calibrated to the observed state increments.

The corresponding mean transition NLL is about `392.4` on validation and
`392.8` on the confirmatory test run.

### 10.3 Force target sensitivity

For DetSkill v3 step 3000 on validation:

| KDE bandwidth | Target norm | Target component std | Predicted norm | Cosine | Degenerate |
|---:|---:|---:|---:|---:|---:|
| `0.10` | `0.1912` | `0.0384` | `0.0117` | approximately `0` | no |
| `0.05` | `0.0555` | `0.0111` | `0.0117` | approximately `0` | no |
| adaptive `0` after leave-one-out fix | `0.00154` | `0.00022` | `0.0117` | `0.0557` | no |

The old adaptive result of exactly zero target force was an evaluator
self-match bug and must be discarded. After the fix, adaptive KDE is finite
but bandwidth-sensitive. The force head still underestimates the target scale,
so force cosine is not a reliable primary optimization target until the KDE
measure and state scale are stabilized.

## 11. Final Phase 1 Bottleneck Diagnosis

### Primary bottleneck: thermodynamic transition-law calibration

The current LSP-WM transition law is the limiting component. The evidence is
consistent across validation and test:

1. protein-disjoint split integrity passes;
2. enabling the paired predictor improves the held-out paired diagnostic;
3. that improvement barely changes the direct LSP-WM endpoint forecast;
4. deterministic drift is almost zero relative to observed displacement;
5. ensemble variance is much too large;
6. transition NLL remains around `392`;
7. energy skill is strongly negative at all horizons.

The first repair target is therefore the one-step physical transition
distribution: latent scale, physical time conversion, drift magnitude,
diffusion magnitude, and ensemble sampling must be calibrated jointly.

### Secondary bottleneck: training/evaluation objective separation

The DetSkill fix trains a paired predictor with:

```yaml
paired_delta_weight: 0.005
lspwm_dynamic_weight: 0.01
lspwm_force_weight: 0.05
lspwm_rollout_enabled: false
lspwm_rollout_trajectory_nll_weight: 0.0
```

The encoders and target encoders are frozen and `encoder_lr_scale` is zero.
This explains why the paired predictor can look strong while the thermodynamic
world model remains poorly calibrated. It is a local endpoint-head repair, not
a transition-closure repair.

### Not the primary bottleneck

- Numerical stability: finite tensors and healthy PSD diffusion pass.
- Paired predictor capacity alone: DetSkill v2/v3 improve this diagnostic.
- Adaptive KDE implementation: the self-match bug is fixed; bandwidth
  sensitivity remains a measurement issue, not proof of a successful model.
- VAMP-2: held-out values around `24-26` show some representation signal, but
  VAMP-2 does not validate the transition distribution.

## 12. Authorization Boundary After Phase 1

Phase 1 authorizes the following next implementation steps:

1. add a dedicated transition-scale diagnostic and validation report;
2. calibrate one-step drift and diffusion against observed increments;
3. compare against persistence and a fitted linear/VAR baseline;
4. only after one-step closure improves, add observed multi-step path likelihood;
5. then evaluate probabilistic state/operator calibration.

Phase 1 does not authorize:

- claiming a new SOTA checkpoint;
- unfreezing the encoder;
- adding Mori memory as the first intervention;
- launching a large training run;
- using the test set for hyperparameter selection.

## 13. Phase 2 Transition Calibration Diagnostic

Date: 2026-09-11

The first Phase 2 diagnostic is implemented in:

```text
scripts/evaluate_lspwm_transition_calibration.py
```

The frozen checkpoint, config, and manifests were:

```text
artifacts/lspwm_v44_stage_b_detskill_fix_v3/checkpoint_step3000.pt
artifacts/lspwm_v44_stage_b_detskill_fix_v3/config.json
manifests/tmjepa_v3.train.jsonl
manifests/tmjepa_v3.validation.jsonl
```

The machine-readable result is:

```text
artifacts/phase2_transition_calibration/transition_calibration.json
```

The reproducible command was:

```bash
cd /liuyutian/protein_jepa_frame
PYTHONPATH=$PWD /liuyutian/protein_jepa/.venv/bin/python \
scripts/evaluate_lspwm_transition_calibration.py \
  --config artifacts/lspwm_v44_stage_b_detskill_fix_v3/config.json \
  --checkpoint artifacts/lspwm_v44_stage_b_detskill_fix_v3/checkpoint_step3000.pt \
  --train-manifest manifests/tmjepa_v3.train.jsonl \
  --validation-manifest manifests/tmjepa_v3.validation.jsonl \
  --output-dir artifacts/phase2_transition_calibration \
  --batch-size 4 \
  --max-train-batches 80 \
  --max-validation-batches 80 \
  --seed 6303 \
  --device cuda:0
```

### 13.1 Protocol

- `960` train rows and `960` validation rows were collected with the same
  hierarchical paired-window sampler.
- The train manifest was used only for baseline fitting.
- Ridge regularization was selected on a train-only protein holdout, separately
  for each horizon and for both target-space and delta-space fits.
- The validation manifest contains `20` unseen proteins at every horizon.
- No model parameter was updated, and the test manifest was not opened.
- The LSP-WM covariance is the executed analytic covariance
  `2 D(q, T) dt`, not an ensemble estimate.

### 13.2 Validation result

| Horizon | Persistence skill | Ridge target skill | Ridge delta skill | LSP-WM skill | LSP-WM drift/target norm | LSP-WM covariance / residual covariance |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | `0.0000` | `0.7287` | `0.7705` | `0.0032` | `0.0030` | `76.3x` |
| 48 | `0.0000` | `0.7122` | `0.7419` | `0.0062` | `0.0058` | `144.5x` |
| 96 | `0.0000` | `0.7188` | `0.7611` | `0.0125` | `0.0119` | `299.1x` |

The current LSP-WM transition NLL is:

```text
H24: 305.28
H48: 392.11
H96: 479.81
macro: 392.40
```

The corresponding LSP-WM model covariance traces are approximately
`431.9`, `863.7`, and `1727.4`, while the empirical residual covariance traces
are only `5.66`, `5.98`, and `5.77`. Even when the deterministic drift error is
included in the comparison, the model covariance is still approximately
`33.7x`, `65.8x`, and `137.1x` larger than the empirical innovation second
moment.

### 13.3 Bottleneck decision

The primary bottleneck is confirmed as **one-step LSP-WM transition-law
calibration**, with two coupled failures:

1. **Drift under-scale:** the physical PMF drift moves only about `0.30%`,
   `0.58%`, and `1.19%` of the observed latent displacement at H24, H48, and
   H96.
2. **Diffusion over-scale:** the executed `2D dt` covariance is one to three
   orders of magnitude broader than the observed latent increment residuals.

The ridge delta baseline beats persistence by a large margin on unseen proteins
at all three horizons. Therefore the current state representation contains
substantial predictive information, and representation insufficiency is not the
first bottleneck. The result also rules out paired predictor capacity as the
primary issue: the paired predictor improvement seen in Phase 1 is not
realized by the thermodynamic transition operator.

The next intervention should be a small, validation-only calibration study
around:

1. latent-state/time-unit calibration;
2. drift and diffusion scale fitting against observed increments;
3. a direct observed one-step transition objective;
4. re-evaluation of NLL, endpoint skill, and covariance closure.

The following remain deferred until that study passes:

- encoder unfreezing;
- a new large training run;
- multi-step path likelihood;
- Mori/memory architecture changes;
- test-set selection or SOTA claims.

### 13.4 Phase 2 status

- transition calibration implementation: complete;
- train-only ridge baseline: complete;
- validation comparison: complete;
- primary bottleneck attribution: complete;
- training or checkpoint modification: not started.

## 14. Phase 3 Physical Calibration Design

Date: 2026-09-11

The first intervention was implemented as a frozen-checkpoint diagnostic. It
does not modify the encoder, target encoder, latent projection, or checkpoint.
It follows the overdamped Langevin / Einstein relation:

```text
dq = -D(q, T) grad A(q, c) dt + sqrt(2 D(q, T) dt) dW
```

The calibration layer uses:

```text
D_calibrated = mobility_scale * D_model
grad A_calibrated = free_energy_scale * grad A_model
```

Therefore the drift is scaled by the product of the two factors, while the
covariance is scaled only by `mobility_scale`. This preserves the
fluctuation-dissipation coupling and does not insert an unconstrained
non-conservative delta predictor.

The calibration parameters are fitted only on a train-protein holdout. The
validation proteins are used only for the final report. The implementation and
result are:

```text
scripts/evaluate_lspwm_physical_calibration.py
artifacts/phase2_physical_calibration/physical_calibration.json
```

The full validation command used `80` train batches, `80` validation batches,
batch size `4`, seed `6303`, and the same frozen DetSkill v3 step-3000
checkpoint as Phase 2.

### 14.1 Scalar calibration result

The fitted scales were:

```text
mobility_scale     = 0.00298868
free_energy_scale  = 21815.5965
drift scale product = 65.1998
```

The large free-energy factor is evidence that the current scalar potential
gradient is strongly under-scaled in the learned latent coordinates. The
mobility factor independently reduces the excessive `2D dt` covariance.

| Horizon | Raw LSP-WM skill | Scalar-calibrated skill | Scalar-calibrated NLL | Covariance / innovation |
|---:|---:|---:|---:|---:|
| 24 | `0.0032` | `0.1733` | `631.04` | `0.12x` |
| 48 | `0.0062` | `0.2734` | `130.65` | `0.27x` |
| 96 | `0.0125` | `0.2839` | `-36.42` | `0.57x` |

This is a meaningful improvement over the raw transition law, but it is not
enough to match the train-fitted ridge delta baseline (`0.7705`, `0.7419`,
`0.7611`). Scalar calibration repairs magnitude and uncertainty scale, but not
the missing force-field direction and state dependence.

The result is stable in direction under a second sampler seed (`6304`):
calibrated skill was `0.2107`, `0.3073`, and `0.1587` at H24, H48, and H96.
The horizon variance means this is a diagnostic success, not a promotion gate.

### 14.2 Conservative force correction

The next candidate preserves the same FDT law and adds a scalar potential
correction in fixed, train-fitted PCA coordinates:

```text
Delta A(z, c) = b^T tanh(z) + tanh(z)^T W c
```

Its force is always the exact gradient of a scalar potential, and the
derivative of `tanh` is bounded. The implementation is:

```text
scripts/evaluate_lspwm_conservative_potential.py
artifacts/phase2_conservative_potential/conservative_potential.json
```

The full validation experiment used state PCA dimension `16`, condition PCA
dimension `8`, `144` correction parameters, train-only ridge selection, and
the same frozen checkpoint.

| Horizon | Scalar skill | Conservative-potential skill | Scalar NLL | Conservative NLL |
|---:|---:|---:|---:|---:|
| 24 | `0.1733` | `0.2504` | `631.04` | `531.96` |
| 48 | `0.2734` | `0.3585` | `130.65` | `73.53` |
| 96 | `0.2839` | `0.1817` | `-36.42` | `-6.95` |

The covariance-to-residual ratios after the conservative correction are
`0.23x`, `0.42x`, and `0.76x`. The bounded correction is numerically stable and
improves H24/H48, but its H96 regression shows that a single stationary
correction is not yet a reliable multi-horizon transition law.

### 14.3 Rejected intervention

An initial quadratic correction,

```text
Delta A = b^T z + 1/2 z^T H z + z^T W c,
```

was rejected after smoke validation. Although it is formally conservative,
the unconstrained Hessian had large negative curvature and generated
validation force magnitudes above `2.4e4`, compared with target displacement
norms around `4`. This is a physical extrapolation failure, not an acceptable
optimization result. The bounded `tanh` form replaced it.

### 14.4 Decision and next implementation

The physical evidence supports a staged frozen-head experiment:

1. retain the scalar mobility/free-energy calibration as the baseline;
2. train only a bounded conservative potential correction;
3. fit it with observed one-step transition NLL plus a conservative-force
   regularizer;
4. keep covariance tied to the same mobility scale through FDT;
5. use horizon-stratified protein-disjoint validation for selection;
6. require positive skill at H24, H48, and H96 and no covariance ratio above
   `2x` before considering encoder unfreezing.

The correction must be horizon-aware only through the physical `dt` already
present in the Langevin integrator. A free-standing horizon-specific delta
head is not authorized because it would bypass the physical transition law.

Still deferred:

- encoder unfreezing;
- changing the latent representation;
- adding Mori memory;
- large-scale training;
- opening the test manifest for selection.

### 14.5 Phase 3 status

- FDT-consistent scalar calibration: complete;
- second-seed stability check: complete;
- bounded conservative-force diagnostic: complete;
- unconstrained quadratic correction: rejected;
- frozen-head implementation: complete;
- frozen-head smoke training: complete;
- scalar-calibration full training: complete;
- endpoint-refinement pilot: complete and passed the one-step endpoint gate.

## 15. Phase 3 Frozen-Head Implementation

Date: 2026-09-11

### 15.1 Code path

The implementation is contained in the canonical LSP-WM path:

- `protein_jepa_frame/models/lspwm.py`
  - `PhysicalTransitionCalibration`;
  - `BoundedConservativePotential`;
  - `calibrated_overdamped_transition_nll`;
  - FDT-consistent calibrated transition parameters;
  - calibrated endpoint sampling.
- `protein_jepa_frame/models/jepa.py`
  - construction from `ModelConfig`;
  - old-checkpoint migration for new calibration keys;
  - model-level freeze contract.
- `protein_jepa_frame/training/train.py`
  - calibrated transition loss and conservative-force regularizer;
  - optimizer group containing only calibration parameters in frozen mode;
  - scale and gradient diagnostics in `train.jsonl`.
- `scripts/evaluate_lspwm_endpoint.py`
  - automatically evaluates the calibrated transition when enabled.

The transition used during calibration is:

```text
D_cal(q, T) = m D(q, T)
grad A_cal(q, c) = s grad A(q, c) + grad DeltaA(q, c)
Cov[q(t + dt) | q(t)] = 2 D_cal(q, T) dt
```

`DeltaA` is a bounded low-rank scalar potential:

```text
DeltaA(q, c) =
  scale * sum_j tanh(q U)_j *
  tanh(b_j + c W_j) / sqrt(rank)
```

The correction amplitudes are initialized to zero. Therefore the newly
constructed model starts with the scalar-calibrated law and cannot inject a
random force field into the existing latent representation.

### 15.2 Configuration

The complete run uses:

```text
configs/lspwm_v44_phase3_fdt_conservative_calibration.yaml
```

Initialization checkpoint:

```text
artifacts/lspwm_v44_stage_b_detskill_fix_v3/checkpoint_step3000.pt
```

Training manifest:

```text
manifests/tmjepa_v3.train.jsonl
```

The configuration trains three epochs of 1,000 steps, with batch size `8`.
The validation manifest is not used by the optimizer. The model has:

- `lspwm_physical_calibration_enabled: true`;
- mobility initialization `0.00298867696`;
- free-energy initialization `21815.5965`;
- conservative correction rank `16`;
- frozen context encoder, temporal encoder, target towers, latent projection,
  original PMF, original diffusion, bath, role readouts, and auxiliary head;
- `lspwm_calibrated_dynamic_weight: 0.01`;
- conservative-force regularization weight `1e-4`.

The resulting trainable parameter set is exactly:

```text
lspwm.world_model.physical_calibration.log_mobility_scale
lspwm.world_model.physical_calibration.log_free_energy_scale
lspwm.world_model.physical_calibration.conservative_correction.raw_state_basis
lspwm.world_model.physical_calibration.conservative_correction.bias
lspwm.world_model.physical_calibration.conservative_correction.condition_basis
```

### 15.3 Verification

Static checks and tests:

```text
python -m py_compile \
  protein_jepa_frame/config.py \
  protein_jepa_frame/models/lspwm.py \
  protein_jepa_frame/models/jepa.py \
  protein_jepa_frame/training/train.py \
  scripts/evaluate_lspwm_endpoint.py

pytest -q
```

Observed result:

```text
161 passed, 56 warnings
```

The 10-step GPU smoke run completed successfully. It reported:

- no non-finite loss;
- context and temporal encoder gradient norms equal to `0`;
- non-zero calibration gradient norm;
- checkpoint reload with no missing keys;
- positive smoke endpoint skill at H24/H48/H96.

Smoke artifacts:

```text
artifacts/lspwm_v44_phase3_fdt_conservative_calibration_smoke/
```

### 15.4 Full training command

The full run was launched with:

```bash
cd /liuyutian/protein_jepa_frame
PYTHONPATH=. python scripts/run_training.py \
  configs/lspwm_v44_phase3_fdt_conservative_calibration.yaml \
  > artifacts/lspwm_v44_phase3_fdt_conservative_calibration/training.log 2>&1
```

Expected final checkpoint:

```text
artifacts/lspwm_v44_phase3_fdt_conservative_calibration/checkpoint_latest.pt
```

The scalar-calibration full run completed and was evaluated on the explicit
validation manifest:

```text
artifacts/lspwm_v44_phase3_fdt_conservative_calibration/endpoint_validation.json
```

That checkpoint is retained as the scalar baseline. The promoted Phase 3
endpoint candidate is the separately trained bounded-correction pilot
documented in Section 16. No SOTA claim is made by checkpoint existence alone.

## 16. Phase 3 Endpoint Refinement Result

Date: 2026-09-12

The frozen-head endpoint refinement pilot completed successfully:

```text
configs/lspwm_v44_phase3_endpoint_refinement_pilot.yaml
artifacts/lspwm_v44_phase3_endpoint_refinement_pilot/checkpoint_step500.pt
artifacts/lspwm_v44_phase3_endpoint_refinement_pilot/endpoint_validation_step500.json
```

The pilot trained only the bounded conservative correction. The encoder,
temporal encoder, target towers, latent projection, original PMF, original
diffusion, bath, and calibrated mobility/free-energy scales remained frozen.
The correction is constructed as the gradient of a scalar potential, so the
drift remains conservative. Because the mobility scale was frozen, the
transition covariance and its FDT relationship were not changed by this
pilot.

The validation comparison used the same explicit protein-disjoint validation
manifest, seed `7311`, batch size `4`, `80` batches, and ensemble size `8`
for both checkpoints:

| Metric | Scalar calibration | Phase 3 correction | Change |
|---|---:|---:|---:|
| H24 deterministic skill | `0.1429` | **`0.2101`** | `+0.0672` |
| H48 deterministic skill | `0.2266` | **`0.3233`** | `+0.0967` |
| H96 deterministic skill | `0.2413` | **`0.2923`** | `+0.0510` |
| H24 energy skill | `0.2182` | **`0.2579`** | `+0.0396` |
| H48 energy skill | `0.2831` | **`0.3362`** | `+0.0531` |
| H96 energy skill | `0.2848` | **`0.3003`** | `+0.0155` |
| Mean transition NLL | `396.17` | **`328.96`** | `-67.21` |
| Mean endpoint skill | `0.2036` | **`0.2752`** | `+0.0716` |

Endpoint MSE also decreased at every horizon:

```text
H24: 0.05314 -> 0.04898
H48: 0.04936 -> 0.04319
H96: 0.04670 -> 0.04356
```

The ensemble spread norms were unchanged at the reported precision
(`1.0750`, `1.5214`, and `2.1445`), which is expected because the mobility
and therefore the FDT covariance were frozen. The improvement is consequently
attributable to the learned conservative drift correction rather than an
unconstrained variance adjustment.

### 16.1 Phase 3 decision

**Phase 3 one-step endpoint gate: passed.** The refinement improves
deterministic skill, energy skill, endpoint MSE, and transition NLL at all
three declared horizons on the held-out validation protocol. The checkpoint
is promoted as the current validated Phase 3 endpoint candidate:

```text
artifacts/lspwm_v44_phase3_endpoint_refinement_pilot/checkpoint_step500.pt
```

This promotion is intentionally narrower than a complete framework or
literature-wide SOTA claim. The following gates remain open:

- three-seed validation and temperature extrapolation;
- full comprehensive evaluation, including rollout and operator diagnostics;
- covariance closure and innovation whiteness under the calibrated evaluator;
- blind test evaluation after selection is frozen;
- multi-step path and memory closure;
- complete environment/action/intervention claims described by the README.

The next authorized phase is therefore **Phase 4: held-out robustness and
multi-step closure evaluation**, using the promoted Phase 3 candidate. No
encoder unfreezing or memory architecture change is authorized before those
checks are completed.
