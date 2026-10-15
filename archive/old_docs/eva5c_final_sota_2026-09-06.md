# EVA5C Final Internal SOTA

Date: 2026-09-06

## Final selection

EVA5C is the final endpoint-energy model for the manuscript freeze. It combines
the frozen U250 thermodynamic stochastic propagator with a causal JEPA-history
projected-force closure and an analytic quadratic short-time gate.

- Checkpoint: `artifacts/eva5c_tmjepa_v3_quadratic_latent_closure/checkpoint_step375_gate2.pt`
- Serialized config: `artifacts/eva5c_tmjepa_v3_quadratic_latent_closure/config.json`
- Evaluation: `artifacts/sota_eval_20260905/eva5c_final_checkpoint_seed6303_ck0.json`
- Parent trained checkpoint: `artifacts/eva5b_tmjepa_v3_frozen_latent_closure/checkpoint_step375.pt`

This is an internal validation SOTA, not an external test-set or literature-wide
SOTA. Checkpoint selection and the gate-timescale comparison used the validation
set, so a held-out test evaluation is still required for the final paper table.

## Model logic

The final endpoint predictor has three parts:

1. U250 supplies the fixed collective-variable measure, Eyring-Einstein
   temperature law, state-dependent diffusion, observation model, and
   thermodynamic rollout.
2. A bounded projected-force head reads a multiscale summary of causal JEPA
   frame states: endpoint state, endpoint-minus-recent mean,
   recent-minus-global mean, and the local latent increment.
3. The endpoint correction is multiplied by
   `(1 - exp(-tau / tau_c))^2`, with `tau_c = 2` base steps. This enforces
   zero correction at zero horizon and the expected `O(tau^2)` short-time
   coordinate response to an unresolved force.

Only the projected-force closure was trained in the EVA5B parent. The context
encoder, temporal encoder, learned CV projection, potential, kinetics,
diffusion, observation noise, and endpoint calibration were frozen. EVA5C adds
no fitted parameter; it embeds the analytic short-time gate in the checkpoint.

The explicit auxiliary Mori bath is disabled in EVA5C. This is deliberate:
EVA4/EVA5 experiments repeatedly showed that the inferred bath had negative or
negligible causal intervention benefit. The history-dependent projected-force
closure is retained because its ablation signal was positive, but the paper
must not claim that the auxiliary bath itself was solved.

## Evaluation protocol

All endpoint comparisons below use the same protocol:

- validation manifest: `manifests/tmjepa_v3.validation.jsonl`
- seed: `6303`
- 50 batches, batch size 4, 200 sampled contexts
- all 20 validation proteins and temperatures 320/348/379/413 K
- horizons: 1/2/4/8/16 ns
- ensemble size: 8
- CK sampling disabled so endpoint RNG is isolated and comparable

## Endpoint results

| Model | Macro skill | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns |
|---|---:|---:|---:|---:|---:|---:|
| U250 encoder dynamics | 28.123% | 34.900% | 34.002% | 30.814% | 23.578% | 17.321% |
| EVA3-250 projected force | 28.572% | 34.872% | 34.314% | 31.020% | 24.269% | 18.383% |
| EVA4-375 CV-history Mori | 28.616% | 34.705% | 34.224% | 31.247% | 24.189% | 18.717% |
| EVA5-500 joint latent-history Mori | 28.491% | 34.862% | 34.188% | 31.068% | 24.028% | 18.307% |
| EVA5B-375 frozen latent closure | 28.480% | 33.905% | 33.920% | 31.153% | 24.403% | 19.019% |
| **EVA5C quadratic latent closure** | **28.687%** | **34.773%** | **34.084%** | **31.175%** | **24.385%** | **19.018%** |

Against U250, EVA5C gains `+0.564` percentage point in macro skill,
`+0.807` at 8 ns, and `+1.697` at 16 ns. Against EVA4-375 it gains
`+0.071` macro point and `+0.301` at 16 ns, while EVA4 remains slightly
better at 4 ns.

The marginal 90% coverages for EVA5C are:

| Horizon | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns |
|---|---:|---:|---:|---:|---:|
| Coverage | 94.62% | 93.05% | 90.77% | 90.97% | 91.17% |

## Closure and physical diagnostics

EVA5C preserves U250's stochastic propagator, so the closure diagnostics are
unchanged except for endpoint statistics:

| Metric | U250 | EVA4-375 | EVA5C |
|---|---:|---:|---:|
| Innovation covariance RMSE | 0.133973 | 0.136956 | 0.133973 |
| Path NLL / dimension | -0.713266 | -0.719738 | -0.713266 |
| Lag-1 corr RMS | 0.04874 | 0.04989 | 0.04874 |
| Lag-2 corr RMS | 0.04001 | 0.04004 | 0.04001 |
| Lag-4 corr RMS | 0.02648 | 0.02656 | 0.02648 |
| Lag-8 corr RMS | 0.02996 | 0.02939 | 0.02996 |
| Minimum diffusion eigenvalue | 3.248e-7 | 3.239e-7 | 3.248e-7 |
| Mean diffusion condition number | 54.61 | 52.69 | 54.61 |
| Maximum diffusion condition number | 70.73 | 67.96 | 70.73 |
| Minimum diffusion increment std | 0.02549 | 0.02545 | 0.02549 |
| Integrated memory/diffusion ratio | 0.0000 | 0.0330 | 0.0000 |

EVA5C passes:

- complete temperature-by-horizon sampling
- complete protein coverage
- positive skill at every temperature and horizon
- diffusion PSD, nondegeneracy, and conditioning gates

EVA5C does not pass:

- strict per-temperature coverage calibration, because 320 K / 1 ns coverage
  is 95.96%, just above the 95% upper bound
- per-temperature covariance eigenvalue closure
- per-temperature multi-lag whiteness

The evaluator labels the Mori gate as vacuously passed when the bath is
disabled. This must not be described as evidence for predictive auxiliary
memory.

## Ablation conclusions

The final experiment tree supports four conclusions:

1. Opening encoder dynamics in U250 was useful but small.
2. Forcing a nonzero Mori-memory magnitude did not make the bath predictive.
3. Causal JEPA latent history contains useful long-range information, but joint
   training and excessive closure learning rates degrade short horizons.
4. Enforcing the correct short-time order on the bounded history closure
   recovers short-horizon accuracy while retaining the 8/16 ns gain.

The `gate=4` control reached 28.607% macro skill, below `gate=2` at 28.687%.
The ungated EVA5B checkpoint reached 28.480%. This establishes that the final
gain is associated with the short-time consistency constraint, not merely
checkpoint promotion.

## Encoder information audit

A fresh frozen-readout audit was completed for the online representation shared
by U250 and EVA5C. The quick protocol used 10 trajectories for VAMP/linear
probes and 2000 labeled clips for the trajectory-heldout MSM probe.

| Readout | VAMP lag-6 dim-5 | VAMP lag-6 dim-10 | MSM accuracy | Balanced accuracy | Macro-F1 | Linear R2 lag-6 |
|---|---:|---:|---:|---:|---:|---:|
| Causal endpoint 64D | 0.3805 | 0.6458 | 40.82% | 35.00% | 30.64% | -0.464 |
| Fixed CV 16D | **0.3834** | **0.6531** | **53.10%** | 43.68% | **38.17%** | not run |
| Learned CV 4D | 0.2902 | 0.2902 | 45.00% | 35.44% | 29.10% | not run |
| Full CV 20D | 0.3815 | 0.6519 | 51.96% | **43.80%** | 37.36% | -0.239 |

The fixed CV supplies most of the useful metastable information. The learned
readout does not improve VAMP or MSM accuracy, and the causal latent is not a
linearly closed state at the first non-overlapping lag. See
`docs/eva5c_encoder_information_audit_2026-09-06.md` for protocol details and
claim boundaries. These quick values require a full 50-trajectory rerun before
being used as final paper estimates.

## Claim boundary

Supported:

> A thermodynamically structured stochastic JEPA with a bounded,
> history-conditioned projected-force closure and a quadratic short-time
> consistency constraint improves internal multi-horizon endpoint energy skill,
> with the largest gain at 16 ns.

Not supported:

- 60% endpoint skill under the current protocol
- complete innovation closure or whiteness
- a predictive auxiliary Mori bath
- external SOTA without a held-out test and matched baseline reruns
- learned-encoder MSM/VAMP superiority; the fresh audit finds the opposite

## Verification

- Final checkpoint strict-load test: passed; embedded gate value is `2.0`.
- Final checkpoint re-evaluation: bitwise-identical horizon skills to the
  command-line gate ablation.
- TM-JEPA V3 focused suite: 21 passed.
- A broader suite run reached 36 passing tests before it was interrupted to
  release shared trajectory I/O for the final evaluations; it did not report a
  failure before interruption.
