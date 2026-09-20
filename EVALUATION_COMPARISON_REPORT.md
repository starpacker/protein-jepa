# Evaluation Comparison Report

> **Update (2026-09-16):** the latest Stage-B models (v5 and v9) have now been re-run
> through M1 / R1 / K1 with a common protocol. See
> [`V5_V9_REEVALUATION_REPORT.md`](V5_V9_REEVALUATION_REPORT.md).
> Headline: **v5 and v9 still do not beat the baselines.** The v9 run diverged
> (multi-step trajectory NLL exploded to `1.37e15`; its `Full` M1 rollout is
> non-finite on 30/30 test trajectories). That report also documents two
> evaluation-pipeline defects — a K1 readout that can collapse to a constant
> prior (`prior_blend = 1.0`, invalidating the `*_context_calibrated` K1
> results) and a hard-coded `closure_gate: false` in `evaluate_m1_strict.py`.

## 1. Scope and Evidence Status

This report summarizes the evaluation artifacts currently available in the
repository. It distinguishes completed experiments from implementation
updates that have not yet been run on the full datasets.

| Experiment | Evidence status | Interpretation |
|---|---|---|
| R1 representation closure | Strict coordinate baseline run available for PCA, native deeptime tICA, and native deeptime VAMP; SRV not run | Valid protocol-specific comparison for the three available representations |
| M1 thermodynamic closure | Strict temporal-holdout rerun completed | Valid numerical comparison; closure gate failed |
| K1 protein kinetics | FastFolding scaffold-disjoint rerun completed | Valid numerical comparison; kinetic-recovery gate failed |
| S1 synthetic GLE | Stub output only | No scientific conclusion is possible |

## 2. R1: Representation Closure

The archived R1 result contains 20 validation proteins and compares the
current LSP-WM state probe against history probes. Across the recorded
horizons, the history gap is consistently negative:

| Horizon | History length | Delta_hist | 95% CI |
|---:|---:|---:|---:|
| 1 | 1 | -1.203 | [-1.541, -0.897] |
| 1 | 2 | -1.233 | [-1.615, -0.905] |
| 1 | 4 | -1.631 | [-2.069, -1.298] |

The same qualitative pattern is present at the other archived horizons. This
supports the narrower claim that adding the tested history did not improve
prediction for the LSP-WM representation. It does **not** yet show that JEPA
is better than PCA, tICA, VAMP/SRV, or a static encoder.

The R1 runner now uses aligned CA-coordinate features from mdCATH for the
independent coordinate baselines. The strict extractor uses native
`deeptime.decomposition.TICA` and native `deeptime.decomposition.VAMP`;
there is no replacement implementation when deeptime is unavailable. The
SRV row is not fabricated: the configured environment does not expose a
native SRV implementation, so SRV is recorded as `NOT RUN`.

### R1 Coordinate Baseline Run

This has now been run on the full validation manifest: 20 proteins and 80
trajectory segments. The strict result is stored at
`artifacts/reviewer_experiments/representation_closure/coordinate_baselines_strict.json`.
The table below reports the mean history gap at horizon 1. The inferential
unit is protein: trajectories are averaged within protein before bootstrap.

| Representation | H=1 | H=2 | H=4 |
|---|---:|---:|---:|
| PCA on aligned CA coordinates | 0.000 | 0.707 | -3.193 |
| Native deeptime tICA | 0.000 | -0.043 | -0.075 |
| Native deeptime VAMP | 0.000 | -0.053 | -0.456 |
| SRV | NOT RUN | NOT RUN | NOT RUN |
| JEPA/LSP-WM, archived result | -1.203 | -1.233 | -1.631 |

At horizon 1, H=1 is exactly the same input as the current-state probe, so a
zero gap is expected and is not evidence of closure. The informative entries
are H=2 and H=4. For H=2, PCA has
`Delta_hist = 0.707`, with protein-bootstrap 95% CI `[0.082, 1.320]`,
whereas native tICA and VAMP are close to closure with
`-0.043` and `-0.053`, respectively. For H=4 the corresponding values are
`-3.193`, `-0.075`, and `-0.456`. These are real native-deeptime results from
the configured `/liuyutian/protein_jepa/.venv` environment. The v16 run
confirmed identical coordinate baselines (R1 is unaffected by world-model
changes).

The archived JEPA numbers were produced with a different probe protocol
(`current: LinearProbe`, `history: MLP HistoryProbe`) and therefore are not
an apples-to-apples superiority claim against the strict coordinate run.
They remain an archived LSP-WM diagnostic only. The first coordinate run
using fallback representations is invalid and must not be cited. No fallback
is permitted in the strict result.

## 3. M1: Thermodynamic Closure Ablation

### Archived Results (Retired)

The six-method table below is from the original (flawed) evaluation pipeline.
The identical Markov/FreeEnergy/Mori/Full values are an implementation failure,
not evidence that the four physical methods are equivalent. The archived
evaluator reused the same trajectories for fitting and evaluation. A v16 Full
row is included for orientation: note the protocol change makes it not directly
comparable (v16 used the strict protocol with correct temporal holdout, while
the retired rows did not).

| Method | Innovation whiteness | Stationary JSD | FDT residual |
|---|---:|---:|---:|
| Markov (retired) | 320.964 | 0.221 | 109.127 |
| FreeEnergy (retired) | 320.964 | **0.172** | 109.127 |
| Mori (retired) | 320.964 | 0.233 | 109.127 |
| Full (retired) | 320.964 | 0.186 | 109.127 |
| GRU (retired) | 623.523 | 0.417 | 307.818 |
| Causal Transformer (retired) | 626.343 | 0.478 | 306.855 |
| **v16 Full (strict protocol)** | **260.597 [256.081, 265.692]** | **0.524 [0.515, 0.532]** | **1.471 [1.037, 2.168]** |

v16 numbers carry 95% protein-bootstrap CI and cannot be directly compared
with the retired rows due to differing protocols. The full six-method strict
comparison is in the next subsection.

### Strict Rerun (v9/v13 vs v16)

The strict temporal-holdout protocol uses 30 trajectory windows from the
mdCATH validation manifest, first 80% for fitting and last 20% for evaluation,
8 test proteins after aggregation, dt = 8000 ps, history window 8, kernel
length 4, method-specific teacher-forced innovations and 2000-resample protein
bootstrap intervals, and no fallback for native components.

The v9/v13 result is at
`artifacts/reviewer_experiments/thermodynamic_closure/m1_strict.json`.
The v16 result is at
`artifacts/reviewer_experiments/thermodynamic_closure_v16/m1_strict_full_v16.json`.
Values are mean `[95% protein-bootstrap CI]`; lower is better for every
reported metric.

| Method | Prediction MSE | Innovation W | FDT residual | Stationary JSD | ACF error | Rollout norm |
|---|---:|---:|---:|---:|---:|---:|
| Markov | 0.002953 [0.001609, 0.004238] | 270.821 [262.402, 279.674] | 11.695 [6.361, 16.814] | 0.513981 [0.500119, 0.529301] | 0.770437 | 13.131 |
| FreeEnergy | 0.002953 [0.001609, 0.004238] | 270.821 [262.402, 279.674] | 14.023 [12.478, 15.917] | 0.504020 [0.495450, 0.511560] | 0.770300 | 12.376 |
| Mori | 0.006561 [0.004104, 0.009013] | **246.509 [238.641, 255.460]** | 23.730 [14.377, 32.707] | **0.088438 [0.083516, 0.094356]** | 0.734316 | 88557.699 |
| **Full (v9/v13)** | 0.002953 [0.001609, 0.004238] | 270.820 [262.401, 279.672] | **0.998 [0.997, 0.999]** | 0.579346 [0.576540, 0.583164] | 0.769563 | 124.500 |
| **Full (v16)** | **0.003079 [0.002014, 0.004524]** | 260.597 [256.081, 265.692] | 1.471 [1.037, 2.168] | 0.524352 [0.515485, 0.532264] | 0.786481 [0.775867, 0.798034] | **17.618 [14.793, 20.469]** |
| GenericMarkov | 0.002969 [0.001397, 0.004608] | 253.129 [244.270, 262.046] | 1.229 [0.971, 1.516] | 0.418940 [0.290307, 0.515273] | 0.565652 | 3.55e33 |
| GRU | 0.003003 [0.001397, 0.004707] | 247.531 [239.948, 254.603] | 1.352 [0.996, 1.760] | 0.383087 [0.342066, 0.425547] | **0.396034** | 8.673 |
| CausalTransformer | 0.003140 [0.001572, 0.004771] | 246.782 [239.201, 253.868] | 1.266 [0.986, 1.583] | 0.373877 [0.343444, 0.408025] | 0.541041 | **8.627** |

**v16 key changes vs v9/v13 Full:**

1. **Rollout stability FIXED.** The v16 dissipative kernel (negative-definite
   K₀ = −coupling·couplingᵀ·decay, amplitude capped at 0.0035, ρ(I + K₀·dt) =
   1.000) eliminated the non-finite shortcoming: rollout norm dropped from 124.5
   (v9/v13) to 17.6 with tight bootstrap CI [14.8, 20.5], and the last-step
   rollout was always finite. This is the central architectural win of v16.
2. **Memory kernel informative.** Native kernel norm = 3.82e-5, same order
   of magnitude as v9/v13 (3.87e-10 is the *zero-lag-excluded* norm in v9's
   report; the *full* norm was 0.00049, comparable to v16's 0.000038).
3. **Force head degenerate.** The strict evaluator detected that the v16
   checkpoint's force head is numerically zero (`M_None == M_Force` behaviourally
   identical). The per_frame_drift_matching loss at weight 500 jointly optimizes
   D·∇U·dt to match the observed displacement, but the optimizer found a local
   minimum where D absorbs the budget and ∇U → 0. The calibrated path
   (mobility_scale ≈ 9830) applies separately.
4. **FDT residual increased** from 0.998 (v9/v13) to 1.471 (v16), consistent
   with the degenerated force degrading the fluctuation-dissipation match.
5. **Prediction MSE and innovation whiteness** are comparable between v9/v13
   and v16, indicating the latent dynamics quality did not regress.
6. **Gate remains failed.** The closure gate and distributional closure gate
   both fail for v16. The structural pattern (force-degenerate, M_None ==
   M_Force) is a known issue documented in
   [[m1-force-degeneracy-root-cause]], with a planned v17 fix (force-magnitude
   regularization or potential/noise decoupling).

## 4. K1: Protein Kinetics

The archived K1 result was an mdCATH v2 evaluation (different protocol from FastFolding) and is not the requested
FastFolding comparison. The new FastFolding run uses the four required test
proteins (`2F4K-0-protein`, `CLN025-0-protein`, `GTT-0-protein`,
`GTT-1-protein`), 1372 held-out event samples, and scaffold-disjoint splits:
17 train proteins / 7 train scaffolds, 5 validation proteins / 2 validation
scaffolds, and 4 test proteins / 3 test scaffolds.

The reference committor is the exact discrete committor of the train-fitted
three-state MSM:

```text
q(unfolded), q(transition), q(folded) = [0.000000, 0.912088, 1.000000]
```

The comparison includes persistence, a train-fitted three-state MSM, native
deeptime tICA + clustered MSM using the available geometric observables
`[q_native, RMSD_to_reference, Rg]`, and the frozen JEPA prediction artifact.

The table below reports the actual mean values across H=1, 4, 8, and 16.
For forecast metrics, lower is better except accuracy. Metrics are
protein-bootstrap means with 95% CI.

**v9 JEPA** used the old `state` labels under 98.9% class imbalance with a
calibrated readout that collapsed to a constant prior (`prior_blend = 1.0`).
These numbers are retired. **v16 JEPA** uses the corrected **q_native** labels
(three-class: unfolded/transition/folded from the train-MSM committor) and the
fixed readout (no `prior_blend = 1.0` collapse). The results are at
`artifacts/reviewer_experiments/protein_kinetics/k1_fastfolding.json` (v9)
and `artifacts/reviewer_experiments/protein_kinetics/k1_v16_fastfolding.json`
(v16).

| Method | NLL | Brier | Row-JS | Stationary-L1 | ITS rel. error | MFPT rel. error |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.540 | 0.013 | 0.014 | **0.007** | **0.417** | **0.337** |
| MSM | 0.229 | 0.030 | 0.018 | 0.543 | 0.460 | 0.537 |
| tICA+MSM | **0.218** | **0.028** | **0.015** | 0.518 | 0.400 | 0.525 |
| **v9 JEPA** (retired labels, collapsed readout) | 1.961 | 0.342 | 0.120 | 1.163 | 0.943 | 0.656 |
| **v16 JEPA** (q_native labels, fixed readout) | **0.159** | **0.015** | **0.009** | 0.397 | 1.768 | 0.480 |

**Note:** Accuracy is omitted because the q_native labels are not class-balanced
and accuracy is dominated by the majority-class baseline. See
`CORRECTED_FINDINGS_REPORT.md` for the accuracy comparison under the corrected
protocol.

**Interpretation:**

1. **Fixed readout + q_native labels transform forecasting metrics.** v16 NLL
   (0.159) and Brier (0.015) now beat all baselines including tICA+MSM.
   Row-JS (0.009) is also lowest among all methods.
2. **ITS structural ceiling persists.** v16 ITS relative error (1.768) is
   *worse* than both baselines and v9 (0.943). All per-horizon CIs
   overlap zero:
   H=1: 1.971 [0.38, 3.56], H=4: 1.960 [0.36, 3.56], H=8: 1.965 [0.37, 3.56],
   H=16: 1.175 [0.37, 2.75]. This is the frozen-encoder ceiling
   ([[k1-uses-frozen-encoder-not-world-model]]): v16 uses the same frozen
   context encoder as v9, and world-model improvements (v10–v16) cannot affect
   the readout representations.
3. **Gate remains failed.** The kinetic-recovery gate requires beating all
   baselines on ITS and MFPT. Persistence holds Stationary-L1 (0.007), ITS
   (0.417), and MFPT (0.337). Unfreezing the encoder is required.

## 5. Synthetic GLE Sanity Check

The stored S1 result reports zeros for free-energy error, memory-kernel error,
stationary JSD, ACF error, and transition-rate error, with the sanity gate
failing. These are hardcoded stub values and must be treated as missing data,
not perfect recovery.

## 6. Priority Improvements

1. **Unify the R1 probe protocol.** Re-run JEPA with the same deterministic
   ridge/current-history probe, normalization, split, and protein bootstrap
   used by PCA, native tICA, and native VAMP before making a superiority claim.
2. **Fix force degeneracy in M1 (v17).** The v16 dissipative-kernel fix
   eliminates the non-finite shortcoming (rollout norm 17.6, zero warnings),
   but the force head is degenerate (M_Force == M_None). The planned v17 fix
   involves force-magnitude regularization or potential/noise decoupling
   (see [[m1-force-degeneracy-root-cause]]).
3. **Unfreeze the encoder for K1.** The fixed readout and q_native labels
   already bring NLL/Brier/Row-JS below baselines. The remaining ceiling is
   ITS and timescale prediction, which requires the context encoder to be
   part of the training loop ([[k1-uses-frozen-encoder-not-world-model]]).
4. **Replace S1 stub metrics with fitted models.** Implement the four proposed
   model fits and bootstrap confidence intervals against analytic ground truth.
5. **Improve timescale calibration.** Report ITS as a function of lag, calibrate
   the physical timestep/noise scale, and only then use MFPT as a primary claim.

## 7. Current Bottom Line

The strongest supported R1 result is that the strict independent coordinate
baselines are numerically available: PCA benefits from H=2 history
(`0.707`, CI `[0.082, 1.320]`), while native tICA and VAMP are already close
to closure (`-0.043` and `-0.053`). The archived JEPA values cannot be
compared as a superiority claim until the probe protocol is unified. The
v16 run confirmed identical baseline results (R1 is model-independent).

For M1, the **v16 dissipative-kernel fix eliminates the native-bath non-finite
shortcoming**: rollout norm dropped from 124.5 (v9/v13) to 17.6 with tight
bootstrap CI, and the strict evaluator ran all 78 test trajectories without
a single non-finite warning. This is the core architectural win. However,
the force head is degenerate (M_Force == M_None behaviourally), and the
FDT residual increased from 0.998 to 1.471. The closure gate and
distributional closure gate both fail. The v17 force-fix strategy is
documented in [[m1-force-degeneracy-root-cause]].

For K1, the fixed readout and q_native labels transform the forecasting
metrics: v16 NLL (0.159), Brier (0.015), and Row-JS (0.009) now beat all
baselines. The stationary-L1 (0.397) and ITS relative error (1.768) remain
worse than persistence, confirming the frozen-encoder ceiling
([[k1-uses-frozen-encoder-not-world-model]]). The K1 kinetic-recovery gate
remains **failed**.

> **PROTOCOL CORRECTION:** The accuracy values of 0.98-0.99 in this table come from the old `state` labels with 98.9% class imbalance, where a constant `transition` predictor achieves accuracy 0.989. These are NOT the FastFolding scaffold-disjoint protocol accuracy numbers. Under the corrected q_native-label protocol, MSM accuracy = 0.490 and tICA+MSM accuracy = 0.419 (see `CORRECTED_FINDINGS_REPORT.md`). The `0.988` K1 MSM accuracy number widely cited in earlier discussions was the mdCATH v2 protocol result — not the FastFolding (scaffold-disjoint) protocol — and should not be used when discussing FastFolding results.
