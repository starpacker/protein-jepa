# Phase 8S TM-JEPA V3 Endpoint/Closure SOTA Report

**Date:** 2026-09-05

**Raw endpoint SOTA checkpoint:** `artifacts/phase8s_tmjepa_v3_slow_cv_floor/checkpoint_step250.pt`

**Final guarded checkpoint:** `artifacts/phase8t_tmjepa_v3_guarded_closure/checkpoint_step250.pt`

**Config:** `configs/phase8s_tmjepa_v3_slow_cv_floor.yaml`

**Primary evaluation:** `artifacts/sota_eval_20260905/s250_slow_cv_floor_seed6303_ck0.json`

## Executive Summary

Phase 8S/S250 is the highest raw endpoint-skill TM-JEPA V3 checkpoint under the Phase-8 validation protocol. Phase 8T/T250 is the recommended manuscript checkpoint: it gives up only `0.262 percentage point` of macro endpoint skill relative to S250 while restoring the diffusion condition-number gate and improving lag-1 innovation whiteness.

Accurate paper-facing claim:

> Phase 8T is the final guarded TM-JEPA V3 model for the manuscript: it preserves the large endpoint-skill gain over G/N while keeping diffusion PSD, nondegenerate, and well-conditioned.

Important limitation:

> Phase 8T is not a full promotion-pass physical closure model. It still fails strict per-temperature coverage, per-temperature covariance closure, and per-temperature multi-lag whiteness gates. These failures are now closure/calibration residuals rather than a visibly ill-conditioned diffusion tensor.

## Evaluation Protocol

| Item | Value |
|---|---:|
| Validation manifest | `manifests/tmjepa_v3.validation.jsonl` |
| Checkpoint | `phase8s/.../checkpoint_step250.pt` |
| Seed | 6303 |
| Batch size | 4 |
| Max batches | 50 |
| Sampled contexts | 200 |
| Horizons | 1/2/4/8/16 ns |
| Closure transitions | 3400 |
| CK samples | 0 |
| Innovation lags | 1/2/4/8 |
| Temperatures covered | 320/348/379/413 K |
| Proteins covered | 20/20 |

Energy skill is `1 - model_energy_score / persistence_energy_score`; higher is better.

## Checkpoint Comparison

All rows below use seed 6303 and the aligned Phase-8 endpoint validation protocol.

| Checkpoint | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns | Macro |
|---|---:|---:|---:|---:|---:|---:|
| N14 endpoint-calibrated previous SOTA | 30.05% | 27.74% | 23.51% | 19.08% | 6.81% | 21.438% |
| O250 fast closure | 31.28% | 29.99% | 26.19% | 16.59% | 6.09% | 22.032% |
| P125 diagonal closure | 31.33% | 30.05% | 26.25% | 16.61% | 6.11% | 22.071% |
| Q375 joint Q/R closure | 34.04% | 33.06% | 29.88% | 22.21% | 15.16% | 26.868% |
| R250 spectral closure | 34.52% | 33.61% | 30.47% | 23.04% | 16.56% | 27.638% |
| **S250 slow-CV floor** | **34.92%** | **33.98%** | **30.87%** | **23.55%** | **17.44%** | **28.150%** |
| **T250 guarded closure** | 34.72% | 33.78% | 30.66% | 23.27% | 17.00% | 27.888% |

Phase 8S improves over N14 by `+6.712 pp` macro, `+4.47 pp` at 8 ns, and `+10.63 pp` at 16 ns. Relative to R250, it adds `+0.512 pp` macro and improves every horizon.

Phase 8T improves over N14 by `+6.450 pp` macro while avoiding the S250 diffusion-condition failure. For the paper, report T250 as the final physically guarded model and S250 as the raw endpoint upper bound.

## Final T250 Metrics

| Horizon | Energy skill | 90% coverage |
|---|---:|---:|
| 1 ns | 34.72% | 95.10% |
| 2 ns | 33.78% | 93.40% |
| 4 ns | 30.66% | 91.25% |
| 8 ns | 23.27% | 90.70% |
| 16 ns | 17.00% | 91.22% |

T250 closure and physics:

| Metric | T250 |
|---|---:|
| Innovation covariance RMSE | 0.1383 |
| Path NLL / dim | -0.7269 |
| Lag-1/2/4/8 corr RMS | 0.0493 / 0.0374 / 0.0266 / 0.0297 |
| Lag-1 diagonal corr RMS | 0.1557 |
| Minimum diffusion increment std | 0.0255 |
| Maximum diffusion condition number | 70.34 |

T250 promotion failures are limited to strict per-temperature coverage/covariance/whiteness. It passes diffusion PSD, nondegenerate diffusion, diffusion well-conditioned, complete grid, complete protein coverage, and positive skill in every temperature/horizon cell.

## Endpoint Metrics

| Horizon | Rows | Model energy score | Persistence score | Energy skill | 90% coverage | Memory-reset ES | Filter advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 ns | 200 | 0.473340 | 0.727363 | 34.924% | 94.925% | 0.477947 | +0.004608 |
| 2 ns | 200 | 0.531291 | 0.804699 | 33.976% | 93.125% | 0.538390 | +0.007099 |
| 4 ns | 200 | 0.614170 | 0.888395 | 30.867% | 90.650% | 0.605762 | -0.008408 |
| 8 ns | 200 | 0.702757 | 0.919183 | 23.545% | 90.175% | 0.691541 | -0.011216 |
| 16 ns | 200 | 0.834606 | 1.010848 | 17.435% | 90.350% | 0.832986 | -0.001620 |

The aggregate coverage is much closer to the target 90% than G/N14. The filter advantage is positive at 1/2 ns and negative at 4/8/16 ns, so the final paper should not claim a universal Mori-filter gain.

## Closure Metrics

| Metric | S250 |
|---|---:|
| Innovation mean RMS | 0.012124 |
| Innovation covariance RMSE | 0.136716 |
| Innovation covariance min eigenvalue | 0.012972 |
| Innovation covariance max eigenvalue | 1.378963 |
| Path NLL per dimension | -0.752880 |

Compared with N14/G, S250 improves covariance closure and lag-1 whiteness substantially, but its path NLL remains worse than the older N14 value reported as about `-0.842`.

## Innovation Lag Diagnostics

| Lag | Cov RMS | Cov max abs | Corr RMS | Corr max abs | Diag corr RMS | Diag mean | Off-diag corr RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.029825 | 0.269611 | 0.054515 | 0.377076 | 0.179751 | -0.072410 | 0.037786 |
| 2 | 0.017655 | 0.105974 | 0.036627 | 0.196267 | 0.081252 | -0.043535 | 0.032630 |
| 4 | 0.013988 | 0.065015 | 0.026534 | 0.098781 | 0.044924 | -0.038193 | 0.025197 |
| 8 | 0.015816 | 0.087445 | 0.029486 | 0.136203 | 0.041994 | -0.007914 | 0.028677 |

The remaining whiteness failure is still concentrated in lag-1 diagonal correlations. The off-diagonal residual is already much smaller.

## Temperature-Resolved Endpoint Metrics

| Temperature | Horizon | Rows | Energy skill | 90% coverage |
|---:|---:|---:|---:|---:|
| 320 K | 1 ns | 52 | 35.083% | 95.962% |
| 320 K | 2 ns | 52 | 34.483% | 94.327% |
| 320 K | 4 ns | 52 | 31.150% | 89.423% |
| 320 K | 8 ns | 52 | 27.530% | 90.096% |
| 320 K | 16 ns | 52 | 13.039% | 89.904% |
| 348 K | 1 ns | 47 | 37.602% | 94.787% |
| 348 K | 2 ns | 47 | 34.949% | 92.979% |
| 348 K | 4 ns | 47 | 30.483% | 92.234% |
| 348 K | 8 ns | 47 | 28.062% | 91.809% |
| 348 K | 16 ns | 47 | 19.573% | 92.872% |
| 379 K | 1 ns | 47 | 33.425% | 94.149% |
| 379 K | 2 ns | 47 | 33.626% | 92.766% |
| 379 K | 4 ns | 47 | 31.087% | 90.319% |
| 379 K | 8 ns | 47 | 23.498% | 90.213% |
| 379 K | 16 ns | 47 | 14.342% | 90.000% |
| 413 K | 1 ns | 54 | 34.165% | 94.722% |
| 413 K | 2 ns | 54 | 33.313% | 92.407% |
| 413 K | 4 ns | 54 | 30.756% | 90.741% |
| 413 K | 8 ns | 54 | 18.027% | 88.796% |
| 413 K | 16 ns | 54 | 20.643% | 88.889% |

All temperature/horizon cells have positive skill. The coverage gate still fails because at least one cell is slightly outside the strict target interval, most visibly 320 K / 1 ns at `95.96%`.

## Temperature-Resolved Closure

| Temperature | Cov RMSE | Min eig | Max eig | Path NLL / dim | Lag-1 corr RMS |
|---:|---:|---:|---:|---:|---:|
| 320 K | 0.148630 | 0.008812 | 1.304005 | -0.962439 | 0.058715 |
| 348 K | 0.147154 | 0.010955 | 1.629201 | -0.823034 | 0.066120 |
| 379 K | 0.141045 | 0.015643 | 1.395866 | -0.712487 | 0.064851 |
| 413 K | 0.135079 | 0.013480 | 1.643399 | -0.525178 | 0.068878 |

The global covariance RMSE is strong, but the per-temperature covariance/whiteness gates remain strict enough that S250 does not pass full promotion.

## Physical Diagnostics

| Metric | S250 |
|---|---:|
| Minimum diffusion eigenvalue | 7.05e-08 |
| Minimum diffusion increment std | 0.011872 |
| Mean diffusion condition number | 244.36 |
| Maximum diffusion condition number | 313.75 |
| Maximum allowed condition number | 100.00 |
| Mean observation/diffusion increment variance ratio | 1.4329 |
| Mean activation energy | 2.9448 kcal/mol |
| Mean memory activation energy | 3.0000 kcal/mol |
| Mean absolute Ito divergence | 1.44e-06 per ps |
| Mean integrated memory/diffusion ratio | 0.0000 |
| Mean memory drift RMS | 0.0000 per ps |
| Mean bath information fraction | -2.06e-07 |
| Mean memory filter advantage | -0.001908 |
| Maximum FDT generator residual | 0.0000 |
| Mean observation noise variance | 0.009292 |
| Integrator | `local_exact_ou_fdt` |

The main physical limitation of S250 is the diffusion condition number. It remains PSD and nondegenerate, but the maximum condition number is above the strict gate.

## Promotion Gates

| Gate | Status |
|---|---:|
| Complete temperature x horizon grid | PASS |
| Complete protein coverage | PASS |
| Positive skill every temperature x horizon | PASS |
| Coverage calibrated every temperature | FAIL |
| Covariance closed every temperature | FAIL |
| Multi-lag white every temperature | FAIL |
| Mori filter predictive | PASS |
| Diffusion PSD | PASS |
| Diffusion nondegenerate | PASS |
| Diffusion well-conditioned | FAIL |
| Full promotion pass | FAIL |

## Legacy MSM/VAMP Metrics

The Phase 8S branch freezes the representation pathway used by the legacy MSM/VAMP probes, so these metrics should be reported as inherited from the G/N line rather than newly improved by S.

| Legacy MSM readout | Accuracy | Balanced accuracy | Macro-F1 | Majority baseline |
|---|---:|---:|---:|---:|
| `tmjepa_cv` | 50.77% | 42.91% | 38.09% | 37.90% |
| `tmjepa_cv_condition_kinetics` | 51.30% | 43.79% | 39.43% | 37.90% |
| V5.1 old probe | 36.22% | 26.88% | 22.42% | 37.90% |

| VAMP readout / baseline | lag1 dim5 | lag1 dim20 | lag10 dim5 | lag10 dim20 |
|---|---:|---:|---:|---:|
| G/N/S `tmjepa_cv` | 0.5483 | 0.4570 | 0.6552 | 0.3744 |
| G/N/S `tmjepa_cv_condition_kinetics` | 0.5468 | 0.4551 | 0.6579 | 0.3810 |
| centered-CA baseline | 0.5731 | 0.4624 | 0.6668 | 0.3920 |

MSM decodability is strong relative to the old V5.1 probe. VAMP remains below centered-CA, so the paper should not claim Koopman slow-mode SOTA from this branch.

## Final Selection

Use Phase 8T/T250 as the manuscript model when emphasizing physical reliability. Use Phase 8S/S250 only when the table is explicitly a raw endpoint-energy leaderboard.

The attempted extra seed/CK8 evaluation on 2026-09-05 was interrupted because the environment stalled during `torch` import/I/O before producing additional JSON. The completed and archived S250 CK0 seed-6303 evaluation is the final available result for this round.
