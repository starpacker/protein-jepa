# Phase 8T Guarded TM-JEPA V3 Manuscript SOTA

Date: 2026-09-05

Final manuscript checkpoint: `artifacts/phase8t_tmjepa_v3_guarded_closure/checkpoint_step250.pt`

Evaluation JSON: `artifacts/sota_eval_20260905/t250_guarded_closure_seed6303_ck0.json`

Training config: `configs/phase8t_tmjepa_v3_guarded_closure.yaml`

Run script: `scripts/run_phase8t_tmjepa_v3_guarded_closure.sh`

## Decision

Use Phase 8T/T250 as the manuscript model. It is slightly below Phase 8S/S250 on raw endpoint energy skill, but it fixes the diffusion-conditioning pathology that made S250 physically weaker. The tradeoff is small on prediction quality and large on physical defensibility.

Phase 8S/S250 remains useful as a raw endpoint upper-bound ablation, not as the main paper checkpoint.

## Main Comparison

| Checkpoint | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns | Macro | Max diffusion condition |
|---|---:|---:|---:|---:|---:|---:|---:|
| N14 previous endpoint SOTA | 30.05% | 27.74% | 23.51% | 19.08% | 6.81% | 21.438% | not archived |
| S250 raw endpoint upper bound | 34.92% | 33.98% | 30.87% | 23.55% | 17.44% | 28.150% | 313.75 |
| T250 guarded manuscript model | 34.72% | 33.78% | 30.66% | 23.27% | 17.00% | 27.888% | 70.34 |

T250 improves over N14 by `+6.450 percentage points` macro endpoint skill. Relative to S250, it gives up only `0.262 percentage point` macro skill while passing the diffusion well-conditioned gate.

## T250 Endpoint Metrics

| Horizon | Energy skill | 90% coverage |
|---|---:|---:|
| 1 ns | 34.72% | 95.10% |
| 2 ns | 33.78% | 93.40% |
| 4 ns | 30.66% | 91.25% |
| 8 ns | 23.27% | 90.70% |
| 16 ns | 17.00% | 91.22% |

## T250 Closure And Physics

| Metric | Value |
|---|---:|
| Innovation covariance RMSE | 0.1383 |
| Path NLL per dimension | -0.7269 |
| Lag-1/2/4/8 corr RMS | 0.0493 / 0.0374 / 0.0266 / 0.0297 |
| Lag-1/2/4/8 diagonal corr RMS | 0.1557 / 0.0764 / 0.0522 / 0.0407 |
| Minimum diffusion increment std | 0.0255 |
| Mean diffusion condition number | 54.25 |
| Maximum diffusion condition number | 70.34 |

T250 passes diffusion PSD, nondegenerate diffusion, well-conditioned diffusion, full temperature-horizon coverage, full protein coverage, and positive skill in every temperature-horizon cell.

It still fails strict per-temperature coverage calibration, per-temperature covariance closure, and per-temperature multi-lag whiteness. These should be described as residual calibration/closure limitations, not as solved properties.

## Temperature-Resolved T250 Endpoint Metrics

| Temperature | Horizon | Energy skill | 90% coverage |
|---:|---:|---:|---:|
| 320 K | 1 ns | 34.88% | 96.15% |
| 320 K | 2 ns | 34.28% | 94.33% |
| 320 K | 4 ns | 31.02% | 89.90% |
| 320 K | 8 ns | 27.29% | 90.29% |
| 320 K | 16 ns | 12.47% | 90.58% |
| 348 K | 1 ns | 37.41% | 95.00% |
| 348 K | 2 ns | 34.78% | 93.40% |
| 348 K | 4 ns | 30.29% | 92.55% |
| 348 K | 8 ns | 27.89% | 92.66% |
| 348 K | 16 ns | 19.16% | 93.40% |
| 379 K | 1 ns | 33.24% | 94.36% |
| 379 K | 2 ns | 33.48% | 93.09% |
| 379 K | 4 ns | 30.86% | 90.74% |
| 379 K | 8 ns | 23.13% | 90.85% |
| 379 K | 16 ns | 13.81% | 90.64% |
| 413 K | 1 ns | 33.94% | 94.81% |
| 413 K | 2 ns | 33.07% | 92.78% |
| 413 K | 4 ns | 30.51% | 91.85% |
| 413 K | 8 ns | 17.74% | 89.26% |
| 413 K | 16 ns | 20.33% | 90.46% |

## Temperature-Resolved T250 Closure

| Temperature | Covariance RMSE | Path NLL / dim | Lag-1 corr RMS |
|---:|---:|---:|---:|
| 320 K | 0.1495 | -0.9423 | 0.0549 |
| 348 K | 0.1483 | -0.7993 | 0.0619 |
| 379 K | 0.1427 | -0.6844 | 0.0604 |
| 413 K | 0.1363 | -0.4934 | 0.0643 |

## Manuscript Framing

Recommended claim:

> A thermodynamically guarded JEPA rollout model improves multi-horizon endpoint prediction over persistence and the previous endpoint-calibrated checkpoint while maintaining PSD, nondegenerate, and well-conditioned stochastic diffusion.

Do not claim:

- Full physical closure promotion pass.
- Koopman/VAMP slow-mode SOTA.
- Universal Mori-filter advantage at every horizon.

Legacy representation probes are inherited from the frozen-representation branch: best MSM readout is `51.30%` accuracy, `43.79%` balanced accuracy, and `39.43%` macro-F1. VAMP remains below the centered-CA baseline.
