# Phase 8U Encoder-Dynamics TM-JEPA V3 SOTA

Date: 2026-09-05

Final checkpoint: `artifacts/phase8u_tmjepa_v3_encoder_dynamics/checkpoint_step250.pt`

Config: `configs/phase8u_tmjepa_v3_encoder_dynamics.yaml`

Evaluation: `artifacts/sota_eval_20260905/u250_encoder_dynamics_seed6303_ck0.json`

## Summary

Phase 8U opens dynamics-driven encoder fine-tuning from the guarded T250 checkpoint. The context encoder and temporal encoder are trainable, while the learned CV projection, potential, endpoint calibration, and memory path remain fixed. This lets the representation adapt to endpoint/transition/path/innovation objectives without moving the established thermodynamic coordinate system too aggressively.

This is now the recommended manuscript model. It recovers almost all of the S250 raw endpoint score, keeps the T250 diffusion-condition fix, improves global covariance closure, improves lag-1 whiteness, and turns the strict coverage-calibration gate from fail to pass.

## Checkpoint Comparison

| Checkpoint | Macro skill | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns | Coverage gate | Max diffusion condition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S250 raw endpoint upper bound | 28.150% | 34.92% | 33.98% | 30.87% | 23.55% | 17.44% | FAIL | 313.75 |
| T250 guarded frozen encoder | 27.888% | 34.72% | 33.78% | 30.66% | 23.27% | 17.00% | FAIL | 70.34 |
| U125 encoder dynamics | 28.081% | 34.86% | 33.90% | 30.81% | 23.55% | 17.28% | PASS | 70.63 |
| **U250 encoder dynamics** | **28.123%** | **34.90%** | **34.00%** | **30.81%** | **23.58%** | **17.32%** | **PASS** | **70.73** |

U250 is only `0.027 percentage point` below the raw S250 endpoint upper bound while avoiding S250's ill-conditioned diffusion tensor. Relative to T250, U250 gains `+0.235 percentage point` macro endpoint skill and improves covariance RMSE from `0.1383` to `0.1340`.

## U250 Metrics

| Horizon | Energy skill | 90% coverage |
|---|---:|---:|
| 1 ns | 34.90% | 93.77% |
| 2 ns | 34.00% | 91.92% |
| 4 ns | 30.81% | 89.03% |
| 8 ns | 23.58% | 88.92% |
| 16 ns | 17.32% | 89.80% |

| Closure / physics metric | Value |
|---|---:|
| Innovation covariance RMSE | 0.1340 |
| Path NLL / dim | -0.7133 |
| Lag-1/2/4/8 corr RMS | 0.0487 / 0.0400 / 0.0265 / 0.0300 |
| Lag-1/2/4/8 diagonal corr RMS | 0.1489 / 0.0844 / 0.0531 / 0.0418 |
| Minimum diffusion increment std | 0.0255 |
| Mean diffusion condition number | 54.61 |
| Maximum diffusion condition number | 70.73 |

## Promotion Gates

| Gate | U250 |
|---|---:|
| Complete temperature x horizon grid | PASS |
| Complete protein coverage | PASS |
| Positive skill every temperature x horizon | PASS |
| Coverage calibrated every temperature | PASS |
| Covariance closed every temperature | FAIL |
| Multi-lag white every temperature | FAIL |
| Mori filter predictive | PASS |
| Diffusion PSD | PASS |
| Diffusion nondegenerate | PASS |
| Diffusion well-conditioned | PASS |
| Full promotion pass | FAIL |

## Interpretation

The encoder-dynamics step is a real improvement over the frozen-encoder guarded model. It does not fully solve physical closure: per-temperature covariance closure and multi-lag whiteness remain the two residual failures. The honest paper claim is therefore not full physical closure, but a thermodynamically guarded JEPA rollout with trainable dynamics-aware representation and improved multi-horizon endpoint skill under stable diffusion.

Recommended manuscript phrasing:

> Opening encoder dynamics under thermodynamic guards recovers raw endpoint performance while preserving a well-conditioned stochastic diffusion operator and improving strict coverage calibration.

Avoid claiming:

- Full closure promotion.
- Complete innovation whiteness.
- VAMP/Koopman SOTA without a fresh representation-probe evaluation.
