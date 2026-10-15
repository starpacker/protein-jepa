# Phase 8W Memory Residual Ablation

Date: 2026-09-05

## Purpose

Phase 8W tests whether Mori memory can be reintroduced as a small residual
closure term on top of the Phase 8U encoder-dynamics SOTA without degrading
endpoint forecasting.

The main design change is conservative:

- initialize from `phase8u_tmjepa_v3_encoder_dynamics/checkpoint_step250.pt`
- keep context and temporal encoders trainable, but reduce encoder LR scale to `0.01`
- enable Mori memory and memory intervention
- disable explicit coupled bath transition
- reduce memory coupling scale to `2.0e-4`
- add a soft integrated-memory/diffusion ratio band `[0.05, 0.20]`

## Files

- Config: `configs/phase8w_tmjepa_v3_memory_residual.yaml`
- Script: `scripts/run_phase8w_tmjepa_v3_memory_residual.sh`
- Checkpoints:
  - `artifacts/phase8w_tmjepa_v3_memory_residual/checkpoint_step125.pt`
  - `artifacts/phase8w_tmjepa_v3_memory_residual/checkpoint_step250.pt`
- Evaluations:
  - `artifacts/sota_eval_20260905/w125_memory_residual_seed6303_ck0.json`
  - `artifacts/sota_eval_20260905/w250_memory_residual_seed6303_ck0.json`

## Validation Metrics

All rows use the same validation protocol: seed `6303`, `max_batches=50`,
`ck_samples=0`, horizons `1/2/4/8/16`.

| Model | Macro skill | Skills 1/2/4/8/16 ns | Coverage 1/2/4/8/16 ns | Cov RMSE | Path NLL | Lag corr RMS 1/2/4/8 | Diag corr RMS 1/2/4/8 | Max cond | Memory ratio | Memory advantage |
|---|---:|---|---|---:|---:|---|---|---:|---:|---:|
| U250 encoder dynamics | 28.123% | 34.90 / 34.00 / 30.81 / 23.58 / 17.32 | 93.77 / 91.92 / 89.03 / 88.92 / 89.80 | 0.133973 | -0.713266 | 0.0487 / 0.0400 / 0.0265 / 0.0300 | 0.1489 / 0.0844 / 0.0531 / 0.0418 | 70.73 | 0.0000 | -0.000796 |
| V250 coupled memory encoder | 27.026% | 34.66 / 33.64 / 30.38 / 21.92 / 14.52 | 94.27 / 93.23 / 90.90 / 90.77 / 91.07 | 0.131919 | -0.686817 | 0.0519 / 0.0399 / 0.0260 / 0.0292 | 0.1585 / 0.0933 / 0.0512 / 0.0402 | 70.87 | 0.1949 | 0.000993 |
| W125 memory residual | 27.929% | 34.90 / 33.94 / 30.79 / 23.23 / 16.79 | 93.90 / 92.47 / 89.78 / 89.53 / 90.38 | 0.131749 | -0.702058 | 0.0493 / 0.0394 / 0.0263 / 0.0293 | 0.1504 / 0.0879 / 0.0543 / 0.0405 | 70.79 | 0.0370 | -0.000239 |
| W250 memory residual | 27.925% | 34.84 / 33.93 / 30.82 / 23.20 / 16.84 | 94.10 / 92.58 / 89.92 / 89.78 / 90.77 | 0.131337 | -0.701766 | 0.0494 / 0.0395 / 0.0262 / 0.0294 | 0.1494 / 0.0879 / 0.0540 / 0.0405 | 70.85 | 0.0330 | -0.000477 |

## Conclusion

Phase 8W is not the manuscript SOTA. Phase 8U remains the main model because it
has the best macro endpoint skill and better path NLL.

Phase 8W is still a useful ablation: it achieves the best covariance closure
seen in this series (`0.131337` vs U250 `0.133973`) while avoiding the large
long-horizon endpoint degradation of Phase 8V. The remaining failure is not
memory magnitude explosion; the new ratio band prevents that. The failure is
that the inferred bath state is not consistently predictive, especially around
the 4/8 ns horizons, so the Mori filter gate remains false.

For the paper, use U250 as the main SOTA and describe W250 as evidence that
Mori residuals improve innovation covariance closure but still require a better
history-to-bath inference objective before replacing the endpoint forecaster.
