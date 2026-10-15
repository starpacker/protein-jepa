# EVA Mori Long-Horizon Ablation

Date: 2026-09-05

## Purpose

EVA tests a more aggressive reactivation of Mori memory on top of the Phase 8U
encoder-dynamics SOTA. The target is not only equilibrium free-energy fitting,
but protein dynamics forecasting under physical constraints, with special
pressure on the 8 ns and 16 ns horizons.

The main design changes are:

- initialize from `phase8u_tmjepa_v3_encoder_dynamics/checkpoint_step250.pt`
- keep both context and temporal encoders trainable with a small encoder LR
- enable Mori memory, coupled bath dynamics, and memory intervention
- add a log-space integrated-memory/diffusion ratio target near `0.10`
- keep the memory ratio inside a physically modest band `[0.06, 0.16]`
- add long-horizon weighted energy score, transition NLL, and path NLL terms
- add long-horizon memory intervention terms so memory is rewarded only when it
  improves 8 ns and 16 ns forecasts
- freeze the Phase 8U potential, learned projection, and endpoint calibration
  so this ablation focuses on kinetics, bath inference, and memory usefulness

## Files

- Config: `configs/eva_tmjepa_v3_mori_longhorizon.yaml`
- Script: `scripts/run_eva_tmjepa_v3_mori_longhorizon.sh`
- Checkpoints:
  - `artifacts/eva_tmjepa_v3_mori_longhorizon/checkpoint_step125.pt`
  - `artifacts/eva_tmjepa_v3_mori_longhorizon/checkpoint_step250.pt`
- Evaluations:
  - `artifacts/sota_eval_20260905/eva125_mori_longhorizon_seed6303_ck0.json`
  - `artifacts/sota_eval_20260905/eva250_mori_longhorizon_seed6303_ck0.json`

## Validation Metrics

All rows use the same validation protocol: seed `6303`, `max_batches=50`,
`ck_samples=0`, horizons `1/2/4/8/16`.

| Model | Macro skill | Skills 1/2/4/8/16 ns | Coverage 1/2/4/8/16 ns | Cov RMSE | Path NLL | Lag corr RMS 1/2/4/8 | Diag corr RMS 1/2/4/8 | Max cond | Memory ratio | Memory advantage |
|---|---:|---|---|---:|---:|---|---|---:|---:|---:|
| U250 encoder dynamics | 28.123% | 34.90 / 34.00 / 30.81 / 23.58 / 17.32 | 93.77 / 91.92 / 89.03 / 88.92 / 89.80 | 0.133973 | -0.713266 | 0.0487 / 0.0400 / 0.0265 / 0.0300 | 0.1489 / 0.0844 / 0.0531 / 0.0418 | 70.73 | 0.0000 | -0.000796 |
| W250 memory residual | 27.925% | 34.84 / 33.93 / 30.82 / 23.20 / 16.84 | 94.10 / 92.58 / 89.92 / 89.78 / 90.77 | 0.131337 | -0.701766 | 0.0494 / 0.0395 / 0.0262 / 0.0294 | 0.1494 / 0.0879 / 0.0540 / 0.0405 | 70.85 | 0.0330 | -0.000477 |
| EVA125 Mori long-horizon | 27.713% | 34.82 / 33.81 / 30.70 / 22.92 / 16.32 | 93.92 / 91.92 / 89.72 / 89.72 / 90.58 | 0.132919 | -0.702286 | 0.0504 / 0.0407 / 0.0261 / 0.0295 | 0.1521 / 0.0908 / 0.0529 / 0.0401 | 70.77 | 0.0792 | 0.000278 |
| EVA250 Mori long-horizon | 27.706% | 34.83 / 33.87 / 30.71 / 22.85 / 16.27 | 94.15 / 92.85 / 90.35 / 90.08 / 90.72 | 0.131486 | -0.697977 | 0.0499 / 0.0400 / 0.0262 / 0.0293 | 0.1512 / 0.0903 / 0.0538 / 0.0407 | 70.74 | 0.0756 | -0.000081 |

## Gate Status

EVA250 passes:

- complete temperature/horizon grid
- complete protein coverage
- positive skill at every temperature and horizon
- diffusion PSD, nondegenerate, and well-conditioned

EVA250 fails:

- coverage calibrated every temperature
- covariance closed every temperature
- multi-lag white every temperature
- Mori filter predictive

## Interpretation

EVA fixes the easiest Mori-memory failure: the memory channel no longer
collapses to zero. The integrated memory ratio is stable around `0.075-0.080`,
which is much closer to the intended physical scale than U250 (`0.0000`) and
W250 (`0.0330`).

The improvement does not yet translate into better long-horizon endpoint skill.
Compared with U250, EVA250 loses about `0.42` macro skill points, `0.72` points
at 8 ns, and `1.05` points at 16 ns. This means the current bath state carries
some physically plausible magnitude, but its inferred memory is not consistently
predictive under the evaluation filter.

The useful gain is closure: EVA250 improves covariance RMSE to `0.131486`
against U250 `0.133973`, with a similar maximum diffusion condition number.
However, W250 remains slightly better on covariance RMSE (`0.131337`), and U250
remains better on endpoint forecasting and path NLL.

## Conclusion

EVA is not the manuscript SOTA. Keep Phase 8U / U250 as the main result.

EVA should be reported as a targeted ablation showing that physically bounded
Mori memory can be made nonzero and numerically stable, and that this improves
innovation covariance closure. The remaining gap is history-to-bath
identifiability: the model needs the latent bath state to encode predictive
slow residuals, not merely to carry a plausible memory magnitude.
