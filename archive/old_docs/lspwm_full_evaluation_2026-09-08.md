# LSP-WM v4.1 Full Evaluation Diagnosis

Date: 2026-09-08

Checkpoint:

```text
/liuyutian/protein_jepa_frame/artifacts/lspwm_v41_mdCATH_20260908_bs16/checkpoint_step4000.pt
```

The checkpoint was evaluated on the fixed `tmjepa_v3` validation and test
manifests. Validation and test each contain 80 held-out trajectories and 3200
paired rows under this evaluator. FastFolding uses a scaffold-disjoint split
over 26 proteins and 100 sampled frames per protein for this diagnostic pass.
The machine-readable outputs are in:

```text
artifacts/lspwm_eval_20260908_step4000/
```

## Executive Conclusion

The current LSP-WM is **numerically stable and representation-informative,
but not yet a successful dynamics world model**.

The main failure is not NaN or diffusion indefiniteness. It is transition
closure and calibration:

```text
learned representation: useful
equilibrium/role readout: partially useful
one-step transition law: poorly calibrated
multi-step rollout: worse than persistence
FastFolding event/operator prediction: catastrophically miscalibrated
```

The current result should not be described as an LSP-WM dynamics SOTA result.
It is a useful negative diagnostic that identifies the transition law as the
limiting component.

## 1. Primary Results

### 1.1 Paired endpoint forecast

The paired evaluator gives exactly zero persistence-relative skill at every
horizon on both validation and test:

| Horizon | Validation skill | Test skill |
| ---: | ---: | ---: |
| 1 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 |
| 4 | 0.000 | 0.000 |
| 8 | 0.000 | 0.000 |
| 16 | 0.000 | 0.000 |

The shuffled-history and shuffled-horizon controls are also identical to the
model prediction. This means the paired transition head is effectively
predicting zero displacement/persistence.

### 1.2 Physical evaluator

| Metric | Validation | Test | Interpretation |
| --- | ---: | ---: | --- |
| MSM hard assignment accuracy | 0.871 | 0.890 | not a held-out transition metric; same readout is compared with itself |
| MSM soft overlap | 0.487 | 0.575 | partial state agreement |
| rollout state retention | 0.322 | 0.362 | weak closure |
| rollout target coverage | 0.311 | 0.356 | ensemble rarely reaches target state |
| mean transition NLL | 489.08 | 488.89 | poor transition likelihood scale |
| raw empirical FDT residual | 2.621 | 2.621 | consistent with finite-sample covariance noise at `d=256`, `N=32`; not a calibration gate |
| detailed-balance readout residual | 0.00144 | 0.00172 | only a soft-MSM diagnostic |
| diffusion PSD | pass | pass | parameterization contract only |
| diffusion condition number | 6.07 | 6.07 | not the main failure |

The high hard MSM accuracy is not evidence of predictive quality. It compares
current and target assignments produced by the same learned role readout. The
rollout retention and target coverage are the meaningful closure signals here,
and both are low.

### 1.3 Multi-step rollout

The rollout evaluator reports validation macro energy skill of `-782.6%` and
trajectory energy skill of `-430.7%`. At the first rollout step:

```text
energy score = 6.6734
persistence   = 0.3818
relative skill = -1647.8%
```

There were no NaN, Inf, or norm-divergence failures. The state norm reached
about `88.98`, below the diagnostic divergence threshold of `838.10`.
Therefore the model is stable in a numerical sense while being dynamically
wrong. The intermediate targets in this evaluator are linear endpoint bridges,
not observed transition paths; this result is a rollout diagnostic, not a path
ensemble claim.

### 1.4 FastFolding event and operator evaluation

The test split contains three scaffold groups and 1372 context/horizon samples.
The transition-head adapter uses the predicted future latent state and a
train-scaffold-only state readout.

| Horizon | State NLL | Accuracy | ECE | Macro-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.196 | 0.242 | 0.720 | 0.130 |
| 4 | 2.517 | 0.242 | 0.575 | 0.130 |
| 8 | 2.971 | 0.241 | 0.592 | 0.130 |
| 16 | 5.196 | 0.244 | 0.724 | 0.131 |

The predicted state probabilities are highly overconfident and mostly wrong.
The per-horizon transition operator has:

```text
row TV:                 0.311 - 0.329
stationary L1 distance: 1.075 - 1.141
MFPT relative error:    0.698 - 0.725
```

The observed test operator places roughly 97% of its mass in the transition
state, while the predicted operator places approximately 40% to 45% in the
folded state. This is a qualitative operator failure, not a small calibration
defect.

The committor result is only a low-sample future-basin proxy and must not be
called a true committor. Its test calibration is also poor, with Brier scores
from approximately `0.27` to `0.99` across the tested horizons.

## 2. Root-Cause Diagnosis

### P0: The paired transition head was disabled during training

The active training configuration contains:

```yaml
paired_delta_weight: 0.0
paired_state_weight: 0.0
```

The training log also reports `grad_norm_paired_predictor: 0.0` and
`paired_skill: 0.0`. Consequently, the zero endpoint skill is expected from
the training contract. This is a configuration failure, not evidence that the
LSP-WM potential itself has solved or failed endpoint prediction.

The paired head must either be trained explicitly or removed from the claimed
world-model evaluation. It cannot remain an untrained output in the primary
table.

### P0: The learned transition law is not calibrated to the learned state

Independent of the paired head, the LSP-WM rollout has poor endpoint-relative
energy skill, low target coverage, and transition NLL near 489. This indicates
a mismatch among:

```text
latent state scale
PMF/mean force
diffusion scale
time-step conversion
memory coupling
```

The PSD diffusion gate passes because it is enforced by parameterization. PSD
does not imply that the diffusion magnitude or drift produces the observed
increment distribution.

The previous interpretation of the raw empirical FDT residual as a failure was
incorrect. Estimating a 256-dimensional covariance from 32 noise draws has an
expected relative Frobenius error of `sqrt((d + 1) / N) = 2.83` even when the
integrator satisfies FDT exactly. The observed value `2.62` is therefore within
ordinary finite-sample noise. The evaluator now reports the analytic FDT
identity and a residual normalized by this Wishart baseline. This correction
does not change the endpoint, rollout, or operator failures above.

### P1: Rollout curriculum does not optimize a reliable observed-path target

At training step 3975, the logged active phase is `observation`; rollout losses
are zero in that phase while the endpoint energy skill remains negative. At
step 2000, the rollout trajectory NLL is already about `827` and endpoint
energy skill is about `-37.4`.

The model therefore reaches the rollout phase without a successful transition
closure, and later curriculum stages do not repair it. The rollout objective
also uses self-consistency/bridge-style targets rather than an observed
multi-step path likelihood at matched physical times.

### P1: MSM role readout is not a held-out kinetic operator

The 32-state readout is useful as a diagnostic but is not itself a learned
held-out transition operator. Its hard assignment accuracy is inflated by the
same-space comparison, and its occupancy is concentrated in a small number of
states. A true operator evaluation must estimate `T_tau` from predicted state
probabilities and compare it with a test-trajectory operator, as done in the
FastFolding benchmark.

### P1: FastFolding exposes a state-space/domain mismatch

The transition head's train-only readout produces very confident but wrong
FastFolding state probabilities. This suggests that the learned 256D endpoint
state is not aligned with folding kinetics across protein scaffolds, or that
the train-only linear observation map is too weak for the transfer task. The
result should be treated as a real transfer failure, not fixed by reporting
accuracy alone.

### P2: Thermodynamic claims remain conditional

This evaluation uses fixed-temperature passive data. FastFolding labels do not
contain temperature, so its adapter explicitly used the checkpoint reference
temperature `348 K`. No environment state, ligand action, mutation action, or
temperature transfer was tested. The current checkpoint therefore cannot
support a complete environment/action-conditioned world-model claim.

## 3. What Passed And What Did Not

### Passed contracts

- checkpoint loads strictly;
- finite tensors and no rollout NaN/Inf;
- diffusion matrix is positive definite;
- conservative force is integrable by construction;
- software and data contracts execute on held-out validation/test data;
- train/validation/test FastFolding scaffold separation is enforced.

### Failed scientific gates

- paired endpoint forecast skill;
- multi-step rollout skill;
- target-state ensemble coverage;
- FastFolding state forecast calibration;
- FastFolding transition operator stationary distribution;
- FastFolding MFPT accuracy.

### Not yet measured

- true multi-shot committor;
- observed transition-path ensemble likelihood/coverage from LSP-WM samples;
- survival calibration from sampled first-passage times;
- temperature transfer;
- environment-state and action-conditioned counterfactual response.

## 4. Recommended Repair Order

1. Turn on and verify a directly supervised/likelihood-trained transition
   objective. Require nonzero paired predictor gradients and positive skill on
   a held-out validation set before enabling the thermodynamic rollout claims.
2. Calibrate the one-step law in physical units: check `dt_ps`, latent scale,
   diffusion magnitude, drift magnitude, and empirical increment covariance.
3. Replace bridge self-consistency with observed multi-step path likelihood and
   endpoint distribution scoring at the same physical horizons.
4. Add an explicit state-operator loss/evaluation using held-out transition
   matrices, stationary occupancy, MFPT, and Chapman-Kolmogorov closure.
5. Only after steps 1-4 pass, add memory-rank and FastFolding committor/path
   evaluation. The current finite bath is not yet shown to improve closure.
6. Keep action/environment claims disabled until the data contract contains
   evolving environment observations and held-out interventions.

## Claim Boundary

The defensible paper statement for this checkpoint is:

> LSP-WM v4.1 learns a structured latent representation with stable PSD
> thermodynamic parameterization and partial slow-state signal, but the current
> transition closure is not predictive: endpoint skill is at persistence,
> rollout skill is negative, FDT calibration fails, and FastFolding state and
> operator transfer is poorly calibrated. The transition law, rather than the
> encoder representation, is the current bottleneck.
