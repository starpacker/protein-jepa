# Current Code SOTA: EVA5C Thermodynamic Mori-JEPA V3

Date: 2026-09-06

## Decision

The current defensible code SOTA is:

**EVA5C quadratic latent closure**

- Checkpoint: `/liuyutian/protein_jepa_frame/artifacts/eva5c_tmjepa_v3_quadratic_latent_closure/checkpoint_step375_gate2.pt`
- Config: `/liuyutian/protein_jepa_frame/artifacts/eva5c_tmjepa_v3_quadratic_latent_closure/config.json`
- Primary evaluation: `/liuyutian/protein_jepa_frame/artifacts/sota_eval_20260905/eva5c_final_checkpoint_seed6303_ck0.json`
- Parent trained checkpoint: `/liuyutian/protein_jepa_frame/artifacts/eva5b_tmjepa_v3_frozen_latent_closure/checkpoint_step375.pt`

This is an **internal validation SOTA** under the TM-JEPA V3 endpoint-energy
protocol. It is not an external literature SOTA and not a held-out test-set
claim.

## SOTA Selection Rule

The leaderboard was filtered by the following rules:

1. Use the TM-JEPA V3 physical endpoint-energy protocol, not older latent-space
   paired-prediction scores.
2. Use the aligned validation evaluation with seed `6303`, CK samples disabled,
   50 batches, batch size 4, and horizons 1/2/4/8/16 ns.
3. Prefer physically admissible checkpoints with PSD, nondegenerate, and
   well-conditioned diffusion.
4. Do not promote a checkpoint purely because one random evaluation seed gives
   a higher raw score.

Important exclusions:

- Phase2 paired-window scores above 90% are excluded from this SOTA definition
  because they measure latent endpoint prediction under a different objective
  and protocol, not the TM-JEPA stochastic endpoint-energy model.
- S250 has a single seed-6304 raw endpoint score of 29.639%, but the same
  checkpoint is 28.150% under seed 6303 and fails the diffusion
  well-conditioned gate with max condition number 313.75. It remains a raw
  endpoint upper-bound ablation, not the main SOTA.
- EVA-LS1 is the current learned-state milestone, but its endpoint macro skill
  is only 0.38% in the quick paired evaluation. It is not the endpoint SOTA.

## Framework Summary

EVA5C is a thermodynamically structured stochastic JEPA endpoint predictor with
a bounded history-dependent projected-force closure.

High-level pipeline:

```text
Protein coordinate clip
        |
        v
Context encoder, frozen in EVA5C
        |
        v
Causal temporal encoder, frozen in EVA5C
        |
        v
Fixed physical CVs + learned 4D latent CV projection
        |
        v
ThermodynamicMoriJEPAV3 stochastic dynamics
        |
        v
History-conditioned projected-force endpoint closure
        |
        v
Ensemble endpoint prediction at 1/2/4/8/16 ns
```

EVA5C should be described as a **hybrid learned-physics JEPA world model**:

- JEPA encoders provide coordinate-derived context and causal history.
- Fixed collective variables define the explicit thermodynamic state.
- Eyring-Einstein kinetics provides the temperature-dependent clock.
- A state-dependent diffusion tensor models stochastic conformational motion.
- A heat-capacity free-energy module supplies the thermodynamic potential.
- A Bayesian observation/noise model handles noisy endpoint observations.
- A projected-force closure uses causal JEPA history to correct unresolved
  long-horizon drift.

## Model Components

### 1. Coordinate JEPA Encoder

The context encoder maps protein coordinate clips to latent frame tokens.
EVA5C uses:

| Item | Value |
|---|---:|
| Latent dimension | 64 |
| Context encoder depth | 2 |
| Temporal encoder depth | 2 |
| Attention heads | 4 |
| Context encoder | Frozen |
| Temporal encoder | Frozen |

The frozen encoder is inherited from the U250 line. This is useful for endpoint
stability, but it also defines an important limitation: EVA5C is not the final
fully online V-JEPA-style learned-state model.

### 2. Thermodynamic State

The TM-JEPA V3 state uses a 20D coordinate:

| State block | Dimension | Role |
|---|---:|---|
| Fixed physical CVs | 16 | Explicit backbone collective variables |
| Learned CV projection | 4 | Latent projection from causal JEPA state |
| Total | 20 | Thermodynamic dynamics coordinate |

The fixed CV statistics are serialized in the checkpoint config through
`tmjepa_cv_mean` and `tmjepa_cv_scale`. The learned CV projection is frozen in
EVA5C.

### 3. Free Energy

The free-energy model is a heat-capacity-aware potential:

```text
F(q, T, context) -> beta F, entropy/kB, heat capacity/kB
```

This lets the same state coordinate respond to temperature rather than learning
four independent temperature-specific predictors.

### 4. Eyring-Einstein Kinetics

The dynamics uses an Eyring-style temperature clock and state-dependent
diffusion. Key physical settings:

| Item | Value |
|---|---:|
| Reference temperature | 348 K |
| Mean activation energy | 2.944 kcal/mol |
| Diffusion rank | 4 |
| Diffusion floor | 5e-7 |
| Base step | 1000 ps |
| Integrator | `local_exact_ou_fdt` |
| State-dependent mobility | Enabled |
| Exact Ito divergence | Enabled |

### 5. Mori Memory Status

EVA5C does not use an active auxiliary Mori bath:

| Item | Value |
|---|---:|
| `tmjepa_memory_enabled` | False |
| `tmjepa_coupled_bath_enabled` | False |
| Mean integrated memory/diffusion ratio | 0.0 |
| Mean memory drift RMS | 0.0 |
| Mean memory filter advantage | 0.0 |

This is deliberate. Earlier EVA4/EVA5 experiments showed that forcing explicit
Mori memory produced negligible or negative causal intervention benefit. The
current best model therefore keeps a history-conditioned closure, but it does
not claim that the auxiliary Mori bath has been solved.

### 6. Projected-Force Closure

The endpoint correction is a bounded, history-conditioned shift:

```text
delta q = max_shift * tanh(MLP(q, auxiliary mean, JEPA history, condition, tau, T))
```

EVA5C uses:

| Item | Value |
|---|---:|
| Projected-force closure | Enabled |
| Hidden dimension | 256 |
| Max shift | 0.55 |
| Latent-history features | Enabled |
| Explicit CV-history features | Disabled |
| Horizon gate | 2 base steps |

The final correction is multiplied by:

```text
(1 - exp(-tau / tau_c))^2, tau_c = 2 base steps
```

This enforces the first-principles short-time constraint that coordinate
response to an unresolved force should scale as `O(tau^2)` near zero horizon.
This is the key EVA5C modification over EVA5B.

## Training Recipe

EVA5C itself is a promoted checkpoint with an analytic gate embedded in the
config. The trainable parent is EVA5B.

Parent training setup:

| Item | Value |
|---|---:|
| Parent | EVA5B frozen latent closure |
| Training steps | 375 |
| Batch size | 16 |
| Context frames | 24 |
| Frame stride | 1 |
| Window stride | 8 |
| Horizons | 1, 2, 4, 8, 16 frames |
| Learning rate | 6e-5 |
| Projected-force LR scale | 5.0 |
| Energy score weight | 1.2 |
| Long-horizon energy score weight | 1.8 |
| Transition NLL weight | 0.08 |
| Long-horizon transition NLL weight | 0.03 |
| Projected-force regularization | 0.01 |

Frozen modules in EVA5C:

- context encoder
- temporal encoder
- learned CV projection
- free-energy potential
- Eyring-Einstein kinetics
- diffusion
- observation noise
- endpoint calibration
- memory path

Trainable component in the EVA5B parent:

- bounded projected-force closure over JEPA latent history

## Evaluation Protocol

Primary endpoint evaluation:

| Item | Value |
|---|---:|
| Manifest | `/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl` |
| Seed | 6303 |
| Batch size | 4 |
| Max batches | 50 |
| Sampled contexts | 200 |
| Temperatures | 320, 348, 379, 413 K |
| Proteins covered | 20 / 20 |
| Horizons | 1, 2, 4, 8, 16 ns |
| Ensemble size | 8 |
| CK samples | 0 |
| Innovation transitions | 3400 |

Endpoint skill is:

```text
energy_skill = 1 - model_energy_score / persistence_energy_score
```

Higher is better. Persistence is the baseline that keeps the current state as
the future endpoint.

## Endpoint SOTA Results

| Model | Macro skill | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns |
|---|---:|---:|---:|---:|---:|---:|
| U250 encoder dynamics | 28.123% | 34.900% | 34.002% | 30.814% | 23.578% | 17.321% |
| EVA3-250 projected force | 28.572% | 34.872% | 34.314% | 31.020% | 24.269% | 18.383% |
| EVA4-375 CV-history Mori | 28.616% | 34.705% | 34.224% | 31.247% | 24.189% | 18.717% |
| EVA5-500 joint latent-history Mori | 28.491% | 34.862% | 34.188% | 31.068% | 24.028% | 18.307% |
| EVA5B-375 frozen latent closure | 28.480% | 33.905% | 33.920% | 31.153% | 24.403% | 19.019% |
| **EVA5C quadratic latent closure** | **28.687%** | **34.773%** | **34.084%** | **31.175%** | **24.385%** | **19.018%** |

EVA5C gains over U250:

| Horizon | Gain |
|---|---:|
| Macro | +0.564 percentage point |
| 8 ns | +0.807 percentage point |
| 16 ns | +1.697 percentage points |

The main gain is long-horizon accuracy, especially 16 ns.

## Endpoint Energy Scores

| Horizon | Rows | Model energy score | Persistence score | Energy skill | 90% coverage |
|---:|---:|---:|---:|---:|---:|
| 1 ns | 200 | 0.476825 | 0.731021 | 34.773% | 94.62% |
| 2 ns | 200 | 0.532826 | 0.808336 | 34.084% | 93.05% |
| 4 ns | 200 | 0.614070 | 0.892224 | 31.175% | 90.77% |
| 8 ns | 200 | 0.697910 | 0.922984 | 24.385% | 90.97% |
| 16 ns | 200 | 0.821555 | 1.014494 | 19.018% | 91.17% |

The marginal 90% coverage is close to the target range at all horizons.

## Temperature-Resolved Results

| Temperature | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns |
|---:|---:|---:|---:|---:|---:|
| 320 K | 34.971% | 34.949% | 30.969% | 28.171% | 14.308% |
| 348 K | 37.636% | 35.314% | 31.234% | 29.005% | 21.713% |
| 379 K | 33.263% | 33.708% | 31.967% | 24.442% | 15.863% |
| 413 K | 33.868% | 33.041% | 30.636% | 18.850% | 22.107% |

All temperature/horizon cells have positive endpoint skill.

Temperature-resolved 90% coverage:

| Temperature | 1 ns | 2 ns | 4 ns | 8 ns | 16 ns |
|---:|---:|---:|---:|---:|---:|
| 320 K | 95.96% | 94.33% | 89.90% | 90.77% | 90.48% |
| 348 K | 94.36% | 92.98% | 92.02% | 92.55% | 93.40% |
| 379 K | 93.72% | 92.23% | 90.96% | 91.38% | 91.06% |
| 413 K | 94.35% | 92.59% | 90.37% | 89.44% | 90.00% |

The strict per-temperature coverage gate fails because at least one cell is
outside the allowed interval, most visibly 320 K / 1 ns at 95.96%.

## Closure And Physical Diagnostics

Global closure:

| Metric | EVA5C |
|---|---:|
| Innovation transitions | 3400 |
| Innovation mean RMS | 0.012623 |
| Innovation covariance RMSE | 0.133973 |
| Innovation covariance min eigenvalue | 0.043643 |
| Innovation covariance max eigenvalue | 1.426771 |
| Path NLL / dimension | -0.713266 |

Lagged innovation diagnostics:

| Lag | Cov RMS | Corr RMS | Diagonal corr RMS | Off-diagonal corr RMS |
|---:|---:|---:|---:|---:|
| 1 | 0.030225 | 0.048737 | 0.148873 | 0.036522 |
| 2 | 0.019018 | 0.040011 | 0.084448 | 0.036191 |
| 4 | 0.014154 | 0.026479 | 0.053127 | 0.024279 |
| 8 | 0.016130 | 0.029961 | 0.041813 | 0.029205 |

Diffusion and physical checks:

| Metric | EVA5C |
|---|---:|
| Minimum diffusion eigenvalue | 3.248e-7 |
| Minimum diffusion increment std | 0.025488 |
| Minimum allowed increment std | 0.010000 |
| Mean diffusion condition number | 54.607 |
| Maximum diffusion condition number | 70.728 |
| Maximum allowed condition number | 100.000 |
| Mean observation/diffusion variance ratio | 1.387 |
| Mean observation noise variance | 0.009337 |
| Maximum FDT generator residual | 0.0 |

Promotion gates:

| Gate | Status |
|---|---:|
| Complete temperature x horizon grid | PASS |
| Complete protein coverage | PASS |
| Positive skill every temperature x horizon | PASS |
| Diffusion PSD | PASS |
| Diffusion nondegenerate | PASS |
| Diffusion well-conditioned | PASS |
| Coverage calibrated every temperature | FAIL |
| Covariance closed every temperature | FAIL |
| Multi-lag white every temperature | FAIL |
| Mori filter predictive | PASS, vacuous because memory is disabled |
| Full promotion pass | FAIL |

The correct physical claim is therefore **stable thermodynamic endpoint
prediction with well-conditioned diffusion**, not full innovation closure.

## Protein-Level Robustness

All 20 validation proteins are covered. Protein-level endpoint skill, computed
from energy score against persistence, is positive for every protein.

| Statistic | Value |
|---|---:|
| Proteins | 20 |
| Mean protein skill | 27.322% |
| Minimum protein skill | 7.815% |
| Maximum protein skill | 34.975% |

The weakest protein is `1g61A00` at 7.815%, so the model is not uniformly strong
across proteins. The endpoint gain is broad but heterogeneous.

## Representation Metrics

The EVA5C encoder audit shows that the endpoint SOTA still relies heavily on
fixed physical CVs.

Quick representation audit:

| Readout | VAMP lag6 dim5 | VAMP lag6 dim10 | MSM accuracy | Balanced accuracy | Macro-F1 | Linear R2 lag6 |
|---|---:|---:|---:|---:|---:|---:|
| Causal endpoint 64D | 0.3805 | 0.6458 | 40.82% | 35.00% | 30.64% | -0.464 |
| Fixed CV 16D | **0.3834** | **0.6531** | **53.10%** | 43.68% | **38.17%** | not run |
| Learned CV 4D | 0.2902 | 0.2902 | 45.00% | 35.44% | 29.10% | not run |
| Full CV 20D | 0.3815 | 0.6519 | 51.96% | **43.80%** | 37.36% | -0.239 |

EVA-LS1 later improved learned-state VAMP quality:

| Readout | VAMP lag6 dim5 | VAMP lag6 dim10 | MSM accuracy | Balanced accuracy | Macro-F1 | Endpoint macro skill |
|---|---:|---:|---:|---:|---:|---:|
| EVA-LS1 slow endpoint | 0.6238 | 0.8536 | 50.10% | 41.15% | 35.15% | 0.38% quick |

This means the learned-state direction is promising, but EVA-LS1 does not yet
replace EVA5C as the endpoint predictor.

## Final Claim Boundary

Supported paper-facing claim:

> A thermodynamically structured stochastic JEPA with a bounded,
> history-conditioned projected-force closure and a quadratic short-time
> consistency constraint improves internal multi-horizon endpoint energy skill,
> especially at long horizons, while preserving PSD, nondegenerate, and
> well-conditioned diffusion.

Also supported:

- Positive endpoint skill at every tested temperature and horizon.
- Complete validation protein coverage.
- Improved macro endpoint skill over U250, EVA3, EVA4, EVA5, and EVA5B under
  the aligned seed-6303 protocol.
- 90% marginal coverage close to target.
- Stronger learned-state representation is possible in EVA-LS1, although its
  transition predictor is not yet competitive.

Not supported:

- External literature SOTA.
- Held-out test-set SOTA.
- 60% endpoint skill under the TM-JEPA V3 endpoint-energy protocol.
- Full physical closure promotion.
- Fully whitened innovations.
- Universal predictive auxiliary Mori memory.
- Learned encoder superiority for EVA5C; EVA5C still mainly depends on fixed
  physical CVs.
- Autonomous long-rollout equilibrium correctness.

## Reproducibility

Primary checkpoint:

```bash
/liuyutian/protein_jepa_frame/artifacts/eva5c_tmjepa_v3_quadratic_latent_closure/checkpoint_step375_gate2.pt
```

Primary evaluation JSON:

```bash
/liuyutian/protein_jepa_frame/artifacts/sota_eval_20260905/eva5c_final_checkpoint_seed6303_ck0.json
```

EVA5C promotion script:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python \
/liuyutian/protein_jepa_frame/scripts/promote_eva5c_quadratic_closure.py \
  --input /liuyutian/protein_jepa_frame/artifacts/eva5b_tmjepa_v3_frozen_latent_closure/checkpoint_step375.pt \
  --output /liuyutian/protein_jepa_frame/artifacts/eva5c_tmjepa_v3_quadratic_latent_closure/checkpoint_step375_gate2.pt \
  --gate-steps 2
```

Related reports:

- `/liuyutian/protein_jepa_frame/docs/eva5c_final_sota_2026-09-06.md`
- `/liuyutian/protein_jepa_frame/docs/eva5c_encoder_information_audit_2026-09-06.md`
- `/liuyutian/protein_jepa_frame/docs/eva_ls1_learned_state_evaluation_2026-09-06.md`

