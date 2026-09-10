# TM-JEPA Phase 6 Pilot Results

**Date:** 2026-09-02
**Status:** implementation complete; memory and learned-CV promotion rejected

## Phase 0 Identifiability

The fixed-CV audit used 3,800 training transitions and 1,900 transitions from
held-out replica 3 across the 20-protein pilot.

| Metric | Result |
|---|---:|
| q-only transition MSE | 0.0016383 |
| q + history transition MSE | 0.0014642 |
| history skill | 10.63% |
| Gaussian conditional mutual information | 1.163 nats |
| q-only innovation lag-1 RMS | 0.2044 |
| history innovation lag-1 RMS | 0.1015 |

This supports testing a finite-memory model. It does not by itself prove that a
particular nonlinear GLE improves held-out dynamics.

Result: `artifacts/tmjepa_identifiability.json`.

## Architecture Correction During Pilot

The first stochastic implementation used a rank-4 diffusion plus a fixed
`1e-8 I` floor in 16D CV space. The remaining 12 directions therefore had
almost no thermal variance. Validation innovation covariance RMSE was about 18.

The accepted implementation uses

$$
D=\operatorname{diag}(d_1,\ldots,d_{16})+B\operatorname{diag}(\lambda)B^T,
$$

with every `d_i > 0`. The diagonal term captures ordinary thermal fluctuation
in every CV, while the rank-4 term captures correlated diffusion and shares its
basis with the Mori coupling. On the first real batch this reduced innovation
covariance loss from about 393 to 0.046 without changing an evaluation gate.

## Final 300-Step Validation

Both models were trained from scratch for 300 steps with identical data,
proper scores, and closure objectives.

| Metric | Overdamped | Rank-4 GLE |
|---|---:|---:|
| Mean energy skill | **16.46%** | 15.43% |
| Innovation mean RMS | 0.0062 | **0.0061** |
| Innovation covariance RMSE | **0.1554** | 0.1570 |
| Innovation lag-1 RMS | 0.3040 | **0.2961** |
| Mean CK energy distance | **0.1661** | 0.1716 |
| Minimum diffusion eigenvalue | 2.40e-7 | 2.35e-7 |
| Mean memory relaxation time | N/A | 1,073 ps |
| Mean absolute memory coupling | 0 | 3.00e-6 / ps |

Per-horizon GLE energy skills for `1,2,4,8,16 ns` were respectively
`23.8%, 23.9%, 18.2%, 10.1%, 1.1%`. All were positive, but long-horizon skill
is weak.

The GLE reduced lag-1 innovation RMS by only 2.58%, below the required 5%, and
made mean CK error 3.32% worse. Mean energy skill also fell by 1.04 percentage
points. `scripts/compare_tmjepa_ablation.py` therefore rejected memory
promotion.

Final artifacts:

- overdamped checkpoint:
  `artifacts/phase6_tmjepa_overdamped_closure_pilot/checkpoint_step300.pt`;
- overdamped evaluation:
  `artifacts/phase6_tmjepa_overdamped_closure_pilot/eval_validation.json`;
- GLE checkpoint:
  `artifacts/phase6_tmjepa_fixed_cv_closure_pilot/checkpoint_step300.pt`;
- GLE evaluation:
  `artifacts/phase6_tmjepa_fixed_cv_closure_pilot/eval_validation.json`;
- ablation decision: `artifacts/tmjepa_ablation_comparison.json`.

## Decision

The thermodynamic stochastic framework is retained. The current rank-4 memory
model is not promoted, and the learned-CV config must not be run as a promoted
Phase 3 experiment yet.

The scientific baseline is now the overdamped TM-JEPA model, not V5.1
persistence. The next memory experiment must improve the auxiliary filtering
or memory-kernel parameterization and pass the automated ablation comparison;
increasing history-ranking loss is not an allowed remedy.

## V2 First-Principles Correction

V2 corrected four issues found after the first pilot:

- removed future-context leakage from intermediate auxiliary states;
- separated the 1000 ps observation scale from a 250 ps Heun integration step;
- added a local path NLL and Mori transition-prior KL;
- evaluated the complete lagged innovation covariance and separated the
  diffusion and reversible-coupling bases.

The derivation and remaining assumptions are documented in
`docs/tmjepa_v2_first_principles_audit_2026-09-02.md`.

### Held-Out Replica 3

Both corrected models were retrained from scratch for 300 steps with seed
6202 and evaluated on the same 500 transitions.

| Metric | V2 overdamped | V2 rank-4 Mori |
|---|---:|---:|
| Mean endpoint energy skill | **21.163%** | 21.096% |
| Path NLL / CV / transition | **-1.80645** | -1.79286 |
| Innovation covariance RMSE | **0.27386** | 0.28663 |
| Full lagged covariance RMS | 0.16739 | **0.16637** |
| Largest absolute lagged covariance | **0.80168** | 0.92195 |
| Mean numerical CK energy distance | 0.12863 | **0.12842** |
| Mean memory relaxation time | N/A | 38.06 ns |
| Mean absolute memory coupling | 0 | 1.87e-6 / ps |

The Mori model preserved endpoint skill but improved full lagged RMS by only
0.61%, below the 5% gate, and worsened path NLL by 0.01359 nat per CV per
transition. The nearly identical numerical CK values support the diagnosis
that the earlier CK difference was largely an integration artifact. Memory
remains rejected by `artifacts/tmjepa_v2_ablation_comparison.json`.

### Frozen Overdamped Tests

After model selection, only the V2 overdamped model was evaluated on replica 4
and the held-out 450 K set.

| Split | Mean energy skill | Path NLL | Covariance RMSE | Lagged RMS |
|---|---:|---:|---:|---:|
| Replica 3 validation | 21.163% | -1.80645 | 0.27386 | 0.16739 |
| Replica 4 test | 22.265% | -1.80747 | 0.26994 | 0.15649 |
| Held-out 450 K | 23.731% | -0.36077 | 1.13698 | 0.43590 |

Replica generalization is stable, but neither in-distribution split passes the
strict innovation closure thresholds. More importantly, the positive endpoint
skill at 450 K hides a severe path-calibration failure. Explicit temperature
input and `F=U-TS` are not sufficient when transport coefficients use an
unconstrained temperature MLP.

## Updated Decision

V2 is a physically cleaner and more predictive overdamped baseline, but it is
not yet a closed coarse stochastic process. Rank-4 memory and learned CVs stay
blocked. The next preregistered change should impose a common thermodynamic
clock, beginning with `D(T)=k_B T M` and a low-parameter viscosity/activation
law for `M(T)`, then test it on a new validation partition before reusing the
450 K result. State-dependent mobility is deferred until its Ito divergence
can be implemented and identified.
