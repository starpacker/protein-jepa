# LSP-WM v4.1 Implementation Map

This document is the compact implementation companion to `docs/new.md`. The
root `README.md` is the normative public contract; this file records exactly
which parts are implemented and which still require a protein-scale experiment.

## Design Decisions

| `new.md` proposal | LSP-WM decision | Code/status |
| --- | --- | --- |
| Coarse-grained dynamics | Predict `z`, not microscopic `x` | `LSPWMWorldModel` |
| Unified high-capacity state | `z in R^256`; role probes see all coordinates | `UnifiedLatentRoleReadout` |
| Soft scientific roles | Koopman 64, MSM 32, geometry 96, auxiliary 64 readouts | `msm_consistency_loss` and role losses |
| Overdamped limit | Momentum is integrated out | `LSPWMWorldModel.step` |
| Koopman compatibility | Empirical proposition, not theorem | `koopman_diagnostics` |
| Mean-force energy | Scalar PMF + KDE score target | `MeanForcePotential`, `kde_mean_force` |
| Constant diffusion | Global Cholesky PSD tensor with `T/T_ref` clock | `ConstantDiffusion` |
| Bath gauge fixing | QR-orthogonal directions + ordered rates | `GaugeFixedBath` |
| Decoder/dynamics separation | Core contains no decoder | integration contract |
| Five losses | APIs for force and transition terms; JEPA/observation remain in host model | curriculum contract |
| GENERIC/FDT claims | FDT is measured from covariance residuals | `fdt_residual` |
| Integrability | Closed-loop conservative-force test | `energy_loop_circulation` |

## Equations Implemented

The potential emits a scalar `A_phi`. Its conservative mean force is computed
as `F_phi = -grad(A_phi)`. Given a KDE target

```text
F_PMF(z) = grad log p_data(z)
```

the force loss is

```text
L_force = mean ||F_phi(z) - F_PMF(z)||^2
```

which is equivalent to `||grad A_phi + F_PMF||^2`. The one-step overdamped
likelihood uses

```text
z_next ~ Normal(z - D(T) grad A_phi dt, 2 D(T) dt)
```

with a zero-mean bath for the local likelihood. Multi-step rollout uses the
Markovian embedding and explicitly exposes the auxiliary state for intervention
tests.

## What Is Intentionally Not Claimed Yet

The current additive implementation does not claim that:

- the learned JEPA encoder already passes the Koopman diagnostics;
- KDE mean force from a finite minibatch is an unbiased PMF estimator;
- the minimal bath rank has been identified on protein trajectories;
- LSP-WM beats EVA5C on the protein endpoint benchmark;
- a role readout dimension is a literal coordinate partition of the JEPA state;
- a synthetic contract result is a biological or literature result.

These are evaluation questions, not properties that can be established by a
unit test. The next experiment must freeze the manifest, split, seed, horizons,
and reporting code before comparing with the EVA5C checkpoint.

## Verification Commands

```bash
python -m pytest -q tests/test_lspwm.py
python scripts/evaluate_lspwm.py --synthetic --output artifacts/lspwm_smoke.json
```

The first command checks structural contracts. The second checks serialization,
diagnostic computation, PSD diffusion, and finite deterministic rollout. A
protein-scale evaluation should be added only after a trained LSP-WM checkpoint
exists and should report the same endpoint metric as
`artifacts/sota_eval_20260905/eva5c_final_checkpoint_seed6303_ck0.json`.
