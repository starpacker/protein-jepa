# TM-JEPA Implementation Guide

The physical derivation is in
`docs/first_principles_thermodynamic_mori_jepa_proposal_2026-09-02.md`.

## Implemented Model

The implementation is centered in
`protein_jepa_frame/models/thermodynamic.py`:

1. `FixedBackboneCollectiveVariables` provides differentiable, SE(3)-invariant
   smooth-contact, shape, segment, and torsion CVs.
2. `ProteinConditionEncoder` uses sequence, residue position, and simulation
   domain, but does not receive horizon or temperature.
3. `CausalAuxiliaryFilter` estimates dimensionless Mori bath variables from
   physical-time CV history.
4. `ThermodynamicMoriDynamics` learns `F(q;c,T)=U(q;c)-T S(q;c)`, constructs
   PSD diffusion, ties noise to dissipation, and performs stochastic
   Heun rollout.

V2 additionally enforces strict prefix causality in the auxiliary observer,
uses separate diffusion and memory-coupling bases, and ties successive
auxiliary posteriors to the same Mori transition used by the generator. The
full audit and derivation are in
`docs/tmjepa_v2_first_principles_audit_2026-09-02.md`.

The augmented dynamics are

$$
\begin{aligned}
dq &= [-D\nabla(\beta F)+Ca]dt+\sqrt{2D}\,dW_q,\\
da &= [-C^T\nabla(\beta F)-\Lambda a]dt+\sqrt{2\Lambda}\,dW_a.
\end{aligned}
$$

`D` is PSD by construction, `C` is the reversible coupling, and `Lambda` is
positive. Random and dissipative terms cannot be tuned independently.

## Fixed CVs

With three segments the physical vector has 16 channels:

```text
global_rg, end_to_end, global_contact, contact_order
segment_0_rg, segment_0_contact
segment_1_rg, segment_1_contact
segment_2_rg, segment_2_contact
segment_0_1_distance, segment_1_2_distance
mean_sin_phi, mean_cos_phi, mean_sin_psi, mean_cos_psi
```

Positive coordinates use `log`, `[0,1]` coordinates use `logit`, and signed
torsion moments use `atanh`. The SDE therefore evolves in an unconstrained
space. `log|dq/dy|` is retained when converting the learned free energy back to
the physical-CV reference measure. Outputs include both forms:

- `tmjepa_current_cv`, `tmjepa_target_cv`, `tmjepa_samples`: SDE coordinates;
- `tmjepa_current_physical_cv`, `tmjepa_target_physical_cv`,
  `tmjepa_sample_physical_cv`: inverse-transformed reporting coordinates.
- `tmjepa_physical_free_energy`: Jacobian-corrected physical-CV free energy.

## Objectives

`tmjepa_terms` implements:

- ensemble energy score;
- low-weight endpoint calibration NLL;
- denoising equilibrium score matching for `-grad(beta F)`;
- innovation mean, covariance, and lag-1 autocorrelation closure;
- local Gaussian path NLL with propagated auxiliary uncertainty;
- auxiliary transition-prior KL;
- auxiliary equilibrium-prior KL;
- optional learned-CV variance, covariance, and VAMP-2 terms.

There is no positive reversed-history loss, hard top-k balance, horizon
calibration, or deterministic endpoint semigroup loss.

## Data Splits

The generated pilot manifests contain 20 mdCATH proteins:

| Manifest | Temperatures | Replicas | Role |
|---|---|---|---|
| `tmjepa_pilot20.train.jsonl` | 320-413 K | 0,1,2 | fit |
| `tmjepa_pilot20.validation.jsonl` | 320-413 K | 3 | select |
| `tmjepa_pilot20.test.jsonl` | 320-413 K | 4 | final replica test |
| `tmjepa_pilot20.temperature_test.jsonl` | 450 K | 0-4 | temperature extrapolation |

Regenerate them with:

```bash
PYTHONPATH=. python scripts/build_tmjepa_pilot_manifests.py \
  --manifest /liuyutian/protein_jepa/artifacts/manifests/mdcath.train.jsonl \
  --output-dir manifests --prefix tmjepa_pilot20 \
  --num-proteins 20 --heldout-temperature-k 450
```

## Phase 0

Determine whether memory is identifiable before fitting a GLE:

```bash
PYTHONPATH=. python scripts/audit_tmjepa_identifiability.py \
  --train-manifest manifests/tmjepa_pilot20.train.jsonl \
  --eval-manifest manifests/tmjepa_pilot20.validation.jsonl \
  --output artifacts/tmjepa_identifiability.json
```

The audit compares `q_t` and `(q_t, history)` transition regressions on a
different replica. It reports held-out MSE, Gaussian conditional mutual
information, and innovation lag-1 autocorrelation.

## Training Sequence

The corrected V2 comparison first fits the no-memory baseline (`C=0`):

```bash
protein-jepa-frame train --config configs/phase6_tmjepa_v2_overdamped_pilot.yaml
```

Then fit four Mori memory modes only if Phase 0 supports memory:

```bash
protein-jepa-frame train --config configs/phase6_tmjepa_v2_mori_pilot.yaml
```

Only after fixed-CV thermodynamics passes, add four JEPA slow coordinates:

```bash
protein-jepa-frame train --config configs/phase6_tmjepa_learned_cv_pilot.yaml
```

The fixed-CV phases use `forward_tmjepa`, which skips unused JEPA encoders.
The learned-CV phase automatically uses the full causal/EMA encoder path.

## Evaluation

```bash
PYTHONPATH=. python scripts/evaluate_tmjepa.py \
  --config configs/phase6_tmjepa_v2_mori_pilot.yaml \
  --checkpoint artifacts/phase6_tmjepa_v2_mori_pilot/checkpoint_step300.pt \
  --manifest manifests/tmjepa_pilot20.validation.jsonl \
  --output artifacts/phase6_tmjepa_v2_mori_pilot/eval_validation.json \
  --max-batches 50 --batch-size 2 --ck-samples 8
```

Repeat against `test` and `temperature_test` only after selection is frozen.
The evaluator reports per-horizon energy skill, coverage, augmented-state CK
energy distance, innovation closure, diffusion eigenvalues, and relaxation
times. Promotion requires positive energy skill at every horizon, PSD
diffusion, low innovation mean, near-identity innovation covariance, and low
lag-1 innovation autocorrelation.

FDT parameter tying, explicit temperature, and horizon-independent drift are
architectural facts recorded in the evaluation JSON.

Compare the GLE against the required overdamped ablation before enabling
learned CVs:

```bash
PYTHONPATH=. python scripts/compare_tmjepa_ablation.py \
  --overdamped artifacts/phase6_tmjepa_v2_overdamped_pilot/eval_validation.json \
  --gle artifacts/phase6_tmjepa_v2_mori_pilot/eval_validation.json \
  --output artifacts/tmjepa_v2_ablation_comparison.json
```

Memory is promoted only when it preserves energy skill, improves full-matrix
innovation autocorrelation by at least 5%, and improves held-out path NLL by
at least 0.001 nat per CV per transition. Augmented-state CK remains a
numerical integrator diagnostic, not a memory-evidence gate.

## Scope Boundary

This model predicts distributions in defined backbone-CV space; it does not
claim atomistic structure generation. mdCATH forces are not used in the first
pilot because a valid CV mean-force target requires coordinate-metric and
Fixman/Jacobian corrections. The previous direct atom-force comparison is not
reused as evidence for a PMF.

Long-run occupancy, MFPT, and committor analysis requires basin definitions
and long generated trajectories. Those are downstream analyses of a promoted
checkpoint, not surrogate training losses.
