# Reviewer-Level Experiment TODO

Date: 2026-09-12

Repository: `/liuyutian/protein_jepa_frame`

## 0. Purpose

This document converts the reviewer recommendations into an executable
experiment checklist. It is intended to guide both implementation and the
Results section of the paper.

The paper should establish one evidence chain:

```text
better representation
    -> smaller extra benefit from history
    -> finite memory explains the remaining non-closure
    -> realistic equilibrium and protein kinetics
```

The main claim is not that JEPA has the lowest endpoint MSE. The main claim is
that the learned state is closer to a dynamically closed state, and that a
thermodynamically structured finite-memory model recovers the remaining
temporal dependence without sacrificing equilibrium behavior.

This file is an experiment plan. It does not promote a checkpoint, change the
model, or authorize use of the blind test set for selection.

## 1. Status Legend

| Mark | Meaning |
|---|---|
| `[x]` | Existing implementation or result has been verified in the repository |
| `[~]` | Partial support exists, but the reviewer-level experiment is incomplete |
| `[ ]` | TODO |
| `P0` | Must be completed before paper-level experiments |
| `P1` | Main evidence required for the paper |
| `P2` | Supporting or robustness experiment |

## 2. Current Repository State

### 2.1 Already available

- `[x]` Protein-disjoint mdCATH manifests:
  `manifests/tmjepa_v3.train.jsonl`,
  `manifests/tmjepa_v3.validation.jsonl`,
  `manifests/tmjepa_v3.test.jsonl`,
  and `manifests/tmjepa_v3.temperature_test.jsonl`.
- `[x]` Protocol audit and checkpoint-screening workflow in
  `docs/revise_plan_implementation_2026-09-11.md`.
- `[x]` Existing VAMP, TICA/MSM, linear-dynamics, transition-operator, MFPT,
  stationary-distribution, semigroup, FDT, and innovation diagnostics.
- `[x]` A FastFolding manifest/label/probe/benchmark workflow.
- `[x]` Synthetic parameterization contract tests and finite-value checks.
- `[x]` Existing TM-JEPA/LSP-WM Mori and overdamped ablations.
- `[x]` Existing transition calibration diagnostic:
  `scripts/evaluate_lspwm_transition_calibration.py`.

### 2.2 Existing but not sufficient for the reviewer claims

- `[~]` Existing synthetic tests verify tensor and parameterization contracts,
  but do not recover a known double-well free energy and known exponential
  memory kernel.
- `[~]` Existing VAMP/MSM/linear probes show kinetic information, but there is
  no single frozen-representation evaluation that reports the reviewer
  `History Gap` over multiple horizons and history lengths.
- `[~]` Existing `evaluate_tmjepa_v3.py` reports innovation covariance,
  lagged correlations, FDT-related diagnostics, and physical gates, but it
  does not provide the complete 2x2 thermodynamic ablation plus a matched
  GRU/causal-Transformer history baseline.
- `[~]` Existing Mori results are useful ablations, but the current reports
  show that the auxiliary bath is not yet a reliable predictive closure. They
  cannot be presented as proof that Mori memory wins.
- `[~]` FastFolding currently supports a `committor_proxy`. The main claim must
  use a committor computed from a reference MSM with fixed boundary conditions.
- `[~]` Existing transition-operator utilities include implied timescales and
  MFPT, but the paper pipeline must put implied-timescale error before MFPT
  and report both per protein and as an aggregate with confidence intervals.

### 2.3 Explicit gaps

- `[ ]` Synthetic double-well GLE parameter-recovery benchmark.
- `[ ]` Alanine-dipeptide benchmark with a declared fixed-CV protocol.
- `[ ]` History Gap evaluator and matched probe training pipeline.
- `[ ]` Generic GRU or causal-Transformer history baseline for closure.
- `[ ]` Unified innovation-whiteness score
  `W = sum_tau ||Corr(epsilon_t, epsilon_{t+tau})||_F`.
- `[ ]` Reference-MSM committor solver and model-vs-reference committor
  calibration.
- `[ ]` Final multi-protein kinetics table with transition operator,
  implied-timescale, committor, MFPT, and basin-occupancy metrics.
- `[ ]` Three-seed confirmatory runs after model/config selection is frozen.

## 3. Non-Negotiable Evaluation Protocol

### 3.1 Split policy

- Use protein-disjoint mdCATH train/validation/test splits.
- Keep all trajectories and windows from one protein in one split.
- If family or topology metadata is available, report family/topology
  stratification in addition to protein-disjoint performance.
- For FastFolding, keep replicas from the same scaffold in one split.
- Fit normalization, state definitions, clustering, MSMs, probe regularizers,
  and calibration parameters on training data only.
- Use validation for model selection and threshold selection.
- Open the blind test set once, after all decisions are frozen.

### 3.2 Reproducibility

Every run must record:

- checkpoint path and SHA256;
- immutable config path and serialized config;
- manifest path and manifest summary;
- code commit;
- random seed;
- device and package environment;
- batch size, number of batches, sampler mode, context length, frame stride,
  window stride, and all horizons;
- train-only fitted objects, including scaler, PCA/tICA/VAMP model, state map,
  MSM, ridge regularizer, and calibration map;
- output directory and machine-readable result path.

Development runs use seed `6303` unless an experiment defines another seed.
Every promoted result requires seeds `6303`, `6304`, and `6305`, with the
same config and evaluator.

### 3.3 Reporting unit and uncertainty

- Report per-protein metrics first, then macro mean across proteins.
- Do not allow long trajectories to dominate a frame-weighted aggregate.
- Report paired bootstrap 95% confidence intervals over proteins.
- Report the number of valid proteins and the number of valid transitions for
  every metric.
- Report every horizon separately. A macro score may be included only after
  the per-horizon table.
- Mark unavailable, degenerate, or under-sampled quantities explicitly.

## 4. Experiment Matrix

| ID | Priority | Experiment | Core question | Dataset | Main output | Status |
|---|---|---|---|---|---|---|
| `P0` | P0 | Protocol and artifact freeze | Can every result be traced and reproduced? | mdCATH, FastFolding, synthetic | audit + run manifest + JSON schema | `[~]` |
| `S1` | P1 | Synthetic GLE recovery | Does `A(z)+D+K(t)` recover known reduced dynamics? | double-well synthetic GLE | `A`, `D`, `K`, equilibrium, ACF, rates | `[ ]` |
| `S2` | P2 | Alanine sanity check | Does the physical parameterization behave on a real small system? | alanine dipeptide | PMF, stationary law, ACF, rates, FDT | `[ ]` |
| `R1` | P1 | Representation closure | Does JEPA reduce the extra predictive value of history? | mdCATH, protein-disjoint | History Gap, residual ACF, VAMP/ITS | `[ ]` |
| `M1` | P1 | Thermodynamic closure ablation | Does free energy plus finite memory explain remaining non-closure? | mdCATH, optional FastFolding subset | innovation whiteness, stationary JSD, FDT, CK | `[~]` |
| `K1` | P1 | Protein kinetics recovery | Does the closed model recover long-time protein kinetics? | FastFolding, mdCATH held-out proteins | operator, ITS, committor, MFPT, occupancy | `[~]` |
| `R2` | P2 | Robustness and transfer | Are conclusions stable across seeds, families, and temperature? | mdCATH temperature test | CI, temperature transfer, failure analysis | `[ ]` |

## 5. P0: Protocol and Artifact Freeze

### Goal

Make data leakage, inconsistent readouts, invalid force diagnostics, and
ambiguous artifact provenance visible before the main experiments begin.

### Checklist

- `[x]` Run the canonical protocol audit:
  `scripts/audit_lspwm_protocol.py`.
- `[x]` Confirm zero protein overlap among mdCATH train, validation, and test.
- `[x]` Keep endpoint evaluation and representation evaluation as separate
  tracks. Do not place LSP-WM endpoint scores and TM-JEPA latent probe scores
  on one leaderboard.
- `[ ]` Define one JSON result schema shared by `S1`, `R1`, `M1`, and `K1`.
- `[ ]` Add a run-provenance block to every new result:
  `protocol`, `data`, `model`, `metrics`, `uncertainty`, `claim_boundary`.
- `[ ]` Add a test that fails when a test manifest is used for selection.
- `[ ]` Add a test that fails when a force metric is promoted with
  `force_target_degenerate=true`.
- `[ ]` Freeze the state definition, MSM discretization, committor boundaries,
  and calibration rules before the final test pass.
- `[ ]` Create one output root:
  `artifacts/reviewer_experiments/`.

### Required artifact layout

```text
artifacts/reviewer_experiments/
  protocol/
  synthetic_gle/
  alanine/
  representation_closure/
  thermodynamic_closure/
  protein_kinetics/
  robustness/
```

Each experiment directory should contain:

```text
config.json
run_manifest.json
results.json
summary.md
figures/
```

## 6. S1: Synthetic Double-Well GLE Recovery

### Scientific question

When the true reduced dynamics is known, can the implementation recover both
equilibrium structure and temporal statistics?

### Data and ground truth

- `[ ]` Implement a seeded generator for a one-dimensional or low-dimensional
  double-well potential `A*(z)`.
- `[ ]` Use known diffusion `D*` and a finite exponential memory kernel
  `K*(t)`, represented by a Markovian auxiliary bath.
- `[ ]` Generate train, validation, and test trajectories from the same
  declared simulator with independent random seeds.
- `[ ]` Save the exact `A*`, `D*`, `K*`, integration step, temperature, and
  trajectory seeds next to the manifest.
- `[ ]` Include a regime with slow memory and a regime with short memory.

### Methods

Run all methods on the same observed trajectories and sampling interval:

1. Markov Langevin baseline.
2. Unconstrained latent SDE baseline.
3. Ordinary finite GLE baseline.
4. Full thermodynamic model with free energy, diffusion, and Mori memory.

The method names, parameter counts, optimizer budget, and observation noise
must be recorded. The full model must not receive more history or more
information than the generic history baselines.

### Primary metrics

- Free-energy recovery error between learned `A_theta(z)` and `A*(z)`,
  after fixing the additive constant.
- Memory-kernel error between learned `K_theta(t)` and `K*(t)`.
- Stationary-distribution Jensen-Shannon divergence.

### Secondary metrics

- Autocorrelation-function error over a fixed lag grid.
- Transition-rate relative error between wells.
- FDT residual.
- Passive detailed-balance diagnostic.
- Long-rollout stability and occupancy drift.

### Expected evidence pattern

- The full model should recover equilibrium and temporal statistics together.
- Removing memory may preserve the stationary distribution while degrading
  ACFs and transition rates.
- Removing the free-energy term should degrade stationary occupancy and may
  produce long-run drift.

### Implementation tasks

- `[ ]` Add `scripts/generate_synthetic_gle.py`.
- `[ ]` Add `scripts/evaluate_synthetic_gle.py`.
- `[ ]` Add a small synthetic manifest format under
  `artifacts/reviewer_experiments/synthetic_gle/`.
- `[ ]` Add analytic or numerical ground-truth comparison utilities.
- `[ ]` Add regression tests for parameter recovery, stationarity, and kernel
  serialization.
- `[ ]` Generate one small smoke result before the full sweep.

### Paper placement

Use as a small sanity panel or appendix figure. A compact true-vs-learned
`A(z)` and `K(t)` inset is enough for the main paper. Do not let this become a
separate Results section.

## 7. S2: Alanine-Dipeptide Sanity Benchmark

### Scientific question

Does the parameterization behave correctly on a real molecular system where
the low-dimensional coordinate and equilibrium law can be independently
checked?

### Data and protocol

- `[ ]` Locate or generate an alanine-dipeptide trajectory with documented
  force field, temperature, timestep, saved-frame interval, and coordinate
  convention.
- `[ ]` Declare the fixed low-dimensional CVs before evaluation.
- `[ ]` Split trajectories or blocks without mixing future windows across
  splits.
- `[ ]` Fit the reference PMF and all calibration statistics on train data.
- `[ ]` Reserve an independent trajectory or block for final confirmation.

### Metrics

- PMF/free-energy recovery error.
- Stationary-distribution JSD.
- ACF error for each declared CV.
- Transition rates between metastable basins.
- FDT residual and detailed-balance diagnostic.
- Geometry and periodic-coordinate validity checks.

### Implementation tasks

- `[ ]` Add a manifest entry and a documented data path.
- `[ ]` Add a fixed-CV evaluator or adapt
  `scripts/evaluate_lspwm_physical.py`.
- `[ ]` Record whether the result is a real-data benchmark or a synthetic
  replay. Do not call a synthetic replay alanine-dipeptide evidence.
- `[ ]` Add a smoke test and one held-out evaluation.

### Exit condition

S2 is a sanity benchmark only. It is not a substitute for the mdCATH
representation experiment or the FastFolding protein kinetics experiment.

## 8. R1: Predictive Representation and Dynamic Closure

### Scientific question

After the representation `z_t` is known, how much does the past
`z_{t-H:t}` still help predict the future? The key quantity is the History Gap,
not raw JEPA endpoint MSE.

### Dataset and split

- Use mdCATH with the canonical protein-disjoint train/validation/test split.
- Use identical context construction and physical horizons for all
  representations.
- Add family/topology stratification if the manifest provides reliable
  metadata.
- Use FastFolding only as an optional downstream confirmation, not as a
  replacement for mdCATH.

### Representations

At minimum, evaluate:

1. PCA or structural baseline.
2. tICA/VAMP/SRV-style kinetic representation.
3. Static encoder representation.
4. One-step JEPA representation.
5. Multi-horizon JEPA representation.

MEMnets or other history models are optional supporting baselines. They should
not replace the matched current-state versus history probe comparison.

### Matched probe design

For every frozen representation, train two probes with matched capacity and
the same training data:

- current-state probe: `z_t -> z_{t+tau}`;
- history probe: `z_{t-H:t} -> z_{t+tau}`.

Repeat across multiple physical horizons and history lengths. Suggested
development grid:

```text
tau: short, medium, long non-overlapping lags
H:   1, 2, 4, 8 history steps
```

The exact grid must be recorded in the run manifest and held fixed across
representations.

### Primary metric

Define:

```text
Delta_hist(tau, H) = L[z_t -> z_(t+tau)]
                     - L[z_(t-H:t) -> z_(t+tau)]
```

where `L` is the same held-out probe loss for both probes. Smaller
`Delta_hist` means that the current representation contains more of the
information needed for the future.

Report:

- per-horizon and per-history-length `Delta_hist`;
- macro `Delta_hist` across proteins;
- paired bootstrap 95% CI;
- history gain relative to the static and one-step baselines.

The central comparison is:

```text
Delta_hist(JEPA) < Delta_hist(static)
```

not merely:

```text
MSE(JEPA) < MSE(baseline)
```

### Secondary metrics

- Residual autocorrelation integral after the current-state probe.
- Residual autocorrelation integral after the history probe.
- VAMP-2 with the current repository's normalized convention.
- Implied-timescale recovery error.
- Multi-horizon NLL or `R^2`.
- Semigroup defect from independently evaluated horizons.
- Stability across history lengths.

### Additional controls

- Shuffle or reverse the history while preserving the current endpoint.
- Match the parameter count between current-state and history probes.
- Keep the encoder frozen during probe fitting.
- Fit all probe regularization on train-only protein groups.
- Report whether the history probe uses raw frames, latent frames, or both.

### Implementation tasks

- `[ ]` Add `scripts/evaluate_history_gap.py`.
- `[ ]` Reuse the existing hierarchical sampler and explicit manifest
  arguments.
- `[ ]` Add per-protein residual ACF and integrated-correlation output.
- `[ ]` Add a common representation export format so each probe uses the same
  saved latent samples.
- `[ ]` Add matched-capacity probe tests.
- `[ ]` Run a smoke comparison on the current State-JEPA champion.
- `[ ]` Run the full validation comparison across all representations.
- `[ ]` Freeze the representation ranking before opening test data.

### Acceptance gate

R1 is successful only if the proposed JEPA representation has a smaller
History Gap than the static baseline on the main horizon range, with the
direction of the difference stable under bootstrap uncertainty. A lower raw
endpoint MSE without a smaller History Gap is not sufficient.

### Paper placement

Main Fig. 2:

- Panel A: History Gap versus horizon.
- Panel B: residual ACF or integrated residual correlation.
- Panel C: VAMP/ITS summary.
- Panel D: representation comparison across proteins or model families.

## 9. M1: Thermodynamic Force and Finite-Memory Closure

### Scientific question

When the current state is not fully Markovian, does a thermodynamically
structured finite Mori memory explain the remaining history dependence better
than either a memory-free model or a generic history network?

The claim must be stronger than `Mori > Markov`:

```text
Mori approximately matches or exceeds generic history prediction
and beats generic history on physical closure.
```

### Fixed representation and data

- Use the representation selected by R1.
- Freeze the representation for the first M1 comparison.
- Use mdCATH held-out proteins as the primary benchmark.
- Add a FastFolding slow-transition subset only as a supporting benchmark.
- Use the same history window, horizon grid, sampler, and number of trainable
  parameters wherever possible.

### Core 2x2 ablation

| ID | Free-energy force | Finite memory | Meaning |
|---|---:|---:|---|
| `M-Markov` | no | no | generic Markov transition baseline |
| `M-FE` | yes | no | thermodynamic free-energy Markov model |
| `M-Memory` | no | yes | memory-only control |
| `M-Full` | yes | yes | proposed free-energy plus Mori memory |

Add the following strong history controls:

- `[ ]` GRU with the same observed history window.
- `[ ]` Causal Transformer with the same observed history window.
- `[ ]` History-shuffled or endpoint-preserving history control.

The generic history baselines must be evaluated with both predictive and
physical metrics. A prediction-only win is not enough.

### Primary metrics

1. Innovation whiteness:

   ```text
   W = sum_(tau > 0) || Corr(epsilon_t, epsilon_(t+tau)) ||_F
   ```

   Use the same lag set for every method and report the individual lag curves.

2. Multi-step/path NLL.
3. Stationary-distribution JSD.
4. Implied-timescale relative error.
5. Chapman-Kolmogorov or semigroup error.

### Secondary metrics

- History Gap after adding the closure model.
- ACF error.
- FDT residual.
- Transition-operator row JS/L1 error.
- Stationary population error.
- Diffusion PSD, minimum eigenvalue, and conditioning.
- Rollout failure rate and norm drift.

### Required interpretation pattern

The ideal result is a functional decomposition:

- free energy mainly improves stationary occupancy;
- memory mainly improves temporal residuals and slow kinetics;
- GRU/Transformer can be competitive on short-term prediction but worse on
  equilibrium, FDT, innovation whiteness, or long-run operator statistics;
- `M-Full` is the most balanced model.

Do not report a generic history network as a strawman. Match its history window,
parameter budget, training examples, and early-stopping rule.

### Existing code to reuse

- `scripts/evaluate_tmjepa_v3.py`
- `scripts/compare_tmjepa_v3_ablation.py`
- `scripts/identify_tmjepa_v3_memory.py`
- `scripts/probe_tmjepa_v3_markov_closure.py`
- `protein_jepa_frame/eval/dynamics_benchmarks.py`

### Implementation tasks

- `[ ]` Define a common adapter interface for Markov, free-energy, Mori,
  GRU, and causal-Transformer predictors.
- `[ ]` Implement the generic history baselines.
- `[ ]` Add the unified innovation-whiteness score `W`.
- `[ ]` Add stationary JSD and FDT outputs to the common result schema.
- `[ ]` Ensure all methods use the same initial-state distribution for rollout.
- `[ ]` Evaluate all four 2x2 cells before changing the proposed architecture.
- `[ ]` Run the existing W250/V250-style Mori artifacts through the new
  evaluator for a direct apples-to-apples comparison.
- `[ ]` Add per-protein paired bootstrap comparisons.

### Acceptance gate

M1 is successful only if the proposed model improves physical closure metrics
without a material degradation in predictive metrics, and the advantage over
generic history is visible in at least two of:

- innovation whiteness;
- stationary distribution;
- FDT residual;
- implied timescales;
- long-rollout stability.

### Paper placement

Main Fig. 3:

- Panel A: 2x2 ablation summary.
- Panel B: innovation ACF and `W`.
- Panel C: stationary-distribution error.
- Panel D: Mori versus generic history, with predictive and physical metrics
  shown side by side.

## 10. K1: Closed Latent Dynamics and Protein Kinetics

### Scientific question

Does the learned closed dynamics recover folding/metastable kinetics and
long-run occupancy, rather than only endpoint prediction?

### Datasets

Primary:

- FastFolding with scaffold-grouped splits.
- All available target proteins and replicas, not only one smoke protein.

Supporting:

- mdCATH held-out proteins with fixed train-fitted metastable state maps.

The current repository documentation names FastFolding examples including
`2F4K`, `CLN025`, `GTT-0`, and `GTT-1`. The final run must record the actual
proteins, scaffold groups, frame counts, and whether the full source dataset
was available in the execution environment.

### Reference state and MSM

- `[ ]` Fit the reference metastable-state assignment on training data only.
- `[ ]` Build a reference MSM from the training trajectories.
- `[ ]` Validate the state map on validation data without refitting.
- `[ ]` Compute reference held-out transition operators for each horizon.
- `[ ]` Define folded and unfolded basins before final evaluation.
- `[ ]` Solve the reference MSM committor:

  ```text
  q_i = sum_j T_ij q_j
  q_A = 0
  q_B = 1
  ```

- `[ ]` Remove the current `committor_proxy` from the primary claim.

### Required methods

At minimum:

1. tICA + MSM.
2. VAMPnet or SRV-style kinetic representation.
3. Black-box latent autoregressive model.
4. JEPA one-step predictor.
5. JEPA plus Markov SDE.
6. Generic-memory history model.
7. Full thermodynamic finite-memory model.
8. Persistence baseline.

Every learned model must start rollouts from the same initial-state
distribution and use the same train-fitted observation/readout map.

### Primary metrics

Report per horizon and per protein:

- Transition-matrix row JS divergence.
- Transition-matrix L1 or total-variation error.
- Implied-timescale relative error:

  ```text
  t_i(tau) = -tau / log(abs(lambda_i(tau)))
  ```

- Reference-MSM committor Pearson correlation.
- Reference-MSM committor Spearman correlation.
- Reference-MSM committor MAE.
- Committor calibration or reliability error.
- MFPT relative error for sufficiently sampled basin pairs.
- Basin-population or stationary-distribution error.

Implied timescales must appear before MFPT in the main table. Current
repository results suggest that transition structure can look reasonable while
absolute time calibration remains poor, so MFPT alone is not an adequate
kinetic summary.

### Secondary metrics

- State forecast NLL and Brier score.
- ECE and macro-F1.
- ACF error.
- VAMP-2.
- Path NLL.
- Transition-path duration distribution.
- Folding/unfolding rate ratio.
- Endpoint skill.
- Rollout invalidity, exploding norm, and out-of-support rate.

### Implementation tasks

- `[ ]` Build the full FastFolding manifest and labels without modifying the
  source dataset.
- `[ ]` Add scaffold-grouped final splits and report split statistics.
- `[ ]` Implement a train-fitted reference MSM builder.
- `[ ]` Implement an exact discrete committor solver.
- `[ ]` Add model rollout adapters that emit state samples/probabilities and,
  where available, full paths.
- `[ ]` Extend `scripts/evaluate_dynamics_benchmarks.py` to report true
  committor metrics and confidence intervals.
- `[ ]` Add independently sampled semigroup checks at multiple horizons.
- `[ ]` Run persistence and linear/VAR baselines before model evaluation.
- `[ ]` Evaluate 2 to 3 representative proteins for figure panels.
- `[ ]` Evaluate the complete protein set for Main Table 1.

### Acceptance gate

K1 is successful only if the full model is stable across:

- equilibrium or basin occupancy;
- slow implied timescales;
- committor structure;
- MFPT;
- long-rollout validity.

The full model does not need to win every short-horizon endpoint metric, but
it should be the most reliable joint model of occupancy and slow kinetics.

### Paper placement

Main Fig. 4:

- Representative transition operators or relaxation curves.
- Implied-timescale comparison for 2 to 3 proteins.
- Reference versus model committor.
- Long-rollout occupancy or folding/unfolding kinetics.

Main Table 1:

- All proteins, mean plus 95% CI.
- Transition operator error.
- Implied-timescale error.
- Committor correlation and MAE.
- MFPT relative error.
- Basin-population error.

## 11. R2: Robustness and Transfer

### Checklist

- `[ ]` Repeat the final selected configuration with seeds `6303`, `6304`,
  and `6305`.
- `[ ]` Report per-protein bootstrap intervals in addition to seed variation.
- `[ ]` Run the frozen model on `manifests/tmjepa_v3.temperature_test.jsonl`.
- `[ ]` Report performance by temperature and horizon.
- `[ ]` Confirm no temperature-specific normalization or calibration used
  blind test information.
- `[ ]` Report family/topology-stratified performance where metadata permits.
- `[ ]` Record negative results and failed gates in `summary.md`.

### Robustness acceptance

A claim is paper-ready only when the direction of the main result is stable
across seeds and is not driven by one protein, one temperature, or one
horizon.

## 12. Execution Order

Use the following order. Do not start with a large architecture sweep.

### Phase A: Freeze and smoke

- `[ ]` Complete P0 result schema and artifact layout.
- `[ ]` Run existing protocol audit.
- `[ ]` Run synthetic GLE smoke generation and evaluation.
- `[ ]` Run a one-protein FastFolding smoke benchmark.
- `[ ]` Add evaluator regression tests.

### Phase B: Sanity benchmarks

- `[ ]` Complete S1 on train/validation/test synthetic splits.
- `[ ]` Complete S2 if alanine data is available.
- `[ ]` Confirm all physical metrics and ground-truth comparisons are finite.

### Phase C: Representation closure

- `[ ]` Run R1 on validation with all representations.
- `[ ]` Select the representation using only validation.
- `[ ]` Freeze the representation and probe configuration.
- `[ ]` Run R1 once on the blind test set.

### Phase D: Thermodynamic closure

- `[ ]` Run the M1 2x2 ablation.
- `[ ]` Add GRU and causal-Transformer history baselines.
- `[ ]` Select the closure model using validation.
- `[ ]` Run the final closure model once on test.

### Phase E: Protein kinetics

- `[ ]` Build the reference MSM and true committor.
- `[ ]` Run K1 baselines.
- `[ ]` Run K1 model rollouts.
- `[ ]` Produce Main Table 1 and Fig. 4.

### Phase F: Confirmatory robustness

- `[ ]` Repeat the selected representation and closure models with three seeds.
- `[ ]` Run temperature/family/topology transfer.
- `[ ]` Freeze all paper numbers and synchronize docs, configs, and artifacts.

## 13. Stop and Promotion Rules

Do not promote a model because of one attractive metric. Use these gates:

- `[ ]` Protocol gate: explicit manifests, no split overlap, reproducible seed,
  complete provenance.
- `[ ]` Sanity gate: synthetic parameter recovery and stationary-law checks
  pass.
- `[ ]` Representation gate: JEPA has a smaller History Gap than the static
  baseline on the main horizon grid.
- `[ ]` Closure gate: full model improves innovation/physical closure and
  remains competitive with generic history prediction.
- `[ ]` Kinetics gate: full model recovers occupancy, implied timescales,
  committor, and MFPT without rollout instability.
- `[ ]` Robustness gate: direction of the result is stable across three seeds
  and per-protein bootstrap uncertainty.

If a gate fails, record the failure and stop the corresponding claim. Do not
repair the evaluator and silently keep the old result.

## 14. Results Recording Template

For every completed experiment, add one row to a running experiment table:

| ID | Date | Commit | Config | Checkpoint | Manifest | Seeds | Primary result | Gate | Artifact |
|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |

Every `summary.md` should answer:

1. What scientific question was tested?
2. What information did each method receive?
3. What was fitted on train, validation, and test?
4. What are the per-protein primary metrics?
5. What is the paired bootstrap interval?
6. Which gate passed or failed?
7. What claim is supported?
8. What claim remains disabled?

## 15. Current Paper Story

The intended main-paper structure is:

1. Methods and protocol, including protein-disjoint evaluation and the
   distinction between representation and transition-law evaluation.
2. Fig. 1: compact synthetic/physical sanity evidence.
3. Fig. 2: JEPA representation and History Gap.
4. Fig. 3: free-energy/memory closure and generic-history comparison.
5. Fig. 4 and Table 1: protein kinetics, committor, implied timescales, MFPT,
   and long-rollout occupancy.

The current repository results already justify a cautious statement that
thermodynamically guarded JEPA endpoint models are numerically stable and can
improve internal multi-horizon endpoint skill. They do not yet establish the
stronger reviewer-level chain above. The missing experiments in this document
are the evidence needed to make that stronger claim responsibly.

## 16. Useful Existing Entry Points

Protocol and endpoint:

- `scripts/audit_lspwm_protocol.py`
- `scripts/evaluate_lspwm_endpoint.py`
- `scripts/evaluate_lspwm_transition_calibration.py`
- `protein_jepa_frame.eval.comprehensive_eval`

Representation and closure:

- `scripts/probe_tmjepa_v3_markov_closure.py`
- `scripts/identify_tmjepa_v3_memory.py`
- `scripts/evaluate_tmjepa_v3.py`
- `scripts/compare_tmjepa_v3_ablation.py`
- `scripts/evaluate_v5_contextual_dynamics.py`

Protein kinetics:

- `scripts/generate_fastfolding_predictions.py`
- `scripts/evaluate_dynamics_benchmarks.py`
- `protein_jepa_frame/eval/dynamics_benchmarks.py`
- `docs/lspwm_benchmark_suite.md`
- `docs/fastfolding_downstream.md`

Existing design and status reports:

- `docs/revise_plan_implementation_2026-09-11.md`
- `docs/lspwm_benchmark_suite.md`
- `docs/eva5c_final_sota_2026-09-06.md`
- `docs/phase8u_encoder_dynamics_sota_2026-09-05.md`
- `docs/phase8w_memory_residual_ablation_2026-09-05.md`

