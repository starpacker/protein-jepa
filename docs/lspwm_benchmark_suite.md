# LSP-WM Dynamics Benchmark Suite

This document defines an evaluation suite for a protein dynamics world model.
It is intentionally stricter than VAMP-2, a latent MSE, or a probe that
classifies a single MSM label. The object being evaluated is the conditional
path law

```text
p(y_{t+tau:t+H} | x_{<=t}, c, a_{t:t+H}, tau),
```

where `y` contains observable kinetic events and coarse states. A latent
representation is useful only if it improves held-out predictions of these
objects.

The suite has two scopes:

1. **Passive protein dynamics**, which can be run with the current mdCATH and
   FastFolding data.
2. **Full LSP-WM tests**, which require environment observations, actions, or
   multiple thermodynamic conditions. These are future gates, not claims about
   the current v4.1 implementation.

## 1. What Counts As Evidence

VAMP-2 measures recovery of slowly varying directions. MSM classification
measures whether a readout contains information about a precomputed label.
Neither metric evaluates whether the model can propagate uncertainty, predict
state transitions, preserve path statistics, or estimate first-passage events.

The primary evidence should therefore be a scorecard with four independent
axes:

| Axis | Question | Primary outputs |
| --- | --- | --- |
| Forecasting | Does the conditional distribution predict the future? | NLL, CRPS/energy score, calibration, persistence-relative skill |
| Kinetic operator | Does it predict transitions and relaxation? | transition-matrix error, occupancy forecast, MFPT, implied timescales, Chapman-Kolmogorov error |
| Reactive paths | Does it produce the right A-to-B paths? | committor, TPT distribution, reactive flux, path coverage |
| Physical/generalization | Does it remain valid across proteins, scales, and conditions? | protein-disjoint transfer, stationarity/detailed-balance checks, temperature/action response |

Do not collapse these axes into one number for the main claim. A model that
has excellent short-time forecast error but incorrect long-time occupancy is
not equivalent to a model with the opposite trade-off.

## 2. Frozen Evaluation Protocol

### 2.1 Splits

The default split is protein-disjoint:

```text
train proteins != validation proteins != test proteins
```

All temporal windows from one trajectory stay in the same split. Splitting
individual windows randomly is invalid because near-neighbor windows make
future prediction look much easier than it is.

For FastFolding, replicas sharing a scaffold must be grouped together. For
example, `A3D-0` and `A3D-1` should not be placed in different splits merely
because their directory names differ. The scaffold prefix or a sequence
identity cluster must be assigned before sampling windows.

The current FastFolding utility uses a blocked chronological split inside one
protein and sets `trajectory_id` to `protein_id`. That is acceptable for a
smoke test, but not for the final benchmark. The final runner must report:

- number of proteins and scaffold groups in every split;
- number of trajectories and frames per split;
- sequence identity and length ranges;
- sampled time gaps and horizon coverage;
- whether a label or normalization statistic was fitted using test data.

### 2.2 No future leakage

For a context ending at `t`, the encoder sees only observations at or before
`t`. The future is used only for scoring. State definitions, clustering,
MSM discretization, probe regularization, calibration, and feature scaling
must be fitted on train data and frozen before the test pass.

Every reported result must include a non-overlapping context rule. For paired
windows, the raw frame spans must not overlap unless the experiment is
explicitly labeled as an overlapping short-lag diagnostic.

### 2.3 Horizons

Report a fixed physical-time grid, not only frame indices. At minimum:

```text
short: 1, 2, 4 sampling intervals
medium: 8, 16, 32 sampling intervals
long: 1/4, 1/2, and 1 estimated slow relaxation time
```

The exact grid is dataset-dependent. A model may not claim long-horizon skill
when the test trajectory does not contain enough future frames to identify
that horizon.

## 3. Benchmark A: Probabilistic Future Forecasting

### Task

Given a context window, predict an ensemble or a conditional distribution of
future coarse observables at each horizon:

```text
q_native, RMSD_to_native, Rg, contact features, torsions, and MSM state.
```

For a model that only emits latent samples, fit a frozen train-only
observation/readout map and apply the same map to all test samples. A latent
MSE by itself is not a sufficient endpoint metric.

### Metrics

Report per horizon and per protein:

- Gaussian or flow NLL when a normalized density is available;
- CRPS for scalar observables;
- energy score for vector observables;
- Wasserstein-1 or MMD between predicted and observed endpoint ensembles;
- persistence-relative skill, with persistence as the mandatory baseline;
- marginal coverage of 50%, 80%, and 95% prediction intervals;
- interval width and calibration error;
- rank histogram or PIT histogram;
- failure rate: NaN, exploding norm, invalid structure, or out-of-support sample.

The primary score is the test-set average of per-protein metrics, not the
frame-weighted average. Otherwise long trajectories dominate the benchmark.

### Required baselines

1. Persistence: `y_{t+tau}=y_t`.
2. Geometry-only readout from the last frame.
3. Train-only empirical MSM / Markov state model.
4. Linear VAR or Koopman model in the same latent dimension.
5. The proposed model with history shuffled or truncated.

The history-shuffled control is important: it tests whether memory is carrying
causal predictive information rather than merely increasing model capacity.

## 4. Benchmark B: MSM State And Transition Prediction

This is the direct version of the proposed idea. The model must predict the
future state distribution, not only classify the state at time `t`.

### 4.1 State definition

Use one of the following, with the choice declared before test evaluation:

- train-fitted MSM states from a fixed observable space;
- physically defined states such as unfolded / transition / folded;
- a train-fitted metastable clustering with a held-out state assignment.

Never fit clusters or state thresholds on the test trajectory. Use the same
state map for all model variants and baselines.

### 4.2 Conditional transition metrics

For each horizon `tau`, compare predicted `p_hat(j | history_t)` with the
held-out outcome:

- multiclass log loss / negative log likelihood;
- Brier score and ranked probability score;
- balanced accuracy and macro-F1 only as secondary hard-decision metrics;
- expected calibration error and reliability diagrams;
- top-1 and top-2 transition accuracy;
- per-source-state recall, especially for rare states.

For an ensemble rollout, estimate `p_hat` by the fraction of samples assigned
to each state. Use a Dirichlet or Laplace smoothing rule fixed on validation.

### 4.3 Transition-operator metrics

Estimate the empirical held-out operator `T_tau` and compare it with the
model operator `T_hat_tau`:

```text
row-wise JS(T_hat_tau, T_tau)
row-wise L1 / total variation
stationary distribution distance
```

Then report kinetic consequences, which are more interpretable than matrix
distance alone:

- implied timescale relative error for the first `k` nontrivial modes;
- relaxation curve error from each source state;
- MFPT error for all sufficiently sampled state pairs;
- forward and reverse flux error;
- Chapman-Kolmogorov error:

```text
||T_hat_(tau1+tau2) - T_hat_tau1 T_hat_tau2||.
```

The semigroup test must use independently sampled rollout horizons. It should
not be satisfied only because the same fitted transition matrix is multiplied
by itself.

### 4.4 Why this is stronger than MSM loss

An MSM loss can be low while the model has the wrong stationary distribution,
wrong rare transition rates, or wrong long-time relaxation. A useful model
must jointly predict:

```text
state occupancy -> transition probabilities -> relaxation -> MFPT.
```

The final table should include all four levels.

## 5. Benchmark C: FastFolding Folding Dynamics

FastFolding is well suited to event-level evaluation because each frame can be
related to a native reference. The existing labels provide `q_native`, RMSD,
radius of gyration, a coarse folded/transition/unfolded state, local direction,
and a future-basin proxy. These are useful labels, but the proxy must be named
as a proxy rather than a true committor.

### C1. Progress and direction

Use context windows ending at matched structural regions and predict:

- future `q_native`, RMSD, and Rg;
- folding vs unfolding vs stationary direction;
- the sign and magnitude of future `Delta q_native`.

Report regression NLL/RMSE, rank correlation, balanced accuracy, macro-F1,
and calibration. The key diagnostic is a **direction-disambiguation split**:
find pairs of frames with similar current `q_native` and geometry but opposite
recent motion, then test whether the contextual state predicts opposite future
directions. A single-frame geometry baseline should be included explicitly.

### C2. Basin hitting and committor

Define `A=unfolded` and `B=folded` using thresholds frozen on train/validation.
For every eligible frame, predict:

```text
P(hit B before A within H)
P(hit B by H)
P(T_A->B <= H)
```

Evaluate with:

- Brier score and log loss;
- AUROC and AUPRC for imbalanced events;
- reliability curve and expected calibration error;
- calibration slope/intercept;
- performance in the transition region only;
- protein-wise bootstrap confidence intervals.

The current one-trajectory future-window label is a **committor proxy**. A
proper committor benchmark needs multiple independent continuations from the
same or matched configurations, or a shooting/reference calculation. Until
then, report both the proxy result and a clearly separate multi-trajectory
committor result when available.

### C3. First-passage and survival prediction

From each context, sample a future path and record the first hitting time of
the target basin. Compare the predicted and observed survival distributions:

- integrated Brier score for censored first-passage events;
- survival calibration at fixed horizons;
- Wasserstein distance and KS distance of first-passage times;
- median and quantile MFPT error;
- time-dependent concordance for ranking contexts by folding likelihood.

This catches a model that predicts the right final basin but crosses at the
wrong rate.

### C4. Transition-path ensemble

For paths that go from `A` to `B`, compare the generated and held-out path
ensembles using:

- transition-path time distribution;
- crossing-time and crossing-coordinate distribution;
- maximum excursion and path length;
- contact-formation order and contact-path features;
- intermediate-state occupancy;
- reactive flux profile;
- path-space MMD/Wasserstein distance on time-aligned kinetic features;
- coverage: fraction of held-out path clusters reached by generated paths.

Do not compare only endpoint RMSD. Two models can have identical endpoints and
different mechanisms.

## 6. Benchmark D: Markov Closure And Memory

This benchmark asks whether the learned state is actually sufficient at the
declared time resolution.

### Tests

1. **History ablation:** last frame, short history, full history, and shuffled
   history.
2. **Same-endpoint/different-history test:** contexts with similar final
   geometry but different recent paths must have different predictions when
   their future outcomes differ.
3. **Innovation test:** one-step residuals should have low autocorrelation after
   conditioning on the declared state and memory.
4. **Lag closure:** fit the transition law at `tau` and test at `2tau`, `4tau`.
5. **Memory-rank sweep:** report the trade-off between memory dimension and
   held-out path likelihood, state-transition accuracy, and residual whiteness.

### Metrics

- conditional future log loss;
- residual autocorrelation and Ljung-Box-style statistics;
- conditional mutual-information proxy between older history and future;
- Chapman-Kolmogorov error;
- long-horizon divergence and support collapse.

A high VAMP score with a large history-dependent residual is evidence that the
state is slow, but not Markov-closed.

## 7. Benchmark E: Equilibrium And Thermodynamic Consistency

For passive data, generated long trajectories should reproduce the stationary
law and its kinetic constraints.

### Required diagnostics

- state occupancy and observable distribution distance;
- free-energy profile error on declared low-dimensional observables;
- stationary distribution error after long rollout;
- forward/reverse transition asymmetry under passive equilibrium;
- probability-current norm, with the equilibrium/nonequilibrium protocol
  declared;
- FDT residual using the same transition/noise law used during rollout;
- temperature dependence of equilibrium populations and barrier rates when
  multiple temperatures exist;
- sensitivity to rollout length, step size, and ensemble size.

The PMF or equilibrium density is only one component of this section. Matching
it does not establish transition-path correctness.

## 8. Benchmark F: Temperature, Environment, And Action Transfer

These are full-world-model gates and should be added once the data contract
contains the relevant variables.

### Temperature transfer

Train on selected temperatures and test on a held-out temperature. Check:

- equilibrium population shift;
- transition-rate ratio and barrier scaling;
- MFPT and survival curves;
- FDT/noise consistency;
- zero-shot versus calibrated temperature conditioning.

Changing diffusion or rate parameters independently at each temperature is not
evidence of thermodynamic consistency.

### Environment transfer

Hold out ligand occupancy, salt, pH, membrane, or crowding conditions. Evaluate
both the protein response and the environment-mediated transition response.
The environment must be a state when it evolves on the prediction timescale;
it must not be treated as a static condition merely to make the task easier.

### Action/counterfactual transfer

For mutation, ligand, force, or temperature-jump actions, report:

- action-conditioned transition NLL;
- response effect size relative to the no-action counterfactual;
- recovery after action removal;
- held-out action-family generalization;
- work/entropy or another declared nonequilibrium quantity when applicable.

The current mdCATH manifests are passive and contain no intervention labels.
No action-conditioned or causal claim should be made from them.

## 9. Recommended Primary Table

The minimum paper table should contain the following rows for every model and
every test split:

| Group | Metrics |
| --- | --- |
| Endpoint forecast | NLL/CRPS/energy score at each horizon, persistence-relative skill, 80% coverage |
| MSM transition | state NLL, Brier, macro-F1, row-wise JS, stationary distance |
| Kinetics | timescale error, MFPT error, Chapman-Kolmogorov error |
| FastFolding events | direction macro-F1, committor Brier/log loss, survival IBS |
| Reactive paths | transition-path time, reactive flux, path MMD, coverage |
| Closure | residual ACF, history-ablation delta, long-horizon support/failure rate |
| Generalization | protein/scaffold-disjoint average and worst-group score |

For each metric report mean, standard error, and a protein/scaffold bootstrap
95% confidence interval. Significance must be computed on matched test
contexts, not by treating individual frames as independent samples.

## 10. Promotion Gates

A checkpoint should not be called a strong dynamics model unless it passes all
of the following qualitative gates:

1. It beats persistence and the train-only MSM on medium-horizon probabilistic
   endpoint forecasts.
2. It improves state-transition NLL/Brier without losing stationary occupancy.
3. It predicts MFPT and survival curves better than a static or one-step
   baseline.
4. Its committor predictions are calibrated, not only discriminative.
5. Its generated transition paths match at least two path-level statistics and
   do not collapse to one route.
6. Its gains survive protein/scaffold-disjoint evaluation.
7. Its claimed memory state reduces held-out residual dependence.
8. Any temperature, environment, or action claim passes the corresponding
   held-out transfer test.

Failure on one gate should be reported as a capability boundary, not hidden by
an aggregate score.

## 11. Mapping To Current Repository

The repository already has useful starting points:

- `docs/fastfolding_downstream.md` and `protein_jepa_frame.fastfolding` for
  read-only FastFolding labels and contextual probes;
- `protein_jepa_frame.eval.paper_eval` for protein-stratified representation
  evaluation and the current MSM probe;
- paired forecast evaluation for short/medium endpoint skill;
- `evaluate_lspwm_rollout.py` for numerical stability and FDT diagnostics.

The next implementation should add a held-out transition/event runner rather
than extend VAMP-2. Its outputs should contain machine-readable arrays for
per-protein, per-horizon, and per-state metrics so that bootstrap confidence
intervals and reliability plots can be regenerated without reopening the test
split.

For the current v4.1 model, the defensible near-term claim is:

> The model learns a candidate protein latent state and can be evaluated for
> passive endpoint forecasting, state-transition prediction, memory closure,
> and FastFolding event prediction. It has not yet demonstrated a complete
> environment/action-conditioned LSP-WM or a correct transition-path ensemble.

