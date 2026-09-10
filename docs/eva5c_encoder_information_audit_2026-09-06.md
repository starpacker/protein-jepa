# EVA5C Encoder Dynamics-Information Audit

Date: 2026-09-06

## Question

Does the EVA5C encoder preserve the slow and metastable information needed for
protein dynamics, or is the downstream dynamics model operating on an
information-deficient representation?

## Checkpoint identity

EVA5C was derived from U250 with the online representation frozen. Direct
checkpoint comparison gives:

| Module | Tensors compared | Different tensors |
|---|---:|---:|
| Context encoder | 42 | 0 |
| Temporal encoder | 38 | 0 |
| Learned-CV projection | 4 | 0 |

Therefore this audit applies to the online representation shared by U250 and
EVA5C. The target towers are not identical because EVA5B synchronized the
target from the online tower at initialization. Target synchronization does not
change the online frozen readouts measured here.

## Protocol

The quick representation audit uses:

- checkpoint: `eva5c_tmjepa_v3_quadratic_latent_closure/checkpoint_step375_gate2.pt`
- mdCATH validation manifest and MSM labels used by the legacy paper pipeline
- 24-frame causal context, frame stride 2, window stride 8
- first non-overlapping evaluation lag 6
- protein-stratified blocked VAMP/linear splits
- per-protein trajectory-heldout MSM probe
- ridge regularization selected inside training trajectories
- VAMP/linear quick subset: 10 trajectories
- MSM subset: 2000 labeled clips

The quick protocol is sufficient for diagnosis but is not the final
paper-scale estimate. A full 50-trajectory rerun is required for final
confidence intervals.

## Readouts

| Readout | Dimension | Information source |
|---|---:|---|
| `causal_endpoint` | 64 | Last causal temporal-encoder state |
| `tmjepa_fixed_cv` | 16 | Explicit backbone collective variables |
| `tmjepa_learned_cv` | 4 | Learned projection of the causal state |
| `tmjepa_cv` | 20 | Concatenated fixed and learned CVs |

## Results

| Readout | VAMP norm lag6 dim5 | VAMP norm lag6 dim10 | MSM accuracy | MSM balanced accuracy | MSM macro-F1 | Linear R2 lag6 |
|---|---:|---:|---:|---:|---:|---:|
| Causal endpoint 64D | 0.3805 | 0.6458 | 40.82% | 35.00% | 30.64% | -0.464 |
| Fixed CV 16D | **0.3834** | **0.6531** | **53.10%** | 43.68% | **38.17%** | not run |
| Learned CV 4D | 0.2902 | 0.2902* | 45.00% | 35.44% | 29.10% | not run |
| Full CV 20D | 0.3815 | 0.6519 | 51.96% | **43.80%** | 37.36% | -0.239 |
| Geometry statistics | 0.3729 | 0.6496 | - | - | - | -0.184 |
| Centered CA | 0.4291 | 0.6641 | - | - | - | -2.517 |

The MSM majority baseline is 43.41%. The learned-CV VAMP dim5/dim10 values are
identical because only four nonconstant learned dimensions are available; the
implementation caps the effective kinetic rank.

Artifacts:

- `artifacts/paper_eval/eva5c_encoder_audit_causal_endpoint_quick`
- `artifacts/paper_eval/eva5c_encoder_audit_tmjepa_cv_quick`
- `artifacts/paper_eval/eva5c_encoder_audit_tmjepa_fixed_cv_quick`
- `artifacts/paper_eval/eva5c_encoder_audit_tmjepa_learned_cv_quick`

## Interpretation

### 1. The representation is not empty

Both the causal state and learned CVs have above-random balanced MSM accuracy,
and the full CV state reaches 43.80% balanced accuracy. The model is not
forecasting from a completely information-free latent.

### 2. Most useful state information is hand-engineered

Removing the four learned CVs improves MSM accuracy from 51.96% to 53.10% and
slightly improves lag-6 VAMP. The learned-only CV is much weaker than the fixed
CV. Therefore the final state-space quality is primarily supplied by explicit
physical collective variables, not by the learned encoder projection.

### 3. The causal latent is not a closed dynamics coordinate

The causal endpoint has negative lag-6 linear R2 and lower MSM accuracy than
the majority baseline. It contains nonlinear conformation information, but a
stable future transition cannot be recovered by a simple blocked linear
operator. The full CV also has negative linear R2. This is direct evidence that
the current representation is not approximately Markov-closed at the tested
resolution.

### 4. Current memory failure may be projection-induced

Residual temporal correlation after conditioning on an incomplete state can
look like physical Mori memory. Since the learned state does not add slow-mode
quality beyond the fixed CV, the failed auxiliary bath experiments cannot yet
distinguish true unresolved physical memory from information discarded by the
encoder.

## Answer

EVA5C preserves a moderate amount of dynamics information, but not enough to
serve as a high-quality learned protein-dynamics state.

The fixed 16D physical CVs preserve metastable-state information reasonably
well. The learned 4D CVs and 64D causal endpoint do not provide a superior slow
representation and are not linearly forecastable at the first non-overlapping
lag. The current encoder is therefore a real bottleneck for autonomous rollout
and identifiable Mori memory.

## Required order of work

### Stage 1: repair the learned dynamics state

- Remove the 16D fixed CV from model inputs, state, and training losses. Keep it
  only as a frozen external probe and matched baseline.
- Replace the single 4D projection with explicit slow and residual subspaces.
- Increase learned slow capacity conservatively, for example 8 dimensions,
  while controlling variance, covariance, and protein-identity leakage.
- Train the slow subspace with multi-lag VAMP/Koopman objectives and
  future-state prediction on non-overlapping windows.
- Keep MSM labels evaluation-only.
- Use a target/online consistency loss so encoder fine-tuning cannot rotate the
  learned coordinate faster than the dynamics model can follow.

Promotion requires all of:

- learned-only VAMP exceeds the fixed-CV baseline
- learned-only VAMP materially exceeds its current 0.290 lag-6 score
- MSM balanced accuracy exceeds the fixed-CV baseline, not only majority class
- blocked lag-6 linear R2 becomes positive and beats geometry statistics
- gains survive a full 50-trajectory protocol

### Stage 2: diagnose conditional non-Markovianity

After Stage 1, compare:

- current-state-only forecast
- current state plus ordered history
- history reversal and truncation controls
- residual autocorrelation conditioned on the complete learned state

Only a reproducible ordered-history advantage justifies a richer adaptive Mori
kernel. A large kernel norm without intervention advantage is not evidence of
memory.

### Stage 3: train autonomous rollout

Once the state passes sufficiency gates:

- train one-step and k-step closed-loop rollout jointly
- propagate the complete latent state, not reconstructed coordinates alone
- use scheduled rollout lengths and stop-gradient target trajectories
- evaluate error growth, equilibrium marginals, CK consistency, and implied
  timescales against MD

Starting autoregressive training before Stage 1 would mainly teach the model to
accumulate errors in an incomplete state space.

### Stage 4: external observable validation

Evaluate fixed-only, learned-only, and full states on held-out:

- contacts and contact order
- radius of gyration
- backbone torsions
- RMSD or aligned distance features
- MSM/VAMP and implied timescales

This separates physical information guaranteed by the fixed CVs from new
information learned by the encoder.

## Manuscript boundary

Supported now:

> EVA5C retains metastable information through its explicit physical CVs and
> obtains improved long-horizon endpoint energy skill with a bounded causal
> history closure.

Not supported now:

- the learned encoder discovers a superior slow coordinate
- the latent state is Markov sufficient
- autonomous long-time equilibrium or implied timescales are correct
- the auxiliary Mori bath captures identifiable physical memory
