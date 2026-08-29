# Phase 2 Diagnosis and Execution Plan

Date: 2026-08-24

## 1. Executive Judgment

The Phase 1 champion is a useful representation-learning baseline, but the
current evidence does not yet support calling it a protein dynamics world
model. The central problem is not that all three headline numbers are merely
too low. Several training and evaluation definitions are misaligned with the
desired claim:

- the positive lag-1 linear result is measured on windows sharing 16 of 24
  input frames, while non-overlapping lag-5 and lag-20 R2 are negative;
- the configured lag-1, lag-2, and lag-4 JEPA losses receive no valid pairs
  under the current future mask and four-frame gap;
- quick VAMP and linear screening uses the first ten trajectories, all from
  one protein (`1ayeA02`);
- physical time is not matched between training and evaluation, and
  temperature is not provided to the model at all;
- a deterministic Smooth L1 future predictor cannot represent the multimodal
  stochastic future of a protein;
- the encoder discards signed torsions, residue-frame orientation, side-chain
  state, solvent, and momenta.

Phase 2 should therefore begin with an evaluation and objective repair, not a
larger model or more naively mixed data.

## 2. Hard Findings From The Repository

### 2.1 The main lag-1 linear number is dominated by overlapping context

The aligned protocol uses 24-frame windows and advances the next window by
eight frames. Consecutive inputs share 16 frames, an overlap ratio of 0.667.
The champion has:

| Horizon in extracted-window units | Linear R2 |
| --- | ---: |
| 1, overlapping | 0.5381 |
| 5, non-overlapping | -0.2185 |
| 20, non-overlapping | -0.5074 |

The model is therefore good at local interpolation or smoothing, but the
current result does not demonstrate durable rollout. Evaluation also forces
`frame_stride=1`, whereas champion training uses `frame_stride=2`. A 24-frame
training clip spans about 47 ns in mdCATH, while the so-called aligned
evaluation clip spans about 23 ns.

Required correction: report physical horizons, expose evaluation frame stride,
and make the primary forecast benchmark non-overlapping.

### 2.2 Three of five configured multi-lag heads receive zero valid pairs

With `future_gap_frames=4`, context ends strictly before the four-frame gap and
the target starts after it. The minimum context-to-target index difference is
five. Consequently, pair masks for lags 1, 2, and 4 are always empty. A
5,000-mask simulation produced:

| Configured lag | Batches with at least one valid pair |
| --- | ---: |
| 1 | 0 / 5000 |
| 2 | 0 / 5000 |
| 4 | 0 / 5000 |
| 8 | 5000 / 5000 |
| 12 | 5000 / 5000 |

The current champion is thus not actually trained with five lag scales.

Required correction: sample explicit `(context endpoint, target, physical
tau)` pairs and assert/log nonzero pair counts independently for every horizon.

### 2.3 Quick model selection is effectively single-protein selection

`_first_n_trajectories` selects records, not proteins. The first ten validation
records are temperatures/replicas of `1ayeA02`. The reported quick VAMP and
linear means contain 30 correlated trajectory-fold rows, but only one protein.
This is insufficient for selecting a cross-protein world model and repeated
ablation selection can overfit this screening set.

Required correction: stratify by protein first, then trajectory, temperature,
and replica. Bootstrap and compare candidates at the protein level. Keep a
final protein test set completely untouched by checkpoint and architecture
selection.

### 2.4 Existing baselines are stronger than the headline table suggests

The champion quick protocol includes only `geometry_stats`. A separate strict
run of the closely related 20k State-JEPA reports:

| Feature | VAMP-CV lag1 dim5 score |
| --- | ---: |
| State-JEPA | 0.5578 |
| Random encoder | 0.5421 |
| Geometry statistics | 0.5304 |
| Centered CA coordinates | 0.6229 |

This means the trained model's gain over the random network is small in that
run, and raw centered coordinates retain more kinetic signal. The strict run is
not the multi-lag champion, so the champion still needs this exact comparison.

The current VAMP normalization is also invalid: `cv_score / dim` can exceed
one (for example 1.075 at lag 20, dim 10). Raw VAMP-2 and a verified kinetic
normalization should replace the current headline normalized score.

### 2.5 The MSM number is a state probe, not a dynamics forecast

The MSM probe:

- labels a clip by the majority state over all 24 frames;
- averages temporal states over the same full clip;
- chooses ridge regularization by maximizing accuracy on the validation
  trajectories themselves.

The trajectory split is useful, but validation-selected regularization inflates
the score. The metric tests window-level state separability and should not be
treated as future-state accuracy.

Required correction: use nested train-only hyperparameter selection, report
balanced accuracy and macro-F1, and add a separate endpoint-to-future-state
transition metric.

### 2.6 Physical conditioning is incomplete

mdCATH contains 320--450 K simulations sampled at about 1 ns per stored frame;
ATLAS is mainly 300 K and about 10 ps per stored frame. The dataset returns
`time_ps`, but drops `temperature_k`, dataset identity, force-field identity,
and other thermodynamic metadata. Therefore the same latent and physical lag
can correspond to different transition kernels without the model knowing why.

The ATLAS experiment also gives 100 ATLAS proteins about 373,500 clips versus
791,458 clips from 603 mdCATH proteins. An ATLAS protein contributes about
three times as many clips as a typical mdCATH protein. Clip-uniform shuffling
therefore changes both time-scale and protein distributions.

Required correction: use hierarchical sampling and condition the transition
model on at least temperature, physical tau, dataset/force-field domain, and
sequence/static structure context.

### 2.7 The representation omits important dynamical geometry

The spatial encoder uses six intra-residue distances and pairwise CA distance
biases. Most intra-residue distances are nearly rigid, pair distances lose
chirality, and distances above the RBF range carry little geometric detail.
Signed phi/psi/omega changes, local residue-frame orientations, side-chain
rotamers, and solvent are absent. Mean residue pooling further suppresses
localized hinge and loop motions.

The EMA target temporal encoder is bidirectional over the full target clip.
Thus a target state at time `t` can contain information from frames after `t`.
That is acceptable for a smoothing representation, but not as the target of a
causal forecaster without an explicit distinction.

Required correction: build a causal endpoint state from orientation-aware
backbone features and preserve several local/domain motion tokens in addition
to one global token.

### 2.8 Training longer currently makes the useful representation worse

The configured run is roughly a multi-epoch, million-step run, yet the champion
is an early 20k checkpoint. At 50k, quick VAMP falls to 0.403, MSM to 0.525,
and linear lag-1 R2 to approximately zero even though JEPA training losses have
continued to decrease. This is direct evidence that the optimized surrogate is
misaligned with the selected dynamics metrics.

Do not scale training duration until this anti-correlation is fixed.

## 3. Define The Achievable Model Before Training It

"Perfect" coordinate prediction is not a well-posed target for finite-
temperature MD. Given only backbone coordinates, the future is stochastic
because velocities, solvent, side chains, and thermostat noise are hidden. The
correct target is a calibrated conditional distribution:

```text
p(z(t + tau), x(t + tau) | history <= t, sequence, T, tau, simulation domain)
```

Phase 2 should separate three claims:

1. **State representation:** preserve metastable states, slow modes, and
   committor-relevant information on unseen proteins.
2. **Latent dynamics:** predict a calibrated distribution over future latent
   states and remain stable under rollout.
3. **Structural generation:** decode physically plausible backbone or all-atom
   ensembles. This is a later stage and needs richer data than the current
   backbone-only input.

## 4. Recommended Model Direction

### 4.1 State encoder

Use a hierarchical state with separate static and dynamic content:

- `c`: sequence/fold context that is constant for a trajectory;
- `z_global(t)`: global slow state;
- `z_local(t, k)`: a small set of domain/loop motion tokens;
- optional residue latents for structural decoding.

Replace or augment distance-only attention with an established
orientation-aware protein structure encoder. Minimum input features should
include signed torsions as sin/cos, local backbone frames, sequence separation,
covalent edges, spatial KNN/contact edges, and missing-atom masks. A residual
path from mean pooling can stabilize learned multi-token pooling.

The temporal encoder used for forecasting must be causal. A separate
bidirectional encoder may still be retained for offline state estimation, but
its metrics and role must be labeled separately.

### 4.2 Continuous-time stochastic transition

Replace independent frame-index MLP heads with one shared physical-time
operator:

```text
q(z(t + tau) | z(<=t), c, log(tau), temperature, domain)
```

Start with a Gaussian or mixture-density latent transition to establish
calibration. If clearly multimodal, move to latent flow matching, diffusion, or
an SDE. Train tau log-uniformly within dataset-supported ranges. ATLAS should
supervise short horizons and mdCATH long horizons; the shared encoder can be
common while horizon/domain adapters remain small.

Add multi-step consistency only after one-step distributional prediction works.
Semigroup consistency on a deterministic point predictor is too restrictive
for a hidden-variable stochastic process.

### 4.3 Learning objectives

Use a balanced objective rather than forcing all behavior into one vector:

- future latent JEPA loss with causal targets;
- proper probabilistic transition loss (NLL or diffusion/flow loss);
- variance, covariance, and effective-rank controls on dynamic latents;
- a carefully implemented batch/trajectory VAMP-E or reversible Koopman
  auxiliary on a low-dimensional slow head;
- optional physical observable heads for torsion change, contacts, RMSD/Q, and
  transition events;
- multi-step rollout consistency at sampled physical horizons.

MSM labels should remain evaluation-only for the primary self-supervised claim.

## 5. Data Strategy

1. Pretrain spatial geometry on broad static structures, then train dynamics on
   MD. The current 603 mdCATH training domains are too narrow to learn both a
   universal protein geometry encoder and transferable kinetics from scratch.
2. Sample hierarchically: dataset -> protein -> trajectory -> time. Control the
   probability of each dataset explicitly instead of inheriting clip counts.
3. Build target pairs by physical tau, not saved-frame index. Maintain separate
   supported horizon buckets for ATLAS and mdCATH.
4. Pass temperature and simulation-domain metadata through dataset, collate,
   model, checkpoint, and evaluation.
5. Split by sequence cluster and structural homology, not only exact domain ID.
   Keep unseen proteins, unseen trajectories, and an optional unseen
   temperature/domain test split.
6. Track the effective number of proteins, trajectories, and non-overlapping
   time blocks per batch. Raw clip count greatly overstates independent data.

## 6. Phase 2 Experiment Matrix

All candidate comparisons should use identical sampled examples, optimizer
steps, three seeds, and a protein-stratified validation set.

| ID | Change | Purpose | Promotion condition |
| --- | --- | --- | --- |
| E0 | Re-evaluate champion with matched frame stride, non-overlap, strong baselines | Establish honest baseline | No promotion; freeze numbers |
| E1 | Fix VAMP normalization and MSM nested CV | Remove metric bias | Unit tests and bounded/verified outputs |
| E2 | Stratify quick eval by protein | Stop single-protein selection | At least 10 proteins in quick, 30+ in full |
| T0 | Fix physical-tau pair sampler and per-lag logging | Make all horizons train | Every requested bucket has nonzero pairs |
| T1 | Hierarchical dataset/protein sampler | Remove ATLAS clip dominance | Stable per-source and per-protein exposure |
| T2 | Add temperature/domain conditioning | Resolve incompatible kernels | Paired gain over T1 on both datasets |
| A0 | Add torsion/orientation-aware spatial features | Beat raw geometry baselines | Paired VAMP and state-probe gain |
| A1 | Causal temporal state plus static/dynamic split | Forecastable endpoint state | Positive non-overlap medium-horizon skill |
| G0 | Probabilistic latent transition | Model stochastic futures | Better NLL/calibration than Gaussian persistence |
| G1 | Multi-step stochastic rollout | World-model behavior | Stable free energies, transitions, and ITS |

Run E0--E2 before new training. Then run T0, T1, and T2 as small controlled
ablations. A0/A1 should be compared after data semantics are fixed. G0/G1 are
the first experiments that can legitimately support a world-model claim.

## 7. Evaluation And Promotion Gates

### Representation gates

- paired protein-level bootstrap confidence interval above `centered_ca`,
  contact/torsion features, random encoder, and static pretrained encoder;
- VAMP/VAMP-E at several physical taus with verified score semantics;
- implied-timescale stability and Chapman-Kolmogorov consistency;
- nested trajectory-heldout balanced MSM accuracy and macro-F1;
- direction/committor probes on FastFolding or an equivalent transition set.

### Forecast gates

- non-overlapping one-step and multi-step metrics in physical time;
- transition-state and basin-arrival probabilities, not only latent R2;
- calibration, NLL/CRPS, and coverage for stochastic predictions;
- rollout free-energy-surface distance, transition-matrix error, state
  population error, and implied-timescale error;
- stability across at least three training seeds and across proteins.

### Promotion rule

Do not retain the Phase 1 rule of improving any two of three lag-1 metrics.
Require all of the following:

1. no statistically credible regression in state representation;
2. positive skill over persistence at a non-overlapping medium horizon;
3. improved distributional rollout or transition metric;
4. improvement on a protein-stratified set with protein-level confidence
   intervals;
5. no access to the final test set during selection.

## 8. Immediate Stop/Go Decisions

- **Stop** adding auxiliary heads to the current objective without fixing pair
  semantics and evaluation.
- **Stop** using mixed clip-uniform mdCATH+ATLAS training.
- **Stop** treating overlapping lag-1 R2 or the current MSM probe as world-model
  evidence.
- **Keep** the 20k multi-lag checkpoint as a frozen Phase 1 diagnostic baseline.
- **Go** first on E0--E2 and T0; these have the highest information gain and the
  lowest compute cost.
- **Go** next on temperature-conditioned continuous-time prediction and an
  orientation-aware causal state encoder.

The near-term success criterion is not a perfect world model. It is an honest,
reproducible model that beats strong geometric baselines on unseen proteins and
has positive non-overlapping, physically timed, distributional forecast skill.
