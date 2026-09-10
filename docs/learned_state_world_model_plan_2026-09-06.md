# Learned-State Protein Dynamics Plan

Date: 2026-09-06

## Decision

The next model must not use the 16 handcrafted backbone collective variables
as encoder inputs, dynamical state, conditioning variables, or training
targets. They remain useful only as frozen external probes and as a matched
baseline after training.

This changes the scientific question from

> Can a structured dynamics model forecast a prescribed physical CV space?

to

> Can a coordinate-based JEPA discover a compact, sufficient dynamical state,
> and can thermodynamically constrained dynamics propagate that state?

EVA5C remains the fixed-CV control. It must not be presented as evidence that
the encoder independently discovered protein state.

The framework keeps the V-JEPA training contract throughout development:

- the online geometry encoder, online learned-state projectors, causal encoder,
  and predictor are optimized together by backpropagation on every training
  step
- the target geometry encoder and target state projectors receive no gradient
  and are updated only by EMA from their online counterparts
- representation evaluation may freeze a checkpoint, but no training stage may
  silently set the online encoder or learned projection learning rate to zero

Here, "predictor" includes the learned transition/dynamics module once the
thermodynamic transition is attached. Encoder and predictor remain jointly
trainable; physics does not replace the V-JEPA objective.

## Why the current learned CV is weak

The problem is structural, not just insufficient training time.

1. The V3 state is `[fixed_cv_16, learned_cv_4]`. The dynamics and endpoint
   losses can improve through the fixed branch without requiring the learned
   branch to become sufficient.
2. The learned head is only `LayerNorm -> Linear(64, 4) -> Tanh`. Four bounded
   scalars are too restrictive for a shared state across diverse proteins and
   make scale-dependent stochastic dynamics difficult to identify.
3. U250 trained the context and temporal encoders while freezing the learned
   projection. The online encoder therefore had to move through a fixed narrow
   readout whose target projection was also effectively fixed.
4. Learned-CV variance, covariance, and VAMP terms each had weight `0.02`,
   while path, innovation, and endpoint losses dominated the optimization.
5. VAMP was computed after subtracting each short clip's mean. This removes
   static protein identity, but also removes metastable offsets that remain
   nearly constant within a clip. It consequently rewards local smooth
   fluctuations rather than necessarily discovering transition coordinates.
6. Training VAMP lags were 1/2/4 adjacent frames. The held-out failure occurs
   at the first non-overlapping lag, so the training objective and sufficiency
   test were not aligned.
7. The temporal state receives temperature and dataset identity. These are
   valid transition conditions but are shortcuts for a conformation state.
8. The shifted future windows overlap the context for all current horizons.
   This is valid for endpoint forecasting, but it is a weak protocol for
   demonstrating that a self-contained state was learned.

The observed result is consistent with these choices: the learned 4D readout
has lag-6 VAMP `0.2902`, while the fixed 16D CV reaches `0.3834`, and adding the
learned readout slightly lowers MSM accuracy.

## Target representation

Use raw backbone coordinates and sequence/topology only:

```text
X_t --SE(3)-invariant geometry encoder--> e_t
                                      |--> s_t: slow state, 8D
                                      `--> r_t: residual state, 16D

(s,r)_{t-L:t} --causal temporal encoder--> h_t: history state

(s_t, r_t, h_t, sequence condition, T, tau)
    --latent predictor/dynamics--> distribution of (s,r)_{t+tau}
```

The complete learned conformation state is `z_t = [s_t, r_t]`.

- `s_t` is optimized to retain slow, metastable information.
- `r_t` retains structural information that is predictive but not slow.
- `h_t` contains ordered history and is not conflated with instantaneous
  conformation. It becomes the candidate Mori state only after `z_t` passes
  sufficiency tests.
- Protein sequence/topology and temperature condition the transition but are
  not concatenated into `z_t`.

Start with 8 slow and 16 residual dimensions. Run 8/16 and 16/16 dimensional
ablations before fixing latent size for the paper. Replace `Tanh` with an MLP
projection followed by a stable whitening/gauge layer; a bounded arbitrary
coordinate is not a physically meaningful gauge.

## Representation training objective

Train the representation before fitting a new Mori model:

```text
L_rep = L_future_JEPA
      + lambda_vamp * L_multilag_VAMP
      + lambda_koopman * L_semigroup
      + lambda_geom * L_masked_geometry
      + lambda_var * L_variance
      + lambda_cov * L_covariance
      + lambda_split * L_slow_residual_decorrelation
```

### Future JEPA

Predict an EMA target state at physical lags 1/2/4/8/16 ns from causal past
only. The target is the future endpoint representation, not the aggregate of
an overlapping future clip. Use all horizons for each sampled context so that
horizon consistency is observable in every update.

The online geometry encoder, causal encoder, slow/residual projectors, and
future predictor are jointly trainable. Their target encoder/projector copies
are EMA updated and receive no gradient. Use a predictor rather than directly
forcing `z_t == z_{t+tau}`, which would reward persistence and erase dynamics.

### Multi-lag variational dynamics

Compute reversible VAMP/TICA statistics from long same-trajectory blocks or a
trajectory-keyed covariance queue, not independently centered 18-frame clips.
Average normalized scores per trajectory/protein so a long trajectory cannot
dominate. Include at least one lag matching the non-overlapping audit.

Do not remove each short clip's metastable offset. Control identity leakage by
keeping static sequence information in a separate condition and by measuring
cross-covariance between dynamic state changes and static condition.

Learn lag-specific Koopman operators and penalize
`K(tau_1 + tau_2) - K(tau_1) K(tau_2)`. This makes Markov closure a training
property rather than an after-the-fact hope.

### Generic structural sufficiency

Decode masked, SE(3)-invariant geometry from the full state: selected backbone
pair distances, local-frame coordinates, and contact probabilities. These are
generic descriptions of the molecular configuration, not a prescribed low-
dimensional CV. The decoder prevents a pure slow objective from discarding
structural degrees of freedom needed by a world model.

MSM labels and the existing 16 physical CVs must not enter this objective.

## Dynamics integration

Only after the representation passes its gates should the thermodynamic model
be attached.

1. Initialize dynamics in learned `z` space and continue joint
   encoder-predictor training. Use a smaller encoder learning rate, but never
   zero, together with EMA-target consistency, whitening statistics, and a
   coordinate trust-region penalty. These control a moving latent gauge without
   violating V-JEPA training.
2. Use state-dependent positive-semidefinite diffusion and the corresponding
   Ito divergence correction. A latent free-energy gradient alone is not
   invariant to nonlinear coordinate changes.
3. Preserve the JEPA future-prediction loss during thermodynamic training so
   the encoder cannot optimize only for an easy physical likelihood while the
   predictor compensates for representation degradation.
4. Diagnose memory by comparing `z_t` against ordered `z` history, reversed
   history, and truncated history. Add a Mori bath only when ordered history
   has reproducible held-out intervention advantage.
5. Train closed-loop multi-step rollout only after one-step learned-state
   closure is positive. Decode rollouts to geometry/observables for scoring;
   latent-space error alone can be manipulated by representation scale.

Physics is retained as an inductive bias in transition laws, stationarity,
temperature dependence, fluctuation-dissipation consistency, and detailed
balance diagnostics. It no longer defines the state by hand.

## Qualification gates

Do not promote the representation based on training VAMP or latent loss.
Freeze it and run trajectory-heldout probes.

Required before dynamics training:

- learned-only lag-6 VAMP dim-5 exceeds the fixed-CV `0.3834` baseline
- learned full-state lag-6 VAMP dim-10 exceeds the fixed-CV `0.6531` baseline
- MSM balanced accuracy exceeds `43.68%` and macro-F1 exceeds `38.17%`
- blocked lag-6 linear prediction R2 is positive and above geometry-statistics
  R2 (`-0.184`)
- no collapsed dimensions; effective state rank and per-trajectory variance
  remain stable on validation trajectories
- masked distance/contact decoding improves over sequence-only and
  persistence baselines
- results reproduce on the full 50-trajectory protocol and at least three
  seeds

Required before calling the result a world model:

- decoded one-step and multi-step predictions beat persistence at every
  reported horizon
- autonomous rollout has bounded error growth
- generated marginal structural observables match held-out MD
- Chapman-Kolmogorov consistency and implied timescales are correct within
  confidence intervals
- gains survive held-out proteins, not only held-out windows from known
  proteins

## Experiment order

1. **LS0 diagnostic implementation:** jointly trained learned-only encoder and
   predictor, EMA endpoint target, long-block VAMP statistics, and a frozen
   checkpoint evaluator.
2. **LS1 representation pilots:** 8/16 and 16/16 state dimensions, three short
   seeds, no thermodynamic rollout.
3. **LS2 full representation run:** only the pilot that passes all quick gates.
4. **LS3 learned-state dynamics:** initialize a new latent thermodynamic
   predictor and train it jointly with the online encoder at separate nonzero
   learning rates; do not reuse fixed-CV potential/diffusion weights whose
   coordinate meaning is incompatible.
5. **LS4 memory and rollout:** conditional memory audit followed by scheduled
   autoregressive rollout.

This order may temporarily reduce the old endpoint metric because that metric
was measured in the handcrafted CV space. That is acceptable. The decisive
result is whether a learned-only state predicts held-out dynamics and decodes
physical observables without receiving those observables as its definition.
