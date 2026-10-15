# V5 Contextual Adaptive Dynamics

## Objective

Predict future protein motion from an observed trajectory history without
assuming that a single conformation has a unique Markov state. V5 defines the
state as a causal function of the observed window and trains state inference
and continuous-time dynamics jointly.

## Model

For a context window `x[0:t]`, the online geometry and causal temporal encoders
produce frame states. A recurrent history encoder maps these states to memory
`h_t`. The model then generates an overcomplete set of context-specific state
slots and activation gates:

```text
h_t = history_encoder(x[0:t])
slots_t, gates_t = slot_generator(h_t)
q_t = assignment(current_frame, h_t, slots_t, gates_t)
```

Slot indices are computational capacity, not global protein-state labels. The
same conformation may receive a different assignment under a different
history, and different proteins may attach different meanings to the same slot
index.

At each integration step, symmetric fluxes and local stationary probabilities
define a valid reversible rate matrix. Both are conditioned on the current
posterior and memory. Posterior propagation updates memory before constructing
the next generator:

```text
Q_n = generator(q_n, h_n, slots_t, gates_t)
q_(n+1) = q_n exp(delta_t * Q_n)
h_(n+1) = memory_update(h_n, q_(n+1) @ slots_t)
```

The integration depth is determined from physical `horizon_time_ps`, not frame
indices. Detailed balance applies to every local generator; the full rollout
is nonstationary because the inferred state and memory evolve.

## Joint Training

Initialize the geometry and temporal encoders from the Phase-1 multi-lag JEPA
checkpoint, then jointly optimize:

- future contextual-assignment prediction;
- future feature prediction in the adaptive slot basis;
- future physical-action prediction (contacts, torsions, radius of gyration,
  aligned RMSD, and local distance changes);
- current and future state reconstruction;
- soft transition likelihood;
- same-context horizon-difference prediction;
- slot diversity and adaptive-capacity regularization.

The EMA target encoder supplies future representation targets. No MSM labels,
fixed sequence segments, global six-state vocabulary, or frozen representation
stage are used.

## Experiment Gates

The first run is a representation-and-identifiability gate, not a long scaling
run. A checkpoint advances only if all of the following hold on validation:

1. positive forecast skill over persistence at every horizon;
2. positive correct-horizon advantage at every horizon;
3. positive same-protein history advantage when the endpoint is held fixed;
4. nonzero target and predicted horizon differences;
5. context slots vary with history and do not collapse to one effective slot;
6. geometry encoder, temporal encoder, slots, and generator all receive finite
   nonzero gradients;
7. no material train/validation divergence by protein and temperature.

After this gate, evaluation must compare against corrected TICA/MSM, VAMPNet,
and SPIB implementations on identical splits, physical lags, inputs, and
forecast targets.

## Formal 2K Run Target

The formal run evaluates checkpoints at steps `250/500/750/1000/1250/1500/1750/2000`.
A checkpoint is promoted only when every evaluated horizon has:

- positive shared-feature skill over persistence;
- positive physical history-order advantage with the endpoint fixed;
- positive physical horizon-shuffle advantage;
- positive correct-horizon versus best-fixed-horizon skill;
- finite local detailed-balance residual below `1e-6`.

The adaptive basis must also remain noncollapsed: mean effective slots must be
greater than `2` and less than `14` out of the maximum capacity of `16`. A high
zero-motion skill alone is not a promotion signal because a history-blind mean
predictor can achieve it.

Run the held-out controls with:

```bash
python scripts/evaluate_v5_contextual_dynamics.py \
  --config configs/phase5_v5_contextual_adaptive_2k.yaml \
  --checkpoint artifacts/phase5_v5_contextual_adaptive_2k/checkpoint_step250.pt \
  --manifest /data/mdcath.validation.jsonl \
  --output artifacts/paper_eval/v5_step250_controls.json \
  --max-batches 50 --batch-size 1 --device cuda:0
```
