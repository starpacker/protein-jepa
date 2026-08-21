# JEPA Improvement Plan

## Current Evidence

The current strongest positive result is not VAMP-CV. It is MSM state recovery
under stricter trajectory-heldout probing:

- `pure_jepa` checkpoint step 1,490,566: 0.596 mean accuracy vs 0.392 majority baseline, 26/30 proteins above baseline.
- `dynamics_aux` checkpoint step 40,000: 0.543 mean accuracy vs 0.392 majority baseline, 25/30 proteins above baseline.

The strict deeptime VAMP-CV run does not currently support a claim that
`dynamics_aux` is better than pure JEPA or strong geometry baselines.

## Likely Failure Modes

1. The evaluated latent is a single-frame spatial encoder output.

   `ProteinGeometryEncoder` performs residue-level geometric attention inside
   each frame. It has no temporal mixing. The predictor receives time embeddings
   and all visible/masked spatiotemporal tokens, but the encoder representation
   used by VAMP-CV is extracted frame-by-frame from the context encoder.

   Consequence: the latent can encode conformation and metastable state, but it
   is not explicitly optimized as a slow kinetic coordinate.

2. The current VAMP auxiliary loss is not aligned with strict deeptime VAMP-CV.

   The training loss computes VAMP-like statistics inside short clips
   (`T=8`, `lag=1`) after a fixed random projection. This is a noisy local
   objective and can be gamed without improving trajectory-level heldout VAMP.

3. The MSM auxiliary head weakens the self-supervised claim.

   `dynamics_aux` uses `msm_state_weight=0.01`. This makes high MSM probe scores
   harder to interpret as emergent self-supervised structure. It should not be
   mixed into the primary model unless the paper is explicitly semi-supervised.

4. The masking task may be too reconstructive and too instantaneous.

   The current target includes masked tokens from the same short clip. If many
   targets can be inferred from static local geometry or residue identity, the
   model can lower JEPA loss without learning slow transition structure.

5. Mean pooling can wash out local transition signals.

   Evaluation currently uses mean-pooled residue latents. Domain-level motions
   may require attention pooling, state tokens, or residue subsets instead of a
   uniform average over all valid residues.

## Recommended Fix Order

1. Do not use MSM state loss in the main claim model.

   Keep MSM labels for evaluation and optional appendix semi-supervised
   variants. The primary model should be pure JEPA or JEPA plus unsupervised
   kinetic regularization only.

2. Add a temporal context encoder or trajectory/state token.

   Add temporal self-attention after per-frame geometric encoding:

   - frame/residue tokens: `[B, T, L, D]`
   - spatial encoder per frame
   - temporal transformer over pooled frame tokens or over residue tracks
   - expose a `state_latent[t]` for VAMP/MSM evaluation

   This directly addresses the mismatch between the training predictor and the
   evaluated representation.

3. Replace the current VAMP auxiliary loss with an official deeptime-aligned
   offline or minibatch objective.

   Preferred first step: do not backprop deeptime itself. Train JEPA normally,
   then periodically evaluate deeptime VAMP-CV. Once the representation is
   stable, add a differentiable kinetic regularizer only if it improves that
   heldout metric.

   If adding a differentiable loss, use longer clips, larger effective sample
   count, and no random projection as the main signal. Use a learned low-rank
   projection or state token instead.

4. Make prediction more future-oriented.

   Add a JEPA target mode where context frames are early frames and target
   frames are future frames only. This forces the representation to model
   dynamics rather than same-frame masked geometry.

5. Evaluate multiple pooling/readout strategies.

   Compare:

   - mean residue pooling
   - learned attention pooling
   - CLS/state token
   - top moving residues or domain-aware pooling

6. Run equal-budget ablations.

   The current `pure_jepa` has ~1.49M steps while `dynamics_aux` is still early.
   Any paper table must compare equal data/step budgets.

## Experiments To Run Next

1. Equal-step pure JEPA vs dynamics variants at 40k, 100k, 250k, 500k.
2. JEPA + VAMP only: `msm_state_weight=0`, `vamp2_weight>0`.
3. Future-only JEPA masking: no same-frame target tokens.
4. Temporal encoder/state-token JEPA.
5. Pooling ablation for VAMP-CV and trajectory-heldout MSM probe.
6. Longer clips: `T=16/32`, larger lag, same total token budget if needed.

## Current Claim Boundary

Supported now:

- JEPA learns protein conformation/state features that transfer across heldout
  trajectories better than a majority MSM baseline.

Not supported yet:

- JEPA+VAMP significantly improves deeptime VAMP-CV over pure JEPA.
- The current latent supports simple linear rollout.
- The model is a full protein dynamics world model.
