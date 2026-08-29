# V4-B Fast/Slow Stochastic World Model

Date: 2026-08-28

## State decomposition

V4-B separates two statistically different processes.

- The fast physical state is the V4-A closed-local branch. V4-A step 2000 is
  frozen as the current mean-prediction champion (`10.10%` held-out skill).
- The slow kinetic state is learned from trajectory-centered per-frame physical
  states. It must carry metastable and horizon information without relying on
  static protein identity.

The first V4-B experiment is B1, a deterministic slow-state gate. Low-rank
Student-t innovations are deferred until B1 demonstrates a non-collapsed,
horizon-sensitive kinetic representation.

## B1 slow kinetic model

Each frame's four-by-twelve physical state is centered by the observed context
trajectory equilibrium and scaled with train-only delta scales. An MLP maps it
to a 16-dimensional continuous latent and eight soft metastable assignments.
A decoder reconstructs centered physical states to retain geometric meaning.

A GRU summarizes the context latent trajectory and predicts one context-
conditioned, row-stochastic T16 matrix. The same matrix is composed 3/4/6/8/12
times for horizons 48/64/96/128/192. An identity-biased initialization gives a
stable neutral kinetic prior.

Training combines multi-lag VAMP-2 at sampled lags 4/8/16, occupancy balance,
moderate assignment sharpening, reconstruction, internal T16 transition, and
long rollout transition losses. Transition targets are stop-gradient views;
VAMP and reconstruction train the representation itself.

## B1 gate

Every 500 steps the held-out protocol reports:

- VAMP loss, assignment entropy and occupancy balance;
- T16 and long-rollout Brier skill over soft metastable assignments;
- second transition eigenvalue as a kinetic-timescale diagnostic;
- shuffled history, shuffled horizon, and all fixed rollout-depth controls;
- unchanged V4-A fast-state metrics to verify exact preservation.

B1 is promoted to low-rank Student-t V4-B2 only if assignments are
non-collapsed, base and rollout skill are positive, correct depth beats the
best fixed depth, and the frozen fast branch retains its V4-A result.
