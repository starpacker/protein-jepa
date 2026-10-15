# V4-D2 Reversible Kinetics

Date: 2026-08-29

D2 continues the D1 scratch run from step 750. It is not initialized from any
pre-V4-D model. The dynamic-only frame encoder, slow decoder, assignment head,
free-energy head, and mobility head are frozen. Only the equilibrium/temperature
conditioner, stationary-state energies, and symmetric flux rates are trainable.

For each protein condition, symmetric nonnegative fluxes and stationary
probabilities define a reversible continuous-time generator. Every horizon is
computed as `matrix_exp(tau * Q)`, preserving row normalization, semigroup
composition, and detailed balance by construction.

Training combines:

- stationary occupancy matching;
- raw-frame-16 base transition Brier loss;
- long-horizon rollout Brier and soft-state cross entropy;
- correct-depth ranking against every wrong fixed depth;
- same-context horizon-difference prediction.

Checkpoints are evaluated every 250 steps. Promotion requires positive Brier
skill at every horizon, positive horizon-shuffle advantage, correct depth to
beat the best fixed depth, non-collapsed `lambda2`, and unchanged D1 switching
and identity-leakage diagnostics.
