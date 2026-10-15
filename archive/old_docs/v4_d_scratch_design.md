# V4-D Scratch Dynamic-State Gate

Date: 2026-08-29

V4-D starts with no checkpoint and does not use learned offline features in
the slow state. Every frame input is a trajectory-centered, fixed low-pass
physical state. Static equilibrium structure and temperature may condition the
reversible generator, but cannot affect state assignment.

The first D1 run is a 1000-step representation gate. It uses six states and
computes VAMP whitening independently for every trajectory. In addition to
global balance, each trajectory must occupy multiple states. Assignment
changes must correlate with low-pass physical changes, and the continuous
latent must retain temporal variance. These constraints directly reject the
globally balanced, one-static-state-per-protein solution found by V4-C1.

Raw instantaneous force matching is disabled because C1 force cosine remained
at zero. A future physics target must use coarse or time-averaged projected
forces. D1 does retain latent free-energy drift shaping, but does not fit the
generator. If D1 passes switching, within-trajectory VAMP, and identity-shuffle
controls, reversible kinetics will be enabled in the same from-scratch run.

Checkpoints are evaluated every 250 steps. Training is stopped early if
trajectory balance falls while switching approaches zero, or if state
assignment becomes predictable from protein identity rather than dynamics.
