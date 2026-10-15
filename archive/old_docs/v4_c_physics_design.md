# V4-C Physics-Constrained Fast/Slow World Model

Date: 2026-08-28

## Decision

V4-B1 is not promoted. Its frozen V4-A fast branch remains the current
physical-state MSE baseline, but its slow branch does not contain validated
slow kinetics.

At step 2000 the mean held-out slow Brier skill is `47.55%`, yet horizon
shuffle changes the score by only `-0.0014%`, shuffled history changes it by
only `0.18%`, and correct rollout depth is `-0.0026%` worse than the best fixed
depth. Meanwhile the mean second transition eigenvalue falls from `0.225` at
step 500 to `0.075` at step 2000. The transition therefore mixes almost
instantly to a nearly horizon-independent occupancy. The apparent Brier gain
is state classification plus mean occupancy prediction, not slow dynamics.

V4-C will repair the representation before adding stochastic innovations.

## Fast/slow separation

The branches should be separated, but strict pointwise Euclidean
orthogonality is not physically appropriate. The current held-out audit finds
a mean state CCA of about `0.682`, top canonical correlations of `0.85-0.89`,
increment CCA of about `0.663`, and about `21.5%` linearly predictable fast
variance from the slow latent. The overlap is excessive. The same audit also
finds that the current slow latent is not more persistent than the fast state.

The desired decomposition is

```text
x_t = D_s(z_t, protein, T) + r_t
```

where `D_s(z_t)` is the conditional slow/equilibrium component and `r_t` is a
zero-mean fast residual. The constraints are distributional:

```text
E[r_t | z_t, protein, T] = 0
L_cross = ||C_zz^(-1/2) C_zr C_rr^(-1/2)||_F^2
L_increment = ||C_dz,dhigh^white||_F^2
```

The covariance is estimated within protein/temperature groups when support
permits. `dhigh` is the high-pass component of the physical-state increment.
The model may still let the variance and relaxation rate of `r_t` depend on
`z_t`; removing that conditional coupling would discard real basin-dependent
fast dynamics.

The current decoder reconstructs the complete fast state from `z_slow`, which
directly rewards leakage. V4-C instead reconstructs low-pass states and slow
observables such as contacts, torsions, radius of gyration, and secondary
structure occupancy. The fast branch models only the residual around that
slow reconstruction.

## Energy-shaped latent

mdCATH provides a useful supervision signal that the current loader ignores:
each trajectory group contains per-frame all-atom `forces` in
`kcal/mol/Angstrom`, aligned with `coords`. The existing backbone atom indices
can gather N/CA/C/O forces without changing topology semantics.

The proposed deterministic relation

```text
dz/dt = alpha * grad(E)
```

has the wrong sign and omits mobility, thermal noise, and the coarse-graining
correction. A physically defensible overdamped latent model is

```text
dz = [-M(z) grad F(z | protein,T) + div M(z)] dt
     + sqrt(2 kB T M(z)) dW
```

Here `F` is a conditional free energy, not the instantaneous atomistic
potential energy. This distinction matters because mdCATH frames are separated
by about `1000 ps`: instantaneous force cannot be equated directly with the
finite-difference displacement across two stored frames.

V4-C uses force matching to shape geometry. With `z = q(R)`, an invariant
latent energy head predicts `F_theta(z, protein,T)` and its atom-space
conservative component is

```text
F_atom_pred = -d F_theta(q(R), protein,T) / dR
```

The robust, masked force loss compares the representable backbone component
with the real backbone force. A force-direction/cosine term is reported
separately from magnitude error. Net translation and rotation are projected
out. This makes the latent geometry sensitive to force directions without
claiming that a 16-dimensional latent can reproduce every high-frequency
atomistic force.

## Boltzmann equilibrium and detailed balance

Equilibrium is conditional:

```text
pi(z | protein,T) proportional to exp[-F(z | protein,T) / (kB T)]
```

Proteins and temperatures must never be pooled into one stationary
distribution. Replicas at the same protein and temperature provide the
appropriate occupancy samples.

The current history-conditioned row-stochastic matrix has no guaranteed
stationary distribution or reversibility. V4-C replaces it with a continuous-
time reversible generator for each protein/temperature condition. For
off-diagonal symmetric nonnegative flux `S_ij = S_ji` and stationary
probability `pi_i`:

```text
Q_ij = S_ij / pi_i                    i != j
Q_ii = -sum_j!=i Q_ij
P(tau) = exp(tau Q)
pi_i P_ij(tau) = pi_j P_ji(tau)
```

The short history infers the current slow state, but does not generate a new
transition matrix for every window. Otherwise the nominal eight-state process
is not Markovian and detailed balance is not well-defined. If history memory
is essential, it must become part of the explicit state.

## Training sequence

### C0: data and diagnostics

- Load and collate masked backbone forces for mdCATH only.
- Add train-only force normalization per atom type.
- Add CCA, residual predictability, autocorrelation, force, Boltzmann,
  detailed-balance, implied-timescale, and Chapman-Kolmogorov diagnostics.
- Keep the V4-A step-2000 fast branch frozen as an unchanged control.

### C1: representation only

- Encode slow state from frozen offline/global temporal features plus physical
  slow observables, rather than only the V4-A fast state.
- Decode a low-pass equilibrium component and define the fast residual.
- Optimize multi-lag variational objectives, slow-observable prediction,
  whitened residual separation, and conservative force matching.
- Do not fit a rollout matrix in this stage.

### C2: reversible kinetics

- Freeze the accepted C1 encoder and estimate `pi(protein,T)` and the
  reversible generator.
- Compose all requested horizons through `matrix_exp(tau * Q)`.
- Match base and rollout transitions and enforce Boltzmann occupancy.
- Use empirical implied timescales as stop-gradient targets; do not reward an
  arbitrary eigenvalue floor that can be gamed by an identity matrix.

### C3: restrained joint tuning

- Jointly tune the slow encoder, free-energy head, mobility, and generator at
  a lower learning rate.
- Add stochastic innovations only after the deterministic/reversible gates
  pass.

## Promotion gates

V4-C is not promoted based on Brier or MSE alone. It must satisfy all of the
following on held-out proteins/trajectories:

- slow implied timescales and autocorrelation exceed the fast residual;
- horizon shuffle advantage is above `2%` and correct depth beats every fixed
  depth by at least `1%`;
- nontrivial held-out VAMP/VAMP-E gain over persistence and state-frequency
  baselines;
- low Chapman-Kolmogorov and detailed-balance residuals;
- predicted stationary occupancy agrees with per-protein-temperature held-out
  occupancy and has better likelihood than a train-frequency baseline;
- force direction and force magnitude beat zero-force and mean-force controls;
- the frozen V4-A fast-state result remains unchanged.

Until these gates pass, longer training and a Student-t rollout head are not
valid next steps.
