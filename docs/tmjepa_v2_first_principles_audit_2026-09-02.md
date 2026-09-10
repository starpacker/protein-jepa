# TM-JEPA V2 First-Principles Audit

**Date:** 2026-09-02
**Scope:** fixed-CV overdamped and rank-4 Mori models

## 1. Continuous-Time Law

The implemented augmented state is `X=(q,a)` with dimensionless potential

$$
\mathcal U(q,a;c,T)=\beta F(q;c,T)+\frac12 a^T a.
$$

For each fixed protein condition and temperature, the kinetic tensors are
independent of the instantaneous state:

$$
S=\begin{bmatrix}D&0\\0&\Lambda\end{bmatrix},\qquad
A=\begin{bmatrix}0&-C\\C^T&0\end{bmatrix}.
$$

The SDE is

$$
dX=-(S+A)\nabla\mathcal U\,dt+\sqrt{2S}\,dW.
$$

Because `S` is symmetric PSD, `A` is antisymmetric, and both are constant in
`X`, its continuous Fokker-Planck operator preserves
`exp(-mathcal U)`. There is no missing Ito divergence in the current model.
If a future version makes `D` or `C` state dependent, it must add
`div(S+A)` to the Ito drift or use an equivalent geometric construction.

FDT does not require the correlated diffusion basis and Mori coupling basis
to coincide. V1 imposed this extra restriction. V2 uses independent
orthonormal bases: the first describes correlated white fluctuation, while
the second defines the memory kernel

$$
K(t)=C\exp(-\Lambda t)C^T.
$$

## 2. Defects Found in V1

### 2.1 The auxiliary filter was not prefix causal

Every history token was centered by the final context CV. Intermediate
posterior states therefore depended on a later observation. Those states were
then used to compute one-step innovations, so the reported whitening could be
optimistically biased.

V2 centers on the first valid observation, uses only backward finite
differences, and includes the observed physical time increment. A prefix
invariance test now verifies that appending future frames cannot change an
earlier posterior.

### 2.2 Euler-Maruyama was too coarse for the learned bath

V1 used a 1000 ps Euler step and learned a mean relaxation time near 1073 ps.
For the scalar OU process, Euler has stationary variance

$$
\operatorname{Var}_{EM}(a)=\frac{1}{1-\Lambda\Delta t/2},
$$

which is about 1.88 at `Lambda Delta t = 0.93`, rather than the FDT target 1.
This also makes direct and two-half-step rollouts different for numerical,
not physical, reasons.

V2 uses additive-noise stochastic Heun with a maximum 250 ps step. The
observation scale remains a separate 1000 ps quantity; numerical precision no
longer changes the observer's units. A Monte Carlo regression test checks
that the auxiliary OU variance remains close to one.

### 2.3 The observer and generator did not define one state-space model

V1 constrained only the endpoint posterior toward `N(0,I)`. It did not require
successive inferred auxiliaries to follow the learned Mori transition.
Consequently the GRU could represent arbitrary history while the generator
used a different dynamical meaning for `a`.

V2 adds a transition-prior KL. Over an observed interval `Delta t`, with the
force held locally constant, the auxiliary prior is

$$
\begin{aligned}
m^-_{t+1} &= e^{-\Lambda\Delta t}m_t
  +\Lambda^{-1}(I-e^{-\Lambda\Delta t})(-C^T\nabla\beta F),\\
V^-_{t+1} &= e^{-2\Lambda\Delta t}V_t + I-e^{-2\Lambda\Delta t}.
\end{aligned}
$$

Training penalizes `KL[q(a_{t+1}|q_{<=t+1}) || p(a_{t+1}|a_t,q_t)]` in
addition to a low-weight equilibrium marginal KL.

### 2.4 Innovation moments were not a proper path score

Matching pooled innovation mean and covariance is not equivalent to maximum
path likelihood. V2 adds the local marginal transition covariance

$$
\Sigma_q=2D\Delta t+\Delta t^2 C V_a C^T
$$

and optimizes the Gaussian negative log likelihood per CV dimension. The
second term propagates uncertainty in the inferred bath instead of treating
its posterior mean as an exact hidden state.

The lag closure now uses the complete matrix
`E[(epsilon_{t+1}-mean)(epsilon_t-mean)^T]`, not only its diagonal. This can
detect memory transferred between different CV channels.

### 2.5 The CK promotion criterion had the wrong interpretation

Both implementations are Markov by construction: one in `q`, one in `(q,a)`.
Direct-versus-composed rollout CK error therefore measures finite-step and
Monte Carlo error, not whether adding memory explains held-out data.

V2 still reports this numerical diagnostic, but memory promotion instead
requires all of the following on a held-out replica:

1. endpoint energy skill is not degraded by more than one percentage point;
2. full lagged innovation RMS improves by at least 5 percent;
3. local path NLL improves by at least 0.001 nat per CV per transition;
4. memory coupling is nonzero and diffusion remains PSD.

## 3. Remaining Scientific Limits

V2 is a corrected pilot, not a complete molecular theory.

- The 1 ns observed transition density is approximated locally as Gaussian.
  It is not an exact atomistic path likelihood.
- Mobility and Mori coupling depend on protein and temperature but not on
  conformation. State-dependent kinetics require the Ito divergence discussed
  above and substantially more data.
- The 16 fixed CVs are coarse and use contiguous segments. They can miss
  domain hinges, nonlocal contact rearrangements, and transition-state
  coordinates.
- Denoising score matching assumes the sampled trajectory frames adequately
  represent each conditional equilibrium ensemble. Burn-in and replica
  stationarity need an independent audit.
- `F=U-TS` is identifiable only through cross-temperature coverage and only up
  to condition-dependent additive gauges. Held-out 450 K evaluation remains
  mandatory.
- Learned JEPA CVs remain blocked until fixed-CV memory passes the V2 gate.

The next scientifically justified expansion after that gate is a
state-dependent mobility with its exact Ito correction, followed by
graph-defined multiscale CVs. Increasing latent width or adding arbitrary
rollout penalties is not a substitute for these checks.
