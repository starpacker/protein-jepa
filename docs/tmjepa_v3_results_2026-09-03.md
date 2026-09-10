# TM-JEPA V3 implementation and validation results (2026-09-03)

## Scope

V3 was implemented to make temperature, memory, and Markov closure testable physical
claims rather than interpretations attached to rollout error. Model selection used only
the protein-disjoint validation split. The blind test proteins and blind 450 K split were
not accessed.

## Implemented physics

- Temperature-dependent free energy uses separate reference enthalpy, entropy, and heat
  capacity fields:
  `beta F_T = (T0/T) h - s + cp [1 - T0/T - log(T/T0)]`.
- Mobility follows an Eyring-Einstein clock with global and bounded state/mode activation
  energies. State-dependent diffusion is PSD by construction and includes the exact Ito
  divergence.
- Diffusion rank and Mori auxiliary rank are independent.
- Every Mori mode has an independent Eyring activation energy. Freezing diffusion no
  longer freezes memory temperature dynamics.
- The auxiliary bath is integrated as an exact local OU process. The q-a process-noise
  cross covariance is retained, and the same transition is used by filtering, path
  likelihood, and rollout.
- A continuous-discrete Kalman filter replaces the learned observer. Its posterior is the
  only history-dependent state supplied to rollout.
- A causal bath-reset intervention and a raw-history innovation probe make memory utility
  and Markov closure falsifiable.

## Data and identification

The V3 manifests contain protein-disjoint train, validation, test, and 450 K temperature
test splits. Robust CV location and scale were fit on training trajectories only.

Residual block-Hankel identification on the overdamped training residuals found:

- observation interval: 1000 ps;
- lag-1 covariance RMS: 0.1372;
- rank required for 90% Hankel energy: 35 (the earlier reported rank 8 was a cap, not the
  identified rank);
- relaxation poles concentrated near 1.08-1.17 ns and 13.8-16 ns;
- lag-1 symmetric residual covariance was entirely negative spectral energy;
- later lags contained roughly 34-52% antisymmetric energy.

The last two observations are incompatible with interpreting the residual as a sum of
independent, completely monotone OU force modes.

## Evaluation corrections

Two errors in the initial protocol were corrected:

1. OD and Mori had been evaluated with 24-frame and 64-frame contexts, respectively.
   All final comparisons use the same 64-frame context and identical sampling seed.
2. The first closure probe regressed raw increments while omitting protein condition and
   temperature. The final probe predicts the standardized one-step innovation, which is
   already conditioned on `(q, a, protein, T)`, from prior raw increments. This directly
   tests innovation whiteness without an out-of-distribution state regressor.

Innovation covariance loss now weights diagonal calibration explicitly, and lag closure
uses correlation rather than unnormalized covariance. This prevents variance inflation
or collapse from trivially reducing the lag loss.

## Validation results

| Candidate | Path NLL improvement vs OD | Multi-lag reduction | Endpoint ES change | Reset advantage | History skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank-8 likelihood Mori | +0.0008 | 1.2% | +0.0050 | -0.0174 | 17.2% |
| rank-8 closure-aware | -0.0226 | 14.3% | +0.0963 | -0.0200 | not promoted |
| rank-8 calibrated joint | +0.0038 | 16.0% | +0.0577 | -0.0192 | 18.4% |
| rank-35 calibrated joint | -0.0083 | 20.5% | +0.0452 | -0.0169 | 17.1% |

Positive endpoint change means degradation. The rank-35 path-NLL bootstrap interval was
`[-0.0278, 0.0144]`. No Mori candidate passed any of the required likelihood,
multi-lag, reset-intervention, or Markov-closure gates. The overdamped V3 checkpoint is
therefore the only retained V3 model; memory is not promoted.

## First-principles diagnosis

Freezing OD diffusion while adding a fluctuation-dissipation-consistent bath is invalid:
the bath adds both colored drift and one-step variance, so instantaneous diffusion must be
re-estimated jointly. Joint fitting fixed this structural error but did not make the
diagonal OU bath sufficient. Increasing rank from 8 to the identified 35 modes also did
not help: likelihood drove the integrated memory/diffusion ratio back to 0.0138.

The remaining residual has negative and antisymmetric spectral structure. The next model
must use a generalized bath generator

`Gamma = S + A`, with `S` positive semidefinite and `A` skew-symmetric,

and bath noise covariance `2 S`. This preserves the Gibbs invariant measure and FDT while
allowing damped rotational/oscillatory hidden modes. Its exact discrete transition and
process covariance must be shared by filter, likelihood, and rollout. A model is not
Markov closed until the protein-disjoint raw-history innovation skill is below 1% and all
existing promotion gates pass.

