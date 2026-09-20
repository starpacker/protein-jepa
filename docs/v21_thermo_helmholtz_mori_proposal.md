# V21 proposal: Thermodynamic Helmholtz--Mori JEPA (THM-JEPA)

Date: 2026-09-19

This is a concrete next experiment, not a claim that the current V20 run has
already validated the idea.

## 1. What the current evidence says

- V20 is still in the representation phase at the currently available step
  (step 390). `lspwm_force_loss`, `lspwm_bath_aware_drift_matching_loss`, and
  `lspwm_conservative_alignment_loss` are all exactly zero. Its endpoint JEPA
  skill is therefore not evidence for force learning yet.
- The corrected v6 evaluation is already strong on M1 distribution closure
  (`stationary_jsd=0.350`, `innovation_whiteness=215.6`) and on pooled long-time
  K1 ITS (`0.397` versus `0.426` for tICA+MSM), but is weaker on K1 short lags,
  NLL, and MFPT (`0.841`).
- The force-identification problem is structural: a bath term can explain most
  of the observed increment, leaving the conservative force with almost no
  gradient signal. A cosine alignment term helps direction, but does not make
  the decomposition identifiable by itself.

## 2. Proposed model

Use a learned latent coordinate `q` and a small auxiliary memory state `a`.
The drift is explicitly split into three physically different pieces:

```text
dq = [ - M(q,c,T) grad F(q,c,T)
       + J(q,c,T) grad H(q,c,T)
       + C(q,c,T) a ] dt + B(q,c,T) dW

da = [ - (S(q,c,T) + A(q,c,T)) a
       - C(q,c,T)^T grad F(q,c,T) ] dt + L(q,c,T) dW_a
```

Here `M` is PSD, `S` is PSD, and `A` is skew-symmetric. `J` is skew-symmetric
and represents a divergence-free reversible/rotational component. Noise is
constructed by `B B^T = 2 k_B T M` and `L L^T = 2 k_B T S`; thus the symmetric
dissipative part obeys FDT while `J` and `A` cannot create or remove
equilibrium probability. The joint `(q,a)` transition is evaluated with the
existing Van Loan exponential, rather than Euler rollout.

The important implementation change is **orthogonal force/memory
identification**. At each observed increment `Delta q`, first compute the
conditional bath estimate from history, `r_bath = C E[a|history]`. Fit the
conservative and reversible terms only to the residual
`r = Delta q/dt - r_bath`, with a stop-gradient copy of `r_bath` in this loss.
The bath is trained separately against the history-predictable component. This
prevents the bath from absorbing all signal and prevents the force head from
chasing noise.

## 3. Losses

For a batch, use the following normalized objective:

```text
L = L_JEPA
  + 1.0 L_transition_NLL
  + 0.25 L_residual_force
  + 0.10 L_force_direction
  + 0.10 L_bath_prediction
  + 0.05 L_semigroup
  + 0.05 L_FDT
  + 0.02 L_detailed_balance
  + 0.02 L_horizon_order
```

`L_residual_force` is a Huber loss on the whitened residual, not raw MSE.
`L_force_direction` is cosine loss after removing the bath prediction.
`L_bath_prediction` predicts the increment from history with the current force
branch detached. `L_FDT` penalizes `||BB^T-2kBTM||` and
`||LL^T-2kBTS||`; `L_detailed_balance` is a reversibility test on paired
forward/reverse transitions. `L_horizon_order` is a same-context ranking loss:
the predicted integrated displacement/variance must be ordered consistently for
`h=1,4,8,16`, which directly targets the V5.1/V20 horizon-collapse failure.

All covariance matrices are parameterized by Cholesky factors or matrix
exponentials; never optimize an unconstrained covariance or unconstrained
relaxation matrix.

## 4. Training schedule

1. **0--500 steps, representation warm start:** load v19/v20, freeze `F,M,S`
   and train only the paired predictor plus the bath observer. Log the
   residual variance fraction.
2. **500--1500, force identification:** unfreeze only `F` and `J`; use the
   orthogonal residual losses above. Keep `M` and the bath scale frozen.
3. **1500--3000, thermodynamic calibration:** unfreeze `M,S,C`, enable FDT,
   semigroup and detailed-balance terms; use exact Van Loan transitions.
4. **3000--5000, closed rollout:** enable 4/8/16-step rollout and horizon
   ranking. Select a checkpoint on validation only.

For the first run, do not change the encoder dimension or dataset. This makes
the result attributable to the decomposition rather than capacity or protocol.

## 5. Required controls and promotion gates

Run the same protein-disjoint, temporal-holdout protocol for:

| Variant | Force | Memory | Reversible term |
|---|---:|---:|---:|
| M0 | no | no | no |
| M1 | yes | no | no |
| M2 | no | yes | no |
| THM-JEPA | yes | yes | yes |

Do not promote a checkpoint unless all of the following hold on validation:

- no non-finite rollout and no diffusion condition number above 100;
- residual-force cosine improves over v20 and the bath fraction is not >0.9;
- M1 beats M0 on force-residual NLL, while M2 beats M0 on innovation
  whiteness; THM-JEPA must beat both on stationary JSD and K1 pooled ITS;
- K1 h=1 and h=4 do not regress by more than 2% relative to v6, and MFPT
  relative error decreases below 0.75;
- horizon-shuffle advantage is positive and same-context horizon ranking is
  significant by protein bootstrap;
- three seeds agree in sign before opening the blind test set.

If the residual cosine remains near zero after 1500 steps, stop increasing
model size: the data do not identify a conservative force under the current
sampling interval, and the honest paper claim should remain “thermodynamically
guarded stochastic JEPA”, not force recovery.

## 6. Immediate action on V20

Let V20 continue only until the first physical curriculum boundary (step 1200),
then evaluate the residual diagnostics. Do not compare its step-320 endpoint
numbers with K1/M1. If the physical terms remain zero or the bath fraction is
near one at step 1500, branch to THM-JEPA rather than extending V20 training.
