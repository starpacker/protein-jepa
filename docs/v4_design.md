# V4-A Compositional Mean-Reverting Dynamics

Date: 2026-08-27

## Evidence entering V4

V3.5 step 2000 improved held-out normalized closed-state skill to `6.21%`
mean and `6.00%` worst horizon. Its trained horizon-pair skill was `-0.41%`.
Zeroing velocity reduced a 20-batch held-out skill audit from `5.06%` to
`0.43%`, while fixing every prediction to horizon 192 improved it to `5.44%`.
The current horizon decoder therefore hurts the otherwise useful context
signal.

Vector audits found absolute prediction/target cosine near `0.23`, but
same-context horizon-difference cosine only `0.00-0.02`. Context finite
differences had negative future cosine from `-0.27` to `-0.44`, identifying
mean reversion rather than ballistic persistence as the useful mechanism.

A cross-protein global linear Koopman map failed persistence because static
protein identity dominated the uncentered state. A context-centered diagonal
OU audit passed the mechanism gate: at raw lag 16 it improved persistence by
about `54%`, and the lag-16 contraction was better than lag-8 and lag-32
operators. Raw lag 32 remained composable from the lag-16 operator.

## V4-A mechanism

V4-A is a deterministic gate before stochastic V4-B.

- Closed physical states are computed at all 24 observed context frames.
- A train-only state mean fitted from 16,000 mdCATH segment endpoints is used
  with the existing train-only delta scales for centered inputs.
- A GRU encodes the physical state history; its memory is recurrently updated
  during rollout.
- One cross-segment attention layer couples the four sequence regions.
- A shared stable transition cell predicts equilibrium displacement,
  positive contraction rate, and residual drift.
- The cell is composed at base step 16: horizons 48/64/96/128/192 use
  3/4/6/8/12 applications. Semigroup structure is architectural.
- A causal context-internal raw-lag-16 transition is supervised together with
  all long rollouts.
- Physical step time and temperature are explicit transition conditions.

## Promotion gate

The 2K mdCATH pilot saves and evaluates every 500 steps. V4-A is promoted to
the low-rank Student-t V4-B only if:

1. mean and worst-horizon state skill exceed V3.5 (`6.21%/6.00%`);
2. correct rollout depth beats the best fixed-depth control on held-out data;
3. raw-lag-16 base-step skill is positive and rollout metrics do not degrade
   after step 1000;
4. horizon pair metrics are logged but are not optimized, because their
   realized target direction was empirically indistinguishable from noise.
