# Submission-Ready V4-D Results

Date: 2026-08-29

## Defensible claims

1. The scratch dynamic-only representation removes the direct frozen-feature
   identity shortcut. D1 has nonzero within-trajectory switching (`0.415`) and
   an identity-leakage ratio of `0.471`, unlike the static V4-C1 assignments.
2. A reversible continuous-time generator gives positive persistence-relative
   Brier skill at all five horizons while satisfying detailed balance and the
   semigroup law by construction.
3. Generator rate calibration exposes a real skill-timescale tradeoff. The
   validation-selected scale `0.25` retains a slow base eigenvalue near `0.8`
   and transfers positive Brier and history signal to the independent test
   split.
4. Full horizon calibration is not solved. Horizon controls remain mixed and
   fixed-depth controls are not consistently beaten.

## Independent test result

The rate scale was selected on validation before evaluating the test manifest.

| Metric | Test |
| --- | ---: |
| Mean Brier skill vs persistence | 41.37% |
| History-shuffle advantage | 5.47% |
| Horizon-shuffle advantage | -0.38% |
| Correct vs best fixed depth | -2.60% |
| Base transition lambda2 | 0.806 |

## Intuitive classification metrics

The following numbers treat the learned six metastable states as classes. They
are easier to interpret than Brier score, but they also show that probability
improvement has not yet translated into better argmax prediction.

| Metric | V4-D | Baseline |
| --- | ---: | ---: |
| Top-1 state accuracy | 23.20% | 25.60% persistence |
| Top-2 state accuracy | 38.20% | 33.33% random |
| Balanced state accuracy | 24.05% | 16.67% random |
| Changed-state exact accuracy | 5.11% | 0% persistence by definition |
| Unchanged-state accuracy | 76.47% | 100% persistence |
| Transition detection precision | 67.52% | - |
| Transition detection recall | 25.85% | - |

Target states differ from the current state in `74.4%` of evaluated examples.
The model is conservative: when it predicts a transition it is often correct,
but it misses most transitions. Its top-1 accuracy is `2.4` percentage points
below persistence. The `41.37%` Brier skill therefore reflects better soft
probabilities, not a `41%` improvement in hard classification accuracy.

These learned-state forecast metrics are distinct from the Phase-1 supervised
MSM probe accuracy (`56.46%`).

| Horizon (raw frames) | Brier skill | History advantage | Horizon advantage | Correct vs fixed |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 39.49% | 9.11% | -7.49% | -8.25% |
| 64 | 40.40% | 8.93% | 3.96% | -2.44% |
| 96 | 42.79% | 4.55% | 3.04% | -0.34% |
| 128 | 44.21% | 3.36% | 0.05% | 0.00% |
| 192 | 39.96% | 1.43% | -1.43% | -1.99% |

## Other established results

- V4-A physical-state mean normalized MSE skill: `10.10%`.
- V4-A worst-horizon normalized MSE skill: `8.96%`.
- Phase-1 champion VAMP-CV lag1 dim5 normalized: `0.5588` under the original
  selected protocol (`0.5757` on the later comparable subset).
- Phase-1 trajectory-heldout MSM accuracy: `0.5646`.

The Phase-1 VAMP/MSM values and V4-D assignment-space VAMP objective are not
the same metric and must not be compared as if they were interchangeable.

## Suggested result wording

> The dynamic-only model improved soft-state prediction over persistence by
> 41.4% on an independent test split while retaining a base transition second
> eigenvalue of 0.806. Shuffling the observed history reduced performance by
> 5.5%, showing that the gain was not solely a stationary-occupancy baseline.
> Time calibration remained incomplete: horizon-shuffle and fixed-depth
> controls were mixed, especially at the shortest and longest evaluated lags.

This wording reports the useful result without claiming solved long-horizon
dynamics.
