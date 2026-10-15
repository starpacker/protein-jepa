# V3.4 Continuous Context-Conditioned Dynamics

Date: 2026-08-27

## Diagnosis

V3.3 standardized the 12 closed-state channels but failed its horizon gate.
Correct history was strongly useful, while swapping horizon changed MSE by
less than `0.3%`. Its five independent experts converged to nearly
interchangeable context-conditioned means and the ranking loss stayed at its
`0.05` margin.

A held-out target audit found substantial same-context target differences:
normalized pairwise MSE ranges from `0.66` for horizons `48-64` to `1.18` for
`48-192`. However, a train-split horizon-centroid classifier achieved `20.2%`
accuracy for five classes, equal to the `20%` chance level. There is no global
horizon-mean shortcut; time evolution must be predicted from context history.

## Mechanism

V3.4 removes independent horizon experts. A horizon-independent trunk predicts
context-conditioned coefficients for one linear drift mode and four continuous
relaxation modes:

```text
tau = horizon / max_training_horizon
b(tau) = [tau, 1-exp(-tau/0.25), 1-exp(-tau/0.5),
          1-exp(-tau/1.0), 1-exp(-tau/2.0)]
state(tau) = state(0) + sum_m coefficient_m(context) * b_m(tau)
```

This construction is continuous in time and exactly equals persistence at
`tau=0`. Every horizon shares the same context representation and dynamical
coefficients. Operator dropout is disabled so duplicated same-context rows do
not receive different random dynamics.

The primary new objective directly supervises horizon differences:

```text
L_difference = mean((((p(h2)-p(h1)) - (y(h2)-y(h1))) / scale)^2)
```

Pairs with fewer than 48 raw frames of separation are excluded from this loss
to avoid spending capacity on the weakest adjacent-time signal. Standardized
absolute-state loss remains active. The failed hinge ranking loss has zero
training weight and is logged only as a diagnostic.

The V3.2 selected checkpoint initializes all protected representation paths.
The complete closed operator is intentionally reinitialized because its
parameterization changed.

## Gate

Checkpoints are evaluated at `500/1000/1500/2000` with the unchanged held-out
manifest and group-aware controls. Promotion requires:

- positive standardized persistence-relative skill at every horizon;
- positive pairwise difference skill for every trained pair with gap >= 48;
- at least `2%` correct-versus-shuffled horizon advantage at every horizon,
  with `5%` retained as the promotion target.

Failure of pairwise difference skill means context history cannot predict the
realized thermal path in this state representation. In that case the next
mechanism must become probabilistic or add velocity/momentum state; extending
deterministic training is not justified.

## 10k Resume Outcome

The first 2k run completed normally and was resumed from
`artifacts/phase3_v3_4_continuous_difference_2k/checkpoint_step2000.pt` to
10k with a longer cosine schedule:

- Resume config:
  `configs/phase3_v3_4_continuous_difference_10k_resume.yaml`.
- Resume artifact:
  `artifacts/phase3_v3_4_continuous_difference_10k_resume`.
- Held-out controls:
  `artifacts/paper_eval/phase3_v3_4_continuous_difference_step{500,1000,1500,2000,2500,3000,4000,6000,8000,10000}_controls.json`.
- Loss curve:
  `artifacts/phase3_v3_4_continuous_difference_10k_resume/v3_4_10k_closed_local_losses.png`.

Held-out summary, normalized metrics:

| step | horizon mean skill | worst horizon skill | horizon shuffle mean | trained-pair mean skill | worst trained-pair skill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.011% | -0.099% | 0.194% | 0.075% | -0.102% |
| 1000 | 0.549% | 0.391% | 0.172% | 0.067% | -0.063% |
| 1500 | 0.655% | 0.363% | 0.096% | 0.085% | -0.144% |
| 2000 | 1.121% | 0.531% | 0.153% | 0.273% | 0.081% |
| 2500 | 0.627% | 0.361% | 0.195% | 0.143% | -0.007% |
| 3000 | 0.860% | 0.665% | 0.216% | 0.034% | -0.017% |
| 4000 | 0.753% | 0.337% | 0.127% | 0.181% | 0.042% |
| 6000 | 1.079% | 0.634% | 0.144% | 0.108% | -0.187% |
| 8000 | 1.665% | 0.901% | 0.168% | 0.308% | 0.001% |
| 10000 | 1.701% | 0.977% | 0.155% | 0.299% | 0.066% |

V3.4 10k slightly exceeds the selected V3.2 closed-state baseline on
normalized persistence-relative skill (`1.70%` mean, `0.98%` worst, versus
V3.2 step 2000 at `1.34%` mean, `0.89%` worst). This is a real but small
closed-state gain.

It still fails the intended world-model gate. Horizon shuffle advantage remains
around `0.16%`, far below the `2%` minimum gate, and pairwise horizon-difference
skill remains near zero (`0.30%` mean at 10k). Training longer therefore helps
absolute state MSE modestly but does not make the deterministic transition
operator learn realized horizon-conditioned dynamics.
