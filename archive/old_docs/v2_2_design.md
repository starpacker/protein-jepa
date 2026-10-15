# V2.2 Dual-State Dynamics Design

V2.2 addresses the V2.1 tradeoff: endpoint forecasting improved strongly,
while the general-purpose VAMP/MSM representation degraded. The model now
has a shared encoder with two explicitly evaluated latent contracts.

## Objectives

- Keep the V2.1 causal endpoint residual forecast at raw horizons `[8, 16, 24, 48]`.
- Add a low-dimensional slow/metastable endpoint state for VAMP/MSM quality.
- Prevent the slow head from collapsing with variance, covariance and in-batch
  temporal contrastive constraints.
- Use the same endpoint readout in training and evaluation; window averages are
  retained only as legacy diagnostics.
- Remain mdCATH-only and scratch-trained for promotion claims.

## Architecture

The online context endpoint is `z_c` and the EMA target future endpoint is
`z_f`. The original dynamic head predicts:

```text
delta_hat = f(z_c, log(tau), temperature)
z_hat = z_c + delta_hat
```

The slow head projects each endpoint to a 32-dimensional state:

```text
s_c = slow_head(z_c)
s_f = EMA_slow_head(z_f)
s_hat = s_c + g(s_c, log(tau), temperature)
```

`s_f` is detached. The slow objective combines delta/state Huber losses,
bidirectional in-batch temporal contrastive loss, per-dimension variance
floor, and off-diagonal covariance suppression. The slow target projector is
updated by the same EMA schedule as the target encoder.

The semigroup gradient remains disabled. It is not reintroduced until direct
held-out forecast skill and slow-state metrics are both stable.

## Evaluation

V2.2 reports three separate readouts:

- `endpoint`: causal final frame of the temporal encoder, full-dimensional;
- `slow_endpoint`: causal final frame passed through the 32-dimensional slow head;
- legacy `state`: window-averaged state, retained for comparison with earlier runs.

Promotion still requires positive non-overlapping forecast R2 and improvement
on at least two of VAMP, nested MSM balanced accuracy, and forecast R2 without
a large regression on the third. Paired persistence-relative skill remains a
diagnostic and cannot by itself promote a world model.

## Ablation interpretation

If `slow_endpoint` improves VAMP/MSM while `endpoint` retains paired skill,
V2.2 resolves the representation conflict through a useful multi-head state.
If both remain weak, the next intervention should target data coverage and
readout capacity rather than adding more consistency losses.
