# V2.1 Paired-Window Dynamics Design

Status: design frozen before implementation

V2.1 changes the training contract so that it matches the primary forecast
benchmark. The unit of prediction is a future trajectory window, not an
arbitrary frame inside the same training clip.

## 1. Scope

- Dataset: mdCATH only.
- Training: from scratch for promotion claims; warm-start runs are diagnostics.
- Primary horizons: raw trajectory-frame offsets `[8, 16, 24, 48]`.
- Primary benchmark: non-overlapping window forecast, with raw offset `48`
  matching the current `sequence_frames=24`, `frame_stride=2`,
  `window_stride=8`, lag-6 protocol.
- Semigroup loss: diagnostic only in V2.1, weight zero during initial training.
- ATLAS and mixed physical time scales: excluded until the mdCATH-only task is
  demonstrably working.

## 2. Data Contract

Each sample contains two windows from the same trajectory:

```text
context: [s, s + 2, ..., s + 46]
future:  [s + h, s + h + 2, ..., s + h + 46]
```

where `h` is sampled from `[8, 16, 24, 48]` in raw trajectory-frame units.
The pair is valid only when the complete future window is inside the same
trajectory. Protein, trajectory, temperature, and domain metadata are shared
between the two windows. Sampling is hierarchical by protein and trajectory,
then balanced across horizons.

The collated batch must expose:

- `context_*` tensors;
- `future_*` tensors;
- `horizon_frames`;
- `horizon_time_ps`;
- shared `temperature_k`, `dataset_index`, `protein_id`, and `trajectory_id`.

The loader must assert that context and future belong to the same trajectory
and that `future_start - context_start == horizon_frames`.

## 3. State Contract

The forecasting state is the causal endpoint state:

```python
z_context = context_frame_states[:, last_valid_context_frame]
z_future = target_frame_states[:, last_valid_future_frame]
```

No mean-over-window state is used by the paired forecast objective. The same
endpoint readout must be used by the evaluation extractor. Offline clip
averages may remain as a separate diagnostic, but cannot be the promotion
readout.

The EMA target endpoint is detached. The online context endpoint receives the
forecast gradient. Both states are projected into the same normalized dynamic
space exactly once.

## 4. V2.1 Transition Head

V2.1 uses a shared horizon-conditioned residual transition rather than one
independent head per frame-index lag:

```text
delta_hat = f(z_context, log(horizon_time_ps), temperature, domain)
z_hat = z_context + delta_hat
```

The initial head is zero-initialized, which makes the first prediction equal to
persistence while the delta objective measures progress beyond persistence.
The head receives physical `tau`, not only an integer lag id.

V2.1 remains deterministic to isolate the representation/data-contract repair.
A stochastic Gaussian or mixture transition is a V2.2 follow-up after the
deterministic paired task obtains positive held-out R2.

## 5. Losses

The initial objective is:

```text
L = 0.10 * residue_JEPA
  + 1.00 * delta_Huber(delta_hat, z_future - z_context.detach())
  + 0.25 * state_Huber(z_hat, z_future)
  + 0.02 * endpoint_variance_regularizer
```

The delta target is detached only on the target side. The online context state
and transition head remain trainable. There is no semigroup gradient in the
first V2.1 run. A semigroup or rollout consistency term can be enabled only
after held-out direct skill is positive and the persistence-relative margin is
stable.

The trainer must log, per horizon:

- absolute state loss;
- delta loss;
- persistence delta loss;
- skill relative to persistence;
- endpoint source/target alignment;
- valid pair count.

## 6. Evaluation Contract

V2.1 evaluation reports the same horizons used in training:

- raw offsets `8`, `16`, `24`, and `48`;
- endpoint latent linear forecast R2;
- persistence-relative MSE and skill;
- VAMP at the corresponding physical horizons;
- endpoint-to-future MSM transition accuracy, balanced accuracy, and macro-F1.

The existing paper pipeline remains the promotion scoreboard, but its `state`
readout must use endpoint states for V2.1. A positive non-overlapping lag-6 R2
is a hard requirement. Promotion still requires improvement on at least two of
VAMP, nested MSM balanced accuracy, and forecast R2 without a large regression
on the third.

## 7. Implementation Order

1. Add `PairedTrajectoryDataset` with deterministic horizon-aware entries.
2. Add paired-window collation and hierarchy/horizon sampler support.
3. Add `encode_endpoint_state` and a shared physical-time `PairedDeltaPredictor`.
4. Add paired forward/losses and per-horizon logging.
5. Add endpoint forecast evaluation and synthetic trajectory tests.
6. Run a 2k mdCATH gate; only continue to 4k/8k if the 48-frame horizon is
   improving relative to persistence.

## 8. Baselines and Promotion

The frozen formal baseline remains Phase 1 E0. `geometry_stats` and
`centered_ca` remain non-learned references. A3/V1 checkpoints are ablation
artifacts, not baselines, because they are warm-started and fail the positive
non-overlapping forecast requirement.
