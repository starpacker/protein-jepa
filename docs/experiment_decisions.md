# Experiment Decisions

## Current Champion

Use the old multi-lag contextual JEPA checkpoint as the current main model:

- config: `configs/mdcath_state_jepa_multilag.yaml`
- checkpoint: `artifacts/mdcath_state_jepa_multilag/checkpoint_step20000.pt`
- convenience link: `artifacts/champion_state_jepa_multilag/checkpoint.pt`

## Why Not Longhorizon MSM

The `mdcath_state_jepa_longhorizon_msm` variant underperformed the old multi-lag model at the same 10k contextual quick protocol.

| metric | old multi-lag 10k | longhorizon+MSM 10k |
| --- | ---: | ---: |
| VAMP-CV norm dim5 | 0.4455 | 0.3924 |
| MSM accuracy | 0.5760 | 0.5108 |
| linear dynamics lag1 R2 | 0.4344 | 0.0932 |

The longhorizon+MSM run should be treated as a failed ablation, not the main line.

## Next Direction

Do not replace the old multi-lag objective wholesale. Future changes should be conservative:

- start from the old multi-lag model or checkpoint;
- add auxiliary objectives with warmup or very small weights;
- keep training and evaluation clip protocols aligned;
- promote a new variant only after it beats `checkpoint_step20000.pt` under the same contextual pipeline.

## Current Conservative Candidate

The next candidate is `configs/mdcath_state_jepa_multilag_refine_delta.yaml`.
It initializes from `artifacts/champion_state_jepa_multilag/checkpoint.pt`,
keeps the old multi-lag objective, lowers the learning rate, and adds only:

- `state_transition_weight: 0.05` with `auxiliary_warmup_steps: 1000`;
- `state_variance_reg: 0.002` with the same warmup;
- no MSM labels, no VAMP training loss, no linear-dynamics head.

Rationale: 20k outperformed 50k on the contextual quick pipeline, so the likely
failure mode is late representation drift rather than undertraining. The delta
auxiliary loss asks future state predictions to match the EMA target state
increments between adjacent target frames, which is label-free and directly
aligned with preserving short-time transition geometry.

## PhysChem Prior Candidate

Use `configs/mdcath_state_jepa_multilag_physchem.yaml` for the first
physics/chemistry prior run against the same baseline. It starts from the same
champion checkpoint and keeps the old multi-lag JEPA losses, then adds:

- kinetic semigroup consistency on existing lag heads;
- contact-persistence and backbone-torsion-change prediction heads;
- the same small state-delta and state-variance auxiliaries as the conservative candidate.

This candidate should be compared directly to
`artifacts/champion_state_jepa_multilag/checkpoint.pt` under the contextual
quick pipeline before it is promoted.

## Semigroup-Only Candidate

Use `configs/mdcath_state_jepa_multilag_semigroup.yaml` to isolate the kinetic
semigroup prior. It starts from the same champion checkpoint and keeps only:

- original residue/state/lagged-state JEPA losses;
- `semigroup_weight: 0.03` with warmup.

It intentionally disables physchem prediction, state-transition delta,
state-variance regularization, MSM labels, VAMP loss, and linear dynamics heads.

## Protocol Alignment Result

The largest reliable gain came from aligning the contextual evaluation clip
length with the champion training clip length:

- old quick protocol: `sequence_frames=16`, `window_stride=5`;
- aligned protocol: `sequence_frames=24`, `window_stride=8`;
- training config: `num_frames=24`, `window_stride=8`.

Under the aligned quick contextual pipeline, the old multi-lag champion remains
the main model:

| metric | old 16-frame protocol | aligned 24-frame protocol |
| --- | ---: | ---: |
| VAMP-CV norm lag1 dim5 | 0.4566 | 0.5588 |
| MSM accuracy | 0.5646 | 0.5646 |
| linear dynamics lag1 R2 | 0.5255 | 0.5381 |

The semigroup-only and physchem variants did not beat the champion overall under
the aligned protocol. Physchem slightly improved aligned VAMP-CV lag1 dim5
(`0.55896` vs `0.55882`) but still reduced lag1 linear dynamics, so it should
remain an ablation rather than a promoted model.

Use `artifacts/paper_eval/pipeline_aligned24_multilag_variants_contextual` as
the current quick comparison artifact for protocol-aligned claims.
