# Ablation Registry

This file records framework changes, evaluation status, and promotion decisions.
Only variants that improve the aligned evaluation metrics should be promoted.

## Current Champion

- Variant: multi-lag State-JEPA with mean frame pooling
- Config: `configs/mdcath_state_jepa_multilag.yaml`
- Checkpoint: `artifacts/mdcath_state_jepa_multilag/checkpoint_step20000.pt`
- Evaluation: `artifacts/paper_eval/pipeline_aligned24_multilag_variants_contextual`
- Decision: keep as current baseline/champion.

| Metric | Champion |
| --- | ---: |
| VAMP lag1 dim5 normalized | 0.5588 |
| MSM accuracy | 0.5646 |
| Linear dynamics lag1 R2 | 0.5381 |

## Rejected Or Not Promoted

| Variant | Main Result | Decision |
| --- | --- | --- |
| Semigroup-only warm-start | Lower VAMP and linear dynamics than champion | Do not promote |
| Physchem warm-start | Tiny VAMP gain, lower MSM/linear dynamics | Do not promote |
| Hard kinetic-query pooling warm-start | Lower VAMP/MSM/linear dynamics | Do not promote |
| Residual kinetic-query pooling warm-start | No clear gain over mean continue control | Do not promote |
| Kinetic-query pooling from scratch 20k | VAMP dim5 0.5649, MSM 0.5251, linear R2 0.3678 | Do not promote |
| Kinetic readout only | Kinetic readout improves VAMP but hurts MSM/linear dynamics | Keep as diagnostic, not champion |
| Time-scale conditioned multi-lag predictor 10k | VAMP dim5 0.5650 and MSM 0.5649, but linear R2 drops to 0.4846 | Do not promote |

## Active Candidates

| ID | Change | Config | Status | Decision |
| --- | --- | --- | --- | --- |
| SG-KIN-FUSE | Stop-gradient VAMPNet kinetic head with `state_kinetic` fusion readout | `configs/mdcath_state_jepa_multilag_stopgrad_kinetic_readout_5k.yaml` | Training | Pending eval |
| STATE-GEOM | Concatenate temporal state with deterministic geometry/physchem statistics | eval readouts `state_geom`, `state_clip_geom`, `state_clip_delta_geom` | Queued after SG-KIN-FUSE | Pending eval |

## Promotion Rule

Promote only if the variant beats the champion under the aligned contextual
quick pipeline on at least two of the three primary metrics without a large
regression on the third:

- VAMP-CV lag1 dim5 normalized;
- MSM trajectory-heldout accuracy;
- linear dynamics lag1 R2.
