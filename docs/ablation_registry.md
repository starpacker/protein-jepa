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
| Stop-gradient kinetic head, state readout | VAMP dim5 improves to 0.5758, but MSM drops to 0.5613 and linear R2 drops to 0.5200 | Do not promote |
| Stop-gradient kinetic head, state+kinetic readout | VAMP dim5 0.5596 and MSM 0.5680, but linear R2 drops to 0.4969 | Do not promote |
| Stop-gradient kinetic head, state+clip+kinetic readout | VAMP dim5 0.5651 and linear R2 rises to 0.6066, but MSM drops sharply to 0.5185 | Promising diagnostic, not champion |
| Stop-gradient kinetic head, state+clip+delta+kinetic readout | Lower VAMP/MSM/linear dynamics than champion | Do not promote |
| State+geometry fusion readouts | State-clip-geometry improves linear R2 to 0.5638, but all geometry fusion readouts hurt VAMP/MSM | Do not promote |
| Scratch stop-gradient kinetic head 20k, state readout | VAMP dim5 0.5626, MSM 0.5469, linear R2 0.4117 | Do not promote |
| Scratch stop-gradient kinetic head 20k, state+kinetic readout | VAMP dim5 0.5482, MSM 0.5477, linear R2 0.4102 | Do not promote |
| Scratch stop-gradient kinetic head 20k, state+clip+kinetic readout | VAMP dim5 0.5512, MSM 0.5267, linear R2 0.4949 | Do not promote |
| Scratch linear dynamics auxiliary 20k, state readout | VAMP dim5 0.5701, but MSM 0.5405 and linear R2 0.3659 | Do not promote |
| Scratch linear dynamics auxiliary 20k, state+clip readout | VAMP dim5 0.5583, MSM 0.5148, linear R2 0.5325 | Do not promote |
| ATLAS-PILOT300 mixed-data scratch 20k | Paired eval: VAMP dim5 0.5370 vs baseline 0.5757, MSM 0.5614 vs 0.5646, linear R2 0.4019 vs 0.5378 | Do not promote |

## Active Candidates

| ID | Change | Config | Status | Decision |
| --- | --- | --- | --- | --- |
| ATLAS-PILOT300-PHYSICAL-TIME | From-scratch champion architecture trained on mdCATH plus ATLAS pilot300 with physical ps-scale time conditioning in temporal/state/lagged predictors | `configs/mdcath_atlas_pilot300_state_jepa_multilag_physical_time_scratch20k.yaml` | Launching | Pending eval |

## Promotion Rule

Promote only if the variant beats the champion under the aligned contextual
quick pipeline on at least two of the three primary metrics without a large
regression on the third:

- VAMP-CV lag1 dim5 normalized;
- MSM trajectory-heldout accuracy;
- linear dynamics lag1 R2.
