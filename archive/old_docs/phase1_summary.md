# Phase 1 Summary

Date: 2026-08-24

This document freezes the Phase 1 outcome before starting the second stage.
The main goal of Phase 1 was to identify the strongest Protein State-JEPA
framework for protein dynamics world-model prediction and to test whether
physics/chemistry priors, kinetic readouts, frame pooling changes, and ATLAS
data can improve the aligned evaluation metrics.

## Executive Conclusion

The current best framework is still the original mdCATH-only multi-lag
State-JEPA with mean frame pooling.

- Framework: multi-lag State-JEPA with mean frame pooling
- Config: `configs/mdcath_state_jepa_multilag.yaml`
- Checkpoint: `artifacts/mdcath_state_jepa_multilag/checkpoint_step20000.pt`
- Convenience link: `artifacts/champion_state_jepa_multilag/checkpoint.pt`
- Main evaluation artifact: `artifacts/paper_eval/pipeline_aligned24_multilag_variants_contextual`
- Decision: keep this as the Phase 1 champion and Phase 2 baseline.

| Metric | Phase 1 Champion |
| --- | ---: |
| VAMP-CV lag1 dim5 normalized | 0.5588 |
| MSM trajectory-heldout accuracy | 0.5646 |
| Linear dynamics lag1 R2 | 0.5381 |

In the later paired ATLAS evaluation, the same checkpoint scored slightly higher
on VAMP because the evaluated trajectory subset changed:

| Metric | Baseline20 In ATLAS Paired Eval |
| --- | ---: |
| VAMP-CV lag1 dim5 normalized | 0.5757 |
| MSM trajectory-heldout accuracy | 0.5646 |
| Linear dynamics lag1 R2 | 0.5378 |

The strict promotion rule remains: promote a new framework only if it improves
at least two of the three primary metrics without a large regression on the
third.

## Evaluation Protocol

The strongest reliable improvement in Phase 1 came from aligning evaluation
clips with the training clip protocol.

- Readout: contextual `state` readout
- Evaluation preset: quick
- Sequence frames: 24
- Window stride: 8
- Linear window stride: 8
- Main selected VAMP metric: lag1, dim5, normalized score
- Main selected linear dynamics metric: lag1 R2
- MSM metric: trajectory-heldout linear probe accuracy

Protocol alignment result:

| Metric | Old 16-frame Protocol | Aligned 24-frame Protocol |
| --- | ---: | ---: |
| VAMP-CV lag1 dim5 normalized | 0.4566 | 0.5588 |
| MSM accuracy | 0.5646 | 0.5646 |
| Linear dynamics lag1 R2 | 0.5255 | 0.5381 |

Phase 2 comparisons should continue to use the aligned contextual protocol
unless explicitly stated otherwise.

## What Worked

The multi-lag JEPA formulation was the most stable design choice. It preserves
short-time kinetic predictability, keeps MSM performance competitive, and gives
the best overall balance across VAMP, MSM, and linear dynamics.

Mean frame pooling remains the best production readout so far. Kinetic pooling
and kinetic heads can improve isolated metrics, but they caused large tradeoffs
in MSM or linear dynamics and therefore did not become the main framework.

Physical time conditioning is useful but insufficient. In mdCATH plus ATLAS
pilot300 training, physical-time conditioning recovered part of the damage from
naive dataset mixing and slightly improved MSM over the paired baseline, but it
still did not match champion VAMP or linear dynamics.

## What Did Not Promote

| Variant | Best Observed Result | Decision |
| --- | --- | --- |
| Longhorizon + MSM 10k | VAMP 0.3924, MSM 0.5108, R2 0.0932 vs old multi-lag 10k VAMP 0.4455, MSM 0.5760, R2 0.4344 | Failed |
| Semigroup-only warm-start | VAMP 0.5472, MSM 0.5568, R2 0.4685 | Do not promote |
| Physchem warm-start | VAMP 0.5590, MSM 0.5613, R2 0.4757 | Tiny VAMP gain, worse MSM/R2 |
| Kinetic-query pooling from scratch 20k | VAMP 0.5649, MSM 0.5251, R2 0.3678 | VAMP up, MSM/R2 collapse |
| Time-scale conditioned multi-lag predictor 10k | VAMP 0.5650, MSM 0.5649, R2 0.4846 | VAMP/MSM small gain, R2 drops |
| Stop-gradient kinetic head, state readout | VAMP 0.5758, MSM 0.5613, R2 0.5200 | Good VAMP diagnostic, not overall champion |
| Stop-gradient kinetic head, state+clip+kinetic readout | VAMP 0.5651, MSM 0.5185, R2 0.6066 | Linear R2 diagnostic, MSM drops sharply |
| State+clip+geometry fusion readout | VAMP 0.5314, MSM 0.5220, R2 0.5638 | Linear R2 up, VAMP/MSM down |
| Scratch linear dynamics auxiliary 20k | VAMP 0.5701, MSM 0.5405, R2 0.3659 | VAMP up, MSM/R2 down |
| ATLAS pilot300 mixed-data scratch 20k | VAMP 0.5370, MSM 0.5614, R2 0.4019 | Naive ATLAS mixing hurts |
| ATLAS pilot300 physical-time scratch 20k | VAMP 0.5559, MSM 0.5698, R2 0.4502 | Better than naive ATLAS, still below champion |

The detailed ablation ledger is maintained in `docs/ablation_registry.md`.

## Dataset Findings

mdCATH and ATLAS are both protein MD trajectory datasets, but they are not
exchangeable under the current clip construction.

| Property | mdCATH | ATLAS pilot300 |
| --- | --- | --- |
| Records used | 15075 | 300 |
| Approximate clips | 791458 | 373500 |
| Clip weight in mixed manifest | about 68% | about 32% |
| Temperature | 320/348/379/413/450 K | 300 K |
| Frame interval | about 1 ns/frame | about 10 ps/frame |
| Frames per trajectory | about 440 | 10001 |
| 24-frame clip with stride 2 spans | about 47 ns | about 0.47 ns |

Interpretation: adding ATLAS does increase trajectory data, but it also injects
a very different physical time scale and distribution. Without careful handling,
ATLAS contributes many short physical-time clips and pulls the representation
away from the mdCATH aligned evaluation target. Physical-time conditioning
partly addresses this but does not solve the distribution mismatch.

## Phase 1 Code And Artifacts To Preserve

Important implementation points added during Phase 1:

- Multi-lag state prediction with lag embeddings remains the main framework.
- Kinetic/semigroup and physchem auxiliary losses are available as ablations.
- Kinetic-query pooling and kinetic readout paths are available as diagnostics.
- Physical-time conditioning is implemented for temporal/state/lagged predictors
  and evaluation readout extraction passes `time_ps` for contextual models.

Important commits:

- `ac550ee Record physical-time ATLAS ablation result`
- `e29002f Add physical time conditioning for mixed MD training`
- `73a15f6 Record ATLAS pilot300 ablation result`
- `68f2660 Add ATLAS pilot mixed-data experiment`
- `7844829 Add scratch-first balanced ablations`
- `e9b9042 Record kinetic and geometry ablation results`
- `5110b3e Record time-conditioned ablation result`

Important artifacts:

| Purpose | Path |
| --- | --- |
| Phase 1 champion checkpoint | `artifacts/mdcath_state_jepa_multilag/checkpoint_step20000.pt` |
| Champion convenience checkpoint | `artifacts/champion_state_jepa_multilag/checkpoint.pt` |
| Champion aligned eval | `artifacts/paper_eval/pipeline_aligned24_multilag_variants_contextual` |
| Kinetic pooling scratch eval | `artifacts/paper_eval/pipeline_aligned24_kinetic_pool_scratch20k_contextual` |
| ATLAS mixed eval | `artifacts/paper_eval/pipeline_aligned24_mdcath_atlas_pilot300_contextual` |
| ATLAS physical-time eval | `artifacts/paper_eval/pipeline_aligned24_mdcath_atlas_pilot300_physical_time_contextual` |

## Phase 2 Starting Point

Phase 2 should start from the mdCATH-only multi-lag State-JEPA champion as the
baseline. The current evidence says that simple auxiliary losses, simple frame
pooling changes, and naive ATLAS expansion are unlikely to produce a large jump
by themselves.

The most promising Phase 2 directions are therefore:

- better cross-dataset training strategy for mdCATH plus ATLAS, not temporal
  normalization but distribution-aware sampling or curriculum;
- model changes that preserve linear dynamics while improving VAMP, because many
  Phase 1 variants improved one metric at the cost of another;
- explicit treatment of physical time and lag semantics as model inputs, but with
  stronger regularization against forgetting mdCATH-scale slow dynamics;
- evaluation-driven promotion only, using the same aligned contextual protocol.

For final claims, every new candidate should be trained from scratch unless the
experiment is explicitly labeled as a warm-start diagnostic.
