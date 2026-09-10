# EVA-LS1 Learned-State V-JEPA Evaluation

Date: 2026-09-06

## Checkpoint

- Model: EVA-LS1 learned-state V-JEPA
- Checkpoint: `/liuyutian/protein_jepa_frame/artifacts/eva_ls1_learned_state_vjepa/checkpoint_step2000.pt`
- Training steps: 2000
- Encoder status: online context encoder, temporal encoder, slow-state projector, Gaussian transition predictor, and slow predictor all received nonzero gradients during training.
- Fixed CV usage: no fixed physical CV is used as model state or loss target. Fixed CVs remain evaluation-only baselines.

## Evaluation Protocol

Paper-grade representation evaluation:

- Output directory: `/liuyutian/protein_jepa_frame/artifacts/paper_eval/eva_ls1_step2000_slow_endpoint_filtered_full`
- Manifest: `/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl`
- MSM labels: `/liuyutian/protein_jepa/artifacts/msm_state_labels/mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered_labels.json`
- Readout: `slow_endpoint`
- Context: 24 frames, frame stride 2, window stride 8
- VAMP/linear lags: 6, 12, 24
- VAMP dims: 5, 10
- VAMP trajectories: 50
- MSM clips: 8000 requested, 6588 labeled clips used
- Split discipline: protein-stratified blocked VAMP/linear splits and per-protein trajectory-heldout MSM probe

Quick paired endpoint evaluation:

- Output: `/liuyutian/protein_jepa_frame/artifacts/paper_eval/eva_ls1_step2000_paired_validation_100b.json`
- Manifest: `/liuyutian/protein_jepa_frame/manifests/tmjepa_v3.validation.jsonl`
- Max batches: 100
- Batch size: 4
- Horizons: 1, 2, 4, 8, 16, 24, 48 frames

## Representation Metrics

Cross-validated normalized VAMP-2:

| Readout / baseline | Lag 6 dim 5 | Lag 6 dim 10 | Lag 12 dim 5 | Lag 12 dim 10 |
|---|---:|---:|---:|---:|
| EVA-LS1 `slow_endpoint` | 0.6238 | 0.8536 | 0.7326 | 0.7143 |
| Geometry statistics | 0.5969 | 0.8579 | 0.7096 | 0.7133 |
| Centered CA | 0.6268 | 0.8521 | 0.7305 | 0.7143 |

Lag-6 dim-5 paired bootstrap:

| Comparison | Mean delta | 95% CI | Fraction positive |
|---|---:|---:|---:|
| Geometry statistics minus EVA-LS1 | -0.0242 | [-0.0404, -0.0066] | 0.300 |
| Centered CA minus EVA-LS1 | 0.0042 | [-0.0139, 0.0224] | 0.533 |

Interpretation: EVA-LS1 has a real learned slow-mode signal. It significantly beats geometry statistics at lag-6 dim-5 and is statistically indistinguishable from centered CA in this quick bootstrap. At dim-10 it is essentially tied with coordinate/geometry baselines.

## MSM State Probe

Trajectory-heldout MSM probe:

| Metric | EVA-LS1 |
|---|---:|
| Accuracy | 50.10% |
| Balanced accuracy | 41.15% |
| Macro-F1 | 35.15% |
| Majority baseline | 37.90% |
| Proteins above baseline | 19 / 30 |

Reference from the EVA5C encoder audit:

| Readout | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| EVA5C fixed CV | 53.10% | 43.68% | 38.17% |
| EVA5C causal endpoint | 40.82% | 35.00% | 30.64% |
| EVA5C learned CV | 45.00% | 35.44% | 29.10% |

Interpretation: EVA-LS1 improves substantially over the old learned-CV and causal-endpoint learned representations, but it does not exceed the EVA5C fixed-CV MSM probe.

## Linear Dynamics Closure

Blocked cross-validated ridge future prediction R2:

| Readout / baseline | Lag 6 R2 | Lag 12 R2 |
|---|---:|---:|
| EVA-LS1 `slow_endpoint` | -1.1248 | -1.5912 |
| Geometry statistics | -0.2522 | -0.3672 |
| Centered CA | -2.3786 | -3.0293 |

Interpretation: the learned state is informative, but it is not yet a closed, linearly forecastable Markov state at the tested resolution. This is the clearest remaining failure mode.

## Paired Endpoint Prediction

Quick 100-batch validation endpoint skill vs persistence:

| Horizon | Full-state skill | Slow-state skill |
|---:|---:|---:|
| 1 | -1.11% | -0.08% |
| 2 | -1.06% | 0.09% |
| 4 | 1.28% | 0.20% |
| 8 | -0.28% | 0.09% |
| 16 | 1.85% | 0.15% |
| 24 | -0.11% | 0.04% |
| 48 | 2.10% | 0.16% |
| Macro | 0.38% | 0.09% |

EVA5C endpoint reference:

- Macro energy skill: 28.6871%
- Horizon skills at 1/2/4/8/16 ns: 34.773%, 34.084%, 31.175%, 24.385%, 19.018%

Interpretation: EVA-LS1 is not endpoint-prediction SOTA. The model learned a better self-supervised slow representation, but the transition predictor is still close to persistence.

## Conclusion

EVA-LS1 succeeds at the specific goal of moving the framework away from handcrafted fixed CVs and toward a genuinely learned coordinate-derived state. The learned `slow_endpoint` is much stronger than the old learned CV and competitive with coordinate baselines on VAMP slow-mode recovery.

EVA-LS1 does not yet solve the full protein dynamics world-model problem. MSM classification remains below the EVA5C fixed-CV probe, linear future prediction is negative, and endpoint prediction skill is near persistence. It should be reported as a learned-state representation milestone, not as the final SOTA dynamics predictor.

For the paper, the defensible statement is:

> Joint online encoder-predictor training can recover a learned slow dynamical representation from coordinates without fixed CV state inputs, but the current transition closure remains the limiting factor for long-horizon world-model accuracy.

