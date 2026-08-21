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
