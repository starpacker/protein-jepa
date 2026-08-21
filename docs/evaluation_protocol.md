# Evaluation Protocol

This project uses `protein_jepa_frame.eval.paper_eval` as the reproducible
evaluation entry point.

## Quick checkpoint screening

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.eval.paper_eval pipeline \
  --checkpoint state_jepa_20k=artifacts/mdcath_state_jepa_future/checkpoint_step20000.pt \
  --output-dir artifacts/paper_eval/pipeline_quick_state_jepa_step20000 \
  --preset quick \
  --batch-size 4 \
  --sequence-frames 16 \
  --readout state
```

`quick` is for checkpoint triage only. It uses fewer trajectories and the
default `window_stride=5`, so adjacent 16-frame clips overlap.

## Strict paper-oriented evaluation

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.eval.paper_eval pipeline \
  --checkpoint state_jepa_20k=artifacts/mdcath_state_jepa_future/checkpoint_step20000.pt \
  --output-dir artifacts/paper_eval/pipeline_strict_state_jepa_step20000 \
  --preset strict \
  --batch-size 4 \
  --window-stride 5 \
  --linear-window-stride 8 \
  --sequence-frames 16 \
  --readout state \
  --include-random \
  --lambda-selection train_holdout
```

The strict preset uses more trajectories, more folds, more lags, and both
`geometry_stats` and `centered_ca` deterministic baselines. Linear dynamics
uses a separate `--linear-window-stride` so the overlap sensitivity can be made
stricter than VAMP-CV/MSM extraction.

## Linear dynamics caveat

Linear dynamics is a diagnostic for how linearly predictable the learned state
space is. It should be reported by horizon. Short-horizon lag-1 scores can be
high when sequential clips overlap; medium and long lags are more informative
for durable dynamics claims.

By default, ridge regularization is selected with `--lambda-selection
train_holdout`, so the validation fold is not used to choose the best lambda.
Use `--lambda-selection validation` only to reproduce legacy quick numbers.
