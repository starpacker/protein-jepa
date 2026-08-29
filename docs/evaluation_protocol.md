# Evaluation Protocol

## V2.3 Paired Forecast And Calibration

V2.3 primary paired evaluation uses raw offsets `48/96/192`, which do not
overlap a 24-frame, stride-2 context window. Run the paired evaluator on a held
out mdCATH manifest after the 2k gate:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame \
/liuyutian/protein_jepa/.venv/bin/python scripts/evaluate_paired_dynamics.py \
  --config configs/phase2_v2_3_s_gaussian_2k.yaml \
  --checkpoint artifacts/phase2_v2_3_s_gaussian_2k/checkpoint_step2000.pt \
  --manifest /liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl \
  --output artifacts/paper_eval/phase2_v2_3_s_gaussian_2k_paired.json \
  --max-batches 500 --batch-size 4
```

For representation evaluation, use `endpoint` for the bidirectional offline
tower and `causal_endpoint` for the causal forecast tower. Report both; neither
may substitute for the other in the promotion table.

This project uses `protein_jepa_frame.eval.paper_eval` as the reproducible
evaluation entry point.

## Quick checkpoint screening

The current default protocol inherits `frame_stride` and `window_stride` from
the checkpoint config. When lags are not supplied, it selects the first lag
whose two contextual windows share no saved frames. For the Phase 1 champion,
24 frames, `frame_stride=2`, and `window_stride=8` produce a 47-saved-frame
context and default lags `[6, 12, 24]`.

Trajectory selection is protein-stratified round-robin. `max_trajs=10` now
means ten proteins before adding a second trajectory from any protein.

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
checkpoint sampling stride. The selected forecast lag, rather than necessarily
the extraction stride, enforces non-overlapping input contexts.

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

## Score Semantics

VAMP output reports the raw cross-validated VAMP-2 score and a kinetic
normalization `(score - 1) / kinetic_dim`, which removes the trivial constant
mode. Older artifacts used `score / dim`; those normalized values are not
directly comparable and could exceed one.

MSM ridge regularization is selected on a train-trajectory group holdout. The
outer validation trajectories are used exactly once for reporting accuracy,
balanced accuracy, and macro-F1.
