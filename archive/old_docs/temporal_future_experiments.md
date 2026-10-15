# Temporal Future JEPA Experiments

This is the next experimental branch after the strict deeptime VAMP-CV audit.
The goal is to test whether explicitly temporal state latents and future-only
masking improve dynamics metrics without using MSM labels in the training
objective.

## Implemented Code Paths

- `model.temporal_depth > 0` enables `TemporalStateEncoder`.
- `mask.mode: future` uses past frames as context and future frames as targets.
- Training optimizer now includes `temporal_encoder` parameters.
- VAMP/MSM auxiliary losses use `frame_states` when the temporal encoder exists.
- `paper_eval.py` supports `--readout mean|state|clip_state` and
  `--sequence-frames`.

## Primary Training Runs

Pure temporal future JEPA:

```bash
cd /liuyutian/protein_jepa_frame
PYTHONPATH=/liuyutian/protein_jepa_frame /liuyutian/protein_jepa/.venv/bin/python \
  -m protein_jepa_frame.cli train \
  --config configs/mdcath_temporal_future.yaml
```

Temporal future JEPA with unsupervised VAMP auxiliary only:

```bash
cd /liuyutian/protein_jepa_frame
PYTHONPATH=/liuyutian/protein_jepa_frame /liuyutian/protein_jepa/.venv/bin/python \
  -m protein_jepa_frame.cli train \
  --config configs/mdcath_temporal_future_vamp.yaml
```

Do not use MSM state loss in the primary model claim. MSM labels should remain
evaluation-only unless the experiment is explicitly marked semi-supervised.

## Evaluation Commands

Use state readout with the same temporal context size as training:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame /liuyutian/protein_jepa/.venv/bin/python \
  -m protein_jepa_frame.eval.paper_eval vamp-cv \
  --checkpoint temporal_future=artifacts/mdcath_temporal_future/checkpoint_latest.pt \
  --checkpoint temporal_future_vamp=artifacts/mdcath_temporal_future_vamp/checkpoint_latest.pt \
  --include-random \
  --output artifacts/paper_eval/vamp_cv_temporal_future.json \
  --sequence-frames 16 \
  --readout state \
  --max-trajs 50 \
  --window-stride 5 \
  --lags 1 2 5 10 20 \
  --dims 2 5 10 20
```

Trajectory-heldout MSM probe:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame /liuyutian/protein_jepa/.venv/bin/python \
  -m protein_jepa_frame.eval.paper_eval msm-probe \
  --checkpoint temporal_future=artifacts/mdcath_temporal_future/checkpoint_latest.pt \
  --checkpoint temporal_future_vamp=artifacts/mdcath_temporal_future_vamp/checkpoint_latest.pt \
  --output artifacts/paper_eval/msm_probe_temporal_future.json \
  --readout state \
  --max-clips 8000
```

Linear latent dynamics:

```bash
PYTHONPATH=/liuyutian/protein_jepa_frame /liuyutian/protein_jepa/.venv/bin/python \
  -m protein_jepa_frame.eval.paper_eval linear-dynamics \
  --checkpoint temporal_future=artifacts/mdcath_temporal_future/checkpoint_latest.pt \
  --checkpoint temporal_future_vamp=artifacts/mdcath_temporal_future_vamp/checkpoint_latest.pt \
  --output artifacts/paper_eval/linear_dynamics_temporal_future.json \
  --sequence-frames 16 \
  --readout state \
  --max-trajs 50 \
  --window-stride 5 \
  --lags 1 2 5 10 20
```

## Decision Rule

Keep the temporal future branch only if it improves at least one strict metric
without regressing the others badly:

- deeptime VAMP-CV improves over pure JEPA, random encoder, `geometry_stats`,
  and `centered_ca` baselines under paired bootstrap comparisons;
- trajectory-heldout MSM probe remains above majority baseline for most
  proteins;
- linear dynamics R2 moves closer to or above zero relative to mean-prediction
  baseline.
