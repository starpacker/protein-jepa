#!/usr/bin/env bash
# Stage A fix3: warm-start from stage_a_abs, add temporal contrast + raise temporal_min_std
# Key changes vs abs:
#   temporal_contrast_weight: 0.0 -> 2.0  (teaches H distance ∝ CA RMSD)
#   temporal_min_std:         0.05 -> 0.40 (prevents H from re-collapsing)
#   temporal_variance_weight: 0.02 -> 0.10 (stronger variance floor backup)
# Architecture and all other hyperparams identical to abs to make comparison clean.

set -e

OUTPUT=artifacts/abc_v3/stage_a_fix3
RESUME=artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt

mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=0 python -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest      artifacts/mdcath_manifest_200dom/train.jsonl \
  --output        "$OUTPUT" \
  --resume        "$RESUME" \
  --steps         10000 \
  --warmup-steps  500 \
  --lr            0.0001 \
  --encoder-lr-scale 0.3 \
  --weight-decay  0.02 \
  --grad-clip     1.0 \
  --batch-size    8 \
  --grad-accum    4 \
  --num-workers   6 \
  --latent-dim    128 \
  --num-factors   4 \
  --encoder-depth 4 \
  --predictor-depth 4 \
  --history-frames 8 \
  --spatial-mask-ratio 0.4 \
  --temporal-lags 0 1 2 4 \
  --recent-frac   0.5 \
  --orth-weight             0.01 \
  --factor-activity-weight  0.05 \
  --encoder-variance-weight 0.05 \
  --temporal-variance-weight 0.10 \
  --temporal-min-std        0.40 \
  --temporal-contrast-weight      2.0 \
  --temporal-contrast-rmsd-scale  2.0 \
  --struct-recon-weight     0.3 \
  --struct-recon-dim        384 \
  --struct-recon-pairs      512 \
  --ema-start     0.998 \
  --ema-end       0.9999 \
  --ema-warmup-steps 5000 \
  --log-every     50 \
  2>&1 | tee "$OUTPUT/train.log"
