#!/usr/bin/env bash
# Stage A fix4: train from scratch with multiscale temporal sampling
#
# Core fix: the context window now contains frames at MULTIPLE timescales:
#   - 4 recent frames at 1 ns spacing (fast dynamics: 0,1,2,3 ns back)
#   - 4 slow frames at 32 ns spacing (slow dynamics: 4,36,68,100 ns back)
# => encoder sees 0-100 ns of trajectory in every sample
#
# Paired future targets at 101,150,200 ns => prediction spans full MD timescale
#
# temporal_contrast_rmsd_scale=5.0 (appropriate for 32-100 ns lags, ~2-6 A RMSD)
# temporal_lags [1,2,4,7] within the 8-frame window => 1-7 ns AND 32-100 ns pairs
# No warm-start from abs (contaminated with 7 ns short-lag collapse)

set -e

OUTPUT=artifacts/abc_v3/stage_a_fix4
mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES=0 python -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest      artifacts/mdcath_manifest_200dom/train.jsonl \
  --output        "$OUTPUT" \
  --steps         15000 \
  --warmup-steps  500 \
  --lr            0.0003 \
  --encoder-lr-scale 0.3 \
  --weight-decay  0.02 \
  --grad-clip     1.0 \
  --batch-size    4 \
  --grad-accum    4 \
  --num-workers   6 \
  --latent-dim    128 \
  --num-factors   4 \
  --encoder-depth 4 \
  --predictor-depth 4 \
  --history-frames       8 \
  --frame-stride         1 \
  --paired-history-mode  multiscale \
  --paired-recent-frames 4 \
  --paired-slow-stride   32 \
  --spatial-mask-ratio   0.4 \
  --temporal-lags 0 1 2 4 \
  --recent-frac   0.5 \
  --paired-horizons-frames 101 150 200 \
  --orth-weight             0.01 \
  --factor-activity-weight  0.05 \
  --encoder-variance-weight 0.05 \
  --encoder-min-std         0.1 \
  --temporal-variance-weight 0.10 \
  --temporal-min-std        0.40 \
  --temporal-contrast-weight      2.0 \
  --temporal-contrast-rmsd-scale  5.0 \
  --struct-recon-weight     0.3 \
  --struct-recon-dim        384 \
  --struct-recon-pairs      512 \
  --ema-start     0.998 \
  --ema-end       0.9999 \
  --ema-warmup-steps 5000 \
  --log-every     50 \
  2>&1 | tee "$OUTPUT/train.log"
