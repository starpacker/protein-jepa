#!/usr/bin/env bash
# Stage A: FULL mdCATH, pure mdCATH (no ATLAS), all 5 temperatures (320-450K).
# Split: artifacts/mdcath_manifest/ (domain-level, 4318 train / 539 val / 541 test domains).
# Plan C: full-data run.  Throughput-optimised config:
#   batch-size 8 with paired_all_horizons=3  -> 24 pairs/optimizer-step,
#   grad-accum 1 (the old batch4 x accum4 did 48 pairs but at ~34 s/step).
#   Here ~1.3 s/step measured, so 130k steps ~= 2 days on one A800.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest      artifacts/mdcath_manifest/train.jsonl \
  --output        artifacts/abc_v3/stage_a_mdcath_full \
  --steps         130000 \
  --warmup-steps  2000 \
  --lr            0.0003 \
  --encoder-lr-scale 0.3 \
  --weight-decay  0.02 \
  --grad-clip     1.0 \
  --batch-size    8 \
  --grad-accum    1 \
  --num-workers   8 \
  --latent-dim    128 \
  --num-factors   4 \
  --encoder-depth 4 \
  --predictor-depth 4 \
  --predictor-spatial-depth 2 \
  --history-frames       8 \
  --frame-stride         1 \
  --paired-history-mode  multiscale \
  --paired-recent-frames 4 \
  --paired-slow-stride   8 \
  --spatial-mask-ratio   0.4 \
  --temporal-lags 0 1 2 \
  --recent-frac   0.5 \
  --paired-horizons-frames 32 48 64 \
  --orth-weight             0.01 \
  --factor-activity-weight  0.05 \
  --encoder-variance-weight 0.05 \
  --encoder-min-std         0.1 \
  --temporal-variance-weight 0.05 \
  --temporal-min-std        0.10 \
  --temporal-contrast-weight      2.0 \
  --temporal-contrast-rmsd-scale  5.0 \
  --struct-recon-weight     0.3 \
  --struct-recon-dim        384 \
  --struct-recon-pairs      512 \
  --ema-start     0.998 \
  --ema-end       0.9999 \
  --ema-warmup-steps 20000 \
  --log-every     100 \
  --save-every    2000 \
  2>&1 | tee artifacts/abc_v3/stage_a_mdcath_full/train.log
