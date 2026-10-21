#!/usr/bin/env bash
# Schema-v2 Stage A. Training settings are not evidence of model skill.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
OUTPUT=${OUTPUT:-artifacts/repaired_v2/full/stage_a}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-artifacts/mdcath_manifest_200dom/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-artifacts/mdcath_manifest_200dom/val.jsonl}
mkdir -p "$OUTPUT"
python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" --output "$OUTPUT" \
  --steps 15000 --warmup-steps 500 --lr 0.0003 --encoder-lr-scale 0.3 \
  --weight-decay 0.02 --grad-clip 1.0 --batch-size 2 --grad-accum 8 --num-workers 6 \
  --latent-dim 128 --num-factors 4 --encoder-depth 4 --predictor-depth 4 \
  --predictor-spatial-depth 2 --history-frames 8 --frame-stride 1 \
  --paired-history-mode uniform --temporal-lags 0 \
  --paired-horizons-frames 1 2 4 8 16 24 32 48 64 96 \
  --opf-mode fixed_orthogonal --spatial-mask-ratio 0.4 \
  --temporal-contrast-weight 2 --struct-recon-weight 0.3 \
  --struct-recon-dim 384 --struct-recon-pairs 512 \
  --ema-warmup-steps 5000 --val-every 250 --val-batches 16 \
  --log-every 50 --save-every 1000 "$@" 2>&1 | tee "$OUTPUT/train.log"
