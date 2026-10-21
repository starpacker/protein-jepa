#!/usr/bin/env bash
# Stage A with structural-displacement readout supervision (P0-A, 2026-10-18).
#
# A single small head (discarded after Stage A) reads the SYNTHESIZED predicted
# future-H change  compose(anchor_factors + pred_factors) - compose(anchor_factors)
# and is supervised by the TRUE future frame's Kabsch-aligned internal CA
# displacement (the SE(3)-quotient quantity B1a's decoder must reproduce).  This
# forces the temporal JEPA objective to carry real structural change instead of a
# self-referential teacher-latent delta.  Minimal added supervision: one head.
#
# Settings are NOT evidence of skill; the acceptance test is the independent H
# readout probe (scripts/probe_h_readout.py):
#     within-protein held-out true-H error <= 0.5 x persistence.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
OUTPUT=${OUTPUT:-artifacts/stage_a_disp_v2}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-artifacts/mdcath_manifest_1000dom/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-artifacts/mdcath_manifest_1000dom/val.jsonl}
mkdir -p "$OUTPUT"
python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" --output "$OUTPUT" \
  --device cuda --torch-threads 4 \
  --steps 20000 --warmup-steps 1000 --lr 0.0003 --encoder-lr-scale 0.3 \
  --weight-decay 0.02 --grad-clip 1.0 --batch-size 8 --grad-accum 2 --num-workers 8 \
  --latent-dim 128 --num-factors 4 --encoder-depth 4 --predictor-depth 4 \
  --predictor-spatial-depth 2 --history-frames 8 --frame-stride 1 \
  --paired-history-mode uniform --temporal-lags 0 \
  --paired-horizons-frames 1 2 4 8 16 24 32 48 64 96 \
  --opf-mode fixed_orthogonal --spatial-mask-ratio 0.4 \
  --temporal-contrast-weight 2 --struct-recon-weight 0.3 \
  --struct-recon-dim 384 --struct-recon-pairs 512 \
  --displacement-weight 1.0 \
  --ema-warmup-steps 5000 --val-every 500 --val-batches 32 \
  --log-every 25 --save-every 500 "$@" 2>&1 | tee "$OUTPUT/train.log"
