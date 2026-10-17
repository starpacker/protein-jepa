#!/usr/bin/env bash
# Stage A UNIFIED (Option B) — training and evaluation on ONE window protocol.
#
# Why this exists (see REVIEWER_BRIEF.md / EVALUATION_PLAN.md §3.2):
#   The previous checkpoints (fix4, mixed) trained with a MULTISCALE context
#   ([0,32,64,96,97..100], 100 ns span) and horizons 101/150/200 ns, while the
#   evaluation harness asks for a UNIFORM K=8 context and horizons 1..96 ns.
#   Train/eval disagreed on BOTH the context geometry and the tau range, so the
#   predictor was fed out-of-distribution lags at eval time and every variant
#   collapsed to persistence.  This run removes that confound.
#
# Window protocol (identical to evaluation/configs/*.yaml):
#   context : K=8 frames, uniform stride 1  -> offsets [0..7], anchor = last
#   horizons: base grid {8,16,24,32,48,64,96} ns
#   lags    : {0,4,6,7} (counted back from the future-clip END, as in training)
#   => supervised physical gaps = {1,2,4,8,9,10,12,16,17,18,20,24,25,26,28,
#      32,41,42,44,48,57,58,60,64,89,90,92,96} ns
#      This SUPERSET covers the plan grid [1,2,4,8,16,24,32,48,64,96] exactly,
#      and evaluation uses the SAME grid -> no train/eval mismatch.
#
# Usage: bash run_stage_a_unified.sh   (set CUDA_VISIBLE_DEVICES first)
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT=${OUTPUT:-artifacts/abc_v3/stage_a_unified}
mkdir -p "$OUTPUT"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest      artifacts/mdcath_manifest_200dom/train.jsonl \
  --output        "$OUTPUT" \
  --steps         15000 \
  --warmup-steps  500 \
  --lr            0.0003 \
  --encoder-lr-scale 0.3 \
  --weight-decay  0.02 \
  --grad-clip     1.0 \
  --batch-size    2 \
  --grad-accum    8 \
  --num-workers   6 \
  --latent-dim    128 \
  --num-factors   4 \
  --encoder-depth 4 \
  --predictor-depth 4 \
  --predictor-spatial-depth 2 \
  --history-frames       8 \
  --frame-stride         1 \
  --paired-history-mode  uniform \
  --spatial-mask-ratio   0.4 \
  --temporal-lags        0 4 6 7 \
  --recent-frac          0.5 \
  --paired-horizons-frames 8 16 24 32 48 64 96 \
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
  --ema-warmup-steps 5000 \
  --log-every     50 \
  --save-every    1000 \
  2>&1 | tee "$OUTPUT/train.log"
