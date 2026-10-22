#!/usr/bin/env bash
# Stage A: mdCATH + ATLAS mixed, unified 1 ns/frame absolute-time contract.
#
# Fixes applied vs previous version:
#   Fix 1: --predictor-spatial-depth 2  — spatial residue attention stage added
#   Fix 2: delta_target defaults to False in code (absolute target + struct_recon coherent)
#   Fix 3: --temporal-lags 0 1 2  — future-clip offsets counted BACK from the
#           horizon end (0 = anchor+full horizon).  Stage A now predicts frames
#           OUTSIDE the context window (target drawn from future_*), so training
#           matches B1a/eval exactly.  frame-stride 1 keeps the nonoverlap
#           contract (span=29 < horizons=32).
#   Fix 4: --temporal-contrast-weight 2.0 (structural supervision ON)
#   Fix 5: --temporal-min-std 0.10 — gentle backstop, not a noise-injection driver
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest      artifacts/mixed_manifest/train.jsonl \
  --output        artifacts/abc_v3/stage_a_mixed \
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
  --ema-warmup-steps 5000 \
  --log-every     50 \
  --save-every    1000 \
  2>&1 | tee artifacts/abc_v3/stage_a_mixed/train.log
