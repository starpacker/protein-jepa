#!/bin/bash
# Stage A v2 screen run: multi-lag temporal objective + rebalanced EMA/variance.
# Short (3k steps) real-data run to validate that H stays temporally
# informative before committing to the full 25k run.
set -e
cd "$(dirname "$0")/.."
python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --output artifacts/abc_v3/stage_a_screen3 \
  --steps 3000 --warmup-steps 300 \
  --batch-size 8 --grad-accum 4 --num-workers 10 \
  --history-frames 8 \
  --temporal-lags 1 2 4 7 \
  --recent-frac 0.5 \
  --paired-horizons-frames 8 16 24 32 48 64 96 \
  --lr 3e-4 --encoder-lr-scale 0.2 \
  --temporal-variance-weight 0.05 \
  --encoder-variance-weight 0.05 \
  --factor-activity-weight 0.05 \
  --ema-start 0.998 --ema-end 0.9999 --ema-warmup-steps 3000 \
  --log-every 50 --save-every 1000 \
  "$@"
