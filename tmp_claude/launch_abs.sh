#!/bin/bash
# Stage A Fix-1: temporal_min_std raised 0.05→0.40 so γ²=0.16 >> natural
# per-step variance ~0.05 → hinge fires from step 1.
# temporal_variance_weight raised 0.02→0.10 for real gradient pressure.
# struct_recon uses context_H_full (unmasked) — bug fixed in prior commit.
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/tmp_claude/scratch
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp_claude/scratch
mkdir -p "$TMPDIR" artifacts/abc_v3/stage_a_fix1
pkill -f train_stage_a_v2 2>/dev/null; sleep 2
nohup python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --output artifacts/abc_v3/stage_a_fix1 \
  --steps 8000 --warmup-steps 300 \
  --batch-size 8 --grad-accum 4 --num-workers 6 \
  --history-frames 8 --temporal-lags 1 2 4 7 --recent-frac 0.5 \
  --paired-horizons-frames 8 16 24 32 48 64 96 \
  --lr 3e-4 --encoder-lr-scale 0.3 \
  --struct-recon-weight 0.3 --struct-recon-pairs 512 --struct-recon-dim 384 \
  --temporal-variance-weight 0.10 --temporal-min-std 0.40 \
  --encoder-variance-weight 0.05 --factor-activity-weight 0.05 \
  --ema-start 0.998 --ema-end 0.9999 --ema-warmup-steps 5000 \
  --log-every 25 --save-every 250 > artifacts/abc_v3/stage_a_fix1/train.log 2>&1 &
echo "launched PID $!"
