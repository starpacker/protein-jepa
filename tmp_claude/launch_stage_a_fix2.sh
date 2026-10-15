#!/bin/bash
# Stage A Fix-2: supervised temporal contrastive loss (Kabsch RMSD → H-distance regression).
# temporal_contrast_weight=2.0 rmsd_scale=2.0 Å.
# temporal_min_std reset to 0.05 (default low), temporal_variance_weight to 0.02
# so the blunt floor doesn't fight the structured contrast signal.
# Training from scratch (no prior checkpoint).
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/tmp_claude/scratch
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp_claude/scratch
mkdir -p "$TMPDIR" artifacts/abc_v3/stage_a_fix2
nohup python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --output artifacts/abc_v3/stage_a_fix2 \
  --steps 8000 --warmup-steps 300 \
  --batch-size 8 --grad-accum 4 --num-workers 6 \
  --history-frames 8 --temporal-lags 1 2 4 7 --recent-frac 0.5 \
  --paired-horizons-frames 8 16 24 32 48 64 96 \
  --lr 3e-4 --encoder-lr-scale 0.3 \
  --struct-recon-weight 0.3 --struct-recon-pairs 512 --struct-recon-dim 384 \
  --temporal-variance-weight 0.02 --temporal-min-std 0.05 \
  --temporal-contrast-weight 2.0 --temporal-contrast-rmsd-scale 2.0 \
  --encoder-variance-weight 0.05 --factor-activity-weight 0.05 \
  --ema-start 0.998 --ema-end 0.9999 --ema-warmup-steps 5000 \
  --log-every 25 --save-every 250 \
  > artifacts/abc_v3/stage_a_fix2/train.log 2>&1 &
echo "launched Stage A fix2 PID $!"
