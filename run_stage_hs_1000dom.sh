#!/bin/bash
# Stage B "H+S (GLE) iteration" on top of Model 1 (1000dom), 2026-10-21.
# Frozen Model-1 encoder+decoder (1000dom) + trainable StageB1b GLE kernel.
# Same data pipeline + same ensemble losses + same weights as Model 1's 1000dom run
# so the comparison is apples-to-apples.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/stage_hs_1000dom}
RESUME=${RESUME:-}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-1}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-3} python -u -m protein_jepa_frame.training.train_stage_b_hs \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 8000 --max-residues 256 --frame-stride 1 \
  --history-frames 4 --n-samples 8 --cond-dims 0 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
  --rollout-steps "$ROLLOUT_STEPS" --temperature-k 320.0 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 6000 --warmup-steps 100 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_hs_1000dom.log
