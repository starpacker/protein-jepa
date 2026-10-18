#!/usr/bin/env bash
# From-none Stage B, held-out-TEMPERATURE split: train on T in {320,348,379,413},
# hold out T=450 entirely. Selection on held-out proteins (val_not450).
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=5
OUTPUT=${OUTPUT:-artifacts/fromnone/stage_cb1b_temp}
TRAIN=${TRAIN:-artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl}
VAL=${VAL:-artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl}
mkdir -p "$OUTPUT"
python -u -m protein_jepa_frame.training.train_stage_cb1b \
  --manifest "$TRAIN" --val-manifest "$VAL" \
  --output "$OUTPUT" \
  --steps 30000 --warmup-steps 1000 --lr 3e-4 --encoder-lr-scale 1.0 \
  --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 6 --max-residues 192 --device cuda \
  --history-frames 8 --multi-lag --frame-stride 1 --step-ps 1000 \
  --latent-dim 128 --encoder-depth 4 --encoder-heads 4 --dropout 0.1 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --history-hidden 128 --coupling-scale 1e-3 --diffusion-scale 1e-4 \
  --c-init-std 0.1 --gamma-scale 2e-4 --gamma-floor 2e-5 --max-gamma-step 1.9 \
  --force-clip 1.0 --tail-init 1.0 \
  --decoder-hidden 256 --decoder-depth 3 --max-update 12.0 \
  --es-weight 0.1 --ca-weight 1.0 --phys-weight 0.1 \
  --es-num-samples 3 \
  --validation-every 500 --validation-batches 24 --validation-horizon-steps 16 \
  --save-every 1000 --log-every 50 --seed 43 \
  "$@" 2>&1 | tee "$OUTPUT/train.log"
