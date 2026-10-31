#!/usr/bin/env bash
# Audited path-distribution fine-tuning. All outputs are separate from v3.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR="$PWD/.tmpwork"
OUT=${OUT:-artifacts/lasp_v6_path}
INIT=${INIT:-artifacts/lasp_v3_velocity/checkpoint_best.pt}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-0} python -u -m protein_jepa_frame.training.train_stage_b_hs \
  --manifest artifacts/mdcath_manifest_full/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_full/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --init-from "$INIT" --output "$OUT" --device cuda --torch-threads 4 \
  --max-train-samples 8000 --max-residues 192 --history-frames 4 --frame-stride 1 \
  --n-samples 4 --multi-horizon --paired-horizons-frames 1 4 16 64 \
  --multi-horizon-max-grad 16 --path-ensemble --per-sample-temperature \
  --ens-rmsf-corr-weight 0.5 --contact-temperature 0.5 \
  --train-encoder --encoder-lr-scale 0.1 --train-decoder --decoder-lr-scale 0.1 \
  --roundtrip-weight 0.5 --roundtrip-frames 2 \
  --rmsf-weight 1 --contact-weight 10 --dist-weight 0.3 --pos-weight 0 \
  --bath-coupling velocity --max-step-norm 20 --bath-dim 8 \
  --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 1000 --warmup-steps 100 --lr 1e-4 --weight-decay 0.01 --grad-clip 1 \
  --batch-size 2 --num-workers 2 --val-every 100 --val-batches 4 \
  --log-every 10 --save-every 100 --seed 42 \
  "$@" 2>&1 | tee "$OUT/train.log"
