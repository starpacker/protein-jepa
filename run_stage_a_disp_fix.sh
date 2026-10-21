#!/bin/bash
# P0-A fix run (modifications 1+2): displacement head reads ONLY synth_delta_H,
# anchor from ONLINE encoder (grad), plus per-residue direction cosine term.
# 2000 steps to evaluate whether H's readout improves past the gate.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=artifacts/stage_a_disp_fix
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-6} python -u -m protein_jepa_frame.training.train_stage_a_v2 \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --output "$OUT" --device cuda --torch-threads 4 \
  --steps 2000 --warmup-steps 400 --lr 0.0003 --encoder-lr-scale 0.3 \
  --weight-decay 0.02 --grad-clip 1.0 --batch-size 8 --grad-accum 2 --num-workers 8 \
  --latent-dim 128 --num-factors 4 --encoder-depth 4 --predictor-depth 4 \
  --predictor-spatial-depth 2 --history-frames 8 --frame-stride 1 \
  --paired-history-mode uniform --temporal-lags 0 \
  --paired-horizons-frames 1 2 4 8 16 24 32 48 64 96 \
  --opf-mode fixed_orthogonal --spatial-mask-ratio 0.4 \
  --temporal-contrast-weight 2 --struct-recon-weight 0.3 \
  --struct-recon-dim 384 --struct-recon-pairs 512 \
  --displacement-weight 1.0 --displacement-direction-weight 1.0 \
  --ema-warmup-steps 1000 --val-every 500 --val-batches 32 \
  --log-every 25 --save-every 250 --seed 0 \
  2>&1 | tee .tmpwork/logs/stage_a_disp_fix.log
