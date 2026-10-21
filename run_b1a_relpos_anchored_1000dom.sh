#!/bin/bash
# B1a anchored decoder RETRAIN on mdCATH-1000dom (2026-10-20).
# Model 1 = Stage-A encoder (already trained on 1000dom) + anchored decoder +
# Gaussian ensemble head.  This script retrains the anchored decoder on the
# 1000dom split so Model 1, MDGen and the from-none GLE evolver all share the
# SAME dataset, enabling an apples-to-apples comparison.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=artifacts/stage_b1a_relpos_anchored_1000dom
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-4} python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  --steps 6000 --warmup-steps 100 --lr 0.0005 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 6 --history-frames 4 --frame-stride 1 \
  --max-residues 256 --paired-horizons-frames 1 2 4 8 \
  --unfreeze-top-n-blocks 0 --encoder-lr-scale 0.1 \
  --decoder-hidden 128 --decoder-blocks 4 --decoder-heads 4 \
  --no-anchor-free --local-position-budget 12.0 \
  --relative-disp-weight 1.0 --ca-mse-weight 1.0 \
  --fape-weight 1.0 --torsion-weight 0.5 --distance-weight 0.2 \
  --phys-future-weight 1.0 --anchor-weight 0.1 \
  --diag-zero-h --val-every 250 --val-batches 8 --fit-normalizer-batches 20 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_b1a_relpos_anchored_1000dom.log
