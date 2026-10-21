#!/bin/bash
# Gaussian ensemble head RETRAIN on mdCATH-1000dom (2026-10-20).
# Model 1 = Stage-A encoder (1000dom) + anchored decoder (1000dom) + ensemble head.
# Trains ONLY the new GaussianHHead (encoder+decoder frozen), matching the
# train_stage_ensemble recipe of the prior dualA run but on the 1000dom split so
# Model 1 / MDGen / from-none GLE all share one dataset.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/stage_ensemble_1000dom}
RESUME=${RESUME:-}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-7} python -u -m protein_jepa_frame.training.train_stage_ensemble \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 8000 --max-residues 256 --frame-stride 1 \
  --history-frames 4 --paired-horizons-frames 1 2 4 8 \
  --head-hidden 128 --n-samples 8 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 --cond-dims 0 \
  --steps 4000 --warmup-steps 100 --lr 0.001 --weight-decay 0.0 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_ensemble_1000dom.log
