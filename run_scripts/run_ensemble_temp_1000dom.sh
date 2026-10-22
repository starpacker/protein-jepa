#!/bin/bash
# Model-1 ensemble head RETRAIN with TEMPERATURE conditioning on the held-out-T
# split (T in {320,348,379,413}, T=450 unseen). cond-dims=2 feeds the head
# [log(horizon_time_ps)/log(1e4), (T-300)/150] so the SAME frozen encoder/decoder
# can express how much flexibility to expect at a given timescale AND temperature.
#
# The original stage_ensemble_1000dom run used --cond-dims 0 (no temperature
# input at all), so it *cannot* generalize over temperature by construction.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/stage_ensemble_1000dom_temp_cond}
RESUME=${RESUME:-}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-7} python -u -m protein_jepa_frame.training.train_stage_ensemble \
  --manifest artifacts/mdcath_manifest_1000dom_temp/train_not450.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom_temp/val_not450.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 8000 --max-residues 256 --frame-stride 1 \
  --history-frames 4 --paired-horizons-frames 1 2 4 8 \
  --head-hidden 128 --n-samples 8 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 --cond-dims 2 \
  --steps 4000 --warmup-steps 100 --lr 0.001 --weight-decay 0.0 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_ensemble_1000dom_temp_cond.log
