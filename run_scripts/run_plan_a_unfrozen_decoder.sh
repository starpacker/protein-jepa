#!/bin/bash
# Plan A variant B: ALSO unfreeze the B1a anchored decoder at a low LR.
# Control (frozen decoder) is the existing run_plan_a_hs_1000dom.sh.
# Everything else identical to terminal-only Plan A so the comparison is clean.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/plan_a_unfrozen_dec_1000dom}
RESUME=${RESUME:-}
H=${H:-8}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-2} python -u -m protein_jepa_frame.training.train_stage_b_hs \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 8000 --max-residues 192 --frame-stride 1 \
  --history-frames 4 --n-samples 6 \
  --paired-horizons-frames "$H" \
  --train-encoder --encoder-lr-scale 0.1 \
  --train-decoder --decoder-lr-scale 0.05 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
  --rollout-steps "$H" --rollout-grad-steps "$H" --temperature-k 320.0 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 6000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/plan_a_unfrozen_dec.log
