#!/bin/bash
# Plan A + DENSE trajectory supervision.
#
# The rollout already returns the WHOLE path [anchor, H_1, ..., H_T] (T+1 frames);
# the terminal-only loss decodes just H_T and throws away T of them.  At H=8 that
# is 1 used / 9 produced.  This run decodes and scores EVERY step against the
# contiguous true future window:
#   * per-step Kabsch CA point loss  (beats persistence; per-frame encoder grad)
#   * energy score on the CA path     (distributional term, proper scoring rule)
#   * ensemble RMSF / RMSF-r / variance on the path (spread matches the true walk)
# Same Stage-A encoder (fine-tuned) + frozen B1a decoder + GLE kernel as Plan A.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/plan_a_dense_1000dom}
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
  --dense-trajectory --rollout-steps "$H" --rollout-grad-steps "$H" \
  --dense-ca-weight 1.0 --dense-es-weight 0.1 \
  --dense-rmsf-weight 0.3 --dense-rmsf-corr-weight 0.5 --dense-var-weight 0.3 \
  --train-encoder --encoder-lr-scale 0.1 --temperature-k 320.0 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 6000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/plan_a_dense_1000dom.log
