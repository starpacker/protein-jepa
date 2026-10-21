#!/usr/bin/env bash
# True-H readout calibration. Run H gate before using this bundle in B1b.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp
OUTPUT=${OUTPUT:-artifacts/repaired_v2/full/stage_b1a}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-artifacts/mdcath_manifest_200dom/train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-artifacts/mdcath_manifest_200dom/val.jsonl}
STAGE_A_CKPT=${STAGE_A_CKPT:-artifacts/repaired_v2/full/stage_a/checkpoint_best.pt}
mkdir -p "$OUTPUT"
python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
  --stage-a-ckpt "$STAGE_A_CKPT" --output "$OUTPUT" --mode oracle \
  --steps 5000 --warmup-steps 200 --batch-size 4 --num-workers 4 \
  --history-frames 8 --frame-stride 1 \
  --paired-horizons-frames 1 2 4 8 16 24 32 48 64 96 \
  --unfreeze-top-n-blocks 0 --lr 0.0001 --weight-decay 0.01 \
  --fit-normalizer-batches 20 --val-every 100 --val-batches 16 \
  --log-every 25 --save-every 500 "$@" 2>&1 | tee "$OUTPUT/train.log"
