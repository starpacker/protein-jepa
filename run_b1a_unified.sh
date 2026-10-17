#!/bin/bash
# Stage B1a UNIFIED — structure-align the unified Stage-A encoder.
# Same recipe as the old unfrozen run (oracle decoder + top-2 blocks unfrozen),
# but on the unified Stage-A checkpoint so train/eval windows agree everywhere.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT=${OUTPUT:-artifacts/abc_v3/stage_b1a_unified}
mkdir -p "$OUTPUT"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_unified/checkpoint_latest.pt \
  --output "$OUTPUT" \
  --steps 5000 --warmup-steps 200 \
  --batch-size 4 --num-workers 4 \
  --history-frames 8 \
  --unfreeze-top-n-blocks 2 \
  --encoder-lr-scale 0.1 \
  --lr 1e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 \
  --torsion-weight 0.2 --distance-weight 0.1 \
  --phys-future-weight 0.5 --anchor-weight 0.1 \
  --fit-normalizer-batches 20 \
  --log-every 25 --save-every 500 \
  2>&1 | tee "$OUTPUT/train.log"
