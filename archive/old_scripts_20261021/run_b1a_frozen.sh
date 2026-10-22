#!/bin/bash
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
mkdir -p artifacts/abc_v3/stage_b1a_frozen
exec python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_delta/checkpoint_latest.pt \
  --output artifacts/abc_v3/stage_b1a_frozen \
  --steps 5000 --warmup-steps 200 \
  --batch-size 4 --num-workers 4 \
  --history-frames 4 \
  --unfreeze-top-n-blocks 0 \
  --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 \
  --torsion-weight 0.2 --distance-weight 0.1 \
  --phys-future-weight 0.5 --anchor-weight 0.0 \
  --fit-normalizer-batches 20 \
  --log-every 25 --save-every 500 \
  > artifacts/abc_v3/stage_b1a_frozen/train.log 2>&1
