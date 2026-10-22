#!/bin/bash
# Stage B1a — ANCHOR-FREE decoder + PREDICTED future H (change C).
#
# Two redesign fixes under test:
#   * decoder_anchor_free: H predicts the whole internal CA cloud; the anchor is
#     only a post-hoc rigid placement (R0, c0).  No "anchor + tiny d_local"
#     shortcut, so real-H must beat zero-H.
#   * --use-predicted-h: decode predictor(context_H), not encode(target).  This
#     calibrates the real H -> H_future -> structure pipeline.
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
mkdir -p artifacts/abc_v3/stage_b1a_anchorfree
exec python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt ${STAGE_A_CKPT:-artifacts/abc_v3/stage_a_mixed/checkpoint_latest.pt} \
  --output artifacts/abc_v3/stage_b1a_anchorfree \
  --steps ${STEPS:-3000} --warmup-steps 100 \
  --batch-size 4 --num-workers 4 \
  --history-frames 8 \
  --unfreeze-top-n-blocks 0 \
  --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 \
  --torsion-weight 0.2 --distance-weight 0.1 \
  --relative-disp-weight 1.0 \
  --ca-mse-weight 1.0 \
  --phys-future-weight 0.5 --anchor-weight 0.0 \
  --coord-scale 40.0 \
  --fit-normalizer-batches 20 \
  --log-every 10 --save-every 500 \
  > artifacts/abc_v3/stage_b1a_anchorfree/train.log 2>&1
