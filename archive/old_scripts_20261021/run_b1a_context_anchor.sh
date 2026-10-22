#!/bin/bash
# Stage B1a — oracle calibration with the invariant x equivariant decomposition
# (STAGE_AB_REDESIGN_PLAN.md §2.0 / §2.0.0).
#
# Decoding:  X_hat = anchor_frame  x  d_local(Q:=H_t, c)
#   - anchor = last observed CONTEXT frame's N/CA/C rigid-body basis (absolute
#     placement only); it never supplies internal conformation.
#   - d_local = internal displacement owned by H; supervised in the anchor-local
#     frame (SE(3) quotient).  This is the MAIN positional loss.
#   - Kabsch-aligned absolute CA is a diagnostic only (weight 0 in the loss).
#
# The run also logs, each interval, the same anchor-local displacement decoded
# from ZERO H and from the wrong-time (context) H.  H is "used" only if
# real-H clearly beats zero-H (plan §2.0 item 4).
#
# Run: bash run_b1a_context_anchor.sh
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
OUT=artifacts/abc_v3/stage_b1a_anchor_local
mkdir -p "$OUT"
exec python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/stage_a_v3_fixed/checkpoint_latest.pt \
  --output "$OUT" \
  --steps 3000 --warmup-steps 200 \
  --batch-size 4 --num-workers 4 \
  --history-frames 4 \
  --unfreeze-top-n-blocks 2 \
  --encoder-lr-scale 0.1 \
  --lr 1e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 \
  --torsion-weight 0.5 --distance-weight 0.2 \
  --relative-disp-weight 5.0 \
  --phys-future-weight 1.0 --anchor-weight 0.1 \
  --local-position-budget 12.0 \
  --fit-normalizer-batches 20 \
  --log-every 25 --save-every 500 \
  > "$OUT/train.log" 2>&1
