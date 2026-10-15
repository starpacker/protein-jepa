#!/bin/bash
# Stage B1a — same-frame oracle smoke test.
#
# Anchor = target frame, so oracle_ca_mse should drop to near-zero (~0-5 Å²)
# within 500 steps if the decoder + direct_h_position head are functional.
# If it does not converge, the decoder architecture is broken independent of H quality.
# Once oracle_ca_mse < 5 Å², proceed to run_b1a_unfrozen.sh for the real calibration.
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
mkdir -p artifacts/abc_v3/stage_b1a_sameframe
exec python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt \
  --output artifacts/abc_v3/stage_b1a_sameframe \
  --steps 1000 --warmup-steps 100 \
  --batch-size 4 --num-workers 4 \
  --history-frames 4 \
  --unfreeze-top-n-blocks 0 \
  --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 --ca-mse-weight 1.0 \
  --torsion-weight 0.2 --distance-weight 0.1 \
  --phys-future-weight 0.0 --anchor-weight 0.0 \
  --same-frame-oracle \
  --fit-normalizer-batches 20 \
  --log-every 25 --save-every 250 \
  > artifacts/abc_v3/stage_b1a_sameframe/train.log 2>&1
