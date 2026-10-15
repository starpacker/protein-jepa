#!/bin/bash
# Stage B1a — controlled unfreezing, balanced loss weights.
#
# Run AFTER run_b1a_sameframe.sh confirms oracle_ca_mse < 5 Å² (decoder works).
# Top-2 encoder blocks unfrozen so CA MSE gradient flows into H.
# ca_mse_weight=1.0 keeps all terms in the same ballpark so nothing dominates.
# anchor_weight=0.1 keeps H from drifting too far from Stage A representations.
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
mkdir -p artifacts/abc_v3/stage_b1a_unfrozen
exec python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_abs/checkpoint_latest.pt \
  --output artifacts/abc_v3/stage_b1a_unfrozen \
  --steps 5000 --warmup-steps 200 \
  --batch-size 4 --num-workers 4 \
  --history-frames 4 \
  --unfreeze-top-n-blocks 2 \
  --encoder-lr-scale 0.1 \
  --lr 1e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --fape-weight 1.0 --ca-mse-weight 1.0 \
  --torsion-weight 0.2 --distance-weight 0.1 \
  --phys-future-weight 0.5 --anchor-weight 0.1 \
  --resume artifacts/abc_v3/stage_b1a_sameframe/checkpoint_latest.pt \
  --fit-normalizer-batches 20 \
  --log-every 25 --save-every 500 \
  > artifacts/abc_v3/stage_b1a_unfrozen/train.log 2>&1
