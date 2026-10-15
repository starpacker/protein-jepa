#!/bin/bash
# Stage B1a Fix-2: unfreeze top 2 encoder blocks, allow joint training with CA MSE gradient flow.
# Stage A checkpoint from fix1 run (temporal_min_std=0.40 floor active).
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/tmp_claude/scratch
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp_claude/scratch
mkdir -p "$TMPDIR" artifacts/abc_v3/stage_b1a_fix2
nohup python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_fix1/checkpoint_latest.pt \
  --output artifacts/abc_v3/stage_b1a_fix2 \
  --steps 6000 --warmup-steps 200 \
  --lr 3e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 \
  --history-frames 4 \
  --unfreeze-top-n-blocks 2 \
  --encoder-lr-scale 0.05 \
  --fape-weight 1.0 \
  --ca-mse-weight 5.0 \
  --torsion-weight 0.5 \
  --distance-weight 0.2 \
  --phys-future-weight 1.0 \
  --anchor-weight 0.1 \
  --fit-normalizer-batches 10 \
  > artifacts/abc_v3/stage_b1a_fix2/train.log 2>&1 &
echo "launched B1a fix2 PID $!"
