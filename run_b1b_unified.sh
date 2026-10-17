#!/bin/bash
# Stage B1b UNIFIED — GLE (H,S) dynamics on the unified Stage-A/B1a base,
# WITH the bath fix from B1B_DIAGNOSIS_AND_PLAN.md 目标 2:
#   coupling_scale 1e-3 -> 0.1, diffusion_scale 1e-4 -> 1e-2, live c_net init.
# Previously B_h P B_h^T was ~1e6x smaller than R, so the bath was dead and the
# rollout barely moved (MSD ~0.009 A^2).  Acceptance: gate full-nobath and
# full-nohist bootstrap CIs exclude 0.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/tmp CLAUDE_CODE_TMPDIR=/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT=${OUTPUT:-artifacts/abc_v3/stage_b1b_unified}
mkdir -p "$OUTPUT"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python -u -m protein_jepa_frame.training.train_stage_b1b \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --stage-a-ckpt artifacts/abc_v3/stage_a_unified/checkpoint_latest.pt \
  --b1a-ckpt artifacts/abc_v3/stage_b1a_unified/checkpoint_latest.pt \
  --output "$OUTPUT" \
  --steps 20000 --warmup-steps 500 \
  --batch-size 8 --num-workers 8 \
  --history-frames 8 --horizon-steps 8 \
  --latent-dim 128 --bath-dim 8 --energy-depth 3 \
  --step-ps 1000.0 \
  --coupling-scale 0.1 \
  --diffusion-scale 1.0e-2 \
  --c-init-std 1.0e-2 \
  --nll-weight 1.0 --es-weight 0.1 --es-warmup-steps 2000 \
  --es-num-samples 4 --es-horizon-steps 8 \
  --rmsd-weight 1.0 --rmsd-horizon-steps 4 \
  --lr 1e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --log-every 20 --save-every 1000 \
  2>&1 | tee "$OUTPUT/train.log"
