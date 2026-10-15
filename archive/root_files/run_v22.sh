#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v22_kde_force
LOG=artifacts/lspwm_v44_stage_b_v22_kde_force/train.log
CUDA_VISIBLE_DEVICES=0 nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v22_kde_force.yaml \
  > "$LOG" 2>&1 &
echo "v22 PID: $!"
echo "Log: $LOG"
