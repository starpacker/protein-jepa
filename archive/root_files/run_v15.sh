#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v15_perframe_drift
LOG=artifacts/lspwm_v44_stage_b_v15_perframe_drift/train.log
CUDA_VISIBLE_DEVICES=0 nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v15_perframe_drift.yaml \
  > "$LOG" 2>&1 &
echo "v15 PID: $!"
echo "Log: $LOG"
