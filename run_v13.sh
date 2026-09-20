#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v13_dt_fix
LOG=artifacts/lspwm_v44_stage_b_v13_dt_fix/train.log

CUDA_VISIBLE_DEVICES=0 nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v13_dt_fix.yaml \
  > "$LOG" 2>&1 &
echo "v13 PID: $!"
echo "Log: $LOG"
