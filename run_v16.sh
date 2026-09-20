#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v16_dissipative_bath
LOG=artifacts/lspwm_v44_stage_b_v16_dissipative_bath/train.log
CUDA_VISIBLE_DEVICES=0 nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v16_dissipative_bath.yaml \
  > "$LOG" 2>&1 &
echo "v16 PID: $!"
echo "Log: $LOG"
