#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v18_dmc_fix
LOG=artifacts/lspwm_v44_stage_b_v18_dmc_fix/train.log
CUDA_VISIBLE_DEVICES=0 nohup /liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v18_dmc_fix.yaml \
  > "$LOG" 2>&1 &
echo "v18 PID: $!"
echo "Log: $LOG"
