#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v19_bath_aware
LOG=artifacts/lspwm_v44_stage_b_v19_bath_aware/train.log
CUDA_VISIBLE_DEVICES=1 nohup /liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v19_bath_aware.yaml \
  > "$LOG" 2>&1 &
echo "v19 PID: $!"
echo "Log: $LOG"
