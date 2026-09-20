#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v14_memory_window
LOG=artifacts/lspwm_v44_stage_b_v14_memory_window/train.log

# v14 on GPU 1 (v13 occupies GPU 0). CUDA_VISIBLE_DEVICES=1 remaps device 1 -> cuda:0 in-process.
CUDA_VISIBLE_DEVICES=1 nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v14_memory_window.yaml \
  > "$LOG" 2>&1 &
echo "v14 PID: $!"
echo "Log: $LOG"
