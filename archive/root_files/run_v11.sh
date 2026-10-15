#!/bin/bash
cd /liuyutian/protein_jepa_frame
LOG=artifacts/lspwm_v44_stage_b_v11_force_reconstruction/train.log

nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v11_force_reconstruction.yaml \
  > "$LOG" 2>&1 &
echo "v11 PID: $!"
echo "Log: $LOG"
