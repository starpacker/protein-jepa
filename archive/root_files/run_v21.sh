#!/bin/bash
cd /liuyutian/protein_jepa_frame
mkdir -p artifacts/lspwm_v44_stage_b_v21_thermo_helmholtz_mori
LOG=artifacts/lspwm_v44_stage_b_v21_thermo_helmholtz_mori/train.log
CUDA_VISIBLE_DEVICES=1 nohup /liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v21_thermo_helmholtz_mori.yaml \
  > "$LOG" 2>&1 &
echo "v21 PID: $!"
echo "Log: $LOG"
