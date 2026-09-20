#!/bin/bash
cd /liuyutian/protein_jepa_frame
LOG=artifacts/lspwm_v44_stage_b_v9_load_bath/stdout.log

# Remove stale config
rm -f artifacts/lspwm_v44_stage_b_v9_load_bath/config.json

nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v9_load_bath.yaml \
  > "$LOG" 2>&1 &
echo "v9 PID: $!"
echo "Log: $LOG"