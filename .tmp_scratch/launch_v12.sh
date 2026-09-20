#!/bin/bash
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/.tmp_scratch
cd /liuyutian/protein_jepa_frame
pkill -f "protein_jepa_frame.cli" 2>/dev/null
sleep 2
mkdir -p artifacts/lspwm_v44_stage_b_v12_improved
nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v12_improved.yaml \
  > artifacts/lspwm_v44_stage_b_v12_improved/train.log 2>&1 &
echo "v12 PID: $!"