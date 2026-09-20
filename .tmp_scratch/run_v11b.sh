#!/bin/bash
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/.tmp_scratch
pkill -f "protein_jepa_frame.cli train" 2>/dev/null
sleep 1
rm -rf /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v11_force_reconstruction
mkdir -p /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v11_force_reconstruction
cd /liuyutian/protein_jepa_frame
nohup python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v11_force_reconstruction.yaml \
  > artifacts/lspwm_v44_stage_b_v11_force_reconstruction/train.log 2>&1 &
echo "v11 PID: $!"