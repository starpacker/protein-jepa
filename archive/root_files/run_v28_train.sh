#!/bin/bash
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTHONPATH=/liuyutian/protein_jepa_frame
mkdir -p /liuyutian/protein_jepa_frame/tmp
cd /liuyutian/protein_jepa_frame
python -m protein_jepa_frame.cli train \
  --config configs/lspwm_v44_stage_b_v28_per_residue.yaml \
  > artifacts/lspwm_v44_stage_b_v28_per_residue/stdout.log 2>&1
