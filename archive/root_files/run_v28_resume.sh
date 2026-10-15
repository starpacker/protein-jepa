#!/bin/bash
# Launch v28 training with OOM fix and resume from step 1100
set -e

export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/tmp

cd /liuyutian/protein_jepa_frame
LOGDIR="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v28_per_residue"

PYTHONPATH="/liuyutian/protein_jepa_frame" exec python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v28_per_residue.yaml \
  >> "${LOGDIR}/stdout.log" 2>&1