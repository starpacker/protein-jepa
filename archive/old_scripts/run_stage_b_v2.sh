#!/usr/bin/env bash
cd /liuyutian/protein_jepa_frame
export PYTHONPATH=/liuyutian/protein_jepa_frame
exec python -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_single_step_thermodynamics_v2.yaml \
  > /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_single_step_thermodynamics_v2/training.log 2>&1