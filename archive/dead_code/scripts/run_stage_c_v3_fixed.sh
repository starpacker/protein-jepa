#!/bin/bash
cd /liuyutian/protein_jepa_frame
CUDA_VISIBLE_DEVICES=0 nohup python -u -m protein_jepa_frame.cli train --config configs/full_abc_stage_c_v3.yaml --stage C > logs/stage_c_v3_fixed.log 2>&1 &
echo "Stage C PID: $!"
