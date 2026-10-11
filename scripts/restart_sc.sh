#!/bin/bash
pkill -f "stage_c_compact" 2>/dev/null
sleep 2
> /liuyutian/protein_jepa_frame/logs/sc_compact.log
CUDA_VISIBLE_DEVICES=1 nohup python -u -m protein_jepa_frame.cli train --config configs/mdcath_stage_c_compact.yaml --stage C > /liuyutian/protein_jepa_frame/logs/sc_compact.log 2>&1 &
PID=$!
echo "Started PID: $PID"
sleep 15
tail -3 /liuyutian/protein_jepa_frame/logs/sc_compact.log

