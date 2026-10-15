#!/bin/bash
set -e
# Launch both Stage-A matched-ablation runs concurrently on two GPUs
nohup python3 -m protein_jepa_frame.cli train --config configs/lspwm_v44_stage_a_opf_k4.yaml > nohup_opf_k4_stagea.log 2>&1 &
PID_K4=$!
nohup python3 -m protein_jepa_frame.cli train --config configs/lspwm_v44_stage_a_opf_k1_control.yaml > nohup_opf_k1_stagea.log 2>&1 &
PID_K1=$!
echo "Launched K4 (PID $PID_K4) and K1-control (PID $PID_K1)"
wait $PID_K4; echo "K4 done rc=$?"
wait $PID_K1; echo "K1 done rc=$?"
