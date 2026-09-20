#!/bin/bash
# Monitor v18 training; when checkpoint_latest reaches final step, signal ready.
cd /liuyutian/protein_jepa_frame
LOG=artifacts/lspwm_v44_stage_b_v18_dmc_fix/train.log
CKPT_DIR=artifacts/lspwm_v44_stage_b_v18_dmc_fix
while true; do
  STEP=$(grep '^{' "$LOG" 2>/dev/null | tail -1 | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get('step',0))" 2>/dev/null)
  PHASE=$(grep '^{' "$LOG" 2>/dev/null | tail -1 | python3 -c "import sys,json;print(json.loads(sys.stdin.read()).get('lspwm_active_phase','?'))" 2>/dev/null)
  ALIVE=$(ps aux | grep 1250821 | grep -v grep | wc -l)
  echo "$(date +%H:%M:%S) step=$STEP phase=$PHASE alive=$ALIVE"
  if [ "$ALIVE" -eq 0 ]; then
    echo "TRAINING_DONE step=$STEP"
    break
  fi
  if [ "$STEP" -ge 5990 ] 2>/dev/null; then
    echo "REACHED_FINAL step=$STEP"
    break
  fi
  sleep 120
done
