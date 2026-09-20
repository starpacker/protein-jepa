#!/bin/bash
cd /liuyutian/protein_jepa_frame
R1_PID=1254883
V18_PID=1250821
R1_LOG=artifacts/reviewer_experiments/representation_closure_v2/r1_coord_256d_pca_fix.log
R1_OUT=artifacts/reviewer_experiments/representation_closure_v2/r1_coord_baselines_256d_v2.json
V18_LOG=artifacts/lspwm_v44_stage_b_v18_dmc_fix/train.log
STALL=0
PREV_U=0
for i in $(seq 1 120); do
  # --- R1 status ---
  if [ -f "$R1_OUT" ]; then R1S="DONE(json written)"; R1ALIVE=0
  elif ps -p $R1_PID >/dev/null 2>&1; then
    U=$(cat /proc/$R1_PID/stat 2>/dev/null | awk '{print $14}')
    VAMP=$(grep -c "^  vamp " "$R1_LOG" 2>/dev/null)
    TICA=$(grep -c "^  tica " "$R1_LOG" 2>/dev/null)
    if [ "$U" = "$PREV_U" ]; then STALL=$((STALL+1)); else STALL=0; fi
    PREV_U=$U
    R1S="running tica=$TICA/20 vamp=$VAMP/20 stall=$STALL"
    R1ALIVE=1
  else R1S="EXITED(no json - failed)"; R1ALIVE=0; fi
  # --- v18 status ---
  if ps -p $V18_PID >/dev/null 2>&1; then
    ST=$(grep '^{' "$V18_LOG" 2>/dev/null | tail -1 | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d.get('step'),d.get('lspwm_active_phase'))" 2>/dev/null)
    V18S="running step=$ST"; V18ALIVE=1
  else V18S="TRAINING_DONE"; V18ALIVE=0; fi
  echo "$(date +%H:%M:%S) R1[$R1S] V18[$V18S]"
  # stall kill: 5 consecutive stalled samples (~5 min)
  if [ "$R1ALIVE" = "1" ] && [ "$STALL" -ge 5 ]; then
    echo "R1 STALLED -> killing $R1_PID"
    kill -9 $R1_PID 2>/dev/null
    R1ALIVE=0
  fi
  # exit when both settled
  if [ "$R1ALIVE" = "0" ] && [ "$V18ALIVE" = "0" ]; then echo "BOTH_SETTLED"; break; fi
  sleep 60
done
