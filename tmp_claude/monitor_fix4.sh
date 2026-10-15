#!/bin/bash
# Monitors fix4 training and runs delta_dij probe at milestones.
# Usage: bash tmp_claude/monitor_fix4.sh &
set -e
CWD=/liuyutian/protein_jepa_frame
CKPT=$CWD/artifacts/abc_v3/stage_a_fix4/checkpoint_latest.pt
LOG=$CWD/artifacts/abc_v3/stage_a_fix4/launch.log
PROBE=$CWD/tmp_claude/probe_h_internal_dist.py
MILESTONES=(5000 8000 12000 15000)
DONE_FILE=$CWD/tmp_claude/monitor_fix4_done.txt
mkdir -p $CWD/tmp_claude
> $DONE_FILE

cd $CWD
for MILESTONE in "${MILESTONES[@]}"; do
  echo "[monitor] waiting for step $MILESTONE ..."
  while true; do
    LATEST=$(grep "^step" $LOG 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -n "$LATEST" ] && [ "$LATEST" -ge "$MILESTONE" ] 2>/dev/null; then
      break
    fi
    sleep 60
  done
  # Check process still alive
  if ! ps -p 2900679 > /dev/null 2>&1; then
    echo "[monitor] fix4 process died at step $LATEST" | tee -a $DONE_FILE
    break
  fi
  echo "[monitor] running delta_dij probe at step $MILESTONE ..." | tee -a $DONE_FILE
  CUDA_VISIBLE_DEVICES=1 timeout 1800 python $PROBE $CKPT \
    --label "fix4_s${MILESTONE}" --n-proteins 60 --steps 1500 2>&1 \
    | tee -a $CWD/tmp_claude/monitor_fix4_probes.log
  echo "---" | tee -a $DONE_FILE
done
echo "[monitor] done" | tee -a $DONE_FILE
