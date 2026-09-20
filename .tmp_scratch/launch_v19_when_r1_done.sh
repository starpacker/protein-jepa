#!/bin/bash
cd /liuyutian/protein_jepa_frame
R1_PID=1254883
R1_OUT=artifacts/reviewer_experiments/representation_closure_v2/r1_coord_baselines_256d_v2.json
R1_LOG=artifacts/reviewer_experiments/representation_closure_v2/r1_coord_256d_pca_fix.log
LAUNCH_LOG=.tmp_scratch/v19_autolaunch.log
echo "$(date) waiting for R1 to finish..." >> $LAUNCH_LOG

while true; do
  VAMP=$(grep -c "^  vamp " "$R1_LOG" 2>/dev/null)
  R1_ALIVE=$(ps -p $R1_PID >/dev/null 2>&1 && echo 1 || echo 0)

  # R1 done when: json exists, OR process died, OR vamp=20
  if [ -f "$R1_OUT" ] || [ "$R1_ALIVE" = "0" ] || [ "$VAMP" -ge 20 ] 2>/dev/null; then
    echo "$(date) R1 done/exited (vamp=$VAMP alive=$R1_ALIVE json=$(ls $R1_OUT 2>/dev/null && echo yes || echo no))" >> $LAUNCH_LOG
    # kill R1 if still alive (json already written or vamp done)
    [ "$R1_ALIVE" = "1" ] && kill $R1_PID 2>/dev/null && echo "$(date) killed R1 PID $R1_PID" >> $LAUNCH_LOG
    sleep 3
    # launch v19 on GPU:1
    mkdir -p artifacts/lspwm_v44_stage_b_v19_bath_aware
    LOG=artifacts/lspwm_v44_stage_b_v19_bath_aware/train.log
    CUDA_VISIBLE_DEVICES=1 nohup /liuyutian/protein_jepa/.venv/bin/python -m protein_jepa_frame.cli train \
      --config configs/lspwm_v44_stage_b_v19_bath_aware.yaml \
      > "$LOG" 2>&1 &
    V19_PID=$!
    echo "$(date) v19 launched PID=$V19_PID log=$LOG" >> $LAUNCH_LOG
    break
  fi
  echo "$(date) waiting... vamp=$VAMP" >> $LAUNCH_LOG
  sleep 60
done
