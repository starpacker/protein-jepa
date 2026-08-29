#!/usr/bin/env bash
set -euo pipefail

REPO="/liuyutian/protein_jepa_frame"
PY="/liuyutian/protein_jepa/.venv/bin/python"
CONFIG="$REPO/configs/phase2_a2_mdcath_split_state_motion.yaml"
RUN_DIR="$REPO/artifacts/phase2_a2_mdcath_split_state_motion"
EVAL_DIR="$REPO/artifacts/paper_eval/phase2_a2_5k_honest_eval"
TRAIN_PID="${1:-}"

if [[ -z "$TRAIN_PID" ]]; then
  TRAIN_PID="$(pgrep -f "phase2_a2_mdcath_split_state_motion.yaml" | head -n 1 || true)"
fi

mkdir -p "$RUN_DIR" "$EVAL_DIR"
echo "$(date -Is) watcher started; train_pid=${TRAIN_PID:-none}"

while [[ ! -f "$RUN_DIR/checkpoint_latest.pt" ]]; do
  sleep 60
done

# Give torch.save a short window to finish the latest checkpoint atomically.
sleep 30

STEP="$("$PY" - <<'PY'
from pathlib import Path
import torch
ckpt = torch.load(Path("/liuyutian/protein_jepa_frame/artifacts/phase2_a2_mdcath_split_state_motion/checkpoint_latest.pt"), map_location="cpu", weights_only=True)
print(ckpt.get("step", "unknown"))
PY
)"
echo "$(date -Is) checkpoint_latest.pt detected at step=${STEP}"

if [[ "$STEP" != "unknown" && ! -f "$RUN_DIR/checkpoint_step${STEP}.pt" ]]; then
  cp "$RUN_DIR/checkpoint_latest.pt" "$RUN_DIR/checkpoint_step${STEP}.pt"
fi

if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
  kill -STOP "$TRAIN_PID" || true
  echo "$(date -Is) paused old train process $TRAIN_PID"
fi

set +e
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO" "$PY" -m protein_jepa_frame.eval.paper_eval pipeline \
  --checkpoint "phase2_a2_5k=$RUN_DIR/checkpoint_latest.pt" \
  --output-dir "$EVAL_DIR" \
  --preset quick \
  --batch-size 8 \
  --sequence-frames 24 \
  --readout state \
  --require-contextual \
  --include-random \
  --baseline geometry_stats \
  --baseline centered_ca
EVAL_STATUS=$?
set -e
echo "$(date -Is) eval finished with status=$EVAL_STATUS"

if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
  kill -TERM "$TRAIN_PID" || true
  echo "$(date -Is) terminated old train process $TRAIN_PID"
fi

setsid bash -lc "cd '$REPO' && CUDA_VISIBLE_DEVICES=0 PYTHONPATH='$REPO' '$PY' - <<'PY'
from protein_jepa_frame.config import load_config
from protein_jepa_frame.training.train import train

cfg = load_config('$CONFIG')
cfg.resume_checkpoint = '$RUN_DIR/checkpoint_latest.pt'
cfg.checkpoint_every_steps = 2000
train(cfg)
PY" >> "$RUN_DIR/launcher_resume.log" 2>&1 &
RESUME_PID=$!
echo "$(date -Is) resumed training with pid=$RESUME_PID"
exit "$EVAL_STATUS"
