#!/bin/bash
# Auto-launcher: waits for Stage B → Stage C → Evaluation
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TMPDIR="$ROOT/tmp"
export CLAUDE_CODE_TMPDIR="$ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
cd "$ROOT"

STAGE_B_CKPT="artifacts/fast_abc_stage_b/checkpoint_latest.pt"
STAGE_B_DONE="artifacts/fast_abc_stage_b/.done"
STAGE_C_CKPT="artifacts/fast_abc_stage_c/checkpoint_latest.pt"
STAGE_C_DONE="artifacts/fast_abc_stage_c/.done"
LOG_DIR="artifacts/fast_abc_evaluation"

echo "AUTO_LAUNCHER_START $(date -Is)"

# ── Wait for Stage B ──
echo "WAITING_FOR_STAGE_B $(date -Is)"
while true; do
    if [[ -s "$STAGE_B_CKPT" ]]; then
        # Check if the checkpoint is the final one (mtime older than 5 min = probably done)
        CKPT_AGE=$(($(date +%s) - $(stat -c %Y "$STAGE_B_CKPT" 2>/dev/null || echo 0)))
        # Also check if process is still running
        if ! pgrep -f "cli train.*fast_abc_stage_b" > /dev/null 2>&1; then
            if [[ $CKPT_AGE -gt 120 ]]; then
                echo "STAGE_B_COMPLETE $(date -Is) (no process, checkpoint age=${CKPT_AGE}s)"
                break
            fi
        fi
    fi
    sleep 60
done
touch "$STAGE_B_DONE"

# ── Launch Stage C ──
echo "LAUNCHING_STAGE_C $(date -Is)"
python -u -m protein_jepa_frame.cli train \
  --stage C \
  --config configs/fast_abc_stage_c_from_scratch.yaml \
  > artifacts/fast_abc_stage_c/stdout.log 2>&1
RC=$?
echo "STAGE_C_EXIT_CODE=$RC $(date -Is)"

if [[ $RC -ne 0 ]]; then
    echo "FATAL: Stage C failed with exit code $RC"
    exit $RC
fi
touch "$STAGE_C_DONE"

# ── Run Evaluation ──
echo "RUNNING_EVALUATION $(date -Is)"
mkdir -p "$LOG_DIR"

run_eval() {
  local name="$1"; shift
  echo "START_EVAL $name $(date -Is)"
  if "$@" > "$LOG_DIR/${name}.log" 2>&1; then
    echo "PASS_EVAL $name $(date -Is)"
  else
    echo "FAIL_EVAL $name (exit=$?) $(date -Is)" | tee -a "$LOG_DIR/${name}.log"
  fi
}

run_eval world_model python scripts/eval_world_model.py \
  --stage-c-ckpt "$STAGE_C_CKPT" --stage-b-ckpt "$STAGE_B_CKPT" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/world_model" --n-val-samples 200 --seeds 42 123 456

run_eval rollout_quality python scripts/eval_rollout_quality.py \
  --stage-b-ckpt "$STAGE_B_CKPT" --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/rollout_quality" --n-val 100 --seeds 42

run_eval baselines python scripts/eval_baselines.py \
  --stage-b-ckpt "$STAGE_B_CKPT" --train-manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/baselines" --n-val 100 --seeds 42

run_eval temperature_intervention python scripts/eval_temperature_intervention.py \
  --stage-b-ckpt "$STAGE_B_CKPT" --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/temperature_intervention" --n-val 50 --seeds 42 123

run_eval sidechain python scripts/eval_sidechain_quality.py \
  --decoder-ckpt "$STAGE_C_CKPT" --stage-b-ckpt "$STAGE_B_CKPT" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/sidechain" --n-val 100

echo "FAST_ABC_COMPLETE $(date -Is)"
