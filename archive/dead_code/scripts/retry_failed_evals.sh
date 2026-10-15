#!/bin/bash
# Retry failed evals after temp intervention finishes
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export TMPDIR="$ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
cd "$ROOT"

STAGE_C="$ROOT/artifacts/fast_abc_stage_c/checkpoint_latest.pt"
STAGE_B="$ROOT/artifacts/fast_abc_stage_b/checkpoint_latest.pt"
LOG_DIR="$ROOT/artifacts/fast_abc_evaluation"
mkdir -p "$LOG_DIR"

# Wait for temp intervention to finish
echo "WAITING for temp intervention to release GPU... $(date -Is)"
while pgrep -f "eval_temperature_intervention" > /dev/null 2>&1; do
    sleep 30
done
echo "GPU FREE $(date -Is)"

echo "=== Retry world_model ==="
python scripts/eval_world_model.py \
  --stage-c-ckpt "$STAGE_C" --stage-b-ckpt "$STAGE_B" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/world_model" --n-val-samples 200 --seeds 42 123 456 \
  > "$LOG_DIR/world_model_retry.log" 2>&1 && echo "PASS world_model" || echo "FAIL world_model"

echo "=== Retry rollout_quality ==="
python scripts/eval_rollout_quality.py \
  --stage-b-ckpt "$STAGE_B" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/rollout_quality" --n-val 100 --seeds 42 \
  > "$LOG_DIR/rollout_quality_retry.log" 2>&1 && echo "PASS rollout" || echo "FAIL rollout"

echo "=== Run sidechain ==="
python scripts/eval_sidechain_quality.py \
  --decoder-ckpt "$STAGE_C" --stage-b-ckpt "$STAGE_B" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$LOG_DIR/sidechain" --n-val 100 \
  > "$LOG_DIR/sidechain_retry.log" 2>&1 && echo "PASS sidechain" || echo "FAIL sidechain"

echo "ALL_RETRIES_DONE $(date -Is)"
