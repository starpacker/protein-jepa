#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

OUT="artifacts/fast_abc_evaluation"
mkdir -p "$OUT"
STAGE_B="artifacts/fast_abc_stage_b/checkpoint_latest.pt"
STAGE_C="artifacts/fast_abc_stage_c/checkpoint_latest.pt"

echo "WAITING_FOR_FAST_FINAL_STAGE_C $(date -Is)"
while [[ ! -s "$STAGE_C" ]]; do sleep 30; done

run_eval() {
  local name="$1"; shift
  echo "START $name $(date -Is)"
  if "$@" >"$OUT/${name}.log" 2>&1; then
    echo "PASS $name $(date -Is)"
  else
    echo "FAIL $name $(date -Is)" | tee -a "$OUT/${name}.log"
  fi
}

run_eval world_model python scripts/eval_world_model.py \
  --stage-c-ckpt "$STAGE_C" --stage-b-ckpt "$STAGE_B" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$OUT/world_model" --n-val-samples 200 --seeds 42 123 456
run_eval rollout_quality python scripts/eval_rollout_quality.py \
  --stage-b-ckpt "$STAGE_B" --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$OUT/rollout_quality" --n-val 100 --seeds 42
run_eval baselines python scripts/eval_baselines.py \
  --stage-b-ckpt "$STAGE_B" --train-manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$OUT/baselines" --n-val 100 --seeds 42
run_eval temperature_intervention python scripts/eval_temperature_intervention.py \
  --stage-b-ckpt "$STAGE_B" --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$OUT/temperature_intervention" --n-val 50 --seeds 42 123
run_eval sidechain python scripts/eval_sidechain_quality.py \
  --decoder-ckpt "$STAGE_C" --stage-b-ckpt "$STAGE_B" \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --output-dir "$OUT/sidechain" --n-val 100

echo "FAST_ABC_EVALUATION_COMPLETE $(date -Is)"
