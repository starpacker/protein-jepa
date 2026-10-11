#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

OUT="${EVAL_OUTPUT_DIR:-artifacts/full_abc_evaluation}"
mkdir -p "$OUT"
STAGE_B="${STAGE_B_CKPT:-artifacts/full_abc_stage_b_canonical_v1/checkpoint_latest.pt}"
STAGE_C="${STAGE_C_CKPT:-artifacts/full_abc_stage_c_canonical_v1/checkpoint_latest.pt}"

echo "WAITING_FOR_FINAL_STAGE_C $(date -Is)"
while [[ ! -s "$STAGE_C" ]]; do
  sleep 60
done
echo "FINAL_CHECKPOINTS_READY $(date -Is)"

# ── Provenance check: require canonical_state_version.txt in each checkpoint dir ──
for CKPT_DIR in "$(dirname "$STAGE_B")" "$(dirname "$STAGE_C")"; do
  VERSION_FILE="$CKPT_DIR/canonical_state_version.txt"
  if [[ ! -s "$VERSION_FILE" ]]; then
    echo "FATAL: Missing canonical_state_version.txt in $CKPT_DIR" >&2
    echo "Refusing to evaluate checkpoints without canonical state provenance." >&2
    exit 1
  fi
  echo "Provenance OK: $(cat "$VERSION_FILE")"
done

# ── Counter for failures ──
FAILURES=0

run_eval() {
  local name="$1"
  shift
  echo "START $name $(date -Is)"
  if "$@" >"$OUT/${name}.log" 2>&1; then
    echo "PASS $name $(date -Is)"
  else
    echo "FAIL $name $(date -Is)" | tee -a "$OUT/${name}.log"
    FAILURES=$((FAILURES + 1))
  fi
}

# ── P0.1: World model evaluation (oracle/direct/autonomous/persistence/qshuffle) ──
run_eval world_model \
  python -u scripts/eval_world_model.py \
    --stage-c-ckpt "$STAGE_C" --stage-b-ckpt "$STAGE_B" \
    --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --output-dir "$OUT/world_model" --n-val-samples 200 \
    --seeds 42 123 456 --base-step-ps 100 --max-substep-dt-ps 100

# ── P0.2: Rollout quality (Q-space drift, ACF) ──
run_eval rollout_quality \
  python -u scripts/eval_rollout_quality.py \
    --stage-b-ckpt "$STAGE_B" \
    --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --output-dir "$OUT/rollout_quality" --n-val 100 --seeds 42

# ── P0.3: Baselines (linear, PCA, ...) ──
run_eval baselines \
  python -u scripts/eval_baselines.py \
    --stage-b-ckpt "$STAGE_B" \
    --train-manifest artifacts/mdcath_manifest_200dom/train.jsonl \
    --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --output-dir "$OUT/baselines" --n-val 100 --seeds 42

# ── P0.4: Temperature intervention ──
run_eval temperature_intervention \
  python -u scripts/eval_temperature_intervention.py \
    --stage-b-ckpt "$STAGE_B" \
    --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --output-dir "$OUT/temperature_intervention" --n-val 50 --seeds 42 123

# ── P0.5: Sidechain quality ──
run_eval sidechain \
  python -u scripts/eval_sidechain_quality.py \
    --decoder-ckpt "$STAGE_C" --stage-b-ckpt "$STAGE_B" \
    --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
    --output-dir "$OUT/sidechain" --n-val 100

echo "FULL_ABC_EVALUATION_COMPLETE $(date -Is) failures=$FAILURES"
if [[ $FAILURES -gt 0 ]]; then
  echo "FATAL: $FAILURES evaluation(s) FAILED"
  exit 1
fi
