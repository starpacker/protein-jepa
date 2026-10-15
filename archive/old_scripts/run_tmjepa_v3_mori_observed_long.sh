#!/usr/bin/env bash
# Fresh training run for TM-JEPA V3 mori_observed (coupled_bath + obs_noise, v2 init)
# Scaled-up coupling, log-uniform memory times, intervention loss enabled, ensemble=8.
# Trains 3000 steps from scratch, then evaluates.
set -euo pipefail

FRAME_REPO=/liuyutian/protein_jepa_frame
CONFIG="$FRAME_REPO/configs/phase7_tmjepa_v3_mori_observed_long.yaml"
RUN_DIR="$FRAME_REPO/artifacts/phase7_tmjepa_v3_mori_observed_long"
EVAL_OUTPUT="$RUN_DIR/eval_validation.json"
VAL_MANIFEST="$FRAME_REPO/manifests/tmjepa_v3.validation.jsonl"
CHECKPOINT="$RUN_DIR/checkpoint_latest.pt"

mkdir -p "$RUN_DIR"

echo "[$(date -Is)] === PHASE 1: TRAINING (fresh, 3000 steps) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" python -m protein_jepa_frame.cli train --config "$CONFIG" 2>&1

echo "[$(date -Is)] === PHASE 2: EVALUATION ON VALIDATION SET ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" python scripts/evaluate_tmjepa_v3.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$VAL_MANIFEST" \
  --output "$EVAL_OUTPUT" \
  --device cuda:0 \
  --batch-size 2 \
  --max-batches 50 \
  --ck-samples 8 \
  --lags 1 2 4 8 \
  --seed 6303 2>&1

echo "[$(date -Is)] === DONE ==="
echo "Training + evaluation completed. Results: $EVAL_OUTPUT"