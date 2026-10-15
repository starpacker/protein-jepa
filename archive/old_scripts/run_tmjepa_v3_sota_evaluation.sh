#!/usr/bin/env bash
set -euo pipefail

FRAME_REPO=/liuyutian/protein_jepa_frame
PYTHON=/liuyutian/protein_jepa/.venv/bin/python
GPU_ID=${GPU_ID:-0}
CONFIG=${CONFIG:-$FRAME_REPO/configs/phase8r_tmjepa_v3_spectral_closure.yaml}
CHECKPOINT=${CHECKPOINT:-$FRAME_REPO/artifacts/phase8r_tmjepa_v3_spectral_closure/checkpoint_step250.pt}
TAG=${TAG:-phase8r_step250}
OUTPUT_DIR=${OUTPUT_DIR:-$FRAME_REPO/artifacts/sota_reproduction/$TAG}

mkdir -p "$OUTPUT_DIR"
cd "$FRAME_REPO"

for seed in 6303 6304 6305; do
  env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$FRAME_REPO" \
    "$PYTHON" scripts/evaluate_tmjepa_v3.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --manifest manifests/tmjepa_v3.validation.jsonl \
    --output "$OUTPUT_DIR/seed${seed}_ck0.json" \
    --device cuda:0 --batch-size 4 --max-batches 50 --ck-samples 0 \
    --lags 1 2 4 8 --seed "$seed"
done

env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$FRAME_REPO" \
  "$PYTHON" scripts/evaluate_tmjepa_v3.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --manifest manifests/tmjepa_v3.validation.jsonl \
  --output "$OUTPUT_DIR/seed6303_ck8.json" \
  --device cuda:0 --batch-size 4 --max-batches 50 --ck-samples 8 \
  --lags 1 2 4 8 --seed 6303
