#!/usr/bin/env bash
set -euo pipefail
cd /liuyutian/protein_jepa_frame
MODE=${MODE:-decoder}
case "$MODE" in
  decoder) extra=() ;;
  joint) extra=(--joint) ;;
  *) echo 'MODE must be decoder or joint' >&2; exit 1 ;;
esac
OUT=${OUT:-artifacts/lasp_v7_${MODE}_pilot}
mkdir -p .tmpwork/logs
export CUDA_VISIBLE_DEVICES=${GPU:-1}
python -u -m protein_jepa_frame.training.train_lasp_calibration \
  --output "$OUT" --steps 300 --max-residues 96 --max-train-samples 1024 \
  --batch-size 1 --val-batches 4 --val-every 50 --warmup 30 --seed 42 \
  "${extra[@]}" "$@" 2>&1 | tee ".tmpwork/logs/lasp_v7_${MODE}_pilot.log"
