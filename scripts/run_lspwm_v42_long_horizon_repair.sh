#!/usr/bin/env bash
set -euo pipefail

ROOT=/liuyutian/protein_jepa_frame
OUT="$ROOT/artifacts/lspwm_v42_long_horizon_repair"
mkdir -p "$OUT"

PYTHONPATH="$ROOT" python -m protein_jepa_frame.cli train \
  --config "$ROOT/configs/lspwm_v42_long_horizon_repair.yaml" \
  > "$OUT/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_paired_dynamics.py" \
  --config "$ROOT/configs/lspwm_v42_long_horizon_repair.yaml" \
  --checkpoint "$OUT/checkpoint_latest.pt" \
  --manifest "$ROOT/manifests/tmjepa_v3.validation.jsonl" \
  --output "$OUT/paired_validation.json" \
  --max-batches 80 --batch-size 4 --seed 6303 --device cuda:0 \
  >> "$OUT/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_lspwm_physical.py" \
  --config "$ROOT/configs/lspwm_v42_long_horizon_repair.yaml" \
  --checkpoint "$OUT/checkpoint_latest.pt" \
  --manifest "$ROOT/manifests/tmjepa_v3.validation.jsonl" \
  --output "$OUT/physical_validation.json" \
  --max-batches 80 --batch-size 4 --ensemble-size 8 --rollout-steps 24 \
  --fdt-samples 256 --seed 6303 --device cuda:0 \
  >> "$OUT/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_lspwm_rollout.py" \
  --config "$ROOT/configs/lspwm_v42_long_horizon_repair.yaml" \
  --checkpoint "$OUT/checkpoint_latest.pt" \
  --manifest "$ROOT/manifests/tmjepa_v3.validation.jsonl" \
  --output "$OUT/rollout_validation.json" \
  --max-batches 80 --batch-size 4 --ensemble-size 8 --rollout-steps 24 \
  --fdt-samples 256 --seed 6303 --device cuda:0 \
  >> "$OUT/training.log" 2>&1
