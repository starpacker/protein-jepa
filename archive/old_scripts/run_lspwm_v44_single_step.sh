#!/usr/bin/env bash
set -euo pipefail

ROOT=/liuyutian/protein_jepa_frame
STAGE_A="$ROOT/configs/lspwm_v44_stage_a_representation.yaml"
STAGE_B="$ROOT/configs/lspwm_v44_stage_b_single_step_thermodynamics.yaml"
OUT_A="$ROOT/artifacts/lspwm_v44_stage_a_representation"
OUT_B="$ROOT/artifacts/lspwm_v44_stage_b_single_step_thermodynamics"
MANIFEST="$ROOT/manifests/tmjepa_v3.validation.jsonl"

mkdir -p "$OUT_A" "$OUT_B"

PYTHONPATH="$ROOT" python -m protein_jepa_frame.cli train \
  --config "$STAGE_A" > "$OUT_A/training.log" 2>&1

PYTHONPATH="$ROOT" python -m protein_jepa_frame.cli train \
  --config "$STAGE_B" > "$OUT_B/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_paired_dynamics.py" \
  --config "$STAGE_A" \
  --checkpoint "$OUT_A/checkpoint_latest.pt" \
  --manifest "$MANIFEST" \
  --output "$OUT_A/paired_validation.json" \
  --max-batches 80 --batch-size 4 --seed 6442 --device cuda:0 \
  >> "$OUT_A/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_paired_dynamics.py" \
  --config "$STAGE_B" \
  --checkpoint "$OUT_B/checkpoint_latest.pt" \
  --manifest "$MANIFEST" \
  --output "$OUT_B/paired_validation.json" \
  --max-batches 80 --batch-size 4 --seed 6442 --device cuda:0 \
  >> "$OUT_B/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_lspwm_endpoint.py" \
  --config "$STAGE_B" \
  --checkpoint "$OUT_B/checkpoint_latest.pt" \
  --manifest "$MANIFEST" \
  --output "$OUT_B/endpoint_validation.json" \
  --max-batches 80 --batch-size 4 --ensemble-size 8 --fdt-samples 256 \
  --seed 6442 --device cuda:0 \
  >> "$OUT_B/training.log" 2>&1

PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_lspwm_physical.py" \
  --config "$STAGE_B" \
  --checkpoint "$OUT_B/checkpoint_latest.pt" \
  --manifest "$MANIFEST" \
  --output "$OUT_B/physical_validation.json" \
  --max-batches 80 --batch-size 4 --ensemble-size 8 --rollout-steps 24 \
  --fdt-samples 256 --seed 6442 --device cuda:0 \
  --framework-label "LSP-WM v4.4" \
  >> "$OUT_B/training.log" 2>&1

# Keep the old rollout evaluator as a diagnostic only. It must never be used
# as a training signal or as the primary V4.4 promotion gate.
PYTHONPATH="$ROOT" python "$ROOT/scripts/evaluate_lspwm_rollout.py" \
  --config "$STAGE_B" \
  --checkpoint "$OUT_B/checkpoint_latest.pt" \
  --manifest "$MANIFEST" \
  --output "$OUT_B/rollout_diagnostic.json" \
  --max-batches 80 --batch-size 4 --ensemble-size 8 --rollout-steps 24 \
  --fdt-samples 256 --seed 6442 --device cuda:0 \
  --framework-label "LSP-WM v4.4" \
  >> "$OUT_B/training.log" 2>&1
