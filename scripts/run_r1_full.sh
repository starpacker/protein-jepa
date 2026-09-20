#!/bin/bash
# ============================================================================
# R1: Committor correlation stability (original settings)
# Horizons 1-64, history lengths 1-8, 50-100 batches
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /liuyutian/protein_jepa_frame

CHECKPOINT="artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_latest.pt"
CONFIG="artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/eval_config.json"
MANIFEST="manifests/tmjepa_v3.validation.jsonl"
OUTDIR="artifacts/reviewer_experiments/protein_kinetics"

echo "[R1 Full] $(date)"
exec /opt/conda/bin/python3 -u scripts/evaluate_protein_kinetics.py \
    --model lspwm \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output-dir "$OUTDIR" \
    --max-batches 100 \
    --batch-size 8 \
    --device cuda \
    --sampler-mode hierarchical \
    --horizons "1,2,4,8,16,32,64" \
    --history-lengths "1,2,4,8" \
    --seed 6303 \
    > /tmp/r1_full.log 2>&1
