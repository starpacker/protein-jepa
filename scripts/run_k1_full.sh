#!/bin/bash
# ============================================================================
# K1: FastFolding kinetics comparison with scaffold-disjoint holdout
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /liuyutian/protein_jepa_frame

OUTDIR="artifacts/reviewer_experiments/protein_kinetics"
LABELS="artifacts/fastfolding/frame_labels.full.jsonl"
PREDICTIONS="artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/fastfolding_predictions.jsonl"

echo "[K1 Full] $(date)"
exec /liuyutian/protein_jepa/.venv/bin/python -u scripts/evaluate_k1_fastfolding.py \
    --labels "$LABELS" \
    --predictions "$PREDICTIONS" \
    --output "$OUTDIR/k1_fastfolding.json" \
    --context-frames 8 \
    --horizons 1 4 8 16 \
    --train-fraction 0.6 \
    --validation-fraction 0.2 \
    --tica-lag 1 \
    --tica-clusters 8 \
    --bootstrap-samples 2000 \
    --seed 6303 \
    > /tmp/k1_full.log 2>&1
