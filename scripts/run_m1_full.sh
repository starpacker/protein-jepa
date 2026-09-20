#!/bin/bash
# ============================================================================
# M1: Strict thermodynamic closure ablation with temporal holdout
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /liuyutian/protein_jepa_frame

CHECKPOINT="artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_latest.pt"
CONFIG="artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/eval_improved/eval_config.json"
MANIFEST="manifests/tmjepa_v3.validation.jsonl"
OUTDIR="artifacts/reviewer_experiments/thermodynamic_closure"

echo "[M1 Full] $(date)"
exec /opt/conda/bin/python3 -u scripts/evaluate_m1_strict.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --output "$OUTDIR/m1_strict.json" \
    --max-batches 50 \
    --batch-size 32 \
    --device cuda \
    --clip-frames 24 \
    --window-stride 8 \
    --history-window 8 \
    --kernel-length 4 \
    --neural-epochs 200 \
    --rollout-steps 256 \
    --seed 6303 \
    > /tmp/m1_full.log 2>&1
