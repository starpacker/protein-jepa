#!/bin/bash
# =============================================================================
# LSP-WM Stage B v3: Improved thermodynamic training + comprehensive evaluation
# 
# This script:
#   1. Trains the LSP-WM with improved config (larger batch, curriculum, etc.)
#   2. After training completes, evaluates the best checkpoint
#
# Run with: setsid bash launch_stage_b_v3.sh &
# Or: bash launch_stage_b_v3.sh  (will exit after starting setsid)
# =============================================================================

set -euo pipefail

cd /liuyutian/protein_jepa_frame

ARTIFACTS="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics"
CONFIG="/liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v3_improved_thermodynamics.yaml"

# Check if already running
if [ -f "${ARTIFACTS}/.running" ]; then
    echo "ERROR: Training appears to be already running (${ARTIFACTS}/.running exists)"
    echo "  If this is a stale file, delete it and re-run."
    exit 1
fi

# Check for GPU
GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
echo "GPU available: ${GPU_AVAILABLE}"

# Mark as running
mkdir -p "${ARTIFACTS}"
date > "${ARTIFACTS}/.running"
echo "PID: $$" >> "${ARTIFACTS}/.running"

# Clear pycache to ensure fresh imports
echo "Clearing pycache..."
find /liuyutian/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Phase 1: Training
# =============================================================================
echo ""
echo "========================================================"
echo "PHASE 1: Training (Stage B v3)"
echo "Config: ${CONFIG}"
echo "Artifacts: ${ARTIFACTS}"
echo "========================================================"
echo ""

python3 -u -m protein_jepa_frame.cli train \
    --config "${CONFIG}" \
    2>&1 | tee "${ARTIFACTS}/training.log"

TRAIN_EXIT=${PIPESTATUS[0]}
date > "${ARTIFACTS}/.training_done"
echo "Training exit code: ${TRAIN_EXIT}" >> "${ARTIFACTS}/.training_done"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo ""
    echo "WARNING: Training exited with code ${TRAIN_EXIT}"
    echo "Proceeding with evaluation of available checkpoints..."
fi

# =============================================================================
# Phase 2: Find best checkpoint
# =============================================================================
echo ""
echo "========================================================"
echo "PHASE 2: Finding best checkpoint"
echo "========================================================"
echo ""

# Find the latest checkpoint
LATEST_CKPT=$(ls -t ${ARTIFACTS}/checkpoint_step*.pt 2>/dev/null | head -1 || echo "")

# Also look for explicit step checkpoints
BEST_CKPT="${LATEST_CKPT}"

# Check if we have step-based checkpoints
for CKPT in $(ls ${ARTIFACTS}/checkpoint_step*.pt 2>/dev/null | sort -t'_' -k2 -n); do
    echo "Found checkpoint: ${CKPT}"
done

# Use checkpoint_step3000 (final trained step) if available, otherwise latest
if [ -f "${ARTIFACTS}/checkpoint_step3000.pt" ]; then
    BEST_CKPT="${ARTIFACTS}/checkpoint_step3000.pt"
    echo "Using checkpoint_step3000.pt (final training step)"
elif [ -f "${ARTIFACTS}/checkpoint_latest.pt" ]; then
    BEST_CKPT="${ARTIFACTS}/checkpoint_latest.pt"
    echo "Using checkpoint_latest.pt"
elif [ -n "${LATEST_CKPT}" ]; then
    BEST_CKPT="${LATEST_CKPT}"
fi

if [ -z "${BEST_CKPT}" ] || [ ! -f "${BEST_CKPT}" ]; then
    echo "ERROR: No checkpoint found in ${ARTIFACTS}"
    echo "  Training may have failed or not yet produced checkpoints."
    rm -f "${ARTIFACTS}/.running"
    exit 1
fi

echo ""
echo "Using checkpoint: ${BEST_CKPT}"

# =============================================================================
# Phase 3: Comprehensive evaluation
# =============================================================================
echo ""
echo "========================================================"
echo "PHASE 3: Comprehensive evaluation"
echo "========================================================"
echo ""

EVAL_OUTPUT="${ARTIFACTS}/eval_improved"
mkdir -p "${EVAL_OUTPUT}"

# Create eval config with batch_size=4 for evaluation
python3 -c "
import json
with open('${CONFIG}') as f:
    cfg = json.load(f) if 'yaml' not in '${CONFIG}' else __import__('yaml').safe_load(f)
cfg['data']['batch_size'] = 4
cfg['model']['lspwm_potential_hidden_depth'] = 2
cfg['model']['lspwm_potential_hidden_dim'] = 128
with open('${EVAL_OUTPUT}/eval_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Eval config created')
"

python3 -u -m protein_jepa_frame.eval.comprehensive_eval \
    --checkpoint "${BEST_CKPT}" \
    --config "${EVAL_OUTPUT}/eval_config.json" \
    --output-dir "${EVAL_OUTPUT}" \
    --max-batches 20 \
    --bandwidth 0.1 \
    --device cuda:0 \
    --fastfolding-labels /tmp/lspwm_fastfolding_labels.jsonl \
    2>&1 | tee "${EVAL_OUTPUT}/eval.log"

EVAL_EXIT=${PIPESTATUS[0]}
date > "${ARTIFACTS}/.eval_done"
echo "Eval exit code: ${EVAL_EXIT}" >> "${ARTIFACTS}/.eval_done"

# =============================================================================
# Done
# =============================================================================
rm -f "${ARTIFACTS}/.running"

echo ""
echo "========================================================"
echo "ALL DONE"
echo "========================================================"
echo ""
echo "Training artifacts:   ${ARTIFACTS}"
echo "Training log:         ${ARTIFACTS}/training.log"
echo "Evaluation results:   ${EVAL_OUTPUT}/lspwm_eval_results.json"
echo "FastFolding results:  ${EVAL_OUTPUT}/fastfolding_results.json"
echo "Combined results:     ${EVAL_OUTPUT}/combined_results.json"
echo ""
echo "To check status: tail -f ${ARTIFACTS}/training.log"