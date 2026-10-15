#!/bin/bash
# ============================================================================
# v22 Hierarchical Mori-Zwanzig World Model — Training Launch
# ============================================================================
# Architecture: Single-frame H_t -> Q_local [B,L,8] + G (pool) + gauge
# Energy graph: A(Q,G) = A_local + A_pair + A_global + A_coupling
# Losses: cv_vamp2 + force_score_G + force_score_Q + force_direction +
#         mori_nll + self_consistency + residue_conditional + stationary +
#         rollout_traj + semigroup + sliced_score
# ============================================================================
set -euo pipefail
LOGFILE=/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v22_hierarchical_mori/launch.log
exec 1>$LOGFILE 2>&1
echo "=== v22 Hierarchical Mori Training Launch at $(date) ==="

# 1. Clear caches
echo "Clearing __pycache__..."
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '*.pyc' -delete 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.egg-link 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.dist-info 2>/dev/null || true
sed -i '/protein.jepa.frame/d' /opt/conda/lib/python3.13/site-packages/easy-install.pth 2>/dev/null || true

# 3. Reinstall from source
echo "Reinstalling..."
cd /liuyutian/protein_jepa_frame
PYTHONPATH="" pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3

# 4. Quick validation
python3 -c "
from protein_jepa_frame.models.lspwm import (
    UnifiedEnergyGraph, force_score_G_loss, force_score_Q_loss,
    self_consistency_loss, stationary_distribution_loss
)
print('All v22 imports OK')
"

# 5. Launch
echo "=== Launching at $(date) ==="
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/tmp
mkdir -p /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v22_hierarchical_mori

cd /liuyutian/protein_jepa_frame
PYTHONPATH="/liuyutian/protein_jepa_frame"
nohup python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v22_hierarchical_mori.yaml \
  >> /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v22_hierarchical_mori/stdout.log 2>&1 &

PID=$!
echo "PID=$PID"
sleep 5
if ps -p $PID > /dev/null 2>&1; then
    echo "Process alive (PID=$PID)"
else
    echo "Process died! Check tail of stdout.log:"
    tail -40 /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v22_hierarchical_mori/stdout.log
fi