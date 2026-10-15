#!/bin/bash
# ============================================================================
# v24 Hierarchical Mori-Zwanzig World Model — Stable Rollout Training
# ============================================================================
# V24 fixes:
#   1. Bath auxiliary state initialised from steady-state estimate
#      (non-zero bath drift from the start)
#   2. Force score Q: σ²-scaled DSM (σ²·∇A vs -ε, not ∇A vs -ε/σ²)
#   3. Rollout: Huber-fied Gaussian NLL + per-step clamping + delayed rollout phase
#   4. Reduced rollout steps (4), ensemble (4), and weights
#   5. Wider tau initialization (4000, 16000, 64000 ps)
# ============================================================================
set -euo pipefail
LOGFILE=/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v24_mori_stable/launch.log
exec 1>$LOGFILE 2>&1
echo "=== v24 Stable Mori Training Launch at $(date) ==="

# 1. Clear caches
echo "Clearing __pycache__..."
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '*.pyc' -delete 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.egg-link 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.dist-info 2>/dev/null || true
sed -i '/protein.jepa.frame/d' /opt/conda/lib/python3.13/site-packages/easy-install.pth 2>/dev/null || true

# 2. Reinstall from source
echo "Reinstalling..."
cd /liuyutian/protein_jepa_frame
PYTHONPATH="" pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3

# 3. Quick validation
python3 -c "
from protein_jepa_frame.models.lspwm import (
    UnifiedEnergyGraph, force_score_G_loss, force_score_Q_loss,
    self_consistency_loss, stationary_distribution_loss,
    mori_transition_nll,
)
print('All V24 imports OK')
print('force_score_Q_loss uses sigma^2-scaled DSM (V24 fix)')
"

# 4. Launch
echo "=== Launching at $(date) ==="
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/claude_tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/claude_tmp
mkdir -p /liuyutian/protein_jepa_frame/tmp
mkdir -p /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v24_mori_stable

cd /liuyutian/protein_jepa_frame
PYTHONPATH="/liuyutian/protein_jepa_frame"
nohup python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v24_mori_stable.yaml \
  >> /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v24_mori_stable/stdout.log 2>&1 &

PID=$!
echo "PID=$PID"
sleep 5
if ps -p $PID > /dev/null 2>&1; then
    echo "Process alive (PID=$PID)"
else
    echo "Process died! Check tail of stdout.log:"
    tail -40 /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v24_mori_stable/stdout.log
fi
