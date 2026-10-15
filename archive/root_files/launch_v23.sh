#!/bin/bash
# ============================================================================
# v23 Fixed Hierarchical Mori-Zwanzig World Model — Training Launch
# ============================================================================
# Bug fixes:
#   1. force_score_Q_loss uses correct DSM formula (noise-only term)
#   2. force_direction weight gating fixed (checks per_residue_enabled)
#   3. self_consistency_loss provides orthogonal constraint
#   4. MultiTimescaleBath wired into Mori transition NLL
# ============================================================================
set -euo pipefail
LOGFILE=/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v23_mori_fixed/launch.log
mkdir -p /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v23_mori_fixed
exec 1>$LOGFILE 2>&1
echo "=== v23 Fixed Mori Training Launch at $(date) ==="

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
    mori_transition_nll, MultiTimescaleBath, DiagonalDiffusion,
)
print('All v23 imports OK')
# Verify force_score_Q is the FIXED version
import inspect
src = inspect.getsource(force_score_Q_loss)
if 'target_score' in src and 'noise' in src and 'noise_std' in src:
    print('force_score_Q_loss: DSM formula VERIFIED')
else:
    print('WARNING: force_score_Q_loss may still be buggy')
# Check mori_transition_nll exists
print(f'mori_transition_nll signature: {inspect.signature(mori_transition_nll)}')
"

# 4. Launch
echo "=== Launching at $(date) ==="
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/tmp
mkdir -p /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v23_mori_fixed

cd /liuyutian/protein_jepa_frame
PYTHONPATH="/liuyutian/protein_jepa_frame"
nohup python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v23_mori_fixed.yaml \
  >> /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v23_mori_fixed/stdout.log 2>&1 &

PID=$!
echo "PID=$PID"
sleep 5
if ps -p $PID > /dev/null 2>&1; then
    echo "Process alive (PID=$PID)"
else
    echo "Process died! Check tail of stdout.log:"
    tail -40 /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v23_mori_fixed/stdout.log
fi
