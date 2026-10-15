#!/bin/bash
exec 1>/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v28_per_residue/launch4.log 2>&1
echo "=== Launch v28 at $(date) ==="

pkill -9 -f "cli train" 2>/dev/null || true
sleep 2

# Aggressive cache clear
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/models/__pycache__ 2>/dev/null || true
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/training/__pycache__ 2>/dev/null || true
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/__pycache__ 2>/dev/null || true
find /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "Caches cleared"

python3 -c "
import ast
ast.parse(open('/liuyutian/protein_jepa_frame/protein_jepa_frame/models/lspwm.py').read())
# Verify force method has enable_grad
with open('/liuyutian/protein_jepa_frame/protein_jepa_frame/models/lspwm.py') as f:
    src = f.read()
assert 'torch.enable_grad()' in src, 'MISSING enable_grad'
print('Syntax OK + enable_grad present')
"

echo "Launching..."
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/tmp
cd /liuyutian/protein_jepa_frame
PYTHONPATH="/liuyutian/protein_jepa_frame"
exec nohup python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v28_per_residue.yaml \
  >> /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v28_per_residue/stdout.log 2>&1