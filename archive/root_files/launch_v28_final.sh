#!/bin/bash
LOGFILE=/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v28_per_residue/launch_final.log
exec 1>$LOGFILE 2>&1
echo "=== Final launch at $(date) ==="

# 1. Kill
pkill -9 -f "python.*cli train" 2>/dev/null || true
sleep 3

# 2. Nuclear cache clear with forced reinstall
echo "Removing all __pycache__..."
find /liuyutian/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find /liuyutian/protein_jepa_frame -name '*.pyc' -delete 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.egg-link 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.dist-info 2>/dev/null || true
sed -i '/protein.jepa.frame/d' /opt/conda/lib/python3.13/site-packages/easy-install.pth 2>/dev/null || true

# 3. Reinstall from source (forces fresh bytecode)
echo "Reinstalling..."
cd /liuyutian/protein_jepa_frame
PYTHONPATH="" pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3

# 4. Verify force method HAS enable_grad
python3 -c "
with open('/liuyutian/protein_jepa_frame/protein_jepa_frame/models/lspwm.py') as f:
    src = f.read()
assert 'torch.enable_grad()' in src, 'FATAL: enable_grad NOT in force()'
# Verify it's in the force method context
idx = src.find('def force(self')
force_block = src[idx:idx+1500]
assert 'torch.enable_grad()' in force_block, 'FATAL: enable_grad not in force method'
assert 'q_leaf = qi.detach().clone().requires_grad_(True)' in force_block, 'FATAL: not using clone for force'
print('force() verification PASSED')
"

# 5. Quick import test
python3 -c "
from protein_jepa_frame.models.lspwm import ResidueMessagePassingPotential
import torch
pot = ResidueMessagePassingPotential(cv_dim=8, condition_dim=64, hidden_dim=16, depth=1)
q = torch.randn(1, 16, 8, requires_grad=True)
d = torch.randn(1, 16, 16)
m = torch.ones(1, 16)
c = torch.randn(1, 64)
f = pot.force(q, d, m, c, create_graph=True)
print(f'force shape={list(f.shape)}, has_grad={f.requires_grad}')
assert f.requires_grad, 'FORCE no grad'
print('Quick test PASSED')
"

# 6. Launch
echo "=== Launching at $(date) ==="
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/tmp

PYTHONPATH="/liuyutian/protein_jepa_frame"
nohup python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v28_per_residue.yaml \
  >> /liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v28_per_residue/stdout.log 2>&1 &

PID=$!
echo "PID=$PID"
sleep 5
ps -p $PID > /dev/null 2>&1 && echo "Process alive" || echo "Process died!"