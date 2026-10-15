#!/bin/bash
set -euo pipefail
ARTIFACT_DIR="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v29_per_residue_fixed"
mkdir -p "$ARTIFACT_DIR"
exec 1>"$ARTIFACT_DIR/launch.log" 2>&1
echo "=== Launch V29 at $(date) ==="

# Clean stale bytecode
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/models/__pycache__
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/models/lasp/__pycache__
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/training/__pycache__
rm -rf /liuyutian/protein_jepa_frame/protein_jepa_frame/__pycache__
echo "Bytecode caches cleared"

# Quick syntax + import check
python3 -c "
import ast
ast.parse(open('/liuyutian/protein_jepa_frame/protein_jepa_frame/models/lasp/energy.py').read())
from protein_jepa_frame.config import load_config
cfg = load_config('configs/lspwm_v44_stage_b_v29_per_residue_fixed.yaml')
print('Config OK')
from protein_jepa_frame.models.jepa import ProteinJEPA
m = ProteinJEPA(cfg.model)
print(f'Model OK: lspwm={type(m.lspwm).__name__}, energy_graph={type(m.lspwm.unified_energy).__name__}')
"
echo "Sanity checks passed"

export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/tmp

cd /liuyutian/protein_jepa_frame
PYTHONPATH="/liuyutian/protein_jepa_frame"
echo "Launching training at $(date)..."

exec nohup python -u -m protein_jepa_frame.cli train \
  --config /liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v29_per_residue_fixed.yaml \
  >> "$ARTIFACT_DIR/stdout.log" 2>&1
