#!/bin/bash
set -euo pipefail
# ============================================================================
# V27 Training + Evaluation Serial Pipeline
# V26 BUGFIX: clamped NLL, delayed per-residue, proper aux context
# ============================================================================

TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S')
LOGDIR="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v27_per_residue_fixed"
LOGFILE="${LOGDIR}/pipeline.log"
CONFIG="/liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v27_per_residue_fixed.yaml"

mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "V27 Train + M1/K1/R1 Eval Pipeline"
echo "Started: $(date)"
echo "Config:  $CONFIG"
echo "Logdir:  $LOGDIR"
echo "=========================================="
echo "V27 CHANGES (BUGFIX):"
echo "  1. Clamped _gaussian_transition_nll_from_parameters (max 100/sample)"
echo "  2. Clamped per_residue_transition_nll (max 50/element)"
echo "  3. Clamped rollout_trajectory_nll (capped at 100)"
echo "  4. Fixed _predict_per_residue_aux with meaningful 3-frame context"
echo "  5. per_residue_warmup_steps=2000 (delayed activation)"
echo "  6. Reduced rollout weights (4-5x lower than V26)"
echo "  7. Increased per_residue_dynamics_weight (0.1→0.2)"
echo "  8. Init from V26 step 5400 (pre-explosion)"
echo "=========================================="

# ── 0. Clean Environment ───────────────────────────────────────
echo ""
echo "[0/5] Cleaning environment..."
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '*.pyc' -delete 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.egg-link 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.dist-info 2>/dev/null || true
sed -i '/protein.jepa.frame/d' /opt/conda/lib/python3.13/site-packages/easy-install.pth 2>/dev/null || true

echo "[0/5] Reinstalling package..."
cd /liuyutian/protein_jepa_frame
PYTHONPATH="" pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3

echo "[0/5] Validating V27 imports + bugfixes..."
python3 -c "
from protein_jepa_frame.models.lspwm import LSPWMProteinWorldModel, _gaussian_transition_nll_from_parameters
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
import torch
from pathlib import Path

cfg = load_config('$CONFIG')
model = ProteinJEPA(cfg.model)
print('  V27 model built OK')
print('  per_residue_enabled:', model.lspwm.per_residue_enabled)
print('  per_residue_warmup_steps:', model.lspwm.per_residue_warmup_steps)
print('  per_residue_step:', hasattr(model.lspwm, 'per_residue_step'))
print('  per_residue_transition_nll:', hasattr(model.lspwm, 'per_residue_transition_nll'))
print('  _predict_per_residue_aux:', hasattr(model.lspwm, '_predict_per_residue_aux'))

# Test V26 ckpt compat
ckpt = torch.load('artifacts/lspwm_v44_stage_b_v26_per_residue/checkpoint_step5400.pt', map_location='cpu')
state = upgrade_lspwm_state_dict(model, ckpt.get('model', {}))
m, u = model.load_state_dict(state, strict=False)
print(f'  V26 ckpt compat: missing={len(m)}, unexpected={len(u)}')
for k in sorted(m):
    print(f'    missing: {k}')
for k in sorted(u):
    print(f'    unexpected: {k}')

# Test NLL clamping
mean = torch.randn(4, 8)
target = torch.randn(4, 8)
dt = torch.ones(4) * 10.0
tiny_diffusion = torch.eye(8).unsqueeze(0).expand(4, -1, -1) * 1e-12
nll = _gaussian_transition_nll_from_parameters(mean, target, tiny_diffusion, dt)
print(f'  NLL clamp test (tiny diffusion): {nll.item():.4f} (should be <= 100)')
assert nll.item() <= 100.1, f'NLL not clamped! Got {nll.item()}'
print('V27 validation: OK')
"

# ── 1. Launch V27 Training ────────────────────────────────────
echo ""
echo "[1/5] Launching V27 training..."
echo "Start time: $(date)"

export CLAUDE_CODE_TMPDIR=/liuyutian/protein_jepa_frame/claude_tmp
export TMPDIR=/liuyutian/protein_jepa_frame/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /liuyutian/protein_jepa_frame/claude_tmp
mkdir -p /liuyutian/protein_jepa_frame/tmp

cd /liuyutian/protein_jepa_frame
PYTHONPATH="/liuyutian/protein_jepa_frame"
nohup python -u -m protein_jepa_frame.cli train \
  --config "$CONFIG" \
  >> "${LOGDIR}/stdout.log" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
sleep 5

if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "FATAL: Training process died immediately!"
    tail -60 "${LOGDIR}/stdout.log"
    exit 1
fi
echo "Training running (PID=$TRAIN_PID)"

# ── 2. Monitor Training ─────────────────────────────────────────
echo ""
echo "[2/5] Monitoring training progress..."
LAST_STEP=0
while ps -p $TRAIN_PID > /dev/null 2>&1; do
    if [ -f "${LOGDIR}/train.jsonl" ]; then
        CURRENT=$(tail -1 "${LOGDIR}/train.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step','?'))" 2>/dev/null || echo "?")
        if [ "$CURRENT" != "$LAST_STEP" ] && [ "$CURRENT" != "?" ]; then
            LINE=$(tail -1 "${LOGDIR}/train.jsonl")
            echo "  [$(date '+%H:%M:%S')] step=$CURRENT $(echo "$LINE" | python3 -c "
import sys,json
d=json.load(sys.stdin)
phase=d.get('phase','?')
loss=float(d.get('loss',0))
mori=float(d.get('mori_nll',0))
G=float(d.get('force_score_G',0))
Q=float(d.get('force_score_Q',0))
skill=float(d.get('skill',0))
per_res_nll=float(d.get('per_residue_nll', -1))
trl_nll=float(d.get('rollout_trajectory_nll', -1))
print(f'phase={phase} loss={loss:.2f} mori={mori:.3f} G={G:.4f} Q={Q:.4f} skill={skill:.4f} per_res_nll={per_res_nll:.2f} trl_nll={trl_nll:.1f}')
" 2>/dev/null || echo "$LINE")"
            LAST_STEP=$CURRENT
        fi
    fi
    sleep 120
done

echo ""
echo "[2/5] Training completed at $(date)"
echo "Final log lines:"
tail -5 "${LOGDIR}/train.jsonl" 2>/dev/null || echo "(no train.jsonl)"

# ── 3. Find Best Checkpoint ────────────────────────────────────
echo ""
echo "[3/5] Finding best checkpoint..."
BEST_CKPT=""
BEST_STEP=0

for CKPT in "${LOGDIR}"/checkpoint_step*.pt; do
    [ -f "$CKPT" ] || continue
    STEP=$(basename "$CKPT" | grep -oP 'step\K[0-9]+')
    if [ -z "$STEP" ]; then continue; fi
    if [ "$STEP" -gt "$BEST_STEP" ]; then
        BEST_STEP=$STEP
        BEST_CKPT="$CKPT"
    fi
done

if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="${LOGDIR}/checkpoint_latest.pt"
    BEST_STEP=$(tail -1 "${LOGDIR}/train.jsonl" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('step',0))" 2>/dev/null || echo 0)
fi
echo "[3/5] Best checkpoint: $BEST_CKPT (step=$BEST_STEP)"

# ── 4. Run M1/K1/R1 Evaluation ──────────────────────────────────
echo ""
echo "[4/5] Running M1/K1/R1 evaluation..."
python3 -c "
import sys, json
from pathlib import Path
from protein_jepa_frame.eval.m1k1r1_benchmark import run_m1k1r1_benchmarks, print_summary_table
from protein_jepa_frame.config import load_config
device = 'cuda'
config = load_config('$CONFIG')
checkpoint_path = Path('$BEST_CKPT')
output_dir = Path('${LOGDIR}/eval')
results = run_m1k1r1_benchmarks(
    checkpoint_path=checkpoint_path,
    config=config,
    device=device,
    output_dir=output_dir,
    m1_batches=100,
    k1_batches=100,
    r1_batches=30,
)
print_summary_table(results)
" 2>&1

echo ""
echo "=========================================="
echo "V27 PIPELINE COMPLETE at $(date)"
echo "Check: ${LOGDIR}/"
echo "=========================================="
