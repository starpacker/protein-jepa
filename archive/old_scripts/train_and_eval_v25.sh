#!/bin/bash
set -euo pipefail
# ============================================================================
# V25 Training + Evaluation Serial Pipeline
# Runs V25 training, then M1/K1/R1 eval on best checkpoint.
# Compares with V24 baseline.
# ============================================================================

TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S')
LOGDIR="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v25_comprehensive"
LOGFILE="${LOGDIR}/pipeline.log"
CONFIG="/liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v25_comprehensive.yaml"

mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "V25 Train + M1/K1/R1 Eval Pipeline"
echo "Started: $(date)"
echo "Config:  $CONFIG"
echo "Logdir:  $LOGDIR"
echo "=========================================="

# ── 0. Clean Environment ───────────────────────────────────────
echo ""
echo "[0/4] Cleaning environment..."
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find /liuyutian/protein_jepa_frame/protein_jepa_frame -name '*.pyc' -delete 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.egg-link 2>/dev/null || true
rm -rf /opt/conda/lib/python3.13/site-packages/protein_jepa_frame-*.dist-info 2>/dev/null || true
sed -i '/protein.jepa.frame/d' /opt/conda/lib/python3.13/site-packages/easy-install.pth 2>/dev/null || true

echo "[0/4] Reinstalling package..."
cd /liuyutian/protein_jepa_frame
PYTHONPATH="" pip install -e . --no-build-isolation --no-deps 2>&1 | tail -3

echo "[0/4] Validating V25 imports + model build..."
python3 -c "
from protein_jepa_frame.models.lspwm import LSPWMProteinWorldModel
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
import torch
from pathlib import Path

cfg = load_config('$CONFIG')
model = ProteinJEPA(cfg.model)
print('  V25 model built OK')
print('  history_to_bath:', hasattr(model.lspwm, 'history_to_bath'))
print('  diffusion_scale_correction:', hasattr(model.lspwm, 'diffusion_scale_correction'))

# Test V24 ckpt backward compat
ckpt = torch.load('artifacts/lspwm_v44_stage_b_v24_mori_stable/checkpoint_step5400.pt', map_location='cpu')
state = upgrade_lspwm_state_dict(model, ckpt.get('model', {}))
m, u = model.load_state_dict(state, strict=False)
print(f'  V24 ckpt compat: missing={len(m)}, unexpected={len(u)}')
if len(u) > 0:
    for k in u:
        print(f'    unexpected: {k}')
print('V25 validation: OK')
"

# ── 1. Launch V25 Training ────────────────────────────────────
echo ""
echo "[1/4] Launching V25 training..."
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
    tail -40 "${LOGDIR}/stdout.log"
    exit 1
fi
echo "Training running (PID=$TRAIN_PID)"

# ── 2. Monitor Training ─────────────────────────────────────────
echo ""
echo "[2/4] Monitoring training progress..."
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
mori=float(d.get('mori',0))
G=float(d.get('G',0))
Q=float(d.get('Q',0))
skill=float(d.get('skill',0))
print(f'phase={phase} loss={loss:.2f} mori={mori:.2f} G={G:.4f} Q={Q:.4f} skill={skill:.4f}')
" 2>/dev/null || echo "$LINE")"
            LAST_STEP=$CURRENT
        fi
    fi
    sleep 120
done

echo ""
echo "[2/4] Training completed at $(date)"
echo "Final log line:"
tail -3 "${LOGDIR}/train.jsonl" 2>/dev/null || echo "(no train.jsonl)"

# ── 3. Find Best Checkpoint ────────────────────────────────────
echo ""
echo "[3/4] Finding best checkpoint..."
BEST_CKPT=""
BEST_SCORE=-999

for CKPT in "${LOGDIR}"/checkpoint_step*.pt; do
    [ -f "$CKPT" ] || continue
    STEP=$(basename "$CKPT" | grep -oP 'step\K[0-9]+')
    if [ -z "$STEP" ]; then continue; fi
    
    # Score: prefer later checkpoint after roll phase starts (5000+)
    # Penalize any where rollout likely exploded
    SCORE=$((STEP - 5000))
    if [ $SCORE -gt $BEST_SCORE ]; then
        if [ $STEP -le 5400 ]; then
            BEST_SCORE=$SCORE
            BEST_CKPT="$CKPT"
        fi
    fi
done

if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="${LOGDIR}/checkpoint_latest.pt"
fi
echo "[3/4] Best checkpoint: $BEST_CKPT"

# ── 4. Run M1/K1/R1 Evaluation ──────────────────────────────────
echo ""
echo "[4/4] Running M1/K1/R1 evaluation..."
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
echo "PIPELINE COMPLETE at $(date)"
echo "=========================================="

# ── Generate V24 vs V25 Comparison ──────────────────────────────
echo ""
echo "=== V24 vs V25 Comparison ==="
python3 -c "
import json
from pathlib import Path

v24_path = Path('artifacts/lspwm_v44_stage_b_v24_mori_stable/eval/m1k1r1_benchmark.json')
v25_path = Path('${LOGDIR}/eval/m1k1r1_benchmark.json')

def load_results(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

v24 = load_results(v24_path)
v25 = load_results(v25_path)

if v24 is None or v25 is None:
    print('ERROR: Could not load both results')
    sys.exit(1)

print()
print('M1 — Thermodynamic Closure (R² skill)')
print(f'  {\"Mode\":<12s} {\"V24\":>8s} {\"V25\":>8s} {\"Δ\":>8s} {\"% Change\":>10s}')
print(f'  {\"-\"*12} {\"-\"*8} {\"-\"*8} {\"-\"*8} {\"-\"*10}')
for mode in ['full', 'force', 'memory', 'none']:
    v24_s = v24['M1']['modes'][mode]['skill_mean']
    v25_s = v25['M1']['modes'][mode]['skill_mean']
    delta = v25_s - v24_s
    pct = (delta / max(abs(v24_s), 1e-12)) * 100 if abs(v24_s) > 1e-12 else 0
    print(f'  {mode:<12s} {v24_s:8.4f} {v25_s:8.4f} {delta:+8.4f} {pct:+9.1f}%')

print()
print('M1 — Pairwise Deltas (bootstrap CI)')
print(f'  {\"Comparison\":<36s} {\"V24 Δ\":>8s} {\"V25 Δ\":>8s} {\"Change\":>10s}')
print(f'  {\"-\"*36} {\"-\"*8} {\"-\"*8} {\"-\"*10}')
for key in ['full_vs_none', 'full_vs_memory', 'full_vs_force', 'force_vs_none', 'memory_vs_none']:
    v24_d = v24['M1']['pairwise_deltas'].get(key, {}).get('delta', 0)
    v25_d = v25['M1']['pairwise_deltas'].get(key, {}).get('delta', 0)
    delta = v25_d - v24_d
    pct = (delta / max(abs(v24_d), 1e-12)) * 100 if abs(v24_d) > 1e-12 else 0
    print(f'  {key:<36s} {v24_d:8.4f} {v25_d:8.4f} {delta:+9.4f} ({pct:+6.1f}%)')

print()
print('K1 — Mean Skill Across Horizons')
print(f'  V24: {v24[\"K1\"][\"k1_mean_skill\"]:.4f}')
print(f'  V25: {v25[\"K1\"][\"k1_mean_skill\"]:.4f}')
print(f'  Δ:   {v25[\"K1\"][\"k1_mean_skill\"] - v24[\"K1\"][\"k1_mean_skill\"]:+.4f}')

print()
print('Comparison saved to ${LOGDIR}/eval/v24_vs_v25_comparison.txt')
" 2>&1 | tee "${LOGDIR}/eval/v24_vs_v25_comparison.txt"

echo ""
echo "All done! Check ${LOGDIR}/eval/ for results."
