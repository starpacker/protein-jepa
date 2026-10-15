#!/bin/bash
set -euo pipefail
# ============================================================================
# V26 Training + Evaluation Serial Pipeline
# V25 baseline + per-residue dynamics integration (q_i ∈ R^(B×L×8))
# ============================================================================

TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S')
LOGDIR="/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v26_per_residue"
LOGFILE="${LOGDIR}/pipeline.log"
CONFIG="/liuyutian/protein_jepa_frame/configs/lspwm_v44_stage_b_v26_per_residue.yaml"

mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "V26 Train + M1/K1/R1 Eval Pipeline"
echo "Started: $(date)"
echo "Config:  $CONFIG"
echo "Logdir:  $LOGDIR"
echo "=========================================="
echo "V26 CHANGES:"
echo "  + per_residue_step()       — integrate per-residue Langevin eqn"
echo "  + per_residue_transition_nll — Gaussian NLL training loss (weight=0.1)"
echo "  + _predict_per_residue_aux — history→bath for per-residue dynamics"
echo "  The per-residue state q_i ∈ R^(B×L×8) now participates in"
echo "  dynamics integration (previously: only global G ∈ R^(B×8))"
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

echo "[0/5] Validating V26 imports + per-residue dynamics..."
python3 -c "
from protein_jepa_frame.models.lspwm import LSPWMProteinWorldModel
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_lspwm_state_dict
import torch
from pathlib import Path

cfg = load_config('$CONFIG')
model = ProteinJEPA(cfg.model)
print('  V26 model built OK')
print('  per_residue_enabled:', model.lspwm.per_residue_enabled)
print('  per_residue_step:', hasattr(model.lspwm, 'per_residue_step'))
print('  per_residue_transition_nll:', hasattr(model.lspwm, 'per_residue_transition_nll'))
print('  _predict_per_residue_aux:', hasattr(model.lspwm, '_predict_per_residue_aux'))
print('  history_to_bath:', hasattr(model.lspwm, 'history_to_bath'))
print('  multi_bath:', hasattr(model.lspwm, 'multi_bath'))
print('  unified_energy:', hasattr(model.lspwm, 'unified_energy'))

# Test V25 ckpt backward compat
# V26 adds new params (per-residue dynamics losses), load with strict=False
ckpt = torch.load('artifacts/lspwm_v44_stage_b_v25_comprehensive/checkpoint_step5400.pt', map_location='cpu')
state = upgrade_lspwm_state_dict(model, ckpt.get('model', {}))
m, u = model.load_state_dict(state, strict=False)
print(f'  V25 ckpt compat: missing={len(m)}, unexpected={len(u)}')
for k in sorted(m):
    print(f'    missing: {k}')
for k in sorted(u):
    print(f'    unexpected: {k}')
print('V26 validation: OK')
"

# ── 1. Launch V26 Training ────────────────────────────────────
echo ""
echo "[1/5] Launching V26 training..."
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
print(f'phase={phase} loss={loss:.2f} mori={mori:.3f} G={G:.4f} Q={Q:.4f} skill={skill:.4f} per_res_nll={per_res_nll:.2f}')
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
    # Choose latest checkpoint before potential rollout explosion (≤5400)
    if [ "$STEP" -gt "$BEST_STEP" ] && [ "$STEP" -le 5400 ]; then
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

# ── 5. V25 vs V26 Comparison ───────────────────────────────────
echo ""
echo "[5/5] V25 vs V26 Comparison..."

python3 << 'PYEOF'
import json, sys
from pathlib import Path

v25_path = Path('artifacts/lspwm_v44_stage_b_v25_comprehensive/eval/m1k1r1_benchmark.json')
v26_path = Path('artifacts/lspwm_v44_stage_b_v26_per_residue/eval/m1k1r1_benchmark.json')

def load(path):
    if not path.exists():
        print(f'ERROR: {path} not found')
        return None
    with open(path) as f:
        return json.load(f)

v25 = load(v25_path)
v26 = load(v26_path)

if v25 is None or v26 is None:
    sys.exit(1)

print()
print('=' * 80)
print('V25 vs V26 COMPARISON')
print('=' * 80)

# ── M1 ──
print('\n--- M1: Thermodynamic Closure (2×2 {force}×{memory}) ---')
modes = ['full', 'force', 'memory', 'none']
print(f'  {"Mode":<10s} {"V25 R²":>10s} {"V26 R²":>10s} {"Δ":>10s} {"% Δ":>8s} {"Sig?":>6s}')
print(f'  {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*8} {"-"*6}')
for mode in modes:
    m25 = v25['M1']['modes'][mode]
    m26 = v26['M1']['modes'][mode]
    s25 = m25['skill_mean']
    s26 = m26['skill_mean']
    d = s26 - s25
    pct = (d / max(abs(s25), 1e-12)) * 100 if abs(s25) > 1e-12 else 0
    sig25 = "✓" if m25.get('significant', False) else "✗"
    sig26 = "✓" if m26.get('significant', False) else "✗"
    print(f'  {mode:<10s} {s25:10.4f} {s26:10.4f} {d:+10.4f} {pct:+7.1f}% {sig25:>3s}>{sig26}')

print()
print('  --- Pairwise Deltas ---')
print(f'  {"Comparison":<30s} {"V25 Δ":>10s} {"V26 Δ":>10s} {"ΔΔ":>10s}')
print(f'  {"-"*30} {"-"*10} {"-"*10} {"-"*10}')
for key in ['full_vs_none', 'full_vs_memory', 'full_vs_force', 'force_vs_none', 'memory_vs_none']:
    d25 = v25['M1']['pairwise_deltas'].get(key, {}).get('delta', float('nan'))
    d26 = v26['M1']['pairwise_deltas'].get(key, {}).get('delta', float('nan'))
    dd = d26 - d25
    print(f'  {key:<30s} {d25:10.4f} {d26:10.4f} {dd:+10.4f}')

# ── K1 ──
print('\n--- K1: Folding Prediction Skill ---')
k25 = v25['K1']
k26 = v26['K1']
print(f'  {"Metric":<25s} {"V25":>10s} {"V26":>10s} {"Δ":>10s}')
print(f'  {"-"*25} {"-"*10} {"-"*10} {"-"*10}')
for h in ['24', '48', '96']:
    hk = f'k1_per_horizon_{h}'
    if hk in k25.get('k1_per_horizon', {}):
        s25 = k25['k1_per_horizon'][hk]['skill_mean']
        s26 = k26['k1_per_horizon'][hk]['skill_mean']
        print(f'  horizon={h:>3s} frames            {s25:10.4f} {s26:10.4f} {s26-s25:+10.4f}')
print(f'  {"k1_mean_skill":<25s} {k25["k1_mean_skill"]:10.4f} {k26["k1_mean_skill"]:10.4f} {k26["k1_mean_skill"]-k25["k1_mean_skill"]:+10.4f}')

# ── R1 ──
print('\n--- R1: Representation Closure ---')
r25 = v25.get('R1', {})
r26 = v26.get('R1', {})
gp25 = r25.get('r1_gate_pass', r25.get('gate_pass', 'N/A'))
gp26 = r26.get('r1_gate_pass', r26.get('gate_pass', 'N/A'))
print(f'  V25 gate pass: {gp25}')
print(f'  V26 gate pass: {gp26}')

# ── Summary ──
print()
print('=' * 80)
print('KEY DIFFERENCES')
print('=' * 80)
m1_25 = v25['M1']['modes']['full']['skill_mean']
m1_26 = v26['M1']['modes']['full']['skill_mean']
k1_25 = v25['K1']['k1_mean_skill']
k1_26 = v26['K1']['k1_mean_skill']
print(f'  M1 M_Full R²:  {m1_25:.4f} → {m1_26:.4f}  (Δ={m1_26-m1_25:+.4f})')
print(f'  K1 Mean Skill: {k1_25:.4f} → {k1_26:.4f}  (Δ={k1_26-k1_25:+.4f})')
print()
if m1_26 > m1_25 and k1_26 > k1_25:
    print('  ✅ V26 IMPROVES BOTH M1 AND K1!')
elif m1_26 > m1_25:
    print('  ✅ V26 improves M1 but K1 unchanged/regressed')
elif k1_26 > k1_25:
    print('  ✅ V26 improves K1 but M1 unchanged/regressed')
else:
    print('  ⚠ V26 did not improve over V25 — check per-residue dynamics')
PYEOF

echo ""
echo "=========================================="
echo "V26 PIPELINE COMPLETE at $(date)"
echo "Check: ${LOGDIR}/"
echo "  - stdout.log     (full training output)"
echo "  - pipeline.log   (this log)"
echo "  - eval/          (M1/K1/R1 results)"
echo "=========================================="
